import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
from trainer.monitor import MonitorModel
from trainer.model import TensorModel
import combination

SEED = 67
tf.keras.utils.set_random_seed(SEED)
# RES = 0.2
# DIMENSIONS = (int(1024*RES), int(768*RES)) # non square
DIMENSIONS = (224, 224)

train_data, test_data = tf.keras.utils.image_dataset_from_directory(
	"datasets/objects100",
	labels="inferred", 
	label_mode="int",
	image_size=DIMENSIONS,
	batch_size=None,
	validation_split=0.2,
	subset="both",
	seed=SEED
)
X_train, y_train = map(np.array, zip(*train_data.as_numpy_iterator()))
X_test,  y_test  = map(np.array, zip(*test_data.as_numpy_iterator()))
class_names = train_data.class_names

optionalTransformations = combination.Options([lambda: [
	tf.keras.layers.RandomFlip("horizontal"),
	tf.keras.layers.RandomRotation(0.15),
	tf.keras.layers.RandomZoom(0.1),
	tf.keras.layers.RandomTranslation(0.1, 0.1)
], lambda: []])
optionalDepth = combination.Options([
	lambda: [],
	lambda: [tf.keras.layers.Dense(64, activation="relu"),], 
	lambda: [tf.keras.layers.Dense(128, activation="relu")],
	lambda: [tf.keras.layers.Dense(256, activation="relu")]
])
learning_rates = combination.Options([0.00005, 0.0001, 0.001])

iterator = combination.Iterator([learning_rates, optionalDepth])
for i in iterator:
	print(f"Doing combination {i}/{iterator.nb_combinations()}")

	base = tf.keras.applications.MobileNetV2(input_shape=(*DIMENSIONS, 3), include_top=False, weights="imagenet")
	base.trainable = False
	model = tf.keras.Sequential(
		[tf.keras.layers.Input(shape=(*DIMENSIONS, 3))] 
		+ [tf.keras.layers.Rescaling(1./127.5, offset=-1),
	 	base,
		tf.keras.layers.GlobalAveragePooling2D()]
		+ optionalDepth.get()()
	 	+ [tf.keras.layers.Dropout(0.3),
		tf.keras.layers.Dense(len(class_names), activation="softmax")]
	)
	model.compile(
		loss = tf.keras.losses.SparseCategoricalCrossentropy(),
		optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rates.get()),
		metrics = ["accuracy"]
	)
	early_stopping = tf.keras.callbacks.EarlyStopping(
		monitor='val_loss',
		patience=5,
		restore_best_weights=True,
		verbose=1
	)

	monitor = MonitorModel("object100_tf_mobilenet_v2", 0)
	monitor.train(TensorModel(model), X_train, y_train, X_test, y_test, nb_epoch=1, class_names=class_names, callbacks=[early_stopping])
