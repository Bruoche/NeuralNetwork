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

optionalFlip = combination.Options([0, 0.5])
optionalFlipVertical = combination.Options([[]])
optionalRotation = combination.Options([0.15])
optionalTranslation = combination.Options([(0.1, 0.1)])
optionalScale = combination.Options([0.1])
optionalDepth = combination.Options([
	[],
	[64],
	[128],
	[64, 64]
])
learning_rates = combination.Options([0.0005, 0.0003])
optionalFineTune = combination.Options([4, 2]) # 3, 2, 
optionalBase = combination.Options([tf.keras.applications.EfficientNetB0])

iterator = combination.Iterator([optionalDepth, learning_rates, optionalFlip, optionalRotation, optionalTranslation, optionalScale, optionalFlipVertical, optionalFineTune, optionalBase])
for i in iterator:
	print(f"Doing combination {i}/{iterator.nb_combinations()}")

	augment = [
		tf.keras.layers.RandomTranslation(optionalTranslation.get()[0], optionalTranslation.get()[1]),
		tf.keras.layers.RandomZoom(optionalScale.get()),
	] + ([tf.keras.layers.RandomFlip("horizontal")] if optionalFlip.get() else []) \
	  + [tf.keras.layers.RandomRotation(optionalRotation.get())] \
	  + optionalFlipVertical.get()

	base = optionalBase.get()(input_shape=(*DIMENSIONS, 3), include_top=False, weights="imagenet")
	base.trainable = False
	model = tf.keras.Sequential(
		[tf.keras.layers.Input(shape=(*DIMENSIONS, 3))]
		+ augment
		+ [base, tf.keras.layers.GlobalAveragePooling2D()]
		+ [tf.keras.layers.Dense(h, activation="relu") for h in optionalDepth.get()]
		+ [tf.keras.layers.Dropout(0.3),
		   tf.keras.layers.Dense(len(class_names), activation="softmax")]
	)
	model.compile(
		loss=tf.keras.losses.SparseCategoricalCrossentropy(),
		optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rates.get()),
		metrics=["accuracy"]
	)
	early_stopping = tf.keras.callbacks.EarlyStopping(
		monitor='val_loss',
		patience=10,
		restore_best_weights=True,
		verbose=1
	)

	monitor = MonitorModel("object100_tf_efficientnet_b0", 0)
	monitor.train(
		TensorModel(model, base=base, fine_tune=optionalFineTune.get(), patience=10),
		X_train, y_train, X_test, y_test,
		nb_epoch=300,
		class_names=class_names,
		callbacks=[early_stopping]
	)
