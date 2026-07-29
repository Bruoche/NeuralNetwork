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
RES = 0.5
DIMENSIONS = (int(1024*RES), int(768*RES))

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
X_train = np.stack([x for x, _ in train_data.as_numpy_iterator()]) / 255.0
y_train = np.array([y for _, y in train_data.as_numpy_iterator()])
X_test  = np.stack([x for x, _ in test_data.as_numpy_iterator()]) / 255.0
y_test  = np.array([y for _, y in test_data.as_numpy_iterator()])
class_names = train_data.classes_names

optionalRotation = combination.Options([[tf.keras.layers.RandomRotation(0.15)], []])
optionalZoom = combination.Options([[tf.keras.layers.RandomZoom(0.1)], []])
optionalTranslation = combination.Options([[tf.keras.layers.RandomTranslation(0.1, 0.1)], []])
optionalConvolution = combination.Options([[
	tf.keras.layers.Conv2D(32, 3, activation="relu"),
	tf.keras.layers.MaxPooling2D(),
	tf.keras.layers.Conv2D(64, 3, activation="relu"),
	tf.keras.layers.MaxPooling2D(),
], []])
weights = combination.Options([256, 128, 64, 32])
learning_rates = combination.Options([0.001, 0.0001, 0.00003, 0.00005])
iterator = combination.Iterator([weights, learning_rates, optionalTranslation, optionalRotation, optionalZoom])

for i in iterator:
	print(f"Doing combination {i}/{iterator.nb_combinations()}")
	model = tf.keras.Sequential([
		tf.keras.layers.Input(shape=(*DIMENSIONS, 3))] 
		+ optionalRotation.get()
		+ optionalZoom.get()
		+ optionalTranslation.get()
		+ [tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(weights.get(), activation="relu"),
		tf.keras.layers.Dropout(0.3),
		tf.keras.layers.Dense(weights.get(), activation="relu"),
		tf.keras.layers.Dropout(0.2),
		tf.keras.layers.Dense(len(np.unique(y_train)), activation="softmax")
	])
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

	monitor = MonitorModel("object100_tf_train", 1)
	monitor.train(TensorModel(model), X_train, y_train, X_test, y_test, nb_epoch=200, class_names=class_names, callbacks=[early_stopping])
