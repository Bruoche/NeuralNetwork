import os, shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
from trainer.monitor import MonitorModel
import combination

tf.keras.utils.set_random_seed(67)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

optionalRotation = combination.Options([[tf.keras.layers.RandomRotation(0.15)], []])
optionalZoom = combination.Options([[tf.keras.layers.RandomZoom(0.1)], []])
optionalTranslation = combination.Options([[tf.keras.layers.RandomTranslation(0.1, 0.1)], []])
optionalConvolution = combination.Options([[
	tf.keras.layers.Conv2D(32, 3, activation="relu"),
	tf.keras.layers.MaxPooling2D(),
	tf.keras.layers.Conv2D(64, 3, activation="relu"),
	tf.keras.layers.MaxPooling2D(),
], []])
weights = combination.Options([32, 64, 128, 256])
learning_rates = combination.Options([0.00003, 0.00005, 0.0001])

iterator = combination.Iterator([weights, learning_rates, optionalConvolution, optionalTranslation, optionalRotation, optionalZoom])
for i in iterator:
	print(f"Doing combination {i}/{iterator.nb_combinations()}")
	model = tf.keras.Sequential([
		tf.keras.layers.Input(shape=(28,28, 1))] 
		+ optionalRotation.get()
		+ optionalZoom.get()
		+ optionalTranslation.get()
		+ optionalConvolution.get()
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

	monitor = MonitorModel("mnist_iterator", 1)
	monitor.train(model, X_train, y_train, X_test, y_test, nb_epoch=200, callbacks=[early_stopping])
