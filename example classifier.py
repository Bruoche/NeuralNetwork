import os, shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
from trainer.monitor import MonitorModel


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28,28, 1)),
	tf.keras.layers.Conv2D(32, 3, activation="relu"),
	tf.keras.layers.MaxPooling2D(),
	tf.keras.layers.Conv2D(64, 3, activation="relu"),
	tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
	tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation="relu"),
	tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(np.unique(y_train)), activation="softmax")
])
model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00005),
    metrics = ["accuracy"]
)

shutil.rmtree("temp", ignore_errors=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard(
	log_dir="temp"
)
early_stopping = tf.keras.callbacks.EarlyStopping(
	monitor='val_loss',
	patience=5,
	restore_best_weights=True,
	verbose=1
)

monitor = MonitorModel("mnist_composite", 1)
monitor.train(model, X_train, y_train, X_test, y_test, nb_epoch=200, callbacks=[tensorboard_callback, early_stopping])
