import os, json, glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
import pandas
import sklearn
import itertools
from pathlib import Path

class MonitorModel:

	MONITORING_DIR = "monitoring"
	MODELS_DIR = "models"
	MAX_RETRIES = 5

	def __init__(self, model_name, major_version):
		self.model_name = model_name
		self.major_version = major_version

	def train(self, model, X_train, y_train, X_test, y_test, nb_epoch = 300, callbacks=[]):
		Path(self.__model_directory()).mkdir(parents=True, exist_ok=True)
		full_name = self.__next_name()
		print(f"Model: {full_name}")
		classes = np.unique(y_train)
		print(f"Detected classes: {classes}")

		monitoring_data = self.__train_model(model, full_name, X_train, y_train, X_test, y_test, nb_epoch, callbacks)

		monitoring_path = f"{MonitorModel.MONITORING_DIR}/{self.model_name}/{full_name}"
		self.__save_all(monitoring_data, full_name, classes, monitoring_path, model, X_test, y_test)
		self.__save_model_params(model, full_name, len(monitoring_data.history["loss"]), classes, callbacks, monitoring_path)
		print("All file saved!")


	def __next_name(self, retry = 0):
		major_name = f"{self.model_name}_{format(self.major_version, '02d')}"
		version = len(glob.glob(f"{self.__model_directory()}/{major_name}_*"))
		full_name = f"{major_name}_{format(version, '02d')}"
		try:
			open(f"{self.__model_directory()}/{full_name}.claim", 'x').close()
		except FileExistsError:
			if retry > MonitorModel.MAX_RETRIES:
				raise ValueError(f"Failed to claim a version for the model after {retry} retries.")
			return self.__next_name(retry+1)
		return full_name

	def __model_directory(self):
		return f"{MonitorModel.MODELS_DIR}/{self.model_name}"

	def __train_model(self, model, full_name, X_train, y_train, X_test, y_test, nb_epoch = 300, callbacks=[]):
		monitoring_data = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=nb_epoch, verbose=2, callbacks=callbacks)
		model.evaluate(X_test, y_test, verbose=2)

		save_path = f"{self.__model_directory()}/{full_name}.keras"
		model.save(save_path)
		print(f"\"{save_path}\" saved.")
		try: os.remove(f"{self.__model_directory()}/{full_name}.claim")
		except OSError: pass
		return monitoring_data

	def __save_all(self, monitoring_data, full_name, classes, monitoring_path, model, X_test, y_test):
		# Per epoch metrics
		training_evolution = (
			pandas.DataFrame(monitoring_data.history)
				.rename_axis("epoch")
				.reset_index()
		)
		training_evolution.insert(0, "model", full_name)
		self.__save_one(training_evolution, "metrics", monitoring_path)
		# Convolution matrix
		prediction = np.argmax(model.predict(X_test, verbose=0), axis=1)
		confusion_matrix = sklearn.metrics.confusion_matrix(y_test, prediction, labels=range(len(classes)))
		confusion_data = []
		for i, j in itertools.product(range(len(classes)), repeat=2):
			confusion_data.append({
			"model": full_name,
			"wanted": classes[i],
			"prediction": classes[j],
			"count": int(confusion_matrix[i, j])
		})
		confusion_dataframe = pandas.DataFrame(confusion_data)
		self.__save_one(confusion_dataframe, "confusion", monitoring_path)

	def __save_one(self, monitor_dataframe, title, monitoring_path):
		Path(monitoring_path).mkdir(parents=True, exist_ok=True)
		monitor_dataframe.to_csv(f"{monitoring_path}/{title}.csv", index=False)

	def __save_model_params(self, model, full_name, nb_epoch, classes, callbacks, monitoring_path):
		model_params = {
			"model": full_name,
			"epochs": nb_epoch,
			"classes": classes,
			"callbacks": callbacks,
			"learning_rate": float(model.optimizer.learning_rate.numpy()),
			"shape": {
				"input": model.input_shape[1:]
			}
		}
		for i, layer in enumerate(model.layers):
			model_params["shape"][f"layer_{i}"] = self.__describe(layer)
		json.dump(
			model_params, 
			open(f"{monitoring_path}/params.json", "w"),
			default=lambda obj: self.__serialize(obj),
			indent=4
		)

	def __describe(self, layer):
		t = type(layer).__name__
		desc = {"type": t}
		if hasattr(layer, "filters"): # Convo2D/1D...
			desc["size"] = layer.filters
			desc["kernel"] = layer.kernel_size
			desc["stride"] = layer.strides
			return desc
		if hasattr(layer, "units"): # Dense
			desc["size"] = layer.units
			return desc
		if hasattr(layer, "pool_size"): # MaxPooling
			desc["pool"] = layer.pool_size
			return desc
		if hasattr(layer, "rate"): # Dropout
			desc["rate"] = layer.rate
		return desc

	def __serialize(self, value):
		if hasattr(value, "tolist"):
			return value.tolist()
		return str(value)