import os, json, glob, time
from datetime import timedelta

import numpy as np
import pandas
import sklearn
import itertools
from pathlib import Path

from trainer.model import Model

class MonitorModel:

	MONITORING_DIR = "monitoring"
	MODELS_DIR = "models"
	MAX_RETRIES = 5

	def __init__(self, model_name, major_version):
		self.model_name = model_name
		self.major_version = major_version

	def train(self, model: Model, X_train, y_train, X_test, y_test, nb_epoch = 300, callbacks=[], class_names=None):
		Path(self.__model_directory()).mkdir(parents=True, exist_ok=True)
		full_name = self.__next_name()
		print(f"Model: {full_name}")
		classes = np.unique(y_train)
		if class_names is None:
			class_names = [str(c) for c in classes]
		print(f"Detected classes: {class_names}")

		history, duration = self.__train_model(model, full_name, X_train, y_train, X_test, y_test, nb_epoch, callbacks)

		monitoring_path = f"{MonitorModel.MONITORING_DIR}/{self.model_name}/{full_name}"
		self.__save_all(history, full_name, classes, class_names, monitoring_path, model, X_test, y_test)
		self.__save_model_params(model, full_name, len(history["loss"]), class_names, callbacks, monitoring_path, duration)
		print("All file saved!")


	def __next_name(self, retry = 0):
		major_name = f"{self.model_name}_{format(self.major_version, '02d')}"
		version = len(glob.glob(f"{self.__model_directory()}/{major_name}_*"))
		full_name = f"{major_name}_{format(version, '02d')}"
		claim_path = f"versions/{full_name}.claim"
		Path(claim_path).mkdir(parents=True, exist_ok=True)
		try:
			open(claim_path, 'x').close()
		except FileExistsError:
			if retry > MonitorModel.MAX_RETRIES:
				raise ValueError(f"Failed to claim a version for the model after {retry} retries.")
			return self.__next_name(retry+1)
		return full_name

	def __model_directory(self):
		return f"{MonitorModel.MODELS_DIR}/{self.model_name}"

	def __train_model(self, model: Model, full_name, X_train, y_train, X_test, y_test, nb_epoch = 300, callbacks=[]):
		start_time = time.perf_counter()
		history = model.fit(X_train, y_train, X_test, y_test, nb_epoch, callbacks)
		duration = time.perf_counter() - start_time
		model.evaluate(X_test, y_test)

		save_path = model.save(f"{self.__model_directory()}/{full_name}")
		print(f"\"{save_path}\" saved.")
		return history, duration

	def __save_all(self, history, full_name, classes, class_names, monitoring_path, model: Model, X_test, y_test):
		# Per epoch metrics
		training_evolution = (
			pandas.DataFrame(history)
				.rename_axis("epoch")
				.reset_index()
		)
		training_evolution.insert(0, "model", full_name)
		self.__save_one(training_evolution, "metrics", monitoring_path)
		# Convolution matrix
		prediction = model.predict(X_test)
		confusion_matrix = sklearn.metrics.confusion_matrix(y_test, prediction, labels=range(len(classes)))
		confusion_data = []
		for i, j in itertools.product(range(len(classes)), repeat=2):
			confusion_data.append({
			"model": full_name,
			"wanted": class_names[i],
			"prediction": class_names[j],
			"count": int(confusion_matrix[i, j])
		})
		confusion_dataframe = pandas.DataFrame(confusion_data)
		self.__save_one(confusion_dataframe, "confusion", monitoring_path)

	def __save_one(self, monitor_dataframe, title, monitoring_path):
		Path(monitoring_path).mkdir(parents=True, exist_ok=True)
		monitor_dataframe.to_csv(f"{monitoring_path}/{title}.csv", index=False)

	def __save_model_params(self, model: Model, full_name, nb_epoch, classes, callbacks, monitoring_path, duration):
		model_params = {
			"model": full_name,
			"epochs": nb_epoch,
			"classes": classes,
			"callbacks": callbacks,
			"learning_rate": model.learning_rate(),
			"training_duration": str(timedelta(seconds=duration)),
			"shape": {
				"input": model.input_shape()
			}
		}
		for i, layer in enumerate(model.describe_layers()):
			model_params["shape"][f"layer_{i}"] = layer
		with open(f"{monitoring_path}/params.json", "w") as file:
			json.dump(
				model_params,
				file,
				default=lambda obj: self.__serialize(obj),
				indent=4
			)

	def __serialize(self, value):
		if hasattr(value, "tolist"):
			return value.tolist()
		return str(value)
