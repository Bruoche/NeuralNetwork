import sys, os
from abc import ABC, abstractmethod
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import combination
from trainer.monitor import MonitorModel
from trainer.model import Model, TensorModel, TorchModel

# Shared transfer-learning driver for both frameworks. object100_transfer.py
# defines the (framework-agnostic) combination.Options once and calls train();
# everything engine-specific lives behind the TrainerFactory interface, so the
# generic code below is written once. Augmentation Options carry canonical
# values (flip probability, rotation as a fraction of a turn, translation as a
# fraction, zoom as a +/- fraction) that each factory turns into its own layers.


class TrainerFactory(ABC):
	"""Engine-specific pieces the generic train() needs, one impl per library."""

	@abstractmethod
	def acronym(self):
		"""Returns the string short name for the trainer, so they can be inserted inside the model name"""

	@abstractmethod
	def setup(self, seed):
		"""Seed the RNGs and do any one-time engine preparation."""

	@abstractmethod
	def load_data(self, dataset, dimensions, seed):
		"""Return (X_train, y_train, X_test, y_test, class_names) as numpy arrays,
		images shaped (N, H, W, C) in the 0-255 range."""

	@abstractmethod
	def build_model(self, num_classes, depth, lr, flip, rotation, translation,
					scale, flip_vertical, fine_tune, base_name, dimensions, patience) -> Model:
		"""Build one combination's model, wrapped as a Model."""

	@abstractmethod
	def callbacks(self, patience):
		"""Callbacks passed to monitor.train for this engine."""


class TensorFactory(TrainerFactory):

	def acronym(self):
		return "tf"
	
	def setup(self, seed):
		import os
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
		os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
		import tensorflow as tf
		tf.keras.utils.set_random_seed(seed)

	def load_data(self, dataset, dimensions, seed):
		import tensorflow as tf
		import numpy as np
		train_data, test_data = tf.keras.utils.image_dataset_from_directory(
			dataset, labels="inferred", label_mode="int", image_size=dimensions,
			batch_size=None, validation_split=0.2, subset="both", seed=seed)
		X_train, y_train = map(np.array, zip(*train_data.as_numpy_iterator()))
		X_test,  y_test  = map(np.array, zip(*test_data.as_numpy_iterator()))
		return X_train, y_train, X_test, y_test, train_data.class_names

	def build_model(self, num_classes, depth, lr, flip, rotation, translation,
					scale, flip_vertical, fine_tune, base_name, dimensions, patience) -> Model:
		import tensorflow as tf
		L = tf.keras.layers
		# keras EfficientNet has preprocessing built in and expects raw [0,255],
		# so there is no Rescaling layer here
		augment = [L.RandomTranslation(translation[0], translation[1]), L.RandomZoom(scale)]
		if flip:
			augment.append(L.RandomFlip("horizontal"))
		augment.append(L.RandomRotation(rotation))
		if flip_vertical:
			augment.append(L.RandomFlip("vertical"))

		base = self.__bases()[base_name](input_shape=(*dimensions, 3), include_top=False, weights="imagenet")
		base.trainable = False
		model = tf.keras.Sequential(
			[L.Input(shape=(*dimensions, 3))]
			+ augment
			+ [base, L.GlobalAveragePooling2D()]
			+ [L.Dense(h, activation="relu") for h in depth]
			+ [L.Dropout(0.3), L.Dense(num_classes, activation="softmax")]
		)
		model.compile(
			loss=tf.keras.losses.SparseCategoricalCrossentropy(),
			optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
			metrics=["accuracy"])
		return TensorModel(model, base=base, fine_tune=fine_tune, patience=patience)

	def callbacks(self, patience):
		import tensorflow as tf
		return [tf.keras.callbacks.EarlyStopping(
			monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1)]

	def __bases(self):
		import tensorflow as tf
		return {
			"efficientnet_b0": tf.keras.applications.EfficientNetB0,
			"mobilenet_v2": tf.keras.applications.MobileNetV2,
		}


class TorchFactory(TrainerFactory):

	def acronym(self):
		return "pt"

	def setup(self, seed):
		import torch
		import numpy as np
		torch.manual_seed(seed)
		np.random.seed(seed)
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		print(f"Device: {self.device}")
		if self.device == "cuda":
			print(torch.cuda.get_device_name())

	def load_data(self, dataset, dimensions, seed):
		import numpy as np
		from torchvision import transforms
		from torchvision.datasets import ImageFolder
		resize = transforms.Resize(dimensions)
		data = ImageFolder(dataset)
		images = np.stack([
			np.asarray(resize(img.convert("RGB")), dtype="float32")
			for img, _ in data
		])
		labels = np.asarray([label for _, label in data], dtype="int64")
		indices = np.random.default_rng(seed).permutation(len(labels))
		val_count = int(0.2 * len(labels))
		test_idx, train_idx = indices[:val_count], indices[val_count:]
		return (images[train_idx], labels[train_idx],
				images[test_idx], labels[test_idx], data.classes)

	def build_model(self, num_classes, depth, lr, flip, rotation, translation,
					scale, flip_vertical, fine_tune, base_name, dimensions, patience) -> Model:
		from torchvision import transforms
		augment = [
			transforms.RandomAffine(degrees=0, translate=translation, scale=(1 - scale, 1 + scale)),
			transforms.RandomHorizontalFlip(flip),
			transforms.RandomRotation(rotation * 360),
		]
		if flip_vertical:
			augment.append(transforms.RandomVerticalFlip(flip_vertical))
		base, base_weights = self.__bases()[base_name]
		return TorchModel(
			num_classes=num_classes, augment=augment, hidden=depth, lr=lr,
			device=self.device, patience=patience, fine_tune=fine_tune,
			base=base, base_weights=base_weights)

	def callbacks(self, patience):
		# torch does early stopping inside TorchModel (via patience); this string
		# is purely what gets logged in params.json
		return [f"EarlyStopping(monitor=val_loss, patience={patience}, restore_best_weights=True)"]

	def __bases(self):
		from torchvision import models
		return {
			"efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
			"mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.IMAGENET1K_V1),
		}


FACTORIES = {
	"tensorflow": TensorFactory(),
	"pytorch": TorchFactory(),
}


def train(trainer, major_version, dataset, dimensions, seed, patience,
		  nb_epoch, depth, learning_rate, flip, rotation, translation, scale,
		  flip_vertical, fine_tune, base):
	"""Run the full transfer-learning sweep for one engine. Every argument from
	`depth` onward is a combination.Options; the iterator order below fixes how
	combination indices map, and is identical for both engines."""
	factory = FACTORIES[trainer]
	factory.setup(seed)
	X_train, y_train, X_test, y_test, class_names = factory.load_data(f"datasets/{dataset}", dimensions, seed)

	options = [depth, learning_rate, flip, rotation, translation, scale,
			   flip_vertical, fine_tune, base]
	iterator = combination.Iterator(options)
	for i in iterator:
		print(f"Doing combination {i}/{iterator.nb_combinations()}")

		model = factory.build_model(
			num_classes=len(class_names),
			depth=depth.get(), lr=learning_rate.get(), flip=flip.get(),
			rotation=rotation.get(), translation=translation.get(), scale=scale.get(),
			flip_vertical=flip_vertical.get(), fine_tune=fine_tune.get(),
			base_name=base.get(), dimensions=dimensions, patience=patience)

		monitor = MonitorModel(f"{dataset}_{factory.acronym()}_{base.get()}", major_version)
		monitor.train(
			model, X_train, y_train, X_test, y_test,
			nb_epoch=nb_epoch, class_names=class_names,
			callbacks=factory.callbacks(patience))
