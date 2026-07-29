from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):
	"""Framework-independent view of a trainable model.

	The monitor only ever talks to this interface, so it stays generic across
	TensorFlow / PyTorch. Each concrete implementation wraps a real model and
	exposes the few things the monitor needs, translated from the underlying
	library.
	"""

	@abstractmethod
	def fit(self, X_train, y_train, X_test, y_test, nb_epoch, callbacks=[]) -> dict:
		"""Train the model and return the per-epoch history as a plain dict of
		metric-name -> list of values (keys: loss, accuracy, val_loss,
		val_accuracy)."""

	@abstractmethod
	def evaluate(self, X_test, y_test) -> None:
		"""Run a final evaluation pass on the test set (side-effect / logging)."""

	@abstractmethod
	def predict(self, X) -> np.ndarray:
		"""Return predicted class indices (argmax already applied), shape (n,)."""

	@abstractmethod
	def save(self, path_without_extension) -> str:
		"""Persist the model, appending the framework's own extension. Returns
		the actual path written."""

	@abstractmethod
	def learning_rate(self) -> float:
		"""Current optimizer learning rate."""

	@abstractmethod
	def input_shape(self):
		"""Input shape without the batch dimension, e.g. (224, 224, 3)."""

	@abstractmethod
	def describe_layers(self) -> list[dict]:
		"""Ordered list of layer descriptions ({"type": ..., ...})."""


class TensorModel(Model):
	"""Model implementation wrapping a compiled tf.keras model."""

	def __init__(self, keras_model):
		self.model = keras_model

	def fit(self, X_train, y_train, X_test, y_test, nb_epoch, callbacks=[]) -> dict:
		history = self.model.fit(
			X_train, y_train,
			validation_data=(X_test, y_test),
			epochs=nb_epoch,
			verbose=2,
			callbacks=callbacks
		)
		return history.history

	def evaluate(self, X_test, y_test) -> None:
		self.model.evaluate(X_test, y_test, verbose=2)

	def predict(self, X) -> np.ndarray:
		return np.argmax(self.model.predict(X, verbose=0), axis=1)

	def save(self, path_without_extension) -> str:
		save_path = f"{path_without_extension}.keras"
		self.model.save(save_path)
		return save_path

	def learning_rate(self) -> float:
		return float(self.model.optimizer.learning_rate.numpy())

	def input_shape(self):
		return self.model.input_shape[1:]

	def describe_layers(self) -> list[dict]:
		return [self.__describe(layer) for layer in self.model.layers]

	def __describe(self, layer) -> dict:
		t = type(layer).__name__
		desc = {"type": t}
		for attr in ("factor", "height_factor", "width_factor", "mode", "fill_mode"):
			if hasattr(layer, attr):
				desc[attr] = getattr(layer, attr)
		if hasattr(layer, "layers"):  # nested/base model (e.g. MobileNetV2)
			desc["base_model"] = layer.name
			desc["size"] = layer.count_params()
			desc["trainable"] = layer.trainable
			return desc
		if hasattr(layer, "filters"):  # Convo2D/1D...
			desc["size"] = layer.filters
			desc["kernel"] = layer.kernel_size
			desc["stride"] = layer.strides
			return desc
		if hasattr(layer, "units"):  # Dense
			desc["size"] = layer.units
			return desc
		if hasattr(layer, "pool_size"):  # MaxPooling
			desc["pool"] = layer.pool_size
			return desc
		if hasattr(layer, "rate"):  # Dropout
			desc["rate"] = layer.rate
		return desc


class TorchModel(Model):
	"""Model implementation wrapping a PyTorch MobileNetV2 transfer-learning
	setup, built to mirror the tf.keras model in object100_transfer_tf.py:

	    [aug] -> normalize -> frozen MobileNetV2 features -> global avg pool
	          -> [Dense(h)+ReLU ...] -> Dropout(0.3) -> Dense(n_classes)

	Data augmentation lives inside the model (applied per-epoch on the training
	set only) exactly like the keras RandomFlip/Rotation/Zoom/Translation
	layers. torch imports are done lazily so this module still imports fine in
	a TF-only environment.
	"""

	BATCH_SIZE = 32       # tf.keras model.fit default
	DROPOUT = 0.3
	# torchvision transform attributes worth reporting, in output order
	AUGMENT_ATTRS = ("p", "degrees", "translate", "scale", "ratio", "shear",
					 "size", "brightness", "contrast", "saturation", "hue",
					 "kernel_size", "sigma", "distortion_scale")

	def __init__(self, num_classes, augment=[], hidden=[], lr=1e-3,
				 device=None, patience=5):
		import torch
		from torch import nn
		from torchvision import models, transforms

		self._torch = torch
		self._nn = nn
		self.num_classes = num_classes
		self.hidden = list(hidden)
		self.lr = lr
		self.patience = patience
		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

		weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
		base = models.mobilenet_v2(weights=weights)
		self.backbone = base.features                 # frozen conv feature extractor
		for p in self.backbone.parameters():
			p.requires_grad = False
		self.backbone.eval()                          # keep BatchNorm stats frozen
		self.pool = nn.AdaptiveAvgPool2d(1)           # GlobalAveragePooling2D -> 1280

		head_layers = []
		in_features = self.backbone[-1].out_channels  # 1280 for MobileNetV2
		for h in self.hidden:
			head_layers += [nn.Linear(in_features, h), nn.ReLU()]
			in_features = h
		# final Dense outputs logits; softmax is folded into CrossEntropyLoss,
		# the equivalent of keras softmax + SparseCategoricalCrossentropy
		head_layers += [nn.Dropout(TorchModel.DROPOUT), nn.Linear(in_features, num_classes)]
		self.head = nn.Sequential(*head_layers)

		self.backbone.to(self.device)
		self.pool.to(self.device)
		self.head.to(self.device)

		self.optimizer = torch.optim.Adam(self.head.parameters(), lr=lr)  # only head trains
		self.loss_fn = nn.CrossEntropyLoss()

		# preprocessing: normalize with the backbone's own ImageNet stats
		# (torch's equivalent of the keras Rescaling(1/127.5, -1) that matched
		# the TF MobileNetV2 pretraining)
		self._mean = weights.transforms().mean
		self._std = weights.transforms().std
		self._normalize = transforms.Normalize(mean=self._mean, std=self._std)
		self.augment = list(augment)
		self._augment_tf = transforms.Compose(self.augment)

	# --- helpers -----------------------------------------------------------
	def __describe_augment(self, transform) -> list[dict]:
		"""Describe one torchvision transform as JSON-friendly dicts, reading
		its parameters off the object instead of hard-coding them. Containers
		(Compose, RandomApply, ...) are flattened into their inner transforms."""
		inner = getattr(transform, "transforms", None)
		if inner is not None:
			return [desc for t in inner for desc in self.__describe_augment(t)]
		desc = {"type": type(transform).__name__}
		for attr in TorchModel.AUGMENT_ATTRS:
			value = getattr(transform, attr, None)
			if value is None:
				continue
			desc[attr] = list(value) if isinstance(value, (tuple, list)) else value
		return [desc]

	def describe_augments(self) -> list[dict]:
		"""Ordered description of the augmentation pipeline (empty if none)."""
		return [desc for t in self.augment for desc in self.__describe_augment(t)]

	def __forward(self, x):
		feats = self.backbone(x)
		pooled = self._torch.flatten(self.pool(feats), 1)
		return self.head(pooled)

	def __to_chw01(self, X):
		# (N, H, W, C) 0-255 float -> (N, C, H, W) float in [0, 1]
		t = self._torch.from_numpy(np.asarray(X, dtype="float32"))
		return t.permute(0, 3, 1, 2).contiguous() / 255.0

	def __loader(self, X, y, train):
		from torch.utils.data import DataLoader, Dataset
		normalize, augment_tf = self._normalize, self._augment_tf if train else None

		class _DS(Dataset):
			def __init__(self, imgs, labels):
				self.imgs = imgs
				self.labels = labels
			def __len__(self):
				return self.imgs.shape[0]
			def __getitem__(self, i):
				img = self.imgs[i]
				if augment_tf is not None:
					img = augment_tf(img)
				return normalize(img), int(self.labels[i])

		imgs = self.__to_chw01(X)
		labels = self._torch.as_tensor(np.asarray(y), dtype=self._torch.long)
		return DataLoader(_DS(imgs, labels), batch_size=TorchModel.BATCH_SIZE,
						  shuffle=train, num_workers=0,
						  pin_memory=(self.device == "cuda"))

	def __run_epoch(self, loader, train):
		torch = self._torch
		self.head.train(train)
		self.backbone.eval()  # always frozen / inference mode
		total_loss = correct = count = 0
		with torch.set_grad_enabled(train):
			for xb, yb in loader:
				xb = xb.to(self.device, non_blocking=True)
				yb = yb.to(self.device, non_blocking=True)
				logits = self.__forward(xb)
				loss = self.loss_fn(logits, yb)
				if train:
					self.optimizer.zero_grad()
					loss.backward()
					self.optimizer.step()
				total_loss += loss.item() * xb.size(0)
				correct += (logits.argmax(1) == yb).sum().item()
				count += xb.size(0)
		return total_loss / count, correct / count

	# --- Model interface ---------------------------------------------------
	def fit(self, X_train, y_train, X_test, y_test, nb_epoch, callbacks=[]) -> dict:
		import copy
		train_loader = self.__loader(X_train, y_train, train=True)
		val_loader = self.__loader(X_test, y_test, train=False)
		history = {"accuracy": [], "loss": [], "val_accuracy": [], "val_loss": []}

		best_val_loss = float("inf")
		best_state = copy.deepcopy(self.head.state_dict())
		wait = 0
		for epoch in range(nb_epoch):
			tr_loss, tr_acc = self.__run_epoch(train_loader, train=True)
			va_loss, va_acc = self.__run_epoch(val_loader, train=False)
			history["loss"].append(tr_loss)
			history["accuracy"].append(tr_acc)
			history["val_loss"].append(va_loss)
			history["val_accuracy"].append(va_acc)
			print(f"Epoch {epoch + 1}/{nb_epoch} - loss: {tr_loss:.4f} - accuracy: {tr_acc:.4f}"
				  f" - val_loss: {va_loss:.4f} - val_accuracy: {va_acc:.4f}")
			# EarlyStopping(monitor='val_loss', patience, restore_best_weights=True)
			if va_loss < best_val_loss:
				best_val_loss = va_loss
				best_state = copy.deepcopy(self.head.state_dict())
				wait = 0
			else:
				wait += 1
				if wait >= self.patience:
					print(f"Epoch {epoch + 1}: early stopping")
					break
		self.head.load_state_dict(best_state)
		return history

	def evaluate(self, X_test, y_test) -> None:
		loader = self.__loader(X_test, y_test, train=False)
		loss, acc = self.__run_epoch(loader, train=False)
		print(f"evaluate - loss: {loss:.4f} - accuracy: {acc:.4f}")

	def predict(self, X) -> np.ndarray:
		torch = self._torch
		self.head.eval()
		self.backbone.eval()
		imgs = self.__to_chw01(X)
		preds = []
		with torch.no_grad():
			for start in range(0, imgs.shape[0], TorchModel.BATCH_SIZE):
				batch = imgs[start:start + TorchModel.BATCH_SIZE]
				batch = self._torch.stack([self._normalize(img) for img in batch]).to(self.device)
				preds.append(self.__forward(batch).argmax(1).cpu().numpy())
		return np.concatenate(preds)

	def save(self, path_without_extension) -> str:
		save_path = f"{path_without_extension}.pt"
		self._torch.save({
			"head": self.head.state_dict(),
			"num_classes": self.num_classes,
			"hidden": self.hidden,
			"augment": self.describe_augments(),
		}, save_path)
		return save_path

	def learning_rate(self) -> float:
		return float(self.optimizer.param_groups[0]["lr"])

	def input_shape(self):
		return (224, 224, 3)

	def describe_layers(self) -> list[dict]:
		layers = self.describe_augments()
		layers.append({"type": "Normalize", "mean": list(self._mean), "std": list(self._std)})
		layers.append({
			"type": "MobileNetV2",
			"base_model": "mobilenet_v2",
			"size": sum(p.numel() for p in self.backbone.parameters()),
			"trainable": False,
		})
		layers.append({"type": "GlobalAveragePooling2D"})
		for h in self.hidden:
			layers.append({"type": "Dense", "size": h})
		layers.append({"type": "Dropout", "rate": TorchModel.DROPOUT})
		layers.append({"type": "Dense", "size": self.num_classes})
		return layers
