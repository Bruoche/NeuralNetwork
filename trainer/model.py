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
				 device=None, patience=5, fine_tune=0, fine_tune_lr=None):
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
		# fine_tune = number of trailing MobileNetV2 feature blocks to unfreeze
		# and train (0 = pure transfer learning, backbone fully frozen). The
		# unfrozen weights train at fine_tune_lr, defaulting to lr/10 (a low
		# rate avoids wrecking the pretrained features on a small dataset).
		self.fine_tune = fine_tune
		self.fine_tune_lr = fine_tune_lr if fine_tune_lr is not None else lr / 10

		weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
		base = models.mobilenet_v2(weights=weights)
		self.backbone = base.features                 # conv feature extractor
		for p in self.backbone.parameters():
			p.requires_grad = False                    # phase 1: backbone fully frozen
		# backbone stays in eval() at all times so its BatchNorm running stats
		# remain frozen even while fine-tuning (matches keras base(training=False))
		self.backbone.eval()
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

		# phase 1 optimizer trains the head only; fine-tuning (phase 2) rebuilds
		# it to also include the unfrozen backbone blocks (see fit / __enable_fine_tuning)
		self.optimizer = torch.optim.Adam(self.head.parameters(), lr=lr)
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
	def __backbone_trainable(self):
		return any(p.requires_grad for p in self.backbone.parameters())

	def __snapshot(self):
		import copy
		# snapshot the backbone too once it has unfrozen (fine-tuning) weights,
		# so restoring the best epoch keeps head and backbone in sync
		state = {"head": copy.deepcopy(self.head.state_dict())}
		if self.__backbone_trainable():
			state["backbone"] = copy.deepcopy(self.backbone.state_dict())
		return state

	def __restore(self, state):
		self.head.load_state_dict(state["head"])
		if "backbone" in state:
			self.backbone.load_state_dict(state["backbone"])

	def __enable_fine_tuning(self):
		# unfreeze the top `fine_tune` blocks and rebuild the optimizer so the
		# head keeps training at lr while the backbone trains at fine_tune_lr
		blocks = list(self.backbone.children())
		for block in blocks[len(blocks) - self.fine_tune:]:
			for p in block.parameters():
				p.requires_grad = True
		param_groups = [
			{"params": self.head.parameters(), "lr": self.lr},
			{"params": [p for p in self.backbone.parameters() if p.requires_grad],
			 "lr": self.fine_tune_lr},
		]
		self.optimizer = self._torch.optim.Adam(param_groups)

	def __train_phase(self, label, train_loader, val_loader, history, nb_epoch,
					  best_val_loss, best_state):
		# one training phase with EarlyStopping(monitor=val_loss, patience,
		# restore_best_weights). best_val_loss/best_state carry across phases so
		# phase 2 can only improve on phase 1, never regress.
		wait = 0
		for epoch in range(nb_epoch):
			tr_loss, tr_acc = self.__run_epoch(train_loader, train=True)
			va_loss, va_acc = self.__run_epoch(val_loader, train=False)
			history["loss"].append(tr_loss)
			history["accuracy"].append(tr_acc)
			history["val_loss"].append(va_loss)
			history["val_accuracy"].append(va_acc)
			print(f"[{label}] Epoch {len(history['loss'])} - loss: {tr_loss:.4f} - accuracy: {tr_acc:.4f}"
				  f" - val_loss: {va_loss:.4f} - val_accuracy: {va_acc:.4f}")
			if va_loss < best_val_loss:
				best_val_loss = va_loss
				best_state = self.__snapshot()
				wait = 0
			else:
				wait += 1
				if wait >= self.patience:
					print(f"[{label}] Epoch {epoch + 1}: early stopping")
					break
		return best_val_loss, best_state

	def fit(self, X_train, y_train, X_test, y_test, nb_epoch, callbacks=[]) -> dict:
		train_loader = self.__loader(X_train, y_train, train=True)
		val_loader = self.__loader(X_test, y_test, train=False)
		history = {"accuracy": [], "loss": [], "val_accuracy": [], "val_loss": []}

		# Phase 1: backbone frozen, train the head to its best (protects the
		# pretrained features from the large gradients of a fresh random head).
		best_val_loss, best_state = self.__train_phase(
			"frozen", train_loader, val_loader, history, nb_epoch, float("inf"), None)
		self.__restore(best_state)

		# Phase 2: unfreeze the top blocks and fine-tune at a lower lr, starting
		# from (and never losing) the best phase-1 weights.
		if self.fine_tune > 0:
			self.__enable_fine_tuning()
			best_val_loss, best_state = self.__train_phase(
				"fine-tune", train_loader, val_loader, history, nb_epoch,
				best_val_loss, self.__snapshot())

		self.__restore(best_state)
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
		data = {
			"head": self.head.state_dict(),
			"num_classes": self.num_classes,
			"hidden": self.hidden,
			"augment": self.describe_augments(),
			"fine_tune": self.fine_tune,
		}
		if self.fine_tune > 0:  # unfrozen backbone weights changed, keep them
			data["backbone"] = self.backbone.state_dict()
		self._torch.save(data, save_path)
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
			"trainable": self.fine_tune > 0,
			"fine_tune_blocks": self.fine_tune,
			"fine_tune_lr": self.fine_tune_lr if self.fine_tune > 0 else None,
		})
		layers.append({"type": "GlobalAveragePooling2D"})
		for h in self.hidden:
			layers.append({"type": "Dense", "size": h})
		layers.append({"type": "Dropout", "rate": TorchModel.DROPOUT})
		layers.append({"type": "Dense", "size": self.num_classes})
		return layers
