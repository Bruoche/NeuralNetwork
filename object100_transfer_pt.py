import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

from trainer.monitor import MonitorModel
from trainer.model import TorchModel
import combination

SEED = 67
torch.manual_seed(SEED)
np.random.seed(SEED)
DIMENSIONS = (224, 224)

resize = transforms.Resize(DIMENSIONS)
dataset = ImageFolder("datasets/objects100")
class_names = dataset.classes

images = np.stack([
	np.asarray(resize(img.convert("RGB")), dtype="float32")
	for img, _ in dataset
])
labels = np.asarray([label for _, label in dataset], dtype="int64")

indices = np.random.default_rng(SEED).permutation(len(labels))
val_count = int(0.2 * len(labels))
test_idx, train_idx = indices[:val_count], indices[val_count:]
X_train, y_train = images[train_idx], labels[train_idx]
X_test,  y_test  = images[test_idx],  labels[test_idx]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
	print(torch.cuda.get_device_name())

optionalDepth = combination.Options([
	[64],
	[128],
	[256],
	[512]
])
optionalFlip = combination.Options([[], [transforms.RandomHorizontalFlip()]])
optionalFlipVertical = combination.Options([[], [transforms.RandomVerticalFlip()]])
optionalRotation = combination.Options([[], [transforms.RandomRotation(0.15 * 360)]])
optionalTranslation = combination.Options([(0.0, 0.0), (0.1, 0.1)])
optionalScale = combination.Options([(1.0, 1.0), (0.9, 1.1)])
learning_rates = combination.Options([0.001, 0.0007, 0.0005])
iterator = combination.Iterator([optionalDepth, learning_rates, optionalFlip, optionalRotation, optionalTranslation, optionalScale, optionalFlipVertical])
for i in iterator:
	print(f"Doing combination {i}/{iterator.nb_combinations()}")

	model = TorchModel(
		num_classes=len(class_names),
		augment=[transforms.RandomAffine(degrees=0, translate=optionalTranslation.get(), scale=optionalScale.get())]
			+ optionalFlip.get()
			+ optionalRotation.get()
			+ optionalFlipVertical.get(),
		hidden=optionalDepth.get()+optionalDepth.get(),
		lr=learning_rates.get(),
		device=device,
		patience=10
	)

	monitor = MonitorModel("object100_pt_mobilenet_v2", 1)
	monitor.train(
		model, X_train, y_train, X_test, y_test,
		nb_epoch=300,
		class_names=class_names,
		callbacks=["EarlyStopping(monitor=val_loss, patience=5, restore_best_weights=True)"]
	)
