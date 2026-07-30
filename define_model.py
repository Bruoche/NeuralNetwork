import combination
from trainer.train import train

# Central entry point for both engines. 
# Switch "trainer" to "tensorflow" or "pytorch" to switch framework; 
train(
	trainer="tensorflow",
	major_version=0,       # Used to differenciate model batches when needed
	dataset="objects150",  # Folder in /datasets (untracked)
	dimensions=(224, 224), # Image dimensions
	seed=67,
	patience=10,
	nb_epoch=1,
	depth=combination.Options([  
		[128],
		[64, 64],
		[64],
		[128, 128]
	]),
	learning_rate=combination.Options([0.0003, 0.0001]),
	flip=combination.Options([0.5]),                    # horizontal flip probability
	rotation=combination.Options([0.15]),               # fraction of a full turn (~54 deg)
	translation=combination.Options([(0.1, 0.1)]),      # (height, width) fraction
	scale=combination.Options([0.1]),                   # +/- zoom fraction
	flip_vertical=combination.Options([0]),             # vertical flip probability (off)
	fine_tune=combination.Options([4, 5, 3]),           # top backbone blocks to unfreeze
	base=combination.Options(["efficientnet_b0"]),
)
