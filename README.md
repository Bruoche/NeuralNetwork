# Training framework

This projects aim to offer a framework allowing training and monitoring models precisely via many helper methods working for both tensorflow and pytorch plugins.

The entrypoint for the project is define_model.py
Here, simply fill in the informations with the desired values.

The combination.Options([...]) in this file serves to choose all the possible values to try training the model with. The combination Iterator will then make all possible iterations with all given values on all Options given to it. This will allow us to make models using all combinations of potentially vallable value. The objective being to allow us to test all intersections of values, as sometimes modifications made in isolation lead to opposite outcome then their combined forces, most notably different combinations of learning_rates depending on the shape of the model.
This util allows us then to choose a selection of plausible value and have all possible combinations tested in a row without manual launch, allowing many short runs to be done consecutively independantly overnight.

For the training itself, this is done in the /trainer package, with train.py having the common training logic for both tensorflow and pytorch, and their specific behavior in their respective Trainer classes. Then monitor.py will do the monitoring and versionning of the resulting models.

WARNING: Datasets are untracked, due to their weight, so you have to import your own dataset. Simply add a folder in /datasets. The sub-folder name is the dataset's name, in this dataset each consecutive sub-folder will be the label for all the images it contains. 

# Monitoring
Monitoring utils are in the /analysis package, with:
- compare_models_training.py (model_name, max): Plots the value of val_loss through epochs during training for all matching models, specifying a max value will cap the results to the n best perfoming models (according to their max val_loss). "model_name" accepts wildcards, such as "object150*_b0_00" to match any model name such as "object150_tf_effectivenet_b0_00_12". No model_name given correspond to selecting all.
- confusion_matrix.py (model_name): Show the confusion matrix for the specified model. Usefull to spot what classes cause trouble to the model.
- model_fitness.py (model_name): Show the comparison of training and test values on both fitness metrics (loss and accuracy) through each epoch. Is the best way to identify issues in the training by clearly showing cases of overfitting or other strange behavior from the model.
- parameter_impact.py (model_name): Selects all the models matching model_name (all if none specified), and compare all the differing parameters among them and their performance, then returns the performance difference between each values of each parameter. Allows identifying trends among models after a batch of training.
~ fitness_per_time.py (model_name): Show a bar graph of best performing models per time of training. The lowest bar being best efficience. This is a rather unreliable metric tho, it is not advised to use it for decision making and is more so informative.

These scripts use the /monitoring artifacts to make all the required calculations. These artifacts being made by trainer.monitor.MonitorModel.

# Using models

Each trained model is saved in models/{model_name}/{model_name}_{run_num}.{ext}.

# Versionning

Models being too large to be committed, and to ensure that the versionning is thread-safe, the versionning is made via .claim files in the /versions folder.
These files must be commited to the repo when collaborating, and must never be deleted to ensure there are no conflicts when versionning models.