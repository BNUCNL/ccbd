import os
from bba.config import ConfigManager
from bba.model_pipeline import BrainBehaviorModel
import numpy as np

# Step 1: Check for the configuration template file
ConfigManager.check_config_exists()
config_path = os.path.join(os.getcwd(), "hyperparameters.json")

# Step 2: Read the configuration file
config = ConfigManager.read_config(config_path)

# Step 3: Build the model
# Initialize the model class
model_pipeline = BrainBehaviorModel()
model_pipeline.build_model(config)  # build_model() now accepts the configuration dictionary as a parameter

# Step 4: Prepare data and fit the model
X = np.random.rand(10, 59412)  # Example feature data
y = np.random.rand(10, 1)      # Example target data
model_pipeline.fit(X, y)

# Step 5: Evaluate model performance
evaluation_results = model_pipeline.evaluate(X, y)
print("Evaluation results:", evaluation_results)

# Step 6: Save results to a CIFTI dscalar file
# Provide a path to the CIFTI template file
model_pipeline.save_results(config["cifti_template"], config["output_path"])
