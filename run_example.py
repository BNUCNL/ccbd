import os
from config import ConfigManager
from model_pipeline import BrainBehaviorModel
import numpy as np
# from check_brain_data import *
# from check_behavior_data import *

# Step 1: Check for the configuration template file
ConfigManager.check_config_exists()
config_path = os.path.join(os.getcwd(), "hyperparameters.json")

# Step 2: Read the configuration file
config = ConfigManager.read_config(config_path)

# Step 3: Build the model
# Initialize the model class
model_pipeline = BrainBehaviorModel()
model_pipeline.build_model(config)  # build_model() now accepts the configuration dictionary as a parameter

# Step 4: Prepare data
# Example data for testing; replace with your real data

X = np.random.rand(100, 180)  # Example feature data
y = np.random.rand(100, 1)      # Example target data

# Step 5: Fit the model
# If z-score normalization and PCA are enabled in the configuration,
# they will be applied automatically in the fit method
# X_preprocessed = model_pipeline.preprocess_data(X)  # Explicitly preprocess the data (z-score and/or PCA)
model_pipeline.fit(X, y)

# Step 6: Evaluate model performance
# Preprocessing (z-score normalization and PCA) will also be applied
# during evaluation if specified in the configuration
evaluation_results = model_pipeline.evaluate(X, y)
print("Evaluation results:", evaluation_results)

# Step 7: Save results to a CIFTI dscalar file
# Provide a path to the CIFTI template file
model_pipeline.save_results(config["template_path"], config["output_path"])
