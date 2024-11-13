import json
import os
import nibabel as nib
from nibabel import cifti2
from sklearn.metrics import r2_score
import importlib


class BrainBehaviorModel:
    def __init__(self):
        """
        Initializes the BrainBehaviorModel class without requiring a configuration dictionary.
        """
        self.model = None
        self.results = None

    def build_model(self, config: dict):
        """
        Builds the model based on the provided configuration dictionary.

        Parameters
        ----------
        config : dict
            A dictionary containing the model name and parameters.

        Raises
        ------
        ValueError
            If the specified model class cannot be found within the sklearn module.
        """
        model_name = config["model_name"]
        parameters = config["parameters"]

        # Dynamically import the model class from sklearn
        module_path = "sklearn"
        model_class = None
        for submodule in ["linear_model", "svm", "tree", "ensemble", "neighbors", "naive_bayes", "neural_network",
                          "cluster", "decomposition"]:
            try:
                model_class = getattr(importlib.import_module(f"{module_path}.{submodule}"), model_name)
                break
            except AttributeError:
                continue

        if not model_class:
            raise ValueError(f"Model class '{model_name}' not found within sklearn modules.")

        # Instantiate the model with provided parameters, filtering out any None values
        self.model = model_class(**{k: v for k, v in parameters.items() if v is not None})
        print(f"Model built: {self.model}")

    def fit(self, X, y):
        """
        Fits the model using the provided data.

        Parameters
        ----------
        X : array-like
            The feature data to train the model.
        y : array-like
            The target data for model fitting.

        Returns
        -------
        dict
            A dictionary containing model coefficients and intercept (if available).

        Raises
        ------
        ValueError
            If the model has not been built prior to fitting.
        """
        if self.model is None:
            raise ValueError("Please call build_model() to create the model before fitting.")

        self.model.fit(X, y)
        self.results = {
            "coefficients": self.model.coef_.tolist(),
            "intercept": self.model.intercept_.tolist() if hasattr(self.model, 'intercept_') else None
        }
        print("Model has been fitted.")
        return self.results

    def evaluate(self, X, y):
        """
        Evaluates the model's performance by calculating the R² score.

        Parameters
        ----------
        X : array-like
            The feature data used for predictions.
        y : array-like
            The true target data to compare against the model's predictions.

        Returns
        -------
        dict
            A dictionary containing the R² score of the model.

        Raises
        ------
        ValueError
            If the model has not been fitted prior to evaluation.
        """
        if self.model is None:
            raise ValueError("Please call fit() to fit the model before evaluation.")

        y_pred = self.model.predict(X)
        r2 = r2_score(y, y_pred)
        return {"R²": r2}

    def save_results(self, template_file_path: str, output_file_path: str):
        """
        Saves the model's coefficients as a CIFTI dscalar file.

        Parameters
        ----------
        template_file_path : str
            The path to the CIFTI template file used to create the CIFTI image.
        output_file_path : str
            The path where the results will be saved.

        Raises
        ------
        ValueError
            If the model has not been fitted or there are no coefficients to save.
        """
        if self.results is None or "coefficients" not in self.results:
            raise ValueError("No results available to save. Please fit the model first.")

        coef = self.model.coef_

        # Load the CIFTI template file
        template_cifti = nib.load(template_file_path)
        cifti_header = template_cifti.header

        # Create the CIFTI image
        cifti_image = cifti2.Cifti2Image(coef, header=cifti_header, nifti_header=template_cifti.nifti_header)

        # Save the CIFTI image
        nib.save(cifti_image, output_file_path)
        print(f"Model coefficients saved to {output_file_path}")
