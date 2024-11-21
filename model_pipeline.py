import json
import os
import nibabel as nib
import numpy as np
from nibabel import cifti2
from sklearn.metrics import r2_score
import importlib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class BrainBehaviorModel:
    def __init__(self):
        """
        Initializes the BrainBehaviorModel class without requiring a configuration dictionary.
        """
        self.model = None
        self.results = None
        # self.scaler = None
        # self.pca = None
        # self.preprocessing_config = None
        self.data_type = None
        self.data_level = None

    def build_model(self, config: dict):
        """
                Builds the model based on the provided configuration dictionary.

                Parameters
                ----------
                config : dict
                    A dictionary containing the model name, parameters, and preprocessing settings.

                Raises
                ------
                ValueError
                    If the specified model class cannot be found within the sklearn module.
                 """
        # self.preprocessing_config = config.get("preprocessing", {})
        #
        # # Handle z-score normalization
        # if self.preprocessing_config.get("z_score", False):
        #     self.scaler = StandardScaler()
        #
        # # Handle PCA
        # if self.preprocessing_config.get("pca", {}).get("enabled", False):
        #     n_components = self.preprocessing_config["pca"].get("n_components", None)
        #     self.pca = PCA(n_components=n_components)
        self.data_type = config['data_type']
        self.data_level = config['data_level']

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

    # def preprocess_data(self, X):
    #     """
    #     Applies preprocessing to the input data, including z-score normalization and PCA.
    #
    #     Parameters
    #     ----------
    #     X : array-like
    #         The input data to preprocess.
    #
    #     Returns
    #     -------
    #     array-like
    #         Preprocessed data.
    #     """
    #     # Z-score normalization
    #     if self.scaler:
    #         print("Applying z-score normalization...")
    #         X = self.scaler.fit_transform(X)
    #
    #     # PCA
    #     if self.pca:
    #         print("Applying PCA...")
    #         X = self.pca.fit_transform(X)
    #
    #     return X

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

    def save_results(self, template_path, output_path):
        """
        保存结果到 NIfTI 或 CIFTI 文件。
        """

        coef = self.model.coef_.T

        if self.data_type == "nifti":
            self._save_nifti(coef, template_path, output_path)
        elif self.data_type == "cifti":
            self._save_cifti(coef, template_path, output_path)
        else:
            raise ValueError(f"Unsupported file type: {self.data_type}")

    def _save_nifti(self, betas, template_path, output_path):
        """
        保存 NIfTI 结果。
        """

        # Step 1: 读取模板文件
        template_img = nib.load(template_path)  # 加载模板文件
        template_data = template_img.get_fdata()  # 获取数据数组
        affine = template_img.affine  # 获取仿射矩阵
        header = template_img.header  # 获取头信息
        if self.data_level == "voxel":
            betas_3d = betas.reshape(template_img.shape)
            output_img = nib.Nifti1Image(betas_3d, affine, header)
            nib.save(output_img, output_path)
            # To do
        elif self.data_level == "roi":

            # Step 2: 创建一个新的数据数组，用于保存贝塔值
            output_data = np.zeros_like(template_data)  # 初始化一个与模板形状相同的数组

            # Step 3: 将贝塔值映射到对应的脑区
            for roi_id in range(1, 181):  # ROI ID从1到180
                output_data[template_data == roi_id] = betas[roi_id - 1]

            # Step 4: 保存新的NIfTI文件
            output_img = nib.Nifti1Image(output_data, affine, header)
            nib.save(output_img, output_path)

        print(f"结果已保存到: {output_path}")

    def _save_cifti(self, betas, template_path, output_path):
        """
        保存 CIFTI 结果。
        """
        template_img = nib.load(template_path)
        img_header = template_img.header

        if self.data_level == "voxel":

            # Create the CIFTI image
            cifti_image = cifti2.Cifti2Image(betas, header=img_header, nifti_header=template_img.nifti_header)

            # Save the CIFTI image
            nib.save(cifti_image, output_path)
        elif self.data_level == "roi":
            '''
            To do
            '''
        print(f"结果已保存到: {output_path}")

    # def save_results(self, template_file_path: str, output_file_path: str, output_type: str):
    #     """
    #     Saves the model's coefficients as a CIFTI dscalar file.
    #
    #     Parameters
    #     ----------
    #     template_file_path : str
    #         The path to the CIFTI template file used to create the CIFTI image.
    #     output_file_path : str
    #         The path where the results will be saved.
    #
    #     Raises
    #     ------
    #     ValueError
    #         If the model has not been fitted or there are no coefficients to save.
    #     """
    #     if self.results is None or "coefficients" not in self.results:
    #         raise ValueError("No results available to save. Please fit the model first.")
    #
    #     coef = self.model.coef_.T
    #
    #     print(coef.shape)
    #
    #
    #
    #     if output_type == "cifiti":
    #         # Load the CIFTI template file
    #         template_cifti = nib.load(template_file_path)
    #         cifti_header = template_cifti.header
    #
    #         # Create the CIFTI image
    #         cifti_image = cifti2.Cifti2Image(coef, header=cifti_header, nifti_header=template_cifti.nifti_header)
    #
    #         # Save the CIFTI image
    #         nib.save(cifti_image, output_file_path)
    #         print(f"Model coefficients saved to {output_file_path}")
    #     elif output_type == "nifiti":
    #         # Step 1: 读取模板文件
    #         atlas_img = nib.load(template_file_path)  # 加载模板文件
    #         atlas_data = atlas_img.get_fdata()  # 获取数据数组
    #         affine = atlas_img.affine  # 获取仿射矩阵
    #         header = atlas_img.header  # 获取头信息
    #
    #         # Step 2: 创建一个新的数据数组，用于保存贝塔值
    #         output_data = np.zeros_like(atlas_data)  # 初始化一个与模板形状相同的数组
    #
    #         # Step 3: 将贝塔值映射到对应的脑区
    #         for roi_id in range(1, 181):  # ROI ID从1到180
    #             output_data[atlas_data == roi_id] = coef[roi_id - 1]
    #
    #         # Step 4: 保存新的NIfTI文件
    #         output_img = nib.Nifti1Image(output_data, affine, header)
    #         nib.save(output_img, output_file_path)
    #
    #         print(f"结果已保存到: {output_file_path}")
