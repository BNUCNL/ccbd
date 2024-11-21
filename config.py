import json
import os


class ConfigManager:
    @staticmethod
    def check_config_exists():
        """
        Check if the configuration file "hyperparameters.json" exists in the current working directory.

        If the configuration file is not found, raises a FileNotFoundError with a prompt to create it.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If the configuration file "hyperparameters.json" is not found in the current working directory.
        """
        current_directory = os.getcwd()
        config_file_path = os.path.join(current_directory, "hyperparameters.json")

        if not os.path.isfile(config_file_path):
            raise FileNotFoundError(
                f"Configuration file 'hyperparameters.json' not found. Please create the configuration file in the current working directory: '{current_directory}'"
            )
        print(f"Configuration file found: {config_file_path}")

    @staticmethod
    def read_config(config_file_path: str) -> dict:
        """
        Reads a specified JSON configuration file and returns a configuration dictionary.

        This function accepts a path to a JSON configuration file, loads its contents, and validates its structure.

        Parameters
        ----------
        config_file_path : str
            Path to the configuration file.

        Returns
        -------
        dict
            Parsed configuration dictionary containing model name and parameters.

        Raises
        ------
        FileNotFoundError
            If the specified configuration file is not found.
        JSONDecodeError
            If the configuration file is not in valid JSON format.
        ValueError
            If the configuration file does not meet the expected structure (missing "model_name" or "parameters" fields).
        """
        with open(config_file_path, 'r') as file:
            config_data = json.load(file)

        # Validate the structure of the configuration file
        if "model_name" not in config_data or "parameters" not in config_data:
            raise ValueError("Invalid configuration file format: missing 'model_name' or 'parameters' fields")

        return config_data
