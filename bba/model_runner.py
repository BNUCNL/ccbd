# bba/model_runner.py

import json
import importlib
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_config(config_path="model_config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def create_model_from_config(config):
    model_name = config["model_name"]
    parameters = config["parameters"]

    # 动态导入模型类
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
        raise ValueError(f"无法找到模型类 {model_name}")

    # 实例化模型
    model = model_class(**{k: v for k, v in parameters.items() if v is not None})
    return model
