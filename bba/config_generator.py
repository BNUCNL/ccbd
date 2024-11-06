# bba/config_generator.py

import json
import inspect
from sklearn.utils import all_estimators

def get_supported_models():
    # 获取所有分类器和回归器模型
    classifiers = all_estimators(type_filter='classifier')
    regressors = all_estimators(type_filter='regressor')

    # 提取模型名称
    classifier_names = [name for name, _ in classifiers]
    regressor_names = [name for name, _ in regressors]

    return {
        "classifiers": classifier_names,
        "regressors": regressor_names
    }

def generate_model_config(model_name: str, output_path="model_config.json"):
    # 获取所有的模型类
    all_models = dict(all_estimators(type_filter='classifier') + all_estimators(type_filter='regressor'))

    if model_name not in all_models:
        raise ValueError(f"模型名称 {model_name} 无效或不受支持。")

    # 获取模型类并提取初始化参数
    model_class = all_models[model_name]
    params = inspect.signature(model_class.__init__).parameters

    # 生成配置字典
    config = {
        "model_name": model_name,
        "parameters": {k: v.default if v.default != inspect.Parameter.empty else None for k, v in params.items() if k != 'self'}
    }

    # 写入JSON文件
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"模型配置文件已生成，保存路径：{output_path}")
