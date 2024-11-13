# bba/__init__.py

# 导入包中的核心类和方法
from .config import ConfigManager
from .model_pipeline import BrainBehaviorModel

# 定义包的公共接口
__all__ = ["ConfigManager", "BrainBehaviorModel"]