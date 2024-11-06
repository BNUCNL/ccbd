import numpy as np
from bba import get_supported_models, generate_model_config, load_config, create_model_from_config
import nibabel as nib
from nibabel import cifti2
from sklearn.metrics import r2_score
import subprocess
import os

np.random.seed(42)

# 模拟数据
N = 100  # 被试数量
num_voxels = 59412  # 体素数量
X = np.random.rand(N, num_voxels)  # 模拟脑成像数据 (N x 59412)
Y = np.random.rand(N, 1)           # 模拟单一行为指标 (N x 1)

model_names = get_supported_models()
print(model_names)
# 1. 生成模型配置文件
model_name = input("请输入模型名称(如LinearRegression):")
generate_model_config(model_name)
try:
    subprocess.call(['vim', 'model_config.json'])
except FileNotFoundError:
    print(f"无法打开编辑器 '{editor}'，请手动编辑 'model_config.json' 文件。")
input("Press Enter to continue...")  # 等待用户修改配置文件

# 2. 加载配置文件并创建模型
config = load_config("model_config.json")
model = create_model_from_config(config)

# 3. 拟合数据并计算 R²
model.fit(X, Y)  # 使用 X 拟合 Y
predictions = model.predict(X)  # 用 X 进行预测
r2 = r2_score(Y, predictions)  # 计算 R²
print(f"模型的 R² 值: {r2:.4f}")

# 4. 提取每个体素的回归系数
coef = model.coef_

# 5. 保存回归系数为 CIFTI 的 dscalar 文件
# 加载 CIFTI 模板文件
template_path = 'Yeo_7Network.dscalar.nii'  # 请确保提供一个 CIFTI 模板文件
template_cifti = nib.load(template_path)  # 加载整个 CIFTI 文件
cifti_header = template_cifti.header  # 提取头部信息

# 创建 CIFTI 图像
cifti_image = cifti2.Cifti2Image(coef, header=cifti_header, nifti_header=template_cifti.nifti_header)

output_path = 'brain_behavior_association_single_behavior.dscalar.nii'
nib.save(cifti_image, output_path)
print(f"回归系数已保存到 {output_path}")
