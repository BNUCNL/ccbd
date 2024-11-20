# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:32:19 2024

@author: masai
"""

import os
import pandas as pd
import nibabel as nib
import numpy as np



#%%
# 检查输入的行为数据
def check_behavior_input(behavior_file_path):
    """

    """
    behavior_input = pd.read_csv(behavior_file_path)
    if behavior_input.isnull().values.any():
        raise ValueError("行为数据中包含缺失值，请进行修改")
    else:
        print("行为数据检查完毕")

# 检查输入的影像数据
def check_brain_input(brain_file_path):
    """

    """
    brain_input = pd.read_csv(brain_file_path)
    # 遍历每一行，检查路径是否存在
    for index, row in brain_input.iterrows():
        subject = row['subject']
        path = row['path']
        # 如果路径不存在，print被试
        if not os.path.exists(path):
            print(f"文件不存在，被试: {subject}, 路径: {path}")
    print("影像数据检查完毕")

#%%
# 将所有被试的影像数据拼接
def concatenate_brain_input(brain_file_path, output_file_path):
    """
    
    """
    brain_input = pd.read_csv(brain_file_path)
    # 初始化一个空列表用于存储每个被试的数据
    all_subject_data = []
    # 遍历每一行，读取影像文件
    for index, row in brain_input.iterrows():
        subject = row['subject']
        path = row['path']
        # 读取影像文件
        if path.endswith('.nii.gz') or path.endswith('.nii'):
            brain_data = nib.load(path).get_fdata()  # NIfTI 文件
        elif path.endswith('.func.gii') or path.endswith('.shape.gii'):
            brain_data = nib.load(path).darrays[0]  # GIFTI 文件
        elif path.endswith('.dscalar.nii'):
            brain_data = nib.load(path).get_fdata()  # CIFTI 文件
        # 将高维数据展平为一维向量
        if brain_data.ndim >= 2:
            brain_data = brain_data.reshape(-1)
        all_subject_data.append(brain_data)
    # 将所有被试的数据拼接为一个矩阵
    concatenated_matrix = np.vstack(all_subject_data)  # 被试数 x 顶点/体素数的矩阵
    # 保存为 .npy 文件
    np.save(output_file_path, concatenated_matrix)
    print(f"影像数据拼接完成，并保存为文件：{output_file_path}")

#%%
# 将体素/顶点级数据转化为ROI级
def transform_analysis_level(concatenate_brain_path, brain_atlas_path, output_file_path):
    """
    将顶点级数据矩阵转换为 ROI 级别矩阵，并保存为 CSV 文件。

    参数：
    ----------
    concatenate_brain_path : str
        拼接好的被试顶点级数据矩阵的路径（.npy 文件）。
    brain_atlas_path : str
        脑图谱文件的路径（.nii.gz 或类似格式的文件），定义 ROI 区域。
    output_csv_path : str
        转换后保存的 CSV 文件路径。

    返回：
    ----------
    None
    """
    # 加载拼接好的被试顶点级数据矩阵
    brain_data = np.load(concatenate_brain_path)  # shape: (num_subjects, num_vertices)
    
    # 加载脑图谱文件并获取 ROI 标签数据
    atlas = nib.load(brain_atlas_path).get_fdata()  # shape: (num_vertices,)
    roi_labels = atlas.reshape(-1)  # 展平为一维
    
    # 获取 ROI 的唯一编号（排除背景或无效区域，通常编号为 0 的区域为背景）
    unique_rois = np.unique(roi_labels)
    unique_rois = unique_rois[unique_rois > 0]  # 排除背景 ROI（编号为 0）
    
    # 初始化 ROI 级别数据矩阵
    roi_level_data = np.zeros((brain_data.shape[0], len(unique_rois)))  # shape: (num_subjects, num_rois)

    # 遍历每个 ROI，计算该 ROI 内顶点的平均值
    for i, roi in enumerate(unique_rois):
        # 获取当前 ROI 的顶点索引
        roi_indices = np.where(roi_labels == roi)[0]
        
        # 计算该 ROI 内所有顶点的平均值
        roi_level_data[:, i] = brain_data[:, roi_indices].mean(axis=1)  # 每个被试的平均值

    # 将 ROI 级别矩阵保存为 CSV 文件
    roi_df = pd.DataFrame(roi_level_data, columns=[f"ROI_{int(roi)}" for roi in unique_rois])
    roi_df.to_csv(output_csv_path, index=False)
    print(f"ROI 级别数据已保存为 CSV 文件：{output_csv_path}")






















