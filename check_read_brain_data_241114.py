#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 02:07:23 2024

@author: zzl-zrl
"""

import os
import pandas as pd
import nibabel as nib
import numpy as np

def check_brain_input(subject_id_list, all_subject_brain_file_path):
    """
    检查脑部数据输入的函数。

    该函数从CSV文件中读取脑部数据，检查指定的被试ID是否存在于数据文件中，并验证其路径的有效性。

    参数:
    ----------
    subject_id_list : list
        包含被试ID的列表，用于指定要检查的数据项。
    all_subject_brain_file_path : str
        CSV文件的路径，其中包含所有被试的编号以及文件路径
        该文件路径必须以".csv"结尾。

    返回:
    ----------
    tuple
        (exist_subject, not_exist_subject)
        exist_subject：list，包含路径存在的被试ID。
        not_exist_subject：list，包含路径不存在或数据缺失的被试ID。

    异常:
    ----------
    ValueError
        当输入的文件格式错误时，会触发该异常。
    FileNotFoundError
        读取CSV文件或访问路径时出错，会触发该异常。

    """
    if not all_subject_brain_file_path.lower().endswith('.csv'):
        print("输入文件格式错误")
        return [], []
    
    try:
        all_subject_brain_data = pd.read_csv(all_subject_brain_file_path)
    except Exception as e:
        print(f"读取CSV文件时出错：{e}")
        return [], []
    
    chosen_subject_brain_data = all_subject_brain_data[all_subject_brain_data['subject'].isin(subject_id_list)]

    exist_subject = []
    not_exist_subject = []

    for subject in subject_id_list:
        subject_brain_data_path = chosen_subject_brain_data[chosen_subject_brain_data['subject'] == subject]['path']
        if not subject_brain_data_path.empty:
            subject_brain_data_path = subject_brain_data_path.iloc[0]  # 取出该被试的路径
            if os.path.exists(subject_brain_data_path):
                exist_subject.append(subject)
            else:
                not_exist_subject.append(subject)
                print(f"被试{subject}的文件路径异常")
        else:
            not_exist_subject.append(subject)
            print(f"被试{subject}的数据在文件中未找到")

    return exist_subject, not_exist_subject

def concatenate_brain_input(exist_subject, all_subject_brain_file_path):
    """
    拼接脑部数据的函数。

    该函数从指定的CSV文件中读取存在的被试ID的脑部数据路径，加载对应的脑部数据文件，并将所有数据进行垂直拼接。

    参数:
    ----------
    exist_subject : list
        包含存在脑部数据路径的被试ID的列表。
    all_subject_brain_file_path : str
        CSV文件的路径，其中包含所有被试的的编号以及文件路径

    返回:
    ----------
    np.ndarray
        拼接后的脑部数据数组，每个被试的脑部数据按行拼接形成一个二维数组。

    异常:
    ----------
    FileNotFoundError
        当指定的脑部数据文件路径不存在时，触发该异常。
    IOError
        当无法读取脑部数据文件时，触发该异常。

    """
    concatenated_data = []
    all_subject_brain_data = pd.read_csv(all_subject_brain_file_path)

    for subject in exist_subject:
        exist_subject_brain_data_path = all_subject_brain_data[all_subject_brain_data['subject'] == subject]['path'].iloc[0]
        exist_subject_brain_data = nib.load(exist_subject_brain_data_path).get_fdata()
        concatenated_data.append(exist_subject_brain_data)

    concatenated_data = np.vstack(concatenated_data)
    return concatenated_data

#check
import pandas as pd
   
sub_all_list = pd.read_csv("/nfs/z1/zhenlab/CCBD/brain_behavior_association_tools/data/test_myelin_data.csv")["subject"].tolist()
sub_check2 = check_brain_input(sub_all_list,"/nfs/z1/zhenlab/CCBD/brain_behavior_association_tools/data/test_myelin_data.csv")
matrix = concatenate_brain_input(sub_check2[0],"/nfs/z1/zhenlab/CCBD/brain_behavior_association_tools/data/test_myelin_data.csv")
