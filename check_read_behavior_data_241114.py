#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:36:36 2024

@author: zzl-zrl
"""

import os
import numpy as np
import pandas as pd

def check_behavior_input(behav_file_path):
    """
    检查行为数据输入的函数。

    该函数用于检查指定的行为数据文件是否为CSV格式，并检查文件中是否存在缺失值。

    参数:
    ----------
    behav_file_path : str
        行为数据文件的路径，文件必须为CSV格式。

    返回:
    ----------
    None
        该函数没有返回值，仅在控制台输出检查结果信息。

    异常:
    ----------
    ValueError
        当文件格式错误时触发该异常。
    数据缺失警告
        当文件中存在缺失值时触发该警告。

    """
    if behav_file_path.lower().endswith('.csv'):
        data = pd.read_csv(behav_file_path)
        if data.isnull().values.any():
            print("文件中包含缺失值，请进行修改。")
        else:
            print("文件数据无异常，可继续执行下一步。")
    else:
        print("文件格式错误，请输入CSV文件。")

        
def read_behavior_input(behav_file_path_checked):
    """
    读取行为数据输入的函数。

    该函数用于读取已检查的行为数据CSV文件，并将数据加载为DataFrame格式。

    参数:
    ----------
    behav_file_path_checked : str
        已检查过格式和内容的行为数据文件路径，应为CSV文件。

    返回:
    ----------
    pd.DataFrame
        包含行为数据的DataFrame对象。

    异常:
    ----------
    FileNotFoundError
        当指定的文件路径不存在时触发该异常。
    IOError
        当文件无法读取时触发该异常。

    """
    data = pd.read_csv(behav_file_path_checked)
    print("数据已正常读取。")
    return data

    
check = check_behavior_input("/nfs/z1/zhenlab/CCBD/brain_behavior_association_tools/data/test_behavior_data.csv")   
check_data =   read_behavior_input("/nfs/z1/zhenlab/CCBD/brain_behavior_association_tools/data/test_behavior_data.csv")
