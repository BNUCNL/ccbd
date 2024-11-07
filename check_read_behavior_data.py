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
    if behav_file_path.lower().endswith('.csv'):
        data = pd.read_csv(behav_file_path)
        if data.isnull().values.any():
            print("Include the missing values in the file, please modify them.")
        else:
            print("No exception occurred in the file data. Continue to the next step.")
    else:
        print("File format error, please enter the csv file.")
        
def read_behavior_input(behav_file_path_checked):
    data = pd.read_csv(behav_file_path_checked)
    print("The data was read in normally.")
    return data
    
check = check_behavior_input("/nfs/z1/zhenlab/CCBD/brain_behavior_association_tools/data/test_behavior_data.csv")   
check_data =   read_behavior_input("/nfs/z1/zhenlab/CCBD/brain_behavior_association_tools/data/test_behavior_data.csv")