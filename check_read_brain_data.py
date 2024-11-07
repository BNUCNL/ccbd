#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 02:21:03 2024

@author: zzl-zrl
"""

import os
import numpy as np
import nibabel as nib

def check_brain_input(sublist,HCPYA_dir):
    exist_sub = []
    not_exist_sub = []
    for sub in sublist:
        myeli_path = os.path.join(HCPYA_dir,str(sub),'MNINonLinear','fsaverage_LR32k',str(sub)+'.MyelinMap_MSMAll.32k_fs_LR.dscalar.nii')
        if os.path.exists(myeli_path):
            exist_sub.append(sub)
        else:
            not_exist_sub.append(sub)
            print(f"Subject {sub} does not have myelination data")
    return exist_sub,not_exist_sub

def concatenate_brain_input(exist_sub,HCPYA_dir):
    concatenated_data = [] 
    for sub in exist_sub:
        myeli_path = os.path.join(HCPYA_dir,str(sub),'MNINonLinear','fsaverage_LR32k',str(sub)+'.MyelinMap_MSMAll.32k_fs_LR.dscalar.nii')
        myeli_data = nib.load(myeli_path).get_fdata()
        concatenated_data.append(myeli_data)
    concatenated_data = np.vstack(concatenated_data)
    return concatenated_data

#check
import pandas as pd
hcp_dir = "/nfs/z1/HCP/HCPYA/sorted_data/"      
sub_all_list = pd.read_csv("/nfs/z1/zhenlab/CCBD/brain_behavior_association_tools/data/test_myelin_data.csv")["subject"].tolist()
sub_check2 = check_brain_input(sub_all_list,hcp_dir)
matrix = concatenate_brain_input(sub_check2[0],hcp_dir)

        