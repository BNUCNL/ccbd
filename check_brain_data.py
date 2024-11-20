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

    详细描述:
    ----------
    根据脑影像数据的文件格式动态调整拼接维度，支持以下格式：
    - NIFTI 文件（.nii 或 .nii.gz）：将每个被试的数据增加一个新的维度，拼接为 4D 或 5D 数据。
    - CIFTI 文件（如 .dscalar.nii, .dtseries.nii, .dlabel.nii）：拼接为 2D 数据（行表示被试，列表示每个时间点或脑部位置）。
    - GIFTI 文件（.gii）：拼接为 2D 数据（行表示被试，列表示每个脑部位置）。

    此函数会根据 CSV 文件中列出的被试 ID 和路径，加载存在的脑影像数据文件并进行拼接。

    注意:
    1. CSV 文件中所有被试的脑影像文件路径必须有效。
    2. 所有被试的脑影像文件格式必须一致（如全部是 .nii 或 .gii）。
    3. 此函数假设所有被试的脑影像文件具有相同的维度和结构。

    参数:
    ----------
    exist_subject : list
        list 类型
        包含存在脑影像数据路径的被试 ID 的列表，仅处理这些被试的数据。
    all_subject_brain_file_path : str
        str 类型
        CSV 文件的路径，文件中需包含以下两列：
        - 'subject': 被试的 ID。
        - 'path': 每个被试对应脑影像文件的路径。

    返回:
    ----------
    np.ndarray
        NumPy 数组类型
        拼接后的脑影像数据数组，其形状取决于文件格式和拼接逻辑：
        - NIFTI 文件：返回一个 4D 或 5D 的数组。
        - CIFTI 和 GIFTI 文件：返回一个 2D 的数组。

    异常:
    ----------
    ValueError
        如果文件格式不一致。
        如果某个被试的文件路径在 CSV 文件中不存在。
        如果加载的文件格式不受支持。
    FileNotFoundError
        如果指定的文件路径不存在。

    """
    concatenated_data = []
    all_subject_brain_data = pd.read_csv(all_subject_brain_file_path)

    # 提取路径中的文件扩展名
    all_file_paths = all_subject_brain_data['path']
    file_formats = set([os.path.splitext(path)[1] for path in all_file_paths])

    # 判断文件格式是否一致
    if len(file_formats) > 1:
        raise ValueError("文件格式不一致。")
    
    # 确定文件格式
    file_format = list(file_formats)[0]
    
    # 根据文件格式加载数据
    for subject in exist_subject:
        subject_row = all_subject_brain_data[all_subject_brain_data['subject'] == subject]
        
        if subject_row.empty:
            raise ValueError(f"被试 {subject} 的路径不存在于 CSV 文件中。")
        
        brain_data_path = subject_row['path'].iloc[0]
        
        if not os.path.exists(brain_data_path):
            raise FileNotFoundError(f"文件 {brain_data_path} 不存在。")
        
        if file_format == '.nii' or file_format == '.nii.gz':  
            brain_data = nib.load(brain_data_path).get_fdata()
            concatenated_data.append(brain_data[np.newaxis, ...])  
        elif file_format == '.gii':  
            gifti_data = nib.load(brain_data_path)
            brain_data = np.hstack([arr.data for arr in gifti_data.darrays])  
            concatenated_data.append(brain_data)  
        elif file_format.endswith('.dscalar.nii') or file_format.endswith('.dtseries.nii') or file_format.endswith('.dlabel.nii'):  
            brain_data = nib.load(brain_data_path).get_fdata()
            concatenated_data.append(brain_data)  
        else:
            raise ValueError(f"不支持的文件格式：{file_format}")
    
    if file_format == '.nii' or file_format == '.nii.gz':  
        concatenated_data = np.concatenate(concatenated_data, axis=0)  
    else:  
        concatenated_data = np.vstack(concatenated_data)  

    return concatenated_data

#check   
sub_all_list = pd.read_csv("/nfs/z1/zhenlab/CCBD/brain_behavior_association_tools/data/test_myelin_data.csv")["subject"].tolist()
sub_check2 = check_brain_input(sub_all_list,"/nfs/z1/zhenlab/CCBD/brain_behavior_association_tools/data/test_myelin_data.csv")
matrix = concatenate_brain_input(sub_check2[0],"/nfs/z1/zhenlab/CCBD/brain_behavior_association_tools/data/test_myelin_data.csv")



