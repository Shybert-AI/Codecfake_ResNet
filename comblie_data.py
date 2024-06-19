# -*- coding:utf-8 -*-
# 合并数据特征到新目录下
import os.path
import torch

data1 = "D:\mywork\preprocess_xls-r-5"
data2 = "preprocess_xls-r-5_wav2vec2-large-xlsr-53-chinese-zh-cn"
data3 = "preprocess_xls-r-5_mms-lid-4017_1280"
data_new = "preprocess_xls-r-5_mms-comblie"

from glob import iglob
import tqdm

path_list = list(iglob(f"{data1}/**/*.pt",recursive=True))
for data1_i in tqdm.tqdm(path_list):
    data2_i = os.path.dirname(data1_i).replace("preprocess_xls-r-5",data2) + "/chinese-wav2vec2-base_"+os.path.basename(data1_i)
    data3_i = os.path.dirname(data1_i).replace("preprocess_xls-r-5", data3) + "/chinese-wav2vec2-base_" + os.path.basename(data1_i)
    featureTensor1 = torch.load(data1_i)
    featureTensor2 = torch.load(data2_i)
    featureTensor3 = torch.load(data3_i)
    wav2vec2 = torch.concat([featureTensor1,featureTensor2,featureTensor3],dim=2)
    new_path = data1_i.replace("preprocess_xls-r-5",data_new)
    os.makedirs(os.path.dirname(new_path),exist_ok=True)
    torch.save(wav2vec2.float(),new_path)