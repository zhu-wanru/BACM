import numpy as np

from tqdm import tqdm
import glob
import os
import sys
import torch
data_path = '/cluster/sc_download/zhuwanru/scannet/val_group'
data_list = sorted(glob.glob(os.path.join(data_path, '*.pth')))
class_label = 7
if not os.path.exists('cls_scannet/'+str(class_label)):
    os.makedirs('cls_scannet/'+str(class_label))
i = 0
for file in tqdm(data_list):
    data = torch.load(file)
    pc = np.array(data[0])
    pc_label = np.array(data[2])
    if class_label in np.unique(pc_label):
        i += 1
        np.savetxt('cls_scannet/'+str(int(class_label))+'/'+str(i)+'.txt', pc[pc_label==class_label])


