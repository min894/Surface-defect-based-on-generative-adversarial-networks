#coding:utf-8
import os
import cv2
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch
from config import configs


f = open('MEMS100/label.txt', 'r')
label_str = f.read().split()
label = list(map(int, label_str))
label = np.array(label)
label = torch.tensor(label.reshape((-1, 1)), dtype=torch.long)


file_lists = os.listdir('MEMS100')
file_lists = file_lists[0:100]
imgs = []
for file in file_lists:
    path = 'MEMS100/' + file
    img = cv2.imread(path)
    img = cv2.resize(img, (configs.imgsize, configs.imgsize))
    img = np.array(img).swapaxes(2, 1)
    img = img.swapaxes(1, 0)
    img = img.reshape((1, 3, configs.imgsize, configs.imgsize))
    imgs.append(img)

data_x = torch.tensor(np.concatenate(imgs, axis=0), dtype=torch.float)


datasets = TensorDataset(data_x, label)





