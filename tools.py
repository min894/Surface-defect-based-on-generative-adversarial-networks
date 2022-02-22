#coding:utf-8
import numpy as np
import cv2
import os


def img_show(x):
    x = np.array(x, dtype=np.uint8)  # (n, 3, 150, 150)
    x = np.swapaxes(x, axis1=1, axis2=2)
    x = np.swapaxes(x, axis1=2, axis2=3)
    x = x.reshape((150, 150, 3))
    x = cv2.resize(x, (600, 600))
    cv2.imshow('1', x)
    cv2.waitKey()

def img_save(x, step):
    path = 'test/' + str(step) + '.jpg'
    x = np.array(x, dtype=np.uint8)  # (n, 3, 150, 150)
    x = np.swapaxes(x, axis1=1, axis2=2)
    x = np.swapaxes(x, axis1=2, axis2=3)
    x = x.reshape((1024, 1024, 3))
    x = cv2.resize(x, (1600, 1600))
    cv2.imwrite(path, x)

def log(step, g_loss, d_loss):
    print(f'step:{step}| g_loss:{g_loss}| d_loss:{d_loss}')


