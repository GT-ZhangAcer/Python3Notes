import os
from PIL import Image
import cv2 as cv
import numpy
import traceback
import paddle.fluid as fluid
import paddle
import numpy as np
from PIL import Image
import Class_OS.o1_获得当前工作目录
import paddle.dataset.mnist as mnist
# 绘图
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import random


def readIMGInDir(path, type=None, onle_name=False):
    '''
    读取文件夹下所有文件的文件名和路径
    :param path: 路径
    type:指定文件类型，如果没有指定则视为jpg类型
    :return: nameL:文件夹内所有路径+文件名 './trainData/ori1/20181024/000030_1_0.jpg' or '000030_1_0.jpg'
    '''
    if type is None:
        type = '.jpg'
    else:
        type = "." + type
    nameL = []  # 保存文件名
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] == type:
                if onle_name is True:
                    nameL.append(str(file).replace("\\", "/"))
                else:
                    nameL.append(str(os.path.join(root, file)).replace("\\", "/"))
    return nameL