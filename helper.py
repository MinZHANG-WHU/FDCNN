# -*- coding: utf-8 -*-
"""
@author: ZHANG Min, Wuhan University
@email: 007zhangmin@whu.edu.cn
"""

import os
import numpy as np
from PIL import Image


def get_custom_path(sensor):
    base_dir = 'datasets'
    t1 = os.path.join(base_dir, sensor, 't1.bmp')
    t2 = os.path.join(base_dir, sensor, 't2.bmp')
    gt = os.path.join(base_dir, sensor, 'gt.bmp')
    t1 = Image.open(t1)
    t1 = np.asarray(t1, dtype=np.float32)
    t2 = Image.open(t2)
    t2 = np.asarray(t2, dtype=np.float32)
    gt = Image.open(gt)
    gt = np.asarray(gt, dtype=np.float32)
    gt[gt == 255] = 0
    return [t1, t2, gt]


def get_fdcnn():
    model_def = r"fdcnn\fdcnn_deploy.prototxt"
    model_weights = r"pre_trained\FDCNN.caffemodel"
    return [model_def, model_weights]


def get_inceptionv3():
    model_def = r'inceptionv3\deploy_inception-v3.prototxt'
    model_weights = r'pre_trained\inception-v3.caffemodel'
    return [model_def, model_weights]


def get_siamese():
    model_def = r'siamese_knn\siamese_deploy.prototxt'
    model_weights = r'pre_trained\siamese.caffemodel'
    return [model_def, model_weights]
