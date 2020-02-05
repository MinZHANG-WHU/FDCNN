# -*- coding: utf-8 -*-
"""
@author: ZHANG Min, Wuhan University
@email: 007zhangmin@whu.edu.cn
"""

import caffe
import numpy as np
import cv2


class ResizeLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 1:
            raise Exception("Need one inputs")
        params = eval(self.param_str)
        self.scale_w = params['scale_w']
        self.scale_h = params['scale_h']

    def reshape(self, bottom, top):
        [n, c, h, w] = bottom[0].data.shape
        new_w = int(w * self.scale_w)
        new_h = int(h * self.scale_h)
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        top[0].reshape(n, c, new_h, new_w)

    def backward(self, top, propagate_down, bottom):
        # print "backward"
        if not propagate_down[0]:
            return
        bottom[0].diff[...] = self.diff

    def forward(self, bottom, top):
        data = bottom[0].data
        [n, c, h, w] = data.shape
        new_w = int(w * self.scale_w)
        new_h = int(h * self.scale_h)
        for i in range(n):
            for j in range(c):
                f = data[i, j, :, :]
                f = np.reshape(f, [h, w])
                # print f.shape;
                f = cv2.resize(f, dsize=(new_h, new_w))
                f = np.reshape(f, [1, 1, new_h, new_w])
                # print f.shape;
                top[0].data[i, j, :, :] = f
