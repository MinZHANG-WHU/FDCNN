# -*- coding: utf-8 -*-
"""
@author: ZHANG Min, Wuhan University
@email: 007zhangmin@whu.edu.cn
"""

import caffe
import numpy as np


class FeatureDifferenceLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need two inputs")

    def reshape(self, bottom, top):
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        top[0].reshape(*bottom[0].data.shape)

    def backward(self, top, propagate_down, bottom):
        # print "backward"
        if not propagate_down[0]:
            return

        bottom[0].diff[...] = self.diff

    def forward(self, bottom, top):
        # print "forward"
        fmap1 = bottom[0].data
        fmap2 = bottom[1].data

        di = np.abs(fmap1 - fmap2)
        maxV = np.max(di)
        di = di / maxV  # [0,1]
        top[0].data[...] = di
