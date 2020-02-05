# -*- coding: utf-8 -*-
"""
@author: ZHANG Min, Wuhan University
@email: 007zhangmin@whu.edu.cn
"""

import caffe
from caffe import layers as L, params as P
import os


class FDCNN(object):
    def __init__(self, fusion_layer_channel):
        self.ff_channel = fusion_layer_channel
        self.dim = 224

    def conv(
            self,
            bottom,
            ks,
            nout,
            stride=1,
            pad=0,
            name_w="",
            name_b="",
            lr_mult_w=1,
            lr_mult_b=1,
            decay_mult=1,
            bias_term=True):
        conv_ = L.Convolution(bottom,
                              kernel_size=ks,
                              stride=stride,
                              num_output=nout,
                              pad=pad,
                              bias_term=bias_term,
                              param=[{'name': name_w,
                                      'lr_mult': lr_mult_w,
                                      'decay_mult': decay_mult},
                                     {'name': name_b,
                                      'lr_mult': lr_mult_b,
                                      'decay_mult': decay_mult}],
                              weight_filler={'type': 'xavier'},
                              bias_filler={'type': 'constant'})
        return conv_

    def conv_sigmoid(self, bottom, ks, nout, stride=1, pad=0):
        conv_ = self.conv(bottom, ks, nout, stride, pad)
        return conv_, L.Sigmoid(conv_, in_place=True)

    def relu(self, bottom):
        return L.ReLU(bottom, in_place=True)

    def upsample(self, bottom, scale):
        return L.Python(bottom, ntop=1,
                        module='resize_layer',
                        layer='ResizeLayer',
                        param_str=str(dict(scale_w=scale, scale_h=scale)))

    def featurediff(self, bottom1, bottom2):
        return L.Python(bottom1, bottom2, ntop=1,
                        module='feature_difference_layer',
                        layer='FeatureDifferenceLayer')

    def concat(self, b1, b2, b3, b4):
        return L.Concat(b1, b2, b3, b4, concat_param={"axis": 1})

    def pooling(self, bottom, ks=2, stride=2):
        return L.Pooling(
            bottom,
            ntop=1,
            pool=P.Pooling.MAX,
            kernel_size=ks,
            stride=stride)

    def model(self, phase="train"):
        n = caffe.NetSpec()

        n.data = L.Input(
            input_param={
                'shape': {
                    'dim': [
                        1,
                        3,
                        self.dim,
                        self.dim]}})
        n.data_p = L.Input(
            input_param={
                'shape': {
                    'dim': [
                        1,
                        3,
                        self.dim,
                        self.dim]}})
        n.data_t12 = L.Input(
            input_param={
                'shape': {
                    'dim': [
                        1,
                        3,
                        self.dim,
                        self.dim]}})

        n.conv1_1 = self.conv(n.data, 3, 64, stride=1, pad=1,
                              name_w="conv1_1_w", name_b="conv1_1_b",
                              lr_mult_w=0, lr_mult_b=0, decay_mult=0)
        n.relu1_1 = self.relu(n.conv1_1)

        n.conv1_2 = self.conv(n.relu1_1, 3, 64, stride=1, pad=1,
                              name_w="conv1_2_w", name_b="conv1_2_b",
                              lr_mult_w=0, lr_mult_b=0, decay_mult=0)
        n.relu1_2 = self.relu(n.conv1_2)

        n.pool1 = self.pooling(n.relu1_2)

        n.conv2_1 = self.conv(n.pool1, 3, 128, stride=1, pad=1,
                              name_w="conv2_1_w", name_b="conv2_1_b",
                              lr_mult_w=0, lr_mult_b=0, decay_mult=0)
        n.relu2_1 = self.relu(n.conv2_1)

        n.conv2_2 = self.conv(n.relu2_1, 3, 128, stride=1, pad=1,
                              name_w="conv2_2_w", name_b="conv2_2_b",
                              lr_mult_w=0, lr_mult_b=0, decay_mult=0)
        n.relu2_2 = self.relu(n.conv2_2)

        n.pool2 = self.pooling(n.relu2_2)

        n.conv3_1 = self.conv(n.pool2, 3, 256, stride=1, pad=1,
                              name_w="conv3_1_w", name_b="conv3_2_b",
                              lr_mult_w=0, lr_mult_b=0, decay_mult=0)
        n.relu3_1 = self.relu(n.conv3_1)

        n.conv3_2 = self.conv(n.relu3_1, 3, 256, stride=1, pad=1,
                              name_w="conv3_2_w", name_b="conv3_2_b",
                              lr_mult_w=0, lr_mult_b=0, decay_mult=0)
        n.relu3_2 = self.relu(n.conv3_2)

        n.conv3_3 = self.conv(n.relu3_2, 3, 256, stride=1, pad=1,
                              name_w="conv3_3_w", name_b="conv3_3_b",
                              lr_mult_w=0, lr_mult_b=0, decay_mult=0)
        n.relu3_3 = self.relu(n.conv3_3)

        # share weight
        n.conv1_1_p = self.conv(n.data_p, 3, 64, stride=1, pad=1,
                                name_w="conv1_1_w", name_b="conv1_1_b",
                                lr_mult_w=0, lr_mult_b=0, decay_mult=0)
        n.relu1_1_p = self.relu(n.conv1_1_p)

        n.conv1_2_p = self.conv(n.relu1_1_p, 3, 64, stride=1, pad=1,
                                name_w="conv1_2_w", name_b="conv1_2_b",
                                lr_mult_w=0, lr_mult_b=0, decay_mult=0)
        n.relu1_2_p = self.relu(n.conv1_2_p)

        n.pool1_p = self.pooling(n.relu1_2_p)

        n.conv2_1_p = self.conv(n.pool1_p, 3, 128, stride=1, pad=1,
                                name_w="conv2_1_w", name_b="conv2_1_b",
                                lr_mult_w=0, lr_mult_b=0, decay_mult=0)
        n.relu2_1_p = self.relu(n.conv2_1_p)

        n.conv2_2_p = self.conv(n.relu2_1_p, 3, 128, stride=1, pad=1,
                                name_w="conv2_2_w", name_b="conv2_2_b",
                                lr_mult_w=0, lr_mult_b=0, decay_mult=0)
        n.relu2_2_p = self.relu(n.conv2_2_p)

        n.pool2_p = self.pooling(n.relu2_2_p)

        n.conv3_1_p = self.conv(n.pool2_p, 3, 256, stride=1, pad=1,
                                name_w="conv3_1_w", name_b="conv3_2_b",
                                lr_mult_w=0, lr_mult_b=0, decay_mult=0)
        n.relu3_1_p = self.relu(n.conv3_1_p)

        n.conv3_2_p = self.conv(n.relu3_1_p, 3, 256, stride=1, pad=1,
                                name_w="conv3_2_w", name_b="conv3_2_b",
                                lr_mult_w=0, lr_mult_b=0, decay_mult=0)
        n.relu3_2_p = self.relu(n.conv3_2_p)

        n.conv3_3_p = self.conv(n.relu3_2_p, 3, 256, stride=1, pad=1,
                                name_w="conv3_3_w", name_b="conv3_3_b",
                                lr_mult_w=0, lr_mult_b=0, decay_mult=0)
        n.relu3_3_p = self.relu(n.conv3_3_p)

        n.fd_1 = self.featurediff(n.relu1_2, n.relu1_2_p)
        n.fd_2 = self.featurediff(n.relu2_2, n.relu2_2_p)
        n.fd_3 = self.featurediff(n.relu3_3, n.relu3_3_p)

        n.up_2 = self.upsample(n.fd_2, 2)
        n.up_3 = self.upsample(n.fd_3, 4)

        n.concat_1 = self.concat(n.data_t12, n.fd_1, n.up_2, n.up_3)
        n.conv_t = self.conv(n.concat_1, 3, self.ff_channel, stride=1, pad=1,
                             name_w="conv_t_w", name_b="conv_t_b",
                             lr_mult_w=1, lr_mult_b=1,
                             decay_mult=1, bias_term=True)
        n.conv_prob = self.conv(n.conv_t, 1, 1, stride=1, pad=0,
                                name_w="conv_prob_w", name_b="conv_prob_b",
                                lr_mult_w=1, lr_mult_b=1,
                                decay_mult=1, bias_term=True)

        n.sig = L.Sigmoid(n.conv_prob, in_place=False)

        return str(n.to_proto())

    def to_proto(self, path, phase="train"):
        with open(os.path.join(path), 'w') as f:
            f.write(self.model(phase))
