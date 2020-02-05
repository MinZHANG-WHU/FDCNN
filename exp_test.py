# -*- coding: utf-8 -*-
"""
@author: ZHANG Min, Wuhan University
@email: 007zhangmin@whu.edu.cn
"""
import caffe
import numpy as np
from PIL import Image
import os
from sklearn import cluster
import time


class clock(object):
    def start(self):
        self.t0 = time.clock()

    def end(self):
        d = time.clock() - self.t0
        print(d)
        return d


def caffe_net(model_def, model_weights):
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(model_def,
                    model_weights,
                    caffe.TEST)
    return net


def pad_edge(im, new_w, new_h, bf=0):
    [h, w, c] = im.shape
    new_im = np.zeros([new_h, new_w, c], dtype=np.float32)
    new_im[bf:h + bf, bf:w + bf, :] = im
    return new_im


def un_pad_edge(im, old_w, old_h, bf=0):
    new_im = im[bf:old_h + bf, bf:old_w + bf]
    return new_im


def hist_match(source, template):
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)
    s_quantiles = np.cumsum(s_counts).astype(np.float32)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float32)
    t_quantiles /= t_quantiles[-1]
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    return interp_t_values[bin_idx].reshape(oldshape)


def block_fdcnn(net, t1_b, t2_b, data_t12, mean_rgb):
    t1 = t1_b - mean_rgb
    t2 = t2_b - mean_rgb

    t1 = t1.transpose((2, 0, 1))
    t1 = t1[np.newaxis, ...]

    t2 = t2.transpose((2, 0, 1))
    t2 = t2[np.newaxis, ...]

    data_t12 = data_t12.transpose((2, 0, 1))
    data_t12 = data_t12[np.newaxis, ...]

    net.blobs['data'].data[...] = t1
    net.blobs['data_p'].data[...] = t2
    net.blobs['data_t12'].data[...] = data_t12

    net.forward()
    cmm = net.blobs['sig'].data[0]
    cmm = cmm.transpose((1, 2, 0))
    return cmm


def block_inceptionv3(net, t1_b, t2_b, mean_rgb):
    t1 = t1_b - mean_rgb
    t2 = t2_b - mean_rgb

    t1 = t1.transpose((2, 0, 1))
    t1 = t1[np.newaxis, ...]

    t2 = t2.transpose((2, 0, 1))
    t2 = t2[np.newaxis, ...]

    net.blobs['data'].data[...] = t1
    net.blobs['data_p'].data[...] = t2

    net.forward()
    cmm = net.blobs['sig'].data[0]
    cmm = cmm.transpose((1, 2, 0))
    return cmm


def block_siamese(net, t1_b, t2_b, mean_rgb):
    t1 = t1_b - mean_rgb
    t2 = t2_b - mean_rgb

    t1 = t1.transpose((2, 0, 1))
    t1 = t1[np.newaxis, ...]

    t2 = t2.transpose((2, 0, 1))
    t2 = t2[np.newaxis, ...]

    net.blobs['data'].data[...] = t1
    net.blobs['data_p'].data[...] = t2

    net.forward()
    t1_data = net.blobs["conv1_5"].data
    t2_data = net.blobs["conv1_5_p"].data
    di = t1_data - t2_data
    dist_sq = np.sum(di ** 2, axis=1, keepdims=True)
    dist = np.sqrt(dist_sq)
    return dist


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_im(im, path):
    im = np.asarray(im, dtype=np.double)
    maxV = np.max(im)
    im = im * 255 / maxV
    im = np.asarray(im, dtype=np.uint8)
    im = Image.fromarray(im)
    im.save(path)


def kmeans(cmm):
    h, w = cmm.shape
    X = cmm.reshape(h * w, 1)
    k_means = cluster.KMeans(n_clusters=2, max_iter=1000)
    labels = k_means.fit_predict(X)
    labels = np.array(labels, dtype=np.uint32)
    labels = labels.reshape(h, w)
    all_count = h * w
    n = np.count_nonzero(labels == 0)
    if n > (all_count / 2.0):
        labels[labels == 0] = 0
        labels[labels == 1] = 255
    else:
        labels[labels == 0] = 255
        labels[labels == 1] = 0
    return labels


def di_threshold(cmm, alpha):
    mean = np.mean(cmm)
    t = alpha * mean
    print 'threshold:', t
    return cmm > t


def acc_evaluation_pixel(result, gt):
    """
                     GT:Changed, Unchanged
      Predicted-Changed:  TP   ,     FP    , b1
    Predicted-Unchanged:  FN   ,     TN    , b2
                          a1   ,     a2
    """
    result_ = np.array(result, dtype=np.uint8)
    result_[result_ >= 0.5] = 1
    gt_ = np.array(gt, dtype=np.uint8)
    gt_[gt_ != 1] = 0
    all_size = gt_.size
    tp = np.count_nonzero((gt_ == result_) & (gt_ > 0))
    tn = np.count_nonzero((gt_ == result_) & (gt_ == 0))
    fp = np.count_nonzero(gt_ < result_)
    fn = np.count_nonzero(gt_ > result_)
    a1 = changes = tp + fn
    a2 = unchanges = fp + tn
    b1 = tp + fp
    b2 = fn + tn
    misdetection = fn * 1.0 / changes
    falsealarms = fp * 1.0 / unchanges
    overallerror = (fp + fn) * 1.0 / all_size
    accuray = (tp + tn) * 1.0 / all_size
    p0 = accuray
    pe = (a1 * b1 + a2 * b2) * 1.0 / (all_size * all_size)
    kappa = (p0 - pe) / (1 - pe)
    print "--------------Accuracy---------------"
    print("     false alarms(FA): {0:.2f} %".format(falsealarms * 100))
    print("     misdetection(MD): {0:.2f} %".format(misdetection * 100))
    print("    overall error(OE): {0:.2f} %".format(overallerror * 100))
    print("             kappa(K): {0:.2f}".format(kappa))
    print "-------------------------------------"
