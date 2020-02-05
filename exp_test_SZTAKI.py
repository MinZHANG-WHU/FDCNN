# -*- coding: utf-8 -*-
"""
@author: ZHANG Min, Wuhan University
@email: 007zhangmin@whu.edu.cn
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import exp_test
import helper
import argparse


def test_fdcnn(alpha):
    # load dataset
    base_dir = r"datasets\SZTAKI"
    image_seleted = r'Szada\1'
    group_dir = os.path.join(base_dir, image_seleted)
    data_t1_path = os.path.join(group_dir, "im1.bmp")
    data_t2_path = os.path.join(group_dir, "im2.bmp")
    label_path = os.path.join(group_dir, "gt.bmp")

    t1 = Image.open(data_t1_path)
    t1 = np.asarray(t1, dtype=np.float32)
    t2 = Image.open(data_t2_path)
    t2 = np.asarray(t2, dtype=np.float32)
    for i in range(3):
        t2[:, :, i] = exp_test.hist_match(t2[:, :, i], t1[:, :, i])
    gt = Image.open(label_path)
    gt = np.asarray(gt, dtype=np.uint8)
    gt = gt.copy()
    gt[gt > 0] = 1

    out_dir = 'output/SZTAKI/' + image_seleted
    exp_test.make_dir(out_dir)

    # parameters
    dim = 224
    bf = 12
    t1 = t1[0:448, 0:784, :]
    t2 = t2[0:448, 0:784, :]
    gt = gt[0:448, 0:784]

    mean_rgb = np.array((101.438, 104.358, 93.970), dtype=np.float32)

    [h, w, c] = t2.shape
    data_t12 = np.abs(t1 - t2)
    maxV = np.max(data_t12)
    data_t12 = data_t12 / maxV

    [model_def, model_weights] = helper.get_fdcnn()
    net = exp_test.caffe_net(model_def, model_weights)

    # Considering the edge
    write_dim = dim - 2 * bf
    h_batch = int(h + write_dim - 1) / write_dim
    w_batch = int(w + write_dim - 1) / write_dim
    new_size = (w_batch * write_dim + 2 * bf, h_batch * write_dim + 2 * bf)
    im1 = exp_test.pad_edge(t1, new_size[0], new_size[1], bf)
    im2 = exp_test.pad_edge(t2, new_size[0], new_size[1], bf)
    im12 = exp_test.pad_edge(data_t12, new_size[0], new_size[1], bf)

    cmm = np.zeros((new_size[1], new_size[0]))

    all_count = h_batch * w_batch
    for i in range(h_batch):
        for j in range(w_batch):
            print "Progress->", all_count
            all_count = all_count - 1
            offset_x = j * write_dim
            offset_y = i * write_dim
            t1_b = im1[offset_y:offset_y + dim, offset_x:offset_x + dim]
            t2_b = im2[offset_y:offset_y + dim, offset_x:offset_x + dim]
            t12_b = im12[offset_y:offset_y + dim, offset_x:offset_x + dim]
            cmm_b = exp_test.block_fdcnn(net, t1_b, t2_b, t12_b, mean_rgb)
            cmm_b = cmm_b.reshape([dim, dim])
            cmm[offset_y + bf:offset_y + bf + write_dim, 
                offset_x + bf:offset_x + bf + write_dim] = cmm_b[bf:bf + write_dim, bf:bf + write_dim]

    cmm = exp_test.un_pad_edge(cmm, w, h, bf)
    bm = exp_test.di_threshold(cmm, alpha)

    exp_test.acc_evaluation_pixel(bm, gt)

    exp_test.save_im(cmm, os.path.join(out_dir, 'CMM.tif'))
    exp_test.save_im(bm, os.path.join(out_dir, 'BM.tif'))

    plt.figure("T1")
    plt.imshow(np.array(t1, dtype=np.uint8))
    plt.show()

    plt.figure("T2")
    plt.imshow(np.array(t2, dtype=np.uint8))
    plt.show()

    plt.figure("GT")
    plt.imshow(np.array(gt, dtype=np.uint8))
    plt.show()

    plt.figure("Binary Map")
    plt.imshow(bm)
    plt.show()

    plt.figure("Change Magnitude Map")
    plt.imshow(cmm)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Test the FDCNN on SZTAKI datasets")
    parser.add_argument('--alpha', '-a', type=float, default=2.66, required=True)
    args = parser.parse_args()
    test_fdcnn(args.alpha)
    print 'Done!'
