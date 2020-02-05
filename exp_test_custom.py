# -*- coding: utf-8 -*-
"""
@author: ZHANG Min, Wuhan University
@email: 007zhangmin@whu.edu.cn
"""
import matplotlib.pyplot as plt
import os
import exp_test
import helper
import argparse
import numpy as np
np.__version__

def test_fdcnn(sensor, alpha):
    dim = 224
    bf = 12

    [t1, t2, gt] = helper.get_custom_path(sensor)
    group_dir = 'output/' + sensor
    exp_test.make_dir(group_dir)

    for i in range(3):
        t2[:, :, i] = exp_test.hist_match(t2[:, :, i], t1[:, :, i])

    [h, w, c] = t2.shape
    data_t12 = np.abs(t1 - t2)
    maxV = np.max(data_t12)
    data_t12 = data_t12 / maxV

    [model_def, model_weights] = helper.get_fdcnn()
    net = exp_test.caffe_net(model_def, model_weights)
    mean_rgb = np.array((101.438, 104.358, 93.970), dtype=np.float32)

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

    exp_test.save_im(bm, os.path.join(group_dir, 'BM.tif'))
    exp_test.save_im(cmm, os.path.join(group_dir, 'CM.tif'))

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
    parser = argparse.ArgumentParser(description="Test the FDCNN")
    parser.add_argument(
        '--sensor',
        '-s',
        default='ZY3',
        required=True,
        help='enum: WV3_1, WV3_2, QB, ZY3')
    parser.add_argument(
        '--alpha',
        '-a',
        default=2.0,
        type=float,
        required=True,
        help='enum:2.3 for WV3, 2.4 for QB,2.0 for ZY3')
    args = parser.parse_args()
    test_fdcnn(args.sensor, args.alpha)
    print 'Done!'
