# -*- coding: utf-8 -*-
"""
@author: ZHANG Min, Wuhan University
@email: 007zhangmin@whu.edu.cn
"""
import numpy as np
from PIL import Image
import os
import exp_test
import helper
import cv2
import argparse


def test_fdcnn(threshold):
    # load dataset
    base_dir = r"datasets\Onera Satellite Change Detection dataset - Images";
    train_set_path = os.path.join(base_dir, 'test.txt');
    with open(train_set_path, 'r') as f:
        train_imgs = f.read();
    name_list = train_imgs.split(',');

    # parameters
    dim = 224;
    mean_rgb = np.array((101.438, 104.358, 93.970), dtype=np.float32);

    [model_def, model_weights] = helper.get_fdcnn();
    net = exp_test.caffe_net(model_def, model_weights);

    for image_seleted in name_list:
        print 'Current image ->', image_seleted
        group_dir = os.path.join(base_dir, 'Test', image_seleted, "pair");
        data_t1_path = os.path.join(group_dir, "img1.png")
        data_t2_path = os.path.join(group_dir, "img2.png")
        out_dir = 'output/OSCD/Test';
        exp_test.make_dir(out_dir);

        t1 = Image.open(data_t1_path);
        t1 = np.asarray(t1, dtype=np.float32);
        t2 = Image.open(data_t2_path);
        t2 = np.asarray(t2, dtype=np.float32);
        for i in range(3):
            t2[:, :, i] = exp_test.hist_match(t2[:, :, i], t1[:, :, i]);

        [h, w, c] = t2.shape;

        # Spatial resolution: 10m -> 5m, [AID dataset: 0.5m-8m]
        t1 = cv2.resize(t1, (2 * w, 2 * h), interpolation=cv2.INTER_LINEAR);
        t2 = cv2.resize(t2, (2 * w, 2 * h), interpolation=cv2.INTER_LINEAR);

        [h, w, c] = t2.shape;
        data_t12 = np.abs(t1 - t2);
        maxV = np.max(data_t12);
        data_t12 = data_t12 / maxV;

        # Considering the edge
        bf = 20;
        write_dim = dim - 2 * bf;
        h_batch = int(h + write_dim - 1) / write_dim;
        w_batch = int(w + write_dim - 1) / write_dim;
        new_size = (w_batch * write_dim + 2 * bf, h_batch * write_dim + 2 * bf);

        im1 = exp_test.pad_edge(t1, new_size[0], new_size[1], bf);
        im2 = exp_test.pad_edge(t2, new_size[0], new_size[1], bf);
        im12 = exp_test.pad_edge(data_t12, new_size[0], new_size[1], bf);
        cmm = np.zeros((new_size[1], new_size[0]));
        for i in range(h_batch):
            for j in range(w_batch):
                offset_x = j * write_dim;
                offset_y = i * write_dim;
                t1_b = im1[offset_y:offset_y + dim, offset_x:offset_x + dim];
                t2_b = im2[offset_y:offset_y + dim, offset_x:offset_x + dim];
                t12_b = im12[offset_y:offset_y + dim, offset_x:offset_x + dim];

                cmm_b = exp_test.block_fdcnn(net, t1_b, t2_b, t12_b, mean_rgb);
                cmm_b = cmm_b.reshape([dim, dim]);
                cmm[offset_y + bf:offset_y + bf + write_dim,
                    offset_x + bf:offset_x + bf + write_dim] = cmm_b[bf:bf + write_dim, bf:bf + write_dim];
        cmm = exp_test.un_pad_edge(cmm, w, h, bf);
        cmm = cv2.resize(cmm, (w / 2, h / 2), interpolation=cv2.INTER_NEAREST);
        maxV = np.max(cmm);
        cmm = cmm * 1.0 / maxV;

        # You can still improve the accuracy by setting thresholds for
        # different image pairs
        bm = cmm > threshold;
        bm = np.asarray(bm, dtype=np.uint8);
        bm[bm > 0] = 1;
        bm[bm == 0] = 0;
        save_name = image_seleted.replace('\\', '_');
        exp_test.save_im(bm, os.path.join(out_dir, save_name + '_BM.tif'));
        exp_test.save_im(cmm, os.path.join(out_dir, save_name + '_CMM.tif'))
        
    print 'The results need be uploaded to the IEEE GRSS DASE websitefor evaluation.'
    
if __name__ == '__main__':
    parser=argparse.ArgumentParser(description="Test the FDCNN on OSCD datasets")
    parser.add_argument('--threshold','-t', default = 0.98,type=float, required=True, help='between 0 to 1')
    args=parser.parse_args()
    test_fdcnn(args.threshold)
    print 'Done!'
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
