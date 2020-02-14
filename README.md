# FDCNN

This repository contains code, network definitions and pre-trained models for change detection on remote sensing images using deep learning.

The implementation uses the [Caffe](https://github.com/BVLC/caffe) framework.

## Motivation


In this work, we use scene-level samples (from the AID data set) of remote sensing scene classification, which are easier to obtain than pixel-level samples, for learning deep features from different remote sensing scenes. These features learned from specific scenes (cultivated land, lakes, vegetation, etc.) are more affected. The changes in these scenes are usually more important. Based on this idea, A new CNN structure and training strategies are proposed for remote sensing image change detection, which is supervised but requires very few pixel-level training samples.


## Content

###  Networks

We provide a deep neural network based on the [VGG16 architecture](https://arxiv.org/abs/1409.1556). It was trained on the AID dataset to learn the deep features from remote sensing images. The pre-trained weights can be download from the [link](https://drive.google.com/file/d/1mAH0Hj9qi2M4GzVaNKe9xJkyeYMf2TLO/view?usp=sharing).

We proposed a novel FDCNN to produce change detection maps from high-resolution RS images.


### Datasets

We will open all data sets after the paper is published. The available datasets can be downloaded from the table below:

| Datasets|  Download |
|--------|--------|
| AID    |  [[official](http://www.lmars.whu.edu.cn/xia/AID-project.html)]|
| WV3_1  |  [drive]|
| WV3_2  |  [drive]|
| ZY3    |  [[drive](https://drive.google.com/file/d/1yoVP5xc4dPA2sDwYIgVkMEjwCz7q9AdC/view?usp=sharing)]|
| QB     |  [drive]|
| OSCD   |  [[drive](https://drive.google.com/file/d/19g9V8LaZhLfmfDeEDpOtrz17cdpJP0sh/view?usp=sharing)][[official](https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection)]|
| SZADA/1|  [[drive](https://drive.google.com/file/d/1nFSqjfen5pY6uQz7lXRDWUig-LtMHRcN/view?usp=sharing)][[official](http://web.eee.sztaki.hu/remotesensing/airchange_benchmark.html)] |


### How to start

1. Install Caffe with Python 2.7

	Follow the instructions in [Installation](http://caffe.berkeleyvision.org/installation.html). Note the version of Python.

2. Training VGG16 & FDCNN

    We will make it public after the paper is published.
  

3. Testing FDCNN

    1. Download the test data sets and unzip them to the "datasets"  subfolder.
    
    2. Using your own trained FDCNN model, or download our pre-trained FDCNN model.
    
    3. Evaluation
    
        - To test the accuracy of FDCNN on the test datasets, run the following commands below:
            ```
            python exp_test_custom.py \
                --sensor=ZY3 \
                --alpha=2.0
            ```

        - To test the accuracy of FDCNN on the SZTAKI datasets, run the following commands below:
            ```
            python exp_test_SZTAKI.py \
                --alpha=2.66
            ```
        
        - To test the accuracy of FDCNN on the OSCD datasets, run the following commands below:
            ```
            python exp_test_OSCD.py \
                --threshold=0.98
            ```
            The ground truth of OSCD remains undisclosed and the results need be uploaded to [the IEEE GRSS DASE website](http://dase.grss-ieee.org/) for evaluation.
    
    Change magnitude map (CMM.tif) and binary image (BM.tif) will be generated under the "output" subfolder.
    

### Accuracy
 
 TODO



## References

If you use this work for your projects, please take the time to cite our paper.



## License

Code and datasets are released under the GPLv3 license for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.


