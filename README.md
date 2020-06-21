# FDCNN

This repository contains code, network definitions and pre-trained models for change detection on remote sensing images using deep learning.

The implementation uses the [Caffe](https://github.com/BVLC/caffe) framework.

## Motivation


In this work, we use scene-level samples (from the AID data set) of remote sensing scene classification, which are easier to obtain than pixel-level samples, for learning deep features from different remote sensing scenes at different scales. These features learned from specific scenes (cultivated land, lakes, vegetation, etc.) are more affected. The changes in these scenes are usually more important. Based on this idea, A new CNN structure and training strategies are proposed for remote sensing image change detection, which is supervised but requires very few pixel-level training samples. Advantageously, it has good generalization ability and multi-scale change detection capabilities.


## Content

###  Networks

We provide a deep neural network based on the [VGG16 architecture](https://arxiv.org/abs/1409.1556). It was trained on the AID dataset to learn the multi-scale deep features from remote sensing images. The pre-trained weights can be download from the [link](https://drive.google.com/open?id=1mAH0Hj9qi2M4GzVaNKe9xJkyeYMf2TLO).

We proposed a novel FDCNN to produce change detection maps from high-resolution RS images. It is suitable for **multi-scale** remote sensing image change detection tasks.


### Datasets

The available datasets can be downloaded from the table below:

<table>
<caption>Tabel 1. Experiment datasets.</caption>
	<tr>
	    <th width="15%">Datasets</th>
	    <th>Description</th>
	    <th width="30%" colspan="2" >Download</th>
	</tr>
    <tr>
	    <td>AID</td>
        <td>10,000 RS images (R, G and B), including 30 different scene types (i.e. labeled 30 types at scene-level), each containing more than 220 images with a size of 600×600 pixels and a spatial resolution of 8 meters to 0.5 meters, collected in different countries (China, USA, UK, France, etc.), at different times and in different imaging conditions</td>
        <td colspan="2">[<a href="http://www.lmars.whu.edu.cn/xia/AID-project.html" target="_blank">official</a>]</td>
	</tr>
    <tr>
	    <td rowspan = "4">Worldview 2 </td>
        <td rowspan = "4">including 2 pilot sites, and each site consists of a ground truth map (labeled changed and unchanged at pixel-level) and two-period Worldview 2 satellite images (Worldview 3 and WV3 were incorrectly written in our paper), located in Shenzhen, China, with a size of 1431×1431 pixels and a spatial resolution of 2 meters, acquired in 2010 and 2015 respectively.</td>
        <td>Site 1 (RGB)</td>
        <td>[<a href="https://drive.google.com/open?id=1ES5bALNZcS5AwiLZKuZW-aBNYiac80Nz" target="_blank">drive</a>]</td>
	</tr>
    <tr>
	    <td>Site 1 (4 bands) </td>
        <td>[<a href="https://drive.google.com/open?id=1FGircw3RANRM1L6hauZJ6iRXWSIVvoly" target="_blank">drive</a>]</td>
    </tr>
    <tr>
	    <td>Site 2 (RGB)</td>
        <td>[<a href="https://drive.google.com/open?id=1x9RdNV6AQpSYHeY0amjVJZWEng-IQpXm" target="_blank">drive</a>]</td>
    </tr>
    <tr>
	    <td>Site 2 (4 bands)</td>
        <td>[<a href="https://drive.google.com/open?id=12HS3eD0iDpqRb9qwR-QEG5qbqea-k9J_" target="_blank">drive</a>]</td>
    </tr>
    <tr>
	    <td>Zi-Yuan 3</td>
        <td>including a ground truth map (labeled changed and unchanged at pixel-level) and two-period Zi-Yuan 3 satellite images, located in Wuhan, Hubei, China, with a size of 458×559 pixels, three bands (R, G and B), and a spatial resolution of 5.8 meters, acquired in 2014 and 2016 respectively.</td>
        <td colspan="2">[<a href="https://drive.google.com/open?id=1yoVP5xc4dPA2sDwYIgVkMEjwCz7q9AdC" target="_blank">drive</a>]</td>
	</tr>
    <tr>
	    <td>Quickbird</td>
        <td>including a ground truth map (labeled changed and unchanged at pixel-level) and two-period Quickbird satellite images, located in Wuhan, Hubei, China, with a size of 1154×740 pixels, three bands (R, G and B), and a spatial resolution of 2.4 meters, acquired in 2009 and 2014 respectively.</td>
        <td colspan="2">[<a href="https://drive.google.com/open?id=1XuiNtqOtH0rQQq-LvVvY9YNt4qIaGEB0" target="_blank">drive</a>]</td>
	</tr>
    <tr>
	    <td>OSCD</td>
        <td> 10 test pairs RS images with a spatial resolution of 10 meters, taken from the Sentinel-2 satellites between 2015 and 2018 with pixel-level change ground truth. Their ground truth remains undisclosed and the results need be uploaded to <a href="http://dase.grss-ieee.org/" target="_blank">the IEEE GRSS DASE website</a> for evaluation</td>
        <td colspan="2">[<a href="https://drive.google.com/open?id=19g9V8LaZhLfmfDeEDpOtrz17cdpJP0sh" target="_blank">drive</a>] [<a href="https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection" target="_blank">official</a>]</td>
	</tr>
    <tr>
	    <td>SZADA/1</td>
        <td>a pair of optical aerial images, labeled changed and unchanged at pixel-level, taken with several years of time differences, with a spatial resolution 1.5 meters.</td>
        <td colspan="2">[<a href="https://drive.google.com/open?id=1nFSqjfen5pY6uQz7lXRDWUig-LtMHRcN" target="_blank">drive</a>] [<a href="http://web.eee.sztaki.hu/remotesensing/airchange_benchmark.html" target="_blank">official</a>]</td>
	</tr>
    
</table> 

### How to start

1. Install Caffe with Python 2.7

	1. Follow the instructions in [Installation](http://caffe.berkeleyvision.org/installation.html). Note the version of Python, or use our [pre-build runtime](https://drive.google.com/open?id=1OLIgpx0Jy6LT0KCkgYLcb0d3FvAJXEA0) (with CUDA 8.0 and for Windows only).
	2. Please add the absolute path of folder "caffe_layers" to the PYTHONPATH so that PyCaffe can search for the layer implementation file.

2. Training VGG16 & FDCNN

    1. Training VGG16 using the AID dataset.
   
    2. Training FDCNN using the WV2 site 1 dataset.


3. Testing FDCNN

    1. Download the test data sets and unzip them to the "datasets"  subfolder.
    
    2. Using your own trained FDCNN model, or download our [pre-trained FDCNN model](https://drive.google.com/open?id=1v1Q9gOqgzk657aaPWfEirSR-aJafF7BS).
    
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
            The ground truth of OSCD remains undisclosed and the results need be uploaded to [the IEEE GRSS DASE website](http://dase.grss-ieee.org/) for evaluation, see figure 1.
            ![](/output/OSCD.png)
            <center>Figure 1. FDCNN accuracy evaluation on the OSCD dataset.</center>
    
    Change magnitude map (CMM.tif) and binary image (BM.tif) will be generated under the "output" subfolder.

## References

If you use this work for your projects, please take the time to cite our paper.

```
@Article{9052762,
AUTHOR = {Zhang, Min and Shi, Wenzhong},
TITLE = {A Feature Difference Convolutional Neural Network-Based Change Detection Method},
JOURNAL = {IEEE Transactions on Geoscience and Remote Sensing},
VOLUME = {},
YEAR = {2020},
NUMBER = {},
URL = {https://ieeexplore.ieee.org/document/9052762},
DOI = {10.1109/TGRS.2020.2981051}
}
```

## License

Code and datasets are released under the GPLv3 license for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.


