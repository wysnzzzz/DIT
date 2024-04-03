# DIT

[![Python](https://img.shields.io/badge/python-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-%237732a8)
![MAC-Group](https://img.shields.io/badge/mac-group-orange.svg)](https://mac.xmu.edu.cn/)

<p align="center">
	<img src="./DIT.png" width="550">
</p>
This is the official implementation of "[Deep Instruction Tuning for Segment Anything Model](https://arxiv.org/pdf/2404.00650.pdf)", which propose two simple yet effective deep instruction tuning (DIT) methods for text-guided SAM.





## Updates
- (2024/3/30) Release our DIT project.
## Installation
```
pip install -r requirements.txt
wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz -O en_vectors_web_lg-2.1.0.tar.gz
pip install en_vectors_web_lg-2.1.0.tar.gz
```
## Data preparation

-  Follow the instructions of  [DATA_PRE_README.md](https://github.com/luogen1996/SimREC/blob/main/DATA_PRE_README.md) to generate training data and testing data.
-  Download the pretrained weights of backbone (vgg, darknet, cspdarknet, DResNet, etc.).  Expect for DResNet, all pretrained backbones are trained  on COCO 2014 *train+val*  set while removing the images appeared in the *val+test* sets of RefCOCO, RefCOCO+ and RefCOCOg (nearly 6500 images).  Please follow the instructions of  [DATA_PRE_README.md](https://github.com/luogen1996/SimREC/blob/main/DATA_PRE_README.md) to download them.

## Training and Evaluation 

1. Prepare your settings. To train a model, you should  modify ``./config/config.yaml``  to adjust the settings  you want. 
2. Train the model. run ` train.py`  under the main folder to start training:
```
python train.py --config ./config/config.yaml
```
3. Test the model.   Then, you can run ` test.py`  by
```
python test.py --eval-weights ./logs/dit/1/weights/seg_best.pth
```
4. Training log.  Logs are stored in ``./logs`` directory, which records the detailed training curve and accuracy per epoch. If you want to log the visualizations, please  set  ``LOG_IMAGE`` to ``True`` in ``config.yaml``.   


## Citation

If this repository is helpful for your research, or you want to refer the provided results in your paper, you can cite the corresponding paper:
```
@article{luo2022what,
  title={What Goes beyond Multi-modal Fusion in One-stage Referring Expression Comprehension: An Empirical Study},
  author={Luo, Gen and Zhou, Yiyi and Sun, Jiamu and Huang, Shubin and Sun, Xiaoshuai and Ye, Qixiang and Wu, Yongjian and Ji, Rongrong},
  journal={arXiv preprint arXiv:2204.07913},
  year={2022}
}
```

## Acknowledgement

 Thanks a lot for the nicely organized code from the following repos
- [OpenVQA](https://github.com/MILVLG/openvqa).
- [Segment Anything](https:////github.com/facebookresearch/segment-anything/)

