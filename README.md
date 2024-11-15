# DIT

[![Python](https://img.shields.io/badge/python-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-%237732a8)

<p align="center">
	<img src="./DIT.png" width="550">
</p>
This is the official implementation of "Deep Instruction Tuning for Segment Anything Model", which propose two simple yet effective deep instruction tuning (DIT) methods for text-guided SAM.

## News

- **2024.07.16: Our work has been accepted as poster by ACM MM 2024.**



## Installation
```
pip install -r requirements.txt
wget https://github.com/explosion/spacy-models/releases/download/en_vectors_web_lg-2.1.0/en_vectors_web_lg-2.1.0.tar.gz -O en_vectors_web_lg-2.1.0.tar.gz
pip install en_vectors_web_lg-2.1.0.tar.gz
```


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

## Model Weights
Following the steps of Data preparation and Training, you can reproduce and get better results in our paper. We provide the model weights for RefCOCO, RefCOCO+, RefCOCOg and GRES. 

1. RefCOCO [Download link](https://drive.google.com/file/d/11gVgwnWI8c0m54gZFJIEcyYMzN7u9lmF/view?usp=sharing)

| val               | test A            | test B            |
| - | - | -|
| 76.2 | 77.85  | 73.53|


2. RefCOCO+ [Download link](https://drive.google.com/file/d/1T3jYiR9BLDJxvThYulySrDv0S2qonq1Z/view?usp=sharing
| val               | test A            | test B            |
| - | - | -|
| 65.94 | 69.78  | 58.89|

3. RefCOCOg [Download link](https://drive.google.com/file/d/1HObuOQLv97NB3eD2X4ss2XsBlAifTc_L/view?usp=sharing)

| val               | test             | 
| - | - | 
| 67.4 | 68.07  | 

4. GRES [Download link](https://drive.google.com/file/d/1v9dcxKwOQM8i2NKXi4YfITJ9ZO00XVur/view?usp=sharing)

| val               | test A            | test B            |
| - | - | -|
| 63.76 | 67.19  | 61.85|


## Citation

```
@inproceedings{
	huang2024deep,
	title={Deep Instruction Tuning for Segment Anything Model},
	author={Xiaorui Huang and Gen Luo and Chaoyang Zhu and Bo Tong and Yiyi Zhou and Xiaoshuai Sun and Rongrong Ji},
	booktitle={ACM Multimedia 2024},
	year={2024}
}
```

## Acknowledgement

 Thanks a lot for the nicely organized code from the following repos
- [Segment Anything](https:////github.com/facebookresearch/segment-anything/)

