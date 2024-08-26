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

## Citation

```
@article{huang2024deep,
  title={Deep Instruction Tuning for Segment Anything Model},
  author={Huang, Xiaorui and Luo, Gen and Zhu, Chaoyang and Tong, Bo and Zhou, Yiyi and Sun, Xiaoshuai and Ji, Rongrong},
  journal={arXiv preprint arXiv:2404.00650},
  year={2024}
}
```

## Acknowledgement

 Thanks a lot for the nicely organized code from the following repos
- [Segment Anything](https:////github.com/facebookresearch/segment-anything/)

