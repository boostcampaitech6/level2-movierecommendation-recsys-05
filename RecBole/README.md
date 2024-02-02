![RecBole Logo](asset/logo.png)

--------------------------------------------------------------------------------

# RecBole

[![PyPi Latest Release](https://img.shields.io/pypi/v/recbole)](https://pypi.org/project/recbole/)
[![Conda Latest Release](https://anaconda.org/aibox/recbole/badges/version.svg)](https://anaconda.org/aibox/recbole)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-RecBole-%23B21B1B)](https://arxiv.org/abs/2011.01731)


## Installation
RecBole works with the following operating systems:

* Linux
* Windows 10
* macOS X

RecBole requires Python version 3.7 or later.

RecBole requires torch version 1.7.0 or later. If you want to use RecBole with GPU,
please ensure that CUDA or cudatoolkit version is 9.2 or later.
This requires NVIDIA driver version >= 396.26 (for Linux) or >= 397.44 (for Windows10).

### Install from conda

```bash
conda install -c aibox recbole
```

### Install from pip

```bash
pip install recbole
```

### Install from source
```bash
git clone https://github.com/RUCAIBox/RecBole.git && cd RecBole
pip install -e . --verbose
```

## Start

```bash
python recbole_dataset.py
python run_recbole.py --model=[model_name]
python run_inference.py --model_path=[saved_model_path]
```

## Cite

```bibtex
@inproceedings{recbole[1.0],
  author    = {Wayne Xin Zhao and Shanlei Mu and Yupeng Hou and Zihan Lin and Yushuo Chen and Xingyu Pan and Kaiyuan Li and Yujie Lu and Hui Wang and Changxin Tian and Yingqian Min and Zhichao Feng and Xinyan Fan and Xu Chen and Pengfei Wang and Wendi Ji and Yaliang Li and Xiaoling Wang and Ji{-}Rong Wen},
  title     = {RecBole: Towards a Unified, Comprehensive and Efficient Framework for Recommendation Algorithms},
  booktitle = {{CIKM}},
  pages     = {4653--4664},
  publisher = {{ACM}},
  year      = {2021}
}
@inproceedings{recbole[2.0],
  title={RecBole 2.0: Towards a More Up-to-Date Recommendation Library},
  author={Zhao, Wayne Xin and Hou, Yupeng and Pan, Xingyu and Yang, Chen and Zhang, Zeyu and Lin, Zihan and Zhang, Jingsen and Bian, Shuqing and Tang, Jiakai and Sun, Wenqi and others},
  booktitle={Proceedings of the 31st ACM International Conference on Information \& Knowledge Management},
  pages={4722--4726},
  year={2022}
}
@inproceedings{recbole[1.2.0],
  author = {Xu, Lanling and Tian, Zhen and Zhang, Gaowei and Zhang, Junjie and Wang, Lei and Zheng, Bowen and Li, Yifan and Tang, Jiakai and Zhang, Zeyu and Hou, Yupeng and Pan, Xingyu and Zhao, Wayne Xin and Chen, Xu and Wen, Ji-Rong},
  title = {Towards a More User-Friendly and Easy-to-Use Benchmark Library for Recommender Systems},
  pages = {2837–2847},
  year = {2023}
}
```