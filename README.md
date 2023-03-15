# RIDCP: Revitalizing Real Image Dehazing via High-Quality Codebook Priors (CVPR2023)

![Python 3.6](https://img.shields.io/badge/python-3.6-g) ![pytorch 1.10.2](https://img.shields.io/badge/pytorch-1.10.2-blue.svg)

This is the official PyTorch codes for the paper.  
>**RIDCP: Revitalizing Real Image Dehazing via High-Quality Codebook Priors**<br>  [Ruiqi Wu](https://rq-wu.github.io/), [Zhengpeng Duan](https://github.com/Adam-duan), [Chunle Guo<sup>*</sup>](https://scholar.google.com/citations?user=RZLYwR0AAAAJ&hl=en), [Zhi Chai](), [Chongyi Li](https://li-chongyi.github.io/)
( <sup>*</sup> indicates corresponding author)<br>
>The IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR), 2023

![framework_img](figs/framework_overview.png)

[Arxiv Paper (TBD)]  [中文版 (TBD)] [Project (TBD)]  [Dataset (TBD)]

## Demo

<img src="figs/mountain.gif" width="390px"/> &nbsp; &nbsp; &nbsp; &nbsp; <img src="figs/car.gif" width="390px"/>

## Dependencies and Installation

- Ubuntu >= 18.04
- CUDA >= 11.0
- Other required packages in `requirements.txt`
```
# git clone this repository
git clone https://github.com/RQ-Wu/RIDCP.git
cd RIDCP

# create new anaconda env
conda create -n ridcp python=3.8
source activate ridcp 

# install python dependencies
pip install -r requirements.txt
BASICSR_EXT=True python setup.py develop
```

