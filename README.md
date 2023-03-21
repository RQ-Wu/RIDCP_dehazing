# RIDCP: Revitalizing Real Image Dehazing via High-Quality Codebook Priors (CVPR2023)

![Python 3.8](https://img.shields.io/badge/python-3.8-g) ![pytorch 1.12.0](https://img.shields.io/badge/pytorch-1.12.0-blue.svg)

This is the official PyTorch codes for the paper.  
>**RIDCP: Revitalizing Real Image Dehazing via High-Quality Codebook Priors**<br>  [Ruiqi Wu](https://rq-wu.github.io/), [Zhengpeng Duan](https://github.com/Adam-duan), [Chunle Guo<sup>*</sup>](https://scholar.google.com/citations?user=RZLYwR0AAAAJ&hl=en), [Zhi Chai](), [Chongyi Li](https://li-chongyi.github.io/) （ * indicates corresponding author)<br>
>The IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR), 2023

![framework_img](figs/framework_overview.png)

[Arxiv Paper (TBD)]  [中文版 (TBD)] [Project (TBD)]  [Dataset (TBD)]

## Demo
<img src="figs/fig1.png" width="800px">
<img src="figs/fig2.png" width="800px">

### Video examples
<img src="https://github.com/RQ-Wu/RIDCP/blob/master/figs/mountain.gif?raw=true" width="390px"/> &nbsp; &nbsp; &nbsp; &nbsp; <img src="https://github.com/RQ-Wu/RIDCP/blob/master/figs/car.gif?raw=true" width="390px"/>

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

## Get Started
### Prepare pretrained models & dataset
TBD

### Quick demo
Run demos to process the images in dir `./examples/` by following commands:
```
python inference_ridcp.py -i examples -w pretrained_models/pretrained_RIDCP.pth -o results
```

### Train RIDCP
Step 1: Pretrain a VQGAN on high-quality dataset
```
TBD
```

Step 2: Train our RIDCP
```
CUDA_VISIBLE_DEVICES=X,X,X,X python basicsr/train.py --opt options/RIDCP.yml
```

Step3: Adjust our RIDCP
```
TBD
```

## Citation
If you find our repo useful for your research, please cite us:
```
@inproceedings{wu2023ridcp,
    title={RIDCP: Revitalizing Real Image Dehazing via High-Quality Codebook Priors},
    author={Wu, Ruiqi and Duan, Zhengpeng and Guo, Chunle and Chai, Zhi and Li, Chongyi},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2023}
}
```

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only.
Any commercial use should get formal permission first.

## Acknowledgement
This repository is maintained by [Ruiqi Wu](https://rq-wu.github.io/).