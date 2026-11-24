# CoDe-SSM: Context-Detail Decoupled State Space Model for Efficient UHD Image Restoration

[![arXiv](https://img.shields.io/badge/arXiv-2607.29595-b31b1b.svg)](https://arxiv.org/pdf/2607.29595)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

This repository provides training and inference code for CoDe-SSM on five UHD image restoration tasks: low-light enhancement, deblurring, dehazing, deraining, and desnowing.

## Installation

### Prerequisites

- Python >= 3.8
- PyTorch >= 1.12
- CUDA >= 11.3

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install Mamba

```bash
pip install causal_conv1d
pip install mamba_ssm
```

## Dataset Preparation

Download the datasets and organize them as follows:

```text
datasets/
 ├── UHD-Blur/
 │   ├── train/
 │   │   ├── gt/
 │   │   └── input/
 │   └── test/
 │       ├── gt/
 │       └── input/
 ├── UHD-Rain/
 │   ├── train/
 │   │   ├── target/
 │   │   └── input/
 │   └── test/
 │       ├── target/
 │       └── input/
 ├── UHD-Haze/
 │   ├── train/
 │   │   ├── gt/
 │   │   └── input/
 │   └── test/
 │       ├── gt/
 │       └── input/
 ├── UHD-LOL4K/
 │   ├── train/
 │   │   ├── high/
 │   │   └── low/
 │   └── test/
 │       ├── high/
 │       └── low/
 └── UHD-Snow/
     ├── train/
     │   ├── target/
     │   └── input/
     └── test/
         ├── target/
         └── input/
```

### Dataset Statistics

| Dataset | Train | Test |
| --- | ---: | ---: |
| UHD-LOL4K | 5,999 | 2,100 |
| UHD-Haze | 2,290 | 231 |
| UHD-Blur | 1,964 | 300 |
| UHD-Snow | 3,000 | 200 |
| UHD-Rain | 3,000 | 200 |

## Training

Training uses **4 NVIDIA RTX 3090 GPUs**. Full-resolution 4K images are randomly cropped to `768 × 768` with batch size `4`. All models are trained for `200K` iterations.

### Training Configuration

| Item | Setting |
| --- | --- |
| GPUs | 4 × NVIDIA RTX 3090 |
| Crop size | `768 × 768` |
| Batch size | `4` |
| Iterations | `200K` |
| Optimizer | AdamW |
| Learning rate | `5e-4` |
| Weight decay | `1e-3` |
| LR schedule | Cosine annealing |

### Train Commands

```bash
bash train.sh uhdblur    # Deblurring
bash train.sh uhdrain    # Deraining
bash train.sh uhdhaze    # Dehazing
bash train.sh uhdlol     # Low-light Enhancement
bash train.sh uhdsnow    # Desnowing
```

Or run directly with `torchrun`:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --standalone --nproc_per_node=4 --master_port=4337 \
    basicsr/train.py -opt options/train_CoDeSSM_uhdlol.yml --launcher pytorch
```

## Inference

```bash
python inference.py \
    -i <input_folder> \
    -g <gt_folder> \
    -w <model_weight_path> \
    -o <output_folder> \
    --device cuda:0 \
    --save_img
```

### Examples

#### Low-Light Enhancement

```bash
python inference.py \
    -i datasets/UHD-LOL4K/test/low \
    -g datasets/UHD-LOL4K/test/high \
    -w pretrained_models/CoDeSSM_uhdlol.pth \
    -o results/uhdlol \
    --save_img
```

#### Deblurring

```bash
python inference.py \
    -i datasets/UHD-Blur/test/input \
    -g datasets/UHD-Blur/test/gt \
    -w pretrained_models/CoDeSSM_uhdblur.pth \
    -o results/uhdblur \
    --save_img
```

#### Dehazing

```bash
python inference.py \
    -i datasets/UHD-Haze/test/input \
    -g datasets/UHD-Haze/test/gt \
    -w pretrained_models/CoDeSSM_uhdhaze.pth \
    -o results/uhdhaze \
    --save_img
```

#### Deraining

```bash
python inference.py \
    -i datasets/UHD-Rain/test/input \
    -g datasets/UHD-Rain/test/target \
    -w pretrained_models/CoDeSSM_uhdrain.pth \
    -o results/uhdrain \
    --save_img
```

#### Desnowing

```bash
python inference.py \
    -i datasets/UHD-Snow/test/input \
    -g datasets/UHD-Snow/test/target \
    -w pretrained_models/CoDeSSM_uhdsnow.pth \
    -o results/uhdsnow \
    --save_img
```

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{su2026codessm,
    title={CoDe-SSM: Context-Detail Decoupled State Space Model for Efficient UHD Image Restoration},
    author={Jiaxu Su and Zhijian Wu and Jun Li and Bo Zhang and Yefeng Zheng},
    year={2026},
    eprint={2607.29595},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2607.29595}
}
```

## Acknowledgments

This codebase is built upon [BasicSR](https://github.com/XPixelGroup/BasicSR), [Mamba](https://github.com/state-spaces/mamba), and [C²SSM](https://github.com/5chen/C2SSM). We thank the authors for their excellent work.
