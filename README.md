## CoDe-SSM: Context-Detail Decoupled State Space Model for Efficient UHD Image Restoration

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/) [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

Ultra-high-definition (UHD) image restoration faces an inherent dilemma: correcting globally similar degradations requires compact context aggregation, yet such coarse compression irreversibly destroys edges, textures, and fine structures essential for perceptual fidelity. Prior methods apply uniform compression—downsampling, window partitioning, or cluster-based token reduction—failing to distinguish between degradation contexts that can be safely shared and structural details that must be strictly preserved.

<p align="center">
  <img src="figs/framework.pdf" width="90%">
</p>

**CoDe-SSM** resolves this tension by explicitly decoupling context modeling from detail recovery into two dedicated, complementary pathways:

- **Global Cluster Scan Module (GCSM)** — The context pathway learns *K* shared restoration prototypes via a neural-parameterized mixture model, performs selective SSM reasoning over this compact centroid sequence, and diffuses global context back through similarity-guided propagation. By operating on a handful of cluster centers rather than millions of pixels, GCSM decouples computational cost from spatial resolution.

- **Local High-Frequency Module (LHFM)** — The detail pathway bypasses cluster compression by extracting the clustering residual. Laplacian and Sobel operators isolate authentic high-frequency structures from clustering-induced averaging artifacts, while a sparse mixture of convolutional experts adaptively reconstructs diverse local patterns.

Experiments across five UHD benchmarks and five degradation types demonstrate that context-detail decoupling yields substantial gains in both restoration quality and efficiency: **2.88M parameters, 25.83G FLOPs**.

---

## Key Results

<details open>
<summary><strong>UHD-LOL4K (Low-Light Enhancement)</strong></summary>

| Methods | Venue | PSNR | SSIM | Param. |
|---------|-------|------|------|--------|
| NSEN | MM | 29.49 | 0.980 | 2.67M |
| UHDFour | ICLR | 36.12 | 0.990 | 17.5M |
| LLFormer | AAAI | 37.33 | 0.988 | 24.5M |
| UHDformer | AAAI | 36.28 | 0.989 | 0.34M |
| Wave-Mamba | MM | 37.43 | 0.990 | 1.25M |
| D2Net | WACV | 37.73 | 0.992 | 5.22M |
| C²SSM | CVPR | 39.61 | 0.992 | 2.71M |
| **CoDe-SSM (Ours)** | **—** | **42.24** | **0.996** | **2.88M** |

</details>

<details>
<summary><strong>UHD-Blur (Deblurring)</strong></summary>

| Methods | Venue | PSNR | SSIM | Param. |
|---------|-------|------|------|--------|
| UHDformer | AAAI | 28.82 | 0.844 | 0.34M |
| UHDDIP | TCSVT | 28.28 | 0.845 | 0.81M |
| DreamUHD | AAAI | 29.33 | 0.852 | 1.45M |
| UHD-Processor | CVPR | 29.43 | 0.855 | 1.60M |
| ERR | CVPR | 29.72 | 0.861 | 1.13M |
| C²SSM | CVPR | 31.53 | 0.890 | 2.71M |
| **CoDe-SSM (Ours)** | **—** | **31.75** | **0.894** | **2.88M** |

</details>

<details>
<summary><strong>UHD-Haze (Dehazing)</strong></summary>

| Methods | Venue | PSNR | SSIM | Param. |
|---------|-------|------|------|--------|
| UHD | ICCV | 18.04 | 0.811 | 34.5M |
| UHDformer | AAAI | 22.59 | 0.942 | 0.34M |
| UHDDIP | TCSVT | 22.14 | 0.941 | 0.81M |
| UHD-Processor | CVPR | 23.24 | 0.953 | 1.60M |
| C²SSM | CVPR | 24.08 | 0.942 | 2.71M |
| **CoDe-SSM (Ours)** | **—** | **27.09** | **0.963** | **2.88M** |

</details>

<details>
<summary><strong>UHD-Rain (Deraining)</strong></summary>

| Methods | Venue | PSNR | SSIM | Param. |
|---------|-------|------|------|--------|
| Uformer | CVPR | 19.49 | 0.716 | 50.9M |
| Restormer | CVPR | 19.41 | 0.711 | 25.3M |
| SFNet | ICLR | 20.10 | 0.709 | 13.3M |
| UHDformer | AAAI | 37.34 | 0.974 | 0.34M |
| UHDDIP | TCSVT | 40.17 | 0.982 | 0.81M |
| **CoDe-SSM (Ours)** | **—** | **42.24** | **0.989** | **2.88M** |

</details>

<details>
<summary><strong>UHD-Snow (Desnowing)</strong></summary>

| Methods | Venue | PSNR | SSIM | Param. |
|---------|-------|------|------|--------|
| Uformer | CVPR | 23.72 | 0.871 | 50.9M |
| Restormer | CVPR | 24.14 | 0.869 | 25.3M |
| SFNet | ICLR | 23.64 | 0.846 | 13.3M |
| UHDformer | AAAI | 36.61 | 0.988 | 0.34M |
| UHDDIP | TCSVT | 41.56 | 0.990 | 0.81M |
| C²SSM | CVPR | 42.45 | 0.990 | 2.71M |
| **CoDe-SSM (Ours)** | **—** | **43.00** | **0.992** | **2.88M** |

</details>

### Efficiency

| Methods | Venue | Param. | FLOPs |
|---------|-------|--------|-------|
| D2Net | WACV | 5.22M | 148.84G |
| UDR-Mixer | TMM | 4.90M | 52.57G |
| Wave-Mamba | MM | 1.25M | 28.73G |
| C²SSM | CVPR | 2.71M | 26.04G |
| **CoDe-SSM (Ours)** | **—** | **2.88M** | **25.83G** |

*FLOPs measured at 512x512 input.*

---

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

---

## Dataset Preparation

Download the datasets and organize them as follows:

```
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
|---------|-------|------|
| UHD-LOL4K | 5,999 | 2,100 |
| UHD-Haze | 2,290 | 231 |
| UHD-Blur | 1,964 | 300 |
| UHD-Snow | 3,000 | 200 |
| UHD-Rain | 3,000 | 200 |

---

## Training

Training uses 4 NVIDIA RTX 3090 GPUs. Images are randomly cropped to 768x768 with batch size 4, trained for 200K iterations with AdamW (lr=5e-4, cosine annealing).

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

---

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

```bash
# Low-light Enhancement
python inference.py \
    -i datasets/UHD-LOL4K/test/low \
    -g datasets/UHD-LOL4K/test/high \
    -w pretrained_models/CoDeSSM_uhdlol.pth \
    -o results/uhdlol \
    --save_img

# Deblurring
python inference.py \
    -i datasets/UHD-Blur/test/input \
    -g datasets/UHD-Blur/test/gt \
    -w pretrained_models/CoDeSSM_uhdblur.pth \
    -o results/uhdblur \
    --save_img

# Dehazing
python inference.py \
    -i datasets/UHD-Haze/test/input \
    -g datasets/UHD-Haze/test/gt \
    -w pretrained_models/CoDeSSM_uhdhaze.pth \
    -o results/uhdhaze \
    --save_img
```

---

## Ablation Studies

*On UHD-LOL4K.*

### Component Analysis

| Variant | PSNR | SSIM | Param. |
|---------|------|------|--------|
| (a) GCSM -> ResBlock | 39.18 | 0.991 | 2.91M |
| (b) LHFM -> FFN | 41.26 | 0.993 | 3.12M |
| (c) w/o GCSM | 37.49 | 0.986 | 2.54M |
| (d) w/o LHFM | 40.51 | 0.992 | 2.83M |
| (e) w/o HFE Filter | 41.92 | 0.995 | 2.88M |
| **(f) Complete model** | **42.24** | **0.996** | **2.88M** |

### Cluster Count *K*

| Metric | [8,12,16] | [12,16,24] | [16,24,32] | [24,32,48] |
|--------|-----------|------------|------------|------------|
| PSNR | 40.41 | 41.53 | **42.24** | 41.14 |
| SSIM | 0.995 | 0.995 | **0.996** | 0.994 |
| Param. | 2.86M | 2.87M | **2.88M** | 2.89M |

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{codessm,
    title={CoDe-SSM: Context-Detail Decoupled State Space Model for Efficient UHD Image Restoration},
    author={},
    booktitle={},
    year={2027}
}
```

---

## Acknowledgments

This codebase is built upon [BasicSR](https://github.com/XPixelGroup/BasicSR) and [C²SSM](https://github.com/5chen/C2SSM). We thank the authors for their excellent work.


