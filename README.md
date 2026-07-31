# CoDe-SSM: Context-Detail Decoupled State Space Model for Efficient UHD Image Restoration

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

> This repository is currently maintained as an anonymous submission for review purposes. Distribution, citation, or public sharing of the manuscript is strictly prohibited. Author and publication information will be updated after acceptance.

## Overview

Ultra-high-definition (UHD) image restoration must simultaneously correct globally similar degradations and preserve localized fine structures. This introduces an inherent **context-detail trade-off**:

- Globally consistent degradations, such as low-light, haze, blur, rain, and snow, often exhibit cross-region regularities and can be efficiently modeled using compact context aggregation.
- However, aggressive or uniform compression, such as downsampling, window partitioning, or cluster-based token reduction, may irreversibly attenuate edges, textures, and fine structures that are essential for perceptual fidelity.

Existing methods usually apply a single compressed representation to the whole image, failing to distinguish between **shareable degradation context** and **unshareable structural details**.

<p align="center">
<img src="figs/framework.pdf" width="90%">
</p>

To address this issue, we propose **CoDe-SSM**, a **Context-Detail Decoupled State Space Model** for efficient UHD image restoration. CoDe-SSM explicitly decouples global context modeling from local detail recovery into two complementary pathways:

- **Global Cluster Scan Module (GCSM)** — the context pathway.  
  GCSM aggregates image features into `K` input-dependent cluster centers through learnable prototype-based soft clustering. It then performs selective SSM reasoning over this compact centroid sequence and propagates the refined global context back to the pixel space through similarity-guided assignment. Since `K << N`, where `N` is the number of spatial tokens, GCSM decouples computational cost from the original spatial resolution.

- **Local High-Frequency Module (LHFM)** — the detail pathway.  
  LHFM bypasses cluster compression by explicitly extracting the clustering residual. A High-Frequency Energy (HFE) filter, built with fixed Laplacian and Sobel operators, isolates authentic high-frequency structures from clustering-induced averaging artifacts. A sparse mixture of convolutional experts with different receptive fields then adaptively reconstructs diverse local patterns, such as edges, textures, rain streaks, and snowflakes.

Extensive experiments on **five UHD benchmarks** and **five degradation types** demonstrate that the proposed context-detail decoupling strategy yields substantial gains in both restoration quality and efficiency.

CoDe-SSM achieves:

- **2.88M parameters**
- **25.83G FLOPs** at `512 × 512` input resolution
- State-of-the-art performance on UHD-LOL4K, UHD-Blur, UHD-Haze, UHD-Rain, and UHD-Snow

## Key Contributions

- We formulate the **context-detail trade-off** in UHD image restoration, where recurring degradation patterns benefit from aggregation, while localized structures are poorly represented by shared compression.
- We propose **CoDe-SSM**, a framework that decouples global context modeling from local detail preservation via two complementary pathways:
  - **GCSM** for prototype-level SSM reasoning over compact cluster centers.
  - **LHFM** for high-frequency-guided sparse expert routing on the clustering residual.
- Extensive experiments across diverse UHD restoration tasks demonstrate the effectiveness of explicit context-detail decoupling, achieving consistent improvements in PSNR, SSIM, and perceptual quality while maintaining high efficiency.

## Method

### Overall Architecture

CoDe-SSM adopts an asymmetric U-Net architecture. Given an input image `X ∈ R^{B×3×H×W}`, a `3×3` convolution projects it into `C` channels. The encoder contains multiple downsampling stages with strided convolutions, while the decoder performs upsampling with skip connections.

The final restoration is predicted as a gated residual:

```math
\hat{X} = X + \sigma(\alpha) \cdot f_{\theta}(X)
```

where:

- `fθ(X)` is the network-predicted residual,
- `σ(·)` is the sigmoid function,
- `α` is a learnable scalar gate.

Each U-Net level contains several **CoDeBlocks**. A CoDeBlock first splits the normalized and projected features into a feature branch and a gate branch. The feature branch is processed by the two decoupled pathways:

```math
F_{out} = F_{global} + F_{local}
```

where:

- `F_global` is produced by GCSM for global context correction,
- `F_local` is produced by LHFM for local detail recovery.

The fused features are then gated, normalized, and refined by a Gated Dconv Feed-forward Network (GDFN).

### Prototype Clustering

For input tokens `Z ∈ R^{B×N×C}`, CoDe-SSM computes value features `V = {v_i}_{i=1}^N` and softly assigns them to `K` learnable prototype anchors `P = {p_k}_{k=1}^K`.

After `ℓ2` normalization, the soft assignment is computed as:

```math
A_{ik} = \mathrm{softmax}_k \left( \frac{\langle \bar{v}_i, \bar{p}_k \rangle}{\tau} \right)
```

where `τ` is a learnable temperature.

Each prototype anchor aggregates its assigned tokens into an input-dependent cluster center:

```math
m_k = \frac{\sum_{i=1}^{N} A_{ik} v_i}{\sum_{i=1}^{N} A_{ik} + \epsilon}
```

The cluster reconstruction and clustering residual are then given by:

```math
\hat{v}_i = \sum_{k=1}^{K} A_{ik} m_k
```

```math
R_i = v_i - \hat{v}_i
```

The clustering front-end produces three complementary outputs:

1. Compact cluster centers `{m_k}_{k=1}^K`
2. Soft assignment matrix `A`
3. Clustering residual `R`

### Global Cluster Scan Module (GCSM)

The cluster centers summarize globally shareable degradation patterns. GCSM performs selective SSM reasoning over this compact sequence:

```math
\{m'_k\}_{k=1}^{K} = \mathrm{SSM}(\{m_k\}_{k=1}^{K}; \theta_{ssm})
```

Since `K << N`, this models cross-region degradation dependencies at a cost decoupled from the original spatial resolution.

The refined cluster centers are mapped back to the pixel space through the soft assignment matrix:

```math
F_{global,i} = \sum_{k=1}^{K} A_{ik} m'_k
```

Because the assignment is image-global, distant pixels with similar degradation signatures can share refined prototypes, avoiding artificial spatial boundaries introduced by local windows or fixed partitions.

### Local High-Frequency Module (LHFM)

Prototype clustering mainly captures low-frequency and globally shared context, while high-frequency details are largely discarded into the clustering residual. However, the raw residual may also contain reconstruction artifacts. Therefore, LHFM introduces a High-Frequency Energy (HFE) filter to isolate authentic structural details.

Given a single-channel grayscale reference map `Y`, obtained from the original input image and resized to the current feature resolution, the HFE map is computed as:

```math
E_{hf} =
\frac{
|L \circledast Y| + \sqrt{(S_x \circledast Y)^2 + (S_y \circledast Y)^2}
}{
\mu_E + \epsilon
}
```

where:

- `L` is the Laplacian kernel,
- `S_x` and `S_y` are horizontal and vertical Sobel kernels,
- `⊛` denotes 2D convolution,
- `μ_E` is the spatial mean of the numerator.

A soft mask with learnable scalar `β` gates the clustering residual:

```math
M = \sigma(\beta (E_{hf} - 1))
```

```math
\tilde{R} = R \odot M
```

This suppresses artifacts in flat regions while preserving authentic high-frequency components.

The refined residual is then processed by a sparse Mixture-of-Experts (MoE) module. For each token, a lightweight router selects the top-2 experts out of `Ne = 4` convolutional experts. The four experts use distinct receptive-field configurations:

| Expert | Kernel / Structure |
| --- | --- |
| Expert 1 | Standard `3×3` depthwise separable convolution |
| Expert 2 | Dilated `3×3` depthwise separable convolution, dilation rate `2` |
| Expert 3 | Cross-shaped `1×5` and `5×1` convolution |
| Expert 4 | Dense `5×5` depthwise separable convolution |

The expert outputs are aggregated and modulated by a channel-wise attention gate to produce `F_local`.

To prevent expert collapse, we use a load-balancing loss:

```math
\mathcal{L}_{bal} = N_e \sum_{e=1}^{N_e} f_e \bar{\pi}_e
```

where:

- `f_e` is the fraction of tokens dispatched to expert `e`,
- `π̄_e` is the mean router probability for expert `e`.

## Key Results

We evaluate CoDe-SSM on five UHD benchmarks spanning five degradation types:

| Dataset | Task | Train | Test |
| --- | --- | ---: | ---: |
| UHD-LOL4K | Low-light enhancement | 5,999 | 2,100 |
| UHD-Haze | Dehazing | 2,290 | 231 |
| UHD-Blur | Deblurring | 1,964 | 300 |
| UHD-Snow | Desnowing | 3,000 | 200 |
| UHD-Rain | Deraining | 3,000 | 200 |

### UHD-LOL4K: Low-Light Enhancement

<details open>
<summary><strong>Results</strong></summary>

| Methods | Venue | PSNR ↑ | SSIM ↑ | Param. |
| --- | --- | ---: | ---: | ---: |
| NSEN | MM | 29.49 | 0.980 | 2.67M |
| UHDFour | ICLR | 36.12 | 0.990 | 17.5M |
| LLFormer | AAAI | 37.33 | 0.988 | 24.5M |
| UHDformer | AAAI | 36.28 | 0.989 | 0.34M |
| Wave-Mamba | MM | 37.43 | 0.990 | 1.25M |
| D2Net | WACV | 37.73 | 0.992 | 5.22M |
| C²SSM | CVPR | 39.61 | 0.992 | 2.71M |
| **CoDe-SSM (Ours)** | — | **42.24** | **0.996** | 2.88M |

</details>

### UHD-Blur: Deblurring

<details>
<summary><strong>Results</strong></summary>

| Methods | Venue | PSNR ↑ | SSIM ↑ | Param. |
| --- | --- | ---: | ---: | ---: |
| UHDformer | AAAI | 28.82 | 0.844 | 0.34M |
| UHDDIP | TCSVT | 28.28 | 0.845 | 0.81M |
| DreamUHD | AAAI | 29.33 | 0.852 | 1.45M |
| UHD-Processor | CVPR | 29.43 | 0.855 | 1.60M |
| ERR | CVPR | 29.72 | 0.861 | 1.13M |
| C²SSM | CVPR | 31.53 | 0.890 | 2.71M |
| **CoDe-SSM (Ours)** | — | **31.75** | **0.894** | 2.88M |

</details>

### UHD-Haze: Dehazing

<details>
<summary><strong>Results</strong></summary>

| Methods | Venue | PSNR ↑ | SSIM ↑ | Param. |
| --- | --- | ---: | ---: | ---: |
| UHD | ICCV | 18.04 | 0.811 | 34.5M |
| UHDformer | AAAI | 22.59 | 0.942 | 0.34M |
| UHDDIP | TCSVT | 22.14 | 0.941 | 0.81M |
| UHD-Processor | CVPR | 23.24 | 0.953 | 1.60M |
| C²SSM | CVPR | 24.08 | 0.942 | 2.71M |
| **CoDe-SSM (Ours)** | — | **27.09** | **0.963** | 2.88M |

</details>

### UHD-Rain: Deraining

<details>
<summary><strong>Results</strong></summary>

| Methods | Venue | PSNR ↑ | SSIM ↑ | Param. |
| --- | --- | ---: | ---: | ---: |
| Uformer | CVPR | 19.49 | 0.716 | 50.9M |
| Restormer | CVPR | 19.41 | 0.711 | 25.3M |
| SFNet | ICLR | 20.10 | 0.709 | 13.3M |
| UHDformer | AAAI | 37.34 | 0.974 | 0.34M |
| UHDDIP | TCSVT | 40.17 | 0.982 | 0.81M |
| **CoDe-SSM (Ours)** | — | **42.24** | **0.989** | 2.88M |

</details>

### UHD-Snow: Desnowing

<details>
<summary><strong>Results</strong></summary>

| Methods | Venue | PSNR ↑ | SSIM ↑ | Param. |
| --- | --- | ---: | ---: | ---: |
| Uformer | CVPR | 23.72 | 0.871 | 50.9M |
| Restormer | CVPR | 24.14 | 0.869 | 25.3M |
| SFNet | ICLR | 23.64 | 0.846 | 13.3M |
| UHDformer | AAAI | 36.61 | 0.988 | 0.34M |
| UHDDIP | TCSVT | 41.56 | 0.990 | 0.81M |
| C²SSM | CVPR | 42.45 | 0.990 | 2.71M |
| **CoDe-SSM (Ours)** | — | **43.00** | **0.992** | 2.88M |

</details>

### Perceptual Quality on UHD-Rain

We additionally report LPIPS on the UHD-Rain dataset to evaluate perceptual quality.

| Methods | Venue | LPIPS ↓ |
| --- | --- | ---: |
| Uformer | CVPR | 0.460 |
| Restormer | CVPR | 0.478 |
| SFNet | ICLR | 0.477 |
| UHDformer | AAAI | 0.055 |
| UHDDIP | TCSVT | 0.030 |
| **CoDe-SSM (Ours)** | — | **0.019** |

The significantly lower LPIPS indicates that the high-frequency details preserved by LHFM contribute directly to perceptual fidelity, beyond what PSNR gains alone can capture.

## Efficiency

FLOPs are measured with an input size of `512 × 512`.

| Methods | Venue | Param. | FLOPs |
| --- | --- | ---: | ---: |
| D2Net | WACV | 5.22M | 148.84G |
| Wave-Mamba | MM | 1.25M | 28.73G |
| UDR-Mixer | TMM | 4.90M | 52.57G |
| C²SSM | CVPR | 2.71M | 26.04G |
| **CoDe-SSM (Ours)** | — | 2.88M | **25.83G** |

Let:

- `N = H_l × W_l` be the number of spatial tokens at a given U-Net level,
- `C` be the feature dimension,
- `K` be the number of cluster centers, with `K << N`.

The per-block complexity can be summarized as:

| Module | Complexity |
| --- | --- |
| GCSM | `O(NKC + KC²)` |
| LHFM | `O(NC)` |
| CoDeBlock | approximately `O(NKC + KC²)` |

Because GCSM operates on only `K` cluster centers instead of all `N` spatial tokens, the computational cost is effectively decoupled from the full spatial resolution.

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
| Encoder levels | `N1 = 3` |
| Encoder / decoder blocks | `N2 = [2, 4, 4]` |
| Bottleneck / refinement blocks | `N3 = N4 = 4` |
| Embedding dimension | `32` |
| Cluster numbers | `[16, 24, 32]` |
| MoE experts | `4` |
| Activated experts | Top-`2` |
| Losses | `ℓ1` loss + FFT loss + load-balancing loss |

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

## Ablation Studies

All ablation studies are conducted on the UHD-LOL4K dataset.

### Component Analysis

| Variant | PSNR ↑ | SSIM ↑ | Param. |
| --- | ---: | ---: | ---: |
| (a) GCSM → ResBlock | 39.18 | 0.991 | 2.91M |
| (b) LHFM → FFN | 41.26 | 0.993 | 3.12M |
| (c) w/o GCSM | 37.49 | 0.986 | 2.54M |
| (d) w/o LHFM | 40.51 | 0.992 | 2.83M |
| (e) w/o HFE Filter | 41.92 | 0.995 | 2.88M |
| (f) w/o MoE | 41.81 | 0.995 | 2.79M |
| (g) Complete model | **42.24** | **0.996** | 2.88M |

Observations:

- Removing GCSM causes a `4.75 dB` PSNR drop, showing that prototype-level global context modeling is crucial for UHD restoration.
- Removing LHFM causes a `1.73 dB` PSNR drop, demonstrating the importance of explicit high-frequency detail recovery.
- Replacing GCSM with a ResBlock leads to a `3.06 dB` degradation, indicating that SSM reasoning over cluster centers is more effective than plain local convolution.
- Replacing LHFM with a standard FFN causes a `0.98 dB` degradation, showing that residual-specific sparse expert routing is beneficial.
- Removing the HFE filter or MoE module also degrades performance, validating both high-frequency masking and sparse expert routing.

### Cluster Count `K`

| Metric | `[8, 12, 16]` | `[12, 16, 24]` | `[16, 24, 32]` | `[24, 32, 48]` |
| --- | ---: | ---: | ---: | ---: |
| PSNR ↑ | 40.41 | 41.53 | **42.24** | 41.14 |
| SSIM ↑ | 0.995 | 0.995 | **0.996** | 0.994 |
| Param. | 2.86M | 2.87M | 2.88M | 2.89M |

The best performance is achieved with `[16, 24, 32]` cluster centers. Too few clusters under-represent fine degradation patterns, while too many clusters dilute prototype compactness.

### Expert Activation in MoE

We study the effect of the number of activated experts in the LHFM MoE module.

| Metric | Top-1 | Top-2 | Top-3 | Top-4 |
| --- | ---: | ---: | ---: | ---: |
| PSNR ↑ | 41.94 | **42.24** | 41.28 | 41.87 |
| SSIM ↑ | 0.993 | **0.996** | 0.993 | 0.992 |
| LPIPS ↓ | 0.0134 | **0.0121** | 0.0145 | 0.0148 |

Top-2 expert activation achieves the best trade-off between representational diversity and sparse computation.

## Limitations and Future Work

Although CoDe-SSM achieves strong performance across five UHD restoration tasks, the current high-frequency energy gate in LHFM relies on global-mean normalization. Under highly non-uniform illumination, this may amplify noise in bright regions or suppress weak details in dark regions.

Future work includes:

- Developing illumination-aware local normalization for more robust high-frequency detail recovery.
- Extending CoDe-SSM toward unified, degradation-agnostic UHD image restoration.
- Exploring more flexible prototype learning and routing strategies for complex real-world degradations.

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

The citation information will be updated after the paper is accepted.

## Acknowledgments

This codebase is built upon [BasicSR](https://github.com/XPixelGroup/BasicSR) and [C²SSM](https://github.com/5chen/C2SSM). We thank the authors for their excellent work.


