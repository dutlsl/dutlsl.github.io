---
title: "[WACV 2025] EW-ViT: Frequency-Domain Refinement of Vision Transformers for Robust Medical Image Segmentation"
date: 2026-07-16T20:41:00+09:00
draft: false
math: true
tags: ["Paper Review", "Medical Image Segmentation", "Vision Transformer", "Wavelet", "WACV 2025"]
categories: ["Paper Review"]
summary: "A review of the EW-ViT paper presented at WACV 2025. It integrates wavelet decomposition into self-attention and introduces the Prompt-Guided High-Frequency Refiner (PGHFR) module to achieve robust segmentation performance even on degraded medical images."
cover:
  image: "/images/ew_vit/ew_vit_architecture.jpeg"
  alt: "EW-ViT Architecture"
---

> **Paper Information**
> - **Title:** Frequency-Domain Refinement of Vision Transformers for Robust Medical Image Segmentation under Degradation
> - **Authors:** Sanaz Karimijafarbigloo, Sina Ghorbani Kolahi, Reza Azad, Ulas Bagci, Dorit Merhof
> - **Affiliations:** University of Regensburg, Tarbiat Modares University, Northwestern University, Fraunhofer MEVIS
> - **Venue:** WACV 2025
> - **Code:** [GitHub](https://github.com/)

---

## 1. TL;DR

A medical image segmentation framework that integrates **wavelet decomposition** into the self-attention block of Vision Transformers to adaptively refine low- and high-frequency components, combines a **Prompt-Guided High-Frequency Refiner (PGHFR)** with **contrastive learning**, and achieves state-of-the-art performance on Synapse, ISIC2018, and ACDC datasets under various degradation conditions including noise and blur.

---

## 2. Background and Motivation

### 2.1. Problem Definition

Medical Image Segmentation is a core process for precise diagnosis, treatment planning, and disease monitoring. CNN-based methods (particularly U-Net) have achieved remarkable results but face **limitations in modeling long-range dependencies**. Vision Transformers (ViTs) capture global contextual information through self-attention, but they have a structural weakness of **insufficient local feature representation**.

### 2.2. Limitations of Existing Methods

To complement ViTs' lack of local features, CNN-Transformer hybrid architectures (TransUNet, HiFormer, CoTr, etc.) have been proposed, but they have the following limitations:

- **High parameter count and computational cost**: TransUNet has 96M parameters and 88.91G FLOPs, making it inefficient.
- **CNN backbone dependency**: HiFormer, CoTr, and others still heavily depend on CNN backbones, with insufficient integration of multi-scale information.
- **Failure to capture high-frequency components**: Research shows that self-attention acts as a **low-pass filter**, failing to sufficiently capture high-frequency components (boundaries, fine textures, etc.).
- **Neglect of degradation conditions**: Existing frequency-based methods (Laplacian-Former, WaveFormer) preserve high-frequency information, but cannot handle situations where **noise, blur, and haze** — common in medical images — corrupt high-frequency components.

### 2.3. Key Contributions

1. **Wavelet-based frequency domain modeling**: Performs wavelet decomposition within self-attention to halve spatial dimensions while simultaneously capturing multi-scale representations, reducing computational complexity.
2. **PGHFR module**: Uses learnable prompts to implicitly encode degradation-related information and dynamically refine high-frequency components to remove noise effects.
3. **Contrastive learning for feature consistency**: Maintains representational consistency through contrastive learning between original and degradation-augmented features, providing indirect guidance to the PGHFR module.
4. **Degradation robustness verification**: Demonstrates model robustness through comprehensive ablation studies across various degradation types including Gaussian noise, blur, and color jittering.

---

## 3. Proposed Method: EW-ViT Framework

### 3.1. Overall Architecture

![EW-ViT Overall Architecture](/images/ew_vit/ew_vit_architecture.jpeg)

*Figure 1: Overall structure of EW-ViT. A U-shaped encoder-decoder configuration where the encoder repeats EW-ViT blocks with patch embedding, and the decoder uses EW-ViT blocks with integrated PGHFR and patch expansion. The Contrastive Head handles feature consistency learning.*

EW-ViT is composed of a U-shaped architecture with an **encoder**, **decoder**, and two heads (**Contrastive Head** and **Segmentation Head**).

- **Encoder**: Consists of 3 layers, each with an EW-ViT block followed by patch embedding (downsampling + dimension expansion).
- **Decoder**: Consists of 3 layers with EW-ViT blocks (including PGHFR) followed by patch expansion (upsampling).
- **Contrastive Head**: Performs contrastive learning using the output from the encoder bottleneck layer.

### 3.2. Enhanced Wave ViT (EW-ViT) Block

![Enhanced Wave Attention Block Details](/images/ew_vit/ew_vit_block.jpeg)

*Figure 2: Detailed structure of the EW-ViT block. DWT separates 4 subbands, and PGHFR refines high-frequency components. Internal operations of CPG and IPI modules are also shown.*

The operation of the EW-ViT block proceeds as follows:

**① Channel Reduction and Wavelet Decomposition**

A linear transformation is applied to the 2D feature map $X \in \mathbb{R}^{H \times W \times D}$ to reduce channels by 1/4 ($\widetilde{X} = XW_d$), followed by **Discrete Wavelet Transform (DWT)**. Using the classical Haar wavelet, low-pass and high-pass filters are applied sequentially along rows and columns, generating 4 subbands:

- $X_{LL}$: Low-frequency component — captures the basic structure of objects at a coarse level
- $X_{LH}, X_{HL}, X_{HH}$: High-frequency components — preserves fine textures and boundary information

Each subband has half the spatial resolution of the original ($H/2 \times W/2$), naturally **reducing the computational cost** of subsequent self-attention.

**② High-Frequency Refinement via PGHFR**

The three high-frequency subbands are concatenated channel-wise as $X_H = [X_{LH}, X_{HL}, X_{HH}]$, and the PGHFR module is applied to obtain degradation-free $X'_H$ (see Section 3.3 for details).

**③ Wavelet-Attention Computation**

The refined high-frequency component $X'_H$ and low-frequency component $X_L$ are combined, then a $3 \times 3$ convolution enforces locality to produce the downsampled feature map $X^d$. This $X^d$ is transformed into keys ($K$) and values ($V$), and multi-head self-attention is performed with queries ($Q$) extracted from the original feature map:

$$head_j = \text{Softmax}\left(\frac{Q_j K_j^T}{\sqrt{D_h}}\right) V_j$$

**④ IDWT Path and Final Output Synthesis**

Separately, **Inverse Discrete Wavelet Transform (IDWT)** is applied to $[X_L, X'_H]$ to restore refined features at the original resolution. The output from this IDWT path is combined with the multi-head attention output to compose the final EW-ViT block output:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_0, \ldots, \text{head}_{N_h}) \widetilde{W}^O$$

### 3.3. Prompt-Guided High-Frequency Refiner (PGHFR)

PGHFR is the core module for removing degradation effects (noise, blur) from high-frequency components, consisting of two sub-modules: **Component Prompt Generation (CPG)** and **Implicit Prompt Interaction (IPI)**.

$$X'_H = \text{IPI}(\text{CPG}(P_c, X_H), X_H)$$

**Component Prompt Generation (CPG)**

Learnable prompt components $P_c \in \mathbb{R}^{N \times \frac{H}{2} \times \frac{W}{2} \times D'}$ are dynamically combined according to the context of input high-frequency features $X_H$.

1. **Global Average Pooling (GAP)** and **Global Max Pooling (GMP)** are applied to $X_H$ to compress spatial information.
2. The two pooling results are summed and passed through **Point-Wise Convolution (PWC)** + Softmax to generate prompt weights $w \in \mathbb{R}^{D'}$.
3. Prompt components are weighted-summed, then refined with a $3 \times 3$ convolution to produce the final prompt $P$.

$$w_i = \text{Softmax}(\text{PWC}(\text{GAP}(X_H) + \text{GMP}(X_H)))$$
$$P = \text{Conv}_{3 \times 3}\left(\sum_{c=1}^{D'} w_i P_c\right)$$

**Implicit Prompt Interaction (IPI)**

The generated prompt $P$ and high-frequency input $X_H$ are concatenated channel-wise, then interacted through an attention block. In this process, the prompt conveys degradation-related information to selectively suppress noise in high-frequency components:

$$X'_H = \text{Conv}_{3 \times 3}(\text{Attention}(X_H, P))$$

### 3.4. Contrastive Learning for Feature Consistency

At the encoder bottleneck layer, **random degradation noise** $G(X_H)$ is injected into high-frequency components, followed by IDWT reconstruction to generate augmented features $X^{aug}$. Pixels from identical spatial locations in both original and augmented features are sampled as contrastive learning candidates.

The class prototype for class $k$ is computed as the mean of pixel features belonging to that class:

$$c_k = \frac{1}{|S_k|} \sum_{k \in S_k} X_k$$

The contrastive loss for aligning augmented candidate pixels with original class prototypes is defined as:

$$\mathcal{L}_{c_k} = -\frac{1}{|S_k|} \sum_{x_k^{aug} \in S_k} \log \frac{\exp(\text{sim}(x_k^{aug}, c_k) / \tau)}{\sum_{j \neq k} \exp(\text{sim}(x_k^{aug}, c_j) / \tau)}$$

where $\text{sim}(\cdot, \cdot)$ is cosine similarity and $\tau$ is the temperature parameter. The overall contrastive loss is summed over all classes, with an additional regularization term $L_{sep} = \sum_{i \neq j} \frac{1}{\|c_i - c_j\|_2}$ to enforce class prototype separation.

The final training objective is:

$$Loss = \lambda_s \mathcal{L}_{Seg} + \lambda_c \mathcal{L}_c$$

---

## 4. Experimental Results

### 4.1. Datasets and Implementation Details

| Dataset | Type | Scale | Segmentation Targets |
|---------|------|-------|---------------------|
| **Synapse** | Abdominal CT | 30 cases, 3,779 slices | 8 organs (spleen, kidneys, etc.) |
| **ISIC2018** | Dermoscopy | — | Skin lesions |
| **ACDC** | Cardiac MRI | 100 scans (train 70/val 10/test 20) | RV, Myo, LV |

- **Optimizer**: Synapse — SGD (lr=0.05, momentum=0.9), ISIC/ACDC — Adam (lr=0.0001)
- **Loss function**: Synapse/ISIC — BDOU + contrastive loss, ACDC — CE + DICE + contrastive loss
- **GPU**: Single RTX 3090

### 4.2. Synapse Dataset Results

| Method | Params (M) | FLOPs (G) | Avg DSC↑ | HD95↓ |
|--------|-----------|-----------|----------|-------|
| TransUNet | 96.07 | 88.91 | 77.49 | 31.69 |
| Swin-UNet | 27.17 | 6.16 | 79.13 | 21.55 |
| MISSFormer | 42.46 | 9.89 | 81.96 | 18.20 |
| HiFormer-B | 25.51 | 8.05 | 80.39 | 14.70 |
| Laplacian-Former | 27.54 | 6.68 | 81.90 | 18.86 |
| WaveFormer | 47.01 | 7.75 | 81.92 | 18.41 |
| VM-UNet | 44.27 | 6.52 | 81.08 | 19.21 |
| **EW-ViT** | **33.35** | **6.03** | **83.51** | **16.68** |

EW-ViT achieves **~1/3 the parameters and ~1/15 the FLOPs** of TransUNet while improving DSC by **6.02%p**. Compared to existing frequency-based methods Laplacian-Former (81.90%) and WaveFormer (81.92%), it shows gains of **1.61%p and 1.59%p**, respectively.

![Visual Comparison on Synapse Dataset](/images/ew_vit/synapse_visual.jpeg)

*Figure 3: Segmentation result comparison between EW-ViT, WaveFormer, and MissFormer on the Synapse dataset. EW-ViT achieves the most precise boundary segmentation for the left kidney (row 1) and stomach (row 2).*

### 4.3. ISIC2018 Dataset Results

| Method | DSC | SE | SP | ACC |
|--------|-----|----|----|-----|
| Swin-UNet | 0.8946 | 0.9056 | 0.9798 | 0.9645 |
| VM-UNet | 0.8971 | 0.9112 | 0.9613 | 0.9491 |
| Laplacian-Former | 0.9128 | 0.9290 | 0.9626 | 0.9640 |
| **EW-ViT** | **0.9164** | **0.9297** | 0.9571 | **0.9643** |

EW-ViT achieved the highest overall performance with DSC of 0.9164 and Sensitivity (SE) of 0.9297. The slightly lower Specificity (SP) of 0.9571 suggests that the high-frequency-based feature enhancement may cause slight over-segmentation by aggressively detecting lesion boundaries.

### 4.4. ACDC Dataset Results

| Method | Avg DICE | RV | Myo | LV |
|--------|----------|-----|------|-----|
| TransUNet | 89.71 | 88.86 | 84.53 | 95.73 |
| Swin-UNet | 90.00 | 88.55 | 85.62 | 95.83 |
| MT-UNet | 90.43 | 86.64 | 89.04 | 95.62 |
| **EW-ViT** | **90.29** | 88.05 | **87.56** | 95.25 |

On ACDC, EW-ViT achieved an Avg DICE of 90.29%, surpassing TransUNet and Swin-UNet by 0.58%p and 0.29%p, respectively. Notably, it recorded 87.56% for myocardium (Myo) segmentation, outperforming Swin-UNet (85.62%) by 1.94%p. While MT-UNet is 0.14%p higher in overall average, EW-ViT significantly outperforms it in robustness under degradation conditions.

### 4.5. Degradation Robustness Analysis

| Method | Gaussian Noise ($\sigma$=0.05) | Gaussian Blur ($\sigma$=1.5) | Color Jittering ($\gamma$=1.75) |
|--------|:---:|:---:|:---:|
| | DSC / $\Delta$↓ | DSC / $\Delta$↓ | DSC / $\Delta$↓ |
| Swin-UNet | 88.20 / 1.80 | 85.49 / 4.51 | 82.54 / 7.46 |
| MT-UNet | 88.12 / 2.31 | 81.53 / 8.90 | 75.53 / 14.90 |
| **EW-ViT** | **89.52 / 0.77** | **86.68 / 3.61** | **83.52 / 6.77** |

On the ACDC dataset, EW-ViT showed the **smallest performance degradation ($\Delta$)** across all degradation types and levels. Notably, under color jittering at $\gamma$=1.75, MT-UNet dropped by 14.90%p while EW-ViT only dropped by 6.77%p.

On the Synapse dataset under high noise ($\sigma$=0.25), EW-ViT maintained DSC of 79.70% ($\Delta$=3.81), while TransUNet dropped to 67.87% ($\Delta$=9.62) and WaveFormer to 74.41% ($\Delta$=7.51).

![Attention Map Visualization under Noise](/images/ew_vit/attention_map_noise.jpeg)

*Figure 4: GradCAM attention maps of EW-ViT under noise-free ($\sigma$=0.0) and high noise ($\sigma$=0.3) conditions. The model accurately focuses on the object of interest even under significant noise.*

---

## 5. Summary of Key Contributions

**① Wavelet Decomposition-Based Frequency Domain Self-Attention Reformulation**

To address the problem of ViT's self-attention acting as a low-pass filter that loses high-frequency components, DWT is integrated within self-attention. This (1) explicitly separates low-frequency (structure) and high-frequency (boundary/texture) components to provide appropriate processing paths for each, and (2) naturally reduces computational complexity since subband spatial resolution is half that of the original, shrinking Key·Value dimensions in self-attention.

**② PGHFR: Prompt-Based Adaptive High-Frequency Refinement**

The problem of noise, blur, and haze corrupting high-frequency components — common in medical images — is addressed through learnable prompts. The CPG module dynamically generates prompts reflecting input feature context, and the IPI module interacts these prompts with high-frequency features via attention to selectively remove degradation. This design enables universal operation without prior knowledge of degradation type or intensity.

**③ Degradation-Invariant Representation Learning via Contrastive Learning**

Contrastive learning between augmented features (with random degradation injected into high-frequency components at the encoder bottleneck) and original features encourages the model to learn consistent representations regardless of degradation. This strategy provides indirect learning signals to the PGHFR module while also strengthening inter-class discriminability through the class prototype separation term.

**④ Comprehensive Degradation Robustness Verification**

Extensive ablation studies were conducted across 9 degradation conditions: Gaussian noise (3 levels), Gaussian blur (3 levels), and color jittering (3 levels). EW-ViT showed the least performance degradation compared to existing methods across all conditions, maintaining 5.29%p higher DSC than WaveFormer even under extreme degradation ($\sigma$=0.25 noise), demonstrating practical applicability in real clinical environments.
