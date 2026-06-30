---
title: "[CVPR 2026] GazeShift: Paper Review on Unsupervised Gaze Estimation Framework for VR"
date: 2026-06-25T21:00:00+09:00
draft: false
math: true
tags: ["Paper Review", "Gaze Estimation", "VR", "Unsupervised Learning", "CVPR 2026"]
categories: ["Paper Review"]
summary: "We review the GazeShift paper presented by Samsung SIRC and Bar-Ilan University at CVPR 2026. The paper proposes an unsupervised gaze estimation framework for VR environments and a large-scale off-axis dataset called VRGaze."
cover:
  image: "/images/gazeshift/_page_1_Figure_0.jpeg"
  alt: "GazeShift Architecture"
  caption: "GazeShift Architecture Overview (Paper Figure 1)"
---

> **Paper Information**
> - **Title:** GazeShift: Unsupervised Gaze Estimation and Dataset for VR
> - **Authors:** Gil Shapira, Ishay Goldin, Evgeny Artyomov, Donghoon Kim, Yosi Keller, Niv Zehngut
> - **Affiliations:** Samsung Semiconductor Israel R&D Center (SIRC), Samsung Electronics, Bar-Ilan University
> - **Conference:** CVPR 2026
> - **Code & Dataset:** [github.com/gazeshift3/gazeshift](https://github.com/gazeshift3/gazeshift)

---

## 1. One-line Summary

The paper proposes a **large-scale gaze dataset (VRGaze, 2.1M images)** tailored to the off-axis near-eye camera environments of VR headsets, and an **attention-based framework (GazeShift)** that learns gaze representations without labels. Using only unsupervised learning, it achieves an **average error of 1.84°** and enables **5 ms inference** on a VR device GPU.

---

## 2. Background and Motivation

### 2.1 Why is Gaze Estimation Important?

Gaze estimation plays a key role in various fields, including HCI, XR, and assistive technologies. In VR/AR specifically, it enables the following applications:
- **Foveated Rendering**: Maximizes computational efficiency by rendering only the area where the gaze is directed in high resolution.
- **Intuitive Input**: Gaze-based UI interaction.
- **Adaptive Content**: Content adjustment based on user attention.

### 2.2 Limitations of Existing Research

| Problem | Description |
|------|------|
| **Data Scarcity** | Existing VR gaze datasets are mostly on-axis camera-based, failing to reflect the off-axis camera geometry of commercial VR headsets. |
| **Labeling Cost** | Gaze labels are inaccurate and expensive because fixation on intended targets cannot be guaranteed. |
| **Domain Gap** | Research based on remote RGB cameras differs significantly from near-eye infrared (IR) image environments. |
| **Model Size** | Existing unsupervised methods require full-face images or use heavy models. |

---

## 3. Contribution 1: VRGaze Dataset

The first contribution of the paper is **VRGaze**, the first large-scale off-axis VR gaze estimation dataset.

### Dataset Specifications

| Item | Details |
|------|------|
| Number of Images | **2.1 Million** (synchronized left/right eye capture) |
| Participants | **68** |
| Camera | Off-axis near-eye infrared (IR) camera |
| Resolution | 400 × 400 |
| Frame Rate | 30 fps |
| Labels | 2D Point of Regard (PoR) |
| Split | Train 61 / Val 7 |

### Comparison with Existing Datasets

- **OpenEDS2020**: 550K images, 80 participants, but **entirely on-axis** — disconnected from the off-axis geometry of commercial VR headsets.
- **NVGaze**: 2.5M images, but off-axis data is limited to about 260K frames (14 subjects) and lacks gaze direction diversity.
- **TEyeD**: Over 20M images, but not collected in a VR environment, and annotated using computational methods.

![VRGaze Dataset Samples and On-axis vs Off-axis Comparison](/images/gazeshift/_page_3_Figure_0.jpeg)
*Figure 2: Off-axis VRGaze samples (left), on-axis OpenEDS2020 samples (center), VRGaze gaze angle distribution (right). Off-axis cameras are mounted at oblique angles to reduce visual obstruction, but this creates strong perspective distortions, forming a fundamentally different distribution from on-axis data.*

### Data Collection Protocol

- Participants followed a **moving target** on a VR display (alternating pursuit and fixation).
- Background brightness was varied to induce different degrees of **pupil dilation**.
- An average of **7 sessions** recorded per participant, combining deterministic and randomized trajectories.
- Ensured diversity in gender, ethnicity, and age (14 females / 54 males, 29 Asians / 39 Caucasians).

---

## 4. Contribution 2: GazeShift Framework

### 4.1 Core Idea

The core insight of GazeShift is as follows:

> **In cameras mounted on VR headsets, most of the appearance variation between frames of the same eye from the same person is due to changes in gaze direction.**

Under this assumption, a **generative pretext task** is set up to transform a source frame into a target frame. For the model to "redirect" the eye shape of the source image to the target's gaze direction, the gaze direction information must necessarily be encoded in the embedding extracted from the target. In other words, the structure naturally learns gaze representations without explicit labels.

During training, frame pairs (source, target) taken at different times from the same eye of the same subject are used, ensuring that the differences between the source and target are primarily due to gaze changes.

### 4.2 Overall Architecture Overview

Figure 1 below shows the overall architecture of GazeShift. The training pipeline consists of **four main modules**: (1) Gaze Encoder, (2) Appearance Encoder, (3) Attention-based Fusion Module, and (4) Decoder. During inference, only the Gaze Encoder and a lightweight calibration module are used, so the heavy Appearance Encoder and Decoder used during training impose no runtime burden.

![GazeShift Architecture](/images/gazeshift/_page_1_Figure_0.jpeg)
*Figure 1: GazeShift overall architecture. The Gaze Encoder on the left extracts the gaze embedding from the target image, while the Appearance Encoder preserves the spatial appearance features of the source image. The central Self-Attention → Cross-Attention block conditions the appearance features with gaze information, and the Decoder finally reconstructs the gaze-redirected image. The Attention Map at the bottom is recycled as a weight for the Gaze-Focused Loss.*

---

### 4.3 Module ①: Separate Gaze & Appearance Encoders

The first noticeable design decision in GazeShift is the **complete separation of encoders for gaze and appearance**. The previous SOTA, Cross-Encoder (Sun et al., ICCV 2021), processed both attributes simultaneously with a single shared encoder. This structure allowed the decoder to access the target's appearance information, leading to the leakage of appearance information into the gaze embedding.

Given a source frame **x_s** and a target frame **x_t**, the roles of the two encoders are as follows:

**Appearance Encoder** `f_app`:
- Input: Source frame x_s
- Output: Appearance feature map **A_s ∈ ℝ^(H × W × C_a)**
- Characteristic: Designed with a **shallow structure** to preserve the 2D spatial structure of the input image as much as possible.

**Gaze Encoder** `f_gaze`:
- Input: Target frame x_t
- Output: Gaze embedding vector **g_t ∈ ℝ^(C_g)**
- Characteristic: **Lightweight design** based on MobileNetV2's **Inverted Bottleneck Blocks** (342K parameters, 55 MFLOPs).

Expressed mathematically:

$$A_s = f_{\text{app}}(x_s), \quad g_t = f_{\text{gaze}}(x_t) \tag{1}$$
**Why is this asymmetric design important?** The key is that **only the gaze encoder is used during inference**. The appearance encoder and decoder function merely as "scaffolding" during training to improve the quality of the gaze representation, leaving only the lightweight 342K parameter gaze encoder for deployment.

---

### 4.4 Module ②: Gaze-Conditioned Global Modulation

This module corresponds to the Self-Attention → Cross-Attention block in the center of Figure 1 and answers the core question: "How should the appearance of the source be transformed into the gaze direction of the target?" It consists of three main steps.

#### Step 1: Refining Appearance Features via Self-Attention

First, **multi-head self-attention** is applied to the appearance feature map A_s output by the appearance encoder:
$$A_s' = \text{SelfAttn}(A_s) \tag{2}$$
Here, self-attention models spatial interactions within the appearance feature map. For example, the relative positional relationship between the iris area and the eyelid boundary, or the spatial correlation between the pupil reflection (glint) and the iris.

Importantly, the **attention weight map w ∈ ℝ^(H × W)** from this self-attention is not just used for feature refinement, but is recycled as a **soft mask** for gaze-relevant regions in the Gaze-Focused Loss described later.

#### Step 2: Injecting Gaze Information via Cross-Attention

Next, the target's gaze embedding g_t is injected into the appearance features. Key design decisions here:

1. The gaze embedding g_t ∈ ℝ^(C_g) is linearly projected and converted into a **single global query q_g ∈ ℝ^(C_a)**.
2. Cross-attention is performed using this single query q_g as the Query, and the refined appearance features A_s' as the Key and Value:
$$c = \text{CrossAttn}(q_g, A_s', A_s') \tag{3}$$
The output **c ∈ ℝ^(C_a)** is a gaze-conditioned global context vector that encodes "how the source appearance features should be globally adjusted from the perspective of the target gaze direction."

**Why a single query?** Since gaze is a global attribute defined as a single direction for the entire frame, a single global query is sufficient, rather than multiple spatial queries.

#### Step 3: Feature Fusion via Residual Connection

The global context vector c is broadcast to the spatial dimensions H × W to create C ∈ ℝ^(H × W × C_a), which is then added as a residual to the original refined appearance features A_s':
$$F = A_s' + C \tag{4}$$
This residual addition acts as a **feature-wise global modulation**: the gaze direction information (C) is uniformly added across the entire space of the appearance features, "steering" the latent representation toward the target gaze direction. Thanks to the residual connection, the spatial structure of the source is preserved.

**Information Leakage Prevention Mechanism:** In this cross-attention structure, the output of the gaze encoder only participates as a query, and there is no direct path (e.g., skip connection) to the decoder. This architectural isolation acts as a buffer layer, fundamentally blocking the leakage of appearance information into the gaze embedding, which was a problem in Cross-Encoder.

---

### 4.5 Module ③: Gaze-Focused Reconstruction Loss

This is the **most original contribution** of GazeShift and the core component that brought the largest performance improvement in the ablation study (2.07° → 1.84°).

#### The Limit of Uniform Pixel Loss

A standard per-pixel MSE loss treats all pixels equally. However, the area where appearance actually changes with gaze is limited to the region around the iris and pupil, while the eyelid boundaries, skin, and background remain mostly the same. Uniform MSE loss wastes capacity reconstructing unnecessary background and degrades the gaze specificity of the gaze embedding.

Previous studies used external detectors for eye masks or hand-crafted geometric priors, but GazeShift proposes a much more elegant method utilizing the model's internal representations.

#### Solution: Recycling the Self-Attention Map

Core idea: The attention weight map generated in the self-attention step of Section 4.4 is recycled as the spatial weight of the loss function.

![Attention Map Visualization](/images/gazeshift/_page_6_Picture_9.jpeg)
*Figure 4: Source appearance images (top) and corresponding self-attention maps (bottom). The model naturally assigns high attention weights to gaze-relevant regions around the iris and pupil without external supervision.*

After upsampling the attention weight map w to the same resolution as the target image, the **Gaze-Focused Reconstruction Loss** is defined by applying a sharpening parameter γ:
$$\mathcal{L}_{\text{focus}} = \frac{1}{\sum_{i} w_i^{\gamma}} \sum_{i} w_i^{\gamma} \cdot (x_{t,i} - \hat{x}_{t,i})^2 \tag{5}$$

Where:
- **i**: Pixel position index
- **w_i**: Upsampled attention weight (high around the iris, low in the background)
- **γ**: Sharpening parameter that controls the attention focus

**Positive Feedback Loop:**
1. Self-attention focuses on gaze-relevant regions → Reconstruction loss for that region is strengthened
2. The strengthened loss improves the precision of the gaze encoder's gaze representation
3. The more precise gaze representation further improves the localization of the attention's gaze region
4. (Repeats from step 1)

---

### 4.6 Gaze Calibration

After unsupervised pretraining, calibration is performed using a small amount of labeled data:

| Setting | Method | Details |
|------|------|------|
| **VR (Per-person)** | Ridge Regression | Learns a linear regressor with a small number of fixed gaze points (17~60). Per-person calibration is essential to correct for the **κ-angle** (difference between optical and visual axes). Also performed per-session to account for headset repositioning. |
| **Remote Camera** | MLP Regressor | Learns a small MLP by pooling 100~200 labeled samples across all subjects. |

---

## 5. Experimental Results

### 5.1 Performance on VRGaze

| Supervision | Method | Calibration | Avg. Error (°) |
|-----------|------|-------------|--------------|
| Supervised | Appearance Based | - | **1.54** |
| Supervised | Feature Based | - | 3.20 |
| Unsupervised | VAE | Per-person | 5.30 |
| Unsupervised | Cross-Encoder | Per-person | 2.15 |
| **Unsupervised** | **GazeShift** | **Per-person** | **1.84** |
| Unsupervised | Cross-Encoder | Person-agnostic (K=200) | 2.26 |
| **Unsupervised** | **GazeShift** | **Person-agnostic (K=200)** | **2.13** |

**Key Result**: Despite being an unsupervised method, GazeShift achieved **1.84°**, which is close to the supervised appearance-based model (1.54°). This demonstrates that practical accuracy can be reached without gaze labels.

### 5.2 Performance on OpenEDS2020

| Method | Per-person Error (°) | Person-agnostic Error (°) |
|------|---------------------|--------------------------|
| Cross-Encoder | 3.69 | 5.20 |
| **GazeShift** | **3.43** | **4.20** |

GazeShift consistently outperforms Cross-Encoder even in on-axis environments.

### 5.3 Cross-Dataset Generalization (On-axis → Off-axis)

When trained on on-axis data (OpenEDS2020) and tested on off-axis data (VRGaze), the error significantly increases to **5.2°** (compared to 1.84° when trained directly on VRGaze). This clearly shows the **need for dedicated off-axis datasets**.

### 5.4 Remote Camera Experiments (MPIIGaze)

| Supervision | Method | Avg. Error (°) | Parameters | FLOPs |
|-----------|------|--------------|---------|-------|
| Supervised | ResNet-18 | 8.35 | 11M | 75M |
| Unsupervised | Cross-Encoder | 8.32 | 11M | 75M |
| **Unsupervised** | **GazeShift (MobileNetV2)** | **8.00** | **1M** | **2M** |
| Unsupervised | GazeShift (ResNet-18) | **7.56** | 11M | 75M |

**After Unsupervised Fine-tuning (Train + Eval on MPIIGaze)**:

| Method | Avg. Error (°) | Parameters |
|------|--------------|---------|
| Cross-Encoder | 7.20 | 11M |
| **GazeShift (MobileNetV2)** | **7.15** | **1M** |

GazeShift outperforms Cross-Encoder with **10x fewer parameters and 35x fewer FLOPs**.

### 5.5 Ablation Study

| # | Separate Encoders | Attention-Based Redirection | Gaze-Focused Loss | Avg. Error (°) |
|---|-----------|-------------------|-------------|--------------|
| 1 | ✗ | ✗ | ✗ | 2.15 |
| 2 | ✓ | ✗ | ✗ | 2.10 |
| 3 | ✓ | ✓ | ✗ | 2.07 |
| 4 | ✓ | ✓ | ✓ | **1.84** |

**Gaze-Focused Loss** brought the largest performance improvement (2.07° → 1.84°), highlighting the paper's most original contribution.

### 5.6 γ Sensitivity Analysis

| γ | 0.5 | 1.0 | 2.0 | 4.0 |
|---|-----|-----|-----|-----|
| Avg. Error (°) | 2.03 | **1.84** | 2.19 | 2.41 |

- **γ = 1 is optimal**: The most balanced result is achieved when the raw attention is used as is.

### 5.7 Model Efficiency and On-device Performance

| Metric | Value |
|------|------|
| Gaze Encoder Parameters | **342K** |
| FLOPs | **55 MFLOPs** |
| VR Headset Inference Time (Both Eyes) | **5 ms** |
| Chipset | Exynos 2200 (Xclipse 920 GPU) |

### 5.8 Disentanglement Verification

![Disentanglement Analysis](/images/gazeshift/_page_5_Figure_10.jpeg)
*Figure 3: Gaze embedding stability under appearance variation (left), appearance embedding stability under gaze variation (right)*

→ Quantitatively confirmed that GazeShift **effectively disentangles gaze and appearance**.

### 5.9 Latent Space Interpolation

![Latent Space Interpolation](/images/gazeshift/_page_7_Figure_11.jpeg)
*Figure 5: Interpolating between two target gaze embeddings generates smooth eye movement while preserving the source's unique appearance characteristics.*

---

## 6. Limitations

Limitations mentioned by the authors:
- **Constraint of the Assumption**: The assumption that most inter-frame appearance variation is due to gaze changes holds well in VR, but requires further verification in **AR/MR environments** where external lighting and reflections change significantly.
- **Non-gaze Appearance Variations**: Blinks, eyelid movements, and pupil dilation are low-frequency in VR, but their impact in other environments is unknown.

---

## 7. Overall Review and Personal Opinion

### Strengths

1. **Value of Dataset Contribution**: VRGaze is the first large-scale off-axis VR gaze dataset, a crucial resource filling a benchmark void in the field. 
2. **Simple Yet Effective Architecture**: Achieving gaze redirection using only standard attention modules without complex geometric priors or warping fields is impressive.
3. **Originality of Gaze-Focused Loss**: The self-guided mechanism of recycling the model's own attention map as a loss function weight reinforces learning on gaze regions without extra modules.
4. **Practicality**: The lightweight gaze encoder with 342K parameters and 5ms inference is highly suitable for actual VR device deployment.

### Areas for Improvement / Future Work

1. **Dataset Diversity**: The demographic composition of the 68 participants is still somewhat limited.
2. **Limited Comparative Baselines**: The comparison targets in off-axis VR are limited to Cross-Encoder and VAE, although there are practical reasons for this due to a lack of reproducible baselines.
3. **AR/MR Expansion**: The issue of non-gaze appearance variation in AR/MR environments seems to be the most important future research direction.
4. **Binocular Utilization**: While the supervised baseline uses a binocular Siamese network, GazeShift is monocular-based. Utilizing binocular information could yield further performance gains.

### Conclusion

GazeShift is an impressive study that simultaneously achieves the three practical elements of **"unsupervised + lightweight + real-time"** while showing accuracy close to supervised learning. In particular, the design philosophy of achieving gaze-appearance disentanglement with only standard attention mechanisms, and the idea of recycling self-attention maps as loss function weights, presents a general framework applicable to other representation learning problems beyond gaze estimation.

---

> **References** can be found in the References section of the original paper.
