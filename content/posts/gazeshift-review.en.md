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

> <b>Paper Information</b>
> - <b>Title:</b> GazeShift: Unsupervised Gaze Estimation and Dataset for VR
> - <b>Authors:</b> Gil Shapira, Ishay Goldin, Evgeny Artyomov, Donghoon Kim, Yosi Keller, Niv Zehngut
> - <b>Affiliations:</b> Samsung Semiconductor Israel R&D Center (SIRC), Samsung Electronics, Bar-Ilan University
> - <b>Conference:</b> CVPR 2026
> - <b>Code & Dataset:</b> [github.com/gazeshift3/gazeshift](https://github.com/gazeshift3/gazeshift)

---

## 1. One-line Summary

The paper proposes a <b>large-scale gaze dataset (VRGaze, 2.1M images)</b> tailored to the off-axis near-eye camera environments of VR headsets, and an <b>attention-based framework (GazeShift)</b> that learns gaze representations without labels. Using only unsupervised learning, it achieves an <b>average error of 1.84°</b> and enables <b>5 ms inference</b> on a VR device GPU.

---

## 2. Background and Motivation

### 2.1 Why is Gaze Estimation Important?

Gaze estimation plays a key role in various fields, including HCI, XR, and assistive technologies. In VR/AR specifically, it enables the following applications:
- <b>Foveated Rendering</b>: Maximizes computational efficiency by rendering only the area where the gaze is directed in high resolution.
- <b>Intuitive Input</b>: Gaze-based UI interaction.
- <b>Adaptive Content</b>: Content adjustment based on user attention.

### 2.2 Limitations of Existing Research

| Problem | Description |
|------|------|
| <b>Data Scarcity</b> | Existing VR gaze datasets are mostly on-axis camera-based, failing to reflect the off-axis camera geometry of commercial VR headsets. |
| <b>Labeling Cost</b> | Gaze labels are inaccurate and expensive because fixation on intended targets cannot be guaranteed. |
| <b>Domain Gap</b> | Research based on remote RGB cameras differs significantly from near-eye infrared (IR) image environments. |
| <b>Model Size</b> | Existing unsupervised methods require full-face images or use heavy models. |

---

## 3. Contribution 1: VRGaze Dataset

The first contribution of the paper is <b>VRGaze</b>, the first large-scale off-axis VR gaze estimation dataset.

### Dataset Specifications

| Item | Details |
|------|------|
| Number of Images | <b>2.1 Million</b> (synchronized left/right eye capture) |
| Participants | <b>68</b> |
| Camera | Off-axis near-eye infrared (IR) camera |
| Resolution | 400 × 400 |
| Frame Rate | 30 fps |
| Labels | 2D Point of Regard (PoR) |
| Split | Train 61 / Val 7 |

### Comparison with Existing Datasets

- <b>OpenEDS2020</b>: 550K images, 80 participants, but <b>entirely on-axis</b> — disconnected from the off-axis geometry of commercial VR headsets.
- <b>NVGaze</b>: 2.5M images, but off-axis data is limited to about 260K frames (14 subjects) and lacks gaze direction diversity.
- <b>TEyeD</b>: Over 20M images, but not collected in a VR environment, and annotated using computational methods.

![VRGaze Dataset Samples and On-axis vs Off-axis Comparison](/images/gazeshift/_page_3_Figure_0.jpeg)
*Figure 2: Off-axis VRGaze samples (left), on-axis OpenEDS2020 samples (center), VRGaze gaze angle distribution (right). Off-axis cameras are mounted at oblique angles to reduce visual obstruction, but this creates strong perspective distortions, forming a fundamentally different distribution from on-axis data.*

### Data Collection Protocol

- Participants followed a <b>moving target</b> on a VR display (alternating pursuit and fixation).
- Background brightness was varied to induce different degrees of <b>pupil dilation</b>.
- An average of <b>7 sessions</b> recorded per participant, combining deterministic and randomized trajectories.
- Ensured diversity in gender, ethnicity, and age (14 females / 54 males, 29 Asians / 39 Caucasians).

---

## 4. Contribution 2: GazeShift Framework

### 4.1 Core Idea

The core insight of GazeShift is as follows:

> <b>In cameras mounted on VR headsets, most of the appearance variation between frames of the same eye from the same person is due to changes in gaze direction.</b>

Under this assumption, a <b>generative pretext task</b> is set up to transform a source frame into a target frame. For the model to "redirect" the eye shape of the source image to the target's gaze direction, the gaze direction information must necessarily be encoded in the embedding extracted from the target. In other words, the structure naturally learns gaze representations without explicit labels.

During training, frame pairs (source, target) taken at different times from the same eye of the same subject are used, ensuring that the differences between the source and target are primarily due to gaze changes.

### 4.2 Overall Architecture Overview

Figure 1 below shows the overall architecture of GazeShift. The training pipeline consists of <b>four main modules</b>: (1) Gaze Encoder, (2) Appearance Encoder, (3) Attention-based Fusion Module, and (4) Decoder. During inference, only the Gaze Encoder and a lightweight calibration module are used, so the heavy Appearance Encoder and Decoder used during training impose no runtime burden.

![GazeShift Architecture](/images/gazeshift/_page_1_Figure_0.jpeg)
*Figure 1: GazeShift overall architecture. The Gaze Encoder on the left extracts the gaze embedding from the target image, while the Appearance Encoder preserves the spatial appearance features of the source image. The central Self-Attention → Cross-Attention block conditions the appearance features with gaze information, and the Decoder finally reconstructs the gaze-redirected image. The Attention Map at the bottom is recycled as a weight for the Gaze-Focused Loss.*

---

### 4.3 Module ①: Separate Gaze & Appearance Encoders

The first noticeable design decision in GazeShift is the <b>complete separation of encoders for gaze and appearance</b>. The previous SOTA, Cross-Encoder (Sun et al., ICCV 2021), processed both attributes simultaneously with a single shared encoder. This structure allowed the decoder to access the target's appearance information, leading to the leakage of appearance information into the gaze embedding.

Given a source frame <b>x_s</b> and a target frame <b>x_t</b>, the roles of the two encoders are as follows:

<b>Appearance Encoder</b> `f_app`:
- Input: Source frame x_s
- Output: Appearance feature map <b>A_s ∈ ℝ^(H × W × C_a)</b>
- Characteristic: Designed with a <b>shallow structure</b> to preserve the 2D spatial structure of the input image as much as possible.

<b>Gaze Encoder</b> `f_gaze`:
- Input: Target frame x_t
- Output: Gaze embedding vector <b>g_t ∈ ℝ^(C_g)</b>
- Characteristic: <b>Lightweight design</b> based on MobileNetV2's <b>Inverted Bottleneck Blocks</b> (342K parameters, 55 MFLOPs).

Expressed mathematically:

$$A_s = f_{\text{app}}(x_s), \quad g_t = f_{\text{gaze}}(x_t) \tag{1}$$
<b>Why is this asymmetric design important?</b> The key is that <b>only the gaze encoder is used during inference</b>. The appearance encoder and decoder function merely as "scaffolding" during training to improve the quality of the gaze representation, leaving only the lightweight 342K parameter gaze encoder for deployment.

---

### 4.4 Module ②: Gaze-Conditioned Global Modulation

This module corresponds to the Self-Attention → Cross-Attention block in the center of Figure 1 and answers the core question: "How should the appearance of the source be transformed into the gaze direction of the target?" It consists of three main steps.

#### Step 1: Refining Appearance Features via Self-Attention

First, <b>multi-head self-attention</b> is applied to the appearance feature map A_s output by the appearance encoder:
$$A_s' = \text{SelfAttn}(A_s) \tag{2}$$
Here, self-attention models spatial interactions within the appearance feature map. For example, the relative positional relationship between the iris area and the eyelid boundary, or the spatial correlation between the pupil reflection (glint) and the iris.

Importantly, the <b>attention weight map w ∈ ℝ^(H × W)</b> from this self-attention is not just used for feature refinement, but is recycled as a <b>soft mask</b> for gaze-relevant regions in the Gaze-Focused Loss described later.

#### Step 2: Injecting Gaze Information via Cross-Attention

Next, the target's gaze embedding g_t is injected into the appearance features. Key design decisions here:

1. The gaze embedding g_t ∈ ℝ^(C_g) is linearly projected and converted into a <b>single global query q_g ∈ ℝ^(C_a)</b>.
2. Cross-attention is performed using this single query q_g as the Query, and the refined appearance features A_s' as the Key and Value:
$$c = \text{CrossAttn}(q_g, A_s', A_s') \tag{3}$$
The output <b>c ∈ ℝ^(C_a)</b> is a gaze-conditioned global context vector that encodes "how the source appearance features should be globally adjusted from the perspective of the target gaze direction."

<b>Why a single query?</b> Since gaze is a global attribute defined as a single direction for the entire frame, a single global query is sufficient, rather than multiple spatial queries.

#### Step 3: Feature Fusion via Residual Connection

The global context vector c is broadcast to the spatial dimensions H × W to create C ∈ ℝ^(H × W × C_a), which is then added as a residual to the original refined appearance features A_s':
$$F = A_s' + C \tag{4}$$
This residual addition acts as a <b>feature-wise global modulation</b>: the gaze direction information (C) is uniformly added across the entire space of the appearance features, "steering" the latent representation toward the target gaze direction. Thanks to the residual connection, the spatial structure of the source is preserved.

<b>Information Leakage Prevention Mechanism:</b> In this cross-attention structure, the output of the gaze encoder only participates as a query, and there is no direct path (e.g., skip connection) to the decoder. This architectural isolation acts as a buffer layer, fundamentally blocking the leakage of appearance information into the gaze embedding, which was a problem in Cross-Encoder.

---

### 4.5 Module ③: Gaze-Focused Reconstruction Loss

This is the <b>most original contribution</b> of GazeShift and the core component that brought the largest performance improvement in the ablation study (2.07° → 1.84°).

#### The Limit of Uniform Pixel Loss

A standard per-pixel MSE loss treats all pixels equally. However, the area where appearance actually changes with gaze is limited to the region around the iris and pupil, while the eyelid boundaries, skin, and background remain mostly the same. Uniform MSE loss wastes capacity reconstructing unnecessary background and degrades the gaze specificity of the gaze embedding.

Previous studies used external detectors for eye masks or hand-crafted geometric priors, but GazeShift proposes a much more elegant method utilizing the model's internal representations.

#### Solution: Recycling the Self-Attention Map

Core idea: The attention weight map generated in the self-attention step of Section 4.4 is recycled as the spatial weight of the loss function.

![Attention Map Visualization](/images/gazeshift/_page_6_Picture_9.jpeg)
*Figure 4: Source appearance images (top) and corresponding self-attention maps (bottom). The model naturally assigns high attention weights to gaze-relevant regions around the iris and pupil without external supervision.*

After upsampling the attention weight map w to the same resolution as the target image, the <b>Gaze-Focused Reconstruction Loss</b> is defined by applying a sharpening parameter γ:
$$\mathcal{L}_{\text{focus}} = \frac{1}{\sum_{i} w_i^{\gamma}} \sum_{i} w_i^{\gamma} \cdot (x_{t,i} - \hat{x}_{t,i})^2 \tag{5}$$

Where:
- <b>i</b>: Pixel position index
- <b>w_i</b>: Upsampled attention weight (high around the iris, low in the background)
- <b>γ</b>: Sharpening parameter that controls the attention focus

<b>Positive Feedback Loop:</b>
1. Self-attention focuses on gaze-relevant regions → Reconstruction loss for that region is strengthened
2. The strengthened loss improves the precision of the gaze encoder's gaze representation
3. The more precise gaze representation further improves the localization of the attention's gaze region
4. (Repeats from step 1)

---

### 4.6 Gaze Calibration

After unsupervised pretraining, calibration is performed using a small amount of labeled data:

| Setting | Method | Details |
|------|------|------|
| <b>VR (Per-person)</b> | Ridge Regression | Learns a linear regressor with a small number of fixed gaze points (17~60). Per-person calibration is essential to correct for the <b>κ-angle</b> (difference between optical and visual axes). Also performed per-session to account for headset repositioning. |
| <b>Remote Camera</b> | MLP Regressor | Learns a small MLP by pooling 100~200 labeled samples across all subjects. |

---

## 5. Experimental Results

### 5.1 Performance on VRGaze

| Supervision | Method | Calibration | Avg. Error (°) |
|-----------|------|-------------|--------------|
| Supervised | Appearance Based | - | <b>1.54</b> |
| Supervised | Feature Based | - | 3.20 |
| Unsupervised | VAE | Per-person | 5.30 |
| Unsupervised | Cross-Encoder | Per-person | 2.15 |
| <b>Unsupervised</b> | <b>GazeShift</b> | <b>Per-person</b> | <b>1.84</b> |
| Unsupervised | Cross-Encoder | Person-agnostic (K=200) | 2.26 |
| <b>Unsupervised</b> | <b>GazeShift</b> | <b>Person-agnostic (K=200)</b> | <b>2.13</b> |

<b>Key Result</b>: Despite being an unsupervised method, GazeShift achieved <b>1.84°</b>, which is close to the supervised appearance-based model (1.54°). This demonstrates that practical accuracy can be reached without gaze labels.

### 5.2 Performance on OpenEDS2020

| Method | Per-person Error (°) | Person-agnostic Error (°) |
|------|---------------------|--------------------------|
| Cross-Encoder | 3.69 | 5.20 |
| <b>GazeShift</b> | <b>3.43</b> | <b>4.20</b> |

GazeShift consistently outperforms Cross-Encoder even in on-axis environments.

### 5.3 Cross-Dataset Generalization (On-axis → Off-axis)

When trained on on-axis data (OpenEDS2020) and tested on off-axis data (VRGaze), the error significantly increases to <b>5.2°</b> (compared to 1.84° when trained directly on VRGaze). This clearly shows the <b>need for dedicated off-axis datasets</b>.

### 5.4 Remote Camera Experiments (MPIIGaze)

| Supervision | Method | Avg. Error (°) | Parameters | FLOPs |
|-----------|------|--------------|---------|-------|
| Supervised | ResNet-18 | 8.35 | 11M | 75M |
| Unsupervised | Cross-Encoder | 8.32 | 11M | 75M |
| <b>Unsupervised</b> | <b>GazeShift (MobileNetV2)</b> | <b>8.00</b> | <b>1M</b> | <b>2M</b> |
| Unsupervised | GazeShift (ResNet-18) | <b>7.56</b> | 11M | 75M |

<b>After Unsupervised Fine-tuning (Train + Eval on MPIIGaze)</b>:

| Method | Avg. Error (°) | Parameters |
|------|--------------|---------|
| Cross-Encoder | 7.20 | 11M |
| <b>GazeShift (MobileNetV2)</b> | <b>7.15</b> | <b>1M</b> |

GazeShift outperforms Cross-Encoder with <b>10x fewer parameters and 35x fewer FLOPs</b>.

### 5.5 Ablation Study

| # | Separate Encoders | Attention-Based Redirection | Gaze-Focused Loss | Avg. Error (°) |
|---|-----------|-------------------|-------------|--------------|
| 1 | ✗ | ✗ | ✗ | 2.15 |
| 2 | ✓ | ✗ | ✗ | 2.10 |
| 3 | ✓ | ✓ | ✗ | 2.07 |
| 4 | ✓ | ✓ | ✓ | <b>1.84</b> |

<b>Gaze-Focused Loss</b> brought the largest performance improvement (2.07° → 1.84°), highlighting the paper's most original contribution.

### 5.6 γ Sensitivity Analysis

| γ | 0.5 | 1.0 | 2.0 | 4.0 |
|---|-----|-----|-----|-----|
| Avg. Error (°) | 2.03 | <b>1.84</b> | 2.19 | 2.41 |

- <b>γ = 1 is optimal</b>: The most balanced result is achieved when the raw attention is used as is.

### 5.7 Model Efficiency and On-device Performance

| Metric | Value |
|------|------|
| Gaze Encoder Parameters | <b>342K</b> |
| FLOPs | <b>55 MFLOPs</b> |
| VR Headset Inference Time (Both Eyes) | <b>5 ms</b> |
| Chipset | Exynos 2200 (Xclipse 920 GPU) |

### 5.8 Disentanglement Verification

![Disentanglement Analysis](/images/gazeshift/_page_5_Figure_10.jpeg)
*Figure 3: Gaze embedding stability under appearance variation (left), appearance embedding stability under gaze variation (right)*

→ Quantitatively confirmed that GazeShift <b>effectively disentangles gaze and appearance</b>.

### 5.9 Latent Space Interpolation

![Latent Space Interpolation](/images/gazeshift/_page_7_Figure_11.jpeg)
*Figure 5: Interpolating between two target gaze embeddings generates smooth eye movement while preserving the source's unique appearance characteristics.*

---

## 6. Limitations

Limitations mentioned by the authors:
- <b>Constraint of the Assumption</b>: The assumption that most inter-frame appearance variation is due to gaze changes holds well in VR, but requires further verification in <b>AR/MR environments</b> where external lighting and reflections change significantly.
- <b>Non-gaze Appearance Variations</b>: Blinks, eyelid movements, and pupil dilation are low-frequency in VR, but their impact in other environments is unknown.

---

## 7. Overall Review and Personal Opinion

### Strengths

1. <b>Value of Dataset Contribution</b>: VRGaze is the first large-scale off-axis VR gaze dataset, a crucial resource filling a benchmark void in the field. 
2. <b>Simple Yet Effective Architecture</b>: Achieving gaze redirection using only standard attention modules without complex geometric priors or warping fields is impressive.
3. <b>Originality of Gaze-Focused Loss</b>: The self-guided mechanism of recycling the model's own attention map as a loss function weight reinforces learning on gaze regions without extra modules.
4. <b>Practicality</b>: The lightweight gaze encoder with 342K parameters and 5ms inference is highly suitable for actual VR device deployment.

### Areas for Improvement / Future Work

1. <b>Dataset Diversity</b>: The demographic composition of the 68 participants is still somewhat limited.
2. <b>Limited Comparative Baselines</b>: The comparison targets in off-axis VR are limited to Cross-Encoder and VAE, although there are practical reasons for this due to a lack of reproducible baselines.
3. <b>AR/MR Expansion</b>: The issue of non-gaze appearance variation in AR/MR environments seems to be the most important future research direction.
4. <b>Binocular Utilization</b>: While the supervised baseline uses a binocular Siamese network, GazeShift is monocular-based. Utilizing binocular information could yield further performance gains.

### Conclusion

GazeShift is an impressive study that simultaneously achieves the three practical elements of <b>"unsupervised + lightweight + real-time"</b> while showing accuracy close to supervised learning. In particular, the design philosophy of achieving gaze-appearance disentanglement with only standard attention mechanisms, and the idea of recycling self-attention maps as loss function weights, presents a general framework applicable to other representation learning problems beyond gaze estimation.

---

> <b>References</b> can be found in the References section of the original paper.
