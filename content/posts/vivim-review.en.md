---
title: "[TCSVT 2025] Vivim: Video Vision Mamba for Ultrasound Video Segmentation"
date: 2026-08-07T20:02:00+09:00
draft: false
math: true
tags: ["Paper Review", "Medical Image Segmentation", "State Space Model", "Mamba", "Video Segmentation", "Ultrasound", "TCSVT 2025"]
categories: ["Paper Review"]
summary: "We propose Vivim, a Video Vision Mamba-based framework for ultrasound video segmentation, which models spatiotemporal long-range dependencies with linear complexity compared to Transformers."
cover:
  image: "/images/vivim/overview.jpeg"
  alt: "Overview of Vivim framework"
---

## 1. One-line Summary

We propose Vivim, a framework that integrates Mamba-based Spatiotemporal Selective Scan (ST-Mamba) into a hierarchical Transformer architecture to model spatiotemporal long-range dependencies with linear complexity, while employing a Boundary-Aware Affine Constraint to enhance segmentation accuracy for ambiguous lesion boundaries.

---

## 2. Research Background and Motivation

### 2.1 Problem Definition

Automatic detection and segmentation of lesions and tissues in ultrasound videos are essential for computer-aided clinical examinations. However, segmenting these medical objects is highly challenging due to several inherent factors:

![Figure 1: Main challenges in ultrasound video segmentation](/images/vivim/_page_1_Figure_2.jpeg)
*Figure 1: Main challenges in ultrasound video segmentation. (a) Ambiguous lesion boundaries due to low contrast and speckle noise, (b) inhomogeneous distributions of lesions among patients, and (c) dynamic changes across frames caused by probe movement and soft tissue deformation.*

- Ambiguous lesion boundaries due to low contrast and speckle noise make precise boundary detection difficult.
- Inhomogeneous distributions of lesions across different patient cases and imaging settings complicate model generalization.
- Dynamic changes across frames, introduced by probe movement and soft tissue deformation, create temporal inconsistencies.

Addressing these challenges requires a robust approach that jointly captures spatiotemporal dependencies.

### 2.2 Limitations of Existing Methods

Traditional CNN-based methods struggle to capture global context due to their limited receptive field. While Transformer-based methods capture global information using Multi-Head Self-Attention (MSA), incorporating temporal self-attention triggers a quadratic increase in computational complexity relative to the sequence length. Specifically, for a video visual sequence $\mathbf{K} \in \mathbb{R}^{1 \times T \times M \times D}$, the computational complexity of global self-attention is:

$$\Omega(\text{self-attention}) = 4(TM)D^2 + 2(TM)^2D$$

This quadratic scaling with respect to the total sequence length $TM$ imposes a massive computational burden in medical video analysis, especially when deploying models on hardware with limited memory. For example, DPSTT requires substantial data augmentation and runs slowly, while FLA-Net has a high memory footprint.

### 2.3 Key Contributions

The key contributions of this paper are summarized as follows:

1. We develop Vivim, a medical video segmentation framework consisting of a Mamba-based encoder to obtain a holistic spatiotemporal understanding and a CNN-based decoder to preserve local details. To the best of our knowledge, this is the first work to introduce state space models into ultrasound video scenarios.
2. Instead of a straightforward application of Mamba, we design a Spatiotemporal Selective Scan mechanism (ST-Mamba) to enhance the global perception capability in video sequences.
3. We employ an improved Boundary-Aware Affine Constraint based on the optimization of affine transformations to mitigate ambiguous boundary predictions.
4. We build and release VTUS, the first video ultrasound thyroid segmentation dataset with pixel-level annotations (100 videos, 9,342 frames), enabling systematic benchmarking of ultrasound video segmentation.

---

## 3. Proposed Method: Vivim Framework

### 3.1 Overview of the Architecture

![Figure 3: Overview of the Vivim framework](/images/vivim/_page_3_Figure_2.jpeg)
*Figure 3: (a) Overview of the proposed Vivim for ultrasound video segmentation. The video sequence is processed via patch embedding and multi-scale Temporal Mamba Blocks. Multi-level features are aggregated to predict the segmentation masks using a CNN-based segmentation head. (b) Component blocks of the Temporal Mamba Block. (c) Multi-way spatiotemporal selective scan mechanism of ST-Mamba.*

The Vivim framework consists of two main modules:

- Hierarchical Encoder: Stacked Temporal Mamba Blocks extract coarse and fine-grained feature sequences at different scales.
- CNN-based Segmentation Head: Fuses multi-level feature sequences to predict the final segmentation masks.

Given a video clip $\mathbf{V} = \{I^1, \dots, I^T\}$ with $T$ frames, we divide them into patches of size $4 \times 4$ using overlapped patch embedding. We feed the sequence of patches into the hierarchical Temporal Mamba Encoder to obtain multi-level spatiotemporal features with resolutions of $\{1/4, 1/8, 1/16, 1/32\}$ of the original frame. Finally, these features are aggregated in the CNN-based segmentation head to yield the final mask.

### 3.2 State Space Model Preliminaries

State Space Models (SSMs) map a 1-D function or sequence $x(t) \in \mathbb{R}$ to an output $y(t) \in \mathbb{R}$ through a latent state $h(t) \in \mathbb{R}^N$. This system is formulated using linear ordinary differential equations (ODEs), parameterized by the evolution matrix $\mathbf{A} \in \mathbb{R}^{N \times N}$ and projection matrices $\mathbf{B} \in \mathbb{R}^{N \times 1}$ and $\mathbf{C} \in \mathbb{R}^{1 \times N}$:

$$h'(t) = \mathbf{A}h(t) + \mathbf{B}x(t), \quad y(t) = \mathbf{C}h(t)$$

Here, ODE (Ordinary Differential Equation) represents the rate of change of the latent state over time using derivatives. SSMs use this ODE formulation to model continuous-signal temporal changes in video.

To process this system digitally, continuous parameters are converted to discrete parameters using a timescale parameter $\Delta$ and Zero-Order Hold (ZOH):

$$\overline{\mathbf{A}} = \exp(\Delta \mathbf{A}), \quad \overline{\mathbf{B}} = (\Delta \mathbf{A})^{-1}(\exp(\Delta \mathbf{A}) - \mathbf{I}) \cdot \Delta \mathbf{B}$$

- Timescale parameter delta ($\Delta$): Represents the discrete step size (sampling rate) used to divide the continuous timeline.
- Zero-Order Hold (ZOH): A standard method for analog-to-digital signal conversion. It assumes that the input signal remains constant (Hold) during each step interval $\Delta$, approximating the continuous input as a staircase-like discrete signal.

The discretized version of the SSM equations is:

$$h_t = \overline{\mathbf{A}} h_{t-1} + \overline{\mathbf{B}} x_t, \quad y_t = \mathbf{C} h_t$$

Mamba (S6) introduces a selective scan mechanism to make these parameters input-dependent, allowing the model to capture long-range dependencies with linear complexity.

### 3.3 Temporal Mamba Block

The Temporal Mamba Block captures spatial and temporal dependencies concurrently:

1. Efficient Spatial Self-Attention: Performs initial spatial context aggregation. To improve efficiency, it uses the sequence reduction technique proposed in SegFormer.
- Sequence Reduction: Because self-attention complexity scales quadratically with patch count, this technique downsamples the spatial key and value features before computing self-attention, reducing the overall sequence length and computational cost.
2. ST-Mamba Layer: For the i-th level feature embedding $\mathbf{F}_i \in \mathbb{R}^{T \times C_i \times H \times W}$, we transpose the channel and temporal dimensions and flatten the features into a 1D sequence $\mathbf{h}_i \in \mathbb{R}^{C_i \times THW}$. This sequence is fed to the ST-Mamba module to capture intra- and inter-frame long-range dependencies.
3. Detail-Specific Feedforward (DSF): Integrates a $3 \times 3 \times 3$ depth-wise convolution into the feedforward network to preserve fine-grained local details.

The equations for the stacked Mamba layers are defined as (where $l \in [1, N_m]$):

$$h^l = \text{ST-Mamba}(\text{LN}(h^{l-1})) + h^{l-1}$$
$$h^l = \text{DSF}(\text{LN}(h^l)) + h^l$$

### 3.4 Spatiotemporal Selective Scan (ST-Mamba)

![Figure 4: Spatiotemporal selective scan mechanism](/images/vivim/_page_4_Figure_2.jpeg)
*Figure 4: Illustration of the spatiotemporal selective scan, which includes temporal forward scan, temporal backward scan, and spatial scan.*

The causal nature of S6 fits sequential temporal data, but video frames contain non-causal 2D spatial structures alongside redundant temporal data. To address this, ST-Mamba scans in three directions in parallel:

- Temporal Forward Scan: Unfolds the patches of each frame along rows and columns, concatenating them in chronological order to form $\mathbf{h}_i^t \in \mathbb{R}^{C_i \times T(HW)}$.
- Temporal Backward Scan: Scans the same sequence in reverse order to capture bidirectional temporal cues.
- Spatial Scan: Stacks the patches along the temporal axis to form a spatial-first sequence $\mathbf{h}_i^s \in \mathbb{R}^{C_i \times (HW)T}$, integrating the state of each pixel across all frames.

This tri-directional scanning balances single-frame spatial coherence and cross-frame temporal continuity under a linear complexity budget:

$$\Omega(\text{SSM}) = 4(TM)(2D)N + (TM)(2D)N^2$$

where the default expansion ratio is 2 and $N$ is set to 16. While self-attention scales quadratically with sequence length $TM$, SSM remains linear.

### 3.5 Boundary-Aware Affine Constraint

![Figure 5: Overview of the training strategy](/images/vivim/_page_4_Figure_4.jpeg)
*Figure 5: Training strategy overview. The patch-level boundary-aware affine constraint L_affine optimizes the model alongside the segmentation loss L_seg and the boundary cross-entropy loss L_bce.*

Training solely with segmentation loss can result in ambiguous, unstructured predictions. To mitigate this, we introduce a patch-level boundary-aware affine constraint inspired by InverseForm to enforce proper boundary structures.

Specifically, we apply the Sobel operator to the ground-truth mask to extract edge boundaries.
- Sobel Operator: An edge-detection filter that calculates the gradient of image intensity at each pixel, highlighting areas of rapid brightness changes to define boundaries.

An auxiliary boundary head, consisting of 3 convolutional layers, processes the feature patches from the Mamba encoder to generate predicted edges.
- Auxiliary Boundary Head: A small sub-network used only during training. It encourages the encoder to retain sharp boundary details by explicitly predicting outline edges. It is discarded during inference.

We calculate the affine transformation matrices between the predicted and ground-truth edges using a pre-trained MLP:
- Affine Transformation: A geometric transformation (e.g., translation, rotation, scaling, shearing) that preserves lines and parallelism.
- Affine Constraint: We measure how much the predicted edges are distorted relative to the ground-truth edges by estimating an affine transformation matrix, and then penalize the model to drive this matrix toward the identity matrix $\mathbb{I}$ (representing zero distortion).

Specifically, we compute:
- $\hat{\theta}_i^t$: The affine matrix between the GT edge $B_{\text{gt}}^t$ and predicted edge $B_{\text{pred}}^t$ of the target frame $I^t$.
- $\hat{\theta}_i^1$: The affine matrix between the GT edge $B_{\text{gt}}^1$ of the first frame $I^1$ and predicted edge $B_{\text{pred}}^t$ of the target frame $I^t$.

The affine constraint loss is defined as:

$$\mathcal{L}_{\text{affine}} = \frac{1}{N_p} \sum_{i=1}^{N_p} \left( \Delta_1 \left| \hat{\theta}_i^t - \mathbb{I} \right|_F - \Delta_2 \left| \hat{\theta}_i^1 - \mathbb{I} \right|_F \right)$$

where $N_p$ is the number of patches, $|\cdot|_F$ is the Frobenius norm, $\Delta_1=1.00$, and $\Delta_2=0.01$. This loss pushes the predicted boundaries toward the correct ground-truth shape while preserving minor cross-frame differences.

The overall objective function is:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{seg}} + \lambda_1 \mathcal{L}_{\text{affine}} + \lambda_2 \mathcal{L}_{\text{bce}}$$

where $\lambda_1 = \lambda_2 = 0.3$.

### 3.6 Decoder

Multi-level features $\{F_1, F_2, F_3, F_4\}$ are passed to an MLP layer to unify channel dimensions. These features are then upsampled, concatenated, fused via another MLP, and finally projected to a segmentation mask $M$ through a $1 \times 1$ convolution.

---

## 4. Experimental Results

### 4.1 Datasets and Implementation Details

We evaluate Vivim on three medical video datasets:
- VTUS (Thyroid Ultrasound): Our newly collected dataset. It contains 100 B-mode video sequences (9,342 frames) annotated by three experts. Split into 70 training and 30 testing videos.
- BUV2022 (Breast Ultrasound): 63 video sequences (4,619 frames) cropped to a resolution of $300 \times 200$.
- CVC-300, CVC-612, ASU-Mayo (Polyp Colonoscopy): Publicly available benchmarks.

Training is performed on a single NVIDIA RTX 4090 GPU for 100 epochs using the Adam optimizer (learning rate $10^{-4} \to 10^{-6}$) and $256 \times 256$ input resolution.

### 4.2 Thyroid and Breast Lesion Segmentation Results

| Methods | Venue | Type | VTUS Dice | VTUS Jaccard | BUV2022 Dice | BUV2022 Jaccard | FPS |
|---|---|---|---|---|---|---|---|
| UNet | MICCAI15 | image | 0.6662 | 0.5328 | 0.7303 | 0.6247 | 88.18 |
| UNet++ | DLMIA18 | image | 0.7656 | 0.6486 | 0.7179 | 0.6124 | 40.90 |
| TransUNet | arXiv21 | image | 0.7461 | 0.6250 | 0.6547 | 0.5358 | 65.10 |
| SETR | CVPR21 | image | 0.7288 | 0.6010 | 0.6649 | 0.5480 | 21.61 |
| DPSTT | MICCAI22 | video | 0.8063 | 0.7117 | 0.8255 | 0.7364 | 30.50 |
| FLA-Net | MICCAI23 | video | 0.8042 | 0.7075 | 0.8232 | 0.7315 | 31.22 |
| MemSAM | CVPR24 | video | 0.7922 | 0.7101 | 0.8149 | 0.7092 | 10.42 |
| Vivim (Ours) | — | video | 0.8324 | 0.7391 | 0.8356 | 0.7450 | 35.33 |

*Table I: Quantitative comparisons on VTUS and BUV2022 datasets. Vivim achieves the highest scores across all metrics.*

Vivim outperforms the second-best method (DPSTT) by +2.61% Dice and +2.74% Jaccard on VTUS, and by +1.01% Dice and +0.86% Jaccard on BUV2022, while running at the fastest speed (35.33 FPS) among all video-based methods.

![Figure 6: Visual thyroid segmentation comparison](/images/vivim/_page_7_Figure_2.jpeg)
*Figure 6: Visual results on thyroid segmentation. Vivim defines and segments boundaries more accurately.*

### 4.3 Video Polyp Segmentation Results

On CVC-300-TV, Vivim achieves a maxDice of 0.901 (+2.7% margin) and a maxIoU of 0.831 (+2.2% margin). It also consistently outperforms existing SOTAs on CVC-612-V and CVC-612-T.

![Figure 7: Visual polyp segmentation comparison](/images/vivim/_page_7_Figure_6.jpeg)
*Figure 7: Qualitative results on CVC-612-T. Vivim segments polyps with much cleaner boundaries.*

### 4.4 Ablation Study

| Config | Tf | Tb | S | BAC | Dice | Jaccard |
|---|---|---|---|---|---|---|
| basic (SegFormer) | - | - | - | - | 0.8144 | 0.7188 |
| C1 (+Forward Temp SSM) | check | - | - | - | 0.8159 | 0.7216 |
| C2 (+Bidirectional Temp SSM) | check | check | - | - | 0.8213 | 0.7264 |
| C3 (+Spatial SSM) | check | check | check | - | 0.8259 | 0.7310 |
| Vivim (full) | check | check | check | check | 0.8324 | 0.7391 |

*Table III: Ablation study on VTUS dataset. Tf: Forward Temporal SSM, Tb: Backward Temporal SSM, S: Spatial SSM, BAC: Boundary-Aware Affine Constraint.*

Key ablation findings:
- Adding Forward Temporal SSM (C1) improves basic SegFormer performance, validating the utility of SSMs in video contexts.
- Bidirectional Temporal SSMs (C2) enhance cross-frame temporal coherence over unidirectional models.
- Incorporating Spatial SSM (C3) to handle non-causal details yields a major boost (+0.46% Dice, +0.83% Recall).
- The Boundary-Aware Affine Constraint (BAC) further refines boundary shapes and boosts final metrics.

### 4.5 Efficiency Analysis of ST-Mamba

| Methods | Core Module | TM (M) | IM (M) | Run-time (s) | Is Global |
|---|---|---|---|---|---|
| M1 | Spatiotemporal Self-attention | OOM | - | - | check |
| M2 | Spatiotemporal Window Self-attention | 25,861 | 7,795 | 0.142 | ✗ |
| M3 | Spatiotemporal Factorized Self-attention | 29,110 | 9,288 | 0.156 | ✗ |
| Vivim | Spatiotemporal Mamba | 19,216 | 5,112 | 0.121 | check |

*Table IV: Comparison of different attention modules with a 32-frame video input at 256p. TM: Training Memory, IM: Inference Memory.*

![Figure 8: Performance and memory scaling](/images/vivim/_page_8_Figure_11.jpeg)
*Figure 8: (a) Dice score change over increasing frame counts — ST-Mamba continues to improve, whereas self-attention degrades. (b) Memory scaling comparison — Vivim allows inference on over 150 frames with a single GPU.*

Global spatiotemporal self-attention (M1) triggers OOM errors on 32-frame inputs. In contrast, Vivim's ST-Mamba handles global dependencies using significantly less memory (19,216M Training, 5,112M Inference) and completes runs faster (0.121s) than localized attention window approximations (M2, M3).

---

## 5. Key Contributions Summary

- Pioneer in SSM-based Medical Video Segmentation: Vivim is the first work to apply State Space Models (Mamba) to ultrasound video segmentation. By replacing the computationally heavy quadratic-complexity self-attention with a linear-complexity Mamba encoder, Vivim allows efficient joint spatiotemporal modeling within tight memory constraints.

- Tri-directional Spatiotemporal Selective Scan: To adapt the causal selective scan of S6 to non-causal visual dimensions, we design a parallel tri-directional scan (temporal forward, temporal backward, and spatial). This scan mechanism captures both cross-frame continuity and single-frame structural layouts.

- Boundary-Aware Affine Regularization: We propose a boundary regularization loss that maps predicted outlines to ground-truth edges using affine matrices, driving them toward zero geometric distortion. This method resolves blurry boundary issues inherent in ultrasound imaging.

- First Video Thyroid Segmentation Dataset (VTUS): We address the lack of public annotated datasets by building and releasing VTUS, which contains 100 video sequences (9,342 frames) annotated by multiple clinical experts, laying down a solid foundation for future research in video-based medical imaging.
