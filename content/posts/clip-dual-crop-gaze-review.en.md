---
title: "[GAZE 2026] Learning to Look: CLIP-Guided Dual-Crop Fusion for Head Position-Invariant Gaze Estimation"
date: 2026-08-28T15:10:14+09:00
draft: false
math: true
tags: ["Paper Review", "Gaze Estimation", "CLIP", "Fusion", "CVPR 2026"]
categories: ["Paper Review"]
summary: "A review of the Mercedes-Benz R&D paper presented at the CVPR 2026 GAZE Workshop. The authors present a dual-stream architecture with Fourier head position embeddings, a learnable pinhole coordinate transformation, and CLIP-guided dynamic fusion to achieve robust head position-invariant 3D gaze estimation across diverse benchmarks."
cover:
  image: "/images/clip-dual-crop-gaze/fig2_architecture.jpeg"
  alt: "Learning to Look Architecture Overview"
---

> Paper Information
> - Title: Learning to Look: CLIP-Guided Dual-Crop Fusion for Head Position-Invariant Gaze Estimation
> - Authors: Sourav Lakhotia, Chaviti Vasantha Lakshmi, Aratrik Chattopadhyay
> - Affiliation: Mercedes-Benz Research and Development, Karnataka, India
> - Venue: The 7th International Workshop on Eye and Gaze in Computer Vision (GAZE 2026) at CVPR 2026

![Learning to Look Architecture Overview](/images/clip-dual-crop-gaze/fig2_architecture.jpeg)

---

## 1. One-Sentence Summary

This paper presents a dual-stream gaze estimation framework combining eye crops and full-face crops with Fourier head position embeddings, a learnable pinhole coordinate transformation, and CLIP-guided vision-language feature fusion, achieving 6.3% to 39.3% error reductions over state-of-the-art baselines across three major benchmarks (IVGaze, GazeGene, and MPIIFaceGaze).

---

## 2. Research Background and Motivation

### 2.1 Problem Definition

Appearance-based 3D gaze estimation aims to directly regress 3D gaze vectors from facial or ocular images. It serves as a foundational building block for Driver Monitoring Systems (DMS), gaze-contingent Human-Computer Interaction (HCI), and driver behavior analytics in modern intelligent vehicles.

However, in-cabin automotive settings present exceptionally challenging conditions for visual gaze estimation models:

First, drastic illumination shifts, strong shadows, and direct solar glare frequently degrade image fidelity depending on the time of day and driving trajectory.
Second, severe occlusions frequently arise from optical glasses reflections, sunglasses, protective masks, and headwear.
Third, drivers continuously shift their 3D head position and rotate their heads across wide ranges, resulting in dynamically varying geometric arrangements between the camera and the subject.

![Figure 1](/images/clip-dual-crop-gaze/fig1_qualitative_teaser.jpeg)

*Figure 1: Qualitative gaze estimation results across challenging scenarios including glasses reflection, mask occlusion, and extreme gaze angles (lizard gaze). Green arrows denote ground truth, blue arrows denote the proposed method (Ours), and red arrows represent the baseline. Angular errors in yaw and pitch (in degrees) are indicated below each image.*

Early laboratory benchmarks (e.g., MPIIFaceGaze, GazeCapture) were collected under constrained head positions and controlled lighting, failing to reflect the harsh variability of automotive environments. The recent release of IVGaze—featuring 44,705 infrared (IR) in-cabin driving images across 125 subjects—has catalyzed rigorous research into real-world in-cabin gaze estimation under challenging conditions.

### 2.2 Limitations of Existing Methods and Related Work

Existing appearance-based gaze estimation approaches suffer from fundamental trade-offs in input cropping and structural vulnerabilities in coordinate normalization.

#### 1. Input Granularity Trade-Off and the Multi-Mapping Dilemma

Prior methods generally adopt one of two input configurations:

- Full-face Input: While full-face models capture macroscopic facial geometry and head orientation effectively, the crucial local ocular features (pupil and iris rotation) become diluted at lower relative resolutions, hindering fine-grained angular precision.
- Eye-crop Input: Focusing exclusively on cropped eye regions preserves high-resolution ocular details. However, it strips away the 3D relative positioning and global facial context.

The most critical defect of eye-crop models is the **multi-mapping dilemma** (many-to-one and one-to-many mappings). Physical gaze direction is the composite result of eyeball orientation and 3D head pose. Under varying 3D head positions and rotations, the exact same absolute gaze direction projects onto the 2D image sensor as distinctly different eye appearances.

![Figure 4](/images/clip-dual-crop-gaze/fig4_multi_mapping.jpeg)

*Figure 4: Illustration of the multi-mapping problem in the IVGaze dataset. Due to variations in 3D head orientation, identical 3D gaze directions produce noticeably different 2D eye appearances as the eyeball center and iris boundaries shift relative to the eye corner landmarks.*

Conversely, visually identical eye crops can correspond to entirely different absolute 3D gaze directions depending on head pose. This non-bijective relationship destabilizes naive regression models during training.

#### 2. Vulnerabilities of Traditional 3D Gaze Normalization

To eliminate multi-mapping, classical pipelines apply 3D geometric gaze normalization, notably Zhang's MPIIGaze normalization and IVGaze axis normalization. These methods estimate the 3D Head Position (HP) and Head Rotation from facial landmarks or external head pose estimators, construct a virtual normalized camera coordinate frame looking directly at the eye center, and perspective-warp the input image accordingly.

Despite its theoretical appeal, traditional normalization exhibits severe failure modes in practice:

- Extreme Sensitivity to HP Estimation Errors: Inaccurate 3D head position estimates from external landmark detectors or pose estimators distort the virtual camera matrix, corrupting the perspective-warped input image.
- Cascading Error Propagation: A mere 10 cm displacement or estimation error in head position causes the normalized gaze error to explode beyond $7^\circ$, leading to catastrophic performance degradation.

### 2.3 Main Contributions

To overcome multi-mapping ambiguity without suffering from the error propagation of explicit 3D normalization, the paper introduces four core contributions:

- A Dual-stream Architecture ($\Phi_{\text{eye}}, \Phi_{\text{face}}$) that processes dedicated eye crops and full-face crops in parallel, jointly capturing fine-grained ocular cues and holistic facial context.
- A Learnable Coordinate Transformation $h(\cdot)$ parameterized by a lightweight MLP based on pinhole camera geometry, mapping crop-space gaze predictions back to global image space without fragile 3D camera recalibration.
- A Fourier Head Position (HP) Embedding that projects 3D head position vectors into high-frequency continuous representations to restore spatial positioning context lost during tight cropping.
- A CLIP-Guided Fusion Module that exploits pre-trained vision-language semantic representations to dynamically weigh eye-branch and face-branch predictions according to visibility and occlusion conditions.

---

## 3. Proposed Framework

![Figure 2](/images/clip-dual-crop-gaze/fig2_architecture.jpeg)

*Figure 2: Overview of the proposed architecture. Given an input face image $I$, HRNet extracts 2D landmarks to crop left/right eye regions ($I_c^L, I_c^R$) and the full face ($I_f$). $\Phi_{\text{eye}}$ (4-stack Hourglass) and $\Phi_{\text{face}}$ (6-stack Hourglass) ingest the cropped images alongside Fourier HP embeddings to regress gaze in crop space ($\mathcal{C}_{\text{crop}}$), which are then mapped to image space ($\mathcal{C}_{\text{img}}$) via $h(\cdot)$. Finally, $\Phi_{\text{fuse}}$ leverages CLIP vision-language features $F_{\text{CLIP}}$ to dynamically fuse the predictions.*

Given an input face image $I \in \mathbb{R}^{H \times W}$, a pre-trained HRNet backbone detects 2D facial landmarks. Based on these landmarks, left/right eye crops $I_c^L, I_c^R \in \mathbb{R}^{H_c \times W_c}$ and a standardized face crop $I_f \in \mathbb{R}^{H_f \times W_f}$ ($H_f=96, W_f=224$) are extracted.

The pipeline comprises four key components: (1) Learnable Coordinate Transformation $h(\cdot)$, (2) Eye Branch $\Phi_{\text{eye}}$, (3) Face Branch $\Phi_{\text{face}}$, and (4) CLIP-Guided Fusion Module $\Phi_{\text{fuse}}$.

### 3.1 Mathematical Derivation of Learnable Coordinate Transformation $h(\cdot)$

The core strategy for resolving multi-mapping is decoupling gaze regression into two coordinate spaces:

1. Gaze regression is first performed within the localized Crop Space ($\mathcal{C}_{\text{crop}}$). In this tight crop, the correspondence between eye appearance and local eyeball orientation remains unique and stable.
2. The localized gaze prediction is subsequently mapped back to the global Image Space ($\mathcal{C}_{\text{img}}$) to evaluate the true line-of-sight relative to the vehicle coordinate frame.

The transformation module $h(\cdot)$ acts as the geometric bridge between these two spaces.

![Figure 3](/images/clip-dual-crop-gaze/fig3_transformation.jpeg)

*Figure 3: Geometric relationship between Crop Space ($\mathcal{C}_{\text{crop}}$) and Image Space ($\mathcal{C}_{\text{img}}$) based on pinhole camera projection and 2D translation matrix $M_T$.*

#### 1. Pinhole Camera Projection Formulation

The standard pinhole projection of a 3D world point onto the 2D image plane is expressed as:

$$[p_{\text{2D}}, 1]^T = K_{\text{img}} [R_{\text{img}} \mid T_{\text{img}}] [P_{\text{3D}}, 1]^T$$

Where:
- $P_{\text{3D}} = (X, Y, Z)$ represents the physical 3D coordinates of the eye center in 3D space. Appending $1$ forms homogeneous coordinates $[X, Y, Z, 1]^T$ to compute 3D rotation and translation via a single linear matrix multiplication.
- $[R_{\text{img}} \mid T_{\text{img}}] \in \mathbb{R}^{3 \times 4}$ denotes the extrinsic camera matrix, where $R_{\text{img}} \in \mathbb{R}^{3 \times 3}$ is the 3D gaze rotation matrix and $T_{\text{img}} \in \mathbb{R}^{3 \times 1}$ is the 3D translation vector (head position relative to the camera).
- $K_{\text{img}} \in \mathbb{R}^{3 \times 3}$ is the intrinsic camera matrix:
  $$K_{\text{img}} = \begin{bmatrix} f & 0 & W/2 \\ 0 & f & H/2 \\ 0 & 0 & 1 \end{bmatrix}$$
  where $f$ is the focal length and $(W/2, H/2)$ is the optical center (principal point).
- $p_{\text{2D}} = (u, v)$ denotes the resulting 2D pixel coordinates on the full image sensor.

#### 2. Virtual Pinhole Projection in Crop Space

Cropping a localized patch around the eye is geometrically equivalent to capturing the eye with a dedicated virtual pinhole camera:

$$[p'_{\text{2D}}, 1]^T = K_c [R_c \mid T_c] [P_{\text{3D}}, 1]^T$$

Here, $p'_{\text{2D}} = (u', v')$ are pixel coordinates in the crop, $R_c$ is the local 3D gaze rotation in crop space, and $K_c$ is the virtual intrinsic matrix:
$$K_c = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$
where $(f_x, f_y)$ represents the virtual focal scaling factor and $(c_x, c_y)$ is the center of the cropped region.

#### 3. Derivation of 2D Translation $M_T$ and Closed-Form Rotation Mapping

The 2D spatial relationship between image coordinates $(u, v)$ and crop coordinates $(u', v')$ is defined by a translation corresponding to the top-left bounding box anchor $(x_s, y_s)$:

$$u' = u - x_s, \quad v' = v - y_s$$

Expressing this translation as a $3 \times 3$ matrix $M_T$:

$$[p'_{\text{2D}}, 1]^T = \begin{bmatrix} 1 & 0 & -x_s \\ 0 & 1 & -y_s \\ 0 & 0 & 1 \end{bmatrix} [p_{\text{2D}}, 1]^T = M_T [p_{\text{2D}}, 1]^T$$

Equating the two projection equations yields:

$$M_T K_{\text{img}} [R_{\text{img}} \mid T_{\text{img}}] = K_c [R_c \mid T_c]$$

Isolating the rotational component $R_{\text{img}}$ by inverting the extrinsic and translation terms yields the closed-form transformation:

$$R_{\text{img}} = K_c \cdot R_c \cdot M_T^{-1} \cdot K_{\text{img}}^{-1}$$

This closed-form formulation:
1. Inverts the original sensor intrinsic distortion ($K_{\text{img}}^{-1}$) and crop translation ($M_T^{-1}$),
2. Re-scales by the crop virtual intrinsic matrix $K_c$, and
3. Accurately maps the local crop gaze rotation $R_c$ back to global image space $R_{\text{img}}$.

#### 4. Learning Virtual Intrinsics $K_c$ via Lightweight MLP

While $K_{\text{img}}$ is known from camera calibration, the crop intrinsics $[f_x, f_y, c_x, c_y]$ vary dynamically with subject distance and bounding box coordinates $b = [x_s, y_s, w, h]$.

Rather than computing $K_c$ with fragile 3D landmark solvers, the framework employs a lightweight MLP $\Phi^M$ that directly regresses the virtual parameters from the 2D bounding box $b$:

$$[f_x, f_y, c_x, c_y]^T = \Phi^M(b), \quad K_c = g_2(\Phi^M(b))$$

The overall transformation function $h(\cdot)$ is thus given by:

$$R_{\text{img}} = g_2(\Phi^M(b)) \cdot R_c \cdot g_1(x_s, y_s)^{-1} \cdot K_{\text{img}}^{-1} = h(\Phi^M(b), b, K_{\text{img}}, R_c)$$

By learning $K_c$ directly from 2D bounding box parameters, the model eliminates dependence on noisy 3D head pose estimation, remaining remarkably resilient against head movements.

### 3.2 Eye Branch Architecture $\Phi_{\text{eye}}$

The eye branch extracts fine-grained ocular features from the left and right eye crops $I_c^L, I_c^R$:

1. Feature Extraction: A weight-shared 4-stack Hourglass CNN $\Phi_e$ extracts localized feature representations:
   $$F^L = \Phi_e(I_c^L), \quad F^R = \Phi_e(I_c^R)$$

2. Spatial Positioning Context (Fourier HP Embedding): To restore 3D spatial awareness stripped away by cropping, an external CNN $\Phi_{\text{HP}}$ extracts the 3D head position $\text{HP} = \Phi_{\text{HP}}(I) \in \mathbb{R}^3$. This vector is mapped to a high-dimensional continuous representation using Fourier embedding $\Psi^{\text{HP}}$ and an MLP $\Phi_e^{\text{HP}}$.

3. Crop-Space Gaze Regression: Feature maps and HP embeddings are concatenated and passed through regressor $R^e$ to yield crop-space gaze predictions:
   $$\tilde{R}_c^L = R^e(\Phi_e^{\text{HP}}(\Psi^{\text{HP}}(\text{HP})) \oplus F^L), \quad \tilde{R}_c^R = R^e(\Phi_e^{\text{HP}}(\Psi^{\text{HP}}(\text{HP})) \oplus F^R)$$

4. Image-Space Mapping and Averaging: Predictions are projected to image space via $h(\cdot)$ using dedicated MLP $\Phi_c^M$, followed by a weighted aggregation $w(\cdot)$:
   $$\tilde{R}_{\text{img}}^L = h(\Phi_c^M(b_c^L), b_c^L, K_{\text{img}}, \tilde{R}_c^L), \quad \tilde{R}_{\text{img}}^R = h(\Phi_c^M(b_c^R), b_c^R, K_{\text{img}}, \tilde{R}_c^R)$$
   $$\tilde{R}_{\text{img}}^e = w(\tilde{R}_{\text{img}}^L, \tilde{R}_{\text{img}}^R)$$

The eye branch is optimized with cosine angular loss:

$$\mathcal{L}_{\text{eye}} = \cos^{-1}((\tilde{R}_{\text{img}}^e)^T R_{\text{img}}^{\text{GT}})$$

### 3.3 Face Branch Architecture $\Phi_{\text{face}}$

The face branch captures macroscopic head pose and facial geometry:

1. Feature Extraction: The full-face crop $I_f$ ($96 \times 224$) is processed by a 6-stack Hourglass CNN $\Phi_f$:
   $$F^f = \Phi_f(I_f)$$

2. HP Embedding and Regression: Combined with face-specific Fourier HP embeddings via MLP $\Phi_f^{\text{HP}}$, face regressor $R^f$ outputs crop-space gaze $\tilde{R}_c^f$:
   $$\tilde{R}_c^f = R^f(\Phi_f^{\text{HP}}(\Psi^{\text{HP}}(\text{HP})) \oplus F^f)$$

3. Image-Space Projection: Mapped to image space using face bounding box $b_f$ and MLP $\Phi_f^M$:
   $$\tilde{R}_{\text{img}}^f = h(\Phi_f^M(b_f), b_f, K_{\text{img}}, \tilde{R}_c^f)$$

The face branch is supervised with angular loss:

$$\mathcal{L}_{\text{face}} = \cos^{-1}((\tilde{R}_{\text{img}}^f)^T R_{\text{img}}^{\text{GT}})$$

### 3.4 CLIP-Guided Fusion Module $\Phi_{\text{fuse}}$

Under mask occlusion, lower facial cues degrade while eye crops remain informative. Conversely, under glasses glare or deep shadows, the eye branch struggles while the face branch provides dependable head pose guidance.

The framework employs pre-trained CLIP ($\Phi_{\text{CLIP}}$) to dynamically evaluate visual reliability:

1. Multimodal Semantic Feature Extraction: The full image $I$ and a fixed semantic prompt context vector $c$ (describing visual eye clarity) are passed through CLIP:
   $$F_{\text{CLIP}} = \Phi_{\text{CLIP}}(I, c)$$

2. Dynamic Gating: $F_{\text{CLIP}}$ is processed by MLP $\Phi_{\text{fuse}}$ to produce softmax-normalized importance weights:
   $$\tilde{R}_{\text{img}}^{\text{fused}} = \sigma(\Phi_{\text{fuse}}(F_{\text{CLIP}})^T) [\tilde{R}_{\text{img}}^e, \tilde{R}_{\text{img}}^f]^T$$

The fusion loss is:

$$\mathcal{L}_{\text{fuse}} = \cos^{-1}((\tilde{R}_{\text{img}}^{\text{fused}})^T R_{\text{img}}^{\text{GT}})$$

The total loss is optimized end-to-end:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{eye}} + \mathcal{L}_{\text{face}} + \mathcal{L}_{\text{fuse}}$$

---

## 4. Experimental Results

### 4.1 Datasets and Evaluation Setup

The method was evaluated across three distinct benchmarks:

| Dataset | Modality / Domain | Scale & Subjects | Protocol |
|---|---|---|---|
| IVGaze | In-cabin Infrared (IR) | 44,705 images, 125 subjects | 3-fold cross-validation |
| GazeGene | Multi-view Synthetic RGB | 9 camera views, 56 characters | Test on final 10 characters |
| MPIIFaceGaze | In-the-wild Webcam RGB | 15 subjects | Leave-one-subject-out (LOSO) |

Metrics: Mean Angular Error (AE, in degrees $^\circ$) and Average Precision (AP, %) across thresholds ($<2^\circ, <4^\circ, <6^\circ, <8^\circ$).

Implementation Details: Optimized with AdamW ($\Phi_{\text{face}}, \Phi_{\text{eye}}$) and Adam ($\Phi_{\text{fuse}}$) at learning rate 0.001 with StepLR schedule for 100 epochs. Inference latency averages 42 ms on an NVIDIA A100 GPU.

### 4.2 Benchmark Comparisons

| Method | IVGaze (IR) | GazeGene (Synth RGB) | MPIIFaceGaze (Wild RGB) |
|---|---|---|---|
| FullFace | 13.67° | 3.54° | 4.93° |
| DWG | 8.82° | - | - |
| Gaze360 | 8.15° | - | - |
| $\text{FullFace}^+$ | 7.48° | - | - |
| GazeTR | 7.33° | 3.37° | 4.00° |
| XGaze | 7.06° | - | - |
| GazePTR | 7.04° | - | - |
| GazeDPTR | 6.71° | - | - |
| Dilated-Net | - | 3.08° | 4.42° |
| ResNet-50 | - | 3.36° | 4.20° |
| RT-Gene | - | - | 4.30° |
| CA-Net | - | - | 4.10° |
| L2CS | - | - | 3.92° |
| Ours | **6.29°** | **1.87°** | **3.66°** |

The proposed framework consistently outperforms all prior baselines across all three benchmarks.

Notably, on GazeGene—which features 9 diverse camera perspectives and wide head position variations—the method achieves a remarkable 39.3% error reduction ($3.08^\circ \to 1.87^\circ$), validating the head position invariance conferred by $h(\cdot)$ and Fourier HP embeddings. On real-world in-cabin IR data (IVGaze), it achieves a 6.3% error reduction over the previous SOTA GazeDPTR.

### 4.3 IVGaze Threshold AP Evaluation

| Method | AP ($< 2^\circ$) | AP ($< 4^\circ$) | AP ($< 6^\circ$) | AP ($< 8^\circ$) |
|---|---|---|---|---|
| FullFace | 2.3% | 8.8% | 17.8% | 28.0% |
| DWG | 6.6% | 21.7% | 33.8% | 53.2% |
| Gaze360 | 9.2% | 27.3% | 44.6% | 58.9% |
| $\text{FullFace}^+$ | 14.2% | 31.1% | 46.7% | 61.3% |
| GazeTR | 17.0% | 32.8% | 45.7% | 64.7% |
| XGaze | 11.7% | 32.7% | 51.6% | 65.2% |
| GazePTR | 17.6% | 34.6% | 49.3% | 66.7% |
| GazeDPTR | **22.1%** | 36.0% | 50.3% | 68.4% |
| Ours | 16.0% | **38.5%** | **58.7%** | **73.5%** |
| Ours (w/o Sunglasses) | 16.2% | 39.1% | 59.6% | 74.6% |

The proposed model dominates across all practical operational bounds ($<4^\circ, <6^\circ, <8^\circ$). In particular, under the $<6^\circ$ threshold, it attains 58.7% AP, surpassing GazeDPTR by 16.69% relative improvement.

### 4.4 Robustness to Accessories and Occlusions

| Method | Glasses | No Glasses | Mask | No Mask | Sunglasses |
|---|---|---|---|---|---|
| FullFace | 14.43° | 12.40° | 15.20° | 13.35° | 21.39° |
| DWG | 9.20° | 8.19° | 9.43° | 8.69° | 17.43° |
| Gaze360 | 8.30° | 7.91° | 8.95° | 7.99° | 17.99° |
| $\text{FullFace}^+$ | 7.59° | 7.30° | 8.37° | 7.30° | 16.50° |
| XGaze | 7.07° | 7.03° | 7.80° | 6.90° | 15.15° |
| GazeTR | 7.40° | 7.22° | 8.12° | 7.17° | 17.49° |
| GazePTR | 7.13° | 6.90° | 7.78° | 6.89° | 16.54° |
| GazeDPTR | 6.77° | 6.63° | 7.44° | 6.57° | **16.41°** |
| Ours | **6.21°** | **5.71°** | **6.72°** | **5.39°** | 18.70° |

Under glasses ($6.21^\circ$) and mask occlusions ($6.72^\circ$), the proposed method achieves superior robustness.

### 4.5 Branch and Fusion Ablation Study

| Condition | $\Phi_{\text{face}}$ (Alone) | $\Phi_{\text{eye}}$ (Alone) | $\Phi_{\text{fuse}}$ (Proposed Fusion) |
|---|---|---|---|
| Glasses | 6.65° | 6.98° | **6.21°** |
| No Glasses | 6.06° | 6.38° | **5.71°** |
| Mask | 7.22° | 6.79° | **6.72°** |
| No Mask | 5.71° | 6.31° | **5.39°** |
| Overall Mean | 6.69° | 6.77° | **6.29°** |

Under mask occlusions, lower face degradation causes $\Phi_{\text{face}}$ to drop ($7.22^\circ$), while $\Phi_{\text{eye}}$ maintains performance ($6.79^\circ$). Under glasses reflection, ocular degradation hampers $\Phi_{\text{eye}}$ ($6.98^\circ$), while $\Phi_{\text{face}}$ provides stable guidance ($6.65^\circ$). Dynamic CLIP fusion $\Phi_{\text{fuse}}$ adaptively balances these signals, reaching $6.29^\circ$ overall.

### 4.6 Qualitative Visualizations

![Figure 5](/images/clip-dual-crop-gaze/fig5_qualitative_ablation.jpeg)

*Figure 5: Qualitative comparisons and ablations on IVGaze (IR) and GazeGene (RGB). The proposed method maintains tight alignment between predicted gaze vectors (blue) and ground truth (green) across severe glasses reflections, mask occlusions, and extreme head poses.*

### 4.7 Sensitivity to Head Position (HP) Perturbation

| Condition | Zhang's Normalization | Ours (Learnable $h(\cdot)$) |
|---|---|---|
| Glasses | 7.10° | **6.65°** |
| No Glasses | 6.32° | **6.06°** |
| Mask | 7.39° | **7.22°** |
| No Mask | 6.72° | **5.71°** |
| Overall Mean | 7.02° | **6.69°** |

Replacing standard Zhang normalization with learnable transformation $h(\cdot)$ reduces mean error from $7.02^\circ$ to $6.69^\circ$.

![Figure 6](/images/clip-dual-crop-gaze/fig6_sensitivity_plot.jpeg)

*Figure 6: Angular error sensitivity under increasing 3D Head Position (HP) perturbation. Traditional IVGaze normalization degrades rapidly past $7^\circ$ under head displacement, while the proposed method retains errors under $2^\circ$, confirming robust position invariance.*

As shown in Figure 6, synthetic HP perturbations cause traditional normalization error to explode past $7^\circ$, whereas the proposed approach remains exceptionally stable below $2^\circ$.

---

## 5. Conclusion and Key Takeaways

This paper addresses the long-standing challenge of head position sensitivity and multi-mapping ambiguity in appearance-based gaze estimation through a geometrically grounded deep learning framework.

The primary insights are summarized as follows:

1. **Decoupled Formulation via Learnable Pinhole Mapping**: By regressing gaze in a canonical crop space and mapping back via a learned pinhole transformation $h(\cdot)$, the framework eliminates multi-mapping while circumventing the fragility of explicit 3D camera normalization.
2. **Multimodal Vision-Language Gating**: Integrating pre-trained CLIP representations enables context-aware dynamic arbitration between ocular and facial branches under diverse occlusions.
3. **Cross-Domain Generalization**: Robust gains across real in-cabin IR (IVGaze), multi-view synthetic RGB (GazeGene), and in-the-wild webcam (MPIIFaceGaze) validate the broad applicability of the architecture.

Future investigations will focus on enhancing context reasoning under total ocular occlusion (such as opaque sunglasses at $18.70^\circ$) and streamlining model latency (currently 42 ms on A100) for real-time edge deployment in vehicle ECUs.
