---
title: "[GAZE 2026] Learning to Look: CLIP-Guided Dual-Crop Fusion for Head Position-Invariant Gaze Estimation"
date: 2026-08-28T15:10:14+09:00
draft: false
math: true
tags: ["Paper Review", "GAZE 2026", "Gaze Estimation", "CLIP", "Fusion", "CVPR 2026"]
categories: ["GAZE 2026", "Paper Review"]
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

![Figure 1: Qualitative Gaze Estimation in In-Cabin Scenarios](/images/clip-dual-crop-gaze/fig1_qualitative_teaser.jpeg)
*Figure 1: Qualitative gaze estimation results across challenging in-cabin scenarios, including glasses reflections, mask occlusions, and extreme gaze deviations. Green arrows denote ground truth, blue arrows indicate the proposed method, and red arrows represent the baseline. Angular errors in yaw and pitch are listed in degrees beneath each image.*

---

## 1. One-Sentence Summary

This paper presents a dual-stream gaze estimation framework combining eye crops and full-face crops with Fourier head position embeddings, a learnable pinhole coordinate transformation, and CLIP-guided vision-language feature fusion, achieving substantial error reductions of 6.3% to 39.3% over state-of-the-art baselines across real in-cabin IR, synthetic multi-view RGB, and in-the-wild webcam benchmarks.

---

## 2. Research Background and Motivation

### 2.1 Problem Definition

Consider driving out of an unlit highway tunnel on a brilliant summer afternoon. The sudden burst of sunlight sweeps across the vehicle cabin, casting sharp shadows over the dashboard while oncoming headlights reflect brightly off the driver's corrective lenses. Concurrently, the driver leans forward, turns their head left and right to inspect the side mirrors, and glances back toward the center lane.

Under such volatile real-world driving conditions, the driver's head moves dynamically across wide three-dimensional translational and rotational ranges, while optical reflections, protective masks, and illumination variations frequently obscure crucial visual features. Appearance-based three-dimensional gaze estimation seeks to regress precise 3D line-of-sight vectors directly from in-cabin camera images. It serves as a vital component for driver monitoring systems and next-generation in-vehicle human-machine interfaces, yet existing vision models suffer severe degradation whenever the driver shifts position or ocular details become partially obstructed.

Legacy benchmarks such as MPIIFaceGaze and GazeCapture were captured under stationary frontal postures and uniformly controlled lighting, leaving the severe spatial and optical variability of real vehicle cabins unaddressed. The recent introduction of the IVGaze dataset—encompassing 44,705 infrared in-cabin images collected during real driving—has catalyzed urgent interest in developing gaze estimation models that maintain strict invariance to head position shifts and visual occlusions.

### 2.2 Limitations of Existing Methods

Existing appearance-based gaze estimation approaches suffer from fundamental limitations in input crop selection and three-dimensional geometric normalization:

First, models face an unresolved resolution trade-off and the multi-mapping dilemma. Prior architectures typically rely either on full-face images or on tightly cropped eye patches. Full-face models preserve global facial geometry and macroscopic head orientation, but the fine-grained rotational cues of the pupil and iris become diluted in downsampled feature maps. Conversely, eye-crop models isolate high-resolution ocular boundaries, yet they completely discard the driver's three-dimensional spatial coordinates relative to the vehicle camera.

![Figure 4: Multi-Mapping Dilemma from Head Pose Variations](/images/clip-dual-crop-gaze/fig4_multi_mapping.jpeg)
*Figure 4: Illustration of the multi-mapping dilemma in the IVGaze dataset. Due to variations in 3D head orientation, identical 3D gaze directions produce noticeably different 2D eye appearances as the eyeball center and iris boundaries shift relative to the eye corner landmarks.*

This isolation triggers the multi-mapping dilemma. Because physical gaze direction is the composite outcome of eyeball rotation and three-dimensional head pose, identical line-of-sight directions yield entirely different two-dimensional eye textures whenever the driver tilts or turns their head. Conversely, visually indistinguishable eye crops correspond to divergent absolute gaze vectors under different head orientations. This non-bijective correspondence severely destabilizes regression training.

Second, traditional three-dimensional gaze normalization suffers from catastrophic error propagation. To resolve multi-mapping, prior works commonly construct a virtual camera coordinate frame centered at the eye and perspective-warp the raw image. However, computing this virtual camera depends heavily on external facial landmark detectors or three-dimensional head pose estimators. If the upstream estimator misjudges head position by even a few centimeters, large geometric distortions are amplified across the perspective warping matrix. In practice, a mere 10 cm shift in estimated head position causes angular gaze errors to surge past $7^\circ$.

### 2.3 Main Contributions

To resolve multi-mapping ambiguity while bypassing the fragility of explicit three-dimensional camera recalibration, the paper establishes three primary contributions:

- It develops a dual-stream architecture that processes dedicated eye crops and full-face crops in parallel, restoring lost spatial geometry through high-frequency continuous Fourier head position embeddings.
- It formulates a learnable pinhole coordinate transformation module parameterized by a lightweight neural network, deriving a closed-form geometric mapping from canonical crop space to global image space based purely on two-dimensional bounding box coordinates.
- It introduces a dynamic fusion mechanism guided by pre-trained CLIP vision-language semantic representations, adaptively arbitrating between eye and face branches based on real-time visual occlusions and surface reflections.

---

## 3. Proposed Framework

The dual-stream formulation can be intuitively understood through the analogy of a harbor control tower navigating ships through dense fog and choppy waters. To track an approaching vessel, the harbor master combines two distinct vantage points: a high-magnification spotting telescope focused on the ship's rudder and compass wheel, alongside a wide-angle observation deck overlooking the entire harbor basin. The telescope view (eye crop) captures millimeter-level adjustments in steering angle but loses track of the vessel's global position. The wide-angle deck view (face crop) tracks the macroscopic hull orientation and mooring channel but misses subtle steering vibrations. To reconcile both observations into a unified navigational chart (image space), the system applies a closed-form coordinate translation. When sea spray blurs the telescope or harbor floodlights glare across the deck, an intelligent supervisor (CLIP) dynamically shifts confidence toward the clearer observation stream.

![Figure 2: Overall System Architecture](/images/clip-dual-crop-gaze/fig2_architecture.jpeg)
*Figure 2: Architecture of the proposed framework. Given an input facial image, HRNet extracts 2D landmarks to segment left/right eye crops and a normalized face crop. The 4-stack Hourglass eye branch and 6-stack Hourglass face branch ingest their respective crops with Fourier head position embeddings to predict gaze in crop space. The predictions are subsequently mapped to image space via the learnable transformation module, and a CLIP-guided fusion module dynamically produces the final integrated gaze vector.*

Given an input face image $I \in \mathbb{R}^{H \times W}$, a pre-trained HRNet backbone detects two-dimensional landmarks to segment left and right eye crops $I_c^L, I_c^R \in \mathbb{R}^{H_c \times W_c}$ and a standardized face crop $I_f \in \mathbb{R}^{H_f \times W_f}$ ($H_f=96, W_f=224$). The pipeline consists of the coordinate transformation module, the eye branch, the face branch, and the CLIP-guided fusion module.

### 3.1 Learnable Coordinate Transformation $h(\cdot)$ and Mathematical Derivation

Decoupling gaze estimation across two coordinate domains resolves the multi-mapping dilemma:
1. Gaze regression is initially executed within a canonical Crop Space ($\mathcal{C}_{\text{crop}}$), where local eyeball rotation corresponds bijectively to pupil and iris textures regardless of global head movement.
2. The localized crop prediction is subsequently translated back into the global Image Space ($\mathcal{C}_{\text{img}}$) to compute vehicle-level metrics and supervision loss.

The transformation module $h(\cdot)$ establishes the formal geometric bridge between these domains.

![Figure 3: Pinhole Geometry and Coordinate Mapping](/images/clip-dual-crop-gaze/fig3_transformation.jpeg)
*Figure 3: Geometric relationship between Crop Space and Image Space based on pinhole camera projection and 2D translation matrix $M_T$.*

Under standard pinhole camera projection, a physical 3D point $P_{\text{3D}} = (X, Y, Z)$ projects onto sensor pixel $p_{\text{2D}} = (u, v)$ according to:

$$[p_{\text{2D}}, 1]^T = K_{\text{img}} [R_{\text{img}} \mid T_{\text{img}}] [P_{\text{3D}}, 1]^T$$

Where:
- $[P_{\text{3D}}, 1]^T$ is the homogeneous coordinate column vector of the physical 3D ocular point, enabling rotation and translation to be evaluated via a single linear transformation.
- $[R_{\text{img}} \mid T_{\text{img}}] \in \mathbb{R}^{3 \times 4}$ denotes the extrinsic camera matrix, where $R_{\text{img}} \in \mathbb{R}^{3 \times 3}$ is the 3D gaze rotation matrix and $T_{\text{img}} \in \mathbb{R}^{3 \times 1}$ is the 3D translation vector representing head position relative to the camera lens.
- $K_{\text{img}} \in \mathbb{R}^{3 \times 3}$ represents the intrinsic camera calibration matrix:
  $$K_{\text{img}} = \begin{bmatrix} f & 0 & W/2 \\ 0 & f & H/2 \\ 0 & 0 & 1 \end{bmatrix}$$
  where $f$ denotes the camera focal length and $(W/2, H/2)$ specifies the optical center principal point.
- $p_{\text{2D}}$ denotes the resulting 2D pixel coordinates on the sensor plane.

Applying the same pinhole formulation to a cropped sub-region $p'_{\text{2D}} = (u', v')$ defines a virtual pinhole camera:

$$[p'_{\text{2D}}, 1]^T = K_c [R_c \mid T_c] [P_{\text{3D}}, 1]^T$$

Here, $R_c$ is the localized ocular rotation within the crop frame, and $K_c$ is the virtual intrinsic matrix of the crop window:

$$K_c = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

The spatial offset between raw image pixel $(u, v)$ and crop pixel $(u', v')$ is defined by the top-left bounding box anchor coordinates $(x_s, y_s)$:

$$u' = u - x_s, \quad v' = v - y_s$$

Formulating this translation as a $3 \times 3$ matrix $M_T$:

$$[p'_{\text{2D}}, 1]^T = \begin{bmatrix} 1 & 0 & -x_s \\ 0 & 1 & -y_s \\ 0 & 0 & 1 \end{bmatrix} [p_{\text{2D}}, 1]^T = M_T [p_{\text{2D}}, 1]^T$$

Multiplying the full-image projection equation by $M_T$ equates both projections:

$$M_T K_{\text{img}} [R_{\text{img}} \mid T_{\text{img}}] = K_c [R_c \mid T_c]$$

Isolating the rotational component $R_{\text{img}}$ by taking inverse operations yields the closed-form transformation:

$$R_{\text{img}} = K_c R_c M_T^{-1} K_{\text{img}}^{-1}$$

The functional operators operate as follows:
- $K_{\text{img}}^{-1}$ inverts the camera sensor intrinsics, re-projecting 2D pixels into 3D directional optical rays.
- $M_T^{-1}$ reverses the bounding box crop offset by adding back $(x_s, y_s)$, realigning the localized patch with the global image origin.
- $R_c$ is the local gaze rotation regressed by the neural network within the canonical crop.
- $K_c$ applies the scaling and optical center of the virtual crop camera.

While camera intrinsic matrix $K_{\text{img}}$ is known from hardware calibration, the virtual crop intrinsics $[f_x, f_y, c_x, c_y]$ vary continuously as the driver moves and alters the bounding box $b = [x_s, y_s, w, h]$. Rather than computing $K_c$ through sensitive 3D geometric optimization, the authors employ a lightweight multi-layer perceptron $\Phi^M$ that infers the virtual parameters directly from the 2D bounding box $b$:

$$[f_x, f_y, c_x, c_y]^T = \Phi^M(b), \quad K_c = g_2(\Phi^M(b))$$

The unified coordinate mapping function $h(\cdot)$ is expressed as:

$$R_{\text{img}} = g_2(\Phi^M(b)) R_c g_1(x_s, y_s)^{-1} K_{\text{img}}^{-1} = h(\Phi^M(b), b, K_{\text{img}}, R_c)$$

By learning the projective parameters directly from 2D bounding box boundaries, the system eliminates reliance on error-prone 3D head pose estimators, remaining impervious to head translation noise.

### 3.2 Eye Branch Architecture $\Phi_{\text{eye}}$

The eye branch extracts fine-grained ocular rotation cues from left and right eye crops $I_c^L, I_c^R$.

A weight-shared 4-stack Hourglass encoder $\Phi_e$ extracts localized ocular feature maps:

$$F^L = \Phi_e(I_c^L), \quad F^R = \Phi_e(I_c^R)$$

To restore 3D spatial awareness lost during tight cropping, a pre-trained head pose estimator $\Phi_{\text{HP}}$ extracts the 3D head position vector $\text{HP} = \Phi_{\text{HP}}(I) \in \mathbb{R}^3$. This vector is mapped to a high-frequency continuous representation via Fourier embedding $\Psi^{\text{HP}}$ and a dedicated MLP $\Phi_e^{\text{HP}}$, then concatenated with ocular features:

$$\tilde{R}_c^L = R^e(\Phi_e^{\text{HP}}(\Psi^{\text{HP}}(\text{HP})) \oplus F^L)$$

$$\tilde{R}_c^R = R^e(\Phi_e^{\text{HP}}(\Psi^{\text{HP}}(\text{HP})) \oplus F^R)$$

The crop-space predictions are transformed into global image space using $h(\cdot)$ with eye-specific MLP $\Phi_c^M$ and aggregated via weighted averaging:

$$\tilde{R}_{\text{img}}^L = h(\Phi_c^M(b_c^L), b_c^L, K_{\text{img}}, \tilde{R}_c^L)$$

$$\tilde{R}_{\text{img}}^R = h(\Phi_c^M(b_c^R), b_c^R, K_{\text{img}}, \tilde{R}_c^R)$$

$$\tilde{R}_{\text{img}}^e = w(\tilde{R}_{\text{img}}^L, \tilde{R}_{\text{img}}^R)$$

The eye branch is trained with cosine angular loss against ground truth vector $R_{\text{img}}^{\text{GT}}$:

$$\mathcal{L}_{\text{eye}} = \cos^{-1}((\tilde{R}_{\text{img}}^e)^T R_{\text{img}}^{\text{GT}})$$

### 3.3 Face Branch Architecture $\Phi_{\text{face}}$

The face branch captures macroscopic head pose and global facial context.

The standardized face crop $I_f$ ($96 \times 224$) is processed by a 6-stack Hourglass encoder $\Phi_f$ to extract facial feature map $F^f = \Phi_f(I_f)$.

Concatenating $F^f$ with face-specific Fourier head position embeddings via MLP $\Phi_f^{\text{HP}}$, the face regressor $R^f$ outputs crop-space gaze $\tilde{R}_c^f$:

$$\tilde{R}_c^f = R^f(\Phi_f^{\text{HP}}(\Psi^{\text{HP}}(\text{HP})) \oplus F^f)$$

The prediction is mapped to image space using face bounding box $b_f$ and MLP $\Phi_f^M$:

$$\tilde{R}_{\text{img}}^f = h(\Phi_f^M(b_f), b_f, K_{\text{img}}, \tilde{R}_c^f)$$

The face branch is supervised via independent angular loss:

$$\mathcal{L}_{\text{face}} = \cos^{-1}((\tilde{R}_{\text{img}}^f)^T R_{\text{img}}^{\text{GT}})$$

### 3.4 CLIP-Guided Fusion Module $\Phi_{\text{fuse}}$

Under face mask usage, lower facial features are degraded while eye crops remain sharp. Conversely, when sunglasses or optical glare obstruct the eyes, the eye branch deteriorates while the face branch provides dependable head orientation cues.

Rather than maintaining fixed fusion weights, the framework utilizes pre-trained CLIP ($\Phi_{\text{CLIP}}$) to assess the semantic visibility of the face.

Full image $I$ and a fixed semantic prompt context vector $c$ describing ocular clarity are fed into CLIP to generate multimodal representation $F_{\text{CLIP}}$:

$$F_{\text{CLIP}} = \Phi_{\text{CLIP}}(I, c)$$

Feature $F_{\text{CLIP}}$ is passed through fusion MLP $\Phi_{\text{fuse}}$ to compute softmax-normalized reliability weights, combining both predictions into final gaze vector $\tilde{R}_{\text{img}}^{\text{fused}}$:

$$\tilde{R}_{\text{img}}^{\text{fused}} = \sigma(\Phi_{\text{fuse}}(F_{\text{CLIP}})^T) [\tilde{R}_{\text{img}}^e, \tilde{R}_{\text{img}}^f]^T$$

The fusion loss is defined as:

$$\mathcal{L}_{\text{fuse}} = \cos^{-1}((\tilde{R}_{\text{img}}^{\text{fused}})^T R_{\text{img}}^{\text{GT}})$$

The framework is trained end-to-end using the joint loss $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{eye}} + \mathcal{L}_{\text{face}} + \mathcal{L}_{\text{fuse}}$.

---

## 4. Experimental Results

### 4.1 Datasets and Evaluation Protocol

Evaluation was conducted across three rigorous benchmarks encompassing real in-cabin infrared data, synthetic multi-view data, and webcam images:

| Dataset | Modality and Domain | Scale and Subjects | Evaluation Protocol |
|---|---|---|---|
| IVGaze | In-cabin infrared camera | 44,705 images across 125 subjects | 3-fold cross-validation |
| GazeGene | Multi-view synthetic rendering | 9 camera views, 56 characters | Test on final 10 characters |
| MPIIFaceGaze | In-the-wild webcam RGB | 15 subjects | Leave-one-subject-out |

Primary evaluation metrics include Mean Angular Error in degrees and Average Precision (%) within error thresholds of 2°, 4°, 6°, and 8° on IVGaze.

Optimization utilized AdamW for feature extractors and Adam for the fusion module at an initial learning rate of 0.001 with StepLR schedule over 100 epochs. Average inference latency is 42 ms on a single NVIDIA A100 GPU, ensuring real-time feasibility.

### 4.2 Benchmark Comparisons

| Method | IVGaze Infrared | GazeGene Synthetic RGB | MPIIFaceGaze Webcam RGB |
|---|---|---|---|
| FullFace | 13.67° | 3.54° | 4.93° |
| DWG | 8.82° | - | - |
| Gaze360 | 8.15° | - | - |
| FullFace+ | 7.48° | - | - |
| GazeTR | 7.33° | 3.37° | 4.00° |
| XGaze | 7.06° | - | - |
| GazePTR | 7.04° | - | - |
| GazeDPTR | 6.71° | - | - |
| Dilated-Net | - | 3.08° | 4.42° |
| ResNet-50 | - | 3.36° | 4.20° |
| RT-Gene | - | - | 4.30° |
| CA-Net | - | - | 4.10° |
| L2CS | - | - | 3.92° |
| Proposed Method | 6.29° | 1.87° | 3.66° |

The proposed framework consistently outperforms prior baselines across all three benchmarks.

The most dramatic gain appears on GazeGene, where wide 3D head movements and 9 camera perspectives challenge conventional models. The proposed method reduces angular error from 3.08° to 1.87°, achieving a 39.3% error reduction and validating the head position invariance of $h(\cdot)$ and Fourier HP embeddings. On real in-cabin IR data (IVGaze), it reduces angular error by 6.3% over the prior state-of-the-art transformer GazeDPTR, reaching 6.29°.

### 4.3 Threshold Average Precision Analysis on IVGaze

| Method | Precision (< 2°) | Precision (< 4°) | Precision (< 6°) | Precision (< 8°) |
|---|---|---|---|---|
| FullFace | 2.3% | 8.8% | 17.8% | 28.0% |
| DWG | 6.6% | 21.7% | 33.8% | 53.2% |
| Gaze360 | 9.2% | 27.3% | 44.6% | 58.9% |
| FullFace+ | 14.2% | 31.1% | 46.7% | 61.3% |
| GazeTR | 17.0% | 32.8% | 45.7% | 64.7% |
| XGaze | 11.7% | 32.7% | 51.6% | 65.2% |
| GazePTR | 17.6% | 34.6% | 49.3% | 66.7% |
| GazeDPTR | 22.1% | 36.0% | 50.3% | 68.4% |
| Proposed Method | 16.0% | 38.5% | 58.7% | 73.5% |
| Proposed Method - Excl. Sunglasses | 16.2% | 39.1% | 59.6% | 74.6% |

Under practical operational thresholds, the proposed method achieves superior accuracy across the 4°, 6°, and 8° bounds. In particular, within the standard 6° safety tolerance, it achieves 58.7% precision, representing an 8.4 percentage point gain over the previous best baseline.

### 4.4 Robustness to Accessories and Occlusions

| Method | Glasses | No Glasses | Mask | No Mask | Sunglasses |
|---|---|---|---|---|---|
| FullFace | 14.43° | 12.40° | 15.20° | 13.35° | 21.39° |
| DWG | 9.20° | 8.19° | 9.43° | 8.69° | 17.43° |
| Gaze360 | 8.30° | 7.91° | 8.95° | 7.99° | 17.99° |
| FullFace+ | 7.59° | 7.30° | 8.37° | 7.30° | 16.50° |
| XGaze | 7.07° | 7.03° | 7.80° | 6.90° | 15.15° |
| GazeTR | 7.40° | 7.22° | 8.12° | 7.17° | 17.49° |
| GazePTR | 7.13° | 6.90° | 7.78° | 6.89° | 16.54° |
| GazeDPTR | 6.77° | 6.63° | 7.44° | 6.57° | 16.41° |
| Proposed Method | 6.21° | 5.71° | 6.72° | 5.39° | 18.70° |

Under glasses reflections (6.21°) and face masks (6.72°), the proposed method maintains substantial leads over competing models. While opaque sunglasses physically blind ocular cues, elevating error to 18.70°, the architecture demonstrates unmatched resilience across all other real-world occlusion modes.

### 4.5 Branch and Fusion Ablation Analysis

| Condition | Face Branch Alone | Eye Branch Alone | Proposed Dynamic Fusion |
|---|---|---|---|
| Glasses | 6.65° | 6.98° | 6.21° |
| No Glasses | 6.06° | 6.38° | 5.71° |
| Mask | 7.22° | 6.79° | 6.72° |
| No Mask | 5.71° | 6.31° | 5.39° |
| Overall Mean | 6.69° | 6.77° | 6.29° |

Single-branch ablations confirm the vital role of adaptive arbitration. Face masks degrade the face branch to 7.22° while the eye branch sustains 6.79°. Conversely, glasses reflections degrade the eye branch to 6.98° while the face branch anchors estimation at 6.65°. The CLIP-guided fusion module dynamically balances these complementary cues, achieving an overall mean error of 6.29°.

Comparing against standard ResNet-50 fusion (6.41°), CLIP multimodal semantics reduce overall error to 6.29°, delivering a 10.16% relative error reduction under unobstructed conditions.

### 4.6 Qualitative Visualizations and Head Position Sensitivity

![Figure 5: Qualitative Predictions under Extreme Conditions](/images/clip-dual-crop-gaze/fig5_qualitative_ablation.jpeg)
*Figure 5: Qualitative results across IVGaze and GazeGene. Predicted gaze vectors from the proposed method (blue arrows) closely align with ground truth vectors (green arrows) despite glasses reflections, mask occlusions, and sharp head rotations.*

Qualitative visualizations illustrate that predicted gaze vectors track ground truth vectors faithfully across severe glasses glare, mask occlusion, and extreme head turning angles.

| Experimental Condition | Zhang's Normalization | Proposed Learnable Transformation |
|---|---|---|
| Glasses | 7.10° | 6.65° |
| No Glasses | 6.32° | 6.06° |
| Mask | 7.39° | 7.22° |
| No Mask | 6.72° | 5.71° |
| Overall Mean | 7.02° | 6.69° |

Replacing standard Zhang normalization with the proposed learnable transformation reduces overall angular error from 7.02° to 6.69° on identical network backbones.

![Figure 6: Error Sensitivity under Head Position Perturbations](/images/clip-dual-crop-gaze/fig6_sensitivity_plot.jpeg)
*Figure 6: Angular error sensitivity under increasing 3D head position perturbation. Traditional axis normalization degrades rapidly past 7° under displacement, whereas the proposed method retains errors under 2°, confirming robust position invariance.*

In sensitivity experiments with synthetic head position perturbation, traditional axis normalization errors explode past 7° at 10 cm displacement, whereas the proposed framework remains below 2°, verifying exceptional invariance to subject translation.

---

## 5. Conclusion and Key Takeaways

This research provides a fundamental re-examination of head position sensitivity and multi-mapping ambiguity in real-world gaze estimation.

Rather than relying on vulnerable three-dimensional sensor readings to perspective-warp images, the proposed framework establishes a closed-form coordinate transformation parameterized directly by two-dimensional bounding box coordinates. This formulation decouples localized ocular feature regression from global geometric projection, achieving robust line-of-sight recovery without image warping artifacts even amidst drastic driver motion.

Furthermore, integrating pre-trained CLIP vision-language semantic regularization into the dual-stream pipeline demonstrates a powerful paradigm for handling real-world occlusions. By arbitrating dynamically between ocular details and macroscopic facial geometry based on scene context, the model achieves consistent state-of-the-art accuracy across infrared, synthetic, and webcam modalities.

Future work will explore enhanced contextual reasoning under complete ocular blockage such as opaque sunglasses and lightweight structural pruning to facilitate millisecond-level deployment on resource-constrained automotive edge processors.
