---
title: "[CVPR 2026] GazeShift: Unsupervised Gaze Estimation and Dataset for VR"
date: 2026-06-25T21:00:00+09:00
draft: false
math: true
tags: ["Paper Review", "GAZE 2026", "Gaze Estimation", "VR", "Unsupervised Learning", "CVPR 2026"]
categories: ["GAZE 2026", "Paper Review"]
summary: "Samsung SIRC and Bar-Ilan University present GazeShift at CVPR 2026. Introducing VRGaze, the first large-scale off-axis near-eye dataset with 2.1M images, and an unsupervised attention-driven redirection framework with self-attention loss modulation that achieves 1.84 degree accuracy and 5 ms inference on mobile VR GPUs."
cover:
  image: "/images/gazeshift/_page_1_Figure_0.jpeg"
  alt: "GazeShift Architecture Overview"
---

> Paper Information
> - Title: GazeShift: Unsupervised Gaze Estimation and Dataset for VR
> - Authors: Gil Shapira, Ishay Goldin, Evgeny Artyomov, Donghoon Kim, Yosi Keller, Niv Zehngut
> - Affiliations: Samsung Semiconductor Israel R&D Center, Samsung Electronics, Bar-Ilan University
> - Venue: CVPR 2026
> - Code & Dataset: https://github.com/gazeshift3/gazeshift

![Figure 1: GazeShift Architecture Overview](/images/gazeshift/_page_1_Figure_0.jpeg)
*Figure 1: Overview of the GazeShift framework. The Gaze Encoder extracts a gaze embedding from the target frame, while the Appearance Encoder preserves the 2D spatial layout of the source. A self-attention and cross-attention module injects the gaze conditioning into the appearance representation, enabling the Decoder to reconstruct the redirected eye image. The internal self-attention map is directly repurposed into the Gaze-Focused Reconstruction Loss without requiring external segmentation masks.*

---

## 1. One-Sentence Summary

GazeShift introduces the first large-scale off-axis infrared eye dataset VRGaze tailored to commercial head-mounted displays, and proposes an unsupervised attention-based gaze redirection framework that repurposes internal self-attention maps into a spatial loss modulation, achieving 1.84 degrees mean angular error and 5 milliseconds real-time inference on an Exynos mobile GPU.

---

## 2. Research Background and Motivation

### 2.1 Problem Definition

Imagine walking into a pitch-black gallery with only a pocket flashlight to admire a massive mural. The narrow circular beam illuminates only a tiny fraction of the canvas at any given moment, leaving the rest immersed in shadow, yet the observer comfortably comprehends the entire artwork. The human visual system operates on the exact same physiological principle, resolving fine details only within the narrow foveal region spanning one to two degrees of the visual field, while perceiving the peripheral surroundings through coarse outlines and motion cues.

Virtual and extended reality headsets exploit this biological property through foveated rendering to bypass the compute limitations of mobile graphics hardware. By directing peak rendering fidelity exclusively to the gaze focal point and rendering peripheral areas at lower resolutions, systems dramatically cut compute workloads while maintaining visual fidelity. Coupled with hands-free gaze interaction and user attention analytics, accurate low-latency gaze tracking has become a foundational pillar of modern immersive computing.

However, moving gaze estimation from controlled desktop setups into commercial wearable headsets encounters three formidable real-world barriers.

The first is camera placement geometry. To keep the visual display completely unobstructed, gaze cameras must be mounted along the lower rim or bridge of the headset in an off-axis configuration, viewing the eye from an acute angle. This oblique perspective produces severe perspective warping and frequent eyelid occlusions compared to frontal on-axis setups.

The second is the prohibitive cost and noise of gaze annotation. Even when subjects are instructed to stare at designated display targets, involuntary saccades and continuous ocular micro-tremors prevent perfect fixation, making pixel-level ground truth annotation time-consuming, expensive, and noisy.

The third is the severe compute ceiling of wearable headsets. Battery life and thermal envelopes mandate lightweight models capable of running at over sixty frames per second, whereas conventional redirection models relying on explicit 3D geometry or warping fields impose prohibitive latency.

### 2.2 Limitations of Existing Methods

Prior gaze estimation methodologies failed to address these hardware and operational constraints.

First, benchmark datasets suffered from a severe geometric mismatch. Widely adopted benchmarks such as OpenEDS2020 provide over five hundred thousand frames, but rely entirely on centered on-axis cameras. While NVGaze contains an off-axis subset, it covers only fourteen subjects with limited angular diversity. When models trained on frontal on-axis imagery are evaluated under off-axis conditions, their angular error spikes past five degrees.

Second, an unbridgeable domain gap separated remote RGB gaze estimation from near-eye infrared modalities. Existing unsupervised methods designed for full-face RGB photographs rely on facial symmetry or head pose priors. In contrast, wearable headsets capture isolated, near-eye monochrome infrared images in a light-isolated chamber, invalidating whole-face geometric assumptions.

Third, prior redirection models suffered from feature leakage and runtime bloat. Methods such as Cross-Encoder employed a shared encoder to process both appearance and gaze simultaneously. This shared pathway allowed appearance cues from the target to leak into the gaze embedding, corrupting gaze specificity. Furthermore, requiring both the appearance encoder and decoder at test time imposed excessive latency on mobile hardware.

### 2.3 Main Contributions

GazeShift overcomes these limitations through three core contributions.

1. The authors construct and release VRGaze, the first large-scale off-axis near-eye gaze dataset comprising 2.1 million infrared frames across 68 diverse subjects.
2. They develop GazeShift, an unsupervised redirection framework that pairs asymmetric encoders with single-query cross-attention to achieve clean gaze-appearance disentanglement without geometric priors.
3. They introduce the Gaze-Focused Reconstruction Loss, which repurposes internal self-attention maps into spatial weighting masks, establishing a self-reinforcing feedback loop that focuses learning on the iris and pupil without external supervision.

---

## 3. Proposed Framework

### 3.1 Overview and the Portrait Painter Analogy

The foundational intuition of GazeShift mirrors that of a master portrait painter. When a painter renders an individual across multiple sittings, structural identity markers including eyelid shape, facial contour, skin texture, and iris pigmentation remain constant. If the subject shifts their gaze to a new vantage point, the artist does not repaint the entire face from scratch, but merely re-renders the pupil and iris at the new orientation.

A near-eye camera inside a VR headset follows this exact physical invariance. Because the camera is anchored to the user's face, almost all visual variance between different frames of the same eye stems entirely from ocular rotation.

![Figure 2: VRGaze Dataset Overview](/images/gazeshift/_page_3_Figure_0.jpeg)
*Figure 2: Sample off-axis frame from VRGaze (left), frontal on-axis sample from OpenEDS2020 (center), and gaze angular distribution in VRGaze (right). Off-axis mounting prevents display interference but introduces perspective warping and occlusion, demonstrating the need for specialized benchmarks.*

Exploiting this physical consistency, GazeShift poses a generative pretext task: redirect the appearance of a source frame to match the gaze orientation of a target frame. To synthesize the target eye faithfully while retaining the source identity, the network must extract a pure gaze representation from the target. Gaze embeddings thus emerge naturally without manual labels.

### 3.2 Asymmetric Encoders for Structural Disentanglement

The architecture begins by decoupling source frame $x_s$ and target frame $x_t$ through two specialized, asymmetric encoders.

$$A_s = f_{\text{app}}(x_s), \quad g_t = f_{\text{gaze}}(x_t)$$

In this formulation, $x_s$ and $x_t$ denote frames sampled from the same eye of an identical subject at distinct timestamps, where the dominant variance is gaze angle.

The appearance encoder $f_{\text{app}}$ maps source frame $x_s$ into a spatial feature map $A_s \in \mathbb{R}^{H \times W \times C_a}$. Because appearance represents high-resolution spatial attributes like skin texture and eyelid boundaries, $f_{\text{app}}$ uses a shallow convolutional layout to preserve 2D coordinate layouts without over-abstracting spatial details.

Conversely, the gaze encoder $f_{\text{gaze}}$ processes target frame $x_t$ to produce a global gaze vector $g_t \in \mathbb{R}^{C_g}$. Because gaze orientation is a compact global attribute defined by yaw and pitch angles, $f_{\text{gaze}}$ utilizes an inverted bottleneck architecture based on MobileNetV2 with only 342K parameters and 55 MFLOPs. This deep compression extracts abstract directional signals while discarding identity cues.

Critically, this asymmetry separates training from deployment. The appearance encoder and decoder serve solely as training scaffolds, while test-time inference on the headset executes only the 342K-parameter gaze encoder.

### 3.3 Gaze-Conditioned Global Modulation via Single-Query Cross-Attention

Once features are extracted, the modulation module steers source appearance toward the target gaze direction.

First, multi-head self-attention refines the source appearance representation.

$$A_s' = \text{SelfAttn}(A_s)$$

This operation captures spatial dependencies across the eye region, including relative distances between iris boundaries, pupil glints, and eyelid contours.

Next, the target gaze embedding is injected via cross-attention.

$$c = \text{CrossAttn}(q_g, A_s', A_s')$$

Here, $q_g \in \mathbb{R}^{C_a}$ is a single global query obtained by linearly projecting $g_t$. Because gaze is uniform across the entire visual field, a single global query suffices. Interacting with the refined appearance map $A_s'$ as Key and Value, it outputs a global modulation context vector $c \in \mathbb{R}^{C_a}$.

This context vector is broadcast spatially to $C \in \mathbb{R}^{H \times W \times C_a}$ and added back to $A_s'$ through a residual connection.

$$F = A_s' + C$$

This addition steers the latent features toward the target gaze angle while preserving the spatial integrity of the source eye. Furthermore, because $g_t$ operates strictly as a query without direct skip connections to the decoder, appearance information cannot leak into the gaze embedding.

### 3.4 Gaze-Focused Reconstruction Loss

Standard generative models train on uniform pixel-wise mean squared error. However, gaze shifts modify only a localized region around the iris and pupil; sclera, skin, and backgrounds remain largely stationary. Uniform losses force the model to waste capacity memorizing static skin textures.

GazeShift solves this by converting its internal self-attention map $w \in \mathbb{R}^{H \times W}$ into a spatial loss weighting mask.

![Figure 3: Attention Map Visualization](/images/gazeshift/_page_6_Picture_9.jpeg)
*Figure 3: Source image (top) alongside the internal self-attention weight map (bottom). Without external labels, the network learns to concentrate attention on gaze-salient regions around the pupil and iris.*

After upsampling $w$ to match the image resolution and applying sharpening parameter $\gamma$, the Gaze-Focused Reconstruction Loss is defined as follows:

$$\mathcal{L}_{\text{focus}} = \frac{1}{\sum_{i} w_i^{\gamma}} \sum_{i} w_i^{\gamma} \cdot (x_{t,i} - \hat{x}_{t,i})^2$$

Index $i$ denotes individual pixel coordinates. Weight $w_i$ reflects self-attention intensity, which peaks around the iris and approaches zero across peripheral skin. The normalization factor $\frac{1}{\sum_{i} w_i^{\gamma}}$ keeps total gradient magnitudes balanced.

Examining the backpropagation gradient magnitude highlights the selective pressure of this formulation:

$$\left\| \frac{\partial \mathcal{L}_{\text{focus}}}{\partial \hat{x}_i} \right\| \propto \frac{w_i^{\gamma}}{\sum_j w_j^{\gamma}}$$

Pixels with high attention generate amplified gradients that demand faithful reconstruction, whereas background gradients are suppressed. This establishes a self-reinforcing feedback loop: sharper self-attention refines gaze reconstruction, which in turn guides the encoder to locate gaze cues with greater spatial precision.

### 3.5 Lightweight Gaze Calibration

Following unsupervised pre-training, the gaze encoder outputs embeddings that encode relative angular differences. To map these vectors to physical screen coordinates or degrees, a lightweight calibration step is performed. Using between 17 and 60 sparse fixation points per user, a Ridge regression head maps frozen gaze embeddings to angular predictions. Because encoder weights remain fixed, this procedure imposes virtually zero compute overhead.

---

## 4. Experimental Results

### 4.1 VR Benchmark Evaluations

Evaluations on the newly released VRGaze benchmark demonstrate the accuracy of GazeShift compared to supervised and unsupervised baselines.

| Training Paradigm | Model | Calibration | Mean Angular Error |
|---|---|---|:---:|
| Supervised | Appearance-Based SOTA | Fully Supervised | 1.54° |
| Supervised | Feature-Based Baseline | Fully Supervised | 3.20° |
| Unsupervised | VAE | Per-person | 5.30° |
| Unsupervised | Cross-Encoder | Per-person | 2.15° |
| Unsupervised | GazeShift (Ours) | Per-person | 1.84° |
| Unsupervised | Cross-Encoder | Person-agnostic K=200 | 2.26° |
| Unsupervised | GazeShift (Ours) | Person-agnostic K=200 | 2.13° |

Without labeled supervision during pre-training, GazeShift achieves a 1.84 degree mean error, coming within 0.3 degrees of the fully supervised baseline (1.54 degrees). It improves upon Cross-Encoder by 14%, passing the practical fidelity threshold required for seamless foveated rendering.

On the on-axis OpenEDS2020 benchmark, GazeShift achieves 3.43 degrees error, outperforming Cross-Encoder at 3.69 degrees.

### 4.2 Cross-Dataset Generalization

Testing dataset transfer between on-axis and off-axis modalities highlights the importance of VRGaze. A model trained on OpenEDS2020 degrades to an error of 5.2 degrees when evaluated on VRGaze, compared to 1.84 degrees when trained directly on off-axis data.

### 4.3 Generalization to Remote RGB Cameras: MPIIGaze

On the remote webcam RGB benchmark MPIIGaze, GazeShift demonstrates consistent architectural efficiency.

| Model | Backbone | Parameters | FLOPs | Mean Angular Error |
|---|---|:---:|:---:|:---:|
| Cross-Encoder | ResNet-18 | 11.0M | 75M | 8.32° |
| GazeShift | ResNet-18 | 11.0M | 75M | 7.56° |
| GazeShift | MobileNetV2 | 1.0M | 2M | 8.00° |

Using a MobileNetV2 backbone, GazeShift reduces model parameters by 10x and compute by 35x compared to Cross-Encoder, while achieving a lower error of 8.00 degrees.

### 4.4 Ablation Study and Hyperparameter Sensitivity

Ablation experiments isolate the contribution of each proposed component.

| Configuration | Separate Encoders | Attention Redirection | Gaze-Focused Loss | Mean Error |
|:---:|:---:|:---:|:---:|:---:|
| Baseline | No | No | No | 2.15° |
| Decoupled Encoders | Yes | No | No | 2.10° |
| Single-Query Attention | Yes | Yes | No | 2.07° |
| Full GazeShift | Yes | Yes | Yes | 1.84° |

Introducing the Gaze-Focused Reconstruction Loss delivers the single largest gain, reducing error from 2.07 to 1.84 degrees.

Varying the sharpening parameter $\gamma$ confirms that $\gamma = 1.0$ provides the optimal balance. Values below 1.0 dilute attention across static backgrounds, while values above 2.0 overly constrict attention to pupil centers, discarding corneal reflections and eyelid cues.

### 4.5 Disentanglement Verification and On-Device Latency

Controlled invariance experiments confirm clean latent separation. Under 100 synthetic illumination and contrast shifts with constant gaze, gaze embeddings remained stable with a cosine distance of 0.08. Conversely, across 80 diverse gaze angles with fixed appearance, the gaze embedding fluctuated significantly (0.17) while appearance embeddings varied by only 0.04.

![Figure 4: Latent Space Interpolation](/images/gazeshift/_page_7_Figure_11.jpeg)
*Figure 4: Latent space interpolation between two target gaze vectors. Gaze shifts smoothly while subject identity and skin textures remain stationary.*

Benchmarked on an Exynos 2200 chipset with an Xclipse 920 GPU processing binocular streams, end-to-end inference executes in 5 milliseconds, readily supporting refresh rates exceeding 120 Hz.

---

## 5. Conclusion and Key Takeaways

GazeShift demonstrates that label-free gaze estimation can achieve commercial-grade precision on resource-constrained wearable hardware. By releasing VRGaze, the authors bridge an important empirical gap for off-axis infrared camera systems.

The meta-insights offered by this work span three dimensions:

First, asymmetric design enables zero-penalty unsupervised learning. By relegating heavy generative decoders and cross-attention blocks strictly to training-time scaffolds, GazeShift leaves behind an ultra-compact 342K encoder that runs in 5 ms on edge GPUs, charting a blueprint for on-device representation learning.

Second, internal attention maps can serve as autonomous supervision signals. Repurposing existing self-attention maps into spatial loss modulators demonstrates that neural networks can bootstrap their own training without brittle heuristics or external segmentation networks.

Third, domain-specific hardware alignment is essential for real-world impact. Confronting the perspective distortions of commercial headsets with a tailored dataset unlocks practical deployment where conventional frontal models fail, illustrating the power of co-designing datasets, loss formulations, and inference architectures.
