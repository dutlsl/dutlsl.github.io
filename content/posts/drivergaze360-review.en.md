---
title: "[CVPR 2026] DriverGaze360: OmniDirectional Driver Attention with Object-Level Guidance"
date: 2026-10-01T17:16:58+09:00
draft: false
math: true
tags: ["Paper Review", "GAZE 2026", "Gaze", "Driver Attention", "Gaze Prediction", "Saliency", "Semantic Segmentation", "360° Vision", "CVPR 2026"]
categories: ["GAZE 2026", "Paper Review"]
summary: "DriverGaze360 breaks through the narrow front-view bottleneck of conventional driver attention estimation with a large-scale 360-degree panoramic gaze dataset and DriverGaze360-Net equipped with auxiliary attended object segmentation."
cover:
  image: "/images/drivergaze360/_page_0_Picture_14.jpeg"
  alt: "DriverGaze360 360-degree OmniDirectional Driver Attention Visualization"
---

> Reference Paper
> - Govil, S., Stricker, D., Rambach, J. "DriverGaze360: OmniDirectional Driver Attention with Object-Level Guidance." CVPR 2026.
> - Project Page: https://dfki-av.github.io/drivergaze360

![Figure 1: DriverGaze360 Overview](/images/drivergaze360/_page_0_Picture_14.jpeg)
*Figure 1: 360-degree omnidirectional attention map in DriverGaze360. Driver gaze distribution is visualized as a continuous saliency heatmap across the entire panoramic surrounding view. Unlike prior datasets confined to narrow forward views, DriverGaze360 captures full spatial gaze dynamics including side mirrors and rear sectors.*

---

## 1. One-Sentence Summary

DriverGaze360 introduces the first large-scale 360-degree panoramic driver gaze dataset to transcend the narrow forward-view bottleneck of prior driver attention modeling, accompanied by DriverGaze360-Net which jointly optimizes attention map prediction and attended object segmentation to achieve superior accuracy even under extreme panoramic gaze sparsity.

---

## 2. Research Background and Motivation

### 2.1 Problem Definition

Consider the visual scanning behavior of a driver navigating through dense urban traffic. A novice driver often suffers from severe tunnel vision, fixating solely on the brake lights of the vehicle immediately ahead. In contrast, an experienced driver's visual field is never locked into a narrow forward cone. While monitoring the road ahead, they glance at the rearview mirror to track following distances, check the side mirrors and perform shoulder checks to inspect blind spots before initiating lane changes, and actively scan cross-traffic and pedestrian crossings when approaching unsignalized intersections. If the side and rear windows were blacked out with opaque curtains, even the most skilled driver would be rendered incapable of perceiving collision threats approaching from the flanks or rear.

Safe driving fundamentally relies on this 360-degree omnidirectional visual scanning dynamic. Road safety depends on multi-directional gaze shifts across the windshield, mirrors, and blind spots. Any intelligent driver monitoring system or autonomous driving platform aiming to anticipate human attention must holistically comprehend this omnidirectional gaze behavior.

Yet for over a decade, driver attention datasets and predictive models have remained tightly constrained to single forward-facing dashboard cameras. This setup effectively blinds the system to everything outside a 60-degree frontal cone, as though driving with the side and rearview mirrors removed. In complex real-world maneuvers such as lane changes, turns, highway merges, and pedestrian interactions—where lateral and rear hazards govern survival—forward-only views inherently fail to capture the true perceptual dynamics of human drivers.

### 2.2 Limitations of Existing Methods: Blacking Out Mirrors and Blind Spots

Examining existing driver attention datasets through the lens of authentic driving reveals fundamental structural bottlenecks.

The DR(eye)VE benchmark was a pioneering effort to synchronize real driving gaze with forward video, yet it only involved 8 drivers over 6 hours of unscripted driving with a single forward camera. LBW expanded the participant pool to 28 drivers but remained strictly confined to a narrow front camera view. BDD-A focused on critical events like sudden braking and busy intersections, but its data was collected by having 1,228 passive participants watch pre-recorded videos on computer monitors rather than actively operating a vehicle. Similarly, DADA-2000 analyzed 2,000 traffic accident sequences, yet it also relied on passive video viewing across narrow forward-facing dashboard camera footage.

All prior benchmarks share an insurmountable physical limitation: the restricted field of view. A dashboard camera captures only a fraction of the full 360-degree sphere that human drivers actively monitor. Critical gaze behaviors—such as glancing at side mirrors, checking rearview mirrors during merges, or tracking cyclists in lateral blind spots—occur entirely outside the camera frame and remain unrecorded. Frontal video simply cannot reconstruct lateral and rearward visual cognition.

Predictive model architectures have inherited this spatial blindness. From multi-sensor CNN pipelines in DR(eye)VE and ConvLSTM models in BDD-A to semantic segmentation fusion in SCAFNet, feedback loops in FBLNet, and bird's-eye-view encoders in SCOUT+, all prior models process narrow forward crops. Because the input lacks lateral and rear information, even the most sophisticated neural architectures cannot synthesize omnidirectional attention dynamics.

### 2.3 Main Contributions

The core contributions of this work are threefold:

- Construction of DriverGaze360, the first large-scale 360-degree omnidirectional driver attention dataset, featuring approximately 1 million annotated gaze frames from 19 licensed drivers across regular driving, lane changes, turns, and critical near-miss hazard scenarios.
- Proposal of DriverGaze360-Net, an end-to-end transformer-based architecture that jointly optimizes 360-degree attention map estimation and attended object semantic segmentation, leveraging object-level guidance to sharpen spatial localization under sparse panoramic attention distributions.
- Formulation of the Attended Object Extraction pipeline, which intersects continuous gaze heatmaps with instance segmentations to isolate only the road users actively monitored by the driver, providing high-quality supervision signals for the auxiliary segmentation head.

---

## 3. Proposed Framework

### 3.1 Data Collection Platform: The 360-Degree Immersive Driving Cockpit

![Figure 2: Experimental Setup](/images/drivergaze360/_page_3_Picture_0.jpeg)
*Figure 2: DriverGaze360 experimental data acquisition setup. Left: First-person cockpit configuration featuring three panoramic display monitors and two digital picture-in-picture rear mirrors. AprilTag fiducial markers on screen bezels enable precise homography alignment between gaze coordinates and simulator space. Right: Generated 360-degree panoramic attention map showing pronounced gaze density on an interacting pedestrian.*

Faithfully replicating the 360-degree visual stimuli experienced by human drivers requires an immersive simulation environment where every surrounding direction is visually unobstructed. DriverGaze360 implements this requirement through a custom 360-degree virtual driving cockpit.

The CARLA simulator serves as the foundational simulation engine, providing realistic physics, high-fidelity graphics rendering, flexible sensor placement, and exact scenario reproducibility across different participants.

The visual display system comprises three primary front displays and two digital rear mirrors rendered as picture-in-picture streams. The three main monitors span front, left, and right sectors with 72 degrees field of view each, while the two upper digital mirrors cover 144 degrees of the rear hemisphere, forming a continuous 360-degree visual sphere. This setup faithfully reproduces windshield, side window, and mirror views. Participants operate the vehicle using force-feedback steering wheels, throttle, brake pedals, and gear shifters in an authentic driving posture.

Gaze tracking is handled by a Pupil Core wearable eye tracker operating at 120 Hz, synchronized with 30 Hz simulation frames. To calibrate gaze coordinates to the panoramic simulator display, AprilTag fiducial markers placed around screen bezels provide anchor points for homography transformations, minimizing geometric projection errors.

### 3.2 Traffic Scenario Design

![Figure 3: Environmental Diversity](/images/drivergaze360/_page_3_Picture_2.jpeg)
*Figure 3: Diverse environmental and traffic conditions in DriverGaze360. Samples encompass day, night, rain, and stormy weather presets across urban, suburban, and highway environments.*

If drivers were only evaluated on quiet straight roads, evasive gaze dynamics in high-risk scenarios would never be captured. DriverGaze360 incorporates three distinct scenario tiers to maximize behavioral diversity and mitigate sim-to-real bias.

First, Unscripted Free Driving allows drivers to choose their routes and maneuvers organically through mixed urban-suburban road networks. Second, Target-Oriented Navigation requires participants to follow voice instructions through road work zones, roundabouts, multi-lane crossings, and highway interchanges. Third, Safety-Critical Events inject five critical near-miss situations: highway emergency braking, abrupt merging, urban jaywalking pedestrians, oncoming vehicles during unsignalized left turns, and aggressive cut-ins.

Each scenario combines randomized weather presets (clear, night, rain, storm), lighting levels, road types, traffic densities, and pedestrian behaviors, ensuring broad visual diversity.

### 3.3 Attention Map Generation and Dataset Statistics

![Figure 4: Gaze Fixation Distribution](/images/drivergaze360/_page_3_Figure_8.jpeg)
*Figure 4: Spatial gaze fixation distribution across the entire panoramic visual field in DriverGaze360. While fixations concentrate predominantly on the forward path, distinct secondary clusters emerge around ±135 degrees and ±180 degrees, reflecting active side and rear mirror surveillance.*

Converting discrete eye fixations into continuous probability distributions involves spatio-temporal kernel smoothing. Gaze coordinates across a temporal window of ±30 frames around current timestamp $t$ are accumulated. Neighboring gaze points are projected onto frame $t$ and convolved with 2D Gaussian kernels of fixed spatial variance. Summing and normalizing these Gaussians yields a continuous 2D probability density map where frequently inspected objects form salient heatmaps.

Out of 21 recruited licensed drivers, data from 19 participants was retained after excluding two due to simulator discomfort or control inconsistency. Participants ranged from 21 to 40 years old, each with over 3 years of regular driving experience.

The key quantitative statistics of DriverGaze360 are summarized below:

| Metric | Value |
|---|---|
| Total Driving Time | ~9 hours |
| Gaze Annotated Frames | ~1,000,000 frames |
| Camera Streams | 5 synchronized streams |
| Frame Resolution | 1280 × 720 |
| Participants | 19 drivers |
| Frames per Participant | 18,000 ~ 126,000 |
| Unscripted Driving Duration | ~80 min |
| Safety-Critical Scenarios | ~85 min |
| Target-Oriented Navigation | ~370 min |
| Rear Gaze Proportion | 6% of total fixations |

Notably, rearward gaze accounts for 6% of total fixations, which aligns precisely with empirical real-world studies reporting 5% to 10% rear-mirror viewing ratios. This confirms that the simulated panoramic setup accurately mirrors human visual scanning habits on real roads.

Dataset splits are stratified geographically by CARLA towns: Towns 2, 3, 4, 7, 10, and 11 form the training set, while Towns 1, 5, and 6 serve as the held-out validation set.

### 3.4 DriverGaze360-Net Architecture: The Omnidirectional Perception Pipeline

![Figure 5: DriverGaze360-Net Architecture](/images/drivergaze360/_page_5_Figure_0.jpeg)
*Figure 5: DriverGaze360-Net overall architecture. Multi-camera RGB streams pass through a Video Swin Transformer encoder to extract multi-scale spatio-temporal features. A shared decoder fuses these representations before dispatching them into dual heads: the Attention Decoder for 360-degree gaze saliency and the Attended Object Decoder for object segmentation.*

Human visual cognition during driving follows a clear three-stage hierarchy: scanning the 360-degree panoramic scene, synthesizing multi-directional cues into an internal spatial representation in the visual cortex, and simultaneously predicting gaze fixations while identifying safety-critical objects. DriverGaze360-Net maps directly to this human cognitive pipeline.

The first component is the Video Swin Transformer encoder, acting as the digital retina scanning the surrounding environment. Consecutive frames across $T$ timestamps from 5 camera viewpoints are horizontally stitched into a $1120 \times 224$ panorama, yielding an input tensor of dimension $T \times 3 \times 224 \times 1120$. The Video Swin Transformer processes this input through four hierarchical stages, extracting rich multi-scale spatio-temporal features.

The choice of Video Swin Transformer is driven by computational efficiency. Standard Vision Transformers exhibit quadratic attention complexity with respect to image area. Since a 360-degree panorama has a 5-fold wider aspect ratio, full attention would cause prohibitive memory consumption. Shifted window attention limits computation to local 3D windows while enabling cross-window connections across layers, achieving linear complexity with respect to input resolution without sacrificing global scene context.

The second component is the Shared Convolutional Decoder, which unifies multi-scale features into a cohesive spatial representation. Multi-scale feature maps from the encoder are progressively upsampled and fused via 3D convolutions (Conv3D) and ReLU activations. This shared representation feeds both downstream decoders, allowing gaze prediction and object recognition to mutually regularize each other.

The third component consists of dual task-specific heads. The Attention Decoder applies Conv3D blocks and bilinear upsampling to produce a $1 \times H \times W$ probability map of 360-degree gaze attention. The Attended Object Decoder mirrors this architecture but terminates in a multi-class pixel classification layer, generating an $N \times H \times W$ segmentation mask. Here $N$ denotes the 5 primary road user categories (vehicles, pedestrians, cyclists, traffic signs, traffic lights) plus background.

### 3.5 Attended Object Extraction: Filtering Driving Hazards

In an expansive 360-degree panorama, hundreds of objects occupy the visual field, including trees, buildings, lane markings, and distant sky. Standard semantic segmentation indiscriminately segments all background geometry. However, human drivers ignore benign infrastructure, focusing gaze selectively on dynamic threats such as merging vehicles or pedestrians stepping into crosswalks.

Prior saliency models incorporated dense semantic segmentation maps without distinguishing whether segmented objects were actually attended to. In narrow forward views, most objects lie along the travel path, making this distinction less critical. In 360-degree panoramas, where gaze is exceptionally sparse, supervising the network to segment the entire scene injects substantial noise that degrades gaze prediction.

To resolve this, the authors introduce the Attended Object Extraction pipeline, which intersects continuous gaze heatmaps with instance masks to isolate only the objects actively monitored by the driver.

The extraction pipeline proceeds in four steps:

First, road user instances $I_{\text{road}}$ are filtered from the full instance segmentation map $I_{\text{inst}}$ according to the target class set $R_{\text{road}}$ (vehicles, pedestrians, cyclists, traffic signs, traffic lights):

$$I_{\text{road}} \leftarrow \{I_i \in I_{\text{inst}} \mid \text{class}(I_i) \in R_{\text{road}}\}$$

Second, the continuous attention map $S_{\text{sal}}$ is thresholded by $\tau$ to yield a binary saliency mask $\hat{S}_{\text{sal}}$:

$$\hat{S}_{\text{sal}} \leftarrow \mathbb{1}[S_{\text{sal}} > \tau]$$

Third, element-wise multiplication between the binary saliency mask and the filtered instance map produces the spatial overlap region $M_{\text{sal}}$:

$$M_{\text{sal}} \leftarrow \hat{S}_{\text{sal}} \odot I_{\text{road}}$$

Fourth, only instances that share non-empty intersections with the saliency mask are retained as attended objects $I_{\text{sal}}$:

$$I_{\text{sal}} \leftarrow \{I_i \in I_{\text{road}} \mid M_{\text{sal}} \cap I_i \neq \emptyset\}$$

Pixels belonging to these attended instances retain their semantic class IDs, while all non-attended pixels and background are zeroed out to form the ground-truth target $S_{\text{obj}}$. This target trains the Attended Object Decoder to focus exclusively on hazards that actively steer driver gaze.

### 3.6 Loss Formulation

The total objective function balances attention saliency prediction and attended object segmentation.

The saliency loss $\mathcal{L}_{\text{sal}}$ combines Kullback-Leibler Divergence (KLD) and Linear Correlation Coefficient (CC):

$$\mathcal{L}_{\text{sal}}(X_{\text{sal}}, Y_{\text{sal}}) = \text{KLD}(P_{X_{\text{sal}}}, P_{Y_{\text{sal}}}) - \text{CC}(X_{\text{sal}}, Y_{\text{sal}})$$

The KLD term measures distribution divergence between ground-truth attention $P_{X_{\text{sal}}}$ and predicted attention $P_{Y_{\text{sal}}}$, penalizing spatial discrepancies toward zero. The CC term evaluates linear correlation between pattern gradients; its negative sign ensures that higher pattern alignment minimizes loss. KLD enforces absolute distribution alignment, while CC aligns structural gradient patterns.

The segmentation loss $\mathcal{L}_{\text{seg}}$ integrates Dice loss, IoU loss, and Cross-Entropy loss:

$$\mathcal{L}_{\text{seg}}(X_{\text{seg}}, Y_{\text{seg}}) = -\text{Dice}(X_{\text{seg}}, Y_{\text{seg}}) - \text{IoU}(X_{\text{seg}}, Y_{\text{seg}}) + \mathcal{L}_{\text{CE}}(X_{\text{seg}}, Y_{\text{seg}})$$

Dice and IoU losses handle extreme class imbalance in panoramic views, where background accounts for over 95% of pixels, by optimizing mask overlap directly. Cross-entropy $\mathcal{L}_{\text{CE}}$ enforces pixel-wise class classification accuracy.

The overall joint objective is expressed as:

$$\mathcal{L}(X, Y) = \lambda_{\text{sal}} \cdot \mathcal{L}_{\text{sal}}(X_{\text{sal}}, Y_{\text{sal}}) + \lambda_{\text{seg}} \cdot \mathcal{L}_{\text{seg}}(X_{\text{seg}}, Y_{\text{seg}})$$

Weights $\lambda_{\text{sal}} = 1$ and $\lambda_{\text{seg}} = 1$ balance gaze localization with object segmentation equally, ensuring coordinated optimization across both modalities.

The model is initialized with a Video Swin-S backbone pretrained on Kinetics-400, trained using AdamW at learning rate $1 \times 10^{-4}$ with batch size 4 for 20 epochs on a single NVIDIA H100 GPU (approx. 24 hours training time).

---

## 4. Experimental Results

### 4.1 Quantitative Evaluation on DriverGaze360

DriverGaze360-Net was benchmarked against five leading driver attention models on DriverGaze360 using standard saliency metrics: KLD (lower is better), SIM, CC, and NSS (higher is better).

| Model | KLD ↓ | SIM ↑ | CC ↑ | NSS ↑ |
|---|:---:|:---:|:---:|:---:|
| Dr(eye)VE | 1.293 | 0.340 | 0.613 | 5.452 |
| BDDA | 2.566 | 0.218 | 0.400 | 2.772 |
| DADANet | 1.269 | 0.476 | 0.618 | 5.598 |
| ViNet++ | 1.251 | 0.442 | 0.611 | 2.772 |
| FBLNet | 1.215 | 0.494 | 0.639 | 6.012 |
| DriverGaze360-Net | 1.067 | 0.515 | 0.667 | 6.309 |

DriverGaze360-Net outperforms all existing methods across every metric, reducing KLD error by 12.18% over the second-best model while improving SIM by 4.24%, CC by 4.51%, and NSS by 4.94%.

Existing baselines falter on panoramic inputs because their architectures assume narrow, dense forward gaze. When presented with 360-degree panoramas where over 80% of the canvas contains no gaze activity, they scatter attention randomly over empty roads and sky. BDDA's KLD escalates to 2.566, demonstrating how front-view models deteriorate when confronted with panoramic sparsity.

![Figure 6: Qualitative Comparisons on DriverGaze360](/images/drivergaze360/_page_6_Figure_12.jpeg)
*Figure 6: Qualitative comparisons on DriverGaze360 during night driving. Compared to Ground Truth, BDDA disperses attention broadly across the scene, and DADANet and ViNet++ produce scattered spurious peaks. DriverGaze360-Net generates sharp, compact attention focused accurately on lead vehicle tail lights.*

Qualitative visualizations confirm this advantage. During nighttime driving, baseline models suffer from dark background noise, placing false peaks on streetlights. In contrast, DriverGaze360-Net locks attention onto relevant vehicle lights and the immediate road path.

### 4.2 Zero-Shot Generalization on Real-World DADA-2000

To verify whether models trained in simulation transfer to real roads, DriverGaze360-Net was evaluated on DADA-2000, a real-world dataset of 2,000 traffic accidents captured by dashcams at $224 \times 224$ resolution, with YOLO-v11 extracting instance masks.

| Model | KLD ↓ | SIM ↑ | CC ↑ | NSS ↑ |
|---|:---:|:---:|:---:|:---:|
| Dr(eye)VE | 2.065 | 0.325 | 0.451 | 2.920 |
| BDDA | 1.820 | 0.290 | 0.440 | 2.805 |
| DADANet | 1.646 | 0.353 | 0.484 | 3.365 |
| ViNet++ | 1.719 | 0.352 | 0.472 | 3.234 |
| FBLNet | 1.818 | 0.369 | 0.480 | 3.305 |
| DriverGaze360-Net | 1.654 | 0.396 | 0.506 | 3.478 |

DriverGaze360-Net secures the highest scores on SIM (+7.32%), CC (+4.55%), and NSS (+3.36%) over prior baselines. In KLD, it achieves 1.654, trailing DADANet's 1.646 by merely 0.48%. This demonstrates that omnidirectional simulation training equips the model with robust spatial representations that readily generalize to narrow front-view real crash sequences.

![Figure 7: Qualitative Comparisons on DADA-2000](/images/drivergaze360/_page_7_Figure_0.jpeg)
*Figure 7: Qualitative comparison on a motorcycle crash from DADA-2000. Dr(eye)VE and ViNet++ diffuse attention across the roadway, and BDDA exhibits excessive full-frame activations. DriverGaze360-Net isolates the exact accident zone with pinpoint accuracy.*

In the motorcycle accident visualization, Dr(eye)VE and ViNet++ spread attention across the asphalt, while BDDA overactivates large swaths of the visual field. DriverGaze360-Net places an accurate, compact hotspot directly over the falling motorcycle and rider.

### 4.3 Ablation Study: Impact of Attended Object Segmentation

To isolate the contribution of auxiliary attended object segmentation, three model variants were compared:

| Configuration | KLD ↓ | SIM ↑ | CC ↑ | NSS ↑ | Dice ↑ | IoU ↑ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Attention Only | 1.127 | 0.510 | 0.654 | 6.158 | - | - |
| + ObjSeg | 1.092 | 0.510 | 0.659 | 6.167 | 0.636 | 0.597 |
| + AttObjSeg | 1.067 | 0.515 | 0.667 | 6.309 | 0.639 | 0.626 |

The Attention Only baseline omits auxiliary segmentation. The + ObjSeg variant includes a standard segmentation head supervising all scene objects unconditionally. The + AttObjSeg variant is the full proposed architecture supervising only attended objects.

Relative to Attention Only, + AttObjSeg reduces KLD by 5.32% and improves SIM (+1.11%), CC (+2.08%), and NSS (+2.45%). Crucially, compared to dense segmentation (+ ObjSeg), attending only to gaze-relevant objects provides an extra 2.29% reduction in KLD and boosts CC (+1.31%) and NSS (+2.30%).

This demonstrates that selectively supervising only attended objects is far more effective than dense scene segmentation. Supervising benign background objects introduces distracting gradients, whereas attending exclusively to hazards guides the network to align gaze with task-critical road users.

![Figure 8: Attention Maps and Attended Object Segmentation](/images/drivergaze360/_page_7_Figure_2.jpeg)
*Figure 8: Attention prediction and attended object segmentation on DriverGaze360. Top: Ground Truth vs. predicted attention maps. Bottom: Ground Truth vs. predicted attended object masks. The model accurately isolates traffic lights and vehicles in task-critical zones.*

![Figure 9: DADA-2000 Attended Object Segmentation](/images/drivergaze360/_page_7_Figure_8.jpeg)
*Figure 9: Attention maps and attended object segmentation on a DADA-2000 motorcycle crash. Attention distributed across motorcycle, rider, and car in Ground Truth is faithfully recovered by the model.*

Figures 8 and 9 illustrate how DriverGaze360-Net isolates task-critical traffic lights, interacting vehicles, and falling motorcycles from surrounding visual clutter, confirming the synergistic bond between gaze estimation and hazard segmentation.

---

## 5. Conclusion and Key Takeaways

DriverGaze360 delivers a profound paradigm shift to driver attention research. For more than a decade, the community has operated within the confines of forward-facing dashboard camera frames, seeking incremental refinements in temporal modeling. Yet real driving is inherently an omnidirectional task: drivers continuously check rearview mirrors, glance at side mirrors, and look over their shoulders to track blind spots. Datasets limited to frontal cones fail to capture these fundamental driving mechanics. By expanding spatial coverage to full 360-degree panoramas, DriverGaze360 breaks free from the forward-bias constraint.

A second takeaway is the strategic viability of high-fidelity simulation for behavioral perception. Instrumenting real vehicles with panoramic gaze tracking during life-threatening collisions poses prohibitive safety and cost hurdles. Building a calibrated 360-degree virtual cockpit in CARLA enables systematic exploration of adverse weather, lighting, and near-miss hazards. The finding that rearward gaze matches the empirical 6% real-world benchmark, combined with strong zero-shot transfer on DADA-2000, establishes simulation as a credible paradigm for driving behavior research.

Finally, DriverGaze360-Net highlights that the quality of auxiliary supervision governs the performance of primary tasks. In sparse panoramic environments, indiscriminately segmenting all background structures injects harmful noise. By filtering supervision down to the objects drivers actually attend to, the Attended Object Extraction pipeline enforces meaningful semantic grounding. This object-level guidance sharpens attention distributions, preventing false activations and driving state-of-the-art accuracy.

Realizing robust autonomous driving and advanced driver assistance systems requires understanding how human drivers allocate attention across 360 degrees of space. DriverGaze360 represents a landmark foundation toward that vision.
