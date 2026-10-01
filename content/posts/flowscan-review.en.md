---
title: "[CVPRW 2026] FlowScan: Self-Supervised Features and Flow Matching Regularization for Gaze Scanpath Prediction"
date: 2026-08-14T18:53:00+09:00
draft: false
math: true
tags: ["Paper Review", "GAZE 2026", "Gaze Prediction", "Scanpath", "Flow Matching", "Self-Supervised Learning", "CVPRW 2026"]
categories: ["GAZE 2026", "Paper Review"]
summary: "Presented at the CVPR 2026 GAZE Workshop, FlowScan introduces a frozen DINOv3 backbone, a deformable pixel decoder, and an auxiliary flow matching head discarded at test time, outperforming the state-of-the-art HAT model across all metrics on COCO-Search18 without adding inference cost."
cover:
  image: "/images/flowscan/flowscan_architecture.jpeg"
  alt: "FlowScan Architecture Overview"
---

> Reference Paper
> - Aklilu, B., Shahar, O. I., Ben-Shahar, O. "FlowScan: Self-Supervised Features and Flow Matching Regularization for Gaze Scanpath Prediction." CVPR 2026 GAZE Workshop.
> - Venue: The 7th International Workshop on Eye and Gaze in Computer Vision (GAZE 2026) at CVPR 2026

![Figure 1: FlowScan Architecture Overview](/images/flowscan/flowscan_architecture.jpeg)
*Figure 1: Overview of the FlowScan architecture. Patch tokens extracted from a frozen DINOv3 backbone pass through a deformable pixel decoder to yield coarse P1 and fine P4 representations. Foveated Working Memory combines peripheral global context with high-resolution crops from previous fixations. A task-conditioned transformer decoder outputs search queries, while an auxiliary flow matching head (dashed lines) injects continuous coordinate regression gradients during training before being removed at inference.*

---

## 1. One-Sentence Summary

FlowScan upgrades human visual search modeling by pairing a frozen DINOv3 vision transformer with a multi-scale deformable pixel decoder, while leveraging an auxiliary flow matching head during training to instill continuous spatial coordinate awareness without imposing test-time compute overhead.

---

## 2. Research Background and Motivation

### 2.1 Problem Definition

Consider stepping into a bustling supermarket to locate a specific brand of beverage among hundreds of densely packed items. Human observers do not scan the shelves linearly like a raster beam from top-left to bottom-right. Instead, peripheral vision rapidly surveys the global aisle geometry and category signage to isolate candidate shelves, after which the eyes execute rapid saccades directly to promising clusters. Observers retain a running working memory of examined locations to prevent redundant inspections, sequentially jumping and fixating until the target is resolved.

This ordered sequence of fixation coordinates executed during visual search is termed a scanpath. Scanpath prediction differs fundamentally from static saliency modeling, which merely predicts an aggregated spatial density of where observers look on average. In contrast, scanpath generation is an autoregressive sequential decision process where each fixation depends dynamically on scene context, task instructions, and personal fixation history.

Accurately simulating human visual search trajectories is vital for intuitive human-robot collaboration, user experience modeling in interface design, and efficient gaze-contingent foveated streaming in immersive extended reality displays.

### 2.2 Limitations of Existing Methods

The prevailing benchmark for goal-directed scanpath prediction is COCO-Search18, where the Human Attention Transformer (HAT) has long served as the dominant architecture. While HAT pioneered combining foveated memory with transformer decoders, it suffers from two core structural bottlenecks.

First, HAT relies on a supervised ResNet-50 backbone trained on ImageNet classification. Because supervised convolutional networks optimize for category discrimination rather than dense spatial geometry, they struggle to preserve object boundaries and spatial relationships crucial for visual search.

Second, training relies solely on focal loss applied to 2D fixation heatmaps. Supervising discrete probability peaks provides indirect spatial guidance, failing to penalize how far predicted distributions deviate in continuous coordinate space. Consequently, internal decoder representations lack smooth geometric awareness of target coordinates.

### 2.3 Main Contributions

FlowScan resolves these limitations through three architectural contributions:

1. A frozen self-supervised DINOv3 vision transformer backbone that replaces supervised convolutional encoders, improving Sequence Score by 12.4% purely through dense semantic representations.
2. An auxiliary Flow Matching head active exclusively during training that provides direct velocity-field regression to internal decoder features, yielding a 12.2% Sequence Score gain with zero test-time inference overhead.
3. Rigorous head-to-head re-evaluation on identical dataset splits against retrained baselines, establishing new state-of-the-art performance across all five benchmark metrics on COCO-Search18.

---

## 3. Proposed Framework

### 3.1 Overview and the Compass Guide Analogy

The operational mechanism of FlowScan can be understood through an explorer navigating fog-laden terrain with the aid of an expert compass guide. The explorer balances distant mountain silhouettes in peripheral vision with clear footholds directly beneath their boots, maintaining an internal mental map of visited landmarks while advancing toward a specified destination.

Attempting to infer spatial coordinates purely from blurry 2D elevation sketches makes navigation error-prone. However, if during practice expeditions a compass guide provides continuous vector arrows indicating the exact velocity and direction toward the goal, the explorer develops an intuitive internal sense of spatial trajectory. During real exploration, even after the compass guide departs, the traveler navigates with precise directionality.

FlowScan adopts this compass guide principle by deploying an auxiliary Flow Matching head during training to instill continuous spatial awareness into the transformer decoder.

### 3.2 Frozen DINOv3 Vision Backbone

Visual search requires fine-grained boundary sensitivity and relational spatial comprehension rather than coarse categorical labels. Pre-trained on the curated LVD-142M dataset, the DINOv3 ViT-B/16 vision transformer extracts rich dense semantic representations without label bias.

FlowScan freezes all 85.7M parameters of this backbone, entirely eliminating feature drift and restricting trainable parameters across the framework to just 11M. An input image of size 320 by 512 is tokenized into a 20 by 32 grid of 768-dimensional visual tokens.

### 3.3 Deformable Pixel Decoder

To bridge peripheral and foveal perception, the Deformable Pixel Decoder expands single-resolution ViT tokens into a multi-scale pyramid.

Linear projections map backbone tokens to dimension $d = 256$, followed by strided convolutions to construct a 3-level Feature Pyramid Network. Six layers of multi-scale deformable attention dynamically route information across scales, producing two distinct outputs:

The P1 feature map preserves the coarse 20 by 32 grid resolution, capturing global peripheral context.

The P4 feature map is upsampled fourfold via transposed convolutions to a resolution of 79 by 127, serving foveal patch crops and high-resolution heatmap prediction.

### 3.4 Foveated Working Memory and Task Decoder

At each generation step, Foveated Working Memory constructs a dynamic token sequence:

Forty-nine peripheral tokens are extracted by applying adaptive average pooling over P1, capturing global scene layout.

For each accumulated fixation history step, forty-nine foveal tokens are cropped from P4 around the corresponding gaze center, preserving localized detail.

Augmented with scale, temporal order, and 2D spatial coordinate embeddings, this token sequence enters a 3-layer Transformer Encoder. Subsequently, a 6-layer Transformer Decoder interacts with a learnable task embedding corresponding to one of the eighteen search categories via cross-attention, compressing next-fixation intent into a 256-dimensional query vector $c$.

### 3.5 Output Heads and Sub-Pixel Coordinate Decoding

The output stage translates query vector $c$ into continuous spatial coordinates and termination decisions.

The Dense Heatmap Head computes pixel-wise dot products between a projection of query $c$ and high-resolution feature map P4, generating a 79 by 127 similarity map. Spatial softmax normalization and lightweight convolutional filtering yield a 2D fixation probability distribution.

Rather than taking a discrete argmax over grid cells, FlowScan computes soft-argmax centroids over a 9 by 9 spatial window surrounding the distribution peak, deriving smooth sub-pixel continuous coordinates.

The Termination Head processes query $c$ through a two-layer MLP to predict the probability that the target object has been located, concluding visual search.

### 3.6 Geometric Regularization via Flow Matching

The central theoretical contribution of FlowScan is the Flow Matching Auxiliary Head.

Standard focal loss operates on 2D discrete grids, neglecting continuous metric distance. FlowScan introduces Flow Matching to construct a vector velocity field that guides points from random Gaussian noise toward ground-truth fixation coordinates.

Let $x_1 \in [0, 1]^2$ represent the normalized 2D ground-truth coordinate of the next fixation, and let $x_0 \sim \mathcal{N}(0, I)$ represent initial Gaussian noise. An optimal linear trajectory for time $t \sim \mathcal{U}(0, 1)$ is formulated as follows:

$$x_t = (1 - t)x_0 + tx_1$$

Along this straight path, the ideal target velocity simplifies to constant vector $v^* = x_1 - x_0$. The Flow Matching head parameterizes a lightweight MLP velocity field $v_\theta(x_t, t \mid c)$ conditioned on decoder query $c$. The objective minimizes mean squared error against target velocity:

$$\mathcal{L}_{\text{FM}} = \| v_\theta(x_t, t \mid c) - (x_1 - x_0) \|^2$$

This formulation forces query vector $c$ to encode continuous metric coordinates. If $c$ fails to preserve precise spatial localization, the velocity head cannot predict direction vector $x_1 - x_0$. Gradients bypass the discrete heatmap to supervise decoder features directly.

The unified training objective balances heatmap accuracy, coordinate regularization, and search termination:

$$\mathcal{L} = \lambda_{\text{heat}} \mathcal{L}_{\text{focal}} + \lambda_{\text{FM}} \mathcal{L}_{\text{FM}} + \lambda_{\text{term}} \mathcal{L}_{\text{term}}$$

Hyperparameter optimization yields weights $\lambda_{\text{heat}} = 1.0, \lambda_{\text{FM}} = 0.5, \lambda_{\text{term}} = 1.0$. At test time, the Flow Matching head is entirely discarded, incurring zero runtime compute or memory penalty.

---

## 4. Experimental Results

### 4.1 Benchmark Evaluation on COCO-Search18

Evaluations were conducted on the target-present split of COCO-Search18. To ensure rigorous benchmarking, the authors retrained the official HAT codebase on an identical test split (seed 42).

| Method | Sequence Score ↑ | Semantic Sequence Score ↑ | conditional Info Gain ↑ | conditional NSS ↑ | conditional AUC ↑ |
|---|:---:|:---:|:---:|:---:|:---:|
| HAT (Retrained Baseline) | 0.575 | 0.499 | 2.491 | 4.649 | 0.915 |
| FlowScan (Ours) | 0.603 | 0.538 | 2.759 | 4.810 | 0.928 |

FlowScan achieves a Sequence Score of 0.603, improving by 4.9% over retrained HAT (0.575). Semantic Sequence Score rises by 7.8% to 0.538, with consistent leads across conditional information gain (cIG), cNSS, and cAUC metrics.

Furthermore, while retrained HAT terminates prematurely with an average scanpath length of 2.58 fixations (compared to human ground truth at 3.68), FlowScan generates an average trajectory length of 3.77, closely matching human search endurance.

### 4.2 Visual Backbone Comparison

Freezing visual encoders across backbones demonstrates the superiority of self-supervised representations.

| Backbone | Supervision | Sequence Score ↑ | Semantic Sequence Score ↑ | conditional Info Gain ↑ | conditional AUC ↑ |
|---|---|:---:|:---:|:---:|:---:|
| ResNet-50 | Supervised | 0.517 | 0.439 | −0.31 | 0.816 |
| ViT-B/16 | Supervised | 0.568 | 0.490 | 1.30 | 0.871 |
| DINO ViT-B/16 | Self-Supervised | 0.559 | 0.473 | 1.51 | 0.887 |
| DINOv2 ViT-B/14 | Self-Supervised | 0.586 | 0.514 | 1.83 | 0.896 |
| DINOv3 ViT-B/16 | Self-Supervised | 0.581 | 0.501 | 2.11 | 0.901 |

All vision transformer backbones significantly outperform supervised ResNet-50. DINOv3 was selected for the final architecture due to its superior step-wise information gain (2.11 cIG) and cAUC (0.901).

### 4.3 Ablation of Flow Matching Regularization

Ablating the auxiliary Flow Matching head isolates its contribution during training.

| Flow Matching Regularization | Sequence Score ↑ | Relative Change | Semantic Sequence Score ↑ | conditional Info Gain ↑ |
|:---:|:---:|:---:|:---:|:---:|
| Disabled | 0.518 | Baseline | 0.434 | 0.78 |
| Enabled (Training Scaffold) | 0.581 | +0.063 (+12.2%) | 0.501 | 2.11 |

Even though the module is discarded before inference, its training presence increases Sequence Score from 0.518 to 0.581, while nearly tripling cIG from 0.78 to 2.11. Integrating the Deformable Pixel Decoder pushes the final score to 0.603.

### 4.4 Hyperparameter Sensitivity

![Figure 2: Loss Weight Sensitivity Curves](/images/flowscan/flowscan_sensitivity.jpeg)
*Figure 2: Sensitivity curves across loss weights. Optimal trade-offs occur at $\lambda_{\text{FM}} = 0.5$ and $\lambda_{\text{term}} = 1.0$, maintaining stable spatial regression without over-constraining the decoder.*

Setting $\lambda_{\text{FM}} = 0.5$ achieves optimal balance, while excessive weighting degrades performance by restricting decoder flexibility. Applying Gaussian blur with $\sigma = 1.5$ pixels on target heatmaps provides optimal smoothness.

### 4.5 Qualitative Search Trajectories

![Figure 3: Qualitative Scanpath Visualizations](/images/flowscan/flowscan_qualitative.jpeg)
*Figure 3: Qualitative scanpath predictions on COCO-Search18. Dashed boxes denote target objects, numbered circles indicate fixation steps, and arrows trace saccades. FlowScan mirrors human search strategies, transitioning smoothly from context exploration to target localization.*

Visual inspections show that while baselines often fixate erratically on irrelevant backgrounds or terminate prematurely, FlowScan explores contextually plausible regions before converging accurately upon the target.

---

## 5. Conclusion and Key Takeaways

FlowScan establishes a refined methodology for sequential gaze scanpath modeling by pairing frozen self-supervised vision representations with continuous geometric regularization.

Broader insights from this research include:

First, generative flow matching can serve effectively as an auxiliary training scaffold. Generative tools need not be restricted to heavy sampling pipelines. Harnessing continuous velocity fields to regularize latent representations during training unlocks substantial accuracy gains at zero inference cost.

Second, self-supervised representations align naturally with human cognitive search. The substantial leap over supervised CNN backbones proves that human visual exploration is driven by fine-grained spatial boundaries and relational semantic contexts rather than discrete categorization labels.

Third, standardized 1-to-1 benchmarking is indispensable. Re-training baselines on identical data splits exposes hidden implementation variations, ensuring that performance claims reflect genuine architectural progress.
