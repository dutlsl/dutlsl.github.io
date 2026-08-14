---
title: "[CVPRW 2026] FlowScan: Self-Supervised Features and Flow Matching Regularization for Gaze Scanpath Prediction"
date: 2026-08-14T18:53:00+09:00
draft: false
math: true
tags: ["Paper Review", "Gaze Prediction", "Scanpath", "Flow Matching", "Self-Supervised Learning", "CVPRW 2026"]
categories: ["Paper Review"]
summary: "A review of the FlowScan paper presented at the CVPR 2026 GAZE Workshop. By integrating a frozen DINOv3 backbone, a deformable pixel decoder, and an auxiliary flow matching head discarded at inference, FlowScan outperforms the state-of-the-art HAT model across all metrics on COCO-Search18."
cover:
  image: "/images/flowscan/flowscan_architecture.jpeg"
  alt: "FlowScan Architecture Overview"
---

> Paper Information
> - Title: FlowScan: Self-Supervised Features and Flow Matching Regularization for Gaze Scanpath Prediction
> - Authors: Brahan Aklilu*, Ofir Itzhak Shahar*, Ohad Ben-Shahar (*Equal contribution)
> - Affiliation: Stein Faculty of Computer and Information Science, Ben-Gurion University of the Negev, Israel
> - Venue: CVPR 2026 GAZE Workshop (7th International Workshop on Eye and Gaze in Computer Vision)

---

## 1. One-Sentence Summary

FlowScan enhances the HAT framework by incorporating a frozen DINOv3 backbone, an adapted deformable pixel decoder, and an auxiliary flow matching head active only during training, outperforming the state-of-the-art on the COCO-Search18 benchmark across all scanpath metrics without adding inference cost.

---

## 2. Research Background and Motivation

### 2.1 Problem Definition

Visual search is a fundamental component of human visual behavior. When searching for a target object in a scene, the human visual system coordinates bottom-up saliency, top-down task guidance, and memory of prior fixations to generate an ordered sequence of eye fixations known as a scanpath.

Scanpath prediction differs from standard static saliency prediction, which only estimates where people look on average. Instead, scanpath prediction is formulated as an autoregressive generation task where each successive fixation depends on the search target, scene context, and the full history of prior fixations. Accurate scanpath modeling plays a crucial role in human-computer interaction (HCI), assistive technologies, image retrieval, and cognitive neuroscience.

### 2.2 Limitations of Existing Methods

HAT (Human Attention Transformer) represents the current state of the art on the COCO-Search18 benchmark. It combines a task-conditioned transformer encoder-decoder, a foveated working memory, and a multi-scale deformable attention pixel decoder. Despite its strong performance, HAT presents two notable architectural limitations:

First, HAT relies on a supervised ResNet-50 backbone trained on ImageNet classification. Supervised CNN features are biased toward class labels and often fail to capture the rich semantic correspondences and precise spatial structures necessary for task-guided visual search. Conversely, modern self-supervised Vision Transformers (such as the DINO family) encode fine-grained semantic boundaries and spatial context without requiring labeled data, providing representations inherently better suited for visual search.

Second, HAT is trained exclusively with a heatmap-based focal loss. While effective at supervising the peak of the 2D spatial distribution, this loss only indirectly constrains the decoder's internal feature representation to encode exact 2D coordinate positions.

### 2.3 Main Contributions

The primary contributions of this work are summarized as follows:

- Frozen Self-Supervised ViT Backbone: Replacing the supervised ResNet-50 with a frozen DINOv3 ViT-B/16 backbone achieves a +12.4% relative gain in Sequence Score (SS).
- Flow Matching Regularization: Introducing an auxiliary flow matching head directly injects continuous coordinate regression signals into the decoder representation. This head is discarded at inference time, delivering a +12.2% SS improvement with zero inference overhead.
- Rigorous Head-to-Head Comparison: Retraining HAT on an open test split (seed 42) ensures a fair comparison under an identical protocol, where FlowScan consistently outperforms HAT across all five evaluation metrics (SS 0.603 vs. 0.575, SemSS 0.538 vs. 0.499).

---

## 3. Proposed Framework

FlowScan builds upon the foundational HAT architecture while introducing three complementary enhancements: self-supervised feature extraction, multi-scale decoding, and auxiliary representation regularization. The overall framework is illustrated in Figure 1.

![FlowScan Architecture](/images/flowscan/flowscan_architecture.jpeg)

*Figure 1: Overview of the FlowScan architecture. A frozen DINOv3 backbone extracts patch tokens, which are processed by a deformable pixel decoder into coarse (P1) and fine-grained (P4) feature maps. The Foveated Working Memory constructs a token sequence combining peripheral context (pooled P1) and foveal crops (from P4). A transformer encoder-decoder generates a task-conditioned query for dense heatmap prediction and soft-argmax decoding. The flow matching head (dashed) acts as an auxiliary regularizer during training and is discarded at inference.*

### 3.1 Frozen DINOv3 Backbone

Visual search requires understanding semantic object boundaries and spatial relationships rather than just categorizing objects. While supervised CNNs are prone to label bias, self-supervised Vision Transformers capture rich spatial semantics and dense contextual representations. FlowScan adopts DINOv3 ViT-B/16, pre-trained via self-distillation on the curated LVD-142M dataset.

Crucially, the backbone is completely frozen during training, which prevents representation collapse and minimizes computational cost. DINOv3 incorporates register tokens to eliminate attention artifacts and improve dense patch representations.

An input image ($320 \times 512$) is partitioned into a $20 \times 32$ grid of 768-dimensional patch tokens. With the backbone's 85.7M parameters frozen, FlowScan contains only ~11M trainable parameters, allowing the entire model to be trained in approximately 4 hours on a single NVIDIA RTX 4090 GPU.

### 3.2 Deformable Pixel Decoder

Human visual exploration relies simultaneously on coarse peripheral vision to maintain scene context and fine foveal vision to resolve detail. To support dual-resolution processing, the Deformable Pixel Decoder expands the single-scale ViT tokens into a multi-scale pyramid and applies sparse deformable attention to selectively aggregate relevant spatial information.

The patch tokens are linearly projected to dimension $d = 256$ and reshaped into a 2D feature map. Strided convolutions construct a 3-level Feature Pyramid Network (FPN). Six layers of Multi-Scale Deformable Attention (MSDeformAttn, 8 heads, 4 sampling points per level) enable cross-scale feature interaction.

The decoder yields two resolution levels:

- P1 (coarse): A linear projection at the original ViT grid resolution ($20 \times 32$), providing global peripheral scene context.
- P4 (fine): A learnable $4\times$ upsampled feature map ($79 \times 127$) generated via transposed convolution, used for foveal crop extraction and high-resolution heatmap prediction.

Both levels maintain dimension $d = 256$. Additionally, the authors reimplemented deformable attention in pure PyTorch, eliminating custom CUDA kernel dependencies.

### 3.3 Foveated Working Memory and Transformer Encoder-Decoder

Mirroring the dual-resolution nature of human vision, the Foveated Working Memory integrates coarse global context with detailed crops from previously visited locations. The Transformer Encoder-Decoder then leverages this memory to generate a task-conditioned query for the next fixation.

At each step, the working memory consists of two token sets:

- Peripheral tokens ($7 \times 7 = 49$ tokens): Obtained by adaptive average pooling of P1, capturing persistent whole-scene context.
- Foveal tokens ($7 \times 7 = 49$ tokens per fixation): Cropped from P4 at each historical fixation location, encoding fine-grained local visual details.

Each token is enriched with scale embeddings (peripheral vs. foveal), temporal embeddings (fixation order), and 2D spatial position embeddings. For $k$ prior fixations, the memory contains $49 + 49k$ tokens in total.

A 3-layer Transformer Encoder (4 heads, FFN dimension 1024, Pre-LN) encodes the memory tokens. Subsequently, a 6-layer Transformer Decoder takes the learned task embedding of the target object category (from the 18 COCO categories) as its initial query and performs cross-attention over the encoded memory. The decoder output is a single $d$-dimensional query vector encoding the next fixation.

### 3.4 Output Heads

The 1D query vector produced by the Transformer Decoder must be translated into spatial fixation coordinates and a termination decision. The Output Heads serve this translation function through three dedicated modules:

First, the Dense Heatmap Head produces a 2D probability distribution over possible next locations. The query vector is projected through a 2-layer MLP ($d \to d$, ReLU) and compared with the fine-grained feature map P4 via pixel-wise dot products. The resulting score map is normalized with a spatial softmax and refined by a lightweight CNN (Conv $1 \to 16$, GroupNorm, ReLU, Conv $16 \to 1$) to generate the final $79 \times 127$ probability heatmap. During training, it is supervised using Focal Loss against a Gaussian-blurred ground-truth map ($\sigma = 1.5$ pixels).

Second, Soft-argmax Decoding extracts sub-pixel continuous coordinates. Instead of taking a discrete argmax that restricts predictions to integer grid positions, FlowScan computes the probability-weighted centroid within a local $9 \times 9$ window around the heatmap peak, yielding smooth continuous coordinates.

Third, the Termination Head determines whether the visual search should end. Implemented as a 2-layer MLP ($256 \to 128 \to 1$, Dropout), it outputs the probability that the current fixation concludes the scanpath and is trained using Binary Cross-Entropy Loss.

### 3.5 Flow Matching Auxiliary Head

Supervising the model solely through a 2D heatmap focal loss concentrates gradients primarily on the peak location. As a result, the decoder's internal features receive weak direct feedback regarding continuous coordinate distances.

FlowScan resolves this by introducing an auxiliary Flow Matching Head during training. Flow matching learns a velocity field that transports random Gaussian noise to the true fixation coordinate. Intuitively, given random starting points across the image, the velocity field acts as a vector flow map that guides noise particles directly to the target location, using the decoder query as conditioning context. To generate an accurate flow map, the decoder query must inherently encode precise target coordinates.

Crucially, this auxiliary head is utilized strictly during training as a representation regularizer and is completely discarded at inference time, incurring zero inference cost.

Formally, the velocity field $v_\theta(x_t, t \mid c)$ is parameterized by a 3-layer MLP conditioned on decoder output $c$. Given ground-truth fixation $x_1 \in [0, 1]^2$, initial Gaussian noise $x_0 \sim \mathcal{N}(0, I)$, and time $t \sim \mathcal{U}(0, 1)$, the linear interpolant is defined as:

$$x_t = (1 - t)x_0 + tx_1$$

The target velocity along the straight path is $v^* = x_1 - x_0$. The flow matching loss is computed as the mean squared error:

$$\mathcal{L}_{\text{FM}} = \| v_\theta(x_t, t \mid c) - (x_1 - x_0) \|^2$$

This auxiliary objective provides three key benefits:

- It creates a smoother, continuous loss landscape around the target fixation.
- It directly backpropagates continuous coordinate error into the decoder features without passing through the 2D heatmap.
- It implicitly enforces spatial locality, as the velocity field must converge toward a single coordinate point.

With only ~150K parameters (~1.4% of trainable parameters), the training overhead is negligible.

### 3.6 Overall Training Objective

The overall training loss $\mathcal{L}$ is formulated as a weighted combination of the three objectives:

$$\mathcal{L} = \lambda_{\text{heat}} \mathcal{L}_{\text{focal}} + \lambda_{\text{FM}} \mathcal{L}_{\text{FM}} + \lambda_{\text{term}} \mathcal{L}_{\text{term}}$$

Hyperparameter tuning confirms that $\lambda_{\text{heat}} = 1.0, \lambda_{\text{FM}} = 0.5, \lambda_{\text{term}} = 1.0$ provides the optimal balance between spatial accuracy, representation regularization, and termination calibration.

### 3.7 Autoregressive Scanpath Generation

During inference, scanpaths are generated autoregressively without the Flow Matching Head. Starting from an initial center fixation, the model sequentially executes four steps at each iteration:

1. Construct the Foveated Working Memory from historical peripheral and foveal tokens.
2. Encode memory tokens and decode a task-conditioned query.
3. Compute the dense heatmap and derive the next fixation coordinate via soft-argmax decoding.
4. Predict termination probability.

Generation proceeds until the termination probability exceeds 0.5 or the maximum step limit $K = 6$ is reached.

---

## 4. Experimental Results

### 4.1 Dataset and Evaluation Protocol

FlowScan is evaluated on COCO-Search18, the standard benchmark for goal-directed visual search. The dataset contains gaze recordings from 10 human participants searching for 18 object categories across 6,202 images under the target-present (TP) condition.

Because the original test split is private, the validation set was partitioned into a 70:30 split (seed 42), producing 977 test scanpaths across 319 image-task pairs. To ensure a strictly fair comparison, HAT was retrained on the exact same split using its official codebase (HAT retrained).

Evaluation metrics include Sequence Score (SS), Semantic Sequence Score (SemSS), and per-step conditional saliency metrics: conditional Information Gain (cIG), conditional NSS (cNSS), and conditional AUC (cAUC).

### 4.2 Main Results

| Method | SS↑ | SemSS↑ | cIG↑ | cNSS↑ | cAUC↑ |
|---|---|---|---|---|---|
| HAT retrained | 0.575 | 0.499 | 2.491 | 4.649 | 0.915 |
| FlowScan | 0.603 | 0.538 | 2.759 | 4.810 | 0.928 |

*Table 1: Main benchmark comparison on COCO-Search18 target-present search under an identical test split (seed 42).*

FlowScan outperforms HAT retrained across all metrics, achieving SS 0.603 (+4.9% relative) and SemSS 0.538 (+7.8% relative), along with clear improvements in cIG (2.759 vs. 2.491), cNSS (4.810 vs. 4.649), and cAUC (0.928 vs. 0.915).

Notably, retraining HAT on the open split yielded SS 0.575 compared to its published 0.470, highlighting metric sensitivity to data splits and reinforcing the necessity of identical test partitions for valid comparisons.

Furthermore, HAT retrained tends to terminate prematurely (mean scanpath length 2.58 with 60.5% consisting of just 2 fixations), whereas FlowScan achieves a mean length of 3.77, closely matching the human ground truth (3.68).

### 4.3 Backbone Ablation

| Backbone | Pre-training | SS↑ | SemSS↑ | cIG↑ | cNSS↑ | cAUC↑ |
|---|---|---|---|---|---|---|
| ResNet-50 | Supervised | 0.517 | 0.439 | −0.31 | 3.25 | 0.816 |
| ViT-B/16 | Supervised | 0.568 | 0.490 | 1.30 | 3.83 | 0.871 |
| DINO ViT-B/16 | SSL | 0.559 | 0.473 | 1.51 | 3.90 | 0.887 |
| DINOv2 ViT-B/14 | SSL | 0.586 | 0.514 | 1.83 | 4.19 | 0.896 |
| DINOv3 ViT-B/16 | SSL | 0.581 | 0.501 | 2.11 | 4.17 | 0.901 |

*Table 2: Backbone comparison using FPN pixel decoder with Flow Matching enabled (all backbones frozen).*

Self-supervised Vision Transformers consistently surpass supervised backbones. DINOv2 achieves the highest SS (0.586) and SemSS (0.514), while DINOv3 leads on conditional saliency metrics (cIG 2.11, cAUC 0.901). DINOv3 was selected as the primary backbone for its superior per-step predictive fidelity. All ViT backbones outperform ResNet-50 by +0.042–0.069 SS, confirming the advantage of frozen transformer features for scanpath modeling.

### 4.4 Flow Matching Ablation

| Flow Matching | SS↑ | ΔSS | SemSS↑ | cIG↑ | cNSS↑ |
|---|---|---|---|---|---|
| Enabled | 0.581 | — | 0.501 | 2.11 | 4.17 |
| Disabled | 0.518 | −0.063 | 0.434 | 0.78 | 3.81 |

*Table 3: Impact of flow matching regularization (DINOv3 backbone, FPN pixel decoder).*

Enabling flow matching yields a +0.063 SS gain (0.518 to 0.581, +12.2% relative) and boosts cIG from 0.78 to 2.11. This substantial improvement demonstrates that auxiliary coordinate supervision effectively shapes the decoder representation even when discarded at test time.

### 4.5 Deformable Pixel Decoder Ablation

Transitioning from an FPN decoder (SS 0.581) to the Deformable Pixel Decoder (SS 0.603) delivers an additional +3.8% relative gain (+0.022 SS) and a +31% increase in cIG (2.11 to 2.76), underscoring the value of multi-scale deformable attention for fine-grained foveal representations.

### 4.6 Hyperparameter Sensitivity

![Sensitivity Analysis](/images/flowscan/flowscan_sensitivity.jpeg)

*Figure 2: Loss weight sensitivity analysis. (a) $\lambda_{\text{heat}} = 1.0$ balances sequence similarity and information gain. (b) $\lambda_{\text{FM}} = 0.5$ provides optimal regularization. (c) $\lambda_{\text{term}} = 1.0$ is essential for calibrated termination.*

Sensitivity analysis reveals that $\lambda_{\text{heat}} = 1.0$ optimalizes both SS and cIG. $\lambda_{\text{FM}} = 0.5$ yields peak cIG (2.76), whereas higher weights over-constrain the decoder. $\lambda_{\text{term}}$ is critical: deviating from 1.0 leads to sharp performance drops in both SS and cIG.

| σ | SS↑ | SemSS↑ | cIG↑ | cNSS↑ |
|---|---|---|---|---|
| 1.0 | 0.595 | 0.528 | 2.518 | 4.758 |
| 1.5 | 0.603 | 0.538 | 2.759 | 4.810 |
| 2.0 | 0.595 | 0.523 | 2.535 | 4.710 |

*Table 4: Sensitivity to heatmap Gaussian blur scale $\sigma$ (DINOv3, deformable decoder, FM).*

Tuning the Gaussian scale indicates that $\sigma = 1.5$ pixels is optimal; $\sigma = 1.0$ produces sparse gradients, while $\sigma = 2.0$ degrades spatial localization precision.

### 4.7 Qualitative Results

![Qualitative Results](/images/flowscan/flowscan_qualitative.jpeg)

*Figure 3: Qualitative scanpath predictions. Each row depicts a test image with the search target on the left. Green dashed boxes denote target objects, numbered circles indicate fixation order, and arrows mark saccade trajectories.*

Qualitative analysis shows that FlowScan replicates human search strategies by logically scanning context before converging on the target, whereas baseline models often wander or terminate prematurely.

---

## 5. Conclusion and Key Takeaways

FlowScan demonstrates that targeted enhancements to feature representation, multi-scale decoding, and loss regularization can significantly improve scanpath prediction on COCO-Search18.

First, frozen self-supervised ViT representations substantially outperform supervised CNN features. The rich semantic groupings and emergent boundary awareness in DINOv3 provide stronger task-target alignment and higher-quality foveal crops.

Second, flow matching regularization serves as an effective mechanism for auxiliary representation learning. By supervising continuous coordinate velocity fields during training and removing the module at inference, FlowScan achieves a +12.2% relative SS gain without incurring any inference penalty. This auxiliary regularization strategy holds broad promise for other sequential spatial prediction tasks.

Third, multi-scale deformable attention provides critical cross-scale feature fusion, boosting conditional information gain by 31% over standard FPN architectures.

Promising future directions include extending FlowScan to target-absent search, exploring stochastic scanpath sampling via ODE integration of the flow matching head, and incorporating space-dependent polar saccadic priors.
