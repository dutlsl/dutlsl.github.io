---
title: "[CVPRW 2026 (GAZE Best Paper)] ECOGaze: How Much Future Helps for Causal Egocentric Gaze Estimation?"
date: 2026-08-12T16:19:00+09:00
draft: false
math: true
tags: ["Paper Review", "Egocentric Gaze Estimation", "Future-Privileged Supervision", "Knowledge Distillation", "Causal Inference", "CVPRW 2026"]
categories: ["Paper Review"]
summary: "We propose ECOGaze, a controlled future-privileged training framework that enhances strictly causal egocentric gaze estimation. We demonstrate that the benefit of future look-ahead peaks within a bounded temporal window of 1.7 to 3.3 seconds."
cover:
  image: "/images/ecogaze/_page_0_Figure_9.jpeg"
  alt: "ECOGaze framework overview"
---

## 1. One-Sentence Summary

ECOGaze is a future-privileged supervision framework that accesses future frames only during training and operates strictly causally at inference. It effectively distills anticipatory cues from future contexts into a causal model, proving that the utility of future look-ahead peaks within a bounded temporal window of approximately 1.7 to 3.3 seconds on the EGTEA Gaze+ and Ego4D benchmarks.

---

## 2. Research Background and Motivation

### 2.1 Problem Definition

Egocentric gaze estimation is the task of predicting the camera wearer's gaze fixation point from first-person video. It is a core technology for various real-time interactive applications, such as augmented reality (AR) assistance, wearable systems, and large-scale attention analysis.

![Figure 1: Comparison between offline and online gaze estimation settings](/images/ecogaze/_page_0_Figure_9.jpeg)
*Figure 1: Existing offline methods access future frames. In contrast, real-time online systems must predict gaze using only past and present observations. ECOGaze utilizes future context exclusively during training, while inference remains strictly causal.*

Most state-of-the-art models assume an offline setting where they can freely look forward and backward in time. Although look-ahead settings that access future frames are not entirely impossible to deploy online—provided we tolerate a latency equivalent to the look-ahead window—even a few frames of delay can severely degrade the user experience in highly interactive environments like AR headsets. Furthermore, allowing look-ahead during evaluation artificially inflates benchmark performance compared to real-world deployment.

Therefore, this study focuses on a strictly causal constraint with zero latency. Mathematically, the predicted gaze map $\hat{G}_t$ at time $t$ must not depend on any future frames $x_s$ ($s > t$), which means $\frac{\partial \hat{G}_t}{\partial x_s} = 0$.

Under this strict condition of not borrowing even a single future frame at inference, this research aims to answer two fundamental questions:

- Does the complete absence of future context inherently deprive causal gaze estimators of valuable anticipatory cues?
- If future context is indeed useful, how much look-ahead horizon should be used to train the causal model most effectively?

### 2.2 Limitations of Existing Methods

Early egocentric gaze estimation methods relied heavily on hand-crafted features, visual saliency, and short-term temporal modeling. Recently, transformer-based architectures like GLC have significantly improved performance by modeling long-range spatiotemporal dependencies. However, even these modern approaches mostly operate in offline settings, relying on bidirectional temporal contexts.

While some recurrent network-based methods support causal inference, systematic attempts to leverage future observations during training to enhance strictly causal inference at test time have been scarce.

Gaze anticipation, which forecasts future gaze locations, is a related but fundamentally distinct task from our objective of estimating the current gaze under a causal constraint. In addition, although privileged supervision and knowledge distillation have been explored in action recognition to manage temporal contexts, no prior work in egocentric gaze estimation has systematically profiled how the utility of look-ahead changes with varying temporal horizons.

### 2.3 Main Contributions

The main contributions of this paper are summarized as follows:

1. A controlled framework for temporal analysis: We formally define egocentric gaze estimation in a causal online setting and propose a training framework that isolates the impact of look-ahead horizons while keeping the inference architecture fixed.
2. Characterization of the optimal future range: Through extensive experiments on EGTEA Gaze+ and Ego4D, we demonstrate that future-privileged supervision consistently improves the causal baseline, with the benefits concentrating within a bounded temporal range of 1.7 to 3.3 seconds.
3. Practical guidance for real-time systems: We show that a lightweight causal decoder can successfully absorb future-aware signals during training while maintaining strict causality at inference, providing actionable insights for designing low-latency wearable systems.

---

## 3. Proposed Method: ECOGaze Framework

### 3.1 Framework Overview

![Figure 2: Overview of the ECOGaze training framework](/images/ecogaze/_page_3_Figure_0.jpeg)
*Figure 2: The training pipeline of ECOGaze. It utilizes a frozen DINOv3 encoder and a shared spatiotemporal decoder. By applying different temporal attention masks, the framework simultaneously computes the outputs of a future-aware teacher (top) and a strictly causal student (bottom). At inference, only the causal student is deployed.*

The ECOGaze framework is carefully designed to control visual representations and model capacity, allowing us to isolate and evaluate the impact of the look-ahead horizon $H$ during training. The key components are:

- Frozen DINOv3: We adopt a pretrained DINOv3 Vision Transformer from Meta as the scene encoder and keep its parameters frozen. This eliminates representation-level confounds (such as feature drift) and ensures that any performance variations are solely attributable to the future-privileged supervision.
- Shared Spatio-Temporal Decoder: This is a lightweight transformer decoder based on Divided Space-Time Attention that models temporal dynamics and eye-hand coordination from the frozen DINOv3 features. During training, two different temporal attention masks are applied:
  1. Future-aware teacher: The temporal attention mask is expanded to allow each token to attend to the past, present, and $H$ future frames.
  2. Strictly causal student: A strict lower-triangular temporal mask is enforced ($H=0$), restricting attention to past and present frames only.
- GLF & Conv Head: A compact prediction pipeline that refines features via Global-Local Focusing (GLF) and projects them through a $1 \times 1$ convolution and a temperature-scaled softmax to yield the final spatial gaze probability map $\hat{G}_t$.

### 3.2 Rationale Behind the Isomorphic Design

Sharing decoder parameters between the teacher and student branches is a critical design choice for controlled analysis. If we were to use a separate, larger teacher network, it would be impossible to determine whether the performance gains stem from the future look-ahead context or simply from the teacher's superior capacity. By sharing parameters, the temporal mask becomes the sole variable, allowing us to cleanly isolate the effect of future context.

### 3.3 Global-Local Focusing and Prediction Head

Features from the final decoder layer are processed sequentially through GLF, a $1 \times 1$ convolution, and a temperature-scaled softmax to produce the final gaze map.

- GLF (Global-Local Focusing): This module dynamically highlights gaze-relevant regions (such as hands and manipulated tools) in the patch features. A query vector, composed of a learnable spatial prior and the frame's global token, computes cosine similarity with all local patch features. This generates a residual gate $\alpha_t$ that suppresses background noise and highlights gaze targets. The gated feature is fused residually:
  
  $X_{t,n}^{\text{focus}} = X_{t,n}^{(L)} + \alpha_{t,n} X_{t,n}^{(L)}$

- $1 \times 1$ Convolution: The focused patch tokens are reshaped back into a 2D spatial grid and upsampled. A $1 \times 1$ convolution then collapses the feature channels into a single channel (gaze map logit) at the target spatial resolution (e.g., 64×64).

- Temperature-scaled Softmax: During training, this operation smooths the output distributions of the teacher and student, facilitating stable knowledge distillation. A standard softmax tends to polarize probabilities, destroying subtle spatial correlations (dark knowledge) across alternative gaze candidates. By dividing logits by a temperature parameter $\tau$ ($\tau=2$ in our setup) before softmax, we preserve these soft probability distributions to ease the transfer of anticipatory knowledge.

### 3.4 Training Objective

During training, the strictly causal student is optimized using a combination of a ground-truth loss and a distillation signal from the future-aware teacher.

Using the temperature-scaled distributions of the student ($\tilde{G}_t^{stu}$) and the teacher ($\tilde{G}_t^{fut}$), we minimize the joint objective:

$\mathcal{L} = \alpha \mathcal{L}_{GT} + \beta \mathcal{L}_{FPS}$

Each loss component is defined via KL divergence:

- $\mathcal{L}_{GT} = \sum_{t} D_{KL}(G_t \parallel \tilde{G}_t^{stu})$: The distance between the ground-truth gaze map and the student's prediction.
- $\mathcal{L}_{FPS} = \sum_{t} D_{KL}(\text{sg}(\tilde{G}_t^{fut}) \parallel \tilde{G}_t^{stu})$: The distance between the teacher's prediction (with stop-gradient applied) and the student's prediction.

Applying the stop-gradient operator $\text{sg}(\cdot)$ to the teacher branch serves two key functions:
First, it ensures that the shared decoder weights are updated primarily to improve the student's causal predictions by routing gradients exclusively through the student path.
Second, it prevents co-degradation. Without the stop-gradient, the teacher might deteriorate its own predictions to match the causal student, rather than guiding the student to anticipate the future. The teacher thus remains a fixed anchor representing optimal future-aware predictions.
We set $\alpha=1$ and $\beta=1$ across all experiments.

---

## 4. Experimental Results

### 4.1 Setup

- Datasets: EGTEA Gaze+ (24 FPS, 28 hours of cooking videos, 8,299 training / 2,022 test clips) and the gaze subset of Ego4D (30 FPS, normalized 2D gaze coordinates).
- Evaluation Metrics: We report the Adaptive F1 Score, Precision, and Recall, which measure the spatial overlap between the predicted and ground-truth gaze maps.
- Implementation Details: Models are trained on a single A5000 GPU using a frozen Meta DINOv3-ViT-S/16 backbone. We use the AdamW optimizer (learning rate $1\times10^{-4}$, weight decay 0.05, and a cosine learning rate schedule) for 25 epochs on EGTEA Gaze+ and 15 epochs on Ego4D.

### 4.2 How Much Future Context Helps

| | EGTEA Gaze+ | | | Ego4D | | |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| H | F1 | Rec. | Prec. | F1 | Rec. | Prec. |
| H = 0 (Baseline) | 44.7 | 60.3 | 35.5 | 40.7 | 56.3 | 31.9 |
| H = 1 | 45.1 | 59.2 | 36.4 | 41.8 | 56.1 | 33.4 |
| H = 3 | 45.5 | 61.5 | 36.1 | 41.5 | 56.8 | 32.7 |
| H = 5 | 45.9 | 61.1 | 36.7 | 41.9 | 55.7 | 33.6 |
| H = 7 | 45.6 | 60.3 | 36.6 | 42.6 | 56.9 | 34.0 |
| H = 10 | 45.9 | 63.6 | 35.9 | 42.7 | 57.6 | 34.0 |
| H = 15 | 45.4 | 61.5 | 36.1 | 41.9 | 56.6 | 33.2 |

*Table 1: Causal gaze estimation performance across different look-ahead horizons $H$. Future-privileged supervision consistently improves performance, peaking around $H \in [5, 10]$ before degrading at an extreme horizon of $H=15$.*

The empirical trends reveal several key characteristics of future context:

- Clear gains over the causal baseline: Incorporating future context during training improves the F1 score from 44.7 to 45.9 ($H=5/10$) on EGTEA Gaze+, and from 40.7 to 42.7 ($H=10$) on Ego4D.
- Non-monotonicity of the look-ahead horizon: Performance does not scale indefinitely with a longer future window. At $H=15$, the F1 score declines on both datasets.
- Optimal temporal window of 1.7 to 3.3 seconds: Given our frame strides, the peak performance corresponds to roughly 1.67–3.33 seconds ($H \in [5, 10]$) on EGTEA Gaze+ and 2.67 seconds ($H=10$) on Ego4D.

This optimal window of 2 to 3 seconds aligns interestingly with cognitive science. While human eye-hand coordination operates on sub-second scales (500–1000 ms), the benefits of future supervision extend further. This is because a 2-to-3-second window captures entire goal-directed actions (e.g., reaching for and grasping a tool) and subsequent object state changes. Conversely, looking too far ahead ($H=15$, ≈4–5 seconds) introduces noise from subsequent, unrelated actions, which contaminates the distillation signal.

### 4.3 Comparison with Prior Methods

| | EGTEA Gaze+ | | | Ego4D | | |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Method | F1 | Rec. | Prec. | F1 | Rec. | Prec. |
| Center Prior | 10.7 | 32.0 | 6.4 | 14.9 | 21.9 | 11.3 |
| GBVS | 15.7 | 45.1 | 9.5 | 18.0 | 47.2 | 11.1 |
| EgoGaze | 16.3 | 16.3 | 16.3 | – | – | – |
| Gaze MLE† | 26.6 | 35.7 | 21.3 | – | – | – |
| Joint Learning† | 34.0 | 42.7 | 28.3 | – | – | – |
| I3D-R50† | 40.9 | 57.2 | 31.8 | – | – | – |
| Attention Transition | 37.2 | 51.9 | 29.0 | 36.4 | 47.5 | 29.5 |
| GLC (Causal) | 41.6 | 57.9 | 32.4 | 41.2 | 56.1 | 32.5 |
| ECOGaze (Ours) | 45.9 | 61.1 | 36.7 | 42.7 | 57.6 | 34.0 |

*Table 2: Comparison with prior methods on egocentric gaze prediction. ECOGaze outperforms all reported baselines on both benchmarks.*

By deploying the best causal student variants ($H=5$ for EGTEA Gaze+ and $H=10$ for Ego4D), ECOGaze achieves state-of-the-art results under strictly causal constraints. It outperforms a causal variant of the state-of-the-art GLC model by +4.3 F1 on EGTEA Gaze+ and +1.5 F1 on Ego4D, proving that our framework is a robust methodology for deriving strong causal predictors.

### 4.4 Efficiency Analysis

![Figure 4: Accuracy-efficiency trade-off](/images/ecogaze/_page_6_Figure_0.jpeg)
*Figure 4: Accuracy-efficiency trade-off on EGTEA Gaze+ and Ego4D. ECOGaze achieves higher F1 scores with fewer GFLOPs per clip compared to the causal GLC baseline.*

| Model | EGTEA Gaze+ | | Ego4D | |
|:---|:---:|:---:|:---:|:---:|
| | Para. (M) | FPS ↑ | Para. (M) | FPS ↑ |
| GLC (Causal) | 70.18 | 30.28 | 70.18 | 29.27 |
| ECOGaze (Ours) | 14.19 | 59.01 | 14.19 | 59.73 |

*Table 3: Model size and inference throughput. ECOGaze is substantially smaller (5× fewer parameters) and faster (2× throughput) than GLC.*

Thanks to its lightweight decoder design, ECOGaze contains only 14.2M parameters—a nearly 5× reduction compared to GLC's 70.2M. Deployed on a single A5000 GPU, it runs at approximately 60 FPS, representing a 2× speedup over GLC (30 FPS). This makes ECOGaze highly suitable for real-time wearable deployments.

### 4.5 Component-wise Ablation

| SAttn | TAttn | GLF | FPS | F1 | Rec. | Prec. |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ✘ | ✘ | ✘ | ✘ | 39.1 | 63.9 | 28.2 |
| ✔ | ✘ | ✘ | ✘ | 42.3 | 58.8 | 33.0 |
| ✔ | ✔ | ✘ | ✘ | 44.4 | 58.7 | 35.8 |
| ✔ | ✔ | ✔ | ✘ | 44.7 | 60.3 | 35.5 |
| ✔ | ✔ | ✔ | ✔ | 45.9 | 61.1 | 36.7 |

*Table 4: Component-wise ablation on EGTEA Gaze+. Spatial attention (SAttn), temporal attention (TAttn), global-local focusing (GLF), and future-privileged supervision (FPS) incrementally improve performance.*

We verified the contributions of each module step-by-step:

- The baseline using only static features (F1 39.1) yields high recall but low precision, indicating scattered and imprecise gaze localization.
- Adding spatial attention (SAttn) improves F1 to 42.3, highlighting the importance of modeling intra-frame patch interactions even on top of frozen representations.
- Integrating temporal attention (TAttn) raises F1 to 44.4, demonstrating the value of short-range temporal dynamics.
- Incorporating the GLF module brings F1 to 44.7 by offering global scene guidance.
- Finally, future-privileged supervision (FPS) boosts F1 to the peak of 45.9, validating our core hypothesis that future-aware signals distill valuable anticipatory cues into the causal student.

### 4.6 Failure Case Analysis

![Figure 6: Typical failure modes of ECOGaze](/images/ecogaze/_page_7_Figure_10.jpeg)
*Figure 6: Representative failure modes under strictly causal inference: (1) early-stage gaze diffusion during visual search before fixation, (2) motion blur from rapid head rotations that degrades visual features, and (3) target ambiguity in cluttered environments.*

Operating strictly causally without future frames at inference leads to performance degradation in several challenging scenarios:

- Early-stage gaze diffusion: When a user is searching for a target but has not yet fixated, the model's predictions tend to diffuse across a wide area.
- Motion blur: Rapid head movements corrupt input frames, degrading the quality of the spatial features extracted by DINOv3.
- Target ambiguity: In cluttered scenes with multiple potential objects, the short-term causal history is sometimes insufficient to resolve the user's specific intent.

Addressing these limitations by integrating longer-term memory or multi-modal inputs remains an important direction for future work.

---

## 5. Conclusion and Key Takeaways

1. Controlled temporal analysis framework: ECOGaze features a parameter-sharing isomorphic decoder structure alongside a frozen DINOv3 encoder. By varying only the temporal attention masks, it isolates and evaluates the pure impact of future context on causal inference without changing representations or model capacity.
2. Characterization of the optimal future look-ahead: Future-privileged supervision consistently improves the causal baseline, peaking within a window of 1.7 to 3.3 seconds. This range successfully captures task-level action progression and object state changes while avoiding noise from unrelated subsequent actions.
3. Compact, real-time causal predictor: ECOGaze requires 5× fewer parameters and runs 2× faster (60 FPS vs. 30 FPS) than GLC. It achieves state-of-the-art causal results on EGTEA Gaze+ (45.9 F1) and Ego4D (42.7 F1), proving its practicality for real-time wearables.
