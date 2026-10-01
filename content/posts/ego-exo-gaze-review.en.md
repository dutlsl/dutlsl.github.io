---
title: "[CVPR 2026 Workshop Best Poster] Ego-Exo Gaze: Learning Ego-Exo Visual Representations for Conversational Gaze Estimation"
date: 2026-08-11T19:12:00+09:00
draft: false
math: true
tags: ["Paper Review", "GAZE 2026", "Gaze Estimation", "Egocentric Vision", "Self-Supervised Learning", "CVPR 2026"]
categories: ["GAZE 2026", "Paper Review"]
summary: "Meta Reality Labs and Idiap Research Institute introduce Ego-Exo Gaze, winner of the Best Poster Award at the CVPR 2026 GAZE Workshop. By aligning first-person egocentric gaze with third-person exocentric observations during training, the model resolves target ambiguity in multi-party conversations during single-frame inference."
cover:
  image: "/images/ego-exo-gaze/_page_0_Picture_10.jpeg"
  alt: "Ego-Exo Gaze Alignment Overview"
---

> Reference Paper
> - Gupta, A., Qian, Y., Gao, R., Ananthabhotla, I., Odobez, J. M., Ithapu, V. K., Murdock, C. "Learning Ego-Exo Visual Representations for Conversational Gaze Estimation." CVPR 2026 GAZE Workshop.
> - Award: The 7th International Workshop on Eye and Gaze in Computer Vision (GAZE 2026) Best Poster Award

![Figure 1: Resolving Egocentric Target Ambiguity via Ego-Exo Alignment](/images/ego-exo-gaze/_page_0_Picture_10.jpeg)
*Figure 1: Single-frame first-person imagery often suffers from target ambiguity when multiple individuals populate the field of view. The proposed framework trains on synchronized pairs of egocentric videos in a Siamese layout to learn shared ego-exo representations, while deploying a single-branch vision transformer for test-time inference on isolated single frames.*

---

## 1. One-Sentence Summary

This research leverages synchronized first-person videos captured simultaneously by pairs of conversational partners to co-train egocentric and exocentric gaze representations via self-supervised alignment, unlocking accurate single-frame egocentric gaze estimation without requiring paired video streams at test time.

---

## 2. Research Background and Motivation

### 2.1 Problem Definition

Picture two friends seated across from each other at a busy café table. The moment one speaker remarks, "Look at that," with a subtle head turn, their conversational partner instinctively deciphers whether the gaze is directed toward the window view behind them or the menu placed between them. Human dialogue is intrinsically structured around this reciprocal coordination, where one person's subjective first-person vantage point and the third-person visual cues of their partner's head and eyes continually cross-reference to align mutual attention.

Egocentric gaze estimation predicts the focal point of a camera wearer directly from their first-person visual stream. It forms a cornerstone technology for wearable augmented reality glasses, enabling adaptive user interfaces, context-aware acoustic beamforming toward active speakers in noisy rooms, and conversational agents that understand multi-party social dynamics.

However, equipping lightweight smart glasses with dedicated eye-tracking hardware imposes severe penalties on industrial design, battery life, and manufacturing cost. Consequently, inferring user attention purely from forward-facing RGB video cameras offers an attractive and practical alternative.

### 2.2 Limitations of Existing Methods

Existing first-person gaze estimation approaches rely predominantly on temporal modeling across video sequences. Yet, maintaining multi-frame temporal buffers and recurrent attention mechanisms increases compute and memory footprints by more than tenfold compared to static single-frame models, quickly exceeding the thermal and power envelopes of slim glasses.

Conversely, static single-frame predictors suffer from severe target ambiguity. In complex social settings where multiple participants gather, a single snapshot contains multiple faces and objects simultaneously, making it virtually impossible to distinguish which conversational partner the wearer is actively engaging with.

In such scenarios, integrating exocentric gaze cues—the third-person perspective captured from the partner's camera showing the wearer's own face—provides a decisive supervisory signal. However, requiring synchronized multi-view streaming at deployment is impractical due to network latency, wireless bandwidth limits, and privacy constraints.

### 2.3 Main Contributions

To overcome this dilemma, the authors propose a self-supervised training paradigm with test-time asymmetry, achieving three primary contributions:

1. They establish an efficient vision transformer baseline demonstrating that single-frame RGB inputs can resolve conversational gaze without temporal sequence overhead.
2. They develop three self-supervised alignment mechanisms—Time Synchronization, Implicit Matching, and Explicit Matching—that bind first-person gaze tokens with third-person head observations without requiring manual gaze annotations.
3. They validate through linear probing that the visual encoder genuinely internalizes third-person social gaze cues, while establishing the Looking at Heads (LAH) evaluation metric to benchmark social attention tracking.

---

## 3. Proposed Framework

### 3.1 Overview and the Ballroom Partner Analogy

The operational principle of Ego-Exo Gaze mirrors two ballroom dancers moving in synchrony. When dancer A looks directly at dancer B (first-person ego gaze), dancer B's field of view simultaneously captures dancer A's head orientation, facial expression, and directional stance (third-person exo gaze).

Because both dancers share the same social interaction, dancer A's subjective gaze intent and dancer B's external observation of dancer A form complementary reflections of a single underlying physical reality. By training the two viewpoints together, dancer A internalizes an awareness of how their body appears in space, allowing them to maintain precise spatial orientation even when dancing solo on stage.

![Figure 2: Ego-Exo Learning Architecture](/images/ego-exo-gaze/_page_3_Figure_0.jpeg)
*Figure 2: The Ego-Exo representation learning architecture. Paired visual features from participants A and B are processed by a shared vision transformer. Egocentric class tokens and exocentric head features are bound via time synchronization or head matching objectives, after which an Ego Decoder produces the predicted gaze heatmaps.*

Under this formulation, the model processes simultaneous frames $I^A$ and $I^B$ through a Siamese architecture during training.

### 3.2 Feature Extraction via Shared Vision Transformer

Input frames from both participants pass through an identical Vision Transformer encoder $V$:

$$F^A = V(I^A), \quad F^B = V(I^B)$$

Frames $I^A$ and $I^B$ represent synchronized RGB images captured at identical timestamps during conversation.

Encoder $V$ extracts patch tokens alongside a global class token, yielding dense feature representations $F^A$ and $F^B$. The encoder processes each perspective independently without parameter divergence.

### 3.3 Self-Supervised Ego-Exo Alignment

The foundational innovation of this framework lies in mathematically binding wearer A's first-person gaze representation to wearer B's observation of wearer A.

#### Strategy 1: Time Synchronization

The global class token of each participant summarizes their perceived environment:

$$G_{ego}^A = \text{CLS}(F^A), \quad G_{exo}^A = \text{CLS}(F^B)$$

Because frames recorded at identical timestamps reflect shared acoustic and environmental context, they form positive pairs. Conversely, frames sampled from different timestamps or disjoint dialogue sessions serve as negative samples.

$$S = \|G_{ego}^A - G_{exo}^A\|_2$$

A triplet margin loss minimizes the Euclidean distance between synchronized ego-exo representations while repelling asynchronous negative samples, aligning global scene semantics without bounding box detectors.

#### Strategy 2: Head Matching

To capture localized directional cues, the network isolates head regions within partner B's camera view. Applying ROI-Align over the detected head bounding box $B^B$ yields a localized exocentric feature:

$$G_{exo}^B = \text{ROI-Align}(F^B, B^B)$$

$$G_{ego}^A = \text{CLS}(F^A)$$

Here, partner B's observation of participant A's head serves as participant A's true exocentric representation. Computing the dot product yields a directional similarity score:

$$S^A = G_{exo}^B \cdot G_{ego}^A$$

Head matching branches into two loss formulations depending on annotation availability:

Explicit Matching applies cross-entropy loss when ground-truth identity labels identify which bounding box in B's view corresponds to subject A, maximizing alignment with the true interaction partner.

Implicit Matching operates fully unsupervised when identity labels are unavailable, applying an entropy loss across similarity scores to encourage the alignment distribution to sharpen autonomously around the most salient conversational target.

### 3.4 Ego Decoder and Composite Optimization

Extracted tokens flow into the Ego Decoder $D_{ego}$, consisting of four transformer layers and a linear projection head:

$$H^A = D_{ego}(F^A), \quad H^B = D_{ego}(F^B)$$

Outputs $H^A$ and $H^B$ represent 2D spatial gaze probability heatmaps corresponding to each wearer's field of view. The composite optimization objective combines task supervision with representation alignment:

$$\mathcal{L} = \mathcal{L}_{gaze}^A + \mathcal{L}_{gaze}^B + \lambda \mathcal{L}_{ego-exo}$$

Loss $\mathcal{L}_{gaze}$ enforces pixel-level cross-entropy between predicted heatmaps and ground-truth fixations. Alignment loss $\mathcal{L}_{ego-exo}$ applies the chosen triplet or cross-entropy objective, driving the visual encoder to internalize partner gaze cues.

---

## 4. Experimental Results

### 4.1 Benchmarks and Metrics

Experiments were evaluated on RLR-CHAT, a large-scale multimodal conversational dataset collected using Meta's Project Aria smart glasses, alongside the Ego4D benchmark.

![Figure 3: RLR-CHAT Dataset Session Distribution](/images/ego-exo-gaze/_page_2_Figure_0.jpeg)
*Figure 3: Session distribution and gaze characteristics within the RLR-CHAT dataset.*

Evaluation metrics track the Euclidean distance between predicted and ground-truth coordinates (Mean and Median Distance), alongside Looking at Heads (LAH) Precision, Recall, and F1 scores measuring whether fixations correctly land on conversational partners.

### 4.2 Egocentric Gaze Estimation Performance

Benchmarking against established baselines on the RLR-CHAT golden subset demonstrates the efficacy of the vision transformer architecture.

| Model | Mean Distance ↓ | Median Distance ↓ | LAH Precision ↑ | LAH Recall ↑ | LAH F1 ↑ |
|---|:---:|:---:|:---:|:---:|:---:|
| Center Prior | 0.107 | 0.093 | 0.633 | 0.146 | 0.237 |
| Average Train Gaze | 0.105 | 0.092 | 0.638 | 0.130 | 0.216 |
| Closest Head Prior | 0.131 | 0.073 | 0.396 | 0.863 | 0.543 |
| U-Net | 0.105 | 0.072 | 0.520 | 0.610 | 0.561 |
| MAV-Gaze | 0.098 | 0.065 | 0.617 | 0.724 | 0.667 |
| EgoGazeViT (Standard Training) | 0.096 | 0.057 | 0.507 | 0.798 | 0.620 |

Operating strictly on isolated static frames, EgoGazeViT achieves a median distance error of 0.057, surpassing recurrent temporal architectures like MAV-Gaze.

Evaluating the proposed Ego-Exo pre-training strategies demonstrates further quantitative gains:

| Initialization Strategy | Mean Distance ↓ | Median Distance ↓ | LAH Precision ↑ | LAH Recall ↑ | LAH F1 ↑ |
|---|:---:|:---:|:---:|:---:|:---:|
| Standard Training | 0.102 | 0.057 | 0.538 | 0.819 | 0.650 |
| Time Synchronization | 0.100 | 0.055 | 0.536 | 0.843 | 0.656 |
| Implicit Matching | 0.101 | 0.056 | 0.533 | 0.833 | 0.650 |
| Explicit Matching | 0.101 | 0.055 | 0.545 | 0.836 | 0.660 |

Explicit Matching achieves the highest LAH F1 score of 0.660, demonstrating that aligning internal representations with third-person observations directly improves partner fixation accuracy at test time.

### 4.3 Probing Exocentric Representations

To confirm whether the visual encoder genuinely acquires third-person social gaze understanding, the authors conducted linear probing experiments using a frozen encoder paired with a lightweight 2-layer MLP to predict partner fixation.

![Figure 4: Exocentric Gaze Probing Framework](/images/ego-exo-gaze/_page_7_Figure_0.jpeg)
*Figure 4: Probing architecture evaluating third-person gaze understanding from frozen visual features.*

| Initialization | Exocentric Gaze Average Precision (AP) ↑ |
|---|:---:|
| Random Initialization | 0.178 |
| Standard Training | 0.262 |
| Implicit Matching | 0.371 |
| Explicit Matching | 0.304 |
| Time Synchronization | 0.498 |

Time Synchronization elevates exocentric probing AP from 0.262 to 0.498, nearly doubling the representation quality and confirming that the encoder internalizes global social coordination across views.

### 4.4 Qualitative Analysis

![Figure 5: Qualitative Gaze Prediction in Dialogue](/images/ego-exo-gaze/_page_7_Figure_4.jpeg)
*Figure 5: Qualitative visualizations on RLR-CHAT. Top rows display egocentric gaze heatmaps with ground-truth coordinates in green. Bottom rows display exocentric head fixation predictions. The aligned model resolves conversational ambiguity across crowded visual fields.*

Visual inspections reveal that while baseline single-frame predictors diffuse attention across multiple nearby faces, the Ego-Exo aligned model sharply isolates the true active listener, mirroring natural conversational turn-taking.

---

## 5. Conclusion and Key Takeaways

Ego-Exo Gaze addresses social ambiguity in conversational gaze tracking through self-supervised multi-view alignment during training, preserving single-frame simplicity at inference.

Key takeaways for wearable computing and spatial AI include:

First, gaze is inherently reciprocal. Visual attention during social interaction cannot be understood in isolation. Co-training egocentric perspectives with third-person observations instills a shared coordinate space that captures communicative intent.

Second, training asymmetry resolves edge device bottlenecks. By distilling multi-view social dynamics into a static single-frame encoder during offline training, the system sidesteps the latency and memory costs of video models on low-power glasses.

Third, the framework lays the groundwork for multimodal conversational agents. Unifying first-person visual representations with third-person partner cues provides an ideal visual foundation for future integration with spatial audio beamforming and speaker diarization in intelligent assistants.
