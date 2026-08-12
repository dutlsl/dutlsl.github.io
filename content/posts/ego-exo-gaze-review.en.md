---
title: "[CVPR 2026 Workshop Best Poster] Learning Ego-Exo Visual Representations for Conversational Gaze Estimation"
date: 2026-08-11T19:12:00+09:00
draft: false
math: true
tags: ["Paper Review", "Gaze Estimation", "Egocentric Vision", "Self-Supervised Learning", "CVPR 2026"]
categories: ["Paper Review"]
summary: "We review the paper presented at the CVPR 2026 GAZE Workshop, which won the Best Poster Award. The authors propose a framework that leverages exocentric gaze information from other individuals in the scene via self-supervised alignment, improving single-frame egocentric gaze estimation."
cover:
  image: "/images/ego-exo-gaze/_page_0_Picture_10.jpeg"
  alt: "Ego-Exo Gaze Alignment Overview"
---

> Paper Information
> - Title: Learning Ego-Exo Visual Representations for Conversational Gaze Estimation
> - Authors: Anshul Gupta, Yijun Qian, Ruohan Gao, Ishwarya Ananthabhotla, Jean-Marc Odobez, Vamsi Krishna Ithapu, Calvin Murdock
> - Affiliations: Meta Reality Labs Research, Idiap Research Institute, EPFL, University of Maryland
> - Venue: CVPR 2026 GAZE Workshop (The 7th International Workshop on Eye and Gaze in Computer Vision). This paper was awarded the Best Poster Award.

---

## 1. One-Sentence Summary

This paper utilizes egocentric videos simultaneously recorded by a pair of wearers in conversational settings to jointly learn ego and exo gaze representations via self-supervised alignment, achieving enhanced egocentric gaze estimation using only a single frame and a single branch during inference.

---

## 2. Research Background and Motivation

### 2.1 Problem Definition

Egocentric gaze estimation is the task of predicting the camera wearer's gaze fixation point within a first-person perspective. It plays a key role in AR/VR wearables for applications such as conversational turn-taking, steering directional microphones in noisy environments, and social interaction analysis.

On lightweight wearable platforms, integrating eye-trackers is often impractical due to hardware cost, calibration complexity, and power consumption constraints. Consequently, estimating gaze solely from scene images has emerged as a viable alternative.

### 2.2 Limitations of Existing Methods

Most state-of-the-art egocentric gaze estimation methods rely on temporal cues across video sequences. However, temporal models consume up to ten times more computation and memory than static models, limiting their real-time usability on hardware-constrained devices.

Conversely, static single-frame models face significant target ambiguity when multiple individuals are present in the field of view. Incorporating exocentric gaze cues, which capture the gaze targets of other individuals in the scene from a third-person perspective, can resolve this ambiguity.

![Figure 1. Resolving egocentric gaze ambiguity using exocentric gaze cues. During training, simultaneous views are aligned via a Siamese architecture. During inference, only a single branch is used to improve gaze estimation.](/images/ego-exo-gaze/_page_0_Picture_10.jpeg)

*Figure 1: Resolving egocentric gaze ambiguity using exocentric gaze cues and Ego-Exo Alignment*

### 2.3 Main Contributions

The main contributions of this work are as follows:

- Single-frame egocentric gaze estimation: Demonstrates that static single-frame models can achieve competitive performance by leveraging modern ViT architectures.
- Ego-exo gaze representation learning: Proposes three self-supervised alignment approaches: Time Synchronization, Implicit Matching, and Explicit Matching.
- Exocentric gaze probing: Validates that the trained encoder successfully captures exocentric gaze information through a frozen probing evaluation.
- Comprehensive evaluation metrics: Introduces Distance and Looking at Heads (LAH) metrics to evaluate spatial and semantic accuracy.

---

## 3. Proposed Framework

The proposed framework employs a Siamese design during training, processing two synchronized egocentric frames $I^A$ and $I^B$ from individuals A and B.

![Figure 3. The proposed ego-exo gaze representation learning architecture. Features are extracted via shared encoders, aligned using Time Synchronization or Head Matching, and decoded to predict gaze heatmaps.](/images/ego-exo-gaze/_page_3_Figure_0.jpeg)

*Figure 3: Overview of the ego-exo gaze representation learning architecture*

### 3.1 Feature Extraction

A ViT-based Encoder $V$ extracts feature representations $F$ from the input frames:

$$F^A = V(I^A)$$
$$F^B = V(I^B)$$

This feature extraction module is applied independently to each view.

### 3.2 Ego-Exo Alignment

The Ego-Exo Alignment module enables self-supervised learning by aligning egocentric and exocentric features. An individual's egocentric gaze features, which encode where they are looking, serve as target supervision to learn the corresponding exocentric representations captured from another person's view.

#### 3.2.1 Time Synchronization

This approach aligns egocentric features across simultaneous views within the same session.

$$G_{ego}^A = \text{CLS}(F^A)$$

The exocentric features for person A are assumed to be captured in the egocentric features of person B: $G_{exo}^A = G_{ego}^B$. The similarity $S$ is calculated using the $L_2$ distance:

$$S = \|G_{ego}^A - G_{exo}^A\|_2$$

A Triplet loss encourages high similarity for positive pairs at the same timestamp while minimizing similarity against negative samples from different sessions or timestamps.

#### 3.2.2 Head Matching

Exocentric features are extracted from the local regions corresponding to head bounding boxes $B^B$ in person B's field of view using ROI-Align.

ROI-Align is a spatial pooling method that crops features from a bounding box and aligns them into a fixed-size vector. Because person A's head in B's view is an exocentric object, cropping this region yields exocentric gaze features of person A captured from B's perspective.

$$G_{exo}^B = \text{ROI-Align}(F^B, B^B)$$
$$G_{ego}^A = \text{CLS}(F^A)$$

The similarity $S^A$ between the normalized ego and exo features is computed via a dot product:

$$S^A = G_{exo}^B \cdot G_{ego}^A$$

Two formulations are explored for alignment:

- Explicit Matching: Utilizes ground truth head box identity information to calculate a Cross-Entropy loss, maximizing the similarity of the matching identity.
- Implicit Matching: Minimizes the entropy of the similarity score distribution $S$, prompting the model to automatically align with one specific head box without identity labels.

### 3.3 Prediction & Loss

The prediction module decodes the extracted visual features to reconstruct the spatial gaze distribution for each user.

It consists of an Ego Decoder $D_{ego}$ containing four Transformer layers and a linear projection layer. The Ego Decoder processes token representations to output the final egocentric gaze heatmaps $H^A$ and $H^B$:

$$H^A = D_{ego}(F^A)$$
$$H^B = D_{ego}(F^B)$$

The end-to-end objective function combines egocentric gaze prediction accuracy and cross-view alignment:

$$L = L_{gaze}^A + L_{gaze}^B + L_{ego-exo}$$

The individual loss terms are defined as follows:

- $L_{gaze}$: Pixel-wise cross-entropy between the predicted heatmap and ground truth target. It guides the model to predict the precise gaze coordinates.
- $L_{ego-exo}$: Alignment loss encouraging cross-view consistency. Depending on the method, it is formulated as Time Synchronization Loss (Triplet loss), Explicit Matching Loss (Cross-Entropy loss), or Implicit Matching Loss (Entropy loss).

---

## 4. Experimental Results

### 4.1 Datasets & Evaluation Metrics

Experiments were conducted on the RLR-CHAT and Ego4D datasets. RLR-CHAT contains multi-person conversational recordings using Aria glasses.

![Figure 2. Session size distribution in the RLR-CHAT dataset](/images/ego-exo-gaze/_page_2_Figure_0.jpeg)

*Figure 2: Session size distribution in the RLR-CHAT dataset*

Performance is evaluated using the Distance metric (mean and median $L_2$ distance) and the Looking at Heads (LAH) semantic metric (Precision, Recall, and F1-score).

### 4.2 Egocentric Gaze Estimation Performance

Baselines comparison on the RLR-CHAT golden subset is shown in Table 2.

| Model | Distance (Mean) ↓ | Distance (Median) ↓ | LAH Prec ↑ | LAH Recall ↑ | LAH F1 ↑ |
|-------|-------------------|---------------------|------------|--------------|----------|
| Predict center | 0.107 | 0.093 | 0.633 | 0.146 | 0.237 |
| Predict avg of train data | 0.105 | 0.092 | 0.638 | 0.130 | 0.216 |
| Predict closest head to center | 0.131 | 0.073 | 0.396 | 0.863 | 0.543 |
| U-Net | 0.105 | 0.072 | 0.520 | 0.610 | 0.561 |
| MAV-Gaze | 0.098 | 0.065 | 0.617 | 0.724 | 0.667 |
| EgoGazeViT (Standard Training) | 0.096 | 0.057 | 0.507 | 0.798 | 0.620 |

*Table 2: Egocentric gaze estimation comparison on RLR-CHAT*

EgoGazeViT trained with standard supervision achieves the best Distance score among image-only models.

| Initialization | Distance (Mean) ↓ | Distance (Median) ↓ | LAH Prec ↑ | LAH Recall ↑ | LAH F1 ↑ |
|----------------|-------------------|---------------------|------------|--------------|----------|
| Standard Training | 0.102 | 0.057 | 0.538 | 0.819 | 0.650 |
| Synchronization | 0.100 | 0.055 | 0.536 | 0.843 | 0.656 |
| Implicit Matching | 0.101 | 0.056 | 0.533 | 0.833 | 0.650 |
| Explicit Matching | 0.101 | 0.055 | 0.545 | 0.836 | 0.660 |

*Table 3: Performance of EgoGazeViT with different initialization methods*

Explicit Matching initialization achieves the highest LAH F1 score of 0.660.

### 4.3 Probing for Exocentric Gaze

To evaluate whether the encoder learns exocentric gaze representations, a probing analysis was conducted by freezing the encoder and training a 2-layer MLP Exo Decoder $D_{exo}$ to predict LAH targets.

![Figure 4. Probing architecture for exocentric gaze representation](/images/ego-exo-gaze/_page_7_Figure_0.jpeg)

*Figure 4: Probing architecture for exocentric gaze representation*

| Initialization | LAH AP ↑ |
|----------------|----------|
| Random init | 0.178 |
| Standard Training | 0.262 |
| Synchronization | 0.498 |
| Implicit Matching | 0.371 |
| Explicit Matching | 0.304 |

*Table 5: Exocentric gaze probing results on RLR-CHAT*

All self-supervised alignment models outperform standard training, with the Synchronization approach achieving the highest AP of 0.498.

### 4.4 Qualitative Results

![Figure 5. Qualitative gaze estimation results on RLR-CHAT. Top: Predicted egocentric gaze heatmap (green dot represents ground truth). Bottom: Predicted exocentric gaze target (LAH).](/images/ego-exo-gaze/_page_7_Figure_4.jpeg)

*Figure 5: Qualitative gaze estimation results on RLR-CHAT*

---

## 5. Conclusion and Key Takeaways

This paper introduced a novel self-supervised learning approach for conversational gaze estimation utilizing cross-view ego-exo alignment.

During inference, only a single branch is used, meaning the model requires no additional computational overhead or simultaneous camera inputs compared to standard training.

Explicit Matching yields consistent improvements in egocentric gaze prediction, while the Time Synchronization approach shows excellent exocentric representation learning and cross-dataset generalization. The study won the Best Poster Award at the CVPR 2026 GAZE Workshop, demonstrating its potential for lightweight wearables. Future work can extend this method to incorporate spatial audio and temporal sequence settings.
