---
title: "[CVPR 2026] VSFOT: Vision-Language Model Guided Source-Free Domain Adaptation via Optimal Transport Paper Review"
date: 2026-07-03T18:52:00+09:00
draft: false
math: true
tags: ["Paper Review", "Domain Adaptation", "Optimal Transport", "Vision-Language Model", "CVPR 2026"]
categories: ["Paper Review"]
summary: "A review of the VSFOT paper presented at CVPR 2026. The paper successfully achieves source-free domain adaptation (SFDA) without pseudo-labels by aligning target data using Optimal Transport guided by the semantic knowledge of a pre-trained Vision-Language Model, enabling bidirectional distillation between the VLM and task model."
cover:
  image: "/images/vsfot/_page_0_Picture_11.jpeg"
  alt: "VSFOT Paradigm"
---

> **Paper Information**
> - **Title:** Vision-Language Model Guided Source-Free Domain Adaptation via Optimal Transport
> - **Authors:** Shuo Han, Xu Tang, Jingjing Ma, Xiangrong Zhang
> - **Affiliation:** Xidian University, China
> - **Conference:** CVPR 2026
> - **Code:** [github.com/TangXu-Group/VSFOT](https://github.com/TangXu-Group/VSFOT)

---

## 1. One-Line Summary

A paper that successfully achieves Source-Free Domain Adaptation (SFDA) without the noise issue of pseudo-labels by aligning target data via Optimal Transport using semantic knowledge from a pre-trained Vision-Language Model (VLM), and establishing "Bidirectional Distillation" to exchange knowledge between the VLM and the task model.

---

## 2. Background and Motivation

### 2.1 The Need for Source-Free Domain Adaptation (SFDA)

Deep learning models suffer significant performance degradation when applied to a target domain with a different distribution from the source domain they were trained on. Traditionally, Unsupervised Domain Adaptation (UDA) methods rely on accessing source data to align it with target data. However, in fields like autonomous driving or healthcare where data privacy and security are paramount, accessing source data is often impossible. SFDA has emerged as a practical alternative to adapt to a target environment using only a pre-trained source model without accessing the original source data.

### 2.2 Limitations of Existing SFDA Methods

| Limitation | Description |
|------|------|
| **Confirmation Bias** | Existing methods typically assign pseudo-labels to target data using the model itself and retrain on them. This causes initial prediction errors to accumulate, exacerbating bias. |
| **Uncontrollable Noise** | Even with techniques like selecting high-confidence samples or entropy minimization, large domain gaps make it fundamentally impossible to eliminate noise embedded in pseudo-labels. |
| **Side Effects of Hard Labels** | Even when using external models like VLMs to provide hard labels, incorrect labels still persist, causing interference that degrades adaptation performance. |

### 2.3 Main Contributions

1. **Bidirectional Distillation Framework**: Proposed an interactive, iterative optimization framework between the source model and an external VLM, enhancing adaptation performance through bidirectional knowledge distillation.
2. **Noise Suppression via OT-based Distribution Alignment**: Reinterpreted domain adaptation as an Optimal Transport (OT) problem, thereby preventing noise interference from incorrect pseudo-labels and ensuring robust predictions.
3. **Achieved SOTA Performance**: Demonstrated the superiority of the proposed framework across 4 major benchmarks, significantly outperforming existing SFDA methods and proving its strong competitiveness.

![Comparison of VSFOT Paradigm](/images/vsfot/_page_0_Picture_11.jpeg)
*Figure 1: (Left) Conventional SFDA adapting with only the source model and unlabeled target data. (Right) VSFOT framework utilizing external prior knowledge (VLM) to guide and enhance adaptation performance.*

---

## 3. VSFOT Framework

### 3.1 Problem Definition

The domain adaptation setup is as follows:

Labeled data distribution in the source domain:
$$ \mathcal{D}_s = \{(x_i^s, y_i^s)\}_{i=1}^{N_s} $$

Unlabeled data distribution in the target domain:
$$ \mathcal{D}_t = \{x_i^t\}_{i=1}^{N_t} $$

- **Task Model**: A model pre-trained on source data (composed of a feature extractor $f$ and a classifier $c$).
- **Goal**: Adapt the task model to the target domain using prior knowledge from an auxiliary VLM, without access to any ground truth labels or source data.

### 3.2 Overall Architecture Overview

VSFOT alternately executes two stages, **VGMA** and **MGVA**, realizing bidirectional distillation between the VLM and the task model.

1. **VGMA Stage**: The VLM guides the adaptation direction of the task model.
2. **MGVA Stage**: The task model refines the VLM using its high-confidence predictions.

![VSFOT Overall Architecture](/images/vsfot/_page_2_Figure_0.jpeg)
*Figure 2: Overall architecture of the VSFOT framework. (Left) VGMA stage (Center) MGVA stage*

---

### 3.3 Stage 1: VGMA (VLM-Guided Model Alignment)

This stage aligns the features of target data with the class prototypes (classifier weight vectors $w_j$) of the source model.

#### Step 1: Defining Joint Distribution Optimal Transport Cost

The transport cost between a target sample $x_i^t$ and a prototype $w_j$ is defined by the distance in feature space and the difference in prediction probabilities:
$$ C_{i,j}^m = \alpha \cdot d(z_i^m, w_j) + L(q_i^m, y_j) $$
- $z_i^m$: Feature representation
- $q_i^m$: Predictive probability distribution
- $d$: Cosine distance
- $L$: Cross-entropy loss

#### Step 2: Injecting VLM Semantic Priors (Cost Matrix Decoupling)

Since the above cost relies solely on the task model's predictions, bias occurs when the domain gap is large. To prevent this, a separate cost based on the VLM's prediction probabilities is defined to plan the transport direction:
$$ C_{i,j}^v = \alpha \cdot (1 - q_{i,j}^v) + L(q_i^m, y_j) $$
Here, $q_i^v$ is derived from the inner product of the VLM's image and text embeddings:
$$ q_i^v = \text{softmax}(z_{\text{img}} \cdot z_{\text{text}}^{\top}) $$
The core idea is decoupling: actual model updates utilize the task model's cost ($C^m$), while the transport plan direction is guided by the VLM's cost ($C^v$).

#### Step 3: Optimal Transport Plan and Loss Function

The optimal transport plan $\Gamma^*$ is computed using the VLM cost matrix $C^v$:
$$ \Gamma^* = \arg\min_{\Gamma} \int C^v d\Gamma $$
The final parameter update is performed using an alignment loss that combines this transport plan with the original model's cost:
$$ \mathcal{L}_{\text{Align}} = \sum_{i} \sum_{j} \Gamma_{i,j}^* C_{i,j}^m $$

---

### 3.4 Stage 2: MGVA (Model-Guided VLM Adaptation)

VLMs generally lack detailed, domain-specific features of the current target data. To compensate, the most reliable predictions (Top-k) from the task model are used to retro-teach the VLM. The VLM is fine-tuned by adding only a lightweight Adapter module.

The KL divergence between the task model's sparse distribution $\tilde{q}_i^m$ and the VLM's prediction $q_i^v$ is minimized:
$$ \mathcal{L}_{\text{VLM}} = \frac{1}{|B|} \sum_{i} \text{KL}(\tilde{q}_i^m \parallel q_i^v) $$
Through this, the VLM learns the target domain characteristics, enabling it to provide more accurate guidance in the subsequent VGMA stage.

---

### 3.5 Overall Training Flow

- **VGMA Stage**: Optimize the task model (freeze the VLM).
- **MGVA Stage**: Refine the VLM adapter (freeze the task model).
This process is repeated alternately, allowing the two models to synergistically improve performance.

---

## 4. Experimental Results

### 4.1 Main Benchmark Results

- **DomainNet-126**: VSFOT achieved 82.63% (a 2.55% improvement over the existing VLM-based DIFO).
- **Office-Home**: VSFOT achieved 85.34% (a 1.14% improvement over existing ProDe).

**Interpretation**: Even under the harsh conditions of lacking ground truth labels (SFDA), the proposed method showed performance equal to or surpassing existing UDA methodologies.

### 4.2 Effects of Bidirectional Distillation and Hyperparameters

- **Bidirectional Distillation**: As training progresses, the VLM and task model interact, and the task model ultimately significantly outperforms the original VLM's zero-shot performance.
- **Top-k Filtering**: Passing only the top 3 ($k=3$) probabilities when the task model teaches the VLM was most effective at suppressing noise.

---

## 5. Summary of Main Contributions

To address the confirmation bias problem of pseudo-labels in existing Source-Free Domain Adaptation (SFDA) research, this paper presents the following three main contributions:

1. **Establishment of a Bidirectional Distillation System**
   Designed an interactive structure between external prior knowledge (VLM) and the task model. The VLM suggests the correct alignment direction to the task model, while the task model feeds back detailed target domain features to the VLM, creating a virtuous cycle that reinforces both.

2. **Noise Suppression Mechanism via Optimal Transport (OT)**
   Reinterpreted the adaptation process as an Optimal Transport problem and introduced a strategy of decoupling the cost matrix. Instead of blindly matching imperfect ground truth pseudo-labels, it performs a distribution-level soft alignment, fundamentally blocking semantic noise interference.

3. **Achieved Overwhelming SOTA Benchmarks**
   Outperformed all existing SFDA models on 4 challenging benchmarks: Office-31, Office-Home, VisDA, and DomainNet-126. Notably, it recorded performance comparable to UDA methods that directly access original source data, strongly demonstrating the framework's practicality.
