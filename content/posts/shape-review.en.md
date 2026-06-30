---
title: "[CVPR 2026] SHAPE: Structure-aware Hierarchical UDA with Plausibility Evaluation for Medical Image Segmentation — Paper Review"
date: 2026-06-30T18:00:00+09:00
draft: false
math: true
tags: ["Paper Review", "Domain Adaptation", "Medical Image Segmentation", "CVPR 2026"]
categories: ["Paper Review"]
summary: "Review of the SHAPE framework presented at CVPR 2026. A new UDA paradigm for medical imaging via hypergraph-based anatomical plausibility evaluation."
cover:
  image: "/images/shape/pipeline.jpeg"
  alt: "SHAPE Framework Pipeline"
---

> **Paper Info**
> - **Title:** SHAPE: Structure-aware Hierarchical Unsupervised Domain Adaptation with Plausibility Evaluation for Medical Image Segmentation
> - **Authors:** Linkuan Zhou, Yinghao Xia, Yufei Shen, Xiangyu Li, Wenjie Du, Cong Cong, Leyi Wei, Ran Su, Qiangguo Jin
> - **Affiliation:** Northwestern Polytechnical University, HIT, USTC, Macquarie University, Macao Polytechnic University, Tianjin University
> - **Conference:** CVPR 2026
> - **Code:** [github.com/BioMedIA-repo/SHAPE](https://github.com/BioMedIA-repo/SHAPE)

---

## 1. One-line Summary

In cross-modality unsupervised domain adaptation (UDA) for medical image segmentation, the authors propose **SHAPE**, a framework integrating **class-aware Hierarchical Feature Modulation (HFM)** built on DINOv3, **Hypergraph Plausibility Estimation (HPE)** for global anatomical validity, and **Structural Anomaly Pruning (SAP)**, achieving SOTA Dice scores of **90.08%** (MRI→CT) and **78.51%** (CT→MRI) on cardiac data, and **87.48%** / **86.89%** on abdominal data.

---

## 2. Background and Motivation

### 2.1. Problem Definition

Medical image segmentation models assume identical training and test distributions, but real clinical environments exhibit severe performance degradation due to variations in **imaging equipment, modalities (CT/MRI), and acquisition parameters**. UDA aims to transfer knowledge from a labeled source domain to an unlabeled target domain, avoiding costly re-annotation.

### 2.2. Limitations of Existing Methods

Existing UDA approaches fall into two categories, each with fundamental limitations:

| Category | Representative Methods | Key Limitation |
|----------|----------------------|----------------|
| **Alignment-based** | CycleGAN, SIFA, SASAN, DDFP | Content-agnostic global transformation → destroys fine-grained cross-class mappings |
| **Pseudo-label-based** | MAPSeg, GenericSSL, IPLC, UPL-SFDA | Relies only on pixel-level confidence (entropy, consistency) → admits anatomically impossible pseudo-labels |

> The core issues reduce to two:
> 1. **Semantically unaware feature alignment:** Monolithic style transfer averages style characteristics across distinct anatomical structures, losing class-specific information
> 2. **Disregard for global anatomical constraints:** Pixel-level validation cannot prevent formation of pseudo-labels with anatomically impossible shapes or spatial arrangements

---

## 3. Proposed Method: SHAPE Framework

### 3.1. Overall Architecture

![SHAPE Framework Pipeline (Paper Figure 1)](/images/shape/pipeline.jpeg)

SHAPE consists of a synergistic pipeline with three core modules:

1. **Hierarchical Feature Modulation (HFM):** Class-aware, spatially-differentiated feature alignment
2. **Hypergraph Plausibility Estimation (HPE):** Global anatomical plausibility assessment via hypergraphs
3. **Structural Anomaly Pruning (SAP):** Cross-view stability-based structural artifact removal

### 3.2. Hierarchical Feature Modulation (HFM)

From a frozen **DINOv3 ViT** encoder, we extract a dense feature map $\mathbf{F} = \Phi(\mathbf{I}) \in \mathbb{R}^{C \times H \times W}$ and apply a **dual-granularity** approach.

**Global Level: AdaIN Style Alignment**

Compute the stylized map $\mathbf{F}_{s \to t}$ aligning textural properties between source $\mathbf{F}_s$ and target $\mathbf{F}_t$:

$$
\mathbf{F}_{s \to t} = \sigma(\mathbf{F}_t) \left( \frac{\mathbf{F}_s - \mu(\mathbf{F}_s)}{\sigma(\mathbf{F}_s) + \epsilon} \right) + \mu(\mathbf{F}_t) \tag{1}
$$

where $\mu(\cdot)$ and $\sigma(\cdot)$ denote channel-wise mean and standard deviation.

**Local Level: Structure-aware Token Mixing**

After upsampling feature maps, each token's **purity score** is computed to classify it as a semantic core (pure) or structural boundary (impure):

$$
\mathcal{P}(\mathbf{m}^i) = \frac{\max_{k \in \{0..K-1\}} \sum_{v \in \mathbf{m}^i} \mathbb{I}(v=k)}{|\mathbf{m}^i|} \tag{2}
$$

A differentiated modulation strategy is then applied:

- **Pure tokens (semantic cores):** Direct linear interpolation with same-class target tokens: $(1-\lambda)\mathbf{f}_s^i + \lambda\mathbf{f}_t^j$
- **Impure tokens (structural boundaries):** Normalization with interpolated statistics from source and target boundary tokens

> [!IMPORTANT]
> **Key insight of HFM:** Conventional AdaIN applies uniform transformation to all features, destroying inter-class separability. HFM distinguishes between anatomical cores and boundaries, applying optimized modulation to each, thereby bridging the domain gap while preserving fine-grained discriminative structure.

### 3.3. Hypergraph Plausibility Estimation (HPE)

Each predicted segmentation map $\mathbf{M} \in \{0,...,K-1\}^{H \times W}$ is modeled as a **multi-level structural hypergraph** $\mathcal{G} = (\mathcal{V}, \mathcal{E})$.

**Vertex Reliability Score:**

$$
S_{\text{vertex}}(\mathcal{G}) = \frac{1}{|\mathcal{V}|} \sum_{p \in \mathcal{V}} w_p \tag{4}
$$

where the weight $w_p = (1 - \frac{H(\bar{\mathbf{M}}_p)}{\log K}) \cdot (1 - \frac{\text{JSD}(\{\mathbf{M}_p^n\})}{\log K})$ combines certainty (mean entropy) and consistency (JSD).

**Intra-class Shape Score:**

Using the **isoperimetric ratio** $\phi(e_k) = 4\pi \cdot \text{Area}(\mathbf{M}_k) / (\text{Perimeter}(\mathbf{M}_k))^2$ as the shape descriptor and Z-score-based softmax-weighted aggregation:

$$
S_{\text{intra}}(\mathcal{G}) = \sum_{k=1}^{K-1} S_{\phi,k} \cdot \frac{\exp(-S_{\phi,k}/\tau)}{\sum_{j=1}^{K-1} \exp(-S_{\phi,j}/\tau)} \tag{5}
$$

**Inter-class Layout Score:**

The layout hyperedge $e_l$ evaluates spatial arrangement plausibility via relative direction cosines $\psi_{ij}$ between class centroids:

$$
S_{\text{inter}}(\mathcal{G}) = \sum_{i,j=1}^{K-1} S_{\psi,ij} \cdot \frac{\exp(-S_{\psi,ij}/\tau)}{\sum_{u,v=1}^{K-1} \exp(-S_{\psi,uv}/\tau)} \tag{6}
$$

**Final Plausibility Score:**

$$
S_{\text{final}} = S_{\text{vertex}}(\mathcal{G}) \cdot (\alpha \, S_{\text{intra}}(\mathcal{G}) + (1-\alpha) \, S_{\text{inter}}(\mathcal{G})) \tag{7}
$$

Samples are selected for self-training only if $S_{\text{final}}$ exceeds a dynamic threshold determined by the top-$\rho$ percentile within the current epoch.

### 3.4. Structural Anomaly Pruning (SAP)

Predictions passing HPE may still contain class-level artifacts (hallucinations). SAP quantifies the **Structural Instability Score** of each class via the coefficient of variation across $N_{\text{ens}}$ teacher predictions:

$$
\Upsilon(k) = \frac{\text{std}(\mathbf{c}_k)}{\bar{c}_k + \epsilon} \tag{8}
$$

where $\mathbf{c}_k = \langle C(\mathbf{M}^1, k), \dots, C(\mathbf{M}^{N_{\text{ens}}}, k) \rangle$ is the vector of class $k$ pixel counts across ensemble predictions. Stable anatomical structures show low variance; model hallucinations show high volatility.

Classes with instability scores exceeding a batch-dynamic threshold $\theta_A$ (the $q$-th percentile of foreground class instability scores) are deemed anomalous and pruned:

$$
\mathcal{K}_{\text{anom}} = \{ k \in \{1, \dots, K-1\} \mid \Upsilon(k) > \theta_A \} \tag{9}
$$

### 3.5. Overall Learning Objective

The total objective is a weighted sum of supervised and unsupervised losses:

- **Source domain supervised loss** over original and HFM-modulated features $\mathcal{F}_s = \{\mathbf{F}_s, \mathbf{F}_{s \to t}, \mathbf{F}_{s,\text{cross}}\}$:

$$
\mathcal{L}_{\text{sup}} = \frac{1}{|\mathcal{F}_s|} \sum_{\mathbf{F}' \in \mathcal{F}_s} \mathcal{L}_{\text{seg}}(\mathcal{D}(\mathbf{F}'), \mathbf{L}_s) \tag{11}
$$

- **Target domain unsupervised loss** guided by high-fidelity pseudo-labels $\mathbf{M}'$ from HPE + SAP validation:

$$
\mathcal{L}_{\text{unsup}} = \frac{1}{|\mathcal{B}_{\text{sel}}|} \sum_{i \in \mathcal{B}_{\text{sel}}} \mathcal{L}_{\text{seg}}(\mathcal{D}(\mathbf{F}_t^i), (\mathbf{M}')^i, w_p^i) \tag{12}
$$

- **Total loss:** $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{sup}} + \gamma_{\text{unsup}} \mathcal{L}_{\text{unsup}}$

The teacher model is an EMA (Exponential Moving Average) of the student decoder, and $\mathcal{L}_{\text{seg}}$ is a combination of Dice + Focal loss.

---

## 4. Experimental Results

### 4.1. Experimental Setup

- **Datasets:** MMWHS (cardiac, 20 CT / 20 MRI each), MICCAI 2015 + ISBI 2019 CHAOS (abdominal, 30 CT / 20 MRI)
- **Implementation:** DINOv3 ViT-S/16 encoder (frozen) + UNet decoder, 200 epochs, RTX 4090, batch size 64
- **Metrics:** Dice Score (DSC), Average Surface Distance (ASD)

### 4.2. Cardiac Dataset Results

**Table 1. Cardiac MRI → CT — DSC(%) / ASD(mm)**

| Method | AA | LAC | LVC | MYO | **Avg DSC** | **Avg ASD** |
|--------|:--:|:---:|:---:|:---:|:-----------:|:-----------:|
| W/o adaptation | 68.29 | 61.41 | 18.24 | 35.71 | 45.91 | 38.02 |
| CycleGAN | 63.29 | 72.50 | 45.98 | 50.73 | 58.13 | 21.28 |
| SIFA | 82.72 | 75.21 | 75.41 | 65.17 | 74.63 | 10.22 |
| GenericSSL | 82.02 | 77.18 | 84.28 | 67.65 | 77.78 | 6.64 |
| IPLC | 87.63 | 78.21 | 86.11 | 71.68 | 80.91 | 6.37 |
| IPLC+ | 63.69 | 86.15 | 86.71 | 89.17 | 81.43 | 3.91 |
| DDFP | 72.03 | 85.30 | 89.64 | 90.86 | 84.46 | 3.79 |
| **SHAPE** | **79.58** | **92.18** | **94.53** | **94.03** | **90.08** | **2.62** |
| Supervised (upper bound) | 91.28 | 92.49 | 95.56 | 94.16 | 93.37 | 2.11 |

> SHAPE achieves a **+5.62% DSC** improvement over the runner-up, narrowing the gap to the supervised upper bound (93.37%) to just **3.29%**.

**Table 2. Cardiac CT → MRI — DSC(%) / ASD(mm)**

| Method | AA | LAC | LVC | MYO | **Avg DSC** | **Avg ASD** |
|--------|:--:|:---:|:---:|:---:|:-----------:|:-----------:|
| W/o adaptation | 36.56 | 46.49 | 49.23 | 15.35 | 36.91 | 23.06 |
| SIFA | 55.47 | 66.43 | 72.52 | 60.69 | 63.78 | 13.84 |
| IPLC+ | 65.09 | 77.56 | 88.95 | 74.33 | 76.48 | 5.37 |
| DDFP | 66.26 | 76.04 | 88.55 | 70.64 | 75.37 | 8.58 |
| **SHAPE** | **70.25** | **79.11** | **86.08** | **78.59** | **78.51** | **4.70** |

### 4.3. Abdominal Dataset Results

**Table 3. Abdominal MRI → CT — DSC(%)**

| Method | LIV | RK | LK | SPL | **Avg DSC** | **Avg ASD** |
|--------|:---:|:--:|:--:|:---:|:-----------:|:-----------:|
| GenericSSL | 84.98 | 84.08 | 84.37 | 84.92 | 84.59 | 7.56 |
| IPLC+ | 85.29 | 88.63 | 85.12 | 83.79 | 85.71 | 5.15 |
| DDFP | 86.40 | 87.66 | 88.05 | 78.55 | 85.17 | 3.14 |
| **SHAPE** | **88.26** | **86.99** | **83.37** | **91.28** | **87.48** | **2.94** |

**Table 4. Abdominal CT → MRI — DSC(%)**

| Method | LIV | RK | LK | SPL | **Avg DSC** | **Avg ASD** |
|--------|:---:|:--:|:--:|:---:|:-----------:|:-----------:|
| GenericSSL | 82.43 | 86.88 | 80.52 | 89.79 | 84.91 | 6.71 |
| DDFP | 82.49 | 86.43 | 87.15 | 89.01 | 86.27 | 5.17 |
| **SHAPE** | **86.83** | **88.86** | **88.30** | **83.58** | **86.89** | **2.81** |

### 4.4. Qualitative Results and Feature Analysis

![Qualitative comparison of SHAPE and representative methods. Yellow arrows indicate areas where SHAPE outperforms competing methods.](/images/shape/qualitative_results.jpeg)

![t-SNE visualization of source-target feature distributions: (a) Before adaptation, (b) After AdaIN, (c) After HFM](/images/shape/tsne_features.jpeg)

Key observations from t-SNE analysis:
- **(a) Before adaptation:** Source (circles) and target (squares) features are completely separated
- **(b) After AdaIN:** Pulls target features toward source, but induces **distributional contraction** that destroys inter-class separability
- **(c) After HFM:** Target class centroids are precisely aligned with source counterparts while **preserving intra-class distributional structure**

### 4.5. Ablation Study

| Configuration | HFM | HPE | SAP | MRI→CT DSC | CT→MRI DSC |
|--------------|:---:|:---:|:---:|:----------:|:----------:|
| (a) Baseline (DINOv3) | | | | 82.02 | 71.58 |
| (b) + HFM | ✓ | | | 85.67 | 75.46 |
| (c) + HPE | | ✓ | | 82.71 | 72.09 |
| (d) + HFM + HPE | ✓ | ✓ | | 85.80 | 75.81 |
| (e) + HFM + SAP | ✓ | | ✓ | 86.03 | 76.23 |
| **(f) SHAPE (Full)** | **✓** | **✓** | **✓** | **90.08** | **78.51** |

> **HFM** alone provides the largest individual gain (+3.65%), while the full integration of all three modules achieves a dramatic synergistic improvement of **+8.06%**.

### 4.6. Hyperparameter Sensitivity Analysis

![Hyperparameter sensitivity analysis](/images/shape/hyperparameter_sensitivity.jpeg)

- The plausibility fusion weight $\alpha$ and anomaly threshold $\theta_A$ exhibit clear optimal regions
- Performance consistently improves with stricter purity threshold $\tau_p$ and larger batch sizes
- HPE benefits from larger batches for stable statistics, but SAP's batch-dynamic threshold ensures effectiveness at smaller sizes

---

## 5. Summary of Key Contributions

1. **Class-aware Feature Modulation (HFM):** Distinguishes semantic cores from structural boundaries for optimized per-category modulation, resolving distributional fidelity issues of monolithic alignment
2. **Hypergraph-based Anatomical Plausibility (HPE):** Goes beyond standard graph pairwise relations to model higher-order interactions of multiple anatomical structures via hypergraphs, validating global structural coherence
3. **Structural Anomaly Pruning (SAP):** Identifies and removes class-level hallucinations through cross-view stability analysis
4. **Paradigm Shift:** Realizes a UDA paradigm shift from pixel-level accuracy to **global anatomical plausibility**
5. **SOTA Performance:** Achieves meaningful improvements across all 4 cross-modality benchmark scenarios on cardiac and abdominal datasets

---

> **References** can be found in the original paper's References section.
