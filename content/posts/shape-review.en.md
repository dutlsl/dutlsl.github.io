---
title: "[CVPR 2026] SHAPE: Structure-aware Hierarchical UDA with Plausibility Evaluation for Medical Image Segmentation — Paper Review"
date: 2026-06-30T17:50:00+09:00
draft: false
math: true
tags: ["Paper Review", "Domain Adaptation", "Medical Image Segmentation", "CVPR 2026"]
categories: ["Paper Review"]
summary: "Review of the SHAPE framework presented at CVPR 2026. A new UDA paradigm for medical imaging via hypergraph-based anatomical plausibility evaluation."
cover:
  image: "/images/shape/pipeline.jpeg"
  alt: "SHAPE Framework Pipeline"
---

> <b>Paper Info</b>
> - <b>Title:</b> SHAPE: Structure-aware Hierarchical Unsupervised Domain Adaptation with Plausibility Evaluation for Medical Image Segmentation
> - <b>Authors:</b> Linkuan Zhou, Yinghao Xia, Yufei Shen, Xiangyu Li, Wenjie Du, Cong Cong, Leyi Wei, Ran Su, Qiangguo Jin
> - <b>Affiliation:</b> Northwestern Polytechnical University, HIT, USTC, Macquarie University, Macao Polytechnic University, Tianjin University
> - <b>Conference:</b> CVPR 2026
> - <b>Code:</b> [github.com/BioMedIA-repo/SHAPE](https://github.com/BioMedIA-repo/SHAPE)

---

## 1. One-line Summary

In cross-modality unsupervised domain adaptation (UDA) for medical image segmentation, the authors propose <b>SHAPE</b>, a framework integrating <b>class-aware Hierarchical Feature Modulation (HFM)</b> built on DINOv3, <b>Hypergraph Plausibility Estimation (HPE)</b> for global anatomical validity, and <b>Structural Anomaly Pruning (SAP)</b>, achieving SOTA Dice scores of <b>90.08%</b> (MRI→CT) and <b>78.51%</b> (CT→MRI) on cardiac data, and <b>87.48%</b> / <b>86.89%</b> on abdominal data.

---

## 2. Background and Motivation

### 2.1. Problem Definition

Medical image segmentation models assume identical training and test distributions, but real clinical environments exhibit severe performance degradation due to variations in <b>imaging equipment, modalities (CT/MRI), and acquisition parameters</b>. UDA aims to transfer knowledge from a labeled source domain to an unlabeled target domain, avoiding costly re-annotation.

### 2.2. Limitations of Existing Methods

Existing UDA approaches fall into two categories, each with fundamental limitations:

| Category | Representative Methods | Key Limitation |
|----------|----------------------|----------------|
| <b>Alignment-based</b> | CycleGAN, SIFA, SASAN, DDFP | Content-agnostic global transformation → destroys fine-grained cross-class mappings |
| <b>Pseudo-label-based</b> | MAPSeg, GenericSSL, IPLC, UPL-SFDA | Relies only on pixel-level confidence (entropy, consistency) → admits anatomically impossible pseudo-labels |

> The core issues reduce to two:
> 1. <b>Semantically unaware feature alignment:</b> Monolithic style transfer averages style characteristics across distinct anatomical structures, losing class-specific information
> 2. <b>Disregard for global anatomical constraints:</b> Pixel-level validation cannot prevent formation of pseudo-labels with anatomically impossible shapes or spatial arrangements

---

## 3. Proposed Method: SHAPE Framework

### 3.1. Overall Architecture

![SHAPE Framework Pipeline (Paper Figure 1)](/images/shape/pipeline.jpeg)

SHAPE consists of a synergistic pipeline with three core modules:

1. <b>Hierarchical Feature Modulation (HFM):</b> Class-aware, spatially-differentiated feature alignment
2. <b>Hypergraph Plausibility Estimation (HPE):</b> Global anatomical plausibility assessment via hypergraphs
3. <b>Structural Anomaly Pruning (SAP):</b> Cross-view stability-based structural artifact removal

### 3.2. Hierarchical Feature Modulation (HFM)

From a frozen <b>DINOv3 ViT</b> encoder, we extract a dense feature map $\mathbf{F} = \Phi(\mathbf{I}) \in \mathbb{R}^{C \times H \times W}$ and apply a <b>dual-granularity</b> approach.

<b>Global Level: AdaIN Style Alignment</b>

Compute the stylized map $\mathbf{F}\_{s \to t}$ aligning textural properties between source $\mathbf{F}\_s$ and target $\mathbf{F}\_t$:

$$\mathbf{F}_{s \to t} = \sigma(\mathbf{F}_t) \left( \frac{\mathbf{F}_s - \mu(\mathbf{F}_s)}{\sigma(\mathbf{F}_s) + \epsilon} \right) + \mu(\mathbf{F}_t) \tag{1}$$

where $\mu(\cdot)$ and $\sigma(\cdot)$ denote channel-wise mean and standard deviation.

<b>Local Level: Structure-aware Token Mixing</b>

After upsampling feature maps, each token's <b>purity score</b> is computed to classify it as a semantic core (pure) or structural boundary (impure):

$$\mathcal{P}(\mathbf{m}^i) = \frac{\max_{k \in \{0..K-1\}} \sum_{v \in \mathbf{m}^i} \mathbb{I}(v=k)}{|\mathbf{m}^i|} \tag{2}$$

A differentiated modulation strategy is then applied:

- <b>Pure tokens (semantic cores):</b> Direct linear interpolation with same-class target tokens: $(1-\lambda)\mathbf{f}\_s^i + \lambda\mathbf{f}\_t^j$
- <b>Impure tokens (structural boundaries):</b> Normalization with interpolated statistics from source and target boundary tokens

> [!IMPORTANT]
> <b>Key insight of HFM:</b> Conventional AdaIN applies uniform transformation to all features, destroying inter-class separability. HFM distinguishes between anatomical cores and boundaries, applying optimized modulation to each, thereby bridging the domain gap while preserving fine-grained discriminative structure.

### 3.3. Hypergraph Plausibility Estimation (HPE)

Each predicted segmentation map $\mathbf{M} \in \{0,...,K-1\}^{H \times W}$ is modeled as a <b>multi-level structural hypergraph</b> $\mathcal{G} = (\mathcal{V}, \mathcal{E})$.

<b>Vertex Reliability Score:</b>

$$S_{\text{vertex}}(\mathcal{G}) = \frac{1}{|\mathcal{V}|} \sum_{p \in \mathcal{V}} w_p \tag{4}$$

where the weight $w\_p = (1 - \frac{H(\bar{\mathbf{M}}\_p)}{\log K}) \cdot (1 - \frac{\text{JSD}(\{\mathbf{M}\_p^n\})}{\log K})$ combines certainty (mean entropy) and consistency (JSD).

<b>Intra-class Shape Score:</b>

Using the <b>isoperimetric ratio</b> $\phi(e\_k) = 4\pi \cdot \text{Area}(\mathbf{M}\_k) / (\text{Perimeter}(\mathbf{M}\_k))^2$ as the shape descriptor and Z-score-based softmax-weighted aggregation:

$$S_{\text{intra}}(\mathcal{G}) = \sum_{k=1}^{K-1} S_{\phi,k} \cdot \frac{\exp(-S_{\phi,k}/\tau)}{\sum_{j=1}^{K-1} \exp(-S_{\phi,j}/\tau)} \tag{5}$$

<b>Inter-class Layout Score:</b>

The layout hyperedge $e\_l$ evaluates spatial arrangement plausibility via relative direction cosines $\psi\_{ij}$ between class centroids:

$$S_{\text{inter}}(\mathcal{G}) = \sum_{i,j=1}^{K-1} S_{\psi,ij} \cdot \frac{\exp(-S_{\psi,ij}/\tau)}{\sum_{u,v=1}^{K-1} \exp(-S_{\psi,uv}/\tau)} \tag{6}$$

<b>Final Plausibility Score:</b>

$$S_{\text{final}} = S_{\text{vertex}}(\mathcal{G}) \cdot (\alpha \, S_{\text{intra}}(\mathcal{G}) + (1-\alpha) \, S_{\text{inter}}(\mathcal{G})) \tag{7}$$

Samples are selected for self-training only if $S\_{\text{final}}$ exceeds a dynamic threshold determined by the top-$\rho$ percentile within the current epoch.

### 3.4. Structural Anomaly Pruning (SAP)

Predictions passing HPE may still contain class-level artifacts (hallucinations). SAP quantifies the <b>Structural Instability Score</b> of each class via the coefficient of variation across $N\_{\text{ens}}$ teacher predictions:

$$\Upsilon(k) = \frac{\text{std}(\mathbf{c}_k)}{\bar{c}_k + \epsilon} \tag{8}$$

where $\mathbf{c}\_k = \langle C(\mathbf{M}^1, k), \dots, C(\mathbf{M}^{N\_{\text{ens}}}, k) \rangle$ is the vector of class $k$ pixel counts across ensemble predictions. Stable anatomical structures show low variance; model hallucinations show high volatility.

Classes with instability scores exceeding a batch-dynamic threshold $\theta\_A$ (the $q$-th percentile of foreground class instability scores) are deemed anomalous and pruned:

$$\mathcal{K}_{\text{anom}} = \{ k \in \{1, \dots, K-1\} \mid \Upsilon(k) > \theta_A \} \tag{9}$$

### 3.5. Overall Learning Objective

The total objective is a weighted sum of supervised and unsupervised losses:

- <b>Source domain supervised loss</b> over original and HFM-modulated features $\mathcal{F}\_s = \{\mathbf{F}\_s, \mathbf{F}\_{s \to t}, \mathbf{F}\_{s,\text{cross}}\}$:

$$\mathcal{L}_{\text{sup}} = \frac{1}{|\mathcal{F}_s|} \sum_{\mathbf{F}' \in \mathcal{F}_s} \mathcal{L}_{\text{seg}}(\mathcal{D}(\mathbf{F}'), \mathbf{L}_s) \tag{11}$$

- <b>Target domain unsupervised loss</b> guided by high-fidelity pseudo-labels $\mathbf{M}'$ from HPE + SAP validation:

$$\mathcal{L}_{\text{unsup}} = \frac{1}{|\mathcal{B}_{\text{sel}}|} \sum_{i \in \mathcal{B}_{\text{sel}}} \mathcal{L}_{\text{seg}}(\mathcal{D}(\mathbf{F}_t^i), (\mathbf{M}')^i, w_p^i) \tag{12}$$

- <b>Total loss:</b> $\mathcal{L}\_{\text{total}} = \mathcal{L}\_{\text{sup}} + \gamma\_{\text{unsup}} \mathcal{L}\_{\text{unsup}}$

The teacher model is an EMA (Exponential Moving Average) of the student decoder, and $\mathcal{L}\_{\text{seg}}$ is a combination of Dice + Focal loss.

---

## 4. Experimental Results

### 4.1. Experimental Setup

- <b>Datasets:</b> MMWHS (cardiac, 20 CT / 20 MRI each), MICCAI 2015 + ISBI 2019 CHAOS (abdominal, 30 CT / 20 MRI)
- <b>Implementation:</b> DINOv3 ViT-S/16 encoder (frozen) + UNet decoder, 200 epochs, RTX 4090, batch size 64
- <b>Metrics:</b> Dice Score (DSC), Average Surface Distance (ASD)

### 4.2. Cardiac Dataset Results

<b>Table 1. Cardiac MRI → CT — DSC(%) / ASD(mm)</b>

| Method | AA | LAC | LVC | MYO | <b>Avg DSC</b> | <b>Avg ASD</b> |
|--------|:--:|:---:|:---:|:---:|:-----------:|:-----------:|
| W/o adaptation | 68.29 | 61.41 | 18.24 | 35.71 | 45.91 | 38.02 |
| CycleGAN | 63.29 | 72.50 | 45.98 | 50.73 | 58.13 | 21.28 |
| SIFA | 82.72 | 75.21 | 75.41 | 65.17 | 74.63 | 10.22 |
| GenericSSL | 82.02 | 77.18 | 84.28 | 67.65 | 77.78 | 6.64 |
| IPLC | 87.63 | 78.21 | 86.11 | 71.68 | 80.91 | 6.37 |
| IPLC+ | 63.69 | 86.15 | 86.71 | 89.17 | 81.43 | 3.91 |
| DDFP | 72.03 | 85.30 | 89.64 | 90.86 | 84.46 | 3.79 |
| <b>SHAPE</b> | <b>79.58</b> | <b>92.18</b> | <b>94.53</b> | <b>94.03</b> | <b>90.08</b> | <b>2.62</b> |
| Supervised (upper bound) | 91.28 | 92.49 | 95.56 | 94.16 | 93.37 | 2.11 |

> SHAPE achieves a <b>+5.62% DSC</b> improvement over the runner-up, narrowing the gap to the supervised upper bound (93.37%) to just <b>3.29%</b>.

<b>Table 2. Cardiac CT → MRI — DSC(%) / ASD(mm)</b>

| Method | AA | LAC | LVC | MYO | <b>Avg DSC</b> | <b>Avg ASD</b> |
|--------|:--:|:---:|:---:|:---:|:-----------:|:-----------:|
| W/o adaptation | 36.56 | 46.49 | 49.23 | 15.35 | 36.91 | 23.06 |
| SIFA | 55.47 | 66.43 | 72.52 | 60.69 | 63.78 | 13.84 |
| IPLC+ | 65.09 | 77.56 | 88.95 | 74.33 | 76.48 | 5.37 |
| DDFP | 66.26 | 76.04 | 88.55 | 70.64 | 75.37 | 8.58 |
| <b>SHAPE</b> | <b>70.25</b> | <b>79.11</b> | <b>86.08</b> | <b>78.59</b> | <b>78.51</b> | <b>4.70</b> |

### 4.3. Abdominal Dataset Results

<b>Table 3. Abdominal MRI → CT — DSC(%)</b>

| Method | LIV | RK | LK | SPL | <b>Avg DSC</b> | <b>Avg ASD</b> |
|--------|:---:|:--:|:--:|:---:|:-----------:|:-----------:|
| GenericSSL | 84.98 | 84.08 | 84.37 | 84.92 | 84.59 | 7.56 |
| IPLC+ | 85.29 | 88.63 | 85.12 | 83.79 | 85.71 | 5.15 |
| DDFP | 86.40 | 87.66 | 88.05 | 78.55 | 85.17 | 3.14 |
| <b>SHAPE</b> | <b>88.26</b> | <b>86.99</b> | <b>83.37</b> | <b>91.28</b> | <b>87.48</b> | <b>2.94</b> |

<b>Table 4. Abdominal CT → MRI — DSC(%)</b>

| Method | LIV | RK | LK | SPL | <b>Avg DSC</b> | <b>Avg ASD</b> |
|--------|:---:|:--:|:--:|:---:|:-----------:|:-----------:|
| GenericSSL | 82.43 | 86.88 | 80.52 | 89.79 | 84.91 | 6.71 |
| DDFP | 82.49 | 86.43 | 87.15 | 89.01 | 86.27 | 5.17 |
| <b>SHAPE</b> | <b>86.83</b> | <b>88.86</b> | <b>88.30</b> | <b>83.58</b> | <b>86.89</b> | <b>2.81</b> |

### 4.4. Qualitative Results and Feature Analysis

![Qualitative comparison of SHAPE and representative methods. Yellow arrows indicate areas where SHAPE outperforms competing methods.](/images/shape/qualitative_results.jpeg)

![t-SNE visualization of source-target feature distributions: (a) Before adaptation, (b) After AdaIN, (c) After HFM](/images/shape/tsne_features.jpeg)

Key observations from t-SNE analysis:
- <b>(a) Before adaptation:</b> Source (circles) and target (squares) features are completely separated
- <b>(b) After AdaIN:</b> Pulls target features toward source, but induces <b>distributional contraction</b> that destroys inter-class separability
- <b>(c) After HFM:</b> Target class centroids are precisely aligned with source counterparts while <b>preserving intra-class distributional structure</b>

### 4.5. Ablation Study

| Configuration | HFM | HPE | SAP | MRI→CT DSC | CT→MRI DSC |
|--------------|:---:|:---:|:---:|:----------:|:----------:|
| (a) Baseline (DINOv3) | | | | 82.02 | 71.58 |
| (b) + HFM | ✓ | | | 85.67 | 75.46 |
| (c) + HPE | | ✓ | | 82.71 | 72.09 |
| (d) + HFM + HPE | ✓ | ✓ | | 85.80 | 75.81 |
| (e) + HFM + SAP | ✓ | | ✓ | 86.03 | 76.23 |
| <b>(f) SHAPE (Full)</b> | <b>✓</b> | <b>✓</b> | <b>✓</b> | <b>90.08</b> | <b>78.51</b> |

> <b>HFM</b> alone provides the largest individual gain (+3.65%), while the full integration of all three modules achieves a dramatic synergistic improvement of <b>+8.06%</b>.

### 4.6. Hyperparameter Sensitivity Analysis

![Hyperparameter sensitivity analysis](/images/shape/hyperparameter_sensitivity.jpeg)

- The plausibility fusion weight $\alpha$ and anomaly threshold $\theta\_A$ exhibit clear optimal regions
- Performance consistently improves with stricter purity threshold $\tau\_p$ and larger batch sizes
- HPE benefits from larger batches for stable statistics, but SAP's batch-dynamic threshold ensures effectiveness at smaller sizes

---

## 5. Summary of Key Contributions

1. <b>Class-aware Feature Modulation (HFM):</b> Distinguishes semantic cores from structural boundaries for optimized per-category modulation, resolving distributional fidelity issues of monolithic alignment
2. <b>Hypergraph-based Anatomical Plausibility (HPE):</b> Goes beyond standard graph pairwise relations to model higher-order interactions of multiple anatomical structures via hypergraphs, validating global structural coherence
3. <b>Structural Anomaly Pruning (SAP):</b> Identifies and removes class-level hallucinations through cross-view stability analysis
4. <b>Paradigm Shift:</b> Realizes a UDA paradigm shift from pixel-level accuracy to <b>global anatomical plausibility</b>
5. <b>SOTA Performance:</b> Achieves meaningful improvements across all 4 cross-modality benchmark scenarios on cardiac and abdominal datasets

