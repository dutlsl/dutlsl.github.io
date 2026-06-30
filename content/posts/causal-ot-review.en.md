---
title: "[CVPR 2026] Causal-OT: Uncertainty-aware UDA for Videos and Time-Series Paper Review"
date: 2026-06-30T17:30:00+09:00
draft: false
math: true
tags: ["Paper Review", "Domain Adaptation", "Optimal Transport", "CVPR 2026"]
categories: ["Paper Review"]
summary: "Review of the Causal-OT framework presented at CVPR 2026."
cover:
  image: "/images/causal_ot/causal_ot_architecture.jpeg"
  alt: "Causal-OT Architecture"
---

> **Paper Info**
> - **Title:** Towards Uncertainty-aware Unsupervised Domain Adaptation for Videos and Time-Series with Causal Optimal Transport
> - **Authors:** Khushboo Mishra, Varun Trivedi, Tanima Dutta
> - **Affiliation:** Indian Institute of Technology (BHU) Varanasi, India
> - **Conference:** CVPR 2026
> - **Code:** [github.com/mynameanonymous/CausalOT](https://github.com/mynameanonymous/CausalOT)

---

## 1. One-line Summary

In Unsupervised Domain Adaptation (UDA) for time-series and video data, the authors propose a unified framework, **Causal-OT**, which integrates **Granger causal graph regularization** into **Optimal Transport (OT)** alignment and combines it with **entropy-based uncertainty-aware pseudo-labeling**, achieving an average improvement of **4.5% in accuracy and 3.8% in F1** across 6 time-series benchmarks, and **2.5% in accuracy** across 4 video benchmarks.

---

## 2. Background and Motivation

### 2.1. Problem Definition

Time-series data (accelerometer, gyroscope, EMG, etc.) exhibit significant cross-domain variations depending on the user, device, and environment. When a model trained on a labeled source domain is applied to an unlabeled target domain, it faces the dual challenge of **statistical distribution discrepancies** and **temporal dependency shifts**.

### 2.2. Limitations of Existing Methods

Existing UDA approaches fail to address the following three key dimensions **simultaneously**:

| Model | Distribution Mismatch | Causality Preservation | Uncertainty Handling |
|------|:---:|:---:|:---:|
| TransPL [15] | ✓ | ✗ | ✗ |
| CauDiTS [36] | ✗ | ✓ | ✗ |
| RAINCOAT [13] | ✓ | ✗ | ✗ |
| **Causal-OT (Ours)** | **✓** | **✓ (Granger)** | **✓ (entropy-aware)** |

> **Reconstruction of Table 1.** Existing methods only handle portions of distribution alignment, causal structure preservation, and prediction uncertainty. Causal-OT integrates all three.

In particular, TransPL relies on a wide range of pseudo-labels, including those with high-entropy predictions, generating noisy supervisory signals. The figure below demonstrates that the prediction uncertainty of the target domain is significantly higher than that of the source:

![Source vs Target Domain Uncertainty Distribution Comparison (SSC, MFD datasets)](/images/causal_ot/uncertainty_distribution.jpeg)

---

## 3. Proposed Method: Causal-OT Framework

### 3.1. Overall Architecture

![Causal-OT Framework Architecture (Paper Figure 3)](/images/causal_ot/causal_ot_architecture.jpeg)

The training pipeline of Causal-OT consists of the following 5 steps:

1. **Causal Graph Construction:** Extract Granger causal graphs G<sub>s</sub>, G<sub>t</sub> from raw time-series of the source and target domains. In the figure, black arrows represent causal relationships between variables, while red arrows indicate non-causal relationships.

2. **Feature Extraction:** A shared feature extractor f<sub>θ</sub> maps the source and target data to the latent feature spaces Z<sub>s</sub>, Z<sub>t</sub>.

3. **Causal Alignment:** Calculate the Optimal Transport (OT) loss ℒ<sub>OT</sub> based on a cost matrix C that combines feature distance and causal graph distance D<sub>G</sub>.

4. **Classification and Pseudo-Labeling:** The classifier h<sub>φ</sub> is trained on the source data (ℒ<sub>src</sub>), and generates high-confidence pseudo-labels on the target domain (ℒ<sub>PL</sub>).

5. **Optimization:** Update model parameters θ, φ using the weighted sum of the three losses.

### 3.2. Granger Causal Graph Extraction

Directional causal relationships between variables are modeled from multivariate time-series X ∈ ℝ<sup>N × T × D</sup>. Specifically:

- **Stationarity Test:** Check signal stationarity with Augmented Dickey-Fuller test.
- **VAR Order Selection:** Determine optimal lag order p using Bayesian Information Criterion (BIC).
- **Significance Filtering:** Retain only edges with p-value < 0.05.
- **Row Normalization:** Normalize the adjacency matrix W such that row sums equal 1.
- **Spectral Embedding:** Derive a k-dimensional embedding Φ(G) ∈ ℝ<sup>d × k</sup> from the Laplacian L = D - W.

> [!IMPORTANT]
> **Hybrid Graph Update Strategy:** The initial causal graph is extracted from raw data X, but is periodically re-estimated and blended from the latent features Z during training.
>
> <b>A<sup>(t)</sup> = α A<sub>X</sub> + (1 - α) A<sub>Z</sub><sup>(t)</sup></b>  (where α ∈ [0.6, 0.9])
>
> This maintains a stable raw signal structure while reflecting causal structural changes as learning progresses.

### 3.3. Causality-Dependent Optimal Transport Cost

The cost between source-target sample pairs is defined by combining feature similarity and causal descriptor consistency:

<b>C<sub>ij</sub> = ‖ f<sub>s</sub>(x<sub>i</sub><sup>s</sup>) - f<sub>t</sub>(x<sub>j</sub><sup>t</sup>) ‖<sub>2</sub><sup>2</sup> + λ ‖ φ<sub>i</sub><sup>s</sup> - φ<sub>j</sub><sup>t</sup> ‖<sub>2</sub><sup>2</sup></b>

Here, φ<sub>i</sub><sup>s</sup> and φ<sub>j</sub><sup>t</sup> are the causal embeddings of the source and target samples, respectively. For this cost matrix, the entropy-regularized OT problem is solved via Sinkhorn iterations to obtain the optimal transport plan γ*:

<b>γ* = argmin<sub>γ ∈ Π(μ<sub>s</sub>, μ<sub>t</sub>)</sub> [ ⟨γ, C⟩ + ε H(γ) ]</b>

The core of this design is to **embed causal graph constraints directly into the cost matrix**, ensuring that the alignment is not only geometrically meaningful but also structurally consistent with temporal dependencies.

### 3.4. Uncertainty-aware Pseudo-Labeling

Soft class predictions from the classifier h<sub>φ</sub> are obtained from the encoded target features Z<sub>t</sub><sup>j</sup>, and uncertainty is estimated using the **entropy** of the predicted distribution:

<b>U<sub>t</sub><sup>j</sup> = - Σ<sub>k=1</sub><sup>K</sup> ŷ<sub>t,j</sub><sup>(k)</sup> log(ŷ<sub>t,j</sub><sup>(k)</sup>)</b>

Only **low-entropy (high-confidence)** samples below an entropy threshold ρ are used as pseudo-labels to prevent noise propagation.

### 3.5. Total Loss Function

The total loss function is a weighted sum of three terms:

<b>ℒ<sub>total</sub> = ℒ<sub>src</sub> + α ℒ<sub>OT</sub> + β ℒ<sub>PL</sub></b>

- **Source Domain Classification Loss (ℒ<sub>src</sub>)**: Cross-Entropy
  <b>ℒ<sub>src</sub> = (1 / N<sub>s</sub>) Σ<sub>i</sub> CE(h<sub>φ</sub>(Z<sub>s</sub><sup>i</sup>), y<sub>s</sub><sup>i</sup>)</b>

- **Causal Structure Preserving OT Alignment Loss (ℒ<sub>OT</sub>)**:
  <b>ℒ<sub>OT</sub> = ⟨γ*, C⟩</b>

- **Uncertainty-filtered Pseudo-Label Loss (ℒ<sub>PL</sub>)**:
  <b>ℒ<sub>PL</sub> = (1 / |I|) Σ<sub>j ∈ I</sub> CE(h<sub>φ</sub>(Z<sub>t</sub><sup>j</sup>), ŷ<sub>t</sub><sup>j</sup>)</b>

### 3.6. Theoretical Analysis

**Proposition 1 (Causal-OT Target Risk Bound):** For a Lipschitz continuous and bounded surrogate loss ℓ:

<b>ℛ<sub>t</sub>(h) ≤ ℛ<sub>s</sub>(h) + L · 𝔼<sub>(x<sub>s</sub>, x<sub>t</sub>) ∼ γ*</sub>[ ‖ f<sub>s</sub>(x<sub>s</sub>) - f<sub>t</sub>(x<sub>t</sub>) ‖ ] + λ · 𝔼<sub>(i,j) ∼ γ*</sub>[ ‖ φ<sub>i</sub><sup>s</sup> - φ<sub>j</sub><sup>t</sup> ‖ ] + 𝒟<sub>ℋ</sub>(P<sub>s</sub>, P<sub>t</sub>)</b>

This inequality formalizes that minimizing the Causal-OT objective simultaneously reduces **feature-level distance, causal structure mismatch, and hypothesis mismatch**, enhancing transferability under domain shift.

### 3.7. Video Data Processing

Video data is integrated into the same pipeline by converting it to time-series during the preprocessing stage, without separate model modifications:

1. Uniformly divide each video clip into T time segments.
2. Extract segment-level embeddings using a ResNet-101 or 3D backbone at each segment.
3. Obtain sequential representations of shape X ∈ ℝ<sup>d × T</sup> (default d=2048) by reducing dimensions with PCA.
4. Apply the same procedures afterward, including causal graph extraction, OT alignment, and pseudo-labeling.

---

## 4. Experimental Results

### 4.1. Time-Series Benchmarks

Evaluated on 6 datasets (UCIHAR, WISDM, HHAR, SSC, MFD, Boiler). Results for WISDM and UCIHAR are below:

**Table 2. WISDM Accuracy (%) — By Source→Target Domain Pair**

| Algorithm | 7→18 | 20→30 | 35→31 | 17→23 | 6→19 | 2→11 | 33→12 | 5→26 | 28→4 | 23→32 | **Average** |
|-----------|------|-------|-------|-------|------|------|-------|------|------|-------|---------|
| No Adapt  | 80.2 | 64.1  | 66.3  | 48.3  | 62.1 | 31.6 | 60.9  | 73.2 | 83.3 | 27.5  | 59.8    |
| TransPL   | 84.9 | 66.0  | 63.9  | 66.7  | 48.5 | 61.8 | 88.5  | 74.4 | 54.5 | 30.4  | 64.0    |
| CoDATS    | 74.5 | 80.6  | 54.2  | 70.0  | 54.5 | 47.4 | 82.8  | 75.6 | 78.8 | 18.8  | 63.7    |
| RAINCOAT  | 53.1 | 58.5  | 40.2  | 30.6  | 58.9 | 77.8 | 37.0  | 37.8 | 67.6 | 11.8  | 47.3    |
| **Ours**  | **86.5** | **84.0** | 64.2 | 66.5 | 51.5 | 62.4 | 87.3 | **84.4** | 58.3 | 35.3 | **68.03** |

**Table 3. UCIHAR Accuracy (%) — By Source→Target Domain Pair**

| Algorithm | 2→11 | 6→23 | 7→13 | 9→18 | 12→16 | 18→27 | 20→5 | 24→8 | 28→27 | 30→20 | **Average** |
|-----------|------|------|------|------|-------|-------|------|------|-------|-------|---------|
| No Adapt  | 60.0 | 60.7 | 79.8 | 40.0 | 60.9  | 65.5  | 48.4 | 58.8 | 47.8  | 47.7  | 57.0    |
| TransPL   | 75.8 | 84.8 | 67.3 | 59.3 | 66.4  | 89.4  | 59.3 | 75.3 | 66.4  | 61.7  | 69.0    |
| SHOT      | 65.3 | 80.8 | 75.5 | 59.3 | 90.3  | 75.2  | 59.3 | 64.7 | 90.3  | 43.0  | 67.8    |
| **Ours**  | **84.4** | **86.2** | 71.4 | **62.0** | 68.2 | **89.7** | **64.5** | **78.2** | 68.6 | **66.5** | **73.97** |

> An improvement of **+4.97%** over TransPL (69.0%), demonstrating the effectiveness of causal structure alignment and uncertainty-aware pseudo-labeling.

### 4.2. Video Benchmarks

**Table 4. UCF101 ↔ HMDB51 (full) Classification Accuracy (%)**

| Method | Backbone | U→H | H→U | **Average** |
|--------|----------|-----|-----|---------|
| Source Only | ResNet101 | 73.9 | 71.7 | 72.8 |
| TA3N | - | 78.3 | 81.8 | 80.1 |
| MA2LT-D | - | 85.0 | 86.6 | 85.8 |
| TransferAttn | - | 88.1 | 88.3 | 88.2 |
| **Ours** | ResNet101 | **90.2** | **89.5** | **89.85** |

### 4.3. Ablation Study

#### OT Solver Comparison

![Accuracy and computational time comparison among Sinkhorn, Greenkhorn, and Unbalanced OT](/images/causal_ot/ot_solver_comparison.jpeg)

Sinkhorn achieves the highest alignment score (0.84) with reasonable training time. Greenkhorn (0.81) and Unbalanced OT (0.79) show marginal trade-offs in stability and computational cost.

#### t-SNE Visualization

![t-SNE visualization of the feature space after Causal-OT alignment (HAR, HHAR, WISDM)](/images/causal_ot/tsne_visualization.jpeg)

The known/unknown target distributions are well separated after alignment, qualitatively confirming the effect of latent space alignment.

#### Transferable Distance Loss Convergence

![Changes in Transferable Distance Loss per Epoch](/images/causal_ot/distance_loss.jpeg)

The loss consistently descends across all domain adaptation scenarios, showing that Causal-OT performs stable and generalizable learning.

### 4.4. Calibration Analysis

![Calibration performance comparison between TransPL (baseline) and Causal-OT (SSC, MFD)](/images/causal_ot/calibration_comparison.jpeg)

| Dataset | TransPL ECE | Causal-OT ECE | Improvement |
|----------|:-----------:|:-------------:|:----:|
| SSC      | 13.55       | 11.23         | ↓ 2.32 |
| MFD      | 4.37        | 3.78          | ↓ 0.59 |

The reduction in Expected Calibration Error (ECE) quantitatively confirms that Causal-OT **improves alignment between prediction probability and actual accuracy**, reducing overconfidence.

---

## 5. Summary of Key Contributions

1. **Unified Framework:** Simultaneously solves distribution alignment (OT), causal structure preservation (Granger), and uncertainty handling (entropy) within a single framework.
2. **Causally-Regularized OT:** Embeds the Granger causal graph directly into the OT cost matrix, realizing structurally consistent knowledge transfer.
3. **Uncertainty-aware Pseudo-Labeling:** Eliminates unreliable target samples using dual filtering based on entropy and causal consistency.
4. **Modality Generality:** The same pipeline achieves SOTA on both 1D time-series (6 benchmarks) and video (4 benchmarks).
5. **Theoretical Backbone:** Formally proves that minimizing the Causal-OT objective reduces the target risk upper bound.
