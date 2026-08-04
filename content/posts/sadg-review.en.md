---
title: "[CVPR 2026 Highlight] SADG: Structure-Aware Domain Generalization for Multi-Task Point Cloud Understanding"
date: 2026-08-03T11:53:00+09:00
draft: false
math: true
tags: ["Paper Review", "Point Cloud", "Domain Generalization", "Mamba", "State Space Model", "CVPR 2026 (Highlight)"]
categories: ["Paper Review"]
summary: "A comprehensive paper review of SADG (CVPR 2026 Highlight), a Mamba-based In-Context Learning framework for multi-task point cloud domain generalization combining structure-aware serialization, hierarchical domain modeling, and spectral graph alignment."
cover:
  image: "/images/sadg/overview.jpeg"
  alt: "SADG Framework Overview"
---

> Paper Information
> - Title: Mamba Learns in Context: Structure-Aware Domain Generalization for Multi-Task Point Cloud Understanding
> - Authors: Jincen Jiang, Qianyu Zhou, Yuhang Li, Kui Su, Meili Wang, Jian Chang, Jian Jun Zhang, Xuequan Lu
> - Affiliation: Bournemouth University, Jilin University, The University of Western Australia, etc.
> - Conference: CVPR 2026 (Highlight)
> - Code: [github.com/Jinec98/SADG](https://github.com/Jinec98/SADG)

---

## 1. Summary

To address the structural drift challenge in multi-domain and multi-task point cloud understanding caused by coordinate-driven serialization, SADG integrates Centroid Distance Spectrum (CDS) and Geodesic Curvature Spectrum (GCS) for structure-aware serialization, Hierarchical Domain-Aware Modeling (HDM), and test-time Spectral Graph Alignment (SGA) into a Mamba-based In-Context Learning framework, achieving consistent state-of-the-art performance across reconstruction, denoising, and registration tasks.

---

## 2. Research Background and Motivation

### 2.1. Problem Definition

3D point cloud understanding encompasses diverse tasks including surface reconstruction, denoising, and point cloud registration. In real-world environments, variations in sensors, viewpoint changes, and scene incompleteness cause severe domain shifts. Conventional single-task or single-domain architectures struggle to simultaneously handle multi-task point cloud understanding under domain shifts.

### 2.2. Limitations of Existing Methods

Existing approaches face limitations across three key dimensions:

| Approach | Key Limitations |
|----------|----------------|
| Transformer-based ICL (DG-PIC) | Quadratic self-attention complexity, lack of explicit token ordering causing structural inconsistency |
| Coordinate-driven Mamba (Z-order, Hilbert) | Sensitive to viewpoint changes and missing regions, breaking topological hierarchy and disrupting state propagation |
| DG Augmentation (PointMixup, PointCutMix) | Sharp performance degradation under multi-domain multi-task benchmarks without structural preservation |

The core difficulty lies in "structural drift". Reconstruction, denoising, and registration all rely on preserving structural hierarchy, including global topology (part-whole spatial organization) and local geometric continuity (surface smoothness and curvature). Under domain shifts, coordinate-driven serialization distorts sequence-local neighborhoods and disrupts intrinsic topological and geometric structures, making Mamba's recurrence fragile.

### 2.3. Core Contributions

- Propose SADG, the first structure-aware Mamba-based In-Context Learning framework for multi-task point cloud domain generalization.
- Design structure-aware serialization using Centroid Distance Spectrum (CDS) and Geodesic Curvature Spectrum (GCS) for transformation-invariant token ordering.
- Introduce Hierarchical Domain-Aware Modeling (HDM) to perform intra-domain structural modeling and inter-domain relational fusion.
- Present Spectral Graph Alignment (SGA) for structure-preserving test-time feature shifting in the spectral domain without parameter updates.
- Introduce MP3DObject, a real-world object scan dataset derived from Matterport3D for synthetic-to-real generalization benchmark.

---

## 3. Proposed Method: SADG Framework

### 3.1. Overall Architecture

![SADG Framework Overview](/images/sadg/overview.jpeg)

*Figure 1: Overview of SADG. (a) During training, multi-source point clouds are serialized via CDS and GCS into structure-aware sequences and processed by Mamba under HDM. (b) At test time, SGA aligns target features with source prototypes in the spectral domain.*

- Structure-Aware Sequence: Rather than coordinate-based or random ordering, this sequence reorganizes tokens into a structure-preserving permutation reflecting global topology (centroid distance) and local surface geometry (geodesic curvature).
- HDM (Hierarchical Domain-Aware Modeling): A cascading mechanism during training that first stabilizes intra-domain structures independently (ISM) and then interleaves tokens from prompt and query domains along their shared structural order for relational fusion (IRF).

SADG learns from $K$ source domains $\{D_s^k\}_{k=1}^K$ and generalizes to an unseen target domain $D_t$ without test-time weight updates. Following the In-Context Learning (ICL) paradigm, SADG processes prompt and query data tokens:
- Prompt Domain: Source domain point cloud tokens that serve as task and domain style guidance hints.
- Query Domain: Target point cloud tokens on which the model performs the required task (reconstruction, denoising, or registration).

### 3.2. Structure-Aware Serialization (SAS)

![Comparison of Serialization Strategies](/images/sadg/serialization_comparison.jpeg)

*Figure 2: Comparison of serialization strategies. CDS and GCS maintain transformation invariance and structural consistency across unaligned real scans.*

Given $N$ local patch tokens $\mathcal{T} = \{t_i\}_{i=1}^N$ extracted via FPS and KNN, SADG constructs a token graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, w)$ and defines token permutations $\pi$.

(1) Centroid Distance Spectrum (CDS)

CDS models global topological layout. Given global centroid $c = \frac{1}{N}\sum_{i=1}^N u_i$, SADG constructs graph $\mathcal{G}_{CDS}$ with Gaussian affinity:

$$
w_{CDS}(i,j) = \exp\left(-\frac{\|u_i - u_j\|_2^2}{\sigma^2}\right)
$$

Beginning from the token $t_r$ closest to centroid $c$, Breadth-First Search (BFS) traverses unvisited neighbors ranked by $w_{CDS}(i,j)$, producing a topologically smooth permutation $\pi_{CDS}$.

(2) Geodesic Curvature Spectrum (GCS)

GCS encodes intrinsic surface geometry through curvature-guided heat diffusion on a geodesic graph. Geodesic distance is computed via shortest path on token center adjacency:

$$
d_{\text{geo}}(i,j) = \min_{\mathcal{P}_{ij}} \sum_{(p,q) \in \mathcal{P}_{ij}} \|u_p - u_q\|_2
$$

Multi-scale self-diffusion coefficients $K_{\tau}(i,i)$ under the Laplace-Beltrami operator $\Delta$ formulate curvature implicitly:

$$
h_i = [K_{\tau_1}(i,i), K_{\tau_2}(i,i), \dots, K_{\tau_S}(i,i)]
$$

Constructing graph $\mathcal{G}_{GCS}$ with affinity $w_{GCS}(i,j) = \exp\left(-\frac{\|h_i - h_j\|_2^2}{\gamma^2}\right)$ and traversing in ascending curvature yields permutation $\pi_{GCS}$.

(3) Unified Sequence Construction

Concatenating bidirectional traversals of CDS and GCS forms the unified sequence:

$$
X_{\text{seq}} = [X_{\pi_{\text{CDS}}}; X_{\text{rev}(\pi_{\text{CDS}})}; X_{\pi_{\text{GCS}}}; X_{\text{rev}(\pi_{\text{GCS}})}]
$$

### 3.3. Hierarchical Domain-Aware Modeling (HDM)

![HDM Structure](/images/sadg/hdm_structure.jpeg)

*Figure 3: HDM cascades intra-domain structural modeling (ISM) and inter-domain relational fusion (IRF).*

(1) Intra-domain Structural Modeling (ISM)

Prompt and query sequences $\{X_{\text{seq}}^p, X_{\text{seq}}^q\}$ are first processed by parallel domain-specific Mamba branches:

$$
Z^p = \operatorname{Mamba}^p(X_{\operatorname{seq}}^p), \quad Z^q = \operatorname{Mamba}^q(X_{\operatorname{seq}}^q)
$$

(2) Inter-domain Relational Fusion (IRF)

Tokens from $\{Z^p, Z^q\}$ are interleaved following shared structural permutation $\pi$:

$$
Z^{pq} = [z_{\pi(1)}^p, z_{\pi(1)}^q, z_{\pi(2)}^p, z_{\pi(2)}^q, \dots, z_{\pi(4N)}^p, z_{\pi(4N)}^q]
$$

This sequence is processed by a Shared Mamba layer:

$$
Z^f = \operatorname{Mamba}^f(Z^{pq})
$$

Shared Mamba denotes a single Mamba block sharing identical weights across prompt and query tokens, enabling parameter-efficient inter-domain feature exchange via recurrent propagation.

### 3.4. Spectral Graph Alignment (SGA)

At test time, SGA performs feature shifting in the graph spectral domain before Mamba processing without parameter updates.

Spectral Domain: The representation space obtained by projecting spatial point signals onto the eigenvector basis of the normalized graph Laplacian $\mathbf{L}_* = \mathbf{D}_* - \mathbf{A}_*$.

Target spectral tokens are adaptively shifted toward source prototypes $\hat{P}_*^s$:

$$
\hat{P}_*^s = (\Phi_*^t)^\top \left( \frac{1}{N_s} \sum_{i=1}^{N_s} X_{\pi_*,i}^s \right)
$$

$$
\hat{X}_{*i}^t \leftarrow \alpha_i \hat{X}_{*i}^t + (1 - \alpha_i)(\hat{P}_*^s - \hat{X}_{*i}^t)
$$

Adaptive weight $\alpha_i$ prevents over-correction, and inverse Graph Fourier Transform restores features back to the spatial domain.

---

## 4. Experimental Results

### 4.1. Benchmark and Setup

Evaluated across five datasets (ModelNet40, ShapeNet, ScanNet, ScanObjectNN, MP3DObject) over seven categories, evaluated under leave-one-domain-out protocol for reconstruction, denoising, and registration using Chamfer Distance (CD, $\times 10^{-3}$).

### 4.2. Main Quantitative Results

| Method | Setting | ModelNet | | | ScanNet | | | MP3DObject | | |
|--------|---------|---------|-------|------|---------|-------|------|------------|-------|------|
| | | Rec. | Den. | Reg. | Rec. | Den. | Reg. | Rec. | Den. | Reg. |
| PointNet | General | 20.56 | 27.15 | 17.19 | 23.73 | 30.27 | 19.49 | 22.63 | 35.24 | 23.17 |
| Point-MAE | General | 14.77 | 21.53 | 13.42 | 16.16 | 24.54 | 16.79 | 20.39 | 27.28 | 17.13 |
| PointMamba | General | 16.47 | 23.13 | 13.65 | 15.61 | 22.43 | 14.30 | 20.16 | 27.08 | 17.40 |
| PointDGMamba | DG | 14.39 | 19.37 | 12.44 | 14.67 | 23.10 | 12.97 | 17.99 | 26.82 | 14.66 |
| DG-PIC | ICL+DG | 6.84 | 9.40 | 5.01 | 5.21 | 9.71 | 5.10 | 5.91 | 10.40 | 5.64 |
| Vanilla Mamba ICL | ICL+DG | 7.69 | 10.81 | 6.22 | 5.45 | 10.75 | 5.56 | 8.28 | 14.19 | 8.44 |
| SADG (Ours) | ICL+DG | 5.99 | 7.98 | 3.81 | 2.97 | 7.67 | 3.63 | 3.55 | 6.61 | 2.84 |

SADG outperforms all baseline models, achieving 3.55 / 6.61 / 2.84 CD error on MP3DObject, demonstrating superior domain generalization ability across both synthetic and real-world scans.

### 4.3. Ablation Study

![Visualization of Serialization Variants](/images/sadg/naive_vs_ours_serialization.jpeg)

*Figure 4: Comparison between naive serialization variants and proposed CDS/GCS.*

Ablation results on MP3DObject (CD, $\times 10^{-3}$):

| Variant | Rec. | Den. | Reg. |
|---------|------|------|------|
| Z-order Scanning | 7.32 | 12.47 | 6.29 |
| Hilbert Curve | 6.23 | 11.13 | 7.68 |
| w/o CDS | 4.75 | 10.69 | 6.17 |
| w/o GCS | 5.82 | 8.92 | 4.43 |
| Full SAS | 3.55 | 6.61 | 2.84 |
| w/o ISM | 5.41 | 11.37 | 7.51 |
| w/o IRF | 6.92 | 9.12 | 7.57 |
| Full HDM | 3.55 | 6.61 | 2.84 |

Removing CDS or GCS degrades performance, confirming that topology and geometry provide complementary structural cues. Removing ISM or IRF also significantly deteriorates accuracy, validating the effectiveness of HDM.

### 4.4. Qualitative Results and Efficiency

![Qualitative Results](/images/sadg/qualitative_results.jpeg)

*Figure 5: Qualitative results on synthetic (ModelNet) and real-scan (MP3DObject) targets.*

![t-SNE Feature Visualization](/images/sadg/tsne_visualization.jpeg)

*Figure 6: t-SNE visualization comparing Vanilla Mamba ICL and SADG.*

SADG achieves 0.75s inference time with 18.87M parameters and 14.89G FLOPs, outperforming DG-PIC (0.94s, 27.57M, 21.07G FLOPs) while delivering higher accuracy.

---

## 5. Key Contributions Summary

1. Structural Drift Solution: Identified structural drift in point cloud DG and proposed structure-aware serialization combining CDS and GCS to preserve global topology and local geometry.
2. Hierarchical Domain-Aware Modeling (HDM): Designed ISM and IRF to stabilize intra-domain structure and fuse inter-domain relations with linear-time efficiency.
3. Spectral Graph Alignment (SGA): Introduced lightweight test-time spectral feature shifting without model weight updates.
4. MP3DObject Dataset: Released a challenging real-world object scan dataset derived from Matterport3D with unaligned orientations and pose variations.
5. SOTA Generalization & Efficiency: Outperformed state-of-the-art methods across multi-domain multi-task benchmarks with 31.5% fewer parameters and 29.3% lower FLOPs.
