---
title: "[CVPR 2026] WeakMed: Rethinking Box Supervision — Bias-Free Weakly Supervised Medical Segmentation Paper Review"
date: 2026-07-29T19:27:00+09:00
draft: false
math: true
tags: ["Paper Review", "Weakly Supervised Learning", "Medical Image Segmentation", "CVPR 2026"]
categories: ["Paper Review"]
summary: "A review of WeakMed (CVPR 2026 Poster), a general-purpose weakly supervised medical image segmentation framework using Mask-to-Box transformation and Scale Consistency loss."
cover:
  image: "/images/weakmed/pipeline.jpeg"
  alt: "WeakMed Pipeline Image"
---

> Paper Information
> - Title: Rethinking Box Supervision: Bias-Free Weakly Supervised Medical Segmentation
> - Authors: Jun Wei, Hui Huang
> - Affiliation: Shenzhen University (VCC, College of Computer Science and Software Engineering)
> - Venue: CVPR 2026 (Poster)

---

## 1. One-line Summary

WeakMed is a general-purpose weakly supervised framework that trains medical image segmentation models using only bounding box annotations, introducing a Mask-to-Box (M2B) transformation to eliminate box-induced structural bias and a Scale Consistency (SC) loss to resolve weak label ambiguity, achieving superior performance across 9 segmentation tasks, 9 datasets, and 6 imaging modalities.

---

## 2. Research Background and Motivation

### 2.1. Problem Definition

Medical image segmentation plays a fundamental role in clinical pipelines such as colonoscopy, skin lesion analysis, glaucoma screening, chest X-ray interpretation, cell nuclei segmentation, and organ delineation. Although convolutional networks (UNet++, nnU-Net, SANet) and vision transformers (TransFuse, MedT) have shown impressive performance, most existing approaches depend heavily on dense pixel-level annotations.

![Figure 1: Medical segmentation tasks and pipeline comparison](/images/weakmed/pipeline.jpeg)
*Figure 1: (a) Examples from 9 medical segmentation tasks (b) Fully supervised pipeline requiring dense pixel-level masks (c) WeakMed framework trained with only bounding box annotations*

Dense pixel-level annotations present severe challenges:
- High Annotation Cost: Labeling every pixel requires extensive time and expert domain knowledge.
- Boundary Ambiguity: Unclear lesion boundaries introduce subjective variability and annotation noise.
- Limited Generalization: Noisy contour labels degrade model generalization on unseen data.

### 2.2. Limitations of Existing Methods

Bounding box supervision offers a cost-effective alternative, but conventional box-based methods suffer from fundamental structural limitations:

| Category | Representative Methods | Core Limitations |
|------|----------|----------|
| Box-based Instance Segmentation | BoxInst, DiscoBox, BoxLevelSet | Rectangular shape of box labels injects structural bias → fails on complex non-rectangular shapes |
| Pseudo-Mask Based | BoxPolyp | Relies on heuristic pseudo-labeling pipelines → error accumulation and unstable training |
| Task-Specific | WeakPolyp | Tailored for specific targets (polyps) → fails to generalize to multi-object or diverse tasks |

> The key challenges are summarized as follows:
> 1. Box-Induced Structural Bias: Direct supervision using bounding boxes causes models to learn rectangular predictions, losing fine-grained shape information.
> 2. Ambiguity in Weak Supervision: Multiple distinct mask predictions collapse into the same box representation, leaving optimization under-constrained.

### 2.3. Key Contributions

1. WeakMed Framework: A model-agnostic, generalizable weakly supervised segmentation framework relying solely on bounding box annotations without modifying network architectures.
2. Mask-to-Box (M2B) Transformation: A differentiable projection module that maps predicted masks into box-aligned representations, alleviating box-induced structural bias.
3. Scale Consistency (SC) Loss: A multi-scale self-supervised regularization loss that compensates for the information loss in M2B and reduces supervision ambiguity.
4. Extensive Validation: Demonstrated consistent superiority over weakly supervised baselines and competitive accuracy with fully supervised models across 9 tasks, 9 datasets, and 6 imaging modalities.

---

## 3. Proposed Method: WeakMed Framework

### 3.1. Overall Architecture

WeakMed operates in a two-stage pipeline: segmentation phase and supervision phase.

![Figure 2: Overview of the WeakMed framework](/images/weakmed/overview.jpeg)
*Figure 2: (a) Two-stage pipeline (b) Mask-to-Box (M2B) transformation (c) Illustration of supervision ambiguity*

- Segmentation Phase: Multi-scale features are extracted using a standard backbone (PVTv2-B2), aligned to a common resolution, and fused to generate mask predictions.
- Supervision Phase: The M2B transformation and SC loss provide effective training signals from bounding box annotations. Both modules are used only during training and impose zero inference cost.

Given an input image $I$, two resized versions $I_1 \in \mathbb{R}^{s_1 \times s_1}$ and $I_2 \in \mathbb{R}^{s_2 \times s_2}$ are independently fed into the segmentation model to produce predictions $P_1, P_2$, which are resized to a common resolution for supervision.

### 3.2. Mask-to-Box (M2B) Transformation

Directly supervising predicted masks $P_1 / P_2$ with ground-truth boxes $B$ causes rectangular bias. M2B reformulates supervision by projecting predictions into box-aligned representations $T_1 / T_2$.

![Figure 3: Geometric constraint enforcement in M2B](/images/weakmed/m2b.jpeg)
*Figure 3: M2B projects inaccurate mask predictions into a box representation to penalize spatial inconsistencies*

① Projection

Given a predicted mask $P \in [0, 1]^{H \times W}$ and a bounding box $[x, y, w, h]$, the local region $P' = P[x:x+w, y:y+h] \in [0, 1]^{h \times w}$ is extracted. Max pooling is performed along horizontal and vertical axes:

$$P_w = \max(P', \text{dim}=0) \in [0, 1]^{1 \times w}$$

$$P_h = \max(P', \text{dim}=1) \in [0, 1]^{h \times 1}$$

This compresses the 2D mask into two 1D descriptors, encoding spatial occupancy while discarding detailed shape noise.

② Back-projection

Using $P_w$ and $P_h$, a box-aligned support representation is reconstructed:

$$\hat{P}_w = \mathbf{1}_h \cdot P_w$$

$$\hat{P}_h = P_h \cdot \mathbf{1}_w^\top$$

$$T' = \min(\hat{P}_w, \hat{P}_h)$$

where $\mathbf{1}_h \in \mathbb{R}^{h \times 1}$ and $\mathbf{1}_w \in \mathbb{R}^{w \times 1}$ are all-one vectors. Column-wise expansion $\hat{P}_w$ and row-wise expansion $\hat{P}_h$ intersect to form $T'$. The final transformed mask $T$ replaces the region in $P$ with $T'$.

③ Supervision Loss

Transformed masks $T_1, T_2$ are supervised against ground-truth box masks $B$ using binary cross-entropy and Dice losses:

$$\mathcal{L}_{M2B} = 0.5[\mathcal{L}_{BCE}(T_1, B) + \mathcal{L}_{BCE}(T_2, B)] + 0.5[\mathcal{L}_{Dice}(T_1, B) + \mathcal{L}_{Dice}(T_2, B)]$$

Interpretation: M2B acts as a projection operator that maps dense mask predictions into a box-constrained space, removing fine-grained ambiguity while preserving spatial extent.

### 3.3. Scale Consistency (SC) Loss

Because M2B discards detailed shape information during projection, optimization can become under-constrained due to many-to-one mapping.

To resolve this ambiguity, Scale Consistency loss enforces probabilistic consistency between predictions $P_1$ and $P_2$ from different input scales within the bounding box region $\Omega_B$:

$$\mathcal{L}_{SC} = \sum_{(i,j) \in \Omega_B} \frac{\text{KL}(P_1^{i,j} \| P_2^{i,j}) + \text{KL}(P_2^{i,j} \| P_1^{i,j})}{2|\Omega_B|}$$

SC complements M2B by providing dense pixel-level regularization across scales without requiring fine contour labels.

### 3.4. Total Training Loss

$$\mathcal{L}_{Total} = \mathcal{L}_{M2B} + \mathcal{L}_{SC}$$

Equal weighting ($\lambda_{M2B} = \lambda_{SC} = 1$) is used across all experiments. WeakMed requires no architectural changes and introduces no computation overhead during inference.

---

## 4. Experimental Results

### 4.1. Experimental Setup

- Datasets: 9 datasets across 6 imaging modalities — SUN-SEG (colonoscopy), Nuclei (microscopy), WBC (blood cell), ISIC (dermoscopy), Chest X-ray (radiography), REFUGE (fundus), Spleen (CT), LiTS (liver CT), KiTS (kidney CT)
- Backbones: PVTv2-B2 (primary), Res2Net-50 (cross-validation)
- Training: SGD (momentum 0.9, weight decay $10^{-4}$), initial lr 0.01, batch size 16, 16 epochs
- Evaluation Metrics: Dice score and IoU (%)

### 4.2. Polyp Segmentation (SUN-SEG)

| Method | Validation Dice | Validation IoU | Testing Dice | Testing IoU |
|------|:-:|:-:|:-:|:-:|
| BoxInst | 76.4 | 66.6 | 77.3 | 67.8 |
| DiscoBox | 75.2 | 65.3 | 72.8 | 63.3 |
| BoxLevelSet | 72.4 | 63.0 | 72.5 | 63.6 |
| BoxTeacher | 77.3 | 66.9 | 78.1 | 68.2 |
| CDSP | 77.9 | 67.1 | 79.1 | 69.4 |
| BoxSup-pvt (Baseline) | 77.3 | 65.3 | 77.7 | 66.0 |
| WeakMed-pvt | 85.6 | 78.2 | 85.9 | 79.0 |

With the PVTv2-B2 backbone, WeakMed achieves 85.9% Dice on the testing set, outperforming BoxSup by +8.2%p and BoxLevelSet by +13.4%p.

### 4.3. Comprehensive Comparison Across 8 Datasets

| Method | Nuclei | WBC | ISIC | X-ray | REFUGE | Spleen | LiTS | KiTS | Average Dice / IoU |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| BoxInst | 33.9 | 72.2 | 52.9 | 89.8 | 63.2 | 51.9 | 75.5 | 76.9 | 64.5 / 53.5 |
| DiscoBox | 60.9 | 89.5 | 76.9 | 90.3 | 64.1 | 84.1 | 72.1 | 83.1 | 77.6 / 69.0 |
| BoxLevelSet | 43.3 | 78.3 | 70.5 | 67.6 | 77.8 | 54.8 | 68.7 | 75.0 | 67.0 / 56.6 |
| BoxSup | 78.5 | 80.4 | 80.1 | 71.8 | 83.0 | 70.2 | 71.1 | 80.2 | 76.9 / 64.1 |
| WeakMed | 85.7 | 96.3 | 84.3 | 94.3 | 91.4 | 91.6 | 91.4 | 92.0 | 90.9 / 84.7 |

WeakMed consistently achieves top performance across all datasets, boosting average Dice to 90.9% (+14.0%p over BoxSup).

![Figure 5: Qualitative comparison across 9 medical segmentation datasets](/images/weakmed/qualitative.jpeg)
*Figure 5: Qualitative segmentation results predicted by weakly supervised methods across 9 medical image datasets*

### 4.4. Ablation Study

| Backbone | BoxSup | M2B | SC | Validation Dice | Testing Dice |
|:---:|:---:|:---:|:---:|:-:|:-:|
| Res2Net | ✓ | | | 76.1 | 75.2 |
| Res2Net | ✓ | | ✓ | 76.9 | 75.6 |
| Res2Net | ✓ | ✓ | | 81.4 | 79.9 |
| Res2Net | ✓ | ✓ | ✓ | 84.0 | 82.4 |
| PVTv2 | ✓ | | | 77.3 | 77.7 |
| PVTv2 | ✓ | | ✓ | 77.6 | 77.9 |
| PVTv2 | ✓ | ✓ | | 83.0 | 83.2 |
| PVTv2 | ✓ | ✓ | ✓ | 85.6 | 85.9 |

- M2B Alone: Improves Dice by +5.5%p (77.7% → 83.2%), confirming its critical role in removing box bias.
- SC Alone: Yields marginal gain (+0.2%p) when used without M2B due to under-constrained box supervision.
- Synergy (M2B + SC): Adding SC on top of M2B brings an extra +2.7%p gain, demonstrating strong complementary effect.

### 4.5. Comparison with Fully Supervised Baselines

WeakMed achieves performance comparable to fully supervised baselines (UNet, UNet++, PraNet, SANet, PNS+). In scenarios with noisy or ambiguous pixel annotations, coarse box supervision combined with M2B and SC can provide more stable learning signals.

### 4.6. Further Analysis

![Figure 4: Comprehensive analysis and ablation studies](/images/weakmed/analysis.jpeg)
*Figure 4: (a) Comparison with fully supervised models (b, c) Data efficiency under varying dataset sizes (d) Robustness across lesion sizes (e) Effect of SC loss*

- Data Efficiency: WeakMed maintains consistent performance gains as training data increases, demonstrating superior scalability compared to BoxSup.
- Lesion Scale Robustness: Performance remains robust across small (< 0.5) and large (> 0.5) lesion size intervals.

---

## 5. Summary of Key Contributions

(1) Universal Weakly Supervised Framework

WeakMed operates purely at the supervision level through lightweight M2B and SC modules. It acts as a plug-and-play component for existing segmentation backbones with zero inference overhead and natural support for multi-object scenes.

(2) Elimination of Box Structural Bias via M2B

Instead of directly comparing predicted masks with rectangular boxes, M2B projects predictions into a box-aligned space before supervision. This allows the model to output flexible non-rectangular shapes while enforcing necessary spatial extent constraints.

(3) Disambiguation via Scale Consistency Loss

SC minimizes symmetric KL divergence between predictions from different input scales, compensating for the many-to-one mapping in M2B and stabilizing training.

(4) Broad Generalization Across Modalities

Across 9 datasets and 6 modalities, WeakMed achieves an average Dice of 90.9%, proving that bounding box supervision can achieve high-quality medical image segmentation in real-world applications.
