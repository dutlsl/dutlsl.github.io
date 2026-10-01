---
title: "[CVPR 2026] See Through the Noise: Improving Domain Generalization in Gaze Estimation"
date: 2026-10-01T18:26:00+09:00
draft: false
math: true
tags: ["Paper Review", "GAZE 2026", "Gaze Estimation", "Domain Generalization", "Noisy Labels", "Semantic Manifold", "CVPR 2026"]
categories: ["GAZE 2026", "Paper Review"]
summary: "Unveiling that label noise inherently corrupted in source gaze datasets severely degrades cross-domain generalization, SeeTN proposes semantic manifold learning over learnable prototypes to filter noisy labels and transfer clean structural order via dual regularization."
cover:
  image: "/images/seetn/_page_2_Figure_0.jpeg"
  alt: "SeeTN Framework Overview"
---

> Reference Paper
> - Peng, Y., Wang, S., Huang, Y., Tian, Y. "See Through the Noise: Improving Domain Generalization in Gaze Estimation." CVPR 2026.
> - Beijing Key Laboratory of Traffic Data Mining and Embodied Intelligence, Beijing Jiaotong University

![Figure 1: SeeTN Framework Overview](/images/seetn/_page_2_Figure_0.jpeg)
*Figure 1: Overall architecture of the SeeTN framework. Extracted visual features are normalized onto a unit hypersphere, followed by semantic manifold construction using twelve learnable prototypes. Cross-entropy between feature affinity and label affinity on this manifold measures the noise indicator $\eta$, partitioning samples into clean and noisy subsets for dual regularization.*

---

## 1. One-Sentence Summary

SeeTN reveals that corrupt label noise in source datasets is the primary culprit behind cross-domain degradation in gaze estimation, introducing prototype-based relative affinity to isolate corrupted labels and transferring structural manifold order to achieve state-of-the-art generalization.

---

## 2. Research Background and Motivation

### 2.1 Problem Definition

Imagine a shoe warehouse receiving thousands of shoe boxes daily, each bearing a printed size label on the outside. Inevitably, the factory mislabels a box, slapping a size 240 sticker onto a large size 270 shoe. If a newly hired clerk blindly memorizes these printed stickers without ever verifying the physical shoe dimensions, the clerk's perceptual judgment becomes corrupted. Once transferred to another branch, the clerk repeatedly hands out completely wrong sizes to customers.

Appearance-based 3D gaze estimation models face an identical dilemma. This task predicts continuous 3D gaze direction vectors directly from eye or face images. Conventional wisdom has long assumed that ground truth gaze labels in training benchmarks are pristine and flawless.

In reality, gaze data collection inherently suffers from human involuntary blinks, subtle head tremors, and optical reflections on eyeglasses, producing significant discrepancies between the recorded label and the actual optical gaze direction. When evaluated within the same training domain, models can easily memorize these corrupt labels. However, when deployed into unseen domains with different camera setups, user demographics, and illumination, this forced over-memorization causes generalization performance to collapse, resulting in massive angular errors.

![Figure 2: Label Noise in Gaze Datasets](/images/seetn/_page_0_Figure_10.jpeg)
*Figure 2: Inherent label noise in gaze datasets and resulting generalization degradation. Even in large-scale benchmarks, micro-movements and reflections cause divergence between real gaze and recorded labels. Models trained blindly on these noisy targets suffer severe error spikes in unseen target domains.*

### 2.2 Limitations of Existing Methods

First, prior domain generalization methods in gaze estimation assumed ground truth labels were unimpeachable, focusing entirely on erasing visual domain shifts such as lighting and subject identities. When ground truth targets themselves are corrupted, forcing domain alignment merely forces the model to memorize wrong labels in an even more contorted feature representation.

Second, established noisy label learning techniques from image classification cannot transfer to gaze estimation. Classification operates over discrete categories, where conflicting predictions can be clearly flagged. Gaze estimation is a continuous regression problem. A discrepancy of several degrees cannot trivially distinguish whether a sample is an outlier label or simply a challenging gaze angle, causing classification-based noise filters to fail catastrophically.

Third, naively discarding high-loss samples introduces severe selection bias. During early training epochs, complex gaze directions naturally incur higher prediction errors. Truncating large-loss samples discards valuable extreme gaze angles, causing the model to learn only trivial frontal gaze directions.

### 2.3 Main Contributions

To overcome these fundamental dilemmas, the authors propose three core contributions:

1. Uncovering that latent label noise in training benchmarks is the primary barrier to domain generalization in gaze estimation.
2. Formulating a prototype-based semantic manifold where the divergence between feature affinity and label affinity yields an explicit noise indicator $\eta$ capable of detecting regression label noise.
3. Establishing a dual regularization strategy that trains clean samples against ground truth targets while steering noisy samples along the clean manifold geometry, achieving superior cross-domain generalization.

---

## 3. Proposed Framework: SeeTN

### 3.1 Pipeline Overview and the Shoe Store Analogy

SeeTN operates analogous to an experienced store manager placing twelve standard reference shoes in ascending sizes along the counter to systematically audit incoming inventory.

Step 1 establishes the reference coordinate frame. Twelve learnable prototype vectors are anchored in feature space, allowing any incoming face sample to be measured against this standard grid.

Step 2 detects mislabeled inventory. The physical appearance similarity between an incoming shoe and the twelve reference shoes is compared against the sticker label similarity. If a shoe physically mirrors the size 270 prototype yet bears a 240 sticker, the sharp contradiction between visual reality and recorded label flags it as a corrupted sample.

Step 3 executes decoupled dual training. Accurately labeled samples are trained directly against their ground truth targets. Mislabeled samples disregard their corrupted numerical labels entirely, instead adopting the smooth relative alignment established by clean samples across the prototype shelf.

### 3.2 Semantic Manifold Construction

Conventional regression optimizes absolute angular differences, which allows head pose and identity variations to fracture feature topology. A semantic manifold is constructed to ensure samples sharing identical gaze directions cluster together irrespective of head pose.

#### 1. Feature Direction Normalization: Eq. 1

Given face image $x_i$, visual backbone features $f_i$ are projected to filter out gaze-irrelevant appearance attributes:

$$z_i = \frac{\text{MLP}(f_i)}{\|\text{MLP}(f_i)\|_2}$$

The MLP isolates gaze-relevant directional cues from identity and lighting variations. The denominator L2 normalization constrains all feature vectors onto a unit hypersphere. Because gaze direction depends purely on angular orientation rather than feature magnitude, this projection enforces strict directional comparisons.

#### 2. Twelve Prototypes and Moving Average Updates: Eq. 2, Eq. 3

Twelve learnable prototype vectors $\mu$ are anchored across the hypersphere:

$$r_{i,k} = \frac{\exp(z_i \cdot \mu_k^T / \tau)}{\sum_{j=1}^K \exp(z_i \cdot \mu_j^T / \tau)}$$

$$\mu_k \leftarrow \alpha \mu_k + (1 - \alpha) \frac{\sum_i r_{i,k} z_i}{\sum_i r_{i,k}}$$

Here $r_{i,k}$ represents the soft assignment probability of feature $z_i$ to the $k$-th prototype. The inner product represents cosine angular similarity between unit vectors. Temperature parameter $\tau$ softens the distribution, ensuring continuous regression angles smoothly span across adjacent prototypes rather than collapsing into hard categorical assignments.

Prototypes are updated via exponential moving average toward the center of mass of assigned features. Momentum $\alpha=0.95$ preserves 95% of previous prototype coordinates while absorbing 5% of new batch statistics, safeguarding against erratic fluctuations caused by transient outliers.

The prototype count $K=12$ is rigorously grounded in ablation studies. $K=8$ fails to capture spatial gaze diversity, while $K \ge 16$ causes prototype boundaries to blur and overfit to source subject identities. Thus $K=12$ provides the optimal trade-off.

#### 3. Relative Affinity Manifold: Eq. 4, Eq. 5, Eq. 6

Projecting feature $z_i$ onto the prototypes yields a 12-dimensional manifold coordinate $p_i$:

$$p_i = z_i \cdot \mu^T$$

Batch-wide relative relationships are modeled via pairwise feature affinity $A^{(m)}$ and ground truth label affinity $A^{(g)}$:

$$A_{i,j}^{(m)} = \frac{p_i \cdot p_j}{\|p_i\|_2 \|p_j\|_2}$$

$$A_{i,j}^{(g)} = \frac{y_i \cdot y_j}{\|y_i\|_2 \|y_j\|_2}$$

$A_{i,j}^{(m)}$ captures similarity between sample representations in manifold space, while $A_{i,j}^{(g)}$ captures cosine similarity between recorded gaze label vectors. Aligning $A^{(m)}$ with $A^{(g)}$ forces features to structure themselves isomorphic to real physical gaze topology regardless of head pose.

### 3.3 Noisy Sample Detection: Eq. 7

Corrupted labels are detected by evaluating the divergence between feature and label affinity distributions across batch neighbors:

$$\hat{y}_i^{(m)} = \text{Softmax}(A_{i,:}^{(m)})$$

$$\hat{y}_i^{(g)} = \text{Softmax}(A_{i,:}^{(g)})$$

For pristine samples, neighbor distributions $\hat{y}_i^{(m)}$ and $\hat{y}_i^{(g)}$ align closely. For corrupted samples, visual features indicate one gaze direction while the false label points elsewhere, inducing severe distributional mismatch.

Cross-entropy measures this divergence to compute the noise indicator $\eta_i$:

$$\eta_i = - \sum_{j=1}^B \hat{y}_{i,j}^{(g)} \log(\hat{y}_{i,j}^{(m)})$$

The leading negative sign converts negative log values into positive penalties. When a label is corrupted, target neighbors have high label probability but near-zero feature probability. Because $\log(0) \to -\infty$, the negative sign causes $\eta_i$ to explode toward positive infinity, reliably isolating corrupted samples.

The top $t\%$ samples with the highest $\eta_i$ within each batch are partitioned into the noisy subset $\mathcal{D}_S^N$, leaving the remainder in clean subset $\mathcal{D}_S^C$. Threshold $t$ is calibrated to dataset noise characteristics, using 5% for laboratory ETH-XGaze and 10% for wild Gaze360. Dynamic per-epoch repartitioning allows initially misclassified samples to be reclaimed as model representations mature.

### 3.4 Dual Regularization Optimization: Eq. 8 ~ Eq. 12

Partitioned subsets are trained under decoupled objectives tailored to their reliability.

#### 1. Clean Sample Supervision: Eq. 8, Eq. 9

Clean subset $\mathcal{D}_S^C$ is optimized directly against ground truth labels:

$$\mathcal{L}_{\text{gaze}} = \frac{1}{B_C} \sum_{x_i \in \mathcal{D}_S^C} |g(x_i) - y_i|$$

$$\mathcal{L}_{\text{align}}^C = \frac{1}{B_C(B_C - 1)} \sum_{x_i \in \mathcal{D}_S^C} \sum_{x_j \in \mathcal{D}_S^C, i \neq j} |A_{i,j}^{(g)} - A_{i,j}^{(m)}|$$

$\mathcal{L}_{\text{gaze}}$ minimizes L1 error between predicted and target gaze vectors. Concurrently, $\mathcal{L}_{\text{align}}^C$ enforces rigid MAE alignment between feature affinity $A^{(m)}$ and label affinity $A^{(g)}$, anchoring a stable geometric backbone across prototypes.

#### 2. Noisy Sample Regularization: Eq. 10, Eq. 11

Noisy subset $\mathcal{D}_S^N$ bypasses corrupted label values completely to avoid harmful memorization, while retaining visual features by regularizing them against clean manifold structure:

$$A_{i,j}^{(f)} = \frac{f_i^N \cdot f_j^C}{\|f_i^N\|_2 \|f_j^C\|_2}$$

$$\mathcal{L}_{\text{align}}^N = - \frac{1}{B_N} \sum_{x_i \in \mathcal{D}_S^N} \frac{A_{i,:}^{(f)} \cdot A_{i,:}^{(m)}}{\|A_{i,:}^{(f)}\|_2 \|A_{i,:}^{(m)}\|_2}$$

$A_{i,j}^{(f)}$ denotes cosine similarity between noisy raw features $f^N$ and clean raw features $f^C$. $\mathcal{L}_{\text{align}}^N$ maximizes cosine similarity between raw affinity vector $A^{(f)}$ and manifold affinity vector $A^{(m)}$, prepended with a minus sign for minimization.

Unlike rigid MAE supervision, smooth cosine alignment avoids crushing intrinsic identity variations, allowing noisy samples to align their directional orientation along the clean manifold without distorting visual expressiveness.

#### 3. Total Objective: Eq. 12

The combined training objective is formulated as:

$$\mathcal{L}_{\text{all}} = \mathcal{L}_{\text{gaze}} + \mathcal{L}_{\text{align}}^C + \lambda \mathcal{L}_{\text{align}}^N$$

Regularization weight $\lambda=0.1$ moderates noisy supervision, ensuring auxiliary manifold guidance enriches feature representations without interfering with primary clean supervision.

During deployment, prototype and affinity modules are stripped away; the trained backbone and gaze head predict 3D gaze vectors with zero computational overhead.

---

## 4. Experimental Results

### 4.1 Quantitative Benchmarks and Cross-Domain Evaluation

Generalization was evaluated across four challenging cross-domain benchmarks, transferring from ETH-XGaze and Gaze360 to MPIIFaceGaze and EyeDiap without accessing target domain data during training.

| Backbone | Method | $\mathcal{D}_E \to \mathcal{D}_M$ | $\mathcal{D}_E \to \mathcal{D}_D$ | $\mathcal{D}_G \to \mathcal{D}_M$ | $\mathcal{D}_G \to \mathcal{D}_D$ | In-Domain $\mathcal{D}_E$ | In-Domain $\mathcal{D}_G$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| ResNet-18 | Baseline | 8.07° | 8.78° | 7.94° | 8.73° | 4.64° | 11.14° |
| ResNet-18 | SeeTN (Ours) | 6.58° (18.5% drop) | 7.18° (18.2% drop) | 6.57° (17.2% drop) | 7.57° (13.3% drop) | 4.40° (5.1% drop) | 10.73° (3.7% drop) |
| ResNet-50 | Baseline | 7.64° | 8.39° | 7.68° | 8.65° | 4.32° | 10.78° |
| ResNet-50 | SeeTN (Ours) | 6.31° (17.4% drop) | 6.84° (18.4% drop) | 6.75° (12.1% drop) | 7.42° (14.2% drop) | 4.16° (3.7% drop) | 10.45° (3.1% drop) |

Across ResNet-18 and ResNet-50, SeeTN consistently delivers 12% to 18.5% angular error reductions across all cross-domain scenarios. Addressing label noise within the source domain alone yields profound generalization gains without requiring target data access.

Furthermore, within-domain accuracy also improves by 3% to 5%. Unlike conventional techniques that sacrifice source accuracy for transferability, SeeTN enhances intrinsic feature quality across both native and unseen domains.

### 4.2 Comparison with Noise Learning Methods and Ablation Studies

Comparative evaluations confirm that classification-oriented noise algorithms fail in continuous regression tasks.

| Method | Mechanism | $\mathcal{D}_E \to \mathcal{D}_M$ | $\mathcal{D}_E \to \mathcal{D}_D$ | $\mathcal{D}_G \to \mathcal{D}_M$ | $\mathcal{D}_G \to \mathcal{D}_D$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Baseline | Standard L1 Loss | 8.07° | 8.78° | 7.94° | 8.73° |
| DivideMix | Semi-Supervised MixMatch | 9.72° | 20.09° | 10.46° | 12.30° |
| SUGE | In-Domain Noise Suppression | 10.00° | 8.74° | 7.04° | 8.32° |
| SeeTN | Manifold Affinity Regularization | 6.58° | 7.18° | 6.57° | 7.57° |

DivideMix degrades performance below baseline levels because categorical mixup disrupts the topological continuity of gaze angles. In contrast, SeeTN preserves continuous geometry through relative affinity matching.

Ablating $\mathcal{L}_{\text{align}}^N$ increases angular error by over 0.6°, demonstrating that guiding noisy representations along clean manifold geometry extracts far more utility than simply discarding corrupted data.

| Prototypes $K$ | $\mathcal{D}_E \to \mathcal{D}_M$ | $\mathcal{D}_G \to \mathcal{D}_M$ | Noise Ratio $t$ | $\mathcal{D}_E \to \mathcal{D}_M$ | $\mathcal{D}_G \to \mathcal{D}_M$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $K = 8$ | 7.37° | 6.90° | $t = 5\%$ | 6.58° | 6.85° |
| $K = 12$ | 6.58° | 6.57° | $t = 10\%$ | 7.25° | 6.57° |
| $K = 16$ | 7.59° | 6.44° | $t = 20\%$ | 7.63° | 6.97° |

$K=12$ achieves peak performance, avoiding both insufficient angular capacity ($K=8$) and domain identity overfitting ($K=16$). Noise ratio thresholds of 5% on ETH-XGaze and 10% on Gaze360 yield optimal balance.

### 4.3 Qualitative Feature Space Visualization

t-SNE visualization of 800 samples from a single subject illustrates the geometric transformation enabled by SeeTN.

![Figure 3: Qualitative t-SNE Feature Visualization](/images/seetn/_page_7_Figure_0.jpeg)
*Figure 3: t-SNE feature space comparison on ETH-XGaze between baseline and SeeTN. Red arrows denote true gaze direction. Baseline features cluster erroneously based on head pose shortcuts, whereas SeeTN eliminates head pose interference to form a continuous circular manifold aligned with gaze direction.*

The baseline model groups features according to head pose shortcuts regardless of actual gaze direction. In contrast, SeeTN clusters identical gaze angles together despite head pose divergence, forming smooth circular trajectories where gaze orientation rotates continuously clockwise and angular deviation expands radially from the center.

![Figure 4: Angular Error Distribution](/images/seetn/_page_7_Figure_8.jpeg)
*Figure 4: Cumulative angular error distribution curves. SeeTN rises sharply toward the upper left, confirming that the vast majority of test samples achieve tight angular error bounds.*

---

## 5. Conclusion and Key Takeaways

SeeTN demonstrates that overlooked label noise within training benchmarks is a fundamental root cause of generalization failure in gaze estimation.

While prior research focused heavily on scaling model capacity or engineering complex domain adaptation modules, SeeTN addresses the fragile foundation of corrupted supervision targets. When training labels are compromised, sophisticated adaptation techniques merely learn distorted representations.

By establishing prototype-based relative affinity, SeeTN solves the long-standing challenge of detecting label noise in continuous regression without disrupting geometric continuity. Decoupling clean supervision from noisy manifold regularization preserves valuable visual features while neutralizing label corruption. This paradigm provides valuable insights for broader continuous regression challenges, including 3D pose estimation, autonomous vehicle steering prediction, and embodied spatial perception.
