---
title: "[ECCV 2026] SegFS: Real-Time Open-Vocabulary Video Instance Segmentation with Dual-Path Processing"
date: 2026-09-22T20:16:59+09:00
draft: false
math: true
tags: ["Paper Review", "Video Instance Segmentation", "Open-Vocabulary", "Real-Time", "Mobile", "Dual-Path", "ECCV 2026"]
categories: ["Paper Review"]
summary: "SegFS alternates between a slow path executing heavy object-centric components only on sparse keyframes and a lightweight fast path predicting masks directly from backbone features, achieving over 30 FPS real-time open-vocabulary video instance segmentation on mobile devices."
cover:
  image: "/images/segfs/_page_5_Figure_2.jpeg"
  alt: "SegFS Dual-Path Architecture Overview"
---

> Reference Paper
> - Barsellotti, L. et al. "Segmenting, Fast and Slow: Real-Time Open-Vocabulary Video Instance Segmentation with Dual-Path Processing." ECCV 2026.

![Figure 1: SegFS Overview](/images/segfs/_page_5_Figure_2.jpeg)
*Figure 1: Overall architecture of SegFS. The slow path extracts object embeddings only on sparse keyframes, while the fast path performs lightweight mask predictions on inter-frames directly from backbone features.*

---

## 1. One-Sentence Summary

Conventional Open-Vocabulary Video Instance Segmentation models suffer from severe inference bottlenecks due to repeatedly running heavy Feature Enhancers on every frame. SegFS resolves this by offloading heavy semantic processing to a sparse slow path while operating an ultra-lightweight fast path on intermediate frames, achieving 30 FPS real-time inference on a Samsung Galaxy S25 Ultra with up to 14x lower latency than prior mobile architectures.

---

## 2. Research Background and Motivation

### 2.1 Problem Definition

Consider watching pedestrians navigating a bustling street intersection. When encountering multiple people and vehicles, the human visual system does not engage in exhaustive, language-level identification dozens of times each second. It does not ask "what exact clothing is that person wearing?" or "is that four-wheeled vehicle a sedan or a truck?" on every single millisecond. Instead, an initial glance establishes semantic identity ("there is a pedestrian in blue and a dark sedan"), after which the visual system ceases complex conceptual analysis for several seconds, relying on rapid perceptual tracking to trace boundaries and motion across retinal inputs. Heavy semantic understanding occurs infrequently, while continuous tracking operates on fast, intuitive sensory reflexes, preventing cognitive overload.

Open-Vocabulary Video Instance Segmentation (OV-VIS) aims to equip AI with this perceptual capability. Given arbitrary text prompts such as "person", "bicycle", or "puppy" directed at a live video stream, the model must detect, track, and segment arbitrary object categories—including unseen classes—in real time at the pixel level.

The standard paradigm for OV-VIS adopts DETR-style object-centric architectures comprising three sequential components:

1. **Visual Backbone**: Extracts a multi-scale feature pyramid from each frame, generating feature maps across varying strides from fine-grained $P_2$ to coarse-grained $P_5$.
2. **Feature Enhancer**: The primary computational bottleneck. A pixel decoder exchanges information across scales via multi-scale deformable attention, while an early fusion module conducts bidirectional vision-language cross-attention to inject text category embeddings into visual representations. Only after this costly stage can feature maps discern distinct semantic categories.
3. **Object Decoder**: Learnable object queries cross-attend to the text-aligned feature map to extract instance-level embeddings. These embeddings predict classification logits via cosine similarity with text embeddings and generate segmentation masks via $1 \times 1$ convolutions over high-resolution feature maps.

Inter-frame tracking typically follows the MinVIS paradigm: without dedicated spatiotemporal tracking modules, identical instances across frames are linked via bipartite matching based on cosine similarity between object embeddings.

### 2.2 Limitations of Existing Methods: The Feature Enhancer Bottleneck

![Figure 2: Efficiency Analysis](/images/segfs/_page_1_Figure_2.jpeg)
*Figure 2: Efficiency analysis across OV-VIS architectures. FLOPs and on-device latency on a Samsung Galaxy S25 Ultra are decomposed across visual backbone, Feature Enhancer, and Object Decoder for MOBIUS-Mini-M, GLEE-Lite, and TROY-VIS.*

Returning to the perceptual analogy, prior OV-VIS methods operate like an observer compulsively re-evaluating "what is that object?" through heavy linguistic-visual cross-attention thirty times a second.

Recent mobile-oriented architectures, such as MOBIUS and TROY-VIS, sought to streamline this process. MOBIUS restricts vision-language fusion and multi-scale attention to a single intermediate scale, while TROY-VIS applies vision-language cross-attention solely at the coarsest scale before propagating features upward.

However, as shown in Figure 2, the Feature Enhancer continues to dominate overall inference costs despite these optimizations. This disparity is particularly stark in on-device latency compared to theoretical FLOPs. Dense operations like multi-scale deformable attention and vision-language cross-attention exhibit memory-bound access patterns that underutilize mobile GPU parallelism. As input resolution and vocabulary size scale, this latency penalty grows severe.

While prior lightweight designs trimmed the computational footprint of these dense interactions, they preserved the structural flaw of running cross-modal fusion on every frame. Similarly, keyframe-based strategies like MobileInst that omit only the Object Decoder on inter-frames yield minimal speedups because the true bottleneck—the Feature Enhancer—remains active across all frames.

### 2.3 Main Contributions

- SegFS introduces a dual-path paradigm that offloads both the Feature Enhancer and the Object Decoder to sparse keyframes, eliminating the primary computational bottleneck of real-time OV-VIS.
- The framework introduces a Fast Feature Aggregator that projects keyframe object embeddings back into raw backbone feature space and fuses them into intermediate frames, enabling high-quality mask prediction directly from backbone features.
- Extensive experiments across multiple OV-VIS benchmarks demonstrate that SegFS preserves the zero-shot capabilities and mask fidelity of heavy object-centric models while exceeding 30 FPS on a Samsung Galaxy S25 Ultra.

---

## 3. Proposed Framework: SegFS

The core intuition behind SegFS is straightforward: visual backbone feature maps already encode sufficient spatial and semantic cues for fine-grained localization. Consequently, the heavy Feature Enhancer does not need to execute on every frame; injecting keyframe object embeddings directly into raw backbone features is sufficient to generate accurate masks.

Backbone pyramids naturally capture boundaries, silhouettes, and regional geometry. While raw backbone features lack the explicit text alignment needed to differentiate a person from a bicycle, this semantic attribution—established once by the slow path on a keyframe—remains valid over short temporal spans. SegFS reuses this semantic judgment across subsequent frames rather than re-computing it from scratch.

### 3.1 Architecture Overview: Dual-Path Processing

SegFS implements this perceptual division of labor by strategically alternating between two sub-networks:

The **Slow Path** encompasses a complete, frozen object-centric OV-VIS model (such as GLEE, MOBIUS, or TROY-VIS). Operating only on sparse keyframes (e.g., every 5 to 6 frames), it processes the full pipeline (Backbone → Feature Enhancer → Object Decoder) to produce text-aligned object embeddings.

The **Fast Path** runs on the remaining $T$ inter-frames. It bypasses the Feature Enhancer and Object Decoder entirely, extracting only multi-scale backbone features. By conditioning these features on the keyframe object embeddings, it synthesizes masks at minimal computational expense.

Inter-frame association maintains the MinVIS paradigm, linking instances across time via cosine similarity matching of keyframe object embeddings without dedicated tracking networks.

### 3.2 Embedding Projection and Bifurcation

In perceptual terms, this stage translates high-level semantic identities identified by the slow path into the representational space of the fast visual pathway.

Object embeddings from the slow path reside in an abstract, text-aligned space, whereas the fast path operates on raw backbone feature maps. To bridge this domain gap, SegFS applies a 3-layer MLP with LayerNorm to project object embeddings into the backbone feature space, subsequently bifurcating them into two functional streams:

1. **Conditioning Tokens**:
   - The top-$K$ object queries are selected based on maximum classification similarity scores against text categories. This filtering discards low-confidence background queries typical of DETR models, preventing spurious distractors from degrading mask synthesis.
   - A learnable background token $t_{\text{bg}}$ is appended to form a set of $K+1$ tokens. Providing an explicit background sink prevents spatial cells from being falsely assigned to foreground instances.
   - The $K+1$ tokens pass through two alternating self-attention and cross-attention blocks, where the tokens serve as queries and the keyframe coarsest backbone feature map $P_5$ serves as key and value. This interaction aligns abstract semantic tokens with the physical layout of the scene.
2. **Mask Kernels**:
   - A parallel 3-layer MLP projects object embeddings into a convolutional kernel space. These act as dynamic filters that scan high-resolution feature maps via $1 \times 1$ convolutions to produce instance mask activations.

### 3.3 Fast Feature Aggregator: Lightweight High-Resolution Feature Map Generation

![Figure 3: Fast Feature Aggregator](/images/segfs/_page_6_Figure_2.jpeg)
*Figure 3: Detailed architecture of the Fast Feature Aggregator. Keyframe object embeddings are injected into the $P_5$ backbone feature map, followed by progressive upsampling and gated fusion to synthesize high-resolution mask features.*

Once an object's identity is recognized, tracking its trajectory across successive milliseconds requires only rapid perceptual alignment between current retinal inputs and stored mental anchors. The Fast Feature Aggregator serves as this fast perceptual engine, taking raw multi-scale backbone features and synthesizing sharp segmentation masks in three stages:

#### 1. Preprocessing and Channel Standardization

Different vision backbones (MobileNetV4, ResNet50, EfficientViT) output varying channel dimensions across their pyramids $F = \{P_2, P_3, P_4, P_5\}$. SegFS projects all levels to a standard channel dimension $D = 256$:

- Coarse maps ($P_4, P_5$): Processed with standard $1 \times 1$ convolutions to reduce channel dimensionality efficiently.
- Fine maps ($P_2, P_3$): Processed with mobile-friendly DSConvGN blocks (depthwise $3 \times 3$ conv, pointwise $1 \times 1$ conv, GroupNorm, and SiLU) to retain spatial detail.

Standardizing channels decouples the computational complexity of the aggregator from backbone capacity. Scaling the candidate object count $K$ from 10 to 100 alters aggregator computation by less than 0.16% (6.323 to 6.333 GFLOPs).

#### 2. Object Guidance at the Coarsest Resolution

![Figure 4: Object Guidance](/images/segfs/_page_6_Picture_4.jpeg)
*Figure 4: Object Guidance module. Keyframe object embeddings are matched with $P_5$ spatial cells to produce object-aware features, which are combined with raw backbone features via gated fusion.*

Object Guidance injects the $K+1$ object embeddings into the lowest-resolution feature map $P_5$.

Restricting semantic injection to $P_5$ balances computation and receptive field. At 1/32 original resolution, $P_5$ has minimal spatial cells while capturing the widest contextual field. Attempting cross-attention over dense grids like $P_2$ or $P_3$ would negate real-time efficiency. Injecting meaning at $P_5$ and propagating it upward provides the optimal trade-off.

The module operates in two steps:

- **Object Injection**: Computes multi-head cosine similarity between each spatial cell in $P_5$ and the $K+1$ tokens, scaled by a learnable temperature $\tau$:
  $$w_{i, k} = \frac{\exp(\tau \cdot \text{sim}(P_{5, i}, t_k))}{\sum_{j=1}^{K+1} \exp(\tau \cdot \text{sim}(P_{5, i}, t_j))}$$
  A weighted sum over tokens yields an object-aware feature map $I_5$. As training progresses, $\tau$ sharpens the softmax distribution, ensuring each spatial cell binds decisively to its matching instance embedding.
- **Gated Fusion**: Combines the semantic identity in $I_5$ with the current frame's raw spatial cues in $P_5$. After smoothing $I_5$ with a DSConvGN block, a delta-gated mechanism fuses the representations:
  $$\tilde{P}_5 = P_5 + \sigma(\text{GateConv}(P_5 \parallel I_5)) \odot \text{DSConvGN}(I_5)$$
  where $\parallel$ denotes concatenation, $\sigma$ is the sigmoid function, and $\text{GateConv}$ predicts a spatial blending mask. The gate channels strong semantic information into object interiors while preserving crisp boundary cues from $P_5$.

#### 3. Progressive Upsampling and Mask Generation

The enriched feature map $\tilde{P}_5$ is progressively upsampled to restore fine structural boundaries:

- $\tilde{P}_5$ is bilinearly upsampled by $2\times$, concatenated with backbone map $P_4$, and fused via a DSConvGN block.
- This process repeats sequentially through $P_3$ up to $P_2$ (1/4 resolution).

Each upsampling stage incorporates spatial detail stored in higher-level backbone features, sharpening mask silhouettes. The final high-resolution feature map at stride 4 is convolved with the $K$ mask kernels via $1 \times 1$ convolution, producing $N$ binary mask predictions.

### 3.4 Training Strategy: Learning Video Segmentation from Static Images

A remarkable design choice of SegFS is that it trains entirely on static image instance segmentation datasets, bypassing expensive video ground-truth annotations. Even when video datasets are used, frames are treated as independent still images.

Labeling pixel-accurate masks across continuous video frames is prohibitively expensive, whereas static image datasets offer vast category diversity. SegFS capitalizes on this by structuring training around the division of labor between slow semantic reasoning and fast mask reconstruction on a single image.

During training, an image passes sequentially through both pathways. The frozen slow network executes a forward pass to extract object queries, category logits, and bounding boxes. These queries are projected to the fast feature space, and the fast network predicts instance masks on the same image's backbone features.

To stabilize bipartite Hungarian matching, the authors implement a hybrid matching cost:

> "For the bipartite Hungarian matching with the ground-truth instance masks and categories, we construct a hybrid matching cost: we utilize the highly accurate category logits and bounding box predictions from the pre-trained slow network, while incorporating the mask predictions from the fast network. By anchoring the matching process with the slow network, we ensure stable and consistent ground-truth assignment."

Allowing an untrained fast network to guide bipartite matching would destabilize optimization: noisy early mask predictions lead to chaotic ground-truth assignments that corrupt gradient updates. Anchoring the matching cost with the mature classification and bounding box predictions of the slow network guarantees consistent assignment. With targets locked, the fast network focuses exclusively on mask refinement under standard Mask Loss (BCE) and DICE Loss supervision.

---

## 4. Experimental Results

### 4.1 Experimental Setup

The training corpus combines image instance segmentation datasets (COCO, LVIS, BDD), video datasets treated as still images (YouTubeVIS19, YouTubeVIS21, OVIS), referring segmentation datasets (RefCOCO, RefCOCO+, RefCOCOg, RVOS), and open-world datasets (UVO, SA-1B). Models are trained on 4 A100 GPUs with batch size 128 for 500,000 iterations. The slow model remains frozen, outputting 300 queries from which the top 50 are selected for Object Guidance.

Evaluations are conducted on seen benchmarks (YouTubeVIS19, OVIS) and zero-shot open-vocabulary benchmarks (BURST, LV-VIS) with inputs resized to a short side of 480 pixels and propagation interval $T=5$. On-device latency is measured on a Samsung Galaxy S25 Ultra via Qualcomm AI Hub.

### 4.2 Quantitative Results

SegFS was benchmarked across five distinct slow networks. Table 1 summarizes performance with MOBIUS-Mini-M against standard baselines:

- **Copy**: Directly copies keyframe masks to subsequent frames without computation (lower bound).
- **Reuse Objects**: Executes the Feature Enhancer on all frames while bypassing only the Object Decoder (upper bound).

| Method | YTVIS19 AP | OVIS AP | BURST HOTA | LV-VIS AP | Amortized FPS |
|---|:---:|:---:|:---:|:---:|:---:|
| Slow Network (Per Frame) | 48.7 | 23.6 | 19.7 | 16.7 | 8.7 |
| Copy Baseline | 20.1 | 5.1 | 11.4 | 10.4 | 51.8 |
| Reuse Objects | 42.1 | 15.0 | 15.2 | 15.8 | 12.7 |
| MPVSS | 39.8 | 13.9 | 13.8 | 14.8 | 16.5 |
| SegFS | 41.6 | 14.1 | 14.7 | 15.2 | 38.2 |

Two observations stand out:

First, SegFS matches the accuracy of Reuse Objects within 1.0 AP across datasets (-0.5 AP on YTVIS19, -0.9 AP on OVIS), confirming that raw backbone features inherently retain sufficient localization cues without requiring full per-frame vision-language enhancement.

Second, SegFS achieves 38.2 amortized FPS—a $3\times$ speedup over Reuse Objects—making it the only MOBIUS configuration exceeding 30 FPS real-time throughput on mobile hardware. With MobileNetV4-CM, fast path latency drops to 8.3 ms on a Galaxy S25 Ultra, representing a 14x to 70x reduction compared to full MOBIUS-Mini-M (115.7 ms) and GLEE-Lite (587.9 ms).

Pairing SegFS with TROY-VIS achieves the highest overall accuracy across configurations, demonstrating synergistic compatibility with backbones that emphasize strong early representations.

### 4.3 Comparison with Optical Flow Methods

Optical flow alternatives (e.g., RAFT, LiteFlowNet2) warp keyframe masks using estimated inter-frame displacement fields. These methods exhibit substantial accuracy degradation compared to SegFS (LiteFlowNet2: 28.6 AP, RAFT: 28.9 AP vs. SegFS: 41.6 AP on YTVIS19).

While MPVSS improves upon standard flow by conditioning motion fields on instance embeddings (39.8 AP), geometric warping inherently fails when objects enter the scene midway or become temporarily occluded, as there is no prior mask to warp. By contrast, SegFS synthesizes masks afresh on each frame by conditioning current backbone features on object embeddings, reliably discovering newly visible instances.

### 4.4 Ablation Study

![Figure 5: K sensitivity and T-FPS tradeoff](/images/segfs/_page_13_Figure_2.jpeg)
*Figure 5: Ablation plots showing AP sensitivity to query count $K$ (left) and trade-offs between propagation interval $T$, AP, and amortized FPS (right).*

- **Propagation Interval $T$**: Semantic embedding propagation in SegFS exhibits graceful degradation as $T$ increases, whereas optical flow warping degrades sharply due to compounding geometric drift.
- **Query Count $K$**: Performance peaks at $K=50$. Exceeding 50 queries introduces noisy, low-confidence background queries that degrade mask synthesis.
- **Module Ablation**: Object Injection and attention-based query adaptation contribute the largest accuracy gains, confirming that direct semantic conditioning is critical for fast path segmentation.

### 4.5 Qualitative Results

![Figure 6: Qualitative Results](/images/segfs/_page_13_Figure_4.jpeg)
*Figure 6: Qualitative segmentation comparisons on YouTubeVIS19 between per-frame MOBIUS, MPVSS, and SegFS.*

Qualitative comparisons highlight the resilience of SegFS. In sequences where optical flow warping (MPVSS) fails to capture newly appearing objects entering from image boundaries, SegFS accurately localizes and tracks instances across severe occlusions and abrupt motions, matching the visual quality of full per-frame slow networks.

---

## 5. Conclusion and Key Takeaways

SegFS demonstrates that real-time Open-Vocabulary Video Instance Segmentation does not require running heavy vision-language cross-attention on every frame. Multi-scale visual backbone features already preserve fine-grained spatial structure; projecting compact semantic embeddings from sparse keyframes into this feature space is sufficient to sustain high mask fidelity across video sequences.

This architecture offers two practical advantages:

1. **Plug-and-Play Compatibility**: Any pretrained object-centric model (GLEE, MOBIUS, TROY-VIS) can serve as the frozen slow path, ensuring immediate compatibility with future architectural advancements.
2. **Standardized Fast Path Efficiency**: Channel standardization bounds aggregator computation regardless of backbone size, isolating latency variations entirely to backbone forward passes.

By decomposing multimodal comprehension and dense mask synthesis along the temporal dimension, SegFS overturns the convention that all video frames require equal compute. Emulating the visual division of labor between deep semantic categorization and rapid spatial tracking brings true open-vocabulary real-time segmentation to edge devices.
