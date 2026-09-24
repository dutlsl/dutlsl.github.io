---
title: "[CVPR 2026] AutoGaze: Efficient and Scalable Video Understanding via Autoregressive Gazing"
date: 2026-09-17T18:18:30+09:00
draft: false
math: true
tags: ["Paper Review", "Video Understanding", "Efficient Inference", "Token Reduction", "MLLM", "ViT", "Reinforcement Learning", "CVPR 2026"]
categories: ["Paper Review"]
summary: "Introducing AutoGaze, a lightweight 3M-parameter autoregressive gazing module placed before the ViT backbone that reduces visual tokens by up to 100x and enables MLLMs to process 1024-frame 4K videos without out-of-memory errors."
cover:
  image: "/images/autogaze/_page_0_Figure_5.jpeg"
  alt: "AutoGaze overview figure"
---

> Reference
> - Shi, B., Fu, S., Lian, L. et al. "Attend Before Attention: Efficient and Scalable Video Understanding via Autoregressive Gazing." CVPR 2026.
> - Project Page: https://autogaze.github.io/

---

## 1. One-Sentence Summary

AutoGaze is a lightweight 3M-parameter neural module placed before the Vision Transformer backbone that prunes spatiotemporal visual redundancy at the raw pixel level, reducing visual tokens by up to $100\times$ and allowing multimodal large language models to seamlessly process 1024-frame 4K-resolution videos without running out of GPU memory.

---

## 2. Research Background and Motivation

### 2.1 Problem Definition

When humans observe dynamic visual environments, they do not allocate equal cognitive resources to every pixel or background detail. Human gaze fixates swiftly on moving subjects, zooms in on critical fine-grained features, and skips static, predictable backgrounds.

In stark contrast, conventional Multimodal Large Language Models (MLLMs) coupled with Vision Transformer (ViT) backbones treat every pixel and every frame uniformly. Despite the overwhelming temporal and spatial redundancy inherent in video data, standard frameworks ingest every single raw patch into the attention pipeline without prior filtering.

![Figure 1: AutoGaze Overview](/images/autogaze/_page_0_Figure_5.jpeg)
*Figure 1: AutoGaze discards redundant patches before ViT encoding, enabling MLLMs to process 1024-frame 4K videos without memory exhaustion. Prior methods only prune tokens after ViT, leaving the ViT itself as an intractable bottleneck.*

This paradigm fails catastrophically when scaled to high-resolution, long-duration videos. Critical real-world applications such as autonomous driving, surveillance, and sports analytics require ingesting minutes of 4K footage. Ingesting these videos generates hundreds of thousands of visual patches, leading to prohibitive computation costs and out-of-memory (OOM) failures.

### 2.2 Limitations of Existing Methods

Prior research attempting to reduce video token counts predominantly applies pruning or merging within the ViT layers or between the ViT and LLM stages. Prominent approaches such as STORM, FastVID, LongVU, and VideoChat-Flash calculate token saliency after tokens pass through the vision backbone.

However, this post-hoc compression suffers from a fundamental architectural flaw: regardless of how aggressively tokens are pruned before the LLM, the preceding ViT must still process every frame and patch across the entire video. Consequently, as resolution and frame count increase, the ViT itself remains a massive computational bottleneck, hitting memory limits before the LLM can even receive the tokens.

The "before ViT" stage introduced by AutoGaze is not an offline data preprocessing step. Instead, it operates as an ultra-lightweight neural module embedded directly within the online inference pipeline immediately preceding the heavy ViT backbone. Before raw video frames enter the vision transformer, the 3M-parameter AutoGaze model rapidly scans the visual stream, identifies spatiotemporal redundancy, and extracts only the minimal subset of raw pixel patches required to reconstruct the video within a calibrated error bound. Because the ViT only processes this tiny fraction of patches, memory bottlenecks are eliminated at the earliest stage.

### 2.3 Main Contributions

The primary contributions of this work are summarized as follows:

- Proposes AutoGaze, a lightweight 3M-parameter autoregressive neural module placed strictly before the ViT backbone to filter out redundant raw patches.
- Achieves up to $100\times$ token reduction, accelerating ViT and MLLM latencies by up to $19\times$ and $10\times$ respectively, and enabling MLLMs to process 1024-frame 4K videos without GPU memory exhaustion.
- Introduces HLVid, a rigorous high-resolution long-form video QA benchmark, demonstrating significant empirical improvements over leading open-source and proprietary MLLMs.

---

## 3. Proposed Framework

### 3.1 Problem Formulation: Video Reconstruction with Minimal Patches

A video consists of a sequence of frames, each partitioned into small grid patches. Standard models feed all patches into the ViT, incurring massive computational overhead.

AutoGaze does not generate new frames; rather, it acts as an intelligent gaze controller that selects the minimal set of patch indices necessary to preserve complete visual semantics. The objective criterion governing this selection is video reconstructability: if a frozen video reconstructor Recon can reconstruct the full video within an allowable error threshold from only the selected subset of patches, the omitted patches represent redundant information that can be safely discarded.

Formally, given a video $X^{1:T}$ with $T$ frames, let $p_k^t$ denote the index of the $k$-th patch selected in frame $t$, and let $N^t$ be the total number of patches selected in that frame. AutoGaze seeks the smallest patch subset $P$ that minimizes the reconstruction loss $L$:

$$
\min_{P} L(X^{1:T}, \text{Recon}(X^1[p_1^1], \dots, X^T[p_{N^T}^T]))
$$

Here, Recon is a VideoMAE-based reconstruction network, and $L$ represents a weighted sum of pixel-level mean squared error and perceptual loss.

![Figure 2: AutoGaze Visualizations](/images/autogaze/_page_1_Figure_0.jpeg)
*Figure 2: Multi-scale patches selected by AutoGaze and their corresponding reconstructions. AutoGaze focuses gaze on dynamic foreground objects, eliminates static background redundancy, and dynamically assigns multi-scale patches based on local visual complexity.*

### 3.2 Model Architecture

![Figure 3: AutoGaze Architecture and Training Pipeline](/images/autogaze/_page_3_Figure_0.jpeg)
*Figure 3: Overview of AutoGaze architecture and two-stage training pipeline. Consisting of a convolutional encoder and an autoregressive transformer decoder, AutoGaze is trained via Stage 1 Next-Token Prediction pre-training and Stage 2 GRPO reinforcement learning post-training.*

AutoGaze comprises two simple components: a convolutional encoder and an autoregressive transformer decoder, totaling only 3 million parameters. Its patch selection mechanism operates via four key designs:

#### 1. Autoregressive Gazing
AutoGaze alternates between extracting visual features and predicting patch coordinates. When the first frame enters, the convolutional encoder extracts a compact spatial feature map. Conditioned on this feature map, the transformer decoder sequentially selects the most informative patch indices.

Starting from the second frame, the decoder conditions on both the previous frame's feature map and the history of previously selected patch indices. This design naturally skips static backgrounds:
- The reconstruction network Recon is conditioned on patches accumulated from both past and current frames.
- Static background regions remain identical across consecutive frames and can be accurately reconstructed using patches selected in preceding frames.
- Selecting a static background patch again yields negligible reduction in reconstruction loss.
- Conversely, newly appearing or moving objects cannot be reconstructed from past history; omitting them causes reconstruction loss to spike.
- Because AutoGaze is trained to maximize reconstruction loss reduction, it naturally bypasses static background regions and prioritizes dynamic changes.

#### 2. Automatic Stopping
Enforcing a fixed patch quota per frame forces models to select redundant patches during static scenes. To prevent this, AutoGaze equips the decoder with an auxiliary regression head that estimates reconstruction loss in real time.

This loss prediction head is not a separate neural network, but a lightweight linear layer built on top of the transformer decoder's hidden states. At each decoding step, it estimates the remaining video reconstruction error as a scalar value. Once the predicted error drops below a user-defined threshold $\epsilon$, AutoGaze immediately halts patch selection for the current frame and advances to the next frame.

#### 3. Multi-scale Gazing
Visual scenes exhibit varying information density. Importantly, patch scaling does not subdivide the convolutional feature map; the convolutional encoder acts solely as a global observer, while the actual multi-scale crops are extracted from the raw video frames.

AutoGaze's vocabulary contains pre-defined patch indices corresponding to four distinct spatial scales ($14 \times 14$, $28 \times 28$, $56 \times 56$, and $112 \times 112$ within a $224 \times 224$ window). When the decoder outputs a patch token, a crop of the corresponding scale and coordinate is sliced from the raw video frame and passed to the downstream ViT.

![Figure 4: Correlation with Optical Flow](/images/autogaze/_page_4_Figure_0.jpeg)
*Figure 4: AutoGaze prioritizes patches with high optical flow across all scales, effectively capturing fast-moving dynamics.*

![Figure 5: Correlation with Detail Complexity](/images/autogaze/_page_4_Figure_7.jpeg)
*Figure 5: Correlation between patch detailedness and gaze scale. AutoGaze allocates fine-grained high-resolution patches to intricate textures while using coarser patches for uniform regions.*

As shown above, AutoGaze frequently selects patches with substantial optical flow and assigns fine-grained scales to regions with high Laplacian variance, while covering broad uniform backgrounds with large, coarse patches.

#### 4. Multi-token Prediction
Standard autoregressive decoding requires sequential step-by-step forward passes, creating latency bottlenecks. AutoGaze incorporates Multi-Token Prediction (MTP) inspired by Meta's architecture.

From a single forward pass of the shared transformer trunk, 10 independent parallel heads simultaneously predict the next 10 patch tokens and their respective loss estimates. These 10 tokens are appended to the input sequence in blocks, reducing decoder trunk invocations by $10\times$ while preserving causal autoregressive context.

### 3.3 Training Pipeline: Offline Greedy Labels and Two-Stage Learning

AutoGaze requires zero human annotations, training in a fully self-supervised manner using the frozen reconstructor Recon as an oracle in a two-stage pipeline.

| Category | Stage 1: Next-Token Prediction Pre-training | Stage 2: GRPO Reinforcement Learning Post-training |
|---|---|---|
| Objective | Imitate offline greedy oracle gaze trajectories | Learn globally optimal video reconstruction policy |
| Label Source | Exhaustive greedy search trajectories via Recon | Zero fixed labels, fully autonomous policy rollouts |
| Loss / Reward | Cross-entropy loss + reconstruction loss $\ell_2$ regression | Negative cumulative discounted reconstruction error |
| Outcome | Reaches myopic step-by-step local optima | Overcomes myopic bounds, reaches global efficiency |

#### Stage 1 Pre-training: Imitating Offline Greedy Trajectories

The training data curation and Stage 1 supervision pipeline are detailed in Section 3.2 of the paper:

- Dataset Scale: The authors collect 800K raw videos spanning egocentric, exocentric, natural, and text-rich domains, sampled at 16 frames and $224 \times 224$ resolution. From this pool, 250K videos are processed to construct offline supervision trajectories.
- Combinatorial Intractability: Finding the globally optimal patch subset that minimizes Equation (2) across 16 frames is an NP-hard combinatorial optimization problem with an exponentially exploding search space.
- Greedy Oracle Search: To make supervision tractable, the authors use the frozen Recon network as an oracle. Starting from the first patch of the first frame, exhaustive greedy search identifies the patch yielding the greatest reduction in video reconstruction loss. This process repeats sequentially until reaching the frame's patch budget, then advances to subsequent frames.
- Pseudo-Ground-Truth Labels: The resulting sequence of patch indices monotonically decreases reconstruction loss at each step, representing an optimal approximation path that serves as the pseudo-ground-truth sequence for imitation learning.
- Pre-training Mechanism: Because running exhaustive greedy search with Recon during real-time inference is computationally impossible, lightweight AutoGaze is trained via next-token prediction to predict this oracle trajectory in a single forward pass directly from raw video features.

#### Stage 2 Post-training: Global Optimization via GRPO

- Limitations of Greedy Labels: Stage 1 greedy labels suffer from myopic decision-making, as greedily minimizing step-wise error traps the sequence in local optima.
- Label-free Exploration: Stage 2 completely discards fixed labels. Using Group Relative Policy Optimization (GRPO), AutoGaze autonomously samples complete gazing trajectories across the video.
- Cumulative Reconstruction Reward: Once a trajectory finishes, the frozen Recon environment evaluates the reconstructed video and returns the negative reconstruction error as a reward signal, enabling AutoGaze to discover globally optimal gazing policies.

#### Standalone Evaluation Methodology of AutoGaze

AutoGaze's standalone capabilities as a visual token selector are evaluated independently of downstream language models through two rigorous axes:

- Reconstruction Loss vs. Gazing Ratio: On unseen test videos, patches selected by AutoGaze are reconstructed using Recon. The gazing ratio required to achieve a target reconstruction loss (0.7) is measured and benchmarked against heuristic baselines (Random, RGB-Difference, Optical Flow), verifying AutoGaze's superior compression efficiency.
- Loss Prediction Calibration: The estimated reconstruction loss from AutoGaze's auxiliary head is compared against the ground-truth loss computed by Recon, ensuring the model reliably halts patch selection without relying on external models.

### 3.4 Downstream Integration: From Spatiotemporal Tiling to ViT Input

AutoGaze operates on local windows of 16 frames at $224 \times 224$ resolution. Ingesting massive 1000-frame 4K videos into downstream MLLMs proceeds through a 6-step pipeline:

1. Spatiotemporal Tiling: Large videos are partitioned into 16-frame $\times 224 \times 224$ spatiotemporal tiles across spatial and temporal dimensions.
2. Per-Tile Gazing: AutoGaze runs independently on each tile, allocating fine $14 \times 14$ patches to intricate details and coarse $112 \times 112$ patches to uniform backgrounds.
3. Global Coordinate Remapping: The local patch coordinates selected within each tile are projected back onto the global 4K canvas coordinate space.
4. Raw Pixel Cropping: Raw pixel patches are sliced directly from the original 4K video at the determined global coordinates. Depending on video resolution and FPS, 4x to 100x patches are discarded, with 30-FPS 4K videos filtering out up to 99% of background redundancy.
5. Downstream ViT Encoding: The extracted multi-scale pixel patches are passed to the downstream ViT. Positional embeddings are interpolated to handle varying patch sizes, and 16-frame collections are structured for video ViT encoding.
6. MLLM Pipelining and Cascading Speedup:
   A video MLLM connects a vision encoder (ViT) and a large language model (LLM) in a serial pipeline: the ViT functions as the visual eyes converting raw pixel patches into visual token vectors, and the LLM acts as the reasoning brain processing visual and text prompt tokens to generate natural language answers. Because the models are chained sequentially, AutoGaze's 99% patch reduction triggers a cascading acceleration across both stages:
   - ViT Acceleration: The input patch matrix ingested by the ViT shrinks dramatically, accelerating ViT latency by up to $19\times$ and completely preventing GPU memory exhaustion.
   - LLM Acceleration: Because the ViT processes 99% fewer pixel patches, the total count of visual token vectors output by the ViT into the LLM also drops by 99%. Consequently, the LLM's attention context window is shortened, accelerating LLM inference latency by up to $10\times$.

### 3.5 HLVid Benchmark

Existing video benchmarks focus primarily on temporal duration while neglecting high-resolution demands. The authors introduce HLVid, featuring 268 high-resolution long-form video QA pairs spanning up to 5-minute 4K videos. Successfully answering HLVid queries strictly requires resolving visual details at $1000 \times 2000$ pixel resolutions.

---

## 4. Experimental Results

### 4.1 Out-of-Distribution Generalization

![Figure 6: OOD Generalization](/images/autogaze/_page_5_Figure_0.jpeg)
*Figure 6: Generalization of AutoGaze to out-of-distribution videos. (a) Robustly tracking moving targets across CCTV footage, robot manipulation, and rapid object identity morphing. (b) Consistent gaze trajectories despite severe visual style and illumination shifts via TokenFlow.*

AutoGaze demonstrates strong zero-shot generalization to unseen semantics and visual styles. It tracks dynamic subjects across CCTV footage, robotic arms, and synthetic clips where human subjects constantly morph into gorillas or robots. When subjected to severe lighting and texture distortions via TokenFlow, AutoGaze maintains invariant gaze tracks on moving targets.

### 4.2 ViT and MLLM Efficiency Gains

![Figure 7: Gazing Ratio Analysis](/images/autogaze/_page_5_Figure_2.jpeg)
*Figure 7: Required gazing ratio across video types. Higher FPS and resolution yield lower gazing ratios to reach identical reconstruction fidelity. 30-FPS 4K videos require only $\sim 1\%$ of patches.*

As frame rate and resolution increase, spatiotemporal redundancy escalates dramatically. Targeting a reconstruction loss threshold of 0.7, typical videos require $4\times$ to $100\times$ fewer patches with less than 0.5% degradation in downstream performance. For 30-FPS 4K videos, gazing at only $\sim 1\%$ of patches is sufficient.

![Figure 8: Latency Comparison](/images/autogaze/_page_6_Figure_0.jpeg)
*Figure 8: Hardware latency improvements on ViT and MLLM. At reconstruction loss 0.7, AutoGaze speeds up ViT execution by up to $19\times$ and overall MLLM pipeline latency by up to $10\times$.*

Wall-clock hardware profiling reveals that at reconstruction loss 0.7, ViT latency is reduced by up to $19\times$ and end-to-end MLLM latency by up to $10\times$. Unmodified ViT baselines crash with OOM errors at 30 FPS and 896 resolution, whereas AutoGaze processes high-resolution video streams effortlessly.

### 4.3 Scaling MLLMs to 1K Frames and 4K Resolution

![Figure 9: Scaling Results](/images/autogaze/_page_6_Figure_2.jpeg)
*Figure 9: Scaling properties over video frames and resolution. While standard baselines fail beyond 256 frames due to OOM, AutoGaze scales cleanly to 1024 frames and 4K resolution, delivering massive accuracy gains on HLVid.*

Equipping NVILA-8B-Video with AutoGaze enables scaling test-time inputs to 1024 frames and 4K resolution, yielding consistent improvements across benchmarks:

| Benchmark | NVILA-8B Baseline | NVILA-8B + AutoGaze | Absolute Gain |
|---|:---:|:---:|:---:|
| VideoMME (w/o subtitles) | 64.2% | 67.0% | +2.8%p |
| VideoMME (w/ subtitles) | 70.0% | 71.8% | +1.8%p |
| LongVideoBench (val) | 57.7% | 61.0% | +3.3%p |
| HLVid (test) | 42.5% | 52.6% | +10.1%p |

On HLVid, which strictly requires resolving fine spatial details over extended durations, AutoGaze boosts performance from 42.5% to 52.6% (+10.1%p), surpassing GPT-4o (49.3%) and Qwen2.5-VL-7B (48.1%).

### 4.4 Comparison with Token Pruning Baselines

Existing token reduction methods (ToMe, VisionZip, FastV, STORM, LongVU) accelerate LLM inference by $3.7\times$ to $13.4\times$, but leave ViT latency fixed at $\sim 2.2$ seconds. In contrast, AutoGaze slashes ViT latency by $4\times$ to 0.55s while keeping LLM latency at 0.10s, maintaining full downstream QA accuracy.

![Figure 10: Comparison with Gazing Baselines](/images/autogaze/_page_6_Figure_4.jpeg)
*Figure 10: Gaze method comparison. AutoGaze achieves identical reconstruction quality with substantially lower patch ratios compared to heuristic alternatives.*

Comparing patch selection algorithms, AutoGaze reaches reconstruction loss 1.0 using only 5% of patches, whereas Random Gaze requires 15%. Simple frame differencing and optical flow baselines perform worse than random selection due to severe fixation on initial padding boundaries.

### 4.5 Ablation Study

Ablations highlight the essential synergy of the two-stage training pipeline. To achieve a reconstruction loss of 0.7, an untrained model requires a 0.263 gazing ratio, Stage 2 RL alone requires 0.209, and Stage 1 NTP pre-training alone achieves 0.102. Combining both stages yields the lowest ratio of 0.094. Furthermore, multi-scale gazing reduces required patch counts from 0.220 to 0.094 compared to single-scale gazing, improving inference speed by $2.3\times$.

---

## 5. Conclusion and Key Takeaways

The fundamental insight of AutoGaze is shifting visual attention from inside the vision transformer to immediately before it. Conventional multimodal frameworks allow vision backbones to process massive spatial and temporal pixel arrays blindly, relying on post-hoc pruning to manage downstream LLM token counts. AutoGaze flips this paradigm, proving that stripping visual redundancy prior to ViT encoding using a tiny 3M-parameter model is a fundamentally superior and scalable architecture.

This front-end pruning enables MLLMs to process 1024-frame 4K videos that were previously impossible due to GPU memory constraints. Combining real-time automatic stopping with adaptive multi-scale patch allocation faithfully mirrors the human visual system's dynamic allocation of focus. Demonstrating an impressive +10.1%p gain on the 4K HLVid benchmark, AutoGaze establishes a compelling foundation for real-time long-form video analysis and edge-device multimodal intelligence.
