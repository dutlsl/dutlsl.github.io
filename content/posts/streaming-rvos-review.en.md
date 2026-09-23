---
title: "[CVPR 2026] StreamingRVOS: Towards Streaming Referring Video Segmentation via Large Language Model"
date: 2026-09-23T19:58:18+09:00
draft: false
math: true
tags: ["Paper Review", "Referring Video Object Segmentation", "RVOS", "MLLM", "Streaming", "SAM2", "CVPR 2026"]
categories: ["Paper Review"]
summary: "Breaking away from the offline paradigm of processing entire videos post hoc, StreamingRVOS introduces Semantic Embedding Recycling and Online Mask Consistency Perception to build an efficient, real-time streaming RVOS pipeline without extra parameters."
cover:
  image: "/images/streaming-rvos/_page_2_Figure_0.jpeg"
  alt: "StreamingRVOS Framework Overview"
---

> Reference Paper
> - Zhang, W., Yang, K., An, X., Li, Q., Feng, Z., Yang, W., Deng, J. "Towards Streaming Referring Video Segmentation via Large Language Model." CVPR 2026.
> - Code: https://github.com/wkzhang636/StreamingRVOS

![Figure 1: StreamingRVOS Framework Overview](/images/streaming-rvos/_page_2_Figure_0.jpeg)
*Figure 1: Overall framework of StreamingRVOS. Without introducing additional modules, Semantic Embedding Recycling propagates semantic context across frames, while Online Mask Consistency Perception balances inference performance and efficiency. Starting from the second frame, SAM2 segments the foreground using SER prompts, and the MLLM is re-invoked only when OMCP detects semantic ambiguity.*

---

## 1. One-Sentence Summary

StreamingRVOS seamlessly extends image-level referring segmentation to video streaming without adding any parameters or preprocessing, pairing Semantic Embedding Recycling to propagate temporal context via recycled tokens with Online Mask Consistency Perception to dynamically wake the MLLM only when masks drift, simultaneously reaching 7 FPS real-time streaming throughput and state-of-the-art accuracy on an A800 GPU.

---

## 2. Research Background and Motivation

### 2.1 Problem Definition

Consider a radio commentator broadcasting live descriptions of scenic landscapes from a moving cross-country bus. As the bus winds through mountain villages, the commentator must narrate the scenery to listeners in real time, describing how a stone cottage appears on the left with a red mailbox standing right before it. Imagine if this commentator could only prepare the broadcast by returning home after the journey concludes, rewinding through the complete recorded footage, and only then noting what appeared at the three-minute mark. Such an approach completely defeats the very essence of live commentary.

Referring Video Object Segmentation, abbreviated as RVOS, is the task of segmenting target objects referred to by natural language expressions across all frames of a video at the pixel level. It serves as a foundational capability for real-time applications ranging from autonomous driving and robotic manipulation to augmented reality.

Yet existing MLLM-based RVOS methodologies have remained confined to an offline paradigm akin to a commentator who can only review recorded footage after the trip ends. They require access to the entire video upfront, run complex sampling strategies to pick a handful of sparse keyframes, execute image-level segmentation via an MLLM, and propagate masks across the remaining frames using auxiliary tracking models like SAM2. In the face of continuous real-time video streams, this offline pipeline breaks down entirely.

### 2.2 Limitations of Existing Methods: The Offline Tape-Reviewer Bottleneck

![Figure 2: Pipeline Comparison with Previous Approaches](/images/streaming-rvos/_page_0_Figure_9.jpeg)
*Figure 2: Comparison between previous RVOS pipelines and StreamingRVOS. While prior approaches require the entire video upfront and separate sparse frame sampling, image segmentation, and mask propagation into disconnected stages, StreamingRVOS directly ingests video streams to output continuous segmentation masks, adaptively skipping MLLM calls starting from the second frame.*

Returning to the radio commentary analogy, existing offline RVOS pipelines suffer from three fundamental bottlenecks.

First, preprocessing constraints. All existing methods depend on meticulously engineered video frame sampling. This is equivalent to requiring the commentator to predetermine the exact time intervals and locations to inspect before embarking on the journey. Such sampling schemes introduce substantial computational overhead and preprocessing complexity.

Second, optimization fragmentation. Frame sampling, image segmentation, and mask propagation operate as isolated modules, making end-to-end unified optimization impossible. It is as if the commentator were trained separately on scene selection, object description, and transitional phrasing by different instructors. Errors incurred in earlier stages propagate unchecked through subsequent stages without any recourse for correction.

Third, the inability to operate in real time. Because these architectures presuppose access to the full video sequence, they cannot function in true online streaming environments where frames arrive sequentially. Even when some methods attempt frame-by-frame processing, invoking the full MLLM on every single frame slows throughput to a crawl, falling far short of real-time requirements.

These three bottlenecks cannot be resolved simply by making individual models smaller or faster. The root of the problem lies in the structural paradigm itself. Giving a commentator a faster rewind button on a tape deck does not transform recorded playback into a live broadcast.

### 2.3 Main Contributions

The primary contributions of this work are threefold:

- The authors propose Semantic Embedding Recycling, a mechanism that recycles the `[SEG]` token generated by the MLLM in the previous frame as a temporal context prompt `[INFO]`, extending image-level segmentation to video streams without introducing any additional parameters.
- The authors design Online Mask Consistency Perception, an adaptive triggering strategy that monitors predicted mask confidence and inter-frame mask consistency in real time, accompanied by a dedicated streaming training pipeline that bridges the gap between training and streaming inference.
- Extensive experiments across multiple benchmarks demonstrate that the 1B variant improves upon Sa2VA by 19.2% on the MeViS dataset while achieving state-of-the-art accuracy across RVOS benchmarks, maintaining an average streaming inference speed of 7 FPS on a single A800 GPU.

---

## 3. Proposed Framework: StreamingRVOS

### 3.1 Foundational Architecture of MLLM-based Segmentation

To grasp the core mechanism of StreamingRVOS, one must first examine how modern MLLM-based segmentation architectures operate. Pioneered by LISA, these models take an input image $I$ and a linguistic expression $R$, generating a specialized segmentation token `[SEG]`. This `[SEG]` token acts as a concentrated semantic seed embodying the multimodal comprehension of the scene. A Segmentation Assistant such as SAM2 then ingests this token alongside the visual features to delineate pixel-accurate masks on the image.

$$\text{[SEG]} = \text{MLLM}(I, R), \quad \text{Mask} = \text{SA}(I, \text{[SEG]})$$

In the radio commentator analogy, the MLLM serves as the commentator's cognitive reasoning, identifying the red mailbox as the focal entity, the `[SEG]` token represents the concise handwritten note containing that insight, and SAM2 functions as the camera operator who takes that note and sharply frames the exact contours of the object.

The breakdown occurs when extending this setup to video. Existing approaches take the full video sequence $I_{1:T}$, sample a sparse subset $I_{1:N}$ for the MLLM, and leave the remaining frames to be propagated blindly by the segmentation assistant. This is tantamount to having the commentator glance at three photographs from a trip and instructing the camera operator to guess the rest.

### 3.2 Semantic Embedding Recycling: The Commentator Reusing Essential Notes

StreamingRVOS tackles this challenge from a fundamentally different perspective by processing video frames sequentially in a streaming fashion as they arrive.

However, processing every frame in isolation strips away all temporal context. If the commentator retains no memory of what came before, treating each second as an unfamiliar landscape, listeners become confused as to whether the red mailbox seen just now is the same one passed moments ago. In RVOS, this manifests as temporal forgetting, characterized by abrupt jumps and inconsistent masks between frames.

Conversely, concatenating all raw visual features from preceding frames causes memory footprints to balloon rapidly, severely undermining both training and inference throughput. This is like a commentator whose notebook becomes so weighed down with exhaustive transcripts of past scenery that pausing to search through it stalls the live broadcast.

Semantic Embedding Recycling, or SER, resolves this dilemma with remarkable simplicity. Rather than discarding the `[SEG]` token produced by the MLLM for the previous frame, the model relabels it as `[INFO]` and feeds it directly into the input of the current frame.

The input to the MLLM is formulated based on frame index $i$:

$$\text{Input} = \begin{cases} I_i + R, & \text{if } i = 1 \\ I_i + R + \text{[INFO]}, & \text{otherwise} \end{cases}$$

For the initial frame, only the current image $I_1$ and language expression $R$ are provided, with `[INFO]` acting as an empty placeholder. From the second frame onward, the input incorporates current image $I_i$, text $R$, and the recycled `[SEG]` token packaged as `[INFO]`.

The elegance of this design stems from the intrinsic nature of the `[SEG]` token. It is a highly condensed vector compressed within the hidden space of the language model, capturing object spatial localization, geometry, and contextual grounding. Passing this single token forward enables the MLLM to understand that the target was positioned at this location with this appearance in the preceding moment, effortlessly preserving temporal continuity.

In the commentator analogy, the commentator slips the concise summary card written during the previous stretch into a pocket, pulling it out as the next bend appears to smoothly explain how the route now passes a blue-roofed shop just beyond that earlier stone cottage. Rather than carrying the entire journey in memory, the single summary card bridges the temporal flow seamlessly.

### 3.3 Online Mask Consistency Perception: An Adaptive Alarm System

While SER maintains temporal context, invoking the full MLLM on every single frame remains computationally prohibitive. Adjacent frames in a video sequence typically share immense visual redundancy. A commentator who undertakes an exhaustive, ground-up analysis every single second across a five-second stretch of nearly identical meadow wastes tremendous energy.

Online Mask Consistency Perception, abbreviated as OMCP, is an adaptive perception mechanism designed to eliminate this inefficiency by continuously monitoring two real-time signals.

The first signal is current frame mask prediction confidence $P_i$. StreamingRVOS repurposes the predicted IoU natively produced by SAM2 during mask generation. Without adding any new modules or parameters, the framework repurposes the internal quality estimate of the Segmentation Assistant. If $P_i$ drops, the quality of the current mask is suspect, much like a commentator feeling a pang of uncertainty about a rushed description.

The second signal is inter-frame mask consistency $P_c$, computed as the intersection over union between current mask $M_i$ and preceding mask $M_{i-1}$:

$$P_c = \frac{|M_i \cap M_{i-1}|}{|M_i \cup M_{i-1}|}$$

A sharp drop in this value indicates that the target object has abruptly vanished or jumped across the scene, signaling potential semantic confusion. This corresponds to the commentator suddenly noticing that the view has transformed unexpectedly.

The overall decision is determined by conjunctively evaluating both criteria against predefined thresholds:

$$\text{OMCP} = (P_i > \tau_1) \land (P_c > \tau_2)$$

When both conditions hold true—meaning the mask exhibits high internal quality and remains continuous with the previous frame—the pipeline skips MLLM invocation entirely, relying on SAM2 and the recycled `[SEG]` token to generate the mask. The commentator notes that the scenery is unfolding steadily and continues with brief notes.

Conversely, if either metric falls below its threshold, semantic ambiguity is detected and the system wakes the MLLM. Crucially, the MLLM is not invoked blindly; an additional instruction explicitly prompts the model to attend to temporal context. The commentator pauses to re-examine the landscape with focused attention. Ingesting the current image, referring expression, temporal context `[INFO]`, and prompt, the MLLM generates a freshly calibrated `[SEG]` token, allowing SAM2 to output an immediately corrected mask.

### 3.4 Streaming Training Pipeline: Bridging the Training-Inference Gap

The primary reason conventional RVOS frameworks struggled with optimization lies in the disconnect between training regimes and inference settings. Standard models train on isolated images to predict masks, but inference demands continuous mask propagation across long temporal sequences. Training a commentator purely on still photo descriptions produces poor handling of real-time scene transitions and past context.

To bridge this gap, StreamingRVOS introduces a two-stage streaming training pipeline.

Stage 1 is Joint Optimization, where the model is jointly trained on mixed image and video datasets. Because the initial frame in streaming segmentation is functionally identical to image referring segmentation, this stage solidifies foundational segmentation competence, giving the commentator strong baseline skills in identifying referred entities in single images.

Stage 2 is Video Semantic Fine-tuning. This stage embeds the OMCP mechanism directly into the training loop, establishing an active learning strategy that replicates true streaming inference. Conventional models train under the idealized condition where the MLLM is queried on every single frame. In real-world streaming inference, however, MLLM calls are bypassed on most frames to preserve throughput, waking only when masks deteriorate. A model trained exclusively under perpetual MLLM supervision cannot cope when left to propagate masks autonomously via SAM2 before abruptly having to recover from severe drift. Stage 2 simulates this exact operational reality.

The video data handling during this stage is pivotal. From long video sequences in the dataset, the pipeline extracts continuous clips of 5 frames. Rather than batching these 5 frames into a single tensor as offline models do, StreamingRVOS separates the clip into 5 individual frames, feeding them into the model one by one in chronological sequence. The choice of a 5-frame clip length satisfies the memory boundaries of an 80GB A800 GPU during unrolled autoregressive and mask backpropagation, while providing an optimal sequence length to capture the complete streaming lifecycle of initialization, frame skipping via SAM2 propagation, and alarm-triggered MLLM re-invocation.

Through this streaming training regime, the model learns to maintain precise masks using only SAM2 and recycled tokens across skipped frames. Simultaneously, when rapid object motion or occlusions cause mask quality to dip and trigger the OMCP alarm, the MLLM masters the ability to intervene with contextual prompts and swiftly rectify semantic drift. The commentator practices live bus rides, glancing lightly at notes during calm stretches and actively taking the floor when dynamic scenery demands sharp correction.

### 3.5 Optimization Objective

The entire architecture is optimized end-to-end using a joint objective that couples text generation loss with mask segmentation loss.

Mask loss $\mathcal{L}_{\text{mask}}$ is a weighted combination of binary cross-entropy loss $\mathcal{L}_{\text{bce}}$ and DICE loss $\mathcal{L}_{\text{dice}}$:

$$\mathcal{L}_{\text{mask}} = \lambda_{\text{bce}} \mathcal{L}_{\text{bce}}(\mathcal{X}_M^s, \hat{\mathcal{X}}_M^s) + \lambda_{\text{dice}} \mathcal{L}_{\text{dice}}(\mathcal{X}_M^s, \hat{\mathcal{X}}_M^s)$$

where $\mathcal{X}_M^s$ and $\hat{\mathcal{X}}_M^s$ represent ground truth and predicted masks, respectively. Binary cross-entropy evaluates pixel-wise classification fidelity, while DICE loss maximizes global geometric overlap and mitigates foreground-background class imbalance.

Total loss $\mathcal{L}_{\text{total}}$ is formulated as:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{txt}} \mathcal{L}_{\text{txt}}(y_{\text{txt}}, \hat{y}_{\text{txt}}) + \lambda_{\text{mask}} \mathcal{L}_{\text{mask}}$$

Here, $y_{\text{txt}}$ and $\hat{y}_{\text{txt}}$ denote ground truth and predicted text tokens. Autoregressive cross-entropy loss $\mathcal{L}_{\text{txt}}$ supervises conversational generation and `[SEG]` token synthesis, while $\mathcal{L}_{\text{mask}}$ directly drives mask precision. Following standard protocol, all hyperparameter weights $\lambda_{\text{bce}}$, $\lambda_{\text{dice}}$, $\lambda_{\text{txt}}$, and $\lambda_{\text{mask}}$ are set to 1.

---

## 4. Experimental Results

### 4.1 Referring Video Object Segmentation Performance

![Figure 3: RVOS Performance Radar Chart](/images/streaming-rvos/_page_1_Figure_0.jpeg)
*Figure 3: Multi-benchmark performance comparison between StreamingRVOS-4B and state-of-the-art MLLM-based RVOS methods. StreamingRVOS consistently outperforms VRS-HQ-7B, VISA-7B, and Sa2VA-4B across all evaluation axes.*

The following table presents quantitative evaluations across standard RVOS benchmarks. Metrics include region similarity $\mathcal{J}$, contour accuracy $\mathcal{F}$, and overall mean $\mathcal{J}\&\mathcal{F}$.

| Model | Scale | Ref-DAVIS17 $\mathcal{J}\&\mathcal{F}$ | Ref-YT-VOS $\mathcal{J}\&\mathcal{F}$ | MeViS $\mathcal{J}\&\mathcal{F}$ | ReVOS $\mathcal{J}\&\mathcal{F}$ |
|---|---|:---:|:---:|:---:|:---:|
| ReferFormer | - | 61.1 | 62.9 | 31.0 | - |
| VISA | 13B | 70.4 | 63.0 | 44.5 | 57.4 |
| VideoLISA | 3.8B | 68.8 | 63.7 | 44.4 | - |
| VRS-HQ | 7B | 76.0 | 70.4 | 50.6 | 62.1 |
| GLUS | 7B | - | 67.3 | 51.3 | 58.3 |
| Sa2VA | 1B | 72.3 | 65.3 | 41.7 | 39.0 |
| Sa2VA | 4B | 73.8 | 70.0 | 46.2 | 59.8 |
| StreamingRVOS | 1B | 76.4 | 69.1 | 49.7 | 59.7 |
| StreamingRVOS | 4B | 76.6 | 70.5 | 50.9 | 63.0 |

StreamingRVOS-4B scores 76.6 on Ref-DAVIS17, 70.5 on Ref-YouTube-VOS, and 63.0 on ReVOS, matching or exceeding the 7B VRS-HQ model across every benchmark. The comparison against Sa2VA, which shares the identical InternVL2.5 backbone, is particularly striking: the 1B variant reaches 49.7 on MeViS compared to 41.7 for Sa2VA-1B (an 8.0-point improvement), while expanding the margin on ReVOS to 59.7 versus 39.0 (a 20.7-point surge).

These findings are especially noteworthy because StreamingRVOS operates in a streaming pipeline, yet surpasses offline state-of-the-art models. The commentator delivers more accurate descriptions live on the road than competitors who study recorded footage in post-production.

### 4.2 Referring Expression Segmentation Performance

StreamingRVOS maintains competitive accuracy on static image segmentation benchmarks alongside its video streaming capabilities.

| Model | Scale | RefCOCO val | RefCOCO+ val | RefCOCOg val |
|---|---|:---:|:---:|:---:|
| LISA | 7B | 74.9 | 65.1 | 67.9 |
| GLaMM | 7B | 79.5 | 72.6 | 74.2 |
| Sa2VA | 4B | 78.9 | 71.7 | 74.1 |
| UniPixel | 3B | 80.5 | 74.3 | 76.3 |
| StreamingRVOS | 1B | 80.3 | 75.0 | 77.5 |
| StreamingRVOS | 4B | 82.5 | 77.9 | 79.9 |

StreamingRVOS-4B achieves 82.5 on RefCOCO val, 77.9 on RefCOCO+ val, and 79.9 on RefCOCOg val, setting new competitive marks across all three datasets. Mastering temporal context propagation through SER does not compromise single-image precision in the slightest, validating both streaming adaptability and static fidelity.

### 4.3 Effectiveness of SER and OMCP

Ablation experiments dissecting the individual contributions of SER and OMCP are summarized below.

| Method | SER | OMCP | MeViS $\text{val}^u$ | ReVOS |
|---|:---:|:---:|:---:|:---:|
| Sa2VA-1B Retrained | ✗ | ✗ | 52.4 | 57.8 |
| Sa2VA-1B-Stream | ✗ | ✗ | 57.1 | 58.0 |
| Ours-1B | ✓ | ✗ | 58.5 | 58.9 |
| Ours-1B | ✓ | ✓ | 59.5 | 59.7 |

Transitioning Sa2VA into a streaming framework boosts MeViS performance from 52.4 to 57.1. Adding SER increases accuracy further to 58.5, and incorporating OMCP-driven video fine-tuning achieves the top mark of 59.5.

These results illustrate the distinct value of both components: SER establishes continuous temporal context to eliminate semantic confusion, while OMCP closes the training-inference gap to deliver enhanced accuracy alongside superior throughput.

### 4.4 Analysis of `[INFO]` Context Token Design

Detailed investigations into the quantity and update strategy of `[INFO]` tokens reveal notable behavioral patterns.

| Number of `[INFO]` Tokens | Ref-DAVIS | MeViS $\text{val}^u$ | ReVOS |
|---|:---:|:---:|:---:|
| 0 - Not used | 75.2 | 52.4 | 57.8 |
| 1 - Default setting | 75.0 | 58.5 | 58.9 |
| 2 | 75.1 | 58.1 | 60.2 |
| 3 | 74.6 | 58.2 | 60.1 |

Omitting the `[INFO]` token causes MeViS performance to plunge to 52.4. Because MeViS centers on complex motion-based referring expressions, lacking temporal context is catastrophic. However, expanding the token queue to 2 or 3 tokens yields no meaningful improvement, and causes Ref-DAVIS accuracy to dip to 74.6. The authors note that accumulating multiple tokens in a basic FIFO queue introduces semantic contamination that degrades feature reliability.

Comparative evaluations of update policies further highlight the superiority of OMCP.

| Update Strategy | Ref-DAVIS | MeViS $\text{val}^u$ | ReVOS | FPS |
|---|:---:|:---:|:---:|:---:|
| First frame only | 70.8 | 54.4 | 56.5 | 8.9 |
| Every frame | 75.4 | 59.2 | 58.9 | 3.4 |
| Every 5 frames | 75.2 | 59.0 | 59.3 | 6.7 |
| Every 10 frames | 75.1 | 58.4 | 58.7 | 8.1 |
| OMCP Adaptive | 76.4 | 59.5 | 59.7 | 6.7 |

Invoking the MLLM on every single frame drops speed to 3.4 FPS, ruling out real-time usage. Fixed intervals of 5 or 10 frames accelerate inference but sacrifice accuracy. OMCP achieves the highest score of 76.4 $\mathcal{J}\&\mathcal{F}$ while sustaining 6.7 FPS.

The advantage of OMCP over fixed intervals is intuitive: fixed intervals react too slowly during volatile scene shifts while wasting MLLM compute during tranquil stretches. OMCP tracks mask fidelity dynamically, intervening only when truly required.

Remarkably, querying the MLLM on every frame performs worse than OMCP. Because Stage 2 video fine-tuning trains under OMCP conditions, performing inference under that same policy minimizes the train-test discrepancy, unlocking optimal performance.

### 4.5 Qualitative Analysis

![Figure 4: Qualitative Comparison](/images/streaming-rvos/_page_7_Figure_0.jpeg)
*Figure 4: Segmentation comparisons between StreamingRVOS and Sa2VA. StreamingRVOS employs a streaming inference pipeline while Sa2VA uses an offline paradigm, highlighting advantages in semantic consistency and online error correction.*

The upper sequence in Figure 4 demonstrates the semantic consistency conferred by SER. When given the expression "a cat crouching down, turning its head, and attacking another cat," Sa2VA confuses the attacking and attacked cats midway through the sequence during offline processing. StreamingRVOS retains continuous target identity by propagating previous segmentation tokens via SER.

The lower sequence illustrates OMCP online error correction. In a scene where "a cat on a bed is attacked by another cat," early errors in Sa2VA propagate across the entire sequence. StreamingRVOS detects the degradation in mask confidence and re-invokes the MLLM at the point labeled "Re-perceive," immediately restoring the correct target mask.

### 4.6 Inference Efficiency Analysis

![Figure 5: FPS and OMCP Threshold Analysis](/images/streaming-rvos/_page_7_Figure_2.jpeg)
*Figure 5: Relationship between inference FPS, OMCP thresholds, and performance on the Ref-YouTube-VOS dataset.*

Figure 5 maps the tradeoff between inference speed and accuracy across OMCP thresholds $\tau_1$ and $\tau_2$. Elevating $\tau_1$ from 0.6 to 0.8 reduces FPS from 7.4 to 5.8 while lifting performance from 68.1 to 69.5, as stricter alarm criteria trigger more frequent MLLM re-evaluations.

Threshold $\tau_2$ exhibits a complementary dynamic: performance scales gradually from 69.0 at $\tau_2=0.0$ to 69.1 at $\tau_2=0.1$ and 69.3 at $\tau_2=0.2$. This confirms that inter-frame mask consistency provides subtle, essential stabilization alongside internal confidence metrics.

---

## 5. Conclusion and Key Takeaways

StreamingRVOS confronts the RVOS research community with a pivotal question: must video segmentation always demand full sequence access upfront and heavy, independent MLLM invocations on every single frame? The findings demonstrate that it does not.

The foundational insight of Semantic Embedding Recycling is disarmingly simple: the `[SEG]` token synthesized by an MLLM already encapsulates object localization, geometry, and semantics in a compressed format. Relaying this single token to the next frame preserves temporal coherence naturally, dispensing with complex memory banks or offline pre-processing.

Online Mask Consistency Perception extends beyond inference optimization to provide deep architectural insights into training design. By replicating adaptive invocation patterns within the training loop, the authors resolve the long-standing mismatch between training setups and inference realities. The fact that invoking the MLLM on every frame yields inferior accuracy proves that aligning training and inference dynamics is far more critical than brute-force computation.

Achieving these milestones without introducing a single extra parameter underscores the elegance of the approach. SER recycles the existing `[SEG]` token, while OMCP reuses the native predicted IoU from SAM2. Rather than piling on architectural complexity, StreamingRVOS rearranges existing components into an intuitive streaming paradigm. Outperforming Sa2VA-1B by 19.2% on MeViS while sustaining 7 FPS real-time throughput proves that transforming the foundational pipeline architecture is the most potent path to practical vision-language intelligence.
