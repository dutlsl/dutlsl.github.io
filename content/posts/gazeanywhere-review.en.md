---
title: "[CVPR 2026] Gaze Target Estimation Anywhere with Concepts"
date: 2026-10-01T14:56:26+09:00
draft: false
math: true
tags: ["Paper Review", "GAZE 2026", "Gaze Target Estimation", "Gaze Estimation", "Promptable Vision", "End-to-End", "Vision Foundation Model", "CVPR 2026"]
categories: ["GAZE 2026", "Paper Review"]
summary: "Introducing Promptable Gaze Target Estimation (PGE) to identify subjects and estimate their gaze targets end-to-end via natural language prompts, supported by the GazeAnywhere framework and the 120K Gaze-Co dataset."
cover:
  image: "/images/gazeanywhere/_page_2_Figure_0.jpeg"
  alt: "GazeAnywhere Architecture Overview"
---

> Reference Paper
> - Cao, X., Yang, H., Gunda, V., Zhou, Z., Xu, T., Kowdle, A., Kim, I., Rehg, J.M. "Gaze Target Estimation Anywhere with Concepts." CVPR 2026.
> - Project: https://github.com/IrohXu/GazeAnywhere

![Figure 1: GazeAnywhere Architecture Overview](/images/gazeanywhere/_page_2_Figure_0.jpeg)
*Figure 1: The GazeAnywhere end-to-end framework for Promptable Gaze Target Estimation. Features extracted from frozen visual encoder DINOv3 and text encoder dino.txt are fused by a trainable Detector Transformer, which simultaneously predicts the subject head bounding box, gaze target heatmap, and in/out-of-frame presence.*

---

## 1. One-Sentence Summary

GazeAnywhere introduces Promptable Gaze Target Estimation to locate target individuals and estimate their gaze destinations in an end-to-end manner from flexible natural language prompts, eliminating brittle multi-stage detector pipelines with a unified architecture and the 120K Gaze-Co benchmark.

---

## 2. Research Background and Motivation

### 2.1 Problem Definition

Consider the air traffic control tower of a major international hub. An air traffic controller must identify a specific aircraft among dozens taxiing across runways and determine its intended flight path. Imagine if the controller first had to manually spot the airframe livery, call an external visual scout to draw coordinates around the fuselage, and only then feed those coordinates into a separate trajectory computer. If the scout misidentifies the aircraft, the entire ground control system grinds to a halt. Modern air traffic control does not operate this way. A controller simply utters a flight callsign into the radio, and the integrated radar immediately highlights the aircraft and tracks its designated landing runway.

Human gaze target estimation has long remained trapped in that cumbersome, multi-device manual pipeline. When an image is fed into conventional pipelines, an Open-Vocabulary Detector must first localize the subject head bounding box. That bounding box is then cropped and forwarded to a dedicated gaze model to predict the line of sight. This sequential multi-stage design is exceptionally fragile to error cascading, where any false positive or missed detection in the initial stage irrecoverably sabotages all subsequent gaze predictions.

GazeAnywhere fundamentally rethinks this architecture. Mirroring the unified control tower where a single callsign activates end-to-end flight guidance, the authors define Promptable Gaze Target Estimation, abbreviated as PGE. In this new paradigm, a concise natural language description such as "the boy in the red shirt" directly guides the model to identify the subject and predict their gaze heatmap in a single forward pass.

### 2.2 Limitations of Existing Methods: The Fragmented Control Tower

![Figure 2: Comparison between traditional pipelines and GazeAnywhere](/images/gazeanywhere/_page_0_Figure_11.jpeg)
*Figure 2: Existing methods rely on a two-stage pipeline where an OVD first detects head bounding boxes before feeding them to a separate gaze model. GazeAnywhere ingests natural language prompts directly to jointly perform subject localization and gaze target estimation end-to-end.*

Returning to the control tower analogy, existing gaze target estimation frameworks suffer from three severe architectural bottlenecks.

The first bottleneck is extreme sequential dependency. State-of-the-art gaze architectures such as ViTGaze, Sharingan, and Gaze-LLE all strictly require a ground-truth or pre-extracted head bounding box as an indispensable input. Generating these boxes in the wild necessitates running separate open-vocabulary detectors like GroundingDINO, OWLv2, or RexSeek. Consequently, the detector accuracy establishes a rigid performance ceiling. In crowded social scenes, poorly illuminated environments, or clinical settings with pediatric faces, detectors frequently confuse nearby heads or fail to detect atypical facial features entirely, causing catastrophic system failure.

The second bottleneck is prohibitive computational latency. Cascading an open-vocabulary detector with a heavy gaze model drastically inflates parameter counts and inference latency. Pairing RexSeek with ViTGaze produces a massive 3B-parameter pipeline requiring 1,160 ms per image, rendering real-time deployment on interactive edge devices or augmented reality smart glasses completely infeasible.

The third bottleneck is the lack of interaction flexibility. Traditional pipelines constrain user queries strictly to pre-defined geometric coordinates or bounding boxes. They cannot interpret intuitive human directives such as "track where the girl in the striped cardigan is looking." This rigidity stands in sharp contrast to the massive progress in promptable vision foundation models like Segment Anything and open-vocabulary grounding.

### 2.3 Main Contributions

The core contributions of this work are three-fold:

- The authors formulate Promptable Gaze Target Estimation (PGE), a novel task that unifies subject identification and gaze target prediction into an end-to-end formulation conditioned on flexible natural language prompts.
- They construct Gaze-Co, the first large-scale PGE benchmark comprising 120K diverse samples synthesized through a scalable, human-in-the-loop data engine that harmonizes heterogeneous gaze datasets with structured concept annotations.
- They propose GazeAnywhere, an efficient end-to-end framework coupling frozen vision-language encoders with a lightweight Detector Transformer, achieving state-of-the-art accuracy across all PGE benchmarks while slashing latency by over twelve-fold compared to two-stage baselines.

---

## 3. Proposed Framework

### 3.1 PGE Task Formulation: Direct Radio-to-Touchdown Control

Imagine directing an aircraft without waiting for ground scouts to report coordinates. PGE establishes an end-to-end protocol where voice instructions directly trigger visual trajectory maps on the control console.

Formally, given an input RGB image $I \in \mathbb{R}^{3 \times H \times W}$ and a user prompt $P$, the objective of PGE is to directly synthesize a gaze target heatmap $\hat{H} \in \mathbb{R}^{H_{\text{out}} \times W_{\text{out}}}$. Each entry $\hat{H}(i, j)$ represents the predicted probability that the subject specified by $P$ is gazing at spatial coordinate $(i, j)$.

The framework accommodates two prompt modalities: visual coordinate prompts specifying a 2D spatial point, and text prompts describing the target subject in natural language.

To prevent ambiguous queries such as "the person in the back" from causing guidance failures, the authors introduce a structured four-category concept protocol:

| Category | Operational Role in Guidance | Representative Example |
|---|---|---|
| Appearance | Stable visual markers identifying the airframe | Adult male with brown hair, spectacles, and blue plaid shirt |
| Location | Spatial canvas coordinates | Bottom left corner region |
| Pose | Static body configuration and facing direction | Standing upright facing directly forward |
| Action | Ongoing physical interactions or movement | Typing document on laptop keyboard |

Crucially, PGE prohibits the model from consuming explicit intermediate representations like pre-computed bounding boxes or skeletal pose keypoints during inference. The network must jointly learn subject identification, spatial grounding, and directional gaze projection within a unified computational graph.

### 3.2 GazeAnywhere Architecture: The Unified Control Center

As depicted in Figure 1, the GazeAnywhere architecture mirrors an integrated command deck where specialized surveillance and communications modules interlock seamlessly.

#### 1. Frozen Encoders: Radar Scanning and Radio Reception

Monitoring airport airspace requires wide-area radar sweeps and clear voice reception channels.

The visual encoder $\phi_V$ acts as a high-altitude surveillance radar. Leveraging a frozen ViT backbone, it partitions the input image $I$ into a spatial grid of $N_V$ local visual patch tokens. To encapsulate global scene semantics, a learnable [CLS] token $c$ is prepended to the sequence. The visual encoder outputs a composite sequence of $1 + N_V$ tokens:

$$\phi_V(I) = [c, s_1, s_2, \ldots, s_{N_V}] \in \mathbb{R}^{(1+N_V) \times D_V}$$

Here, $c \in \mathbb{R}^{D_V}$ encapsulates global environmental context, while $s_i \in \mathbb{R}^{D_V}$ captures fine-grained spatial evidence within the $i$-th patch.

The text encoder $\phi_T$ functions as a digital communications transceiver. A frozen transformer encoder tokenizes the natural language prompt $T$ into an initial embedding sequence $T_E$, padded to a fixed context length $L_T$:

$$\phi_T(T_E) = [t_1, t_2, \ldots, t_{N_T}, t_{\text{eos}}, \ldots, t_{\text{pad}}] \in \mathbb{R}^{L_T \times D_T}$$

Here, $t_i \in \mathbb{R}^{D_T}$ denotes the embedding for the $i$-th content word token, $t_{\text{eos}} \in \mathbb{R}^{D_T}$ denotes the end-of-sentence token condensing holistic prompt semantics, and $t_{\text{pad}} \in \mathbb{R}^{D_T}$ represents padding tokens ensuring uniform context length across varying prompt lengths.

Both encoders are strictly frozen throughout training. Rather than rewiring the internal hardware of proven radar and radio transceivers, freezing preserves rich pre-trained multi-modal representations while allowing lightweight downstream adapters to specialize exclusively on cross-modal gaze reasoning.

#### 2. Linear Projection Layers: Standardizing Signal Protocols

Surveillance radar signals of dimension $D_V$ and radio voice signals of dimension $D_T$ reside in entirely disparate feature spaces. To display them on a common tactical screen, both signals must be converted to a unified communication protocol.

Two learnable linear projection matrices, $W_V$ and $W_T$, project the representations into a shared latent space of dimension $D$:

$$Z_V = W_V \cdot \phi_V(I) \in \mathbb{R}^{(1+N_V) \times D}$$

$$Z_T = W_T \cdot \phi_T(T_E) \in \mathbb{R}^{L_T \times D}$$

Here, $W_V \in \mathbb{R}^{D \times D_V}$ and $W_T \in \mathbb{R}^{D \times D_T}$ perform token-wise linear transformations.

The authors deliberately enforce $D < \min(D_V, D_T)$. Compressing features into a lower-dimensional bottleneck eliminates modality-specific high-frequency noise, forcing the representations to retain only the most salient cross-modal cues necessary for subject-gaze alignment.

#### 3. Task-Specific Embeddings: Specialized Mission Tags

Beyond standard feature projection, an air traffic controller must attach two explicit mission tags to the tracking board: "Where is the designated aircraft located on the tarmac?" and "Is its final destination inside our regional airfield or exiting our sector?"

The Head Token $t_h$ serves as an explicit localization query:

$$t_h = t'_{\text{eos}} \in \mathbb{R}^D$$

Because the projected end-of-sentence token $t'_{\text{eos}}$ condenses the complete semantic description of the target individual, initializing the head query with $t'_{\text{eos}}$ provides an optimal inductive bias to seek out the matching subject in visual space.

The Target Presence Token $t_p$ predicts whether the gaze destination resides inside or outside the camera frame:

$$t_p = c' + \mathbf{E}_{\text{presence}} \in \mathbb{R}^D$$

Here, $c'$ is the projected visual [CLS] token and $\mathbf{E}_{\text{presence}}$ is a learnable guidance embedding.

Decoupling presence prediction from spatial localization reflects a profound architectural insight. Determining whether gaze exits the frame requires broad spatial context across the entire image canvas, whereas pinpointing the exact landing coordinates requires localized fine-grained inspection. Forcing a single query to simultaneously perform both objectives creates optimization conflict. Introducing an independent global token $t_p$ completely resolves this tension.

#### 4. Detector Transformer: The Integrated Tactical Radar Screen

All task tokens, text instructions, and visual canvas patches are concatenated into a single input sequence $F$:

$$F = [t_h, \mathbf{t}', \mathbf{s}', t_p] \in \mathbb{R}^{(N_T + N_V + 2) \times D}$$

The ordering strictly mirrors operational hierarchy: the head localization token $t_h$ leads the sequence, followed by text content tokens $\mathbf{t}'$, visual patch tokens $\mathbf{s}'$, and finally the global presence token $t_p$.

To preserve spatial and syntactic geometry, 1D sinusoidal position embeddings are added to $\mathbf{t}'$, while 2D sinusoidal position embeddings are injected into visual tokens $\mathbf{s}'$.

The Detector Transformer $\psi$, comprising $k$ standard transformer blocks, applies multi-head self-attention across the combined sequence. Through self-attention, visual patches attend to text tokens to identify the described subject, the head query hones in on the subject facial features, and visual patches exchange spatial vectors to track gaze direction.

#### 5. Decoders: Dedicated Operational Terminals

Following transformer refinement, distinct tokens are routed to specialized output heads:

The Gaze Tracker takes refined visual patch tokens $\hat{\mathbf{s}} \in \mathbb{R}^{N_V \times D}$, rearranges them into a 2D spatial grid, and passes them through two transposed convolutional layers to upsample the features into the final $64 \times 64$ gaze heatmap $\hat{H}$.

The Head Tracker passes the refined head token $\hat{t}_h \in \mathbb{R}^D$ through a 3-layer feed-forward network with ReLU activations, regressing a 4D normalized bounding box $[x, y, w, h]$. While auxiliary, this objective provides a strong supervision signal that tightly binds language concepts to visual heads.

The Presence Predictor routes the refined presence token $\hat{t}_p \in \mathbb{R}^D$ through a 2-layer FFN to output a scalar logit for binary in/out-of-frame classification.

### 3.3 Learning Objectives: Calibrating the Guidance Instruments

GazeAnywhere is optimized end-to-end via a multi-task composite loss function:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{gaze}} + \mathcal{L}_{\text{presence}} + \mathcal{L}_{\text{head}}$$

Each component calibrates a vital navigational sensory capability.

#### 1. Gaze Heatmap Loss: Soft Touchdown Guidance Zones

Supervising a gaze point as an isolated single-pixel Dirac delta yields an extraordinarily sparse learning signal, leading to vanishing gradients across the vast pixel space. Ground-truth guidance must form a smooth Gaussian basin around the landing zone.

The authors smooth the ground-truth gaze coordinate with a 2D Gaussian filter ($\sigma = 3$) to generate target heatmap $Y$, supervised via pixel-wise binary cross-entropy:

$$\mathcal{L}_{\text{gaze}} = -\frac{1}{N} \sum_{p=1}^N \left[ y_p \log(\hat{y}_p) + (1 - y_p) \log(1 - \hat{y}_p) \right]$$

Here, $N = H_{\text{out}} \times W_{\text{out}} = 64 \times 64 = 4,096$ denotes total pixel count, normalizing loss scale across resolutions. $y_p \in [0, 1]$ represents the Gaussian ground-truth confidence, and $\hat{y}_p \in [0, 1]$ is the predicted probability.

In target regions where $y_p \approx 1$, predicting $\hat{y}_p \to 1$ drives $\log(\hat{y}_p) \to 0$, vanishing the penalty. Conversely, failing to activate on the true target drives the logarithm toward negative infinity, which converts via the outer minus sign into a massive positive loss penalty. In vast non-target background areas where $y_p = 0$, the $(1 - y_p) \log(1 - \hat{y}_p)$ term penalizes false alarms, ensuring clean, sharp heatmap peaks.

#### 2. Presence Loss: Focal Modulation for Boundary Outliers

In social datasets, subjects gaze inside the frame far more frequently than outside, creating severe class imbalance. Standard cross-entropy would allow the model to score artificially high accuracy by trivially predicting in-frame presence while failing on difficult out-of-frame instances.

The authors deploy Focal Loss to dynamically modulate gradient focus:

$$\mathcal{L}_{\text{presence}} = \mathcal{L}_{\text{focal}}(Y_{\text{presence}}, \hat{Y}_{\text{presence}})$$

Here, $Y_{\text{presence}} \in \{0, 1\}$ is the ground-truth binary label, and $\hat{Y}_{\text{presence}} \in [0, 1]$ is the predicted probability.

Focal Loss introduces the dynamic modulating factor $(1 - p_t)^\gamma$. When the model classifies an easy in-frame sample with high confidence ($p_t \to 1$), the factor shrinks toward zero, effectively zeroing its contribution to the backpropagation gradient. Conversely, on ambiguous boundary cases where gaze glances near the frame border, the modulating weight remains high, compelling the optimizer to concentrate updates on challenging out-of-frame discriminations.

#### 3. Head Bounding Box Loss: Anchoring Subject Localization

Verifying that the model identifies the intended subject requires regressing head bounding boxes. The loss must balance precise coordinate regression with robust gradients when predictions fail to overlap:

$$\mathcal{L}_{\text{head}} = \lambda_{l_1} \|b - \hat{b}\|_1 + \lambda_{\text{iou}} \mathcal{L}_{\text{iou}}(b, \hat{b})$$

Here, $b = [x, y, w, h]$ denotes normalized ground-truth box coordinates and $\hat{b} = [\hat{x}, \hat{y}, \hat{w}, \hat{h}]$ is the predicted box.

The $L^1$ loss $\|b - \hat{b}\|_1 = |x - \hat{x}| + |y - \hat{y}| + |w - \hat{w}| + |h - \hat{h}|$ provides steady linear gradients without being oversensitized by extreme coordinate outliers.

The Generalized IoU loss $\mathcal{L}_{\text{iou}}(b, \hat{b})$ circumvents standard IoU gradient vanishing. Under standard IoU, if two bounding boxes have zero overlap, the intersection area is zero and gradients vanish entirely. GIoU computes the smallest convex enclosing box encompassing both boxes, penalizing the empty volume not covered by the predicted and ground-truth boxes. This guarantees non-zero pull gradients even when boxes are entirely disjoint.

The weighting hyperparameters $\lambda_{l_1} = 5$ and $\lambda_{\text{iou}} = 2$ follow established DETR and OWLViT conventions, harmonizing coordinate scale differences with area ratios for balanced multi-task convergence.

### 3.4 Gaze-Co Data Engine: Synthesizing the Standardized Flight Manual

No advanced air traffic system can operate without standardized flight manuals pairing voice calls with aircraft locations and flight vectors. Pre-existing gaze datasets like GazeFollow, VAT, and ChildPlay contained only coordinate annotations and lacked natural language descriptions entirely.

To bridge this gap, the authors engineered a 3-stage human-in-the-loop data engine.

![Figure 3: Data Engine Workflow](/images/gazeanywhere/_page_4_Figure_10.jpeg)
*Figure 3: Overview of the Gaze-Co data engine pipeline. Heterogeneous source datasets are harmonized and filtered, structured concept phrases are synthesized via Gemini 2.5 Pro, and an MLLM-first human-in-the-loop review ensures high data purity.*

#### Stage 1: Data Alignment and Filter

Source datasets exhibit inconsistent coordinate systems and annotation policies. The data engine standardizes all records into explicit head box pixel coordinates and normalized gaze vectors $(g_x / W, g_y / H)$.

Rigorous geometric and sharpness filters are subsequently applied. Head boxes must satisfy width $\ge 30$ px, height $\ge 40$ px, area $\ge 2,500$ px$^2$, box-to-image area ratio between $0.008$ and $0.3$, and pass Tenengrad focus metrics. Extremely miniature, blurred, or excessively oversized instances lacking spatial context are purged.

#### Stage 2: Concept Generation

Filtered samples are queried against the Gemini 2.5 Pro VLM API to generate concise lowercase concept phrases structured around the four standardized fields.

Attribute captures invariant visual traits such as hair color, eyeglasses, and clothing patterns, ending strictly with visual category tokens like man, woman, boy, girl, infant, or child.

Position specifies relative canvas quadrants such as bottom left corner.

Action and Pose are enforced as strictly non-overlapping fields, where Action describes dynamic object interactions and Pose defines static body orientations. Unobservable attributes are labeled as none.

#### Stage 3: Verification

To prevent hallucinations from contaminating training, the engine deploys an MLLM-first, human-in-the-loop verification protocol.

Gemini 2.5 Pro first screens all synthesized concept phrases, flagging records as pass or fail. Human annotators subsequently audit random subsets of passed batches to verify batch success rates across three criteria: Consistency (text matches the designated head), Completeness (all four non-conflicting fields are present), and Privacy (no identifying or sensitive personal data). If batch accuracy falls below threshold, prompt rules are refined and re-executed until the observed error rate drops below $1\%$.

The resulting Gaze-Co training corpus contains 120K vetted samples with complete promptable annotations.

---

## 4. Experimental Results

### 4.1 SOTA Performance on PGE Benchmarks

The authors benchmark GazeAnywhere against twelve strong two-stage baselines formed by coupling three leading gaze models (ViTGaze, Sharingan, Gaze-LLE) with four open-vocabulary detectors (GroundingDINO, LLMDet, OWLv2, RexSeek).

| Model | Detector | Parameters | Latency | GazeFollow AUC ↑ | GazeFollow Avg L2 ↓ | VAT L2 ↓ | VAT AP ↑ | ChildPlay L2 ↓ | ChildPlay AP ↑ | Child-SC L2 ↓ | Child-SC AP ↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ViTGaze | RexSeek | 3B | 1160ms | 0.945 | 0.119 | 0.123 | 0.841 | 0.117 | 0.903 | 0.136 | 0.818 |
| Gaze-LLE | RexSeek | 3B | 1183ms | 0.954 | 0.108 | 0.121 | 0.861 | 0.119 | 0.914 | 0.172 | 0.846 |
| GazeAnywhere | CLIP-L | 430M | 35ms | 0.953 | 0.105 | 0.137 | 0.874 | 0.104 | 0.915 | 0.146 | 0.868 |
| GazeAnywhere | DINOv3-L | 870M | 96ms | 0.958 | 0.099 | 0.123 | 0.879 | 0.098 | 0.906 | 0.090 | 0.902 |

GazeAnywhere-DINOv3-L establishes state-of-the-art performance across all benchmark datasets. Two major takeaways emerge from the evaluation:

First, simultaneous breakthroughs in inference speed and predictive precision. The strongest baseline, Gaze-LLE combined with RexSeek, requires 3B parameters and 1,183 ms latency. GazeAnywhere-DINOv3-L outperforms this baseline with only 870M parameters and 96 ms latency, representing a 3.4x parameter reduction and a 12x speedup. Furthermore, the lightweight GazeAnywhere-CLIP-L variant executes in 35 ms, enabling real-time edge processing.

Second, exceptional out-of-domain robustness. On the private IRB-approved Child Social Communication (Child-SC) video dataset, GazeAnywhere-DINOv3-L records an L2 error of 0.090 and an AP of 0.902, outperforming the best two-stage competitor by over 33% in L2 distance. In pediatric scenes where off-the-shelf detectors frequently fail to detect child faces (detection rates hovering around 70%), the end-to-end paradigm decisively prevents error cascading.

### 4.2 Comparison with Generalist VLMs

| Model | Parameters | GazeFollow Avg L2 ↓ | GazeFollow Min L2 ↓ | VAT L2 ↓ | VAT AP ↑ |
|---|---|---|---|---|---|
| Qwen3-VL-8B | 8B | 0.201 | 0.137 | 0.286 | 0.651 |
| Gemini 2.5 Flash | - | 0.216 | 0.156 | 0.292 | 0.661 |
| GazeAnywhere-DINOv3-L | 870M | 0.099 | 0.050 | 0.123 | 0.879 |

Evaluating state-of-the-art vision-language foundation models in zero-shot PGE reveals significant performance gaps. Qwen3-VL-8B records an average L2 error of 0.201 on GazeFollow, whereas GazeAnywhere achieves 0.099, cutting spatial error by more than half. Generalist VLMs lack the fine-grained geometric grounding required for subtle eye and head vectoring, demonstrating the clear necessity of dedicated PGE architectures.

### 4.3 Prompting Strategy Analysis

| Prompt Modality | GazeFollow AUC ↑ | GazeFollow Avg L2 ↓ | VAT AUC ↑ | VAT L2 ↓ | VAT AP ↑ |
|---|---|---|---|---|---|
| No Prompting | 0.944 | 0.144 | 0.875 | 0.210 | 0.796 |
| Visual Prompting | 0.958 | 0.100 | 0.914 | 0.131 | 0.894 |
| Text: Appearance only | 0.952 | 0.113 | 0.904 | 0.153 | 0.840 |
| Text: Position only | 0.953 | 0.124 | 0.893 | 0.188 | 0.826 |
| Text: Action only | 0.949 | 0.129 | 0.897 | 0.180 | 0.839 |
| Text: Pose only | 0.952 | 0.121 | 0.909 | 0.163 | 0.859 |
| Text: Full Composition | 0.958 | 0.099 | 0.928 | 0.123 | 0.879 |

Integrating all four text concept categories matches visual coordinate prompting performance. Decomposing individual prompt components reveals that Appearance and Pose serve as the two most decisive conditioning cues. Appearance resolves subject identity unambiguously, while Pose provides indispensable directional priors for line-of-sight projection.

### 4.4 Loss and Encoder Ablations

Ablation studies on the multi-task objective confirm that incorporating auxiliary head localization loss $\mathcal{L}_{\text{head}}$ improves both gaze heatmap quality and target presence classification. Forcing the network to localize the subject head establishes strong semantic feature alignment that directly enriches downstream gaze attention maps.

Backbone comparisons establish DINOv3-L as the premier encoder, surpassing CLIP, SigLIP2, and MetaCLIP2. MetaCLIP2-H (1.9B parameters) fails to match DINOv3-L (870M parameters), demonstrating that spatial visual granularity and fine-grained visual-text alignment supersede raw model scale.

### 4.5 Qualitative Visualizations and AR Agent Deployment

![Figure 4: Qualitative Gaze Visualizations](/images/gazeanywhere/_page_7_Figure_0.jpeg)
*Figure 4: Qualitative gaze target estimation results across diverse scenarios. GazeAnywhere maintains high localization fidelity from simple sparse scenes to complex crowded gatherings using only textual descriptions of target appearance. The bottom rows illustrate robust out-of-domain performance on Child-SC.*

![Figure 5: AnyGaze Agent Architecture](/images/gazeanywhere/_page_7_Figure_2.jpeg)
*Figure 5: The AnyGaze Agent workflow deployed on AR smart glasses. Environmental video and microphone audio captured by DigiLens ARGO are routed to Gemini 2.5, which converts speech to text via Whisper v3 and invokes GazeAnywhere as a specialized perceptual tool.*

The authors demonstrate real-world deployment through the AnyGaze Agent integrated with DigiLens ARGO augmented reality glasses. The smart glasses capture ego-centric audio and video, Whisper v3 transcribes speech queries, and Gemini 2.5 calls GazeAnywhere as an external perception tool:

| Architecture Setup | Gaze Shift MAE/min ↓ | Eye Contact MAE/min ↓ |
|---|---|---|
| Gemini 2.5 Flash Standalone | 9.247 | 14.763 |
| Gemini 2.5 Flash + AnyGaze Agent | 2.337 | 4.756 |
| Gemini 2.5 Pro Standalone | 6.063 | 10.268 |
| Gemini 2.5 Pro + AnyGaze Agent | 2.546 | 6.672 |

Equipping MLLMs with the specialized GazeAnywhere agent tool cuts Gaze Shift Mean Absolute Error by four-fold, validating the efficacy of tool-augmented perception for interactive embodied AI.

---

## 5. Conclusion and Key Takeaways

GazeAnywhere marks a pivotal paradigm shift for gaze target estimation and promptable spatial intelligence.

First, the transition from fragmented multi-stage detector pipelines to unified end-to-end learning eliminates error cascading while delivering order-of-magnitude gains in speed and efficiency. Analogous to replacing multiple manual visual spotters with a unified radar console, eliminating brittle intermediate handoffs removes systemic vulnerabilities at their source.

Second, validating that natural language prompting rivals explicit visual coordinate cues establishes a human-centric interaction foundation. Clinical practitioners, educators, and roboticists can now query gaze patterns dynamically through conversational language without manually drawing bounding boxes on video frames.

Third, the Gaze-Co data engine offers a reusable blueprint for augmenting existing computer vision datasets with concept-level language annotations through MLLM-human collaborative loops.

Finally, the real-world deployment on AR glasses underscores the clinical and commercial promise of promptable gaze intelligence, providing a tangible path forward for automated social communication assessment in neurodevelopmental screening and embodied robotic assistance.
