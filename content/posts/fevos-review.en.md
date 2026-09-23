---
title: "[ECCV 2026] FeVOS: Foresight Expression Video Object Segmentation by Anticipating Future Actions and Pixel-Level Grounding"
date: 2026-09-23T16:35:00+09:00
draft: false
math: true
tags: ["Paper Review", "Video Object Segmentation", "RVOS", "MLLM", "Reinforcement Learning", "Chain-of-Thought", "ECCV 2026"]
categories: ["Paper Review"]
summary: "Moving beyond traditional RVOS that retrospectively grounds observed events, FeVOS introduces a predictive segmentation task where models anticipate future events from antecedent spatio-temporal cues in observed frames, alongside FeVOS-R1 combining CoT-SFT and GRPO reinforcement learning."
cover:
  image: "/images/fevos/_page_1_Picture_2.jpeg"
  alt: "FeVOS Task Overview"
---

> Reference Paper
> - Lan, K., Ying, K., Ding, H. "FeVOS: Foresight Expression Video Object Segmentation." ECCV 2026.
> - Project Page: https://henghuiding.com/FeVOS/

![Figure 1: FeVOS Task Overview](/images/fevos/_page_1_Picture_2.jpeg)
*Figure 1: Comparison between existing RVOS datasets and FeVOS. Ref-DAVIS grounds observable static attributes such as "the sponge in the sink", while MeViS handles motion expressions across frames like "the sponge moved". In contrast, FeVOS queries future events that have not yet occurred, such as "What tool will be used?", requiring models to anticipate and segment the object involved in the upcoming action based solely on antecedent spatio-temporal cues in observed frames.*

---

## 1. One-Sentence Summary

FeVOS defines a novel task that requires models to anticipate objects involved in forthcoming events from spatio-temporal cues in observed video frames and segment them at the pixel level, proposing FeVOS-R1 that combines Chain-of-Thought supervised fine-tuning with IoU-reward-guided GRPO reinforcement learning to advance video object segmentation from passive observation to proactive predictive reasoning.

---

## 2. Research Background and Motivation

### 2.1 Problem Definition

Consider an experienced sous-chef working alongside a head chef in a busy kitchen. The moment the head chef squirts dish soap into a greasy pot and grasps its handle firmly with the left hand, the skilled assistant does not stand idly waiting for the chef to shout "bring me a sponge" or reach out and pick one up. By instantaneously assessing the current state of the dirty pot and the physical constraint of the occupied hand, the assistant directs attention preemptively to the sponge resting on the right counter, the most convenient tool for the chef's free hand. Human visual perception does not passively record past occurrences; it proactively anticipates upcoming causal events from subtle antecedent cues to prepare swift, decisive actions.

Referring Video Object Segmentation, abbreviated as RVOS, is a research domain aiming to train artificial intelligence to integrate natural language expressions with video streams and segment the referent objects at the pixel level across entire video sequences. It serves as a foundational technology for video editing, autonomous driving, and robotic manipulation planning.

However, all existing RVOS studies have operated under a comfortable premise: the actions or events described by the textual expressions are already captured within the observed video frames. For physical robots or intelligent agents to truly assist humans in real-world environments, verifying what has already happened is insufficient. They must anticipate which object will be required next and what action will unfold. FeVOS reformulates observation-centric video segmentation into proactive predictive reasoning.

From the perspective of an intelligent assistant, solving this task requires overcoming three unique challenges:

| Challenge | Description |
|---|---|
| Inability to Direct Vision-Language Alignment | The referring expression queries an unobserved future event, making direct word-to-visual pattern matching impossible |
| Predictive Spatio-Temporal Reasoning | Models must synthesize temporal context, spatial affordance, physical dynamics, and human intent to deduce the target of the upcoming action |
| Pixel-Level Fine-Grained Segmentation | Models must extract sharp, spatially and temporally consistent pixel masks for the anticipated object across observed frames |

### 2.2 Limitations of Existing Methods: Passive Observers in the Kitchen

Returning to the kitchen analogy, conventional RVOS models behave like an unobservant assistant who waits until the chef has already picked up the sponge and scrubbed the pot before remarking that the object in hand is a sponge.

Early benchmarks like Ref-DAVIS and Ref-YouTube-VOS focused on static attributes easily discernible from single frames, such as a sponge sitting in a sink. MeViS advanced the benchmark by introducing motion expressions that capture spatio-temporal dynamics across frames, while ReVOS and ReasonVOS incorporated complex reasoning and common-sense world knowledge.

Nonetheless, across all these benchmarks, the target events are always visible within the provided video clips. Models never needed to look beyond the frame boundary. Meanwhile, video understanding tasks exploring future prediction have largely adhered to simplistic question-answering formats, failing to provide the dense pixel-level visual grounding essential for real-world robotic interaction.

Recent multimodal large language models such as Sa2VA, VISA, and VideoLISA also falter when confronted with this predictive challenge. While excelling on traditional benchmarks, they lack explicit causal reasoning trajectories connecting antecedent visual evidence to future events. When the direct link between language description and observable pixels is severed, these models struggle to interpret temporal context, misdirecting their attention to irrelevant objects.

### 2.3 Main Contributions

The core contributions of this paper are summarized into three key aspects:

- Proposing Foresight Expression Video Object Segmentation, a novel task requiring models to anticipate which objects will participate in future events from observed spatio-temporal cues, shifting RVOS from passive observation to proactive reasoning.
- Constructing the FeVOS dataset, comprising 968 video clips, 14,525 foresight expressions, 37,412 pixel-level segmentation masks, and 2,904 Chain-of-Thought annotations that provide explicit causal supervision.
- Developing FeVOS-R1, a two-stage training framework that establishes foundational reasoning via CoT supervised fine-tuning and directly optimizes segmentation quality via IoU-reward-guided GRPO reinforcement learning, achieving state-of-the-art results on FeVOS while demonstrating superior generalization on conventional RVOS benchmarks.

---

## 3. Proposed Framework

### 3.1 Data Construction Pipeline

![Figure 2: Data Construction Pipeline](/images/fevos/_page_4_Figure_2.jpeg)
*Figure 2: The five-stage data construction pipeline of FeVOS: Video Collection, Automatic Filtration, Manual Video Splitting, Expression Annotation, and Mask Annotation.*

To train an assistant to anticipate future actions, the training material must feature scenarios where early cues foreshadow subsequent events with deterministic clarity. The authors developed a rigorous five-stage pipeline to curate data with causal inevitability.

In Video Collection, candidate clips were gathered from diverse benchmarks, including first-person activities in EPIC-KITCHENS-VISOR, instructional videos in COIN, daily human actions in STAR, physical simulations in CLEVR, and unexpected events in OOPS. These domains ensure rich spatio-temporal dynamics where early visual cues naturally anticipate forthcoming occurrences.

In Automatic Filtration, Qwen2.5-VL served as a validator to evaluate predictive suitability. Videos were screened based on whether the antecedent segment provided a clear cause-and-effect narrative to deterministically deduce the future segment, whether multiple objects interacted, and whether observable and actionable cues were present, retaining the top 30.5% of candidates.

In Manual Video Splitting, human annotators employed an online tool to split filtered videos into observation and future segments. The split ensured that the observation clip contained visible target objects with clear contextual foreshadowing, while the future clip captured a definite, predictable action. About 44.0% of clips satisfied these criteria.

In Expression Annotation, a two-stage cross-validation process was implemented. First, an annotator with access to both observation and future clips selected the target object and formulated a predictive expression. Subsequently, an independent validator, provided only with the expression and the observation clip, attempted to identify the referent through predictive reasoning. Overly trivial queries inferable from single-frame appearances, as well as ambiguous questions lacking sufficient cues, were eliminated, retaining 85.9% of well-balanced expressions.

Finally, in Mask Annotation, interactive annotation tools powered by SAM2 were utilized to generate precise pixel-level masks for target objects across all observation frames, ensuring temporal consistency.

### 3.2 Automatic CoT Annotation Generation: Teaching the Assistant How to Reason

![Figure 3: Samples from FeVOS with CoT Annotations](/images/fevos/_page_2_Picture_2.jpeg)
*Figure 3: Representative samples from FeVOS with Chain-of-Thought annotations, illustrating three core predictive reasoning challenges: Physically-Aligned Prediction, Procedure-Grounded Prediction, and Intention-Guided Prediction.*

Once video-expression-mask triplets were curated, Chain-of-Thought annotations were synthesized to instill structured, step-by-step reasoning patterns into the model.

By overlaying ground-truth masks onto video frames as visual prompts, Qwen2.5-VL was tasked with articulating why the highlighted object constitutes the correct answer to the predictive question. Three distinct reasoning paths were generated per video to provide diverse perspectives.

In our assistant analogy, this corresponds to teaching an apprentice to verbally justify why a specific sponge must be selected. As illustrated in the curling example in Figure 3, the model traces the trajectory of the sliding red stone, observes its linear alignment with the lowermost yellow stone, rules out collision with other stones outside the trajectory, and concludes that the lowermost yellow stone will be struck.

### 3.3 Dataset Statistics and Three Predictive Reasoning Challenges

![Figure 4: Word Clouds](/images/fevos/_page_6_Figure_2.jpeg)
*Figure 4: Word clouds illustrating the distributions of referring expressions and reasoning traces. Foresight expressions frequently feature future-oriented markers like "will", while reasoning traces frequently employ grounding markers like "Given".*

The finalized FeVOS dataset encompasses 968 video clips, 14,525 expressions, and 37,412 segmentation masks across 30,125 frames. The training subset contains 779 videos with 11,708 expressions, while the validation subset consists of 189 videos with 2,817 expressions, supported by 2,904 synthetic CoT annotations.

The predictive reasoning challenges required by FeVOS fall into three distinct categories:

| Reasoning Challenge | Required Capability | Representative Scenario |
|---|---|---|
| Physically-Aligned Prediction | Cross-frame physical trajectory estimation | Tracking a sliding curling stone across frames to identify which opposing stone will be struck |
| Procedure-Grounded Prediction | Understanding procedural task sequences | Observing a knife halving a melon and inferring that a spoon will be required to scoop out seeds |
| Intention-Guided Prediction | Deducing human intent from gaze and gesture | Differentiating between a presenter explaining food and an onlooker about to taste it |

### 3.4 FeVOS-R1 Architecture: Eyes, Brain, and Hands Working in Harmony

![Figure 5: Overview of FeVOS-R1](/images/fevos/_page_8_Figure_2.jpeg)
*Figure 5: Overview of the FeVOS-R1 two-stage framework. Built upon Sa2VA, Stage 1 establishes structured reasoning via CoT-SFT, and Stage 2 directly optimizes segmentation quality via IoU-reward-guided GRPO.*

FeVOS-R1 adopts Sa2VA as its architectural backbone, integrating the multimodal large language model InternVL2.5 with the foundation segmentation model SAM2. The architecture operates as a harmonious union of eyes, brain, and hands.

Given an input video $V$ with a sampled sequence of $T$ frames $\{I_t\}_{t=1}^T$, information flows through three coordinated components:

First, the Visual Encoder serves as the assistant's retina, extracting spatio-temporal visual features from the video frames and mapping them into rich visual embeddings. Physical scene states, such as a soiled pot or the placement of hands on a counter, are digitized into structured representations.

Second, the Large Language Model acts as the assistant's cerebral cortex. It receives the visual embeddings alongside a natural language prompt $P$. Rather than immediately outputting a mask or a brief answer, the LLM first generates an autoregressive chain-of-thought text detailing antecedent cues and causal logic. Upon reaching its conclusion, it emits a special segmentation token [SEG].

Third, SAM2's Mask Decoder acts as the assistant's motor execution system. The hidden states of the [SEG] token are linearly projected into prompt embeddings for the decoder. Conditioned on this signal, the decoder scans across all observation frames to produce a pixel-level mask sequence $\{\hat{M}_t\}_{t=1}^T$ with precise boundaries.

Standard Sa2VA struggled with predictive reasoning because it lacked explicit causal pathways within its LLM, relying merely on standard supervised learning that matches language descriptions with co-occurring pixels. FeVOS-R1 resolves this limitation through a dedicated two-stage training paradigm.

### 3.5 Stage 1: Supervised Fine-Tuning with Chain-of-Thought (Constructing the Scaffolding)

If an untrained assistant is placed into a demanding kitchen without instruction, the novice is overwhelmed by chaotic stimuli. Similarly, applying reinforcement learning to a raw model without prior structure causes it to collapse into degenerate outputs across vast token and pixel spaces.

Therefore, a cold start via Supervised Fine-Tuning, abbreviated as SFT, is essential before reinforcement learning. SFT provides models with paired inputs and expert demonstrations, training model weights to replicate structured reasoning paths and segmentation masks. The authors fine-tune the model on the 2,904 synthetic CoT annotations.

During this stage, the model is trained to articulate an interpretable reasoning chain analyzing temporal context and causal relationships before emitting the [SEG] token. The objective combines three loss terms:

$$\mathcal{L}_{\text{total}} = \alpha_{\text{ce}} \mathcal{L}_{\text{ce}} + \alpha_{\text{dice}} \mathcal{L}_{\text{dice}} + \alpha_{\text{text}} \mathcal{L}_{\text{text}}$$

Each loss term refines a distinct capability:

- Pixel-wise Cross-Entropy Loss $\mathcal{L}_{\text{ce}}$: Evaluates whether each individual pixel belongs to the referent object or background.
- Dice Loss $\mathcal{L}_{\text{dice}}$: Mitigates extreme class imbalance between small target objects and large background regions, optimizing global shape alignment.
- Text Generation Loss $\mathcal{L}_{\text{text}}$: Supervises the autoregressive generation of the causal reasoning text preceding the [SEG] token.

Starting from pretrained Sa2VA-4B, LoRA with rank 128 is applied to both the LLM and the SAM2 mask decoder. The model is trained for 4 epochs using a learning rate of $2 \times 10^{-5}$ with cosine annealing, a batch size of 4, and gradient accumulation over 4 steps. This stage establishes the apprentice's fundamental ability to reason before acting.

### 3.6 Stage 2: Reinforcement Learning with End-to-End IoU Rewards (Sharpening Motor Precision)

While SFT instills basic reasoning patterns, memorized explanations do not necessarily translate into flawless physical grasping. If an assistant articulates logical steps yet grabs the wrong tool, the assistance fails. In practice, SFT reasoning trajectories remain suboptimal and loosely coupled to dense mask quality.

To bridge this gap, Stage 2 employs Group Relative Policy Optimization, or GRPO, optimizing the model's reasoning process directly for pixel segmentation accuracy.

#### 1. Value-Free Group Relative Policy Optimization

Conventional PPO algorithms maintain a heavy value network (Critic) to estimate state advantages. In high-dimensional multimodal video tasks, maintaining an additional critic alongside large vision-language models incurs prohibitive memory overhead.

GRPO eliminates the critic network entirely. For each query $q$, the policy generates a group of outputs $G = \{o_i\}_{i=1}^{|G|}$ where the group size $|G|$ is set to 4. After evaluating the reward $r_i$ for each response $o_i$, the advantage $A_i$ is normalized against the group's mean $\bar{r}_G$ and standard deviation $\sigma_G$:

$$A_i = \frac{r_i - \bar{r}_G}{\sigma_G}$$

By comparing four candidate reasoning-and-segmentation attempts against each other within the same scenario, the algorithm gauges relative excellence. The policy $\pi_{\theta}$ is updated by maximizing:

$$\mathcal{J}(\theta) = \mathbb{E}_G \left[ \frac{1}{|G|} \sum_{i \in G} \left( \min \left( s_i A_i, \operatorname{clip} \left( s_i, 1 - \epsilon, 1 + \epsilon \right) A_i \right) - \beta \mathbb{D}_{\mathrm{KL}}(\pi_{\theta} || \pi_{\text{ref}}) \right) \right]$$

The components in this objective function act as precision stabilizers:

- Probability Ratio $s_i = \frac{\pi_{\theta}(o_i|q)}{\pi_{\text{old}}(o_i|q)}$: Measures the likelihood ratio between the updated and previous policies.
- Clipping Operator $\operatorname{clip}(s_i, 1 - \epsilon, 1 + \epsilon)$: Restricts policy updates within $1 - \epsilon$ and $1 + \epsilon$ to prevent destructive policy divergence.
- Kullback-Leibler Divergence Penalty $\mathbb{D}_{\mathrm{KL}}(\pi_{\theta} || \pi_{\text{ref}})$: Constrains the updating policy $\pi_{\theta}$ from drifting excessively away from the frozen reference model $\pi_{\text{ref}}$ obtained from Stage 1 SFT. Because KL divergence measures distribution divergence, it is strictly non-negative. In reward-driven reinforcement learning, models can succumb to reward hacking or language drift, generating nonsensical gibberish that happens to trigger high mask scores. The negative sign subtracts a penalty whenever the model deviates from coherent language syntax, serving as a regularizer that confines policy exploration within acceptable linguistic bounds.

#### 2. Reward Innovation: Omitting Format Constraints for Pure End-to-End IoU Rewards

A pivotal design insight in FeVOS-R1 lies in its reward structure.

Previous vision-language reinforcement learning works often forced models to emit bounding boxes in rigid JSON formats, necessitating explicit format rewards to enforce bracket and tag compliance.

However, FeVOS-R1 connects the LLM's [SEG] token hidden states directly to SAM2's mask decoder in an end-to-end manner, bypassing intermediate JSON coordinate parsing. Consequently, mask accuracy can be optimized directly via an IoU reward:

$$R_{\text{IoU}} = \frac{1}{T} \sum_{t=1}^{T} \text{IoU}(\hat{M}_t, M_t)$$

where $\hat{M}_t$ and $M_t$ denote the predicted and ground-truth masks at frame $t$.

In our assistant analogy, once an apprentice has already learned standard kitchen etiquette during basic training, grading the apprentice on penmanship during a dinner rush only distracts from fetching the correct knife. Because FeVOS-R1's reasoning tags are lightweight and fully mastered during SFT, imposing format rewards dissipates model capacity on redundant structural constraints. Evaluating models purely on the physical accuracy of the grasped object directs reinforcement exploration entirely toward refining causal reasoning for mask fidelity.

During Stage 2, the SAM2 mask decoder is frozen, and only the LLM parameters are fine-tuned via LoRA for 2 epochs using a learning rate of $1 \times 10^{-5}$ on 4 NVIDIA RTX 4090 GPUs. This isolates training to sharpen the brain's predictive decisions while keeping motor execution stable.

---

## 4. Experimental Results

### 4.1 Benchmark Evaluation on FeVOS

We evaluate recent state-of-the-art video segmentation models on the FeVOS validation set using standard metrics: region similarity $\mathcal{J}$, boundary accuracy $\mathcal{F}$, and their mean $\mathcal{J}\&\mathcal{F}$.

| Method | Backbone | $\mathcal{J}$ | $\mathcal{F}$ | $\mathcal{J}\&\mathcal{F}$ |
|---|---|:---:|:---:|:---:|
| ReferFormer | ResNet-50 | 16.4 | 20.0 | 18.2 |
| LMPM | Swin-T | 17.1 | 20.7 | 18.9 |
| VISA | Chat-UniVi-7B | 22.9 | 28.2 | 25.6 |
| VideoLISA | LLaVA-Phi-3-V-3.8B | 22.9 | 29.4 | 26.1 |
| VideoGLaMM | Phi3-Mini-3.8B | 21.7 | 26.7 | 24.2 |
| VRS-HQ | Chat-UniVi-7B | 28.8 | 33.3 | 31.0 |
| GLUS | Chat-UniVi-7B | 27.4 | 31.7 | 29.6 |
| Sa2VA | InternVL2.5-4B | 23.7 | 27.2 | 25.4 |
| GLUS* | Chat-UniVi-7B | 31.0 | 35.9 | 33.5 |
| Sa2VA* | InternVL2.5-4B | 33.1 | 38.4 | 35.8 |
| FeVOS-R1 | InternVL2.5-4B | 39.5 | 45.1 | 42.3 |

Zero-shot models demonstrate severe deficiencies on predictive reasoning. Traditional RVOS methods designed for grounding explicit descriptions, such as ReferFormer (18.2) and LMPM (18.9), struggle markedly. Video-based MLLMs like VideoLISA (26.1) and VideoGLaMM (24.2) yield moderate improvements, while reasoning-oriented models like VRS-HQ (31.0) achieve the top zero-shot score.

Directly fine-tuning Sa2VA on FeVOS via standard supervised learning substantially improves performance from 25.4 to 35.8, demonstrating that domain adaptation captures temporal dynamics.

Integrating CoT-guided reasoning and IoU-guided RL, FeVOS-R1 achieves 42.3 $\mathcal{J}\&\mathcal{F}$, securing an additional +6.5 gain over the fine-tuned baseline. This highlights the indispensable role of explicit causal chains and reinforcement exploration in isolating subtle antecedent signals. Notably, the absolute score of 42.3 remains substantially lower than scores typical of retrospective benchmarks, reflecting the heightened difficulty of foresight visual understanding.

### 4.2 Generalization to Traditional RVOS Benchmarks

An astute assistant capable of anticipating future events naturally excels at analyzing past events. To evaluate cross-domain transferability, we benchmark FeVOS-R1 on ReVOS and MeViS without additional tuning.

| Method | Backbone | ReVOS Ref. | ReVOS Reas. | ReVOS All | MeViS $\mathcal{J}\&\mathcal{F}$ |
|---|---|:---:|:---:|:---:|:---:|
| VISA | Chat-UniVi-7B | 50.9 | 43.0 | 46.9 | 43.5 |
| VRS-HQ | Chat-UniVi-7B | 62.1 | 56.1 | 59.1 | 50.6 |
| GLUS | Chat-UniVi-7B | 58.3 | 51.4 | 54.9 | 51.3 |
| Sa2VA | InternVL2.5-4B | 62.5 | 55.6 | 59.1 | 46.4 |
| Sa2VA* | InternVL2.5-4B | 61.0 | 55.2 | 58.1 | 46.5 |
| FeVOS-R1 | InternVL2.5-4B | 62.8 | 57.8 | 60.3 | 49.5 |

The results are revealing. Standard fine-tuning of Sa2VA on FeVOS leads to domain overfitting, with ReVOS All scores slipping from 59.1 to 58.1. Conversely, FeVOS-R1 reaches 60.3, outperforming both zero-shot and fine-tuned baselines. The improvement is especially pronounced on the ReVOS Reasoning subset, rising from 55.2 to 57.8.

On MeViS, FeVOS-R1 reaches 49.5 $\mathcal{J}\&\mathcal{F}$, outperforming the fine-tuned baseline by 3.0 points and leading all comparable 4B models. This confirms that predictive spatio-temporal reasoning sharpens visual perception for complex, reasoning-intensive video understanding in general.

### 4.3 Ablation Studies

Ablation on training strategies demonstrates the complementarity of the two stages:

| Training Strategy | CoT-SFT | GRPO RL | $\mathcal{J}\&\mathcal{F}$ |
|:---:|:---:|:---:|:---:|
| SFT Only | ✓ | ✗ | 37.2 |
| RL Only | ✗ | ✓ | 36.0 |
| Two-Stage Pipeline | ✓ | ✓ | 42.3 |

Training with RL alone yields 36.0, underperforming SFT alone (37.2) because the unguided policy produces trivial responses without viable reasoning trajectories. Coupling SFT scaffolding with RL exploration unlocks the peak score of 42.3, verifying that SFT establishes structured reasoning while RL aligns reasoning with dense segmentation.

Ablations on reward configurations reinforce our design hypothesis:

| Reward Design | IoU Reward | Format Reward | $\mathcal{J}\&\mathcal{F}$ |
|:---:|:---:|:---:|:---:|
| Format Only | ✗ | ✓ | 37.7 |
| Joint Reward | ✓ | ✓ | 40.9 |
| IoU Only | ✓ | ✗ | 42.3 |

Relying exclusively on the IoU reward yields 42.3, surpassing the joint reward by 1.4 points and format reward alone by 4.6 points. Imposing format constraints when formatting has already been internalized during SFT needlessly diverts model capacity away from task-relevant spatio-temporal reasoning.

### 4.4 Qualitative Comparison

![Figure 6: Qualitative Comparison](/images/fevos/_page_13_Figure_2.jpeg)
*Figure 6: Qualitative comparison between the fine-tuned Sa2VA baseline and FeVOS-R1. Given the question "What will fly out?", the baseline incorrectly segments the entire bottle, whereas FeVOS-R1 reasons about internal pressure and cork behavior to accurately isolate the cork.*

Qualitative visualizations illustrate the power of explicit causal deduction. When asked "What will fly out?" during a champagne opening sequence, the baseline model fails to distinguish between the container and the ejected component, segmenting the entire bottle. FeVOS-R1 articulates an explicit reasoning chain analyzing internal pressure buildup and cork kinetics, isolating the cork with surgical precision.

Similarly, when asked what will soon be filled with clothes, FeVOS-R1 interprets the ongoing folding action to accurately identify the laundry bag rather than adjacent apparel.

---

## 5. Conclusion and Key Takeaways

FeVOS addresses a fundamental question in computer vision: toward which temporal horizon should intelligent perception be oriented? For years, referring video object segmentation focused on retrospective verification—segmenting objects that had already completed their actions within the frame. However, for robots collaborating with humans or autonomous systems avoiding collisions, retrospective vision offers little proactive utility. FeVOS establishes an anticipatory grounding benchmark, shifting artificial intelligence from passive observation to proactive foresight.

The two-stage framework of FeVOS-R1 offers a pragmatic blueprint for coupling large reasoning models with dense visual grounding. Raw reasoning capabilities in large language models risk producing superficial narratives without proper alignment. Grounding causal trajectories via CoT supervised fine-tuning and subsequently refining mask boundaries via pure IoU-reward reinforcement learning creates an effective bridge between cognitive deduction and motor precision.

Furthermore, eliminating format penalties in favor of end-to-end IoU rewards provides a valuable methodological takeaway for vision-language reinforcement learning. When foundational formatting is established during initial supervision, shedding auxiliary syntactic penalties allows the model to concentrate its representational capacity entirely on task objectives.

The concurrent performance gains achieved on traditional reasoning-intensive benchmarks like ReVOS and MeViS confirm that learning to anticipate future actions strengthens fundamental spatio-temporal understanding. Although an absolute score of 42.3 indicates that foresight pixel intelligence is still in its infancy, FeVOS marks a critical transition point for artificial agents evolving from passive observers into proactive partners.
