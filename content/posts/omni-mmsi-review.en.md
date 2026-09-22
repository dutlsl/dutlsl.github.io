---
title: "[arXiv 2025] Omni-MMSI: Reference-Guided Social Interaction Understanding from Raw Video"
date: 2026-09-22T11:53:32+09:00
draft: false
math: true
tags: ["Paper Review", "Social Interaction", "Multi-modal LLM", "Identity Attribution", "Chain-of-Thought", "arXiv 2025"]
categories: ["Paper Review"]
summary: "Unlike prior works assuming clean oracle social cues, Omni-MMSI formulates the task of multi-party social interaction understanding directly from raw audio-video, resolving identity attribution with reference profiles and structured CoT reasoning in Omni-MMSI-R."
cover:
  image: "/images/omni-mmsi/_page_0_Figure_10.jpeg"
  alt: "Omni-MMSI Overview"
---

> Reference Paper
> - Li, X., Lai, B., Chen, H. et al. "Omni-MMSI: Toward Identity-attributed Social Interaction Understanding." arXiv 2025.
> - Project Page: https://sampson-lee.github.io/omni-mmsi-project-page

![Omni-MMSI Overview Diagram](/images/omni-mmsi/_page_0_Figure_10.jpeg)
*Figure 1: Overview of the Omni-MMSI task and the Omni-MMSI-R pipeline. Unlike prior works that rely on idealized oracle social cues, Omni-MMSI tackles social interaction understanding directly from raw audio-video streams.*

---

## 1. One-Sentence Summary

Omni-MMSI establishes a realistic benchmark for understanding multi-party social interactions directly from raw, uncurated audio-video by simultaneously resolving identity attribution and conversational reasoning, and introduces Omni-MMSI-R, a reference-guided framework pairing specialized perceptual tools with structured chain-of-thought reasoning to achieve state-of-the-art performance.

---

## 2. Research Background and Motivation

### 2.1 Problem Definition

Imagine sitting around a table in a bustling cafe with five colleagues. Someone asks, "Did you finish the slide deck?" As a human participant, you instantly recognize who is asking and precisely whom they are addressing. You seamlessly integrate the acoustic timbre and spatial origin of the voice with visual cues like head orientation, gaze direction, and subtle conversational turn-taking. This effortless synthesis of multi-modal signals forms the bedrock of human social intelligence.

Multi-modal Multi-party Social Interaction Understanding (MMSI) is the research endeavor aimed at endowing artificial intelligence with this exact social perceptual ability. Given an audio-video stream of multi-party conversations, the objective is to extract communicative behaviors—such as transcribed speech, spatial trajectories, facial expressions, and mutual eye contact—and reason over them to comprehend the underlying social dynamics.

This work focuses on two fundamental social interaction tasks:
1. **Speaking Target Identification (STI)**: Determining the intended recipient of an utterance when a speaker uses second-person pronouns such as "you" or "your."
2. **Pronoun Coreference Resolution (PCR)**: Disambiguating which specific conversational participant is referenced by third-person pronouns such as "he," "she," or "they."

### 2.2 Limitations of Existing Methods

Returning to the cafe analogy, existing MMSI benchmarks operate under an unrealistic premise. They essentially hand the AI a pristine, pre-compiled dossier containing ground-truth speech transcripts, perfect speaker diarization, and exact bounding boxes for every participant across all video frames. Relieved of the burden of perceptual ground-truth extraction, models only need to solve the high-level reasoning problem of "who spoke to whom."

In real-world applications—such as embodied AI agents, social robotics, or smart glasses—no such oracle dossier exists. An AI agent is confronted with raw, unsegmented video streams and overlapping, noisy audio tracks. The system must independently isolate who spoke, locate where each individual is in 3D space, bind the acoustic voice to the visual persona, and infer social intentionality.

The primary bottleneck in this transition from oracle to raw input is **identity attribution**: the capability to correctly map extracted acoustic utterances and visual bounding boxes to consistent individual identities across modalities and time.

![Figure 2: Challenges in Omni-MMSI](/images/omni-mmsi/_page_2_Figure_0.jpeg)
*Figure 2: Sharp performance degradation when transitioning from oracle cues to raw inputs. Conventional pipelines drop by an average of 28.1%, while state-of-the-art Omni-LLMs drop by 9.52%. Representative identity attribution failure modes from Gemini 2.5 Pro illustrate spatial ordering bias and cross-modal mismatch.*

When existing MMSI models are evaluated on raw multi-modal inputs rather than oracle annotations, their accuracy collapses by an average of 28.1%. Even cutting-edge foundation models like Gemini 2.5 Pro suffer an average drop of 9.52%.

A closer inspection of Gemini 2.5 Pro's failure cases reveals two critical weaknesses:
- **Spatial Ordering Assumption**: In visual attribution, foundation models often assume participants are neatly arranged from left to right. When participants move, overlap, or experience occlusion, this spatial heuristic fails, causing identity swaps across the scene.
- **Cross-Modal Mismatch**: In acoustic attribution, speech recognition often transcribes the words accurately, but the model incorrectly binds the voice to an erroneous visual identity, decoupling the speaker from their physical presence.

### 2.3 Main Contributions

The principal contributions of this work are summarized as follows:

- **Formalization of Omni-MMSI**: The authors introduce Omni-MMSI, a realistic and challenging task formulation that requires AI models to understand multi-party social interactions directly from raw audio-video streams without oracle assistance.
- **The Omni-MMSI-R Framework**: A novel reference-guided framework is proposed. It employs external, specialized tools to extract identity-attributed verbal and non-verbal social cues using enrolled reference profiles, followed by structured Chain-of-Thought (CoT) reasoning within a fine-tuned Omni-LLM.
- **Curated Multi-Modal Reference Benchmark**: The authors establish 69 individual audio-visual reference profiles across two benchmark datasets (Ego4D and YouTube), along with structured CoT reasoning annotations to support future research.
- **State-of-the-Art Empirical Results**: Omni-MMSI-R outperforms prior pipelines by up to 15.1% in social interaction accuracy and surpasses leading Omni-LLMs by up to 23.7% in identity attribution accuracy, demonstrating the effectiveness of modular perceptual anchoring.

---

## 3. Proposed Framework

### 3.1 Core Idea and Overall Architecture

Extending the cafe analogy, if five strangers engage in conversation, distinguishing their voices and keeping track of their spatial movements is difficult. However, if each participant briefly introduces themselves beforehand—providing a headshot photograph and a five-second voice sample—an observer can use these registered profiles as reliable anchors to cross-reference ambiguous conversational cues.

Omni-MMSI-R adopts this exact human-inspired cognitive strategy. In practical smart-device ecosystems, such reference profiles are routinely available through enrollment or biometric authentication routines on smart glasses and video conferencing platforms.

![Figure 3: Omni-MMSI-R Pipeline Architecture](/images/omni-mmsi/_page_3_Figure_0.jpeg)
*Figure 3: Overall architecture of the Omni-MMSI-R pipeline. Task-specific tools extract identity-attributed verbal and non-verbal cues using reference audio-vision pairs. The fine-tuned Qwen2.5 Omni 7B model then receives the raw streams, references, and attributed cues to perform two-step Chain-of-Thought reasoning for final referent prediction.*

The framework operates via a coordinated two-stage design:
1. **Perceptual Cue Attribution**: Specialized tools leverage reference profiles to anchor and extract structured, identity-attributed communicative cues from raw sensor data.
2. **Cognitive Social Reasoning**: An Omni-LLM digests these attributed cues alongside raw sensory streams to perform multi-step social reasoning.

Formally, given a system prompt $P$, raw audio-video segment $I_{\text{AV}}$, and a set of reference profiles $\mathcal{R}$, the system objective is formulated as:

$$f: (P, I_{\text{AV}}, \mathcal{R}) \to X_{\text{answer}}$$

### 3.2 Reference Guidance

![Figure 4: Construction of Reference Pairs](/images/omni-mmsi/_page_3_Figure_2.jpeg)
*Figure 4: Construction of reference audio-vision pairs for each conversational participant. Upper-body image crops and five-second clean voice clips serve as anchors for cross-modal identity attribution.*

For each of the $N$ participants in a scene, the system maintains an enrolled reference set $\mathcal{R} = \{(a_i, v_i)\}_{i=1}^N$, where $a_i$ denotes a clean acoustic voice clip and $v_i$ represents an upper-body visual portrait. In total, 69 reference profiles were curated across the experimental datasets.

These reference profiles serve as persistent identity anchors across modalities and time. They mitigate common multi-party tracking pitfalls, including identity swaps under severe occlusion and cross-modal mismatch between unseen speech and visible faces.

### 3.3 Tool-Based Social Cue Extraction

Once reference profiles are established, Omni-MMSI-R deploys specialized, lightweight perceptual tools to extract identity-attributed social cues from the raw inputs:

- **Acoustic Tool Pipeline**: Whisper is first applied to transcribe the continuous audio stream into discrete, timestamped utterance segments. For each utterance, SpeechBrain extracts acoustic speaker embeddings and computes cosine similarities against all enrolled voice references in $\mathcal{R}$. The identity with the highest similarity is assigned as the speaker. This yields identity-attributed verbal cues specifying *who said what*.
- **Visual Tool Pipeline**: YOLO detects all visible human bounding boxes in the final frame of the video query. For each bounding box, OSNet extracts visual re-identification embeddings and measures cosine similarity against all enrolled portrait references in $\mathcal{R}$. The reference with the highest visual match determines the participant's identity. This yields identity-attributed non-verbal cues specifying *who is located where*.

In the cafe analogy, these tools act as automated registration desks: matching speech snippets to registered voiceprints and matching detected faces to photo IDs. Together, they construct a structured intermediate representation $\mathcal{S}$, establishing that Player2 is located at coordinates `[0.18, 0.47, 0.33, 0.78]` and uttered `"Do you need the script?"`.

Importantly, this tool extraction stage is strictly a perceptual pre-processing phase. It solves identity attribution—determining who spoke and where participants are—but does not yet determine *whom the speaker is addressing*. The high-level social interaction reasoning is executed in the subsequent stage.

### 3.4 Multi-Modal LLM with Structured CoT Inference

With the identity-attributed cue set $\mathcal{S}$ prepared, the raw audio-video segment $I_{\text{AV}}$, reference set $\mathcal{R}$, and system prompt $P$ are fed into an Omni-LLM to infer the social interaction:

$$X_{\text{answer}}, X_{\text{think}} = f_{\theta}^{\text{Omni-LLM}}(P, I_{\text{AV}}, \mathcal{R}, \mathcal{S})$$

In cognitive science and foundational natural language processing, the fundamental premise of Chain-of-Thought (CoT) reasoning is that complex, multi-hop problems cannot be solved reliably by directly mapping high-dimensional inputs to final outputs. Forcing an autoregressive model to predict a final answer immediately compresses all intermediate deduction into a single step, resulting in hallucinations and reliance on spurious correlations. Generating explicit intermediate reasoning tokens provides a computational scratchpad, allowing each deduction step to serve as a conditioning context that sharpens the model's multi-modal attention onto relevant evidence.

In multi-party social interaction understanding, direct prediction frequently causes models to confuse the speaker with the referent or choose candidates based on superficial camera angles. Omni-MMSI-R addresses this by structuring the reasoning trace $X_{\text{think}}$ into two sequential, causal phases:
1. **Last Speaker Confirmation**: The model cross-examines the audio embedding similarity and lip synchrony in the final video frames to definitively confirm the active speaker's identity.
2. **Speaker's Referent Inference**: Conditioned on the confirmed speaker, the model analyzes conversational turn-taking dynamics, mutual eye contact (mutual gaze), body orientation, and deictic pointing gestures to deduce whom the speaker is addressing.

During runtime inference, the model produces structured reasoning traces before emitting the final answer:

> `<think>` Last speaker confirmation: The last speaker is Player2, confirmed by voice matching and active lip movement. Speaker's referent inference: Based on the turn-taking context of the dialogue and sustained mutual gaze between Player2 and Player3, Player2's question is directed toward Player3. `</think>` Final Response: Player3

By decomposing complex social intentionality into verifiable causal steps, the inference pipeline achieves both high accuracy and clear interpretability.

---

## 4. Experimental Results

To train the lightweight Qwen2.5-Omni-7B model to master this two-step CoT reasoning, the authors implemented a generate-and-filter data curation pipeline using Gemini 2.5 Pro as a teacher model.

![Figure 5: Construction of CoT Datasets](/images/omni-mmsi/_page_4_Figure_0.jpeg)
*Figure 5: CoT dataset curation pipeline. Gemini 2.5 Pro generates reasoning traces and answers, which undergo rejection sampling based on ground-truth consistency followed by lightweight human verification.*

Under this pipeline, reasoning traces generated by Gemini 2.5 Pro were retained only if they strictly concluded with the ground-truth target via rejection sampling (with up to 10 regeneration attempts per sample). A lightweight human review subsequently eliminated any reasoning traces containing visually contradictory statements. The student model, Qwen2.5-Omni-7B, was fine-tuned using LoRA with a learning rate of $1 \times 10^{-4}$, 3 training epochs, and a context window of 16,384 tokens. Query video segments averaged 14 seconds covering 5 conversational turns, with reference audio clips standardized to 5 seconds.

Evaluation was conducted on two distinct subsets of the Werewolf Among Us dataset: **YouTube** (3,255 STI and 2,679 PCR instances) and **Ego4D** (832 STI and 503 PCR instances), with an average of 5 participants per scene.

### 4.1 Social Interaction Understanding Performance

Omni-MMSI-R achieved state-of-the-art results across both benchmarks, attaining an average accuracy of 43.06% on Ego4D and 47.04% on YouTube. It surpassed conventional MMSI pipelines by 12.06% on Ego4D and 15.13% on YouTube, while outperforming Gemini 2.5 Pro by 5.36% on Ego4D.

| Pipeline | STI | PCR | Average Accuracy |
|---|:---:|:---:|:---:|
| Qwen2.5 Omni 7B | 26.29 | 28.57 | 27.43 |
| Gemini 2.5 Pro | 36.12 | 39.28 | 37.70 |
| Lee et al. | 28.98 | 32.14 | 30.56 |
| Li et al. | 29.73 | 32.27 | 31.00 |
| Omni-MMSI-R | 40.57 | 45.54 | 43.06 |

*Table 1: Main performance comparison on the Ego4D benchmark dataset.*

### 4.2 Identity Attribution Performance

Evaluating identity attribution accuracy in isolation highlights the decisive advantage of the reference-guided architecture. Omni-MMSI-R reached 78.79% attribution accuracy on Ego4D and 76.95% on YouTube, outperforming leading Omni-LLMs by 23.68% and 18.91%, respectively.

| Pipeline | Verbal Attribution | Non-Verbal Attribution | Average |
|---|:---:|:---:|:---:|
| OmniVinci | 54.04 | 27.42 | 40.73 |
| Qwen3 Omni 30B | 52.61 | 57.61 | 55.11 |
| Gemini 2.5 Pro | 44.75 | 26.52 | 35.64 |
| Omni-MMSI-R | 71.09 | 86.48 | 78.79 |

*Table 2: Identity attribution accuracy comparison on Ego4D.*

The non-verbal attribution score of 86.48% is particularly compelling, demonstrating that YOLO and OSNet combined with reference image matching anchor human identities across scene clutter with high precision.

### 4.3 Qualitative Comparison

![Figure 6: Qualitative Comparison](/images/omni-mmsi/_page_5_Figure_0.jpeg)
*Figure 6: Qualitative comparison between Gemini 2.5 Pro and Omni-MMSI-R. Gemini 2.5 Pro misattributes the utterance to the wrong visual identity, leading to an incorrect referent. Omni-MMSI-R correctly aligns verbal and non-verbal cues and verifies mutual gaze via CoT reasoning.*

In a representative qualitative example, Gemini 2.5 Pro erroneously attributes Player0's utterance (*"And so you saw that you were the Insomniac?"*) to Player3, predicting Player1 as the referent. In contrast, Omni-MMSI-R correctly grounds the speaker to Player0 through acoustic matching, and its CoT trace identifies that Player0 turns toward Player2 while Player2 reciprocates with mutual eye contact, correctly predicting Player2 as the referent.

### 4.4 Effects of Reference-Guided Input Components

An ablation study on the Ego4D dataset dissects the individual contributions of reference pairs and tool-extracted cues:

| Ref. Audio | Ref. Visual | Verbal Cues | Non-Verbal Cues | Average Accuracy |
|:---:|:---:|:---:|:---:|:---:|
| ✗ | ✗ | ✗ | ✗ | 33.97 |
| ✓ | ✓ | ✗ | ✗ | 35.98 |
| ✗ | ✗ | ✓ | ✓ | 39.44 |
| ✓ | ✓ | ✓ | ✓ | 43.06 |

*Table 3: Ablation study of input components on the Ego4D dataset.*

The raw audio-video baseline achieves 33.97%. Providing raw reference pairs alone increases accuracy to 35.98%, confirming that reference signals implicitly aid the model. Introducing tool-extracted cues without references boosts accuracy to 39.44%, showing that explicit identity attribution directly benefits social reasoning. Peak accuracy of 43.06% is achieved only when providing both reference pairs and attributed cues, indicating that the Omni-LLM cross-verifies tool-extracted cues against raw reference evidence to compensate for noisy detections.

### 4.5 Effectiveness of CoT Reasoning Depth

The impact of CoT reasoning granularity was evaluated on the Ego4D dataset:

| Reference Input | Reasoning Granularity | Average Accuracy |
|:---:|---|:---:|
| ✗ | Direct Output | 33.97 |
| ✗ | Standard CoT | 35.45 |
| ✓ | Direct Output | 39.41 |
| ✓ | 1-Step CoT | 39.70 |
| ✓ | 2-Step CoT | 43.06 |
| ✓ | 3-Step CoT | 34.43 |

*Table 4: Impact of CoT reasoning granularity on Ego4D.*

While introducing 2-step CoT reasoning elevates accuracy to 43.06%, extending the reasoning depth to 3 steps causes performance to plummet to 34.43%. The authors attribute this drop to three factors: excessively long generation traces dilute the model's attention away from key causal paths, explicit extraction of redundant social signals exceeds the perceptual capacity of the 7B backbone, and available training data is insufficient to supervise deep multi-stage reasoning.

---

## 5. Conclusion and Key Takeaways

Omni-MMSI exposes a critical gap in artificial social intelligence: identity attribution. While earlier benchmarks bypassed this challenge by supplying idealized oracle annotations, real-world deployment demands that models identify who spoke and where participants are directly from uncurated multi-modal streams.

Omni-MMSI-R addresses this challenge by mimicking human social cognition. When humans enter a group conversation, they register facial appearances and voices in working memory as anchors. The reference-guided architecture operationalizes this cognitive strategy by combining enrolled reference profiles with lightweight specialized tools, achieving a 23.68% margin in identity attribution accuracy on Ego4D.

Ablation results provide vital engineering insights for embodied AI systems. Expecting a single end-to-end multi-modal LLM to resolve both low-level perceptual attribution and high-level social reasoning is inefficient and prone to hallucination at smaller parameter scales. Offloading perceptual identity attribution to specialized, modular tools while reserving the LLM for structured reasoning provides a far more robust and scalable paradigm. Furthermore, the sharp performance drop observed in 3-step CoT demonstrates that reasoning depth must be carefully calibrated to model capacity and data scale. Omni-MMSI-R proves that principled cognitive architectures often triumph over brute-force model scaling in complex social perception.
