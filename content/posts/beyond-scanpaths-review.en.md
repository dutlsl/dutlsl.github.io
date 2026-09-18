---
title: "Beyond Scanpaths: Graph-Based Gaze Simulation in Dynamic Scenes"
date: 2026-09-18T18:36:00+09:00
draft: false
math: true
tags: ["Paper Review", "Gaze Prediction", "Graph Neural Network", "Heterogeneous Graph Transformer", "Driver Attention", "Dynamical Systems", "GlimpseML 2025"]
categories: ["Paper Review"]
summary: "Modeling human gaze as a graph-based dynamical simulation in driving scenes, introducing a unified framework that generates raw gaze sequences, scanpath dynamics, and saliency maps from a single model."
cover:
  image: "/images/beyond-scanpaths/_page_0_Picture_11.jpeg"
  alt: "Beyond Scanpaths overview figure"
---

> Reference
> - Palmer, L., Palasek, P., Abdelkawy, H. "Beyond Scanpaths: Graph-Based Gaze Simulation in Dynamic Scenes." GlimpseML / Toyota Motor Europe.
> - Project Page: https://glimpse.ml/beyond-scanpaths

---

## 1. One-Sentence Summary

By formulating driving scenes, road geometry, and driver gaze as a spatiotemporal heterogeneous graph, learning relative relationships via the Affinity Relation Transformer, and predicting next-step gaze distributions with the Object Density Network, this paper presents a unified dynamical simulation framework that concurrently generates raw gaze timeseries, scanpath dynamics, and saliency maps from a single model.

---

## 2. Research Background and Motivation

### 2.1 Problem Definition

When humans observe dynamic visual environments, they do not distribute attention uniformly across every detail. In driving scenarios, a driver's gaze shifts rapidly toward suddenly cutting-in vehicles, tracks pedestrians crossing the street, and ignores static road surfaces. Human attention allocation is inherently an active dynamical process evolving in continuous interaction with the physical environment.

However, prior attention modeling frameworks largely strip away this rich temporal structure. Saliency map approaches estimate spatial gaze distributions at isolated timesteps to produce per-frame heatmaps, while scanpath methods predict discrete sequences of fixation coordinates. Both paradigms either treat smooth pursuit and transition dynamics between saccades implicitly or omit them entirely.

![Figure 1: System Overview](/images/beyond-scanpaths/_page_0_Picture_11.jpeg)
*Figure 1: Driving scenes are encoded as heterogeneous scene graphs, which are processed by the Affinity Relation Transformer to predict Gaussian mixture distributions for next-step gaze positions. This dynamical systems approach generates raw gaze timeseries, scanpaths, and saliency maps from a single model.*

In video stimuli, converting continuous raw gaze trajectories into discrete fixations introduces severe artifacts and data loss. Fixation detection algorithms were originally designed for static imagery; in dynamic videos, they routinely misclassify smooth pursuit as fixations or drop short fixations altogether. This preprocessing corrupts temporal continuity and creates a profound disconnect with real human oculomotor behavior.

### 2.2 Limitations of Existing Methods

Existing video gaze modeling literature can be divided into two primary categories.

The first category relies on saliency maps. Models utilizing CNN-LSTM backbones, vision transformers, or generative adversarial networks estimate 2D spatial gaze probability heatmaps for individual video frames. While driving-specific architectures incorporate optical flow, 3D convolutions, or spatial graph convolutional networks to boost spatial fidelity, they aggregate multiple observer trajectories into static density maps, thereby discarding individual gaze paths and temporal ordering entirely.

The second category focuses on scanpaths. Using diffusion models, transformers, Markov models, or reinforcement learning, these methods predict discrete sequences of fixation coordinates. Although temporal ordering is preserved, these models strictly depend on fragile fixation filtering during preprocessing. In dynamic scenes where fixation algorithms exhibit high error rates, the ground-truth training signal itself is inherently corrupted.

Only a single prior study has explored driving scanpath prediction, and that framework relies on a CNN-Transformer to generate spatial fixation sequences without predicting fixation durations.

Crucially, previous works treat saliency maps, scanpaths, and raw gaze sequences as disjoint modeling tasks requiring distinct architectures and separate training pipelines. A unified approach capable of learning the underlying gaze-generating process to produce all three representations from a single model has remained unexplored.

### 2.3 Main Contributions

The core contributions of this work are fourfold:

- Formulating driving scenes, traffic agents, and driver gaze as a unified gaze-centric spatiotemporal heterogeneous graph.
- Introducing the Affinity Relation Transformer, which directly injects pairwise relative relational features into graph attention.
- Designing the Object Density Network, an adaptive mixture density head that dynamically matches the number of Gaussian mixture components to scene object density.
- Releasing Focus100, a large-scale raw gaze dataset collected from 30 participants viewing egocentric driving video clips.

---

## 3. Proposed Framework

### 3.1 Spatiotemporal Heterogeneous Scene Graph

The framework begins by transforming continuous video frames into a spatiotemporal heterogeneous scene graph.

Given $T$ consecutive input frames, each detected traffic-relevant entity forms a node in the graph. In addition to standard physical objects such as vehicles, pedestrians, and traffic signs, the graph incorporates two specialized node types: a gaze node centered on the driver's foveal field of view at each timestep, and a structure node capturing drivable road topology.

| Node Type | Description |
|---|---|
| vehicle | Detected dynamic agents including cars, bicycles, motorcycles, buses, and trucks |
| person | Detected pedestrian bounding boxes |
| static | Static traffic infrastructure such as traffic lights and stop signs |
| gaze | Driver foveated field of view centered at observed gaze coordinates |
| structure | Drivable road surface segmentation |

Each node feature vector comprises 2D bounding box center coordinates, box geometry, detection confidence scores, a 128-dimensional appearance vector extracted from a pretrained VGG network, estimated monocular depth, and a one-hot semantic category label. For the gaze node, a bounding box scaled to 20% of image height and 10% of image width is constructed around the ground-truth gaze point to crop the appearance embedding.

Edges connecting nodes fall into two classes: bidirectional spatial edges linking all node pairs within the same timestep, and strictly causal temporal edges directed from past timesteps to future timesteps. Temporal edges span predefined time dilation intervals to capture multi-scale temporal context efficiently.

Every edge is parameterized by an affinity feature vector containing 3D coordinate displacement, temporal offset, and cosine appearance similarity between the connected nodes. Edges between gaze nodes and object nodes capture whether the driver tracked an object in the past or is currently attending to it, while temporal edges linking sequential gaze nodes encode gaze history as conditions for autoregressive rollout.

![Figure 2: Full Architecture](/images/beyond-scanpaths/_page_3_Figure_0.jpeg)
*Figure 2: Synchronized driving video and gaze data are converted into a spatiotemporal heterogeneous scene graph. ART blocks process the graph before ODN predicts a Gaussian mixture distribution for the next gaze position. Training optimizes negative log-likelihood of ground-truth gaze; during inference, autoregressive rollout produces raw gaze trajectories that are post-processed into scanpaths and saliency maps without additional training.*

### 3.2 Affinity Relation Transformer

Node and edge feature vectors undergo batch normalization and are mapped through node-type-specific linear projections, augmented by alternating sinusoidal temporal encodings.

The design of the Affinity Relation Transformer (ART) directly addresses a fundamental structural limitation in standard Heterogeneous Graph Transformers (HGT). HGT applies distinct linear projections for queries, keys, and values according to discrete node and edge categories, propagating messages via type-specific scaled dot-product attention. While this architecture differentiates node semantic types, its edge transformations depend solely on categorical edge labels rather than continuous pairwise attributes.

Consequently, all edges belonging to the same relation category share identical transformation matrices. Whether a vehicle is 2 meters directly ahead of the driver or 50 meters away in the periphery, standard HGT applies the exact same weight matrix, leaving continuous 3D spatial separation, elapsed time, and appearance similarity invisible to the attention calculation.

ART resolves this deficiency by mapping the continuous pairwise affinity vector $a_{i,j}$ into independent key and value embeddings that are injected directly into the attention mechanism.

![Figure 3: ART Architecture](/images/beyond-scanpaths/_page_4_Figure_0.jpeg)
*Figure 3: ART computes messages for each connected source node $v_j$ and destination node $v_i$, directly incorporating their relative affinity $a_{i,j}$. The relative affinity embeddings are highlighted in red.*

Specifically, each edge affinity vector $a_{i,j}$ passes through a two-layer encoder composed of a linear projection, batch normalization, ReLU activation, and a second linear projection, producing a $d$-dimensional key embedding $p^K_{i,j}$ and value embedding $p^V_{i,j}$. These embeddings are added directly to the HGT key and value vectors.

This mechanism can be understood intuitively through an everyday analogy of interpersonal communication in a workplace. Standard HGT behaves like evaluating someone based strictly on their job title or name badge. Treating every colleague with the exact same preset protocol ignores whether that colleague sits 50 centimeters away or across the hallway, or whether you spoke one minute ago versus three months ago.

In contrast, ART explicitly accounts for actual physical distance, conversational recency, and mutual familiarity alongside the job title.

First, adding the key embedding $p^K_{i,j}$ establishes attention priority. Just as someone speaking right next to you naturally commands immediate attention over someone shouting from afar, the spatial and temporal proximity encoded in the key directly modulates attention scores.

Second, adding the value embedding $p^V_{i,j}$ embeds relational context into the transmitted message itself. A whisper close by carries a vastly different meaning and urgency compared to a distant sound, ensuring the transmitted representation retains precise relative geometric and temporal context.

By combining source node features $x_j$ with continuous pairwise relative vectors, ART ensures both attention weights and aggregated messages explicitly reflect spatial, temporal, and visual affinity.

Each ART block adheres to the Pre-LN Transformer architecture: LayerNorm, ART attention, LayerNorm, and a two-layer feed-forward network. Type-specific learnable gating parameters $\lambda^\tau$ adaptively regulate residual connections. Stacking $L$ ART blocks forms the Graph Processor, whose final representations feed into the ODN.

### 3.3 Object Density Network

Node representations from the final timestep $T$ generated by the $L$-th ART block serve as input to the Object Density Network (ODN). ODN is an adaptive mixture density head that forecasts the driver's next gaze point at timestep $T+1$ as a 2D Gaussian Mixture Model.

Revisiting the meeting room analogy clarifies the essential architectural advantage of ODN. Having observed the colleagues and surroundings, an observer must now choose where to look next.

A standard Mixture Density Network (MDN) predetermines an arbitrary fixed number of candidate targets, such as 10 static points on the walls or floor, regardless of where people are actually seated. When only two people occupy the room, 10 fixed targets waste capacity; when dozens gather in a crowded auditorium, 10 targets are woefully insufficient to cover the scene.

Instead, ODN dynamically instantiates candidate gaze distributions matched exactly to the entities present in the room. For every node present in the scene graph at timestep $T$, ODN assigns exactly one Gaussian component ($K = |V_T|$). The updated node vector $x'_k$ passes through node-type-specific linear layers to predict the component's mean offset, standard deviations, correlation coefficient, and mixture weight $\pi_k$.

The component mean $\mu_k$ is anchored to the entity's actual image location $\mu^0_k$ adjusted by a small learned displacement $\Delta\mu_k$ bounded by 0.05. This corresponds to directing gaze toward a colleague's general vicinity while fine-tuning the fixation on their eyes or gestures.

This formulation naturally captures the dual modalities of human gaze shifts:

First, a dominant mixture weight $\pi$ on the current gaze node corresponds to fixation maintenance, holding attention near the current point of regard.

Second, elevated mixture weights $\pi$ on surrounding traffic entity nodes represent attentional shifts, redirecting gaze toward an emerging object or pedestrian.

The final predicted gaze distribution is defined as the weighted mixture across all candidate components:

$$p(x, y) = \sum_{k=1}^K \pi_k \mathcal{N}(x, y \mid \mu_k, \sigma_k, \rho_k)$$

### 3.4 Learning Objectives and Gaze Simulation

The entire framework is trained end-to-end by minimizing the negative log-likelihood of observed human gaze coordinates. Given a training batch of spatiotemporal scene graphs, the network maximizes the log probability of ground-truth gaze points $g^{\text{GT}}$ under the predicted Gaussian mixture distribution.

During inference, trajectories are simulated via autoregressive rollout.

This process mirrors the continuous flow of social interaction. Once an observer glances at a colleague, that chosen landing point becomes the new gaze origin for the subsequent timestep. The scene graph updates with this new gaze node, and the model samples the next fixation target from the updated Gaussian mixture, iteratively unrolling gaze trajectories over time.

These simulated continuous gaze coordinates readily yield multiple attention representations through simple post-processing:

Filtering the trajectory with standard fixation criteria extracts discrete scanpaths and fixation durations. Furthermore, running 50 stochastic rollout simulations per video clip and convolving accumulated fixation points with a Gaussian kernel constructs dense spatial saliency maps.

By modeling the underlying generative dynamics of raw gaze directly, the framework produces raw timeseries, scanpath dynamics, and saliency maps without requiring auxiliary models or task-specific fine-tuning.

---

## 4. Experimental Results

### 4.1 Focus100 Dataset

To address the limitations of existing driving gaze benchmarks, the authors curated and released Focus100.

Thirty participants fitted with 60Hz Tobii Pro Nano eye trackers watched 100 egocentric driving clips. Footage was recorded in Brussels and Leuven across urban, suburban, and highway environments over two weeks, capturing balanced distributions of traffic density. Viewers performed a hazard perception task mirroring the UK driving theory examination, engaging active attention comparable to real-world driving.

![Figure 4: Dataset Comparison](/images/beyond-scanpaths/_page_5_Figure_0.jpeg)
*Figure 4: Distributions of vehicle and pedestrian densities across Focus100, MAAD, and DR(eye)VE datasets. Focus100 provides superior diversity in pedestrian densities, offering twice the driving video duration and three times the gaze data volume of MAAD.*

While DR(eye)VE suffered from temporal alignment flaws and limited diversity, and MAAD offered improvements on a restricted scale, Focus100 delivers 100 total minutes of driving video paired with 7 to 12 raw gaze recordings per frame. It also includes bounding box annotations for hazard objects, facilitating rigorous evaluation of time-to-first-fixation on road hazards.

### 4.2 Quantitative Results

Model performance is evaluated across three complementary dimensions: temporal sequence alignment (DTW, Temporal Correlation, Levenshtein distance), scanpath dynamics (fixation duration, fixation rate, time-to-first-fixation), and spatial saliency fidelity (NSS, IG, AUC).

On Focus100, ART outperformed or matched all baseline models across temporal sequence metrics. It achieved the lowest DTW score of 42.31, indicating optimal temporal alignment with human trajectories, alongside the best LEV score of 1.23. Regarding scanpath dynamics, human observers exhibited an average fixation duration of 0.44s and a fixation rate of 1.61, while ART achieved 0.41s and 1.64 respectively, closely mirroring human behavior. Baseline architectures degraded significantly, generating unnaturally short fixation durations around 0.12s.

In spatial saliency metrics, ART attained NSS 4.864, IG 9.728, and AUC 0.945, surpassing models trained with explicit saliency supervision. This demonstrates that modeling raw continuous gaze dynamics inherently recovers accurate spatial attention density.

On the MAAD dataset, ART demonstrated an even clearer advantage, reaching a Temporal Correlation of 0.46 (matching the human benchmark of 0.42) with DTW and LEV closely matching human statistics.

### 4.3 Qualitative Analysis

![Figure 5: Qualitative Comparison](/images/beyond-scanpaths/_page_7_Figure_0.jpeg)
*Figure 5: Qualitative comparison of gaze sequences and saliency maps on a 15-second Focus100 clip. Left shows human gaze versus model trajectories; right displays observed human fixations alongside model saliency maps on the same frame. Blue markers denote detected fixations.*

Qualitative analysis highlights the natural realism of ART trajectories. The frequency and duration of fixations produced by ART closely replicate human gaze records, evidenced by continuous clusters of blue fixation markers.

In contrast, baseline models rarely produce sustained fixation clusters, merely generating jittery noise around the central field of view. Moreover, trajectory variance generated by ART mirrors the natural dispersion across human observers, whereas competing models collapse to over-smoothed mean paths.

### 4.4 Ablation Study

| Component | Temporal Window | Head | TC $\uparrow$ | DTW $\downarrow$ | LEV $\downarrow$ |
|---|---|---|---|---|---|
| ART | 20 | ODN | $0.22 \pm 0.05$ | $42.31 \pm 4.88$ | $1.23 \pm 0.10$ |
| HGT | 20 | ODN | $0.21 \pm 0.06$ | $42.72 \pm 5.17$ | $1.28 \pm 0.09$ |
| HEAT | 20 | ODN | $0.13 \pm 0.05$ | $59.50 \pm 7.17$ | $1.47 \pm 0.08$ |
| ART | 20 | MDN $k=10$ | $0.14 \pm 0.04$ | $44.78 \pm 4.80$ | $1.30 \pm 0.10$ |
| ART | 20 | MDN $k=20$ | $0.14 \pm 0.04$ | $45.69 \pm 3.51$ | $1.32 \pm 0.09$ |
| ART | 8 | ODN | $0.17 \pm 0.04$ | $43.46 \pm 4.50$ | $1.26 \pm 0.09$ |
| ART | 1 | ODN | $0.17 \pm 0.07$ | $42.35 \pm 5.24$ | $1.24 \pm 0.09$ |

The ablation experiments validate the contribution of each architectural component:

First, shortening the input temporal window $T$ from 20 to 8 or 1 markedly degrades sequence alignment with human gaze, confirming the necessity of long temporal context for resolving long-range oculomotor dependencies.

Second, replacing ART with standard HGT or HEAT causes noticeable performance declines. ART's relative affinity encoding explicitly supplies pairwise spatial, temporal, and appearance relationships, yielding far richer relational representations than architectures constrained to discrete entity types.

Third, output head ablation demonstrates the critical utility of ODN. Replacing ODN with standard MDN heads utilizing fixed component counts ($k=10$ or $k=20$) causes Temporal Correlation to collapse from 0.22 to 0.14. Because standard MDN assigns static component counts irrespective of scene layout, it over-parameterizes barren scenes while starving dense urban scenes of expressive capacity. Dynamic allocation via ODN proves essential for robust gaze distribution modeling.

---

## 5. Conclusion and Key Takeaways

The principal conceptual breakthrough of this work lies in reconceptualizing human gaze not as collapsed static saliency maps or filtered fixation points, but as an active dynamical system evolving in continuous interaction with the visual world. By organizing driving scene agents and driver gaze into a spatiotemporal heterogeneous graph and unrolling gaze trajectories via graph-based simulation, a single model trained once concurrently produces raw gaze timeseries, realistic scanpath dynamics, and spatial saliency heatmaps.

This integration rests on two architectural innovations. The Affinity Relation Transformer injects pairwise relative 3D displacement, temporal intervals, and appearance similarity directly into the graph attention mechanism, explicitly capturing interactions between driver gaze and road elements. Meanwhile, the Object Density Network dynamically assigns Gaussian mixture components to every entity present in the scene graph, seamlessly adapting distribution complexity from empty highways to crowded intersections.

By bypassing fixation preprocessing and learning directly from raw gaze signals, the framework preserves subtle oculomotor dynamics such as smooth pursuit. Empirical results demonstrate that ART achieves human-level fixation durations and rates while faithfully capturing spatial attention density without direct supervision.

The authors acknowledge key limitations: Focus100 was collected in a controlled laboratory simulator rather than on-road vehicles, ART depends on upstream perception stacks where detection failures can cascade into gaze errors, and high-level driver intent is not explicitly modeled. Incorporating explicit driver intentionality represents a compelling avenue for future research.
