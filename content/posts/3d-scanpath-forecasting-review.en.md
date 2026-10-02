---
title: "[CVPR 2026] Forecasting 3D Scanpaths in Egocentric Video"
date: 2026-10-02T18:41:00+09:00
draft: false
math: true
tags: ["Paper Review", "GAZE 2026", "Gaze", "Gaze Forecasting", "Scanpath Prediction", "Egocentric Vision", "3D Gaze", "Transformer", "CVPR 2026"]
categories: ["GAZE 2026", "Paper Review"]
summary: "We formulate the novel task of forecasting future 3D gaze scanpaths in egocentric video in world coordinates by integrating head pose and visual context into a canonical-frame Transformer architecture."
cover:
  image: "/images/3d-scanpath-forecasting/_page_2_Figure_0.jpeg"
  alt: "3D Scanpath Forecasting Architecture Overview"
---

> Reference Paper
> - Ryan, F., Ananthabhotla, I., Qian, Y., Hoffman, J., Rehg, J. M., Ithapu, V. K., Murdock, C. "Forecasting 3D Scanpaths in Egocentric Video." CVPR 2026.
> - Affiliations: Georgia Institute of Technology, Meta Reality Labs Research, UC Irvine, UIUC

![Figure 1: Architecture Overview](/images/3d-scanpath-forecasting/_page_2_Figure_0.jpeg)
*Figure 1: Overall pipeline of the proposed architecture. Global image features from past video frames, dense patch features from the canonical frame, and 6-DoF head pose features are fused in the Visual Context Encoder. The Trajectory Decoder then processes observed 3D scanpaths and learnable query tokens via cross-modal attention to forecast future 3D scanpaths.*

---

## 1. One-Sentence Summary

Transcending conventional 2D scanpath prediction on static images, this paper defines the novel task of forecasting future 3D gaze fixation locations and dwell durations in world coordinates from egocentric video and head pose, proposing a canonical-frame Transformer architecture.

---

## 2. Research Background and Motivation

### 2.1 Problem Definition

Consider a sports broadcast camera director preparing the next sequence of shots during a live soccer match. The camera tracking the match continually pans and tilts across the stadium pitch. When the camera abruptly pans toward the left touchline, a defender in the penalty box who occupied the center of the frame moments earlier vanishes beyond the screen boundary. If one attempts to predict the next location of action solely based on pixel coordinates on the camera screen, the coordinates of the exact same player will fluctuate wildly depending on which direction the camera pans, rendering consistent forecasting impossible.

The solution to this dilemma is straightforward: instead of forecasting in pixel coordinates of the camera viewport, one establishes predictions relative to fixed absolute coordinates on the stadium pitch. Regardless of how the camera pans or rotates, the center circle coordinates remain constant, allowing predictions across disparate moments to be compared and linked coherently.

Predicting the gaze of a user wearing AR/VR glasses faces an identical geometric dilemma. As the wearer walks through an environment and turns their head, the coordinate frame of the wearable camera translates and rotates continuously. Prior scanpath prediction literature has almost exclusively operated under laboratory settings where stationary observers inspect static 2D pictures displayed on a desktop monitor. In that static world, the subject's head never rotates and objects never leave the field of view. This fundamentally diverges from real-world human vision, where observers physically walk, turn their heads, and explore dynamic 3D environments.

This paper formally introduces the task of Egocentric 3D Scanpath Forecasting, aiming to predict sequences of future 3D gaze fixations in world coordinates given past gaze trajectories, egocentric video frames, and head pose data.

| Core Dilemma | Physical Phenomenon in Live Broadcasting | Functional Requirement for 1st-Person Gaze Models |
|---|---|---|
| Coordinate Instability | Panning the camera alters the pixel position of the same player | The model must predict in environment-anchored 3D world coordinates |
| Field-of-View Exit | Rotating the lens drops tracked players out of frame | The model must track and anticipate positions outside the current view frustum |
| Open Temporal Continuity | The match runs continuously for 90 minutes without arbitrary cuts | Unlike viewing static images, continuous gaze behaviors must be forecast from partial observations |

### 2.2 Limitations of Existing Methods

First, existing scanpath models are bound to stationary 2D images. Landmark architectures developed on datasets such as MIT1003 or COCO-Search18, including IOR-ROI LSTM, DeepGaze III, and PathGAN, assume a seated participant inspecting a picture for a brief 3-second window. The observer cannot turn their head or walk around. Even extensions to 360-degree equirectangular panoramas remain restricted to directional angular forecasting from a static tripod viewpoint, failing to address metric 3D coordinates in physical space.

Second, prior egocentric gaze forecasting approaches formulate future gaze targets on unobserved future 2D frames. Landmark works such as Deep Future Gaze follow this paradigm. Returning to the sports broadcasting analogy, this forces the camera director to solve a compounding dual prediction task: first guessing where the camera will be oriented three seconds into the future, and simultaneously estimating where the ball will land within that uncaptured viewpoint. If either prediction falters, the entire trajectory breaks down. Defining targets in environment-anchored coordinates eliminates this dual dependency because the physical coordinates of the target remain invariant to camera motion.

Third, point process formulations successful on static images, such as TPP-Gaze, conflict with the intrinsic dynamics of egocentric vision. When observing static pictures, human eyes execute broad, discontinuous saccades across disparate regions. In egocentric vision, however, the observer actively rotates their head to align objects of interest near the optical axis, inducing a strong center bias that produces smooth, clustered trajectories. Adapting static point process models to 3D fails to capture this continuous behavioral pattern.

### 2.3 Main Contributions

1. Formulated the task of Egocentric 3D Scanpath Forecasting, predicting future 3D gaze fixation locations and durations in world coordinates from first-person video and head pose.
2. Introduced a canonical coordinate frame anchored at the final observation camera pose to eliminate rotational distortion, coupled with an integrated Transformer architecture fusing visual context, head pose, and past scanpaths.
3. Conducted extensive benchmarking and ablation studies on the Aria Digital Twin dataset, demonstrating that multi-frame visual context and 6-DoF head pose are pivotal for reducing 3D depth errors.

---

## 3. Proposed Framework

### 3.1 Pipeline Overview and the Broadcast Director Analogy

A trajectory forecasting system operating on a sports pitch requires four coordinated mechanisms:

First, an absolute coordinate system must be established. Because pitch landmarks remain static regardless of lens orientation, action locations can be logged systematically on a single map. The proposed framework implements this by defining a canonical coordinate frame centered at the final observed camera position.

Second, the system must integrate granular spatial details of the current shot, historical memory of preceding plays, and real-time camera orientation. These correspond to the $16 \times 16$ dense patch tokens of the canonical frame, global context vectors from past frames, and 7-DoF relative head pose embeddings.

Third, the director plans the upcoming ten-shot sequence simultaneously rather than iteratively, preventing early errors from propagating across subsequent transitions. The model implements this through $N_f$ learnable query tokens and bidirectional attention.

Fourth, because locating players accurately takes precedence over the exact shot duration, the training objective heavily weights 3D position accuracy over dwell time.

### 3.2 Formulating Canonical Coordinates

#### 1. Defining Observed Scanpaths: Equation 1

To resolve coordinate instability induced by camera motion, gaze observations must be grounded in physical 3D space. The observed scanpath comprises $N_o$ discrete fixations:

$$S_o = \{g_1, g_2, \dots, g_{N_o}\}$$

Each fixation $g_i \in \mathbb{R}^4$ is parameterized by a 3D position and dwell duration:

$$g_i = (x_i^W, m_i)$$

Here, $x_i^W \in \mathbb{R}^3$ represents the 3D fixation coordinate in the world frame, capturing physical intersection points in the environment. The scalar $m_i \in \mathbb{R}$ denotes dwell duration in seconds. Combining them produces a 4-dimensional vector.

The target output is a sequence of $N_f$ future fixations:

$$S_f = \{g_{N_o+1}, g_{N_o+2}, \dots, g_{N_o+N_f}\}$$

Fixing $N_f = 10$ isolates spatial sequence modeling from temporal duration variations, treating duration as an auxiliary regression task.

#### 2. Canonical Coordinate Transformation: Equation 2

The key to resolving viewpoint rotation is anchoring coordinates to a stable local reference. The framework defines the canonical coordinate frame $C$ at the observer camera position $p_{N_o}^W$ of the final observation step:

$$S_o^C = \text{Transform}(S_o, p_{N_o}^W)$$

$$S_f^C = \text{Transform}(S_f, p_{N_o}^W)$$

All past observations and future targets are transformed into this shared coordinate frame. Setting the origin at the final observation step ensures minimal geometric variance relative to upcoming fixations. Consequently, targets that leave the current visual frustum remain continuous and well-defined in 3D space.

### 3.3 Visual Context Encoding: Integrating Local Detail and Temporal Flow

#### 1. Asymmetric Dense and Global Visual Encoding: Equation 3

A director preparing an upcoming camera move needs both high-resolution detail of the immediate frame and macroscopic memory of prior plays.

Given video stream $V$, frames $\{v_1, v_2, \dots, v_{N_o}\}$ corresponding to each observation fixation are sampled and fed into a frozen DINOv2-B backbone $\psi$.

For canonical frame $v_{N_o}$, all $16 \times 16 = 256$ dense patch tokens are retained to preserve spatial arrangements of nearby candidate targets. For preceding frames $v_{1:N_o-1}$, only a single global feature token per frame is extracted. Preserving all dense patches across past frames would cause computational complexity to escalate linearly with time, whereas coarse semantic context suffices to track the wearer's general environmental path.

$$F_{\text{dense}} = E_{\text{dense}} \cdot \psi_{\text{patch}}(v_{N_o}), \quad F_{\text{global}} = E_{\text{global}} \cdot \psi_{\text{global}}(v_{1:N_o-1})$$

Linear projection matrices $E_{\text{dense}}, E_{\text{global}} \in \mathbb{R}^{d_\psi \times d}$ compress DINOv2 feature dimensions from $d_\psi = 768$ down to internal Transformer dimension $d = 256$, filtering task-irrelevant visual noise.

#### 2. 7-DoF Head Pose and Temporal Positional Embeddings: Equation 4

In sports broadcasting, camera orientation provides critical context for anticipated motion. Knowing whether the lens points toward the goal or mid-pitch dictates where action will develop next.

Transforming absolute camera poses $p_i^W$ into canonical coordinates yields relative 7-DoF pose vectors $p_i^C$:

$$p_i^C = (q_i^C, t_i^C) \in \mathbb{R}^7$$

Here, $q_i^C \in \mathbb{R}^4$ is a unit quaternion representing 3D head rotation, and $t_i^C \in \mathbb{R}^3$ denotes relative 3D translation from the canonical origin.

A linear projection $E_{\text{pose}} \in \mathbb{R}^{7 \times d}$ maps these vectors to dimension $d = 256$, concatenating them with visual features to form visual context sequence $C_{\text{visual}}$:

$$C_{\text{visual}} = \text{Concat}([F_{\text{dense}}, F_{\text{global}}, E_{\text{pose}} \cdot P^C])$$

Sinusoidal temporal embeddings are added to synchronize visual tokens and pose vectors belonging to identical timestamps. The sequence is processed by a 2-layer self-attention Transformer encoder:

$$C'_{\text{visual}} = \text{TransformerEncoder}_{\text{2-layer}}(C_{\text{visual}} + \text{Pos}_{\text{temporal}})$$

Through self-attention, canonical patch tokens cross-reference historical global tokens, while head pose injects geometric constraints into the visual representations.

### 3.4 3D Scanpath Decoding: Holistic Sequence Generation

#### 1. Trajectory Tokens and Learnable Queries: Equation 5

Rather than planning camera shots one by one, a director designs the entire transition sequence holistically. Sequential greedy choices risk compounding early directional errors into catastrophic drift.

Observed fixation sequence $S_o^C$ is linearly projected to dimension $d = 256$, yielding trajectory sequence $C_{\text{traj}}$. This sequence is concatenated with $N_f$ learnable query tokens $Q$:

$$Q = \{q_1, q_2, \dots, q_{N_f}\}, \quad q_i \in \mathbb{R}^d$$

Each query $q_i$ acts as a dedicated slot initialized to decode the $i$-th future fixation.

#### 2. Bidirectional Decoding and Final Projection: Equation 6

The combined sequence $[C_{\text{traj}}, Q]$ is input to a 2-layer Transformer decoder:

$$[C'_{\text{traj}}, Q'] = \text{TransformerDecoder}_{\text{2-layer}}([C_{\text{traj}}, Q], C'_{\text{visual}})$$

Within the decoder, query tokens perform self-attention across all future steps simultaneously, while cross-attention gathers spatial cues from encoded context $C'_{\text{visual}}$.

Conventional autoregressive scanpath models generate fixations step by step, allowing early inaccuracies to accumulate. Bidirectional attention circumvents error accumulation by jointly optimizing all $N_f$ fixations into a coherent physical path.

Decoded queries $Q'$ are projected to final predictions via a linear head:

$$S_f^{\text{pred}} = \text{Proj}(Q')$$

$$g_i^{\text{pred}} = (x_i^{C_{\text{pred}}}, m_i^{\text{pred}}) \in \mathbb{R}^4$$

Each predicted fixation specifies a 3D coordinate and duration in canonical space.

### 3.5 Multi-Task Loss and Implementation Details

#### 1. Weighted Multi-Task Objective: Equation 7

Tracking target positions accurately is fundamentally more critical than estimating exact dwell times. The multi-task objective balances spatial regression against temporal duration estimation:

$$\mathcal{L}(S_f^{\text{pred}}, S_f) = \lambda_1 \mathcal{L}_{\text{pos}}(X_f^{\text{pred}}, X_f) + \lambda_2 \mathcal{L}_{\text{dur}}(M_i^{\text{pred}}, M_i)$$

Hyperparameters $\lambda_1$ and $\lambda_2$ dictate relative task weighting, with $\lambda_1$ prioritized.

#### 2. Position Loss Formulation: Equation 8

$$\mathcal{L}_{\text{pos}} = \frac{1}{N_f} \sum_{i=1}^{N_f} \|x_i^{C_{\text{pred}}} - x_i^C\|_2^2$$

Mean squared error penalizes Euclidean deviations between predicted and ground-truth coordinates. Because the squared penalty scales quadratically with distance, large spatial offsets are heavily suppressed across all $N_f$ steps.

#### 3. Duration Loss Formulation: Equation 9

$$\mathcal{L}_{\text{dur}} = \frac{1}{N_f} \sum_{i=1}^{N_f} (m_i^{\text{pred}} - m_i)^2$$

Dwell duration errors in seconds are penalized via squared error. Because $\lambda_2 < \lambda_1$, spatial positioning remains the primary optimization signal.

#### 4. Implementation Details

Both observation length $N_o$ and prediction horizon $N_f$ are set to 10. Images are resized to $224 \times 224$, yielding a $16 \times 16 \times 768$ feature map from DINOv2-B. Hidden dimension $d = 256$. Optimization utilizes AdamW with learning rate $2 \times 10^{-4}$ and weight decay $1 \times 10^{-2}$ across 3 epochs over 87,000 sliding-window sub-sequences.

---

## 4. Experimental Results

### 4.1 Datasets and Evaluation Protocol

Evaluations are conducted on the Aria Digital Twin (ADT) dataset collected via Project Aria smart glasses, encompassing 30Hz video, 30Hz eye-tracking, 6-DoF SLAM trajectories, and dense 3D meshes. Ground-truth 3D fixations are derived by intersecting gaze vectors with the reconstructed 3D surface mesh. The dataset splits 184 sequences into 147 train, 18 validation, and 19 test sequences, with testing performed on 646 non-overlapping segments averaging 2.8 seconds.

Metrics include Dynamic Time Warping (DTW), Euclidean Distance (EUC), Frechet Distance (FRE), Eyeanalysis (EYE), and Time Delay Embedding (TDE) in meters. MultiMatch metrics assess Shape (Sh), Direction (Dir), Length (Len), Position (Pos), and Duration (Dur).

### 4.2 3D Scanpath Forecasting Performance

| Method | DTW↓ | EUC↓ | FRE↓ | EYE↓ | TDE↓ | Sh↓ | Dir↓ | Len↓ | Pos↓ | Dur↓ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Dataset average | 2.014 | 2.014 | 2.948 | 3.199 | 1.445 | 0.520 | 1.247 | 0.325 | 1.975 | 0.713 |
| Center prior at avg depth | 2.035 | 2.035 | 2.962 | 3.249 | 1.459 | 0.520 | 1.247 | 0.325 | 2.004 | - |
| Average of observations | 1.816 | 1.816 | 2.781 | 2.773 | 1.310 | 0.520 | 1.247 | 0.325 | 1.779 | - |
| Last observed point | 1.533 | 1.533 | 2.646 | 1.977 | 1.310 | 0.520 | 1.247 | 0.325 | 1.504 | - |
| Linear extrapolation | 4.642 | 4.660 | 8.133 | 5.506 | 1.504 | 0.984 | 1.568 | 0.603 | 4.309 | - |
| TPP-Gaze + GT depth | 3.278 | 3.309 | 4.668 | 4.451 | 1.917 | 1.244 | 1.381 | 1.008 | 3.174 | 0.776 |
| TPP-Gaze (3D modified) | 1.972 | 2.102 | 3.245 | 2.158 | 0.947 | 1.347 | 1.191 | 0.938 | 1.840 | 0.598 |
| Ours - Trajectory only | 1.450 | 1.456 | 2.280 | 1.901 | 0.930 | 0.507 | 1.254 | 0.406 | 1.419 | 0.654 |
| Ours - Single image, no pose | 1.410 | 1.421 | 2.212 | 1.836 | 0.860 | 0.509 | 1.185 | 0.377 | 1.367 | 0.671 |
| Ours - Single image, pose | 1.402 | 1.421 | 2.163 | 1.823 | 0.852 | 0.506 | 1.191 | 0.364 | 1.368 | 0.698 |
| Ours - Video, no pose | 1.395 | 1.412 | 2.215 | 1.814 | 0.853 | 0.503 | 1.188 | 0.373 | 1.344 | 0.740 |
| Ours - Video, pose (Full) | 1.377 | 1.382 | 2.173 | 1.800 | 0.859 | 0.507 | 1.194 | 0.384 | 1.350 | 0.630 |

A notable takeaway is the competitiveness of the Last Observed Point baseline. While static 2D scanpaths feature broad exploratory leaps, egocentric viewing exhibits strong center bias due to natural head movements centering objects in the viewport. The final observed gaze point in canonical space frequently aligns near this center, making static persistence surprisingly hard to outperform. This confirms that 2D heuristics cannot be directly transferred to 3D egocentric tasks without careful calibration.

Adapting TPP-Gaze to predict 3D coordinates results in an EUC of 2.102, over 1.5 times worse than the proposed model (1.382), underscoring that stochastic point process models struggle with smooth continuous scanpaths.

Ablations highlight incremental benefits across modalities: the trajectory-only baseline achieves EUC 1.456, single images improve this to 1.421, and adding head pose reduces FRE from 2.212 to 2.163. The full model combining video and pose achieves the best overall EUC of 1.382.

### 4.3 Axis-Wise Geometric Analysis and 2D Projections

| Video Input | Head Pose Input | X-Y Plane Error | Z Depth Error |
|:---:|:---:|:---:|:---:|
| × | × | 0.916 (+0.018) | 0.927 (+0.037) |
| ✓ | × | 0.907 (+0.009) | 0.918 (+0.028) |
| ✓ | ✓ | 0.898 | 0.890 |

Decomposing errors along individual axes reveals that head pose reduces X-Y planar error by 0.009 meters, whereas it reduces Z-axis depth error by 0.028 meters—a threefold improvement. While monocular video alone offers ambiguous depth cues, 6-DoF head motion provides critical motion parallax cues that resolve depth uncertainty.

| Training Domain | Method | DTW↓ | EUC↓ | FRE↓ | EYE↓ | TDE↓ |
|---|---|:---:|:---:|:---:|:---:|:---:|
| 2D (zero-shot) | TPP-Gaze | 0.151 | 0.165 | 0.266 | 0.160 | 0.071 |
| 2D (finetuned) | TPP-Gaze | 0.121 | 0.125 | 0.209 | 0.133 | 0.058 |
| 2D (finetuned) | Ours | 0.115 | 0.119 | 0.176 | 0.165 | 0.079 |
| 3D (finetuned) | TPP-Gaze | 0.128 | 0.136 | 0.214 | 0.142 | 0.061 |
| 3D (finetuned) | Ours (Projected to 2D) | 0.088 | 0.091 | 0.151 | 0.105 | 0.051 |

Projecting 3D predictions onto canonical 2D image coordinates yields an EUC of 0.091, decisively surpassing the same model trained directly in 2D space (0.119). Enforcing 3D geometric consistency acts as a powerful regularizer, yielding superior 2D predictions compared to direct 2D training.

### 4.4 Context Length and Attention Structure

| Observed Fixations | DTW↓ | EUC↓ | FRE↓ | EYE↓ | TDE↓ |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 1.720 | 1.666 | 2.524 | 2.561 | 1.166 |
| 1 | 1.450 | 1.461 | 2.248 | 1.923 | 0.880 |
| 3 | 1.425 | 1.430 | 2.217 | 1.877 | 0.882 |
| 5 | 1.417 | 1.429 | 2.217 | 1.871 | 0.881 |
| 10 | 1.377 | 1.382 | 2.173 | 1.800 | 0.859 |

Providing a single observed fixation dramatically reduces error from EUC 1.666 to 1.461 by grounding trajectory origination. Extending observations to 10 fixations captures curvature and velocity trends, lowering EUC to 1.382.

| Decoder Attention Scheme | DTW↓ | EUC↓ | FRE↓ | EYE↓ | TDE↓ |
|---|:---:|:---:|:---:|:---:|:---:|
| Fully Causal | 1.419 | 1.427 | 2.234 | 1.857 | 0.895 |
| Partially Causal | 1.403 | 1.421 | 2.225 | 1.801 | 0.854 |
| Bidirectional (Ours) | 1.377 | 1.382 | 2.173 | 1.800 | 0.859 |

Comparing decoder attention mechanisms shows that bidirectional attention achieves EUC 1.382, outperforming causal autoregression (1.427). Because human gaze trajectories align with broader task goals, jointly optimizing the entire future sequence proves superior to sequential autoregressive prediction.

### 4.5 Qualitative Evaluation and Failure Modes

![Figure 2: Qualitative Comparison](/images/3d-scanpath-forecasting/_page_6_Picture_0.jpeg)
*Figure 2: Qualitative comparison. Observed scanpaths, Trajectory-only baseline, 3D modified TPP-Gaze, the proposed model, and ground truth projected onto the canonical frame. Circle radii indicate fixation duration.*

Trajectory-only models drift erratically based solely on momentum, detached from scene objects. TPP-Gaze exhibits erratic, scattered jumps inherited from static image priors. In contrast, the proposed method produces smooth paths grounded in meaningful scene geometry.

![Figure 3: Temporal Visual Context](/images/3d-scanpath-forecasting/_page_7_Figure_0.jpeg)
*Figure 3: Impact of temporal visual context. In scenarios with hand-object manipulation (Row 1) or substantial head motion (Row 2), the full model accurately anticipates interaction targets compared to single-frame baselines.*

When users manipulate objects or rotate their heads quickly, multi-frame context and head pose tracking ensure accurate alignment with moving visual targets.

![Figure 4: Failure Modes](/images/3d-scanpath-forecasting/_page_7_Figure_2.jpeg)
*Figure 4: Representative failure modes. Underpredicting trajectory range (Rows 1-2) and predicting plausible but unselected interaction targets (Row 3).*

The primary failure mode is underestimating scanpath dispersion. Daily activities feature long fixation clusters on single objects, biasing models toward conservative, low-movement predictions. Furthermore, in Figure 4 (Row 3), the model anticipated gaze shifting toward a bowl during a grasping motion, whereas the user shifted attention elsewhere. While the model's prediction was functionally plausible, it was penalized against the single ground truth. Because egocentric data intrinsically captures a single observer per trial, developing multi-observer benchmarks remains an important direction for future research.

---

## 5. Conclusion and Key Takeaways

This work fundamentally elevates scanpath forecasting from 2D pixel planes into 3D physical environments. While static 2D scanpath research has modeled gaze transitions within stationary monitor screens, real-world human vision involves continuous physical movement, head rotation, and 3D spatial exploration. By shifting the forecasting arena into 3D world space, this paper transitions scanpath modeling from controlled laboratory setups to embodied real-world environments.

The canonical frame formulation represents a critical engineering insight. Predicting gaze in frame-varying 2D coordinates forces models to simultaneously forecast viewpoint shifts and gaze targets. Anchoring coordinates to the final observed camera pose decouples head rotation from gaze targets, allowing the model to focus purely on spatial gaze forecasting.

The finding that models trained on 3D coordinates decisively outperform 2D baselines when projected back onto 2D image planes is especially significant. Learning 3D physical constraints provides an inductive bias that enhances lower-dimensional performance, demonstrating the value of embodied 3D representations.

While single-observer biases and subtle cross-modal contributions leave room for future architectural refinement, this work establishes the essential mathematical and structural foundation for 3D scanpath forecasting, marking a vital milestone for spatial computing and embodied artificial intelligence.
