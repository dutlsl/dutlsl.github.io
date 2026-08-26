---
title: "[arXiv 2026] Mamba-3: Improved Sequence Modeling using State Space Principles"
date: 2026-08-26T17:37:42+09:00
draft: false
math: true
tags: ["Paper Review", "State Space Model", "Mamba", "SSM", "Linear Attention", "Efficient Inference", "arXiv 2026"]
categories: ["Paper Review"]
summary: "A paper review of Mamba-3, which advances the performance-efficiency Pareto frontier by introducing three methodological innovations from the SSM perspective: exponential-trapezoidal discretization, complex-valued state transitions, and multi-input multi-output (MIMO) architectures."
cover:
  image: "/images/mamba3/_page_10_Picture_0.jpeg"
  alt: "Mamba-3 Architecture Overview"
---

> Paper Information
> - Title: Mamba-3: Improved Sequence Modeling using State Space Principles
> - Authors: Aakash Lahoti*, Kevin Y. Li*, Berlin Chen*, Caitlin Wang*, Aviv Bick, J. Zico Kolter, Tri Dao†, Albert Gu†
> - Affiliation: Carnegie Mellon University, Princeton University, Together AI, Cartesia AI
> - Venue: arXiv preprint (2026)
> - Code: [github.com/state-spaces/mamba](https://github.com/state-spaces/mamba)

![Mamba-3 Architecture Overview](/images/mamba3/_page_10_Picture_0.jpeg)

---

## 1. One-Sentence Summary

From an inference-first perspective, Mamba-3 advances the SSM recurrence of Mamba-2 along three complementary axes: exponential-trapezoidal discretization for higher expressivity, complex-valued state transitions to resolve state-tracking failures, and multi-input multi-output (MIMO) architectures to boost decoding hardware efficiency without increasing hidden state memory footprint. At the 1.5B scale, Mamba-3 MIMO achieves an average downstream accuracy improvement of +2.2 points over Transformers and +1.9 points over Mamba-2.

---

## 2. Research Background and Motivation

### 2.1. Problem Definition

As test-time compute scaling emerges as a central driver of large language model capabilities, inference efficiency has become a defining metric in neural architecture design. Standard Transformer architectures incur severe computational and memory penalties over long context windows due to the linear memory scaling of KV caching and the quadratic time complexity of full self-attention. Consequently, state space models (SSMs) and linear attention mechanisms operating with constant memory and linear time complexity have garnered substantial research focus.

### 2.2. Limitations of Existing Methods and Theoretical Background

While existing linear sequence models achieve superior theoretical efficiency, they exhibit pronounced expressivity and functional limitations:

| Limitation Area | Manifestation | Root Mechanism |
|---|---|---|
| Reduced Expressivity | Mamba-2 trails Mamba-1 on associative reasoning tasks | Discretization relies on first-order Euler approximation for state input integrals, yielding numerical imprecision |
| State Tracking Failures | Inability to solve simple discrete state-tracking tasks such as parity or modular arithmetic | Constraining state transition matrix $A$ to negative real scalars prevents rotational dynamics in state space |
| Decoding Hardware Inefficiency | GPU compute cores sit idle despite theoretical linear complexity during autoregressive token generation | Single-Input Single-Output (SISO) structures yield an arithmetic intensity of $\approx 2.5$ ops/byte, causing severe memory bandwidth bottlenecks |

The theoretical mechanisms underlying these limitations are elaborated below.

#### Negative Real Constraints on State Transition Matrix $A$ and Lack of Rotational Dynamics

The state transition matrix $A$ governs the decay rate of hidden state information as $h_{t-1}$ transitions to $h_t$ within the SSM block.

To prevent numerical divergence and ensure stable optimization, Mamba-1 and Mamba-2 restrict the eigenvalues of $A$ to strictly negative real scalars ($A_t = a_t I, a_t < 0$). Because the analytical solution of the continuous differential equation follows $\exp(At)$, a positive $A$ causes states to blow up exponentially over time, whereas a negative $A$ forces $\exp(At)$ into the range $(0, 1)$, ensuring stable exponential decay.

However, restricting $A$ strictly to the real domain forces the hidden state vector to undergo simple contraction toward zero without any directional rotation. In contrast, rotational dynamics leverage Euler's formula $e^{i\theta} = \cos \theta + i \sin \theta$ to rotate the state vector by an angle $\theta$ across a 2D plane. By rotating the state rather than shrinking its magnitude, properties such as phase, periodicity, and token occurrence counts can be preserved indefinitely across sequence steps.

This absence of rotational dynamics directly causes failure on exact state tracking. Unlike general language modeling that merely requires fuzzy historical context summarization, algorithmic state tracking requires exact discrete state tracking across an automata graph (e.g., parity classification of binary sequences or cyclic modular arithmetic).

To compute parity, each arrival of token 1 requires the state vector to rotate exactly 180 degrees ($\pi$ radians) to invert its sign ($h_t = -h_{t-1}$). Real-valued decaying SSMs cannot invert state signs and inevitably decay past state information toward zero as sequence length grows, reducing their test accuracy on algorithmic tasks to random guessing.

#### Heuristic Discretization in Mamba-1/2

Mapping continuous-time SSM differential equations to discrete token sequences requires discretization. Traditional continuous SSMs (such as S4) utilize Zero-Order Hold (ZOH) transformations to preserve exact exponential integrals.

However, to minimize computational complexity under input-dependent step sizes, Mamba-1 and Mamba-2 substituted ZOH in practice with a heuristic first-order exponential-Euler approximation. This rule was adopted for implementation convenience without formal numerical error bounds or theoretical derivation.

#### SISO Architectures and Decoding Memory Bandwidth Bottlenecks

Autoregressive decoding generates text sequentially by feeding previously generated tokens back into the model as input for subsequent steps.

Because subsequent tokens depend strictly on prior token outputs, decoding cannot parallelize across the time dimension. In this regime, Mamba-2 operates as a Single-Input Single-Output (SISO) system, processing one 1D scalar/vector input per channel to produce one output.

At every decoding step, the GPU must fetch the large hidden state tensor $h_{t-1} \in \mathbb{R}^{N \times P}$ from VRAM, perform a single vector-vector outer product with the incoming token, and write $h_t$ back to VRAM. While modern GPU Tensor Cores provide massive matrix multiplication compute density, VRAM memory bandwidth is orders of magnitude slower.

Because SISO performs only a minimal vector update after transferring the entire hidden state tensor across memory buses, its arithmetic intensity drops to $\approx 2.5$ ops/byte. Consequently, GPU Tensor Cores spend the vast majority of their time idling waiting for memory transfers, bottlenecking decoding throughput entirely on memory bandwidth.

### 2.3. Main Contributions

- Exponential-Trapezoidal Discretization: Establishes a generalized discretization framework for linear time-varying (LTV) SSMs, formalizing previous Euler approximations and deriving a 3-term recurrence based on second-order accurate trapezoidal quadrature.
- Complex-Valued SSMs via the RoPE Trick: Proves that complex state transitions are mathematically equivalent to applying data-dependent Rotary Position Embeddings (RoPE) directly to Key ($B$) and Query ($C$) projections, resolving state-tracking failures while maintaining real-valued state computation.
- Multi-Input Multi-Output (MIMO) Architecture: Upgrades decoding outer products to matrix multiplications, boosting arithmetic intensity and expanding compute FLOPs up to $4\times$ without increasing state memory size or wall-clock latency.
- Elimination of Short Causal Convolutions: Demonstrates that the implicit internal convolution induced by exponential-trapezoidal discretization combined with learned BC biases renders external short causal convolution layers completely redundant.

---

## 3. Proposed Framework: Mamba-3

### 3.1. Background: SSM Recurrence in Mamba-2

The continuous-time state space model is formulated as:

$$h'(t) = A(t) h(t) + B(t) x(t), \quad y(t) = C(t)^T h(t)$$

where $x(t)$ represents the external input sequence, $A(t)$ and $B(t)$ are internal control parameter tensors governing decay rates and input projections, and $y(t)$ is the projected output from the hidden state $h(t)$ via readout filter $C(t)$.

Discretizing this system yields the sequential recurrence of Mamba-2:

$$h_t = \alpha_t h_{t-1} + \gamma_t B_t x_t, \quad y_t = C_t^T h_t$$

where $\alpha_t = \exp(\Delta_t A_t)$ acts as the historical decay multiplier and $\gamma_t = \Delta_t$ weights the incoming token update.

To parallelize training across sequence lengths, the State Space Duality (SSD) framework represents this recurrence as a masked matrix transformation:

$$Y = (L \odot CB^T) X$$

The transformation matrix $L$ is a strictly lower-triangular matrix, ensuring that historical tokens cannot attend to future tokens.

Unrolling the sequential recurrence across three time steps ($t = 1, 2, 3$) demonstrates how $L$ encodes cumulative decay:
- Output at $t = 1$: $y_1 = \gamma_1 x_1$
- Output at $t = 2$: $y_2 = \alpha_2 y_1 + \gamma_2 x_2 = \alpha_2 \gamma_1 x_1 + \gamma_2 x_2$
- Output at $t = 3$: $y_3 = \alpha_3 y_2 + \gamma_3 x_3 = \alpha_3 \alpha_2 \gamma_1 x_1 + \alpha_3 \gamma_2 x_2 + \gamma_3 x_3$

This aligns into the $3 \times 3$ lower-triangular mask matrix $L$:

$$\begin{bmatrix} y_1 \\ y_2 \\ y_3 \end{bmatrix} = \begin{bmatrix} \gamma_1 & 0 & 0 \\ \alpha_2 \gamma_1 & \gamma_2 & 0 \\ \alpha_3 \alpha_2 \gamma_1 & \alpha_3 \gamma_2 & \gamma_3 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}$$

Each row of $L$ implements a cumulative decay gate:
- Row 1: Only $x_1$ is incorporated with weight $\gamma_1$; future inputs $x_2, x_3$ are strictly zeroed.
- Row 2: Past input $x_1$ is attenuated by decay factor $\alpha_2$, while current input $x_2$ is added with weight $\gamma_2$.
- Row 3: Two-step historical input $x_1$ is attenuated by chained product $\alpha_3 \alpha_2$, one-step historical input $x_2$ is attenuated by $\alpha_3$, and current input $x_3$ is added.

Through matrix $L$, sequential recurrence updates are mapped directly into a single parallel matrix multiplication on GPUs.

### 3.2. Exponential-Trapezoidal Discretization

![Exponential-Trapezoidal Discretization](/images/mamba3/_page_3_Figure_0.jpeg)

*Figure 1: (Left) The structured mask induced by the exponential-trapezoidal rule decomposes into the product of a decay matrix and a 2-band convolutional matrix. (Right) Comparison of Euler (right-endpoint) and Trapezoidal (endpoint averaging) quadrature rules.*

Numerical integration error scales with step size $\Delta_t$. In discrete sequence modeling where $\Delta_t \le 0.1$, higher-order polynomial error terms drastically reduce cumulative integration error.

The exponential-Euler rule used in Mamba-1 and Mamba-2 approximates integrals with piecewise constant rectangles, yielding a first-order local error of $O(\Delta_t^2)$. At $\Delta_t = 0.1$, residual error remains at the $0.01$ scale.

In contrast, Mamba-3 adopts an exponential-trapezoidal rule that connects endpoints linearly, achieving second-order accuracy with local error scaling as $O(\Delta_t^3)$. At $\Delta_t = 0.1$, residual error drops to $0.001$—over a $10\times$ reduction in numerical error.

The resulting Mamba-3 recurrence equation is formulated as:

$$h_t = \alpha_t h_{t-1} + \beta_t B_{t-1} x_{t-1} + \gamma_t B_t x_t$$

where $\alpha_t = \exp(\Delta_t A_t)$, $\beta_t = (1 - \lambda_t) \Delta_t \exp(\Delta_t A_t)$, $\gamma_t = \lambda_t \Delta_t$, and $\lambda_t \in [0, 1]$ is a learned data-dependent scalar.

By incorporating the prior input term $\beta_t B_{t-1} x_{t-1}$, Mamba-3 blends adjacent inputs $x_{t-1}$ and $x_t$ directly during hidden state updates. This operation intrinsically performs a width-2 data-dependent causal convolution inside the recurrence, eliminating the need for standalone external convolution layers.

In the parallel SSD formulation, the transformation mask $L$ factorizes into the product of a 1-semiseparable matrix and a 2-band matrix. The 1-semiseparable matrix encapsulates historical decay chains, while the 2-band matrix handles adjacent token mixing across main and sub-diagonals.

### 3.3. Complex-Valued State Space Models and the RoPE Trick

Standard real-valued SSMs fail to solve algorithmic discrete state tracking. Mamba-3 introduces complex-valued state transitions combined with the RoPE Trick to retain optimal efficiency.

#### Intuition: Real Contraction vs. Complex Rotation

The fundamental distinction between real and complex transitions is analogous to 1D elastic contraction versus 2D clock hand rotation:

- Real Decay: Analogous to points on a 1D line. Multiplying repeated real fractions (e.g., $0.9 \times 0.9 \dots$) contracts coordinates monotonically toward zero. Direction cannot change, and magnitude continuously fades.
- Complex Rotation: Adding an imaginary axis $i$ creates a 2D plane. Under Euler's formula $e^{i\theta} = \cos \theta + i \sin \theta$, multiplying by a complex exponential rotates the state vector by angle $\theta$ without altering its length.

For binary parity tracking, each occurrence of input 1 rotates the state hand by 180 degrees:
- 0 inputs: 12 o'clock position (even state)
- 1 input: 6 o'clock position (odd state, sign inversion)
- 2 inputs: Returns to 12 o'clock position (even state)

Because magnitude is preserved along the angular coordinate, information survives across arbitrarily long sequences without decaying to zero.

#### Formulation of Complex SSMs

Mamba-3 defines continuous diagonal complex state transitions as:

$$h'(t) = \text{Diag}(A(t) + i \theta(t)) h(t) + (B(t) + i \hat{B}(t)) x(t)$$

where $A(t)$ modulates real decay and $\theta(t)$ determines angular velocity.

Discretizing this system yields block-diagonal transition matrices composed of $2 \times 2$ rotation blocks $R_t$:

$$h_t = \exp(\Delta_t A_t) R_t h_{t-1} + \Delta_t B_t x_t$$

where rotation matrix $R_t$ rotates 2D coordinates by angle $\Delta_t \theta_t$:

$$R_t = \begin{bmatrix} \cos(\Delta_t \theta_t) & -\sin(\Delta_t \theta_t) \\ \sin(\Delta_t \theta_t) & \cos(\Delta_t \theta_t) \end{bmatrix}$$

#### Resolving Overhead via the RoPE Trick

Directly multiplying the massive hidden state tensor $h_t$ by complex rotation matrices quadruples arithmetic operations and severely degrades decoding latency.

Mamba-3 resolves this through the RoPE Trick mathematical equivalence:
- Core Mechanism: Instead of rotating the entire internal state $h_t$, the internal state is updated purely with real decay. Rotations are pre-applied exclusively to input projection Key ($B_t$) and output projection Query ($C_t$).
- Mathematical Equivalence: Expanding the recurrence shows that rotating $h_t$ sequentially is mathematically identical to applying cumulative rotation matrices directly to $B_t$ and $C_t$:

$$h_t = \exp(\Delta_t A_t) h_{t-1} + \left(\prod_{j=1}^t R_j^T\right) \Delta_t B_t x_t, \quad y_t = \left[\left(\prod_{j=1}^t R_j^T\right) C_t\right]^T h_t$$

- Synergy with Transformer Kernels: Standard Transformer architectures already deploy highly optimized GPU kernels for Rotary Position Embeddings (RoPE). Mamba-3 leverages these exact kernels, substituting fixed positional angles with learned data-dependent angles $\theta_t$.

This formulation grants Mamba-3 full complex rotational expressivity (100% parity accuracy) while maintaining real-valued compute speeds.

### 3.4. Multi-Input Multi-Output (MIMO) SSMs

![Mamba-2 vs Mamba-3 Architecture](/images/mamba3/_page_10_Picture_0.jpeg)

*Figure 2: Architectural comparison between Mamba-2 and Mamba-3. Highlights exponential-trapezoidal discretization, data-dependent RoPE, MIMO projections, normalization layers, and learned parameter biases.*

To resolve the decoding memory bandwidth bottleneck ($\approx 2.5$ ops/byte), Mamba-3 upgrades the internal recurrence to a Multi-Input Multi-Output (MIMO) architecture:

- Input dimension expansion: $x_t \in \mathbb{R}^P \to x_t \in \mathbb{R}^{P \times R}$
- State input projection expansion: $B_t \in \mathbb{R}^N \to B_t \in \mathbb{R}^{N \times R}$
- State output projection expansion: $C_t \in \mathbb{R}^N \to C_t \in \mathbb{R}^{N \times R}$

Hidden state memory footprint $h_t \in \mathbb{R}^{N \times P}$ remains identical, preserving low memory bandwidth overhead. Crucially, outer product updates transition into $R$-dimensional matrix-matrix multiplications, scaling FLOPs by $R\times$. Because these additional operations execute on Tensor Cores concurrently with memory I/O transfers, wall-clock decoding latency remains unaffected.

| Metric | SISO (Single-Input Single-Output) | MIMO (Rank $R$) |
|---|---|---|
| Arithmetic Intensity | $\approx 2.5$ ops/byte | $\approx 2.5 \times R$ ops/byte |
| Decoding FLOPs | $5NP - P$ | $4NPR + NP - PR$ |
| Hidden State Memory Size | $N \times P$ | $N \times P$ (Identical) |

During training, MIMO factorizes into $R^2$ parallel SISO computations. By tuning chunk sizes to $C / R$, training overhead is bounded within $R\times$ ($2\times$ slowdown at $R=4$).

To minimize parameter growth, rank $R$ is applied exclusively to $B$ and $C$ projections, while remaining projections receive learned channel-wise scalar multipliers. MLP intermediate dimensions are slightly reduced to maintain strict parameter parity with baseline models.

### 3.5. Architectural Details

Mamba-3 adopts a Pre-Norm Llama-style backbone interleaving Mamba-3 and SwiGLU blocks. Key modifications from Mamba-2 include:

- Updated SSM Recurrence: SSD layers are upgraded to complex exponential-trapezoidal recurrences with the RoPE Trick. Real components are handled in the SSD engine, while imaginary components are processed via RoPE kernels.
- Projection Normalization: RMSNorm is added directly following $B$ and $C$ projections, stabilizing training and permitting removal of post-gate RMSNorm layers.
- Learned Channel Biases: Head- and channel-specific learned biases are appended after projection normalization, inducing convolutional behavior that replaces external short causal convolution layers.

---

## 4. Experimental Results

### 4.1. Language Modeling

All models were pre-trained on FineWeb-Edu across 100B tokens under 2K context lengths using the Llama-3.1 tokenizer. Benchmark evaluations at the 1.5B scale are summarized below:

| Model | FW-Edu ppl↓ | LAMB. acc↑ | HellaS. acc_n↑ | PIQA acc↑ | Arc-E acc↑ | Arc-C acc_n↑ | WinoGr. acc↑ | OBQA acc↑ | Avg acc↑ |
|---|---|---|---|---|---|---|---|---|---|
| Transformer-1.5B | 10.51 | 50.3 | 60.6 | 73.8 | 74.0 | 40.4 | 58.7 | 29.6 | 55.4 |
| GDN-1.5B | 10.45 | 49.2 | 61.3 | 74.3 | 75.3 | 41.2 | 58.0 | 31.6 | 55.8 |
| Mamba-2-1.5B | 10.47 | 47.8 | 61.4 | 73.6 | 75.3 | 41.8 | 57.5 | 32.6 | 55.7 |
| Mamba-3-SISO-1.5B | 10.35 | 49.4 | 61.9 | 73.6 | 75.9 | 42.7 | 59.4 | 32.0 | 56.4 |
| Mamba-3-MIMO-1.5B | 10.24 | 51.7 | 62.3 | 75.3 | 76.5 | 44.5 | 60.6 | 32.6 | 57.6 |

Mamba-3 SISO outperforms Mamba-2 and GDN baselines across all model sizes without external convolutions, while Mamba-3 MIMO ($R=4$) achieves an additional +1.2 point gain in average accuracy over SISO.

### 4.2. State Tracking Capabilities

| Model | Parity ↑ | Arith. w/o brackets ↑ | Arith. w/ brackets ↑ |
|---|---|---|---|
| Mamba-3 | 100.00 | 98.51 | 87.75 |
| Mamba-3 (Standard RoPE) | 1.56 | 20.70 | 2.62 |
| Mamba-3 (No RoPE) | 2.27 | 1.49 | 0.72 |
| Mamba-2 | 0.90 | 47.81 | 0.88 |
| GDN [-1,1] | 100.00 | 99.25 | 93.50 |

Data-dependent RoPE in Mamba-3 achieves 100% accuracy on parity tracking and near-perfect resolution of modular arithmetic. Removing RoPE or replacing it with data-independent standard RoPE collapses performance to random guessing, demonstrating the critical necessity of complex rotational dynamics.

### 4.3. Component Ablation Studies

| Model Variant | ppl↓ |
|---|---|
| Mamba-3 − bias − trap | 16.68 |
| Mamba-3 − bias | 16.49 |
| Mamba-3 | 15.72 |
| Mamba-3 + conv | 15.85 |

Combining learned biases with exponential-trapezoidal discretization yields major perplexity gains ($16.68 \to 15.72$). Re-introducing external short convolutions slightly degrades perplexity ($15.72 \to 15.85$), confirming that internal implicit convolutions fully supersede external convolutional layers.

### 4.4. Inference Efficiency

![State Size vs Pretraining Perplexity Pareto Frontier](/images/mamba3/_page_14_Figure_0.jpeg)

*Figure 3: Pareto frontier between hidden state size and pre-training perplexity. Mamba-3 matches baseline perplexity at half the state size of Mamba-2, while MIMO further advances the frontier without increasing state memory.*

In kernel benchmarks with bf16 and state dimension 128, Mamba-3 SISO surpasses both Mamba-2 and GDN reference kernels. Mamba-3 MIMO ($R=4$) also maintains faster execution than Mamba-2. Across 16K sequence lengths, end-to-end prefill and decode latency reaches 140.61ms for SISO and 151.81ms for MIMO, substantially outperforming attention-based vLLM (976.50ms).

### 4.5. Retrieval Capabilities and Hybrid Models

While Mamba-3 performs competitively on associative recall and question answering (TQA, SQuAD), fixed state capacities inherently limit extraction over semi-structured and unstructured data (SWDE, FDA). Interleaving Mamba-3 with self-attention layers in a 5:1 hybrid ratio overcomes this boundary, outperforming pure Transformers while dramatically boosting length generalization via pre-gate group RMSNorm.

---

## 5. Conclusion and Key Takeaways

1. Generalized Discretization Framework: Mamba-3 establishes a rigorous theoretical foundation for time-varying SSM discretization, upgrading first-order exponential-Euler approximations to second-order accurate exponential-trapezoidal quadrature. The resulting 3-term recurrence performs an implicit width-2 causal convolution that, combined with learned biases, eliminates external short convolutions and streamlines the architecture.

2. Complex State Transitions via Data-Dependent RoPE: By establishing mathematical equivalence between complex block-diagonal state transitions and data-dependent RoPE on Key and Query projections, Mamba-3 introduces rotational dynamics without computational overhead. This resolves longstanding state-tracking failures on parity and modular arithmetic, achieving 100% accuracy where prior real-valued models failed.

3. MIMO Scaling for Hardware Utilization: To overcome decoding memory bandwidth bottlenecks ($\approx 2.5$ ops/byte), Mamba-3 extends recurrences to multi-input multi-output structures. Upgrading outer products to rank-$R$ matrix multiplications expands FLOPs by $R\times$ without altering state memory footprint or increasing inference latency.

4. Unique SSM-Driven Innovation: The three core innovations—discretization, complex transitions, and MIMO—arise naturally from continuous-time state space principles, opening distinct architectural design spaces inaccessible to standard linear attention or test-time training frameworks.
