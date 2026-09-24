---
title: "Transformer vs Mamba: Structural Contrast Between Self-Attention and State Space Models"
date: 2026-09-07T20:44:13+09:00
draft: false
math: true
tags: ["Paper Review", "Transformer", "Mamba", "SSM", "Attention", "State Space Model", "Sequence Modeling"]
categories: ["Paper Review"]
summary: "A structural comparison between Transformer (Attention Is All You Need) and the Mamba series (1/2/3) across identical axes: sequence processing philosophy, memory architecture, computational complexity, hardware efficiency, and expressive power."
cover:
  image: "/images/transformer-vs-mamba/transformer_arch.jpeg"
  alt: "Transformer vs Mamba Architecture Comparison"
---

> Referenced Papers
> - Vaswani, A. et al. "Attention Is All You Need." NeurIPS 2017.
> - Gu, A. & Dao, T. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." ICLR 2024.
> - Dao, T. & Gu, A. "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality." ICML 2024.
> - Lahoti, A. et al. "Mamba-3: Improved Sequence Modeling using State Space Principles." ICLR 2026.

---

## 1. One-Sentence Summary

While Transformer achieves powerful context awareness via self-attention that directly references all token pairs at quadratic cost with sequence length, the Mamba series achieves linear complexity and constant-memory inference through a recurrence structure that selectively compresses information into a fixed-size state. This post contrasts the two paradigms across identical axes to delineate how each architectural choice creates trade-offs between performance and efficiency.

---

## 2. Fundamental Sequence Processing Philosophy

### 2.1 Transformer: Global Reference Seeing Everything at Once

![Transformer Architecture](/images/transformer-vs-mamba/transformer_arch.jpeg)
*Figure 1: Transformer model architecture. An Encoder-Decoder structure that alternately stacks Multi-Head Attention and Feed-Forward Networks in each layer.*

The operational flow within a Transformer block is intuitive. Positional encoding is added to input token vectors to convey order before entering the layers. Inside each layer, computation proceeds in two distinct stages.

First is Multi-Head Attention, which gathers surrounding context. Each token issues a Query to every other token across the sequence. By comparing against each token's Key, attention scores are computed, and information from their Values is weighted and aggregated accordingly. Instead of waiting sequentially for previous token computations, all tokens examine each other simultaneously in a single pass to gather necessary information.

Second is the Feed-Forward Network (FFN), which refines features. Drawing upon the aggregated context from attention, each token undergoes an independent non-linear transformation. Between stages, Residual Connections preserve raw signals and Layer Normalization stabilizes feature scales, ensuring steady signal propagation across deep stacks.

Thus, Transformer alternates between context aggregation (Attention) and feature transformation (FFN) across two separate sub-layers. The core attention operation is formulated as:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

The pivotal advantage of this design is that the maximum path length between any two tokens is $O(1)$. No matter how far apart the first and last words are, they connect directly in a single attention operation, structurally resolving the vanishing gradient problem that plagued RNNs on long sequences.

However, referencing all token pairs incurs $O(T^2)$ computation and $O(T^2)$ memory for a sequence of length $T$. Processing 1,000 tokens requires 1 million pairwise calculations, while 10,000 tokens require 100 million.

### 2.2 Mamba: Linear Recurrence Advancing with a Compact Summary Notebook

![Mamba-1 Architecture](/images/transformer-vs-mamba/mamba1_arch.jpeg)
*Figure 2: Mamba-1 block design. Integrates the SSM block and MLP block into a single homogeneous block.*

The operational flow of the Mamba-1 block adopts a fundamentally different philosophy from Transformer. Rather than stacking separate Attention and MLP layers, it unifies local context mixing and long-term memory within a single homogeneous block.

Upon entering the block, the token vector undergoes linear projection that doubles the channel dimension and splits the signal into two branches.

The main branch first passes through a 1D Convolution, which blends local context across immediately adjacent tokens. Next comes the core module: the Selective SSM. By examining the current token directly, it dynamically determines how much new information to retain and how much prior memory to discard (via input-dependent parameters $\Delta, B, C$). It updates a fixed-size hidden state notebook, forgetting irrelevant past details while extracting a distilled output.

The second branch passes through a SiLU activation to act as a gating mechanism. Multiplying the SSM output element-wise with this gating signal filters out residual noise. Finally, an output linear projection contracts channels back to the model dimension.

Instead of constructing a massive pairwise correlation matrix, Mamba operates via an input-dependent recurrence:

$$ h_t = \bar{A}h_{t-1} + \bar{B}x_t, \quad y_t = Ch_t $$

It compresses historical context into a compact hidden state $h_t$, updating it upon each new incoming token. While Transformer retains every past token in view, Mamba advances carrying only a single distilled summary notebook.

This yields $O(T)$ computational complexity with respect to sequence length $T$, and $O(1)$ memory during inference (governed solely by the state size). Even over sequences of 1 million tokens, resource requirements scale strictly linearly.

### 2.3 Fundamental Trade-off: Global Reference vs Compressed Memory

| Property | Transformer | Mamba |
|----------|-------------|-------|
| Information Access | Directly references all past tokens | Compresses into a fixed-size state |
| Maximum Path Length | $O(1)$ — Direct connection at any distance | $O(T)$ — Indirect propagation via recurrence |
| Information Loss | None (all tokens preserved) | Present (lossy compression beyond state capacity) |
| Computational Cost | $O(T^2)$ — Quadratic scaling | $O(T)$ — Linear scaling |

This fundamental trade-off dictates the divergent characteristics of both paradigms: Transformer preserves information completely at high cost, while Mamba operates with extreme efficiency by summarizing context into a finite state.

---

## 3. Memory Architecture: KV Cache vs Recursive State

### 3.1 Transformer's KV Cache: A Growing Storage of Historical Keys and Values

As described in Section 2.1, Transformer requires each token to match Queries against Keys and aggregate Values across all tokens. During training, the entire sequence is available at once, enabling this operation via a single parallel matrix multiplication.

However, during autoregressive decoding (generating text token by token), the operational dynamics shift dramatically. Whenever a new token is generated, it must re-attend to all previously generated tokens.

If previous Keys and Values were discarded, generating each new word would require recomputing neural activations from layer 1 all the way to the top for the entire sequence from scratch. This redundant recalculation quickly becomes prohibitively slow.

The KV Cache was introduced precisely to prevent this waste. It serves as temporary GPU storage preserving the Key and Value vectors computed by all prior tokens. When generating a new token, the model computes only its own Query, fetches previously cached Keys and Values from GPU memory, and executes attention promptly.

Because Transformer lacks an internal memory state, it maintains context by accumulating every past token's physical Key and Value vectors in GPU RAM:

$$ \text{KV Cache Memory} = 2 \times \text{Layers} \times \text{Heads} \times \text{Head Dimension} \times \text{Sequence Length} $$

Because cache size scales linearly with sequence length, GPU memory is rapidly exhausted on long contexts. In a 7B model with a 32K context window, the KV cache alone demands several gigabytes, restricting batch sizes and forming a severe bottleneck on serving throughput.

### 3.2 Mamba's Recursive State: Advancing with a Single-Page Summary Notebook

Why does Mamba eliminate the need for a KV cache while maintaining full context awareness during generation?

The key reason is that Mamba does not store raw Keys and Values of historical tokens. As detailed in Section 2.2, the Selective SSM evaluates the incoming token upon arrival and directly overwrites and updates the fixed-size hidden state ($h_t$).

To produce the next token, Mamba does not need to search through past records. It simply conditions on the single, latest summary notebook (the recursive state) and updates it with the new token:

$$ \text{Recursive State Memory} = \text{Layers} \times \text{Heads} \times \text{State Size}(N) \times \text{Head Dimension}(P) $$

Whether the sequence spans 1,000 tokens or 1,000,000 tokens, the memory footprint remains entirely constant. In production serving environments, this offers substantial advantages: freeing GPU memory from bulky KV caches allows serving vastly larger batch sizes concurrently, delivering over $5\times$ higher inference throughput compared to Transformers on identical hardware.

### 3.3 Memory Structure Comparison

| Metric | Transformer (KV Cache) | Mamba (Recursive State) |
|--------|------------------------|-------------------------|
| Sequence Length Dependency | $O(T)$ — Linear growth | $O(1)$ — Constant |
| Size at 1M Tokens | Dozens of gigabytes | Dozens of megabytes |
| Batch Size Scalability | Constrained by memory contention | Favorable due to minimal overhead |
| Information Completeness | Lossless (all raw tokens kept) | Lossy (compressed into finite state) |

---

## 4. Computational Complexity and Hardware Efficiency

### 4.1 Training Computation Comparison

| Model | Layer Complexity | Sequential Operations | Dominant Operation |
|-------|------------------|-----------------------|--------------------|
| Transformer | $O(T^2 \cdot d)$ | $O(1)$ | Matrix Multiplication (Tensor Cores) |
| Mamba-1 | $O(T \cdot N \cdot d)$ | $O(\log T)$ | Element-wise scan (Generic ALUs) |
| Mamba-2 (SSD) | $O(T \cdot N^2)$ | $O(T/Q)$ | Matrix Multiplication (Tensor Cores) |
| Mamba-3 (MIMO) | $O(T \cdot N^2 \cdot R)$ | $O(T/Q)$ | Matrix Multiplication (Tensor Cores) |

Transformer was designed from inception around matrix multiplication (matmul), leveraging GPU Tensor Cores to near-peak utilization. Together with optimizations such as FlashAttention, real-world training efficiency far outpaces what raw theoretical quadratic complexity suggests.

While Mamba-1 achieved linear theoretical complexity, its selective scan relied on element-wise recurrences, bypassing Tensor Cores and executing on general-purpose arithmetic units.

Mamba-2's SSD algorithm resolved this discrepancy. By proving that structured SSMs are mathematically dual to attention (State Space Duality), it introduced a hybrid formulation: computing dense block matmuls inside chunks via Tensor Cores while maintaining recurrence across chunks. This achieved $2\times \sim 8\times$ faster training speeds over Mamba-1.

### 4.2 Inference Computation Comparison (Token Generation)

| Model | Compute per Token | Memory Reads | Arithmetic Intensity |
|-------|-------------------|--------------|----------------------|
| Transformer | $O(T \cdot d)$ | Entire KV cache | High (matmul) |
| Mamba-1/2 (SISO) | $O(N \cdot P)$ | State tensor | ~2.5 ops/byte |
| Mamba-3 (MIMO, R=4) | $O(N \cdot P \cdot R)$ | State tensor (identical) | ~10 ops/byte |

During autoregressive decoding, Transformer must read the entire KV cache for every new token, progressively slowing down as context elongates. Mamba reads only the fixed-size state tensor.

However, the Single-Input Single-Output (SISO) design of Mamba-1 and Mamba-2 updated state vectors via outer products, yielding an extremely low arithmetic intensity (~2.5 ops/byte). GPU Tensor Cores spent most cycles idling, waiting on memory I/O.

Mamba-3's Multiple-Input Multiple-Output (MIMO) design directly overcomes this bottleneck. By projecting inputs and outputs into rank $R$, it elevates the outer product into a dense matrix-matrix multiplication while maintaining the same underlying state dimension. Arithmetic intensity increases by a factor of $R$ with minimal additional memory transfers, driving Tensor Core utilization and boosting model expressiveness without increasing decode latency.

---

## 5. Expressive Power: Strengths and Limitations

### 5.1 Transformer Strengths and Limitations

Transformer self-attention provides exceptional expressive capacity:

- Arbitrary-distance dependencies: Any token can reference any prior token directly.
- Multi-pattern representations: Multi-Head Attention learns heterogeneous relationship subspaces concurrently.
- Precise associative recall: Key-value matching retrieves exact details reliably.

Nonetheless, Softmax attention normalizes weights to sum to 1, causing attention to diffuse as context scales. Furthermore, attention lacks intrinsic spatial awareness, requiring external mechanisms such as Positional Encodings or RoPE.

### 5.2 Expressive Evolution of the Mamba Series

Mamba's expressive capability expanded progressively across three generations:

#### Mamba-1: Selective Compression

Traditional SSMs (such as S4) were Linear Time-Invariant (LTI) systems with static parameters across all time steps, unable to distinguish critical tokens from noise. Mamba-1 introduced the Selection Mechanism, parameterizing $B, C, \Delta$ as functions of the input.

When $\Delta$ increases, the model quickly forgets historical states and prioritizes the current input; when $\Delta$ decreases, it preserves context long-term. This dynamic gating proved mathematically dual to RNN forget/input gates, enabling Mamba-1 to achieve over 99% accuracy on synthetic tasks like Selective Copying and Induction Heads where earlier SSMs failed.

#### Mamba-2: Expanding State Capacity

Mamba-2 simplified the transition matrix $A$ to a scalar multiple of the identity while expanding head dimension $P$ from 1 to 64~128.

Crucially, SSD enabled scaling the state dimension $N$ from $N=16$ up to $N=64 \sim 256$ at virtually no extra computational cost. On Multi-Query Associative Recall (MQAR), where Mamba-1 struggled due to state saturation, Mamba-2 matched softmax attention baselines.

#### Mamba-3: Rotational Dynamics and Precision

Prior to Mamba-3, fundamental limitations remained:

- Inability to track states: Constraining $A$ to negative real scalars permitted decay but prohibited rotation. On parity tasks requiring state sign flips, models hovered around random guess rates (0.9%).
- Numerical drift: First-order Euler discretization accumulated $O(\Delta_t^2)$ truncation errors over long sequences.

Mamba-3 tackled these via three innovations:

1. Complex state transitions: Adding imaginary components $i\theta(t)$ to diagonal transitions enables rotational dynamics alongside decay, reusing Transformer RoPE kernels to solve parity tasks with 100% accuracy.
2. Exponential-trapezoidal discretization: Replacing Euler with a second-order trapezoidal integration cuts errors to $O(\Delta_t^3)$ and implicitly embeds a width-2 convolution within the recurrence.
3. MIMO: Expanding inputs and outputs to rank $R$ enhances expressiveness and decoding arithmetic intensity simultaneously.

### 5.3 Expressive Power Summary

| Capability | Transformer | Mamba-1 | Mamba-2 | Mamba-3 |
|------------|-------------|---------|---------|---------|
| Long-range Dependencies | $O(1)$ path, unrestricted | Linear recurrence, selective preservation | Improved via state expansion | Lossless preservation via complex rotation |
| Associative Recall (MQAR) | Flawless | Constrained by $N=16$ | Matches attention with $N=256$ | Further elevated with $N=256$ + MIMO |
| State Tracking (Parity) | Capable | Incapable (decay only) | Incapable (0.9%) | 100% (complex rotation) |
| Selective Copying | Capable | 99.8% | Capable | Capable |
| Precise Retrieval | Superior | Constrained | Improved | Improved (hybrid recommended) |

---

## 6. Conclusion: Coexistence, Not Replacement

Transformer and Mamba are not mutually exclusive rivals; they achieve their greatest potential when deployed together in a complementary partnership.

Transformer excels at precise associative retrieval and complex global reasoning through pairwise attention, but encounters an unavoidable quadratic barrier as context expands. Mamba achieves linear-time processing and constant-memory inference by selectively compressing sequences into compact states, though finite capacity introduces potential compression trade-offs.

As illuminated by Mamba-2's State Space Duality, both paradigms represent two computational paths of the same underlying mathematical structure. The widespread adoption of hybrid architectures in industry—interleaving Mamba layers with periodic attention layers—reflects this theoretical reality. Leveraging Mamba's linear efficiency for the vast majority of sequence processing while deploying attention where global associative search is indispensable represents today's optimal balance of performance and efficiency.
