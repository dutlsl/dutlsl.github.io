---
title: "From Vanilla RNN to ConvLSTM: Deep Learning Architectures for Time-Series and Spatiotemporal Data"
date: 2026-07-21T18:36:41+09:00
draft: false
math: true
tags: ["Paper Review", "Deep Learning", "RNN", "LSTM", "GRU", "ConvLSTM", "Spatiotemporal"]
categories: ["Paper Review"]
summary: "An engineering breakdown of deep learning architectures for time-series and spatiotemporal data, covering Vanilla RNN BPTT limitations, LSTM Cell States, GRU gate simplifications, and 3D tensor-based ConvLSTM."
cover:
  image: "/images/convlstm/transform_tensor.jpeg"
  alt: "ConvLSTM 3D Tensor Transformation"
---

> **Summary**: This post analyzes the evolution of recurrent architectures for sequence processing, starting from the BPTT vanishing gradient limits of **Vanilla RNN**, to the Cell State control mechanism of **LSTM**, the lightweight gate simplification of **GRU**, and finally **ConvLSTM**, which integrates 3D tensors and convolution operations to enable spatiotemporal sequence modeling.

---

## 1. Lineage of Recurrent Neural Network Architectures

The architectural progression from processing 1D temporal dependencies in sequence data to preserving spatial structures in image/video data follows this trajectory:

```mermaid
flowchart TD
    A["<b>Vanilla RNN</b><br/>Single Recurrent Operation"] 
    B["<b>LSTM</b><br/>Cell State + 3 Gates"]
    C["<b>GRU</b><br/>Reset & Update Gates"]
    D["<b>ConvLSTM</b><br/>3D Tensor & Spatial Grid"]

    A -->|1. Solves BPTT Vanishing Gradient| B
    B -->|2. Gate Simplification & Lightweight Params| C
    B -->|3. Preserves Spatial Structure & Conv Operation| D
```

---

## 2. Vanilla RNN: Recurrent Operations and BPTT Limitations

### 2.1 Architecture and Recurrent Operations
Vanilla RNN is the simplest recurrent structure that updates the current hidden state $h_t$ by combining the previous hidden state $h_{t-1}$ and the current input $x_t$.

![](/images/convlstm/feedforward_vs_recurrent.jpeg)
*Figure 1: Information flow and recurrent structure comparison between Feedforward NN and Recurrent NN*

- Hidden State Calculation: $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$
- Output Calculation: $O_t = \sigma(W_{ho} h_t + b_o)$

### 2.2 BPTT (Backpropagation Through Time) and Vanishing Gradient
During backpropagation across unrolled time steps, the gradient propagated back to a past time step $k$ involves the consecutive multiplication of the weight matrix $W_{hh}^T$:

$$\frac{\partial \mathcal{L}}{\partial W_{hh}} \propto \sum_{k=1}^{t} (W_{hh}^T)^{t-k} h_k$$

As sequence length $(t - k)$ increases, if the eigenvalues of $W_{hh}$ are less than 1, the gradient exponentially decays to 0 (**Vanishing Gradient**), rendering the network incapable of learning long-term dependencies.

---

## 3. LSTM (Long Short-Term Memory): Cell State-Based Information Control

### 3.1 Architectural Structure
LSTM introduces a **Cell State ($C_t$)** that acts as a highway for uninterrupted information flow, governed by three gating mechanisms (Forget, Input, Output Gates).

![](/images/convlstm/lstm_architecture.jpeg)
*Figure 2: Internal cell architecture and three gating structures of LSTM*

### 3.2 Gate-by-Gate Operations
1. **Forget Gate ($f_t$)**: Decides the proportion (0 to 1) of information to retain from the previous Cell State $C_{t-1}$.
   $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
2. **Input Gate ($i_t$)**: Controls how much new information from current input $x_t$ should be stored in the Cell State.
   $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
   $$\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$ *(New candidate cell state)*
3. **Cell State Update ($C_t$)**: Scales the previous cell state by the forget gate and adds the gated new candidate state.
   $$C_t = f_t \circ C_{t-1} + i_t \circ \tilde{C}_t$$
4. **Output Gate ($o_t$ & $h_t$)**: Computes the external hidden state $h_t$ based on the updated Cell State.
   $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
   $$h_t = o_t \circ \tanh(C_t)$$

---

## 4. GRU (Gated Recurrent Unit): Hidden State Integration and Gate Simplification

### 4.1 Architectural Structure
GRU simplifies the LSTM architecture by merging the Cell State and Hidden State into a single state ($h_t$) and reducing the gates to two (Reset Gate, Update Gate).

![](/images/convlstm/gru_architecture.jpeg)
*Figure 3: Internal cell architecture of GRU (Reset and Update Gates)*

### 4.2 Gate-by-Gate Operations
1. **Reset Gate ($r_t$)**: Determines how much of the previous hidden state $h_{t-1}$ to forget when computing the new candidate state.
   $$r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)$$
2. **Update Gate ($z_t$)**: Controls the interpolation ratio between the previous state $h_{t-1}$ and the candidate state $\tilde{h}_t$.
   $$z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)$$
3. **Candidate Hidden State ($\tilde{h}_t$)**:
   $$\tilde{h}_t = \tanh(W \cdot [r_t \circ h_{t-1}, x_t] + b)$$
4. **Hidden State Update ($h_t$)**:
   $$h_t = (1 - z_t) \circ h_{t-1} + z_t \circ \tilde{h}_t$$

---

## 5. Limitations of LSTM and GRU: Loss of Spatial Structure

LSTM and GRU operate on 1D vectors. Flattening 2D/3D image sequences (e.g., Height $\times$ Width) into 1D vectors introduces significant engineering drawbacks:

1. **Destruction of Spatial Locality**: Pixel adjacency relations across rows and columns are completely lost.
2. **Parameter Inefficiency**: All pixel nodes are fully connected, causing weight parameters and computation to scale redundantly with input dimensions.

---

## 6. ConvLSTM (Convolutional LSTM): Integrated Spatiotemporal Architecture

### 6.1 3D Tensor Input/Output Structure
ConvLSTM replaces matrix multiplication with **Convolution operations ($*$)**, maintaining inputs, cell states, hidden states, and all gates as **3D Tensors (Channels $\times$ Height $\times$ Width)**.

![](/images/convlstm/transform_tensor.jpeg)
*Figure 4: Transformation of 2D input sequence into a spatial grid-preserving 3D tensor*

### 6.2 ConvLSTM Internal Gate Operations
ConvLSTM applies convolution kernels in both input-to-state and state-to-state transitions.

![](/images/convlstm/inner_structure.jpeg)
*Figure 5: Internal gate operation architecture of ConvLSTM*

- **Input Gate**: $$i_t = \sigma(W_{xi} * \mathcal{X}_t + W_{hi} * \mathcal{H}_{t-1} + W_{ci} \circ \mathcal{C}_{t-1} + b_i)$$
- **Forget Gate**: $$f_t = \sigma(W_{xf} * \mathcal{X}_t + W_{hf} * \mathcal{H}_{t-1} + W_{cf} \circ \mathcal{C}_{t-1} + b_f)$$
- **Cell State Update**: $$\mathcal{C}_t = f_t \circ \mathcal{C}_{t-1} + i_t \circ \tanh(W_{xc} * \mathcal{X}_t + W_{hc} * \mathcal{H}_{t-1} + b_c)$$
- **Output Gate**: $$o_t = \sigma(W_{xo} * \mathcal{X}_t + W_{ho} * \mathcal{H}_{t-1} + W_{co} \circ \mathcal{C}_t + b_o)$$
- **Hidden State Update**: $$\mathcal{H}_t = o_t \circ \tanh(\mathcal{C}_t)$$

> **Impact of Convolution Kernel Size**:
> In ConvLSTM, larger kernel sizes cover a broader receptive field to capture fast motions across spatial grids, whereas smaller kernels excel at tracking fine-grained localized dynamics.

### 6.3 Encoding-Forecasting Architecture
For spatiotemporal sequence forecasting (e.g., precipitation nowcasting, video frame prediction), multiple ConvLSTM layers are stacked into an **Encoding-Forecasting** structure.

![](/images/convlstm/encoding_forecasting.jpeg)
*Figure 6: Encoding-Forecasting ConvLSTM network architecture for precipitation nowcasting*

1. **Encoding Network**: Consumes past image sequences sequentially to compress and accumulate spatiotemporal features into 3D hidden/cell state tensors.
2. **Forecasting Network**: Takes the final states of the encoding network to unfold 3D tensor states and generate future image predictions.

---

## 7. Architectural Comparison Summary

| Architecture | Data Representation | Core Operation | Memory / Gate Structure | Spatiotemporal Modeling Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **Vanilla RNN** | 1D Vector | Fully-Connected | No Gate (Single Tanh) | Captures short-term temporal dynamics (BPTT limit) |
| **LSTM** | 1D Vector | Fully-Connected | Forget, Input, Output Gate / **Separate Cell State** | Controls long-term temporal dependencies (Loss of spatial structure) |
| **GRU** | 1D Vector | Fully-Connected | Reset, Update Gate / **Unified Hidden State** | Controls long-term temporal dependencies with lightweight parameters |
| **ConvLSTM** | 3D Tensor | Convolution ($*$) | Forget, Input, Output Gate / **3D Grid Preserved** | **Jointly models long-term temporal dependencies and spatial locality** |
