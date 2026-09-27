# TURKLM Model Architecture

TURKLM is a 443M parameter Turkish autoregressive language model based on a modern **LLaMA-style decoder-only Transformer** architecture.

---

## 1. Architectural Specifications (A100 Profile)

| Hyperparameter | Value | Description |
|---|---|---|
| **Architecture** | LLaMA | Decoder-only autoregressive transformer |
| **Parameters** | 443.07 Million | Total trainable parameters |
| **Hidden Size ($d_{model}$)** | 1024 | Representation dimension |
| **Intermediate Size ($d_{ff}$)** | 4096 | Feed-forward / SwiGLU expansion dimension |
| **Layers ($N$)** | 24 | Number of Transformer blocks |
| **Attention Heads ($h_q$)** | 16 | Number of Query attention heads |
| **Key/Value Heads ($h_{kv}$)** | 8 | Grouped Query Attention (GQA) 2:1 ratio |
| **Head Dimension ($d_k$)** | 64 | Dimension per attention head ($1024 / 16$) |
| **Vocabulary Size** | 64,000 | Custom Byte-Level BPE with Turkish morphology |
| **Context Length (Pretraining)** | 2048 | Maximum sequence length for base training |
| **Context Length (SFT)** | 1024 | Optimized sequence length for instruction tuning |
| **Activation Function** | SwiGLU | SiLU gated linear unit |
| **Normalization** | RMSNorm ($\epsilon = 10^{-5}$) | Root Mean Square layer normalization |
| **Positional Embedding** | RoPE ($\theta = 10000$) | Rotary Positional Embedding |
| **Weight Tying** | Enabled | Input embeddings tied to LM head (`tie_word_embeddings=True`) |

---

## 2. Parameter Breakdown

With a vocabulary size of $V = 64,000$ and hidden size $d = 1024$:

### A. Embedding Layer (Tied with LM Head)
- **Token Embeddings:** $64,000 \times 1024 = 65,536,000$ parameters (~65.5M)

### B. Per-Transformer Layer Breakdown ($N = 24$)
1. **Self-Attention with GQA:**
   - $W_q$ (Query Projection): $1024 \times 1024 = 1,048,576$
   - $W_k$ (Key Projection - 8 heads): $1024 \times 512 = 524,288$
   - $W_v$ (Value Projection - 8 heads): $1024 \times 512 = 524,288$
   - $W_o$ (Output Projection): $1024 \times 1024 = 1,048,576$
   - **Attention Subtotal per layer:** $3,145,728$ parameters (~3.15M)

2. **Feed-Forward Network (SwiGLU):**
   - $W_{gate}$ (Gate Projection): $1024 \times 4096 = 4,194,304$
   - $W_{up}$ (Up Projection): $1024 \times 4096 = 4,194,304$
   - $W_{down}$ (Down Projection): $4096 \times 1024 = 4,194,304$
   - **FFN Subtotal per layer:** $12,582,912$ parameters (~12.58M)

3. **Normalization:**
   - Input RMSNorm + Post-Attention RMSNorm: $2 \times 1024 = 2,048$ parameters

- **Single Layer Total:** $\approx 15,730,688$ parameters (~15.73M)
- **24 Layers Total:** $24 \times 15,730,688 \approx 377,536,512$ parameters (~377.5M)

### C. Final Normalization Layer
- **Final RMSNorm:** $1024$ parameters

### D. Grand Total
$$\text{Total Parameters} = 65,536,000 + 377,536,512 + 1024 = 443,073,536 \approx \mathbf{443.07\text{ M}}$$

---

## 3. Key Design Choices

### Grouped Query Attention (GQA)
Instead of standard Multi-Head Attention (16 KV heads) or Multi-Query Attention (1 KV head), TURKLM uses GQA with 16 query heads and 8 key/value heads (2 queries per KV group). This yields:
- **50% smaller KV cache** during autoregressive generation compared to standard MHA.
- Higher inference throughput and larger batch capacity.
- Preservation of full model capacity and reasoning quality.

### SwiGLU Gated Activation
Standard ReLU/GELU activations are replaced with SwiGLU:
$$\text{SwiGLU}(x) = (\text{SiLU}(x W_{gate}) \otimes (x W_{up})) W_{down}$$
This provides richer non-linear representations and faster loss convergence.

### RoPE (Rotary Position Embeddings)
Absolute positional embeddings are replaced by Rotary Position Embeddings (RoPE) applied to Query and Key representations with $\theta = 10000.0$. RoPE preserves relative distance information naturally and supports length extrapolation.

### Tied Embeddings
By sharing the weights of the input embedding matrix and the final classification head (`lm_head`), the model saves $65.5\text{ M}$ parameters while stabilizing token representation learning in vocabulary-heavy languages like Turkish.

---

## 4. Hardware Profiles

The codebase detects GPU memory at runtime and activates the appropriate profile:

| Profile | Target GPU | Min VRAM | Hidden / Intermediate | Layers | Heads (Q/KV) | Context |
|---|---|---|---|---|---|---|
| **A100_HIGH** | A100 40GB/80GB | 35.0 GB | 1024 / 4096 | 24 | 16 / 8 | 2048 (Base) / 1024 (SFT) |
| **L4_MID** | NVIDIA L4 24GB | 20.0 GB | 1024 / 4096 | 24 | 16 / 8 | 2048 (Base) / 1024 (SFT) |
| **T4_BUDGET** | NVIDIA T4 16GB | < 20.0 GB | 768 / 2048 | 12 | 12 / 12 | 1024 |
