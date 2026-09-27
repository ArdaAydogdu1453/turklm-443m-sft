# TURKLM Training Log & Engineering Journey

This document chronicles the real-world development, failures, audits, and breakthrough milestones of the **TURKLM (443M)** Turkish Language Model project.

---

## Run Summary Table

| Run Version | Stage | Dataset | Epochs / Steps | LR | Eval Loss | Perplexity | Outcome |
|---|---|---|---|---|---|---|---|
| **v10.0_turbo** | Base Pretrain | Prepared Wikipedia (Packed 2048) | 5,000 steps | 2.5e-4 | **3.166** | **23.71** | ✅ **Success** (Rock-solid base) |
| **v10.1_sft** | SFT Attempt 1 | Merve + Alpaca (~100K) | 2 epochs (840 steps) | 2.0e-5 | **3.488** | **32.74** | ❌ **Failed** (Base degraded, underfitted) |
| **v10.2_sft_fix** | SFT Recovery | Merve + Alpaca (103,477) | 3 epochs (2,457 steps) | 1.0e-4 | **2.820** | **16.78** | ✅ **Success** (PPL halved, instructions learned) |
| **v10.3_merve** | SFT Ablation | SADECE Merve (51,563) | 2 epochs (~850 steps) | 1.0e-4 | *In Progress* | *Target < 2.75* | 🔬 Clean data vs diversity ablation |
| **v11_base_v2** | 3-Stage Curriculum | 15B Token Multi-Tier Data | 8,000 steps | 2.5e-4 $\to$ 8e-5 | *Planned* | *Target < 12.0* | 🚀 Next-gen foundation model |

---

## Phase 1: Base Pretraining (`v10.0_turbo`)

### Objectives
Train a 443M parameter Turkish decoder-only Transformer from scratch using a custom 64,000 BPE vocabulary and packed 2048-token context windows.

### Execution Highlights
- Hardware: NVIDIA A100-SXM4-80GB on Google Colab.
- Backend: SDPA fallback with PyTorch Fused AdamW and TF32 acceleration.
- Pretraining completed 5,000 optimization steps seamlessly with atomic Drive synchronization every 500 steps.
- **Milestone Metric:** Reached **Eval Loss: 3.166** and **Perplexity: 23.71**. Language structure, Turkish morphology, and factual context were solidly learned.

---

## Phase 2: The SFT Failure & Post-Mortem (`v10.1_sft`)

### The Problem
When the v10.0 base model was fine-tuned for instruction following in run `v10.1_sft`, **the model degraded**:
- Eval loss increased from 3.166 $\to$ **3.488**.
- Perplexity worsened from 23.71 $\to$ **32.74**.
- The model failed to follow instructions cleanly and exhibited repetitive user-prompt echoing.

### Root Cause Audit
Through systematic code and telemetry inspection, 4 fatal flaws were uncovered:
1. **Learning Rate Too Low ($2\times 10^{-5}$):** For a 443M parameter model trained from scratch, $2\times 10^{-5}$ was 10x below the pretraining learning rate ($2.5\times 10^{-4}$). The weights were barely nudged.
2. **Underfitting (Only 2 Epochs):** Training was terminated while the training curve was still steep; 840 steps was insufficient for 100K data.
3. **Dataset Schema Bugs:** In Hugging Face's `merve/turkish_instructions`, the real columns were `['Unnamed: 0', 'talimat', ' giriş', ' çıktı']` with leading spaces on ` giriş` and ` çıktı`. Naive `.get("instruction")` calls resulted in empty strings that degraded data quality.
4. **Data Collator Memory Crash:** Using `group_by_length=True` with `DataCollatorForSeq2Seq` caused nondeterministic crash bugs due to residual `length` fields.

---

## Phase 3: The SFT Breakthrough (`v10.2_sft_fix`)

### Engineering Fixes
1. **Learning Rate 5x Boost:** Increased from $2\times 10^{-5} \to 1\times 10^{-4}$ to match the base model's learning scale.
2. **Epoch Extension:** Increased from 2 to 3 full epochs (~2,457 steps) for thorough convergence.
3. **Robust Schema Normalization:** Explicit `rename_columns` handling Turkish column names with leading spaces.
4. **Assistant-Only Loss Masking:** Full `-100` label masking on user prompt tokens so the model optimizes purely on assistant answers.
5. **Memory & Collator Stabilizations:** `group_by_length` set to `False`, `length` column discarded from tokenization output.

### Results
- **Eval Loss:** Decreased dramatically from 3.488 to **2.820**.
- **Perplexity:** Dropped from 32.74 to **16.78** (perplexity was cut in half!).
- **Benchmark Smoke Test:** Successfully passed all 5 benchmark questions in Turkish (capital of Turkey, Python reverse list, Atatürk, chemical formula of water, Solar system planets).

---

## Phase 4: SFT Data Ablation (`v10.3_merve_only`)

### Hypothesis
`TFLai/Turkish-Alpaca` contains synthetic machine-translated artifacts. Does fine-tuning exclusively on human-curated Turkish instructions (`merve/turkish_instructions`, 51K samples) produce a lower loss than the noisy 103K mixture?

- Setup: 51,563 clean samples, 2 epochs, LR $1\times 10^{-4}$, evaluated every 100 steps.

---

## Phase 5: The Roadmap (`train_base_v2.py` / v11)

To push TURKLM beyond the limitations of single-source Wikipedia pretraining, the **v11 architecture** introduces a **15-Billion Token 3-Stage Curriculum Pretraining Pipeline** using `musabc/nanogpt-tr-v5-data`:

1. **Stage 1 (Web Foundation - 2.94B Tokens):** OSCAR, mC4, FineWeb-HQ ($\text{LR} = 2.5\times 10^{-4}$).
2. **Stage 2 (Domain Refinement - 9.03B Tokens):** BellaTurca, CulturaX, Cosmopedia ($\text{LR} = 1.8\times 10^{-4}$).
3. **Stage 3 (High-Precision Knowledge - 2.97B Tokens):** Wikipedia, Academic Theses, Dizin ($\text{LR} = 8\times 10^{-5}$).

Implemented with fork-safe lazy `np.memmap` datasets, fast subset evaluations, and automated multi-stage Drive checkpointing.
