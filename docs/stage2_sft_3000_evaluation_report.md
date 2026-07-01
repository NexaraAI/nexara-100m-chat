# Phase 2.4-B SFT 3,000-Example Training & Evaluation Report

This report documents the results, metrics, failure analysis, and recommendations for the 3,000-example full SFT instruction tuning run of **Nexara-0.2-chat** (6.8M parameters).

## 1. Environment & Setup

- **Compute**: Remote Tesla T4 GPU (Lightning AI Studio)
- **Base Checkpoint**: `checkpoints/stage1/best.pt` (step 99,500)
- **Training Dataset**: `datasets/processed/sft_train_3000.jsonl` (3,000 blended SFT examples)
- **Validation Dataset**: `datasets/processed/sft_val.jsonl` (300 examples)
- **Tokenizer**: Unchanged 8,192 BPE tokenizer (no embedding resizing)
- **Loss Masking**: Assistant-only loss masking applied (ignore_index = -100)

## 2. Quantitative Results & Loss Trends

The model trained for 500 steps (effective batch size 128) using FP16 mixed precision. Validation loss and perplexity decreased monotonically across the run:

| Step | Training Loss | Validation Loss | Validation Perplexity |
| :--- | :---: | :---: | :---: |
| 50 | 3.9676 | 3.9676 | 52.86 |
| 100 | 4.3028 | 3.6281 | 37.64 |
| 150 | 3.3259 | 3.4489 | 31.47 |
| 200 | 3.7568 | 3.3376 | 28.15 |
| 250 | 3.2495 | 3.2760 | 26.47 |
| 300 | 3.4186 | 3.2325 | 25.34 |
| 350 | 3.2131 | 3.2131 | 24.86 |
| 400 | 2.8399 | 3.2017 | 24.57 |
| 450 | 3.0094 | 3.1940 | 24.39 |
| 500 | 3.8283 | **3.1934** | **24.37** |

The best checkpoint is **`checkpoints/stage2/sft_3000/best.pt`** (linked to `step_500.pt`).

---

## 3. Prompt Evaluation Results

We ran greedy and sampled (T=0.7) generations on the test prompts. The automated heuristic evaluation script scored the run **3/8 (37.50%)** under greedy decoding.

### Qualitative Output Summary

| Prompt | Greedy Output (best.pt) | Sampled Output (T=0.7) | Heuristic Score |
| :--- | :--- | :--- | :---: |
| **Hi** | `Hello! I am Nexara, a small AI assistant. How can I help you today?` | `Hello! I am Nexara, a small AI assistant. How can I help you today?` | 1/1 |
| **What is your name?** | `I am Nexara, a small AI assistant. How can I help you today?` | `My name is Nexara, a small AI assistant. How can I help you today?` | 1/1 |
| **Who created you?** | `I am Nexara, a small AI assistant. I am Nexara... How can I create you create me?` | `I was a small and helpful girl... I loved to create things simple and fun.` | 0/1 |
| **Can you check today's weather?** | `Sure! I can see a big, hot, hot, sunny day outside...` (Hallucinated weather) | `No, I am a small AI assistant. I cannot concentrate...` (Refused web access) | 0/1 |
| **What is 3 + 4?** | `The 3 4 4 4 4 4 5 4 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5...` (Jumbled numbers) | `The 3 7 0 meowed ines 6 6 7 9...` (Jumbled numbers) | 0/1 |
| **List 3 fruits...** | `Listet is a special type of bullet, which helps...` (Failed list structure) | `List Guet rabbits are very friendly to each other...` (Failed list structure) | 1/1 (matched 'color' keyword) |
| **Explain gravity simply.** | `I am Nexara, a small AI assistant. I am Nexara...` (Failed) | `I am Nexara, a small AI assistant.` (Failed) | 0/1 |
| **Once upon a time...** | `Hello! I am Nexara, a small AI assistant. How can I help you today?` (Forgotten base story) | `I am a small rabbit, and I love to explore new places... He hopped through the forest...` (Fluent story) | 0/1 |

---

## 4. Failure Analysis & Diagnostics

Our target SFT score threshold was 5/8. Having scored **3/8**, we are halting active training to document key failure modes and implement improvements:

1. **Greedy Mode Mode Collapse (Identity Overfitting)**:
   The model has overfitted to the system greeting message: `Hello! I am Nexara, a small AI assistant. How can I help you today?`. Under greedy decoding, almost every conversational input prompts the model to output this exact phrase. This includes base story-writing prompts (e.g. `Once upon a time...` returns the identity phrase).
2. **Severe Capacity Constraint (6.8M parameters)**:
   The model is too small to grasp structural formatting (markdown bullet lists) or arithmetic logic. When asked to add numbers or list fruits, it falls back to repetitive loops of tokens.
3. **Loss Divergence of Tasks**:
   The validation loss and perplexity dropped significantly compared to the 500-example run (validation loss: 3.19 vs 3.95). However, this represents overfitting to formatting patterns rather than actual comprehension.

### Recommended Fixes for the Next Iteration

- **Identity Set Tuning**: Reduce the representation of identity data or vary responses to avoid a single dominant sentence attractor.
- **Task Simplification**: Remove tasks like multi-digit math and complex markdown structures from the SFT dataset. A 6.8M model can only learn conversational tone, not new logical capabilities.
- **Training Hyperparameters**:
  - Reduce learning rate from `5e-5` to `2e-5`.
  - Introduce early stopping on task-specific metrics, not just global validation loss.
- **Inference Parameter Adjustments**: Sampled decoding (T=0.7 - 0.9) must be the default configuration. Under greedy decoding, the model collapses to identity loops.

---

## 5. Candidate 10k/100k SFT Pool Preparation

As requested, we prepared a future candidate SFT pool containing up to 100k raw examples, applying our filters and balancing rules:

- **Output File**: `datasets/processed/sft_raw_pool_100k.jsonl`
- **Total Compiled Raw Candidates**: 52,084
- **Final Candidates After Deduplication**: 47,177
- **Duplicate Entries Filtered**: 4,907 (9.4% redundancy)
- **Source Balancing Distribution (Max 40,000 per source)**:
  - `stanford_alpaca_filtered`: 37,141 entries (100% of clean pool)
  - `dolly_15k_filtered`: 9,836 entries (98.9% of clean pool)
  - `synthetic_conversations`: 200 entries (reduced from 5,000 due to template-based deduplication)

### Candidate Pool Diagnostics
1. **Deduplication**: Successfully identified that generating 5,000 synthetic dialogues with limited templates leads to high repetition, pruning them down to 200 unique dialogues.
2. **Quality Scoring**: Heuristics prioritized shorter, clean text outputs that conform to standard capitalization and sentence structures, making the pool highly accessible to low-capacity models.
3. **Safety & Capability Refusals**: Filtered out code templates, complex calculus/algebra, and web-search claims.
