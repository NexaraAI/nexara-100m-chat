# Nexara Tiny 100M Pretraining Final Report

This report summarizes the pretraining phase of the **Nexara Tiny 100M** parameter model, documenting model architecture, training performance, quantitative loss convergence, and final checkpoint details.

---

## 1. Model Architecture & Config

* **Architecture**: Decoder-Only Transformer (Causal Language Model)
* **Total Parameters**: **97,555,968** (with tied embeddings)
* **Layers (`n_layers`)**: 12
* **Attention Heads (`n_heads`)**: 12
* **Embedding Dimension (`embedding_dim`)**: 768
* **FFN Hidden Dimension**: 3072 (`mlp_ratio`: 4.0)
* **Max Sequence Length**: 512 tokens
* **Vocabulary Size**: 16,384 (16k BPE trained on the pretraining corpus)
* **Positional Embeddings**: RoPE (Rotary Position Embeddings)

---

## 2. Quantitative Performance & Loss Convergence

Stage C pretraining completed early at step **97,000** due to convergence of validation metrics. The run showed highly stable gradient norms and clean loss decay.

### Key Milestones Table

| Metric / Milestone | Stage A (1,000 steps) | Stage B (10,000 steps) | Stage C Resume (40,500 steps) | Stage C Best (94,500 steps) | Stage C Final (97,000 steps) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Training Loss** | ~2.95 | ~1.55 | 1.2770 | 1.2510 | 1.2524 |
| **Validation Loss** | — | 1.4375 | 1.3137 | **1.2510** | 1.2524 |
| **Validation Perplexity** | 10.01 | 4.2105 | 3.7199 | **3.4937** | 3.4986 |
| **Throughput** | ~22k tokens/s | ~212k tokens/s | ~212k tokens/s | ~212k tokens/s | ~212k tokens/s |

* **Hardware Optimization**: NVIDIA A100-SXM4-40GB GPU running TF32, `torch.compile` compilation, and asynchronous multi-worker dataloading.
* **Loss Curve Trend**: Constant downward trend during pretraining, showing clean convergence. Early stopping was triggered as validation loss stabilized around **1.25**.

---

## 3. Checkpoint Inventory & Hashes

The final models and metadata have been downloaded locally to `C:\Nexara\checkpoints\stage1_100m\stage_c_100000\` and integrity-verified:

* **`best.pt`** (1,221,118,373 bytes)
  * *Description*: Checkpoint at the lowest validation loss (step 94,500).
  * *SHA256*: `c7f81098708aa54dbf81d4eb648555004802d1cb0794e4bc0122f13dfa9abab5`
* **`latest.pt`** (1,221,119,819 bytes)
  * *Description*: The latest checkpoint saved before pretraining stopped (step 97,000).
  * *SHA256*: `fffb6c4a074b524f1c48bbf0a43f8d480252b3f7f0b1ab31e8fe9a7c1e86c7cc`
* **`final.pt`** (1,221,119,352 bytes)
  * *Description*: Final copy of the weights at step 97,000.
  * *SHA256*: `fe33efdd33003af276dcc6ff0bd711e03a454972901b1543a3ffcb741d075313`

---

## 4. Next Phase: Downstream Fine-Tuning (Stage 2 SFT)

With a strong, fluent base model checkpoint (`best.pt` with Val Loss `1.2510`), the model is ready for Stage 2 Supervised Fine-Tuning:
1. **Curriculum Blending**: Merge the SFT prompt-response identity dataset with task instructions.
2. **Instruction Following**: Train with assistant-only loss masking to teach the model formatting, conversational flow, and task obedience.
