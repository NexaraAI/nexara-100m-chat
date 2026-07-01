# Nexara Tiny 100M Training Plan

This document outlines the training strategy, datasets, licensing, compute requirements, and risks for pretraining the **Nexara Tiny 100M** parameter model.

---

## 1. Chosen Model Architecture

* **Backbone**: Decoder-Only Transformer (Causal Language Model)
* **Model Parameters**: **97,555,968** (under tied embeddings)
* **Number of Layers (`n_layers`)**: 12
* **Number of Attention Heads (`n_heads`)**: 12
* **Embedding Dimension (`embedding_dim`)**: 768
* **Feed-Forward Hidden Dimension**: 3072 (`mlp_ratio`: 4.0)
* **Max Sequence Length (`max_seq_len`)**: 512 tokens
* **Dropout**: 0.1
* **Tied Embeddings**: True
* **Bias Terms**: False
* **Positional Embeddings**: RoPE (Rotary Position Embeddings)

---

## 2. Tokenizer Decision & Tradeoffs

For **Nexara Tiny 100M**, we adopt a **16,384 (16k) BPE Tokenizer** trained on our mixed pretraining corpus.

### Tradeoff Evaluation

| Metric | 8,192 Vocab (Old) | 16,384 Vocab (Chosen) | 32,768 Vocab |
| :--- | :---: | :---: | :---: |
| **Embedding Size** | ~6.3 MB | ~12.6 MB | ~25.2 MB |
| **% of 100M Budget** | ~6.4% | ~12.8% | ~25.8% |
| **Text Compression** | Poor (~4.1 char/token) | Good (~4.8 char/token) | Excellent (~5.2 char/token) |
| **Context Window Density**| Low (fewer words in 512) | Medium | High (more words in 512) |
| **Out-of-Vocab (OOV)** | High on world text | Low on world text | Very low |

* **Decision Rationale**: A 16k vocabulary provides the ideal sweet spot. It achieves a 17% improvement in text compression and semantic density in the context window over the old 8k tokenizer, while reserving 87.2M parameters for the transformer layers (compared to 32k which would consume a massive 25.8% of the parameter budget).
* **Compatibility**: The old 8192 tokenizer config (`tokenizer/nexara-bpe.json`) remains completely unmodified to preserve the archived 6.8M prototypes.

---

## 3. Data Strategy & Pretraining Mix

To move beyond the narrative-only limitation of TinyStories and build basic world knowledge, we use a clean, curated mixed pretraining dataset:

1. **TinyStories (50% weight)**:
   * *Role*: Maintains strong syntactic fluency, pronoun resolution, and simple narrative structures.
   * *Licensing*: MIT (released by Microsoft Research).
2. **Simple Wikipedia (30% weight)**:
   * *Role*: Introduces factual knowledge, history, geography, and science simplified for readability.
   * *Licensing*: CC-BY-SA 4.0.
3. **Project Gutenberg - Selected Educational/Public Domain (10% weight)**:
   * *Role*: Long-context vocabulary enrichment, classical children's literature, and basic logic.
   * *Licensing*: Public Domain / Out of Copyright.
4. **WikiText-103 / OpenWebText Clean Subset (10% weight)**:
   * *Role*: Exposure to general world facts, structured writing, and basic programming/logical reasoning tokens.
   * *Licensing*: CC-BY-SA / Public Domain.

---

## 4. Compute, Storage, and Timeline Estimates

We target a pretraining dataset size of **500 Million tokens** (approximately 5 epochs of training for a 100M token unique corpus).

* **Storage Needs**:
  * Raw text: ~1.2 GB
  * Binary token cache (`uint16` representations): ~1.0 GB
  * Checkpoints (100M parameters, fp32 for training): ~400 MB per checkpoint
  * Disk space required: ~5 GB total.
* **Compute Needs**:
  * GPU: Single NVIDIA Tesla T4 GPU (available on Lightning AI).
  * Precision: `fp16` mixed precision training.
* **Expected Training Time**:
  * With a throughput of ~60,000 tokens/sec on Tesla T4:
    * 100,000 steps (batch size 32, grad accum 4 = 65,536 tokens per step):
    * Total tokens processed: \(100,000 \times 65,536 = 6.55\) Billion tokens (approx 13 epochs over the 500M token corpus).
    * Total training hours: \(\frac{6.55 \times 10^9}{60,000 \times 3600} \approx 30\) hours.

---

## 5. Risks and Mitigations

1. **CUDA Out of Memory (OOM)**:
   * *Risk*: The 100M model uses ~14x more activation memory than the 6.8M model.
   * *Mitigation*: Enable gradient accumulation (`steps=4`), reduce per-device batch size if needed, and use mixed-precision `fp16` or `bf16`.
2. **Catastrophic Forgetting / Loss Divergence**:
   * *Risk*: Large models can easily diverge at early stages with high learning rates.
   * *Mitigation*: Implement linear learning rate warmup (1,000 steps) and cosine learning rate decay. Maintain NaNs detection checks.
3. **Storage Exhaustion**:
   * *Risk*: Storing too many checkpoints (400MB each) will fill the disk.
   * *Mitigation*: Maintain a strict checkpoint rotation policy (retaining only `best.pt`, `latest.pt`, and the last 5 step checkpoints).

---

## 6. Staged Training Plan

Training is structured into four distinct verification stages to guarantee stability:

* **Stage A (Smoke Validation)**: 1,000 steps to check for NaN/Inf anomalies, gradients, and validation losses.
* **Stage B (Early Pretraining)**: 10,000 steps. Monitor learning rate decay and vocabulary utilization.
* **Stage C (Mid-Pretraining)**: 50,000 steps. Run generation benchmarks to check coherence.
* **Stage D (Full Convergence)**: 100,000 steps or until validation loss plateaus.
