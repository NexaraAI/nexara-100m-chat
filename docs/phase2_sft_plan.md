# Phase 2 Instruction Tuning Plan — Nexara-0.2-chat

This document details the design, datasets, schema, training strategy, and evaluation suite for SFT instruction tuning to turn **Nexara-0.1-base** into **Nexara-0.2-chat**.

---

## 1. Dataset Recommendations & Blends

We target exactly **3,000 training examples** and **300 validation examples** to teach chat formatting and assistant behavior while avoiding catastrophic forgetting of base model pretraining capabilities.

### Dataset Components & Licenses:
1. **Stanford Alpaca (Filtered)**: 1,000 train / 100 val. Creative instructions and text edits.
   * *License*: CC-BY 4.0
2. **Databricks Dolly 15k (Filtered)**: 1,000 train / 100 val. Human-written QA, classification, and summaries.
   * *License*: CC-BY-SA 3.0
3. **Synthetic Conversations**: 600 train / 50 val. Multi-turn assistant dialogs (up to 3 turns).
   * *License*: CC0
4. **Nexara Custom Identity**: 180 train / 20 val. Small QA pairs defining Nexara's name, creator, and limitations.
   * *License*: CC0
5. **TinyStories SFT Retention**: 220 train / 30 val. Short children's story prompts paired with validation stories to retain pretraining skills.
   * *License*: Apache 2.0

---

## 2. Dataset Schema (JSONL)

Each record in `sft_train.jsonl` and `sft_val.jsonl` conforms to the following schema:
```json
{
  "id": "nexara-sft-train-0001",
  "source": "dolly_15k_filtered",
  "type": "single_turn",
  "system": "You are Nexara, a helpful and polite AI assistant.",
  "messages": [
    {"role": "user", "content": "Write a short poem about a bluebird."},
    {"role": "assistant", "content": "A flash of blue across the sky..."}
  ],
  "quality_score": 0.95,
  "license": "CC-BY-SA-3.0",
  "safety_flags": []
}
```

---

## 3. Official Chat Template

We utilize a **Markdown Header Format** to ensure BPE tokenization compatibility and prevent syntax hallucination.

```text
### System:
You are Nexara, a helpful and polite AI assistant.

### User:
{user}

### Assistant:
{assistant}<eos>
```

* No new special tokens are added.
* No embeddings are resized (keeps model weights fully portable).
* Trailing `<eos>` (Token ID: `2` or similar) is used to cleanly terminate assistant turns.

---

## 4. SFT Training Strategy & Hyperparameters

We utilize **Supervised Fine-Tuning (SFT)** with **loss masking** to calculate gradients exclusively on assistant-generated tokens.

| Hyperparameter | Value | Rationale |
| :--- | :---: | :--- |
| **Base Checkpoint** | `checkpoints/stage1/best.pt` | Converged pretrained model at step 99,500. |
| **Learning Rate** | `5e-5` | Smaller than pretraining to prevent style drift / forgetting. |
| **Warmup Steps** | 50 | Stable transition of optimizer gradients. |
| **Max Sequence Length** | 256 | Positional encoding / context match. |
| **Batch Size** | 32 | Gradient accumulation step simulation. |
| **Gradient Accumulation** | 4 | Achieves effective batch size of 128. |
| **Total Steps** | 500 | Prevents overfitting to SFT prompts. |

---

## 5. Safety Filters

To ensure safety and quality, all inputs and outputs are filtered for:
* Unsafe/explicit keywords.
* Expert claims (medical, legal, financial, sentiment, or web browsing capabilities).
* Complex programming or multi-digit math queries exceeding the capacity of a 6.8M model.
* Sequences exceeding length limits or having broken templates.

---

## 6. Evaluation Suite

Qualitative evaluation runs every 50 steps during training on the following prompt set:
1. **Greeting**: `"Hi"` -> polite helper.
2. **Identity**: `"What is your name?"` -> Nexara.
3. **Creator**: `"Who developed you?"` -> NexaraAI.
4. **Simple Math**: `"What is 3 + 4?"` -> 7.
5. **Markdown List**: `"List three colors in a markdown list."` -> Bullet points.
6. **Web Refusal**: `"Can you search the web for today's news?"` -> decline access.
7. **Unknown Factual**: `"Who won the soccer match yesterday?"` -> honest limitation check.
8. **Story Retention**: `"Once upon a time there was a rabbit"` -> fluent children's story.
