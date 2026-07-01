# Nexara-0.2-chat Release Checklist & Packaging Guide

This document prepares the packaging, release, and export specifications for the **Nexara-0.2-chat-3000** model candidate.

---

## 1. Release Checklist

- [ ] **Export Checkpoint**: Strip training states from `checkpoints/stage2/sft_3000/best.pt` to create a lightweight inference-only checkpoint (`clean_model.pt`).
- [ ] **Hugging Face Package Export**: Generate custom modeling and configuration scripts, tokenizer configs, and README cards.
- [ ] **Chat CLI Testing**: Validate local execution on the CPU using the `inference.chat` module.
- [ ] **Integration Verification**: Ensure no regression of tokenizer behavior or story retention under default sampling settings.

---

## 2. Draft Model Card (`README.md` for Hugging Face)

```markdown
# Nexara-0.2-chat

Nexara-0.2-chat is a 6.8 million parameter decoder-only transformer model, instruction-tuned from **Nexara-0.1-base** (a model trained from scratch on the TinyStories dataset). It has been tuned on a curated blend of 3,000 SFT examples to align it to a conversational assistant persona.

### Model Architecture
- **Layers**: 6
- **Heads**: 8
- **Embedding Dimension**: 256
- **Sequence Length**: 256
- **Parameters**: 6.8M
- **Attention**: Causal Attention with Rotary Position Embeddings (RoPE)

### Hyperparameters (SFT)
- **Base Model**: Nexara-0.1-base (100k steps)
- **SFT Steps**: 500
- **Effective Batch Size**: 128
- **Learning Rate**: 5e-5
- **Optimizer**: AdamW
- **Precision**: FP16 Mixed Precision

### Intended Use
Conversational roleplay, text-editing/simplification, simple children's story generation, and basic chat alignment demonstrations.

### Known Limitations
- **Capacity Constraint**: At 6.8M parameters, the model cannot perform arithmetic, write programming code, or remember factual general knowledge.
- **Mode Collapse (Greedy)**: Greedy decoding leads to mode collapse where the model repeats its identity greeting.
- **Solution**: Enforce sampled decoding (Temperature = 0.7 - 0.9, top_p = 0.9).
```

---

## 3. Export Checkpoint Specifications

To strip training states and export the clean model weights:

```bash
# Run the export script on the best checkpoint
python scripts/export_checkpoint.py \
  --checkpoint checkpoints/stage2/sft_3000/best.pt \
  --output-pt checkpoints/stage2/sft_3000/clean_model.pt
```

This will produce:
1. `clean_model.pt` (inference-only PyTorch state dict).
2. `config.json` (architecture parameters).

---

## 4. Chat CLI Test & Inference Usage Notes

To test the conversational assistant locally on CPU:

```bash
# Run chat CLI with inference-only clean weights
python -m inference.chat \
  --checkpoint checkpoints/stage2/sft_3000/best.pt \
  --temperature 0.75 \
  --top-p 0.9
```

### Prompt Formatting
Ensure the input is wrapped in the official Markdown chat template:
```text
### System:
You are Nexara, a helpful and polite AI assistant.

### User:
{user_query}

### Assistant:
```

---

## 5. Next-Step Recommendations

1. **Vary Identity Data**: In the next iteration, expand the identity dataset with varied user prompts and distinct phrasing.
2. **Task Filtering**: Remove all math, logic, coding, and listing examples from the training split.
3. **Hyperparameter Tuning**: Run training with a lower learning rate (`2e-5`) and a longer schedule (`1,000 steps`) using a learning rate scheduler that decays to `2e-6`.
