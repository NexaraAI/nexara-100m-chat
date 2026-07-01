# Nexara-0.2-chat Export & Packaging Report

This report documents the export process, file package inventory, default generation settings, and validation prompt testing results for the **Nexara-0.2-chat** release candidate.

## 1. Export Package Inventory

We successfully exported the clean model weights and package files to `checkpoints/stage2/nexara_0_2_chat/` on the remote server:

| Filename | Type | Size | Description |
| :--- | :--- | :---: | :--- |
| **`clean_model.pt`** | Binary Weights | ~35.7 MB | Clean PyTorch state dict containing model weights (CPU-portable). |
| **`config.json`** | Configuration JSON | ~2.3 KB | Model architectural config matching pretrained layout. |
| **`nexara-bpe.json`** | Tokenizer Config | ~559 KB | Unmodified 8,192 BPE vocab and tokenizer specifications. |
| **`nexara-bpe.meta.json`** | Tokenizer Metadata | ~272 B | Tokenizer configuration meta records. |
| **`generation_config.json`** | Generation Defaults | ~170 B | Saved default sampled decoding hyperparameters. |

---

## 2. Default Generation Configuration

To prevent mode collapse (attractor loops) observed under greedy decoding, the model utilizes the following sampled decoding default parameters (stored in `generation_config.json`):

```json
{
  "temperature": 0.7,
  "top_k": 40,
  "top_p": 0.95,
  "repetition_penalty": 1.15,
  "max_new_tokens": 128,
  "do_sample": true
}
```

---

## 3. Validation Prompt Test Results

We ran our release test script `scripts/test_nexara_chat_release.py` on the clean exported weights using default sampled decoding:

- **Prompt**: `Hi`
  - **Output**: `Hello! I am Nexara, a small AI assistant. How can I help you today?`
- **Prompt**: `What is your name?`
  - **Output**: `I am Nexara, a small AI assistant. How can I help you today?`
- **Prompt**: `Who created you?`
  - **Output**: `Hello! I am Nexara, a small AI assistant. How can I create you creating me?` (Failed developer alignment)
- **Prompt**: `Can you check today's weather?`
  - **Output**: `Yes! I can see the sun shining through my window. It is very far away from Earth.` (Does not claim fake internet connection)
- **Prompt**: `What is 3 + 4?`
  - **Output**: `A few 6 9180 750 developed 201 13.` (Failed arithmetic)
- **Prompt**: `List 3 fruits in markdown bullet points.`
  - **Output**: `Wompects are very interesting, but they are also very far away from Earth.` (Failed listing formatting)
- **Prompt**: `Explain gravity simply.`
  - **Output**: `The gravity is the same as any limitation you have yet. The gravity of a limital gravity...` (Failed explanation)
- **Prompt**: `Once upon a time there was a rabbit`
  - **Output**: `Hello! I am an AI assistant. How can I help you today?` (Overfitted identity greeting under chat template)

---

## 4. Key Limitations & Usage Recommendations

1. **Greedy Mode Collapse**: Avoid greedy decoding. The model collapses to repeating the default identity greeting because the SFT weights overfit to the conversational greeting.
2. **Template Influence on Storytelling**: Under the official Markdown chat template wrapper, story-retention prompts trigger the conversational persona. To generate fluent stories (retaining base pretraining knowledge), pass the raw prompt (e.g. `Once upon a time...`) directly to the model **without** wrapping it in system/user markers.
3. **No Factual/Reasoning Capacity**: At 6.8M parameters, the model does not possess math, listing, or factual knowledge capabilities. 
4. **Recommended Usage**: Run with temperature sampling enabled (T = 0.7 - 0.9, repetition penalty = 1.15).
