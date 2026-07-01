# Model Card for Nexara-0.2-chat

Nexara-0.2-chat is a 6.8 million parameter decoder-only transformer model, instruction-tuned from **Nexara-0.1-base** (trained on TinyStories-style data). It is designed to demonstrate basic chat alignment, polite conversational helper responses, and name-identity compliance.

## Model Details

- **Developed by**: NexaraAI
- **Model Type**: Decoder-Only Transformer (Causal Language Model)
- **Base Model**: Nexara-0.1-base (100,000 steps pretraining)
- **Language(s)**: English (simplified children's vocabulary)
- **License**: MIT
- **Tuning Method**: Supervised Fine-Tuning (SFT) on a 3,000-example blended instruction dataset

## Architectural Specifications

- **Layers**: 6
- **Attention Heads**: 8
- **Embedding Dimension**: 256
- **Vocabulary Size**: 8,192 (Byte-Pair Encoding)
- **Maximum Sequence Length**: 256 tokens
- **Tied Embeddings**: True
- **Positional Embeddings**: RoPE (Rotary Position Embeddings)

## Intended Use

- **Conversational Demos**: Polite greetings, identity verification, and simple conversational turns.
- **Storytelling**: Story generation starting from custom prompts (when run without the chat template wrapper).
- **Educational / Research**: Small-scale demonstration of instruction tuning, alignment, and parameter-efficiency bounds.

## Limitations

- **Logical Capabilities**: Cannot perform arithmetic (fails on `3 + 4`), writing code, or structured formatting (e.g. markdown lists).
- **Mode Collapse (Greedy)**: Greedy decoding path is heavily biased toward the identity greeting. **Do not use greedy decoding.**
- **Factual Claims**: May hallucinate simple facts. It does not have access to real-time information or the internet.

## Recommended Decoding Configurations

We recommend using the following sampling settings for all conversational interfaces:

- **Temperature**: 0.7 - 0.9
- **Top-K**: 40
- **Top-P**: 0.95
- **Repetition Penalty**: 1.15 - 1.2
- **Do Sample**: True
- **Max New Tokens**: 128
