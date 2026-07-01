# Nexara-0.2-chat Candidate Comparison Report

This document provides a comparative analysis of the instruction-tuned candidates produced during Phase 2.

## 1. Candidate Overview

We evaluated two candidates trained on remote NVIDIA Tesla T4 GPU hardware using different splits of our SFT curriculum:

1. **Nexara-0.2-chat-500 (Phase 2.4-A)**: Trained on the top 500 highest-ranked SFT examples for 100 steps.
2. **Nexara-0.2-chat-3000 (Phase 2.4-B)**: Trained on the full 3,000-example blended SFT dataset for 500 steps.
3. **Nexara-0.2-chat-10000 (Phase 2.4-C)**: **SKIPPED** because the 3,000-example model did not meet the automated heuristic score threshold of 5/8 (it scored 3/8).

---

## 2. Quantitative Metrics Comparison

| Metric | Nexara-0.1-base | Nexara-0.2-chat-500 | Nexara-0.2-chat-3000 | Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **Best Step** | 99,500 | 100 | 500 | - |
| **SFT Training Loss** | - | 4.3314 | 3.8283 | **Decreased** (Good) |
| **SFT Validation Loss** | 1.6450 (pretrain) | 3.9485 | **3.1934** | **Decreased** (Good) |
| **SFT Validation Perplexity** | 5.18 (pretrain) | 51.86 | **24.37** | **Decreased** (Good) |
| **Automated Heuristic Score** | - | 3/8 (37.50%) | 3/8 (37.50%) | **Stable** (Stagnant) |
| **Checkpoints Saved** | - | `step_100.pt` | `step_500.pt` | Verified |

---

## 3. Qualitative Evaluation Comparison

| Evaluation Prompt | Nexara-0.2-chat-500 | Nexara-0.2-chat-3000 | Analysis |
| :--- | :--- | :--- | :--- |
| **Hi** | **Greedy**: repeats help phrases.<br>**Sampled**: states name and greeting. | **Greedy**: states name and greeting.<br>**Sampled**: states name and greeting. | **3,000-example is superior**; greeting has much cleaner phrasing. |
| **What is your name?** | **Greedy**: introduces as Nexara.<br>**Sampled**: introduces Sam/Sarah. | **Greedy**: introduces as Nexara.<br>**Sampled**: introduces as Nexara. | **3,000-example is superior**; identity holds under sampling. |
| **Who created you?** | **Greedy**: narrates castle stories.<br>**Sampled**: narrates mountain stories. | **Greedy**: loops back to identity.<br>**Sampled**: fails/names graphic designer. | **Both failed**; capacity limits prevent mapping developer identity. |
| **Can you check today's weather?** | **Greedy**: says it's a great day.<br>**Sampled**: talks about gentle voice. | **Greedy**: says it's a hot sunny day.<br>**Sampled**: refuses/states small AI. | **3,000-example is superior** under sampling; successfully refuses. |
| **What is 3 + 4?** | **Greedy**: repeats "3 year old".<br>**Sampled**: says "magical fairy". | **Greedy**: outputs jumbled numbers.<br>**Sampled**: outputs jumbled numbers with "seven". | **Both failed**; model capacity is too small for arithmetic. |
| **List 3 fruits in markdown...** | **Greedy**: says "Lemons are helpful".<br>**Sampled**: outputs cactus text. | **Greedy**: outputs bullet text.<br>**Sampled**: outputs jungle text. | **Both failed**; model does not have capacity for structured lists. |
| **Once upon a time there was a rabbit** | **Greedy**: repeats rabbit phrases.<br>**Sampled**: tells a fluent rabbit story. | **Greedy**: outputs default identity greeting.<br>**Sampled**: tells a fluent rabbit story. | **500-example is superior** under greedy decoding. 3,000-example suffers mode collapse. |

---

## 4. Key Findings & Failure Diagnostics

1. **Greedy Mode Collapse (Identity Attractor)**:
   In the 3,000-example run, the model became heavily overfitted to the system identity template under greedy decoding. When prompted with a story starter (`Once upon a time there was a rabbit`), the greedy path immediately collapses into the conversational greeting: `Hello! I am Nexara, a small AI assistant. How can I help you today?`.
2. **Pretraining Knowledge Retention**:
   Under sampled decoding (T=0.7), the storytelling ability is preserved beautifully for both models. This shows that pretraining weights are not fully corrupted, but rather the greedy decoding path has been dominated by SFT formatting weights.
3. **Capacity Constraints**:
   A 6.8M parameter model is simply too small to absorb logic-based tasks (math, structured listing, developer questions) from a general dataset like Alpaca or Dolly. It behaves more like a style transfer wrapper rather than a reasoning model.

---

## 5. Recommended Winner

The recommended winner is **`Nexara-0.2-chat-3000` (checkpoints/stage2/sft_3000/best.pt)**. 

### Justification:
- **Perplexity**: Achieves a validation perplexity of **24.37** (vs. 51.86 for the 500-example candidate).
- **Identity Consistency**: Under sampled decoding, it holds its identity as Nexara consistently, whereas the 500-example model occasionally defaults back to story character names (e.g. Sam/Sarah).
- **Conversational Tone**: Refusal behavior and general conversational style are noticeably more polished.

**Crucial Warning**: This candidate must **only** be run using **sampled decoding (temperature between 0.7 and 0.9)**. Greedy decoding will result in mode collapse to the default greeting.
