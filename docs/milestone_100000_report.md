# Phase 1.4B GPU Training — Step 100,000 Milestone Report

This milestone report covers the evaluation of the 6.8M parameter Nexara model at step 100,000 of training, comparing metrics and generation quality against previous milestones.

## 1. Quantitative Evaluation Summary

Validation loss and perplexity converged cleanly to their final values at 100,000 steps. Validation perplexity reached **5.18**, and the heuristic coherence score finished at **73.90%**.

| Metric | Step 5,000 | Step 10,000 | Step 15,000 | Step 20,000 | Step 30,000 | Step 40,000 | Step 50,000 | Step 75,000 | Step 100,000 (Final) | Trend / Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Training Loss** | 2.23 | 1.94 | 1.86 | 1.73 | 1.86 | 1.78 | 1.83 | 1.77 | 1.81 | **Decreasing** (Good) |
| **Validation Loss** | 2.13 | 1.90 | 1.82 | 1.78 | 1.74 | 1.71 | 1.69 | 1.66 | 1.64 | **Decreasing** (Good) |
| **Validation Perplexity** | 8.40 | 6.71 | 6.20 | 5.95 | 5.71 | 5.54 | 5.44 | 5.26 | 5.18 | **Decreasing** (Good) |
| **Heuristic Coherence Score** | 63.64% | 74.74% | 78.56% | 73.13% | 68.91% | 77.03% | 69.14% | 80.58% | 73.90% | **Stable** (Good) |
| **Bigram Repetition Rate** | 7.65% | 7.98% | 7.51% | 7.35% | 7.92% | 8.93% | 7.20% | 7.26% | 7.04% | **Stable** (Good) |
| **Trigram Repetition Rate** | 1.61% | 1.87% | 1.45% | 1.24% | 1.26% | 2.25% | 0.92% | 1.30% | 1.05% | **Stable** (Good) |
| **Token Entropy** | 7.2787 | 7.1681 | 7.2842 | 7.2208 | 7.3447 | 7.2660 | 7.2968 | 7.2310 | 7.2814 | **Stable** (Good) |
| **Average Sentence Length** | 7.62 words | 7.73 words | 8.78 words | 8.63 words | 8.04 words | 8.77 words | 8.92 words | 7.91 words | 7.80 words | **Stable** (Good) |

---

## 2. Qualitative Generation

Below is the final text generated from standard prompts at the step 100,000 milestone.

### PROMPT: "Once upon a time"
* **Output**: *"Once upon a time there was a little girl called Annie. Annie was only three years old, and she loved to play. One day Annie went to the park with her mom. She saw a big slide and ran to it. It felt so soft in her hands. Annie's mom said "Let's go play on the slide together". So they slid down the slide together. Annie liked it so much that she ran after her mom. When she got to the bottom, Annie had a big smile on her face. Her mom said "It's nice to share things with each other". Annie laughed and said "Yes, I love sharing." Then they both"*

### PROMPT: "The little dog was"
* **Output**: *"The little dog was running around in the garden. He loved to chase butterflies and jump high. But then, suddenly he slipped on a big rock. It hurt his foot while he was running. The little dog started to cry because he couldn't catch any leaves or flowers. Then, he saw his friend, the rabbit. The rabbit had a big smile on her face. "What's wrong?" the little dog asked. The rabbit replied, "I can't catch anything, but I have a plan." The rabbit went and got some tape and attached it to the rock. "Now you can play with it again!" said the little dog. From that"*

### PROMPT: "The rabbit saw a flower"
* **Output**: *"The rabbit saw a flower in the garden. He hopped over and stopped to smell it. It was soft, sweet and smelled nice. The rabbit decided to explore the flower and he started walking around it. But suddenly, he heard something. It was loud coming from the flower. It said "Hello! Who are you?" The rabbit was confused. He didn't know who was talking to him. He slowly came closer and said "Hi!" The flower replied "I'm your new friend!" They became friends and they played together every day. Every morning the rabbit would wake up to say goodbye to the flower and go back to playing."*

---

## 3. Export Status & Checkpoint Inventory

Model export has successfully completed, stripping training states and optimizer history to generate clean inference weights.

* **Export Status**: Completed (clean weights generated successfully).
* **Inventory of Remote Checkpoints** (stored on Lightning AI under `/home/zeus/Nexara/checkpoints/stage1/`):
  - `clean_model.pt` (35 MB): Clean inference-only model weights.
  - `best.pt` (87 MB): Full training checkpoint at the best validation step.
  - `latest.pt` (87 MB): Full training checkpoint at step 100,000.
  - `final.pt` (87 MB): Duplicate of the final step 100,000 training checkpoint.
  - `config.json` (2.3 KB): Architectural configuration file.

---

## 4. Key Performance Highlights

* **Final Training Step**: 100,000
* **Best Checkpoint Step**: 99,500 (Validation Loss: **1.64441**, Perplexity: **5.1780**)
* **Final Training Loss**: 1.8065
* **Final Validation Loss**: 1.6450 (Validation Perplexity: **5.1810**)
* **Heuristic Coherence Score**: 73.90%
* **Bigram / Trigram Repetition Rate**: 7.04% / 1.05%

---

## 5. Phase 2 Recommendations & Next Steps

With pretraining complete, the next phase is **Phase 2: Instruction Tuning**.
1. **Model Release**: Deploy `clean_model.pt` and `config.json` to Hugging Face or GitHub Releases.
2. **Supervised Fine-Tuning (SFT)**: Fine-tune the base model on story-oriented prompt/response instruction datasets to enable better instruction-following capabilities (e.g. generating stories with specific endings, morals, or characters on command).
3. **Evaluation Framework**: Establish a prompt-evaluation benchmark suite to verify fine-tuning alignment.
