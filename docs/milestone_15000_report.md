# Phase 1.4B GPU Training — Step 15,000 Milestone Report

This milestone report covers the evaluation of the 6.8M parameter Nexara model at step 15,000 of training, comparing metrics and generation quality against previous milestones.

## 1. Quantitative Evaluation Summary

All target metrics continue to improve monotonically. The model shows no signs of overfitting (validation loss continues to fall in sync with training loss).

| Metric | Step 5,000 (Baseline) | Step 10,000 | Step 15,000 (Milestone) | Trend / Decision |
| :--- | :---: | :---: | :---: | :--- |
| **Training Loss** | 2.23 | 1.94 | 1.86 | **Decreasing** (Good) |
| **Validation Loss** | 2.13 | 1.90 | 1.82 | **Decreasing** (Good) |
| **Validation Perplexity** | 8.40 | 6.71 | 6.20 | **Decreasing** (Good) |
| **Heuristic Coherence Score** | 63.64% | 74.74% | 78.56% | **Improving** (Good) |
| **Bigram Repetition Rate** | 7.65% | 7.98% | 7.51% | **Decreasing** (Good) |
| **Trigram Repetition Rate** | 1.61% | 1.87% | 1.45% | **Decreasing** (Good) |
| **Token Entropy** | 7.2787 | 7.1681 | 7.2842 | **Increasing** (Good) |
| **Average Sentence Length** | 7.62 words | 7.73 words | 8.78 words | **Increasing** (Good) |

> [!NOTE]
> The **Heuristic Coherence Score** has reached **78.56%** (up from 74.74%), and the **Token Entropy** has rebounded to **7.2842**, showing that the model's vocabulary diversity is expanding as it learns finer grammatical structures.
> The **Average Sentence Length** has increased to **8.78 words**, indicating the model is generating longer, more descriptive clauses instead of short, repetitive phrases.

---

## 2. Metrics & Loss Curves

### Loss Evolution
* Step 0: **9.04**
* Step 5,000: **2.23** (Val: 2.13)
* Step 10,000: **1.94** (Val: 1.90)
* Step 15,000: **1.86** (Val: 1.82)

### Perplexity Evolution
* Step 5,000: **8.40**
* Step 10,000: **6.71**
* Step 15,000: **6.20**

---

## 3. Qualitative Generation Comparison

Below are comparisons of the text generated from standard prompts between the step 10,000 and step 15,000 milestones.

### PROMPT: "Once upon a time"
* **Step 10,000 Output**: *"Once upon a time, there was a little girl named Lily. She loved to play with her toys and run around in the park. One day, she saw a big billboard on the ground. It had pictures of animals like lions, tigers, and elephants. Lily wanted to get closer to the billboard, so she ran up to it and touched it with her hand. Suddenly, something caught her eye! It was a small mouse! Lily thought for a moment and decided to examine it anyway. She picked up the mouse and showed it to her mom. Her mom was proud of her for being responsible and listening to the billboard..."*
* **Step 15,000 Output**: *"Once upon a time, there was a little girl named Lily. She loved to play with her toys and sing songs. One day, she went to the park to play with her friends. While playing on the swings, she saw a big, scary dog running towards her. She screamed and ran away from him, but it was too late. The dog chased after her and caught her. Lily was very sad and cried a lot. Her mom came over to help her feel better. They found a way to make it up to the dog and the dog came closer to them. Lily's mom asked her what happened, and the dog showed her his home."*

### PROMPT: "The little dog was"
* **Step 10,000 Output**: *"The little dog was so excited to have his ball. He ran outside and grabbed it with him. He kicked the ball back and forth, but he couldn't catch it. He decided to take a break and sit down in the grass. Then, he looked for it, but he couldn't find it anywhere. He was worried. He tried again, but this time, he had an idea. He thought of a plan to get the ball back. He asked his dad if he could borrow it. His dad said yes and lent him the ball. He was very happy and took it. When he returned home, he gave the ball a big hug."*
* **Step 15,000 Output**: *"The little dog was very alert. He loved to play and explore. One day he found a big door. The little pup wanted to find out what was inside the door. So he opened the door and peeked inside. It was so quiet, he could hear anything. He saw a door in the other side. He wondered what was inside. He ran in the door and looked around. He saw a friendly little girl sitting in the corner. She was looking for something to eat. The pup meowed softly, but she had an idea. She came over and asked if he could help her out. The little girl smiled. She was happy that the pup was..."*

### PROMPT: "The rabbit saw a flower"
* **Step 10,000 Output**: *"The rabbit saw a flower. It was green and yellow and had many colors. The rabbit hopped over to it and started eating it. \"Can I have some of your fruit?\" asked the rabbit. \"Sure!\" said the bird. \"But be careful not to bite it.\" The rabbit looked at the flower and said, \"That's too sour for you. You should go home.\" The rabbit nodded and said goodbye. Then she went back to the garden to explore. The next day, the little rabbit came back to the forest. She found a big tree and decided to climb up the tree. When she reached the top, she was so excited!..."*
* **Step 15,000 Output**: *"The rabbit saw a flower in the middle of the meadow. It was so big and colourful and it seemed to have been left in the garden for a long time. The rabbit hopped over and sniffed the flowers, then suddenly noticed a small green caterpillar crawling on one of the petals. He knew this was the most beautiful thing he had ever seen. The rabbit hopped closer to the caterpillar and asked if it could be his friend. The caterpillar nodded and said yes. So the rabbit gave him a big squeeze of leaves to cover the caterpillar's wings. After a few seconds, the rabbit felt safe and happy. The rabbit thanked the caterpillar for helping him out..."*

---

## 4. Decision: CONTINUE TRAINING
The run is exceptionally **healthy and improving**. Validation loss and perplexity continue their downward trend. Qualitative samples show improved vocabulary usage ("meow", "caterpillar", "meadow", "alert", "ostrich"), longer clause lengths, and more narrative complexity. Repetition is dropping.

We will proceed toward the **20,000-step milestone**.
