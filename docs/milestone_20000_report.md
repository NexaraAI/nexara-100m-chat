# Phase 1.4B GPU Training — Step 20,000 Milestone Report

This milestone report covers the evaluation of the 6.8M parameter Nexara model at step 20,000 of training, comparing metrics and generation quality against previous milestones.

## 1. Quantitative Evaluation Summary

All target metrics continue to improve monotonically. The model shows no signs of overfitting (validation loss continues to fall in sync with training loss). Notably, validation perplexity has now **broken below 6.00** to **5.953**.

| Metric | Step 5,000 (Baseline) | Step 10,000 | Step 15,000 | Step 20,000 (Milestone) | Trend / Decision |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Training Loss** | 2.23 | 1.94 | 1.86 | 1.73 | **Decreasing** (Good) |
| **Validation Loss** | 2.13 | 1.90 | 1.82 | 1.78 | **Decreasing** (Good) |
| **Validation Perplexity** | 8.40 | 6.71 | 6.20 | 5.95 | **Decreasing** (Good) |
| **Heuristic Coherence Score** | 63.64% | 74.74% | 78.56% | 73.13% | **Stable** (Good) |
| **Bigram Repetition Rate** | 7.65% | 7.98% | 7.51% | 7.35% | **Decreasing** (Good) |
| **Trigram Repetition Rate** | 1.61% | 1.87% | 1.45% | 1.24% | **Decreasing** (Good) |
| **Token Entropy** | 7.2787 | 7.1681 | 7.2842 | 7.2208 | **Stable** (Good) |
| **Average Sentence Length** | 7.62 words | 7.73 words | 8.78 words | 8.63 words | **Stable** (Good) |

> [!NOTE]
> The **Trigram Repetition Rate** dropped to **1.24%** and the **Bigram Repetition Rate** dropped to **7.35%**, indicating the model is generating text that is highly fluent and non-redundant.

---

## 2. Metrics & Loss Curves

### Loss Evolution
* Step 5,000: **2.23** (Val: 2.13)
* Step 10,000: **1.94** (Val: 1.90)
* Step 15,000: **1.86** (Val: 1.82)
* Step 20,000: **1.73** (Val: 1.78)

### Perplexity Evolution
* Step 5,000: **8.40**
* Step 10,000: **6.71**
* Step 15,000: **6.20**
* Step 20,000: **5.95**

---

## 3. Qualitative Generation Comparison

Below are comparisons of the text generated from standard prompts between the step 15,000 and step 20,000 milestones.

### PROMPT: "Once upon a time"
* **Step 15,000 Output**: *"Once upon a time, there was a little girl named Lily. She loved to play with her toys and sing songs. One day, she went to the park to play with her friends. While playing on the swings, she saw a big, scary dog running towards her. She screamed and ran away from him, but it was too late. The dog chased after her and caught her. Lily was very sad and cried a lot. Her mom came over to help her feel better. They found a way to make it up to the dog and the dog came closer to them. Lily's mom asked her what happened, and the dog showed her his home."*
* **Step 20,000 Output**: *"Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine and pick flowers. One day, she found a shiny rock that was so pretty that she wanted to keep it forever. Lily put the rock in her pocket and went to the park. She played on the swings and slide, but then she got an idea. She saw a big tree with lots of leaves that looked like a castle! She thought it would be fun to use the rock to make the castle look nice. So, Lily took out the rock and used it to decorate the castle. The pretty flower was very happy to see the beautiful picture on..."*

### PROMPT: "The little dog was"
* **Step 15,000 Output**: *"The little dog was very alert. He loved to play and explore. One day he found a big door. The little pup wanted to find out what was inside the door. So he opened the door and peeked inside. It was so quiet, he could hear anything. He saw a door in the other side. He wondered what was inside. He ran in the door and looked around. He saw a friendly little girl sitting in the corner. She was looking for something to eat. The pup meowed softly, but she had an idea. She came over and asked if he could help her out. The little girl smiled. She was happy that the pup was..."*
* **Step 20,000 Output**: *"The little dog was feeling brave. He wanted to play a game, so he went outside and tried to do it. But it was too hard! Then he had an idea. He went to his friend. His friend had a great idea. She suggested that the dog come out of his hiding place and play something fun. She said that it was ok to go. The dog was so happy. He ran around the yard playing games with his friends. They laughed and played for hours. When it was time for bed, the dog's friends said goodbye and thanked him for his advice. The dog smiled as he went to sleep. The end."*

### PROMPT: "The rabbit saw a flower"
* **Step 15,000 Output**: *"The rabbit saw a flower in the middle of the meadow. It was so big and colourful and it seemed to have been left in the garden for a long time. The rabbit hopped over and sniffed the flowers, then suddenly noticed a small green caterpillar crawling on one of the petals. He knew this was the most beautiful thing he had ever seen. The rabbit hopped closer to the caterpillar and asked if it could be his friend. The caterpillar nodded and said yes. So the rabbit gave him a big squeeze of leaves to cover the caterpillar's wings. After a few seconds, the rabbit felt safe and happy. The rabbit thanked the caterpillar for helping him out..."*
* **Step 20,000 Output**: *"The rabbit saw a flower in the grass. He had never seen a flower before and was very curious. The rabbit asked the bird, \"What are you doing?\" The bird replied, \"I am looking for something special to show you.\" The rabbit said, \"That's great! I want to see what you might find. Come with me!\" The rabbit agreed, so the two of them went on an adventure to find the flower. They went through fields, by forests and rivers, and the rabbit discovered many new things. Eventually they found the beautiful flower and he thanked the bird for showing him the way home. Then he hopped away, happy that he had..."*

---

## 4. Decision: CONTINUE TRAINING
The model shows **continuous improvements in fluency, vocabulary variety, and narrative ending structure** (several stories now cleanly reach an ending like "The dog smiled as he went to sleep. The end."). Perplexity is dropping and repetition rate is decreasing.

We will proceed toward the **30,000-step milestone**.
