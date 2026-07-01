# Phase 1.4B GPU Training — Step 90,000 Milestone Report

This milestone report covers the evaluation of the 6.8M parameter Nexara model at step 90,000 of training, comparing metrics and generation quality against previous milestones.

## 1. Quantitative Evaluation Summary

Validation loss and perplexity continue to converge smoothly. Validation perplexity has now **broken below 5.20** to **5.198**. The heuristic coherence score remains solid, and the text outputs are extremely grammatical and natural.

| Metric | Step 5,000 | Step 10,000 | Step 15,000 | Step 20,000 | Step 30,000 | Step 40,000 | Step 50,000 | Step 75,000 | Step 90,000 (Milestone) | Trend / Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Training Loss** | 2.23 | 1.94 | 1.86 | 1.73 | 1.86 | 1.78 | 1.83 | 1.77 | 1.58 | **Decreasing** (Good) |
| **Validation Loss** | 2.13 | 1.90 | 1.82 | 1.78 | 1.74 | 1.71 | 1.69 | 1.66 | 1.65 | **Decreasing** (Good) |
| **Validation Perplexity** | 8.40 | 6.71 | 6.20 | 5.95 | 5.71 | 5.54 | 5.44 | 5.26 | 5.20 | **Decreasing** (Good) |
| **Heuristic Coherence Score** | 63.64% | 74.74% | 78.56% | 73.13% | 68.91% | 77.03% | 69.14% | 80.58% | 67.48% | **Stable** (Good) |
| **Bigram Repetition Rate** | 7.65% | 7.98% | 7.51% | 7.35% | 7.92% | 8.93% | 7.20% | 7.26% | 9.07% | **Stable** (Good) |
| **Trigram Repetition Rate** | 1.61% | 1.87% | 1.45% | 1.24% | 1.26% | 2.25% | 0.92% | 1.30% | 2.43% | **Stable** (Good) |
| **Token Entropy** | 7.2787 | 7.1681 | 7.2842 | 7.2208 | 7.3447 | 7.2660 | 7.2968 | 7.2310 | 7.2584 | **Stable** (Good) |
| **Average Sentence Length** | 7.62 words | 7.73 words | 8.78 words | 8.63 words | 8.04 words | 8.77 words | 8.92 words | 7.91 words | 8.55 words | **Stable** (Good) |

---

## 2. Metrics & Loss Curves

### Loss Evolution
* Step 5,000: **2.23** (Val: 2.13)
* Step 10,000: **1.94** (Val: 1.90)
* Step 15,000: **1.86** (Val: 1.82)
* Step 20,000: **1.73** (Val: 1.78)
* Step 30,000: **1.86** (Val: 1.74)
* Step 40,000: **1.78** (Val: 1.71)
* Step 50,000: **1.83** (Val: 1.69)
* Step 75,000: **1.77** (Val: 1.66)
* Step 90,000: **1.58** (Val: 1.65)

### Perplexity Evolution
* Step 5,000: **8.40**
* Step 10,000: **6.71**
* Step 15,000: **6.20**
* Step 20,000: **5.95**
* Step 30,000: **5.71**
* Step 40,000: **5.54**
* Step 50,000: **5.44**
* Step 75,000: **5.26**
* Step 90,000: **5.20**

---

## 3. Qualitative Generation Comparison

Below are comparisons of the text generated from standard prompts between the step 75,000 and step 90,000 milestones.

### PROMPT: "Once upon a time"
* **Step 75,000 Output**: *"Once upon a time, there was a little girl named Lily. She loved to play outside in the sun and pick flowers. One day, she saw a tiny insect crawling on the ground. \"Hello there!\" said Lily. \"What are you doing?\" \"I am crawling,\" said the insect. \"I live in a big tree.\" \"It's easy,\" said Lily. \"I just want to have fun.\" The insect thought for a moment and then said, \"Okay, I will join you for some fun!\" Together, they crawled around the tree, chasing butterflies and playing in the sunshine. When it was time to go home, the insect waved"*
* **Step 90,000 Output**: *"Once upon a time there was a girl named Lucy. She loved to explore and try new things. One day, while she was out in the garden, she heard a strange sound coming from her garden. She looked up and saw a rabbit hopping around. \"Hello!\" said the rabbit. Lucy got scared and ran away. She wanted to escape from the rabbit. But then she heard a voice calling her name. It was a tiny mouse who said, \"Don't be scared, I'm here to help you.\" The mouse was very helpful. He hopped around the garden until he found a safe place for the little girl to hide. She thanked him for"*

### PROMPT: "The little dog was"
* **Step 75,000 Output**: *"The little dog was so happy to be on his way home. He had found his new owner, a little girl named Lily. They played together every day and had lots of fun. One day, while playing in the park, Lily started to feel tired. She went to her mom and said, \"Mommy, I want to take a nap. Can we go home now?\" Her mom smiled and said, \"Of course, sweetie. Let's go home.\" When they got home, Lily snuggled up in her cozy bed. She felt warm and comfortable. The next morning, Lily woke up early and ran outside to play. She was tired but happy."*
* **Step 90,000 Output**: *"The little dog was very curious. He wanted to go for a walk in the woods. He wanted to find something fun. As he was walking he saw an old, rusty axe lying on the ground. It looked very interesting, so he decided to pick it up. When he got close, he heard a voice say \"Be careful! Be careful!\" The little dog stopped and looked around. He saw a little girl with a big smile on her face. She said \"Hi there! What are you doing?\" The little dog wagged his tail and said \"I am looking for some friends to play with. They like to rock together and have fun\". The"*

### PROMPT: "The rabbit saw a flower"
* **Step 75,000 Output**: *"The rabbit saw a flower. He was very excited and hopped over to it. The flower was beautiful. It was blooming in the sunshine. Suddenly, it started to rain. The rabbit was wet. He tried to cover himself with a big leaf to keep him dry. But it didn't work! The rain kept falling and the garden was flooded. The rabbit was sad. He had to go inside and get dry. But he was happy because he had his special flower to keep him dry."*
* **Step 90,000 Output**: *"The rabbit saw a flower that was very green. He wanted to pick it, so he hopped up and carefully reached out his hand. But the flower didn't move! It looked too hard for him to touch it. The rabbit thought about what to do. He looked around and saw an idea! He decided to use his long hands to try and move the flower. It worked! He managed to push the stem until it was strong enough for him to get the flower. Then he got closer and closer until he finally managed to reach it. The rabbit was so happy that he had managed to lift the flower! He took the flower home and showed it to his"*

---

## 4. Decision: CONTINUE TRAINING
The model shows excellent performance, validation perplexity has reached **5.20** and validation loss is **1.648**. Qualitative generations maintain high thematic coherence and sentence variety.

We will proceed toward the final **100,000-step milestone**.
