# Phase 1.4B GPU Training — Step 75,000 Milestone Report

This milestone report covers the evaluation of the 6.8M parameter Nexara model at step 75,000 of training, comparing metrics and generation quality against previous milestones.

## 1. Quantitative Evaluation Summary

Validation loss and perplexity continue to decrease nicely. Validation perplexity has now **broken below 5.26** to **5.259**. The heuristic coherence score has reached a new peak of **80.58%**, demonstrating the model's growing capability to produce structurally sound and grammatically correct TinyStories.

| Metric | Step 5,000 | Step 10,000 | Step 15,000 | Step 20,000 | Step 30,000 | Step 40,000 | Step 50,000 | Step 75,000 (Milestone) | Trend / Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Training Loss** | 2.23 | 1.94 | 1.86 | 1.73 | 1.86 | 1.78 | 1.83 | 1.77 | **Decreasing** (Good) |
| **Validation Loss** | 2.13 | 1.90 | 1.82 | 1.78 | 1.74 | 1.71 | 1.69 | 1.66 | **Decreasing** (Good) |
| **Validation Perplexity** | 8.40 | 6.71 | 6.20 | 5.95 | 5.71 | 5.54 | 5.44 | 5.26 | **Decreasing** (Good) |
| **Heuristic Coherence Score** | 63.64% | 74.74% | 78.56% | 73.13% | 68.91% | 77.03% | 69.14% | 80.58% | **Increasing** (Good) |
| **Bigram Repetition Rate** | 7.65% | 7.98% | 7.51% | 7.35% | 7.92% | 8.93% | 7.20% | 7.26% | **Stable** (Good) |
| **Trigram Repetition Rate** | 1.61% | 1.87% | 1.45% | 1.24% | 1.26% | 2.25% | 0.92% | 1.30% | **Stable** (Good) |
| **Token Entropy** | 7.2787 | 7.1681 | 7.2842 | 7.2208 | 7.3447 | 7.2660 | 7.2968 | 7.2310 | **Stable** (Good) |
| **Average Sentence Length** | 7.62 words | 7.73 words | 8.78 words | 8.63 words | 8.04 words | 8.77 words | 8.92 words | 7.91 words | **Stable** (Good) |

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

### Perplexity Evolution
* Step 5,000: **8.40**
* Step 10,000: **6.71**
* Step 15,000: **6.20**
* Step 20,000: **5.95**
* Step 30,000: **5.71**
* Step 40,000: **5.54**
* Step 50,000: **5.44**
* Step 75,000: **5.26**

---

## 3. Qualitative Generation Comparison

Below are comparisons of the text generated from standard prompts between the step 50,000 and step 75,000 milestones.

### PROMPT: "Once upon a time"
* **Step 50,000 Output**: *"Once upon a time, there was a little girl named Lily. She loved to eat spaghetti with sauce. One day, her mommy made spaghetti for lunch, but it was so messy that she didn't want to clean it up. Lily's mommy told her not to worry because they could clean it together. So, Lily and her mommy cleaned the spaghetti with a cloth. After dinner, Lily's tummy felt better. She took a big bite of spaghetti, but then she started to complain. Her mommy said it was okay and gave her a kiss. From that day on, Lily always remembered to wipe up after herself. And her mommy was always there to help"*
* **Step 75,000 Output**: *"Once upon a time, there was a little girl named Lily. She loved to play outside in the sun and pick flowers. One day, she saw a tiny insect crawling on the ground. \"Hello there!\" said Lily. \"What are you doing?\" \"I am crawling,\" said the insect. \"I live in a big tree.\" \"It's easy,\" said Lily. \"I just want to have fun.\" The insect thought for a moment and then said, \"Okay, I will join you for some fun!\" Together, they crawled around the tree, chasing butterflies and playing in the sunshine. When it was time to go home, the insect waved"*

### PROMPT: "The little dog was"
* **Step 50,000 Output**: *"The little dog was feeling very lonely. He wanted someone to hug him, but he couldn't. He heard a voice coming from the tree nearby. The voice said \"I'm here for you!\" It was a little kitten. She looked up and saw the little dog's friendly face. She said \"Hi! I'm Lucy. Would you like to be my friend?\" The little boy smiled and nodded his head. Lucy gave the little kitten a big hug and they both played together until the sun went down. They hugged and cuddled until the sun set, and then the little girl went back home. She gave the little kitten a warm kiss on the cheek"*
* **Step 75,000 Output**: *"The little dog was so happy to be on his way home. He had found his new owner, a little girl named Lily. They played together every day and had lots of fun. One day, while playing in the park, Lily started to feel tired. She went to her mom and said, \"Mommy, I want to take a nap. Can we go home now?\" Her mom smiled and said, \"Of course, sweetie. Let's go home.\" When they got home, Lily snuggled up in her cozy bed. She felt warm and comfortable. The next morning, Lily woke up early and ran outside to play. She was tired but happy."*

### PROMPT: "The rabbit saw a flower"
* **Step 50,000 Output**: *"The rabbit saw a flower in the garden and hopped over to it. He was so excited that he wanted to take a closer look. The rabbit slowly approached the flower and the flower seemed to smile. Suddenly, the rabbit heard a voice behind him. \"Who are you?\" it said. The rabbit looked up and saw a friendly bunny. He smiled back and asked, \"What are you doing here?\". The rabbit said, \"I'm looking for something special. Come with me!\" So the rabbit followed the bunny together. They walked around the garden until they found something amazing. It was an ugly, old lady. She had a big, shiny key in her"*
* **Step 75,000 Output**: *"The rabbit saw a flower. He was very excited and hopped over to it. The flower was beautiful. It was blooming in the sunshine. Suddenly, it started to rain. The rabbit was wet. He tried to cover himself with a big leaf to keep him dry. But it didn't work! The rain kept falling and the garden was flooded. The rabbit was sad. He had to go inside and get dry. But he was happy because he had his special flower to keep him dry."*

---

## 4. Decision: CONTINUE TRAINING
The model shows excellent grammatical structures, coherent dialogue, and strong contextual progression. Both validation loss (**1.660**) and validation perplexity (**5.26**) have set new records.

We will proceed toward the **90,000-step milestone**.
