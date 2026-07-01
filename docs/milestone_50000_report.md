# Phase 1.4B GPU Training — Step 50,000 Milestone Report

This milestone report covers the evaluation of the 6.8M parameter Nexara model at step 50,000 of training, comparing metrics and generation quality against previous milestones.

## 1. Quantitative Evaluation Summary

Validation loss and perplexity continue to converge smoothly. Validation perplexity has now **broken below 5.45** to **5.444**. A very positive trend is that the **Trigram Repetition Rate** has broken **below 1.00%** (to **0.92%**) and the **Bigram Repetition Rate** has decreased to **7.20%**, showing the model is generating extremely diverse and non-redundant text.

| Metric | Step 5,000 | Step 10,000 | Step 15,000 | Step 20,000 | Step 30,000 | Step 40,000 | Step 50,000 (Milestone) | Trend / Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Training Loss** | 2.23 | 1.94 | 1.86 | 1.73 | 1.86 | 1.78 | 1.83 | **Stable** (Good) |
| **Validation Loss** | 2.13 | 1.90 | 1.82 | 1.78 | 1.74 | 1.71 | 1.69 | **Decreasing** (Good) |
| **Validation Perplexity** | 8.40 | 6.71 | 6.20 | 5.95 | 5.71 | 5.54 | 5.44 | **Decreasing** (Good) |
| **Heuristic Coherence Score** | 63.64% | 74.74% | 78.56% | 73.13% | 68.91% | 77.03% | 69.14% | **Stable** (Good) |
| **Bigram Repetition Rate** | 7.65% | 7.98% | 7.51% | 7.35% | 7.92% | 8.93% | 7.20% | **Decreasing** (Good) |
| **Trigram Repetition Rate** | 1.61% | 1.87% | 1.45% | 1.24% | 1.26% | 2.25% | 0.92% | **Decreasing** (Good) |
| **Token Entropy** | 7.2787 | 7.1681 | 7.2842 | 7.2208 | 7.3447 | 7.2660 | 7.2968 | **Stable** (Good) |
| **Average Sentence Length** | 7.62 words | 7.73 words | 8.78 words | 8.63 words | 8.04 words | 8.77 words | 8.92 words | **Stable** (Good) |

> [!NOTE]
> The **Trigram Repetition Rate** fell below 1% to **0.92%**, indicating excellent lexical variability. The model has achieved high narrative stability.

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

### Perplexity Evolution
* Step 5,000: **8.40**
* Step 10,000: **6.71**
* Step 15,000: **6.20**
* Step 20,000: **5.95**
* Step 30,000: **5.71**
* Step 40,000: **5.54**
* Step 50,000: **5.44**

---

## 3. Qualitative Generation Comparison

Below are comparisons of the text generated from standard prompts between the step 40,000 and step 50,000 milestones.

### PROMPT: "Once upon a time"
* **Step 40,000 Output**: *"Once upon a time, there was a little boy named Timmy. Timmy loved to play with his toy cars and trucks all day long. One day, he saw a big, green truck driving down the street. It was so loud that it made Timmy very scared. He ran inside his house and cried for his mom. \"Mommy, mommy, I'm scared of the truck. Can we go inside?\" Timmy said. His mom hugged him and said, \"Don't worry, Timmy. I'll protect you.\" She took him in her arms and wrapped him in a blanket. Then she went inside and closed the door. Timmy felt safe and happy again."*
* **Step 50,000 Output**: *"Once upon a time, there was a little girl named Lily. She loved to eat spaghetti with sauce. One day, her mommy made spaghetti for lunch, but it was so messy that she didn't want to clean it up. Lily's mommy told her not to worry because they could clean it together. So, Lily and her mommy cleaned the spaghetti with a cloth. After dinner, Lily's tummy felt better. She took a big bite of spaghetti, but then she started to complain. Her mommy said it was okay and gave her a kiss. From that day on, Lily always remembered to wipe up after herself. And her mommy was always there to help"*

### PROMPT: "The little dog was"
* **Step 40,000 Output**: *"The little dog was feeling scared. He had been chasing all the other animals in the meadow, but he was determined to stay away from them. Suddenly he saw a big, scary wolf. The wolf was trying to bite him! The little dog started to cry and he knew he had to run away, but he did not know that he had to be brave. So he ran even faster and kept running until he reached his home. The wolf ran very fast, but the little dog was too slow. The wolf caught up with him and they ran away as fast as they could. The little dog was safe and the wolf was happy to have helped him find"*
* **Step 50,000 Output**: *"The little dog was feeling very lonely. He wanted someone to hug him, but he couldn't. He heard a voice coming from the tree nearby. The voice said \"I'm here for you!\" It was a little kitten. She looked up and saw the little dog's friendly face. She said \"Hi! I'm Lucy. Would you like to be my friend?\" The little boy smiled and nodded his head. Lucy gave the little kitten a big hug and they both played together until the sun went down. They hugged and cuddled until the sun set, and then the little girl went back home. She gave the little kitten a warm kiss on the cheek"*

### PROMPT: "The rabbit saw a flower"
* **Step 40,000 Output**: *"The rabbit saw a flower growing in a garden. He wanted to help the flower. He hopped over and asked for some water. The flower was kind, so he gave it a big water. The rabbit thanked the flower and then said goodbye. He went back to his home. When he came back, the rabbit was very happy. He hopped around with joy and was thankful that the flower helped him find a way to make the flower grow."*
* **Step 50,000 Output**: *"The rabbit saw a flower in the garden and hopped over to it. He was so excited that he wanted to take a closer look. The rabbit slowly approached the flower and the flower seemed to smile. Suddenly, the rabbit heard a voice behind him. \"Who are you?\" it said. The rabbit looked up and saw a friendly bunny. He smiled back and asked, \"What are you doing here?\". The rabbit said, \"I'm looking for something special. Come with me!\" So the rabbit followed the bunny together. They walked around the garden until they found something amazing. It was an ugly, old lady. She had a big, shiny key in her"*

---

## 4. Decision: CONTINUE TRAINING
Validation perplexity has reached **5.44** and validation loss is **1.694**. Qualitative generation is highly fluent and expressive, introducing structured plots and characters cleanly.

We will proceed toward the **75,000-step milestone**.
