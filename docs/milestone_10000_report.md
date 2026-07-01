# Phase 1.4B GPU Training — Step 10,000 Milestone Report

This milestone report covers the evaluation of the 6.8M parameter Nexara model at step 10,000 of training, comparing metrics and generation quality against the step 5,000 baseline.

## 1. Quantitative Evaluation Summary

Training continues to show excellent progress. The validation loss and perplexity have decreased consistently without stagnation or instability.

| Metric | Step 5,000 (Baseline) | Step 10,000 (Milestone) | Trend / Decision |
| :--- | :---: | :---: | :--- |
| **Training Loss** | 2.23 | 1.94 | **Decreasing** (Good) |
| **Validation Loss** | 2.13 | 1.90 | **Decreasing** (Good) |
| **Validation Perplexity** | 8.40 | 6.71 | **Decreasing** (Good) |
| **Heuristic Coherence Score** | 63.64% | 74.74% | **Improving** (Good) |
| **Bigram Repetition Rate** | 7.65% | 7.98% | **Stable** (Good) |
| **Trigram Repetition Rate** | 1.61% | 1.87% | **Stable** (Good) |
| **Token Entropy** | 7.2787 | 7.1681 | **Stable** (Good) |
| **Average Sentence Length** | 7.62 words | 7.73 words | **Stable** (Good) |

> [!NOTE]
> The **Heuristic Coherence Score** increased by over **11%**, reflecting much more consistent sentence capitalization, punctuation endings, and natural word frequencies.

---

## 2. Metrics & Loss Curves

### Loss Evolution
* Step 0: **9.04**
* Step 1,000: **2.99** (Val: 2.97)
* Step 2,500: **2.27** (Val: 2.29)
* Step 5,000: **2.23** (Val: 2.13)
* Step 7,500: **1.98** (Val: 1.98)
* Step 10,000: **1.94** (Val: 1.90)

### Perplexity Evolution
* Step 1,000: **19.50**
* Step 2,500: **9.84**
* Step 5,000: **8.40**
* Step 7,500: **7.25**
* Step 10,000: **6.71**

---

## 3. Qualitative Generation Comparison

Below are comparisons of the text generated from standard prompts between the 5k baseline and 10k milestone.

### PROMPT: "Once upon a time"
* **Step 5,000 Output**: *"Once upon a time, there was a little girl named Lily. She loved to play outside in the garden. One day, she saw a big, dark spider crawling on a leaf. Lily ran to the spider and said, \"Oh no! My hand is hurt!\" The spider looked at Lily and said, \"Don't worry, Lily. We can get you some honey.\" Lily held on tight and started to walk slowly. As they walked, Lily realized that the spider was not afraid of a little bug. She asked her mom, \"What's wrong?\" Her mom told her that the spider might bite us and they can help it grow up again.\""*
* **Step 10,000 Output**: *"Once upon a time, there was a little girl named Lily. She loved to play with her toys and run around in the park. One day, she saw a big billboard on the ground. It had pictures of animals like lions, tigers, and elephants. Lily wanted to get closer to the billboard, so she ran up to it and touched it with her hand. Suddenly, something caught her eye! It was a small mouse! Lily thought for a moment and decided to examine it anyway. She picked up the mouse and showed it to her mom. Her mom was proud of her for being responsible and listening to the billboard. From then on, Lily always..."*

### PROMPT: "The little dog was"
* **Step 5,000 Output**: *"The little dog was so excited to see what he had to do. He ran over to the man and asked for a tool. \"Can I use this tool?\" The man smiled and said, \"Yes, of course!\" But then he saw an old man sitting at him. The man told him that the axe belonged to his friend. The old man was very surprised and asked the boy why he wanted to put it in his pocket. The old man said no because he had lost it. So the old man tried to repair the axe, but it wouldn't budge. He felt sad and started to cry. Just then, he heard a voice in front"*
* **Step 10,000 Output**: *"The little dog was so excited to have his ball. He ran outside and grabbed it with him. He kicked the ball back and forth, but he couldn't catch it. He decided to take a break and sit down in the grass. Then, he looked for it, but he couldn't find it anywhere. He was worried. He tried again, but this time, he had an idea. He thought of a plan to get the ball back. He asked his dad if he could borrow it. His dad said yes and lent him the ball. He was very happy and took it. When he returned home, he gave the ball a big hug."*

### PROMPT: "Lily wanted to"
* **Step 5,000 Output**: *"Lily wanted to go on a holiday with her mom. She saw many people at the beach, like lions, monkeys, and elephants. She also liked to see the animals and the animals. She put on her shoes and said hello. She was very excited. Lily wanted to watch the animals. She jumped in front of them and waved to them. \"Hello, animals,\" she said to her mom. She held her hand and smiled. She ran to the shore. She walked into the boat. It was red and green and had a long legs. She reached out her hand to touch it. She felt a tickle in her nose. She looked around"*
* **Step 10,000 Output**: *"Lily wanted to play with her computer. She asked her mom if she could help. Her mom said yes and gave Lily a small desk in the living room. Lily was happy to see her computer. She played with it all day long and had so much fun. After a while, Lily's mom called out for lunch. Lily looked at her computer and saw that it was broken. She knew her mom was upset because it was dirty. She went to the bathroom and washed her hands. She told her mom what happened and she told her about the computer. Her mom cleaned her hands and hugged her. Then they went to find their friends. They were"*

---

## 4. Decision: CONTINUE TRAINING
The run is **healthy and improving**. Validation loss and perplexity continue their downward trend. The model has begun introducing more complex, modern nouns (like "billboard", "computer", "desk", "station", "motorcycle", "sirens") in a grammatically consistent manner. Repetition is stable and token entropy remains high.

We will proceed toward the **15,000-step milestone**.
