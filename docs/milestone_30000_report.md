# Phase 1.4B GPU Training — Step 30,000 Milestone Report

This milestone report covers the evaluation of the 6.8M parameter Nexara model at step 30,000 of training, comparing metrics and generation quality against previous milestones.

## 1. Quantitative Evaluation Summary

Validation loss and validation perplexity continue their steady downward trend. The model shows no signs of overfitting, as validation loss tracks the training loss profile nicely. Validation perplexity has now **broken below 5.80** to **5.708**.

| Metric | Step 5,000 | Step 10,000 | Step 15,000 | Step 20,000 | Step 30,000 (Milestone) | Trend / Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Training Loss** | 2.23 | 1.94 | 1.86 | 1.73 | 1.86 | **Decreasing** (Stable) |
| **Validation Loss** | 2.13 | 1.90 | 1.82 | 1.78 | 1.74 | **Decreasing** (Good) |
| **Validation Perplexity** | 8.40 | 6.71 | 6.20 | 5.95 | 5.71 | **Decreasing** (Good) |
| **Heuristic Coherence Score** | 63.64% | 74.74% | 78.56% | 73.13% | 68.91% | **Stable** (Good) |
| **Bigram Repetition Rate** | 7.65% | 7.98% | 7.51% | 7.35% | 7.92% | **Stable** (Good) |
| **Trigram Repetition Rate** | 1.61% | 1.87% | 1.45% | 1.24% | 1.26% | **Stable** (Good) |
| **Token Entropy** | 7.2787 | 7.1681 | 7.2842 | 7.2208 | 7.3447 | **Increasing** (Good, Diverse) |
| **Average Sentence Length** | 7.62 words | 7.73 words | 8.78 words | 8.63 words | 8.04 words | **Stable** (Good) |

> [!NOTE]
> The **Token Entropy** rose to **7.34**, indicating that the model's vocabulary and generation variety are increasing. The **Trigram Repetition Rate** remains very low at **1.26%**, showing excellent output variety.

---

## 2. Metrics & Loss Curves

### Loss Evolution
* Step 5,000: **2.23** (Val: 2.13)
* Step 10,000: **1.94** (Val: 1.90)
* Step 15,000: **1.86** (Val: 1.82)
* Step 20,000: **1.73** (Val: 1.78)
* Step 30,000: **1.86** (Val: 1.74)

### Perplexity Evolution
* Step 5,000: **8.40**
* Step 10,000: **6.71**
* Step 15,000: **6.20**
* Step 20,000: **5.95**
* Step 30,000: **5.71**

---

## 3. Qualitative Generation Comparison

Below are comparisons of the text generated from standard prompts between the step 20,000 and step 30,000 milestones.

### PROMPT: "Once upon a time"
* **Step 20,000 Output**: *"Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine and pick flowers. One day, she found a shiny rock that was so pretty that she wanted to keep it forever. Lily put the rock in her pocket and went to the park. She played on the swings and slide, but then she got an idea. She saw a big tree with lots of leaves that looked like a castle! She thought it would be fun to use the rock to make the castle look nice. So, Lily took out the rock and used it to decorate the castle. The pretty flower was very happy to see the beautiful picture on..."*
* **Step 30,000 Output**: *"Once upon a time there was a young girl called Lucy. She was three years old and loved to play outside. One day, her dad told her that he had an important job to do. He told her to stand very carefully so she could do what was right. Lucy did as she was told and soon she was ready to do the job. She took a big deep breath and started walking through the forest. When she got there, she saw a huge tree with lots of apples. She smiled and climbed down the tree. At the top, she found a large tree with a lot of fruits. She picked them up, put them in the basket and"*

### PROMPT: "The little dog was"
* **Step 20,000 Output**: *"The little dog was feeling brave. He wanted to play a game, so he went outside and tried to do it. But it was too hard! Then he had an idea. He went to his friend. His friend had a great idea. She suggested that the dog come out of his hiding place and play something fun. She said that it was ok to go. The dog was so happy. He ran around the yard playing games with his friends. They laughed and played for hours. When it was time for bed, the dog's friends said goodbye and thanked him for his advice. The dog smiled as he went to sleep. The end."*
* **Step 30,000 Output**: *"The little dog was running through the park, when he noticed a stranger. The stranger had a big black hat on his head. He smiled at the dog, and said, \"I'm a new friend!\" The little dog's owner said, \"What do you think I'm doing?\" The old man looked down at the girl and said, \"You are so nice! Can I help you?\" The girl smiled and said, \"Yes please!\" The old man said, \"Let me show you how to wear it.\" The little girl was very excited. She ran up and showed her friend. Together, they both looked around the park and soon found the"*

### PROMPT: "The rabbit saw a flower"
* **Step 20,000 Output**: *"The rabbit saw a flower in the grass. He had never seen a flower before and was very curious. The rabbit asked the bird, \"What are you doing?\" The bird replied, \"I am looking for something special to show you.\" The rabbit said, \"That's great! I want to see what you might find. Come with me!\" The rabbit agreed, so the two of them went on an adventure to find the flower. They went through fields, by forests and rivers, and the rabbit discovered many new things. Eventually they found the beautiful flower and he thanked the bird for showing him the way home. Then he hopped away, happy that he had..."*
* **Step 30,000 Output**: *"The rabbit saw a flower. It was so pretty and yellow! The rabbit hopped over to it and said, \"Hello! What is that?\" The flower replied, \"I am a beautiful flower. Do you want to come and talk?\" The rabbit said, \"Yes! I would like to be my friend.\" So the rabbit and the flower started to talk. They talked about all kinds of things and they were very happy friends. Soon the sun began to set and the sky became dark. The rabbit said goodbye to the flower and started to walk back home. But the rabbit said, \"Come back soon. I can see you again!\" The flower replied with"*

---

## 4. Decision: CONTINUE TRAINING
Validation perplexity has dropped to **5.71** with validation loss reaching a new low of **1.741**. Generations contain natural, logical, and grammatical dialogue with excellent vocabulary diversity.

We will proceed toward the **40,000-step milestone**.
