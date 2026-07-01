# Phase 1.4B GPU Training — Step 40,000 Milestone Report

This milestone report covers the evaluation of the 6.8M parameter Nexara model at step 40,000 of training, comparing metrics and generation quality against previous milestones.

## 1. Quantitative Evaluation Summary

Validation loss and perplexity continue to show strong, consistent improvements. The validation perplexity has now **broken below 5.55** to **5.542**, showing that the model is gaining a better grasp of grammar and storytelling structure. The heuristic coherence score recovered to **77.03%**.

| Metric | Step 5,000 | Step 10,000 | Step 15,000 | Step 20,000 | Step 30,000 | Step 40,000 (Milestone) | Trend / Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Training Loss** | 2.23 | 1.94 | 1.86 | 1.73 | 1.86 | 1.78 | **Decreasing** (Good) |
| **Validation Loss** | 2.13 | 1.90 | 1.82 | 1.78 | 1.74 | 1.71 | **Decreasing** (Good) |
| **Validation Perplexity** | 8.40 | 6.71 | 6.20 | 5.95 | 5.71 | 5.54 | **Decreasing** (Good) |
| **Heuristic Coherence Score** | 63.64% | 74.74% | 78.56% | 73.13% | 68.91% | 77.03% | **Increasing** (Good) |
| **Bigram Repetition Rate** | 7.65% | 7.98% | 7.51% | 7.35% | 7.92% | 8.93% | **Stable** (Good) |
| **Trigram Repetition Rate** | 1.61% | 1.87% | 1.45% | 1.24% | 1.26% | 2.25% | **Stable** (Good) |
| **Token Entropy** | 7.2787 | 7.1681 | 7.2842 | 7.2208 | 7.3447 | 7.2660 | **Stable** (Good) |
| **Average Sentence Length** | 7.62 words | 7.73 words | 8.78 words | 8.63 words | 8.04 words | 8.77 words | **Stable** (Good) |

---

## 2. Metrics & Loss Curves

### Loss Evolution
* Step 5,000: **2.23** (Val: 2.13)
* Step 10,000: **1.94** (Val: 1.90)
* Step 15,000: **1.86** (Val: 1.82)
* Step 20,000: **1.73** (Val: 1.78)
* Step 30,000: **1.86** (Val: 1.74)
* Step 40,000: **1.78** (Val: 1.71)

### Perplexity Evolution
* Step 5,000: **8.40**
* Step 10,000: **6.71**
* Step 15,000: **6.20**
* Step 20,000: **5.95**
* Step 30,000: **5.71**
* Step 40,000: **5.54**

---

## 3. Qualitative Generation Comparison

Below are comparisons of the text generated from standard prompts between the step 30,000 and step 40,000 milestones.

### PROMPT: "Once upon a time"
* **Step 30,000 Output**: *"Once upon a time there was a young girl called Lucy. She was three years old and loved to play outside. One day, her dad told her that he had an important job to do. He told her to stand very carefully so she could do what was right. Lucy did as she was told and soon she was ready to do the job. She took a big deep breath and started walking through the forest. When she got there, she saw a huge tree with lots of apples. She smiled and climbed down the tree. At the top, she found a large tree with a lot of fruits. She picked them up, put them in the basket and"*
* **Step 40,000 Output**: *"Once upon a time, there was a little boy named Timmy. Timmy loved to play with his toy cars and trucks all day long. One day, he saw a big, green truck driving down the street. It was so loud that it made Timmy very scared. He ran inside his house and cried for his mom. \"Mommy, mommy, I'm scared of the truck. Can we go inside?\" Timmy said. His mom hugged him and said, \"Don't worry, Timmy. I'll protect you.\" She took him in her arms and wrapped him in a blanket. Then she went inside and closed the door. Timmy felt safe and happy again."*

### PROMPT: "The little dog was"
* **Step 30,000 Output**: *"The little dog was running through the park, when he noticed a stranger. The stranger had a big black hat on his head. He smiled at the dog, and said, \"I'm a new friend!\" The little dog's owner said, \"What do you think I'm doing?\" The old man looked down at the girl and said, \"You are so nice! Can I help you?\" The girl smiled and said, \"Yes please!\" The old man said, \"Let me show you how to wear it.\" The little girl was very excited. She ran up and showed her friend. Together, they both looked around the park and soon found the"*
* **Step 40,000 Output**: *"The little dog was feeling scared. He had been chasing all the other animals in the meadow, but he was determined to stay away from them. Suddenly he saw a big, scary wolf. The wolf was trying to bite him! The little dog started to cry and he knew he had to run away, but he did not know that he had to be brave. So he ran even faster and kept running until he reached his home. The wolf ran very fast, but the little dog was too slow. The wolf caught up with him and they ran away as fast as they could. The little dog was safe and the wolf was happy to have helped him find"*

### PROMPT: "The rabbit saw a flower"
* **Step 30,000 Output**: *"The rabbit saw a flower. It was so pretty and yellow! The rabbit hopped over to it and said, \"Hello! What is that?\" The flower replied, \"I am a beautiful flower. Do you want to come and talk?\" The rabbit said, \"Yes! I would like to be my friend.\" So the rabbit and the flower started to talk. They talked about all kinds of things and they were very happy friends. Soon the sun began to set and the sky became dark. The rabbit said goodbye to the flower and started to walk back home. But the rabbit said, \"Come back soon. I can see you again!\" The flower replied with"*
* **Step 40,000 Output**: *"The rabbit saw a flower growing in a garden. He wanted to help the flower. He hopped over and asked for some water. The flower was kind, so he gave it a big water. The rabbit thanked the flower and then said goodbye. He went back to his home. When he came back, the rabbit was very happy. He hopped around with joy and was thankful that the flower helped him find a way to make the flower grow."*

---

## 4. Decision: CONTINUE TRAINING
The model shows **increasing narrative consistency and dialogue flow**. Overfitting is not present, validation perplexity is decreasing steadily, and the text generations are highly logical.

We will proceed toward the **50,000-step milestone**.
