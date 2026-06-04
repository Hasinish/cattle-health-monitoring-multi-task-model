# Pre-Thesis II Presentation Study Guide
## Key Technical Concepts (Simplified)

---

### Part 1: Multi-Task Learning (MTL)
*(Solving different problems using only one AI model)*

#### The Problem:
Normally, to monitor cows, you would need to run **four separate convolutional neural networks** *(AI models designed to read images)*: one for Body Condition Score (BCS), one for Behavior, one for Lameness, and one for ID.
* This takes too much memory and processing power.
* It has high **inference latency** *(takes too long for the AI to process an image and make a guess)*.
* It cannot run on low-power **edge devices** *(small, cheap computers placed directly on the farm)*.

#### The Solution: Hard Parameter Sharing
*(Forcing different tasks to share the exact same base layers)*

Instead of four separate models, the project uses **one shared backbone** *(the base part of the AI that extracts general shapes and patterns from the image)*.

* The backbone reads the cow image and outputs a single **feature vector** *(a list of numbers that represents the visual details of the image)*.
* We then attach **four separate prediction heads** *(small mini-networks attached to the end of the backbone)* to solve the specific tasks:
  * **BCS Head** outputs the body condition score.
  * **Behavior Head** outputs what the cow is doing.
  * **Lameness Head** outputs if the cow is limping.
  * **ID Head** outputs which specific cow it is.

#### How the model learns all four tasks at the same time:
To train the model, we use a **Multi-Task Loss Function** *(a mathematical formula that calculates the total mistakes of the AI across all tasks)*:
* **Total Loss = $w_1 \cdot \text{Loss}_{\text{BCS}} + w_2 \cdot \text{Loss}_{\text{Behavior}} + w_3 \cdot \text{Loss}_{\text{Lameness}} + w_4 \cdot \text{Loss}_{\text{ID}}$**
* The variables $w_1, w_2, w_3, w_4$ are **loss weights** *(numbers that control how much the AI prioritizes each task during training)*. We tune these weights so that one task doesn't dominate the training and make the AI forget the other tasks.

#### The Backbone Selection Process:
We trained and evaluated single-task baseline models for five different backbones:
* **ResNet-18** (assigned to Hasin)
* **MobileNetV3-Small** (assigned to Namira)
* **ResNet-50** (assigned to Bithi)
* **DenseNet121** (assigned to Shouvik)
* **EfficientNetB0** (assigned to Nusrat)
* **Objective:** Find the best backbone that has the lowest error rates but uses the fewest parameters, making it the most efficient "shared brain" for our multi-task model.

#### Why this is important for the thesis:
* **Lightweight:** It keeps the model under **10 million parameters** *(the variables the AI uses to learn)*, which easily fits on farm cameras.
* **Regularization effect:** *(Forcing the AI to learn general features rather than memorizing details)*. Since the backbone has to satisfy four tasks at once, it cannot overfit on any single task. This makes the model much better at understanding new, unseen cows.

---

### Part 2: Ordinal Regression (CORAL Framework)
*(Predicting categories that have a natural, ordered scale)*

#### The Problem:
Cattle Body Condition Scoring uses ordered numbers (like `3.25, 3.5, 3.75, 4.0, 4.25`).
* If you use standard classification, the AI treats all wrong guesses the same. If the real score is `3.5`, standard classification penalizes a guess of `3.75` exactly the same as a guess of `4.25`. This is bad because `3.75` is a much closer and better guess.
* If you use standard regression, the AI outputs a continuous decimal number (like `3.64`). You then have to use arbitrary rounding rules to get a class, which is unstable and lacks probabilistic meaning.

#### The Solution: The CORAL Framework
*(Consistent Rank Logits - a method that splits rank prediction into separate Yes/No questions)*

Instead of choosing one class out of five, the model's output layer has **4 nodes** *(connection points)* that answer **binary classification** *(deciding between two choices, like Yes or No)* questions:
* **Node 1:** Is the score higher than 0? (Yes/No)
* **Node 2:** Is the score higher than 1? (Yes/No)
* **Node 3:** Is the score higher than 2? (Yes/No)
* **Node 4:** Is the score higher than 3? (Yes/No)

To get the final score, we check the probability of each node. If it passes a **threshold** *(a cutoff value, usually 0.5)*, we count it as a `1` (Yes) and sum them up:
* If the outputs are: `[1, 1, 0, 0]`, the sum is `2`. The prediction is **Class 2** (representing BCS `3.75`).
* If the outputs are: `[1, 1, 1, 0]`, the sum is `3`. The prediction is **Class 3** (representing BCS `4.0`).

#### Why this is important for the thesis:
It teaches the AI the natural order of the numbers. It drastically reduces the **MAE (Mean Absolute Error)** *(the average distance between the model's guesses and the real values)* because it penalizes far-away guesses much more than close guesses.

---

### Part 3: Focal Loss (Addressing Class Imbalance)
*(Forcing the AI to focus on hard-to-learn, rare images rather than easy, common ones)*

#### The Problem: Class Imbalance
In our behavior dataset (MmCows), the cow activities are extremely imbalanced. For example, cows spend a huge amount of time *Lying* or *Standing* (thousands of images), but very little time *Drinking* or *Licking* (very few images).
* If we use standard **Cross-Entropy Loss** *(the normal formula used to calculate classification mistakes)*, the AI will realize it can get a 95% score just by predicting "Lying" or "Standing" for almost everything.
* The model gets lazy, ignores the rare categories, and fails to learn the difference between drinking, feeding, and licking.

#### The Solution: Focal Loss
Focal Loss modifies the standard loss function by adding a **modulating factor** $(1 - p_t)^\gamma$ to dynamically scale the loss based on prediction confidence. 

In simple terms, it tells the model: *"If you are already very confident and correct about an image, ignore it. If you are struggling and incorrect, multiply the penalty so you pay more attention to it."*

#### Key Parameters Explained:
1. **Gamma ($\gamma = 2$) - The Focusing Parameter**:
   * Controls how heavily we discount "easy" examples.
   * If the AI is 90% sure about a common image (e.g., a cow lying down), a $\gamma=2$ reduces the loss for that image to almost zero.
   * This forces the gradient updates *(the adjustments made to the AI's internal settings during training)* to be driven entirely by the hard, rare images.
2. **Alpha ($\alpha = 0.25$) - The Balancing Parameter**:
   * Weighs the overall importance of classes. It helps balance the frequency difference between positive and negative classes.

#### Why this is important for the thesis:
Without Focal Loss, the model's **Macro F1-Score** *(a metric that averages performance across all classes equally, regardless of how many images they have)* would be extremely low because rare classes (like drinking) would have near-zero accuracy. Focal Loss ensures the AI learns all 7 behaviors effectively.
