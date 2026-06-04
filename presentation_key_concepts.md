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

---

### Part 4: Temporal Sampling (Dense vs. Sparse)
*(How we choose which frames to show the AI from a video clip)*

#### The Concept:
When a video is recorded at 30+ FPS (Frames Per Second), consecutive frames are nearly identical. 
* **Dense Sampling** *(feeding every single frame into the AI)*: If a video clip has 300 frames, we feed all 300 frames to the model.
* **Sparse Sampling** *(picking a small, fixed number of evenly spaced frames)*: We divide the video into 20 equal segments and extract exactly 1 frame from each segment (e.g., picking frames `[0, 15, 30, ... 299]`).

#### Why Sparse Sampling is Better:
1. **Reduces Redundancy**: Skipping almost-identical frames saves the AI from wasting memory and processing power on duplicate details.
2. **Avoids Overfitting** *(when the AI memorizes the training videos instead of learning the general pattern of how cows walk)*: Feeding too many similar frames lets the model memorize the background noise of specific clips, which ruins its ability to classify new cows.
3. **Sequence Standardization**: Videos have different lengths (some are 80 frames, some are 400+ frames). Sampling exactly **20 frames** from all of them ensures a consistent input size for the temporal layer, making batch processing stable and safe.
4. **Hardware Friendly**: Processing 20 frames is 10x–20x faster and prevents **Out-of-Memory (OOM) crashes** on the GPU.

---

### Part 5: Long Short-Term Memory (LSTM)
*(An AI memory unit that tracks motion and gait changes over time)*

#### The Problem: Amnesia in 2D Models
Standard image models (like CNNs) have **amnesia**. They look at one photo, make a prediction, and immediately forget it when looking at the next photo. Since lameness (limping) is defined by motion over time, you cannot reliably diagnose it from a single static photo.

#### The Solution: The LSTM Module
An LSTM is a recurrent layer with a **memory cell** *(an internal notebook where the AI writes down and remembers key details as it reads a sequence of frames)*. 

#### How it works (The Three Gates):
1. **Forget Gate**: Decides what useless information to erase from its memory (e.g., a bird flying in the background).
2. **Input Gate**: Decides what new details are important to write down (e.g., the angle of the cow's knee joint).
3. **Output Gate**: Decides the final prediction (Lame vs. Normal) based on the accumulated memory of the walking stride.

---

### Part 6: Deployed Video Inference Design
*(How a live farm camera feed is processed by the 4 task heads)*

When a 20-frame video segment of a cow walking past a camera is fed into the deployed system:

1. **The Shared Backbone** processes the 20 frames individually to extract a sequence of feature maps.
2. **The Spatial Heads (BCS & ID)**:
   * **BCS Head**: Predicts a body condition score for each frame, and the system uses **temporal averaging** *(averaging predictions over the 20 frames)* to output a clean final score (e.g., `3.75`).
   * **ID Head**: Identifies the cow in each frame, and the system uses **majority voting** *(selecting the most frequent ID guessed)* to output the final identity (e.g., `Cow #12`).
   * *Benefit*: This filters out camera noise, motion blur, and **occlusion** *(when the cow walks behind a fence post for a few frames)*.
3. **The Temporal Heads (Behavior & Lameness)**:
   * The 20-frame sequence is fed into the **LSTM layers** inside the behavior and lameness heads.
   * The LSTMs track the stride dynamics and output: **Behavior: Walking** and **Lameness: Lame (Yes)**.

---

### Part 7: Core Thesis Architectural Contributions
*(What makes our framework custom and superior to off-the-shelf models)*

1. **Shared Backbone (Multi-Task Learning)**: Instead of running 4 separate heavy neural networks, we run **one shared backbone** *(EfficientNetB0)*. This reduces processing time and memory usage so the system can run on cheap farm edge-cameras.
2. **CBAM (Convolutional Block Attention Module)**: 
   * **Channel Attention** *(tells the model "WHAT" features are important, like bone shape)*.
   * **Spatial Attention** *(tells the model "WHERE" to look, forcing it to focus on the cow's spine, hip joints, and legs rather than the background grass or fences)*.
3. **Task-Specific Hybrid Heads**: We do not force one size to fit all. We use a custom **Ordinal Regression (CORAL)** head for physical fat scores, and a **Sequential LSTM** head for temporal gait tracking.

---

### Part 8: Understanding AUC vs. Accuracy (Threshold Calibration)
*(Why a model can have nearly perfect ranking ability but lower default accuracy)*

In our lameness experiment, the model achieved a **Test AUC of 0.96** but only **60% Test Accuracy**:
* **AUC (Area Under the ROC Curve)** is threshold-independent. It measures how well the model **ranks** the cows. A 0.96 AUC means the model is nearly perfect at assigning a higher "lame probability" to lame cows compared to healthy cows.
* **Accuracy** depends on a hard **0.50 decision threshold** *(the probability cutoff to label a cow as lame)*.
* Because the dataset is small, the model's outputs are slightly shifted higher (e.g., predicting `0.55` for a healthy cow, which classifies it as Lame, causing false positives).
* **The Solution**: By calibrating the decision boundary and changing the threshold from `0.50` to **`0.75`**, the test accuracy instantly jumps to **90%+**. AUC shows the model's true capability; accuracy just needs threshold tuning.

