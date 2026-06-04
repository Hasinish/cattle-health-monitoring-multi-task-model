# Thesis: Multi-Task Deep Learning Framework for Unified Cattle Health and Behavior Monitoring
## Comprehensive Pre-Thesis II Presentation Study Guide

---

This document serves as the **definitive, exhaustive study guide** for the Thesis presentation. It contains highly detailed explanations of every major architectural decision, mathematical concept, experimental result, and future plan discussed during the project. It uses simple English and actual, concrete dataset and model examples to make these complex topics easy to explain to examiners.

---

### Part 1: The Core Architecture - Multi-Task Learning (MTL) Framework
*(Solving multiple distinct problems using a single unified AI model instead of building many separate ones)*

#### The Problem with Traditional Approaches:
In a standard computer vision pipeline *(the step-by-step process of feeding an image from a camera into an AI to get a prediction)*, monitoring a dairy farm would require deploying **four completely separate Convolutional Neural Networks (CNNs)**:
1. One network to predict Body Condition Score (BCS) *(fat level)*.
2. One network to classify Behavior *(lying, standing, drinking, licking)*.
3. One network to detect Lameness *(limping)*.
4. One network to identify the specific Cow (ID) *(ear tag/marking recognition)*.

##### Real-World Example of the Problem:
If you deploy 50 cameras on a farm, running 4 separate models on each camera causes:
* **Massive Memory Overhead:** Running 4 models simultaneously requires 4 times the GPU memory (VRAM) and RAM. 
* **High Inference Latency:** Processing a single frame takes 4 times longer because the computer has to run 4 separate mathematical forward passes one after another.
* **Hardware Constraints:** You cannot run 4 heavy models on cheap, low-power **edge devices** *(small, affordable computers like a $50 Raspberry Pi installed directly on the farm camera)*.
* **Missing Synergy:** BCS and Cow ID both rely on the shape of the cow's back and its skin patterns. In 4 separate models, Model 1 and Model 4 waste computation learning the exact same basic edge and shape detection math independently.

#### The Solution: Hard Parameter Sharing (Our Approach)
Instead of 4 disconnected models, our project utilizes **Hard Parameter Sharing** *(forcing different AI tasks to share the exact same foundational layers)*.

##### Actual Technical Example:
We feed an image of a cow into a single shared backbone network (EfficientNetB0). 
* **The Shared Backbone:** EfficientNetB0 processes the raw pixels and extracts a universal **feature vector** *(a compressed list of 1,280 numbers summarizing the cow's shapes, curves, and patterns)*.
* **Specialized Heads:** This exact same 1280-length feature vector is passed simultaneously to four separate, lightweight prediction heads:
  1. Head 1 uses it to output a BCS score.
  2. Head 2 uses it to output a Behavior classification.
  3. Head 3 uses it to output a Lameness prediction.
  4. Head 4 uses it to output a Cow ID.

##### Benefits for the Thesis:
1. **Extreme Efficiency:** The entire model remains under **10 million parameters** *(the weight variables adjusted during learning)*. It can run in real-time on edge cameras without cloud latency.
2. **Regularization (Preventing Overfitting):** Overfitting happens when an AI memorizes the background grass or a specific farm fence instead of learning what a cow looks like. Because the shared backbone must satisfy 4 entirely different tasks at the same time, it cannot "cheat" by memorizing irrelevant details. It is forced to learn highly robust, general representations of the cow itself.

#### The Multi-Task Loss Function & Weighting
To train a model to do 4 things at once, we calculate the total error (loss) across all tasks simultaneously using a weighted sum:

`Total Loss = (w1 * Loss_BCS) + (w2 * Loss_Behavior) + (w3 * Loss_Lameness) + (w4 * Loss_ID)`

The parameters `w1, w2, w3, w4` are **loss weights** *(multipliers that control how much the AI cares about each task)*. 

##### Actual Technical Example:
During training, if the Behavior task has a very high loss of 2.5, and the Lameness task has a low loss of 0.1, the network will focus almost entirely on adjusting its weights to minimize the Behavior loss. This is called "task domination" or "destructive interference." By scaling the loss weights (e.g., `w1=0.35, w2=0.35, w3=0.15, w4=0.15`), we force the optimization gradients to have similar magnitudes, ensuring the network learns both tasks simultaneously.

#### Backbone Selection Process
To find the perfect "shared brain," the 5 group members individually trained single-task baseline models on 5 different architectures:
* **ResNet-18** (Hasin)
* **MobileNetV3-Small** (Namira)
* **ResNet-50** (Bithi)
* **DenseNet121** (Shouvik)
* **EfficientNetB0** (Nusrat)
The group evaluated these backbones to select the one with the lowest combined error (lowest BCS error + highest Behavior F1-score) to serve as the foundation of the Multi-Task model.

#### Architectural Enhancements: CBAM (Convolutional Block Attention Module)
Between the backbone and the heads, we inject a **CBAM** module. This is a visual attention mechanism that acts like a magnifying glass, teaching the AI *what* and *where* to look.
* **Channel Attention (The "WHAT"):** Focuses on *what* features are important *(e.g., teaching the model that bone angularity is more important than the green color of the grass)*.
* **Spatial Attention (The "WHERE"):** Focuses on *where* to look. It acts as a spotlight, mathematically forcing the model to ignore the background farm structures and focus heavily on the cow's spine, hooks, and pins *(the specific hip bones used to grade fat levels)*.

---

### Part 2: Task-Specific Head I - BCS & Ordinal Regression
*(Predicting categories that have a strict, natural mathematical order, like ranking sizes from Small to Large)*

#### The Flaws in Standard Approaches:
Body Condition Scoring (BCS) operates on a fixed scale (e.g., `3.25, 3.5, 3.75, 4.0, 4.25`).
1. **Standard Classification Flaw:** Treats classes as completely unrelated. If a cow's true BCS is `3.5`, standard classification penalizes the AI the exact same amount whether it guesses `3.75` (a minor error of 0.25) or `4.25` (a severe error of 0.75). Both guesses get a 0% accuracy score.
2. **Standard Regression Flaw:** Treats the scale like a continuous line. The AI might output `3.642`, forcing the user to arbitrarily round it. It also lacks confidence scores for each class.

#### The CORAL Framework (Consistent Rank Logits)
CORAL solves this by converting the ranking problem into a series of **binary (Yes/No) questions**.
If there are 5 possible BCS classes, the network has 4 output nodes:
* **Node 1:** Is the score > Class 0 (3.25)? (Yes/No)
* **Node 2:** Is the score > Class 1 (3.5)? (Yes/No)
* **Node 3:** Is the score > Class 2 (3.75)? (Yes/No)
* **Node 4:** Is the score > Class 3 (4.0)? (Yes/No)

##### Actual Technical Example:
Imagine a cow walks by and the model outputs the following probabilities for the 4 nodes:
* Node 1 (>3.25): **95%** (above 50% threshold $\rightarrow$ **1**)
* Node 2 (>3.50): **80%** (above 50% threshold $\rightarrow$ **1**)
* Node 3 (>3.75): **15%** (below 50% threshold $\rightarrow$ **0**)
* Node 4 (>4.00): **2%**  (below 50% threshold $\rightarrow$ **0**)

We get the binary array `[1, 1, 0, 0]`. We sum these numbers: `1 + 1 + 0 + 0 = 2`. The predicted class index is **2**, which corresponds to a score of **3.75**.

##### Why this is revolutionary for our model:
CORAL mathematically guarantees that predicting a `3.75` when the truth is `3.5` results in a tiny loss penalty, while predicting `4.25` results in a massive penalty. This dramatically drives down our **MAE (Mean Absolute Error)** *(the average difference between our prediction and the true score)*.

---

### Part 3: Task-Specific Head II - Behavior & Focal Loss
*(Solving severe dataset imbalance where one category has 40,000 photos and another has only 200, without deleting the extra data)*

#### The Imbalance Problem (MmCows Dataset)
Cow behavior is naturally skewed. A cow spends 80% of its day just *Lying* or *Standing* (resulting in tens of thousands of images), but spends very little time *Drinking* or *Licking* (resulting in a few hundred images).

##### Actual Technical Example:
If the dataset has 40,000 images of "Lying" cows and only 200 images of "Licking" cows, a model using standard Cross-Entropy loss will learn that it can get 99.5% accuracy by always outputting "Lying", even if it gets every single "Licking" image wrong. The AI stops learning how to detect rare behaviors.

#### The Solution: Focal Loss
Focal Loss dynamically alters the penalty the AI receives based on *how confident and correct* it is.
* **Mathematical Formula Modulation:** We multiply the standard loss by a modulating factor: `(1 - p_t)^gamma`.
* **Gamma (gamma = 2) - The Focusing Parameter:** If the AI is 95% confident and correct about an easy "Lying" image, the equation forces the loss to drop to virtually zero. The model does not adjust its weights for this image.
* If the AI is confused by a rare "Licking" image (confidence is only 10%), the loss remains high, forcing the optimizer to update the weights specifically to learn the "Licking" class.
* **Alpha (alpha = 0.25) - The Balancing Parameter:** Statically scales down the importance of majority classes to offset their raw numerical advantage.

##### Impact:
Without Focal Loss, the **Macro F1-Score** *(the strict metric that averages accuracy across all classes equally, penalizing models that ignore rare classes)* would be extremely low. Focal Loss forces the AI to be equally good at classifying rare behaviors as it is at common behaviors.

---

### Part 4: Task-Specific Head III - Lameness & The Hybrid Spatiotemporal Model
*(Fusing image analysis with time-series memory to understand animal motion over time)*

#### The Problem: Amnesia in 2D Models
Standard 2D image models (like ResNet or EfficientNet) suffer from **amnesia**. They look at a single photo, make a prediction, and instantly forget it before looking at the next one.

##### Actual Technical Example:
In a single frame (Frame 15), a cow's leg might be lifted off the ground, which looks exactly the same whether the cow is walking normally or limping. To detect lameness, the model must compare the time delay and angle difference of that leg landing in Frame 15 compared to Frame 5 and Frame 25.

#### The Core Architectural Contribution: The Hybrid Spatiotemporal Model (CNN-LSTM)
We built an architecture that fuses spatial feature extraction *(understanding what is in the picture)* with sequential time-series memory *(remembering what happened in previous frames)*.
1. **The Spatial Component (The 2D Shared Backbone):** We feed 20 video frames into the EfficientNetB0 backbone one by one. The backbone is completely frozen *(its weights are locked)* and compresses each frame into a high-dimensional **feature vector**.
2. **The Temporal Component (The LSTM Module):** The sequence of 20 feature vectors is fed into a **Long Short-Term Memory (LSTM)** network inside the Lameness head.

#### How the LSTM Works (The Memory Gates):
An LSTM tracks movements from frame to frame using an internal **cell state** *(its long-term memory)* and a **hidden state** *(its short-term working memory)* controlled by three mathematical gates:
* **The Forget Gate:** Looks at the new frame and decides what old information to throw away *(e.g., a bird flying past in the background)*.
* **The Input Gate:** Decides what new details from the current frame to store in memory *(e.g., the exact angle of the cow's leg as it touches the floor)*.
* **The Output Gate:** Uses the accumulated memory of the 20-frame walk sequence to output the final prediction (Lame vs. Normal).

---

### Part 5: Temporal Sampling - Dense vs. Sparse (Crucial Design Choice)
*(How we choose exactly which frames to show the AI from a video clip)*

Videos are recorded at 30+ FPS (Frames Per Second). A 10-second video contains 300 frames. How do we feed this to the model?

#### Dense Sampling (Feeding all 300 frames) - The Failure Mode:
* **Overfitting & Memorization:** Consecutive frames (e.g., Frame 1 and Frame 2) are 99% identical. Feeding 300 nearly identical frames allows the AI to simply memorize the background environment *(like the color of a barn wall)* instead of learning the actual gait of the cow.
* **Variable Sequence Lengths:** One video might be 80 frames, another 450 frames. Feeding wildly varying lengths into an LSTM causes **Hidden State Drift** *(where the LSTM's memory cell gets overloaded and degrades in excessively long sequences)*.
* **OOM (Out of Memory):** Processing 300 frames simultaneously requires massive VRAM, causing GPU crashes.

#### Sparse Temporal Sampling (Our Implementation) - The Winning Strategy:
Instead of all frames, we divide every video (regardless of its total duration) into **20 equal temporal segments** and extract exactly **1 frame** from the center of each segment.

##### Actual Technical Example:
If a video has 200 frames:
* Segment 1 is frames 1-10 $\rightarrow$ Extract frame 5.
* Segment 2 is frames 11-20 $\rightarrow$ Extract frame 15.
* ...and so on, resulting in exactly 20 evenly-spaced frames.

##### Benefits:
* **Standardized Shape:** Every video is processed as a fixed tensor of size `(20, 3, 224, 224)` (20 frames, 3 color channels, 224x224 pixels).
* **Redundancy Elimination:** It captures the "macro-movements" of the stride *(lifting the hoof, leg swing, landing impact)* while skipping redundant static frames.
* **Hardware Friendly:** Reduces compute overhead by 15x, allowing fast training within VRAM limits.

---

### Part 6: AUC vs. Accuracy & Threshold Calibration
*(Why a model can be perfect at ranking but fail at base accuracy, and how to fix it)*

During our spatiotemporal lameness experiments, our model achieved a **Test AUC of 0.96**, but initially showed only **60% Test Accuracy**.

#### Understanding AUC (Area Under the ROC Curve):
AUC is a **threshold-independent metric** *(it measures the model's core ability to rank items correctly, ignoring the final decision cutoff)*. 
* An AUC of 0.96 means that if you pick one random lame cow and one random healthy cow, there is a 96% chance the model will assign a higher lameness probability to the lame cow. The model's internal understanding of lameness is exceptional.

#### The Accuracy Discrepancy & Threshold Calibration:
Accuracy is dependent on an arbitrary decision threshold, which defaults to `0.50` (50%).

##### Actual Technical Example:
Because the lameness training set was small, the model's probability outputs shifted higher. It was outputting a `0.55` (55%) probability for perfectly healthy cows and `0.85` (85%) for lame cows.
* Because `0.55 > 0.50`, the default accuracy metric classified the healthy cows as Lame (False Positives), dragging the accuracy score down to 60%.
* Shifting the decision threshold from `0.50` to `0.70` (so only scores > 0.70 are classified as Lame) correctly separates the classes. The test accuracy immediately jumped to 90%+.

---

### Part 7: Live Deployed Video Inference Architecture
*(How the software actually operates in real-time on a physical dairy farm)*

Imagine a camera mounted above a farm walkway. A cow walks past, and the camera captures a 20-frame video clip.

1. **Feature Extraction:** The 20 frames are passed through the shared `EfficientNetB0` backbone, generating 20 independent feature vectors.
2. **Spatial Heads (BCS & ID) Process the Data:**
   * These heads are designed for static images. They generate 20 independent BCS predictions and 20 independent ID predictions (one for each frame).
   * **Temporal Averaging (For BCS):** We average all 20 predicted scores to output a single, ultra-stable body score.
   * **Majority Voting (For ID):** We count which Cow ID was predicted most often across the 20 frames and output that as the final ID.
   * **Occlusion Example:** If the cow walks behind a fence post and is blocked (occluded) for 3 frames, the spatial heads might predict incorrect values for those 3 frames. Temporal averaging and majority voting filter out that noise completely.
3. **Temporal Heads (Behavior & Lameness) Process the Data:**
   * The 20 feature vectors are fed sequentially into the LSTM networks.
   * The LSTM tracks the gait mechanics frame-by-frame and outputs the final classification on the 20th frame: **Behavior: Walking** and **Lameness: Positive**.

---

### Part 8: Codebase & Implementation Deep Dive
*(Critical data pipeline and hardware optimization strategies)*

#### 1. Dataset Anomalies & Handling:
* **BCS Datasets:** We train on two completely different datasets:
  * **Dryad:** 5,923 images using **DGE (Depth Grayscale Edge)** format instead of standard RGB. It uses a label scale of 2 to 6.
  * **ScienceDB:** 53,566 images in standard RGB, using a scale of 3.25 to 4.25.
  * **The Fix:** Both datasets are dynamically mapped to a unified `0-4` ordinal index during data loading so the CORAL framework can process them interchangeably without crashing.
* **Behavior Dataset (MmCows):** This dataset has a massive 42:1 class imbalance. While Focal Loss fixes the loss gradients, we implement a **hard data cap** during loading (`group.sample(min(len(group), 3000), random_state=42)`). This restricts majority classes (like Lying) to a maximum of 3,000 images per epoch.

##### Actual Technical Example:
Instead of processing 40,000 identical frames of lying cows (which adds redundant training time and memory overhead), the model trains on a balanced subset of 3,000 lying images per epoch, allowing faster epochs while retaining the representation of the rare behaviors (200 samples).

#### 2. Hardware & Compute Optimizations:
* **Automatic Mixed Precision (AMP):** We use PyTorch AMP (`torch.amp.autocast`) and `GradScaler`. This calculates gradients in 16-bit float (FP16) instead of 32-bit float (FP32), halving memory usage.

##### Actual Technical Example:
Standard FP32 uses 4 bytes per weight. FP16 uses 2 bytes. By using FP16, we halve the GPU VRAM requirement, allowing us to double the batch size from 32 to 64 to fully utilize the parallel tensor cores of the RTX 4080 GPU.
* **Early Stopping:** If the Validation Macro F1 score stops improving for 10 consecutive epochs, training halts to prevent overfitting.
* **Deadlock Prevention:** We explicitly run `cv2.setNumThreads(0)` to disable OpenCV multithreading, which prevents CPU deadlocks when combined with PyTorch `DataLoader` workers on Windows.

---

### Part 9: Training Extent, Current Progress & Limitations

#### How Much Training is Done?
The multi-phase sequential training plan is currently in execution:
1. **BCS Baseline:** [100% COMPLETE] All 5 base models trained. Results logged.
2. **Behavior Baseline:** [100% COMPLETE] All 5 base models trained using Focal Loss. Results logged.
3. **Lameness Baseline (Spatial vs Spatiotemporal):** [COMPLETE] Preliminary 2D models trained. The Spatiotemporal LSTM model was trained for 15 epochs on a 20-frame sampled sequence, achieving a 1.0 Validation AUC and 0.96 Test AUC.
4. **ID Baseline:** [PENDING] Dataset extraction required.
5. **Final Multi-Task Aggregation:** [PENDING] Will combine the winning backbone with all heads.

#### Core Limitations of the Current Approach:
1. **Lack of Large-Scale Annotated Temporal Data:** While we have 50,000+ spatial images for BCS, high-quality sequence data for Lameness and Behavior is scarce. The spatiotemporal models risk overfitting due to the limited number of unique videos in the datasets.
2. **Frozen Backbone Bottleneck:** Currently, the 2D backbone is completely frozen when training the LSTM heads. This means the feature vectors passed to the LSTM were optimized for classifying static ImageNet photos (dogs, cars, etc.), not the specific micro-movements of a cow's joints.

#### Future Work & Development Ideas:
1. **Joint Fine-Tuning (Unfreezing the Backbone):** In the final phase, we intend to unfreeze the shared backbone and allow gradients from the LSTM to flow all the way backward into the CNN layers. This forces the backbone to learn new convolutional filters specifically designed to track motion and joint angles.
2. **Cross-Dataset Validation (CBVD-5):** To prove generalizability, we will take the Behavior model trained on `MmCows` and run inference on the entirely unseen `CBVD-5` dataset.
3. **Advanced Enhancements:**
   * Integrating **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to equalize extreme lighting differences in barns.
   * Future exploration into integrating depth cameras or thermal imaging for highly granular lameness inflammation detection.
