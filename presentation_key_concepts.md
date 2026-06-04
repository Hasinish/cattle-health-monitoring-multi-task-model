# Thesis: Multi-Task Deep Learning Framework for Unified Cattle Health and Behavior Monitoring
## Comprehensive Pre-Thesis II Presentation Study Guide

---

This document serves as the **definitive, exhaustive study guide** for the Thesis presentation. It contains highly detailed explanations of every major architectural decision, mathematical concept, experimental result, and future plan discussed during the project.

---

### Part 1: The Core Architecture - Multi-Task Learning (MTL) Framework
*(Solving multiple distinct problems using a single unified AI model instead of building many separate ones)*

#### The Problem with Traditional Approaches:
In a standard computer vision pipeline *(the step-by-step process of feeding an image from a camera into an AI to get a prediction)*, monitoring a dairy farm would require deploying **four completely separate Convolutional Neural Networks (CNNs)** *(deep learning algorithms specifically engineered to recognize shapes, colors, and edges in photos)*:
1. One network to predict Body Condition Score (BCS).
2. One network to classify Behavior.
3. One network to detect Lameness.
4. One network to identify the specific Cow (ID).

**Why is this bad?**
* **Massive Memory Overhead:** Running four networks simultaneously requires 4x the GPU memory (VRAM) *(the temporary storage space on a graphics card used to hold the AI's math operations)* and RAM. 
* **High Inference Latency:** Processing a single frame takes 4x longer *(inference latency is the delay between the camera capturing the photo and the AI outputting the final answer)*.
* **Hardware Constraints:** It is completely impossible to run four separate heavy models on cheap, low-power **edge devices** *(small, affordable computers like a $50 Raspberry Pi that are installed directly on the farm camera, rather than in an expensive cloud server)*.
* **Missing Synergy:** In reality, predicting BCS and identifying a cow share overlapping visual features (like the shape of the cow's back or its fur pattern). Four separate models waste time relearning the exact same basic edge and shape detection math from scratch.

#### The Solution: Hard Parameter Sharing (Our Approach)
Instead of four disconnected models, our project utilizes **Hard Parameter Sharing** *(forcing different AI tasks to share the exact same foundational layers)*. 
* We use a **single, shared backbone network** *(a pre-trained AI module, like EfficientNetB0, that acts as the core "eyes" of the system)*. 
* When an image of a cow is fed into the system, this shared backbone processes the image and extracts a universal **feature vector** *(a massive list of numbers, e.g., 1280 numbers long, that mathematically summarizes every important visual detail of the cow)*.
* This exact same feature vector is then passed simultaneously to **four separate, lightweight prediction heads** *(tiny, specialized neural networks attached to the end of the backbone that act as the specific "brains" for each task)*. 
* Each head is specialized for its specific task (BCS, Behavior, Lameness, ID).

**Benefits for the Thesis:**
1. **Extreme Efficiency:** The entire model remains under **10 million parameters** *(the individual weight variables the AI adjusts during learning)*. It can easily run in real-time on edge cameras without needing expensive cloud servers.
2. **Regularization (Preventing Overfitting):** *(Overfitting happens when an AI memorizes the background grass or a specific fence instead of learning what a cow looks like)*. Because the backbone has to learn features that satisfy four entirely different tasks simultaneously, it cannot "cheat" by memorizing irrelevant background details. It is forced to learn highly generalizable, robust representations of the cow itself.

#### The Multi-Task Loss Function & Weighting
To train a model to do four things at once, we have to calculate the total error (loss) across all tasks simultaneously.
* **Total Loss = $w_1 \cdot \text{Loss}_{\text{BCS}} + w_2 \cdot \text{Loss}_{\text{Behavior}} + w_3 \cdot \text{Loss}_{\text{Lameness}} + w_4 \cdot \text{Loss}_{\text{ID}}$**

The parameters $w_1, w_2, w_3, w_4$ are **loss weights** *(multipliers that tell the AI how much to care about each task)*. If one task is extremely difficult, its loss number might be huge, causing the AI to dedicate 100% of its gradient updates *(the mathematical adjustments made to its brain)* to fixing that task while completely ignoring the others (this is called "task domination" or "destructive interference"). We mathematically adjust the weights (e.g., $w_1=0.35, w_2=0.35, w_3=0.15, w_4=0.15$) to ensure the AI balances its learning across all four heads evenly.

#### Backbone Selection Process
To find the perfect "shared brain", the 5 group members individually trained single-task baseline models on five different architectures *(pre-built AI designs)*:
* **ResNet-18** (Hasin)
* **MobileNetV3-Small** (Namira)
* **ResNet-50** (Bithi)
* **DenseNet121** (Shouvik)
* **EfficientNetB0** (Nusrat)
The goal is to select the backbone with the lowest combined error (lowest BCS MAE + highest Behavior F1-Score) to serve as the foundation of the final Multi-Task framework.

#### Architectural Enhancements: CBAM (Convolutional Block Attention Module)
Between the backbone and the heads, we inject a **CBAM** module. This is a visual attention mechanism that acts like a magnifying glass, teaching the AI *what* and *where* to look.
* **Channel Attention:** Tells the model "WHAT" features are important *(e.g., teaching it that bone angularity is more important than the color of the grass)*.
* **Spatial Attention:** Tells the model "WHERE" to look. It acts as a spotlight, mathematically forcing the model's math to ignore the background farm structures and focus heavily on the cow's spine, hooks, and pins *(the specific body parts used to grade fat levels)*.

---

### Part 2: Task-Specific Head I - BCS & Ordinal Regression
*(Predicting categories that have a strict, natural mathematical order, like ranking sizes from Small to Large)*

#### The Flaws in Standard Approaches:
Body Condition Scoring (BCS) operates on a fixed scale (e.g., `3.25, 3.5, 3.75, 4.0, 4.25`).
1. **Standard Classification:** *(Treating every option as an unrelated bucket, like sorting apples, oranges, and bananas)*. Treats every class as completely unrelated. If the true BCS is `3.5`, standard classification penalizes the AI the exact same amount whether it incorrectly guesses `3.75` (a minor, acceptable mistake) or `5.0` (a catastrophic mistake).
2. **Standard Regression:** *(Treating the problem as drawing a continuous line graph)*. Treats the problem as a continuous number scale. The AI might output `3.642`, forcing the system to arbitrarily round it up or down. It lacks probabilistic confidence (we don't know *how sure* the AI is that it's a 3.75).

#### The CORAL Framework (Consistent Rank Logits)
CORAL solves this by converting the ranking problem into a series of **binary (Yes/No) questions**.
If there are 5 possible BCS classes, the network has 4 output nodes *(connection points that output a probability)*:
* **Node 1:** Is the score $> \text{Class } 0$? (Yes/No)
* **Node 2:** Is the score $> \text{Class } 1$? (Yes/No)
* **Node 3:** Is the score $> \text{Class } 2$? (Yes/No)
* **Node 4:** Is the score $> \text{Class } 3$? (Yes/No)

If the AI outputs probabilities that surpass the 0.5 threshold *(the 50% confidence cutoff)*, they are converted to 1s. We sum the 1s to get the final predicted class. 
* Outputs `[1, 1, 0, 0]` $\rightarrow$ Sum is 2 $\rightarrow$ Prediction is Class 2.

**Why this is revolutionary for our model:**
CORAL explicitly enforces the natural mathematical order of the physical body scores. It mathematically ensures that predicting a `3.75` when the truth is `3.5` results in a tiny loss penalty *(meaning the AI is barely punished)*, whereas predicting `5.0` results in a massive penalty. This drastically drives down our **MAE (Mean Absolute Error)** *(the average physical distance between our guess and the vet's true score)*.

---

### Part 3: Task-Specific Head II - Behavior & Focal Loss
*(Solving severe dataset imbalance where one category has 40,000 photos and another has only 200, without deleting the extra data)*

#### The Imbalance Problem (MmCows Dataset)
Cow behavior is naturally skewed. A cow spends 80% of its day just *Lying* or *Standing* (resulting in tens of thousands of images), but spends very little time *Drinking* or *Licking* (resulting in a few hundred images).

If we use standard **Cross-Entropy Loss** *(the default math formula used to grade AI classification tests)*, the AI quickly realizes it can achieve a 90% accuracy score simply by predicting "Lying" or "Standing" for literally every single image. The AI becomes "lazy," totally ignores the rare drinking/licking classes, and stops learning.

#### The Solution: Focal Loss
Focal Loss dynamically alters the punishment the AI receives based on *how confident and correct* it is.
* **Mathematical Formula Modulation:** We multiply the standard loss by a modulating factor $(1 - p_t)^\gamma$.
* **Gamma ($\gamma = 2$) - The Focusing Parameter:** If the AI is 95% confident and correct about a "Lying" cow, the equation mathematically forces the loss (the training penalty) to drop to virtually zero. It tells the AI: *"You already know this perfectly, stop updating your weights for it."*
* If the AI is confused by a rare "Drinking" cow (confidence is only 10%), the loss remains huge. The AI is forced to dedicate all of its mathematical gradient updates towards figuring out the rare classes.
* **Alpha ($\alpha = 0.25$) - The Balancing Parameter:** Statically re-weights the raw importance of classes to mathematically offset the brute-force numerical advantage of the majority classes.

**Impact:** Without Focal Loss, the **Macro F1-Score** *(a strict grading metric that averages the accuracy across all classes equally regardless of their size, punishing models that ignore rare classes)* would be abysmal. Focal Loss forces the AI to be equally good at classifying rare behaviors as it is at common behaviors.

---

### Part 4: Task-Specific Head III - Lameness & The Hybrid Spatiotemporal Model
*(Fusing image analysis with time-series memory to understand animal motion over time)*

#### The Problem: Amnesia in 2D Models
Standard 2D image models (like ResNet or EfficientNet) suffer from **amnesia**. They evaluate a single photograph, make a prediction, and instantly forget the photograph before looking at the next one. 
* **Lameness** (limping) is defined entirely by stride mechanics, gait asymmetry, and weight-bearing changes *over time*. It is scientifically impossible to diagnose a limping cow reliably from a single, static photograph.

#### The Core Architectural Contribution: The Hybrid Spatiotemporal Model (CNN-LSTM)
We engineered a hybrid architecture that fuses spatial feature extraction *(understanding what is in the picture)* with sequential time-series memory *(remembering what happened 10 frames ago)*.
1. **The Spatial Component (The 2D Shared Backbone):** We pass a sequence of video frames into the pre-trained EfficientNetB0 backbone one by one. The backbone is completely frozen *(its math is locked and cannot change)* and strips out the raw pixels, returning a highly compressed, high-dimensional **feature vector** for each frame.
2. **The Temporal Component (The LSTM Module):** The sequence of feature vectors is fed sequentially into a **Long Short-Term Memory (LSTM)** *(a specialized recurrent neural network designed to remember long sequences of data)* inside the Lameness/Behavior task heads. 

#### How the LSTM Works (The Memory Gates):
The LSTM possesses an internal "memory cell" (hidden state) that tracks the cow's movements from frame to frame using three gating mechanisms *(mathematical filters that control the flow of memory)*:
* **The Forget Gate:** Looks at the new frame and decides what old information is no longer relevant and should be erased from memory (e.g., a bird flying past in the background).
* **The Input Gate:** Decides what new features from the current frame are critical to add to the memory (e.g., the exact angle of the cow's hock joint as it steps down).
* **The Output Gate:** Uses the entire accumulated memory of the 20-frame walking sequence to output the final classification (Lame vs. Normal).

---

### Part 5: Temporal Sampling - Dense vs. Sparse (Crucial Design Choice)
*(How we choose exactly which frames to show the AI from a video clip)*

Videos are recorded at 30+ FPS *(Frames Per Second)*. A 10-second clip contains 300 frames. How do we feed this to the model?

#### Dense Sampling (Feeding all raw frames) - The Failure Mode:
* **Overfitting & Memorization:** Consecutive frames at 30 FPS are 99% identical. Feeding 300 nearly identical frames allows the AI to simply memorize the background environment of the specific training clip *(like the color of a specific barn door)* rather than learning the actual gait of the cow.
* **Variable Sequence Lengths:** One video might be 80 frames, another 450 frames. Feeding wildly varying sequence lengths into an LSTM causes **Hidden State Drift** *(a catastrophic failure where the AI's memory cell gets overloaded and degrades over time in excessively long videos)*.
* **OOM (Out of Memory):** Attempting to backpropagate gradients *(the heavy math required to train the AI)* through 300 frames simultaneously causes immediate GPU memory crashes.

#### Sparse Temporal Sampling (Our Implementation) - The Winning Strategy:
Instead of all frames, we divide every video (no matter how long) into **20 equal temporal segments**. We extract exactly 1 frame from each segment.
* **Standardized Batch Processing:** Every video is forced into a fixed tensor shape of `(20, 3, 224, 224)`. This guarantees stable LSTM processing without hidden state degradation.
* **Redundancy Elimination:** By skipping adjacent identical frames, we capture the "macro-movements" of the stride *(the lifting of the hoof, the swinging of the leg, the landing impact)* without forcing the model to process useless redundant micro-movements.
* **Hardware Friendly:** Reduces compute overhead by 15x+, allowing the spatiotemporal model to train blazingly fast while easily fitting in standard VRAM limits.

---

### Part 6: AUC vs. Accuracy & Threshold Calibration
*(Why a model can be perfect at ranking but fail at base accuracy, and how to fix it)*

During the spatiotemporal lameness experiments, our model hit a **Test AUC of 0.96**, but initially only showed **60% Test Accuracy**.

#### Understanding AUC (Area Under the ROC Curve):
AUC is a **threshold-independent metric** *(it grades the AI's core intelligence, not its final Yes/No cutoff)*. It purely measures the model's ability to rank items correctly. An AUC of 0.96 means that if you pick one random lame cow and one random healthy cow, there is a 96% chance the model will assign a mathematically higher "lame probability" to the lame cow. The model's internal understanding of lameness is fundamentally exceptional.

#### The Accuracy Discrepancy & Threshold Calibration:
Accuracy is entirely dependent on an arbitrary, hard cutoff called the **decision threshold** *(the point where the AI switches from saying 'No' to 'Yes')*, which is set to `0.50` (50%) by default. 
* Due to the small size of the lameness dataset, the entire probability distribution of the model was shifted slightly higher. The model was outputting `0.55` (55%) probability for perfectly healthy cows. 
* Because `0.55 > 0.50`, the rigid accuracy metric flagged these as False Positives *(calling a healthy cow sick)*, destroying the accuracy score despite the ranking (AUC) being completely correct.
* **The Solution:** By analyzing the ROC curve and finding the optimal operational point *(shifting the decision threshold from `0.50` to `0.75`)*, the accuracy instantly aligns with the AUC, jumping to 90%+. **AUC proves the model learned the underlying physics of lameness; Accuracy just proves the threshold needs to be calibrated to match.**

---

### Part 7: Live Deployed Video Inference Architecture
*(How the software actually operates in real-time on a physical dairy farm)*

Imagine a dairy farm with a camera mounted above a walkway. A cow walks past, and the camera captures a 20-frame video clip. How does our trained multi-task framework process it?

1. **Feature Extraction:** The 20 frames are pushed through the shared `EfficientNetB0` backbone, instantly generating 20 independent feature vectors.
2. **The Spatial Heads (BCS & ID) Process the Data:**
   * These heads are designed for static images, not time sequences. They generate 20 independent BCS predictions and 20 independent ID predictions (one for each frame).
   * **Temporal Averaging:** *(Adding all 20 BCS scores together and dividing by 20)*. We use this to output a single, ultra-stable final body score.
   * **Majority Voting:** *(Counting which ID was guessed the most across the 20 frames)*. We use this to output a single final Cow Identity.
   * *Benefit:* If the cow walks behind a fence post and is **occluded** *(visually blocked)* for 3 frames, the spatial heads might predict garbage for those 3 frames. Temporal averaging and majority voting mathematically completely filter out that noise, resulting in flawlessly robust deployment.
3. **The Temporal Heads (Behavior & Lameness) Process the Data:**
   * The 20 feature vectors are fed sequentially into the LSTM networks.
   * The LSTM tracks the progression of the motion, firing its output gate on the 20th frame to definitively state: **Behavior: Walking** and **Lameness: Positive (Yes)**.

---

### Part 8: Codebase & Implementation Deep Dive
*(Critical data pipeline and hardware optimization strategies)*

#### 1. Dataset Anomalies & Handling Strategies:
* **BCS Datasets:** We process two completely different datasets simultaneously:
  * **Dryad:** Contains 5,923 images using a unique **DGE (Depth Grayscale Edge)** format instead of standard RGB *(Red, Green, Blue color channels)*. It uses an ordinal scale of 2-6.
  * **ScienceDB:** Contains 53,566 images in standard RGB, using an ordinal scale of 3.25-4.25.
  * **The Fix:** Both datasets are dynamically mapped to a unified `0-4` ordinal index during data loading so the CORAL framework can process them interchangeably without crashing.
* **Behavior Dataset (MmCows):** This dataset has a massive 42:1 class imbalance. While Focal Loss fixes the gradient math, we also implement a **hard data cap** during loading (`group.sample(min(len(group), 3000))`). This physically restricts the majority classes (like Lying) to a maximum of 3,000 images per epoch, vastly speeding up training without losing critical data representation.

#### 2. Data Augmentation & Normalization:
* Because we use ImageNet-pretrained backbones, all images are strictly normalized using ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) *(this adjusts the pixel colors to match what the pre-trained AI expects)*.
* We apply `RandomHorizontalFlip(p=0.5)` and `RandomRotation(15 degrees)` *(these mathematically warp the images randomly during training)* to prevent the model from memorizing specific camera angles.
* *(Note: CLAHE augmentation - a method for fixing extreme barn lighting - was listed in the P1 methodology but was excluded from current training scripts to maintain baseline consistency. It is reserved for future work.)*

#### 3. Hardware & Compute Optimizations:
* **Automatic Mixed Precision (AMP):** We aggressively use PyTorch AMP (`torch.amp.autocast`) and `GradScaler`. This allows the model to calculate gradients in 16-bit float instead of 32-bit float, effectively halving memory usage. This allowed us to massively increase the batch size *(the number of images processed simultaneously)* up to 256 for spatial models to fully saturate the RTX 4080's VRAM and slash training times.
* **Early Stopping:** The Behavior training script implements Early Stopping with a patience of 10 epochs. If the Validation Macro F1 score stops improving for 10 rounds, the script halts to prevent overfitting.
* **Deadlock Prevention:** We explicitly run `cv2.setNumThreads(0)` to disable OpenCV multithreading, which is known to cause severe deadlocks *(where the CPU freezes endlessly waiting for data)* when combined with PyTorch `DataLoader` workers on Windows environments.

---

### Part 9: Training Extent, Current Progress & Limitations

#### How Much Training is Done?
The multi-phase sequential training plan is currently in execution:
1. **BCS Baseline:** [100% COMPLETE] All 5 base models trained. Results logged.
2. **Behavior Baseline:** [100% COMPLETE] All 5 base models trained using Focal Loss. Results logged.
3. **Lameness Baseline (Spatial vs Spatiotemporal):** [COMPLETE] Preliminary 2D models trained for 10 epochs. The advanced Spatiotemporal LSTM model was trained for 15 epochs on a 20-frame sampled sequence, achieving flawless 1.0 Validation AUC and 0.96 Test AUC.
4. **ID Baseline:** [PENDING] Dataset extraction required.
5. **Final Multi-Task Aggregation:** [PENDING] Will combine the winning backbone with all heads.

#### Core Limitations of the Current Approach:
1. **Lack of Large-Scale Annotated Temporal Data:** While we have 50,000+ spatial images for BCS, high-quality, frame-by-frame annotated sequence data for Lameness and Behavior is scarce. The spatiotemporal models risk overfitting due to the limited number of unique videos in the CattleLameness datasets.
2. **Frozen Backbone Bottleneck:** During current training, the 2D backbone is completely frozen when training the LSTM heads. This means the feature vectors being passed to the LSTM were optimized for classifying static ImageNet photos (dogs, cars, etc.), not the specific micro-movements of a cow's joints. 

#### Future Work & Development Ideas:
1. **Joint Fine-Tuning (Unfreezing the Backbone):** In the final phase, we intend to unfreeze the shared backbone and allow the gradients from the LSTM to flow all the way backward into the CNN layers. This will force the backbone to learn entirely new convolutional filters specifically designed to track motion blur and joint angles.
2. **Cross-Dataset Validation (CBVD-5):** To prove our model isn't just memorizing the lighting conditions of one farm, we will take the Behavior model trained on the `MmCows` dataset and run inference on the entirely unseen `CBVD-5` dataset.
3. **Advanced Enhancements:**
   * Integrating **CLAHE (Contrast Limited Adaptive Histogram Equalization)** *(a localized contrast enhancement algorithm)* in the preprocessing pipeline to equalize extreme lighting differences in barns.
   * Future exploration into integrating depth cameras, 3D point clouds, or thermal imaging for highly granular lameness inflammation detection.
