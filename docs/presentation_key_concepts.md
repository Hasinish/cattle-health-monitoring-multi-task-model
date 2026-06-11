# Thesis: Multi-Task Deep Learning Framework for Unified Cattle Health and Behavior Monitoring
## Comprehensive Pre-Thesis II Presentation Study Guide

---

This document serves as the **definitive, exhaustive study guide** for the Thesis presentation. It contains highly detailed explanations of every major architectural decision, mathematical concept, experimental result, and future plan discussed during the project. It uses simple English and actual, concrete dataset and model examples to make these complex topics easy to explain. Every technical term is immediately defined in brackets for absolute clarity. All numbers in this guide are aligned with the final Thesis II (P2) report.

---

### Part 1: The Core Architecture - Multi-Task Learning (MTL) Framework
*(Solving multiple distinct problems using a single unified AI model instead of building many separate ones)*

#### The Problem with Traditional Approaches:
In a standard computer vision pipeline *(the step-by-step process of feeding an image from a camera into an AI to get a prediction)*, monitoring a dairy farm would require deploying **four completely separate Convolutional Neural Networks (CNNs)** *(deep learning algorithms specifically engineered to recognize shapes, colors, and edges in photos)*:
1. One network to predict Body Condition Score (BCS) *(a standardized visual grading system used to estimate a cow's body fat reserves)*.
2. One network to classify Behavior *(lying, standing, drinking, licking, etc.)*.
3. One network to detect Lameness *(limping)*.
4. One network to identify the specific Cow (ID) *(recognizing the individual animal from its visual appearance, like skin patterns)*.

##### Real-World Example of the Problem:
If you deploy 50 cameras on a farm, running 4 separate models on each camera causes:
* **Massive Memory Overhead:** Running 4 models simultaneously requires 4 times the GPU memory (VRAM) *(Video Random Access Memory, the temporary storage space on a graphics card used to hold the AI's math operations)* and RAM *(Random Access Memory, the computer's primary temporary workspace)*. Four separate EfficientNet-B0 models total ~21.2 million parameters versus one shared 5.3 million parameter encoder — roughly a **75% reduction** in memory.
* **High Inference Latency:** Processing a single frame takes 4 times longer because the computer has to run 4 separate forward passes *(the mathematical process of pushing input data forward through the neural network layers to generate a prediction)* one after another.
* **Hardware Constraints:** You cannot run 4 heavy models on cheap, low-power **edge devices** *(small, affordable computers like a Jetson Orin Nano or Raspberry Pi installed directly on the farm camera, rather than in an expensive cloud server)*.
* **Missing Synergy:** BCS and Cow ID both rely on the shape of the cow's back and its skin patterns. In 4 separate models, Model 1 and Model 4 waste computation learning the exact same basic edge and shape detection math independently.

#### The Three-Stage Pipeline (Our Approach)
Our framework is split into three connected segments rather than four disconnected models:
1. **Detection & Cropping:** A **YOLOv8-Nano** detector *(You Only Look Once, version 8, "Nano" = the smallest/fastest size; a tiny object-detection network that locates the cow and draws a box around it)* finds the cow and extracts its **bounding box** *(the tight rectangular box drawn around the detected animal)*. This removes background farm clutter (gates, grass, posts) and normalizes the input so the model only sees the cow.
2. **Feature Extraction:** The cropped image (or video sequence) is processed by **one shared backbone** enhanced with a CBAM attention module (explained below).
3. **Task Prediction:** The extracted features are routed into four task-specific heads.

#### Hard Parameter Sharing
The heart of the design is **Hard Parameter Sharing** *(forcing different AI tasks to share the exact same foundational layers)*.

##### Actual Technical Example:
We feed the YOLO-cropped image of a cow into a single shared backbone network *(a pre-trained neural network module, here EfficientNet-B0, that acts as the core feature extractor or "eyes" of the system)*.
* **The Shared Backbone:** The backbone processes the raw pixels and extracts a universal **feature vector** *(a compressed list of 1,280 numbers summarizing the cow's shapes, curves, and patterns)*.
* **Specialized Routing Paths:** The extracted feature vector is routed into two separate pathways:
  * **Static Path (A):** The vector is passed directly to lightweight prediction heads *(small neural network layers attached to the end of the backbone that perform the final classification or regression for each task)*:
    1. **Head 1** uses it to output a Body Condition Score (BCS).
    2. **Head 2** uses it to output a Behavior classification (frame-level).
    3. **Head 3** uses it to output a Cow ID.
  * **Sequence Path (B):** For video data, a sequence of 20 feature vectors is passed into a memory module:
    4. **Head 4** uses the sequence to output a Lameness prediction. *(In the full spatiotemporal model, Behavior is also routed through this sequence path to gain temporal context.)*

##### Benefits for the Thesis:
1. **Extreme Efficiency:** The entire model remains under **10 million parameters** *(the individual weight variables adjusted during learning)* — EfficientNet-B0 itself is only **5.3 million** parameters. It can run in real-time on edge cameras without cloud latency.
2. **Regularization (Preventing Overfitting):** Overfitting *(when an AI memorizes specific training details like the background grass rather than learning generalizable concepts)* is prevented. Because the shared backbone must satisfy 4 entirely different tasks at the same time, it cannot "cheat" by memorizing irrelevant details. It is forced to learn highly robust, general representations of the cow itself.

#### The Multi-Task Loss Function & Weighting
To train a model to do 4 things at once, we calculate the total error (loss) across all tasks simultaneously using a weighted sum of individual loss functions *(mathematical formulas that calculate how wrong the model's predictions are compared to the actual truth)*:

`Total Loss = (0.35 * Loss_BCS) + (0.35 * Loss_Behavior) + (0.15 * Loss_Lameness) + (0.15 * Loss_ID)`

The parameters `0.35, 0.35, 0.15, 0.15` are **loss weights** *(multipliers that control how much each task's error contributes to the total loss, determining its influence on learning)*. BCS and Behavior get the higher 0.35 weight because they are complex visual tasks backed by very large datasets (53,566 and ~21,000 images), while Lameness (50 videos) and ID (4,736 images) get 0.15. This stops the shared backbone from over-specializing on the small datasets.

##### Actual Technical Example:
During training, if the Behavior task has a very high loss of 2.5, and the Lameness task has a low loss of 0.1, the network will focus almost entirely on adjusting its weights to minimize the Behavior loss. This is called task domination or destructive interference *(when one task's large error updates override and erase the progress made on other tasks)*. By scaling the loss weights, we force the optimization gradients *(the directional math values indicating how to adjust the model's weights to reduce the error)* to have similar magnitudes, ensuring the network learns all tasks together. This is handled by the optimizer *(the algorithm, here Adam, that uses gradients to update the model's weights during training)*.

#### Backbone Selection Process & Baseline Results
To find the perfect "shared brain," the 5 group members individually trained single-task baseline models *(simple, single-task models trained to establish a performance benchmark before building the multi-task model)* on 5 different architectures *(the specific structure and arrangement of layers in a neural network)*. The primary metrics tracked are BCS Test MAE *(Mean Absolute Error on the unified 0-4 scale, lower is better)*, Behavior Test Macro F1 *(higher is better)*, and Lameness Test AUC *(Area Under the ROC Curve, higher is better)*.

Here is the baseline performance table showing the complete results of each base model across all 4 tasks:

| Person | Model | BCS MAE ↓ | BCS Exact Acc ↑ | BCS ±1 Acc ↑ | Behavior F1 ↑ | Lameness Acc ↑ | Lameness AUC ↑ | Cow ID Acc ↑ |
|--------|-------|-----------|----------------|-------------|--------------|---------------|----------------|-------------|
| **Nusrat** ✓ | EfficientNet-B0 | **0.557** | **55.06%** | **90.50%** | **0.745** | **93.05%** | **0.983** | **86.49%** |
| **Hasin** | ResNet-18 | 0.580 | 54.81% | 89.02% | 0.713 | 75.30% | 0.720 | 45.56% |
| **Shouvik** | DenseNet121 | 0.629 | 49.51% | 88.78% | 0.737 | 73.98% | 0.794 | 82.46% |
| **Bithi** | ResNet-50 | 0.649 | 48.43% | 88.20% | 0.704 | 80.97% | 0.744 | 53.02% |
| **Namira** | MobileNetV3-S | 0.709 | 44.67% | 86.47% | 0.681 | 75.66% | 0.723 | 78.83% |

*BCS metrics shown are on ScienceDB (the primary RGB dataset). ✓ = selected backbone.*

The backbone with the best overall rank is **EfficientNet-B0** (Nusrat). It achieves the best BCS MAE (`0.5566`), the highest Behavior F1 (`0.7445`), the best Lameness spatial Accuracy (`93.05%`) and AUC (`0.9829`) by a large margin, and the highest ID Top-1 Accuracy (`86.49%`), all while staying at only 5.3M parameters. It serves as the shared backbone for the final Multi-Task model.

*Important honesty note: the high lameness spatial AUC of 0.9829 is likely **inflated** because the entire lameness dataset is only 50 videos; frame-level splitting lets the same cows/backgrounds appear in both train and test, so this is an upper bound, not true generalization.*

#### Architectural Enhancements: CBAM (Convolutional Block Attention Module)
Between the backbone and the heads, we inject a **CBAM** module *(Convolutional Block Attention Module — a visual attention mechanism that guides the network to focus on relevant features and locations)*.
* **Channel Attention (The "WHAT"):** Focuses on *what* features are important *(e.g., teaching the model that bone angularity is more important than the green color of the grass)*. It uses average-pooling and max-pooling *(operations that summarize a feature map by taking its average or its strongest value)*.
* **Spatial Attention (The "WHERE"):** Focuses on *where* to look using a 7×7 convolution. It acts as a spotlight, mathematically forcing the model to ignore background farm structures and focus heavily on the cow's spine, hooks, and pins *(the specific hip bones used to grade fat levels)*.

The attention-weighted feature map is then reduced by **Adaptive Average Pooling** *(an operation that shrinks any feature map down to a fixed size by averaging)* and flattened to the 1,280-dimensional feature vector.

---

### Part 2: Task-Specific Head I - BCS & Ordinal Regression
*(Predicting categories that have a strict, natural mathematical order, like ranking sizes from Small to Large)*

#### The Flaws in Standard Approaches:
Body Condition Scoring (BCS) operates on a fixed scale (e.g., `3.25, 3.50, 3.75, 4.00, 4.25`).
1. **Standard Classification Flaw:** Standard classification *(treating categories as completely independent, unrelated buckets)* treats classes as completely unrelated. If a cow's true BCS is `3.5`, standard classification penalizes the AI the exact same amount whether it guesses `3.75` (a minor error of 0.25) or `4.25` (a severe error of 0.75). Both guesses get a 0% accuracy score.
2. **Standard Regression Flaw:** Standard regression *(predicting continuous numerical values along a continuous scale)* treats the scale like a continuous line. The AI might output `3.642`, forcing the user to arbitrarily round it. It also lacks confidence scores for each class.

#### The CORAL Framework (Consistent Rank Logits)
CORAL *(COnsistent RAnk Logits — an ordinal-regression method that preserves the natural order of classes)* solves this by converting the ranking problem into a series of **binary (Yes/No) questions** outputted by output nodes *(the final units in a neural network layer that output a prediction probability)*.
For `K` possible classes (here K=5), the network has `K-1 = 4` output nodes:
* **Node 1:** Is the score > Class 0 (3.25)? (Yes/No)
* **Node 2:** Is the score > Class 1 (3.50)? (Yes/No)
* **Node 3:** Is the score > Class 2 (3.75)? (Yes/No)
* **Node 4:** Is the score > Class 3 (4.00)? (Yes/No)

Each node is squashed by a **sigmoid** *(an S-shaped function that turns any number into a probability between 0 and 1)*, and the loss is the sum of **binary cross-entropies** *(the standard yes/no loss measuring the gap between predicted probability and the true 0/1 answer)* over the 4 nodes.

##### Actual Technical Example:
Imagine a cow walks by and the model outputs the following probabilities for the 4 nodes:
* Node 1 (>3.25): **95%** (above 50% threshold *(the cutoff value used to convert a continuous probability into a binary Yes/No decision)* → **1**)
* Node 2 (>3.50): **80%** (above 50% threshold → **1**)
* Node 3 (>3.75): **15%** (below 50% threshold → **0**)
* Node 4 (>4.00): **2%**  (below 50% threshold → **0**)

We get the binary array *(a list containing only 0s and 1s representing binary decisions)* `[1, 1, 0, 0]`. We sum these numbers: `1 + 1 + 0 + 0 = 2`. The predicted class index *(the integer position of a class in a list, starting at 0)* is **2**, which corresponds to a score of **3.75**.

##### Why this is revolutionary for our model:
CORAL mathematically guarantees that predicting a `3.75` when the truth is `3.5` results in a tiny loss penalty, while predicting `4.25` results in a massive penalty. This dramatically drives down our **MAE (Mean Absolute Error)** *(the average absolute difference between the predicted value and the actual true score)*. (Ablation result: CORAL gives MAE 0.5566 vs. 0.6940 for plain Cross-Entropy — a 19.8% improvement.)

---

### Part 3: Task-Specific Head II - Behavior & Focal Loss
*(Solving severe dataset imbalance where one category has tens of thousands of photos and another has only a couple thousand, without deleting the extra data)*

#### The Imbalance Problem (MmCows Dataset)
Cow behavior is naturally skewed. A cow spends most of its day just *Lying* or *Standing* (resulting in tens of thousands of images), but spends very little time *Drinking* or *Licking* (resulting in only a couple thousand images).

##### The 7 Behavior Classes (with actual training counts):
| Class | Behavior | Training images |
|-------|----------|----------------|
| Class 1 | Walking | 4,118 |
| Class 2 | Standing | 70,107 |
| Class 3 | Feeding head up | 19,080 |
| Class 4 | Feeding head down | 31,255 |
| Class 5 | Licking | 2,009 |
| Class 6 | Drinking | 3,311 |
| Class 7 | Lying | 83,806 |

This is a roughly **42:1 imbalance** (83,806 Lying ÷ 2,009 Licking ≈ 42).
*Note: During data loading, these are mapped to 0-indexed labels `0-6` by subtracting 1 (e.g. Class 1 becomes Label 0).*

##### Actual Technical Example:
With 70,107 images of "Standing" cows (Class 2) and only 2,009 images of "Licking" cows (Class 5), a model using standard **Cross-Entropy loss** *(the standard loss function for classification that measures the difference between predicted probability distributions and target classes)* will learn that it can get a high accuracy score by always outputting the majority class, even if it gets every single "Licking" image wrong. The AI stops learning rare behaviors due to **class imbalance** *(a dataset state where some categories have vastly more samples than others)*.

#### The Solution: Focal Loss
Focal Loss *(an altered loss function that dynamically scales the loss based on prediction confidence to focus training on hard, rare examples)* dynamically alters the penalty the AI receives based on *how confident and correct* it is.
* **Mathematical Formula Modulation:** We multiply the standard loss by a modulating factor: `(1 - p_t)^gamma`, where `p_t` is the model's predicted probability for the correct class.
* **Gamma (γ = 2) - The Focusing Parameter:** Gamma *(the focusing parameter that controls how heavily easy-to-classify examples are down-weighted)* scales down the loss for easy images. If the AI is 95% confident and correct about an easy "Lying" image, the equation forces the loss to drop to virtually zero. The model does not adjust its weights for this image.
* If the AI is confused by a rare "Licking" image (confidence only 10%), the loss remains high, forcing the optimizer to update the weights specifically to learn the "Licking" class.
* **Alpha (α = 0.25) - The Balancing Parameter:** Alpha *(the balancing parameter that scales class loss to offset numerical class imbalance)* statically scales down the importance of majority classes. *(In our implementation it is applied as a single uniform scalar rather than a per-class vector — a deliberate simplification.)*

##### Impact:
Without Focal Loss, the **Macro F1-Score** *(a metric that averages the F1-score across all classes individually, giving equal weight to each class regardless of its size)* would be low. Focal Loss raises the Behavior Test Macro F1 from `0.7074` (Cross-Entropy) to `0.7445` (Focal Loss), forcing the AI to be equally good at rare and common behaviors.

---

### Part 4: Task-Specific Head III - Lameness & The Hybrid Spatiotemporal Model
*(Fusing image analysis with time-series memory to understand animal motion over time)*

#### The Problem: Amnesia in 2D Models
Standard 2D image models *(networks that analyze single, static images without any time-series awareness)* suffer from amnesia. They look at a single photo, make a prediction, and instantly forget it before looking at the next one.

##### Actual Technical Example:
In a single frame (Frame 15), a cow's leg might be lifted off the ground, which looks exactly the same whether the cow is walking normally or limping. To detect lameness, the model must compare the time delay and angle difference of that leg landing in Frame 15 compared to Frame 5 and Frame 25. This requires tracking **stride mechanics** *(the physical kinematics of walking, such as stride length, joint flexion, and contact duration)* and **gait asymmetry** *(unequal walking movements between left and right limbs, a primary indicator of limping)* over time.

#### The Core Architectural Contribution: The Hybrid Spatiotemporal Model (CNN-LSTM)
We built a spatiotemporal model *(an architecture that processes both spatial details from images and temporal changes over time)* that fuses spatial feature extraction with sequential time-series memory *(the capacity to store and sequence information across consecutive points in time)*.
1. **The Spatial Component (The 2D Shared Backbone):** We feed 20 video frames into the EfficientNet-B0 backbone one by one. The backbone is **frozen** *(its weights are set to be non-trainable so they do not change during backpropagation)* and compresses each frame into a high-dimensional feature vector.
2. **The Temporal Component (The LSTM Module):** The sequence of 20 feature vectors is fed into a **Long Short-Term Memory (LSTM)** *(a specialized recurrent neural network designed to track and process sequential data over time without losing memory)* network inside the Lameness head. Our LSTM is a **single layer** with a **hidden size of 256** *(the number of memory units inside the LSTM)* and **dropout = 0.5** *(randomly switching off half the connections during training to prevent overfitting)*, followed by a linear layer that outputs one binary (Lame/Normal) score.

#### How the LSTM Works (The Memory Gates):
An LSTM tracks movements from frame to frame using an internal **cell state** *(its long-term memory that retains information across long sequences)* and a **hidden state** *(its short-term memory and output used for local sequence processing)* controlled by three memory gates *(mathematical filters that control the flow of information into and out of the LSTM memory cell)*:
* **The Forget Gate:** Looks at the new frame and decides what old information to throw away from the cell state *(e.g., a bird flying past in the background)*.
* **The Input Gate:** Decides what new details from the current frame to store in the cell state *(e.g., the exact angle of the cow's leg as it touches the floor)*.
* **The Output Gate:** Uses the accumulated cell state memory of the 20-frame walk sequence to output the hidden state prediction (Lame vs. Normal).

---

### Part 5: Temporal Sampling - Dense vs. Sparse (Crucial Design Choice)
*(How we choose exactly which frames to show the AI from a video clip)*

Videos are recorded at 30+ FPS *(Frames Per Second, the speed at which video frames are recorded or displayed)*. A 10-second video contains 300 frames. How do we feed this to the model?

#### Dense Sampling (Feeding all 300 frames) - The Failure Mode:
* **Overfitting & Memorization:** Consecutive frames (e.g., Frame 1 and Frame 2) are 99% identical. Feeding 300 nearly identical frames allows the AI to simply memorize the background environment *(like the color of a barn wall)* instead of learning the actual gait of the cow.
* **Variable Sequence Lengths:** One video might be 80 frames, another 450 frames. Feeding wildly varying lengths into an LSTM causes **Hidden State Drift** *(a degradation where an LSTM's memory becomes overloaded or corrupted over excessively long sequences)*.
* **OOM (Out of Memory):** Processing 300 frames simultaneously requires massive VRAM, causing OOM *(a hardware error where the GPU runs out of physical VRAM during training calculations)* crashes when we try to **backpropagate** *(the mathematical process of computing gradients backward from the loss through the network to update weights)*.

#### Sparse Temporal Sampling (Our Implementation) - The Winning Strategy:
Instead of all frames, we divide every video (regardless of its total duration) into **20 equal temporal segments** *(equal intervals of time split across the total duration of a video)* and extract exactly **1 frame** from the center of each segment.

##### Actual Technical Example:
If a video has 200 frames:
* Segment 1 is frames 1-10 → Extract frame 5.
* Segment 2 is frames 11-20 → Extract frame 15.
* ...and so on, resulting in exactly 20 evenly-spaced frames.

##### Benefits:
* **Standardized Shape:** Every video is processed as a fixed **tensor** *(a multi-dimensional math array used by deep learning frameworks to represent data)* of size `(20, 3, 224, 224)` (20 frames, 3 color channels *(the individual color layers, typically Red, Green, and Blue, that make up a digital image)*, 224×224 pixels *(the tiny individual colored dots that make up a digital image)*).
* **Redundancy Elimination:** It captures the "macro-movements" of the stride *(lifting the hoof, leg swing, landing impact)* while skipping redundant static frames.
* **Hardware Friendly:** Reduces compute overhead *(the amount of CPU/GPU processing power and time required to execute a program)*, allowing fast training within VRAM limits.

---

### Part 6: AUC vs. Accuracy & Threshold Calibration
*(Why a model can be perfect at ranking but fail at base accuracy, and why "fixing" it on the test set is cheating)*

During our spatiotemporal lameness experiments, the multi-task model achieved a **Test AUC of 1.00**, but under the default 0.50 threshold showed only **80.00% Test Accuracy**.

#### Understanding AUC (Area Under the ROC Curve):
AUC *(Area Under the Receiver Operating Characteristic Curve, a metric measuring a model's ability to rank positive cases higher than negative cases across all thresholds)* is a **threshold-independent metric** *(a grading metric that measures the model's core discriminative ability without relying on a specific decision cutoff)* based on the ROC curve *(Receiver Operating Characteristic curve, a graph plotting the True Positive Rate against the False Positive Rate at various thresholds)*.
* An AUC of 1.00 means that if you pick one random lame cow and one random healthy cow, there is a 100% chance the model assigns a higher lameness probability to the lame cow.

#### The Accuracy Discrepancy & Threshold Calibration:
Accuracy depends on an arbitrary **decision threshold** *(the probability cutoff value used to assign a sample to a class)*, which defaults to `0.50` (50%).

##### Actual Technical Example:
Because the lameness training set was tiny (50 videos total), the model's probability outputs shifted higher. It was outputting `0.55` (55%) for healthy cows and `0.85` (85%) for lame cows.
* Because `0.55 > 0.50`, the default metric classified healthy cows as Lame, generating **False Positives** *(healthy cases incorrectly flagged by the model as diseased or abnormal)* and dragging accuracy down to 80%.
* Shifting the threshold from `0.50` to `0.70` separated the classes and accuracy jumped to 100.00%.

##### ⚠️ The Critical Honesty Caveat (this is the key takeaway):
**This perfect 100% result is treated as INVALID for final evaluation.** Tuning the threshold *directly on the test set* is **post-hoc threshold tuning** *(adjusting the decision cutoff after already seeing the test results)*, which causes **test-set leakage** *(using information from the supposedly-unseen test set to tune the model, which artificially inflates the score)*. A rigorous method calibrates the threshold on a **separate validation split** *(a held-out subset used for tuning, kept distinct from the final test set)* and leaves the test set untouched. Combined with the tiny 50-video sample size, this perfect score is reported only as a **preliminary proof-of-concept**, not a clinical benchmark. Our honest headline number is the **uncalibrated 80.00% accuracy**.

---

### Part 7: Live Deployed Video Inference Architecture
*(How the software actually operates in real-time on a physical dairy farm)*

Imagine Deployed Video Inference *(running a trained model in a live production environment to generate predictions on new video feeds)* where a camera is mounted above a farm walkway. A cow walks past, and the system captures frames through a **sliding window** *(a moving buffer that keeps the most recent frames; here it fills until 20 cow crops are queued)*.

1. **Detection & Feature Extraction:** YOLOv8-Nano crops the cow from each frame; the 20 crops are passed through the shared `EfficientNet-B0` backbone, generating 20 independent feature vectors.
2. **Static Spatial Heads (BCS, ID, & Behavior) Process the Data:**
   * These heads generate 20 independent BCS predictions, 20 ID predictions, and 20 Behavior predictions (one per frame).
   * **Temporal Averaging (For BCS):** We use temporal averaging *(calculating the mathematical mean of predictions made across multiple frames over time)* to average all 20 predicted scores into a single, stable body score.
   * **Majority Voting (For ID & Behavior):** We use majority voting *(selecting the classification class that was predicted most frequently across a sequence of frames)* to output the Cow ID and Behavior predicted most often across the 20 frames.
   * **Occlusion Example:** If the cow walks behind a fence post and suffers from **occlusion** *(the state of being visually blocked or hidden from the camera's view by another object)* for 3 frames, the static heads might predict wrong values for those 3 frames. Temporal averaging and majority voting filter out that noise completely.
3. **Spatiotemporal Sequence Head (Lameness) Processes the Data:**
   * The sequence of 20 feature vectors is fed into the LSTM, which tracks gait mechanics frame-by-frame and outputs a single, stable locomotion score: **Lameness: Positive/Negative**.

---

### Part 8: Codebase & Preprocessing Deep Dive
*(Critical data pipeline, dataset splits, augmentation, and training strategy)*

#### 1. The Six Datasets (all publicly available):
| Dataset | Task | Size | Format | Labels |
|---------|------|------|--------|--------|
| **ScienceDB** | BCS (primary) | 53,566 images | RGB | 5 classes {3.25–4.25}, step 0.25 |
| **Dryad** | BCS | 5,923 images | DGE | 5 integer classes {2–6} |
| **MmCows** | Behavior | 213,686 frames (16 cows) | RGB | 7 classes |
| **CattleLameness** | Lameness | 50 videos (25 Lame / 25 Normal) | RGB video | binary {0 Normal, 1 Lame} |
| **OpenCows2020** | Cow ID | 4,736 images | RGB | 46 identity classes |
| **CBVD-5** | Behavior (test-only) | 2,000 balanced images | RGB | 5 behaviors (4 overlap ours) |

* **DGE (Depth Grayscale Edge)** *(an image format where the three channels are grayscale intensity, depth coordinates, and a Canny edge map, instead of plain Red/Green/Blue)*. **Canny edge map** *(an algorithm output that extracts the object's outlines/edges)*. DGE encodes physical 3D structure, not just color.
* **CBVD-5** is used **only as an independent test set** to measure generalization under domain shift; its "foraging" class is mapped to our "Feeding".

##### Splits and counts (cow-wise where applicable, 70/15/15):
* **ScienceDB:** Train 37,688 / Val 7,580 / Test 8,298 images.
* **Dryad:** Train 4,163 (102 cows) / Val 1,360 (22 cows) / Test 400 (23 cows).
* **MmCows:** Train 148,401 (11 cows) / Val 25,134 (2 cows) / Test 40,151 (3 cows).
* **CattleLameness:** Train 34 videos (680 frames) / Val 6 (120) / Test 10 (200).

*Note: Both BCS datasets are dynamically mapped to a unified `0-4` ordinal index during loading so the CORAL framework can process them interchangeably.*

#### 2. Augmentations & Normalization:
All training images are resized to **224×224** and **normalized with ImageNet statistics** *(subtract a fixed mean [0.485, 0.456, 0.406] and divide by a fixed standard deviation [0.229, 0.224, 0.225], the values computed from the ImageNet dataset so pre-trained weights work properly)*. **Data augmentation** *(artificially expanding the dataset by applying random transformations so the model sees more variety)* is applied to training only: **random horizontal flips** (probability 0.5) and **random rotations** (±15°). Validation/test images are resized and normalized but **not** augmented.

#### 3. Class Balancing via Capping:
The MmCows behavior dataset has a severe 42:1 imbalance. While Focal Loss fixes the loss gradients, we also apply a **hard data cap** *(a strict upper limit on the number of samples used per class each epoch)* of **3,000 images per class** during loading (`group.sample(min(len(group), 3000), random_state=42)`). This shrinks the ~213k frames down to a **balanced ~21,000-image** training subset per **epoch** *(one complete pass of the training data through the network)* — fast epochs while keeping the rare classes (e.g. 2,009 Licking) fully represented.

#### 4. Phased Sequential Training (Six Phases):
To avoid destructive interference, components are trained in order rather than all at once:
* **Phase 1 — Encoder Initialization:** EfficientNet-B0 is loaded with **ImageNet-pretrained weights** *(weights already learned on the 1.2M-image ImageNet dataset; reusing them is called **transfer learning**)*.
* **Phase 2 — BCS Head:** Backbone frozen; CBAM + BCS head trained 30 epochs with the **Adam optimizer** *(Adaptive Moment Estimation, an optimizer that adapts the step size per weight)*, **learning rate** *(the step size for weight updates)* `1e-3`, with a **scheduler** *(a rule that lowers the learning rate over time)* halving it every 10 epochs.
* **Phase 3 — Behavior Head:** Backbone frozen; trained 30 epochs with Focal Loss on capped MmCows; **early stopping** with patience 10.
* **Phase 4 — Lameness Sequence:** Backbone frozen; YOLOv8-Nano crops extracted; LSTM + head trained on 20-frame sequences for 15 epochs.
* **Phase 5 — Cow ID:** Backbone frozen; ID head trained on OpenCows2020 for 10 epochs.
* **Phase 6 — Multi-Task Joint Optimization:** All heads attached; **backbone unfrozen** and trained end-to-end with the weighted multi-task loss (0.35/0.35/0.15/0.15). This lets the backbone co-adapt to all tasks (but also introduces negative transfer — see Part 9/Part 10).

#### 5. Hardware & Compute Optimizations:
* **Automatic Mixed Precision (AMP):** *(a training technique that dynamically uses both 16-bit and 32-bit floating-point numbers to accelerate training and reduce memory)* via PyTorch `torch.amp.autocast` and `GradScaler` *(a PyTorch utility that prevents underflow by scaling gradients when using 16-bit precision)*. **FP32** *(32-bit float, 4 bytes/number)* vs **FP16** *(16-bit float, 2 bytes/number)*: using FP16 halves VRAM, letting us double the batch size and fully use the **tensor cores** *(specialized GPU units that accelerate matrix multiplication)* of the RTX 4080.
* **Early Stopping:** *(halting training when a monitored validation metric stops improving, to prevent overfitting)*. With a **patience** *(the number of epochs allowed without improvement before stopping)* of 10.
* **Deadlock Prevention:** We run `cv2.setNumThreads(0)` to disable OpenCV multithreading, preventing CPU **deadlocks** *(a freeze where processes block each other forever)* when combined with PyTorch **DataLoader workers** *(background CPU threads that load and preprocess batches in parallel)* on Windows.

---

### Part 9: Training Extent, Current Progress & Limitations

#### How Much Training is Done? (Status as of June 9, 2026)
The multi-phase sequential training plan *(training individual components of a complex model in separate, ordered stages)* current status:
1. **BCS Baseline:** ✅ COMPLETE — All 5 base models trained on Dryad + ScienceDB.
2. **Behavior Baseline:** ✅ COMPLETE — All 5 base models trained. Cross-dataset eval on CBVD-5 also done (Macro F1: 0.1245 — domain shift confirmed).
3. **Lameness Baseline (ALL 5 members):** ✅ COMPLETE — All 5 trained on YOLO-cropped CattleLameness.
   - Best spatial: EfficientNet-B0 (93.05% Acc, 0.9829 AUC) *(but inflated by leakage on 50 videos)*.
4. **ID Baseline (ALL 5 members):** ✅ COMPLETE — All 5 trained on OpenCows2020 (46 classes).
   - Best: EfficientNet-B0 (86.49%) | Worst: ResNet-18 (45.56%).
5. **Multi-Task Static (Frame-Level) Model:** ✅ COMPLETE — `train_multitask.py`.
   - Results: BCS MAE 0.7266, Behavior F1 0.3771, Lameness Acc 95.28% (AUC 0.9921), ID Acc 94.96%.
6. **Spatiotemporal Multi-Task Model:** ✅ COMPLETE — `train_multitask_temporal.py`.
   - Results: BCS MAE 0.7827, Behavior F1 0.4948, Lameness Acc **80.00% uncalibrated** (AUC 1.0000 — *perfect-separation result invalid for final eval due to test-set leakage*), ID Acc 97.58%.
7. **Ablation Studies:** ✅ COMPLETE — ScienceDB Cross-Entropy (MAE 0.6940 vs CORAL 0.5566); Dryad No-CBAM (MAE 0.7000 vs CBAM 0.6175); Behavior Cross-Entropy (F1 0.7074 vs Focal 0.7445); CBVD-5 cross-dataset (F1 0.1245).

#### Core Limitations of the Current Approach (honest baseline framing):
1. **Severe Negative Transfer:** Joint training degraded the spatial tasks — BCS MAE rose ~41% (0.5566 → 0.7827) and Behavior F1 dropped ~34–36% (0.7445 → 0.4948). The current joint model is presented as an **honest baseline characterizing negative transfer**, not a deployment-ready system.
2. **Tiny Lameness Data:** Only 50 videos → high variance; results are preliminary.
3. **Post-hoc Threshold Result Invalid:** The perfect lameness accuracy used test-set feedback (leakage).
4. **Narrow BCS Range:** ScienceDB only covers 3.25–4.25, so the model is not validated on very thin or obese cattle.
5. **Domain Shift:** CBVD-5 shows behavior recognition does not yet generalize across farms.
6. **No Variance Estimates:** All metrics are single-run point estimates (no mean ± standard deviation).
7. **Edge Numbers are Estimates:** Edge feasibility is computed from model size/FLOPs, not measured on a real Jetson/Raspberry Pi.

#### Future Work & Development Ideas:
1. **Joint Fine-Tuning:** ✅ Already implemented (Phase 6 unfreezes the backbone).
2. **Cross-Dataset Validation (CBVD-5):** ✅ COMPLETE — Macro F1 0.1245; generalizes to Standing (90.34%) but fails on Feeding (8.39%), Drinking (2.99%), Lying (24.31%).
3. **Gradient Alignment (Future):** **GradNorm** *(an algorithm that rescales each task's gradient so no task dominates)* or **PCGrad** *(Projecting Conflicting Gradients — projects one task's gradient so it stops opposing another's)* to mitigate negative transfer.
4. **Advanced Enhancements (Future):**
   * **CLAHE (Contrast Limited Adaptive Histogram Equalization)** *(a localized image-processing technique that improves contrast under uneven barn lighting)*.
   * **Federated learning** *(training across many farms without centralizing their raw data)* for cross-farm generalization.
   * **Synthetic data via 3D simulation** *(generating perfectly-labeled virtual cow videos in a physics/game engine like Unreal Engine to overcome the 50-video bottleneck)*.
   * Depth/thermal imaging for finer lameness inflammation detection.

---

### Part 10: Ablation Studies
*(Systematically removing or replacing parts of our system to prove that each design choice actually improves performance)*

Ablation studies *(experiments where specific components of a model are systematically removed or replaced to measure their individual impact)* isolate the value of each key design decision:

#### 1. CORAL vs. Standard Cross-Entropy (For BCS)
Same EfficientNet-B0 + CBAM on ScienceDB. CORAL → MAE **0.5566** (Exact 55.06%, ±1 90.50%); Cross-Entropy → MAE **0.6940** (Exact 46.31%, ±1 86.62%). **19.8% MAE reduction**, confirming that encoding ordinal structure helps.

#### 2. With CBAM vs. Without CBAM (For BCS)
On Dryad: without CBAM → MAE **0.7000**; with CBAM → MAE **0.6175**. Spatial attention filters background noise and focuses on spine/hooks/pins, improving the regression.

#### 3. Cross-Dataset Modality Comparison (RGB vs. RGB+Depth)
*(Presented as a cross-dataset observation, NOT a controlled ablation, because the datasets differ in size, breed, and environment.)* EfficientNet-B0 on RGB ScienceDB → MAE 0.5566; on DGE Dryad → MAE 0.6175. A shallower ResNet-18 on the same DGE input → 0.8675, showing backbone capacity matters for depth shapes, and that depth contours help offset Dryad's much smaller size (5,923 vs 53,566 images).

#### 4. Focal Loss vs. Standard Cross-Entropy (For Behavior)
On imbalanced MmCows: Cross-Entropy → Test Macro F1 **0.7074** (favors majority classes); Focal Loss (γ=2, α=0.25) → **0.7445**, by forcing the model to learn rare behaviors.

#### 5. Cross-Dataset Evaluation (MmCows → CBVD-5)
Tests **out-of-distribution generalization** *(performance on data from a different distribution — different farm, lighting, camera)*. MmCows-trained EfficientNet-B0 on unseen CBVD-5 (2,000 images) → Macro F1 **0.1245**. Generalizes to Standing (90.34%) but collapses on Feeding head down (8.39%), Drinking (2.99%), Lying (24.31%) — proving the model overfit to MmCows' specific angles/lighting (**domain shift**).

#### 6. Backbone Selection Comparison
5 architectures compared (ResNet-18, MobileNetV3-Small, ResNet-50, DenseNet121, EfficientNet-B0). EfficientNet-B0 wins: lowest BCS MAE (0.5566), highest Behavior F1 (0.7445), high lameness AUC, while staying at 5.3M parameters.

#### 7. Single-Task vs. Multi-Task Learning
Sharing one backbone forces general features (animal contours) that help all tasks. The spatiotemporal multi-task model achieved perfect-but-uncalibrated lameness sequence results and improved ID (97.58%, via majority voting). BUT due to negative transfer the spatial tasks degraded (BCS MAE → 0.7827, Behavior F1 → 0.4948). **The core trade-off: ~75% memory/latency savings and strong temporal performance, at the cost of degraded static spatial accuracy.**

---

### Part 11: Data Leakage Prevention via Task-Specific Splitting
*(Enforcing strict partition boundaries between training and testing subjects to prevent over-optimistic results)*

**Data Leakage** *(when training data contains information the model would not have in real deployment, leading to over-optimistic results but poor generalization)* is a major concern when many images come from the same subject.

#### The Risk: Identity Leakage
If many images of Cow #1042 (same camera, lighting, background) are randomly split, some land in train and some in test/validation.
* **The Result:** **Identity leakage** *(a form of data leakage where the model memorizes the subject's appearance rather than the task-specific features)*. The model memorizes Cow #1042's ear tag, markings, or horn shape to output a BCS score, instead of learning general body-fat markers — giving 99% validation accuracy but failing on new cows.

#### Our Solution: Splitting is TASK-SPECIFIC (an important nuance)
We do **not** use one universal split. The split method is chosen per task:
1. **BCS & Behavior → Cow-Wise Group Split:** *(all images of a given cow are kept entirely in one split — train, val, OR test — never shared)*. This acts as a **GroupKFold** *(a cross-validation scheme where the same group/subject never appears in two folds)*. It guarantees the test set contains only **out-of-distribution** *(from a different distribution than training)* cows the backbone has never seen.
2. **Lameness → Video-Wise Split:** no video is shared across train/val/test (so the same walking clip can't leak).
3. **Cow ID → Image-Wise Split *within each identity*:** here the task *is* to recognize known individuals, so **every cow appears in train, val, and test**, but no single image is shared across them. (Using a cow-wise split would make ID impossible, because the test cows would never have been seen.)

##### Why this distinction matters:
For BCS/Behavior we want the model to generalize to **new cows**, so cows are held out. For ID we want the model to re-recognize **known cows**, so identities must be present in all splits but with different photos. Matching the split strategy to the task's real-world goal is what makes the evaluation honest.

---

### Part 12: Hardware, Software & Edge Deployment Specifications
*(What the system was built on and what it needs to run live on a farm)*

#### Training Workstation:
* **GPU:** NVIDIA GeForce RTX 4080 (16GB VRAM, Ada Lovelace architecture, supports FP8/FP16 tensor ops).
* **CPU:** Intel Core i9-13900K. **RAM:** 64GB DDR5. **Storage:** 2TB NVMe SSD.

#### Software Stack:
* **OS:** Windows 11 / Ubuntu 22.04. **Language:** Python 3.12.3. **Framework:** PyTorch 2.5.1 + CUDA 12.1.
* **timm** *(PyTorch Image Models — a library of ready-made pre-trained vision backbones)* for loading EfficientNet-B0/MobileNetV3.
* **OpenCV 4.9** (video processing), **Albumentations 1.4** (augmentation), Pandas/NumPy/Scikit-learn (splits & metrics).

#### Edge Deployment Target:
* Runs on resource-constrained edge boxes like the **Jetson Orin Nano** *(a small, low-power NVIDIA AI computer for the edge; ~40 TOPS)* or Raspberry Pi 5.
* Model stays **under 10M parameters** (< 40MB file in FP16).
* **Compute cost:** backbone is ~0.39 **GFLOPs** *(Giga Floating-Point Operations — billions of math operations per frame; a measure of compute load)* per frame; a full 20-frame sequence is ~7.8 GFLOPs — well within the Jetson's **40 TOPS** *(Tera Operations Per Second — trillions of operations per second the chip can perform)*.
* *Caveat: these edge figures are estimated from model size/FLOPs, not measured on real hardware yet.*

---

### Part 13: Standards, Impact & Economic Analysis
*(The non-technical context the report covers: why this matters and what it costs)*

#### Standards Followed:
* **BCS:** the globally recognized **Elanco 5-point scale** *(the standard 1–5 body-condition chart, in 0.25 increments)* (ScienceDB), and integer scores (Dryad).
* **Lameness:** the **Sprecher locomotion scoring system** *(a standard 1–5 gait-scoring scheme used to classify cattle as sound or lame)*.
* **Software:** PEP 8 Python style; standard metric definitions (Macro F1, MAE, AUC-ROC).

#### Impacts:
* **Societal:** earlier disease detection, support for **Precision Livestock Farming (PLF)** *(using sensors/AI to monitor individual animals continuously instead of manual herd checks)*, and relief for farm labor shortages.
* **Environmental:** "green computing" — one shared 5.3M backbone instead of four cuts duplicated computation and energy (a 75% parameter reduction); sparse sampling further reduces training GPU time.
* **Ethical:** **non-invasive** vision monitoring avoids painful **RFID** *(Radio-Frequency Identification — electronic ear tags read wirelessly)* tags/branding; local **edge processing** keeps farm data private (no cloud upload).

#### Economic Cost-Benefit (vs. traditional sensor systems):
* **Traditional:** wearable collars/RFID at $80–120 per cow → **$40,000+** for a 500-cow herd, plus ~15% annual tag/collar loss and cloud fees.
* **Proposed MTL edge system:** standard IP cameras (~$150 each) + one edge unit (~$300) ≈ **$1,500 total**, offline, near-zero recurring cost.
* Net effect: lowers the entry investment for small/medium farms by **over 90%**.

#### Team Roles (Project Management):
* **Hasin Ishrak:** dataset preprocessing, baseline training pipelines, ResNet-18 baseline, and the CBAM and RGB-vs-Depth ablations.
* **Namira Abrar Haque:** MobileNetV3-Small baseline (compressed-network edge study).
* **Sanjida Akter Bithi:** ResNet-50 baseline (deeper-architecture study).
* **Shouvik Banik:** DenseNet121 baseline (dense connectivity for behavior).
* **Nusrat Lamya Faruk:** winning EfficientNet-B0 baseline, the final spatiotemporal multi-task model, LSTM integration, and the real-time visualizer.

---

### Part 14: Negative Transfer & State-of-the-Art Context
*(The central scientific finding, and how our numbers sit beside published work)*

#### Negative Transfer (the headline finding):
When all four tasks share one EfficientNet-B0 backbone, the gradient updates for **temporal** tasks (Lameness, Behavior) conflict with those for **spatial** tasks (BCS, ID). BCS needs precise anatomical contours (spine, hooks, pins); the Lameness LSTM needs dynamic motion trajectories. Forcing both into one backbone degrades both — a textbook case of **negative transfer** *(when sharing parameters makes tasks hurt each other)* / **gradient conflict** *(when two tasks' weight-update directions point opposite ways)*. Result: ~41% BCS MAE increase, ~36% Behavior F1 drop. The fix (future work) is gradient decoupling via GradNorm/PCGrad.

#### Comparison with Published Work (contextual only — NOT a ranking):
These studies use different datasets, modalities, and metrics, so they are **not directly comparable**; the table only positions the scale of results.

| Task | Reference | Their Setup | Their Result | Our Best |
|------|-----------|-------------|--------------|----------|
| BCS | Yao et al. (2026) | Different dataset/metric | N/A (non-equivalent) | 90.50% ±0.25 Acc / 0.5566 MAE |
| Behavior | Asim et al. (2026) | Sensor data | 91.11% Acc | 0.7445 Macro F1 (vision, severe imbalance) |
| Lameness | Sohan et al. (2025) | RGB-D, different dataset | N/A (non-equivalent) | 80.00% uncalibrated sequence Acc |

Key framing: our entire framework uses **only public datasets** with documented preprocessing and task-specific splits — a transparency advantage over the ~88% of prior cattle studies (29 of 33 reviewed) that rely on private data. The core value proposition is **computational efficiency** (one 5.3M backbone, ~75% memory savings), with negative transfer identified as the clear next research target.
