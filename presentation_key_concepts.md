# Thesis: Multi-Task Deep Learning Framework for Unified Cattle Health and Behavior Monitoring
## Comprehensive Pre-Thesis II Presentation Study Guide

---

This document serves as the **definitive, exhaustive study guide** for the Thesis presentation. It contains highly detailed explanations of every major architectural decision, mathematical concept, experimental result, and future plan discussed during the project. It uses simple English and actual, concrete dataset and model examples to make these complex topics easy to explain. Every technical term is immediately defined in brackets for absolute clarity.

---

### Part 1: The Core Architecture - Multi-Task Learning (MTL) Framework
*(Solving multiple distinct problems using a single unified AI model instead of building many separate ones)*

#### The Problem with Traditional Approaches:
In a standard computer vision pipeline *(the step-by-step process of feeding an image from a camera into an AI to get a prediction)*, monitoring a dairy farm would require deploying **four completely separate Convolutional Neural Networks (CNNs)** *(deep learning algorithms specifically engineered to recognize shapes, colors, and edges in photos)*:
1. One network to predict Body Condition Score (BCS) *(a standardized visual grading system from 1 to 5 used to estimate a cow's body fat reserves)*.
2. One network to classify Behavior *(lying, standing, drinking, licking)*.
3. One network to detect Lameness *(limping)*.
4. One network to identify the specific Cow (ID) *(ear tag/marking recognition)*.

##### Real-World Example of the Problem:
If you deploy 50 cameras on a farm, running 4 separate models on each camera causes:
* **Massive Memory Overhead:** Running 4 models simultaneously requires 4 times the GPU memory (VRAM) *(Video Random Access Memory, the temporary storage space on a graphics card used to hold the AI's math operations)* and RAM *(Random Access Memory, the computer's primary temporary workspace)*. 
* **High Inference Latency:** Processing a single frame takes 4 times longer because the computer has to run 4 separate forward passes *(the mathematical process of pushing input data forward through the neural network layers to generate a prediction)* one after another.
* **Hardware Constraints:** You cannot run 4 heavy models on cheap, low-power **edge devices** *(small, affordable computers like a $50 Raspberry Pi installed directly on the farm camera, rather than in an expensive cloud server)*.
* **Missing Synergy:** BCS and Cow ID both rely on the shape of the cow's back and its skin patterns. In 4 separate models, Model 1 and Model 4 waste computation learning the exact same basic edge and shape detection math independently.

#### The Solution: Hard Parameter Sharing (Our Approach)
Instead of 4 disconnected models, our project utilizes **Hard Parameter Sharing** *(forcing different AI tasks to share the exact same foundational layers)*.

##### Actual Technical Example:
We feed an image of a cow into a single shared backbone network *(a pre-trained neural network module, like EfficientNetB0, that acts as the core feature extractor or "eyes" of the system)*. 
* **The Shared Backbone:** The backbone processes the raw pixels and extracts a universal **feature vector** *(a compressed list of 1,280 numbers summarizing the cow's shapes, curves, and patterns)*.
* **Specialized Heads:** This exact same 1280-length feature vector is passed simultaneously to four separate, lightweight prediction heads *(small neural network layers attached to the end of the backbone that perform the final classification or regression for each task)*:
  1. Head 1 uses it to output a BCS score.
  2. Head 2 uses it to output a Behavior classification.
  3. Head 3 uses it to output a Lameness prediction.
  4. Head 4 uses it to output a Cow ID.

##### Benefits for the Thesis:
1. **Extreme Efficiency:** The entire model remains under **10 million parameters** *(the individual weight variables adjusted during learning)*. It can run in real-time on edge cameras without cloud latency.
2. **Regularization (Preventing Overfitting):** Overfitting *(when an AI memorizes specific training details like the background grass rather than learning generalizable concepts)* is prevented. Because the shared backbone must satisfy 4 entirely different tasks at the same time, it cannot "cheat" by memorizing irrelevant details. It is forced to learn highly robust, general representations of the cow itself.

#### The Multi-Task Loss Function & Weighting
To train a model to do 4 things at once, we calculate the total error (loss) across all tasks simultaneously using a weighted sum of individual loss functions *(mathematical formulas that calculate how wrong the model's predictions are compared to the actual truth)*:

`Total Loss = (w1 * Loss_BCS) + (w2 * Loss_Behavior) + (w3 * Loss_Lameness) + (w4 * Loss_ID)`

The parameters `w1, w2, w3, w4` are **loss weights** *(multipliers that control how much weight each task's error contributes to the total loss, determining its influence on learning)*. 

##### Actual Technical Example:
During training, if the Behavior task has a very high loss of 2.5, and the Lameness task has a low loss of 0.1, the network will focus almost entirely on adjusting its weights to minimize the Behavior loss. This is called task domination or destructive interference *(when one task's large error updates override and erase the progress made on other tasks)*. By scaling the loss weights (e.g., `w1=0.35, w2=0.35, w3=0.15, w4=0.15`), we force the optimization gradients *(the directional math values indicating how to adjust the model's weights to reduce the error)* to have similar magnitudes, ensuring the network learns both tasks simultaneously. This is handled by the optimizer *(the algorithm, like Adam or SGD, that uses gradients to update the model's weights during training)*.

#### Backbone Selection Process
To find the perfect "shared brain," the 5 group members individually trained single-task baseline models *(simple, single-task models trained to establish a performance benchmark before building the multi-task model)* on 5 different architectures *(the specific structure and arrangement of layers in a neural network)*:
* **ResNet-18** (Hasin)
* **MobileNetV3-Small** (Namira)
* **ResNet-50** (Bithi)
* **DenseNet121** (Shouvik)
* **EfficientNetB0** (Nusrat)
The group evaluated these backbones to select the one with the lowest combined error (lowest BCS error + highest Behavior F1-score) to serve as the foundation of the Multi-Task model.

#### Architectural Enhancements: CBAM (Convolutional Block Attention Module)
Between the backbone and the heads, we inject a **CBAM** module *(a visual attention mechanism that guides the network to focus on relevant features and locations)*.
* **Channel Attention (The "WHAT"):** Focuses on *what* features are important *(e.g., teaching the model that bone angularity is more important than the green color of the grass)*.
* **Spatial Attention (The "WHERE"):** Focuses on *where* to look. It acts as a spotlight, mathematically forcing the model to ignore the background farm structures and focus heavily on the cow's spine, hooks, and pins *(the specific hip bones used to grade fat levels)*.

---

### Part 2: Task-Specific Head I - BCS & Ordinal Regression
*(Predicting categories that have a strict, natural mathematical order, like ranking sizes from Small to Large)*

#### The Flaws in Standard Approaches:
Body Condition Scoring (BCS) operates on a fixed scale (e.g., `3.25, 3.5, 3.75, 4.0, 4.25`).
1. **Standard Classification Flaw:** Standard classification *(treating categories as completely independent, unrelated buckets)* treats classes as completely unrelated. If a cow's true BCS is `3.5`, standard classification penalizes the AI the exact same amount whether it guesses `3.75` (a minor error of 0.25) or `4.25` (a severe error of 0.75). Both guesses get a 0% accuracy score.
2. **Standard Regression Flaw:** Standard regression *(predicting continuous numerical values along a continuous scale)* treats the scale like a continuous line. The AI might output `3.642`, forcing the user to arbitrarily round it. It also lacks confidence scores for each class.

#### The CORAL Framework (Consistent Rank Logits)
CORAL solves this by converting the ranking problem into a series of **binary (Yes/No) questions** outputted by output nodes *(the final units in a neural network layer that output a prediction probability)*.
If there are 5 possible BCS classes, the network has 4 output nodes:
* **Node 1:** Is the score > Class 0 (3.25)? (Yes/No)
* **Node 2:** Is the score > Class 1 (3.5)? (Yes/No)
* **Node 3:** Is the score > Class 2 (3.75)? (Yes/No)
* **Node 4:** Is the score > Class 3 (4.0)? (Yes/No)

##### Actual Technical Example:
Imagine a cow walks by and the model outputs the following probabilities for the 4 nodes:
* Node 1 (>3.25): **95%** (above 50% threshold *(the cutoff value used to convert a continuous probability into a binary Yes/No decision)* $\rightarrow$ **1**)
* Node 2 (>3.50): **80%** (above 50% threshold $\rightarrow$ **1**)
* Node 3 (>3.75): **15%** (below 50% threshold $\rightarrow$ **0**)
* Node 4 (>4.00): **2%**  (below 50% threshold $\rightarrow$ **0**)

We get the binary array *(a list containing only 0s and 1s representing binary decisions)* `[1, 1, 0, 0]`. We sum these numbers: `1 + 1 + 0 + 0 = 2`. The predicted class index *(the integer position of an class in a list, starting at 0)* is **2**, which corresponds to a score of **3.75**.

##### Why this is revolutionary for our model:
CORAL mathematically guarantees that predicting a `3.75` when the truth is `3.5` results in a tiny loss penalty, while predicting `4.25` results in a massive penalty. This dramatically drives down our **MAE (Mean Absolute Error)** *(the average absolute difference between the predicted value and the actual true score)*.

---

### Part 3: Task-Specific Head II - Behavior & Focal Loss
*(Solving severe dataset imbalance where one category has 40,000 photos and another has only 200, without deleting the extra data)*

#### The Imbalance Problem (MmCows Dataset)
Cow behavior is naturally skewed. A cow spends 80% of its day just *Lying* or *Standing* (resulting in tens of thousands of images), but spends very little time *Drinking* or *Licking* (resulting in a few hundred images).

##### Actual Technical Example:
If the dataset has 40,000 images of "Lying" cows and only 200 images of "Licking" cows, a model using standard Cross-Entropy loss *(the standard loss function used for classification tasks to measure the difference between predicted probability distributions and target classes)* will learn that it can get 99.5% accuracy by always outputting "Lying", even if it gets every single "Licking" image wrong. The AI stops learning how to detect rare behaviors due to class imbalance *(a dataset state where some categories have vastly more samples than others)*.

#### The Solution: Focal Loss
Focal Loss *(an altered loss function that dynamically scales the loss based on prediction confidence to focus training on hard, rare examples)* dynamically alters the penalty the AI receives based on *how confident and correct* it is.
* **Mathematical Formula Modulation:** We multiply the standard loss by a modulating factor: `(1 - p_t)^gamma`.
* **Gamma (gamma = 2) - The Focusing Parameter:** Gamma *(the focusing parameter in Focal Loss that controls how heavily easy-to-classify examples are down-weighted)* scales down the loss for easy images. If the AI is 95% confident and correct about an easy "Lying" image, the equation forces the loss to drop to virtually zero. The model does not adjust its weights for this image.
* If the AI is confused by a rare "Licking" image (confidence is only 10%), the loss remains high, forcing the optimizer to update the weights specifically to learn the "Licking" class.
* **Alpha (alpha = 0.25) - The Balancing Parameter:** Alpha *(the balancing parameter in Focal Loss that scales class loss to offset numerical class imbalances)* statically scales down the importance of majority classes to offset their raw numerical advantage.

##### Impact:
Without Focal Loss, the **Macro F1-Score** *(a performance metric that calculates the average F1-score across all classes individually, giving equal weight to each class regardless of its size)* would be extremely low. Focal Loss forces the AI to be equally good at classifying rare behaviors as it is at common behaviors.

---

### Part 4: Task-Specific Head III - Lameness & The Hybrid Spatiotemporal Model
*(Fusing image analysis with time-series memory to understand animal motion over time)*

#### The Problem: Amnesia in 2D Models
Standard 2D image models *(networks that analyze single, static images without any time-series awareness)* suffer from amnesia. They look at a single photo, make a prediction, and instantly forget it before looking at the next one.

##### Actual Technical Example:
In a single frame (Frame 15), a cow's leg might be lifted off the ground, which looks exactly the same whether the cow is walking normally or limping. To detect lameness, the model must compare the time delay and angle difference of that leg landing in Frame 15 compared to Frame 5 and Frame 25. This requires stride mechanics *(the physical kinematics of walking, such as stride length, joint flexion, and contact duration)* and gait asymmetry *(unequal walking movements between left and right limbs, a primary indicator of limping)* tracking over time.

#### The Core Architectural Contribution: The Hybrid Spatiotemporal Model (CNN-LSTM)
We built a spatiotemporal model *(an architecture that processes both spatial details from images and temporal changes over time)* that fuses spatial feature extraction with sequential time-series memory *(the capacity to store and sequence information across consecutive points in time)*.
1. **The Spatial Component (The 2D Shared Backbone):** We feed 20 video frames into the EfficientNetB0 backbone one by one. The backbone is completely frozen *(setting the model weights to be non-trainable so they do not change during backpropagation)* and compresses each frame into a high-dimensional feature vector.
2. **The Temporal Component (The LSTM Module):** The sequence of 20 feature vectors is fed into a **Long Short-Term Memory (LSTM)** *(a specialized recurrent neural network designed to track and process sequential data over time without losing memory)* network inside the Lameness head.

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
* **OOM (Out of Memory):** Processing 300 frames simultaneously requires massive VRAM, causing OOM *(a hardware error where the GPU runs out of physical VRAM during training calculations)* crashes when we try to backpropagate *(the mathematical process of computing gradients backward from the loss through the network to update weights)*.

#### Sparse Temporal Sampling (Our Implementation) - The Winning Strategy:
Instead of all frames, we divide every video (regardless of its total duration) into **20 equal temporal segments** *(equal intervals of time split across the total duration of a video)* and extract exactly **1 frame** from the center of each segment.

##### Actual Technical Example:
If a video has 200 frames:
* Segment 1 is frames 1-10 $\rightarrow$ Extract frame 5.
* Segment 2 is frames 11-20 $\rightarrow$ Extract frame 15.
* ...and so on, resulting in exactly 20 evenly-spaced frames.

##### Benefits:
* **Standardized Shape:** Every video is processed as a fixed tensor *(a multi-dimensional math array used by deep learning frameworks to represent data)* of size `(20, 3, 224, 224)` (20 frames, 3 color channels *(the individual color layers, typically Red, Green, and Blue, that make up a digital image)*, 224x224 pixels *(the tiny individual colored dots that make up a digital image)*).
* **Redundancy Elimination:** It captures the "macro-movements" of the stride *(lifting the hoof, leg swing, landing impact)* while skipping redundant static frames.
* **Hardware Friendly:** Reduces compute overhead *(the amount of CPU/GPU processing power and time required to execute a program)* by 15x, allowing fast training within VRAM limits.

---

### Part 6: AUC vs. Accuracy & Threshold Calibration
*(Why a model can be perfect at ranking but fail at base accuracy, and how to fix it)*

During our spatiotemporal lameness experiments, our model achieved a **Test AUC of 0.96**, but initially showed only **60% Test Accuracy**.

#### Understanding AUC (Area Under the ROC Curve):
AUC *(Area Under the Receiver Operating Characteristic Curve, a metric measuring a model's ability to rank positive cases higher than negative cases across all thresholds)* is a **threshold-independent metric** *(a grading metric that measures the model's core discriminative ability without relying on a specific decision cutoff)* based on the ROC curve *(Receiver Operating Characteristic curve, a graph plotting the True Positive Rate against the False Positive Rate at various thresholds)*.
* An AUC of 0.96 means that if you pick one random lame cow and one random healthy cow, there is a 96% chance the model will assign a higher lameness probability to the lame cow. The model's internal understanding of lameness is exceptional.

#### The Accuracy Discrepancy & Threshold Calibration:
Accuracy is dependent on an arbitrary decision threshold *(the probability cutoff value used to assign a sample to a class)*, which defaults to `0.50` (50%).

##### Actual Technical Example:
Because the lameness training set was small, the model's probability outputs shifted higher. It was outputting a `0.55` (55%) probability for perfectly healthy cows and `0.85` (85%) for lame cows.
* Because `0.55 > 0.50`, the default accuracy metric classified the healthy cows as Lame, generating False Positives *(healthy cases incorrectly flagged by the model as diseased or abnormal)* and dragging the accuracy score down to 60%.
* Shifting the decision threshold from `0.50` to `0.70` (so only scores > 0.70 are classified as Lame) correctly separates the classes. The test accuracy immediately jumped to 90%+.

---

### Part 7: Live Deployed Video Inference Architecture
*(How the software actually operates in real-time on a physical dairy farm)*

Imagine Deployed Video Inference *(running a trained model in a live production environment to generate predictions on new video feeds)* where a camera is mounted above a farm walkway. A cow walks past, and the camera captures a 20-frame video clip.

1. **Feature Extraction:** The 20 frames are passed through the shared `EfficientNetB0` backbone, generating 20 independent feature vectors.
2. **Spatial Heads (BCS & ID) Process the Data:**
   * These heads are designed for static images. They generate 20 independent BCS predictions and 20 independent ID predictions (one for each frame).
   * **Temporal Averaging (For BCS):** We use temporal averaging *(calculating the mathematical mean of predictions made across multiple frames over time)* to average all 20 predicted scores to output a single, stable body score.
   * **Majority Voting (For ID):** We use majority voting *(selecting the classification class that was predicted most frequently across a sequence of frames)* to count which Cow ID was predicted most often across the 20 frames and output that as the final ID.
   * **Occlusion Example:** If the cow walks behind a fence post and suffers from occlusion *(the state of being visually blocked or hidden from the camera's view by another object)* for 3 frames, the spatial heads might predict incorrect values for those 3 frames. Temporal averaging and majority voting filter out that noise completely.
3. **Temporal Heads (Behavior & Lameness) Process the Data:**
   * The 20 feature vectors are fed sequentially into the LSTM networks.
   * The LSTM tracks the gait mechanics frame-by-frame and outputs the final classification on the 20th frame: **Behavior: Walking** and **Lameness: Positive**.

---

### Part 8: Codebase & Implementation Deep Dive
*(Critical data pipeline and hardware optimization strategies)*

#### 1. Dataset Anomalies & Handling:
We resolve dataset anomalies *(inconsistencies, missing values, or format mismatches present in raw data)* during preprocessing:
* **BCS Datasets:** We train on two completely different datasets:
  * **Dryad:** 5,923 images using **Depth Grayscale Edge (DGE)** format *(an image format where pixel intensities represent distance from the sensor rather than visible color, combined with edge enhancement)* instead of standard RGB *(Red, Green, Blue, the standard color channel format for digital images)*. It uses a label scale of 2 to 6.
  * **ScienceDB:** 53,566 images in standard RGB, using a scale of 3.25 to 4.25.
  * **The Fix:** Both datasets are dynamically mapped to a unified `0-4` ordinal index during data loading so the CORAL framework can process them interchangeably without crashing.
* **Behavior Dataset (MmCows):** This dataset has a massive 42:1 class imbalance. While Focal Loss fixes the loss gradients, we implement a **hard data cap** *(setting a strict upper limit on the number of samples processed per class during an epoch)* during loading (`group.sample(min(len(group), 3000), random_state=42)`). This restricts majority classes (like Lying) to a maximum of 3,000 images per epoch *(one complete pass of the entire training dataset through the neural network)*.

##### Actual Technical Example:
Instead of processing 40,000 identical frames of lying cows (which adds redundant training time and memory overhead), the model trains on a balanced subset of 3,000 lying images per epoch, allowing faster epochs while retaining the representation of the rare behaviors (200 samples).

#### 2. Hardware & Compute Optimizations:
* **Automatic Mixed Precision (AMP):** We use Automatic Mixed Precision (AMP) *(a training technique that dynamically uses both 16-bit and 32-bit floating-point numbers to accelerate training and reduce memory)* via PyTorch `torch.amp.autocast` and `GradScaler` *(a PyTorch utility that prevents underflow by scaling gradients when using 16-bit precision)*.

##### Actual Technical Example:
Standard FP32 *(32-bit floating-point format, using 4 bytes of memory per number)* uses 4 bytes per weight. FP16 *(16-bit floating-point format, using 2 bytes of memory per number)* uses 2 bytes. By using FP16, we halve the GPU VRAM requirement, allowing us to double the batch size from 32 to 64 to fully utilize the parallel tensor cores *(specialized hardware units on modern GPUs designed to accelerate matrix multiplications)* of the RTX 4080 GPU.
* **Early Stopping:** We use Early Stopping *(halting training when a monitored validation metric stops improving to prevent overfitting)*. If the Validation Macro F1 score stops improving for a patience *(the number of epochs the model is allowed to train without improvement before early stopping triggers)* of 10 consecutive epochs, training halts.
* **Deadlock Prevention:** We explicitly run `cv2.setNumThreads(0)` to disable OpenCV multithreading *(running multiple processing threads concurrently to speed up tasks like image decoding)*, which prevents CPU deadlocks *(a freeze state where multiple processes block each other endlessly while waiting for resources)* when combined with PyTorch DataLoader workers *(background CPU threads responsible for loading and preprocessing batches of data in parallel)* on Windows.

---

### Part 9: Training Extent, Current Progress & Limitations

#### How Much Training is Done?
The multi-phase sequential training plan *(training individual components of a complex model in separate, ordered stages)* is currently in execution:
1. **BCS Baseline:** [100% COMPLETE] All 5 base models trained. Results logged.
2. **Behavior Baseline:** [100% COMPLETE] All 5 base models trained using Focal Loss. Results logged.
3. **Lameness Baseline (Spatial vs Spatiotemporal):** [COMPLETE] Preliminary 2D models trained. The Spatiotemporal LSTM model was trained for 15 epochs on a 20-frame sampled sequence, achieving a 1.0 Validation AUC and 0.96 Test AUC.
4. **ID Baseline:** [PENDING] Dataset extraction required.
5. **Final Multi-Task Aggregation:** [PENDING] Will combine the winning backbone with all heads.

#### Core Limitations of the Current Approach:
1. **Lack of Large-Scale Annotated Temporal Data:** While we have 50,000+ spatial images for BCS, high-quality, annotated *(labeled with ground truth information by human experts like veterinarians)* sequence data for Lameness and Behavior is scarce. The spatiotemporal models risk overfitting due to the limited number of unique videos in the datasets.
2. **Frozen Backbone Bottleneck:** Currently, the 2D backbone is completely frozen when training the LSTM heads. This means the feature vectors passed to the LSTM were optimized for classifying static ImageNet photos (dogs, cars, etc.), not the specific micro-movements of a cow's joints.

#### Future Work & Development Ideas:
1. **Joint Fine-Tuning (Unfreezing the Backbone):** In the final phase, we intend to perform Fine-Tuning *(unfreezing pretrained layers and training them with a very small learning rate to adapt them to a new, specific task)* by unfreezing the shared backbone and allowing gradients from the LSTM to flow all the way backward into the CNN layers. This forces the backbone to learn new convolutional filters specifically designed to track motion and joint angles.
2. **Cross-Dataset Validation (CBVD-5):** To prove generalizability, we will perform Cross-Dataset Validation *(testing a model trained on one dataset on an entirely different, independent dataset to prove generalizability)* by taking the Behavior model trained on `MmCows` and running inference on the entirely unseen `CBVD-5` dataset.
3. **Advanced Enhancements:**
   * Integrating **CLAHE (Contrast Limited Adaptive Histogram Equalization)** *(a localized image processing technique used to improve contrast and details under uneven lighting)* to equalize extreme lighting differences in barns.
   * Future exploration into integrating depth cameras or thermal imaging for highly granular lameness inflammation detection.
