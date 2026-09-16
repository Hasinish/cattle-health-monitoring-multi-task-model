<style>
  body, .markdown-body {
    font-family: Cambria, Georgia, serif !important;
    color: black !important;
  }
</style>

### **🐄 CATTLE HEALTH MONITORING — THESIS Q&amp;A STUDY GUIDE 📋**

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q1: In simple terms, what is your thesis about?</b></span>

Our thesis is about creating a single ***multi-task deep learning model*** to monitor cow health on a farm. Instead of using four separate models, our model shares ***one shared backbone*** to perform four tasks at once: ***body condition scoring***, ***behavior classification***, ***lameness detection***, and ***cow identification***. This saves ***75% of computer memory***, allowing the model to run on cheap farm hardware.

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q2: What is Body Condition Scoring (BCS) and why is it important?</b></span>

***Body Condition Scoring (BCS)*** is a method used to check how much fat is on a cow's body. 

It is done by looking at the cow from behind to see how visible its bones are, specifically the ***spine, hooks, and pins*** (the hip bones). The score follows a ***five-point scale*** (with steps of 0.25, like 3.5 or 4.0), where a lower score means the cow is thin and a higher score means the cow is fat.

This is important to monitor because:
1. ***Low BCS (too thin):*** The cow has low energy reserves and produces ***low milk yield***.
2. ***High BCS (too fat):*** The cow faces ***calving difficulties*** (trouble giving birth) during labor.

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q3: Why do we need to monitor behavior and why is it important?</b></span>

Monitoring ***behavior*** (daily actions like eating, drinking, standing, and lying) is important because it tells us about the cow's health:

1. ***Sickness warning:*** If a cow is sick, it will stop eating or lie down for too long.
2. ***Breeding time:*** If a cow walks or stands much more than usual, it means they are in ***estrus*** (ready to breed).
3. ***Rest time:*** Cows will stand instead of lying down if they are in pain or suffering from ***heat stress*** (too hot). If they do not rest, their ***milk production*** drops.

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q4: What is lameness and why is it important to monitor?</b></span>

***Lameness*** is when a cow limps because of pain or injury in its hooves or legs. 

It is important to monitor because:
1. ***Drop in milk:*** Lame cows are in pain, so they eat less and produce less milk.
2. ***Animal welfare:*** It is highly painful, and early detection prevents suffering.
3. ***High costs:*** Catching it early requires simple treatment. Catching it late is extremely expensive and might force the farmer to replace the cow.

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q5: Why do we need individual cow identification and why is it important?</b></span>

***Cow identification*** (recognizing each individual cow) connects all tasks together. It is important because it tells the farmer ***which exact cow*** is sick and needs treatment, automatically tracks health records over time, and serves as a ***non-invasive*** replacement for physical ear tags that can easily fall out or get lost.

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q6: What is a backbone and does it exist in both single-task and multi-task models?</b></span>

A ***backbone*** is the visual core (the "eyes") of the AI model. It is the main part of the neural network that reads input images and extracts general shapes, outlines, and features. A backbone exists in ***both*** single-task and multi-task models: in a single-task model, it extracts features for just one specific task (like behavior only), whereas in a multi-task model, a single backbone is shared to extract features for all four tasks at the same time. These standard backbones (like EfficientNet-B0) are initialized with pretrained weights.

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q7: What is Multi-Task Learning (MTL)?</b></span>

***Multi-Task Learning (MTL)*** is a machine learning approach where a single ***neural network*** is trained to solve several different problems. 

An MTL model works by using a ***shared backbone***, which is the ***main core network*** that reads input images and extracts general visual features like edges, shapes, and outlines. 

The output from this shared core is then passed to separate ***task heads***, which are the small, final layers of the network designed to handle specific tasks like ***body condition scoring*** or ***behavior classification***. 

This allows the system to generate multiple ***final decisions*** at the same time using a fraction of the computer memory.

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q8: What is hard parameter sharing?</b></span>

***Hard parameter sharing*** is when multiple tasks share the exact same ***backbone*** (the core neural network), with separate ***task heads*** at the end.

This means the core weights are identical for all tasks, which drastically reduces the ***VRAM memory footprint*** and prevents the model from overfitting.

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q9: How does the model reduce VRAM and parameter usage by 75%?</b></span>

Running ***four separate models*** means loading and running four full networks. Our ***multi-task model*** combines them so they share one core network. This means instead of running four networks, we only run one. 

Saving three out of four models results in a ***75% reduction*** in memory. 

This cuts both the ***static model parameters*** (stored weights) and the ***image processing activations*** (temporary calculations) by exactly 75%.

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q10: How does your model balance the training of four different tasks at the same time?</b></span>

We combine the individual errors of all four tasks into a single ***total loss*** using this formula: 

Total Loss = (0.35 * Loss_BCS) + (0.35 * Loss_Behavior) + (0.15 * Loss_Lameness) + (0.15 * Loss_ID). 

We assign higher weights of 0.35 to ***body condition scoring*** and ***behavior classification*** because they are complex tasks with large datasets, and lower weights of 0.15 to ***lameness detection*** and ***cow identification***. This prevents the small datasets (like the 50 lameness videos) from dominating the learning process and causing the model to overfit.

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q11: What is your proposed plan/methodology?</b></span>

Our proposed methodology is a ***three-stage pipeline***:
1. First, we use a lightweight ***YOLOv8-Nano*** detector, which crops the cow and removes background noise.
2. Second, we extract visual features from the crop using a shared ***EfficientNet-B0 backbone*** and a ***CBAM attention module*** (which highlights key body parts).
3. Third, we branch these features into four ***task-specific heads***. Both the behavior and lameness heads use ***LSTM networks*** to analyze video sequences over time, while the body condition and cow identity heads perform static frame predictions.

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q12: What is regularization?</b></span>

***Regularization*** is a technique used to prevent a model from ***overfitting*** (memorizing training data instead of learning general patterns). 

In ***our*** model, ***Multi-Task Learning*** acts as a built-in regularizer. By forcing the shared core to learn features for four tasks at the same time, it cannot overfit to any single task, forcing it to learn more robust features.

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q13: Why did you choose EfficientNet-B0 as your shared core instead of the other options?</b></span>

We chose ***EfficientNet-B0*** based on our ***single-task baseline comparison***, where each team member trained a different backbone independently on all four tasks. EfficientNet-B0 achieved the ***best overall performance*** rank across all tasks, beating larger networks like ResNet-50 and DenseNet121. Additionally, it is extremely lightweight, containing only ***5.3 million parameters***, making it perfect for ***resource-constrained edge devices*** (like the Jetson Orin Nano).

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q14: How did your team divide the work of training and evaluating these backbone networks?</b></span>

Each of the five team members worked on their own assigned ***model backbone*** and dataset splits, running the training processes individually on their own workstations. We trained our assigned models on the tasks separately, resulting in individual baseline weights. Once all training runs were completed, we collected and compared everyone's results to select the best-performing model.

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q15: What was the reasoning behind choosing these specific five architectures for your baseline comparison?</b></span>

We chose five of the most popular and standard ***CNN base models*** used in computer vision. This allowed us to compare different well-known architectures and find the most accurate and efficient one for our tasks.

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q16: How did you perform the single-task baseline training?</b></span>

Each of the five team members worked on their own assigned ***model backbone*** and dataset splits, running the training processes individually on their own workstations. We trained our assigned models on the tasks separately, resulting in individual baseline weights. Once all training runs were completed, we collected and compared everyone's results to select the best-performing model.

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q17: How did you preprocess the datasets for training?</b></span>

We followed a consistent preprocessing pipeline across all datasets:
1. ***YOLO cropping:*** We ran a lightweight ***YOLOv8-Nano*** model to detect the cow in each frame and crop it, removing background farm noise.
2. ***Standardization:*** We resized all cropped images to ***224x224 pixels*** and normalized them using standard ImageNet parameters.
3. ***Data balancing (Capping):*** To handle class imbalance in behavior, we capped the frames to ***3,000 per class*** to prevent class bias.
4. ***Modality conversion:*** For the Dryad dataset, images were converted to ***DGE (Depth-Grayscale-Edge) maps*** to highlight contours.
5. ***Sequence sampling:*** For video clips (like lameness), we extracted exactly ***20 evenly spaced frames*** to create consistent video sequences and prevent GPU memory crashes.

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q18: What do the performance metrics used in our thesis mean in simple terms?</b></span>

These metrics measure our model's accuracy and efficiency:
* ***MAE (Mean Absolute Error):*** Tells us how far off the model's guesses are on average. Lower is better.
* ***±1 Accuracy (Primary BCS Metric):*** Checks if the model guessed within one step (0.25 points) of the real score (like guessing 3.25 or 3.75 for a true 3.50). We use this as our main metric because even human experts cannot tell the exact difference between 3.50 and 3.75 in real life, so being within one step is considered a perfect guess in farming.
* ***F1-score:*** Measures behavior accuracy fairly. It makes sure the model doesn't cheat by only predicting common behaviors (like lying) while ignoring rare ones (like drinking).
* ***AUC (Area Under the Curve):*** Measures how good the model is at separating healthy cows from lame cows. 1.0 is perfect, and 0.5 is random guessing.
* ***Top-1 Accuracy:*** The percentage of times the model's absolute best guess is the correct cow.
* ***GFLOPs:*** Tells us how many mathematical calculations the model does (in billions). Lower is better because the model runs faster on cheap edge hardware.

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q19: What is CBAM and why did you use it?</b></span>

***CBAM (Convolutional Block Attention Module)*** is an attention module that helps the model focus only on the cow and ignore background noise. It does this in two ways:
* ***Channel Attention (The "WHAT"):*** Selects *what* visual features to focus on. It tells the model to prioritize channels detecting shapes/edges (like bone outlines) and ignore channels detecting irrelevant info (like grass color).
* ***Spatial Attention (The "WHERE"):*** Spotlights *where* to look. It tells the model to look only at the pixel regions containing the cow's body (like the ***spine, hooks, and pins***) and ignore background pixels (like fences).

Adding CBAM reduced our Dryad BCS error (MAE) from ***0.7000 to 0.6175***.

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q20: How did you train the single-task BCS model?</b></span>

We trained the model separately on two datasets: ***ScienceDB*** (53,566 rear-view RGB images with scores 3.25 to 4.25) and ***Dryad*** (5,923 Depth-Grayscale-Edge images with scores 2 to 6). The training steps were:
1. ***Dataset Split:*** We used a ***cow-wise split*** (70% train, 15% validation, 15% test) to make sure images of the same cow were never in both training and testing.
2. ***Backbone and Attention:*** We used the ***EfficientNet-B0*** backbone initialized with ImageNet weights, and added our custom ***CBAM*** attention module at the end.
3. ***Loss Function:*** We trained it using ***CORAL Loss*** (Consistent Rank Logits) because BCS scores have a natural order (from thin to fat).
4. ***Hyperparameters:*** 
   * ***Optimizer:*** Adam (learning rate of ***1e-3***, halved every 10 epochs).
   * ***Batch size:*** 32.
   * ***Epochs:*** 30.
5. ***Training Execution:*** We trained on an RTX 4080 GPU, saving the best checkpoint based on the lowest ***Validation MAE*** (Mean Absolute Error) to prevent overfitting.

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q21: What is CORAL Loss and why did you use it?</b></span>

***CORAL (Consistent Rank Logits)*** is a loss function designed for ***ordinal regression*** (predicting categories that have a natural mathematical order, like body condition scores from thin to fat).

We use it because it teaches the model the order of scores. For example, if a cow's true score is ***3.50***:
* ***With CORAL:*** Guessing ***3.75*** is treated as "less wrong" (small penalty), while guessing ***4.50*** is treated as "more wrong" (big penalty).
* ***Without CORAL (Normal AI):*** Both ***3.75*** and ***4.50*** are treated as equally wrong (same penalty).

By teaching the model the order, we get much more accurate predictions. Using CORAL reduced our ScienceDB BCS error (MAE) from ***0.5566*** (compared to ***0.6940*** when using standard Cross-Entropy).

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q22: What were the body condition scoring (BCS) results, and how did the multi-task model compare to the single-task model?</b></span>

Our primary clinical metric was ***±1 Accuracy*** (predicting within 0.25 points of the real score). The results were:
* ***Single-Task BCS (EfficientNet-B0):*** Achieved ***90.50%*** accuracy on ScienceDB (RGB) and ***85.75%*** on Dryad (Depth).
* ***Frame-Level Multi-Task BCS:*** Dropped slightly to ***87.64%*** on ScienceDB.
* ***Spatiotemporal Multi-Task BCS:*** Dropped to ***84.82%*** on ScienceDB.

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q23: Why did the BCS performance degrade in the multi-task models, and how do you plan to improve it?</b></span>

The performance degraded because of ***negative transfer*** (gradient conflict). The tasks conflict: BCS needs static shapes (bone outlines), while Lameness needs moving sequences (joints bending). Forcing both into one backbone causes their updates to fight and degrade BCS accuracy. 

To improve this in the future, we plan to use gradient alignment algorithms like ***PCGrad*** (which mathematically stops conflicting gradients from fighting) and ***GradNorm*** (which dynamically balances task learning weights).

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q24: How did you train the single-task behavior recognition model?</b></span>

We trained the model on the ***MmCows*** dataset (213,686 cropped RGB frames across 7 behavior classes: walking, standing, feeding head up, feeding head down, licking, drinking, lying). The training steps were:
1. ***Dataset Split:*** We used a ***cow-wise split*** (11 cows for training, 2 for validation, 3 for testing) to ensure the model was tested on unseen cows.
2. ***Data Capping (Class Balancing):*** Because the dataset was highly imbalanced (e.g., 83,000 lying frames vs. only 2,009 licking frames), we capped each training class at ***3,000 frames*** per epoch to create a balanced subset of ~21,000 images and prevent majority-class bias.
3. ***Backbone and Attention:*** We used the ***EfficientNet-B0*** backbone with our custom ***CBAM*** attention module.
4. ***Loss Function:*** We trained it using ***Focal Loss*** (gamma = 2, alpha = 0.25) to down-weight easy, majority classes and force the model to focus on hard, rare classes (like drinking or licking).
5. ***Hyperparameters:*** Trained for 30 epochs (batch size 128) using Adam. We stopped training early (early stopping) once the validation F1-score stabilized.

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q25: What is Focal Loss and why did you use it?</b></span>

***Focal Loss*** is a training rule that tells the AI to ignore easy photos it already knows and focus only on hard, confusing photos. Swapping to Focal Loss raised our Behavior F1-score from ***0.7074 to 0.7445***.

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q26: Why did you use Focal Loss if you already balanced the dataset by capping it?</b></span>

We used it because capping only balances the ***number*** of photos, not the ***difficulty*** of the task. Lying is very easy for the AI to recognize, while licking is hard. Without Focal Loss, the AI gets lazy by focusing on the easy lying photos. Focal Loss turns off the error for easy guesses (like lying), forcing the AI to spend 100% of its learning power on the hard behaviors (like licking).

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q27: What were the behavior recognition results for the single-task baseline model?</b></span>

Our single-task ***EfficientNet-B0*** baseline achieved a ***Macro F1-score of 0.7445***. The individual class accuracies on the test set were:

* ***Lying:*** 98.77%
* ***Licking:*** 83.96%
* ***Drinking:*** 83.01%
* ***Feeding Head Up:*** 71.97%
* ***Standing:*** 70.00%
* ***Feeding Head Down:*** 66.07%
* ***Walking:*** 59.78%

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q28: Why does the behavior model's performance vary so much, and why does it struggle on specific classes?</b></span>

The performance varies across behaviors due to the following reasons:

1. ***Lying (Easiest - 98.77%):*** It has a distinct, static posture that is visually completely different from all other classes.
2. ***Walking vs. Standing (Worst - 59.78% / 70.00%):*** Because this is a frame-level (2D CNN) model, a walking cow captured mid-stride looks physically identical to a standing cow. Without temporal context (video motion over time), the model cannot distinguish between them.
3. ***Licking and Drinking (Rare Behaviors - 83.96% / 83.01%):*** These suffer from ***severe class imbalance*** (e.g., only 2,009 licking frames vs. over 83,000 lying frames), giving the model far less data to learn from. They also require capturing micro-movements of the mouth/head that are difficult to see in a single static crop.
4. ***Feeding Head Up vs. Down (Similar Postures - 71.97% / 66.07%):*** Both classes are visually almost identical, differing only by a minor change in head height, leading to high classification confusion.

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q29: How did behavior recognition perform in the multi-task models compared to the single-task baseline?</b></span>

The overall behavior performance across the three configurations is:
* ***Single-Task Baseline:*** Achieved a ***Macro F1-score of 0.7445***.
* ***Frame-Level Multi-Task (Spatial):*** Dropped severely to ***0.3771***. This was due to ***negative transfer*** (gradient conflict) as the shared backbone struggled to balance static tasks (like BCS bone outlines) with behavior tasks.
* ***Spatiotemporal Multi-Task (Temporal):*** Recovered to ***0.4948***. By processing video sequences and using ***temporal majority voting***, the model was able to filter out single-frame prediction errors.

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q30: How did you train the single-task lameness detection model?</b></span>

We trained the model on the ***CattleLameness*** dataset (50 videos total: 25 lame, 25 normal). The training steps were:
1. ***Video-wise Split:*** We split the data into 34 videos for training, 6 for validation, and 10 for testing to prevent walking clip leakage.
2. ***Sampling:*** We cropped the cow in each frame using YOLOv8-Nano and extracted exactly ***20 evenly spaced frames*** per video.
3. ***Sequence Model:*** We ran the crops through the ***EfficientNet-B0 backbone*** and fed the features into a ***single-layer LSTM*** (256 hidden units) to track walking motion over time.
4. ***Loss and Hardware:*** We trained the model on an RTX 4080 GPU for 15 epochs using ***Binary Cross-Entropy (BCE) loss***.

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q31: What is an LSTM and why did you use it in your model?</b></span>

An ***LSTM (Long Short-Term Memory)*** is a type of neural network designed to analyze ***sequential data*** (like video frames over time) rather than just single static images. 

We used it because tasks like ***lameness detection*** and ***behavior recognition*** cannot be diagnosed from a single photo. A leg lifted off the ground looks identical whether a cow is walking normally or limping. By using an LSTM, the model connects the sequence of 20 frames to track ***stride mechanics*** and ***gait asymmetry*** over time to make a correct decision.

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q32: What were the lameness detection results, and why is there a controversy about the 100% accuracy score?</b></span>

Our lameness detection results across the different configurations were:
* ***Single-Task Spatial:*** Achieved ***93.05% accuracy*** (AUC 0.9829), but this score is too high and misleading due to ***identity/background leakage*** from frame-level splitting.
* ***Single-Task Spatiotemporal (LSTM):*** Achieved an honest, leak-free ***80.00% accuracy*** (AUC 0.8400).
* ***Multi-Task Spatiotemporal:*** Achieved an uncalibrated ***80.00% accuracy*** (AUC 1.0000).

The ***100.00% accuracy*** is ***invalid*** because we changed the pass/fail score threshold after looking at the test results (known as ***post-hoc threshold calibration***). 

Because the dataset was tiny (only 50 videos), the model's output scores were shifted upwards (predicting 55% for healthy cows and 85% for lame cows). Since 55% is above the default 50% threshold, healthy cows were wrongly classified as lame. By manually raising the threshold to 70%, we got 100% accuracy. However, tuning the model on the test set is ***test-set leakage*** (cheating), so the honest final accuracy is ***80.00%***.

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q33: Why are the lameness detection results not fully reliable or considered "bad" for practical use?</b></span>

The lameness results are not fully trustable for actual farm use due to three main reasons:
1. ***Tiny Dataset Size:*** The dataset contains only ***50 videos*** in total. This extremely small size makes it very easy for the model to ***overfit*** (memorizing the specific walking patterns of these few cows instead of learning general limping behavior).
2. ***Tiny Test Set:*** With our split, the test set consists of only ***10 videos***. Because of this, a single correct or incorrect guess changes the accuracy by 10%, making the final results statistically ***unreliable***.
3. ***Poor Dataset Quality:*** The videos have messy backgrounds, bad farm lighting, and blocked views. Since the clips were filmed from different angles, in different areas, on different farms, and with different cows, the model struggles to learn general walking patterns. A dataset this small and messy is not enough to build a reliable tool.

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q34: How did you train the single-task cow identification model?</b></span>

We trained the model on the ***OpenCows2020*** dataset (4,736 images across 46 classes of individual cows) using the following pipeline:
1. ***Image-wise Split:*** We split the images of each cow into 70% for training, 15% for validation, and 15% for testing. Every cow must be in the training set so the model can learn to recognize it.
2. ***Cropping and Resize:*** We cropped the cow from each image using YOLOv8-Nano and resized it to ***224x224 pixels***.
3. ***Backbone and Head:*** We ran the crops through the ***EfficientNet-B0 backbone*** and used a standard linear classifier head to predict the cow's identity.
4. ***Loss & Optimization:*** We trained the model on an RTX 4080 GPU for 10 epochs using ***Cross-Entropy loss*** and the ***Adam optimizer*** (learning rate of 1e-3).

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q35: What were the cow identification results, and how did the multi-task models compare to the baseline?</b></span>

Our cow identification results across the different configurations were:
* ***Single-Task Baseline:*** Achieved ***86.49% accuracy*** on single images.
* ***Frame-Level Multi-Task:*** Rose unexpectedly to ***94.96% accuracy*** (an unexplained gain, as joint training usually hurts static tasks).
* ***Spatiotemporal Multi-Task:*** Achieved our best score of ***97.58% accuracy***.

The ***97.58% accuracy*** was achieved by using ***temporal majority voting***. Instead of guessing the cow's identity from a single frame, the model predicts the identity across 20 frames of a video clip and selects the most frequent prediction as the final answer. This majority voting filters out temporary errors, such as frames where the cow is briefly blocked by a gate or out of focus.

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q36: How would your model recognize a new cow?</b></span>

Our current model cannot recognize a new cow because it only knows the 46 cows it was trained on. 

To recognize a new cow, we must:
1. Collect pictures of the new cow from different angles.
2. Add those pictures to our training dataset.
3. ***Retrain the final layer*** of the model so it learns the new cow's identity.

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q37: What are the current limitations of your cow identification model?</b></span>

Our cow identification has three main limitations:
1. ***Closed-Set Limit:*** The model can only recognize the 46 cows it was trained on. It cannot identify any new cows unless the final layer is retrained.
2. ***Unfair Baseline Comparison:*** During baseline testing, all models were trained for only 10 epochs. Deeper models like ResNet-50 did not fully converge (finish learning) in that short time, making the comparison slightly unfair to them.
3. ***YOLO Dependence:*** The ID accuracy depends heavily on the YOLOv8-Nano crop. If the cow's face/body is blocked and YOLO makes a bad crop, the model will fail to identify the cow.

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q38: How did you train the spatiotemporal multi-task model?</b></span>

We trained the multi-task model using a ***6-phase training pipeline*** to ensure all tasks converged without interfering with each other:

1. ***Phase 1: Backbone Initialization:*** Pretrained ImageNet weights are loaded into the shared ***EfficientNet-B0 backbone*** to establish a robust foundation for extracting general visual features.
2. ***Phase 2: BCS Head Warmup (Backbone Frozen):*** With the backbone weights frozen, we trained only the BCS classification head using *CORAL Loss* for ***5 epochs*** (learning rate: ***1e-3***).
3. ***Phase 3: Behavior Head Warmup (Backbone Frozen):*** With the backbone frozen, we trained only the Behavior LSTM and classifier head using *Focal Loss* for ***5 epochs*** (learning rate: ***1e-3***).
4. ***Phase 4: Lameness Head Warmup (Backbone Frozen):*** With the backbone frozen, we trained only the Lameness LSTM and classifier head using *Binary Cross-Entropy (BCE) Loss* for ***5 epochs*** (learning rate: ***1e-3***).
5. ***Phase 5: Cow ID Head Warmup (Backbone Frozen):*** With the backbone frozen, we trained only the Cow Identification linear head using *Cross-Entropy Loss* for ***5 epochs*** (learning rate: ***1e-3***).
6. ***Phase 6: Joint Fine-Tuning (Backbone Unfrozen):*** We unfroze the entire network (backbone + all heads) and trained them simultaneously for ***10 epochs*** using a lower learning rate of ***1e-4***. All four individual losses were combined into a weighted ***total loss*** using this formula:
   * **Total Loss = 0.35 * L_BCS + 0.35 * L_BEH + 0.15 * L_LAM + 0.15 * L_ID**

***Why did we use this warmup strategy?***
If we train all tasks jointly from scratch, the random initial gradients from the un-warmed task heads will conflict with each other, causing the shared backbone to learn chaotic features. Freezing the backbone and warming up each head first (Phases 2-5) ensures that they are already capable of reading the backbone's features before we unfreeze the backbone for joint optimization (Phase 6).

---

<span style="font-size:13pt; background-color: #b6d7a8; color: black;"><b>Q41: What ablation studies did you perform in your thesis and what were the key findings?</b></span>

We performed **five systematic ablation and generalization experiments** to isolate the impact of our design choices:

1. ***Ablation 1: CORAL Ordinal Loss vs. Cross-Entropy Loss (BCS on ScienceDB):***
   * **Goal:** Test if treating BCS scores as ordered categories (ordinal regression) is better than treating them as unrelated classes (standard classification).
   * **Results:** Using **CORAL Loss** achieved a Mean Absolute Error (MAE) of **0.5566** (90.50% ±1 accuracy), whereas standard **Cross-Entropy** yielded a worse MAE of **0.6940** (86.62% ±1 accuracy).
   * **Finding:** Explicitly modeling the ordinal structure (thin to fat progression) reduced error by **19.8%**.

2. ***Ablation 2: Contribution of CBAM Attention Module (BCS on Dryad):***
   * **Goal:** Test if adding spatial and channel attention (CBAM) helps filter background noise.
   * **Results:** Without CBAM, the model got a Test MAE of **0.7000** (80.75% ±1 accuracy). With CBAM, the Test MAE dropped to **0.6175** (85.75% ±1 accuracy).
   * **Finding:** CBAM successfully directed attention to key anatomical regions (spine, hooks, pins) and improved scoring accuracy.

3. ***Ablation 3: Cross-Dataset Modality Comparison (RGB vs. Depth):***
   * **Goal:** Compare performance on standard color images (RGB) vs. Depth Grayscale Edge (DGE) maps.
   * **Results:** RGB images (ScienceDB) got an MAE of **0.5566** (trained on 53,566 images). Depth contours (Dryad) got a comparable MAE of **0.6175** (trained on only 5,923 images).
   * **Finding:** Depth contour maps contain strong geometric details that allow the model to learn effectively even when the dataset is nearly 10 times smaller.

4. ***Ablation 4: Focal Loss vs. Cross-Entropy Loss (Behavior on MmCows):***
   * **Goal:** Test how to handle the severe 42:1 class imbalance in cow behavior.
   * **Results:** Standard Cross-Entropy got a Macro F1-score of **0.7074** (cheating by only predicting common classes like lying). **Focal Loss** improved the F1-score to **0.7445**.
   * **Finding:** Focal Loss is essential because it down-weights easy, common classes and forces the model to learn rare classes (like licking and drinking).

5. ***Ablation 5: Cross-Dataset Behavior Evaluation (Generalization on CBVD-5):***
   * **Goal:** Test if a model trained on one farm (MmCows) can generalize to an unseen farm (CBVD-5).
   * **Results:** The Macro F1-score crashed from **0.7445 to 0.1245**. Standing posture generalized well (90.34% accuracy), but lying down (24.31%) and drinking (2.99%) degraded heavily.
   * **Finding:** Farm backgrounds, camera angles, and lighting represent a severe domain shift, meaning future models must train on multi-farm datasets to work in the real world.

---

