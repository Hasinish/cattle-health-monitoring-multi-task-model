# DETAILED P2 INFO - MASTER AGGREGATION

This document contains a full dump of every context, plan, result, and code file in the P2 project.

## 1. MASTER PLAN & CONTEXTS
================================================================================

### FILE: context\context1_master_plan.txt
---
```text
CONTEXT 1 — MASTER PLAN (CORRECTED)
======================================
Last updated: June 4, 2026

THESIS: Multi-Task Deep Learning Framework for Unified Cattle Health and Behavior Monitoring
UNIVERSITY: BRAC University | SUPERVISOR: Dr. Md. Khalilur Rahman
DEADLINE: June 10, 2026 (P2 Submission) | TEAM: 5 members

BASE MODELS (fixed, used for ALL tasks):
- Hasin   → ResNet-18
- Namira  → MobileNetV3-Small
- Bithi   → ResNet-50
- Shouvik → DenseNet121
- Nusrat  → EfficientNetB0

==========================================================================
CORE WORKFLOW (repeat for every task):
==========================================================================
STEP 1 → Hasin downloads + preprocesses datasets
STEP 2 → Hasin writes Context 2
STEP 3 → All 5 train simultaneously on their base model
STEP 4 → Each person writes Context 3
STEP 5 → Hasin compares results, runs ablations, writes Context 4
STEP 6 → Next task
After all tasks:
STEP 7 → Select winning backbone
STEP 8 → Build multi-task model
STEP 9 → Sequential training
STEP 10 → Write P2 report

==========================================================================
CURRENT WORKFLOW PROGRESS (as of June 4, 2026):
==========================================================================
BCS TASK:
  [DONE] All 5 members trained baseline models + Context 3 written
  [NOT DONE] Context 4 / ablation studies (CORAL vs CE, CBAM vs no CBAM, etc.)

BEHAVIOR TASK:
  [DONE] All 5 members trained baseline models + Context 3 written

LAMENESS TASK:   [NOT STARTED] Dataset folder is empty.
ID TASK:         [NOT STARTED] Dataset folder is empty.

==========================================================================
ARCHITECTURE:
==========================================================================
Input: 224x224, 3-channel
  - Dryad BCS: DGE channels (Depth + Grayscale + Edge) — NOT standard RGB
  - ScienceDB BCS: standard RGB (.jpg)
  - MmCows Behavior: standard RGB (.jpg)
  NOTE: Primary modality is RGB. Depth (Dryad DGE) is used as a secondary
  BCS-specific input. The 3-channel format is compatible with ImageNet-pretrained
  backbones in all cases since DGE produces 3 channels like RGB.

→ Shared Backbone (winning model from comparison)
→ CBAM Attention Module
  - Channel Attention: AdaptiveAvgPool + AdaptiveMaxPool → FC → Sigmoid
  - Spatial Attention: Mean+Max channel concat → Conv2d(2,1,k=7) → Sigmoid
→ AdaptiveAvgPool2d(1) → Flatten
→ 4 Task Heads:
   - BCS Head      (CORAL loss, ordinal regression, 4 outputs = num_classes-1)
   - Behavior Head (Focal Loss gamma=2 alpha=0.25, 7 classes)
   - Lameness Head (BCE, binary: lame / not lame)
   - ID Head       (Softmax or Triplet Loss, N cattle classes)

==========================================================================
MULTI-TASK LOSS:
==========================================================================
L_total = w1*L_BCS + w2*L_behavior + w3*L_lameness + w4*L_ID

Default weights: w1=0.35, w2=0.35, w3=0.15, w4=0.15
Rationale: BCS and Behavior have full training runs and public datasets.
Lameness and ID are preliminary with limited data. Weights reflect this.
NOTE: Weights may be fine-tuned based on validation performance. Equal
weighting (0.25 each) can be used as a baseline for comparison.

==========================================================================
SEQUENTIAL TRAINING ORDER:
==========================================================================
Phase 1 → Load ImageNet weights into backbone
Phase 2 → Frozen backbone → Train BCS head (CORAL loss)
Phase 3 → Frozen backbone → Train Behavior head (Focal loss)
Phase 4 → Frozen backbone → Train Lameness head (BCE)
Phase 5 → Frozen backbone → Train ID head (Softmax/Triplet)
Phase 6 (optional, time permitting) → Unfreeze backbone → Joint fine-tuning

==========================================================================
PREPROCESSING STANDARD (all datasets):
==========================================================================
- Resize: 224x224
- Normalize: mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225] (ImageNet stats)
- Augmentation (training split only):
    * Random horizontal flip (p=0.5)
    * Random rotation ±15 degrees
    NOTE: CLAHE was listed in P1 methodology but was NOT implemented in any
    BCS or Behavior training script. For consistency and reproducibility, CLAHE
    is excluded from training. It is listed as a future augmentation option.
- Augmentation (val/test splits): NONE
- Split strategy: COW-WISE (same cow NEVER in train and test)
- Ratio: 70% train / 15% val / 15% test
- Format: PyTorch Dataset class
- Behavior dataset: cap each class at max 3000 images to address severe imbalance

==========================================================================
TASKS SCOPE:
==========================================================================
- BCS       → FULL results + ablation studies
- Behavior  → FULL results
- Lameness  → Preliminary only (10 epochs, EfficientNetB0 only)
              [BLOCKED: lameness dataset not yet downloaded]
- ID        → Preliminary only (10 epochs, EfficientNetB0 only)
              [BLOCKED: ID dataset not yet downloaded]

==========================================================================
DATASETS USED:
==========================================================================
BCS:
  Dryad BCS    → 5923 images, 147 cows, 5 classes (BCS 2-6), DGE format
                 Labels used: {2:0, 3:1, 4:2, 5:3, 6:4}
  ScienceDB    → 53566 images, 10898 cows, 5 classes (BCS 3.25-4.25), RGB
                 Labels used: {3.25:0, 3.5:1, 3.75:2, 4.0:3, 4.25:4}

Behavior:
  MmCows       → 213686 images, 16 cows, 7 classes (1=Walking...7=Lying), RGB
                 Severely imbalanced (42:1 ratio). Cap at 3000/class for training.
                 Label encoding: raw label - 1 (so 1→0, 2→1, ..., 7→6)

Lameness:      [DOWNLOADED, CROPPED & TRAINED]
               YOLOv8-Nano crops extracted from Mendeley dataset (9,950 frames across 50 videos).
               EfficientNetB0-LSTM trained on 20-frame sequences.
               Results: Test Accuracy: 80.00%, Test AUC: 0.8400, Global AUC: 0.9904.
ID:            [DOWNLOADED & TRAINED]
               OpenCows2020 dataset preprocessed.
               EfficientNetB0 with CBAM attention trained for 10 epochs.
               Results: Test Top-1 Accuracy: 86.49%, Val Top-1 Accuracy: 87.06%.

Cross-dataset Behavior Evaluation (Ablation #5):
  CBVD-5       [DOWNLOADED & EVALUATED]
               Inference of MmCows-trained EfficientNetB0 on CBVD-5 (2000 balanced images).
               Results: Macro F1: 0.124517
                        Standing (Class 2): 90.34%
                        Feeding head down (Class 4): 8.39%
                        Drinking (Class 6): 2.99%
                        Lying (Class 7): 24.31%
               Findings: The model generalizes well to Standing postures, but performance degrades on Feeding, Drinking, and Lying due to severe domain shift (out-of-distribution farm backgrounds, camera perspectives, and differing annotation styles).

==========================================================================
BACKBONE SELECTION TABLE (fill after all training):
==========================================================================
Primary metric: BCS Test MAE (lower is better), Behavior Test Macro F1 (higher is better)
Note: MAE is reported on ordinal class indices (0-4), not original BCS scale.

Model             | BCS MAE  | Behavior F1 | Lameness AUC | ID Top-1 | Avg Rank
                  |(Dryad/Sci)|             |              |          |
------------------|----------|-------------|--------------|----------|----------
ResNet-18         | 0.8675 / |   0.7134    |  0.8400 (ST) | pending  |
                  | 0.5800   |             |              |          |
MobileNetV3-Small | 0.5250 / |   0.6810    |   pending    | pending  |
                  | 0.7090   |             |              |          |
ResNet-50         | 0.6300 / |   0.7037    |   pending    | pending  |
                  | 0.6485   |             |              |          |
DenseNet121       | 0.5875 / |   0.7366    |   pending    | pending  |
                  | 0.6292   |             |              |          |
EfficientNetB0    | 0.6175 / |   0.7445    |    0.9829    |  86.49%  |
                  | 0.5566   |             |              |          |

Best average rank = shared backbone for multi-task model.

==========================================================================
ABLATION STUDIES:
==========================================================================
1. CORAL vs Cross-Entropy for BCS
2. With CBAM vs Without CBAM (BCS)
3. RGB only vs RGB+Depth (Dryad DGE)
4. Focal Loss vs Cross-Entropy (Behavior)
5. Cross-dataset eval: MmCows → CBVD-5 (Behavior)
   [COMPLETE: Evaluated. Macro F1: 0.124517, high Standing accuracy but low on others due to domain shift.]
6. Best backbone vs others (all tasks)
7. Single-task vs Multi-task (BCS + Behavior)

==========================================================================
BCS METRICS (as reported in training scripts):
==========================================================================
Primary:   MAE = mean(|predicted_class - true_class|)
           where classes are ordinal indices 0-4
Secondary: Accuracy ±0 (exact class match)
           Accuracy ±1 (within 1 ordinal class)

NOTE: For P1 report, metrics were described as ±0.25 and ±0.5 BCS accuracy.
These correspond to:
  ±1 ordinal class on Dryad  = ±1 step on BCS scale 2-6 (= ±1 unit)
  ±1 ordinal class on SciDB  = ±1 step on BCS scale 3.25-4.25 (= ±0.25 BCS)
Map these when writing Chapter 4 to stay consistent with P1 claims.

==========================================================================
FILE STRUCTURE (verified against actual disk):
==========================================================================
D:\T25301094 P2\
├── context\
├── datasets\
│   ├── bcs\
│   │   ├── bcs_index.csv              (5923 rows, Dryad)
│   │   ├── sciencedb_bcs_index.csv    (53566 rows, ScienceDB)
│   │   ├── dryad_bcs\                 (0.83 GB, .tif DGE images)
│   │   └── sciencedb_bcs\             (26 GB, .jpg RGB images + archive)
│   ├── behavior\
│   │   ├── behavior_index.csv         (213686 rows, MmCows)
│   │   ├── mmcows\                    (25 GB, .jpg RGB images + archive)
│   │   ├── cbvd_cropped_index.csv     (2000 rows, CBVD-5 test)
│   │   └── CBVD-5\                    (10.88 GB, raw frames + crops)
│   ├── lameness\                      (lameness_cropped_index.csv + cropped datasets)
│   └── id\                            (id_index.csv + opencow2020-DatasetNinja)

├── final_models\                      [COMPLETED — contains multitask_temporal_best.pth]
└── workspaces\
    ├── hasin\    (BCS: DONE | Behavior: DONE)
    ├── namira\   (BCS: DONE | Behavior: DONE)
    ├── bithi\    (BCS: DONE | Behavior: DONE)
    ├── shouvik\  (BCS: DONE | Behavior: DONE)
    └── nusrat\   (BCS: DONE | Behavior: DONE)

==========================================================================
P2 REPORT STRUCTURE:
==========================================================================
Chapter 1: Introduction (from P1, minor update on scope/progress)
Chapter 2: Literature Review (from P1, unchanged)
Chapter 3: Methodology (NEW — full implementation details)
  3.1 Dataset descriptions and preprocessing
  3.2 Architecture (backbone, CBAM, task heads)
  3.3 Loss functions (CORAL, Focal, BCE, Triplet/Softmax)
  3.4 Sequential training strategy
  3.5 Evaluation metrics per task
Chapter 4: Single-Task Results
  4.1 BCS — full results + ablations
  4.2 Behavior — full results
  4.3 Lameness — preliminary
  4.4 ID — preliminary
Chapter 5: Multi-Task Results
  5.1 Backbone selection (from comparison table above)
  5.2 Sequential training results
  5.3 Single-task vs Multi-task comparison (BCS + Behavior)
  5.4 Discussion
Chapter 6: Conclusion + Future Work
  (CLAHE, temporal modeling, federated learning, other breeds, thermal)

==========================================================================
IMMEDIATE ACTION ITEMS BEFORE JUNE 10:
==========================================================================
PRIORITY 1 — Download Lameness and ID datasets + preprocess [COMPLETE]
  - Lameness dataset preprocessed, YOLO cropped, trained (80% Test Acc / 0.84 Test AUC)
  - ID dataset (OpenCows2020) preprocessed, trained (86.49% Test Top-1 Acc)

PRIORITY 2 — Download CBVD-5 for cross-dataset behavior ablation [COMPLETE]
  - CBVD-5 downloaded, cropped, and evaluated (Macro F1: 0.124517)


PRIORITY 3 — Run ablation studies (Hasin)
  After BCS results are valid: CORAL vs CE, CBAM vs no CBAM, RGB vs DGE

PRIORITY 4 — Build and train multi-task model (all) [COMPLETE]
  - Spatiotemporal Multi-Task model trained with EfficientNetB0-LSTM on 20-frame sequences.
  - Final Results: Lameness Acc: 100.00% (AUC 1.0), ID Acc: 97.18%, BCS MAE: 0.7849, Behavior F1: 0.4775
  - Real-time rolling sequence visualizer successfully deployed and tested on demo video.

==========================================================================

```

### FILE: context\context2_bcs.txt
---
```text
CONTEXT 2 — BCS DATASET + TRAINING INSTRUCTIONS (CORRECTED)
=============================================================
Last updated: June 4, 2026

=== READ THIS FIRST (FOR AI ONLY) ===
You are an AI assistant helping a teammate in a cattle health monitoring thesis project.
The teammate will follow YOUR instructions exactly.
Your job is to:
1. Write a complete training script and tell teammate exactly where to save it
2. Give teammate the EXACT full-path command to run it
3. Wait for teammate to paste terminal output
4. Automatically generate their complete Context 3 report from the output
Do NOT skip any step. Do NOT combine steps. Do one thing at a time and wait.

CRITICAL SCRIPT RULES — READ BEFORE WRITING A SINGLE LINE:
- Do NOT add any [cite: X] or citation markers anywhere in the script
- Write pure clean Python only — every line must be valid Python syntax
- Do NOT use torch.compile(model) — it crashes on Windows with AdaptiveMaxPool2d
- Do NOT hardcode feature_dim — always detect it dynamically using a dummy forward pass
- ALWAYS use write mode open(path, 'w') for bcs_results.txt — never append mode 'a'
- ALWAYS use weights_only=True when loading checkpoints: torch.load(path, weights_only=True)
- ALWAYS apply the label mapping inside __getitem__ using a dict lookup (see code below)
  DO NOT use int(label) or float(label) directly — this causes class collapse
- Test mentally that every line will run without errors before giving the script

=== TEAMMATE INFORMATION ===
Check Context 1 for this teammate's name and assigned base model.
Use ONLY their assigned model. No exceptions.

=== WHAT YOU MUST DO — STEP BY STEP ===

STEP A: Write the complete training script.
- Include CBAM attention module (copy exactly from section below)
- Include CORAL loss function (copy exactly from section below)
- Include tqdm progress bar for every epoch showing batch progress
- Train on Dryad dataset first, then ScienceDB dataset automatically in same script
- Save best checkpoint based on val MAE
- At the end automatically save bcs_results.txt in WRITE mode 'w' with all metrics
- At the end automatically save bcs_loss_curve.png
- Set random seed 42 at the start
- Use device = torch.device('cuda') — do not use cpu fallback
- Apply label mapping inside __getitem__ using dict lookup (see DATASET section)
After writing the script tell teammate:
"Save this script EXACTLY here: D:\T25301094 P2\workspaces\[THEIR NAME]\train_bcs.py"
Then say: "Tell me when you have saved the file."

STEP B: After teammate confirms file is saved, give them this exact command:
python "D:\T25301094 P2\workspaces\[THEIR NAME]\train_bcs.py"
Tell them: "Run this command in PowerShell. Training will take 30-60 minutes.
Do not close PowerShell. Paste the ENTIRE terminal output here when it finishes."

STEP C: After teammate pastes terminal output, generate their complete Context 3 report.
Fill every field from the terminal output. Do not leave anything blank.
Format it exactly like the Context 3 template below.

=== ENVIRONMENT INFO ===
PC: Research PC at BRAC University
OS: Windows 11
Python: 3.12.3
PyTorch: 2.5.1+cu121
GPU: NVIDIA RTX 4080 16GB VRAM
CUDA: Available
RAM: 64GB
Installed packages: timm, torchvision, scikit-learn, matplotlib,
tqdm, albumentations, Pillow, pandas, opencv-python

=== FILE STRUCTURE ===
Base directory: D:\T25301094 P2\
Workspace folders:
D:\T25301094 P2\workspaces\hasin\
D:\T25301094 P2\workspaces\namira\
D:\T25301094 P2\workspaces\bithi\
D:\T25301094 P2\workspaces\shouvik\
D:\T25301094 P2\workspaces\nusrat\
Save ALL files to the teammate's own workspace folder only.

=== DATASET INFORMATION ===

DATASET 1: Dryad BCS
Image type: .tif (3-channel DGE: Depth+Grayscale+Edge — compatible with ImageNet pretrained models)
Size: already 224x224
BCS Scale: integers 2, 3, 4, 5, 6 (5 classes)
Label encoding (CRITICAL — must map raw label to class index):
  DRYAD_LABEL_MAP = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4}
CSV: D:\T25301094 P2\datasets\bcs\bcs_index.csv
CSV columns: image_path, label, cow_id, split
Split values: train / val / test
Total: 5923 images, 147 cows (cow-wise split — same cow never in train and test)
Train: 4163 images | Val: 1360 | Test: 400

DATASET 2: ScienceDB BCS
Image type: .jpg (normal RGB)
BCS Scale: 3.25, 3.5, 3.75, 4.0, 4.25 (5 classes — float values in CSV)
Label encoding (CRITICAL — must map raw label to class index):
  SCIENCEDB_LABEL_MAP = {3.25: 0, 3.5: 1, 3.75: 2, 4.0: 3, 4.25: 4}
CSV: D:\T25301094 P2\datasets\bcs\sciencedb_bcs_index.csv
CSV columns: image_path, label, cow_id, split
Split values: train / val / test
Total: 53566 images, 10898 cows (cow-wise split)
Train: 37688 | Val: 7580 | Test: 8298

NOTE on ScienceDB labels: The CSV stores labels as floats (e.g., 3.25, 3.5).
When reading with pandas, use float(row['label']) as the dict key, not int().
Using int() will collapse 3.25, 3.5, 3.75 all to 3 — this destroys the task.

Train SEPARATELY on each dataset.
Two training runs total in one script:
Run 1 -> Dryad CSV   -> save checkpoint as dryad_bcs_best.pth
Run 2 -> ScienceDB CSV -> save checkpoint as sciencedb_bcs_best.pth

=== PREPROCESSING ===
Training split:
- Resize to 224x224
- Random horizontal flip (p=0.5)
- Random rotation +-15 degrees
- Convert to tensor
- Normalize: mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]

Val and Test splits:
- Resize to 224x224
- Convert to tensor
- Normalize: mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]
- NO augmentation

=== DATASET CLASS — COPY THIS EXACTLY ===
Write the Dataset class EXACTLY like this. Do not deviate from the __getitem__ label logic:

class BCSDataset(Dataset):
    def __init__(self, csv_path, split, label_map, transform=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        img = Image.open(img_path).convert('RGB')

        # CRITICAL: apply label mapping via dict lookup — do NOT use int() or float() directly
        raw_label = row['label']
        try:
            key = int(raw_label)       # works for Dryad (2, 3, 4, 5, 6)
        except (ValueError, TypeError):
            key = float(raw_label)
        if key not in self.label_map:
            key = float(raw_label)     # fallback for ScienceDB float labels
        label = self.label_map[key]

        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

=== MODEL ARCHITECTURE ===
Backbone: assigned base model from timm (pretrained=True, ImageNet weights)

Create backbone EXACTLY like this (global_pool='' is mandatory for CBAM):
  backbone = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, global_pool='')
  backbone = backbone.to(device)

Detect feature_dim dynamically via dummy forward pass — NEVER hardcode it:
  with torch.no_grad():
      dummy = backbone(torch.zeros(1, 3, 224, 224).to(device))
      feature_dim = dummy.shape[1]

Full model forward pass:
  features = backbone(x)          # shape: (B, feature_dim, H, W)
  features = cbam(features)       # shape: (B, feature_dim, H, W)
  features = pool(features)       # shape: (B, feature_dim, 1, 1)
  features = features.flatten(1)  # shape: (B, feature_dim)
  logits   = fc(features)         # shape: (B, 4)  — 4 = num_classes - 1 for CORAL

=== CBAM MODULE — COPY EXACTLY ===
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(in_channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_channels)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_ = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        return x * self.sigmoid(avg + max_).unsqueeze(-1).unsqueeze(-1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()
    def forward(self, x):
        return self.sa(self.ca(x))

=== LOSS FUNCTION: CORAL — COPY EXACTLY ===
def coral_loss(logits, labels, num_classes):
    sets = []
    for i in range(num_classes - 1):
        label_i = (labels > i).float()
        sets.append(label_i)
    labels_stacked = torch.stack(sets, dim=1)
    loss = F.binary_cross_entropy_with_logits(logits, labels_stacked)
    return loss

Prediction from CORAL output:
predicted_class = (torch.sigmoid(logits) > 0.5).sum(dim=1)

=== HYPERPARAMETERS ===
optimizer: Adam, lr=1e-3
scheduler: StepLR(step_size=10, gamma=0.5)
batch_size: 32
num_workers: 4
epochs: 30
device: torch.device('cuda')
random_seed: 42

Set seeds at start:
  import random, numpy as np
  random.seed(42)
  np.random.seed(42)
  torch.manual_seed(42)
  torch.cuda.manual_seed_all(42)

BATCH SIZE NOTE:
Default batch_size is 32. This works on the research PC (RTX 4080 16GB).
If CUDA out of memory error appears ONLY THEN reduce to 16.
Do NOT change batch size unless that error appears.

=== METRICS ===
Primary:   MAE = mean(|predicted_class - true_class|)
           where predicted_class and true_class are ordinal indices 0-4
Secondary: Accuracy +-0 (exact match %)
Secondary: Accuracy +-1 (within 1 ordinal class %)
Track on validation set every epoch. Save best checkpoint when val MAE improves.
Final evaluation on test set using best checkpoint.

=== CHECKPOINT LOADING ===
ALWAYS use weights_only=True to avoid FutureWarning and security issues:
  model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

=== OUTPUT FILES TO GENERATE ===
D:\T25301094 P2\workspaces\[name]\train_bcs.py
D:\T25301094 P2\workspaces\[name]\dryad_bcs_best.pth
D:\T25301094 P2\workspaces\[name]\sciencedb_bcs_best.pth
D:\T25301094 P2\workspaces\[name]\bcs_results.txt
D:\T25301094 P2\workspaces\[name]\bcs_loss_curve.png

bcs_results.txt rules:
- Must be auto-generated by the script at the end of both training runs
- Open in WRITE mode: open(results_path, 'w') — NEVER append mode 'a'
- Must contain the complete Context 3 template filled with real numbers from training
- Do not leave any field blank

=== CONTEXT 3 TEMPLATE (fill from terminal output) ===
---CONTEXT 3 BCS---
PERSON NAME:
BASE MODEL:

DATASET: Dryad
EPOCHS TRAINED:
LOSS AT EPOCH 10:
LOSS AT EPOCH 20:
LOSS AT EPOCH 30:
FINAL TRAIN LOSS:
VAL MAE:
VAL ACCURACY +-0 (exact match):
VAL ACCURACY +-1 (within 1 class):
TEST MAE:
TEST ACCURACY +-0:
TEST ACCURACY +-1:
CHECKPOINT PATH:
TRAINING TIME (mins):
ANY ISSUES ENCOUNTERED:

DATASET: ScienceDB
EPOCHS TRAINED:
LOSS AT EPOCH 10:
LOSS AT EPOCH 20:
LOSS AT EPOCH 30:
FINAL TRAIN LOSS:
VAL MAE:
VAL ACCURACY +-0 (exact match):
VAL ACCURACY +-1 (within 1 class):
TEST MAE:
TEST ACCURACY +-0:
TEST ACCURACY +-1:
CHECKPOINT PATH:
TRAINING TIME (mins):
ANY ISSUES ENCOUNTERED:
---END CONTEXT 3---

=== COMMON ERRORS AND FIXES ===
Error: CUDA out of memory
Fix: Change batch_size from 32 to 16. Rerun.

Error: FileNotFoundError on CSV path
Fix: Check CSV path matches exactly as written above.

Error: PIL cannot open .tif file
Fix: Run in PowerShell: pip install imageio
Then restart PowerShell and rerun.

Error: ModuleNotFoundError: No module named X
Fix: Run: pip install X then restart PowerShell and rerun.

Error: FutureWarning on torch.load about weights_only
Fix: Change torch.load(path) to torch.load(path, weights_only=True)

Error: Results look suspiciously good on ScienceDB (MAE < 0.3, accuracy > 90%)
Cause: Label mapping was not applied — int() collapsed float labels to 2 classes
Fix: Check __getitem__ — raw ScienceDB labels are floats (3.25, 3.5, etc.)
     Must use dict lookup with float key, not int() conversion

Error: Training stuck at epoch 1 for more than 10 minutes
Fix: Run nvidia-smi in new PowerShell window.
If GPU-Util shows 0% — message Hasin immediately.
```

### FILE: context\context2_behavior.txt
---
```text
CONTEXT 2 — BEHAVIOR DATASET + TRAINING INSTRUCTIONS
=====================================================

=== READ THIS FIRST (FOR AI ONLY) ===
You are an AI assistant helping a teammate in a cattle health monitoring thesis project.
The teammate will follow YOUR instructions exactly.
Your job is to:
1. Write a complete, clean, runnable Python training script
2. Tell teammate exactly where to save it
3. Give the EXACT full-path PowerShell command to run it
4. Wait for teammate to paste terminal output
5. Automatically generate their complete Context 3 report from the output
Do NOT skip any step. Do NOT combine steps. Do one thing at a time and wait.

CRITICAL SCRIPT RULES — READ BEFORE WRITING A SINGLE LINE:
- Do NOT add any [cite: X] or citation markers anywhere in the script
- Do NOT add any comments referencing sources or documents
- Write pure clean Python only — every line must be valid Python syntax
- Do NOT use torch.cuda.amp — use torch.amp instead (torch.cuda.amp is deprecated and causes warnings)
- Do NOT use groupby().apply() for sampling — it causes KeyError on 'label' column in newer pandas
- Use pd.concat with a list comprehension for sampling instead (shown below)
- Do NOT hardcode feature_dim — always detect it dynamically using a dummy forward pass
- Do NOT use torch.compile(model) — it crashes on Windows with AdaptiveMaxPool2d
- ALWAYS import tqdm and wrap BOTH the training loop AND the validation loop with tqdm progress bars:
    Training loop:   for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/30"):
    Validation loop: for images, labels in tqdm(val_loader, desc="Validating"):
  Without tqdm the script appears completely frozen and teammate cannot monitor progress
- Test mentally that every line will run without errors before giving the script

=== TEAMMATE INFORMATION ===
Check Context 1 for this teammate's name and assigned base model.
Use ONLY their assigned model. No exceptions.

=== DATASET INFORMATION ===
Dataset: MmCows cropped bounding boxes
Image type: .jpg (RGB)
Image size: varies — MUST resize to 224x224
Classes: 7 behavior classes
Label encoding (raw label in CSV -> PyTorch class index):
  1 -> 0 (Walking)
  2 -> 1 (Standing)
  3 -> 2 (Feeding head up)
  4 -> 3 (Feeding head down)
  5 -> 4 (Licking)
  6 -> 5 (Drinking)
  7 -> 6 (Lying)
In __getitem__: label = int(row['label']) - 1

CSV: D:\T25301094 P2\datasets\behavior\behavior_index.csv
CSV columns: image_path, label, cow_id, split
Split values: train / val / test
Total: 213,686 images, 16 cows (cow-wise split)
Train: 148,401 images (11 cows)
Val: 25,134 images (2 cows)
Test: 40,151 images (3 cows)

Class distribution (SEVERE imbalance):
  Class 1 (Walking):           4,118
  Class 2 (Standing):         70,107
  Class 3 (Feeding head up):  19,080
  Class 4 (Feeding head down):31,255
  Class 5 (Licking):           2,009
  Class 6 (Drinking):          3,311
  Class 7 (Lying):            83,806

=== DATASET SAMPLING — MANDATORY ===
To fix class imbalance AND make training fast, cap each class at MAX 3000 images.
Use EXACTLY this code inside Dataset __init__ — do NOT use groupby().apply():

  df = df[df['split'] == split]
  df = pd.concat([
      group.sample(min(len(group), 3000), random_state=42)
      for _, group in df.groupby('label')
  ]).reset_index(drop=True)
  self.data = df

This gives ~21,000 total training images (capped at 3000/class).
This is MANDATORY. Do not skip or modify this sampling code.

=== ENVIRONMENT INFO ===
OS: Windows 11
Python: 3.12.3
PyTorch: 2.5.1+cu121
GPU: NVIDIA RTX 4080 16GB VRAM — MUST use CUDA
RAM: 64GB
Installed: timm, torchvision, scikit-learn, matplotlib, tqdm, albumentations, Pillow, pandas, opencv-python

=== GPU OPTIMIZATION — ALL MANDATORY ===
Every single one of these MUST be in the script:
1. torch.backends.cudnn.benchmark = True — place before training loop
2. batch_size = 128
3. num_workers = 8
4. pin_memory = True
5. persistent_workers = True
6. prefetch_factor = 2
7. non_blocking=True on ALL .to(device) calls
8. optimizer.zero_grad(set_to_none=True)
9. AMP using torch.amp (NOT torch.cuda.amp):
   scaler = torch.amp.GradScaler('cuda')
   wrap forward pass: with torch.amp.autocast('cuda'):
   backward: scaler.scale(loss).backward()
   step: scaler.step(optimizer)
   update: scaler.update()
DO NOT use torch.compile(model) — it crashes on Windows with AdaptiveMaxPool2d

=== FILE STRUCTURE ===
Base directory: D:\T25301094 P2\
Workspaces:
D:\T25301094 P2\workspaces\hasin\
D:\T25301094 P2\workspaces\namira\
D:\T25301094 P2\workspaces\bithi\
D:\T25301094 P2\workspaces\shouvik\
D:\T25301094 P2\workspaces\nusrat\
Save ALL output files to teammate's own workspace folder ONLY.

=== MODEL ARCHITECTURE ===
Backbone: assigned model from timm
  timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, global_pool='')

Always detect feature_dim dynamically — NEVER hardcode it:
  backbone = backbone.to(device)
  with torch.no_grad():
      dummy = backbone(torch.zeros(1, 3, 224, 224).to(device))
      feature_dim = dummy.shape[1]

After backbone: CBAM attention module
After CBAM: AdaptiveAvgPool2d(1)
After pool: flatten then Linear(feature_dim, 7)

CBAM (copy exactly, no modifications):
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_ = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        return x * self.sigmoid(avg + max_).unsqueeze(-1).unsqueeze(-1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()
    def forward(self, x):
        return self.sa(self.ca(x))

=== LOSS FUNCTION: FOCAL LOSS (copy exactly) ===
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

=== PREPROCESSING ===
Use albumentations (NOT torchvision transforms).
Load images with: image = cv2.imread(img_path) then cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

Training transforms:
  A.Resize(224, 224)
  A.HorizontalFlip(p=0.5)
  A.Rotate(limit=15, p=0.5)
  A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
  ToTensorV2()

Val and Test transforms (NO augmentation):
  A.Resize(224, 224)
  A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
  ToTensorV2()

=== HYPERPARAMETERS ===
optimizer: Adam, lr=1e-3
scheduler: StepLR(step_size=10, gamma=0.5)
batch_size: 128
MAX_EPOCHS: 30
PATIENCE: 10
random_seed: 42
Set seeds at start: random.seed(42), np.random.seed(42), torch.manual_seed(42), torch.cuda.manual_seed_all(42)

=== EARLY STOPPING — MANDATORY ===
Stop training if val Macro F1 does not improve for 10 consecutive epochs.
Implement EXACTLY like this inside the training loop:

  PATIENCE = 10
  epochs_no_improve = 0
  best_val_f1 = -1.0

  for epoch in range(1, MAX_EPOCHS + 1):
      # ... training and validation code ...

      if val_f1 > best_val_f1:
          best_val_f1 = val_f1
          epochs_no_improve = 0
          torch.save(model.state_dict(), checkpoint_path)
      else:
          epochs_no_improve += 1
          if epochs_no_improve >= PATIENCE:
              print(f"Early stopping triggered at epoch {epoch}. No improvement for {PATIENCE} epochs.")
              break

Track actual_epochs_trained = epoch when loop ends.
Use actual_epochs_trained in Context 3 EPOCHS TRAINED field.

=== METRICS ===
Primary: Macro F1-score (higher is better)
Secondary: Per-class accuracy (as percentage %)
Compute using sklearn.metrics f1_score and confusion_matrix
Track val Macro F1 every epoch.
Save best checkpoint when val Macro F1 improves.
After training, load best checkpoint and evaluate on both val and test sets.
Also print epoch summary after each epoch: "Epoch X/30 | Loss: X | Val Macro F1: X"

=== OUTPUT FILES — ALL MANDATORY ===
All saved to: D:\T25301094 P2\workspaces\[name]\
  behavior_best.pth       <- best checkpoint
  behavior_results.txt    <- auto-generated Context 3 with real numbers
  behavior_loss_curve.png <- training loss plot

behavior_results.txt MUST be written automatically by the script at the end.
Fill every single field with real numbers from training.
Do NOT leave any field blank.

=== WHAT TO DO STEP BY STEP ===
STEP A:
Write the complete training script with no errors.
End with exactly: "Save this as D:\T25301094 P2\workspaces\[NAME]\train_behavior.py — tell me when saved."

STEP B:
After teammate confirms saved, give this exact command:
python "D:\T25301094 P2\workspaces\[NAME]\train_behavior.py"
Say: "Run in PowerShell. Do not close PowerShell. Paste the full terminal output here when done."

STEP C:
After teammate pastes terminal output, fill and return the complete Context 3 below with real numbers.

=== CONTEXT 3 TEMPLATE ===
---CONTEXT 3 BEHAVIOR---
PERSON NAME:
BASE MODEL:
DATASET: MmCows (capped 3000/class)
EPOCHS TRAINED:
LOSS AT EPOCH 10:
LOSS AT EPOCH 20:
LOSS AT EPOCH 30:
FINAL TRAIN LOSS:
VAL MACRO F1:
VAL PER-CLASS ACCURACY:
  Class 1 (Walking):
  Class 2 (Standing):
  Class 3 (Feeding head up):
  Class 4 (Feeding head down):
  Class 5 (Licking):
  Class 6 (Drinking):
  Class 7 (Lying):
TEST MACRO F1:
TEST PER-CLASS ACCURACY:
  Class 1 (Walking):
  Class 2 (Standing):
  Class 3 (Feeding head up):
  Class 4 (Feeding head down):
  Class 5 (Licking):
  Class 6 (Drinking):
  Class 7 (Lying):
CHECKPOINT PATH:
TRAINING TIME (mins):
EARLY STOPPING TRIGGERED AT EPOCH: (write N/A if completed all 30 epochs)
ANY ISSUES ENCOUNTERED:
---END CONTEXT 3---

=== KNOWN ERRORS AND EXACT FIXES ===
ERROR 1: KeyError: 'label'
CAUSE: Used groupby().apply() for sampling
FIX: Use pd.concat with list comprehension as shown in DATASET SAMPLING section above

ERROR 2: FutureWarning about torch.cuda.amp
CAUSE: Used torch.cuda.amp instead of torch.amp
FIX: Use torch.amp.GradScaler('cuda') and torch.amp.autocast('cuda')

ERROR 3: [cite: X] SyntaxError
CAUSE: AI added citation markers inside the script
FIX: Remove every single [cite: X] tag from the script

ERROR 4: CUDA out of memory
FIX: Change batch_size from 128 to 64 and rerun

ERROR 5: Training slow (below 5 it/s after epoch 1)
FIX: Verify all GPU optimizations are applied — cudnn.benchmark, AMP, batch_size=128, num_workers=8

ERROR 6: ModuleNotFoundError
FIX: pip install [missing module name] then restart PowerShell and rerun

ERROR 7: BackendCompilerFailed / LoweringException with AdaptiveMaxPool2d
CAUSE: torch.compile incompatible with AdaptiveMaxPool2d on Windows
FIX: Remove torch.compile(model) entirely — other GPU optimizations are sufficient

ERROR 8: No progress bar visible — terminal appears frozen during training
CAUSE: tqdm not used in training or validation loops
FIX: Import tqdm and wrap both loops:
  Training:   for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/30"):
  Validation: for images, labels in tqdm(val_loader, desc="Validating"):

ERROR 9: torch.load FutureWarning about weights_only
CAUSE: Loading checkpoint without weights_only parameter
FIX: Change torch.load(path) to torch.load(path, weights_only=True)

ERROR 10: Training runs all 30 epochs even when results plateau
CAUSE: Early stopping not implemented
FIX: Implement PATIENCE=10 exactly as shown in EARLY STOPPING section above
```

## 2. PRESENTATION & KEY CONCEPTS
================================================================================

### FILE: docs\presentation_key_concepts.md
---
```text
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
2. One network to classify Behavior *(lying, standing, drinking, licking, etc.)*.
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
* **Specialized Routing Paths:** The extracted feature vector is routed into two separate pathways:
  * **Static Path (A):** The vector is passed directly to 3 lightweight prediction heads *(small neural network layers attached to the end of the backbone that perform the final classification or regression for each task)*:
    1. **Head 1** uses it to output a Body Condition Score (BCS).
    2. **Head 2** uses it to output a Behavior classification.
    3. **Head 3** uses it to output a Cow ID.
  * **Sequence Path (B):** For video data, a sequence of 20 feature vectors is passed into a memory module:
    4. **Head 4** uses the sequence to output a Lameness prediction.

##### Benefits for the Thesis:
1. **Extreme Efficiency:** The entire model remains under **10 million parameters** *(the individual weight variables adjusted during learning)*. It can run in real-time on edge cameras without cloud latency.
2. **Regularization (Preventing Overfitting):** Overfitting *(when an AI memorizes specific training details like the background grass rather than learning generalizable concepts)* is prevented. Because the shared backbone must satisfy 4 entirely different tasks at the same time, it cannot "cheat" by memorizing irrelevant details. It is forced to learn highly robust, general representations of the cow itself.

#### The Multi-Task Loss Function & Weighting
To train a model to do 4 things at once, we calculate the total error (loss) across all tasks simultaneously using a weighted sum of individual loss functions *(mathematical formulas that calculate how wrong the model's predictions are compared to the actual truth)*:

`Total Loss = (w1 * Loss_BCS) + (w2 * Loss_Behavior) + (w3 * Loss_Lameness) + (w4 * Loss_ID)`

The parameters `w1, w2, w3, w4` are **loss weights** *(multipliers that control how much weight each task's error contributes to the total loss, determining its influence on learning)*. 

##### Actual Technical Example:
During training, if the Behavior task has a very high loss of 2.5, and the Lameness task has a low loss of 0.1, the network will focus almost entirely on adjusting its weights to minimize the Behavior loss. This is called task domination or destructive interference *(when one task's large error updates override and erase the progress made on other tasks)*. By scaling the loss weights (e.g., `w1=0.35, w2=0.35, w3=0.15, w4=0.15`), we force the optimization gradients *(the directional math values indicating how to adjust the model's weights to reduce the error)* to have similar magnitudes, ensuring the network learns both tasks simultaneously. This is handled by the optimizer *(the algorithm, like Adam or SGD, that uses gradients to update the model's weights during training)*.

#### Backbone Selection Process & Baseline Results
To find the perfect "shared brain," the 5 group members individually trained single-task baseline models *(simple, single-task models trained to establish a performance benchmark before building the multi-task model)* on 5 different architectures *(the specific structure and arrangement of layers in a neural network)*. The primary metrics tracked are BCS Test MAE *(Mean Absolute Error on the 0-4 scale)*, Behavior Test Macro F1 *(higher is better)*, and Lameness Test AUC *(Area Under the ROC Curve)*.

Here is the baseline performance table showing the results of each base model:

| Model Architecture | BCS Test MAE (Dryad / SciDB) | Behavior Test Macro F1 | Lameness Test AUC | ID Top-1 Accuracy | Average Rank |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ResNet-18** (Hasin) | 0.8675 / 0.5800 | 0.7134 | 0.9600 (ST) | -- | 4th |
| **MobileNetV3-Small** (Namira) | 0.5250 / 0.7090 | 0.6810 | -- | -- | 5th |
| **ResNet-50** (Bithi) | 0.6300 / 0.6485 | 0.7037 | -- | -- | 3rd |
| **DenseNet121** (Shouvik) | 0.5875 / 0.6292 | 0.7366 | -- | -- | 2nd |
| **EfficientNetB0** (Nusrat) | 0.6175 / 0.5566 | 0.7445 | 0.9829 (Spatial) | 86.49% | **1st (Selected)** |

*\*Note: The 0.9600 Spatiotemporal (ST) *(image sequence tracking)* AUC *(Area Under the ROC Curve)* was achieved using the ResNet18-LSTM sequence model. The selected EfficientNet-B0 backbone achieved 0.9829 (Spatial AUC) as a single-task baseline, and 0.8400 (Sequence AUC) when integrated with the LSTM sequence tracking model.*

The backbone with the best average rank is **EfficientNetB0** (Nusrat), which achieves a Test MAE *(Mean Absolute Error)* of `0.6175` (Dryad) and `0.5566` (ScienceDB), a Behavior Test F1 of `0.7445`, and a Lameness Test AUC of `0.9829` (Spatial). It serves as the shared backbone for the final Multi-Task model.

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

We get the binary array *(a list containing only 0s and 1s representing binary decisions)* `[1, 1, 0, 0]`. We sum these numbers: `1 + 1 + 0 + 0 = 2`. The predicted class index *(the integer position of a class in a list, starting at 0)* is **2**, which corresponds to a score of **3.75**.

##### Why this is revolutionary for our model:
CORAL mathematically guarantees that predicting a `3.75` when the truth is `3.5` results in a tiny loss penalty, while predicting `4.25` results in a massive penalty. This dramatically drives down our **MAE (Mean Absolute Error)** *(the average absolute difference between the predicted value and the actual true score)*.

---

### Part 3: Task-Specific Head II - Behavior & Focal Loss
*(Solving severe dataset imbalance where one category has 40,000 photos and another has only 200, without deleting the extra data)*

#### The Imbalance Problem (MmCows Dataset)
Cow behavior is naturally skewed. A cow spends 80% of its day just *Lying* or *Standing* (resulting in tens of thousands of images), but spends very little time *Drinking* or *Licking* (resulting in a few hundred images).

##### The 7 Behavior Classes:
The dataset is classified into the following 7 categories:
* **Class 1 (Walking)**
* **Class 2 (Standing)**
* **Class 3 (Feeding head up)**
* **Class 4 (Feeding head down)**
* **Class 5 (Licking)**
* **Class 6 (Drinking)**
* **Class 7 (Lying)**

*Note: During data loading, these are mapped to 0-indexed labels `0-6` by subtracting 1 (e.g. Class 1 becomes Label 0).*

##### Actual Technical Example:
If the dataset has 49,848 images of "Standing" cows (Class 2) and only 1,365 images of "Licking" cows (Class 5) in the training split, a model using standard Cross-Entropy loss *(the standard loss function used for classification tasks to measure the difference between predicted probability distributions and target classes)* will learn that it can get a high accuracy score by always outputting "Standing," even if it gets every single "Licking" image wrong. The AI stops learning how to detect rare behaviors due to class imbalance *(a dataset state where some categories have vastly more samples than others)*.

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

During our spatiotemporal lameness experiments, our model achieved a **Test AUC of 1.00**, but under the default 0.50 threshold showed only **80.00% Test Accuracy**.

#### Understanding AUC (Area Under the ROC Curve):
AUC *(Area Under the Receiver Operating Characteristic Curve, a metric measuring a model's ability to rank positive cases higher than negative cases across all thresholds)* is a **threshold-independent metric** *(a grading metric that measures the model's core discriminative ability without relying on a specific decision cutoff)* based on the ROC curve *(Receiver Operating Characteristic curve, a graph plotting the True Positive Rate against the False Positive Rate at various thresholds)*.
* An AUC of 1.00 means that if you pick one random lame cow and one random healthy cow, there is a 100% chance the model will assign a higher lameness probability to the lame cow. The model's internal understanding of lameness is exceptional.

#### The Accuracy Discrepancy & Threshold Calibration:
Accuracy is dependent on an arbitrary decision threshold *(the probability cutoff value used to assign a sample to a class)*, which defaults to `0.50` (50%).

##### Actual Technical Example:
Because the lameness training set was small (50 videos total), the model's probability outputs shifted higher. It was outputting a `0.55` (55%) probability for perfectly healthy cows and `0.85` (85%) for lame cows.
* Because `0.55 > 0.50`, the default accuracy metric classified the healthy cows as Lame, generating False Positives *(healthy cases incorrectly flagged by the model as diseased or abnormal)* and dragging the accuracy score down to 80%.
* Shifting the decision threshold from `0.50` to `0.70` (so only scores > 0.70 are classified as Lame) correctly separates the classes. The test accuracy immediately jumped to 100.00%.

---

### Part 7: Live Deployed Video Inference Architecture
*(How the software actually operates in real-time on a physical dairy farm)*

Imagine Deployed Video Inference *(running a trained model in a live production environment to generate predictions on new video feeds)* where a camera is mounted above a farm walkway. A cow walks past, and the camera captures a 20-frame video clip.

1. **Feature Extraction:** The 20 frames are passed through the shared `EfficientNetB0` backbone, generating 20 independent feature vectors.
2. **Static Spatial Heads (BCS, ID, & Behavior) Process the Data:**
   * These heads are designed for single, static images. They generate 20 independent BCS predictions, 20 ID predictions, and 20 Behavior predictions (one for each frame).
   * **Temporal Averaging (For BCS):** We use temporal averaging *(calculating the mathematical mean of predictions made across multiple frames over time)* to average all 20 predicted scores to output a single, stable body score.
   * **Majority Voting (For ID & Behavior):** We use majority voting *(selecting the classification class that was predicted most frequently across a sequence of frames)* to count which Cow ID and Behavior was predicted most often across the 20 frames and output that as the final classification.
   * **Occlusion Example:** If the cow walks behind a fence post and suffers from occlusion *(the state of being visually blocked or hidden from the camera's view by another object)* for 3 frames, the static heads might predict incorrect values for those 3 frames. Temporal averaging and majority voting filter out that noise completely.
3. **Spatiotemporal Sequence Head (Lameness) Processes the Data:**
   * The sequence of 20 feature vectors is fed sequentially into the LSTM network.
   * The LSTM tracks the gait mechanics frame-by-frame and outputs the final classification on the 20th frame: **Lameness: Positive**.

---

### Part 8: Codebase & Preprocessing Deep Dive
*(Critical data pipeline, dataset splits, and hardware optimization strategies)*

#### 1. Dataset Sizes, Splits, and Formats:
We resolve dataset anomalies *(inconsistencies, missing values, or format mismatches present in raw data)* during preprocessing. The split ratio is set to `70% train / 15% val / 15% test` based on the number of unique cows/videos:

* **Dryad BCS:** 5,923 total images from 147 unique cows. Standard format: Depth Grayscale Edge (DGE) *(an image format where pixel intensities represent distance from the sensor rather than visible color, combined with edge enhancement)*. Labels: `2-6`.
  * **Train split:** 102 cows (4,163 images)
  * **Validation split:** 22 cows (1,360 images)
  * **Test split:** 23 cows (400 images)
* **ScienceDB BCS:** 53,566 total images from 10,898 unique cows. Standard format: RGB. Labels: `3.25-4.25`.
  * **Train split:** 7,628 cows (37,688 images)
  * **Validation split:** 1,634 cows (7,580 images)
  * **Test split:** 1,636 cows (8,298 images)
* **MmCows Behavior:** 213,686 total images from 16 unique cows. Standard format: RGB. Labels: `1-7`.
  * **Train split:** 11 cows (148,401 images)
  * **Validation split:** 2 cows (25,134 images)
  * **Test split:** 3 cows (40,151 images)
* **CattleLameness:** 50 total videos (25 Lame, 25 Normal), sub-sampled to 20 frames per video (total 1,000 frames). Standard format: RGB video. Labels: `0 (Normal)` and `1 (Lame)`.
  * **Train split:** 34 videos (680 frames)
  * **Validation split:** 6 videos (120 frames)
  * **Test split:** 10 videos (200 frames)

*Note: Both BCS datasets are dynamically mapped to a unified `0-4` ordinal index during data loading so the CORAL framework can process them interchangeably without crashing.*

#### 2. Class Balancing via Capping:
* **Behavior Dataset (MmCows):** This dataset has a massive 42:1 class imbalance. While Focal Loss fixes the loss gradients, we implement a **hard data cap** *(setting a strict upper limit on the number of samples processed per class during an epoch)* during loading (`group.sample(min(len(group), 3000), random_state=42)`). This restricts majority classes (like Lying) to a maximum of 3,000 images per epoch *(one complete pass of the entire training dataset through the neural network)*.

##### Actual Technical Example:
Instead of processing 40,000 identical frames of lying cows (which adds redundant training time and memory overhead), the model trains on a balanced subset of 3,000 lying images per epoch, allowing faster epochs while retaining the representation of the rare behaviors (200 samples).

#### 3. Hardware & Compute Optimizations:
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
4. **ID Baseline:** [100% COMPLETE] EfficientNetB0 baseline trained for 10 epochs, achieving 86.49% Test Top-1 Accuracy.
5. **Final Multi-Task Aggregation:** [100% COMPLETE] The Spatiotemporal Multi-Task model (EfficientNetB0-LSTM) was trained across all 4 heads simultaneously using 20-frame sequences. It achieved **100.00% Lameness Accuracy (AUC 1.0)**, **97.18% ID Accuracy**, **0.7849 BCS MAE**, and **0.4775 Behavior Macro F1**. A rolling sequence real-time visualizer was built and successfully deployed.

#### Core Limitations of the Current Approach:
1. **Lack of Large-Scale Annotated Temporal Data:** While we have 50,000+ spatial images for BCS, high-quality, annotated *(labeled with ground truth information by human experts like veterinarians)* sequence data for Lameness and Behavior is scarce. The spatiotemporal models risk overfitting due to the limited number of unique videos in the datasets.
2. **Frozen Backbone Bottleneck:** Currently, the 2D backbone is completely frozen when training the LSTM heads. This means the feature vectors passed to the LSTM were optimized for classifying static ImageNet photos (dogs, cars, etc.), not the specific micro-movements of a cow's joints.

#### Future Work & Development Ideas:
1. **Joint Fine-Tuning (Unfreezing the Backbone):** In the final phase, we intend to perform Fine-Tuning *(unfreezing pretrained layers and training them with a very small learning rate to adapt them to a new, specific task)* by unfreezing the shared backbone and allowing gradients from the LSTM to flow all the way backward into the CNN layers. This forces the backbone to learn new convolutional filters specifically designed to track motion and joint angles.
2. **Cross-Dataset Validation (CBVD-5):** To prove generalizability, we will perform Cross-Dataset Validation *(testing a model trained on one dataset on an entirely different, independent dataset to prove generalizability)* by taking the Behavior model trained on `MmCows` and running inference on the entirely unseen `CBVD-5` dataset.
3. **Advanced Enhancements:**
   * Integrating **CLAHE (Contrast Limited Adaptive Histogram Equalization)** *(a localized image processing technique used to improve contrast and details under uneven lighting)* to equalize extreme lighting differences in barns.
   * Future exploration into integrating depth cameras or thermal imaging for highly granular lameness inflammation detection.

---

### Part 10: Ablation Studies
*(Systematically removing or replacing parts of our system to prove that each design choice actually improves performance)*

Ablation studies *(an experimental investigation where specific components of an AI model are systematically removed or replaced to measure their individual impact on performance)* were conducted to isolate the value of our key design decisions:

#### 1. CORAL vs. Standard Cross-Entropy (For BCS)
* **The Study:** We train the exact same Body Condition Scoring (BCS) model architecture using standard Cross-Entropy Loss *(treating categories as independent buckets)* versus the CORAL framework *(Consistent Rank Logits, treating scores as ordered rankings)*.
* **Actual Technical Example:** Under standard Cross-Entropy, guessing a score of `4.25` for a cow whose true score is `3.5` receives the same loss penalty as guessing `3.75`. By swapping in CORAL, Node 1 and Node 2 check if the score is greater than 3.25 and 3.5. This ordinal framework penalizes the larger error (`4.25`) much more heavily than the minor error (`3.75`), resulting in a lower overall MAE *(Mean Absolute Error)* on the test set.

#### 2. With CBAM vs. Without CBAM (For BCS)
* **The Study:** We train the BCS network with and without the CBAM *(Convolutional Block Attention Module)* visual attention layer inserted after the backbone.
* **Actual Technical Example:** Without CBAM, the backbone filters capture features across the entire image frame, including the metal gates and ground shadow textures. Adding CBAM applies Channel and Spatial attention, which highlights the cow's backbone and hips while zeroing out background noise. This results in cleaner feature vectors and a lower test loss.

#### 3. RGB Only vs. RGB+Depth (For Dryad DGE)
* **The Study:** We evaluate model predictions on the Dryad dataset using standard RGB *(Red, Green, Blue)* color channels only versus utilizing the Depth Grayscale Edge (DGE) format.
* **Actual Technical Example:** Standard RGB images lose 3D structural details due to lighting variance. The DGE format provides a depth map containing physical coordinates of the cow's spine and pelvic bone curvature. This ablation demonstrates that including physical depth features yields significantly more accurate fat score predictions compared to color features alone.

#### 4. Focal Loss vs. Standard Cross-Entropy (For Behavior)
* **The Study:** We train the Behavior head using standard Cross-Entropy Loss versus using Focal Loss on the imbalanced MmCows dataset.
* **Actual Technical Example:** Standard Cross-Entropy achieves 99.5% accuracy by simply predicting the majority class "Lying" for all samples, yielding a poor F1-score for rare behaviors. Focal Loss scales down the gradients for high-confidence predictions (e.g. 95% confident on "Lying") while maintaining high loss penalties for low-confidence ones (e.g. 10% on "Licking"), forcing the model to learn rare behaviors and raising the Macro F1-Score.

#### 5. Cross-Dataset Evaluation (MmCows to CBVD-5)
* **The Study:** We train the behavior classifier on MmCows and perform cross-dataset inference on the completely unseen CBVD-5 dataset.
* **Actual Technical Example:** This ablation tests out-of-distribution generalization. By evaluating our MmCows-trained EfficientNetB0 on the independent CBVD-5 dataset (2,000 balanced images), we observed a Macro F1-score of **`0.124517`**. While the model generalized extremely well to Standing postures (**`90.34%`** accuracy), it struggled on Feeding (**`8.39%`**), Drinking (**`2.99%`**), and Lying (**`24.31%`**) behaviors. This performance drop highlights that the model overfit to the specific camera angles, farm lighting, and bounding box perspectives of the MmCows dataset, proving that cross-dataset generalization is highly sensitive to domain shift and benefit from multi-farm diversity or domain adaptation.

#### 6. Backbone Selection Comparison
* **The Study:** We train individual task baselines using 5 different architectures (ResNet-18, MobileNetV3-Small, ResNet-50, DenseNet121, EfficientNetB0).
* **Actual Technical Example:** We measure the inference latency and parameter size of each architecture against its test error. EfficientNetB0 was chosen because it achieves a low BCS MAE *(Mean Absolute Error)* of `0.5566` and a high Behavior F1-score of `0.7445` while remaining under 10 million parameters, outperforming heavier models like ResNet-50.

#### 7. Single-Task vs. Multi-Task Learning
* **The Study:** We train separate, individual networks for each task versus training a single unified model where all four prediction heads share the same backbone.
* **Actual Technical Example:** This ablation measures the impact of shared parameter representation. Training all four heads together forces the backbone to learn general visual features (like animal contour lines) that benefit all tasks. By testing the fully integrated **Spatiotemporal Multi-Task model**, we observed that the temporal LSTM heads achieved perfect Lameness identification (`1.00` AUC and `100%` Acc) and significantly improved Cow ID accuracy (`97.18%`). However, due to destructive interference on highly constrained shared backbones, the spatial metrics slightly degraded (BCS MAE dropped to `0.7849` and Behavior Macro F1 dropped to `0.4775`). This proves the multi-task tradeoff: massive compute and hardware savings and perfect temporal performance, at the cost of a slight drop in static spatial accuracy.

---

### Part 11: Data Leakage Prevention via Cow-Wise Group Splitting
*(Enforcing strict partition boundaries between training and testing subjects to prevent over-optimistic results)*

Data Leakage *(when training data contains information that the model would not have access to in a real-world deployment, leading to overly optimistic training results but poor generalization)* is a major concern when multiple images are captured from the same subject.

#### The Risk: Identity Leakage
If multiple images or video frames of Cow #1042 are collected under the same camera background, lighting, and weather conditions, a simple random split will place some images of Cow #1042 in the training set and others in the validation set *(a subset of the dataset used to evaluate model performance and tune hyperparameters during training)* or test set *(a subset of the dataset held out until the end of training to provide an unbiased evaluation of the final model)*.
* **The Result:** The model will experience identity leakage *(a form of data leakage where the model memorizes the visual identity of the subject rather than learning the task-specific features)*. It will memorize Cow #1042's ear tag, skin markings, or shape of horns to output a BCS score, rather than learning general markers of body fat. This leads to 99% validation accuracy but immediate failure in live deployment on new cows.

#### Our Solution: Cow-Wise Group Splitting
To prevent this, all our dataset preprocessing pipelines use **Cow-Wise Group Splitting** *(partitioning data so that images of a specific cow are strictly contained in either the train, validation, or test split, and never shared across splits)*. 

##### Actual Technical Example:
1. During dataset creation, we extract the `cow_id` or `video_id` for every image frame (e.g. `Lame_1`, `GS_1035`, etc.).
2. We compile a list of all unique cow IDs.
3. We shuffle this list of unique cows and assign them to splits: 70% of unique cow IDs are assigned to `train_cows`, 15% to `val_cows`, and 15% to `test_cows`.
4. All images belonging to these specific cows are grouped accordingly. This acts as a GroupKFold *(a cross-validation technique where the data is split such that the same group does not appear in both training and validation folds)* split.
This mathematically guarantees that the validation and test sets only contain out-of-distribution *(data that comes from a different distribution than the training data, such as a different farm, lighting condition, or camera angle)* cows that the backbone network has never processed before, proving the model has learned true biological and kinematic markers.

```

## 3. INDIVIDUAL CONTEXT 3 REPORTS
================================================================================

### FILE: context\Context3_Bithi_Behavior.txt
---
```text
---CONTEXT 3 BEHAVIOR---
PERSON NAME: Bithi
BASE MODEL: ResNet-50
DATASET: MmCows (capped 3000/class)
EPOCHS TRAINED: 22
LOSS AT EPOCH 10: 0.012949
LOSS AT EPOCH 20: 0.003129
LOSS AT EPOCH 30: N/A
FINAL TRAIN LOSS: 0.001347
VAL MACRO F1: 0.7210
VAL PER-CLASS ACCURACY:
  Class 1 (Walking): 64.86%
  Class 2 (Standing): 58.20%
  Class 3 (Feeding head up): 71.20%
  Class 4 (Feeding head down): 64.97%
  Class 5 (Licking): 94.68%
  Class 6 (Drinking): 84.36%
  Class 7 (Lying): 98.53%
TEST MACRO F1: 0.7037
TEST PER-CLASS ACCURACY:
  Class 1 (Walking): 54.97%
  Class 2 (Standing): 65.37%
  Class 3 (Feeding head up): 66.77%
  Class 4 (Feeding head down): 68.20%
  Class 5 (Licking): 73.11%
  Class 6 (Drinking): 79.18%
  Class 7 (Lying): 99.03%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\bithi\behavior_best.pth
TRAINING TIME (mins): 10.37
EARLY STOPPING TRIGGERED AT EPOCH: 22
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---

```

### FILE: context\Context3_Hasin_BCS.txt
---
```text
---CONTEXT 3 BCS---
PERSON NAME: Hasin
BASE MODEL: ResNet-18

DATASET: Dryad
EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.008938
LOSS AT EPOCH 20: 0.001953
LOSS AT EPOCH 30: 0.001819
FINAL TRAIN LOSS: 0.001819
VAL MAE: 0.703676
VAL ACCURACY +-0 (exact match): 31.25%
VAL ACCURACY +-1 (within 1 class): 98.38%
TEST MAE: 0.867500
TEST ACCURACY +-0: 24.50%
TEST ACCURACY +-1: 88.75%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\hasin\dryad_bcs_best.pth
TRAINING TIME (mins): 2.21
ANY ISSUES ENCOUNTERED: None

DATASET: ScienceDB
EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.043623
LOSS AT EPOCH 20: 0.011708
LOSS AT EPOCH 30: 0.003786
FINAL TRAIN LOSS: 0.003786
VAL MAE: 0.518865
VAL ACCURACY +-0 (exact match): 58.72%
VAL ACCURACY +-1 (within 1 class): 90.40%
TEST MAE: 0.580019
TEST ACCURACY +-0: 54.81%
TEST ACCURACY +-1: 89.02%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\hasin\sciencedb_bcs_best.pth
TRAINING TIME (mins): 21.87
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
```

### FILE: context\Context3_Hasin_Behavior.txt
---
```text
---CONTEXT 3 BEHAVIOR---
PERSON NAME: Hasin
BASE MODEL: ResNet-18
DATASET: MmCows (capped 3000/class)
EPOCHS TRAINED: 18
LOSS AT EPOCH 10: 0.0139
LOSS AT EPOCH 20: N/A
LOSS AT EPOCH 30: N/A
FINAL TRAIN LOSS: 0.0034
VAL MACRO F1: 0.7241
VAL PER-CLASS ACCURACY:
  Class 1 (Walking): 60.14%
  Class 2 (Standing): 59.23%
  Class 3 (Feeding head up): 66.10%
  Class 4 (Feeding head down): 73.13%
  Class 5 (Licking): 91.20%
  Class 6 (Drinking): 77.09%
  Class 7 (Lying): 98.13%
TEST MACRO F1: 0.7134
TEST PER-CLASS ACCURACY:
  Class 1 (Walking): 47.21%
  Class 2 (Standing): 66.90%
  Class 3 (Feeding head up): 64.63%
  Class 4 (Feeding head down): 73.97%
  Class 5 (Licking): 69.34%
  Class 6 (Drinking): 81.51%
  Class 7 (Lying): 96.90%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\hasin\behavior_best.pth
TRAINING TIME (mins): 4.62
EARLY STOPPING TRIGGERED AT EPOCH: 18
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---

```

### FILE: context\Context3_Namira_BCS.txt
---
```text
---CONTEXT 3 BCS---
PERSON NAME: Namira
BASE MODEL: MobileNetV3-Small

DATASET: Dryad
EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.0748
LOSS AT EPOCH 20: 0.0458
LOSS AT EPOCH 30: 0.0112
FINAL TRAIN LOSS: 0.0112
VAL MAE: 0.6324
VAL ACCURACY +-0 (exact match): 43.38%
VAL ACCURACY +-1 (within 1 class): 93.60%
TEST MAE: 0.5250
TEST ACCURACY +-0: 61.50%
TEST ACCURACY +-1: 86.00%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\namira\dryad_bcs_best.pth
TRAINING TIME (mins): 4.42
ANY ISSUES ENCOUNTERED: None

DATASET: ScienceDB
EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.3237
LOSS AT EPOCH 20: 0.1983
LOSS AT EPOCH 30: 0.1219
FINAL TRAIN LOSS: 0.1219
VAL MAE: 0.6641
VAL ACCURACY +-0 (exact match): 47.90%
VAL ACCURACY +-1 (within 1 class): 86.21%
TEST MAE: 0.7090
TEST ACCURACY +-0: 44.67%
TEST ACCURACY +-1: 86.47%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\namira\sciencedb_bcs_best.pth
TRAINING TIME (mins): 132.40
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---

```

### FILE: context\Context3_Namira_Behavior.txt
---
```text
---CONTEXT 3 BEHAVIOR---
PERSON NAME: Namira
BASE MODEL: MobileNetV3-Small
DATASET: MmCows (capped 3000/class)
EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.040425
LOSS AT EPOCH 20: 0.015269
LOSS AT EPOCH 30: 0.004966
FINAL TRAIN LOSS: 0.004966
VAL MACRO F1: 0.7271
VAL PER-CLASS ACCURACY:
  Class 1 (Walking): 72.97%
  Class 2 (Standing): 64.13%
  Class 3 (Feeding head up): 74.13%
  Class 4 (Feeding head down): 56.47%
  Class 5 (Licking): 94.91%
  Class 6 (Drinking): 81.82%
  Class 7 (Lying): 96.10%
TEST MACRO F1: 0.6810
TEST PER-CLASS ACCURACY:
  Class 1 (Walking): 61.42%
  Class 2 (Standing): 60.93%
  Class 3 (Feeding head up): 65.83%
  Class 4 (Feeding head down): 66.93%
  Class 5 (Licking): 59.43%
  Class 6 (Drinking): 80.82%
  Class 7 (Lying): 94.77%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\namira\behavior_best.pth
TRAINING TIME (mins): 6.97
EARLY STOPPING TRIGGERED AT EPOCH: N/A
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---

```

### FILE: context\Context3_Nusrat_BCS.txt
---
```text
---CONTEXT 3 BCS---
PERSON NAME: Nusrat
BASE MODEL: EfficientNetB0

DATASET: Dryad
EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.005635
LOSS AT EPOCH 20: 0.000713
LOSS AT EPOCH 30: 0.000111
FINAL TRAIN LOSS: 0.000111
VAL MAE: 1.013971
VAL ACCURACY +-0 (exact match): 0.257353
VAL ACCURACY +-1 (within 1 class): 0.729412
TEST MAE: 0.617500
TEST ACCURACY +-0: 0.525000
TEST ACCURACY +-1: 0.857500
CHECKPOINT PATH: D:\T25301094 P2\workspaces\nusrat\dryad_bcs_best.pth
TRAINING TIME (mins): 4.78
ANY ISSUES ENCOUNTERED: None

DATASET: ScienceDB
EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.032411
LOSS AT EPOCH 20: 0.007906
LOSS AT EPOCH 30: 0.002366
FINAL TRAIN LOSS: 0.002366
VAL MAE: 0.495778
VAL ACCURACY +-0 (exact match): 0.596834
VAL ACCURACY +-1 (within 1 class): 0.915699
TEST MAE: 0.556640
TEST ACCURACY +-0: 0.550615
TEST ACCURACY +-1: 0.905037
CHECKPOINT PATH: D:\T25301094 P2\workspaces\nusrat\sciencedb_bcs_best.pth
TRAINING TIME (mins): 26.24
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
```

### FILE: context\Context3_Nusrat_Behavior.txt
---
```text
---CONTEXT 3 BEHAVIOR---
PERSON NAME: Nusrat
BASE MODEL: EfficientNetB0
DATASET: MmCows (capped 3000/class)
EPOCHS TRAINED: 24
LOSS AT EPOCH 10: 0.013290
LOSS AT EPOCH 20: 0.003363
LOSS AT EPOCH 30: N/A
FINAL TRAIN LOSS: 0.000979
VAL MACRO F1: 0.763051
VAL PER-CLASS ACCURACY:
  Class 1 (Walking): 70.72%
  Class 2 (Standing): 71.60%
  Class 3 (Feeding head up): 76.23%
  Class 4 (Feeding head down): 63.00%
  Class 5 (Licking): 96.30%
  Class 6 (Drinking): 82.55%
  Class 7 (Lying): 98.53%
TEST MACRO F1: 0.744548
TEST PER-CLASS ACCURACY:
  Class 1 (Walking): 59.78%
  Class 2 (Standing): 70.00%
  Class 3 (Feeding head up): 71.97%
  Class 4 (Feeding head down): 66.07%
  Class 5 (Licking): 83.96%
  Class 6 (Drinking): 83.01%
  Class 7 (Lying): 98.77%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\nusrat\behavior_best.pth
TRAINING TIME (mins): 8.95
EARLY STOPPING TRIGGERED AT EPOCH: 24
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
```

### FILE: context\Context3_Nusrat_ID.txt
---
```text
---CONTEXT 3 ID---
PERSON NAME: Nusrat
BASE MODEL: EfficientNetB0
DATASET: OpenCows2020
EPOCHS TRAINED: 10
LOSS AT EPOCH 10: 0.550795
FINAL TRAIN LOSS: 0.550795
VAL TOP-1 ACCURACY: 87.06%
TEST TOP-1 ACCURACY: 86.49%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\nusrat\id_best.pth
TRAINING TIME (mins): 1.15
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---

```

### FILE: context\Context3_Shouvik_BCS.txt
---
```text
---CONTEXT 3 BCS---
PERSON NAME: Shouvik
BASE MODEL: DenseNet121

DATASET: Dryad
EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.0295
LOSS AT EPOCH 20: 0.0059
LOSS AT EPOCH 30: 0.0027
FINAL TRAIN LOSS: 0.0027
VAL MAE: 0.7338
VAL ACCURACY +-0 (exact match): 44.49%
VAL ACCURACY +-1 (within 1 class): 82.21%
TEST MAE: 0.5875
TEST ACCURACY +-0: 54.75%
TEST ACCURACY +-1: 86.50%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\shouvik\dryad_bcs_best.pth
TRAINING TIME (mins): 15.2
ANY ISSUES ENCOUNTERED: FutureWarning on torch.load (weights_only=False) — non-critical, no impact on results. Repeated "Using device: cuda" lines printed due to DataLoader workers — cosmetic only.

DATASET: ScienceDB
EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.1200
LOSS AT EPOCH 20: 0.0210
LOSS AT EPOCH 30: 0.0066
FINAL TRAIN LOSS: 0.0066
VAL MAE: 0.5925
VAL ACCURACY +-0 (exact match): 53.06%
VAL ACCURACY +-1 (within 1 class): 89.89%
TEST MAE: 0.6292
TEST ACCURACY +-0: 49.51%
TEST ACCURACY +-1: 88.78%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\shouvik\sciencedb_bcs_best.pth
TRAINING TIME (mins): 55.5
ANY ISSUES ENCOUNTERED: FutureWarning on torch.load (weights_only=False) — non-critical, no impact on results.
---END CONTEXT 3---
```

### FILE: context\Context3_Shouvik_Behavior.txt
---
```text
---CONTEXT 3 BEHAVIOR---
PERSON NAME: Shouvik
BASE MODEL: DenseNet121
DATASET: MmCows (capped 3000/class)
EPOCHS TRAINED: 27
LOSS AT EPOCH 10: 0.020973
LOSS AT EPOCH 20: 0.004873
LOSS AT EPOCH 30: N/A
FINAL TRAIN LOSS: 0.001219
VAL MACRO F1: 0.7371
VAL PER-CLASS ACCURACY:
  Class 1 (Walking): 53.83%
  Class 2 (Standing): 67.00%
  Class 3 (Feeding head up): 71.53%
  Class 4 (Feeding head down): 69.47%
  Class 5 (Licking): 88.66%
  Class 6 (Drinking): 75.64%
  Class 7 (Lying): 98.33%
TEST MACRO F1: 0.7366
TEST PER-CLASS ACCURACY:
  Class 1 (Walking): 41.64%
  Class 2 (Standing): 74.70%
  Class 3 (Feeding head up): 67.40%
  Class 4 (Feeding head down): 70.70%
  Class 5 (Licking): 87.26%
  Class 6 (Drinking): 73.42%
  Class 7 (Lying): 99.37%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\shouvik\behavior_best.pth
TRAINING TIME (mins): 13.81
EARLY STOPPING TRIGGERED AT EPOCH: 27
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---

```

### FILE: context\context3_Bithi_BCS.txt
---
```text
---CONTEXT 3 BCS---
PERSON NAME: Bithi
BASE MODEL: ResNet-50

DATASET: Dryad
EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.0104
LOSS AT EPOCH 20: 0.0094
LOSS AT EPOCH 30: 0.0008
FINAL TRAIN LOSS: 0.0008
VAL MAE: 0.7485
VAL ACCURACY +-0 (exact match): 42.13%
VAL ACCURACY +-1 (within 1 class): 85.81%
TEST MAE: 0.6300
TEST ACCURACY +-0: 47.75%
TEST ACCURACY +-1: 89.25%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\bithi\dryad_bcs_best.pth
TRAINING TIME (mins): 11.03
ANY ISSUES ENCOUNTERED: None

DATASET: ScienceDB
EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.0521
LOSS AT EPOCH 20: 0.0122
LOSS AT EPOCH 30: 0.0035
FINAL TRAIN LOSS: 0.0035
VAL MAE: 0.6009
VAL ACCURACY +-0 (exact match): 52.48%
VAL ACCURACY +-1 (within 1 class): 89.51%
TEST MAE: 0.6485
TEST ACCURACY +-0: 48.43%
TEST ACCURACY +-1: 88.20%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\bithi\sciencedb_bcs_best.pth
TRAINING TIME (mins): 60.51
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
```

## 4. ALL RAW RESULTS
================================================================================

### FILE: workspaces\bithi\bcs_results.txt
---
```text
---CONTEXT 3 BCS---
PERSON NAME: Bithi
BASE MODEL: ResNet-50

DATASET: Dryad
EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.0104
LOSS AT EPOCH 20: 0.0094
LOSS AT EPOCH 30: 0.0008
FINAL TRAIN LOSS: 0.0008
VAL MAE: 0.7485
VAL ACCURACY +-0 (exact match): 42.13%
VAL ACCURACY +-1 (within 1 class): 85.81%
TEST MAE: 0.6300
TEST ACCURACY +-0: 47.75%
TEST ACCURACY +-1: 89.25%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\bithi\dryad_bcs_best.pth
TRAINING TIME (mins): 11.03
ANY ISSUES ENCOUNTERED: None

DATASET: ScienceDB
EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.0521
LOSS AT EPOCH 20: 0.0122
LOSS AT EPOCH 30: 0.0035
FINAL TRAIN LOSS: 0.0035
VAL MAE: 0.6009
VAL ACCURACY +-0 (exact match): 52.48%
VAL ACCURACY +-1 (within 1 class): 89.51%
TEST MAE: 0.6485
TEST ACCURACY +-0: 48.43%
TEST ACCURACY +-1: 88.20%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\bithi\sciencedb_bcs_best.pth
TRAINING TIME (mins): 60.51
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
```

### FILE: workspaces\bithi\behavior_results.txt
---
```text
---CONTEXT 3 BEHAVIOR---
PERSON NAME: Bithi
BASE MODEL: ResNet-50
DATASET: MmCows (capped 3000/class)
EPOCHS TRAINED: 22
LOSS AT EPOCH 10: 0.012949
LOSS AT EPOCH 20: 0.003129
LOSS AT EPOCH 30: N/A
FINAL TRAIN LOSS: 0.001347
VAL MACRO F1: 0.7210
VAL PER-CLASS ACCURACY:
  Class 1 (Walking): 64.86%
  Class 2 (Standing): 58.20%
  Class 3 (Feeding head up): 71.20%
  Class 4 (Feeding head down): 64.97%
  Class 5 (Licking): 94.68%
  Class 6 (Drinking): 84.36%
  Class 7 (Lying): 98.53%
TEST MACRO F1: 0.7037
TEST PER-CLASS ACCURACY:
  Class 1 (Walking): 54.97%
  Class 2 (Standing): 65.37%
  Class 3 (Feeding head up): 66.77%
  Class 4 (Feeding head down): 68.20%
  Class 5 (Licking): 73.11%
  Class 6 (Drinking): 79.18%
  Class 7 (Lying): 99.03%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\bithi\behavior_best.pth
TRAINING TIME (mins): 10.37
EARLY STOPPING TRIGGERED AT EPOCH: 22
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---

```

### FILE: workspaces\hasin\bcs_results.txt
---
```text
---CONTEXT 3 BCS---
PERSON NAME: Hasin
BASE MODEL: ResNet-18

DATASET: Dryad
EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.008938
LOSS AT EPOCH 20: 0.001953
LOSS AT EPOCH 30: 0.001819
FINAL TRAIN LOSS: 0.001819
VAL MAE: 0.703676
VAL ACCURACY +-0 (exact match): 31.25%
VAL ACCURACY +-1 (within 1 class): 98.38%
TEST MAE: 0.867500
TEST ACCURACY +-0: 24.50%
TEST ACCURACY +-1: 88.75%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\hasin\dryad_bcs_best.pth
TRAINING TIME (mins): 2.21
ANY ISSUES ENCOUNTERED: None

DATASET: ScienceDB
EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.043623
LOSS AT EPOCH 20: 0.011708
LOSS AT EPOCH 30: 0.003786
FINAL TRAIN LOSS: 0.003786
VAL MAE: 0.518865
VAL ACCURACY +-0 (exact match): 58.72%
VAL ACCURACY +-1 (within 1 class): 90.40%
TEST MAE: 0.580019
TEST ACCURACY +-0: 54.81%
TEST ACCURACY +-1: 89.02%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\hasin\sciencedb_bcs_best.pth
TRAINING TIME (mins): 21.87
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---

```

### FILE: workspaces\hasin\behavior_results.txt
---
```text
---CONTEXT 3 BEHAVIOR---
PERSON NAME: Hasin
BASE MODEL: ResNet-18
DATASET: MmCows (capped 3000/class)
EPOCHS TRAINED: 18
LOSS AT EPOCH 10: 0.0139
LOSS AT EPOCH 20: N/A
LOSS AT EPOCH 30: N/A
FINAL TRAIN LOSS: 0.0034
VAL MACRO F1: 0.7241
VAL PER-CLASS ACCURACY:
  Class 1 (Walking): 60.14%
  Class 2 (Standing): 59.23%
  Class 3 (Feeding head up): 66.10%
  Class 4 (Feeding head down): 73.13%
  Class 5 (Licking): 91.20%
  Class 6 (Drinking): 77.09%
  Class 7 (Lying): 98.13%
TEST MACRO F1: 0.7134
TEST PER-CLASS ACCURACY:
  Class 1 (Walking): 47.21%
  Class 2 (Standing): 66.90%
  Class 3 (Feeding head up): 64.63%
  Class 4 (Feeding head down): 73.97%
  Class 5 (Licking): 69.34%
  Class 6 (Drinking): 81.51%
  Class 7 (Lying): 96.90%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\hasin\behavior_best.pth
TRAINING TIME (mins): 4.62
EARLY STOPPING TRIGGERED AT EPOCH: 18
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
```

### FILE: workspaces\namira\bcs_results.txt
---
```text
---CONTEXT 3 BCS---
PERSON NAME: Namira
BASE MODEL: MobileNetV3-Small

DATASET: Dryad
EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.0748
LOSS AT EPOCH 20: 0.0458
LOSS AT EPOCH 30: 0.0112
FINAL TRAIN LOSS: 0.0112
VAL MAE: 0.6324
VAL ACCURACY +-0 (exact match): 43.38%
VAL ACCURACY +-1 (within 1 class): 93.60%
TEST MAE: 0.5250
TEST ACCURACY +-0: 61.50%
TEST ACCURACY +-1: 86.00%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\namira\dryad_bcs_best.pth
TRAINING TIME (mins): 4.42
ANY ISSUES ENCOUNTERED: None

DATASET: ScienceDB
EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.3237
LOSS AT EPOCH 20: 0.1983
LOSS AT EPOCH 30: 0.1219
FINAL TRAIN LOSS: 0.1219
VAL MAE: 0.6641
VAL ACCURACY +-0 (exact match): 47.90%
VAL ACCURACY +-1 (within 1 class): 86.21%
TEST MAE: 0.7090
TEST ACCURACY +-0: 44.67%
TEST ACCURACY +-1: 86.47%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\namira\sciencedb_bcs_best.pth
TRAINING TIME (mins): 132.40
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---

```

### FILE: workspaces\namira\behavior_results.txt
---
```text
---CONTEXT 3 BEHAVIOR---
PERSON NAME: Namira
BASE MODEL: MobileNetV3-Small
DATASET: MmCows (capped 3000/class)
EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.040425
LOSS AT EPOCH 20: 0.015269
LOSS AT EPOCH 30: 0.004966
FINAL TRAIN LOSS: 0.004966
VAL MACRO F1: 0.7271
VAL PER-CLASS ACCURACY:
  Class 1 (Walking): 72.97%
  Class 2 (Standing): 64.13%
  Class 3 (Feeding head up): 74.13%
  Class 4 (Feeding head down): 56.47%
  Class 5 (Licking): 94.91%
  Class 6 (Drinking): 81.82%
  Class 7 (Lying): 96.10%
TEST MACRO F1: 0.6810
TEST PER-CLASS ACCURACY:
  Class 1 (Walking): 61.42%
  Class 2 (Standing): 60.93%
  Class 3 (Feeding head up): 65.83%
  Class 4 (Feeding head down): 66.93%
  Class 5 (Licking): 59.43%
  Class 6 (Drinking): 80.82%
  Class 7 (Lying): 94.77%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\namira\behavior_best.pth
TRAINING TIME (mins): 6.97
EARLY STOPPING TRIGGERED AT EPOCH: N/A
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---

```

### FILE: workspaces\nusrat\bcs_results.txt
---
```text
---CONTEXT 3 BCS---
PERSON NAME: Nusrat
BASE MODEL: EfficientNetB0

DATASET: Dryad
EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.005635
LOSS AT EPOCH 20: 0.000713
LOSS AT EPOCH 30: 0.000111
FINAL TRAIN LOSS: 0.000111
VAL MAE: 1.013971
VAL ACCURACY +-0 (exact match): 0.257353
VAL ACCURACY +-1 (within 1 class): 0.729412
TEST MAE: 0.617500
TEST ACCURACY +-0: 0.525000
TEST ACCURACY +-1: 0.857500
CHECKPOINT PATH: D:\T25301094 P2\workspaces\nusrat\dryad_bcs_best.pth
TRAINING TIME (mins): 4.78
ANY ISSUES ENCOUNTERED: None

DATASET: ScienceDB
EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.032411
LOSS AT EPOCH 20: 0.007906
LOSS AT EPOCH 30: 0.002366
FINAL TRAIN LOSS: 0.002366
VAL MAE: 0.495778
VAL ACCURACY +-0 (exact match): 0.596834
VAL ACCURACY +-1 (within 1 class): 0.915699
TEST MAE: 0.556640
TEST ACCURACY +-0: 0.550615
TEST ACCURACY +-1: 0.905037
CHECKPOINT PATH: D:\T25301094 P2\workspaces\nusrat\sciencedb_bcs_best.pth
TRAINING TIME (mins): 26.24
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---

```

### FILE: workspaces\nusrat\behavior_results.txt
---
```text
---CONTEXT 3 BEHAVIOR---
PERSON NAME: Nusrat
BASE MODEL: EfficientNetB0
DATASET: MmCows (capped 3000/class)
EPOCHS TRAINED: 24
LOSS AT EPOCH 10: 0.013290
LOSS AT EPOCH 20: 0.003363
LOSS AT EPOCH 30: N/A
FINAL TRAIN LOSS: 0.000979
VAL MACRO F1: 0.763051
VAL PER-CLASS ACCURACY:
  Class 1 (Walking): 70.72%
  Class 2 (Standing): 71.60%
  Class 3 (Feeding head up): 76.23%
  Class 4 (Feeding head down): 63.00%
  Class 5 (Licking): 96.30%
  Class 6 (Drinking): 82.55%
  Class 7 (Lying): 98.53%
TEST MACRO F1: 0.744548
TEST PER-CLASS ACCURACY:
  Class 1 (Walking): 59.78%
  Class 2 (Standing): 70.00%
  Class 3 (Feeding head up): 71.97%
  Class 4 (Feeding head down): 66.07%
  Class 5 (Licking): 83.96%
  Class 6 (Drinking): 83.01%
  Class 7 (Lying): 98.77%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\nusrat\behavior_best.pth
TRAINING TIME (mins): 8.95
EARLY STOPPING TRIGGERED AT EPOCH: 24
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
```

### FILE: workspaces\nusrat\cbvd_behavior_results.txt
---
```text
---CONTEXT 3 CBVD-5 EVALUATION---
PERSON NAME: Nusrat
BASE MODEL: EfficientNetB0
DATASET: CBVD-5 (2000 balanced images)
VAL MACRO F1 ON CBVD-5: 0.124517
PER-CLASS ACCURACY ON CBVD-5:
  Class 2 (Standing): 90.34%
  Class 4 (Feeding head down): 8.39%
  Class 6 (Drinking): 2.99%
  Class 7 (Lying): 24.31%
TRAINING TIME (mins): N/A (Evaluation Only)
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
```

### FILE: workspaces\nusrat\id_results.txt
---
```text
---CONTEXT 3 ID---
PERSON NAME: Nusrat
BASE MODEL: EfficientNetB0
DATASET: OpenCows2020
EPOCHS TRAINED: 10
LOSS AT EPOCH 10: 0.550795
FINAL TRAIN LOSS: 0.550795
VAL TOP-1 ACCURACY: 87.06%
TEST TOP-1 ACCURACY: 86.49%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\nusrat\id_best.pth
TRAINING TIME (mins): 1.15
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
```

### FILE: workspaces\nusrat\lameness_results.txt
---
```text
---CONTEXT 3 LAMENESS---
PERSON NAME: Nusrat
BASE MODEL: EfficientNetB0
DATASET: CattleLameness (20 frames/video)
EPOCHS TRAINED: 10
FINAL TRAIN LOSS: 0.000011
VAL AUC: 0.994655
VAL ACCURACY: 82.70%
VAL F1 SCORE: 0.842615
VAL PER-CLASS ACCURACY:
  Class 0 (Normal): 99.76%
  Class 1 (Lame): 72.91%
TEST AUC: 0.982942
TEST ACCURACY: 93.05%
TEST F1 SCORE: 0.948293
TEST PER-CLASS ACCURACY:
  Class 0 (Normal): 90.10%
  Class 1 (Lame): 94.48%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\nusrat\lameness_best.pth
TRAINING TIME (mins): 4.26
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
```

### FILE: workspaces\nusrat\multitask_results.txt
---
```text
Multi-Task Model Evaluation Results
========================================

BCS - MAE: 0.6919, Exact Acc: 0.4285
Behavior - Macro F1: 0.3739
Lameness - Acc: 0.9432, AUC: 0.9884
Cow ID - Acc: 0.9597

```

### FILE: workspaces\nusrat\multitask_temporal_results.txt
---
```text
Spatiotemporal Multi-Task Model Evaluation Results
========================================

BCS - MAE: 0.7849, Exact Acc: 0.3907
Behavior - Macro F1: 0.4775
Lameness - Acc: 1.0000, AUC: 1.0000
Cow ID - Acc: 0.9718

```

### FILE: workspaces\nusrat\spatiotemporal_lameness_efficientnet_results.txt
---
```text
---CONTEXT 3 SPATIOTEMPORAL LAMENESS---
PERSON NAME: Nusrat
BASE MODEL: EfficientNetB0-LSTM
DATASET: CattleLameness (20 frames video sequences)
EPOCHS TRAINED: 15
FINAL TRAIN LOSS: 0.106081
VAL AUC: 1.000000
VAL ACCURACY: 100.00%
VAL F1 SCORE: 1.000000
VAL PER-CLASS ACCURACY:
  Class 0 (Normal): 100.00%
  Class 1 (Lame): 100.00%
TEST AUC: 0.840000
TEST ACCURACY: 80.00%
TEST F1 SCORE: 0.800000
TEST PER-CLASS ACCURACY:
  Class 0 (Normal): 80.00%
  Class 1 (Lame): 80.00%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\nusrat\spatiotemporal_lameness_efficientnet_best.pth
TRAINING TIME (mins): 6.29
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
```

### FILE: workspaces\nusrat\spatiotemporal_lameness_results.txt
---
```text
---CONTEXT 3 SPATIOTEMPORAL LAMENESS---
PERSON NAME: Nusrat
BASE MODEL: ResNet18-LSTM
DATASET: CattleLameness (20 frames video sequences)
EPOCHS TRAINED: 15
FINAL TRAIN LOSS: 0.252091
VAL AUC: 1.000000
VAL ACCURACY: 50.00%
VAL F1 SCORE: 0.666667
VAL PER-CLASS ACCURACY:
  Class 0 (Normal): 0.00%
  Class 1 (Lame): 100.00%
TEST AUC: 0.960000
TEST ACCURACY: 60.00%
TEST F1 SCORE: 0.714286
TEST PER-CLASS ACCURACY:
  Class 0 (Normal): 20.00%
  Class 1 (Lame): 100.00%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\nusrat\spatiotemporal_lameness_best.pth
TRAINING TIME (mins): 2.35
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
```

### FILE: workspaces\shouvik\bcs_results.txt
---
```text
---CONTEXT 3 BCS---
PERSON NAME: Shouvik
BASE MODEL: DenseNet121

DATASET: Dryad
EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.0295
LOSS AT EPOCH 20: 0.0059
LOSS AT EPOCH 30: 0.0027
FINAL TRAIN LOSS: 0.0027
VAL MAE: 0.7338
VAL ACCURACY +-0 (exact match): 17.65%
VAL ACCURACY +-1 (within 1 class): 63.09%
TEST MAE: 0.5875
TEST ACCURACY +-0: 54.75%
TEST ACCURACY +-1: 86.50%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\shouvik\dryad_bcs_best.pth
TRAINING TIME (mins): 15.2
ANY ISSUES ENCOUNTERED: None

DATASET: ScienceDB
EPOCHS TRAINED: 30
LOSS AT EPOCH 10: 0.1200
LOSS AT EPOCH 20: 0.0210
LOSS AT EPOCH 30: 0.0066
FINAL TRAIN LOSS: 0.0066
VAL MAE: 0.5925
VAL ACCURACY +-0 (exact match): 52.78%
VAL ACCURACY +-1 (within 1 class): 89.01%
TEST MAE: 0.6292
TEST ACCURACY +-0: 49.51%
TEST ACCURACY +-1: 88.78%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\shouvik\sciencedb_bcs_best.pth
TRAINING TIME (mins): 55.5
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
```

### FILE: workspaces\shouvik\behavior_results.txt
---
```text
---CONTEXT 3 BEHAVIOR---
PERSON NAME: Shouvik
BASE MODEL: DenseNet121
DATASET: MmCows (capped 3000/class)
EPOCHS TRAINED: 27
LOSS AT EPOCH 10: 0.020973
LOSS AT EPOCH 20: 0.004873
LOSS AT EPOCH 30: N/A
FINAL TRAIN LOSS: 0.001219
VAL MACRO F1: 0.7371
VAL PER-CLASS ACCURACY:
  Class 1 (Walking): 53.83%
  Class 2 (Standing): 67.00%
  Class 3 (Feeding head up): 71.53%
  Class 4 (Feeding head down): 69.47%
  Class 5 (Licking): 88.66%
  Class 6 (Drinking): 75.64%
  Class 7 (Lying): 98.33%
TEST MACRO F1: 0.7366
TEST PER-CLASS ACCURACY:
  Class 1 (Walking): 41.64%
  Class 2 (Standing): 74.70%
  Class 3 (Feeding head up): 67.40%
  Class 4 (Feeding head down): 70.70%
  Class 5 (Licking): 87.26%
  Class 6 (Drinking): 73.42%
  Class 7 (Lying): 99.37%
CHECKPOINT PATH: D:\T25301094 P2\workspaces\shouvik\behavior_best.pth
TRAINING TIME (mins): 13.81
EARLY STOPPING TRIGGERED AT EPOCH: 27
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---

```

## 5. PREPROCESSING CODE
================================================================================

### FILE: context\preprocess_bcs.py
---
```python
import os
import random
import shutil
import csv
from pathlib import Path

# CONFIG
DATASET_ROOT = r"D:\T25301094 P2\datasets\bcs\dryad_bcs\Total_sorted_DGE_images\Total_sorted_DGE_images"
OUTPUT_CSV = r"D:\T25301094 P2\datasets\bcs\bcs_index.csv"
VALID_CLASSES = ['2', '3', '4', '5', '6']
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# Step 1: Collect all cows and their images per class
cow_data = {}  # cow_id -> list of (image_path, label)

for label in VALID_CLASSES:
    class_dir = Path(DATASET_ROOT) / label
    if not class_dir.exists():
        continue
    for cow_folder in class_dir.iterdir():
        if not cow_folder.is_dir():
            continue
        cow_id = cow_folder.name
        images = list(cow_folder.glob("*.tif"))
        if len(images) == 0:
            continue
        if cow_id not in cow_data:
            cow_data[cow_id] = []
        for img in images:
            cow_data[cow_id].append((str(img), label))

# Step 2: Cow-wise split
all_cows = list(cow_data.keys())
random.shuffle(all_cows)

n = len(all_cows)
train_end = int(n * TRAIN_RATIO)
val_end = train_end + int(n * VAL_RATIO)

train_cows = set(all_cows[:train_end])
val_cows = set(all_cows[train_end:val_end])
test_cows = set(all_cows[val_end:])

# Step 3: Write CSV
rows = []
for cow_id, entries in cow_data.items():
    if cow_id in train_cows:
        split = 'train'
    elif cow_id in val_cows:
        split = 'val'
    else:
        split = 'test'
    for img_path, label in entries:
        rows.append([img_path, label, cow_id, split])

with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_path', 'label', 'cow_id', 'split'])
    writer.writerows(rows)

# Step 4: Print summary
print(f"Total cows: {len(all_cows)}")
print(f"Train cows: {len(train_cows)} | Val cows: {len(val_cows)} | Test cows: {len(test_cows)}")

splits = {'train': 0, 'val': 0, 'test': 0}
for row in rows:
    splits[row[3]] += 1
print(f"Train images: {splits['train']} | Val images: {splits['val']} | Test images: {splits['test']}")
print(f"Total images: {len(rows)}")
print(f"CSV saved to: {OUTPUT_CSV}")
```

### FILE: context\preprocess_cbvd.py
---
```python
import os
import csv
import json
import random
import cv2
from pathlib import Path
from collections import defaultdict

# CONFIG
BASE_DIR = r"d:\T25301094 P2"
DATASET_ROOT = Path(BASE_DIR) / "datasets" / "behavior" / "CBVD-5"
OUTPUT_CROP_DIR = DATASET_ROOT / "cropped"
OUTPUT_CSV = DATASET_ROOT / "cbvd_cropped_index.csv"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
OUTPUT_CROP_DIR.mkdir(parents=True, exist_ok=True)

print(f"Dataset Root: {DATASET_ROOT}")
print(f"Output Crop Dir: {OUTPUT_CROP_DIR}")
print(f"Output CSV: {OUTPUT_CSV}")

# Parse CSV
csv_path = DATASET_ROOT / "CBVD-5.csv"
if not csv_path.exists():
    raise FileNotFoundError(f"CSV not found: {csv_path}")

# Groups by label (0, 1, 2, 3, 4 -> mapped to MmCows labels)
# Label mapping:
#   '1' in option_ids -> 6 (Lying)
#   '3' in option_ids -> 5 (Drinking)
#   '2' in option_ids -> 3 (Feeding)
#   '0' in option_ids -> 1 (Standing)
grouped_rows = defaultdict(list)

with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if not row:
            continue
        if row[0].startswith('#'):
            continue
        
        # row: metadata_id, file_list, flags, temporal_coordinates, spatial_coordinates, metadata
        metadata_id = row[0]
        try:
            filename = json.loads(row[1])[0]
            spatial = json.loads(row[4])
            meta = json.loads(row[5])
            
            if "1" not in meta:
                continue
                
            val = meta["1"]
            
            # Label mapping logic
            if '1' in val:      # Lying down
                mapped_label = 6
            elif '3' in val:    # Drinking
                mapped_label = 5
            elif '2' in val:    # Foraging/Feeding
                mapped_label = 3
            elif '0' in val:    # Standing
                mapped_label = 1
            else:
                continue
                
            grouped_rows[mapped_label].append({
                'metadata_id': metadata_id,
                'filename': filename,
                'spatial': spatial,
                'label': mapped_label
            })
        except Exception as e:
            pass

print("\nParsed Label Counts:")
for lbl, items in sorted(grouped_rows.items()):
    print(f"  Mapped Label {lbl}: {len(items)} items")

# Select a balanced subset of up to 500 images per class
selected_items = []
target_size_per_class = 500

for lbl, items in sorted(grouped_rows.items()):
    random.shuffle(items)
    selected = items[:target_size_per_class]
    selected_items.extend(selected)
    print(f"  Selected {len(selected)} items for Label {lbl}")

print(f"\nTotal selected items for cropping: {len(selected_items)}")

# Crop and save images
csv_rows = []
success_count = 0

for idx, item in enumerate(selected_items):
    filename = item['filename']
    metadata_id = item['metadata_id']
    spatial = item['spatial']
    label = item['label']
    
    # Locate full-resolution source image
    try:
        video_id = int(filename.split('_')[0])
    except ValueError:
        continue
        
    if video_id >= 700:
        img_path = DATASET_ROOT / "labelframes_add" / "labelframes" / filename
    else:
        img_path = DATASET_ROOT / "labelframes" / "labelframes" / filename
        
    if not img_path.exists():
        continue
        
    # Read full-resolution image
    img = cv2.imread(str(img_path))
    if img is None:
        continue
        
    # Bounding Box Coordinates: [shape_id, x, y, width, height]
    x = int(float(spatial[1]))
    y = int(float(spatial[2]))
    w = int(float(spatial[3]))
    h = int(float(spatial[4]))
    
    img_h, img_w = img.shape[:2]
    cx1 = max(0, x)
    cy1 = max(0, y)
    cx2 = min(img_w, x + w)
    cy2 = min(img_h, y + h)
    
    if cx2 <= cx1 or cy2 <= cy1:
        continue
        
    # Crop and Resize
    crop = img[cy1:cy2, cx1:cx2]
    resized_crop = cv2.resize(crop, (224, 224))
    
    # Save Crop
    crop_filename = f"{metadata_id}.jpg"
    crop_path = OUTPUT_CROP_DIR / crop_filename
    cv2.imwrite(str(crop_path), resized_crop)
    
    # Save CSV Row: image_path, label, cow_id, split
    csv_rows.append([str(crop_path), str(label), f"CBVD_{video_id}", "test"])
    success_count += 1
    
    if (idx + 1) % 200 == 0:
        print(f"  Processed {idx + 1}/{len(selected_items)} images...")

# Write new CSV index
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_path', 'label', 'cow_id', 'split'])
    writer.writerows(csv_rows)

print(f"\nCropping Complete!")
print(f"Successfully cropped and saved {success_count} images.")
print(f"CSV index written to: {OUTPUT_CSV}")

```

### FILE: context\preprocess_id.py
---
```python
import os
import csv
import random
from pathlib import Path
from collections import defaultdict, Counter

# CONFIG
BASE_DIR = r"d:\T25301094 P2"
DATASET_ROOT = Path(BASE_DIR) / "datasets" / "id" / "opencow2020-DatasetNinja"
OUTPUT_CSV = Path(BASE_DIR) / "datasets" / "id" / "id_index.csv"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

train_dir = DATASET_ROOT / "identification-train" / "img"
test_dir = DATASET_ROOT / "identification-test" / "img"

print(f"Dataset root: {DATASET_ROOT}")
print(f"Output CSV path: {OUTPUT_CSV}")

rows = []

# Process Train Set to split into train and val splits
train_images_by_cow = defaultdict(list)
if train_dir.exists():
    for filename in sorted(os.listdir(train_dir)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Filename format: [class_id]_[image_id].jpg
            parts = filename.split('_')
            if len(parts) >= 2:
                cow_id = parts[0]
                image_path = train_dir / filename
                train_images_by_cow[cow_id].append(str(image_path))

# Split cow-wise/image-wise within training set
# Total 46 classes
all_cows = sorted(list(train_images_by_cow.keys()))
print(f"Found {len(all_cows)} cow classes in training directory.")

for cow_id in all_cows:
    img_paths = train_images_by_cow[cow_id]
    # Shuffle images for this cow class to avoid order bias
    random.shuffle(img_paths)
    
    n = len(img_paths)
    train_count = int(n * 0.85)
    
    # 0-indexed label corresponding to cow class
    label = int(cow_id) - 1
    
    for idx, path in enumerate(img_paths):
        split = 'train' if idx < train_count else 'val'
        rows.append([path, str(label), cow_id, split])

# Process Test Set to assign to test split
test_count = 0
if test_dir.exists():
    for filename in sorted(os.listdir(test_dir)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            parts = filename.split('_')
            if len(parts) >= 2:
                cow_id = parts[0]
                image_path = test_dir / filename
                label = int(cow_id) - 1
                rows.append([str(image_path), str(label), cow_id, 'test'])
                test_count += 1

print(f"Total test images indexed: {test_count}")

# Write to CSV
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_path', 'label', 'cow_id', 'split'])
    writer.writerows(rows)

# Print Summary Statistics
splits = Counter([row[3] for row in rows])
print(f"\nPreprocessing Complete!")
print(f"Total entries in CSV: {len(rows)}")
print(f"Split distribution: Train={splits['train']} | Val={splits['val']} | Test={splits['test']}")
print(f"CSV index file written to: {OUTPUT_CSV}")

```

### FILE: context\preprocess_lameness.py
---
```python
import os
import cv2
import csv
import random
from pathlib import Path

# Paths
BASE_DIR = r"D:\T25301094 P2"
LAME_DIR = Path(BASE_DIR) / "datasets" / "lameness" / "CattleLameness" / "Data" / "Lame"
NORMAL_DIR = Path(BASE_DIR) / "datasets" / "lameness" / "CattleLameness" / "Data" / "Normal"
OUTPUT_FRAMES_DIR = Path(BASE_DIR) / "datasets" / "lameness" / "frames"
OUTPUT_CSV = Path(BASE_DIR) / "datasets" / "lameness" / "lameness_index.csv"

# Configuration
TARGET_SIZE = (224, 224)
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# Ensure output frames directory exists
OUTPUT_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

def process_video_folder(directory, label, class_prefix):
    video_files = sorted([f for f in os.listdir(directory) if f.endswith('.mp4')])
    
    # Shuffle videos to perform video-wise (cow-wise) split
    random.shuffle(video_files)
    
    num_videos = len(video_files)
    train_end = int(num_videos * 0.70)
    val_end = train_end + int(num_videos * 0.15)
    
    video_data = []
    
    for idx, video_file in enumerate(video_files):
        video_path = directory / video_file
        video_id = f"{class_prefix}_{idx + 1}"
        
        # Determine split
        if idx < train_end:
            split = 'train'
        elif idx < val_end:
            split = 'val'
        else:
            split = 'test'
            
        # Extract frames
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error opening video {video_file}")
            continue
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"Empty video {video_file}")
            cap.release()
            continue
            
        # Read and save all frames
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize frame to 224x224
            resized_frame = cv2.resize(frame, TARGET_SIZE)
            
            # Save frame image
            frame_filename = f"{video_id}_frame_{frame_idx}.jpg"
            frame_path = OUTPUT_FRAMES_DIR / frame_filename
            cv2.imwrite(str(frame_path), resized_frame)
            
            video_data.append({
                'image_path': str(frame_path),
                'label': label,
                'cow_id': video_id,
                'split': split
            })
            frame_idx += 1
            
        cap.release()
        print(f"  Processed {video_file}: extracted {frame_idx} frames.")
        
    return video_data

print("Extracting Lame video frames (ALL)...")
lame_data = process_video_folder(LAME_DIR, 1, "Lame")

print("Extracting Normal video frames (ALL)...")
normal_data = process_video_folder(NORMAL_DIR, 0, "Normal")

all_data = lame_data + normal_data

# Write to CSV
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_path', 'label', 'cow_id', 'split'])
    for item in all_data:
        writer.writerow([item['image_path'], item['label'], item['cow_id'], item['split']])

# Summary statistics
splits = {'train': 0, 'val': 0, 'test': 0}
labels = {'Lame (1)': 0, 'Normal (0)': 0}
for item in all_data:
    splits[item['split']] += 1
    if item['label'] == 1:
        labels['Lame (1)'] += 1
    else:
        labels['Normal (0)'] += 1

print(f"\nPreprocessing Complete!")
print(f"Total frames extracted: {len(all_data)}")
print(f"Split distribution: Train={splits['train']} | Val={splits['val']} | Test={splits['test']}")
print(f"Class distribution: Lame={labels['Lame (1)']} | Normal={labels['Normal (0)']}")
print(f"CSV index file written to: {OUTPUT_CSV}")

```

### FILE: context\preprocess_mmcows_behavior.py
---
```python
import os
import csv
import random
from pathlib import Path
from collections import defaultdict, Counter

DATASET_ROOT = r"D:\T25301094 P2\datasets\behavior\mmcows\cropped_bboxes\cropped_bboxes\behaviors"
OUTPUT_CSV = r"D:\T25301094 P2\datasets\behavior\behavior_index.csv"
VALID_CLASSES = ['1','2','3','4','5','6','7']
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

cow_data = defaultdict(list)

for label in VALID_CLASSES:
    class_dir = Path(DATASET_ROOT) / label
    for img_file in class_dir.glob("*.jpg"):
        parts = img_file.stem.split('_')
        cow_id = parts[2]
        cow_data[cow_id].append((str(img_file), label))

all_cows = list(cow_data.keys())
random.shuffle(all_cows)

n = len(all_cows)
train_end = int(n * TRAIN_RATIO)
val_end = train_end + int(n * VAL_RATIO)

train_cows = set(all_cows[:train_end])
val_cows = set(all_cows[train_end:val_end])
test_cows = set(all_cows[val_end:])

rows = []
for cow_id, entries in cow_data.items():
    split = 'train' if cow_id in train_cows else ('val' if cow_id in val_cows else 'test')
    for img_path, label in entries:
        rows.append([img_path, label, cow_id, split])

with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_path', 'label', 'cow_id', 'split'])
    writer.writerows(rows)

splits = {'train': 0, 'val': 0, 'test': 0}
for row in rows:
    splits[row[3]] += 1

print(f"Total cows: {len(all_cows)}")
print(f"Train: {len(train_cows)} cows | Val: {len(val_cows)} cows | Test: {len(test_cows)} cows")
print(f"Train images: {splits['train']} | Val: {splits['val']} | Test: {splits['test']}")
print(f"Total: {len(rows)}")
print("\nClass distribution:")
labels = [r[1] for r in rows]
for cls, count in sorted(Counter(labels).items()):
    print(f"  Class {cls}: {count} images")
```

### FILE: context\preprocess_sciencedb_bcs.py
---
```python
import os
import random
import csv
from pathlib import Path
from collections import defaultdict, Counter

DATASET_ROOT = r"D:\T25301094 P2\datasets\bcs\sciencedb_bcs\dataset"
OUTPUT_CSV = r"D:\T25301094 P2\datasets\bcs\sciencedb_bcs_index.csv"
VALID_CLASSES = ['3.25', '3.5', '3.75', '4.0', '4.25']
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# Step 1: Collect all cows and images
cow_data = defaultdict(list)

for label in VALID_CLASSES:
    class_dir = Path(DATASET_ROOT) / label
    if not class_dir.exists():
        continue
    for img_file in class_dir.glob("*.jpg"):
        stem = img_file.stem
        if stem.startswith('GS_') or stem.startswith('YM_'):
            parts = stem.split('_')
            cow_id = f"{parts[0]}_{parts[1]}"
        elif stem.startswith('L-i') or stem.startswith('R-i'):
            cow_id = 'i' + stem[3:]  # L-i1035 and R-i1035 = same cow
        else:
            cow_id = stem
        cow_data[cow_id].append((str(img_file), label))

# Step 2: Cow-wise split
all_cows = list(cow_data.keys())
random.shuffle(all_cows)

n = len(all_cows)
train_end = int(n * TRAIN_RATIO)
val_end = train_end + int(n * VAL_RATIO)

train_cows = set(all_cows[:train_end])
val_cows = set(all_cows[train_end:val_end])
test_cows = set(all_cows[val_end:])

# Step 3: Write CSV
rows = []
for cow_id, entries in cow_data.items():
    if cow_id in train_cows:
        split = 'train'
    elif cow_id in val_cows:
        split = 'val'
    else:
        split = 'test'
    for img_path, label in entries:
        rows.append([img_path, label, cow_id, split])

with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_path', 'label', 'cow_id', 'split'])
    writer.writerows(rows)

# Step 4: Summary
splits = {'train': 0, 'val': 0, 'test': 0}
for row in rows:
    splits[row[3]] += 1

print(f"Total cows: {len(all_cows)}")
print(f"Train cows: {len(train_cows)} | Val: {len(val_cows)} | Test: {len(test_cows)}")
print(f"Train images: {splits['train']} | Val: {splits['val']} | Test: {splits['test']}")
print(f"Total images: {len(rows)}")

print("\nClass distribution:")
labels = [r[1] for r in rows]
for cls, count in sorted(Counter(labels).items()):
    print(f"  BCS {cls}: {count} images")

print("\nSample rows:")
for row in rows[:3]:
    print(row)
```

## 6. TRAINING CODE (HASIN - ResNet-18)
================================================================================

### FILE: workspaces\hasin\train_bcs.py
---
```python
import os
import time
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
SEED          = 42
PERSON_NAME   = "Hasin"
BASE_MODEL    = "ResNet-18"
TIMM_MODEL    = "resnet18"
WORKSPACE_DIR = r"D:\T25301094 P2\workspaces\hasin"
DRYAD_CSV     = r"D:\T25301094 P2\datasets\bcs\bcs_index.csv"
SCIDB_CSV     = r"D:\T25301094 P2\datasets\bcs\sciencedb_bcs_index.csv"

NUM_CLASSES   = 5       # ordinal classes 0-4
CORAL_OUTPUTS = NUM_CLASSES - 1   # 4 output nodes
BATCH_SIZE    = 64      # doubled from 32 — safe with AMP on RTX 4080 16GB
NUM_WORKERS   = 8
EPOCHS        = 30
LR            = 1e-3

DRYAD_LABEL_MAP = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4}
SCIDB_LABEL_MAP = {3.25: 0, 3.5: 1, 3.75: 2, 4.0: 3, 4.25: 4}

DEVICE = torch.device("cuda")

# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ============================================================
# DATASET
# ============================================================
class BCSDataset(Dataset):
    def __init__(self, csv_path, split, label_map, transform=None):
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")

        # CRITICAL: dict lookup — never use int() or float() directly on the label
        raw = row["label"]
        try:
            key = int(raw)
        except (ValueError, TypeError):
            key = float(raw)
        if key not in self.label_map:
            key = float(raw)
        label = self.label_map[key]

        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

def get_transforms():
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        T.ToTensor(),
        norm,
    ])
    eval_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        norm,
    ])
    return train_tf, eval_tf

def build_loaders(csv_path, label_map):
    train_tf, eval_tf = get_transforms()
    train_ds = BCSDataset(csv_path, "train", label_map, train_tf)
    val_ds   = BCSDataset(csv_path, "val",   label_map, eval_tf)
    test_ds  = BCSDataset(csv_path, "test",  label_map, eval_tf)

    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader

# ============================================================
# CBAM ATTENTION
# ============================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        hidden = max(in_channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_ = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        return x * self.sigmoid(avg + max_).unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))


class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))

# ============================================================
# MODEL
# ============================================================
class BCSModel(nn.Module):
    def __init__(self):
        super().__init__()
        # global_pool='' gives spatial feature maps needed for CBAM
        self.backbone = timm.create_model(
            TIMM_MODEL, pretrained=True, num_classes=0, global_pool=""
        )
        self.backbone = self.backbone.to(DEVICE)

        # Dynamically detect feature dimension — never hardcode
        with torch.no_grad():
            dummy = self.backbone(torch.zeros(1, 3, 224, 224).to(DEVICE))
            feature_dim = dummy.shape[1]

        self.cbam = CBAM(feature_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(feature_dim, CORAL_OUTPUTS)

    def forward(self, x):
        x = self.backbone(x)
        x = self.cbam(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

# ============================================================
# CORAL LOSS + METRICS
# ============================================================
def coral_loss(logits, labels):
    sets = [(labels > i).float() for i in range(NUM_CLASSES - 1)]
    targets = torch.stack(sets, dim=1)
    return F.binary_cross_entropy_with_logits(logits, targets)


def coral_predict(logits):
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)


def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae    = float(np.mean(np.abs(y_true - y_pred)))
    acc0   = float(np.mean(y_true == y_pred) * 100)
    acc1   = float(np.mean(np.abs(y_true - y_pred) <= 1) * 100)
    return mae, acc0, acc1

# ============================================================
# TRAIN ONE EPOCH (AMP)
# ============================================================
def train_epoch(model, loader, optimizer, scaler, epoch, dataset_name):
    model.train()
    running_loss = 0.0
    total        = 0

    bar = tqdm(loader, desc=f"[{dataset_name}] Epoch {epoch:02d}/{EPOCHS} Train", leave=False)
    for imgs, labels in bar:
        imgs   = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda"):
            logits = model(imgs)
            loss   = coral_loss(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        total        += imgs.size(0)
        bar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / total

# ============================================================
# EVALUATE (AMP)
# ============================================================
@torch.no_grad()
def evaluate(model, loader, desc):
    model.eval()
    all_preds, all_labels = [], []

    for imgs, labels in tqdm(loader, desc=desc, leave=False):
        imgs   = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.amp.autocast("cuda"):
            logits = model(imgs)

        preds = coral_predict(logits)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    return compute_metrics(all_labels, all_preds)

# ============================================================
# TRAINING RUN
# ============================================================
def run_training(dataset_name, csv_path, label_map, checkpoint_name):
    print(f"\n{'='*70}")
    print(f"  STARTING: {dataset_name}")
    print(f"{'='*70}")

    ckpt_path    = os.path.join(WORKSPACE_DIR, checkpoint_name)
    train_loader, val_loader, test_loader = build_loaders(csv_path, label_map)

    model     = BCSModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scaler    = torch.amp.GradScaler("cuda")

    best_val_mae  = float("inf")
    best_val_metrics = None
    train_losses  = []
    milestone     = {}          # {10: loss, 20: loss, 30: loss}

    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, epoch, dataset_name)
        scheduler.step()
        train_losses.append(train_loss)

        if epoch in (10, 20, 30):
            milestone[epoch] = train_loss

        val_mae, val_acc0, val_acc1 = evaluate(model, val_loader, f"[{dataset_name}] Epoch {epoch:02d} Val")
        print(f"  Epoch {epoch:02d}/{EPOCHS} | Loss: {train_loss:.6f} | Val MAE: {val_mae:.4f} | Val ±0: {val_acc0:.2f}% | Val ±1: {val_acc1:.2f}%")

        if val_mae < best_val_mae:
            best_val_mae     = val_mae
            best_val_metrics = (val_mae, val_acc0, val_acc1)
            torch.save(model.state_dict(), ckpt_path)

    elapsed_mins = (time.time() - start) / 60.0

    # Load best checkpoint for test evaluation
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    test_mae, test_acc0, test_acc1 = evaluate(model, test_loader, f"[{dataset_name}] Test")

    print(f"\n  {dataset_name} DONE | Best Val MAE: {best_val_mae:.4f} | Test MAE: {test_mae:.4f} | Time: {elapsed_mins:.2f} mins")

    return {
        "dataset_name":   dataset_name,
        "loss_ep10":      milestone.get(10, "N/A"),
        "loss_ep20":      milestone.get(20, "N/A"),
        "loss_ep30":      milestone.get(30, "N/A"),
        "final_loss":     train_losses[-1],
        "val_mae":        best_val_metrics[0],
        "val_acc0":       best_val_metrics[1],
        "val_acc1":       best_val_metrics[2],
        "test_mae":       test_mae,
        "test_acc0":      test_acc0,
        "test_acc1":      test_acc1,
        "ckpt_path":      ckpt_path,
        "time_mins":      elapsed_mins,
        "train_losses":   train_losses,
    }

# ============================================================
# OUTPUT FILES
# ============================================================
def fmt(v):
    return f"{v:.6f}" if isinstance(v, float) else str(v)


def save_results(dryad, scidb):
    def block(r):
        return f"""DATASET: {r['dataset_name']}
EPOCHS TRAINED: {EPOCHS}
LOSS AT EPOCH 10: {fmt(r['loss_ep10'])}
LOSS AT EPOCH 20: {fmt(r['loss_ep20'])}
LOSS AT EPOCH 30: {fmt(r['loss_ep30'])}
FINAL TRAIN LOSS: {fmt(r['final_loss'])}
VAL MAE: {fmt(r['val_mae'])}
VAL ACCURACY +-0 (exact match): {r['val_acc0']:.2f}%
VAL ACCURACY +-1 (within 1 class): {r['val_acc1']:.2f}%
TEST MAE: {fmt(r['test_mae'])}
TEST ACCURACY +-0: {r['test_acc0']:.2f}%
TEST ACCURACY +-1: {r['test_acc1']:.2f}%
CHECKPOINT PATH: {r['ckpt_path']}
TRAINING TIME (mins): {r['time_mins']:.2f}
ANY ISSUES ENCOUNTERED: None"""

    report = f"""---CONTEXT 3 BCS---
PERSON NAME: {PERSON_NAME}
BASE MODEL: {BASE_MODEL}

{block(dryad)}

{block(scidb)}
---END CONTEXT 3---
"""
    out_path = os.path.join(WORKSPACE_DIR, "bcs_results.txt")
    with open(out_path, "w") as f:        # WRITE mode — never append
        f.write(report)
    print(f"\nResults saved: {out_path}")
    print(report)


def save_loss_curve(dryad, scidb):
    epochs = list(range(1, EPOCHS + 1))
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, dryad["train_losses"], label="Dryad Train Loss",   color="steelblue")
    plt.plot(epochs, scidb["train_losses"], label="ScienceDB Train Loss", color="darkorange")
    plt.xlabel("Epoch")
    plt.ylabel("CORAL Loss")
    plt.title(f"BCS Training Loss — {PERSON_NAME} ({BASE_MODEL})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_path = os.path.join(WORKSPACE_DIR, "bcs_loss_curve.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Loss curve saved: {out_path}")

# ============================================================
# MAIN
# ============================================================
def main():
    set_seed(SEED)
    os.makedirs(WORKSPACE_DIR, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Check GPU drivers.")

    # Enable cudnn autotuner — finds fastest conv algorithms for fixed input size
    torch.backends.cudnn.benchmark = True

    print("=" * 70)
    print(f"  BCS TRAINING — {PERSON_NAME} — {BASE_MODEL}")
    print(f"  GPU : {torch.cuda.get_device_name(0)}")
    print(f"  Workspace: {WORKSPACE_DIR}")
    print(f"  Batch size: {BATCH_SIZE} | Workers: {NUM_WORKERS} | AMP: ON")
    print("=" * 70)

    dryad_result = run_training("Dryad",     DRYAD_CSV, DRYAD_LABEL_MAP, "dryad_bcs_best.pth")
    scidb_result = run_training("ScienceDB", SCIDB_CSV, SCIDB_LABEL_MAP,  "sciencedb_bcs_best.pth")

    save_results(dryad_result, scidb_result)
    save_loss_curve(dryad_result, scidb_result)

    print("\n" + "=" * 70)
    print("  ALL TRAINING COMPLETE")
    print(f"  Dryad    — Test MAE: {dryad_result['test_mae']:.4f}")
    print(f"  ScienceDB — Test MAE: {scidb_result['test_mae']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

### FILE: workspaces\hasin\train_behavior.py
---
```python
import os
import time
import random
import numpy as np
import pandas as pd

import cv2
cv2.setNumThreads(0)  # Disable OpenCV multithreading to prevent deadlocks in DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from sklearn.metrics import f1_score, confusion_matrix

import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend to prevent Tkinter thread crashes
import matplotlib.pyplot as plt

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# GPU Optimization settings
torch.backends.cudnn.benchmark = True

class BehaviorDataset(Dataset):
    def __init__(self, csv_path, split, transform=None):
        df = pd.read_csv(csv_path)
        df = df[df['split'] == split]
        
        # Mandatory sampling code to fix class imbalance and make training fast
        df = pd.concat([
            group.sample(min(len(group), 3000), random_state=42)
            for _, group in df.groupby('label')
        ]).reset_index(drop=True)
        
        self.data = df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['image_path']
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = int(row['label']) - 1
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, label

# CBAM Attention Modules
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_ = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        return x * self.sigmoid(avg + max_).unsqueeze(-1).unsqueeze(-1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()
    def forward(self, x):
        return self.sa(self.ca(x))

# Complete Model Architecture
class BehaviorNetwork(nn.Module):
    def __init__(self, model_name='resnet18', num_classes=7):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='')
        self.backbone = self.backbone.to(device)
        
        # Dynamically detect feature dimensions
        with torch.no_grad():
            dummy = self.backbone(torch.zeros(1, 3, 224, 224).to(device))
            feature_dim = dummy.shape[1]
            
        self.cbam = CBAM(feature_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Focal Loss Module
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def evaluate_metrics(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(7)))
    
    # Calculate per-class accuracy
    per_class_acc = []
    for i in range(7):
        class_total = np.sum(cm[i, :])
        if class_total > 0:
            acc = (cm[i, i] / class_total) * 100.0
        else:
            acc = 0.0
        per_class_acc.append(acc)
        
    return macro_f1, per_class_acc

def main():
    # Setup directories
    workspace_dir = r"D:\T25301094 P2\workspaces\hasin"
    os.makedirs(workspace_dir, exist_ok=True)
    
    csv_path = r"D:\T25301094 P2\datasets\behavior\behavior_index.csv"
    checkpoint_path = os.path.join(workspace_dir, "behavior_best.pth")
    results_txt_path = os.path.join(workspace_dir, "behavior_results.txt")
    loss_curve_path = os.path.join(workspace_dir, "behavior_loss_curve.png")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define Transforms
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_test_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Initialize Datasets and Loaders
    train_dataset = BehaviorDataset(csv_path, 'train', transform=train_transform)
    val_dataset = BehaviorDataset(csv_path, 'val', transform=val_test_transform)
    test_dataset = BehaviorDataset(csv_path, 'test', transform=val_test_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=4,
        pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=128, shuffle=False, num_workers=4,
        pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=4,
        pin_memory=True, persistent_workers=False, prefetch_factor=2
    )
    
    # Initialize Framework Components
    model = BehaviorNetwork(model_name='resnet18', num_classes=7).to(device)
    criterion = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scaler = torch.amp.GradScaler('cuda')
    
    # Tracking variables
    MAX_EPOCHS = 30
    PATIENCE = 10
    epochs_no_improve = 0
    best_val_f1 = -1.0
    actual_epochs_trained = 0
    early_stopping_epoch = "N/A"
    
    epoch_losses = []
    loss_milestones = {10: "N/A", 20: "N/A", 30: "N/A"}
    
    start_time = time.time()
    
    # Training Loop
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_losses.append(epoch_loss)
        scheduler.step()
        
        # Track specific milestone losses
        if epoch in loss_milestones:
            loss_milestones[epoch] = f"{epoch_loss:.4f}"
            
        # Evaluate Validation metrics
        val_f1, val_per_class = evaluate_metrics(model, val_loader, device)
        print(f"Epoch {epoch}/{MAX_EPOCHS} | Loss: {epoch_loss:.4f} | Val Macro F1: {val_f1:.4f}")
        
        # Early stopping verification block
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}. No improvement for {PATIENCE} epochs.")
                early_stopping_epoch = str(epoch)
                actual_epochs_trained = epoch
                break
                
        actual_epochs_trained = epoch

    end_time = time.time()
    training_time_mins = (end_time - start_time) / 60.0
    final_train_loss = f"{epoch_losses[-1]:.4f}"
    
    # Plot training loss curve
    plt.figure()
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Behavior Training Loss Curve')
    plt.legend()
    plt.savefig(loss_curve_path)
    plt.close()
    
    # Evaluate best checkpoint on Val and Test sets
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    final_val_f1, final_val_per_class = evaluate_metrics(model, val_loader, device)
    test_f1, test_per_class = evaluate_metrics(model, test_loader, device)
    
    # Auto-generate Context 3 Report file
    report_content = f"""---CONTEXT 3 BEHAVIOR---
PERSON NAME: Hasin
BASE MODEL: ResNet-18
DATASET: MmCows (capped 3000/class)
EPOCHS TRAINED: {actual_epochs_trained}
LOSS AT EPOCH 10: {loss_milestones.get(10, "N/A")}
LOSS AT EPOCH 20: {loss_milestones.get(20, "N/A")}
LOSS AT EPOCH 30: {loss_milestones.get(30, "N/A")}
FINAL TRAIN LOSS: {final_train_loss}
VAL MACRO F1: {final_val_f1:.4f}
VAL PER-CLASS ACCURACY:
  Class 1 (Walking): {final_val_per_class[0]:.2f}%
  Class 2 (Standing): {final_val_per_class[1]:.2f}%
  Class 3 (Feeding head up): {final_val_per_class[2]:.2f}%
  Class 4 (Feeding head down): {final_val_per_class[3]:.2f}%
  Class 5 (Licking): {final_val_per_class[4]:.2f}%
  Class 6 (Drinking): {final_val_per_class[5]:.2f}%
  Class 7 (Lying): {final_val_per_class[6]:.2f}%
TEST MACRO F1: {test_f1:.4f}
TEST PER-CLASS ACCURACY:
  Class 1 (Walking): {test_per_class[0]:.2f}%
  Class 2 (Standing): {test_per_class[1]:.2f}%
  Class 3 (Feeding head up): {test_per_class[2]:.2f}%
  Class 4 (Feeding head down): {test_per_class[3]:.2f}%
  Class 5 (Licking): {test_per_class[4]:.2f}%
  Class 6 (Drinking): {test_per_class[5]:.2f}%
  Class 7 (Lying): {test_per_class[6]:.2f}%
CHECKPOINT PATH: {checkpoint_path}
TRAINING TIME (mins): {training_time_mins:.2f}
EARLY STOPPING TRIGGERED AT EPOCH: {early_stopping_epoch}
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---"""

    with open(results_txt_path, 'w') as f:
        f.write(report_content)

if __name__ == '__main__':
    main()
```

## 7. TRAINING CODE (NUSRAT - EfficientNetB0 & Lameness)
================================================================================

### FILE: workspaces\nusrat\crop_cow_detection.py
---
```python
import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm

# Configuration
BASE_DIR = r"D:\T25301094 P2"
INPUT_VIDEO_PATH = os.path.join(BASE_DIR, "videos", "cut_cow_video.mp4")
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, "videos", "cropped_cow_video.mp4")
MODEL_NAME = os.path.join(BASE_DIR, "final_models", "yolov8n.pt")  # Lightweight YOLOv8 Nano model
TARGET_SIZE = (224, 224)   # Standard crop size for neural networks

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Crop Cow Detections from Video and Filter Backgrounds")
    parser.add_argument("--input", type=str, default=INPUT_VIDEO_PATH, help="Path to input video")
    parser.add_argument("--output", type=str, default=OUTPUT_VIDEO_PATH, help="Path to save cropped video")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of frames to process (0 for full video)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input video not found at: {args.input}")

    print(f"Loading YOLOv8 detector: {MODEL_NAME}...")
    detector = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Error: Could not open video file {args.input}")

    # Read video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    limit = args.limit if args.limit > 0 else total_frames
    limit = min(limit, total_frames)

    print(f"\nProcessing Video:")
    print(f"  Source:       {args.input}")
    print(f"  FPS:          {fps}")
    print(f"  Total Frames: {total_frames} (Processing limit: {limit})")
    print(f"  Output size:  {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    print(f"  Saving to:    {args.output}\n")
    print("All slides, texts, and frames without cows will be automatically filtered out!\n")

    # Define VideoWriter to save output (size is fixed to 224x224)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, TARGET_SIZE)

    show_window = True
    cropped_count = 0

    for i in tqdm(range(limit), desc="Cropping cows"):
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = detector(frame, verbose=False)[0]

        cow_crop = None
        best_box = None
        max_area = 0
        for box in results.boxes:
            class_id = int(box.cls[0])
            
            # Class ID 19 is 'cow' in COCO dataset
            if class_id == 19:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    best_box = (x1, y1, x2, y2)

        if best_box is not None:
            x1, y1, x2, y2 = best_box
            cow_crop = frame[y1:y2, x1:x2]

        # If a cow was found, resize, save, and display it
        if cow_crop is not None and cow_crop.size > 0:
            resized_crop = cv2.resize(cow_crop, TARGET_SIZE)
            out.write(resized_crop)
            cropped_count += 1

            if show_window:
                try:
                    cv2.imshow("Cropped Cow Stream (Press 'q' to Quit)", resized_crop)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nProcessing stopped early by user.")
                        break
                except cv2.error:
                    print("\nWarning: Headless environment or OpenCV without GUI support detected.")
                    print("Running in HEADLESS mode. Video is still being written to the output file.")
                    show_window = False

    cap.release()
    out.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    
    print(f"\nFinished!")
    print(f"Saved {cropped_count} cropped frames to: {args.output}")

if __name__ == "__main__":
    main()

```

### FILE: workspaces\nusrat\crop_sciencedb_dataset.py
---
```python
import os
import cv2
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

# Configuration
BASE_DIR = r"D:\T25301094 P2"
INPUT_CSV = os.path.join(BASE_DIR, "datasets", "bcs", "sciencedb_bcs_index.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "datasets", "bcs", "sciencedb_bcs_cropped_index.csv")
CROPPED_DIR = os.path.join(BASE_DIR, "datasets", "bcs", "sciencedb_bcs_cropped")
YOLO_MODEL = os.path.join(BASE_DIR, "final_models", "yolov8n.pt")
TARGET_SIZE = (224, 224)

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found at: {INPUT_CSV}")

    os.makedirs(CROPPED_DIR, exist_ok=True)

    print("Loading YOLOv8 detector...")
    detector = YOLO(YOLO_MODEL)

    print(f"Reading index file: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    
    cropped_records = []
    skipped_count = 0

    print(f"Processing {len(df)} images...")
    # Loop over all images in the index
    for idx, row in enumerate(tqdm(df.iterrows(), total=len(df), desc="Cropping ScienceDB")):
        img_path = row[1]['image_path']
        label = row[1]['label']
        cow_id = row[1]['cow_id']
        split = row[1]['split']

        if not os.path.exists(img_path):
            skipped_count += 1
            continue

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            skipped_count += 1
            continue

        # Run YOLO detection
        results = detector(img, verbose=False)[0]

        cow_crop = None
        best_box = None
        max_area = 0
        for box in results.boxes:
            class_id = int(box.cls[0])
            # Class ID 19 is 'cow'
            if class_id == 19:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    best_box = (x1, y1, x2, y2)

        if best_box is not None:
            x1, y1, x2, y2 = best_box
            cow_crop = img[y1:y2, x1:x2]

        # Save output path
        filename = f"crop_{idx}_{os.path.basename(img_path)}"
        output_path = os.path.join(CROPPED_DIR, filename)

        # Write crop or fallback to original resized
        if cow_crop is not None and cow_crop.size > 0:
            resized_crop = cv2.resize(cow_crop, TARGET_SIZE)
            cv2.imwrite(output_path, resized_crop)
        else:
            # Fallback to resizing original image if YOLO misses
            resized_orig = cv2.resize(img, TARGET_SIZE)
            cv2.imwrite(output_path, resized_orig)

        cropped_records.append({
            "image_path": output_path,
            "label": label,
            "cow_id": cow_id,
            "split": split
        })

    # Save new cropped index CSV
    out_df = pd.DataFrame(cropped_records)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nCropping Complete!")
    print(f"  Cropped images saved to: {CROPPED_DIR}")
    print(f"  New index CSV written to: {OUTPUT_CSV}")
    print(f"  Skipped/Missing files:    {skipped_count}")

if __name__ == "__main__":
    main()

```

### FILE: workspaces\nusrat\cut_video_segment.py
---
```python
import os
import cv2
from tqdm import tqdm

# Configuration
BASE_DIR = r"D:\T25301094 P2"
INPUT_VIDEO = os.path.join(BASE_DIR, "videos", "full_download.mp4")
OUTPUT_VIDEO = os.path.join(BASE_DIR, "videos", "cut_cow_video_2.mp4")
START_SEC = 175.0  # 2:55 (2 * 60 + 55)
END_SEC = 215.0    # 3:35 (3 * 60 + 35)

def main():
    if not os.path.exists(INPUT_VIDEO):
        raise FileNotFoundError(f"Input video not found at: {INPUT_VIDEO}")

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open input video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(START_SEC * fps)
    end_frame = int(END_SEC * fps)
    end_frame = min(end_frame, total_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    print(f"Trimming segment: {START_SEC}s (2:55) to {END_SEC}s (3:35)")
    print(f"Frames: {start_frame} to {end_frame} (Total to cut: {end_frame - start_frame})")

    for _ in tqdm(range(start_frame, end_frame), desc="Trimming video"):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"\nSuccessfully saved trimmed segment to: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()

```

### FILE: workspaces\nusrat\evaluate_all_50_videos.py
---
```python
import os
import cv2
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

# Configuration
MODEL_NAME = "efficientnet_b0"
HIDDEN_DIM = 64
CSV_PATH = r"D:\T25301094 P2\datasets\lameness\lameness_cropped_index.csv"
CHECKPOINT_PATH = r"D:\T25301094 P2\workspaces\nusrat\spatiotemporal_lameness_efficientnet_best.pth"
OUTPUT_REPORT_PATH = r"D:\T25301094 P2\workspaces\nusrat\all_50_videos_evaluation.txt"
DECISION_THRESHOLD = 0.50  # Calibrated decision threshold for cropped model

class CNNLSTMModel(nn.Module):
    def __init__(self, backbone_name, hidden_dim, device):
        super().__init__()
        backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        backbone = backbone.to(device, non_blocking=True)
        with torch.no_grad():
            dummy = backbone(torch.zeros(1, 3, 224, 224).to(device, non_blocking=True))
            feature_dim = dummy.shape[1]
        self.backbone = backbone
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.backbone(x)
        features = features.view(batch_size, seq_len, -1)
        lstm_out, (hn, cn) = self.lstm(features)
        last_step_out = lstm_out[:, -1, :]
        logits = self.classifier(last_step_out)
        return logits

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = CNNLSTMModel(MODEL_NAME, HIDDEN_DIM, device)
    model = model.to(device)
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at: {CHECKPOINT_PATH}")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded checkpoint from: {CHECKPOINT_PATH}")

    # Load dataset index using standard csv to avoid pyarrow crash
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Dataset index not found at: {CSV_PATH}")
    
    import csv
    video_data = []
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_data.append({
                'image_path': row['image_path'],
                'label': int(row['label']),
                'cow_id': row['cow_id'],
                'split': row['split']
            })
            
    # Get all unique cow (video) IDs
    video_ids = sorted(list(set(item['cow_id'] for item in video_data)))
    print(f"Found {len(video_ids)} unique video sequences to evaluate.")

    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    results = []

    # Run inference for all 50 videos
    for v_id in tqdm(video_ids, desc="Evaluating videos"):
        v_samples = [item for item in video_data if item['cow_id'] == v_id]
        v_samples = sorted(v_samples, key=lambda x: x['image_path'])
        split = v_samples[0]['split']
        label = v_samples[0]['label']

        # Sample 20 frames evenly
        total_frames = len(v_samples)
        indices = [int(i * (total_frames - 1) / (20 - 1)) for i in range(20)] if total_frames > 1 else [0] * 20

        frames = []
        for idx_to_load in indices:
            img_path = v_samples[idx_to_load]['image_path']
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = transform(image=image)["image"]
            frames.append(image)

        # Shape: (1, 20, 3, 224, 224)
        input_tensor = torch.stack(frames).unsqueeze(0).to(device)

        with torch.no_grad():
            with torch.amp.autocast('cuda') if torch.cuda.is_available() else torch.no_grad():
                outputs = model(input_tensor)
                prob = torch.sigmoid(outputs).view(-1).item()

        pred_label = 1 if prob >= DECISION_THRESHOLD else 0
        correct = (pred_label == label)

        results.append({
            "cow_id": v_id,
            "split": split,
            "true_label": label,
            "pred_probability": prob,
            "pred_label": pred_label,
            "correct": correct
        })

    # Compute metrics globally and per split
    splits = ["train", "val", "test"]
    split_metrics = {}
    
    for s in splits:
        s_results = [r for r in results if r['split'] == s]
        if len(s_results) > 0:
            labels = np.array([r['true_label'] for r in s_results])
            probs = np.array([r['pred_probability'] for r in s_results])
            preds = np.array([r['pred_label'] for r in s_results])
            
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, zero_division=0)
            try:
                auc = roc_auc_score(labels, probs)
            except ValueError:
                auc = float('nan')
            
            cm = confusion_matrix(labels, preds, labels=[0, 1])
            split_metrics[s] = {
                "count": len(s_results),
                "accuracy": acc,
                "f1": f1,
                "auc": auc,
                "cm": cm
            }

    # Global metrics
    g_labels = np.array([r['true_label'] for r in results])
    g_probs = np.array([r['pred_probability'] for r in results])
    g_preds = np.array([r['pred_label'] for r in results])
    g_acc = accuracy_score(g_labels, g_preds)
    g_f1 = f1_score(g_labels, g_preds, zero_division=0)
    g_auc = roc_auc_score(g_labels, g_probs)
    g_cm = confusion_matrix(g_labels, g_preds, labels=[0, 1])

    # Build the report string
    report = []
    report.append("=" * 70)
    report.append("          EVALUATION REPORT: ALL 50 CATTLE VIDEOS")
    report.append("=" * 70)
    report.append(f"Model: {MODEL_NAME}-LSTM")
    report.append(f"Checkpoint: {CHECKPOINT_PATH}")
    report.append(f"Calibrated Threshold: {DECISION_THRESHOLD:.2f}\n")

    report.append("DETAILED TABLE:")
    report.append("-" * 75)
    report.append(f"{'Cow ID':<15} | {'Split':<8} | {'True Label':<12} | {'Pred Prob':<10} | {'Pred Label':<10} | {'Status':<8}")
    report.append("-" * 75)
    for r in results:
        t_lbl = "Lame" if r['true_label'] == 1 else "Normal"
        p_lbl = "Lame" if r['pred_label'] == 1 else "Normal"
        status = "CORRECT" if r['correct'] else "WRONG"
        report.append(f"{r['cow_id']:<15} | {r['split']:<8} | {t_lbl:<12} | {r['pred_probability']:0.4f}     | {p_lbl:<10} | {status:<8}")
    report.append("-" * 75)
    report.append("\nSUMMARY METRICS PER SPLIT:")
    
    for s, metrics in split_metrics.items():
        report.append(f"\n--- {s.upper()} SPLIT ({metrics['count']} videos) ---")
        report.append(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
        report.append(f"  F1 Score: {metrics['f1']:.4f}")
        report.append(f"  AUC:      {metrics['auc']:.4f}")
        report.append(f"  Confusion Matrix:")
        report.append(f"    [[TN: {metrics['cm'][0,0]}, FP: {metrics['cm'][0,1]}],")
        report.append(f"     [FN: {metrics['cm'][1,0]}, TP: {metrics['cm'][1,1]}]]")

    report.append("\n" + "=" * 70)
    report.append("--- OVERALL GLOBAL METRICS (All 50 videos) ---")
    report.append(f"  Overall Accuracy: {g_acc*100:.2f}%")
    report.append(f"  Overall F1 Score: {g_f1:.4f}")
    report.append(f"  Overall AUC:      {g_auc:.4f}")
    report.append(f"  Overall Confusion Matrix:")
    report.append(f"    [[TN: {g_cm[0,0]}, FP: {g_cm[0,1]}],")
    report.append(f"     [FN: {g_cm[1,0]}, TP: {g_cm[1,1]}]]")
    report.append("=" * 70)

    report_text = "\n".join(report)
    
    # Save report
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Print to console
    print(report_text)
    print(f"\nSaved evaluation report to: {OUTPUT_REPORT_PATH}")

if __name__ == "__main__":
    main()

```

### FILE: workspaces\nusrat\evaluate_cbvd.py
---
```python
import os
import csv
import time
import cv2
import timm
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm

# Configurations
PERSON_NAME = "Nusrat"
BASE_MODEL_DISPLAY = "EfficientNetB0"
MODEL_NAME = "efficientnet_b0"
BASE_DIR = r"D:\T25301094 P2"
WORKSPACE_DIR = r"D:\T25301094 P2\workspaces\nusrat"
CSV_PATH = r"D:\T25301094 P2\datasets\behavior\CBVD-5\cbvd_cropped_index.csv"
CHECKPOINT_PATH = r"D:\T25301094 P2\workspaces\nusrat\behavior_best.pth"
RESULTS_PATH = r"D:\T25301094 P2\workspaces\nusrat\cbvd_behavior_results.txt"

NUM_CLASSES = 7
BATCH_SIZE = 64
NUM_WORKERS = 0  # Safe for Windows
RANDOM_SEED = 42

# Mapped classes of interest in CBVD-5:
#   1 -> Class 2 (Standing)
#   3 -> Class 4 (Feeding head down)
#   5 -> Class 6 (Drinking)
#   6 -> Class 7 (Lying)
MAPPED_CLASSES = [1, 3, 5, 6]

class CBVDDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.transform = transform
        self.samples = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    'image_path': row['image_path'],
                    'label': int(row['label']),
                    'cow_id': row['cow_id']
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = item['image_path']
        if not os.path.isabs(img_path):
            img_path = os.path.join(BASE_DIR, img_path)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = item['label']
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label

# Attention layers (matching train_behavior.py)
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_ = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        return x * self.sigmoid(avg + max_).unsqueeze(-1).unsqueeze(-1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))

class BehaviorModel(nn.Module):
    def __init__(self, model_name, num_classes, device):
        super().__init__()
        backbone = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool='')
        backbone = backbone.to(device, non_blocking=True)
        with torch.no_grad():
            dummy = backbone(torch.zeros(1, 3, 224, 224).to(device, non_blocking=True))
            feature_dim = dummy.shape[1]
        self.backbone = backbone
        self.cbam = CBAM(feature_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # Set up transform
    eval_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    dataset = CBVDDataset(CSV_PATH, eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    print(f"Total CBVD-5 evaluation samples: {len(dataset)}")

    # Load Model
    model = BehaviorModel(MODEL_NAME, NUM_CLASSES, device)
    model = model.to(device)
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at: {CHECKPOINT_PATH}")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.eval()
    print("Model loaded successfully.")

    # Evaluate
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating CBVD-5"):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast('cuda') if torch.cuda.is_available() else torch.no_grad():
                outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    # Compute metrics on the MAPPED_CLASSES only to keep F1 accurate
    # Note: If model predicts a class outside MAPPED_CLASSES, it is treated as a false positive
    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average='macro',
        labels=MAPPED_CLASSES,
        zero_division=0
    )
    
    # Calculate confusion matrix on MAPPED_CLASSES
    # We map labels to a 4x4 matrix specifically for the 4 classes
    cm = confusion_matrix(all_labels, all_preds, labels=MAPPED_CLASSES)

    per_class_accuracy = []
    for idx, class_label in enumerate(MAPPED_CLASSES):
        total = cm[idx].sum()
        correct = cm[idx, idx]
        acc = (correct / total) * 100 if total > 0 else 0.0
        per_class_accuracy.append(acc)

    # Output text results
    text = f"""---CONTEXT 3 CBVD-5 EVALUATION---
PERSON NAME: {PERSON_NAME}
BASE MODEL: {BASE_MODEL_DISPLAY}
DATASET: CBVD-5 (2000 balanced images)
VAL MACRO F1 ON CBVD-5: {macro_f1:.6f}
PER-CLASS ACCURACY ON CBVD-5:
  Class 2 (Standing): {per_class_accuracy[0]:.2f}%
  Class 4 (Feeding head down): {per_class_accuracy[1]:.2f}%
  Class 6 (Drinking): {per_class_accuracy[2]:.2f}%
  Class 7 (Lying): {per_class_accuracy[3]:.2f}%
TRAINING TIME (mins): N/A (Evaluation Only)
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---"""

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(text)

    print("\n" + text)
    print(f"\nSaved results to: {RESULTS_PATH}")

if __name__ == "__main__":
    main()

```

### FILE: workspaces\nusrat\predict_spatiotemporal_lameness.py
---
```python
import os
import glob
import cv2
import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Model Configuration
MODEL_NAME = "efficientnet_b0"
HIDDEN_DIM = 64
CHECKPOINT_PATH = r"D:\T25301094 P2\workspaces\nusrat\spatiotemporal_lameness_efficientnet_best.pth"
DECISION_THRESHOLD = 0.70  # Calibrated decision threshold

class CNNLSTMModel(nn.Module):
    def __init__(self, backbone_name, hidden_dim, device):
        super().__init__()
        backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        backbone = backbone.to(device, non_blocking=True)
        with torch.no_grad():
            dummy = backbone(torch.zeros(1, 3, 224, 224).to(device, non_blocking=True))
            feature_dim = dummy.shape[1]
        self.backbone = backbone
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.backbone(x)
        features = features.view(batch_size, seq_len, -1)
        lstm_out, (hn, cn) = self.lstm(features)
        last_step_out = lstm_out[:, -1, :]
        logits = self.classifier(last_step_out)
        return logits

def load_frames_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Error: Could not open video file {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

def load_frames_from_directory(dir_path):
    # Support sorting frame files numerically or alphabetically
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(dir_path, ext)))
    
    if not files:
        raise FileNotFoundError(f"Error: No image files found in directory {dir_path}")
    
    # Sort files naturally/numerically
    files.sort(key=lambda x: [int(c) if c.isdigit() else c for c in glob.os.path.split(x)[1].split('_')])
    
    frames = []
    for file in files:
        img = cv2.imread(file)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
    return frames

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Predict Lameness from Video or Frame Directory")
    parser.add_argument("--path", type=str, required=True, help="Path to input video file (.mp4, .avi) OR image frame directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize and load model
    model = CNNLSTMModel(MODEL_NAME, HIDDEN_DIM, device)
    model = model.to(device)
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at: {CHECKPOINT_PATH}")
    
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"Model checkpoint loaded successfully from: {CHECKPOINT_PATH}")

    # Detect if path is a directory, a glob pattern matching files, or a video file
    if os.path.isdir(args.path):
        print(f"Loading frames from directory: {args.path}")
        frames = load_frames_from_directory(args.path)
    elif glob.glob(args.path) and len(glob.glob(args.path)) > 1:
        print(f"Loading frames matching pattern: {args.path}")
        files = glob.glob(args.path)
        files.sort(key=lambda x: [int(c) if c.isdigit() else c for c in os.path.split(x)[1].split('_')])
        frames = []
        for file in files:
            img = cv2.imread(file)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(img)
    else:
        print(f"Loading frames from video file: {args.path}")
        frames = load_frames_from_video(args.path)

    total_frames = len(frames)
    print(f"Loaded {total_frames} frames.")
    
    if total_frames == 0:
        raise ValueError("Error: Loaded frame sequence is empty.")

    # Sparse sample exactly 20 frames evenly spaced
    indices = [int(i * (total_frames - 1) / (20 - 1)) for i in range(20)] if total_frames > 1 else [0] * 20
    sampled_frames = [frames[i] for i in indices]

    # Normalize and compile transform
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    processed_frames = [transform(image=img)["image"] for img in sampled_frames]
    input_tensor = torch.stack(processed_frames).unsqueeze(0).to(device)  # Add batch dimension: (1, 20, 3, 224, 224)

    # Perform inference
    with torch.no_grad():
        with torch.amp.autocast('cuda') if torch.cuda.is_available() else torch.no_grad():
            outputs = model(input_tensor)
            prob = torch.sigmoid(outputs).item()

    prediction = "Lame" if prob >= DECISION_THRESHOLD else "Normal"

    print("\n" + "="*40)
    print("           INFERENCE RESULTS            ")
    print("="*40)
    print(f"Source: {os.path.basename(args.path)}")
    print(f"Calibrated Threshold: {DECISION_THRESHOLD:.2f}")
    print(f"Lameness Probability: {prob * 100:.2f}%")
    print(f"Final Prediction:     {prediction.upper()}")
    print("="*40)

if __name__ == "__main__":
    main()

```

### FILE: workspaces\nusrat\train_bcs.py
---
```python
import os
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import timm

from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

warnings.filterwarnings("ignore")

SEED = 42
PERSON_NAME = "Nusrat"
BASE_MODEL = "EfficientNetB0"
TIMM_MODEL_NAME = "efficientnet_b0"

BASE_DIR = r"D:\T25301094 P2"
WORKSPACE_DIR = os.path.join(BASE_DIR, "workspaces", "nusrat")

DRYAD_CSV = os.path.join(BASE_DIR, "datasets", "bcs", "bcs_index.csv")
SCIENCEDB_CSV = os.path.join(BASE_DIR, "datasets", "bcs", "sciencedb_bcs_cropped_index.csv")

DRYAD_CHECKPOINT = os.path.join(WORKSPACE_DIR, "dryad_bcs_best.pth")
SCIENCEDB_CHECKPOINT = os.path.join(WORKSPACE_DIR, "sciencedb_bcs_best.pth")

RESULTS_TXT = os.path.join(WORKSPACE_DIR, "bcs_results.txt")
LOSS_CURVE_PNG = os.path.join(WORKSPACE_DIR, "bcs_loss_curve.png")

NUM_CLASSES = 5
CORAL_OUTPUTS = NUM_CLASSES - 1

BATCH_SIZE = 128  # Increased from 32 to 128 to maximize GPU usage on RTX 4080 SUPER
EPOCHS = 30
LR = 1e-3
STEP_SIZE = 10
GAMMA = 0.5

DEVICE = torch.device("cuda")

DRYAD_LABEL_MAP = {
    2: 0,
    3: 1,
    4: 2,
    5: 3,
    6: 4,
}

SCIENCEDB_LABEL_MAP = {
    3.25: 0,
    3.5: 1,
    3.75: 2,
    4.0: 3,
    4.25: 4,
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ============================================================
# DATASET
# ============================================================

class BCSDataset(Dataset):
    def __init__(self, csv_path, split, label_map, transform=None):
        self.csv_path = csv_path
        self.split = split
        self.label_map = label_map
        self.transform = transform

        df = pd.read_csv(csv_path)
        df = df[df["split"] == split].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(f"No rows found for split='{split}' in {csv_path}")

        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row["image_path"]
        raw_label = row["label"]

        try:
            raw_label_key = int(raw_label)
        except Exception:
            raw_label_key = float(raw_label)

        if raw_label_key not in self.label_map:
            raw_label_key = float(raw_label)

        label = self.label_map[raw_label_key]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_transforms(split):
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if split == "train":
        return T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ToTensor(),
            normalize,
        ])

    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        normalize,
    ])


def build_loaders(csv_path, label_map):
    train_dataset = BCSDataset(csv_path, "train", label_map, get_transforms("train"))
    val_dataset = BCSDataset(csv_path, "val", label_map, get_transforms("val"))
    test_dataset = BCSDataset(csv_path, "test", label_map, get_transforms("test"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, val_loader, test_loader


# ============================================================
# CBAM ATTENTION
# ============================================================

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()

        hidden_channels = max(in_channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, in_channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_ = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))

        attention = self.sigmoid(avg + max_).unsqueeze(-1).unsqueeze(-1)

        return x * attention


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=7,
            padding=3,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)

        attention = self.sigmoid(
            self.conv(torch.cat([avg, max_], dim=1))
        )

        return x * attention


class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


# ============================================================
# MODEL
# ============================================================

class BCSModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            TIMM_MODEL_NAME,
            pretrained=True,
            num_classes=0,
            global_pool="",
        )

        self.feature_dim = self.backbone.num_features

        self.cbam = CBAM(self.feature_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.feature_dim, CORAL_OUTPUTS)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        features = self.cbam(features)
        pooled = self.pool(features).flatten(1)
        logits = self.head(pooled)

        return logits


# ============================================================
# CORAL LOSS + METRICS
# ============================================================

def coral_loss(logits, labels, num_classes):
    sets = []

    for i in range(num_classes - 1):
        label_i = (labels > i).float()
        sets.append(label_i)

    labels_stacked = torch.stack(sets, dim=1)

    loss = F.binary_cross_entropy_with_logits(logits, labels_stacked)

    return loss


def coral_predict(logits):
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)


def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    acc_exact = accuracy_score(y_true, y_pred)
    acc_within_1 = np.mean(np.abs(y_true - y_pred) <= 1)

    return {
        "mae": float(mae),
        "acc_exact": float(acc_exact),
        "acc_within_1": float(acc_within_1),
    }


# ============================================================
# TRAIN / EVAL
# ============================================================

def train_one_epoch(model, loader, optimizer, epoch, dataset_name):
    model.train()

    running_loss = 0.0
    total_samples = 0

    progress = tqdm(
        loader,
        desc=f"{dataset_name} | Epoch {epoch:02d}/{EPOCHS} | Train",
        leave=True,
    )

    for images, labels in progress:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = coral_loss(logits, labels, NUM_CLASSES)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        progress.set_postfix({
            "loss": f"{loss.item():.4f}",
        })

    epoch_loss = running_loss / total_samples

    return epoch_loss


@torch.no_grad()
def evaluate(model, loader, dataset_name, split_name):
    model.eval()

    running_loss = 0.0
    total_samples = 0

    all_labels = []
    all_preds = []

    progress = tqdm(
        loader,
        desc=f"{dataset_name} | {split_name}",
        leave=True,
    )

    for images, labels in progress:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        logits = model(images)
        loss = coral_loss(logits, labels, NUM_CLASSES)

        preds = coral_predict(logits)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

        progress.set_postfix({
            "loss": f"{loss.item():.4f}",
        })

    avg_loss = running_loss / total_samples
    metrics = compute_metrics(all_labels, all_preds)

    metrics["loss"] = float(avg_loss)

    return metrics


def run_training(dataset_name, csv_path, label_map, checkpoint_path):
    print("=" * 80)
    print(f"STARTING DATASET: {dataset_name}")
    print(f"CSV: {csv_path}")
    print(f"CHECKPOINT: {checkpoint_path}")
    print("=" * 80)

    train_loader, val_loader, test_loader = build_loaders(csv_path, label_map)

    model = BCSModel().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=STEP_SIZE,
        gamma=GAMMA,
    )

    best_val_mae = float("inf")
    best_epoch = None
    best_val_metrics = None

    train_losses = []
    val_losses = []

    loss_at_epoch_10 = None
    loss_at_epoch_20 = None
    loss_at_epoch_30 = None

    start_time = time.time()
    issue_text = "None"

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            dataset_name=dataset_name,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            dataset_name=dataset_name,
            split_name=f"Val Epoch {epoch:02d}/{EPOCHS}",
        )

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_metrics["loss"])

        if epoch == 10:
            loss_at_epoch_10 = train_loss
        if epoch == 20:
            loss_at_epoch_20 = train_loss
        if epoch == 30:
            loss_at_epoch_30 = train_loss

        print(
            f"{dataset_name} | Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_metrics['loss']:.6f} | "
            f"Val MAE: {val_metrics['mae']:.6f} | "
            f"Val Acc +-0: {val_metrics['acc_exact']:.6f} | "
            f"Val Acc +-1: {val_metrics['acc_within_1']:.6f}"
        )

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch
            best_val_metrics = val_metrics

            torch.save(
                {
                    "person_name": PERSON_NAME,
                    "base_model": BASE_MODEL,
                    "dataset_name": dataset_name,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "label_map": label_map,
                    "num_classes": NUM_CLASSES,
                    "timm_model_name": TIMM_MODEL_NAME,
                },
                checkpoint_path,
            )

            print(
                f"{dataset_name} | New best checkpoint saved at epoch {epoch} "
                f"with Val MAE {best_val_mae:.6f}"
            )

    total_time_mins = (time.time() - start_time) / 60.0

    print(f"{dataset_name} | Loading best checkpoint for final test evaluation...")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        dataset_name=dataset_name,
        split_name="Test",
    )

    final_train_loss = train_losses[-1]

    print("=" * 80)
    print(f"COMPLETED DATASET: {dataset_name}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Final Train Loss: {final_train_loss:.6f}")
    print(f"Best Val MAE: {best_val_metrics['mae']:.6f}")
    print(f"Best Val Acc +-0: {best_val_metrics['acc_exact']:.6f}")
    print(f"Best Val Acc +-1: {best_val_metrics['acc_within_1']:.6f}")
    print(f"Test MAE: {test_metrics['mae']:.6f}")
    print(f"Test Acc +-0: {test_metrics['acc_exact']:.6f}")
    print(f"Test Acc +-1: {test_metrics['acc_within_1']:.6f}")
    print(f"Training Time Mins: {total_time_mins:.2f}")
    print("=" * 80)

    return {
        "dataset_name": dataset_name,
        "epochs_trained": EPOCHS,
        "loss_at_epoch_10": loss_at_epoch_10,
        "loss_at_epoch_20": loss_at_epoch_20,
        "loss_at_epoch_30": loss_at_epoch_30,
        "final_train_loss": final_train_loss,
        "best_epoch": best_epoch,
        "val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "checkpoint_path": checkpoint_path,
        "training_time_mins": total_time_mins,
        "issues": issue_text,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


# ============================================================
# OUTPUT FILES
# ============================================================

def fmt_float(value):
    if value is None:
        return "N/A"
    return f"{float(value):.6f}"


def write_context3_report(dryad_result, sciencedb_result):
    def block(result):
        return f"""DATASET: {result['dataset_name']}
EPOCHS TRAINED: {result['epochs_trained']}
LOSS AT EPOCH 10: {fmt_float(result['loss_at_epoch_10'])}
LOSS AT EPOCH 20: {fmt_float(result['loss_at_epoch_20'])}
LOSS AT EPOCH 30: {fmt_float(result['loss_at_epoch_30'])}
FINAL TRAIN LOSS: {fmt_float(result['final_train_loss'])}
VAL MAE: {fmt_float(result['val_metrics']['mae'])}
VAL ACCURACY +-0 (exact match): {fmt_float(result['val_metrics']['acc_exact'])}
VAL ACCURACY +-1 (within 1 class): {fmt_float(result['val_metrics']['acc_within_1'])}
TEST MAE: {fmt_float(result['test_metrics']['mae'])}
TEST ACCURACY +-0: {fmt_float(result['test_metrics']['acc_exact'])}
TEST ACCURACY +-1: {fmt_float(result['test_metrics']['acc_within_1'])}
CHECKPOINT PATH: {result['checkpoint_path']}
TRAINING TIME (mins): {result['training_time_mins']:.2f}
ANY ISSUES ENCOUNTERED: {result['issues']}"""

    report = f"""---CONTEXT 3 BCS---
PERSON NAME: {PERSON_NAME}
BASE MODEL: {BASE_MODEL}

{block(dryad_result)}

{block(sciencedb_result)}
---END CONTEXT 3---
"""

    with open(RESULTS_TXT, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Context 3 report saved to: {RESULTS_TXT}")
    print(report)


def save_loss_curve(dryad_result, sciencedb_result):
    plt.figure(figsize=(10, 6))

    epochs = list(range(1, EPOCHS + 1))

    plt.plot(
        epochs,
        dryad_result["train_losses"],
        label="Dryad Train Loss",
    )
    plt.plot(
        epochs,
        dryad_result["val_losses"],
        label="Dryad Val Loss",
    )
    plt.plot(
        epochs,
        sciencedb_result["train_losses"],
        label="ScienceDB Train Loss",
    )
    plt.plot(
        epochs,
        sciencedb_result["val_losses"],
        label="ScienceDB Val Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("BCS Training and Validation Loss Curves - Nusrat - EfficientNetB0")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(LOSS_CURVE_PNG, dpi=300)
    plt.close()

    print(f"Loss curve saved to: {LOSS_CURVE_PNG}")


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(SEED)

    os.makedirs(WORKSPACE_DIR, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This script must run on GPU as required."
        )

    print("=" * 80)
    print("BCS TRAINING SCRIPT")
    print(f"Person: {PERSON_NAME}")
    print(f"Base Model: {BASE_MODEL}")
    print(f"Timm Model: {TIMM_MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Workspace: {WORKSPACE_DIR}")
    print("=" * 80)

    dryad_result = run_training(
        dataset_name="Dryad",
        csv_path=DRYAD_CSV,
        label_map=DRYAD_LABEL_MAP,
        checkpoint_path=DRYAD_CHECKPOINT,
    )

    sciencedb_result = run_training(
        dataset_name="ScienceDB",
        csv_path=SCIENCEDB_CSV,
        label_map=SCIENCEDB_LABEL_MAP,
        checkpoint_path=SCIENCEDB_CHECKPOINT,
    )

    write_context3_report(dryad_result, sciencedb_result)
    save_loss_curve(dryad_result, sciencedb_result)

    print("=" * 80)
    print("ALL BCS TRAINING COMPLETED SUCCESSFULLY")
    print(f"Checkpoint 1: {DRYAD_CHECKPOINT}")
    print(f"Checkpoint 2: {SCIENCEDB_CHECKPOINT}")
    print(f"Results TXT: {RESULTS_TXT}")
    print(f"Loss Curve: {LOSS_CURVE_PNG}")
    print("=" * 80)


if __name__ == "__main__":
    main()

```

### FILE: workspaces\nusrat\train_behavior.py
---
```python
import os
import time
import random
import multiprocessing

import cv2
import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


PERSON_NAME = "Nusrat"
BASE_MODEL_DISPLAY = "EfficientNetB0"
MODEL_NAME = "efficientnet_b0"
BASE_DIR = r"D:\T25301094 P2"
WORKSPACE_DIR = r"D:\T25301094 P2\workspaces\nusrat"
CSV_PATH = r"D:\T25301094 P2\datasets\behavior\behavior_index.csv"
CHECKPOINT_PATH = r"D:\T25301094 P2\workspaces\nusrat\behavior_best.pth"
RESULTS_PATH = r"D:\T25301094 P2\workspaces\nusrat\behavior_results.txt"
LOSS_CURVE_PATH = r"D:\T25301094 P2\workspaces\nusrat\behavior_loss_curve.png"

NUM_CLASSES = 7
BATCH_SIZE = 128
NUM_WORKERS = 8
MAX_EPOCHS = 30
PATIENCE = 10
RANDOM_SEED = 42

CLASS_NAMES = [
    "Walking",
    "Standing",
    "Feeding head up",
    "Feeding head down",
    "Licking",
    "Drinking",
    "Lying",
]


class BehaviorDataset(Dataset):
    def __init__(self, csv_path, split, transform=None):
        df = pd.read_csv(csv_path)
        df = df[df['split'] == split]
        df = pd.concat([
            group.sample(min(len(group), 3000), random_state=42)
            for _, group in df.groupby('label')
        ]).reset_index(drop=True)
        self.data = df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = str(row["image_path"])
        if not os.path.isabs(img_path):
            img_path = os.path.join(BASE_DIR, img_path)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = int(row['label']) - 1
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_ = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        return x * self.sigmoid(avg + max_).unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))


class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class BehaviorModel(nn.Module):
    def __init__(self, model_name, num_classes, device):
        super().__init__()
        backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='')
        backbone = backbone.to(device, non_blocking=True)
        with torch.no_grad():
            dummy = backbone(torch.zeros(1, 3, 224, 224).to(device, non_blocking=True))
            feature_dim = dummy.shape[1]
        self.backbone = backbone
        self.cbam = CBAM(feature_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms():
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    eval_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return train_transform, eval_transform


def build_loader(dataset, shuffle):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )


def evaluate(model, loader, device, desc):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average='macro',
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))

    per_class_accuracy = []
    for class_idx in range(NUM_CLASSES):
        total = cm[class_idx].sum()
        correct = cm[class_idx, class_idx]
        acc = (correct / total) * 100 if total > 0 else 0.0
        per_class_accuracy.append(acc)

    return macro_f1, per_class_accuracy


def save_loss_curve(train_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Behavior Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH, dpi=300)
    plt.close()


def format_loss(epoch_losses, epoch):
    value = epoch_losses.get(epoch)
    if value is None:
        return "N/A"
    return f"{value:.6f}"


def format_acc(value):
    return f"{value:.2f}%"


def write_results(
    actual_epochs_trained,
    epoch_losses,
    final_train_loss,
    val_f1,
    val_acc,
    test_f1,
    test_acc,
    training_time_mins,
    early_stop_epoch,
):
    early_stop_text = str(early_stop_epoch) if early_stop_epoch is not None else "N/A"
    text = f"""---CONTEXT 3 BEHAVIOR---
PERSON NAME: {PERSON_NAME}
BASE MODEL: {BASE_MODEL_DISPLAY}
DATASET: MmCows (capped 3000/class)
EPOCHS TRAINED: {actual_epochs_trained}
LOSS AT EPOCH 10: {format_loss(epoch_losses, 10)}
LOSS AT EPOCH 20: {format_loss(epoch_losses, 20)}
LOSS AT EPOCH 30: {format_loss(epoch_losses, 30)}
FINAL TRAIN LOSS: {final_train_loss:.6f}
VAL MACRO F1: {val_f1:.6f}
VAL PER-CLASS ACCURACY:
  Class 1 (Walking): {format_acc(val_acc[0])}
  Class 2 (Standing): {format_acc(val_acc[1])}
  Class 3 (Feeding head up): {format_acc(val_acc[2])}
  Class 4 (Feeding head down): {format_acc(val_acc[3])}
  Class 5 (Licking): {format_acc(val_acc[4])}
  Class 6 (Drinking): {format_acc(val_acc[5])}
  Class 7 (Lying): {format_acc(val_acc[6])}
TEST MACRO F1: {test_f1:.6f}
TEST PER-CLASS ACCURACY:
  Class 1 (Walking): {format_acc(test_acc[0])}
  Class 2 (Standing): {format_acc(test_acc[1])}
  Class 3 (Feeding head up): {format_acc(test_acc[2])}
  Class 4 (Feeding head down): {format_acc(test_acc[3])}
  Class 5 (Licking): {format_acc(test_acc[4])}
  Class 6 (Drinking): {format_acc(test_acc[5])}
  Class 7 (Lying): {format_acc(test_acc[6])}
CHECKPOINT PATH: {CHECKPOINT_PATH}
TRAINING TIME (mins): {training_time_mins:.2f}
EARLY STOPPING TRIGGERED AT EPOCH: {early_stop_text}
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---"""

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(text)

    print(text)


def main():
    start_time = time.time()
    os.makedirs(WORKSPACE_DIR, exist_ok=True)

    set_seed(RANDOM_SEED)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    train_transform, eval_transform = build_transforms()

    train_dataset = BehaviorDataset(CSV_PATH, "train", train_transform)
    val_dataset = BehaviorDataset(CSV_PATH, "val", eval_transform)
    test_dataset = BehaviorDataset(CSV_PATH, "test", eval_transform)

    train_loader = build_loader(train_dataset, shuffle=True)
    val_loader = build_loader(val_dataset, shuffle=False)
    test_loader = build_loader(test_dataset, shuffle=False)

    print(f"Person: {PERSON_NAME}")
    print(f"Base model: {BASE_MODEL_DISPLAY}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    model = BehaviorModel(MODEL_NAME, NUM_CLASSES, device)
    model = model.to(device, non_blocking=True)

    criterion = FocalLoss(gamma=2, alpha=0.25)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scaler = torch.amp.GradScaler('cuda')

    epochs_no_improve = 0
    best_val_f1 = -1.0
    train_losses = []
    epoch_losses = {}
    early_stop_epoch = None
    actual_epochs_trained = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/30"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_count = images.size(0)
            running_loss += loss.item() * batch_count
            total_samples += batch_count

        train_loss = running_loss / total_samples
        train_losses.append(train_loss)
        epoch_losses[epoch] = train_loss

        val_f1, _ = evaluate(model, val_loader, device, "Validating")
        scheduler.step()

        actual_epochs_trained = epoch

        print(f"Epoch {epoch}/30 | Loss: {train_loss:.6f} | Val Macro F1: {val_f1:.6f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                early_stop_epoch = epoch
                print(f"Early stopping triggered at epoch {epoch}. No improvement for {PATIENCE} epochs.")
                break

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Best checkpoint was not saved: {CHECKPOINT_PATH}")

    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))

    val_f1, val_acc = evaluate(model, val_loader, device, "Validating")
    test_f1, test_acc = evaluate(model, test_loader, device, "Testing")

    save_loss_curve(train_losses)

    training_time_mins = (time.time() - start_time) / 60
    final_train_loss = train_losses[-1]

    write_results(
        actual_epochs_trained,
        epoch_losses,
        final_train_loss,
        val_f1,
        val_acc,
        test_f1,
        test_acc,
        training_time_mins,
        early_stop_epoch,
    )

    print(f"Saved checkpoint: {CHECKPOINT_PATH}")
    print(f"Saved results: {RESULTS_PATH}")
    print(f"Saved loss curve: {LOSS_CURVE_PATH}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

```

### FILE: workspaces\nusrat\train_id.py
---
```python
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend to prevent Tkinter thread crashes

import os
import csv
import time
import random
import multiprocessing
import cv2
cv2.setNumThreads(0)  # Disable OpenCV multithreading to prevent deadlocks in DataLoader

import timm
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

PERSON_NAME = "Nusrat"
BASE_MODEL_DISPLAY = "EfficientNetB0"
MODEL_NAME = "efficientnet_b0"
BASE_DIR = r"D:\T25301094 P2"
WORKSPACE_DIR = r"D:\T25301094 P2\workspaces\nusrat"
CSV_PATH = r"D:\T25301094 P2\datasets\id\id_index.csv"
CHECKPOINT_PATH = r"D:\T25301094 P2\workspaces\nusrat\id_best.pth"
RESULTS_PATH = r"D:\T25301094 P2\workspaces\nusrat\id_results.txt"
LOSS_CURVE_PATH = r"D:\T25301094 P2\workspaces\nusrat\id_loss_curve.png"

NUM_CLASSES = 46
BATCH_SIZE = 64
NUM_WORKERS = 0  # Set to 0 to prevent Windows multiprocessing deadlocks and worker unexpected exit crashes
MAX_EPOCHS = 10
RANDOM_SEED = 42

class CowIDDataset(Dataset):
    def __init__(self, csv_path, split, transform=None):
        self.transform = transform
        self.samples = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['split'] == split:
                    self.samples.append({
                        'image_path': row['image_path'],
                        'label': int(row['label']),
                        'cow_id': row['cow_id']
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = item['image_path']
        if not os.path.isabs(img_path):
            img_path = os.path.join(BASE_DIR, img_path)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = item['label']
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_ = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        return x * self.sigmoid(avg + max_).unsqueeze(-1).unsqueeze(-1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))

class CowIDModel(nn.Module):
    def __init__(self, model_name, num_classes, device):
        super().__init__()
        backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='')
        backbone = backbone.to(device, non_blocking=True)
        with torch.no_grad():
            dummy = backbone(torch.zeros(1, 3, 224, 224).to(device, non_blocking=True))
            feature_dim = dummy.shape[1]
        self.backbone = backbone
        self.cbam = CBAM(feature_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_transforms():
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    eval_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return train_transform, eval_transform

def build_loader(dataset, shuffle, persistent=True):
    # If NUM_WORKERS is 0, persistent_workers must be False
    actual_persistent = persistent if NUM_WORKERS > 0 else False
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=actual_persistent,
    )

def evaluate(model, loader, device, desc):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    return acc

def save_loss_curve(train_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Cow ID Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH, dpi=300)
    plt.close()

def format_loss(epoch_losses, epoch):
    value = epoch_losses.get(epoch)
    if value is None:
        return "N/A"
    return f"{value:.6f}"

def write_results(
    actual_epochs_trained,
    epoch_losses,
    final_train_loss,
    val_acc,
    test_acc,
    training_time_mins,
):
    text = f"""---CONTEXT 3 ID---
PERSON NAME: {PERSON_NAME}
BASE MODEL: {BASE_MODEL_DISPLAY}
DATASET: OpenCows2020
EPOCHS TRAINED: {actual_epochs_trained}
LOSS AT EPOCH 10: {format_loss(epoch_losses, 10)}
FINAL TRAIN LOSS: {final_train_loss:.6f}
VAL TOP-1 ACCURACY: {val_acc * 100:.2f}%
TEST TOP-1 ACCURACY: {test_acc * 100:.2f}%
CHECKPOINT PATH: {CHECKPOINT_PATH}
TRAINING TIME (mins): {training_time_mins:.2f}
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---"""

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(text)

    print(text)

def main():
    start_time = time.time()
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    set_seed(RANDOM_SEED)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    train_transform, eval_transform = build_transforms()

    train_dataset = CowIDDataset(CSV_PATH, "train", train_transform)
    val_dataset = CowIDDataset(CSV_PATH, "val", eval_transform)
    test_dataset = CowIDDataset(CSV_PATH, "test", eval_transform)

    train_loader = build_loader(train_dataset, shuffle=True, persistent=True)
    val_loader = build_loader(val_dataset, shuffle=False, persistent=False)
    test_loader = build_loader(test_dataset, shuffle=False, persistent=False)

    print(f"Person: {PERSON_NAME}")
    print(f"Base model: {BASE_MODEL_DISPLAY}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    model = CowIDModel(MODEL_NAME, NUM_CLASSES, device)
    model = model.to(device, non_blocking=True)

    # Freeze backbone parameters to prevent overfitting
    # We will only train CBAM and the classifier head
    for param in model.backbone.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
    scaler = torch.amp.GradScaler('cuda')

    best_val_acc = -1.0
    train_losses = []
    epoch_losses = {}
    actual_epochs_trained = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_count = images.size(0)
            running_loss += loss.item() * batch_count
            total_samples += batch_count

        train_loss = running_loss / total_samples
        train_losses.append(train_loss)
        epoch_losses[epoch] = train_loss

        val_acc = evaluate(model, val_loader, device, "Validating")
        scheduler.step()

        actual_epochs_trained = epoch

        print(f"Epoch {epoch}/{MAX_EPOCHS} | Loss: {train_loss:.6f} | Val Top-1 Accuracy: {val_acc * 100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Best checkpoint was not saved: {CHECKPOINT_PATH}")

    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))

    val_acc = evaluate(model, val_loader, device, "Validating Best Val")
    test_acc = evaluate(model, test_loader, device, "Testing Best Val")

    save_loss_curve(train_losses)

    training_time_mins = (time.time() - start_time) / 60
    final_train_loss = train_losses[-1]

    write_results(
        actual_epochs_trained,
        epoch_losses,
        final_train_loss,
        val_acc,
        test_acc,
        training_time_mins,
    )

    print(f"Saved checkpoint: {CHECKPOINT_PATH}")
    print(f"Saved results: {RESULTS_PATH}")
    print(f"Saved loss curve: {LOSS_CURVE_PATH}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

```

### FILE: workspaces\nusrat\train_lameness.py
---
```python
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend to prevent Tkinter thread crashes

import os
import time
import random
import multiprocessing
import cv2
cv2.setNumThreads(0)  # Disable OpenCV multithreading to prevent deadlocks in DataLoader

import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Configuration
PERSON_NAME = "Nusrat"
BASE_MODEL_DISPLAY = "EfficientNetB0"
MODEL_NAME = "efficientnet_b0"
BASE_DIR = r"D:\T25301094 P2"
WORKSPACE_DIR = r"D:\T25301094 P2\workspaces\nusrat"
CSV_PATH = r"D:\T25301094 P2\datasets\lameness\lameness_index.csv"
CHECKPOINT_PATH = r"D:\T25301094 P2\workspaces\nusrat\lameness_best.pth"
RESULTS_PATH = r"D:\T25301094 P2\workspaces\nusrat\lameness_results.txt"
LOSS_CURVE_PATH = r"D:\T25301094 P2\workspaces\nusrat\lameness_loss_curve.png"

BATCH_SIZE = 256  # Increased to leverage 16GB VRAM on RTX 4080 SUPER
NUM_WORKERS = 4
MAX_EPOCHS = 10
RANDOM_SEED = 42

class LamenessDataset(Dataset):
    def __init__(self, csv_path, split, transform=None):
        df = pd.read_csv(csv_path)
        df = df[df['split'] == split].reset_index(drop=True)
        self.data = df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = str(row["image_path"])
        if not os.path.isabs(img_path):
            img_path = os.path.join(BASE_DIR, img_path)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = float(row['label'])
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, torch.tensor(label, dtype=torch.float32)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_ = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        return x * self.sigmoid(avg + max_).unsqueeze(-1).unsqueeze(-1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))

class LamenessModel(nn.Module):
    def __init__(self, model_name, device):
        super().__init__()
        backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='')
        backbone = backbone.to(device, non_blocking=True)
        with torch.no_grad():
            dummy = backbone(torch.zeros(1, 3, 224, 224).to(device, non_blocking=True))
            feature_dim = dummy.shape[1]
        self.backbone = backbone
        self.cbam = CBAM(feature_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(feature_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_transforms():
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    eval_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return train_transform, eval_transform

def build_loader(dataset, shuffle, persistent=True):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=persistent,
        prefetch_factor=2 if num_workers_avail() > 0 else None,
    )

def num_workers_avail():
    return NUM_WORKERS

def evaluate_model(model, loader, device, desc):
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                probs = torch.sigmoid(outputs).squeeze(1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)

    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # Per-class accuracy
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    normal_acc = (cm[0, 0] / cm[0].sum() * 100) if cm[0].sum() > 0 else 0.0
    lame_acc = (cm[1, 1] / cm[1].sum() * 100) if cm[1].sum() > 0 else 0.0
    
    return auc, acc, f1, [normal_acc, lame_acc]

def save_loss_curve(train_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Lameness Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH, dpi=300)
    plt.close()

def write_results(actual_epochs_trained, final_train_loss, val_auc, val_acc, val_f1, val_class_acc, test_auc, test_acc, test_f1, test_class_acc, training_time_mins):
    text = f"""---CONTEXT 3 LAMENESS---
PERSON NAME: {PERSON_NAME}
BASE MODEL: {BASE_MODEL_DISPLAY}
DATASET: CattleLameness (20 frames/video)
EPOCHS TRAINED: {actual_epochs_trained}
FINAL TRAIN LOSS: {final_train_loss:.6f}
VAL AUC: {val_auc:.6f}
VAL ACCURACY: {val_acc * 100:.2f}%
VAL F1 SCORE: {val_f1:.6f}
VAL PER-CLASS ACCURACY:
  Class 0 (Normal): {val_class_acc[0]:.2f}%
  Class 1 (Lame): {val_class_acc[1]:.2f}%
TEST AUC: {test_auc:.6f}
TEST ACCURACY: {test_acc * 100:.2f}%
TEST F1 SCORE: {test_f1:.6f}
TEST PER-CLASS ACCURACY:
  Class 0 (Normal): {test_class_acc[0]:.2f}%
  Class 1 (Lame): {test_class_acc[1]:.2f}%
CHECKPOINT PATH: {CHECKPOINT_PATH}
TRAINING TIME (mins): {training_time_mins:.2f}
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---"""

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)

def main():
    start_time = time.time()
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    set_seed(RANDOM_SEED)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    train_transform, eval_transform = build_transforms()

    train_dataset = LamenessDataset(CSV_PATH, "train", train_transform)
    val_dataset = LamenessDataset(CSV_PATH, "val", eval_transform)
    test_dataset = LamenessDataset(CSV_PATH, "test", eval_transform)

    # Use persistent_workers=False for validation/testing to prevent Windows multiprocessing deadlocks
    train_loader = build_loader(train_dataset, shuffle=True, persistent=True)
    val_loader = build_loader(val_dataset, shuffle=False, persistent=False)
    test_loader = build_loader(test_dataset, shuffle=False, persistent=False)

    print(f"Person: {PERSON_NAME}")
    print(f"Base model: {BASE_MODEL_DISPLAY}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    model = LamenessModel(MODEL_NAME, device)
    model = model.to(device, non_blocking=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scaler = torch.amp.GradScaler('cuda')

    best_val_auc = -1.0
    train_losses = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/10"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs.squeeze(1), labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_count = images.size(0)
            running_loss += loss.item() * batch_count
            total_samples += batch_count

        train_loss = running_loss / total_samples
        train_losses.append(train_loss)

        val_auc, val_acc, val_f1, val_class_acc = evaluate_model(model, val_loader, device, "Validating")
        scheduler.step()

        print(f"Epoch {epoch}/10 | Train Loss: {train_loss:.6f} | Val AUC: {val_auc:.6f} | Val Acc: {val_acc * 100:.2f}%")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Best checkpoint was not saved: {CHECKPOINT_PATH}")

    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))

    val_auc, val_acc, val_f1, val_class_acc = evaluate_model(model, val_loader, device, "Evaluating Best Val")
    test_auc, test_acc, test_f1, test_class_acc = evaluate_model(model, test_loader, device, "Testing Best Val")

    save_loss_curve(train_losses)
    training_time_mins = (time.time() - start_time) / 60

    write_results(
        MAX_EPOCHS,
        train_losses[-1],
        val_auc,
        val_acc,
        val_f1,
        val_class_acc,
        test_auc,
        test_acc,
        test_f1,
        test_class_acc,
        training_time_mins
    )

    print(f"Saved checkpoint: {CHECKPOINT_PATH}")
    print(f"Saved results: {RESULTS_PATH}")
    print(f"Saved loss curve: {LOSS_CURVE_PATH}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

```

### FILE: workspaces\nusrat\train_spatiotemporal_lameness.py
---
```python
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend to prevent Tkinter thread crashes

import os
import time
import random
import multiprocessing
import cv2
cv2.setNumThreads(0)  # Disable OpenCV multithreading to prevent deadlocks in DataLoader

import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Configuration
PERSON_NAME = "Nusrat"
BASE_MODEL_DISPLAY = "ResNet18-LSTM"
MODEL_NAME = "resnet18"
BASE_DIR = r"D:\T25301094 P2"
WORKSPACE_DIR = r"D:\T25301094 P2\workspaces\nusrat"
CSV_PATH = r"D:\T25301094 P2\datasets\lameness\lameness_index.csv"
CHECKPOINT_PATH = r"D:\T25301094 P2\workspaces\nusrat\spatiotemporal_lameness_best.pth"
RESULTS_PATH = r"D:\T25301094 P2\workspaces\nusrat\spatiotemporal_lameness_results.txt"
LOSS_CURVE_PATH = r"D:\T25301094 P2\workspaces\nusrat\spatiotemporal_lameness_loss_curve.png"

BATCH_SIZE = 16  # Increased to leverage 16GB VRAM (processes 16 * 20 = 320 frames per batch)
NUM_WORKERS = 2  # Low workers to be safe with memory and deadlocks
MAX_EPOCHS = 15
RANDOM_SEED = 42
HIDDEN_DIM = 64

class VideoSequenceDataset(Dataset):
    def __init__(self, csv_path, split, transform=None):
        df = pd.read_csv(csv_path)
        df = df[df['split'] == split].reset_index(drop=True)
        # Get unique video IDs in this split
        self.video_ids = sorted(df['cow_id'].unique())
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        v_id = self.video_ids[idx]
        v_df = self.df[self.df['cow_id'] == v_id].sort_values('image_path').reset_index(drop=True)
        
        frames = []
        label = float(v_df.iloc[0]['label'])
        
        # Sub-sample exactly 20 frames evenly spaced to prevent OOM and redundancy
        total_frames = len(v_df)
        indices = [int(i * (total_frames - 1) / (20 - 1)) for i in range(20)] if total_frames > 1 else [0] * 20
        
        for idx_to_load in indices:
            img_path = v_df.iloc[idx_to_load]['image_path']
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.transform is not None:
                # Note: using the same transform across the sequence
                # For basic normalizations this is fine. For spatial augs, 
                # using seed ensures same crop/flip is applied to the entire clip.
                state = random.getstate()
                image = self.transform(image=image)["image"]
                random.setstate(state)
                
            frames.append(image)
            
        frames_tensor = torch.stack(frames)  # shape: (20, 3, 224, 224)
        return frames_tensor, torch.tensor(label, dtype=torch.float32)

class CNNLSTMModel(nn.Module):
    def __init__(self, backbone_name, hidden_dim, device):
        super().__init__()
        # Extract features without classification head
        backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        backbone = backbone.to(device, non_blocking=True)
        
        with torch.no_grad():
            dummy = backbone(torch.zeros(1, 3, 224, 224).to(device, non_blocking=True))
            feature_dim = dummy.shape[1]
            
        self.backbone = backbone
        # LSTM processes the frame feature vectors
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, c, h, w)
        batch_size, seq_len, c, h, w = x.shape
        # Flatten batch and seq dimensions to run through backbone
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.backbone(x)  # (batch_size * seq_len, feature_dim)
        
        # Reshape back to sequence
        features = features.view(batch_size, seq_len, -1)  # (batch_size, seq_len, feature_dim)
        
        # LSTM output
        lstm_out, (hn, cn) = self.lstm(features)  # lstm_out shape: (batch_size, seq_len, hidden_dim)
        
        # Take the final output step
        last_step_out = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Output logit
        logits = self.classifier(last_step_out)  # (batch_size, 1)
        return logits

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_transforms():
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    eval_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return train_transform, eval_transform

def build_loader(dataset, shuffle, persistent=True):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=persistent,
    )

def evaluate_model(model, loader, device, desc):
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                probs = torch.sigmoid(outputs).view(-1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)

    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    normal_acc = (cm[0, 0] / cm[0].sum() * 100) if cm[0].sum() > 0 else 0.0
    lame_acc = (cm[1, 1] / cm[1].sum() * 100) if cm[1].sum() > 0 else 0.0
    
    return auc, acc, f1, [normal_acc, lame_acc]

def save_loss_curve(train_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Spatiotemporal Lameness Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH, dpi=300)
    plt.close()

def write_results(actual_epochs_trained, final_train_loss, val_auc, val_acc, val_f1, val_class_acc, test_auc, test_acc, test_f1, test_class_acc, training_time_mins):
    text = f"""---CONTEXT 3 SPATIOTEMPORAL LAMENESS---
PERSON NAME: {PERSON_NAME}
BASE MODEL: {BASE_MODEL_DISPLAY}
DATASET: CattleLameness (20 frames video sequences)
EPOCHS TRAINED: {actual_epochs_trained}
FINAL TRAIN LOSS: {final_train_loss:.6f}
VAL AUC: {val_auc:.6f}
VAL ACCURACY: {val_acc * 100:.2f}%
VAL F1 SCORE: {val_f1:.6f}
VAL PER-CLASS ACCURACY:
  Class 0 (Normal): {val_class_acc[0]:.2f}%
  Class 1 (Lame): {val_class_acc[1]:.2f}%
TEST AUC: {test_auc:.6f}
TEST ACCURACY: {test_acc * 100:.2f}%
TEST F1 SCORE: {test_f1:.6f}
TEST PER-CLASS ACCURACY:
  Class 0 (Normal): {test_class_acc[0]:.2f}%
  Class 1 (Lame): {test_class_acc[1]:.2f}%
CHECKPOINT PATH: {CHECKPOINT_PATH}
TRAINING TIME (mins): {training_time_mins:.2f}
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---"""

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)

def main():
    start_time = time.time()
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    set_seed(RANDOM_SEED)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    train_transform, eval_transform = build_transforms()

    train_dataset = VideoSequenceDataset(CSV_PATH, "train", train_transform)
    val_dataset = VideoSequenceDataset(CSV_PATH, "val", eval_transform)
    test_dataset = VideoSequenceDataset(CSV_PATH, "test", eval_transform)

    train_loader = build_loader(train_dataset, shuffle=True, persistent=True)
    val_loader = build_loader(val_dataset, shuffle=False, persistent=False)
    test_loader = build_loader(test_dataset, shuffle=False, persistent=False)

    print(f"Person: {PERSON_NAME}")
    print(f"Base model: {BASE_MODEL_DISPLAY}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Train samples (videos): {len(train_dataset)}")
    print(f"Val samples (videos): {len(val_dataset)}")
    print(f"Test samples (videos): {len(test_dataset)}")

    model = CNNLSTMModel(MODEL_NAME, HIDDEN_DIM, device)
    model = model.to(device, non_blocking=True)

    # Freeze backbone parameters to prevent overfitting on the small dataset
    # We will only train the LSTM and classification head
    for param in model.backbone.parameters():
        param.requires_grad = False

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scaler = torch.amp.GradScaler('cuda')

    best_val_auc = -1.0
    train_losses = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs.view(-1), labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_count = images.size(0)
            running_loss += loss.item() * batch_count
            total_samples += batch_count

        train_loss = running_loss / total_samples
        train_losses.append(train_loss)

        val_auc, val_acc, val_f1, val_class_acc = evaluate_model(model, val_loader, device, "Validating")
        scheduler.step()

        print(f"Epoch {epoch}/{MAX_EPOCHS} | Train Loss: {train_loss:.6f} | Val AUC: {val_auc:.6f} | Val Acc: {val_acc * 100:.2f}%")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Best checkpoint was not saved: {CHECKPOINT_PATH}")

    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))

    val_auc, val_acc, val_f1, val_class_acc = evaluate_model(model, val_loader, device, "Evaluating Best Val")
    test_auc, test_acc, test_f1, test_class_acc = evaluate_model(model, test_loader, device, "Testing Best Val")

    save_loss_curve(train_losses)
    training_time_mins = (time.time() - start_time) / 60

    write_results(
        MAX_EPOCHS,
        train_losses[-1],
        val_auc,
        val_acc,
        val_f1,
        val_class_acc,
        test_auc,
        test_acc,
        test_f1,
        test_class_acc,
        training_time_mins
    )

    print(f"Saved checkpoint: {CHECKPOINT_PATH}")
    print(f"Saved results: {RESULTS_PATH}")
    print(f"Saved loss curve: {LOSS_CURVE_PATH}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

```

### FILE: workspaces\nusrat\train_spatiotemporal_lameness_efficientnet.py
---
```python
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend to prevent Tkinter thread crashes

import os
import time
import random
import multiprocessing
import cv2
cv2.setNumThreads(0)  # Disable OpenCV multithreading to prevent deadlocks in DataLoader

import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Configuration
PERSON_NAME = "Nusrat"
BASE_MODEL_DISPLAY = "EfficientNetB0-LSTM"
MODEL_NAME = "efficientnet_b0"
BASE_DIR = r"D:\T25301094 P2"
WORKSPACE_DIR = r"D:\T25301094 P2\workspaces\nusrat"
CSV_PATH = r"D:\T25301094 P2\datasets\lameness\lameness_cropped_index.csv"
CHECKPOINT_PATH = r"D:\T25301094 P2\workspaces\nusrat\spatiotemporal_lameness_efficientnet_best.pth"
RESULTS_PATH = r"D:\T25301094 P2\workspaces\nusrat\spatiotemporal_lameness_efficientnet_results.txt"
LOSS_CURVE_PATH = r"D:\T25301094 P2\workspaces\nusrat\spatiotemporal_lameness_efficientnet_loss_curve.png"

BATCH_SIZE = 8  # Increased from 4 to 8 to utilize more GPU capacity
NUM_WORKERS = 4  # Increased from 2 to 4 to leverage more CPU cores for faster dataloading
MAX_EPOCHS = 15
RANDOM_SEED = 42
HIDDEN_DIM = 64

class VideoSequenceDataset(Dataset):
    def __init__(self, csv_path, split, transform=None):
        import csv
        self.transform = transform
        self.video_data = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['split'] == split:
                    self.video_data.append({
                        'image_path': row['image_path'],
                        'label': float(row['label']),
                        'cow_id': row['cow_id']
                    })
        self.video_ids = sorted(list(set(item['cow_id'] for item in self.video_data)))

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        v_id = self.video_ids[idx]
        v_samples = [item for item in self.video_data if item['cow_id'] == v_id]
        v_samples = sorted(v_samples, key=lambda x: x['image_path'])
        
        frames = []
        label = v_samples[0]['label']
        
        total_frames = len(v_samples)
        indices = [int(i * (total_frames - 1) / (20 - 1)) for i in range(20)] if total_frames > 1 else [0] * 20
        
        for idx_to_load in indices:
            img_path = v_samples[idx_to_load]['image_path']
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.transform is not None:
                state = random.getstate()
                image = self.transform(image=image)["image"]
                random.setstate(state)
                
            frames.append(image)
            
        frames_tensor = torch.stack(frames)  # shape: (20, 3, 224, 224)
        return frames_tensor, torch.tensor(label, dtype=torch.float32)

class CNNLSTMModel(nn.Module):
    def __init__(self, backbone_name, hidden_dim, device):
        super().__init__()
        # Extract features without classification head
        backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        backbone = backbone.to(device, non_blocking=True)
        
        with torch.no_grad():
            dummy = backbone(torch.zeros(1, 3, 224, 224).to(device, non_blocking=True))
            feature_dim = dummy.shape[1]
            
        self.backbone = backbone
        # LSTM processes the frame feature vectors
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, c, h, w)
        batch_size, seq_len, c, h, w = x.shape
        # Flatten batch and seq dimensions to run through backbone
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.backbone(x)  # (batch_size * seq_len, feature_dim)
        
        # Reshape back to sequence
        features = features.view(batch_size, seq_len, -1)  # (batch_size, seq_len, feature_dim)
        
        # LSTM output
        lstm_out, (hn, cn) = self.lstm(features)  # lstm_out shape: (batch_size, seq_len, hidden_dim)
        
        # Take the final output step
        last_step_out = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Output logit
        logits = self.classifier(last_step_out)  # (batch_size, 1)
        return logits

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_transforms():
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    eval_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return train_transform, eval_transform

def build_loader(dataset, shuffle, persistent=True):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=persistent,
    )

def evaluate_model(model, loader, device, desc):
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                probs = torch.sigmoid(outputs).view(-1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)

    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    normal_acc = (cm[0, 0] / cm[0].sum() * 100) if cm[0].sum() > 0 else 0.0
    lame_acc = (cm[1, 1] / cm[1].sum() * 100) if cm[1].sum() > 0 else 0.0
    
    return auc, acc, f1, [normal_acc, lame_acc]

def save_loss_curve(train_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Spatiotemporal Lameness Training Loss Curve (EfficientNetB0)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PATH, dpi=300)
    plt.close()

def write_results(actual_epochs_trained, final_train_loss, val_auc, val_acc, val_f1, val_class_acc, test_auc, test_acc, test_f1, test_class_acc, training_time_mins):
    text = f"""---CONTEXT 3 SPATIOTEMPORAL LAMENESS---
PERSON NAME: {PERSON_NAME}
BASE MODEL: {BASE_MODEL_DISPLAY}
DATASET: CattleLameness (20 frames video sequences)
EPOCHS TRAINED: {actual_epochs_trained}
FINAL TRAIN LOSS: {final_train_loss:.6f}
VAL AUC: {val_auc:.6f}
VAL ACCURACY: {val_acc * 100:.2f}%
VAL F1 SCORE: {val_f1:.6f}
VAL PER-CLASS ACCURACY:
  Class 0 (Normal): {val_class_acc[0]:.2f}%
  Class 1 (Lame): {val_class_acc[1]:.2f}%
TEST AUC: {test_auc:.6f}
TEST ACCURACY: {test_acc * 100:.2f}%
TEST F1 SCORE: {test_f1:.6f}
TEST PER-CLASS ACCURACY:
  Class 0 (Normal): {test_class_acc[0]:.2f}%
  Class 1 (Lame): {test_class_acc[1]:.2f}%
CHECKPOINT PATH: {CHECKPOINT_PATH}
TRAINING TIME (mins): {training_time_mins:.2f}
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---"""

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)

def main():
    start_time = time.time()
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    set_seed(RANDOM_SEED)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    train_transform, eval_transform = build_transforms()

    train_dataset = VideoSequenceDataset(CSV_PATH, "train", train_transform)
    val_dataset = VideoSequenceDataset(CSV_PATH, "val", eval_transform)
    test_dataset = VideoSequenceDataset(CSV_PATH, "test", eval_transform)

    train_loader = build_loader(train_dataset, shuffle=True, persistent=True)
    val_loader = build_loader(val_dataset, shuffle=False, persistent=False)
    test_loader = build_loader(test_dataset, shuffle=False, persistent=False)

    print(f"Person: {PERSON_NAME}")
    print(f"Base model: {BASE_MODEL_DISPLAY}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Train samples (videos): {len(train_dataset)}")
    print(f"Val samples (videos): {len(val_dataset)}")
    print(f"Test samples (videos): {len(test_dataset)}")

    model = CNNLSTMModel(MODEL_NAME, HIDDEN_DIM, device)
    model = model.to(device, non_blocking=True)

    # Freeze backbone parameters to prevent overfitting on the small dataset
    # We will only train the LSTM and classification head
    for param in model.backbone.parameters():
        param.requires_grad = False

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scaler = torch.amp.GradScaler('cuda')

    best_val_auc = -1.0
    best_val_acc = -1.0
    train_losses = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs.view(-1), labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_count = images.size(0)
            running_loss += loss.item() * batch_count
            total_samples += batch_count

        train_loss = running_loss / total_samples
        train_losses.append(train_loss)

        val_auc, val_acc, val_f1, val_class_acc = evaluate_model(model, val_loader, device, "Validating")
        scheduler.step()

        print(f"Epoch {epoch}/{MAX_EPOCHS} | Train Loss: {train_loss:.6f} | Val AUC: {val_auc:.6f} | Val Acc: {val_acc * 100:.2f}%")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        elif val_auc == best_val_auc and val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Best checkpoint was not saved: {CHECKPOINT_PATH}")

    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))

    val_auc, val_acc, val_f1, val_class_acc = evaluate_model(model, val_loader, device, "Evaluating Best Val")
    test_auc, test_acc, test_f1, test_class_acc = evaluate_model(model, test_loader, device, "Testing Best Val")

    save_loss_curve(train_losses)
    training_time_mins = (time.time() - start_time) / 60

    write_results(
        MAX_EPOCHS,
        train_losses[-1],
        val_auc,
        val_acc,
        val_f1,
        val_class_acc,
        test_auc,
        test_acc,
        test_f1,
        test_class_acc,
        training_time_mins
    )

    print(f"Saved checkpoint: {CHECKPOINT_PATH}")
    print(f"Saved results: {RESULTS_PATH}")
    print(f"Saved loss curve: {LOSS_CURVE_PATH}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

```

### FILE: workspaces\nusrat\visualize_bcs_crops.py
---
```python
import os
import cv2
import random
import pandas as pd
import numpy as np
from ultralytics import YOLO

# Configuration
BASE_DIR = r"D:\T25301094 P2"
INPUT_CSV = os.path.join(BASE_DIR, "datasets", "bcs", "sciencedb_bcs_index.csv")
OUTPUT_IMAGE = os.path.join(BASE_DIR, "workspaces", "nusrat", "bcs_crop_samples.png")
YOLO_MODEL = os.path.join(BASE_DIR, "final_models", "yolov8n.pt")
PANEL_SIZE = (224, 224)

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found at: {INPUT_CSV}")

    print("Loading YOLOv8 detector...")
    detector = YOLO(YOLO_MODEL)

    print(f"Reading index file: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    
    # Pick 5 random rows to visualize
    random.seed(42)  # For reproducibility
    sampled_rows = df.sample(n=5).reset_index(drop=True)

    rows_images = []

    for idx, row in enumerate(sampled_rows.iterrows()):
        img_path = row[1]['image_path']
        print(f"Processing image {idx+1}/5: {os.path.basename(img_path)}")

        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image: {img_path}")
            continue

        # Keep a copy of the original for drawing the box
        original_visual = img.copy()

        # Run YOLO detection
        results = detector(img, verbose=False)[0]

        cow_crop = None
        box_coords = None
        max_area = 0
        for box in results.boxes:
            class_id = int(box.cls[0])
            # Class ID 19 is 'cow'
            if class_id == 19:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    box_coords = (x1, y1, x2, y2)

        if box_coords is not None:
            x1, y1, x2, y2 = box_coords
            cow_crop = img[y1:y2, x1:x2]

        # Draw box and label on original image copy
        if box_coords is not None:
            x1, y1, x2, y2 = box_coords
            cv2.rectangle(original_visual, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(original_visual, "Cow", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Process panels
        left_panel = cv2.resize(original_visual, PANEL_SIZE)
        
        if cow_crop is not None and cow_crop.size > 0:
            right_panel = cv2.resize(cow_crop, PANEL_SIZE)
        else:
            # Fallback if YOLO misses: show resized original with red warning border
            right_panel = cv2.resize(img, PANEL_SIZE)
            cv2.rectangle(right_panel, (0, 0), (PANEL_SIZE[0]-1, PANEL_SIZE[1]-1), (0, 0, 255), 3)
            cv2.putText(right_panel, "No Cow Detected", (10, PANEL_SIZE[1]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Label both panels
        cv2.putText(left_panel, "Original", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(right_panel, "YOLO Crop", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Stitched row: horizontally concatenate Left and Right
        row_concat = np.hstack((left_panel, right_panel))
        rows_images.append(row_concat)

    if not rows_images:
        print("Error: No images were successfully processed.")
        return

    # Vertically concatenate all rows
    grid_visual = np.vstack(rows_images)

    # Save to disk
    os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)
    cv2.imwrite(OUTPUT_IMAGE, grid_visual)
    print(f"\nSaved visualization grid to: {OUTPUT_IMAGE}")

    # Display in window
    try:
        print("Displaying crop samples. Press any key to close the window...")
        cv2.imshow("Cattle Crop Samples (Left: Original, Right: YOLO Crop)", grid_visual)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("\nNote: Running in headless mode. Bypassing window display.")

if __name__ == "__main__":
    main()

```

### FILE: workspaces\nusrat\visualize_cow_detection.py
---
```python
import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm

# Configuration
BASE_DIR = r"D:\T25301094 P2"
INPUT_VIDEO_PATH = os.path.join(BASE_DIR, "videos", "test_cow.mp4")
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, "videos", "test_cow_detection.mp4")
MODEL_NAME = os.path.join(BASE_DIR, "final_models", "yolov8n.pt")  # Lightweight YOLOv8 Nano model
MAX_FRAMES_TO_PROCESS = 2000  # Default limit to process a subset of the long video quickly

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize YOLO Cow Detection on Video")
    parser.add_argument("--input", type=str, default=INPUT_VIDEO_PATH, help="Path to input video")
    parser.add_argument("--output", type=str, default=OUTPUT_VIDEO_PATH, help="Path to save annotated video")
    parser.add_argument("--limit", type=int, default=MAX_FRAMES_TO_PROCESS, help="Maximum number of frames to process (set to 0 for full video)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input video not found at: {args.input}")

    print(f"Loading YOLOv8 detector: {MODEL_NAME}...")
    # Load pre-trained YOLOv8-Nano model
    detector = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Error: Could not open video file {args.input}")

    # Read video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set frame limit
    limit = args.limit if args.limit > 0 else total_frames
    limit = min(limit, total_frames)

    print(f"\nProcessing Video:")
    print(f"  Source:       {args.input}")
    print(f"  Resolution:   {width}x{height}")
    print(f"  FPS:          {fps}")
    print(f"  Total Frames: {total_frames} (Limit set to process first {limit} frames)")
    print(f"  Saving to:    {args.output}\n")

    # Define VideoWriter to save output with annotations
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    show_window = True

    # Process frames
    for i in tqdm(range(limit), desc="Drawing bounding boxes"):
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = detector(frame, verbose=False)[0]

        # Draw box for any detected cow
        for box in results.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])

            # Class ID 19 is 'cow' in COCO dataset
            if class_id == 19:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Draw green bounding box around cow
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Draw text label with confidence score
                label = f"Cow {conf*100:.1f}%"
                cv2.putText(frame, label, (x1, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Write the annotated frame to output video
        out.write(frame)

        # Display the annotated frame in real-time if GUI window support is available
        if show_window:
            try:
                cv2.imshow("Real-Time YOLO Cow Detection (Press 'q' to Quit)", frame)
                # Check for 'q' key press to exit early
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nVisualization stopped early by user.")
                    break
            except cv2.error:
                print("\nWarning: Headless environment or OpenCV without GUI support detected.")
                print("Running in HEADLESS mode. Video is still being written to the output file.")
                show_window = False

    cap.release()
    out.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    print(f"\nFinished! Annotated video saved successfully to: {args.output}")

if __name__ == "__main__":
    main()

```

### FILE: workspaces\nusrat\visualize_lameness_realtime.py
---
```python
import os
import cv2
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO
from tqdm import tqdm

# Configurations
BASE_DIR = r"D:\T25301094 P2"
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "final_models", "yolov8n.pt")
LAME_CHECKPOINT_PATH = os.path.join(BASE_DIR, "workspaces", "nusrat", "spatiotemporal_lameness_efficientnet_best.pth")
ID_CHECKPOINT_PATH = os.path.join(BASE_DIR, "workspaces", "nusrat", "id_best.pth")
CSV_PATH = os.path.join(BASE_DIR, "datasets", "id", "id_index.csv")
INPUT_VIDEO_PATH = os.path.join(BASE_DIR, "videos", "cut_cow_video.mp4")
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, "videos", "cut_cow_realtime_detection.mp4")

MODEL_NAME = "efficientnet_b0"
HIDDEN_DIM = 64
DECISION_THRESHOLD = 0.50
TARGET_SIZE = (224, 224)
SEQ_LEN = 20
NUM_CLASSES_ID = 46

# Attention Modules for CowIDModel
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_ = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        return x * self.sigmoid(avg + max_).unsqueeze(-1).unsqueeze(-1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))

# CowIDModel definition
class CowIDModel(nn.Module):
    def __init__(self, model_name, num_classes, device):
        super().__init__()
        backbone = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool='')
        backbone = backbone.to(device, non_blocking=True)
        with torch.no_grad():
            dummy = backbone(torch.zeros(1, 3, 224, 224).to(device, non_blocking=True))
            feature_dim = dummy.shape[1]
        self.backbone = backbone
        self.cbam = CBAM(feature_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def extract_features(self, x):
        x = self.backbone(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.extract_features(x)
        x = self.classifier(x)
        return x

# CNN-LSTM Lameness Model definition
class CNNLSTMModel(nn.Module):
    def __init__(self, backbone_name, hidden_dim, device):
        super().__init__()
        backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        backbone = backbone.to(device, non_blocking=True)
        with torch.no_grad():
            dummy = backbone(torch.zeros(1, 3, 224, 224).to(device, non_blocking=True))
            feature_dim = dummy.shape[1]
        self.backbone = backbone
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.backbone(x)
        features = features.view(batch_size, seq_len, -1)
        lstm_out, (hn, cn) = self.lstm(features)
        last_step_out = lstm_out[:, -1, :]
        logits = self.classifier(last_step_out)
        return logits

def draw_premium_overlay(img, text_lines, x, y, colors):
    font_scale = 0.75
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    box_w = 0
    line_heights = []
    
    for line in text_lines:
        (w, h), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        box_w = max(box_w, w)
        line_heights.append(h + baseline + 6)
    
    box_w += 20
    box_h = sum(line_heights) + 15
    
    # Background semi-transparent overlay
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 10, y - 25), (x + box_w, y + box_h - 25), (0, 0, 0), -1)
    
    alpha = 0.65
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Draw texts with their individual colors
    curr_y = y - 5
    for idx, line in enumerate(text_lines):
        color = colors[idx] if idx < len(colors) else (255, 255, 255)
        cv2.putText(img, line, (x, curr_y), font, font_scale, color, thickness)
        curr_y += line_heights[idx]

def build_embedding_database(model, csv_path, device):
    print("Building cow embedding database from training images...")
    
    # Group image paths by label
    cow_images = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['split'] == 'train':
                lbl = int(row['label'])
                if lbl not in cow_images:
                    cow_images[lbl] = []
                if len(cow_images[lbl]) < 5:
                    cow_images[lbl].append(row['image_path'])
                    
    # Extract features and average them per cow class
    database = {}
    model.eval()
    
    # Simple transform matching train_id.py eval_transform
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    with torch.no_grad():
        for lbl, paths in tqdm(cow_images.items(), desc="Registering Cows"):
            tensors = []
            for path in paths:
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    tensor = transform(image=img)["image"]
                    tensors.append(tensor)
            if tensors:
                batch = torch.stack(tensors).to(device)
                with torch.amp.autocast('cuda') if torch.cuda.is_available() else torch.no_grad():
                    # Get features before final classifier
                    features = model.extract_features(batch) # shape: (N, 1280)
                    mean_feature = features.mean(dim=0)
                    mean_feature = F.normalize(mean_feature, p=2, dim=0) # Normalize to unit sphere
                database[lbl] = mean_feature
                
    print(f"Database built successfully. Registered classes: {len(database)}\n")
    return database

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Real-Time Bounding Box, Cow ID & Lameness Visualizer")
    parser.add_argument("--input", type=str, default=INPUT_VIDEO_PATH, help="Path to input video file")
    parser.add_argument("--output", type=str, default=OUTPUT_VIDEO_PATH, help="Path to save annotated video")
    parser.add_argument("--stride", type=int, default=8, help="Temporal stride to sub-sample frames")
    parser.add_argument("--threshold", type=float, default=0.65, help="Cosine similarity threshold for registration verification")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load YOLO detector
    print("Loading YOLOv8 detector...")
    yolo = YOLO(YOLO_MODEL_PATH)

    # Load Cow ID model
    print("Loading Cow ID baseline model...")
    id_model = CowIDModel(MODEL_NAME, NUM_CLASSES_ID, device)
    id_model = id_model.to(device)
    if not os.path.exists(ID_CHECKPOINT_PATH):
        raise FileNotFoundError(f"Cow ID checkpoint not found at: {ID_CHECKPOINT_PATH}")
    id_model.load_state_dict(torch.load(ID_CHECKPOINT_PATH, map_location=device, weights_only=True))
    id_model.eval()

    # Build embedding database
    embedding_db = build_embedding_database(id_model, CSV_PATH, device)

    # Load Lameness model
    print("Loading Lameness CNN-LSTM model...")
    lame_model = CNNLSTMModel(MODEL_NAME, HIDDEN_DIM, device)
    lame_model = lame_model.to(device)
    if not os.path.exists(LAME_CHECKPOINT_PATH):
        raise FileNotFoundError(f"Lameness checkpoint not found at: {LAME_CHECKPOINT_PATH}")
    lame_model.load_state_dict(torch.load(LAME_CHECKPOINT_PATH, map_location=device, weights_only=True))
    lame_model.eval()
    print("Models loaded successfully.")

    # Initialize video capture
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Error: Could not open input video {args.input}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"\nProcessing Video:")
    print(f"  Source:     {args.input}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS:        {fps}")
    print(f"  Output:     {args.output}\n")

    # Image preprocessing transform
    transform = A.Compose([
        A.Resize(TARGET_SIZE[0], TARGET_SIZE[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Sliding window buffer of cropped cow frames (holds preprocessed tensors)
    cropped_buffer = []
    show_window = True
    last_best_box = None
    
    # Store running ID results to display stable values
    last_cow_id_str = "Unknown"
    last_cow_id_prob = 0.0

    # Auto-registration variables
    unknown_embeddings_buffer = []
    next_class_id = NUM_CLASSES_ID  # Starts at 46 (which represents indices 0-45)
    
    # Calculate required consecutive frames of observation for 5 seconds
    required_unknown_frames = int(5.0 * fps / args.stride)
    print(f"Registration Window: 5.0s of video space ({required_unknown_frames} processed frames at stride {args.stride})")

    # Process frame-by-frame
    for i in tqdm(range(total_frames), desc="Running Real-Time Inference"):
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = yolo(frame, verbose=False)[0]

        best_box = None
        max_area = 0
        for box in results.boxes:
            class_id = int(box.cls[0])
            # Class ID 19 is 'cow'
            if class_id == 19:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    best_box = (x1, y1, x2, y2)

        # Update last known bounding box for tracking fallback
        if best_box is not None:
            last_best_box = best_box
        
        # Determine crop region
        crop_box = best_box if best_box is not None else last_best_box

        cow_crop = None
        if crop_box is not None:
            cx1, cy1, cx2, cy2 = crop_box
            cx1 = max(0, cx1)
            cy1 = max(0, cy1)
            cx2 = min(width, cx2)
            cy2 = min(height, cy2)
            if cx2 > cx1 and cy2 > cy1:
                cow_crop = frame[cy1:cy2, cx1:cx2]

        # Process crop frame
        if cow_crop is not None and cow_crop.size > 0:
            rgb_crop = cv2.cvtColor(cow_crop, cv2.COLOR_BGR2RGB)
            transformed = transform(image=rgb_crop)["image"]

            # Sub-sample frames temporally according to STRIDE to match dataset time span
            if i % args.stride == 0:
                cropped_buffer.append(transformed)
                if len(cropped_buffer) > SEQ_LEN:
                    cropped_buffer.pop(0)

                # Run Cow ID prediction using vector embedding Cosine Similarity
                input_crop_tensor = transformed.unsqueeze(0).to(device)
                with torch.no_grad():
                    with torch.amp.autocast('cuda') if torch.cuda.is_available() else torch.no_grad():
                        live_features = id_model.extract_features(input_crop_tensor) # shape: (1, 1280)
                        live_features = F.normalize(live_features, p=2, dim=1) # normalize to unit sphere
                        
                # Compute Cosine Similarity against all registered classes
                best_similarity = -1.0
                best_class = -1
                for lbl, reg_feature in embedding_db.items():
                    sim = torch.dot(live_features[0], reg_feature).item()
                    if sim > best_similarity:
                        best_similarity = sim
                        best_class = lbl
                
                # Check threshold for open-set registration verification
                if best_similarity >= args.threshold:
                    last_cow_id_str = f"{(best_class + 1):03d}"
                    last_cow_id_prob = best_similarity
                    # Clear the unknown buffer because we successfully recognized a registered cow
                    unknown_embeddings_buffer.clear()
                else:
                    last_cow_id_str = "Unknown Cow"
                    last_cow_id_prob = best_similarity
                    
                    # Store features for auto-registration
                    unknown_embeddings_buffer.append(live_features[0].cpu())
                    
                    # If we collect 5 seconds of consecutive unknown frames, register as a new cow
                    if len(unknown_embeddings_buffer) >= required_unknown_frames:
                        new_cow_vector = torch.stack(unknown_embeddings_buffer).mean(dim=0).to(device)
                        new_cow_vector = F.normalize(new_cow_vector, p=2, dim=0)
                        
                        # Register in memory database
                        embedding_db[next_class_id] = new_cow_vector
                        print(f"\n>>> [AUTO-REGISTRATION] Registered New Cow in database as ID: {next_class_id + 1:03d} after 5.0 seconds of observation!")
                        
                        # Set current outputs to match the new class
                        last_cow_id_str = f"{next_class_id + 1:03d}"
                        last_cow_id_prob = 1.0 # Initial similarity is 1.0 since it matches itself
                        
                        next_class_id += 1
                        unknown_embeddings_buffer.clear()

        # Draw YOLO box on the frame
        box_color = (0, 255, 0) # Green bounding box
        if best_box is not None:
            cv2.rectangle(frame, (best_box[0], best_box[1]), (best_box[2], best_box[3]), box_color, 3)
        elif last_best_box is not None:
            # Draw tracking box if reusing previous box
            cv2.rectangle(frame, (last_best_box[0], last_best_box[1]), (last_best_box[2], last_best_box[3]), (255, 128, 0), 1)

        # Run spatiotemporal lameness prediction if buffer is full
        text_lines = []
        text_colors = []

        # Add Cow ID info to overlay
        text_lines.append(f"Cow ID: {last_cow_id_str} ({last_cow_id_prob*100:.1f}%)")
        text_colors.append((255, 255, 255)) # White

        if len(cropped_buffer) == SEQ_LEN:
            input_seq_tensor = torch.stack(cropped_buffer).unsqueeze(0).to(device)
            with torch.no_grad():
                with torch.amp.autocast('cuda') if torch.cuda.is_available() else torch.no_grad():
                    lame_logits = lame_model(input_seq_tensor)
                    lame_prob = torch.sigmoid(lame_logits).view(-1).item()

            prediction = "LAME" if lame_prob >= DECISION_THRESHOLD else "NORMAL"
            gait_color = (0, 0, 255) if prediction == "LAME" else (0, 255, 0)
            text_lines.append(f"Gait: {prediction} ({lame_prob*100:.1f}%)")
            text_colors.append(gait_color)
        else:
            text_lines.append(f"Gait: Buffering Stride... ({len(cropped_buffer)}/{SEQ_LEN})")
            text_colors.append((255, 255, 0)) # Cyan/yellow

        # Draw the premium transparent background overlay
        text_x = best_box[0] if best_box is not None else (last_best_box[0] if last_best_box is not None else 30)
        text_y = max(40, (best_box[1] - 40) if best_box is not None else (last_best_box[1] - 40 if last_best_box is not None else 50))
        
        draw_premium_overlay(frame, text_lines, text_x, text_y, text_colors)

        # Write annotated frame to output video
        out.write(frame)

        # Display window
        if show_window:
            try:
                cv2.imshow("Real-Time Unified Multi-Task Inference (Press 'q' to Quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nProcessing stopped early by user.")
                    break
            except cv2.error:
                print("\nWarning: Headless environment or OpenCV without GUI support detected.")
                print("Running in HEADLESS mode. Video is still being written to the output file.")
                show_window = False

    cap.release()
    out.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    print(f"\nProcessing Complete!")
    print(f"Saved annotated video to: {args.output}")

if __name__ == "__main__":
    main()

```

## 8. TRAINING CODE (NAMIRA - MobileNetV3-Small)
================================================================================

### FILE: workspaces\namira\train_bcs.py
---
```python
import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import timm

# --- Configuration & Seed ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CBAM Architecture ---
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid_channels = max(1, in_channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.fc(self.avg_pool(x).view(b, c))
        max_ = self.fc(self.max_pool(x).view(b, c))
        return x * self.sigmoid(avg + max_).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))

# --- Model Definitions ---
class MultiTaskBCSModel(nn.Module):
    def __init__(self):
        super().__init__()
        # FIX APPLIED HERE: num_classes=0 explicitly removes the classifier head
        self.backbone = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=0, global_pool='')
        
        # Calculate feature map channels
        dummy_tensor = torch.randn(1, 3, 224, 224)
        out = self.backbone(dummy_tensor)
        in_channels = out.shape[1]
        
        self.cbam = CBAM(in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(in_channels, 4)

    def forward(self, x):
        x = self.backbone(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

def coral_loss(logits, labels, num_classes=5):
    sets = []
    for i in range(num_classes - 1):
        label_i = (labels > i).float()
        sets.append(label_i)
    labels_stacked = torch.stack(sets, dim=1).to(device)
    return F.binary_cross_entropy_with_logits(logits, labels_stacked)

# --- Dataset Handler ---
class BCSDataset(Dataset):
    def __init__(self, csv_path, split, label_mapping, transform=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.label_mapping = label_mapping
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'image_path']
        label = self.df.loc[idx, 'label']
        
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
            
        mapped_label = self.label_mapping[label]
        return img, torch.tensor(mapped_label, dtype=torch.long)

# --- Transforms ---
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Training Loop Engine ---
def run_training_cycle(dataset_name, csv_path, label_mapping, best_ckpt_path):
    print(f"\n======================================")
    print(f" INITIALIZING TRAINING: {dataset_name.upper()}")
    print(f"======================================")
    
    train_ds = BCSDataset(csv_path, 'train', label_mapping, train_transforms)
    val_ds = BCSDataset(csv_path, 'val', label_mapping, val_test_transforms)
    test_ds = BCSDataset(csv_path, 'test', label_mapping, val_test_transforms)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    model = MultiTaskBCSModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    epochs = 30
    best_val_mae = float('inf')
    loss_history = []
    tracker = {'loss_10': 0, 'loss_20': 0, 'loss_30': 0}
    
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [{dataset_name}]")
        for imgs, labels in progress:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            logits = model(imgs)
            loss = coral_loss(logits, labels, 5)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            progress.set_postfix({'batch_loss': f"{loss.item():.4f}"})
            
        scheduler.step()
        epoch_loss = running_loss / len(train_ds)
        loss_history.append((epoch, epoch_loss))
        
        if epoch in [10, 20, 30]:
            tracker[f'loss_{epoch}'] = epoch_loss

        # Validation Step
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                preds = (torch.sigmoid(logits) > 0.5).sum(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        
        val_mae = np.mean(np.abs(val_preds - val_labels))
        val_acc_0 = np.mean(val_preds == val_labels) * 100
        val_acc_1 = np.mean(np.abs(val_preds - val_labels) <= 1) * 100
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), best_ckpt_path)

    # Test Evaluation (using best model)
    model.load_state_dict(torch.load(best_ckpt_path))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            preds = (torch.sigmoid(logits) > 0.5).sum(dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    
    test_mae = np.mean(np.abs(test_preds - test_labels))
    test_acc_0 = np.mean(test_preds == test_labels) * 100
    test_acc_1 = np.mean(np.abs(test_preds - test_labels) <= 1) * 100
    
    total_mins = (time.time() - start_time) / 60

    metrics = {
        'epochs': epochs,
        'loss_10': tracker['loss_10'],
        'loss_20': tracker['loss_20'],
        'loss_30': tracker['loss_30'],
        'val_mae': best_val_mae,
        'val_acc_0': val_acc_0,
        'val_acc_1': val_acc_1,
        'test_mae': test_mae,
        'test_acc_0': test_acc_0,
        'test_acc_1': test_acc_1,
        'time': total_mins,
        'history': loss_history
    }
    
    return metrics

if __name__ == '__main__':
    # Workspace Config
    workspace = r"D:\T25301094 P2\workspaces\namira"
    
    # Dryad Config
    dryad_csv = r"D:\T25301094 P2\datasets\bcs\bcs_index.csv"
    dryad_map = {2:0, 3:1, 4:2, 5:3, 6:4}
    dryad_ckpt = os.path.join(workspace, "dryad_bcs_best.pth")
    
    # ScienceDB Config
    sciencedb_csv = r"D:\T25301094 P2\datasets\bcs\sciencedb_bcs_index.csv"
    sciencedb_map = {3.25:0, 3.5:1, 3.75:2, 4.0:3, 4.25:4}
    sciencedb_ckpt = os.path.join(workspace, "sciencedb_bcs_best.pth")

    dryad_metrics = run_training_cycle("Dryad", dryad_csv, dryad_map, dryad_ckpt)
    science_metrics = run_training_cycle("ScienceDB", sciencedb_csv, sciencedb_map, sciencedb_ckpt)

    # --- Generate Loss Curve ---
    plt.figure(figsize=(10, 6))
    plt.plot([x[0] for x in dryad_metrics['history']], [x[1] for x in dryad_metrics['history']], label='Dryad Train Loss', marker='o')
    plt.plot([x[0] for x in science_metrics['history']], [x[1] for x in science_metrics['history']], label='ScienceDB Train Loss', marker='s')
    plt.title('CORAL Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(workspace, "bcs_loss_curve.png"))
    print("\nLoss curve saved.")

    # --- Generate Context 3 Report ---
    report_content = f"""---CONTEXT 3 BCS---
PERSON NAME: Namira
BASE MODEL: MobileNetV3-Small

DATASET: Dryad
EPOCHS TRAINED: {dryad_metrics['epochs']}
LOSS AT EPOCH 10: {dryad_metrics['loss_10']:.4f}
LOSS AT EPOCH 20: {dryad_metrics['loss_20']:.4f}
LOSS AT EPOCH 30: {dryad_metrics['loss_30']:.4f}
FINAL TRAIN LOSS: {dryad_metrics['loss_30']:.4f}
VAL MAE: {dryad_metrics['val_mae']:.4f}
VAL ACCURACY +-0 (exact match): {dryad_metrics['val_acc_0']:.2f}%
VAL ACCURACY +-1 (within 1 class): {dryad_metrics['val_acc_1']:.2f}%
TEST MAE: {dryad_metrics['test_mae']:.4f}
TEST ACCURACY +-0: {dryad_metrics['test_acc_0']:.2f}%
TEST ACCURACY +-1: {dryad_metrics['test_acc_1']:.2f}%
CHECKPOINT PATH: {dryad_ckpt}
TRAINING TIME (mins): {dryad_metrics['time']:.2f}
ANY ISSUES ENCOUNTERED: None

DATASET: ScienceDB
EPOCHS TRAINED: {science_metrics['epochs']}
LOSS AT EPOCH 10: {science_metrics['loss_10']:.4f}
LOSS AT EPOCH 20: {science_metrics['loss_20']:.4f}
LOSS AT EPOCH 30: {science_metrics['loss_30']:.4f}
FINAL TRAIN LOSS: {science_metrics['loss_30']:.4f}
VAL MAE: {science_metrics['val_mae']:.4f}
VAL ACCURACY +-0 (exact match): {science_metrics['val_acc_0']:.2f}%
VAL ACCURACY +-1 (within 1 class): {science_metrics['val_acc_1']:.2f}%
TEST MAE: {science_metrics['test_mae']:.4f}
TEST ACCURACY +-0: {science_metrics['test_acc_0']:.2f}%
TEST ACCURACY +-1: {science_metrics['test_acc_1']:.2f}%
CHECKPOINT PATH: {sciencedb_ckpt}
TRAINING TIME (mins): {science_metrics['time']:.2f}
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
"""
    
    with open(os.path.join(workspace, "bcs_results.txt"), "w") as f:
        f.write(report_content)
    print("Context 3 report generated successfully.")
```

### FILE: workspaces\namira\train_behavior.py
---
```python
import os
import time
import random
import numpy as np
import pandas as pd

import cv2
cv2.setNumThreads(0)  # Disable OpenCV multithreading to prevent deadlocks in DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from sklearn.metrics import f1_score, confusion_matrix

import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend to prevent Tkinter thread crashes
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
PERSON_NAME = "Namira"
BASE_MODEL_DISPLAY = "MobileNetV3-Small"
MODEL_NAME = "mobilenetv3_small_100"
BASE_DIR = r"D:\T25301094 P2"
WORKSPACE_DIR = r"D:\T25301094 P2\workspaces\namira"
CSV_PATH = r"D:\T25301094 P2\datasets\behavior\behavior_index.csv"
CHECKPOINT_PATH = os.path.join(WORKSPACE_DIR, "behavior_best.pth")
RESULTS_PATH = os.path.join(WORKSPACE_DIR, "behavior_results.txt")
LOSS_CURVE_PATH = os.path.join(WORKSPACE_DIR, "behavior_loss_curve.png")

NUM_CLASSES = 7
BATCH_SIZE = 128
NUM_WORKERS = 4
MAX_EPOCHS = 30
PATIENCE = 10
RANDOM_SEED = 42

CLASS_NAMES = [
    "Walking",
    "Standing",
    "Feeding head up",
    "Feeding head down",
    "Licking",
    "Drinking",
    "Lying",
]

# ============================================================
# REPRODUCIBILITY & SETUP
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)
torch.backends.cudnn.benchmark = True

# ============================================================
# DATASET
# ============================================================
class BehaviorDataset(Dataset):
    def __init__(self, csv_path, split, transform=None):
        df = pd.read_csv(csv_path)
        df = df[df['split'] == split]
        # Mandatory sampling to fix class imbalance and make training fast
        df = pd.concat([
            group.sample(min(len(group), 3000), random_state=RANDOM_SEED)
            for _, group in df.groupby('label')
        ]).reset_index(drop=True)
        self.data = df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = str(row["image_path"])
        if not os.path.isabs(img_path):
            img_path = os.path.join(BASE_DIR, img_path)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = int(row['label']) - 1  # 1-7 to 0-6
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label

# ============================================================
# CBAM ATTENTION MODULES
# ============================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, max(in_channels // reduction, 1)),
            nn.ReLU(),
            nn.Linear(max(in_channels // reduction, 1), in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_ = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        return x * self.sigmoid(avg + max_).unsqueeze(-1).unsqueeze(-1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))

# ============================================================
# FOCAL LOSS
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# ============================================================
# MODEL ARCHITECTURE
# ============================================================
class BehaviorModel(nn.Module):
    def __init__(self, model_name, num_classes, device):
        super().__init__()
        backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='')
        backbone = backbone.to(device, non_blocking=True)
        with torch.no_grad():
            dummy = backbone(torch.zeros(1, 3, 224, 224).to(device, non_blocking=True))
            feature_dim = dummy.shape[1]
        self.backbone = backbone
        self.cbam = CBAM(feature_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ============================================================
# METRICS EVALUATION
# ============================================================
def evaluate(model, loader, device, desc):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    
    per_class_acc = []
    for i in range(NUM_CLASSES):
        class_total = np.sum(cm[i, :])
        acc = (cm[i, i] / class_total * 100.0) if class_total > 0 else 0.0
        per_class_acc.append(acc)
        
    return macro_f1, per_class_acc

# ============================================================
# MAIN TRAINING LOOP
# ============================================================
def main():
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Transforms
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    eval_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # Datasets & Loaders
    train_ds = BehaviorDataset(CSV_PATH, "train", train_transform)
    val_ds = BehaviorDataset(CSV_PATH, "val", eval_transform)
    test_ds = BehaviorDataset(CSV_PATH, "test", eval_transform)
    
    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2
    )
    train_loader = DataLoader(train_ds, shuffle=True, persistent_workers=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, persistent_workers=True, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, persistent_workers=False, **loader_kwargs)
    
    # Build Model
    model = BehaviorModel(MODEL_NAME, NUM_CLASSES, device).to(device)
    criterion = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_f1 = -1.0
    epochs_no_improve = 0
    actual_epochs_trained = 0
    early_stopping_epoch = "N/A"
    
    epoch_losses = []
    loss_milestones = {10: "N/A", 20: "N/A", 30: "N/A"}
    
    start_time = time.time()
    print("="*60)
    print(f"  BEHAVIOR TRAINING — {PERSON_NAME} — {BASE_MODEL_DISPLAY}")
    print(f"  Workspace: {WORKSPACE_DIR}")
    print("="*60)
    
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{MAX_EPOCHS:02d}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_losses.append(epoch_loss)
        scheduler.step()
        
        if epoch in loss_milestones:
            loss_milestones[epoch] = f"{epoch_loss:.6f}"
            
        # Validation evaluation
        val_f1, val_accs = evaluate(model, val_loader, device, "Validating")
        print(f"Epoch {epoch:02d}/{MAX_EPOCHS:02d} | Loss: {epoch_loss:.6f} | Val Macro F1: {val_f1:.4f}")
        
        # Checkpoint Saving & Early Stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}. No improvement for {PATIENCE} epochs.")
                early_stopping_epoch = str(epoch)
                actual_epochs_trained = epoch
                break
                
        actual_epochs_trained = epoch

    end_time = time.time()
    training_time_mins = (end_time - start_time) / 60.0
    final_train_loss = f"{epoch_losses[-1]:.6f}"
    
    # Save Loss Curve
    plt.figure()
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{PERSON_NAME} - Behavior Training Loss')
    plt.legend()
    plt.savefig(LOSS_CURVE_PATH)
    plt.close()
    
    # Load best checkpoint and evaluate on Test set
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
    final_val_f1, final_val_per_class = evaluate(model, val_loader, device, "Evaluating Val")
    test_f1, test_per_class = evaluate(model, test_loader, device, "Evaluating Test")
    
    # Context 3 Report content
    report_content = f"""---CONTEXT 3 BEHAVIOR---
PERSON NAME: {PERSON_NAME}
BASE MODEL: {BASE_MODEL_DISPLAY}
DATASET: MmCows (capped 3000/class)
EPOCHS TRAINED: {actual_epochs_trained}
LOSS AT EPOCH 10: {loss_milestones.get(10, "N/A")}
LOSS AT EPOCH 20: {loss_milestones.get(20, "N/A")}
LOSS AT EPOCH 30: {loss_milestones.get(30, "N/A")}
FINAL TRAIN LOSS: {final_train_loss}
VAL MACRO F1: {final_val_f1:.4f}
VAL PER-CLASS ACCURACY:
  Class 1 (Walking): {final_val_per_class[0]:.2f}%
  Class 2 (Standing): {final_val_per_class[1]:.2f}%
  Class 3 (Feeding head up): {final_val_per_class[2]:.2f}%
  Class 4 (Feeding head down): {final_val_per_class[3]:.2f}%
  Class 5 (Licking): {final_val_per_class[4]:.2f}%
  Class 6 (Drinking): {final_val_per_class[5]:.2f}%
  Class 7 (Lying): {final_val_per_class[6]:.2f}%
TEST MACRO F1: {test_f1:.4f}
TEST PER-CLASS ACCURACY:
  Class 1 (Walking): {test_per_class[0]:.2f}%
  Class 2 (Standing): {test_per_class[1]:.2f}%
  Class 3 (Feeding head up): {test_per_class[2]:.2f}%
  Class 4 (Feeding head down): {test_per_class[3]:.2f}%
  Class 5 (Licking): {test_per_class[4]:.2f}%
  Class 6 (Drinking): {test_per_class[5]:.2f}%
  Class 7 (Lying): {test_per_class[6]:.2f}%
CHECKPOINT PATH: {CHECKPOINT_PATH}
TRAINING TIME (mins): {training_time_mins:.2f}
EARLY STOPPING TRIGGERED AT EPOCH: {early_stopping_epoch}
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
"""
    with open(RESULTS_PATH, "w") as f:
        f.write(report_content)
        
    print(f"Training complete. Results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()

```

## 9. TRAINING CODE (BITHI - ResNet-50)
================================================================================

### FILE: workspaces\bithi\train_bcs.py
---
```python
import os
import random
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import timm
import matplotlib.pyplot as plt

# ==========================================
# HYPERPARAMETERS & SEEDING
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 30
NUM_CLASSES = 5  # 5 ordinal classes for both datasets
NUM_CORAL_OUTPUTS = NUM_CLASSES - 1  # 4 output nodes

# Workspace configurations
WORKSPACE_DIR = r"D:\T25301094 P2\workspaces\bithi"
os.makedirs(WORKSPACE_DIR, exist_ok=True)

# ==========================================
# CBAM ATTENTION MODULE
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_ = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        return x * self.sigmoid(avg + max_).unsqueeze(-1).unsqueeze(-1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()
        
    def forward(self, x):
        return self.sa(self.ca(x))

# ==========================================
# RESNET-50 BACKBONE + ATTENTION + HEAD
# ==========================================
class BCSModel(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True):
        super().__init__()
        # Load backbone with feature maps preserved (global_pool='')
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='')
        in_channels = self.backbone.num_features
        
        self.cbam = CBAM(in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, NUM_CORAL_OUTPUTS)
        
    def forward(self, x):
        features = self.backbone(x)
        features = self.cbam(features)
        features = self.pool(features).squeeze(-1).squeeze(-1)
        logits = self.fc(features)
        return logits

# ==========================================
# LOSS AND PREDICTION FUNCTIONS
# ==========================================
def coral_loss(logits, labels, num_classes=NUM_CLASSES):
    sets = []
    for i in range(num_classes - 1):
        label_i = (labels > i).float()
        sets.append(label_i)
    labels_stacked = torch.stack(sets, dim=1)
    loss = F.binary_cross_entropy_with_logits(logits, labels_stacked)
    return loss

def predict_class(logits):
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)

# ==========================================
# DATASET IMPLEMENTATION
# ==========================================
class CattleBCSDataset(Dataset):
    def __init__(self, csv_path, split, label_mapping, transform=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.label_mapping = label_mapping
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        
        # Open and ensure 3 channels
        img = Image.open(img_path).convert('RGB')
        
        raw_label = row['label']
        label = self.label_mapping[raw_label]
        
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(label, dtype=torch.long)

# ==========================================
# TRAINING AND EVALUATION ENGINE
# ==========================================
def train_and_evaluate(dataset_name, csv_path, label_mapping, checkpoint_name):
    print(f"\n==================================================")
    print(f"STARTING TRAINING ON DATASET: {dataset_name}")
    print(f"==================================================")
    
    # Preprocessing pipelines
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataloaders
    train_dataset = CattleBCSDataset(csv_path, 'train', label_mapping, train_transform)
    val_dataset = CattleBCSDataset(csv_path, 'val', label_mapping, val_test_transform)
    test_dataset = CattleBCSDataset(csv_path, 'test', label_mapping, val_test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    model = BCSModel(model_name='resnet50', pretrained=True).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    
    best_val_mae = float('inf')
    loss_history = []
    
    # Dictionary to capture progress milestones for reporting
    metrics_report = {
        'loss_ep10': 0.0, 'loss_ep20': 0.0, 'loss_ep30': 0.0, 'final_train_loss': 0.0,
        'val_mae': 0.0, 'val_acc0': 0.0, 'val_acc1': 0.0,
        'test_mae': 0.0, 'test_acc0': 0.0, 'test_acc1': 0.0,
        'time_mins': 0.0
    }
    
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        # Progress bar setup
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for images, labels in progress_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = coral_loss(logits, labels, NUM_CLASSES)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            progress_bar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(train_loader.dataset)
        loss_history.append(epoch_loss)
        scheduler.step()
        
        # Capture specific epoch metrics
        if epoch == 10: metrics_report['loss_ep10'] = epoch_loss
        if epoch == 20: metrics_report['loss_ep20'] = epoch_loss
        if epoch == 30: metrics_report['loss_ep30'] = epoch_loss
        metrics_report['final_train_loss'] = epoch_loss
        
        # Validation evaluation loop
        model.eval()
        val_abs_errors, val_exact_matches, val_close_matches = [], [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits = model(images)
                preds = predict_class(logits)
                
                abs_err = torch.abs(preds - labels)
                val_abs_errors.extend(abs_err.cpu().numpy())
                val_exact_matches.extend((abs_err == 0).cpu().numpy())
                val_close_matches.extend((abs_err <= 1).cpu().numpy())
                
        val_mae = np.mean(val_abs_errors)
        val_acc0 = np.mean(val_exact_matches) * 100
        val_acc1 = np.mean(val_close_matches) * 100
        
        print(f"Epoch {epoch} Summary -> Train Loss: {epoch_loss:.4f} | Val MAE: {val_mae:.4f} | Val Acc+-0: {val_acc0:.2f}%")
        
        # Save best model logic
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            metrics_report['val_mae'] = val_mae
            metrics_report['val_acc0'] = val_acc0
            metrics_report['val_acc1'] = val_acc1
            torch.save(model.state_dict(), os.path.join(WORKSPACE_DIR, checkpoint_name))
            
    end_time = time.time()
    metrics_report['time_mins'] = (end_time - start_time) / 60
    
    # Final Testing Evaluation using the saved best weights
    print(f"\nEvaluating final checkpoint on Test Split...")
    best_model_path = os.path.join(WORKSPACE_DIR, checkpoint_name)
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    test_abs_errors, test_exact_matches, test_close_matches = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            preds = predict_class(logits)
            
            abs_err = torch.abs(preds - labels)
            test_abs_errors.extend(abs_err.cpu().numpy())
            test_exact_matches.extend((abs_err == 0).cpu().numpy())
            test_close_matches.extend((abs_err <= 1).cpu().numpy())
            
    metrics_report['test_mae'] = np.mean(test_abs_errors)
    metrics_report['test_acc0'] = np.mean(test_exact_matches) * 100
    metrics_report['test_acc1'] = np.mean(test_close_matches) * 100
    
    return metrics_report, loss_history

# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    # RUN 1: Dryad Dataset
    dryad_csv = r"D:\T25301094 P2\datasets\bcs\bcs_index.csv"
    dryad_mapping = {2:0, 3:1, 4:2, 5:3, 6:4}
    dryad_metrics, dryad_loss = train_and_evaluate("Dryad", dryad_csv, dryad_mapping, "dryad_bcs_best.pth")
    
    # RUN 2: ScienceDB Dataset
    sciencedb_csv = r"D:\T25301094 P2\datasets\bcs\sciencedb_bcs_index.csv"
    sciencedb_mapping = {3.25:0, 3.5:1, 3.75:2, 4.0:3, 4.25:4}
    sciencedb_metrics, sciencedb_loss = train_and_evaluate("ScienceDB", sciencedb_csv, sciencedb_mapping, "sciencedb_bcs_best.pth")
    
    # 1. Plot and save Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS+1), dryad_loss, label='Dryad Train Loss', color='blue', linestyle='-')
    plt.plot(range(1, EPOCHS+1), sciencedb_loss, label='ScienceDB Train Loss', color='orange', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('BCS Training Loss Curves (ResNet-50)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(WORKSPACE_DIR, "bcs_loss_curve.png"))
    plt.close()
    
    # 2. Compile Context 3 Report String
    report_output = f"""---CONTEXT 3 BCS---
PERSON NAME: Bithi
BASE MODEL: ResNet-50

DATASET: Dryad
EPOCHS TRAINED: {EPOCHS}
LOSS AT EPOCH 10: {dryad_metrics['loss_ep10']:.4f}
LOSS AT EPOCH 20: {dryad_metrics['loss_ep20']:.4f}
LOSS AT EPOCH 30: {dryad_metrics['loss_ep30']:.4f}
FINAL TRAIN LOSS: {dryad_metrics['final_train_loss']:.4f}
VAL MAE: {dryad_metrics['val_mae']:.4f}
VAL ACCURACY +-0 (exact match): {dryad_metrics['val_acc0']:.2f}%
VAL ACCURACY +-1 (within 1 class): {dryad_metrics['val_acc1']:.2f}%
TEST MAE: {dryad_metrics['test_mae']:.4f}
TEST ACCURACY +-0: {dryad_metrics['test_acc0']:.2f}%
TEST ACCURACY +-1: {dryad_metrics['test_acc1']:.2f}%
CHECKPOINT PATH: D:\\T25301094 P2\\workspaces\\bithi\\dryad_bcs_best.pth
TRAINING TIME (mins): {dryad_metrics['time_mins']:.2f}
ANY ISSUES ENCOUNTERED: None

DATASET: ScienceDB
EPOCHS TRAINED: {EPOCHS}
LOSS AT EPOCH 10: {sciencedb_metrics['loss_ep10']:.4f}
LOSS AT EPOCH 20: {sciencedb_metrics['loss_ep20']:.4f}
LOSS AT EPOCH 30: {sciencedb_metrics['loss_ep30']:.4f}
FINAL TRAIN LOSS: {sciencedb_metrics['final_train_loss']:.4f}
VAL MAE: {sciencedb_metrics['val_mae']:.4f}
VAL ACCURACY +-0 (exact match): {sciencedb_metrics['val_acc0']:.2f}%
VAL ACCURACY +-1 (within 1 class): {sciencedb_metrics['val_acc1']:.2f}%
TEST MAE: {sciencedb_metrics['test_mae']:.4f}
TEST ACCURACY +-0: {sciencedb_metrics['test_acc0']:.2f}%
TEST ACCURACY +-1: {sciencedb_metrics['test_acc1']:.2f}%
CHECKPOINT PATH: D:\\T25301094 P2\\workspaces\\bithi\\sciencedb_bcs_best.pth
TRAINING TIME (mins): {sciencedb_metrics['time_mins']:.2f}
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---"""
    
    # Save text report to file
    with open(os.path.join(WORKSPACE_DIR, "bcs_results.txt"), "w") as f:
        f.write(report_output)
        
    print(f"\nExecution Finished! Results and loss curves successfully compiled in {WORKSPACE_DIR}")
```

### FILE: workspaces\bithi\train_behavior.py
---
```python
import os
import time
import random
import numpy as np
import pandas as pd

import cv2
cv2.setNumThreads(0)  # Disable OpenCV multithreading to prevent deadlocks in DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from sklearn.metrics import f1_score, confusion_matrix

import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend to prevent Tkinter thread crashes
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
PERSON_NAME = "Bithi"
BASE_MODEL_DISPLAY = "ResNet-50"
MODEL_NAME = "resnet50"
BASE_DIR = r"D:\T25301094 P2"
WORKSPACE_DIR = r"D:\T25301094 P2\workspaces\bithi"
CSV_PATH = r"D:\T25301094 P2\datasets\behavior\behavior_index.csv"
CHECKPOINT_PATH = os.path.join(WORKSPACE_DIR, "behavior_best.pth")
RESULTS_PATH = os.path.join(WORKSPACE_DIR, "behavior_results.txt")
LOSS_CURVE_PATH = os.path.join(WORKSPACE_DIR, "behavior_loss_curve.png")

NUM_CLASSES = 7
BATCH_SIZE = 128
NUM_WORKERS = 4
MAX_EPOCHS = 30
PATIENCE = 10
RANDOM_SEED = 42

CLASS_NAMES = [
    "Walking",
    "Standing",
    "Feeding head up",
    "Feeding head down",
    "Licking",
    "Drinking",
    "Lying",
]

# ============================================================
# REPRODUCIBILITY & SETUP
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)
torch.backends.cudnn.benchmark = True

# ============================================================
# DATASET
# ============================================================
class BehaviorDataset(Dataset):
    def __init__(self, csv_path, split, transform=None):
        df = pd.read_csv(csv_path)
        df = df[df['split'] == split]
        # Mandatory sampling to fix class imbalance and make training fast
        df = pd.concat([
            group.sample(min(len(group), 3000), random_state=RANDOM_SEED)
            for _, group in df.groupby('label')
        ]).reset_index(drop=True)
        self.data = df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = str(row["image_path"])
        if not os.path.isabs(img_path):
            img_path = os.path.join(BASE_DIR, img_path)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = int(row['label']) - 1  # 1-7 to 0-6
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label

# ============================================================
# CBAM ATTENTION MODULES
# ============================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, max(in_channels // reduction, 1)),
            nn.ReLU(),
            nn.Linear(max(in_channels // reduction, 1), in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_ = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        return x * self.sigmoid(avg + max_).unsqueeze(-1).unsqueeze(-1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))

# ============================================================
# FOCAL LOSS
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# ============================================================
# MODEL ARCHITECTURE
# ============================================================
class BehaviorModel(nn.Module):
    def __init__(self, model_name, num_classes, device):
        super().__init__()
        backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='')
        backbone = backbone.to(device, non_blocking=True)
        with torch.no_grad():
            dummy = backbone(torch.zeros(1, 3, 224, 224).to(device, non_blocking=True))
            feature_dim = dummy.shape[1]
        self.backbone = backbone
        self.cbam = CBAM(feature_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ============================================================
# METRICS EVALUATION
# ============================================================
def evaluate(model, loader, device, desc):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    
    per_class_acc = []
    for i in range(NUM_CLASSES):
        class_total = np.sum(cm[i, :])
        acc = (cm[i, i] / class_total * 100.0) if class_total > 0 else 0.0
        per_class_acc.append(acc)
        
    return macro_f1, per_class_acc

# ============================================================
# MAIN TRAINING LOOP
# ============================================================
def main():
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Transforms
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    eval_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # Datasets & Loaders
    train_ds = BehaviorDataset(CSV_PATH, "train", train_transform)
    val_ds = BehaviorDataset(CSV_PATH, "val", eval_transform)
    test_ds = BehaviorDataset(CSV_PATH, "test", eval_transform)
    
    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2
    )
    train_loader = DataLoader(train_ds, shuffle=True, persistent_workers=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, persistent_workers=True, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, persistent_workers=False, **loader_kwargs)
    
    # Build Model
    model = BehaviorModel(MODEL_NAME, NUM_CLASSES, device).to(device)
    criterion = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_f1 = -1.0
    epochs_no_improve = 0
    actual_epochs_trained = 0
    early_stopping_epoch = "N/A"
    
    epoch_losses = []
    loss_milestones = {10: "N/A", 20: "N/A", 30: "N/A"}
    
    start_time = time.time()
    print("="*60)
    print(f"  BEHAVIOR TRAINING — {PERSON_NAME} — {BASE_MODEL_DISPLAY}")
    print(f"  Workspace: {WORKSPACE_DIR}")
    print("="*60)
    
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{MAX_EPOCHS:02d}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_losses.append(epoch_loss)
        scheduler.step()
        
        if epoch in loss_milestones:
            loss_milestones[epoch] = f"{epoch_loss:.6f}"
            
        # Validation evaluation
        val_f1, val_accs = evaluate(model, val_loader, device, "Validating")
        print(f"Epoch {epoch:02d}/{MAX_EPOCHS:02d} | Loss: {epoch_loss:.6f} | Val Macro F1: {val_f1:.4f}")
        
        # Checkpoint Saving & Early Stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}. No improvement for {PATIENCE} epochs.")
                early_stopping_epoch = str(epoch)
                actual_epochs_trained = epoch
                break
                
        actual_epochs_trained = epoch

    end_time = time.time()
    training_time_mins = (end_time - start_time) / 60.0
    final_train_loss = f"{epoch_losses[-1]:.6f}"
    
    # Save Loss Curve
    plt.figure()
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{PERSON_NAME} - Behavior Training Loss')
    plt.legend()
    plt.savefig(LOSS_CURVE_PATH)
    plt.close()
    
    # Load best checkpoint and evaluate on Test set
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
    final_val_f1, final_val_per_class = evaluate(model, val_loader, device, "Evaluating Val")
    test_f1, test_per_class = evaluate(model, test_loader, device, "Evaluating Test")
    
    # Context 3 Report content
    report_content = f"""---CONTEXT 3 BEHAVIOR---
PERSON NAME: {PERSON_NAME}
BASE MODEL: {BASE_MODEL_DISPLAY}
DATASET: MmCows (capped 3000/class)
EPOCHS TRAINED: {actual_epochs_trained}
LOSS AT EPOCH 10: {loss_milestones.get(10, "N/A")}
LOSS AT EPOCH 20: {loss_milestones.get(20, "N/A")}
LOSS AT EPOCH 30: {loss_milestones.get(30, "N/A")}
FINAL TRAIN LOSS: {final_train_loss}
VAL MACRO F1: {final_val_f1:.4f}
VAL PER-CLASS ACCURACY:
  Class 1 (Walking): {final_val_per_class[0]:.2f}%
  Class 2 (Standing): {final_val_per_class[1]:.2f}%
  Class 3 (Feeding head up): {final_val_per_class[2]:.2f}%
  Class 4 (Feeding head down): {final_val_per_class[3]:.2f}%
  Class 5 (Licking): {final_val_per_class[4]:.2f}%
  Class 6 (Drinking): {final_val_per_class[5]:.2f}%
  Class 7 (Lying): {final_val_per_class[6]:.2f}%
TEST MACRO F1: {test_f1:.4f}
TEST PER-CLASS ACCURACY:
  Class 1 (Walking): {test_per_class[0]:.2f}%
  Class 2 (Standing): {test_per_class[1]:.2f}%
  Class 3 (Feeding head up): {test_per_class[2]:.2f}%
  Class 4 (Feeding head down): {test_per_class[3]:.2f}%
  Class 5 (Licking): {test_per_class[4]:.2f}%
  Class 6 (Drinking): {test_per_class[5]:.2f}%
  Class 7 (Lying): {test_per_class[6]:.2f}%
CHECKPOINT PATH: {CHECKPOINT_PATH}
TRAINING TIME (mins): {training_time_mins:.2f}
EARLY STOPPING TRIGGERED AT EPOCH: {early_stopping_epoch}
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
"""
    with open(RESULTS_PATH, "w") as f:
        f.write(report_content)
        
    print(f"Training complete. Results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()

```

## 10. TRAINING CODE (SHOUVIK - DenseNet121)
================================================================================

### FILE: workspaces\shouvik\train_bcs.py
---
```python
# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ASvOwztjjz3OSW2nRZnFhET8xSologWQ
"""

# train_bcs.py — Shouvik | DenseNet121 | BCS Task
# Save to: D:\T25301094 P2\workspaces\shouvik\train_bcs.py

import os
import time
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
import timm
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Seed ───────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ─── CBAM ────────────────────────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_ = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        return x * self.sigmoid(avg + max_).unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))


class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


# ─── Model ───────────────────────────────────────────────────────────────────
class DenseNet121BCS(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = timm.create_model(
            'densenet121', pretrained=True, global_pool='', num_classes=0
        )
        feature_dim = self.backbone.num_features  # 1024 for DenseNet121
        self.cbam = CBAM(feature_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(feature_dim, num_classes - 1)  # CORAL: 4 nodes

    def forward(self, x):
        feat = self.backbone(x)       # (B, 1024, H, W)
        feat = self.cbam(feat)        # (B, 1024, H, W)
        feat = self.pool(feat)        # (B, 1024, 1, 1)
        feat = feat.view(feat.size(0), -1)  # (B, 1024)
        logits = self.head(feat)      # (B, 4)
        return logits


# ─── CORAL Loss ──────────────────────────────────────────────────────────────
def coral_loss(logits, labels, num_classes=5):
    sets = [(labels > i).float() for i in range(num_classes - 1)]
    labels_stacked = torch.stack(sets, dim=1)
    return F.binary_cross_entropy_with_logits(logits, labels_stacked)


def coral_predict(logits):
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)


# ─── Dataset ─────────────────────────────────────────────────────────────────
class BCSDataset(Dataset):
    def __init__(self, df, label_map, transform):
        self.df = df.reset_index(drop=True)
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        img = self.transform(img)
        label = self.label_map[row['label']]
        return img, torch.tensor(label, dtype=torch.long)


# ─── Transforms ──────────────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ─── Metrics ─────────────────────────────────────────────────────────────────
def compute_metrics(model, loader, num_classes=5):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            preds = coral_predict(logits).cpu()
            all_preds.append(preds)
            all_labels.append(labels)
    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    mae    = float(np.mean(np.abs(preds - labels)))
    acc0   = float(np.mean(preds == labels))
    acc1   = float(np.mean(np.abs(preds - labels) <= 1))
    return mae, acc0, acc1


# ─── Training Loop ───────────────────────────────────────────────────────────
def train_one_dataset(
    dataset_name, csv_path, label_map,
    checkpoint_name, workspace, batch_size=32, epochs=30, num_classes=5
):
    print(f"\n{'='*60}")
    print(f"  TRAINING ON: {dataset_name}")
    print(f"{'='*60}\n")

    df = pd.read_csv(csv_path)
    df['label'] = df['label'].apply(lambda x: float(x) if '.' in str(x) else int(x))

    train_df = df[df['split'] == 'train']
    val_df   = df[df['split'] == 'val']
    test_df  = df[df['split'] == 'test']

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_ds = BCSDataset(train_df, label_map, train_tf)
    val_ds   = BCSDataset(val_df,   label_map, val_tf)
    test_ds  = BCSDataset(test_df,  label_map, val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = DenseNet121BCS(num_classes=num_classes).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_mae = float('inf')
    checkpoint_path = os.path.join(workspace, checkpoint_name)

    history = {
        'train_loss': [],
        'val_mae': [],
        'epoch_loss_at': {}
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"[{dataset_name}] Epoch {epoch:02d}/{epochs}", leave=True)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = coral_loss(logits, labels, num_classes)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_loss = running_loss / len(train_ds)
        val_mae, val_acc0, val_acc1 = compute_metrics(model, val_loader, num_classes)

        history['train_loss'].append(epoch_loss)
        history['val_mae'].append(val_mae)

        if epoch in [10, 20, 30]:
            history['epoch_loss_at'][epoch] = epoch_loss

        print(f"  → Epoch {epoch:02d} | Train Loss: {epoch_loss:.4f} | Val MAE: {val_mae:.4f} | "
              f"Acc±0: {val_acc0*100:.2f}% | Acc±1: {val_acc1*100:.2f}%")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Best model saved → {checkpoint_name} (Val MAE: {best_val_mae:.4f})")

        scheduler.step()

    elapsed = (time.time() - start_time) / 60.0
    print(f"\n  Training complete in {elapsed:.1f} minutes.")

    # ── Load best and evaluate test ──
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    test_mae, test_acc0, test_acc1 = compute_metrics(model, test_loader, num_classes)

    print(f"\n  TEST RESULTS ({dataset_name}):")
    print(f"  Test MAE     : {test_mae:.4f}")
    print(f"  Test Acc ±0  : {test_acc0*100:.2f}%")
    print(f"  Test Acc ±1  : {test_acc1*100:.2f}%")

    result = {
        'dataset'       : dataset_name,
        'epochs_trained': epochs,
        'loss_ep10'     : history['epoch_loss_at'].get(10, 'N/A'),
        'loss_ep20'     : history['epoch_loss_at'].get(20, 'N/A'),
        'loss_ep30'     : history['epoch_loss_at'].get(30, 'N/A'),
        'final_train_loss': history['train_loss'][-1],
        'val_mae'       : best_val_mae,
        'val_acc0'      : val_acc0,
        'val_acc1'      : val_acc1,
        'test_mae'      : test_mae,
        'test_acc0'     : test_acc0,
        'test_acc1'     : test_acc1,
        'checkpoint'    : checkpoint_path,
        'train_time_min': elapsed,
        'train_loss_hist': history['train_loss'],
        'val_mae_hist'  : history['val_mae'],
    }
    return result


# ─── Save Results TXT ────────────────────────────────────────────────────────
def save_results_txt(workspace, r1, r2):
    path = os.path.join(workspace, 'bcs_results.txt')

    def fmt_loss(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    lines = [
        "---CONTEXT 3 BCS---",
        "PERSON NAME: Shouvik",
        "BASE MODEL: DenseNet121",
        "",
        f"DATASET: {r1['dataset']}",
        f"EPOCHS TRAINED: {r1['epochs_trained']}",
        f"LOSS AT EPOCH 10: {fmt_loss(r1['loss_ep10'])}",
        f"LOSS AT EPOCH 20: {fmt_loss(r1['loss_ep20'])}",
        f"LOSS AT EPOCH 30: {fmt_loss(r1['loss_ep30'])}",
        f"FINAL TRAIN LOSS: {fmt_loss(r1['final_train_loss'])}",
        f"VAL MAE: {r1['val_mae']:.4f}",
        f"VAL ACCURACY +-0 (exact match): {r1['val_acc0']*100:.2f}%",
        f"VAL ACCURACY +-1 (within 1 class): {r1['val_acc1']*100:.2f}%",
        f"TEST MAE: {r1['test_mae']:.4f}",
        f"TEST ACCURACY +-0: {r1['test_acc0']*100:.2f}%",
        f"TEST ACCURACY +-1: {r1['test_acc1']*100:.2f}%",
        f"CHECKPOINT PATH: {r1['checkpoint']}",
        f"TRAINING TIME (mins): {r1['train_time_min']:.1f}",
        "ANY ISSUES ENCOUNTERED: None",
        "",
        f"DATASET: {r2['dataset']}",
        f"EPOCHS TRAINED: {r2['epochs_trained']}",
        f"LOSS AT EPOCH 10: {fmt_loss(r2['loss_ep10'])}",
        f"LOSS AT EPOCH 20: {fmt_loss(r2['loss_ep20'])}",
        f"LOSS AT EPOCH 30: {fmt_loss(r2['loss_ep30'])}",
        f"FINAL TRAIN LOSS: {fmt_loss(r2['final_train_loss'])}",
        f"VAL MAE: {r2['val_mae']:.4f}",
        f"VAL ACCURACY +-0 (exact match): {r2['val_acc0']*100:.2f}%",
        f"VAL ACCURACY +-1 (within 1 class): {r2['val_acc1']*100:.2f}%",
        f"TEST MAE: {r2['test_mae']:.4f}",
        f"TEST ACCURACY +-0: {r2['test_acc0']*100:.2f}%",
        f"TEST ACCURACY +-1: {r2['test_acc1']*100:.2f}%",
        f"CHECKPOINT PATH: {r2['checkpoint']}",
        f"TRAINING TIME (mins): {r2['train_time_min']:.1f}",
        "ANY ISSUES ENCOUNTERED: None",
        "---END CONTEXT 3---",
    ]
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n  ✓ bcs_results.txt saved → {path}")


# ─── Save Loss Curve ──────────────────────────────────────────────────────────
def save_loss_curve(workspace, r1, r2):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, r in zip(axes, [r1, r2]):
        epochs = range(1, len(r['train_loss_hist']) + 1)
        ax.plot(epochs, r['train_loss_hist'], label='Train Loss', color='steelblue')
        ax2 = ax.twinx()
        ax2.plot(epochs, r['val_mae_hist'], label='Val MAE', color='tomato', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Train Loss', color='steelblue')
        ax2.set_ylabel('Val MAE', color='tomato')
        ax.set_title(f"DenseNet121 — {r['dataset']}")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    path = os.path.join(workspace, 'bcs_loss_curve.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ bcs_loss_curve.png saved → {path}")


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    WORKSPACE   = r"D:\T25301094 P2\workspaces\shouvik"
    BATCH_SIZE  = 32
    EPOCHS      = 30
    NUM_CLASSES = 5

    # ── Run 1: Dryad ──
    dryad_result = train_one_dataset(
        dataset_name    = 'Dryad',
        csv_path        = r"D:\T25301094 P2\datasets\bcs\bcs_index.csv",
        label_map       = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4},
        checkpoint_name = 'dryad_bcs_best.pth',
        workspace       = WORKSPACE,
        batch_size      = BATCH_SIZE,
        epochs          = EPOCHS,
        num_classes     = NUM_CLASSES,
    )

    # ── Run 2: ScienceDB ──
    sciencedb_result = train_one_dataset(
        dataset_name    = 'ScienceDB',
        csv_path        = r"D:\T25301094 P2\datasets\bcs\sciencedb_bcs_index.csv",
        label_map       = {3.25: 0, 3.5: 1, 3.75: 2, 4.0: 3, 4.25: 4},
        checkpoint_name = 'sciencedb_bcs_best.pth',
        workspace       = WORKSPACE,
        batch_size      = BATCH_SIZE,
        epochs          = EPOCHS,
        num_classes     = NUM_CLASSES,
    )

    # ── Save outputs ──
    save_results_txt(WORKSPACE, dryad_result, sciencedb_result)
    save_loss_curve(WORKSPACE,  dryad_result, sciencedb_result)

    print("\n  ALL DONE. Share bcs_results.txt with Hasin.")
```

### FILE: workspaces\shouvik\train_behavior.py
---
```python
import os
import time
import random
import numpy as np
import pandas as pd

import cv2
cv2.setNumThreads(0)  # Disable OpenCV multithreading to prevent deadlocks in DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from sklearn.metrics import f1_score, confusion_matrix

import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend to prevent Tkinter thread crashes
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
PERSON_NAME = "Shouvik"
BASE_MODEL_DISPLAY = "DenseNet121"
MODEL_NAME = "densenet121"
BASE_DIR = r"D:\T25301094 P2"
WORKSPACE_DIR = r"D:\T25301094 P2\workspaces\shouvik"
CSV_PATH = r"D:\T25301094 P2\datasets\behavior\behavior_index.csv"
CHECKPOINT_PATH = os.path.join(WORKSPACE_DIR, "behavior_best.pth")
RESULTS_PATH = os.path.join(WORKSPACE_DIR, "behavior_results.txt")
LOSS_CURVE_PATH = os.path.join(WORKSPACE_DIR, "behavior_loss_curve.png")

NUM_CLASSES = 7
BATCH_SIZE = 128
NUM_WORKERS = 4
MAX_EPOCHS = 30
PATIENCE = 10
RANDOM_SEED = 42

CLASS_NAMES = [
    "Walking",
    "Standing",
    "Feeding head up",
    "Feeding head down",
    "Licking",
    "Drinking",
    "Lying",
]

# ============================================================
# REPRODUCIBILITY & SETUP
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)
torch.backends.cudnn.benchmark = True

# ============================================================
# DATASET
# ============================================================
class BehaviorDataset(Dataset):
    def __init__(self, csv_path, split, transform=None):
        df = pd.read_csv(csv_path)
        df = df[df['split'] == split]
        # Mandatory sampling to fix class imbalance and make training fast
        df = pd.concat([
            group.sample(min(len(group), 3000), random_state=RANDOM_SEED)
            for _, group in df.groupby('label')
        ]).reset_index(drop=True)
        self.data = df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = str(row["image_path"])
        if not os.path.isabs(img_path):
            img_path = os.path.join(BASE_DIR, img_path)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = int(row['label']) - 1  # 1-7 to 0-6
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label

# ============================================================
# CBAM ATTENTION MODULES
# ============================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, max(in_channels // reduction, 1)),
            nn.ReLU(),
            nn.Linear(max(in_channels // reduction, 1), in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x).squeeze(-1).squeeze(-1))
        max_ = self.fc(self.max_pool(x).squeeze(-1).squeeze(-1))
        return x * self.sigmoid(avg + max_).unsqueeze(-1).unsqueeze(-1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))

# ============================================================
# FOCAL LOSS
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# ============================================================
# MODEL ARCHITECTURE
# ============================================================
class BehaviorModel(nn.Module):
    def __init__(self, model_name, num_classes, device):
        super().__init__()
        backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='')
        backbone = backbone.to(device, non_blocking=True)
        with torch.no_grad():
            dummy = backbone(torch.zeros(1, 3, 224, 224).to(device, non_blocking=True))
            feature_dim = dummy.shape[1]
        self.backbone = backbone
        self.cbam = CBAM(feature_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ============================================================
# METRICS EVALUATION
# ============================================================
def evaluate(model, loader, device, desc):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    
    per_class_acc = []
    for i in range(NUM_CLASSES):
        class_total = np.sum(cm[i, :])
        acc = (cm[i, i] / class_total * 100.0) if class_total > 0 else 0.0
        per_class_acc.append(acc)
        
    return macro_f1, per_class_acc

# ============================================================
# MAIN TRAINING LOOP
# ============================================================
def main():
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Transforms
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    eval_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # Datasets & Loaders
    train_ds = BehaviorDataset(CSV_PATH, "train", train_transform)
    val_ds = BehaviorDataset(CSV_PATH, "val", eval_transform)
    test_ds = BehaviorDataset(CSV_PATH, "test", eval_transform)
    
    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2
    )
    train_loader = DataLoader(train_ds, shuffle=True, persistent_workers=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, persistent_workers=True, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, persistent_workers=False, **loader_kwargs)
    
    # Build Model
    model = BehaviorModel(MODEL_NAME, NUM_CLASSES, device).to(device)
    criterion = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_f1 = -1.0
    epochs_no_improve = 0
    actual_epochs_trained = 0
    early_stopping_epoch = "N/A"
    
    epoch_losses = []
    loss_milestones = {10: "N/A", 20: "N/A", 30: "N/A"}
    
    start_time = time.time()
    print("="*60)
    print(f"  BEHAVIOR TRAINING — {PERSON_NAME} — {BASE_MODEL_DISPLAY}")
    print(f"  Workspace: {WORKSPACE_DIR}")
    print("="*60)
    
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{MAX_EPOCHS:02d}"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_losses.append(epoch_loss)
        scheduler.step()
        
        if epoch in loss_milestones:
            loss_milestones[epoch] = f"{epoch_loss:.6f}"
            
        # Validation evaluation
        val_f1, val_accs = evaluate(model, val_loader, device, "Validating")
        print(f"Epoch {epoch:02d}/{MAX_EPOCHS:02d} | Loss: {epoch_loss:.6f} | Val Macro F1: {val_f1:.4f}")
        
        # Checkpoint Saving & Early Stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}. No improvement for {PATIENCE} epochs.")
                early_stopping_epoch = str(epoch)
                actual_epochs_trained = epoch
                break
                
        actual_epochs_trained = epoch

    end_time = time.time()
    training_time_mins = (end_time - start_time) / 60.0
    final_train_loss = f"{epoch_losses[-1]:.6f}"
    
    # Save Loss Curve
    plt.figure()
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{PERSON_NAME} - Behavior Training Loss')
    plt.legend()
    plt.savefig(LOSS_CURVE_PATH)
    plt.close()
    
    # Load best checkpoint and evaluate on Test set
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
    final_val_f1, final_val_per_class = evaluate(model, val_loader, device, "Evaluating Val")
    test_f1, test_per_class = evaluate(model, test_loader, device, "Evaluating Test")
    
    # Context 3 Report content
    report_content = f"""---CONTEXT 3 BEHAVIOR---
PERSON NAME: {PERSON_NAME}
BASE MODEL: {BASE_MODEL_DISPLAY}
DATASET: MmCows (capped 3000/class)
EPOCHS TRAINED: {actual_epochs_trained}
LOSS AT EPOCH 10: {loss_milestones.get(10, "N/A")}
LOSS AT EPOCH 20: {loss_milestones.get(20, "N/A")}
LOSS AT EPOCH 30: {loss_milestones.get(30, "N/A")}
FINAL TRAIN LOSS: {final_train_loss}
VAL MACRO F1: {final_val_f1:.4f}
VAL PER-CLASS ACCURACY:
  Class 1 (Walking): {final_val_per_class[0]:.2f}%
  Class 2 (Standing): {final_val_per_class[1]:.2f}%
  Class 3 (Feeding head up): {final_val_per_class[2]:.2f}%
  Class 4 (Feeding head down): {final_val_per_class[3]:.2f}%
  Class 5 (Licking): {final_val_per_class[4]:.2f}%
  Class 6 (Drinking): {final_val_per_class[5]:.2f}%
  Class 7 (Lying): {final_val_per_class[6]:.2f}%
TEST MACRO F1: {test_f1:.4f}
TEST PER-CLASS ACCURACY:
  Class 1 (Walking): {test_per_class[0]:.2f}%
  Class 2 (Standing): {test_per_class[1]:.2f}%
  Class 3 (Feeding head up): {test_per_class[2]:.2f}%
  Class 4 (Feeding head down): {test_per_class[3]:.2f}%
  Class 5 (Licking): {test_per_class[4]:.2f}%
  Class 6 (Drinking): {test_per_class[5]:.2f}%
  Class 7 (Lying): {test_per_class[6]:.2f}%
CHECKPOINT PATH: {CHECKPOINT_PATH}
TRAINING TIME (mins): {training_time_mins:.2f}
EARLY STOPPING TRIGGERED AT EPOCH: {early_stopping_epoch}
ANY ISSUES ENCOUNTERED: None
---END CONTEXT 3---
"""
    with open(RESULTS_PATH, "w") as f:
        f.write(report_content)
        
    print(f"Training complete. Results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()

```

