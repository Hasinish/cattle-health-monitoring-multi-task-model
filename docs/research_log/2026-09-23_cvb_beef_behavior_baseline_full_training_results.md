# Research Log: CVB + Kaggle Beef RGB Behavior Baseline Full 30-Epoch Training Results & Held-Out Evaluation

**Date:** 2026-09-23  
**Status:** CANONICAL / COMPLETED & 100% CERTIFIED ✅  
**Log Type:** Full Empirical Model Training & Held-Out Test Evaluation  
**Task:** Behavior Recognition (Run 2 of 8 in Deadline Execution Plan / Phase 3 Step 4.2)  
**Parent Blueprint:** [`phase3_canonical_roadmap.md`](file:///d:/cattle-health-monitoring-multi-task-model/phase3_canonical_roadmap.md)  
**Deadline Priority Overlay:** [`phase3_deadline_execution_2026-09-26.md`](file:///d:/cattle-health-monitoring-multi-task-model/phase3_deadline_execution_2026-09-26.md)  
**Associated Deliverables:**
- `scripts/train_cvb_beef_behavior_baseline.py` (Local/remote training script)
- `scripts/modal_train_cvb_beef_behavior.py` (Modal cloud launcher on L40S)
- `artifacts/behavior_baseline/behavior_baseline_metrics.json` (Full empirical evaluation metrics)
- `artifacts/behavior_baseline/behavior_baseline_30epoch_summary.md` (Summary artifact)

---

## 1. Executive Summary

We have successfully executed the full 30-epoch training and held-out test evaluation of the **Phase 3 Step 4.2 Behavior RGB Single-Task Baseline** (Run 2 of 8 in the active Deadline Execution Plan) on Modal cloud using an **NVIDIA L40S GPU (48GB Ada Lovelace)** with 8 CPUs and 32GB RAM.

The pipeline operated on the canonical, source-video-disjoint and session-disjoint CVB + Kaggle Beef dataset (`datasets/behavior/cvb_beef/`; Train 3,785, Val 680, Test 809; 5 classes: Standing, Lying, Feeding, Drinking, Walking; Seed 2026). All video midpoint crops were pre-cached to persistent cloud storage (`behavior-checkpoints/behavior_cache`) to completely eliminate video decoding bottlenecks during optimization.

All 30 epochs completed in **821.5 seconds (~13.69 minutes)**. Global best validation performance was achieved at **Epoch 25** (Val Accuracy: 87.06%, Val Balanced Accuracy: 73.40%, Val Macro-F1: 0.7399). Final evaluation on the held-out sequence-safe test split (809 unseen clips from 44 completely unseen source videos/sessions) established the definitive Behavior single-task reference:

- **Overall Test Accuracy:** **88.88%** (719 / 809 correct)
- **Balanced Accuracy:** **71.72%** (over 3.5x higher than random chance of 20%)
- **Macro-F1:** **0.7413**
- **Test Cross-Entropy Loss:** **0.5312**
- **Kaggle Beef Sub-Accuracy:** **94.06%** (Macro-F1: 0.9113)
- **CVB Sub-Accuracy:** **84.12%** (Macro-F1: 0.6541)

This establishes the certified empirical baseline (E0 control) for Behavior against which the upcoming perception-enhanced temporal model (Run 5) and final multi-task architectures (Runs 7 and 8) will be benchmarked.

---

## 2. Experimental Setup & Hardware Configuration

| Component | Specification |
| :--- | :--- |
| **Cloud Platform** | Modal (`tigerwood693` profile) |
| **App ID** | `ap-ZrBKKvGcVzM7IB1AYMs2LH` |
| **Hardware** | NVIDIA L40S (48 GB VRAM, Ada Lovelace architecture) |
| **Container Resources** | 8.0 CPUs, 32,768 MB (32 GB) RAM |
| **Storage Mounts** | `/mnt/cvb` (`cvb-data`), `/mnt/beef` (`beef-behavior-data`), `/checkpoints` (`behavior-checkpoints`) |
| **Network & I/O Strategy** | Pre-cached 224x224 RGB midpoint crops stored directly on persistent volume |
| **Backbone Architecture** | Standard ResNet-18 (ImageNet-pretrained weights) |
| **Input Resolution** | 224 x 224 px RGB |
| **Loss Function** | Standard CrossEntropyLoss |
| **Optimizer** | AdamW (`lr=1e-4`, `weight_decay=1e-2`, `betas=(0.9, 0.999)`) |
| **Scheduler** | CosineAnnealingLR (`T_max=30`, `eta_min=1e-6`) |
| **Batch Size** | 64 (Train: 60 batches/epoch; Val: 11 batches; Test: 13 batches) |
| **Epoch Count** | 30 full epochs |
| **Training Duration** | 821.47 seconds (~13.69 minutes) |

---

## 3. Training & Validation Progression

The training progressed rapidly at **~8.3 seconds per epoch** once the initial video decoding cache was assembled.

- **Epoch 1:** Train Loss 0.4722 | Val Loss 0.6481 | Val Macro-F1 0.4912 | Val Acc 75.74%
- **Epoch 2:** Train Loss 0.2055 | Val Loss 0.5429 | Val Macro-F1 0.6186 | Val Acc 81.18%
- **Epoch 5:** Train Loss 0.0894 | Val Loss 0.6102 | Val Macro-F1 0.6698 | Val Acc 83.38%
- **Epoch 10:** Train Loss 0.0381 | Val Loss 0.6558 | Val Macro-F1 0.7012 | Val Acc 85.00%
- **Epoch 20:** Train Loss 0.0142 | Val Loss 0.6214 | Val Macro-F1 0.7250 | Val Acc 86.47%
- **Epoch 25 (BEST CHECKPOINT):** Train Loss 0.0078 | Val Loss 0.5833 | **Val Macro-F1 0.7399** | **Val Bal Acc 73.40%** | **Val Acc 87.06%**
- **Epoch 30:** Train Loss 0.0049 | Val Loss 0.6012 | Val Macro-F1 0.7310 | Val Acc 86.76%

Global best checkpoint (`behavior_baseline_best.pth`) was captured at **Epoch 25** and restored for final evaluation.

---

## 4. Final Held-Out Test Evaluation Results

The final model was evaluated on the held-out test split (`datasets/behavior/cvb_beef/test.csv`), consisting of **809 samples across 44 completely unseen groups** (422 CVB track segments and 387 Kaggle Beef clips).

### 4.1 Global Aggregate Performance

| Metric | Score | Context / Interpretation |
| :--- | :--- | :--- |
| **Overall Accuracy** | **88.88%** | 719 out of 809 test clips classified correctly |
| **Balanced Accuracy** | **71.72%** | Unweighted average of recall across all 5 classes (>3.5x random baseline) |
| **Macro-F1** | **0.7413** | Harmonic mean across per-class F1 scores |
| **Cross-Entropy Loss** | **0.5312** | Clean convergence on unseen real-world video frames |

### 4.2 Per-Class Performance Breakdown

| Class | Precision | Recall | F1-Score | Support (N) | Analysis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Lying** | **0.9608** | **0.9496** | **0.9552** | 258 | Near-perfect recognition; recumbent body posture is visually unambiguous |
| **Feeding** | **0.8647** | **0.9885** | **0.9225** | 349 | Outstanding recall (345/349); head-down feeding posture is strongly detected |
| **Drinking** | **0.8793** | **0.8095** | **0.8430** | 63 | High precision; water trough interactions reliably separated |
| **Standing** | **0.9481** | **0.6460** | **0.7684** | 113 | High precision; standing cows occasionally confused with feeding |
| **Walking** | **0.2500** | **0.1923** | **0.2174** | 26 | Severely challenged on static single frames; confused with feeding/standing |

### 4.3 Confusion Matrix Analysis

Row: True Class, Column: Predicted Class  
Order: `[Standing, Lying, Feeding, Drinking, Walking]`

```text
               Pred Stand  Pred Lie  Pred Feed  Pred Drink  Pred Walk  Total
True Standing:     73         9         14          7          10       113
True Lying:         1       245         10          0           2       258
True Feeding:       1         0        345          0           3       349
True Drinking:      1         0         11         51           0        63
True Walking:       1         1         19          0           5        26
```

#### Key Confusion Insight (The Walking Kinetic Bottleneck):
- **19 of the 26 walking instances were misclassified as Feeding.**
- In CVB pasture footage, cows often walk slowly with heads angled toward the grass (grazing while moving). On a single static midpoint frame, a walking cow in pasture is visually indistinguishable from a grazing/feeding cow without temporal motion cues.
- **Scientific Significance:** This low static walking F1 (0.2174) provides the **exact empirical justification** for Phase 3 Step 8 / Deadline Run 5 (Temporal TCN model). Static 2D frames cannot resolve gait kinematics; sequential modeling across time is theoretically and practically required.

### 4.4 Sub-Dataset Domain Generalization

| Sub-Dataset | Test Support | Overall Accuracy | Balanced Accuracy | Macro-F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Kaggle Beef** | 387 clips | **94.06%** | **90.84%** | **0.9113** |
| **CVB** | 422 clips | **84.12%** | **60.62%** | **0.6541** |

Kaggle Beef clips are tight, pre-cropped single-cow videos in controlled feedlots, yielding exceptional accuracy (94.06%). CVB features wide-angle open-pasture footage with bounding-box cropping, multiple distant cows, and all 26 walking instances, providing a realistic uncurated test environment.

---

## 5. Artifact & Checkpoint Registry

All model weights and logs are permanently archived on the persistent Modal volume `behavior-checkpoints`:
- Best Model Checkpoint: `behavior-checkpoints/behavior_baseline/behavior_baseline_best.pth`
- Latest Epoch Checkpoint: `behavior-checkpoints/behavior_baseline/behavior_baseline_latest.pth`
- Local Metrics JSON: `artifacts/behavior_baseline/behavior_baseline_metrics.json`
- Local Metrics Summary: `artifacts/behavior_baseline/behavior_baseline_30epoch_summary.md`

---

## 6. Next Steps in Deadline Execution Plan

With Run 1 (BCS RGB) and Run 2 (Behavior RGB) fully certified:
1. **Run 3 (Re-ID RGB Baseline):** Implement and train ResNet-18 identity baseline on SideViewCows2026.
2. **Perception Caching (Step 3):** Cache combined perception features (box, soft mask, pose keypoints) for BCS, Behavior, and Re-ID.
3. **Run 4, 5, 6:** Perception-enhanced single-task runs.
4. **Run 7, 8:** Final MTL integration (E1 hard sharing vs E3 modular adapters).
