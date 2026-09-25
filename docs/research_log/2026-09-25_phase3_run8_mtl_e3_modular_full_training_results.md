# Research Log: Phase 3 Run 8 E3 Modular Multi-Task Learning Full Training Results

**Date**: 2026-09-25  
**Author**: Hasin Ishrak  
**Execution Environment**: Modal Cloud Profile `hasinishrak2015`  
**Hardware Tier**: NVIDIA L40S GPU (48 GB VRAM, 8 vCPUs, 32 GB RAM)  
**Modal App ID**: `ap-H0BerVczDO938LjXBIhLy0` (launched via `modal run --detach`)  
**Git Commit at Execution**: `50c9b06797544602d482e5a5ac76df668067ec07`  
**Target Milestone**: Phase 3 Step 7 / Run 8 of 8 in Deadline Execution Plan (E3 Modular Multi-Task Learning with Task-Private Residual Bottleneck Adapters)

---

## 1. Executive Summary

Full 30-epoch training and validation of **Phase 3 Run 8 (E3 Modular Multi-Task Learning / Task-Private Residual Bottleneck Adapters)** were successfully completed on Modal profile `hasinishrak2015` using an NVIDIA L40S GPU (total training duration 1,634.37s / ~27.24 minutes, exit code 0).

Run 8 investigates whether modular multi-task routing (allocating task-private residual bottleneck adapters alongside a shared spatial trunk) improves multi-task optimization and mitigates held-out negative transfer compared to the monolithic hard-shared baseline (Run 7 E1).

### Key Empirical Findings:
* **Throughput & Efficiency**: Using full in-memory RAM preloading (total dataset footprint ~9.5 GB in 32 GB RAM), training executed across all 537 multi-task super-steps per epoch at a steady **~53.5s per epoch** (total training time: 1,634.37s / ~27.24 minutes).
* **Global Best Multi-Task Objective (`val_e3_objective = 0.38888`) at Epoch 2**:
  * **Surpassed Run 7 E1 Best Objective**: Run 8 achieved `0.38888` at Epoch 2 vs Run 7's `0.40036` at Epoch 3 (lower composite validation loss).
  * **BCS**: Val Loss 0.4550, Real MAE **0.1906** BCS units (vs Run 7 E1 best checkpoint: 0.1968; -0.0062 error reduction), Acc@1 (+/- 0.25 tolerance) **87.80%**, Acc@0 **37.33%**, Balanced Acc **34.09%**, Macro-F1 **0.3277**.
  * **Behavior**: Val Loss 0.4134 (vs Run 7 E1: 0.4597), Accuracy **84.76%**, Balanced Accuracy **68.02%**, Macro-F1 **0.6725** (surges to **0.7577** at Epoch 18).
  * **Re-ID**: Val Loss 0.2982 (vs Run 7 E1: 0.3029), Top-1 Accuracy **92.84%**, Balanced Accuracy **92.88%**, Macro-F1 **0.9199** (reaches **96.20%** at Epoch 20).
* **Task-Private Adapter Dynamics**:
  * The lightweight residual adapters (395,904 parameters total; +3.32% capacity over E1) enabled faster initial joint alignment, reaching peak multi-task validation loss at Epoch 2 while keeping BCS clinical error lower (`0.1906` vs `0.1968` MAE).
* **Checkpoint & Metric Integrity**:
  * Checkpoints committed to `/mtl-checkpoints/mtl_e3_modular/mtl_e3_best.pth` and `mtl_e3_latest.pth`.
  * Verified zero test-set access or held-out evaluation during the training run.

---

## 2. Experimental Setup & Architecture

### 2.1 Multi-Task Modular Architecture (`MTLE3ModularModel`)
* **Shared Spatial Trunk**: 4-channel ResNet-18 (`nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)`). Channels 0-2 initialized from ImageNet; Channel 3 initialized from RGB channel mean. Spatial pooling outputs a unified 512-D cattle representation.
  * Parameters: **11,179,648** (90.72% of total model capacity).
* **Task-Private Modular Adapters (x3)**:
  * Structure: `Linear(512, 128) -> LayerNorm(128) -> GELU -> Dropout(0.1) -> Linear(128, 512) + Residual Skip`.
  * Initialization: Houlsby et al. (2019) zero-weight up-projection. At step 0, each adapter acts as an identity mapping ($h_t \equiv h_{\text{shared}}$).
  * Parameters: **131,968** per task x 3 = **395,904** params (3.21% of total capacity).
* **Task Heads (100% matched to Runs 4-7)**:
  * **BCS Head**: Frank & Hall ordinal classification head (`BCSOrdinalHead`, 4 thresholds over 5 classes, `Linear(512, 4)` = **2,052** params).
  * **Behavior Head**: 1D Temporal Convolutional Network (`BehaviorTemporalTCN`, 2 Conv1d blocks + Linear(256, 5) = **723,973** params).
  * **Re-ID Head**: Linear classifier over 41 representation learning cows (`ReIDHead`, `Linear(512, 41)` = **21,033** params).
  * Total Head Parameters: **747,058** params (6.06% of total capacity).
* **Total Trainable Parameters**: **12,322,610** (+395,904 params / +3.32% capacity vs Run 7 E1).

### 2.2 Super-Step Training Schedule
* Fixed, equal task loss weights: $w_{\text{bcs}} = 1.0, w_{\text{beh}} = 1.0, w_{\text{reid}} = 1.0$.
* Batch sizes: BCS = 64, Behavior = 8 sequences (64 frames), Re-ID = 32 images.
* Super-step loop: 537 super-steps per epoch (aligned to BCS batch count).
* Optimizer: AdamW ($lr=10^{-4}, wd=10^{-4}$), CosineAnnealingLR ($T_{\max}=30, \eta_{\min}=10^{-6}$).

---

## 3. Training & Validation Trajectory

| Epoch | Duration (s) | Train Total Loss | Val E3 Objective | BCS Val Loss | BCS Real MAE | Beh Val Loss | Beh Macro-F1 | Re-ID Val Loss | Re-ID Top-1 Acc |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 62.5 | 0.4723 | 0.4400 | 0.4808 | 0.2076 | 0.4796 | 0.6379 | 0.3597 | 92.32% |
| **2 (BEST)** | **53.9** | **0.1931** | **0.3889** | **0.4550** | **0.1906** | **0.4134** | **0.6725** | **0.2982** | **92.84%** |
| **3** | 54.2 | 0.1431 | 0.4234 | 0.5374 | 0.1977 | 0.5001 | 0.7105 | 0.2328 | 94.93% |
| **5** | 52.9 | 0.0878 | 0.4665 | 0.5571 | 0.1937 | 0.5306 | 0.6881 | 0.3118 | 91.76% |
| **8** | 52.6 | 0.0414 | 0.5560 | 0.7903 | 0.1964 | 0.6883 | 0.7117 | 0.1895 | 94.67% |
| **12** | 53.4 | 0.0207 | 0.6759 | 1.0524 | 0.2049 | 0.8199 | 0.6892 | 0.1554 | 95.71% |
| **18** | 52.3 | 0.0083 | 0.7192 | 1.3723 | 0.2017 | 0.6282 | **0.7577** | 0.1572 | 95.08% |
| **22** | 53.4 | 0.0034 | 0.7771 | 1.4254 | 0.1919 | 0.7583 | 0.7319 | 0.1475 | 95.45% |
| **25** | 52.6 | 0.0011 | 0.7675 | 1.5534 | 0.1967 | 0.6032 | 0.7343 | 0.1460 | 95.34% |
| **28** | 56.5 | 0.0005 | 0.7623 | 1.4992 | 0.1937 | 0.6401 | 0.7404 | 0.1476 | 95.49% |
| **30** | 54.1 | 0.0007 | 0.7858 | 1.5459 | 0.1960 | 0.6741 | 0.7217 | 0.1374 | **95.83%** |

---

## 4. Multi-Task Validation Comparison: Run 7 E1 vs Run 8 E3

| Metric / Objective | Run 7 E1 (Hard Sharing) | Run 8 E3 (Modular MTL) | Delta (E3 vs E1) | Direction |
| :--- | :--- | :--- | :--- | :--- |
| **Best Val Objective** | **0.40036** (Epoch 3) | **0.38888** (Epoch 2) | **-0.01148** | Better Fit (Lower Loss) |
| **BCS Val Loss (at best)** | 0.4384 | 0.4550 | +0.0166 | Comparable |
| **BCS Real MAE (at best)** | 0.1968 | **0.1906** | **-0.0062** | Improved (+3.1% error drop) |
| **BCS Acc@1 (at best)** | 86.90% | **87.80%** | **+0.90 pp** | Improved |
| **Behavior Val Loss (at best)**| 0.4597 | **0.4134** | **-0.0463** | Improved (-10.1% loss) |
| **Behavior Macro-F1 (at best)**| **0.7175** | 0.6725 | -0.0450 | Peak later (0.7577 @ Ep 18) |
| **Re-ID Val Loss (at best)** | 0.3029 | **0.2982** | **-0.0047** | Improved |
| **Re-ID Top-1 Acc (at best)** | **94.19%** | 92.84% | -1.35 pp | Peak later (96.20% @ Ep 20) |

---

## 5. Artifact Registry & Checkpoint Locations

| Artifact | Location | Description |
| :--- | :--- | :--- |
| **Best Model Checkpoint** | `/mtl-checkpoints/mtl_e3_modular/mtl_e3_best.pth` | Best validation model state dict (Epoch 2, ValObj 0.38888) |
| **Latest Model Checkpoint** | `/mtl-checkpoints/mtl_e3_modular/mtl_e3_latest.pth` | Final 30-epoch training checkpoint |
| **Full Metrics JSON** | `artifacts/mtl_e3_metrics.json` | Complete epoch-by-epoch training/validation trajectory |
| **Training Engine Script** | `scripts/train_mtl_e3_modular.py` | PyTorch Modular MTL model and training implementation |
| **Modal Runner Script** | `scripts/modal_train_mtl_e3_modular.py` | Cloud dispatch and volume orchestration wrapper |

---

## 6. Strategic Takeaways & Next Steps

1. **Training Completed Successfully**: Phase 3 Run 8 E3 Modular MTL is fully trained and verified on Modal L40S.
2. **Defensible Validation Results**: The E3 modular configuration achieved a superior composite validation objective (`0.38888` vs `0.40036`) and lower BCS Real MAE (`0.1906` vs `0.1968`) compared to the hard-shared baseline.
3. **Immediate Next Step**: Implement and execute the official **Held-Out Test Evaluation of Run 8 E3** across ScienceDB BCS (7,549 images), Behavior (780 sequences), and SideViewCows2026 Re-ID Protocol A (69 held-out cows) using the exact matched protocol from Run 7 E1 to evaluate held-out generalization.
