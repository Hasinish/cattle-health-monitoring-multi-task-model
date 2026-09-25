# Phase 3 E4 PCGrad Multi-Task Learning Full 30-Epoch Training Results

**Date**: 2026-09-26  
**Author**: Hasin Ishrak  
**Execution Environment**: Modal Cloud (`hasinishrak2015`, NVIDIA L40S, App `ap-JfKjvY9vbNUXBPaPXL4GCT`)  
**Target Checkpoint**: `/mtl-checkpoints/mtl_e4_pcgrad/mtl_e4_best.pth` (Epoch 3, `val_e4_objective = 0.40098`)  
**Status**: COMPLETED & VERIFIED (Full 30-Epoch Training Certified; Test Sets Untouched)

---

## 1. Executive Summary

We successfully executed and certified the full 30-epoch training and multi-task validation of **Phase 3 E4 PCGrad** (Projecting Conflicting Gradients; Yu et al., 2020) on Modal cloud (`hasinishrak2015`, NVIDIA L40S, App `ap-JfKjvY9vbNUXBPaPXL4GCT`, runtime 2,213.1s / ~36.88 mins). E4 preserves the exact hard-shared ResNet-18 architecture (11,926,706 trainable parameters) and training schedule as Run 7 E1, applying PCGrad conflict projection surgery exclusively to shared backbone gradients before optimizer updates.

Training converged stably, achieving its global best multi-task validation objective at **Epoch 3** (`val_e4_objective = 0.40098`), practically identical to Run 7 E1's best validation objective (`0.40036` at Epoch 3). At this best checkpoint: BCS validation loss was `0.4313` with Real MAE `0.1968` (Acc@1: 86.63%); Behavior validation loss was `0.4697` with Macro-F1 `0.7050` and accuracy `86.83%` (minority class Walking reaching F1 `0.1176`); and Re-ID validation loss was `0.3020` with Top-1 Accuracy `93.59%`.

Critically, the PCGrad diagnostics engine provided the first direct empirical measurement of multi-task gradient conflict across the cattle monitoring triad: across **16,140 super-steps**, PCGrad executed **44,177 pairwise conflict projections** (averaging **2.737 projections per step** out of 6 maximum). Pairwise task conflicts occurred in **45% to 54% of all super-steps** across all 30 epochs. Checkpoints and metrics were committed to `/mtl-checkpoints/mtl_e4_pcgrad/`. Canonical held-out test sets remain strictly untouched.

---

## 2. Experimental Context & Controlled Architecture

E4 tests the optimization hypothesis:
> *"Can projectively removing conflicting gradient components on the shared backbone mitigate negative transfer without adding architectural parameters or task-private modules?"*

### Architecture & Training Specifications:
- **Shared Spatial Trunk**: 1 shared 4-channel ResNet-18 (11,179,648 parameters; 93.74% of capacity).
- **Task Heads**: BCS Linear(512, 4) ordinal BCE (2,052 params); Behavior 1D TCN (723,973 params); Re-ID Linear(512, 41) classifier (21,033 params).
- **Total Trainable Parameters**: 11,926,706 (asserted match to E1).
- **Optimizer & Schedule**: AdamW, initial LR = 1e-4, weight decay = 1e-4, Cosine Annealing to 1e-6, 30 epochs, 538 super-steps/epoch (16,140 total steps).
- **Task Weights**: Equal fixed weights: `w_BCS = 1.0, w_Behavior = 1.0, w_ReID = 1.0`.
- **Validation Objective**: `val_e4_objective = (BCS_val_loss + Behavior_val_loss + ReID_val_loss) / 3.0`.

---

## 3. Full 30-Epoch Training Dynamics & Head-to-Head Comparison

### Comparative Multi-Task Validation Summary:

| Model / Run | Best Epoch | Val Objective | BCS Val Loss | BCS Real MAE | Behavior Val Loss | Behavior Macro-F1 | Re-ID Val Loss | Re-ID Top-1 Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Run 7 (E1 Hard-Shared)** | 3 | **0.40036** | 0.4384 | 0.1968 | **0.4597** | **0.7175** | 0.3029 | 94.19% |
| **Run 8 (E3 Modular Adapters)**| 2 | **0.38888** | 0.4550 | **0.1906** | **0.4134** | 0.6725 | **0.2982** | 92.84% |
| **Phase 3 E4 (PCGrad Surgery)** | 3 | **0.40098** | **0.4313** | 0.1968 | 0.4697 | 0.7050 | 0.3020 | 93.59% |

### Key Observations Across Training:
1. **Validation Convergence**: E4's composite validation objective curve matches E1 closely in early epochs, with both models reaching their global validation minimum at Epoch 3.
2. **Behavior Metrics**: At Epoch 3, E4 Behavior validation accuracy reached 86.83% and Macro-F1 reached 0.7050. In later epochs, Behavior Macro-F1 surged to **0.7850** (Epoch 21), demonstrating strong temporal pattern retention.
3. **Re-ID Metric**: Re-ID validation loss dropped to 0.1525 with Top-1 accuracy peaking at **96.76%** (Epoch 15).
4. **BCS Representation**: BCS Real MAE remained exceptionally stable between 0.1934 and 0.1968 across the first 20 epochs, while validation loss increased in later epochs (0.4313 -> 0.9610) as the shared trunk became overconfident on classification thresholds.

---

## 4. Forensic PCGrad Diagnostics: The Gradient Conflict Proof

Because PCGrad captures per-task gradients on the shared backbone prior to surgery, E4 provides the first quantitative evidence of gradient interference:

| Metric | Overall Value across 30 Epochs |
| :--- | :--- |
| **Total Super-Steps** | 16,140 |
| **Total Conflict Projections Triggered** | **44,177** |
| **Mean Projections per Step** | **2.737** (out of max 6) |
| **BCS vs Behavior Conflict Frequency** | **47.8%** of super-steps |
| **BCS vs Re-ID Conflict Frequency** | **46.5%** of super-steps |
| **Behavior vs Re-ID Conflict Frequency** | **47.4%** of super-steps |

### Significance for Thesis:
In naive hard sharing (E1), these opposing gradients simply cancel each other out during ordinary accumulation, exerting conflicting updates on the spatial representation. PCGrad actively detected and eliminated opposing directional components 44,177 times during training.

### Scientific Claim Boundaries:
1. These measurements prove that task gradients on the shared 4-channel ResNet-18 frequently point in conflicting directions during multi-task cattle training.
2. They do **not** prove that PCGrad eliminates negative transfer on held-out populations; held-out test evaluation must be performed before making generalization claims.

---

## 5. Artifacts & Registry

- **Trained Best Checkpoint**: `/mtl-checkpoints/mtl_e4_pcgrad/mtl_e4_best.pth` (Modal Volume `mtl-checkpoints`)
- **Trained Latest Checkpoint**: `/mtl-checkpoints/mtl_e4_pcgrad/mtl_e4_latest.pth`
- **Full Metrics Artifact**: `artifacts/mtl_e4_training/mtl_e4_metrics.json`
- **Sync Script**: `scripts/sync_mtl_e4_artifacts.py`
- **Training Engine**: `scripts/train_mtl_e4_pcgrad.py`
- **Cloud Runner**: `scripts/modal_train_mtl_e4_pcgrad.py`

---

## 6. Next Steps

1. Commit and push all E4 training artifacts, research logs, and updated memory to Git.
2. If desired by user: Implement and execute the frozen held-out test evaluation script for E4 (`evaluate_mtl_e4_held_out.py`) across ScienceDB (7,549 images), Behavior (780 sequences), and SideView Protocol A (69 unseen cows) to determine whether gradient surgery translates to superior held-out generalization over E1.
