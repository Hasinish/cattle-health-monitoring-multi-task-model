# Research Log: Phase 3 ScienceDB RGB Single-Task BCS Baseline Pipeline

**Date**: 2026-09-23  
**Author**: Autonomous AI Pair Programmer (Thesis Research Workspace)  
**Topic**: Clean Phase 3 ScienceDB RGB Single-Task BCS Baseline Pipeline, CORAL Ordinal Regression, Real BCS Metrics Engine, Checkpoint Determinism, and Smoke Test on Local GTX 1050 Ti  

---

## 1. Executive Summary

We designed, implemented, and empirically validated the canonical Phase 3 ScienceDB RGB single-task Body Condition Scoring (BCS) baseline pipeline (`scripts/train_sciencedb_bcs_baseline.py`). The pipeline strictly uses the canonical leakage-safe 5,653-burst-group train/val/test split (`datasets/bcs/sciencedb/`), explicitly rejecting outdated Phase 2 split assumptions and formally classifying ScienceDB as **burst-group-disjoint / sequence-safe** (true biological cow IDs are not released by the publisher).

The architecture couples an ImageNet-pretrained ResNet-18 backbone with a CORAL (Consistent Rank Logits) ordinal regression head (or linear classification head). We implemented a rigorous metrics engine that converts categorical indices (0 to 4) directly to true physiological BCS scores (3.25 to 4.25, step 0.25) to report authentic physiological MAE alongside Balanced Accuracy, Macro-F1, Acc@0, Acc@1, and per-class breakdowns. Checkpoint save/resume was verified to be 100% bit-identical. A 2-epoch smoke test was executed on the local GTX 1050 Ti (exiting with code 0 in ~16 seconds), confirming full pipeline stability without running full 30-epoch training.

---

## 2. Context & Motivation

Under Phase 3 Step 4 of our canonical roadmap (`phase3_canonical_roadmap.md`), establishing a trusted, uncorrupted single-task RGB baseline is mandatory before introducing cattle-centered visual representations (localization crops, segmentation masks, pose keypoints, or viewpoint conditioning).

Previous Phase 2 BCS training scripts relied on `sciencedb_bcs_index.csv`, which suffered from severe stereo sequence leakage (94.64% cross-split contamination) and falsely treated passage identifiers as individual cows. Furthermore, legacy metrics reported class-index MAE (0 to 4) rather than physical BCS units, leading to misleading error scales.

This task resolves those legacy flaws by establishing a clean, reproducible baseline pipeline that:
1. Operates exclusively on the canonical 5,653-burst-group stratified split (`train.csv`: 37,045 images, `val.csv`: 8,481 images, `test.csv`: 8,040 images).
2. Explicitly enforces the burst-group-disjoint protocol and disclaims cow-disjointness.
3. Implements true physiological BCS MAE calculation across the 3.25 to 4.25 range.
4. Preserves full provenance (Git commit, split SHA-256 hashes, seed, hyperparameters).
5. Validates checkpoint save/resume determinism.
6. Conducts low-cost smoke testing on local hardware before dispatching full training runs to the RTX 5090 or cloud instances.

---

## 3. Forensic Findings & Implementation Details

### 3.1 Split Provenance & Invariant Enforcement
The pipeline directly consumes the repaired canonical splits generated on 2026-09-20:
- `datasets/bcs/sciencedb/train.csv` (SHA256: `3b4694406201b1b046a7828e6789965b75a4dd0a0fead0cbfa5160cbbe0e7161`, 37,045 samples across 3,957 burst groups)
- `datasets/bcs/sciencedb/val.csv` (SHA256: `085601dbf6cbf9cf0f69a84a6a575a40a8d67a9a12c40c810d7a0c863dcf8c0a`, 8,481 samples across 848 burst groups)
- `datasets/bcs/sciencedb/test.csv` (SHA256: `1f7e02b74b1ef40d4f13ee1c7f4625b1b4632cb51a1d13f9c6d1d2b8b981d7f1`, 8,040 samples across 848 burst groups)
- **Grouping Protocol**: 5,653 connected burst groups clustered via perceptual hashing and Disjoint Set Union (DSU) with normalized pixel MAE <= 5.0. Zero passage overlap exists across partitions.
- **Scientific Classification**: Burst-group-disjoint / sequence-safe. **NOT cow-disjoint**.

### 3.2 Architecture & Head Formulation
- **Backbone**: ImageNet-1k pretrained ResNet-18 (`torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`).
- **Feature Pooling**: Global Average Pooling (512-dimensional embedding).
- **Head Options**:
  1. `coral` (Default): Rank-consistent ordinal regression predicting 4 binary logits corresponding to cumulative thresholds: P(y > 3.25), P(y > 3.50), P(y > 3.75), P(y > 4.00). Loss is cumulative binary cross-entropy. Discrete prediction: `sum(sigmoid(logits) > 0.5)`.
  2. `linear`: 5-way multi-class classification with Cross-Entropy Loss. Discrete prediction: `argmax(logits)`.

### 3.3 True Physiological BCS Metrics Engine
In animal husbandry and veterinary scoring (Ferguson et al., 1994), BCS error must be reported in actual BCS points, not integer class steps:
- Conversion: `real_bcs = 3.25 + class_idx * 0.25`
- Primary Metric: Real BCS MAE = mean(|y_true_real - y_pred_real|)
- Secondary Metrics:
  - Class-Index MAE = mean(|y_true_idx - y_pred_idx|) = Real BCS MAE / 0.25
  - Exact Accuracy (Acc@0) = % exact class agreement
  - Within-0.25 Tolerance (Acc@1) = % predictions within +/- 0.25 BCS units (adjacent ordinal class)
  - Balanced Accuracy = unweighted mean of class-specific recalls
  - Macro-F1 = unweighted mean of class-specific F1 scores
  - Per-class breakdown: Support, Precision, Recall, F1, Per-Class Real MAE, 5x5 Confusion Matrix.

### 3.4 Checkpoint Save/Resume Determinism Check
The pipeline includes an automated round-trip test (`test_checkpoint_roundtrip`):
1. Captures full state dictionary payload: `epoch`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `best_val_mae`, `config`, `git_info`, `split_provenance`, `history`.
2. Restores into a newly instantiated model on the target execution device.
3. Verifies that all parameter tensors are bit-identical (`torch.equal`).
4. Verifies forward pass output parity on a random tensor (`torch.allclose(out1, out2, atol=1e-6)`).
- **Result**: PASSED with 100% bit-identical parity.

### 3.5 GTX 1050 Ti Smoke Test Execution Evidence
- **Command**: `python scripts/train_sciencedb_bcs_baseline.py --smoke --epochs 2 --max_samples 250 --test_save_resume`
- **Execution Device**: NVIDIA GeForce GTX 1050 Ti (4.0 GB VRAM)
- **Samples Used**: 250 train, 125 val, 125 test (stratified across the 5 classes, 25 samples/class)
- **Timing**: Epoch 1: 7.1s; Epoch 2: 4.9s (Total smoke run: ~16 seconds)
- **Convergence**:
  - Epoch 1: Train Loss 0.6773 | Val Loss 0.7464 | Val Real MAE: 0.4300 BCS | Bal Acc: 14.40%
  - Epoch 2: Train Loss 0.4622 | Val Loss 0.7507 | Val Real MAE: 0.4200 BCS | Bal Acc: 18.40%
- **Final Test Set Evaluation (Best Epoch 2 Checkpoint)**:
  - Real BCS MAE: **0.3700 BCS units**
  - Class Index MAE: 1.4800 steps
  - Exact Accuracy: 20.00%
  - Within-0.25 (Acc@1): 53.60%
  - Balanced Accuracy: 20.00%
  - Macro-F1: 0.1921
- **Resume Verification**:
  - Resuming via `--resume artifacts/bcs_baseline/bcs_baseline_latest.pth --epochs 3` successfully restored Epoch 2 state, trained Epoch 3 (Train Loss 0.2824, Val Real MAE 0.3720 BCS), and improved test Real BCS MAE to 0.3480 BCS units (Acc@1: 59.20%).

---

## 4. Architectural Decisions & Action Plan

1. **Step 4 Ready for Full Execution on Research PC / Modal**:
   - The script is fully functional, parameterized, and self-contained.
   - For full training (30 epochs on 37k images), execution should be dispatched to the BRACU Lab Research PC (RTX 5090) or Modal GPU, avoiding local laptop GPU wear.
2. **Provenance Sealed**:
   - Split metadata and Git commit hashes are embedded in all saved checkpoints and summary JSON files.
3. **No Unwarranted Cross-Task Changes**:
   - Viewpoint work (Step 2.4), perception caching (Step 3), and canonical roadmap (`phase3_canonical_roadmap.md`) remain untouched.

---

## 5. Artifacts & File Registry

| Artifact | File Path | Purpose / Description |
| :--- | :--- | :--- |
| **BCS Baseline Script** | `scripts/train_sciencedb_bcs_baseline.py` | Complete training, evaluation, checkpointing, and smoke-testing pipeline. |
| **Smoke Metrics JSON** | `artifacts/bcs_baseline/bcs_baseline_metrics.json` | Structured JSON containing metadata, provenance, configs, history, and test metrics. |
| **Smoke Summary Markdown** | `artifacts/bcs_baseline/bcs_baseline_smoke_summary.md` | Human-readable markdown report of smoke test performance and per-class metrics. |
| **Research Log Entry** | `docs/research_log/2026-09-23_sciencedb_bcs_baseline_pipeline.md` | This research log document. |

---

## 6. Next Steps

- [x] Implement Phase 3 ScienceDB RGB single-task BCS baseline pipeline.
- [x] Integrate real physiological BCS MAE calculation (scale 3.25 to 4.25).
- [x] Embed Git commit and split provenance tracking.
- [x] Verify checkpoint save/resume bit-identical determinism.
- [x] Execute 2-epoch smoke test on local GTX 1050 Ti.
- [x] Author research log and update research log index.
- [ ] Update `memory/state.md` and `memory/history.md`.
- [ ] Stage, commit, and push changes to `origin/main`.
- [ ] Mirror customizations/memory to `D:\custom-antigravity`.
