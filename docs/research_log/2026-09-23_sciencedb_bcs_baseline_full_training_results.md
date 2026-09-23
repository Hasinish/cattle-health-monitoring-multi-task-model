# Research Log: Phase 3 ScienceDB RGB Single-Task BCS Baseline Full 30-Epoch Training Results

**Date**: 2026-09-23  
**Author**: Hasin Ishrak (investigation & execution assisted by Antigravity)  
**Topic**: Full 30-Epoch Baseline Training Completion, Hardware Throughput Optimization, and Final Held-Out Evaluation for Phase 3 Step 4 (ScienceDB RGB BCS Baseline)  
**Status**: COMPLETE & VERIFIED (Canonical Held-Out Test Set Evaluated; All Checkpoints & Metrics Committed to Cloud Volume)  

---

## 1. Executive Summary

Phase 3 Step 4 (Single-Task ScienceDB RGB Body Condition Scoring Baseline) successfully completed its full 30-epoch training schedule on Modal cloud (`tigerwood697`, App `ap-nDiJmX29K9gJnbvGYEr8Zu`) using an ImageNet-pretrained ResNet-18 with Frank & Hall (2001) independent cumulative ordinal BCE loss (`ordinal_bce`). Following empirical throughput diagnostics showing that an NVIDIA L4 with 4 CPUs / 16GB RAM was heavily disk-bound and CPU-starved (~1.5s/batch, projecting to 8.5+ hours and ~$7.00), the execution target was upgraded to an NVIDIA L40S with 8 vCPUs, 32GB RAM, and 8 DataLoader workers. This configuration reduced batch latency by 20x (achieving 10.5–15.9 batches/s once warm-cached in RAM) and completed the entire 30-epoch training plus test evaluation in exactly **33.7 minutes** (2,023.2s total elapsed runtime) for **~$1.15 total cloud compute cost**.

The best model checkpoint was achieved at **Epoch 8** with a Validation Real BCS MAE of **0.1719 BCS units** and Balanced Accuracy of **42.49%**. Evaluating this best checkpoint on the completely held-out, sequence-safe canonical test split (8,040 images across 845 unseen burst groups) yielded:
- **Primary Metric — Real BCS MAE**: **0.1848 BCS units** (scale 3.25 to 4.25, step 0.25).
- **Secondary Metric — Within $\pm$1 Class Accuracy (Acc@1 / Within $\pm$0.25)**: **86.74%** (6,974 / 8,040 test images).
- **Exact Accuracy (Acc@0)**: **41.63%** (3,347 / 8,040 exact matches across 5 classes).
- **Balanced Accuracy**: **40.41%** (Macro-average recall across all 5 classes; 2x random baseline of 20%).
- **Macro-F1 Score**: **0.4110** (Macro Precision: 0.4360, Macro Recall: 0.4041).

All model weights (`bcs_baseline_best.pth`, `bcs_baseline_latest.pth`), evaluation metrics (`bcs_baseline_metrics.json`), and summaries are permanently stored and committed on Modal persistent volume `sciencedb-checkpoints` and synced locally to `artifacts/bcs_baseline/`.

---

## 2. Context & Motivation

Phase 3 Step 4 establishes the fundamental single-task RGB baseline for Body Condition Scoring (BCS) in the multi-task research roadmap. The central thesis question investigates whether cattle-centered visual representations (localization, soft masks, pose/anatomy, viewpoint) reduce shortcut learning and improve robustness over generic RGB representations. To answer this question, a rigorous, non-leaking, well-tuned RGB baseline must first be established.

Previous attempts had uncovered:
1. Extraction corruption on Modal volume `sciencedb-data` (1,753 zero-byte stub files causing `PIL.UnidentifiedImageError`), which was completely repaired and exhaustively verified (53,566/53,566 images decoded).
2. A severe I/O bottleneck on low-tier hardware: 1024x576 JPEG decoding over network volumes on 4 CPUs caused the GPU to stall, consuming 1.5 seconds per batch.
3. The need to verify that ordinal loss formulation (`ordinal_bce`) produces clinically meaningful predictions that respect the ordering of body condition scores (3.25 < 3.50 < 3.75 < 4.00 < 4.25).

---

## 3. Forensic Findings & Experimental Results

### 3.1 Hardware Throughput Comparison (L4 vs L40S)

| Parameter | NVIDIA L4 (Aborted Run `ap-QYGwdrz5jKGHSAq77okZ6b`) | NVIDIA L40S (Completed Run `ap-nDiJmX29K9gJnbvGYEr8Zu`) |
| :--- | :--- | :--- |
| **GPU / VRAM** | NVIDIA L4 (24GB VRAM) | NVIDIA L40S (48GB VRAM) |
| **Host Resources** | 4.0 vCPU, 16GB RAM | 8.0 vCPU, 32GB RAM |
| **DataLoader Workers**| 4 workers | 8 workers |
| **Cold Throughput** | ~0.67 batch/s (1.50s / batch) | ~2.88 batch/s (0.35s / batch) |
| **Warm Throughput** | Stalled on RAM / disk thrashing | **10.50 – 15.91 batch/s (0.06 – 0.09s / batch)** |
| **Train Epoch Time** | ~14.5 minutes | **~42 to 46 seconds** |
| **Total 30-Epoch Time**| ~8.5 to 9.0 hours (projected) | **33.7 minutes (actual)** |
| **Total Cloud Cost** | ~$6.80 – $7.20 (projected) | **~$1.15 (actual)** |

*Root Cause*: ScienceDB's full dataset is ~30GB of 1024x576 JPEG images. With 32GB host RAM and 8 vCPUs, the entire active dataset fits in the Linux OS page cache after the initial epoch, dropping disk I/O latency to zero and allowing 8 parallel workers to saturate the L40S GPU.

---

### 3.2 Training Trajectory & Convergence

Training proceeded with AdamW (`lr=1e-4`, `weight_decay=1e-4`, batch size 64) and a Cosine Annealing learning rate schedule decaying to `1e-6` at Epoch 30:

| Epoch | Duration | Train Loss | Val Loss | Val Real MAE | Val Bal Acc | Val Macro-F1 | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 147.3s | 0.3652 | 0.4252 | 0.1851 | 35.91% | 0.3648 | [BEST] |
| 2 | 56.2s | 0.2442 | 0.4747 | 0.1864 | 37.65% | 0.3702 | |
| 3 | 62.1s | 0.1764 | 0.5562 | 0.1853 | 38.08% | 0.3770 | |
| 4 | 53.3s | 0.1301 | 0.5449 | 0.1793 | 39.91% | 0.4059 | [BEST] |
| 5 | 58.4s | 0.0951 | 0.6408 | 0.1900 | 38.06% | 0.3917 | |
| 6 | 58.2s | 0.0725 | 0.6445 | 0.1764 | 40.69% | 0.4223 | [BEST] |
| 7 | 64.9s | 0.0578 | 0.6983 | 0.1841 | 38.83% | 0.3957 | |
| **8** | **54.9s** | **0.0486** | **0.6781** | **0.1719** | **42.49%** | **0.4366** | **[GLOBAL BEST]** |
| 9 | 68.5s | 0.0375 | 0.8285 | 0.1924 | 37.30% | 0.3766 | |
| 10 | 79.0s | 0.0317 | 0.8224 | 0.1825 | 37.34% | 0.3729 | |
| 15 | 60.3s | 0.0125 | 0.8872 | 0.1802 | 39.51% | 0.3978 | |
| 20 | 56.7s | 0.0051 | 0.9412 | 0.1784 | 39.12% | 0.3955 | |
| 25 | 56.8s | 0.0022 | 1.0114 | 0.1781 | 39.20% | 0.3961 | |
| 30 | 56.5s | 0.0014 | 1.0317 | 0.1777 | 39.04% | 0.3969 | |

*Key Phenomenon*: Peak validation performance occurred at **Epoch 8** (`Val Real MAE: 0.1719 BCS`). While training loss steadily decreased from `0.0486` down to `0.0014`, validation loss plateaued and gradually expanded due to mild feature memorization. Because `bcs_baseline_best.pth` was preserved at the global minimum, overfitting in epochs 16–30 had zero negative impact on the final model.

---

### 3.3 Final Held-Out Test Set Evaluation (8,040 Images)

The canonical test split was evaluated exclusively once using `bcs_baseline_best.pth`:

| Evaluation Metric | Test Set Score | Interpretation & Benchmarks |
| :--- | :--- | :--- |
| **Real BCS MAE (Primary)** | **0.1848 BCS units** | Average physical error is well below a single Ferguson sub-grade (0.25). |
| **Within $\pm$0.25 (Acc@1 / $\pm$1 Class)** | **86.74%** | **6,974 out of 8,040 test images** are either exact or within $\pm$0.25 BCS. |
| **Exact Accuracy (Acc@0)** | **41.63%** | 3,347 / 8,040 exact predictions across 5 ordinal classes. |
| **Balanced Accuracy** | **40.41%** | Unweighted average recall across all 5 classes; 2.02x above random guessing (20%). |
| **Macro-F1 Score** | **0.4110** | Macro Precision: 0.4360, Macro Recall: 0.4041. |
| **Class Index MAE** | **0.7391 steps** | Literature parity metric ($0.7391 \times 0.25 = 0.1848$). |
| **Test Loss** | **0.7515** | Cross-entropy ordinal binary threshold cumulative loss. |

---

### 3.4 Per-Class Granular Performance

| Class Index | Real BCS Class | Test Support | Exact Accuracy | Precision | Recall | F1 Score | Real BCS MAE | Within $\pm$1 Class Acc |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | **3.25** | 1,209 | 37.72% | 0.5455 | 0.3772 | 0.4460 | 0.2333 | 77.01% (931/1209) |
| 1 | **3.50** | 1,999 | 47.47% | 0.4460 | 0.4747 | 0.4599 | 0.1683 | 86.34% (1726/1999) |
| 2 | **3.75** | 2,232 | 33.24% | 0.3684 | 0.3324 | 0.3495 | 0.1873 | 91.85% (2050/2232) |
| 3 | **4.00** | 1,750 | 54.34% | 0.3812 | 0.5434 | 0.4481 | 0.1406 | 90.57% (1585/1750) |
| 4 | **4.25** | 850 | 29.29% | 0.4392 | 0.2929 | 0.3514 | 0.2388 | 80.24% (682/850) |
| **Overall** | — | **8,040** | **41.63%** | **0.4360** | **0.4041** | **0.4110** | **0.1848** | **86.74%** |

---

### 3.5 Test Confusion Matrix (Rows = Ground Truth, Columns = Predicted)

```text
               Pred 3.25   Pred 3.50   Pred 3.75   Pred 4.00   Pred 4.25     Total (GT)
GT 3.25 [0]       456         475         192          75          11         1,209
GT 3.50 [1]       268         949         509         250          23         1,999
GT 3.75 [2]        89         522         742         786          93         2,232
GT 4.00 [3]        20         145         443         951         191         1,750
GT 4.25 [4]         3          37         128         433         249           850
Total Pred        836       2,128       2,014       2,495         567         8,040
```

*Key Confusion Observations*:
1. **Strongly Tridiagonal Structure**: Errors are heavily clustered directly adjacent to the main diagonal (e.g. GT 3.50 misclassified as 3.25 or 3.75; GT 4.00 misclassified as 3.75 or 4.25).
2. **Minimal Severe Hallucinations**: Extreme off-diagonal errors are virtually nonexistent (only 11 cows of BCS 3.25 predicted as 4.25, and only 3 cows of BCS 4.25 predicted as 3.25).
3. **No Class Collapse**: Unlike unweighted cross-entropy which frequently collapses rare classes (3.25 and 4.25) to zero predictions, the Frank & Hall ordinal cumulative formulation successfully predicted 836 samples for 3.25 and 567 samples for 4.25, maintaining healthy recall across all grades.

---

## 4. Architectural Decisions & Baseline Role in Thesis

1. **Step 4 RGB Baseline Established**:
   The single-task RGB ResNet-18 baseline is officially established and certified with a canonical test MAE of **0.1848 BCS units** and **86.74% within $\pm$1 class accuracy**.
2. **Primary Benchmark Target for Step 6**:
   Any subsequent cattle-centered visual representation model (e.g., localized cow crop, soft-masked cow, pose-guided anatomical feature pooling, or viewpoint-conditioned multi-task network) must outperform **0.1848 Real BCS MAE** and **86.74% Acc@1** on this identical test split to prove that representation engineering delivers genuine inductive bias.
3. **Hardware Policy Reaffirmation**:
   Modal cloud profiles using NVIDIA L40S (8 vCPUs, 32GB RAM) provide optimal cost-to-speed efficiency for image datasets over persistent network volumes. Training costs ~$1.15 for 30 epochs while finishing in 33 minutes.

---

## 5. Artifacts & File Registry

| File / Artifact Path | Description & Purpose |
| :--- | :--- |
| `scripts/train_sciencedb_bcs_baseline.py` | Core training script with ordinal BCE, ResNet-18, progress streaming, path resolver, and checkpoint commit hooks. |
| `scripts/modal_train_sciencedb_bcs.py` | Modal wrapper configured for L40S, 8 CPUs, 32GB RAM, persistent volumes `/data` and `/checkpoints`. |
| `artifacts/bcs_baseline/bcs_baseline_metrics.json` | Complete machine-readable results JSON containing git info, split hashes, full 30-epoch history, per-class metrics, and confusion matrix. |
| `artifacts/bcs_baseline/bcs_baseline_30epoch_summary.md` | Formal human-readable summary report downloaded from cloud volume. |
| Modal Volume `sciencedb-checkpoints/bcs_baseline_best.pth` | Canonical best model weights (Epoch 8 checkpoint, 44.7 MB). |
| Modal Volume `sciencedb-checkpoints/bcs_baseline_latest.pth` | Final Epoch 30 model weights (44.7 MB). |

---

## 6. Next Steps

1. **Update `memory/state.md`**: Mark Phase 3 Step 4 single-task ScienceDB RGB BCS baseline full training as 100% COMPLETE & VERIFIED.
2. **Step 2 Qualitative Visual Reviews**: Schedule human review of the Kaggle Beef perception contact sheets (`beef_sam_prompt_rescue`, `beef_A4_A5_fresh40`, `beef_rtdetr_failure_fallback`).
3. **Step 5 / Behavior Baseline Planning**: Prepare the single-task dense-video Behavior baseline pipeline on the canonical CVB + Kaggle Beef protocol (`datasets/behavior/cvb_beef/`).
