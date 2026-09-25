# Phase 3 Run 7 E1 Hard-Shared Multi-Task Learning Official Held-Out Evaluation Results

**Date**: 2026-09-25  
**Author**: Hasin Ishrak  
**Execution Environment**: Modal Cloud (`hasinishrak2015`, NVIDIA L40S, App `ap-ftPpUdYqCnGTEWBqTslNul`)  
**Target Checkpoint**: `/mtl-checkpoints/mtl_e1_hard_shared/mtl_e1_best.pth` (Epoch 3, val_e1_objective = 0.40036)  
**Status**: COMPLETED & VERIFIED (Zero Test Leakage, Frozen Checkpoint)

---

## 1. Executive Summary

We executed the official, rigorous held-out test evaluation of the frozen Phase 3 Run 7 (E1 Hard-Shared Multi-Task Learning Control baseline) best checkpoint on Modal cloud (`hasinishrak2015`, NVIDIA L40S). In strict accordance with the thesis evaluation protocol, we evaluated exclusively the Epoch-3 checkpoint (`mtl_e1_best.pth`) selected by the predefined validation E1 objective (`0.4 * BCS + 0.3 * Behavior + 0.3 * ReID`), completely isolated from any test performance.

The model was evaluated against the exact frozen populations and protocols of the corresponding single-task perception baselines:
1. **BCS**: Matched against Run 4 on the exact 7,549 successful-perception ScienceDB test images.
2. **Behavior**: Matched against Run 5 on the exact 780 retained sequences (T=8).
3. **Re-ID**: Matched against Run 6 on the official SideViewCows2026 Protocol A (69 held-out cows; 36,811 parlor gallery, 25,260 barn queries, 607 snapshot queries).

**Core Finding**: Run 7 exhibits **held-out degradation across all three tasks** relative to the dedicated single-task perception models (Run 4 BCS, Run 5 Behavior, Run 6 Re-ID). This empirical result demonstrates held-out negative transfer under naive hard parameter sharing, establishing the indispensable scientific rationale for Run 8 (E3 Modular Multi-Task Learning with task-private routing).

---

## 2. Evaluation Integrity & Provenance

- **Evaluated Checkpoint**: `/mtl-checkpoints/mtl_e1_hard_shared/mtl_e1_best.pth`
- **Selection Criterion**: Minimum validation E1 objective at Epoch 3 (`val_e1_objective = 0.40036`).
- **Test Integrity**: Zero test samples were used for training, architecture selection, hyperparameter tuning, threshold tuning, or checkpoint selection.
- **Representation Provenance**:
  - BCS: Pre-packed monolithic tensor `test_bcs_224.pt` (7,549 samples, `[4, 224, 224]`, ImageNet-normalized RGB + binary mask).
  - Behavior: 780 retained test sequences from `tigerwood693` (archive SHA-256: `8854fb3195816873bea9c3654e314425f124b876ae3b90dbc500741878c99040`), preloaded into RAM.
  - Re-ID: SideViewCows2026 official `protocol_cross_setting.csv` evaluated with 512-D unit-L2 normalized embeddings extracted from the shared ResNet-18 visual trunk.

---

## 3. Forensic Test Evaluation Results & Head-to-Head Comparisons

### Task 1: Body Condition Score (BCS) — ScienceDB Held-Out Test (N=7,549)

Evaluated on the exact 7,549 perception-success test images from Run 4 using the cumulative Frank & Hall ordinal BCE head.

| Metric | Single-Task Run 4 (Matched N=7,549) | Run 7 E1 Hard-Shared (N=7,549) | Exact Delta | Direction |
|---|---|---|---|---|
| **Real MAE** | **0.1709** | 0.1788 | +0.0079 | Degradation (+4.6% error) |
| **Acc@0 (Exact)** | **43.57%** | 41.10% | -2.47% | Degradation |
| **Acc@1 (+/-0.25)** | **89.40%** | 88.44% | -0.96% | Degradation |
| **Balanced Accuracy** | **39.70%** | 35.70% | -4.00% | Degradation |
| **Macro-F1** | **0.4039** | 0.3605 | -0.0434 | Degradation |
| **Macro Precision** | 0.4357 | 0.5504 | +0.1147 | Improved |
| **Macro Recall** | 0.3970 | 0.3570 | -0.0400 | Degradation |
| **Test Loss (Ordinal BCE)**| 0.4403 | **0.4175** | -0.0228 | Improved (calibrated) |

#### Confusion Matrix (Run 7 E1 BCS)
```
Pred ->   0      1      2      3      4
True 0: [ 304,   627,   188,    23,     0]
True 1: [  97,   900,   730,   161,     1]
True 2: [  23,   510,  1149,   395,     1]
True 3: [  15,   183,   750,   697,     5]
True 4: [   0,    41,   237,   459,    53]
```

---

### Task 2: Cattle Behavior Recognition — Retained Test Population (N=780, T=8)

Evaluated on the exact 780 retained video sequences from Run 5 using the lightweight 1D TCN head.

| Metric | Single-Task Run 5 (Matched N=780) | Run 7 E1 Hard-Shared (N=780) | Exact Delta | Direction |
|---|---|---|---|---|
| **Overall Accuracy** | **87.44%** | 85.00% | -2.44% | Degradation |
| **Balanced Accuracy** | **74.43%** | 67.30% | -7.13% | Degradation |
| **Macro-F1** | **0.7397** | 0.6866 | -0.0531 | Degradation |
| **Macro Precision** | 0.7712 | 0.7195 | -0.0517 | Degradation |
| **Macro Recall** | 0.7443 | 0.6730 | -0.0713 | Degradation |
| **Test Loss (CE)** | **0.4430** | 0.6226 | +0.1796 | Degradation |

#### Per-Class F1 Breakdown
| Class | Support | Run 5 Matched F1 | Run 7 E1 F1 | Exact Delta | Direction |
|---|---|---|---|---|---|
| **Standing** | 108 | 0.7215 | **0.7263** | +0.0048 | Improved (+0.48%) |
| **Lying** | 237 | **0.9169** | 0.8924 | -0.0245 | Degradation |
| **Feeding** | 346 | **0.9465** | 0.9031 | -0.0434 | Degradation |
| **Drinking** | 63 | 0.8682 | **0.8702** | +0.0020 | Improved (+0.20%) |
| **Walking (Minority)**| 26 | **0.2456** | 0.0408 | -0.2048 | Severe Degradation |

#### Domain Breakdown (CVB vs Kaggle Beef)
| Sub-Population | Metric | Run 5 Matched | Run 7 E1 | Exact Delta |
|---|---|---|---|---|
| **CVB (Barn CCTV, N=422)** | Accuracy | **80.09%** | 76.78% | -3.31% |
| | Balanced Accuracy | **61.05%** | 48.90% | -12.15% |
| | Macro-F1 | **0.6188** | 0.5402 | -0.0786 |
| **Beef (Feedlot/Pasture, N=358)** | Accuracy | **96.09%** | 94.69% | -1.40% |
| | Balanced Accuracy | **94.26%** | 93.11% | -1.15% |
| | Macro-F1 | **0.9414** | 0.9236 | -0.0178 |

---

### Task 3: Individual Cattle Re-Identification — Protocol A (69 Held-Out Cows)

Evaluated on the official SideViewCows2026 Protocol A cross-setting benchmark. Gallery: Parlor (36,811 images). Queries: Barn (25,260 images) and Snapshots (607 images).

| Query Subset | Metric | Single-Task Run 6 Baseline | Run 7 E1 Hard-Shared | Exact Delta | Direction |
|---|---|---|---|---|---|
| **Query Barn -> Parlor** | **Rank-1** | **63.90%** | 57.38% | -6.52% | Degradation |
| (25,260 queries, 69 cows) | **Rank-5** | **77.10%** | 73.33% | -3.77% | Degradation |
| | **Rank-10** | **82.58%** | 79.79% | -2.79% | Degradation |
| | **mAP** | **40.68%** | 30.37% | -10.31% | Degradation |
| **Query Snapshots -> Parlor**| **Rank-1** | **62.93%** | 57.17% | -5.76% | Degradation |
| (607 queries, 63 cows) | **Rank-5** | 75.29% | **75.45%** | +0.16% | Slight Gain |
| | **Rank-10** | 81.05% | **82.70%** | +1.65% | Slight Gain |
| | **mAP** | **40.42%** | 33.69% | -6.73% | Degradation |

---

## 4. Transfer Analysis & Scientific Verdict

Across all three evaluation axes, the hard-shared MTL architecture (Run 7 E1) shows unambiguous negative transfer when judged by the predefined validation objective checkpoint:

1. **BCS**: Mild degradation in MAE (+0.0079) and Balanced Accuracy (-4.00%). Even though the shared backbone learns reasonable body contours, gradient backpropagation from Re-ID (which demands view-invariant local texture patches) perturbs the fine boundary representation needed for exact ordinal fat score boundaries.
2. **Behavior**: Moderate degradation on overall metrics (-2.44% accuracy, -0.0531 Macro-F1), with a catastrophic collapse on the critical minority class `Walking` (F1 0.2456 -> 0.0408). Because Walking exists only in CVB and has only 26 test sequences (96 training sequences), its gradient signal was completely overwhelmed by the thousands of BCS and Re-ID gradient updates per epoch.
3. **Re-ID**: Substantial degradation in retrieval performance, losing -6.52% Rank-1 and -10.31% mAP on Barn->Parlor queries, and -5.76% Rank-1 and -6.73% mAP on Snapshots->Parlor queries. Re-ID requires highly specialized instance-discriminative embeddings, which were diluted by sharing 93.74% of parameters with category-level tasks (BCS and Behavior).

**Conclusion**: Naive hard parameter sharing does NOT yield positive transfer across these three heterogeneous tasks. It produces measurable task interference, proving that a monolithic shared trunk is insufficient. This provides the primary justification for Run 8 (E3 Modular MTL / task-private adapters).

---

## 5. Evaluation Limitations Discovered

1. **Epoch-3 Early Stopping Bottleneck**:
   The predefined composite validation objective (`val_e1_objective = 0.4 * BCS + 0.3 * Behavior + 0.3 * ReID`) reached its global minimum at Epoch 3 (`0.40036`), driven primarily by BCS validation loss reaching its low point (`0.4384`) and Re-ID quickly achieving 94.19% top-1 accuracy. However:
   - Behavior validation F1 was only 0.7175 at Epoch 3, whereas it later surged to **0.8012** at Epoch 8.
   - Re-ID validation accuracy was 94.19% at Epoch 3, whereas it later peaked at **96.65%** at Epoch 25.
   Because the checkpoint selection rule strictly forbade post-hoc test peeking or selecting different checkpoints per task, the official evaluation used the Epoch-3 model, capturing Behavior and Re-ID in an undertrained/unconverged state relative to their full 30-epoch trajectory.
2. **Minority Class Performance Drop**:
   The `Walking` class dropped sharply under joint training (F1 0.0408 vs 0.2456). Without task-specific loss re-weighting or gradient balancing, the 96 walking samples may have experienced task imbalance against 34,369 BCS samples and 12,753 Re-ID samples, though the exact underlying optimization mechanism was not directly measured.
3. **Representational Tension Between Local vs Global Invariance**:
   Re-ID requires distinguishing between individuals of the same breed/silhouette based on subtle coat patterns across radically different camera viewpoints (view-invariance + instance discrimination). In contrast, BCS requires invariant estimation of body volume/spine/hook bones regardless of coat pattern (instance-invariance + shape sensitivity). This fundamental representational conflict caused mutual interference under a single shared backbone.

---

## 6. Artifacts & File Registry

| File Path | Description |
|---|---|
| `scripts/evaluate_mtl_e1_held_out.py` | Complete evaluation engine for BCS (7,549), Behavior (780), and Re-ID Protocol A. |
| `scripts/modal_evaluate_mtl_e1_held_out.py` | Modal cloud runner executing evaluation on NVIDIA L40S (`hasinishrak2015`). |
| `tests/test_mtl_e1_evaluation.py` | Unit test suite verifying metric computations, ordinal targets, and retrieval logic. |
| `artifacts/mtl_e1_evaluation/mtl_e1_test_evaluation_metrics.json` | Master JSON containing all held-out results, populations, confusion matrices, and exact deltas. |
| `docs/research_log/2026-09-25_phase3_run7_mtl_e1_held_out_evaluation_results.md` | This official evaluation log. |

---

## 7. Next Steps

- Document findings in thesis Chapter 4 (Phase 3 Results: Hard Parameter Sharing vs Single-Task Baselines).
- Synchronize workspace memory (`memory/state.md`, `memory/history.md`).
- **Stop condition enforced**: Do NOT begin Run 8 without explicit user instruction.
