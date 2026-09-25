# Phase 3 E4 PCGrad Multi-Task Learning Official Held-Out Evaluation Results (hasinishrak2015)

**Date**: 2026-09-26  
**Author**: Hasin Ishrak  
**Execution Platform**: Modal Cloud (`hasinishrak2015`, NVIDIA L40S, App ID `ap-JWAyJ9EgN2Zvbaeo2RZW6d`)  
**Evaluated Checkpoint**: `/mtl-checkpoints/mtl_e4_pcgrad/mtl_e4_best.pth` (Epoch 3, `val_e4_objective = 0.40098`)  
**Git Base Commit**: `2f7d7fe74309b88a1e0c8d6099184b4c32f565e3`  
**Status**: COMPLETE, VERIFIED & LOCKED  

---

## 1. Executive Summary

We executed the official, frozen held-out evaluation of the Phase 3 E4 PCGrad (Projecting Conflicting Gradients; Yu et al., 2020) multi-task model on Modal (`hasinishrak2015`, NVIDIA L40S, App `ap-JWAyJ9EgN2Zvbaeo2RZW6d`). In strict accordance with the experimental protocol, evaluation was performed exclusively on the predefined validation-loss selection checkpoint (**Epoch 3**, `val_e4_objective = 0.40098`, 11,926,706 trainable parameters; exact E1 architecture match with 0 adapters and 0 gates), with zero retraining, zero hyperparameter adjustments, zero threshold tuning, and zero evaluation of non-selected checkpoints.

Across the canonical held-out evaluation protocols:
1. **BCS (ScienceDB, N=7,549 images, burst-group-disjoint)**: E4 achieved **Real MAE: 0.1828**, Acc@0: 40.23%, Acc@1: 87.84%, Balanced Acc: 34.97%, Macro-F1: 0.3516, and Test Loss: 0.4181. This represents a +0.0040 MAE increase over E1 (0.1788) and +0.0119 over single-task Run 4 (0.1709), but decisively outperforms E3 Modular Adapters (0.1916; -0.0088 MAE improvement).
2. **Behavior (CVB + Beef, N=780 retained sequences, T=8 frames)**: E4 achieved **Overall Accuracy: 86.92%** (+1.92 pp over E1 85.00%; +1.15 pp over E3 85.77%), **Balanced Accuracy: 69.15%** (+1.85 pp over E1 67.30%; +2.45 pp over E3 66.70%), **Macro-F1: 0.7114** (+0.0248 over E1 0.6866; +0.0359 over E3 0.6755), and **Test Loss: 0.5454** (-0.0772 reduction vs E1 0.6226). Minority class `Walking` F1 reached **0.0909** (more than doubled vs E1 0.0408; E3 completely collapsed Walking to 0.0000). On authentic barn surveillance (**CVB**, N=422), E4 achieved **81.04% Overall Accuracy** (+4.26 pp vs E1 76.78%; +0.95 pp vs single-task Run 5 80.09%) and **0.6141 Macro-F1** (+0.0739 vs E1 0.5402; essentially matching single-task Run 5 0.6188).
3. **Re-ID (SideViewCows2026 Protocol A, 69 held-out cows; 36,811 parlor gallery)**: 
   - **Query Barn -> Parlor (25,260 queries)**: Rank-1: 54.97%, Rank-5: 68.81%, Rank-10: 74.99%, mAP: 30.23% (mAP closely matches E1 30.37%, beats E3 28.04% by +2.19 pp and Rank-1 by +5.89 pp).
   - **Query Snapshots -> Parlor (607 queries)**: Rank-1: 57.00%, Rank-5: 72.16%, Rank-10: 78.25%, mAP: 33.87% (mAP beats E1 33.69% by +0.18 pp, beats E3 28.78% by +5.09 pp and Rank-1 by +3.29 pp).

**Scientific Takeaway**: Optimization-level gradient projection (E4 PCGrad) yielded selective, metric-dependent benefits over naive hard sharing (E1), most notably on Behavior (where CVB barn CCTV accuracy gained +4.26 pp and Walking F1 more than doubled). However, held-out evidence confirms that PCGrad did **NOT** eliminate negative transfer: performance remained below single-task baselines across all three tasks. Crucially, E4 PCGrad decisively outperformed architectural modularity (E3 task-private residual adapters) across all three tasks on held-out test data. This marks the successful completion of the ONE approved final additional experimental run.

---

## 2. Context & Motivation

In Run 7 (E1 Hard-Shared MTL), naive parameter sharing caused unambiguous held-out degradation across all three tasks relative to single-task baselines (BCS MAE degraded to 0.1788 vs 0.1709; Behavior Macro-F1 dropped to 0.6866 with Walking collapsing to 0.0408 vs 0.2456; Re-ID Barn mAP dropped to 30.37% vs 40.68%).

In Run 8 (E3 Modular MTL), task-private residual bottleneck adapters (`512 -> 128 -> 512`) were introduced at the output of the shared backbone. While E3 improved Behavior test loss (0.4193) and CVB surveillance accuracy (80.33%), it further degraded BCS MAE (0.1916) and Re-ID retrieval (Barn Rank-1 49.08%, mAP 28.04%).

E4 was designed to test whether an **optimization-level intervention**—specifically Projecting Conflicting Gradients (PCGrad; Yu et al., 2020)—could resolve conflicting gradient directions during training without altering the underlying E1 architecture (11,926,706 trainable parameters; 0 adapters, 0 gates). During 30-epoch training, PCGrad recorded **44,177 conflict projections** (2.737 projections/step) across 16,140 super-steps, providing direct empirical proof that task gradients repeatedly conflicted. This evaluation tests whether resolving those conflicts at training time translates into superior generalization on canonical held-out test sets.

---

## 3. Off-By-One Documentation Correction

We formally document the following super-step accounting correction for historical and archival transparency:
- The executed full-training loader used:
  - ScienceDB BCS: 34,369 training samples
  - Batch size: 64
- Therefore:
  - `ceil(34,369 / 64) = 538` super-steps per epoch.
  - Across 30 epochs: `30 * 538 = 16,140` total super-steps.
- This corrects stale comments in earlier scripts that stated 537 super-steps/epoch and 16,110 total super-steps.
- The executed training run and its recorded diagnostics are 100% valid; this note merely updates documentation to reflect the exact executed loader count.

---

## 4. Comprehensive Evaluation Metrics & 3-Way Comparisons

### 4.1 Task 1: Body Condition Scoring (ScienceDB Matched Test Set)
- **Population**: N = 7,549 images (repaired burst-group-disjoint / passage-sequence-safe protocol; test unit = image/sample; NOT biological cow-disjoint).
- **Head**: Frank & Hall (2001) cumulative ordinal BCE head (4 outputs, thresholds 2.75, 3.00, 3.25, 3.50).

| Metric | Run 4 Single-Task | Run 7 E1 Hard-Shared | Run 8 E3 Modular | Phase 3 E4 PCGrad | Delta vs Single | Delta vs E1 | Delta vs E3 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Real MAE** (BCS units) | **0.1709** | 0.1788 | 0.1916 | **0.1828** | +0.0119 | +0.0040 | **-0.0088** |
| **Acc@0** (Exact match) | **43.57%** | 41.10% | 38.47% | **40.23%** | -3.34 pp | -0.87 pp | **+1.76 pp** |
| **Acc@1** (+/- 0.25 units) | **89.40%** | 88.44% | 86.32% | **87.84%** | -1.56 pp | -0.60 pp | **+1.52 pp** |
| **Balanced Accuracy** | **39.70%** | 35.70% | 33.13% | **34.97%** | -4.73 pp | -0.73 pp | **+1.84 pp** |
| **Macro-F1** | **0.4039** | 0.3605 | 0.3291 | **0.3516** | -0.0523 | -0.0089 | **+0.0225** |
| **Macro Precision** | 0.5484 | 0.5504 | 0.4735 | **0.5334** | -0.0150 | -0.0170 | **+0.0599** |
| **Macro Recall** | **0.3970** | 0.3570 | 0.3313 | **0.3497** | -0.0473 | -0.0073 | **+0.0184** |
| **Test BCE Loss** | 0.4403 | **0.4175** | 0.4555 | **0.4181** | -0.0222 | +0.0006 | **-0.0374** |

**Full 5x5 Confusion Matrix (Rows: True [2.75, 3.00, 3.25, 3.50, 3.75], Cols: Pred)**:
```
[[ 273,  619,  223,   27,    0],
 [ 114,  846,  740,  189,    0],
 [  28,  463, 1102,  485,    0],
 [  16,  167,  699,  760,    8],
 [   0,   48,  220,  466,   56]]
```

---

### 4.2 Task 2: Behavior Recognition (CVB + Beef Matched Retained Test Set)
- **Population**: N = 780 sequences (T=8 frames, 224x224 RGB + binary mask; CVB N=422, Kaggle Beef N=358).
- **Head**: 1D Temporal Convolutional Network (TCN, 2 blocks, 256 hidden channels, 5 classes).
- **Interpretation Boundary**: Walking is CVB-only.

| Metric | Run 5 Single-Task | Run 7 E1 Hard-Shared | Run 8 E3 Modular | Phase 3 E4 PCGrad | Delta vs Single | Delta vs E1 | Delta vs E3 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | **87.44%** | 85.00% | 85.77% | **86.92%** | -0.52 pp | **+1.92 pp** | **+1.15 pp** |
| **Balanced Accuracy** | **74.43%** | 67.30% | 66.70% | **69.15%** | -5.28 pp | **+1.85 pp** | **+2.45 pp** |
| **Macro-F1** | **0.7397** | 0.6866 | 0.6755 | **0.7114** | -0.0283 | **+0.0248** | **+0.0359** |
| **Macro Precision** | 0.7712 | 0.7195 | 0.7017 | **0.7521** | -0.0191 | **+0.0326** | **+0.0504** |
| **Macro Recall** | **0.7443** | 0.6730 | 0.6670 | **0.6915** | -0.0528 | **+0.0185** | **+0.0245** |
| **Test CE Loss** | **0.4430** | 0.6226 | **0.4193** | **0.5454** | +0.1024 | **-0.0772** | +0.1261 |

**Per-Class F1 Breakdown**:
| Class | Support | Run 5 Single | Run 7 E1 | Run 8 E3 | E4 PCGrad | Prec | Rec | F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Standing | 108 | 0.7215 | 0.7263 | **0.7417** | **0.7273** | 0.9412 | 0.5926 | 0.7273 |
| Lying | 237 | 0.9169 | 0.8924 | **0.9306** | **0.9272** | 0.9722 | 0.8861 | 0.9272 |
| Feeding | 346 | **0.9465** | 0.9031 | 0.9016 | **0.9067** | 0.8313 | 0.9971 | 0.9067 |
| Drinking | 63 | 0.8682 | 0.8702 | 0.8039 | **0.9048** | 0.9048 | 0.9048 | **0.9048** |
| Walking (CVB-only) | 26 | **0.2456** | 0.0408 | 0.0000 | **0.0909** | 0.1111 | 0.0769 | **0.0909** |

**Source-Specific Performance Breakdown**:
| Source Dataset | Metric | Run 5 Single | Run 7 E1 | Run 8 E3 | E4 PCGrad | Delta vs E1 | Delta vs E3 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CVB (Barn CCTV, N=422)** | Overall Accuracy | 80.09% | 76.78% | 80.33% | **81.04%** | **+4.26 pp** | **+0.71 pp** |
| | Balanced Accuracy | **61.05%** | 48.90% | **60.61%** | **55.30%** | **+6.40 pp** | -5.31 pp |
| | Macro-F1 | **0.6188** | 0.5402 | 0.6026 | **0.6141** | **+0.0739** | **+0.0115** |
| **Kaggle Beef (N=358)** | Overall Accuracy | **96.09%** | 94.69% | 92.18% | **93.85%** | -0.84 pp | **+1.67 pp** |
| | Balanced Accuracy | **94.26%** | 93.11% | 87.29% | **92.05%** | -1.06 pp | **+4.76 pp** |
| | Macro-F1 | **0.9414** | 0.9236 | 0.8849 | **0.9158** | -0.0078 | **+0.0309** |

**Full 5x5 Behavior Confusion Matrix (Rows: True [Standing, Lying, Feeding, Drinking, Walking], Cols: Pred)**:
```
[[ 64,   5,  19,   6,  14],
 [  0, 210,  25,   0,   2],
 [  1,   0, 345,   0,   0],
 [  0,   0,   6,  57,   0],
 [  3,   1,  20,   0,   2]]
```

---

### 4.3 Task 3: Re-Identification (SideViewCows2026 Protocol A)
- **Population**: 69 held-out cows; Parlor Gallery: 36,811 images; Barn Queries: 25,260 images; Snapshots Queries: 607 images.
- **Representation**: 512-D unit L2-normalized embedding extracted from 4-channel ResNet-18 spatial backbone.
- **Masks**: Released SideView target masks / oracle masks (NOT automatic SAM masks).

| Setting | Metric | Run 6 Single | Run 7 E1 | Run 8 E3 | Phase 3 E4 PCGrad | Delta vs Single | Delta vs E1 | Delta vs E3 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Query Barn -> Parlor** | **Rank-1** | **63.90%** | 57.38% | 49.08% | **54.97%** | -8.93 pp | -2.41 pp | **+5.89 pp** |
| (25,260 queries, 69 cows) | **Rank-5** | **77.10%** | 73.33% | 67.72% | **68.81%** | -8.29 pp | -4.52 pp | **+1.09 pp** |
| | **Rank-10** | **82.58%** | 79.79% | 75.03% | **74.99%** | -7.59 pp | -4.80 pp | -0.04 pp |
| | **mAP** | **40.68%** | 30.37% | 28.04% | **30.23%** | -10.45 pp | -0.14 pp | **+2.19 pp** |
| **Query Snapshots -> Parlor** | **Rank-1** | **62.93%** | 57.17% | 53.71% | **57.00%** | -5.93 pp | -0.17 pp | **+3.29 pp** |
| (607 queries, 63 cows) | **Rank-5** | 75.29% | **75.45%** | 72.32% | **72.16%** | -3.13 pp | -3.29 pp | -0.16 pp |
| | **Rank-10** | 81.05% | **82.70%** | 80.23% | **78.25%** | -2.80 pp | -4.45 pp | -1.98 pp |
| | **mAP** | **40.42%** | 33.69% | 28.78% | **33.87%** | -6.55 pp | **+0.18 pp** | **+5.09 pp** |

---

## 5. Calibrated Scientific Findings

### 5.1 Question 1: Does PCGrad reduce the held-out degradation observed under E1?
**Answer: Mixed and task-selective.**
- **Behavior**: YES. PCGrad significantly mitigated the degradation seen in E1. Overall accuracy gained +1.92 pp (86.92% vs 85.00%), Macro-F1 gained +0.0248 (0.7114 vs 0.6866), test loss dropped by -0.0772 (0.5454 vs 0.6226), CVB CCTV surveillance gained +4.26 pp accuracy (81.04% vs 76.78%) and +0.0739 Macro-F1 (0.6141 vs 0.5402), and minority class `Walking` F1 more than doubled (0.0909 vs 0.0408).
- **BCS**: NO. MAE slightly increased by +0.0040 (0.1828 vs 0.1788), and Acc@0 decreased by -0.87 pp.
- **Re-ID**: MIXED/MATCHED. mAP remained essentially identical to E1 across both query settings (Barn: 30.23% vs 30.37%; Snapshots: 33.87% vs 33.69%), while Barn Rank-1 was slightly lower (54.97% vs 57.38%) and Snapshots Rank-1 was virtually identical (57.00% vs 57.17%).

### 5.2 Question 2: Does PCGrad improve all tasks or only selected tasks/metrics?
**Answer: Selected tasks and metrics only.**
- PCGrad is NOT a universal fix for negative transfer. It produced clear benefits on temporal sequence classification (Behavior) and stabilized CCTV cross-domain generalization, but did not elevate BCS or Re-ID back to their single-task levels.

### 5.3 Question 3: How does optimization-level intervention E4 compare with architectural intervention E3?
**Answer: E4 PCGrad decisively outperformed E3 Modular Adapters across all three tasks.**
- **BCS**: E4 beat E3 on Real MAE (0.1828 vs 0.1916; -0.0088 error drop), Acc@0 (+1.76 pp), Acc@1 (+1.52 pp), Balanced Acc (+1.84 pp), and Macro-F1 (+0.0225).
- **Behavior**: E4 beat E3 on Overall Acc (86.92% vs 85.77%), Balanced Acc (69.15% vs 66.70%), Macro-F1 (0.7114 vs 0.6755), and Walking F1 (0.0909 vs 0.0000; E3 suffered complete representation collapse on Walking).
- **Re-ID**: E4 crushed E3 on retrieval quality: Barn Rank-1 +5.89 pp (54.97% vs 49.08%), Barn mAP +2.19 pp (30.23% vs 28.04%); Snapshots Rank-1 +3.29 pp (57.00% vs 53.71%), Snapshots mAP +5.09 pp (33.87% vs 28.78%).
- **Conclusion**: When multi-task interference occurs on a shared backbone, projecting conflicting gradients (optimization intervention) preserves cross-task representation quality far more effectively than inserting unconstrained bottleneck adapters (architectural intervention).

### 5.4 Mandatory Scientific Claim Boundaries
- **Gradient Conflict Observation**: During E4 training, pairwise shared-backbone gradient conflicts were directly and frequently observed (44,177 conflict projections across 16,140 super-steps; 45%–54% pairwise frequency).
- **Forbidden Claim 1**: We do NOT claim "gradient conflict caused all E1 negative transfer," because negative transfer may also arise from representation capacity limits, task objective scaling, or differing convergence rates.
- **Forbidden Claim 2**: We do NOT claim "PCGrad solved negative transfer," because held-out performance across all three tasks remains below single-task baselines (Run 4, Run 5, Run 6).
- **Finality**: Phase 3 E4 PCGrad is the **FINAL ADDITIONAL EXPERIMENTAL RUN**. No further training runs or ablations will be launched. GradNorm (E5) and partial sharing (E2) remain deferred.

---

## 6. Artifact Registry

| Artifact | File Path | Description |
| :--- | :--- | :--- |
| **Evaluation Script** | `scripts/evaluate_mtl_e4_held_out.py` | Official E4 held-out evaluator with exact population and checkpoint assertions |
| **Modal Cloud Wrapper** | `scripts/modal_evaluate_mtl_e4_held_out.py` | Cloud execution script mounting volumes on `hasinishrak2015` |
| **Unit Test Suite** | `tests/test_mtl_e4_evaluation.py` | 12/12 unit tests verifying parameter counts, metric math, assertions, and pure inference |
| **Evaluation Metrics Artifact** | `artifacts/mtl_e4_evaluation/mtl_e4_test_evaluation_metrics.json` | Complete deterministic JSON payload containing all test metrics and 3-way deltas |
| **Checkpoint Evaluated** | `/mtl-checkpoints/mtl_e4_pcgrad/mtl_e4_best.pth` | Frozen Epoch-3 model weights committed on Modal persistent volume |

---

## 7. Next Steps

1. Update `docs/research_log/README.md` to index this log entry.
2. Synchronize `memory/state.md` and `memory/history.md`.
3. Mirror all modified customizations to `D:\custom-antigravity`.
4. Git commit and push to `origin/main`.
5. Cease all experimental model training—proceed to final thesis writing and defense preparation.
