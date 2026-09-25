# Phase 3 Run 8 E3 Modular Multi-Task Learning Official Held-Out Evaluation Results

**Date**: 2026-09-25  
**Author**: Hasin Ishrak  
**Execution Environment**: Modal Cloud (`hasinishrak2015`, NVIDIA L40S, App `ap-muXQHq9UeSnLniVrUfSQhc`)  
**Target Checkpoint**: `/mtl-checkpoints/mtl_e3_modular/mtl_e3_best.pth` (Epoch 2, val_e3_objective = 0.38888)  
**Status**: COMPLETED & VERIFIED (Held-Out Test Isolation Preserved, Frozen Checkpoint)

---

## 1. Executive Summary

We executed the official held-out test evaluation of the frozen Phase 3 Run 8 (E3 Modular Multi-Task Learning / Task-Private Residual Bottleneck Adapters) best checkpoint on Modal cloud (`hasinishrak2015`, NVIDIA L40S, App `ap-muXQHq9UeSnLniVrUfSQhc`). In strict accordance with the thesis evaluation protocol, we evaluated exclusively the Epoch-2 checkpoint (`mtl_e3_best.pth`) selected by the predefined composite validation E3 objective, with held-out test-set isolation preserved.

The model was evaluated against the exact frozen populations and protocols of both the single-task perception baselines (Runs 4, 5, 6) and the hard-shared control (Run 7 E1):
1. **BCS**: Matched against Run 4 and Run 7 on the exact 7,549 successful-perception ScienceDB test images.
2. **Behavior**: Matched against Run 5 and Run 7 on the exact 780 retained sequences (T=8).
3. **Re-ID**: Matched against Run 6 and Run 7 on the official SideViewCows2026 Protocol A (69 held-out cows; 36,811 parlor gallery, 25,260 barn queries, 607 snapshot queries).

### Key Scientific Findings:
1. **Behavior Results Over Run 7 E1 (Mixed Outcome)**:
   - E3 produced metric-dependent Behavior improvements over E1, particularly in test loss and CVB surveillance performance, but did not improve all Behavior metrics and therefore does not demonstrate complete removal of negative transfer.
   - Overall Behavior accuracy improved by +0.77 pp (85.77% vs 85.00%) and test cross-entropy loss dropped by -0.2033 (0.4193 vs 0.6226).
   - On authentic barn surveillance CCTV footage (**CVB**), Run 8 E3 achieved 80.33% Accuracy (+3.55 pp over Run 7's 76.78%, compared with single-task Run 5's 80.09%) and 0.6026 Macro-F1 (+6.24 pp over Run 7's 0.5402).
   - Per-class F1 on Standing was 0.7417 (vs Run 7 0.7263, Run 5 0.7215) and Lying was 0.9306 (vs Run 7 0.8924, Run 5 0.9169).
   - However, balanced accuracy decreased by 0.60 pp (66.70% vs 67.30%), overall Macro-F1 decreased by 0.0111 (0.6755 vs 0.6866), and minority class Walking F1 dropped from 0.0408 to 0.0000.
2. **Held-Out Re-ID Retrieval**:
   - E3 degraded held-out Re-ID retrieval relative to both E1 and the single-task Run 6 reference. Possible explanations include task interference, insufficient task-specific capacity, or checkpoint-selection trade-offs, but the mechanism was not directly measured.
   - On Protocol A (69 unseen cows): Barn -> Parlor retrieval reached 49.08% Rank-1 and 28.04% mAP (vs Run 7 57.38% Rank-1, 30.37% mAP; vs Run 6 63.90% Rank-1, 40.68% mAP); Snapshots -> Parlor reached 53.71% Rank-1 and 28.78% mAP (vs Run 7 57.17% Rank-1, 33.69% mAP; vs Run 6 62.93% Rank-1, 40.42% mAP).
3. **BCS Test Outcome**:
   - Run 8 achieved a BCS MAE of 0.1916 on the matched held-out test population (vs Run 7 E1 0.1788 and Run 4 0.1709), with exact accuracy (Acc@0) at 38.47% and Acc@1 at 86.32%.

---

## 2. Evaluation Integrity & Provenance

- **Evaluated Checkpoint**: `/mtl-checkpoints/mtl_e3_modular/mtl_e3_best.pth`
- **Selection Criterion**: Minimum validation E3 objective at Epoch 2 (`val_e3_objective = 0.38888`).
- **Test Integrity**: Held-out test-set isolation was preserved according to the implemented evaluation protocol and recorded checks. No test samples were used during training, architecture selection, hyperparameter tuning, threshold tuning, or checkpoint selection.
- **Representation Provenance**:
  - BCS: Pre-packed monolithic tensor `test_bcs_224.pt` (7,549 samples, `[4, 224, 224]`, ImageNet-normalized RGB + binary mask).
  - Behavior: 780 retained test sequences from `mtl-data/behavior_test`, preloaded into RAM.
  - Re-ID: SideViewCows2026 official `protocol_cross_setting.csv` evaluated with 512-D unit-L2 normalized embeddings extracted via `model.forward_reid()` (shared trunk + Re-ID private adapter).

---

## 3. Forensic Test Evaluation Results & Head-to-Head Comparisons

### Task 1: Body Condition Score (BCS) — ScienceDB Held-Out Test (N=7,549)

Evaluated on the exact 7,549 perception-success test images from Run 4 using the cumulative Frank & Hall ordinal BCE head.

| Metric | Single-Task Run 4 (Matched N=7,549) | Run 7 E1 Hard-Shared (N=7,549) | Run 8 E3 Modular (N=7,549) | Delta vs Run 4 | Delta vs Run 7 E1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Real MAE** | **0.1709** | 0.1788 | 0.1916 | +0.0207 | +0.0128 |
| **Acc@0 (Exact)** | **43.57%** | 41.10% | 38.47% | -5.10% | -2.63% |
| **Acc@1 (+/-0.25)** | **89.40%** | 88.44% | 86.32% | -3.08% | -2.12% |
| **Balanced Accuracy** | **39.70%** | 35.70% | 33.13% | -6.57% | -2.57% |
| **Macro-F1** | **0.4039** | 0.3605 | 0.3291 | -0.0748 | -0.0314 |
| **Macro Precision** | 0.4357 | **0.5504** | 0.4735 | +0.0378 | -0.0769 |
| **Macro Recall** | **0.3970** | 0.3570 | 0.3313 | -0.0657 | -0.0257 |
| **Test Loss (Ordinal BCE)**| 0.4403 | **0.4175** | 0.4555 | +0.0152 | +0.0380 |

#### Confusion Matrix (Run 8 E3 BCS)
```
Pred ->   0      1      2      3      4
True 0: [ 269,   603,   241,    29,     0]
True 1: [ 133,   819,   690,   245,     2]
True 2: [  38,   459,  1090,   489,     2]
True 3: [  22,   132,   792,   691,    13]
True 4: [   4,    47,   271,   433,    35]
```

---

### Task 2: Cattle Behavior Recognition — Retained Test Population (N=780, T=8)

Evaluated on the exact 780 retained video sequences from Run 5 using the lightweight 1D TCN head.

| Metric | Single-Task Run 5 (Matched N=780) | Run 7 E1 Hard-Shared (N=780) | Run 8 E3 Modular (N=780) | Delta vs Run 5 | Delta vs Run 7 E1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Overall Accuracy** | **87.44%** | 85.00% | **85.77%** | -1.67% | **+0.77%** |
| **Balanced Accuracy** | **74.43%** | 67.30% | 66.70% | -7.73% | -0.60% |
| **Macro-F1** | **0.7397** | 0.6866 | 0.6755 | -0.0642 | -0.0111 |
| **Macro Precision** | **0.7712** | 0.7195 | 0.7017 | -0.0695 | -0.0178 |
| **Macro Recall** | **0.7443** | 0.6730 | 0.6670 | -0.0773 | -0.0060 |
| **Test Loss (CE)** | 0.4430 | 0.6226 | **0.4193** | **-0.0237** | **-0.2033** |

#### Per-Class F1 Breakdown
| Class | Support | Run 5 Matched F1 | Run 7 E1 F1 | Run 8 E3 F1 | Delta vs Run 7 E1 | Direction |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standing** | 108 | 0.7215 | 0.7263 | **0.7417** | **+0.0154** | **Improved (+1.54 pp)** |
| **Lying** | 237 | 0.9169 | 0.8924 | **0.9306** | **+0.0382** | **Improved (+3.82 pp)** |
| **Feeding** | 346 | **0.9465** | 0.9031 | 0.8940 | -0.0091 | Slight Drop |
| **Drinking** | 63 | 0.8682 | **0.8702** | 0.8113 | -0.0589 | Slight Drop |
| **Walking (Minority)**| 26 | **0.2456** | 0.0408 | 0.0000 | -0.0408 | Unconverged at Ep 2 |

#### Domain Breakdown (CVB vs Kaggle Beef)
| Sub-Population | Metric | Run 5 Matched | Run 7 E1 | Run 8 E3 | Delta vs Run 7 E1 | Direction |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CVB (Barn CCTV, N=422)** | Accuracy | 80.09% | 76.78% | **80.33%** | **+3.55%** | **Surpasses Run 5 & E1** |
| | Balanced Accuracy | 61.05% | 48.90% | **60.61%** | **+11.71%** | **Massive Recovery** |
| | Macro-F1 | 0.6188 | 0.5402 | **0.6026** | **+0.0624** | **+6.24 pp over E1** |
| **Beef (Feedlot/Pasture, N=358)** | Accuracy | **96.09%** | 94.69% | 92.18% | -2.51% | Slight Drop |
| | Balanced Accuracy | **94.26%** | 93.11% | 87.29% | -5.82% | Slight Drop |
| | Macro-F1 | **0.9414** | 0.9236 | 0.8849 | -0.0387 | Slight Drop |

---

### Task 3: Individual Cattle Re-Identification — Protocol A (69 Held-Out Cows)

Evaluated on the official SideViewCows2026 Protocol A cross-setting benchmark. Gallery: Parlor (36,811 images). Queries: Barn (25,260 images) and Snapshots (607 images).

| Query Subset | Metric | Single-Task Run 6 Baseline | Run 7 E1 Hard-Shared | Run 8 E3 Modular | Delta vs Run 6 | Delta vs Run 7 E1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Query Barn -> Parlor** | **Rank-1** | **63.90%** | 57.38% | 49.08% | -14.82% | -8.30% |
| (25,260 queries, 69 cows) | **Rank-5** | **77.10%** | 73.33% | 67.72% | -9.38% | -5.61% |
| | **Rank-10** | **82.58%** | 79.79% | 75.03% | -7.55% | -4.76% |
| | **mAP** | **40.68%** | 30.37% | 28.04% | -12.64% | -2.33% |
| **Query Snapshots -> Parlor**| **Rank-1** | **62.93%** | 57.17% | 53.71% | -9.22% | -3.46% |
| (607 queries, 63 cows) | **Rank-5** | 75.29% | **75.45%** | 72.32% | -2.97% | -3.13% |
| | **Rank-10** | 81.05% | **82.70%** | 80.23% | -0.82% | -2.47% |
| | **mAP** | **40.42%** | 33.69% | 28.78% | -11.64% | -4.91% |

---

## 4. Multi-Task Comparative Analysis: E3 Modular vs E1 Hard-Shared vs Single-Task

| Task & Metric | Single-Task Perception | Run 7 E1 (Hard Sharing) | Run 8 E3 (Modular Routing) | Comparative Outcome | Observation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **BCS Real MAE** | **0.1709** | 0.1788 | 0.1916 | Single-Task lowest error | Run 8 achieved BCS MAE of 0.1916 on matched test population; error increased vs E1 (0.1788) and single-task (0.1709) |
| **Behavior Overall Acc** | **87.44%** | 85.00% | **85.77%** | Run 8 E3 > Run 7 E1 (+0.77%) | Overall accuracy improved over E1, though below single-task Run 5 |
| **Behavior Barn CCTV Acc**| 80.09% | 76.78% | **80.33%** | Run 8 E3 highest | CVB CCTV accuracy improved over both Run 7 (+3.55 pp) and Run 5 (+0.24 pp) |
| **Behavior Test Loss** | 0.4430 | 0.6226 | **0.4193** | Run 8 E3 lowest | Cross-entropy test loss dropped by -0.2033 vs E1, lower than Run 5 |
| **Behavior Balanced Acc** | **74.43%** | 67.30% | 66.70% | Single-Task highest | Balanced accuracy dropped by 0.60 pp vs E1; Walking class F1 reached 0.0000 |
| **Behavior Macro-F1** | **0.7397** | 0.6866 | 0.6755 | Single-Task highest | Macro-F1 dropped by 0.0111 vs E1, indicating mixed Behavior outcomes |
| **Re-ID Barn Rank-1** | **63.90%** | 57.38% | 49.08% | Single-Task highest | E3 degraded held-out retrieval vs E1 and Run 6; mechanism was not directly measured |
| **Re-ID Snapshots Rank-1**| **62.93%** | 57.17% | 53.71% | Single-Task highest | E3 degraded held-out retrieval vs E1 and Run 6; mechanism was not directly measured |

---

## 5. Summary of Experimental Findings & Thesis Implications

1. **Selective, Metric-Dependent Behavior Changes (Run 8 E3)**:
   - Allocating lightweight task-private residual adapters (395k parameters, +3.32% trainable capacity) produced metric-dependent Behavior changes: test loss dropped from 0.6226 to **0.4193** and overall accuracy rose from 85.00% to **85.77%**.
   - On surveillance barn CCTV video (**CVB**), E3 reached **80.33% accuracy** and **0.6026 Macro-F1**, improving over E1 (76.78% Acc, 0.5402 Macro-F1).
   - However, balanced accuracy decreased by 0.60 pp (66.70% vs 67.30%), Macro-F1 decreased by 0.0111 (0.6755 vs 0.6866), and minority class Walking F1 dropped from 0.0408 to 0.0000. Therefore, E3 produced metric-dependent Behavior improvements over E1, particularly in test loss and CVB surveillance performance, but did not improve all Behavior metrics and therefore does not demonstrate complete removal of negative transfer.
2. **Held-Out Re-ID Degradation**:
   - E3 degraded held-out Re-ID retrieval relative to both E1 and the single-task Run 6 reference across both query conditions (Barn Rank-1: 49.08% vs 57.38% and 63.90%, mAP 28.04% vs 30.37% and 40.68%; Snapshots Rank-1: 53.71% vs 57.17% and 62.93%, mAP 28.78% vs 33.69% and 40.42%).
   - Possible explanations include task interference, insufficient task-specific capacity, or checkpoint-selection trade-offs, but the underlying mechanism was not directly measured.
3. **Overall Multi-Task Evaluation Synthesis**:
   - The defensible overall conclusion across the three tasks is:
     - **BCS**: E3 degraded relative to both E1 (MAE 0.1916 vs 0.1788) and the single-task Run 4 reference (0.1709).
     - **Behavior**: Mixed outcomes; selected metrics improved substantially (test loss -0.2033, CVB CCTV accuracy +3.55 pp), while others worsened (balanced accuracy -0.60 pp, Macro-F1 -0.0111, Walking F1 to 0.0000).
     - **Re-ID**: E3 degraded relative to both E1 and the single-task Run 6 reference.
   - Run 7 demonstrated held-out performance degradation under hard sharing, consistent with negative transfer at the outcome level. The underlying optimization mechanism was not directly measured.
   - Therefore, task-private residual adapters showed selective, metric-dependent benefit rather than universal negative-transfer mitigation.

---

## 6. Artifact Registry

| File Path | Description |
| :--- | :--- |
| `scripts/evaluate_mtl_e3_held_out.py` | Complete evaluation engine for BCS, Behavior, and Re-ID Protocol A on E3 |
| `scripts/modal_evaluate_mtl_e3_held_out.py` | Modal cloud runner executed on NVIDIA L40S (`hasinishrak2015`, App `ap-muXQHq9UeSnLniVrUfSQhc`) |
| `tests/test_mtl_e3_evaluation.py` | Unit test suite (5/5 tests passed) |
| `artifacts/mtl_e3_evaluation/mtl_e3_test_evaluation_metrics.json` | Master JSON containing all held-out results, confusion matrices, and exact deltas |
| `docs/research_log/2026-09-25_phase3_run8_mtl_e3_modular_held_out_evaluation_results.md` | This official held-out evaluation research log |
