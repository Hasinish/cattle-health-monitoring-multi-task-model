# Phase 3 E4 PCGrad Multi-Task Learning Architecture, Implementation & Cloud Smoke Certification

**Date**: 2026-09-25  
**Author**: Hasin Ishrak  
**Execution Environment**: Modal Cloud (`hasinishrak2015`, Tesla T4, App `ap-xbVM5Th8GNiWyToOw8lMsx`)  
**Target Architecture**: Hard-shared 4-channel ResNet-18 (11,926,706 trainable params) with PCGrad (Yu et al., 2020) on shared backbone  
**Status**: IMPLEMENTED, CERTIFIED & SMOKE-TESTED (Full Training Awaiting Manual User Launch)

---

## 1. Executive Summary

We designed, implemented, unit-tested, and smoke-certified the final additional multi-task learning experiment: **E4 PCGrad** (Projecting Conflicting Gradients; Yu et al., 2020) on Modal cloud profile `hasinishrak2015`. E4 serves as the user-approved final additional optimization-control experiment to answer a single controlled scientific question: *Does PCGrad improve the hard-shared E1 multi-task model when the architecture, data, task heads, training schedule, losses, optimizer, and validation-selection rule are otherwise kept matched?*

All 12 local unit tests passed (3.31s). Zero-GPU cloud readiness verification (`ap-Iy8ywTox8KxYyCI7kEc4hD`) certified dataset integrity, exact parameter matching, and held-out test isolation. A 2-epoch cheap GPU smoke test on an NVIDIA Tesla T4 (`ap-xbVM5Th8GNiWyToOw8lMsx`, runtime 3.7s) confirmed finite loss decrease (1.7787 -> 1.4754), active execution of PCGrad conflict projections (4 projections/epoch; 2.0 projections/step), finite gradient cosines and conflict fractions, and bit-identical checkpoint reload (`max_logit_diff == 0.00000000`). Canonical held-out test sets remain strictly untouched. Full 30-epoch training was **not** launched and awaits explicit manual user command.

---

## 2. Context & Controlled Scientific Purpose

Phase 3 multi-task learning investigates the mitigation of negative transfer across three disparate agricultural computer vision tasks:
1. **Body Condition Scoring (BCS)** (ScienceDB, 4-channel perception crop + binary mask, ordinal classification).
2. **Behavior Recognition** (CVB barn CCTV + Kaggle Beef feedlot, T=8 temporal sequence, 1D TCN).
3. **Individual Cow Re-Identification (Re-ID)** (SideViewCows2026, 4-channel perception crop + binary mask, metric retrieval).

### The Scientific Triad:
- **E1 Control Baseline (Run 7)**: Ordinary hard parameter sharing with naive accumulated unit-weight task gradients (`w_BCS = 1.0, w_Behavior = 1.0, w_ReID = 1.0`).
- **E3 Modular Intervention (Run 8)**: Architectural task-private residual adapter intervention (+3.32% parameter capacity; 12,322,610 trainable parameters) preserving the shared backbone while allowing task-specific representational routing.
- **E4 PCGrad Intervention (Final Additional Control)**: Optimization intervention. Uses the **EXACT SAME** hard-shared ResNet-18 model as E1 (11,926,706 trainable parameters; 0 adapters, 0 gates, 0 private visual trunks), but projects conflicting task gradients on the shared backbone prior to the optimizer step.

*Note on Deferred Experiments*: GradNorm (E5) and partial sharing (E2) remain deferred post-deadline. E4 is the sole user-approved additional experimental run brought forward to isolate optimization vs architectural effects.

---

## 3. Architecture Specification & Parameter Invariance

E4 preserves the exact architectural parameter count and structure of E1:

| Component | Architecture Specification | Trainable Parameters | Status |
| :--- | :--- | :--- | :--- |
| **Shared Spatial Backbone** | 4-channel ResNet-18 (conv1 RGB + mask mean init, layers 1-4, avgpool) | 11,179,648 | Shared across all 3 tasks |
| **BCS Head** | Cumulative Frank & Hall (2001) Ordinal Linear(512, 4) | 2,052 | Task-specific private |
| **Behavior Head** | Lightweight 1D Temporal Convolutional Network (TCN, 2 Conv1d blocks + Linear(256, 5)) | 723,973 | Task-specific private |
| **Re-ID Head** | 512-D Unit L2 metric embedding + Linear(512, 41) classifier | 21,033 | Task-specific private |
| **TOTAL** | **Exact match to E1 Hard-Shared Model** | **11,926,706** | **Certified (0 adapters, 0 gates)** |

---

## 4. PCGrad Algorithm & Mathematical Implementation

The standard PCGrad algorithm (Yu et al., 2020) is implemented in `PCGradProjector`:
- Applied **ONLY** to the shared ResNet-18 backbone (11,179,648 parameters). Task heads are never projected against each other; each head receives only its own task gradient.
- For each super-step:
  1. Forward BCS batch -> compute ordinal BCE loss -> backward -> capture shared backbone grad $g_{BCS}$ and head grad $h_{BCS}$.
  2. Clear gradients.
  3. Forward Behavior batch -> compute cross-entropy loss -> backward -> capture shared backbone grad $g_{Beh}$ and head grad $h_{Beh}$.
  4. Clear gradients.
  5. Forward Re-ID batch -> compute cross-entropy loss -> backward -> capture shared backbone grad $g_{ReID}$ and head grad $h_{ReID}$.
  6. Clear gradients.
  7. Apply PCGrad to the 3 shared-backbone task vectors:
     - For each task $i \in \{BCS, Beh, ReID\}$:
       - Shuffle remaining tasks $j \in \{1, 2, 3\} \setminus \{i\}$ using deterministic RNG initialized from seed 2026.
       - If $g_i \cdot g_j^{orig} < 0$:
         $$g_i \leftarrow g_i - \frac{g_i \cdot g_j^{orig}}{\|g_j^{orig}\|^2 + \epsilon} g_j^{orig}$$
         Where $g_j^{orig}$ is the original comparison gradient.
  8. Sum projected shared gradients: $g_{shared} = g_{BCS}^{proj} + g_{Beh}^{proj} + g_{ReID}^{proj}$ (summation preserves the unit-weight gradient accumulation scale of E1).
  9. Restore $g_{shared}$ to `model.backbone`, restore $h_{BCS}$ to `model.bcs_head`, $h_{Beh}$ to `model.behavior_tcn`, $h_{ReID}$ to `model.reid_head`.
  10. Single optimizer update: `optimizer.step()`.

### Gradient Diagnostics:
The engine records per super-step and aggregates per epoch:
- Shared gradient norms: $\|g_{BCS}\|, \|g_{Beh}\|, \|g_{ReID}\|$.
- Pre-projection cosines: $\cos(g_i, g_j) = \frac{g_i \cdot g_j}{\|g_i\| \|g_j\| + \epsilon}$.
- Conflict fractions: fraction of super-steps where $g_i \cdot g_j < 0$.
- Projections triggered: total number and rate per step.
- Post-projection cosines: alignment following gradient surgery.

### Scientific Claim Boundaries:
1. These diagnostics establish measured gradient-direction conflict frequency during E4 training.
2. They do NOT retroactively prove gradient conflict caused E1's held-out degradation (since E1 did not record individual task gradients).
3. They do NOT prove PCGrad "solves negative transfer" prior to held-out evaluation.

---

## 5. Verification Results

### A. Local Unit Test Suite (`tests/test_mtl_e4_pcgrad.py`)
Executed 12 rigorous unit tests:
1. `test_01_exact_e1_architecture_retained`: PASS (11,926,706 params).
2. `test_02_no_adapters_or_private_visual_trunks`: PASS (`assert_hard_sharing()` true, 0 adapter modules).
3. `test_03_pcgrad_conflict_projection_math`: PASS (numerical verification of orthogonalization formula).
4. `test_04_non_conflicting_gradients_remain_unchanged`: PASS (0 projections, 0 drift).
5. `test_05_pcgrad_operates_only_on_shared_backbone`: PASS (heads unprojected, backbone projected).
6. `test_06_bcs_head_receives_only_bcs_gradients`: PASS (zero cross-talk).
7. `test_07_behavior_tcn_receives_only_behavior_gradients`: PASS (zero cross-talk).
8. `test_08_reid_head_receives_only_reid_gradients`: PASS (zero cross-talk).
9. `test_09_deterministic_projected_gradients`: PASS (seed 2026 determinism verified).
10. `test_10_forward_shapes`: PASS (BCS [B, 4], Beh [B, 5], ReID [B, 41] + [B, 512] unit L2).
11. `test_11_checkpoint_save_reload_bit_identity`: PASS (diff = 0.00000000; PCGrad RNG state restored).
12. `test_12_sideview_protocol_disjointness`: PASS (41 train vs 69 eval cows; 0 overlap).

*Result*: 12/12 passed in 3.311s. E1 unit tests (`tests/test_mtl_e1_hard_shared.py`) re-verified with 6/6 passed in 2.420s.

### B. Modal Cloud Zero-GPU Readiness (`hasinishrak2015`, App `ap-Iy8ywTox8KxYyCI7kEc4hD`)
- Volume mounts verified: `mtl-data`, `mtl-checkpoints`, `sideview-data`.
- BCS tensors verified: 34,369 train samples (6.42 GB), 7,817 val samples (1,496.3 MB).
- Behavior sequences verified: 3,641 train seqs, 630 val seqs.
- Re-ID cows verified: 41 train cows, 69 held-out eval cows, 0 overlap.
- Checkpoint directories writable: `/mtl-checkpoints/mtl_e4_pcgrad_smoke/` and `/mtl-checkpoints/mtl_e4_pcgrad/`.
- Held-out test isolation confirmed: 0 test files loaded.
- Status: `CERTIFIED_READY_FOR_E4_SMOKE`.

### C. Modal Cloud GPU Smoke Test (`hasinishrak2015`, Tesla T4, App `ap-xbVM5Th8GNiWyToOw8lMsx`)
- Hardware: NVIDIA Tesla T4 (`gpu="T4"`, cpu=4.0, memory=16384 MB).
- Execution time: 3.7 seconds (2 epochs, batch sizes 8/4/8).
- Training loss drop: 1.7787 (Epoch 1) -> 1.4754 (Epoch 2).
- Validation E4 objective: 1.7799 (best at Epoch 1).
- PCGrad projections executed:
  - Epoch 1: 4 projections triggered (2.00 / step; conflicts: BCS-Beh 50%, Beh-ReID 50%, BCS-ReID 0%).
  - Epoch 2: 4 projections triggered (2.00 / step; conflicts: BCS-ReID 50%, Beh-ReID 50%, BCS-Beh 0%).
  - Total: 8 projections triggered across 4 super-steps.
- Checkpoint reload bit-identity: `max_logit_diff == 0.00000000`.
- Checkpoints committed to `/mtl-checkpoints/mtl_e4_pcgrad_smoke/` on volume `mtl-checkpoints`.
- Local smoke artifact synced: `artifacts/mtl_e4_pcgrad_smoke/mtl_e4_pcgrad_smoke_metrics.json`.

---

## 6. Artifacts & Registry

- Implementation script: `scripts/train_mtl_e4_pcgrad.py`
- Modal cloud wrapper: `scripts/modal_train_mtl_e4_pcgrad.py`
- Unit test suite: `tests/test_mtl_e4_pcgrad.py`
- Smoke test metrics artifact: `artifacts/mtl_e4_pcgrad_smoke/mtl_e4_pcgrad_smoke_metrics.json`
- Research log: `docs/research_log/2026-09-25_phase3_mtl_e4_pcgrad_implementation_and_smoke.md`

---

## 7. Next Steps & Execution Control

1. **Awaiting Manual User Approval**: Full 30-epoch training on NVIDIA L40S was **NOT** launched.
2. **Exact Manual Launch Command** (when ready):
   ```bash
   modal run --detach --profile hasinishrak2015 scripts/modal_train_mtl_e4_pcgrad.py::main --epochs 30
   ```
3. Post-training: Evaluate held-out test sets across all 3 tasks using the frozen `mtl_e4_best.pth` checkpoint and perform head-to-head comparison against E1 (Run 7) and E3 (Run 8).
