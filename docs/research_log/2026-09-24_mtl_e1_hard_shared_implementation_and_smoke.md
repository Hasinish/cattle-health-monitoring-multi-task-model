# Research Log — 2026-09-24: Phase 3 Run 7 E1 Hard-Shared Multi-Task Learning Implementation & Cloud Smoke Certification

## 1. Executive Summary
Implemented, hardened, and verified **Phase 3 Run 7: E1 Hard-Shared Multi-Task Learning Control** across ScienceDB Body Condition Scoring (BCS), CVB + Kaggle Beef Behavior Recognition, and SideViewCows2026 Cow Re-Identification (Re-ID). Executed zero-GPU cloud readiness audit and a cheap GPU smoke test on NVIDIA Tesla T4 (`hasinishrak2015`, App `ap-C6fr97nFN503dgBR5PHTrt`). Proved that exactly ONE 4-channel ResNet-18 spatial feature extractor is shared across all three tasks (11,926,706 total trainable parameters; 11,179,648 shared backbone). Verified finite losses across all three tasks, joint backpropagation into the single backbone, optimizer updates, and bit-identical checkpoint reload (`max_logit_diff == 0.00000000`). Canonical test splits and 69 held-out evaluation cows remained strictly untouched. **Full 30-epoch training was NOT launched and awaits manual user execution.**

---

## 2. Scientific Purpose of Run 7
Run 7 is the primary **E1 Hard-Shared Multi-Task Learning Control baseline**. Its purpose is to discover whether forcing the three disparate tasks to share a single visual trunk produces negative transfer relative to the selected single-task models:
- BCS: Run 4 (ScienceDB 4-channel ResNet-18 Ordinal BCE)
- Behavior: Run 5 (CVB + Beef 4-channel ResNet-18 + lightweight 1D TCN)
- Re-ID: Run 6 (SideViewCows2026 4-channel ResNet-18 512-D embedding)

To maintain strict scientific control integrity, E1 excludes:
- Private task trunks / parallel ResNets
- Adapters / gating mechanisms (reserved for Run 8 / E3)
- Multi-objective dynamic gradient balancers (PCGrad, GradNorm)
- Dynamic task loss re-weighting or uncertainty weighting
- Multi-task pretraining or external foundation models

---

## 3. Architecture & Shared/Private Boundary

```text
                               Shared 4-Channel ResNet-18 Trunk
                            [conv1, bn1, relu, maxpool, layer1-4, avgpool]
                                          (11,179,648 params)
                                                   │
                  ┌────────────────────────────────┼────────────────────────────────┐
                  │                                │                                │
           [B_bcs, 512]                     [B_beh*8, 512]                   [B_reid, 512]
                  │                                │                                │
                  │                      reshape [B_beh, 8, 512]                    │
                  │                                │                         F.normalize (L2)
                  │                                │                                │
            BCS Ordinal Head                  Behavior TCN                      Re-ID Head
            Linear(512, 4)             2 Conv1d blocks + Linear(256, 5)        Linear(512, 41)
             (2,052 params)                   (723,973 params)                 (21,033 params)
                  │                                │                                │
           [B_bcs, 4] logits              [B_beh, 5] logits               [B_reid, 41] logits
                  │                                │                                │
          BCEWithLogitsLoss                 CrossEntropyLoss                 CrossEntropyLoss
```

### Exact Parameter Distribution:
- **Shared Spatial Backbone (ResNet-18)**: 11,179,648 parameters (+3,136 over standard 3-channel 11,176,512 backbone due to 4th mask channel in `conv1`).
- **BCS Ordinal Head (`Linear(512, 4)`)**: 2,052 parameters.
- **Behavior 1D TCN (`TemporalConvNet`)**: 723,973 parameters.
  * Block 1: `Conv1d(512, 256, 3, pad=1)` + `BatchNorm1d(256)` + `GELU()` + `Dropout(0.2)` + residual projection `Conv1d(512, 256, 1)`.
  * Block 2: `Conv1d(256, 256, 3, pad=1)` + `BatchNorm1d(256)` + `GELU()` + `Dropout(0.2)` + identity skip.
  * Pooling: `AdaptiveAvgPool1d(1)` -> `squeeze(-1)`.
  * Classifier: `Linear(256, 5)`.
- **Re-ID Linear Classifier (`Linear(512, 41)`)**: 21,033 parameters.
- **Total E1 Trainable Parameters**: exactly **11,926,706**.

### Programmatic Hard Sharing Proof:
- Verified that `model.backbone` is the ONLY instance of `ResNet18SharedBackbone`.
- Programmatically asserted that no 2D spatial convolutions exist in any task head.
- Verified that all three tasks pass gradients into the identical `model.backbone.conv1.weight.grad` tensor.

### Behavior Temporal Modeling Clarification:
The Behavior pathway is inherently sequence-based (T=8 frames). Behavior does NOT introduce a separate spatial ResNet; instead, each frame in `[B, 8, 4, 224, 224]` is reshaped to `[B*8, 4, 224, 224]` and passed through the **same shared visual ResNet-18 trunk** to produce `[B, 8, 512]` features, which then feed the lightweight Behavior TCN. The TCN is part of the task output head, maintaining E1 as a strict hard-shared visual-trunk baseline.

---

## 4. Input Shapes, Losses & Metrics

| Task | Input Tensor Shape | Representation & Channels | Loss Function | Metrics |
| :--- | :--- | :--- | :--- | :--- |
| **BCS** | `[B_bcs, 4, 224, 224]` | ImageNet-normalized RGB + SAM 2.1 binary mask in {0.0, 1.0} | `BCEWithLogitsLoss` (cumulative ordinal on 4 thresholds) | Real BCS MAE, Acc@1 (+/-0.25), Acc@0, Balanced Acc, Macro-F1 |
| **Behavior** | `[B_beh, 8, 4, 224, 224]` | T=8 frames; each frame ImageNet RGB + SAM 2.1 binary mask | `CrossEntropyLoss` (5 classes) | Accuracy, Balanced Acc, Macro-F1, per-class F1 |
| **Re-ID** | `[B_reid, 4, 224, 224]` | ImageNet-normalized RGB + SideView GT binary mask | `CrossEntropyLoss` (41 training cows) | Top-1 Accuracy, Balanced Acc, Macro-F1 |

---

## 5. Task-Balanced Update Schedule & Epoch Math
- **Batch Sizes**: BCS = 64, Behavior = 8 (sequences of 8 frames), Re-ID = 32.
- **Fixed Task Weights**:
  * `w_BCS = 1.0`
  * `w_Behavior = 1.0`
  * `w_ReID = 1.0`
- **One Super-Step**:
  ```python
  optimizer.zero_grad()
  (1.0 * loss_bcs).backward()       # Activations freed immediately
  (1.0 * loss_behavior).backward()  # Activations freed immediately
  (1.0 * loss_reid).backward()      # Activations freed immediately
  optimizer.step()
  ```
- **Definition of One Epoch**: Defined by the primary anchor dataset loader (`train_loader_bcs`), yielding **537 super-steps per epoch**.
- **Oversampling via Deterministic Cycling**:
  * **BCS**: 537 batches drawn per epoch = 34,368 samples (~1.000x of 34,369 samples).
  * **Behavior**: 537 batches drawn per epoch = 4,296 sequences (1.180x oversampling of 3,641 sequences; cycles deterministically).
  * **Re-ID**: 537 batches drawn per epoch = 17,184 images (1.349x oversampling of 12,753 images; cycles deterministically).
- **Composite Model Selection Objective**:
  `val_E1_objective = (BCS_val_loss + Behavior_val_loss + ReID_val_loss) / 3.0`
  Best checkpoint is selected strictly by minimum `val_E1_objective`.

---

## 6. Dataset Partitioning & Anti-Leakage Proofs
- **BCS (ScienceDB)**:
  * Train: 34,369 samples (`train_bcs_224.pt`, 6.42 GB, SHA-256 `b45b3b14...`)
  * Validation: 7,817 samples (`val_bcs_224.pt`, 1.46 GB, SHA-256 `3be6daac...`)
  * Canonical test data: strictly absent from volume (`bcs_test_absent: true`).
- **Behavior (CVB + Beef)**:
  * Train: 3,641 sequences (29,128 frames + 29,128 masks)
  * Validation: 630 sequences (5,040 frames + 5,040 masks)
  * Train/Val Sequence Disjointness: 0 overlapping sequence IDs (`train_val_disjoint: true`).
  * Canonical test data: strictly absent from volume (`behavior_test_absent: true`).
- **Re-ID (SideViewCows2026 Zero-Copy Policy)**:
  * Train: 12,753 image/mask pairs across strictly 41 training cows.
  * Validation: 2,683 image/mask pairs across strictly 41 training cows.
  * Held-Out Evaluation Cows: 69 biological cows.
  * **Train vs Held-Out Cow Overlap: 0** (`held_out_cow_overlap: 0`).
  * Canonical Protocol A gallery/queries: never loaded or evaluated.

---

## 7. Forensic Audit Receipts & Smoke Test Results

### Phase A: Zero-GPU Cloud Readiness Audit (`verify_readiness`)
- App ID: `ap-dJw1WALcst0rsfjbDvKQ3z`
- Profile: `hasinishrak2015`
- Hardware: minimal CPU container (cpu=2.0, memory=16384 MB, NO GPU)
- Result: **100% PASS** (all mounts, hashes, tensors, manifests, disjointness checks certified).

### Phase B: Cloud GPU Smoke Test (`smoke_test`)
- App ID: `ap-C6fr97nFN503dgBR5PHTrt`
- Profile: `hasinishrak2015`
- Hardware: NVIDIA Tesla T4 (14.75 GB VRAM)
- Subsets: BCS: 16 train / 8 val; Behavior: 8 train / 4 val; Re-ID: 16 train / 8 val.
- Runtime: 3.6s (App lifetime: ~45s)
- Train Objective Progression: 1.7787 (Epoch 1) -> 1.4754 (Epoch 2).
- Validation Objective: 1.77984 (Epoch 1) -> 1.88237 (Epoch 2). Best: Epoch 1 (`val_e1_objective = 1.77984`).
- Gradient Accumulation: Verified that all 3 tasks' backward passes accumulated directly into `model.backbone.conv1.weight.grad`.
- Checkpoint Reload Verification: **Bit-Identical** (`max_logit_diff == 0.00000000`).
- Checkpoints committed to `/mtl-checkpoints/mtl_e1_smoke/` on volume `mtl-checkpoints`.
- Local receipt saved at `artifacts/mtl_e1_smoke/mtl_e1_smoke_metrics.json`.

---

## 8. Artifact & File Registry
1. `scripts/train_mtl_e1_hard_shared.py`: Standalone PyTorch training and validation engine.
2. `scripts/modal_train_mtl_e1_hard_shared.py`: Modal cloud wrapper for profile `hasinishrak2015`.
3. `tests/test_mtl_e1_hard_shared.py`: Comprehensive local unit test suite (all 6 tests passed in 2.18s).
4. `artifacts/mtl_e1_smoke/mtl_e1_smoke_metrics.json`: Smoke test verification receipt.
5. `docs/research_log/2026-09-24_mtl_e1_hard_shared_implementation_and_smoke.md`: This official research log.

---

## 9. Immediate Action: Full Training Execution Command
Full 30-epoch training was **NOT launched**. To launch the real training run on NVIDIA L40S, execute:

```bash
modal run --detach --profile hasinishrak2015 scripts/modal_train_mtl_e1_hard_shared.py::main --epochs 30
```
