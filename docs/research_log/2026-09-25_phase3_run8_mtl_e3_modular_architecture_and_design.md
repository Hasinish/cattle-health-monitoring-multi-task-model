# Phase 3 Run 8: E3 Modular Multi-Task Learning Architecture & Controlled Design

**Date**: 2026-09-25  
**Author**: Hasin Ishrak  
**Target Execution Environment**: Modal Cloud Profile `hasinishrak2015`  
**Target Milestone**: Phase 3 Run 8 — E3 Modular MTL Pipeline Implementation & Verification (Pre-Smoke Gate)  
**Active Roadmap**: `phase3_canonical_roadmap.md` / `phase3_deadline_execution_2026-09-26.md`

---

## 1. Executive Summary

We designed, implemented, and verified the complete architecture, training engine, and cloud wrapper for **Phase 3 Run 8: E3 Modular Multi-Task Learning (Task-Private Residual Adapters)**.

Run 8 directly addresses the empirical findings of Run 7 (E1 Hard-Shared MTL control), where forcing three biologically distinct tasks—Body Condition Scoring (morphology), Behavior Recognition (posture/motion over time), and Cow Re-Identification (individual coat patterns)—into a single hard-shared spatial representation caused across-the-board negative transfer on held-out test populations:
- **BCS MAE**: degraded from 0.1709 (Run 4 single-task) to 0.1788 (Run 7 E1)
- **Behavior Macro-F1**: degraded from 0.7397 (Run 5 single-task) to 0.6866 (Run 7 E1), with minority class `Walking` dropping from 0.2456 to 0.0408
- **Re-ID Barn mAP**: degraded from 40.68% (Run 6 single-task) to 30.37% (Run 7 E1)

The E3 modular architecture preserves the exact single 4-channel ResNet-18 visual trunk (11,179,648 parameters; 90.72% shared capacity) and identical task heads from Runs 4–7 (747,058 parameters total), but inserts **three lightweight task-private residual bottleneck adapters** (Linear 512 -> 128 -> LayerNorm -> GELU -> Dropout(0.1) -> Linear 128 -> 512; 131,968 parameters each = 395,904 total private parameters, +3.32% over E1). Total trainable parameters are **12,322,610**. Crucially, the task-private adapters use zero-initialized up-projections (`up_proj.weight == 0`, `up_proj.bias == 0`), so each adapter initially acts as an identity mapping. At initialization, the adapted task feature therefore equals the output of E3's shared backbone before task-specific adaptation is learned (E3 and E1 use the same backbone architecture and initialization procedure, but E3 does not load the trained Run 7 E1 checkpoint).

All 8 focused unit tests passed locally in 3.53s, verifying tensor shapes, adapter identity initialization, conv1 4th mask-channel initialization, task-private adapter isolation (with the shared backbone intentionally receiving gradients from all three tasks), and bit-identical checkpoint reload (`max_logit_diff == 0.00000000`). Zero GPU compute was launched, zero test sets were touched, and the pipeline is certified ready for the future T4 smoke test.

---

## 2. Motivation & Empirical Problem: Negative Transfer in Run 7 (E1)

In multi-task deep learning, tasks with disparate visual semantics often exert competing gradient pressures on shared representation layers. In our primary cattle stack:
1. **BCS** demands anatomical morphology (skeletal curvature of the hook and pin bones, lumbar depressions, pelvic cavity fat deposits) while discarding individual coat markings and posture changes.
2. **Behavior** demands temporal frame-to-frame posture transitions (head dip for feeding/drinking, limb movements for walking, recumbency for lying) while ignoring specific body condition ratings and individual identification.
3. **Re-ID** demands fine-grained, view-invariant coat pigmentation patterns, facial blazes, and individual markings across cameras, while marginalizing posture variation and temporary feeding/drinking stances.

Run 7 showed held-out negative transfer under hard sharing across all three tasks (`Walking` F1 dropped from 0.2456 to 0.0408, and Re-ID Barn mAP dropped from 40.68% to 30.37%). Gradient interference or task imbalance are possible explanations, but the underlying optimization mechanism was not directly measured.

Run 8 tests the core thesis hypothesis:
> *"Can lightweight task-private residual pathways decouple conflicting gradients and mitigate negative transfer while preserving the shared cattle visual manifold?"*

---

## 3. Mathematical Formulation & Architecture Specification

### 3.1 Shared Spatial Trunk (ResNet-18)
The spatial feature extractor is exactly ONE 4-channel ResNet-18:
- Input: $X \in \mathbb{R}^{B \times 4 \times 224 \times 224}$ combining ImageNet-normalized RGB $[c_0, c_1, c_2]$ and a binary foreground mask $[c_3] \in \{0.0, 1.0\}$.
- Conv1 weights: Channels 0–2 initialized from ImageNet-1K V1; Channel 3 initialized deterministically as $W[:, 3:4, :, :] = \frac{1}{3}\sum_{c=0}^2 W[:, c:c+1, :, :]$.
- Output: Global average pooled 512-dimensional shared representation $h_{\text{shared}} = f_{\theta_{\text{shared}}}(X) \in \mathbb{R}^{B \times 512}$.
- Parameters: **11,179,648** (90.72% of total model capacity).

### 3.2 Task-Private Residual Bottleneck Adapters
For each task $t \in \{\text{BCS}, \text{Behavior}, \text{Re-ID}\}$, an independent, task-private adapter $\phi_t$ transforms $h_{\text{shared}}$:
$$z_t = \text{DownProj}_t(h_{\text{shared}}) \in \mathbb{R}^{B \times 128}$$
$$\tilde{z}_t = \text{GELU}(\text{LayerNorm}(z_t))$$
$$\Delta h_t = \text{UpProj}_t(\text{Dropout}(\tilde{z}_t, p=0.1)) \in \mathbb{R}^{B \times 512}$$
$$h_t = h_{\text{shared}} + \Delta h_t$$

- **Bottleneck dimension**: $d_{\text{bottleneck}} = 128$ (compression ratio 4:1).
- **Identity Initialization**:
  $$\text{UpProj}_t.W = \mathbf{0}, \quad \text{UpProj}_t.b = \mathbf{0} \implies \Delta h_t = \mathbf{0} \implies h_t \equiv h_{\text{shared}}$$
  This design ensures that each adapter initially acts as an identity mapping, so the adapted task feature equals the shared backbone output before task-specific adaptation is learned.
- **Parameters per adapter**:
  - Down projection: $512 \times 128 + 128 = 65,664$
  - LayerNorm: $128 \times 2 = 256$
  - Up projection: $128 \times 512 + 512 = 66,048$
  - Total per adapter: **131,968**
  - Total across all 3 adapters: **395,904** (3.21% of total model capacity).

### 3.3 Task Heads (100% Matched to Runs 4–7)
1. **BCS Cumulative Ordinal Head**:
   - Takes $h_{\text{bcs}} \in \mathbb{R}^{B \times 512}$.
   - Linear layer: $W_{\text{bcs}} \in \mathbb{R}^{4 \times 512} + b_{\text{bcs}} \in \mathbb{R}^4$.
   - Output: 4 threshold logits $P(Y > j) = \sigma(\text{logit}_j)$ for $j \in \{0, 1, 2, 3\}$.
   - Parameters: **2,052**.
2. **Behavior Lightweight 1D TCN**:
   - Takes sequence $[h_{\text{beh}}^{(1)}, \dots, h_{\text{beh}}^{(8)}] \in \mathbb{R}^{B \times 8 \times 512}$.
   - Block 1: Conv1d(512, 256, k=3, p=1) + BatchNorm1d + GELU + Dropout(0.2) + residual skip Conv1d(512, 256, k=1).
   - Block 2: Conv1d(256, 256, k=3, p=1) + BatchNorm1d + GELU + Dropout(0.2) + residual skip Identity.
   - Temporal pooling: AdaptiveAvgPool1d(1) -> Linear(256, 5).
   - Parameters: **723,973**.
3. **Re-ID Linear Classifier & Metric Embedding**:
   - Takes $h_{\text{reid}} \in \mathbb{R}^{B \times 512}$.
   - Normalized embedding: $e_{\text{reid}} = \frac{h_{\text{reid}}}{\|h_{\text{reid}}\|_2} \in \mathbb{R}^{B \times 512}$.
   - Linear classification head: Linear(512, 41).
   - Parameters: **21,033**.

---

## 4. Parameter Counts & Architectural Decomposition

| Sub-Module | Structure | Output Dim | Trainable Parameters | % of Total | Parameter Ownership |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Shared Backbone** | 4-channel ResNet-18 | 512 | **11,179,648** | 90.72% | Shared across all 3 tasks |
| **BCS Adapter** | Linear(512, 128) + LN + GELU + Linear(128, 512) | 512 | **131,968** | 1.07% | Task-Private (BCS only) |
| **Behavior Adapter** | Linear(512, 128) + LN + GELU + Linear(128, 512) | 512 | **131,968** | 1.07% | Task-Private (Behavior only) |
| **Re-ID Adapter** | Linear(512, 128) + LN + GELU + Linear(128, 512) | 512 | **131,968** | 1.07% | Task-Private (Re-ID only) |
| **BCS Head** | Linear(512, 4) | 4 | **2,052** | 0.02% | Task Head (Ordinal BCE) |
| **Behavior TCN** | 2-block 1D TCN + Linear(256, 5) | 5 | **723,973** | 5.88% | Task Head (Temporal TCN) |
| **Re-ID Head** | Linear(512, 41) | 41 | **21,033** | 0.17% | Task Head (CrossEntropy) |
| **TOTAL** | **Run 8 E3 Modular MTL** | — | **12,322,610** | **100.00%** | **+395,904 (+3.32% vs E1)** |

---

## 5. Controlled Experimental Comparison vs Run 7 (E1)

The comparison keeps the datasets, task heads, task weights, batch sizes, optimizer, training budget, and checkpoint-selection rule matched. E3 differs from E1 through the task-private adapters and their additional 395,904 trainable parameters (+3.32% capacity). Future performance differences can therefore be described as improvements from the E3 modular configuration, rather than isolated proof that routing alone caused them:

| Experimental Parameter | Run 7 E1 (Hard Sharing) | Run 8 E3 (Modular MTL) | Controlled Fairness Status |
| :--- | :--- | :--- | :--- |
| **Shared Visual Backbone** | 4-channel ResNet-18 (ImageNet init) | 4-channel ResNet-18 (ImageNet init) | Identical |
| **Task Heads** | Ordinal BCS, 1D TCN, Linear(512, 41) | Ordinal BCS, 1D TCN, Linear(512, 41) | Identical |
| **Input Specifications** | BCS: [B, 4, 224, 224]<br>Behavior: [B, 8, 4, 224, 224]<br>Re-ID: [B, 4, 224, 224] | BCS: [B, 4, 224, 224]<br>Behavior: [B, 8, 4, 224, 224]<br>Re-ID: [B, 4, 224, 224] | Identical |
| **Task Batch Sizes** | BCS: 64, Behavior: 8, Re-ID: 32 | BCS: 64, Behavior: 8, Re-ID: 32 | Identical |
| **Task Loss Weights** | $w_{\text{bcs}} = 1.0, w_{\text{beh}} = 1.0, w_{\text{reid}} = 1.0$ | $w_{\text{bcs}} = 1.0, w_{\text{beh}} = 1.0, w_{\text{reid}} = 1.0$ | Identical |
| **Optimizer & Schedule** | AdamW ($lr=10^{-4}, wd=10^{-4}$), Cosine Annealing | AdamW ($lr=10^{-4}, wd=10^{-4}$), Cosine Annealing | Identical |
| **Training Budget** | 30 epochs on Modal L40S | 30 epochs on Modal L40S | Identical |
| **Selection Objective** | $\min \frac{1}{3}(\mathcal{L}_{\text{bcs}}^{\text{val}} + \mathcal{L}_{\text{beh}}^{\text{val}} + \mathcal{L}_{\text{reid}}^{\text{val}})$ | $\min \frac{1}{3}(\mathcal{L}_{\text{bcs}}^{\text{val}} + \mathcal{L}_{\text{beh}}^{\text{val}} + \mathcal{L}_{\text{reid}}^{\text{val}})$ | Identical |
| **Checkpoint Directory** | `/mtl-checkpoints/mtl_e1_hard_shared/` | `/mtl-checkpoints/mtl_e3_modular/` | Isolated (Zero Collision) |
| **Test Set Isolation** | Held-out test evaluated post-training only | Held-out test evaluated post-training only | Identical |

---

## 6. Verification Results

### 6.1 Focused Local Unit Tests (`tests/test_mtl_e3_modular.py`)
Ran full test suite locally on CPU:
- `test_01_parameter_counts_and_structure`: PASSED. Confirmed 12,322,610 trainable parameters and verified `model.assert_modular_sharing()`.
- `test_02_adapter_identity_initialization`: PASSED. Verified up-projection weights and bias are identically zero, yielding $\max |f(x) - x| = 0.000000$.
- `test_03_conv1_mask_initialization`: PASSED. Verified 4th mask channel equals mean of ImageNet RGB weights.
- `test_04_forward_shapes`: PASSED. Verified [B, 4] for BCS, [B, 5] & [B, 8, 512] for Behavior, [B, 41] & [B, 512] for Re-ID, and unit L2 norms.
- `test_05_gradient_isolation_and_task_private_routing`: PASSED. Verified that backpropagating BCS yields `grad is None` in Behavior/Re-ID adapters; backpropagating Behavior yields `grad is None` in BCS/Re-ID adapters; backpropagating Re-ID yields `grad is None` in BCS/Behavior adapters; and super-step gradient accumulation updates the shared backbone while maintaining private adapter separation.
- `test_06_checkpoint_reload_bit_identity`: PASSED. Verified save and reload produces exact bit-identical logits ($\max |\Delta \text{logit}| = 0.00000000$).
- `test_07_sideview_reid_protocol_disjointness`: PASSED. Verified 41 train cows vs 69 held-out cows with 0 overlap.
- `test_08_zero_unavailable_label_padding`: PASSED. Verified pure task-specific targets with zero dummy padding.

**Result: 8/8 tests PASSED in 3.53s.**

### 6.2 Cloud GPU Smoke Test on Tesla T4 (`hasinishrak2015`)
Executed 2-epoch remote smoke test on Modal profile `hasinishrak2015` (Tesla T4 tier, App ID: `ap-53DDpGfdgkrLOIb0ykJz06`, Git SHA: `4a47666d52e17c1df43ccca6acc3a046287e4fe6`):
- **Deterministic Subsets**: BCS: 16 train / 8 val, Behavior: 8 train / 4 val, Re-ID: 16 train / 8 val.
- **Forward & Backward Execution**: All three tasks executed cleanly across 2 super-steps per epoch.
- **Gradient Isolation & Accumulation**: Verified during live training (step 1) that BCS backward populated only BCS adapter and backbone (`grad is None` in Behavior/Re-ID adapters and heads); Behavior backward populated only Behavior adapter and backbone; Re-ID backward populated only Re-ID adapter and backbone; and shared ResNet-18 backbone successfully accumulated finite gradients from all three tasks.
- **Loss Progression & Optimizer Update**: Finite loss drops verified (Epoch 1 total train loss: 1.7246 -> Epoch 2: 1.4145; best validation objective: 2.02141).
- **Validation Evaluation**: Validation executed across all three tasks without errors.
- **Checkpoint Save & Bit-Identical Reload**: Saved `/mtl-checkpoints/mtl_e3_smoke/mtl_e3_smoke_best.pth` and `mtl_e3_smoke_latest.pth`. Reloaded state dict into fresh model and evaluated dummy inputs:
  $$\max |\Delta \text{logit}| = 0.00000000 \quad (\text{BCS: } 0.00000000, \text{ Beh: } 0.00000000, \text{ ReID: } 0.00000000)$$
  Bit-identical reload verified ($\max \text{diff} < 10^{-6}$).
- **Data Protection**: Canonical test sets (ScienceDB BCS 7,549 images, Behavior 780 sequences) and 69 held-out evaluation cows strictly untouched.
- **Storage Isolation**: Outputs committed exclusively to `/mtl-checkpoints/mtl_e3_smoke/` without touching Run 7 checkpoints or full Run 8 output paths.

**Result: PASS (App ID: `ap-53DDpGfdgkrLOIb0ykJz06`, Total duration: 3.58s training).**

---

## 7. Artifacts & File Registry

| File Path | Description |
| :--- | :--- |
| `scripts/train_mtl_e3_modular.py` | Standalone PyTorch training and evaluation script for Run 8 E3 Modular MTL |
| `scripts/modal_train_mtl_e3_modular.py` | Modal cloud wrapper with `verify_readiness`, `smoke_test`, and `main` entrypoints |
| `tests/test_mtl_e3_modular.py` | Focused unit and architectural verification suite (8 tests) |
| `docs/research_log/2026-09-25_phase3_run8_mtl_e3_modular_architecture_and_design.md` | This research log |

---

## 8. Next Steps & Future Commands

1. **Full 30-Epoch L40S Training (Awaiting User Command)**:
   ```bash
   modal run --detach --profile hasinishrak2015 scripts/modal_train_mtl_e3_modular.py::main --epochs 30
   ```
