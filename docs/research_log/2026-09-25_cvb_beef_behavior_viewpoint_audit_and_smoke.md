# CVB + Kaggle Beef Behavior Viewpoint Feasibility Audit & Cloud Smoke Certification

## 1. Executive Summary
To evaluate whether the certified frozen real-cattle 3-class viewpoint model (front, side, rear) transfers reliably to the primary Phase 3 Run 5 Behavior stack (CVB + Kaggle Beef, T=8 sequences, cattle-centered RGB crops + binary masks), we conducted a forensic feasibility audit and controlled 2-epoch smoke certification on Modal profile `hasinishrak2015` (NVIDIA Tesla T4).

### Key Empirical Findings:
1. **Provenance & Checkpoint Integrity**: Transferred `viewpoint_resnet18_real_best.pth` from `tigerwood693` (`viewpoint-checkpoints`) to `hasinishrak2015` (`mtl-checkpoints/viewpoint_aux/`). Verified bit-identical SHA-256: `a93b9232e640388447f994cbffe93e6115d1af18e9188aa32cd117aeca454d1a` (134,275,929 bytes).
2. **Transfer Sanity Audit (N=54 sequences, 432 frames, train/val only)**:
   - **Predicted Distribution**: 41.90% front, 43.29% side, 14.81% rear. Non-degenerate: **PASS**.
   - **Confidence & Entropy**: Mean confidence: **0.7402** (median 0.7510); Mean entropy: **0.6122 nats** (maximum entropy for 3 classes is 1.0986 nats); Low-confidence rate (<0.50): **12.04%**.
   - **Temporal Consistency**: Adjacent frame transition agreement: **82.54%**; Temporally constant sequence rate: **57.41%** (31/54 sequences with zero viewpoint flips across all 8 frames); Mean class switches: **1.22** per sequence.
   - **Source / Camera Shortcut Risk**: **HIGH (Total Variation Distance = 0.7646)**. Predictions are sharply split by source dataset: CVB (indoor barn CCTV) predicts 7.9% front, 70.8% side, 21.2% rear, whereas Kaggle Beef (pasture/feedlot camera angles) predicts 84.4% front, 8.8% side, 6.8% rear.
   - *Transfer Sanity Disclaimer*: These are transfer sanity indicators, NOT viewpoint accuracy, as no target ground-truth viewpoint annotations exist on CVB or Kaggle Beef.
3. **Run 5 + Viewpoint Controlled Smoke Test**:
   - Model: Frozen ResNet-18 (RGB, 3 classes, zero gradients asserted) -> 3 probabilities -> MLP(3 -> 16 -> 16, 400 params) -> fused per-frame with 4-channel ResNet-18 visual feature (512-D) -> 528-D -> TCN (in_features=528, hidden_dim=256, 5 classes).
   - Total Trainable Parameters: **11,920,405** (+16,784 / +0.14% vs Run 5 baseline). Frozen Parameters: **11,178,051**. Total: **23,098,456**.
   - 2-Epoch Smoke: Train loss dropped monotonically from **1.8819 -> 0.6371** (-1.2448). Validation accuracy: 75.0%.
   - Checkpoint Reload Determinism: **PASS** (`max_logit_diff == 0.00000000`).
   - Test Isolation: **PASS** (held-out test split strictly untouched, 0 evaluations).
4. **Roadmap Recommendation**: While technically usable from an inference standpoint, full Behavior + Viewpoint training is **deferred / not prioritized** due to the severe camera/source shortcut risk (TVD = 0.7646) that could encourage the temporal model to exploit camera angle rather than anatomical dynamics.

---

## 2. Context & Motivation
In Phase 3 Run 5, cattle behavior recognition is modeled using a 4-channel ResNet-18 spatial backbone coupled to a 1D Temporal Convolutional Network (TCN) operating over T=8 chronological frames per sequence. Run 5 achieved 74.43% balanced accuracy and 0.4430 test loss on the matched test evaluation.

Before committing GPU compute to an ablation that injects viewpoint conditioning into the Behavior representation, the user mandated:
1. Verify the physical presence and bit-identity of the certified real-cattle viewpoint model on Modal profile `hasinishrak2015`.
2. Perform a transfer sanity audit on authentic train/val CVB and Kaggle Beef sequences to inspect prediction confidence, temporal stability, non-degeneracy, and potential dataset shortcut learning.
3. If technically usable, build and smoke-certify a controlled Run 5 + Viewpoint fusion architecture.
4. Strictly respect the stop condition: do not launch full 30-epoch training.

---

## 3. Viewpoint Model Provenance & Transfer Audit

### 3.1 Checkpoint Verification & Bit Identity
The certified 3-class real-cattle viewpoint model (`viewpoint_resnet18_real_best.pth`) was previously trained on real cattle crops from Moo/Synthetic transfer experiments and stored on Modal volume `viewpoint-checkpoints` under profile `tigerwood693`.
- **Target Location**: Volume `mtl-checkpoints` at `/checkpoints/viewpoint_aux/viewpoint_resnet18_real_best.pth` on profile `hasinishrak2015`.
- **File Size**: 134,275,929 bytes (~128.05 MB).
- **Verified SHA-256**: `a93b9232e640388447f994cbffe93e6115d1af18e9188aa32cd117aeca454d1a`.
- **Architecture**: Standard torchvision ResNet-18 with modified `fc = nn.Linear(512, 3)` predicting classes: `0: front`, `1: side`, `2: rear`.

### 3.2 Audit Sampling & Boundary Conditions
- **Data Scope**: Staged authentic Run 5 sequences (`/mtl-data/behavior/{sample_id}/frame_{t:02d}.jpg`, `mask_{t:02d}.png`) on volume `mtl-data`.
- **Splits**: Strictly train and validation splits (`retained_train.csv`, `retained_val.csv`). Held-out test split was isolated with zero access.
- **Stratified Sample**: Deterministic random seed `2026`. Exactly 6 sequences (4 train, 2 val) sampled per available source x class cell. Across 9 cells (5 CVB classes + 4 Beef classes; Walking exists only in CVB), this produced 54 sequences (432 frames).

---

## 4. Quantitative Audit Findings

### 4.1 Overall Transfer Sanity Summary
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Total Sequences Evaluated** | **54** | Deterministic stratified sample (seed 2026, 6 per cell) |
| **Total Frames Evaluated** | **432** | T=8 frames per sequence |
| **Predicted Distribution (Front)** | **41.90%** (181/432) | Non-degenerate; active distribution across all classes |
| **Predicted Distribution (Side)** | **43.29%** (187/432) | Plausible side predominance in barn side-angle setups |
| **Predicted Distribution (Rear)** | **14.81%** (64/432) | Lower frequency, consistent with cattle orientations |
| **Mean Softmax Confidence** | **0.7402** | Median: 0.7510; indicates decisive predictions |
| **Mean Prediction Entropy** | **0.6122 nats** | Well below uniform maximum of ln(3) = 1.0986 nats |
| **Low-Confidence Rate (<0.50)** | **12.04%** (52/432) | Only 12% of frames show ambiguous viewpoint outputs |
| **Mean Temporal Stability** | **82.54%** | Fraction of adjacent frame pairs with identical viewpoint |
| **Temporally Constant Sequences** | **57.41%** (31/54) | Exactly 0 viewpoint flips across all 8 frames |
| **Mean Class Switches per Sequence**| **1.22** | Minor boundary jitter; robust overall sequence consistency |
| **Non-Degenerate Verdict** | **PASS** | No class collapse observed |
| **Source Shortcut Risk Level** | **HIGH** | Total Variation Distance (TVD) = 0.7646 between sources |

---

### 4.2 Cross-Domain Source Shortcut Risk: CVB vs. Kaggle Beef
Comparing predictions by source dataset reveals an acute shortcut risk:
| Metric | CVB (Barn CCTV) | Kaggle Beef (Pasture/Feedlot) | Total Variation Distance |
| :--- | :--- | :--- | :--- |
| **Sequences / Frames** | 30 seqs / 240 frames | 24 seqs / 192 frames | - |
| **Predicted Front** | **7.92%** (19/240) | **84.38%** (162/192) | **+76.46%** in Beef |
| **Predicted Side** | **70.83%** (170/240) | **8.85%** (17/192) | **+61.98%** in CVB |
| **Predicted Rear** | **21.25%** (51/240) | **6.77%** (13/192) | **+14.48%** in CVB |
| **Mean Confidence** | 0.6942 | 0.7978 | Higher confidence on Beef |
| **Mean Entropy** | 0.7222 nats | 0.4746 nats | Lower entropy on Beef |
| **Low Confidence (<0.50)** | 13.75% | 9.90% | Low ambiguity in both |
| **Temporal Stability** | 82.38% | 82.74% | High adjacent frame agreement |
| **Temporally Constant Seqs** | 53.33% (16/30) | 62.50% (15/24) | Over half 100% constant |
| **Mean Class Switches** | 1.23 | 1.21 | Identical temporal dynamics |

**Forensic Takeaway on Shortcut Learning**:
In the CVB dataset, cameras are mounted overhead and to the side of feeding bunks and pens, leading the viewpoint model to predict 70.8% `side`. In the Kaggle Beef dataset, mobile/pasture cameras frequently face approaching cattle, causing the viewpoint model to predict 84.4% `front`.
Because the class balance differs between CVB and Beef (e.g. Walking is CVB-only; Drinking has different pen structures), giving the TCN direct access to viewpoint features risks enabling it to recognize the **source dataset / camera angle** rather than intrinsic cattle motion dynamics.

---

### 4.3 Results by Canonical Behavior Class
| Canonical Class | Seqs / Frames | Front % | Side % | Rear % | Mean Conf | Entropy | Low Conf % | Temp Stability | Constant % | Switches |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standing** | 12 / 96 | 39.58% | 42.71% | 17.71% | 0.6703 | 0.7302 | 16.67% | 76.19% | 58.33% | 1.67 |
| **Lying** | 12 / 96 | 41.67% | 46.88% | 11.46% | 0.6627 | 0.7661 | 15.62% | 79.76% | 41.67% | 1.42 |
| **Feeding** | 12 / 96 | 47.92% | 27.08% | 25.00% | 0.7594 | 0.5742 | 9.38% | 78.57% | 50.00% | 1.50 |
| **Drinking** | 12 / 96 | 51.04% | 46.88% | 2.08% | 0.8844 | 0.3338 | 5.21% | 94.05% | 83.33% | 0.42 |
| **Walking** | 6 / 48 | 16.67% | 62.50% | 20.83% | 0.7085 | 0.7009 | 14.58% | 85.71% | 50.00% | 1.00 |

---

### 4.4 Results Across All 9 Source x Class Cells
| Cell Identifier | Source | Class | Seqs | Front % | Side % | Rear % | Mean Conf | Entropy | Temp Stab | Constant % | Switches |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `cvb_Standing` | CVB | Standing | 6 | 10.42% | 66.67% | 22.92% | 0.6668 | 0.7694 | 92.86% | 83.33% | 0.50 |
| `cvb_Lying` | CVB | Lying | 6 | 6.25% | 77.08% | 16.67% | 0.6289 | 0.8501 | 80.95% | 50.00% | 1.33 |
| `cvb_Feeding` | CVB | Feeding | 6 | 4.17% | 54.17% | 41.67% | 0.6577 | 0.7868 | 64.29% | 16.67% | 2.50 |
| `cvb_Drinking` | CVB | Drinking | 6 | 2.08% | 93.75% | 4.17% | 0.8092 | 0.5039 | 88.10% | 66.67% | 0.83 |
| `cvb_Walking` | CVB | Walking | 6 | 16.67% | 62.50% | 20.83% | 0.7085 | 0.7009 | 85.71% | 50.00% | 1.00 |
| `beef_Standing` | Beef | Standing | 6 | 68.75% | 18.75% | 12.50% | 0.6739 | 0.6910 | 59.52% | 33.33% | 2.83 |
| `beef_Lying` | Beef | Lying | 6 | 77.08% | 16.67% | 6.25% | 0.6966 | 0.6822 | 78.57% | 33.33% | 1.50 |
| `beef_Feeding` | Beef | Feeding | 6 | 91.67% | 0.00% | 8.33% | 0.8612 | 0.3617 | 92.86% | 83.33% | 0.50 |
| `beef_Drinking` | Beef | Drinking | 6 | 100.00%| 0.00% | 0.00% | 0.9595 | 0.1636 | 100.00%| 100.00%| 0.00 |

---

## 5. Controlled Run 5 + Viewpoint Model Architecture & Smoke Certification

### 5.1 Model Architecture & Tensor Flow
The model preserves the entire Run 5 perception trunk and temporal TCN, integrating viewpoint as an auxiliary conditioning channel:
1. **Per-Frame Visual Extraction**:
   - Input: Cattle-centered RGB crop + binary mask `[B, T, 4, 224, 224]`.
   - Run 5 ResNet-18 Backbone: `ResNet18-4ch` (conv1 modified to 4 channels; 11,179,648 parameters).
   - Spatial Feature: `[B*T, 512]`.
2. **Auxiliary Viewpoint Conditioning**:
   - Input: RGB channels only `[B*T, 3, 224, 224]` with standard ImageNet normalization.
   - Frozen Viewpoint Backbone: Pretrained 3-class ResNet-18 (11,178,051 parameters, `requires_grad=False`, `eval()`).
   - Logits: `[B*T, 3]` -> `F.softmax(dim=-1)` -> 3-class probabilities `[B*T, 3]`.
   - Viewpoint Projection MLP:
     ```
     Linear(3, 16) -> LayerNorm(16) -> ReLU() -> Linear(16, 16) -> LayerNorm(16)
     ```
     (400 parameters).
3. **Per-Frame Feature Fusion**:
   - Concatenation: `torch.cat([visual_feat, vp_feat], dim=-1)` -> `[B*T, 528]`.
   - Reshaped to temporal sequence: `[B, 528, T]` (with T=8).
4. **Temporal Modeling**:
   - 1D Temporal Convolutional Network (TCN):
     - Conv1d Block 1: `Conv1d(528, 256, kernel_size=3, padding=1) -> BatchNorm1d(256) -> ReLU() -> Dropout(0.2)`
     - Conv1d Block 2: `Conv1d(256, 256, kernel_size=3, padding=1) -> BatchNorm1d(256) -> ReLU() -> Dropout(0.2)`
     - Global Temporal Pooling: `AdaptiveAvgPool1d(1)` -> `[B, 256]`.
     - Behavior Classification Head: `Linear(256, 5)` -> `[B, 5]`.
     - TCN Parameters: 740,357 parameters (+16,384 params vs Run 5 due to input channel expansion 512 -> 528).

### 5.2 Parameter Census
| Module | Trainable Parameters | Frozen Parameters | Total Parameters | Delta vs Run 5 Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Visual 4-channel ResNet-18** | 11,179,648 | 0 | 11,179,648 | 0 (identical) |
| **Viewpoint ResNet-18** | 0 | 11,178,051 | 11,178,051 | +11,178,051 (frozen) |
| **Viewpoint MLP** | 400 | 0 | 400 | +400 |
| **Temporal TCN (528 -> 5)** | 740,357 | 0 | 740,357 | +16,384 (+2.26%) |
| **Total** | **11,920,405** | **11,178,051** | **23,098,456** | **+16,784 (+0.14%)** |

### 5.3 Cloud Smoke Test Results (Modal Tesla T4)
Executed under Modal profile `hasinishrak2015` (App ID `ap-zvPm0AfMAK9vPE8pQpsuWj`):
- **Batch Size**: 4 sequences (32 frames total).
- **Epochs**: 2.
- **Viewpoint Gradient Invariance**: **VERIFIED**. `param.grad is None` across all 11,178,051 viewpoint parameters after `loss.backward()`.
- **Loss Progression**:
  - Epoch 1 Train Loss: **1.8819**
  - Epoch 2 Train Loss: **0.6371** (-1.2448 drop)
  - Epoch 2 Val Accuracy: **75.00%** (12/16 correct on val sample)
- **Checkpoint Save / Reload Determinism**:
  - Saved checkpoint to `/checkpoints/smoke_behavior_viewpoint.pth`.
  - Reloaded model in separate instance and evaluated on identical batch.
  - `max_logit_difference == 0.00000000` (**PASS**).
- **Test Set Isolation**:
  - `test.csv` was never opened, loaded, or evaluated. **PASS**.

---

## 6. Artifact Registry
| File Path | Description |
| :--- | :--- |
| `scripts/audit_cvb_beef_viewpoint_transfer.py` | Standalone audit script computing viewpoint transfer sanity metrics across stratified sequences |
| `scripts/train_cvb_beef_behavior_viewpoint.py` | Behavior + Viewpoint fused model class and 2-epoch smoke test training loop |
| `scripts/modal_audit_and_smoke_behavior_viewpoint.py` | Modal runner executing audit and smoke test on `hasinishrak2015` with persistent volume mount |
| `artifacts/behavior_viewpoint_audit/behavior_viewpoint_frame_results.csv` | Frame-level predictions, confidences, and entropy across 432 frames |
| `artifacts/behavior_viewpoint_audit/behavior_viewpoint_sequence_summary.csv` | Sequence-level distributions, temporal stability, and switch counts across 54 sequences |
| `artifacts/behavior_viewpoint_audit/behavior_viewpoint_transfer_sanity_metrics.json` | Comprehensive machine-readable transfer sanity metrics (overall, by source, by class, by cell) |
| `artifacts/behavior_viewpoint_audit/behavior_viewpoint_all_classes_contact_sheet.jpg` | Visual contact sheet depicting predicted viewpoints across representative sequences |
| `artifacts/behavior_viewpoint_smoke/behavior_viewpoint_smoke_metrics.json` | Exact architectural shapes, parameter census, and 2-epoch smoke training progression |
| `docs/audits/assets/behavior_viewpoint_audit/behavior_viewpoint_all_classes_contact_sheet.jpg` | Version-controlled copy of the visual contact sheet |

---

## 7. Next Steps & Recommendation
1. **Full Ablation Recommendation**: **DEFER / DO NOT LAUNCH**.
   - Although the viewpoint model transfers with high confidence (0.74) and temporal consistency (82.5%), the severe source shortcut risk (TVD = 0.7646 between CVB side-views and Beef front-views) poses a high risk of confounding temporal behavior learning.
   - Run 5 baseline already achieves 74.43% balanced accuracy (+3.13% over Run 2) without viewpoint.
2. **Canonical Roadmap Integrity**: Preserve the 13-step roadmap without deviation. Keep Run 7 (E1 Hard-Shared MTL) and Run 8 (E3 Modular MTL) as the immediate primary deliverables.
