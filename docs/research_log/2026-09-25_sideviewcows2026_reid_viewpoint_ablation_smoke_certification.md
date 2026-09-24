# SideViewCows2026 Re-ID + Viewpoint Ablation Cloud Smoke Certification (dryousufmozumder)

## 1. Executive Summary
To evaluate whether viewpoint priors from a frozen, certified real-cattle classifier improve the oracle-segmented visual Re-ID representation established in Phase 3 Run 6, we implemented, audited, and smoke-certified a controlled Re-ID + Viewpoint ablation on Modal profile `dryousufmozumder`. The certified checkpoint `viewpoint_resnet18_real_best.pth` (86.26% test accuracy, 84.96% balanced accuracy on its source domain) was transferred directly cloud-to-cloud from `tigerwood693` (`viewpoint-checkpoints`) to `dryousufmozumder` (`reid-checkpoints/viewpoint_aux/`), verifying bit-identical SHA-256 provenance (`a93b9232e640388447f994cbffe93e6115d1af18e9188aa32cd117aeca454d1a`). A cross-domain transfer sanity audit on 50 representative Protocol D train/val crops from the 41 training cows revealed predicted class distributions of 34.0% front, 26.0% side, and 40.0% rear, with a mean softmax confidence of 0.7014 (median 0.6829), mean entropy of 0.6875 nats, and a 14.0% low-confidence rate (<50%). The fused architecture couples the 4-channel ResNet-18 visual trunk (512-D) with a Viewpoint MLP (3 -> 16-D) into a unit L2-normalized 528-D embedding feeding a `Linear(528, 41)` identity classifier. Total trainable parameters are 11,201,737 (+1,056 / +0.0094% vs Run 6), with the 11,178,051-parameter viewpoint network strictly frozen. Remote readiness and a 2-epoch smoke test on Tesla T4 (`ap-zLS0oDRARAgklDTDyJ7GIS`) passed all checks: verified tensor shapes, finite loss drop (3.9830 -> 3.3459), zero viewpoint gradients, bit-identical reload (`max_logit_diff == 0.00000000`), and zero access to the 69 Protocol A held-out cows. In strict adherence to the user stop condition, full 30-epoch training was NOT launched.

---

## 2. Context & Scientific Motivation
In Phase 3 Run 6, providing ground-truth cow silhouette masks to the ResNet-18 spatial backbone produced massive gains in cross-setting retrieval over the Run 3 RGB baseline:
- Snapshots -> Parlor: Rank-1 rose from 38.88% to 62.93% (+24.05% absolute; +61.9% relative surge).
- Barn -> Parlor: Rank-1 rose from 58.64% to 63.90% (+5.26% absolute).

However, Run 6 operates without explicit 3D orientation awareness. In multi-camera surveillance and pasture tracking, cow coat patterns appear radically distorted under perspective changes (e.g. lateral flank vs cranial facial markings vs caudal rump). The scientific question is:
**Does conditioning the identity representation on a continuous viewpoint probability prior help the model disentangle pose/angle variation from biological coat identity, or does domain shift in viewpoint predictions introduce noise?**

### Controlled Experimental Rules
1. Exact Run 6 visual representation: same cow crop (GT mask bbox + 5% proportional margin), same 224x224 resolution, same 4-channel `[R, G, B, Mask]` input tensor.
2. Exact Run 6 partitions: strictly 41 training cows from Protocol D (12,753 train, 2,683 val), seed=2026.
3. Frozen viewpoint network: certified weights from `viewpoint_resnet18_real_best.pth`, zero fine-tuning, zero gradients.
4. No metadata leakage: camera IDs, subset names, recording IDs, and cow IDs are strictly excluded from viewpoint features.
5. Strict evaluation isolation: the 69 held-out Protocol A cows remain untouched during preparation and smoke testing.

---

## 3. Step A — Checkpoint Transfer & Provenance
The certified real-cattle viewpoint model was transferred directly from source to destination volume without modifying or deleting the source:
- **Source Profile**: `tigerwood693`
- **Source Volume**: `viewpoint-checkpoints`
- **Source Path**: `viewpoint_real_finetune/viewpoint_resnet18_real_best.pth`
- **Destination Profile**: `dryousufmozumder`
- **Destination Volume**: `reid-checkpoints`
- **Destination Path**: `viewpoint_aux/viewpoint_resnet18_real_best.pth`
- **Size**: 134,275,929 bytes (128.06 MB)
- **SHA-256 Provenance**: `a93b9232e640388447f994cbffe93e6115d1af18e9188aa32cd117aeca454d1a`
- **Identity Verification**: Bit-for-bit SHA-256 identity verified on both source download and destination read-back (`assert dst_hash == src_hash`).

---

## 4. Step B — Physical Audit on `dryousufmozumder`
Executed remote physical readiness verification (`verify_readiness_remote`, App `ap-LAFkI98X5Prjto03qXyGk9`, Tesla T4):
- **Volume `sideview-data`**: Mounted at `/data`. Verified all 80,260 RGB images and 80,260 GT masks across 110 unique cow identities:
  - `parlor`: 54,393 RGB images, 54,393 masks (54 cameras)
  - `barn`: 25,260 RGB images, 25,260 masks (18 cameras)
  - `snapshots`: 607 RGB images, 607 masks (handheld cameras)
- **Protocol Integrity**:
  - Protocol A: 41 training cows (`setting_role == "train"`), 69 evaluation cows (`setting_role != "train"`), 0 overlap.
  - Protocol D: 12,753 train images, 2,683 val images across the 41 training cows.
- **Volume `reid-checkpoints`**: Mounted at `/checkpoints`. Writable probe test passed.
- **Transferred Checkpoint**: `/checkpoints/viewpoint_aux/viewpoint_resnet18_real_best.pth` exists, size=134,275,929 bytes, SHA-256 verified, PyTorch load into `models.resnet18(num_classes=3)` succeeded with 0 missing and 0 unexpected keys.

---

## 5. Step C — SideView Viewpoint Cross-Domain Transfer Sanity Audit
Evaluated the frozen viewpoint model on 50 deterministically sampled Protocol D crops (25 train, 25 val from 41 cows, seed=2026, held-out cows: 0):
- **Target Crop**: Exact Run 6 GT-mask-derived bbox + 5% proportional margin, resized to 224x224 RGB, ImageNet normalized.
- **Crucial Clarification**: SideViewCows2026 does NOT contain ground-truth viewpoint annotations. Camera names and subset identifiers must not be treated as viewpoint labels. This audit reports transfer behavior, confidence, and entropy as a **cross-domain transfer sanity check**, not an accuracy benchmark.

### Quantitative Sanity Metrics
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| Evaluated Samples | 50 | 25 Train, 25 Val across 41 Protocol-D training cows |
| Predicted Front | **34.0%** (17 / 50) | Balanced representation across classes |
| Predicted Side | **26.0%** (13 / 50) | Consistent lateral body orientation detection |
| Predicted Rear | **40.0%** (20 / 50) | Plausible parlor rear/chute orientations |
| Mean Softmax Confidence | **0.7014** (std: 0.1743) | Substantial certainty on most crops |
| Median Softmax Confidence | **0.6829** | Consistent confidence distribution |
| Min / Max Confidence | 0.3581 / 0.9956 | Dynamic range from ambiguous to unambiguous |
| Mean Shannon Entropy | **0.6875 nats** | Well below max uniform entropy (ln(3) = 1.0986 nats) |
| Low-Confidence Rate (<50%) | **14.0%** (7 / 50) | Minority of ambiguous or borderline crops |
| Low-Confidence Rate (<60%) | **30.0%** (15 / 50) | Reflects oblique 3/4 camera angles |

### Qualitative Observations
- **High-Confidence Cases (Confidence > 0.85)**: Clear side-profile stanchion views (e.g. `442_0219.jpg`: Side 98.4%, H=0.09) and clear rear milking positions (e.g. `209_0086.jpg`: Rear 96.6%, H=0.17).
- **Borderline / Low-Confidence Cases (Confidence < 0.50)**: Ambiguous oblique 45-degree angles where the cow is entering the stall or turning (e.g. `565_0083.jpg`: Rear 39.7%, Front 36.2%, Side 24.1%, H=1.08; `405_0028.jpg`: Side 40.1%, Front 36.7%, Rear 23.3%, H=1.07). These correctly produce diffuse probability distributions rather than extreme overconfident errors.
- **Contact Sheet**: Generated and saved to `artifacts/reid_viewpoint_ablation/viewpoint_transfer_contact_sheet.jpg` displaying all 50 crops with predicted class, probabilities, confidence, entropy, and individual IDs.

---

## 6. Step D — Re-ID + Viewpoint Architecture & Parameter Verification
- **Visual Spatial Trunk**: 4-channel ResNet-18 identical to Run 6 (`conv1` expanded for `[R, G, B, Mask]`, 11,179,648 trainable parameters). Extracts 512-D visual feature.
- **Frozen Viewpoint Classifier**: Pretrained ResNet-18 (`viewpoint_resnet18_real_best.pth`, 11,178,051 parameters, strictly `requires_grad=False`, evaluated under `torch.no_grad()`). Outputs 3-class probability vector `p = [p_front, p_side, p_rear]`.
- **Viewpoint MLP**: `Linear(3, 16) -> LayerNorm(16) -> ReLU() -> Linear(16, 16) -> LayerNorm(16)` (400 trainable parameters).
- **Fusion**: Concatenates visual 512-D feature and viewpoint 16-D embedding into 528-D representation.
- **Retrieval Metric Embedding**: Unit L2-normalized 528-D vector (`norm_embedding = F.normalize(fused, p=2, dim=1)`).
- **Identity Classifier Head**: `nn.Linear(528, 41)` (21,689 trainable parameters).

### Exact Parameter Comparison
| Model Component | Phase 3 Run 6 (Baseline) | Run 6 + Viewpoint Ablation | Difference |
| :--- | :--- | :--- | :--- |
| Visual Spatial Trunk | 11,179,648 | 11,179,648 | 0 |
| Viewpoint MLP (3 -> 16) | 0 | 400 | +400 |
| Identity Classifier Head | 21,033 (`Linear(512, 41)`) | 21,689 (`Linear(528, 41)`) | +656 |
| **Total Trainable Parameters** | **11,200,681** | **11,201,737** | **+1,056 (+0.0094%)** |
| Frozen Viewpoint Network | 0 | 11,178,051 | +11,178,051 |
| **Total Model Parameters** | **11,200,681** | **22,379,788** | **+11,179,107** |

---

## 7. Step E — Protocol Preservation & Held-Out Isolation
- **Training Population**: Strictly 41 biological cows (`setting_role == "train"` in Protocol A).
- **Canonical Partitions**: Protocol D closed-set train (12,753 images) and val (2,683 images).
- **Held-Out Isolation**: The 69 Protocol A evaluation cows (36,811 parlor gallery images, 25,260 barn queries, 607 snapshots queries) were NOT accessed, loaded, or evaluated during readiness, sanity auditing, or smoke testing.
- **Evaluation Gate**: Retrieval metrics (Barn->Parlor and Snapshots->Parlor Rank-1/5/10 and mAP) remain strictly deferred until post-training evaluation.

---

## 8. Step F — Smoke Certification Results
Executed on Modal profile `dryousufmozumder` (`smoke_test_remote`, App `ap-zLS0oDRARAgklDTDyJ7GIS`, Tesla T4 GPU):
1. **Full-Data Readiness**: All 80,260 images/masks and protocol splits verified.
2. **Tensor Shapes Verified**:
   - Input batch: `[64, 4, 224, 224]`
   - Visual feature: `[64, 512]`
   - Viewpoint probabilities: `[64, 3]`
   - Viewpoint embedding: `[64, 16]`
   - Fused raw feature: `[64, 528]`
   - Normalized embedding: `[64, 528]` (min L2 norm: 0.99999988, max: 1.00000000)
   - Logits: `[64, 41]`
3. **Training Dynamics**:
   - Epoch 1: Train Loss = 3.9830, Train Acc = 6.25%
   - Epoch 2: Train Loss = 3.3459, Train Acc = 9.38% (finite loss drop: -0.6371)
4. **Zero Viewpoint Gradients**: Programmatically asserted `param.grad is None` across all parameters of `viewpoint_model` after backward pass (`model.assert_frozen_viewpoint()`).
5. **Checkpoint Reload Determinism**:
   - Logits before reload vs after reload: `max_logit_difference == 0.00000000`
   - Bit-identical reproduction: `True`
6. **Held-Out Data Isolation**:
   - Held-out Protocol A images loaded: `0`
   - Protocol A evaluation executed: `False`
7. **STOP CONDITION RESPECTED**: Full 30-epoch training was NOT launched.

---

## 9. Artifact Registry
- `scripts/train_sideview_reid_viewpoint.py` — Core Re-ID + Viewpoint ablation training engine
- `scripts/modal_train_sideview_reid_viewpoint.py` — Modal cloud wrapper targeting `dryousufmozumder`
- `scripts/audit_sideview_viewpoint_transfer.py` — Standalone cross-domain viewpoint transfer sanity audit script
- `scripts/transfer_viewpoint_to_dryousuf.py` — Inter-profile certified viewpoint checkpoint transfer utility
- `artifacts/reid_viewpoint_ablation/reid_viewpoint_smoke_metrics.json` — Official smoke test verification receipt
- `artifacts/reid_viewpoint_ablation/viewpoint_transfer_sanity_metrics.json` — Transfer sanity audit quantitative summary
- `artifacts/reid_viewpoint_ablation/viewpoint_transfer_samples.csv` — Per-sample predictions, confidences, and entropies
- `artifacts/reid_viewpoint_ablation/viewpoint_transfer_contact_sheet.jpg` — High-resolution 50-crop visual audit sheet
- `artifacts/reid_viewpoint_ablation/run6_vs_viewpoint_design_note.md` — Controlled comparison documentation

---

## 10. Scientific Limitations Before Launch
1. **Cross-Domain Viewpoint Transfer Uncertainty**: While the viewpoint model demonstrated plausible distribution on SideView parlor crops (34% front, 26% side, 40% rear; mean conf 0.7014), the model was trained on outdoor/pasture and loose pen crops (`self_clean_v1_rtdetr_crop/`), not indoor herringbone/parallel milking parlors with metal stanchion bars.
2. **Ambiguity on Oblique Perspectives**: 30% of crops had confidence < 0.60, primarily due to oblique 3/4 views that do not strictly conform to the 3-class front/side/rear taxonomy.
3. **Identity Shortcut vs Viewpoint Invariance**: If cow identity correlates spuriously with camera angle during training (e.g. certain cows frequenting specific parlor stalls), the model might inadvertently use viewpoint as an identity shortcut. However, the frozen continuous representation and low added parameter count (+1,056) mitigate this risk compared to discrete camera ID injection.

---

## 11. Next Steps
- Await manual execution of full 30-epoch ablation run by user via:
  ```bash
  modal run --detach --profile dryousufmozumder scripts/modal_train_sideview_reid_viewpoint.py::main --epochs 30 --batch-size 64
  ```
