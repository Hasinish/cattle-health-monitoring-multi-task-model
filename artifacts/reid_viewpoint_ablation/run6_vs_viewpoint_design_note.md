# Run 6 vs. Re-ID + Viewpoint Ablation Controlled Design Note

## 1. Experimental Objective
This ablation evaluates whether injecting viewpoint priors from a frozen, certified real-cattle viewpoint classifier (`viewpoint_resnet18_real_best.pth`, trained on authentic cattle crops to classify `front`, `side`, and `rear`) improves or alters the individual identification representation learned by the Phase 3 Run 6 perception baseline on SideViewCows2026.

---

## 2. Controlled Comparison Matrix

| Component | Phase 3 Run 6 (Baseline) | New Condition: Re-ID + Viewpoint | Controlled Match Rationale |
| :--- | :--- | :--- | :--- |
| **Input Crop Geometry** | Official GT mask bounding box + 5% proportional margin | Exact same GT mask bounding box + 5% proportional margin | Guarantees identical cow crop coordinates and spatial context |
| **Input Resolution** | 224 x 224 pixels | 224 x 224 pixels | Bilinear interpolation matching Run 6 |
| **Visual Input Channels** | 4 channels: `[R, G, B, Mask]` (ImageNet RGB + {0, 1} binary mask) | 4 channels: `[R, G, B, Mask]` (ImageNet RGB + {0, 1} binary mask) | Identical visual perception representation |
| **Visual Spatial Backbone** | ResNet-18 (`conv1` expanded to 4 channels; 11,179,648 params) | ResNet-18 (`conv1` expanded to 4 channels; 11,179,648 params) | Matched capacity and initialization |
| **Visual Feature Dim** | 512-D (post AdaptiveAvgPool2d + flatten) | 512-D (post AdaptiveAvgPool2d + flatten) | Bit-matched visual trunk output |
| **Auxiliary Prior** | None | Frozen ResNet-18 Viewpoint Classifier (3 classes: front, side, rear) | Isolated ablation of viewpoint information |
| **Auxiliary Prior Input** | N/A | Channels 0-2 (`visual_input[:, :3, :, :]`) | Viewpoint evaluated strictly on the cow crop |
| **Auxiliary Prior Gradients** | N/A | Strictly FROZEN (`requires_grad=False`, `torch.no_grad()`, zero grads) | Prevents Re-ID loss from distorting certified viewpoint prior |
| **Viewpoint Projection** | N/A | `ViewpointMLP` (3 -> 16 -> 16, LayerNorm, ReLU; 400 params) | Compact, continuous representation without dimension explosion |
| **Fused Representation** | 512-D visual feature | 528-D concatenation (`[visual_512, vp_16]`) | Fused representation |
| **Retrieval Metric Embedding** | Unit L2-normalized 512-D vector | Unit L2-normalized 528-D vector | Cosine similarity via chunked dot product |
| **Identity Classifier Head** | `nn.Linear(512, 41)` (21,033 params) | `nn.Linear(528, 41)` (21,689 params) | Classification over 41 training cows |
| **Trainable Parameters** | **11,200,681** | **11,201,737** | **Minimal delta: +1,056 params (+0.0094%)** |
| **Frozen Parameters** | 0 | 11,178,051 (Viewpoint ResNet-18) | Certified checkpoint unchanged |
| **Dataset Partitions** | 41 training cows (Protocol D: 12,753 train, 2,683 val) | 41 training cows (Protocol D: 12,753 train, 2,683 val) | Exact same identities and samples |
| **Evaluation Isolation** | 69 held-out Protocol A cows (parlor, barn, snapshots) | 69 held-out Protocol A cows (parlor, barn, snapshots) | Strictly untouched during preparation and smoke testing |

---

## 3. Strict Metadata Exclusion Policy
Under NO circumstances are acquisition setting metadata used as viewpoint or identification features:
- Camera IDs: **EXCLUDED**
- Recording IDs: **EXCLUDED**
- Subsets (`parlor`, `barn`, `snapshots`): **EXCLUDED**
- Cow IDs: **EXCLUDED from input**
- Image filenames: **EXCLUDED**

Viewpoint information is derived strictly and exclusively from the frozen model's inference on the RGB visual pixels of the cow crop.

---

## 4. Architectural Summary
```text
Run 6 Cow Crop (GT Mask BBox + 5% margin)
           │
           ├──> RGB Channels [3, 224, 224] ──> Frozen Viewpoint ResNet-18 ──> Softmax [3] ──> ViewpointMLP (3->16) ──┐
           │                                                                                                          │
           └──> 4-Channel Input [4, 224, 224] ──> Visual Backbone (ResNet-18) ───────────────> Visual Feature [512] ──┴──> Concat [528]
                                                                                                                             │
                                                                          Identity Head Linear(528, 41) <── Unit L2 Norm <───┘
```
