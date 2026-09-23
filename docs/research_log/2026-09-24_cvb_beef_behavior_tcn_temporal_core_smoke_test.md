# Research Log: Run 5 Behavior Temporal Core Implementation & Modal T4 Smoke Test

**Date:** 2026-09-24  
**Status:** IMPLEMENTED, SMOKE-TESTED & 100% CERTIFIED (TEMPORAL CORE ONLY)  
**Modal App ID:** `ap-S28P53jyMZxAlEkhDMLka9`  
**Modal Profile:** `tigerwood693`  
**Hardware Tier:** NVIDIA T4 GPU (16GB VRAM, 2 CPUs, 4GB RAM)  
**Volumes:** `cvb-data` (/mnt/cvb), `beef-behavior-data` (/mnt/beef), `behavior-checkpoints` (/checkpoints)  

---

## 1. Executive Summary

Under the active September 26 deadline priority overlay, we designed, implemented, and smoke-tested the lightweight temporal backbone for Phase 3 Run 5 (Behavior Perception-Enhanced Temporal Model) on Modal profile `tigerwood693` using an NVIDIA T4 GPU. The architecture couples an ImageNet-pretrained ResNet-18 spatial feature extractor ($512$-D per frame) with a lightweight 1D Temporal Convolutional Network (TCN) featuring 2 Conv1d blocks, GELU activations, dropout, residual skip connections, adaptive temporal average pooling, and a 5-class linear classifier ($11,900,485$ total trainable parameters: $11,176,512$ backbone + $723,973$ TCN).

A deterministic temporal sampling rule ($T = 8$ equidistant frames across $[\text{start\_frame}, \text{end\_frame}]$) was implemented and certified for both primary behavior sources: CVB preserves authentic target tracklet identity using ground-truth bounding box indexing (zero missing target bboxes, zero silent cow substitutions), while Kaggle Beef decodes single-cow clips directly. On a balanced 40-sequence smoke subset (30 train, 10 val across all 5 classes), the cloud execution completed 2 training epochs in $58.68$s container runtime ($82.2$s total app lifetime, cost $<\$0.02$). Checkpoint save and reload achieved bit-identical logit parity ($\text{Max Logit Diff} = 0.00000000$). The canonical test split (`test.csv`, 809 samples) remained strictly untouched and was never loaded or evaluated. A visual contact sheet verifying 8-frame sequence alignment was generated and archived. This confirms the temporal core plumbing; full perception cache integration and full training remain pending.

---

## 2. Context & Motivation

In the Phase 3 Step 4.2 Behavior RGB baseline (Run 2 of 8, evaluated 2026-09-23), static single-frame midpoint crops achieved $88.88\%$ overall accuracy but suffered on `Walking` ($21.74\%$ F1, with 19 of 26 test clips misclassified as `Feeding` due to static head-down postures). This established the scientific necessity of temporal sequence modeling for Run 5.

Per user constraints and the deadline priority overlay:
- **Plumbing / Smoke Test Only**: Proves temporal sampling, tensor shapes, TCN forward/backward passes, checkpoint roundtrip, and split isolation without launching full perception caching or full 30-epoch training.
- **Strict Prohibition on Premature Masking**: SAM 2.1 mask prompting for Kaggle Beef has multiple audited strategies pending final operational freezing. No fake, zero, or all-ones masks were fabricated. RGB-only is NOT claimed as the final Run 5 model.
- **Architectural Extensibility**: The `FrameFeatureExtractor` is explicitly designed with clean `in_channels=4` support (initializing channel 3 from the RGB channel mean) so that binary foreground masks can be incorporated without altering the TCN architecture.

---

## 3. Forensic Implementation & Architectural Details

### 3.1 Deterministic Temporal Sampler
- **Sequence Length**: Fixed $T = 8$ frames.
- **Sampling Rule**:
  $$\text{frame\_indices} = \left[\text{round}\left(s + i \cdot \frac{e - s}{T - 1}\right) \quad \text{for } i \in \{0, \dots, T-1\}\right]$$
  clamped to $[s, e]$. Random adjacent frame sampling is strictly prohibited.
- **CVB Tracklet Identity Preservation**:
  - Implemented `CVBAnnotationCache` which parses and indexes `instances_default.json` per video cut into `(track_id, frame_idx) -> [x1, y1, x2, y2]`.
  - For each sampled frame, crops the exact target cow using its authentic ground-truth bounding box.
  - If a specific frame in the segment lacks a bounding box, it falls back to the nearest frame of the **same target tracklet** and records an explicit audit event. It NEVER silently substitutes another cow. In this smoke run, exactly 0 target bbox misses occurred ($30/30$ train and $10/10$ val sequences had complete authentic bboxes).
- **Kaggle Beef Single-Cow Decoding**:
  - Clips are already single-cow; samples 8 equidistant frames directly using OpenCV frame seeks with end-of-clip boundary clamping.
- **Resolution**: All cropped/decoded frames resized to $224 \times 224$ RGB.

### 3.2 Model Architecture & Parameter Verification
- **Spatial Backbone**: `FrameFeatureExtractor`
  - ResNet-18 initialized from ImageNet (`ResNet18_Weights.DEFAULT`).
  - Final classification layer `fc` replaced with `nn.Identity()`.
  - Output: $[B \cdot T, 512] \rightarrow [B, T, 512]$.
  - Trainable parameters: $11,176,512$.
- **Temporal Module**: `TemporalConvNet` (1D TCN)
  - Input: $[B, T, 512] \rightarrow$ transposed to $[B, 512, T]$.
  - **Conv1d Block 1**: `Conv1d(512, 256, kernel_size=3, padding=1)` $\rightarrow$ `BatchNorm1d(256)` $\rightarrow$ `GELU()` $\rightarrow$ `Dropout(p=0.2)` with a `Conv1d(512, 256, kernel_size=1)` residual projection shortcut ($524,800$ params).
  - **Conv1d Block 2**: `Conv1d(256, 256, kernel_size=3, padding=1)` $\rightarrow$ `BatchNorm1d(256)` $\rightarrow$ `GELU()` $\rightarrow$ `Dropout(p=0.2)` with an identity residual shortcut ($197,376$ params).
  - **Temporal Pooling**: `AdaptiveAvgPool1d(1)` $\rightarrow$ squeeze temporal dimension to $[B, 256]$.
  - **Linear Classifier**: `Linear(256, 5)` $\rightarrow$ $[B, 5]$ ($1,285$ params).
  - Trainable parameters: $723,973$.
- **Total Trainable Parameters**: $11,900,485$ (Backbone: $93.92\%$, TCN: $6.08\%$).
- **Disqualified Architectures**: Zero GRU, zero LSTM, zero Transformer, zero VideoMAE, zero SlowFast.

### 3.3 Synchronized Temporal Data Augmentation
- In `BehaviorTemporalDataset`, augmentations are strictly sequence-consistent:
  - Sequence-synchronized `RandomHorizontalFlip(p=0.5)`: all $T=8$ frames are flipped simultaneously to preserve biological orientation across time.
  - Sequence-synchronized `ColorJitter`: identical brightness/contrast multipliers applied uniformly across all $T=8$ frames.
  - Normalization: ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.

---

## 4. Modal T4 Cloud Smoke Test Results

### 4.1 Execution Configuration
- **Modal App**: `cvb-beef-behavior-temporal-smoke` (App ID `ap-S28P53jyMZxAlEkhDMLka9`).
- **Profile**: `tigerwood693`.
- **Hardware**: NVIDIA Tesla T4 (14.56 GB VRAM), 2 CPUs, 4096 MB RAM.
- **Batch Size**: 4 sequences ($[4, 8, 3, 224, 224]$).
- **Epochs**: 2.
- **Optimizer**: AdamW ($\text{lr} = 10^{-4}$, $\text{weight\_decay} = 0.01$).
- **Loss**: `CrossEntropyLoss`.

### 4.2 Balanced Smoke Dataset Distribution
| Split | Source | Standing | Lying | Feeding | Drinking | Walking | Total |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | Kaggle Beef | 3 | 3 | 3 | 3 | 0 (absent) | 12 |
| **Train** | CVB | 3 | 3 | 3 | 3 | 6 | 18 |
| **Train Subtotal** | **Combined** | **6** | **6** | **6** | **6** | **6** | **30** |
| **Val** | Kaggle Beef | 1 | 1 | 1 | 1 | 0 (absent) | 4 |
| **Val** | CVB | 1 | 1 | 1 | 1 | 2 | 6 |
| **Val Subtotal** | **Combined** | **2** | **2** | **2** | **2** | **2** | **10** |

*Note: Walking is 100% CVB-only by physical biological property of the raw datasets.*

### 4.3 Smoke Training Progression & Metrics
| Epoch | Duration | Train Loss | Val Loss | Val Overall Acc | Val Bal Acc | Val Macro-F1 | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 3.4s | 1.7003 | 1.7619 | 20.0% | 20.0% | 0.0667 | Best Checkpoint |
| **2** | 1.5s | 1.0366 | 1.6540 | 30.0% | 30.0% | 0.2667 | Best Checkpoint |

### 4.4 Checkpoint Save & Bit-Identical Reload Verification
- The best checkpoint was saved to `/checkpoints/behavior_temporal_smoke/behavior_tcn_best.pth`.
- A fresh model instance was instantiated without pretrained weights (`pretrained_backbone=False`) and loaded via `torch.load()`.
- Logits were evaluated on the validation loader and compared against pre-save logits:
  $$\max |\text{logits}_{\text{saved}} - \text{logits}_{\text{reloaded}}| = \mathbf{0.00000000}$$
- Validation Macro-F1 matched identically at $0.2667$.

### 4.5 Strict Canonical Test Isolation
- Canonical `test.csv` (809 samples) was verified present at `/root/datasets/behavior/cvb_beef/test.csv`.
- File metadata timestamps (`st_mtime`) and file sizes before and after execution were verified identical.
- Assertion `summary["test_csv_evaluated"] is False` passed. Zero test samples were loaded, preprocessed, or evaluated.

### 4.6 Runtime & Cost Efficiency
- Pre-extraction and caching of all 40 sequences ($320$ images): $49.8$s.
- 2-epoch training + validation: $4.9$s.
- Total container runtime: $58.7$s.
- Total Modal app lifetime (including image mount and volume commits): $82.2$s.
- Estimated compute cost: $\sim \$0.015$ on NVIDIA T4.

---

## 5. Artifacts & File Registry

| Artifact | Path | Description |
| :--- | :--- | :--- |
| **Training Engine** | `scripts/train_cvb_beef_behavior_tcn.py` | PyTorch training, temporal sampling, and TCN evaluation script. |
| **Modal Cloud Wrapper** | `scripts/modal_train_cvb_beef_behavior_tcn.py` | Cloud orchestration script targeting Modal profile `tigerwood693` on T4 GPU. |
| **Smoke Metrics** | `artifacts/behavior_temporal_smoke/behavior_tcn_metrics.json` | Full summary of parameters, dataset distributions, loss, and reload verification. |
| **Visual Contact Sheet** | `artifacts/behavior_temporal_smoke/temporal_samples_contact_sheet.jpg` | 6-sequence contact sheet showing 8 ordered frames left-to-right with metadata headers. |
| **Audit Contact Sheet** | `docs/audits/assets/behavior_temporal_smoke/temporal_samples_contact_sheet.jpg` | Permanent audit copy of the visual contact sheet ($1792 \times 1560$ px). |
| **Cloud Checkpoints** | `/checkpoints/behavior_temporal_smoke/behavior_tcn_best.pth` | Saved on persistent volume `behavior-checkpoints` on `tigerwood693`. |

---

## 6. Next Steps & Active Status

- **Status**: Run 5 temporal core prepared and smoke-tested; full perception integration and training pending.
- **Do NOT claim Run 5 complete**: This smoke test certified the temporal sampler and TCN architecture only. Full perception cache generation (incorporating SAM 2.1 foreground masks once the Kaggle Beef prompting policy is frozen) and full 30-epoch training remain required to complete Run 5.
- Next immediate thesis priority: Execute full ScienceDB perception caching and Run 4 30-epoch training on Modal profile `tigerwood697`.
