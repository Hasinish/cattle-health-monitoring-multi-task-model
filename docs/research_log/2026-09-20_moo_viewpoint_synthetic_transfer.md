# Research Log: MOO Synthetic-to-Real Viewpoint Transfer Experiment & Scientific Integrity Audit

**Date**: 2026-09-20  
**Phase**: Phase 3 (Step 2.4 — Cattle Viewpoint Perception Feasibility)  
**Author / Operator**: Hasin Ishrak (in pair programming with Antigravity)  
**Status**: AUDITED / STEP 2.4 REOPENED (POST-HOC CORRECTION AUDITED; STEP 2 IN PROGRESS)  

---

## 1. Executive Summary & Audit Findings

In Step 2.4 of the Phase 3 Canonical Roadmap, frozen zero-shot Vision-Language Models (OpenAI CLIP, OpenCLIP LAION-2B, Google SigLIP) were evaluated on a 100-sample human-verified and cross-checked benchmark of real farm cattle with RT-DETR-L target crops. All three VLMs failed to deliver acceptable operational performance (OpenAI CLIP: 33.68% physical accuracy, 19 false fronts; OpenCLIP: 15.79%; SigLIP: 12.63%, 66 false fronts).

To test the remaining canonical roadmap route—**synthetic viewpoint supervision**—we set up a zero-waste cloud pipeline on Modal to ingest the CEA Kalisteo MOO (Multi-view Oriented Observations) dataset (128,000 synthetic cattle images, 1,000 cattle IDs, continuous $(\phi, \theta)$ angles).

### Key Empirical Findings & Scientific Status:

1. **MOO Synthetic Validation (93.2% Accuracy)**:
   A linear head trained on 2,500 MOO synthetic images using a frozen ImageNet ResNet-18 achieved **93.2% validation accuracy** on held-out synthetic renders. This is a clean result, valid as synthetic-domain validation only.

2. **Raw Synthetic-to-Real Transfer (12.63% Accuracy — Clean Baseline)**:
   Direct evaluation on N=95 non-ambiguous real cattle crops yielded **12.63% accuracy**, **Macro-F1 0.0674**, and **40 false-front predictions**. This is the **clean, untuned real transfer result** from this experiment, demonstrating a substantial sim-to-real domain gap on zero-shot transfer.

3. **Post-Hoc 63.16% Result & Scientific Integrity Audit**:
   An apparent surge to 63.16% accuracy / 0.3600 Macro-F1 was obtained after applying a remapping heuristic (`front/rear -> side`, `side -> rear`). A subsequent scientific integrity audit revealed:
   - The mapping was selected dynamically from 6 candidate permutations evaluated against the SAME 95 real benchmark labels.
   - The winning mapping is not a geometrically valid 90° rotation (it maps both front and rear to side, never predicts front, and collapsed left/right into a single side class).
   - Therefore, the 63.16% result is classified explicitly as an:
     `exploratory post-hoc benchmark fit; invalid as final held-out evaluation evidence`.
   - It must NOT be described as a discovered physical coordinate correction or proof of successful 90-degree alignment.

4. **Independent MOO Orientation Verification**:
   To independently test whether MOO had an underlying 90° coordinate mismatch, raw synthetic renders for a single cow identity (`cow388`) across `front`, `back`, `left`, and `right` were extracted from Modal volume `moo-data` and visually inspected without referencing real data:
   - `front` visually shows the cow HEAD / cranial end.
   - `back` visually shows the cow TAIL / caudal end.
   - `left` and `right` visually show lateral SIDE views.
   - Therefore: `Independent evidence does NOT support the claimed 90-degree anatomical mismatch.`
   MOO's 3D coordinate system is already anatomically aligned. The 12.63% real accuracy reflects the genuine sim-to-real domain gap (texture, lighting, background, and morphology shifts), not an uncorrected camera rotation.

5. **Operational Generator Status**:
   - Operational viewpoint generator: **NOT YET SELECTED**.
   - MOO remains a valid roadmap-approved candidate, but current real-transfer evidence is insufficient.
   - Status: **STEP 2.4 REOPENED | STEP 2 IN PROGRESS**.

---

## 2. Methodology & Experimental Setup

### 2.1 Remote Archive Inspection & Cloud Pipeline
- **Archive**: `MOO.zip` (34.03 GB compressed) hosted by CEA Kalisteo.
- **Remote Inspection**: Verified via HTTP Range requests in `scratch/test_remote_zip.py` that the archive contains `data.hdf5` (55.24 GB uncompressed) and `metadata.json` (136.36 MB uncompressed).
- **Modal Cloud Pipeline** (`scripts/modal_moo_pipeline.py`):
  - Persistent Volume: `moo-data` attached at `/data`.
  - Minimal container footprint: `cpu=1.0, memory=2048` for download/I/O per `.agents/rules/modal_cost_optimization.md`.
  - Ingestion: Downloaded via `aria2c` with 16 parallel connections in 2,133.6s (16.3 MB/s avg); extracted via single-threaded `unzip` in 928.8s; `MOO.zip` deleted to preserve storage quota; volume committed via `volume.commit()`.
  - Total compute and storage cost: ~$0.30 on Modal burner account `hasinishrak74001`.

### 2.2 Model Architecture & Training
- **Backbone**: Standard `torchvision.models.resnet18(weights="IMAGENET1K_V1")`, **100% frozen** (requires_grad = False).
- **Classification Head**: Single linear layer `fc = nn.Linear(512, 5)` (2,565 trainable parameters).
- **Training Strategy**:
  - Sampled 500 balanced images per class across 5 physical classes = 2,500 training images, 500 validation images from `data.hdf5`.
  - Pre-extracted 512-dimensional features through frozen ResNet-18 in 54.7s on 4 vCPUs.
  - Trained linear head for 15 epochs in 0.64s on CPU (Adam, lr=2e-3, CrossEntropyLoss).
  - Synthetic validation accuracy: **93.2%** (clean synthetic validation).
  - Checkpoint: `moo_resnet18_viewpoint.pth` (42.7 MB).

### 2.3 Evaluation on Real Cattle Benchmark
- **Benchmark Manifest**: `artifacts/perception_audit/viewpoint_expanded_agent_review_manifest.csv` (N=100 across ScienceDB, MmCows, SideViewCows2026).
- **Input Crops**: Step 2.1 RT-DETR-L target bounding boxes recovered from `artifacts/perception_audit/localization_detections_expanded.csv` (86 crops, 14 full-image fallbacks; exact same input as zero-shot audit).
- **Evaluation Script**: `scripts/evaluate_moo_viewpoint.py`.

---

## 3. Results & Comparative Scoreboard

### 3.1 5-Class Physical Evaluation (N=95 Non-Ambiguous Real Samples)

| Model / Approach | Supervision | Accuracy | Macro-F1 | False Fronts (GT=0) | Scientific Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Google SigLIP** (ViT-B/16-224) | Zero-Shot VLM | 12.63% | 0.0958 | 66 / 95 | Clean zero-shot baseline (Rejected) |
| **OpenCLIP** (LAION-2B ViT-B/32) | Zero-Shot VLM | 15.79% | 0.1149 | 13 / 95 | Clean zero-shot baseline (Rejected) |
| **OpenAI CLIP** (ViT-B/32) | Zero-Shot VLM | 33.68% | 0.2711 | 19 / 95 | Clean zero-shot baseline (Rejected) |
| **MOO ResNet-18 (Raw Unaligned)** | Synthetic Linear Head | **12.63%** | **0.0674** | **40 / 95** | **Clean untuned real transfer result** |
| **MOO ResNet-18 (Post-Hoc Permutation Fit)** | Synthetic Linear Head | 63.16% | 0.3600 | 0 / 95 | **Exploratory post-hoc benchmark fit; invalid as final held-out evidence** |

### 3.2 Confusion Matrices

#### A. Raw Unaligned Confusion Matrix (Clean Transfer):
```
                    Pred_front  Pred_front-oblique  Pred_side  Pred_rear-oblique  Pred_rear
True_front                   0                   0          0                  0          0
True_front-oblique           1                   0          0                  0          1
True_side                   28                   2          9                  0         15
True_rear-oblique            3                   1          4                  0          4
True_rear                    8                   0         16                  0          3
```

#### B. Post-Hoc Permutation Fit Confusion Matrix (Exploratory Only):
```
                    Pred_front  Pred_front-oblique  Pred_side  Pred_rear-oblique  Pred_rear
True_front                   0                   0          0                  0          0
True_front-oblique           0                   0          2                  0          0
True_side                    0                   0         43                  2          9
True_rear-oblique            0                   0          7                  1          4
True_rear                    0                   0         11                  0         16
```

---

## 4. Scientific Significance for Thesis

1. **Definitive Refutation of Generic Zero-Shot VLMs**:
   Pretrained multimodal foundation models without cattle domain adaptation produce unacceptably high error rates and catastrophic false-front hallucinations (up to 69.5% on SigLIP).
2. **Substantial Sim-to-Real Domain Gap**:
   While synthetic 3D CAD data (MOO) achieves high accuracy in-domain (93.2%), out-of-the-box linear probe transfer to real farm imagery suffers a major performance drop (12.63% accuracy), primarily misclassifying sides as fronts and rears.
3. **Rigorous Scientific Integrity & Benchmark Protection**:
   Label remappings chosen by testing permutations against the test benchmark must never be passed off as physical coordinate discoveries. Independent visual verification confirmed standard camera orientation in MOO (`Independent evidence does NOT support the claimed 90-degree anatomical mismatch`).
4. **Current Status**:
   Step 2.4 is **REOPENED**. Immediate next action: Establish a scientifically valid MOO-supervised viewpoint experiment without post-hoc label remapping on the real evaluation benchmark.
