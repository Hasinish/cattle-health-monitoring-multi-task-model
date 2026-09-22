# Research Log: Phase 3 ScienceDB RGB Single-Task BCS Baseline Pipeline

**Date**: 2026-09-23  
**Author**: Autonomous AI Pair Programmer (Thesis Research Workspace)  
**Topic**: Clean Phase 3 ScienceDB RGB Single-Task BCS Baseline Pipeline, Ordinal BCE vs CORAL Formulation, Real BCS Metrics Engine, Checkpoint Determinism, Smoke Test on GTX 1050 Ti, and Compute Directive Update  

---

## 1. Executive Summary

We designed, implemented, and empirically validated the canonical Phase 3 ScienceDB RGB single-task Body Condition Scoring (BCS) baseline pipeline (`scripts/train_sciencedb_bcs_baseline.py`). The pipeline strictly uses the canonical leakage-safe 5,653-burst-group train/val/test split (`datasets/bcs/sciencedb/`), explicitly rejecting outdated Phase 2 split assumptions and formally classifying ScienceDB as **burst-group-disjoint / sequence-safe** (true biological cow IDs were not released by the publisher).

Following forensic audit and code refinements:
1. The CLI was corrected with `argparse.BooleanOptionalAction` supporting `--smoke`, `--no-smoke`, `--full-run`, and `--dry-run`.
2. Strict test-set isolation was implemented: smoke mode completely bypasses `test.csv` and evaluates strictly on validation subsets, ensuring zero leakage or exposure of the canonical held-out test split.
3. Split statistics and SHA-256 hashes were recomputed directly from the active files and reconciled 100% with `datasets/bcs/sciencedb/split_report.md`.
4. Ordinal regression head terminology was mathematically clarified: distinguishing Frank & Hall (2001) independent cumulative BCE (`ordinal_bce`) from Cao et al. (2020) weight-shared CORAL (`coral`).
5. A 2-epoch smoke test was executed on the local GTX 1050 Ti (~32s runtime, code 0) verifying train, val, and 100% bit-identical checkpoint save/resume.
6. Compute policy was realigned: the BRACU Lab Research PC (RTX 5090) is disqualified due to an unrecoverable locked/bloated OS environment; all heavy preprocessing, full training, and ablations are assigned to rotating Modal cloud profiles.

---

## 2. Context & Motivation

Under Phase 3 Step 4 of our canonical roadmap (`phase3_canonical_roadmap.md`), establishing a trusted, uncorrupted single-task RGB baseline is mandatory before introducing cattle-centered visual representations (localization crops, segmentation masks, pose keypoints, or viewpoint conditioning).

Previous Phase 2 BCS training scripts relied on `sciencedb_bcs_index.csv`, which suffered from severe stereo sequence leakage (94.64% cross-split contamination) and falsely treated passage identifiers as individual cows. Furthermore, legacy metrics reported class-index MAE (0 to 4) rather than physical BCS units, leading to misleading error scales.

This task resolves those legacy flaws by establishing a clean, reproducible baseline pipeline that:
1. Operates exclusively on the canonical 5,653-burst-group stratified split (`train.csv`: 37,045 images, `val.csv`: 8,481 images, `test.csv`: 8,040 images).
2. Explicitly enforces the burst-group-disjoint protocol and disclaims cow-disjointness.
3. Implements true physiological BCS MAE calculation across the 3.25 to 4.25 range.
4. Preserves full provenance (Git commit, split SHA-256 hashes, seed, hyperparameters).
5. Validates checkpoint save/resume determinism.
6. Conducts low-cost smoke testing on local hardware without touching the held-out test set before dispatching full training runs to rotating Modal cloud profiles.

---

## 3. Forensic Findings & Implementation Details

### 3.1 Split Provenance & Recomputed Hashes
The pipeline directly consumes the repaired canonical splits generated on 2026-09-20. All metrics and hashes were re-verified directly from disk:
- `datasets/bcs/sciencedb/train.csv`:
  - SHA-256: `9f6b0bc716e01a2ab22208daff1c49e49fd450a4d7cf0a57b2979275ed33497a`
  - Samples: 37,045 images (69.2%) across 3,958 burst groups (70.0%)
- `datasets/bcs/sciencedb/val.csv`:
  - SHA-256: `e223e3c4c081ca5c9f993f7156dc791df97b6ea6d011b8b4b6f068590b3d975d`
  - Samples: 8,481 images (15.8%) across 850 burst groups (15.0%)
- `datasets/bcs/sciencedb/test.csv`:
  - SHA-256: `eae459e031d06c4b1150ce2cbcdcb8259724b070b831341222b15c99e3626e5f`
  - Samples: 8,040 images (15.0%) across 845 burst groups (14.9%)
- **Total**: 53,566 images across exactly 5,653 connected burst groups (100.0% burst-group disjoint, 0 cross-passage overlaps, 0 byte duplicates).
- **Scientific Classification**: Burst-group-disjoint / sequence-safe. **NOT cow-disjoint**.

### 3.2 Architecture & Head Formulation
- **Backbone**: ImageNet-1k pretrained ResNet-18 (`torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`).
- **Feature Pooling**: Global Average Pooling (512-dimensional embedding).
- **Head Formulations**:
  1. `ordinal_bce` (Default): Cumulative Link / Extended Binary Classification (Frank & Hall, 2001). Implements $K-1$ independent binary classifiers with unconstrained weights and biases ($w_k^T x + b_k$). Binary targets: $t_k = \mathbb{I}(y > k)$. Loss: cumulative binary cross-entropy. Discrete prediction: $\sum_{k=1}^{K-1} \mathbb{I}(\sigma(\text{logit}_k) > 0.5)$.
  2. `coral` (Cao et al., 2020): Consistent Rank Logits with weight sharing. A single shared weight vector $w \in \mathbb{R}^D$ is combined with individual rank bias thresholds ($w^T x + b_k$). Guaranteed rank consistency when biases are strictly ordered.
  3. `linear`: Standard 5-way softmax classification with Cross-Entropy Loss. Discrete prediction: $\text{argmax}(\text{logits})$.

### 3.3 True Physiological BCS Metrics Engine
In veterinary practice (Ferguson et al., 1994; Elanco BCS chart), BCS is evaluated on a continuous-like quarter-point scale:
- Mapping: $\text{real\_bcs} = 3.25 + \text{class\_idx} \times 0.25$
- Primary Metric: Real BCS MAE = $\frac{1}{N} \sum_{i=1}^N |\text{real\_bcs}_i - \widehat{\text{real\_bcs}}_i|$
- Secondary Metrics:
  - Class-Index MAE = $\frac{\text{Real BCS MAE}}{0.25}$
  - Exact Accuracy (Acc@0) = % exact class agreement
  - Within-0.25 Tolerance (Acc@1) = % predictions within $\pm 0.25$ BCS units (adjacent ordinal class)
  - Balanced Accuracy = Macro-averaged recall across all 5 classes
  - Macro-F1 = Macro-averaged F1 score
  - Per-class breakdowns: Support, Precision, Recall, F1, Per-Class Real MAE, 5x5 Confusion Matrix.

### 3.4 CLI & Test-Set Isolation Protocols
- **CLI Options**:
  - `--smoke` / `--no-smoke`: Toggle smoke testing mode.
  - `--full-run`: Explicit alias for `--no-smoke`.
  - `--dry-run`: Parse configs, check paths, print execution plan, and exit cleanly without training.
- **Canonical Test Set Protection**:
  - In smoke mode (`--smoke`), `test.csv` is **never loaded**.
  - Validation subset (`val_subset`) is evaluated at epoch ends and post-training.
  - The held-out canonical test split is preserved completely untouched until genuine finalized baseline runs.

### 3.5 Checkpoint Determinism & GTX 1050 Ti Smoke Test Evidence
- **Execution Command**:
  ```powershell
  python scripts/train_sciencedb_bcs_baseline.py --smoke --epochs 2 --max_samples 250 --test_save_resume
  ```
- **Execution Device**: NVIDIA GeForce GTX 1050 Ti (4.0 GB VRAM)
- **Samples Used**: 250 train (50/class), 125 val (25/class). Test split: UNTOUCHED.
- **Timing**: Epoch 1: 5.7s; Epoch 2: 4.8s; Total runtime: ~32s (including roundtrip test).
- **Progression**:
  - Epoch 1: Train Loss 0.6773 | Val Loss 0.7464 | Val Real MAE: 0.4300 BCS | Bal Acc: 14.40%
  - Epoch 2: Train Loss 0.4622 | Val Loss 0.7507 | Val Real MAE: 0.4200 BCS | Bal Acc: 18.40%
- **Checkpoint Save/Resume Determinism Test**:
  - Round-trip save/load into new model: PASSED with 100% bit-identical parameters (`torch.equal`) and forward logits (`torch.allclose` atol=1e-6).
- **Post-Training Smoke Evaluation (Validation Subset)**:
  - Real BCS MAE: **0.4200 BCS units**
  - Class Index MAE: 1.6800 steps
  - Exact Accuracy (Acc@0): 18.40%
  - Within-0.25 (Acc@1): 50.40%
  - Balanced Accuracy: 18.40%
  - Macro-F1: 0.1827
  - Canonical Test Split Status: **PRESERVED UNTOUCHED (0 samples evaluated)**.

---

## 4. Compute Execution Policy & Cloud Strategy

1. **BRACU Lab Research PC (RTX 5090) Disqualification**:
   - The lab workstation environment is heavily bloated, locked by institutional policies, and exhibits unrecoverable dependency conflicts.
   - It is permanently **disqualified** as an execution target for this project.
   - Future experiments will NOT be scheduled or run on the RTX 5090.
2. **Local Machine (GTX 1050 Ti)**:
   - Dedicated strictly to sanity checks, pipeline path verifications, tensor dimension assertions, loss/metric unit testing, checkpoint determinism, and 1–2 epoch smoke tests.
   - Full 30-epoch training on 37k images will NOT be run locally.
3. **Rotating Modal Cloud Profiles**:
   - Heavy preprocessing, perception feature caching (Step 3), full single-task training (Step 4), and multi-task ablations (Step 6+) will run on Modal.
   - Modal accounts/profiles rotate dynamically based on available monthly credits and team resource pools. Scripts and documentation must not hard-code a single universal profile.

---

## 5. Artifacts & File Registry

| Artifact | File Path | Purpose / Description |
| :--- | :--- | :--- |
| **BCS Baseline Script** | `scripts/train_sciencedb_bcs_baseline.py` | Complete training, evaluation, checkpointing, and smoke-testing pipeline with `--no-smoke`, `--full-run`, `--dry-run`, `ordinal_bce`, and `coral`. |
| **Smoke Metrics JSON** | `artifacts/bcs_baseline/bcs_baseline_metrics.json` | Provenance metadata, Git commit, split SHA-256 hashes, configs, training loss curves, and smoke validation metrics. |
| **Smoke Summary Markdown** | `artifacts/bcs_baseline/bcs_baseline_smoke_summary.md` | Human-readable markdown report of smoke test performance and per-class metrics on the validation subset. |
| **Research Log Entry** | `docs/research_log/2026-09-23_sciencedb_bcs_baseline_pipeline.md` | This research log document. |

---

## 6. Next Steps

- [x] Fix CLI to support `--no-smoke`, `--full-run`, and `--dry-run`.
- [x] Isolate canonical test split during smoke tests (evaluate on val subset only).
- [x] Recompute and reconcile split statistics and SHA-256 hashes with `split_report.md`.
- [x] Accurately distinguish Frank & Hall (2001) `ordinal_bce` from Cao et al. (2020) `coral`.
- [x] Re-run GTX 1050 Ti smoke test and verify bit-identical save/resume.
- [x] Update compute policy in `AGENTS.md`, `.agents/rules/`, `docs/`, and `memory/state.md`.
- [ ] Commit and push all changes to `origin/main`.
- [ ] Mirror customizations and memory to `D:\custom-antigravity`.
