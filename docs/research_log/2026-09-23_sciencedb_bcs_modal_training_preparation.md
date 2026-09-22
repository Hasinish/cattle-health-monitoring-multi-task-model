# Research Log: Phase 3 ScienceDB RGB Single-Task BCS Baseline Modal Training Preparation & Readiness Audit

**Date**: 2026-09-23  
**Author**: Autonomous AI Pair Programmer (Thesis Research Workspace)  
**Topic**: Cloud Preparation, Runtime Path Remapping, TQDM Progress Bar Implementation, Persistent Checkpoint Mount, and Pre-Flight Verification on Modal (`tigerwood697`)  

---

## 1. Executive Summary

We prepared and audited the full 30-epoch training setup for the canonical Phase 3 ScienceDB RGB single-task Body Condition Scoring (BCS) baseline on Modal profile `tigerwood697`. The underlying training engine (`scripts/train_sciencedb_bcs_baseline.py`) was augmented with clean `tqdm` progress bars, runtime path remapping for cross-platform compatibility (resolving local Windows paths to the mounted Linux volume), configurable data/split paths, and periodic volume commit hooks. A dedicated Modal cloud wrapper (`scripts/modal_train_sciencedb_bcs.py`) was constructed, mounting the dataset volume `sciencedb-data` at `/data` and a persistent checkpoint volume `sciencedb-checkpoints` at `/checkpoints`.

An exhaustive pre-flight readiness audit was executed directly inside a Tesla T4 GPU container on Modal (`tigerwood697`, App `ap-TrHVaxRLZvyJBANOPX4ODu`). All 6 verification checks passed with 100% success:
1. **CUDA / GPU**: Tesla T4 (14.56 GB VRAM) verified.
2. **Dataset Volume**: Exact image root `/data/dataset/` verified; all 53,566 images across 5 discrete ordinal classes (`3.25`: 7,536; `3.5`: 13,256; `3.75`: 14,255; `4.0`: 12,556; `4.25`: 5,963) confirmed present.
3. **Canonical Splits**: 100% hash and record alignment verified against canonical burst-group-disjoint splits (`train.csv`: 37,045 imgs / 3,958 groups; `val.csv`: 8,481 imgs / 850 groups; `test.csv`: 8,040 imgs / 845 groups).
4. **Runtime Path Resolution**: 15/15 representative image paths across train, val, and test resolved and verified with PIL Image (1024x576 px).
5. **Persistent Storage**: `/checkpoints/bcs_baseline` verified writable, persistent, and committed to cloud volume `sciencedb-checkpoints`.
6. **Progress Streaming**: Live updating `tqdm` batch-level progress bars tested and confirmed streaming cleanly.

The environment is officially certified **READY** for the user to execute the final 30-epoch run.

---

## 2. Technical Modifications & Enhancements

### 2.1 Runtime Path Remapping Engine (`resolve_image_path`)
Canonical split files (`datasets/bcs/sciencedb/{train,val,test}.csv`) contain host-specific absolute Windows paths (e.g. `D:\cattle-health-monitoring-multi-task-model\datasets\bcs\sciencedb_bcs\dataset\4.25\GS_1_1.jpg`). To preserve scientific split files untouched without rewriting paths or altering SHA-256 hashes, a runtime resolution mechanism was introduced:
- Uses `pathlib.PureWindowsPath` to reliably extract terminal parts `('dataset', '<class>', '<filename>.jpg')` regardless of host OS.
- Resolves against `--data_root` (default `/data` or `/data/dataset` on Modal).
- Preserves local paths when running on Windows where local files exist.

### 2.2 Live Batch-Level TQDM Progress Bars
`scripts/train_sciencedb_bcs_baseline.py` previously lacked visible batch-level progress indicators. Clean `tqdm` progress bars were integrated into training, validation, and final test routines:
- **Train Bar**: `Epoch X/30 | Train | XX%|████████  | N/1158 batches [elapsed<remaining, rate, loss=0.XXXX]`
- **Val Bar**: `Epoch X/30 | Val | XX%|████████  | N/266 batches [elapsed<remaining, rate, loss=0.XXXX]`
- **Test Bar**: `Test | XX%|████████  | N/252 batches [elapsed<remaining, rate, loss=0.XXXX]`
- Features dynamic column scaling, in-place updates, running loss tracking, and clean closing before displaying epoch summary statistics.

### 2.3 Persistent Cloud Checkpoints & Volume Commits
- Full training outputs (`bcs_baseline_latest.pth`, `bcs_baseline_best.pth`, `bcs_baseline_metrics.json`, summary markdown) are routed to `/checkpoints/bcs_baseline/`.
- Mounted on persistent Modal Volume `sciencedb-checkpoints`.
- An `on_epoch_end_callback` executes `checkpoint_vol.commit()` after every epoch, ensuring crash-safe resumption even under spot/container evictions.

---

## 3. Pre-Flight Readiness Scorecard

| Check Item | Target Requirement | Empirical Result | Status |
| :--- | :--- | :--- | :---: |
| **Modal Profile** | `tigerwood697` active | Balance `$29.14` healthy | **PASS** |
| **Execution Hardware** | NVIDIA GPU with CUDA | Tesla T4 (14.56 GB VRAM) | **PASS** |
| **Dataset Volume** | `sciencedb-data` mounted at `/data` | `/data/dataset` found, 53,566 images | **PASS** |
| **Class Distribution** | 5 discrete ordinal classes | `3.25`: 7,536; `3.5`: 13,256; `3.75`: 14,255; `4.0`: 12,556; `4.25`: 5,963 | **PASS** |
| **Split Hashes** | Unmodified canonical hashes | Train `9f6b0b...`, Val `e223e3...`, Test `eae459...` verified | **PASS** |
| **Path Resolution** | Remap Windows CSV paths to Linux | 15/15 samples opened successfully via PIL (1024x576) | **PASS** |
| **Checkpoint Storage** | Persistent Modal volume `/checkpoints` | `/checkpoints/bcs_baseline` created, test write & commit verified | **PASS** |
| **Progress Streaming** | TQDM batch updates | Live batch-level bars confirmed streaming | **PASS** |
| **Overall Verdict** | All pre-flight conditions met | **100% READY FOR TRAINING** | **READY** |

---

## 4. Canonical Full Run Specification

- **Script**: `scripts/modal_train_sciencedb_bcs.py`
- **Encoder**: ImageNet-pretrained ResNet-18
- **Head**: `ordinal_bce` (Frank & Hall 2001 cumulative logits)
- **Epochs**: 30
- **Batch Size**: 32 (1,158 train batches, 266 val batches, 252 test batches per epoch)
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4) with CosineAnnealingLR (eta_min=1e-6)
- **Seed**: 42
- **Best Model Selection**: Lowest validation Real BCS MAE
- **Test Set Evaluation**: Held-out canonical test set (`test.csv`, 8,040 images) evaluated exactly once using best checkpoint.

---

## 5. Hardware Target Upgrade: NVIDIA Tesla T4 -> NVIDIA L4

To accelerate the 30-epoch training throughput and take advantage of modern Ada Lovelace tensor cores and expanded memory headroom, the execution target in `scripts/modal_train_sciencedb_bcs.py` was officially updated from `gpu="T4"` to `gpu="L4"`.

### 5.1 Historical Provenance Preservation
The initial pre-flight readiness audit was successfully conducted on an **NVIDIA Tesla T4** (App `ap-TrHVaxRLZvyJBANOPX4ODu`, 14.56 GB VRAM), verifying pipeline integrity, volume mounts, split hashes, and path resolution. That historical audit is permanently preserved in Section 3 above.

### 5.2 NVIDIA L4 Pre-Flight Readiness Scorecard
A separate pre-flight readiness audit was executed directly on an **NVIDIA L4** GPU container on Modal (`tigerwood697`, App `ap-LpbnMu603XOremldE0aTYr`):

| Check Item | Target Requirement | Empirical Result (L4 Audit) | Status |
| :--- | :--- | :--- | :---: |
| **Modal Profile** | `tigerwood697` active | Balance `$29.14` healthy | **PASS** |
| **Execution Hardware** | NVIDIA L4 GPU with CUDA | NVIDIA L4 (22.03 GB VRAM) | **PASS** |
| **Dataset Volume** | `sciencedb-data` mounted at `/data` | `/data/dataset` found, 53,566 images | **PASS** |
| **Class Distribution** | 5 discrete ordinal classes | `3.25`: 7,536; `3.5`: 13,256; `3.75`: 14,255; `4.0`: 12,556; `4.25`: 5,963 | **PASS** |
| **Split Hashes** | Unmodified canonical hashes | Train `9f6b0b...`, Val `e223e3...`, Test `eae459...` verified | **PASS** |
| **Path Resolution** | Remap Windows CSV paths to Linux | 15/15 samples opened successfully via PIL (1024x576) | **PASS** |
| **Checkpoint Storage** | Persistent Modal volume `/checkpoints` | `/checkpoints/bcs_baseline` created, test write & commit verified | **PASS** |
| **Progress Streaming** | TQDM batch updates | Live batch-level bars confirmed streaming | **PASS** |
| **Overall Verdict** | All pre-flight conditions met on L4 | **100% READY FOR L4 TRAINING** | **READY** |

### 5.3 Manual Launch Command for Full Run on L4
```powershell
$env:PYTHONIOENCODING="utf-8"; modal run --profile tigerwood697 scripts/modal_train_sciencedb_bcs.py --epochs 30 --head-type ordinal_bce --batch-size 32
```
