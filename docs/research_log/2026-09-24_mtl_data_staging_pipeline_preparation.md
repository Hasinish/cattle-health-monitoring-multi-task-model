# Research Log: High-Speed Resumable Multi-Task Learning (MTL) Data Staging Pipeline Preparation

**Date**: 2026-09-24  
**Author**: Hasin Ishrak  
**Target Profile**: `hasinishrak2015`  
**Target Volumes**: `mtl-data`, `mtl-checkpoints`  
**Purpose**: Prepare and verify the maximum-speed, resumable, low-disk data staging pipeline to provision all Train and Validation inputs for Run 7 (E1 Hard-Shared MTL) and Run 8 (E3 Modular/Task-Private MTL).

---

## 1. Executive Summary & Architectural Decisions

To stage the unified MTL dataset on the target Modal profile `hasinishrak2015` with maximum throughput, minimum local disk consumption, and zero test leakage, we designed an asymmetric, task-optimized transfer pipeline:

1. **Task A — BCS (Body Condition Scoring)**:
   - **Source**: Modal profile `tigerwood697`, persistent volume `sciencedb-perception-cache`.
   - **Strategy**: Reuses already-certified Run 4 monolithic binary tensors (`train_bcs_224.pt` at ~6.89 GB and `val_bcs_224.pt` at ~1.57 GB) + manifests.
   - **Avoided Overhead**: Zero re-download of raw ScienceDB (~4.11 GB) and zero recomputation of RT-DETR + SAM perception.
   - **Transfer Method**: Source-side chunking into 1024 MB parts with SHA-256 hashes, relayed sequentially via PC buffer, uploaded to `/mtl-data/bcs/staging/`, verified, and reassembled on target. Local chunk files are deleted immediately after upload (peak PC disk footprint <= 1024 MB).
   - **Leakage Protection**: `test_bcs_224.pt` and `test_perception.csv` are strictly excluded and verified absent.

2. **Task B — Behavior Recognition**:
   - **Source**: Modal profile `tigerwood693`, persistent volume `behavior-perception-cache`.
   - **Strategy**: Reuses certified production cache (`/cache/production/`). Packages ONLY the 4,271 retained sequence directories (3,641 Train + 630 Val) into an uncompressed sequential tar archive (`behavior_retained.tar`, ~900 MB - 1.1 GB) on ephemeral storage inside `tigerwood693`.
   - **Avoided Overhead**: Zero re-download of CVB (~14.29 GB) and Kaggle Beef (~48.55 GB); avoids ~58 minutes of heavy GPU SAM perception computation. Avoids transferring 72,607 tiny files individually over network I/O.
   - **Transfer Method**: Split into 1024 MB parts, relayed sequentially via PC buffer, uploaded to `/mtl-data/behavior/staging/`, verified, and extracted directly into `/mtl-data/behavior/`. Local parts deleted immediately.
   - **Leakage Protection**: `production_test/`, `retained_test.csv`, and `failed_test.csv` are strictly excluded and verified absent.

3. **Task C — Individual Cow Re-Identification (Re-ID)**:
   - **Source**: Zenodo record `21605650` (SideViewCows2026).
   - **Strategy**: **Direct Cloud Download** executed directly INSIDE `hasinishrak2015` minimal container.
   - **Transfer Method**: Multi-threaded HTTP Range downloader (16 parallel workers) pulls ONLY `parlor.zip` (~9.59 GB) to ephemeral `/tmp/sideview_mtl/parlor.zip`.
   - **PC Relay**: **0 BYTES** relayed through local PC.
   - **Selective Extraction**: Uses canonical protocol CSVs (`protocol_closed_set.csv` and `protocol_cross_setting.csv`) to resolve the exact 15,436 Train/Val records for the 41 representation learning cows (12,753 Train, 2,683 Val). Selectively extracts only the 15,436 image files and 15,436 mask files (30,872 files total) into `/mtl-data/reid/parlor/`.
   - **Immediate Purge**: Temporary `parlor.zip` is purged immediately post-extraction.
   - **Leakage Protection**: Zero persistent `barn/` or `snapshots/` data; 0 overlap with the 69 held-out Protocol A evaluation cows.

---

## 2. Quantitative Transfer & Storage Metrics

| Metric | Task A: BCS | Task B: Behavior | Task C: Re-ID | Total / Peak |
| :--- | :--- | :--- | :--- | :--- |
| **Source Location** | `tigerwood697` | `tigerwood693` | Zenodo (Record 21605650) | Mixed Cloud / Cross-Profile |
| **Source Volume** | `sciencedb-perception-cache` | `behavior-perception-cache` | Zenodo CDN | Persistent & Ephemeral |
| **Target Path** | `/mtl-data/bcs/` | `/mtl-data/behavior/` | `/mtl-data/reid/` | `/mtl-data/` |
| **Measured/Est. Size** | ~8.46 GB | ~1.05 GB (uncompressed tar) | ~9.60 GB (`parlor.zip`) | ~19.11 GB total input |
| **PC-Relayed Bytes** | ~8.46 GB | ~1.05 GB | **0 bytes** | **~9.51 GB** |
| **Direct Cloud Bytes** | 0 bytes | 0 bytes | **~9.60 GB** | **~9.60 GB** |
| **Peak Local PC Disk** | ~1024 MB (1 chunk) | ~1024 MB (1 chunk) | 0 MB | **<= 1024 MB** |
| **Train Samples** | 37,045 images | 3,641 sequences | 12,753 pairs | Unified Train Set |
| **Val Samples** | 8,481 images | 630 sequences | 2,683 pairs | Unified Val Set |
| **Test Data Transferred** | **0 (Strictly Excluded)** | **0 (Strictly Excluded)** | **0 (Strictly Excluded)** | **ZERO TEST LEAKAGE** |

---

## 3. Pipeline Implementation & Deliverables

1. **`scripts/stage_mtl_workspace.py`** (Master Local Controller):
   - Supports `--task {all,bcs,behavior,reid}`, `--dry-run`, `--fast`, `--yes`, `--verify`, `--workers`, `--chunk-size-mb`, `--keep-temp`.
   - Thread-safe live progress bar (`CleanProgressBar`) rendering real-time carriage-return updates: `%`, downloaded/uploaded bytes, rolling `MB/s`, elapsed time, and dynamic `ETA`.
   - Chunked resume engine tracking completion status in `.part_state.json`.
   - Immediate unlinking of local temporary parts after target verification.

2. **`scripts/modal_export_mtl_sources.py`** (Source Modal Worker):
   - Runs on minimal containers (`cpu=2.0, memory=4096, NO GPU`) in `tigerwood697` and `tigerwood693`.
   - `prepare_bcs_export_remote`: splits `.pt` files and copies manifests to `/cache/export_bcs/`.
   - `prepare_behavior_export_remote`: creates uncompressed tar of 4,271 retained sequences to `/cache/export_behavior/`.
   - `cleanup_bcs_export_remote` & `cleanup_behavior_export_remote`: clears export staging on source volumes.

3. **`scripts/modal_stage_mtl_target.py`** (Target Modal Worker):
   - Runs on minimal containers (`cpu=2.0, memory=4096, NO GPU`) in `hasinishrak2015`.
   - `stage_reid_direct_remote`: 16-thread HTTP Range direct download of `parlor.zip` -> selective extraction of 15,436 Train/Val pairs -> deletion of zip.
   - `reassemble_bcs_remote`: concatenates chunks, verifies whole-file SHA-256, verifies PyTorch tensor loadability, unlinks chunks.
   - `reassemble_behavior_remote`: concatenates tar chunks, verifies SHA-256, extracts 4,271 sequences, verifies 8 frames + 8 masks per sequence, unlinks tar.
   - `verify_mtl_workspace_remote`: comprehensive audit asserting zero test leakage, creating `/mtl-data/staging_manifest.json`, and committing volumes.

4. **`artifacts/mtl_staging/staging_manifest_schema.json`**:
   - Complete schema tracking volume provenance, task sample counts, and leakage audit flags.

---

## 4. Verification & Readiness

- Local syntax compilation (`python -m py_compile`) passed on all three scripts with exit code 0.
- **BCS Staging Verification Hardening Patch**:
  - Corrected tensor payload schema expectation from `payload["images"]` to authentic Run 4 keys: `payload["tensors"]` (`torch.uint8` tensor with shape `[N, 4, 224, 224]`), `payload["targets"]` (`torch.long`), and `payload["raw_labels"]` (`torch.float32`).
  - Upgraded Modal container RAM from 4096 MB to 16384 MB (16 GB) in both `reassemble_bcs_remote` and `verify_mtl_workspace_remote` to safely accommodate `train_bcs_224.pt` (~6.42 GiB in RAM) without risk of OOM.
  - Enforced memory-safe sequential verification: loads `train_bcs_224.pt`, validates shape `[N, 4, 224, 224]` and keys, deletes payload, triggers `gc.collect()`, then sequentially loads and validates `val_bcs_224.pt`, deletes payload, and triggers `gc.collect()`. Never loads train and val simultaneously.
  - Synthetic unit test (`scratch/verify_bcs_staging_schema.py`) verified 100% of sequential verification logic and confirmed rejection of faulty `images` schemas.
- Dry-run verification (`python scripts/stage_mtl_workspace.py --task all --dry-run`) verified all execution plans, file counts, and estimated payloads without initiating full data movement or launching GPU instances.
- Zero GPU compute was consumed; zero full transfers were automatically initiated.
