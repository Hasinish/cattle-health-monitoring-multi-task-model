# Research Log: Kaggle Beef Cattle Behavior Dataset Scientific & Provenance Audit

**Date**: 2026-09-22  
**Author**: Autonomous AI Pair Programmer (Thesis Research Workspace)  
**Topic**: Forensic Acquisition Verification, Modal Inode Quota Analysis, Behavior Taxonomy Census, Identity Provenance, Video Stream Metrics, Leakage Assessment, and Visual Evidence Review for Kaggle Beef Cattle Behavior Dataset  

---

## 1. Executive Summary

We conducted a comprehensive physical verification and forensic scientific audit of the Kaggle Beef Cattle Behavior Data Set (`lucyfirst/beef-cattle-behavior-data-set`, Version 1, Released Dec 17, 2024) hosted on Modal Persistent Volume `beef-behavior-data` (`/data/beef_behavior/` under profile `tigerwood693`). We confirmed that the 48.55 GB master archive (`archive.zip`, 48,553,721,000 bytes) is 100% physically intact and fully downloaded. We audited all 4,337 single-cow cropped MP4 video clips (~11.84 hours of continuous 25.0 FPS video at 224x224 px) across 5 official behaviors (`ruminate`: 1,544 clips; `lie`: 1,362 clips; `stand`: 638 clips; `eat`: 546 clips; `drink`: 247 clips).

Crucially, our audit established three fundamental scientific limitations:
1. **Walking is 100% ABSENT**: Zero walking clips or annotations exist in the entire 48.55 GB dataset. Consequently, Kaggle Beef cannot independently support our canonical 4-class Phase 3 behavior benchmark (`Standing`, `Lying`, `Walking`, `Feeding`).
2. **Zero Biological Cow IDs**: Although 6 physical cows were monitored in a captive pen, the released clips use ephemeral ByteTrack multi-target tracker IDs (`_1_`, `_2_` ... `_77_`). Frequent occlusion spawned over 100 tracker IDs for 6 animals. True biological identities cannot be preserved across sessions, ruling out cow-disjoint evaluation.
3. **Modal Volume Inode Quota Saturation**: Attempting to extract the archive's 1.14 million individual JPEG frames (`Labelframes/`) exhausted Modal's 500,000 inode quota (100% saturation). The cache was safely remediated down to <1% inode usage while keeping the complete 48.55 GB master archive intact alongside all 4,337 playable behavior video clips.

Final Suitability Verdict: **PARTIALLY SUITABLE — WITH SPECIFIC LIMITATIONS**. Kaggle Beef is cataloged as `Behavior (Candidate)` for auxiliary 25 FPS video action modeling. MmCows remains the canonical **Primary Behavior Dataset**, and CVB remains **Optional External Validation**.

---

## 2. Context & Motivation

In Phase 3 of our thesis, behavior recognition is one of the three primary tasks (alongside BCS regression and Individual Re-Identification). MmCows is currently designated as the canonical Primary Behavior benchmark (213,686 crops, 16 verified biological cows, 4 CCTV cameras, 21.0 hours). However, MmCows is sampled at 15-second discrete intervals (0.067 Hz), designed for macro-behavior ethograms rather than continuous high-frame-rate video motion modeling.

Kaggle Beef was identified as a candidate benchmark due to its continuous video clips and automated TimeSformer baseline. A Modal download pipeline (`scripts/modal_beef_behavior_pipeline.py`) was initiated to acquire the dataset. Before incorporating Kaggle Beef into any experimental protocols or multi-task architectures, a rigorous provenance and scientific audit was required to:
1. Physically verify whether the 48.55 GB download completed and inspect archive integrity.
2. Determine the exact behavior taxonomy and check for presence of `Walking`.
3. Investigate cattle identifier provenance (biological cow ID vs. automated tracker ID).
4. Measure temporal continuity, frame rate, duration, and visual crop quality.
5. Identify data leakage risks and determine safe partitioning boundaries.
6. Evaluate taxonomy compatibility with our Phase 3 4-class compact schema.

---

## 3. Forensic Findings & Data

### 3.1 Physical Volume Status & Inode Quota Management
- **Modal Volume**: `beef-behavior-data` mounted at `/data/beef_behavior/` (profile `tigerwood693`).
- **Archive Size**: Exactly **48,553,721,000 bytes (45.219 GiB / 48.55 GB)** at `/data/beef_behavior/archive.zip`. Fully intact, zero partial chunks.
- **Archive Structure**: Total of **1,152,458 archive entries**:
  - `Category Videos/cows/`: **4,337 single-cow MP4 video clips** (~6.53 GB).
  - `Labelframes/`: **1,143,130 JPEG frames** (224x224 RGB image crops of tracked cows).
  - `videos_cut/videos/`: **500 full-scene surveillance MP4 videos** (~25-30 GB).
  - `Target detection dataset/`: **1,500 YOLO-format .txt label files** (`class 0 x_c y_c w h`).
- **Modal Inode Quota Resolution**: Modal enforces a strict **500,000 inode limit per volume**. Uncompressing `Labelframes/` (1.14M files) hit this limit at 500,000 inodes (100% saturation, `OSError: [Errno 28] No space left on device`). The uncompressed frame cache was safely purged to restore the volume to lean state (4,347 inodes, <1%), preserving the master archive and all 4,337 clips.

### 3.2 Exact Behavior Taxonomy (4,337 Clips)
Exhaustive evaluation of all extracted video clips revealed 5 official behaviors:

| Behavior | Class Name | Video Clips | % of Dataset | Mean Duration | Total Duration | Resolution | FPS |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `ruminate` | 1,544 | 35.60% | 9.85 s | ~4.22 hours | 224x224 | 25.0 |
| 2 | `lie` | 1,362 | 31.40% | 9.81 s | ~3.71 hours | 224x224 | 25.0 |
| 3 | `stand` | 638 | 14.71% | 9.83 s | ~1.74 hours | 224x224 | 25.0 |
| 4 | `eat` | 546 | 12.59% | 9.86 s | ~1.50 hours | 224x224 | 25.0 |
| 5 | `drink` | 247 | 5.70% | 9.78 s | ~0.67 hours | 224x224 | 25.0 |
| **Total** | **5 Behaviors** | **4,337** | **100.00%** | **9.83 s** | **~11.84 hours** | **224x224** | **25.0** |

- **Complete Absence of Walking**: Zero walking clips or frames exist. In a captive barn with 6 cows, locomotion was brief and unannotated. Kaggle Beef cannot independently support the 4-class behavior task.

### 3.3 Identity Provenance & ByteTrack Fragmentation
- **Monitored Animals**: 6 physical commercial beef cattle in a single captive barn over 168 hours (7 days) via 1 surveillance camera.
- **Zero Biological Cow IDs**: Animal ear tags or physical identities were never labeled.
- **Tracker ID Proliferation**: Filenames follow `<session_id>_<track_id>_clip_<clip_number>.mp4` (e.g., `00000000082000000_77_clip_69.mp4`). Over 100 distinct tracker IDs (`1, 2, 3 ... 77, 83...`) were generated across the 6 animals by ByteTrack due to occlusions and bunk feeding.
- **Tracker IDs must NEVER be equated with biological cows.**

### 3.4 Video Stream Metrics & Crop Quality
- **Sampling & Frame Rate**: Exactly **25.0 FPS** (dt = 0.040 s).
- **Temporal Structure**: 100% consecutive frames. Median clip length is 10.0 seconds (250 frames). Total duration is ~11.84 hours.
- **Crop Resolution**: Fixed **224 x 224 pixels** (50,176 px^2).
  - Pre-resized square crops introduce mild distortion for rectangular cow bounding boxes (w/h ~ 1.2 to 1.8).
  - Pixel area is **2.87x smaller than MmCows** (median 390x370 px, 144,300 px^2), but **5.68x larger than CVB** (median 104x85 px, 8,840 px^2).

### 3.5 Leakage & Split Safety
- **Single Pen, Single Camera Confinement**: High risk of background and temporal leakage.
- **Mandatory Splitting Unit**: Must group strictly by **Recording Session (`session_id`)** or **Source Surveillance Video (`videos_cut`)**.
- Random clip or frame splitting is strictly prohibited.

---

## 4. Architectural Decisions & Action Plan

1. **Retain Canonical Dataset Roles**:
   - **MmCows** remains the canonical **Primary Behavior Dataset** (16 biological cows, 4 CCTV cameras, 213k native-resolution crops, 4-class compatibility including walking).
   - **CVB** remains the **Optional External Validation** benchmark (open-pasture, 30 FPS, grazing/walking).
   - **Kaggle Beef** is cataloged as `Behavior (Candidate)` for dense 25 FPS video transformer experiments.
2. **Phase 3 Roadmap Untouched**:
   - `phase3_canonical_roadmap.md` remains 100% frozen.
3. **Registry Update**:
   - Ingest Kaggle Beef into `datasets/dataset_registry.csv` with status `AVAILABLE` and role `Behavior (Candidate)`.

---

## 5. Artifacts & File Registry

| Artifact | File Path | Purpose / Description |
| :--- | :--- | :--- |
| **Comprehensive Audit Report** | `docs/audits/phase3_beef_behavior_scientific_audit.md` | Formal 274-line scientific audit document with complete provenance, stream metrics, and decision boundary analysis. |
| **Research Log Entry** | `docs/research_log/2026-09-22_beef_behavior_scientific_audit.md` | This research log document. |
| **Clip Manifest** | `datasets/behavior/beef_cattle_behavior/manifest.csv` | Exhaustive CSV manifest of all 4,337 video clips with session, track, behavior, and duration. |
| **Summary Metrics** | `artifacts/behavior_audit/beef_behavior_audit_summary.csv` | Summary table of class distributions, clip counts, and durations. |
| **Visual Evidence Pack** | `docs/audits/assets/beef_behavior_audit/` | 12 verified visual review assets (10 consecutive filmstrips + 2 diagnostic edge cases). |
| **Audit Scripts** | `scripts/audit_beef_behavior_stage1.py` - `stage4.py` | Modal cloud audit probing scripts (archive inspection, video probing, manifest generation). |
| **Asset Fetcher** | `scripts/fetch_beef_audit_assets.py` | Automated local hydration script for visual review pack. |

---

## 6. Next Steps

- [x] Complete physical verification of Kaggle Beef on Modal volume `beef-behavior-data`.
- [x] Perform exhaustive behavior taxonomy census and video stream probing.
- [x] Generate visual review pack and download to `docs/audits/assets/beef_behavior_audit/`.
- [x] Author comprehensive audit report `docs/audits/phase3_beef_behavior_scientific_audit.md`.
- [x] Create research log entry `docs/research_log/2026-09-22_beef_behavior_scientific_audit.md`.
- [ ] Update `docs/research_log/README.md` index table.
- [ ] Update `scripts/build_dataset_registry.py` and regenerate `datasets/dataset_registry.csv`.
- [ ] Update `memory/state.md` and `memory/history.md`.
- [ ] Commit and push all audit assets and code to Git `origin/main`.
- [ ] Mirror customizations and memory to `D:\custom-antigravity`.
