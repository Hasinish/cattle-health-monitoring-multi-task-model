# Research Log: Canonical CVB + Kaggle Beef Behavior Protocol Generation & Leakage Verification

**Date**: 2026-09-23  
**Author**: Autonomous AI Pair Programmer (Thesis Research Workspace)  
**Topic**: Canonical Leakage-Safe CVB + Kaggle Beef Behavior Dataset Protocol (Gate 1 Cleared for Behavior)  

---

## 1. Executive Summary

Following the approved Phase 3 Behavior stack correction designating **CVB + Kaggle Beef Cattle Behavior** as the canonical primary dense-video Behavior training stack, we engineered and verified the unified, leak-free partition protocol (`datasets/behavior/cvb_beef/`). 

Rather than relying on coarse cut-level directory tags (`primary_beh_tag`), we executed deep annotation parsing across all 502 CVB `instances_default.json` files on Modal volume `cvb-data` (profile `tigerwood693`) to extract authentic annotation-level tracklet segments (`attributes.behavior`). We combined these 2,481 canonical CVB segments with the 2,793 canonical Kaggle Beef continuous clips across 5 shared behavioral classes (`Standing`, `Lying`, `Feeding`, `Drinking`, `Walking`).

Using a deterministic multi-objective stratified group partitioner (Seed 2026), we partitioned the combined 5,274 samples into:
- **Train**: 3,785 samples (71.8%) across 184 groups (44 CVB source videos + 140 Beef sessions)
- **Validation**: 680 samples (12.9%) across 39 groups (10 CVB source videos + 29 Beef sessions)
- **Test**: 809 samples (15.3%) across 44 groups (12 CVB source videos + 32 Beef sessions)

Zero cross-partition group leakage was proven for both datasets. The reopened Behavior Gate 1 is officially **CLEARED**.

---

## 2. Context & Motivation

On 2026-09-23, our roadmap was corrected (`docs/research_log/2026-09-23_behavior_primary_stack_correction.md`) because MmCows' scan-sampled 15-second interval representation lacks the dense temporal motion required for temporal video action modeling. CVB (30 FPS 1080p open-pasture video) and Kaggle Beef (25 FPS barn CCTV video) provide true continuous video suitable for the thesis's temporal modeling goals.

Step 1 and Gate 1 were reopened specifically for the Behavior task to construct and verify the unified protocol under strict scientific constraints:
1. **No random frame or clip splitting**: All cuts, frames, and tracks from a CVB source video or Beef recording session must reside in a single partition.
2. **Annotation-level CVB labeling**: Labels must reflect authentic bounding-box `attributes.behavior` rather than cut-level tags.
3. **Explicit Walking limitation**: Kaggle Beef contains zero Walking; 100% of Walking samples originate from CVB.
4. **Preservation of MmCows**: Existing cow-disjoint MmCows splits remain untouched for external identity-aware validation.

---

## 3. Forensic Implementation Details & Data

### 3.1 CVB Annotation Extraction via Modal
Using profile `tigerwood693` on minimal resources (`cpu=1.0, memory=2048`, zero GPU), `scripts/extract_cvb_track_segments.py` parsed all 502 `instances_default.json` files:
- Parsed 1,163,408 bounding-box annotations into contiguous single-behavior track segments.
- Extracted 3,693 total track segments across 452 cuts and 66 unique source videos (`arm01_{camera}_{date}_{time}`).
- Applied canonical mapping:
  - `grazing` -> `Feeding`: 1,295 segments
  - `resting-lying` -> `Lying`: 486 segments
  - `resting-standing` -> `Standing`: 399 segments
  - `walking` -> `Walking`: 171 segments
  - `drinking` -> `Drinking`: 130 segments
- Excluded 1,212 segments (`hidden`: 350, `none`: 329, `ruminating-lying`: 185, `other`: 163, `grooming`: 86, `ruminating-standing`: 77, `running`: 22).
- Saved master CVB tracks manifest: `datasets/behavior/cvb/cvb_tracks_manifest.csv`.

### 3.2 Kaggle Beef Behavior Manifest Integration & ffprobe Physical Verification
From `datasets/behavior/beef_cattle_behavior/manifest.csv`:
- Extracted 4,337 single-cow MP4 clips across 203 recording sessions.
- Executed physical forensic inspection via `ffprobe` directly on Modal persistent volume `beef-behavior-data` (`/data/beef_behavior/Category Videos/cows/`, profile `tigerwood693`, `cpu=1.0, memory=2048`, zero GPU) using `scripts/probe_beef_clips_ffprobe.py`.
- Probed all 4,337 clips with multi-threaded ffmpeg/ffprobe to extract exact `fps`, `n_frames`, and `duration_sec`:
  - Exactly 25.0 FPS across 100% of clips.
  - Duration ranges from 0.080s to 10.000s (frame counts range from 2 to 250 frames; mean 248.09 frames).
  - 78 total clips in the raw dataset deviate from the default 250 frames (1.80%).
  - In the canonical 2,793-clip subset, exactly 58 clips deviate from 250 frames.
  - Defined explicit frame indexing: zero-based inclusive (`start_frame = 0`, `end_frame = n_frames - 1`).
- Applied canonical mapping:
  - `lie` -> `Lying`: 1,362 clips
  - `stand` -> `Standing`: 638 clips
  - `eat` -> `Feeding`: 546 clips
  - `drink` -> `Drinking`: 247 clips
- Excluded 1,544 `ruminate` clips.
- Retained 2,793 canonical clips across 201 sessions with exact verified frame and duration metadata.

### 3.3 Stratified Group Partitioning (Seed 2026)
We implemented `scripts/build_cvb_beef_behavior_protocol.py` with multi-objective search satisfying:
- CVB: 66 source videos grouped atomically into Train (44), Val (10), Test (12).
- Beef: 201 sessions grouped atomically into Train (140), Val (29), Test (32).
- Enforced all present classes > 0 in all three partitions.

---

## 4. Anti-Leakage Verification Evidence

| Verification Dimension | Condition Checked | Result | Status |
| :--- | :--- | :--- | :--- |
| **CVB Source Video Disjointness** | Train ∩ Val | 0 overlapping videos | **PASS** |
| **CVB Source Video Disjointness** | Train ∩ Test | 0 overlapping videos | **PASS** |
| **CVB Source Video Disjointness** | Val ∩ Test | 0 overlapping videos | **PASS** |
| **Beef Session Disjointness** | Train ∩ Val | 0 overlapping sessions | **PASS** |
| **Beef Session Disjointness** | Train ∩ Test | 0 overlapping sessions | **PASS** |
| **Beef Session Disjointness** | Val ∩ Test | 0 overlapping sessions | **PASS** |
| **Sample Uniqueness** | Across All Splits | 0 collisions (5,274 unique keys) | **PASS** |
| **Walking Confounding** | Beef Walking Count | Exactly 0 (171 from CVB only) | **PASS** |
| **Label Hygiene** | Excluded Classes | Exactly 0 samples entered manifests | **PASS** |
| **Frame Indexing Integrity** | Beef start_frame/end_frame | Exactly 0-based inclusive (end = n_frames - 1) | **PASS** |

### Split Hashes (Deterministic Provenance Recomputed Directly From Disk)
- `datasets/behavior/cvb_beef/manifest.csv`: `cfe54ba2dc939c4329fd5683e2ff832d1fd3376d263a1400452b206f779d5c36`
- `datasets/behavior/cvb_beef/train.csv`: `117d3191b175f4a6f43dc3cfb92f1ecbe42230f7f46a01c2d67cb81d84177e30`
- `datasets/behavior/cvb_beef/val.csv`: `897105d6266eba01b2b7bd45e2a7eb63bca7e9107faa202b07ba82e6d866b925`
- `datasets/behavior/cvb_beef/test.csv`: `0a67faf182a5ce8d3c02188656553310a6720a54d23f114b80d8b6093e00b30e`
- `datasets/behavior/cvb_beef/label_mapping.csv`: `08f1482f6ee1014885ee3dbb8bacc671d178d0570c56aa8726c008f5005a482f`
- `datasets/behavior/cvb/cvb_tracks_manifest.csv`: `0887e302e10a04d9e33b889055a350b20892d060b4fb8b1d04f1e4219758d2f9`
- `datasets/behavior/beef_cattle_behavior/manifest.csv`: `3f3ef4aa10fa5fda01b0d3365cfb28a13a0a690e3a96826f3723713d8fe10e69`

---

## 5. Partition Statistics & Class Breakdown

### 5.1 Combined 5-Class Distribution

| Behavior | Train Count (%) | Val Count (%) | Test Count (%) | Combined Total (%) | Origin |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Standing** | 821 (21.7%) | 103 (15.1%) | 113 (14.0%) | 1,037 (19.7%) | Both (CVB: 399, Beef: 638) |
| **Lying** | 1,337 (35.3%) | 253 (37.2%) | 258 (31.9%) | 1,848 (35.0%) | Both (CVB: 486, Beef: 1,362) |
| **Feeding** | 1,239 (32.7%) | 253 (37.2%) | 349 (43.1%) | 1,841 (34.9%) | Both (CVB: 1,295, Beef: 546) |
| **Drinking** | 269 (7.1%) | 45 (6.6%) | 63 (7.8%) | 377 (7.1%) | Both (CVB: 130, Beef: 247) |
| **Walking** | 119 (3.1%) | 26 (3.8%) | 26 (3.2%) | 171 (3.2%) | **CVB ONLY (CVB: 171, Beef: 0)** |
| **Total** | **3,785 (71.8%)** | **680 (12.9%)** | **809 (15.3%)** | **5,274 (100.0%)** | |

### 5.2 Sub-Dataset Breakdown
- **CVB Sub-Dataset** (2,481 segments across 66 source videos):
  - Train: 1,747 segments (44 source videos) | Drinking: 102, Feeding: 851, Lying: 363, Standing: 312, Walking: 119
  - Val: 312 segments (10 source videos) | Drinking: 10, Feeding: 169, Lying: 57, Standing: 50, Walking: 26
  - Test: 422 segments (12 source videos) | Drinking: 18, Feeding: 275, Lying: 66, Standing: 37, Walking: 26
- **Kaggle Beef Sub-Dataset** (2,793 clips across 201 sessions):
  - Train: 2,038 clips (140 sessions) | Drinking: 167, Feeding: 388, Lying: 974, Standing: 509, Walking: 0
  - Val: 368 clips (29 sessions) | Drinking: 35, Feeding: 84, Lying: 196, Standing: 53, Walking: 0
  - Test: 387 clips (32 sessions) | Drinking: 45, Feeding: 74, Lying: 192, Standing: 76, Walking: 0

---

## 6. Scientific Decisions & Governance Status

1. **Step 1 / Gate 1 Cleared for Behavior**: All primary Phase 3 datasets now possess fully audited, leakage-free, Git-tracked partitions.
2. **Scientific Classification**: The primary Behavior stack is classified as **`source-video / session-disjoint`**, NOT cow-disjoint.
3. **External Validation Benchmarks**:
   - `MmCows`: Frozen cow-disjoint split preserved as external identity-aware stress test.
   - `CBVD-5`: Preserved as secondary external validation benchmark.
4. **Standalone Verification**: Verified via `python scripts/build_cvb_beef_behavior_protocol.py --verify-only`.

---

## 7. Artifacts & File Registry

| Artifact | File Path | Description |
| :--- | :--- | :--- |
| **CVB Track Extraction Script** | `scripts/extract_cvb_track_segments.py` | Modal script extracting annotation-level CVB tracks on profile `tigerwood693`. |
| **Beef ffprobe Metadata Script** | `scripts/probe_beef_clips_ffprobe.py` | Modal script probing exact fps, n_frames, and duration_sec across all 4,337 Beef clips on profile `tigerwood693`. |
| **Master CVB Tracks Manifest** | `datasets/behavior/cvb/cvb_tracks_manifest.csv` | Master manifest of 3,693 CVB track segments with bounding-box metrics. |
| **Master Kaggle Beef Manifest** | `datasets/behavior/beef_cattle_behavior/manifest.csv` | 4,337 clips with exact verified n_frames and duration_sec metadata. |
| **Protocol Generation Script** | `scripts/build_cvb_beef_behavior_protocol.py` | Deterministic group-stratified partition generator and verification suite. |
| **Combined Manifest** | `datasets/behavior/cvb_beef/manifest.csv` | Master manifest of 5,274 canonical samples across CVB and Beef. |
| **Train Partition** | `datasets/behavior/cvb_beef/train.csv` | 3,785 training samples across 184 groups. |
| **Val Partition** | `datasets/behavior/cvb_beef/val.csv` | 680 validation samples across 39 groups. |
| **Test Partition** | `datasets/behavior/cvb_beef/test.csv` | 809 test samples across 44 groups. |
| **Label Mapping** | `datasets/behavior/cvb_beef/label_mapping.csv` | Official-to-canonical behavior mapping and exclusion notes. |
| **Split Report** | `datasets/behavior/cvb_beef/split_report.md` | Comprehensive forensic split report. |
| **Dataset Registry** | `datasets/dataset_registry.csv` | Updated with `CVB_Beef_Behavior` protocol entry. |

---

## 8. Next Steps

- [x] Extract authentic annotation-level track segments from all 502 CVB JSONs on Modal.
- [x] Probe physical Kaggle Beef clips with ffprobe on Modal and record exact frame counts and durations.
- [x] Build unified `cvb_beef` schema and map to canonical 5 classes with zero-based inclusive frame indexing for Beef.
- [x] Run deterministic group-stratified splitting with Seed 2026.
- [x] Verify 100% disjointness, sample uniqueness, label hygiene, and Walking provenance.
- [x] Save Git-tracked manifests, label mapping, and split report.
- [x] Update `datasets/dataset_registry.csv`, `memory/state.md`, `phase3_canonical_roadmap.md`, and research log index.
- [ ] Commit and push changes to `origin/main`.
- [ ] Mirror to `D:\custom-antigravity`.
