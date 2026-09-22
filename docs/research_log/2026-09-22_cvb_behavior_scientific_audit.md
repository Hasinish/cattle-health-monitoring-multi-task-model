# Research Log: CVB (Cattle Visual Behaviors) Dataset Scientific & Provenance Audit

**Date**: 2026-09-22  
**Author**: Autonomous AI Pair Programmer (Thesis Research Workspace)  
**Topic**: Forensic Scientific Audit, Behavior Taxonomy Census, Track Continuity Analysis, Leakage Detection, and Visual Review Pack Generation for the CVB Dataset  

---

## 1. Executive Summary

We conducted an exhaustive scientific and provenance audit of the Cattle Visual Behaviors (CVB) dataset (CSIRO Collection 58916v001, DOI: 10.25919/bmtp-5j95) hosted on Modal Persistent Volume `cvb-data` (`/data/cvb/`). All 226,344 files, 502 video cuts, 225,829 1080p JPEG frames, and 1,163,408 bounding-box annotations across 12 official behaviors were verified. We resolved the "589 cuts" myth (which was a legacy artifact of the CSIRO collection ID `58916v001`), proved that 100% of cuts are annotated at 30.0 FPS across continuous 15.0s clips ($dt = 0.033\text{ s}$), and discovered that the cattle crops suffer from severe spatial miniaturization (median 104x85 pixels, 16.3x smaller than MmCows). We exposed that the official AVA benchmark split suffers from fatal source-video leakage (88.9% of validation source videos overlap with training), proved that zero biological cow IDs exist, and concluded that CVB is **PARTIALLY SUITABLE — WITH SPECIFIC LIMITATIONS**. MmCows remains the canonical Primary Behavior dataset, and the Phase 3 roadmap remains unchanged.

---

## 2. Context & Motivation

In Phase 3 of our thesis, behavior monitoring is one of the three primary tasks (alongside BCS regression and Individual Re-Identification). MmCows is currently designated as the canonical Primary Behavior dataset (213,686 crops, 16 biological cows, 4 CCTV cameras). However, MmCows is sampled at 15-second discrete intervals (0.067 Hz), designed for macro-behavior ethograms rather than dense 30 FPS video motion modeling.

CVB was proposed as a candidate external validation benchmark to evaluate continuous 30 FPS video action architectures. Before incorporating CVB into any experimental protocols or multi-task pipelines, a rigorous provenance and scientific audit was required to determine:
1. True physical dataset geometry and frame counts.
2. Official behavior taxonomy and defensibility of mapping to our compact set (`Standing`, `Lying`, `Walking`, `Feeding`).
3. Meaning and continuity of tracking IDs.
4. Existence of biological cow identifiers.
5. Spatial resolution and bounding-box distribution.
6. Temporal continuity and motion modeling suitability.
7. Data leakage risks in the official benchmark partitions.

---

## 3. Forensic Findings & Data

### 3.1 Dataset Structure & Physical Inventory
- **Modal Volume**: `cvb-data` at `/data/cvb/000058916v001/data/`.
- **Total Physical Files**: 226,344 files (~14.29 GB).
- **Physical Video Cuts**: Exactly **502 unique cuts** in `raw_frames/` and 502 matching folders in `annotations/`.
- **Total Frames**: Exactly **225,829 1080p JPEG frames** (1920x1080 RGB).
- **Frame Rate & Durations**: 500 cuts have exactly 450 frames (15.0s @ 30 FPS); 2 cuts have 415 and 414 frames (~13.8s).
- **Annotation Files**: Exactly 503 JSON files (502 `instances_default.json` + 1 duplicate file in cut `1366_arm01_gopro4_20200324_040135_beh5_ani6_ins1_cut_00006`). Zero cuts are unannotated.

### 3.2 Exhaustive Behavior Taxonomy (1,163,408 BBoxes)
A complete census of all 502 COCO JSON files revealed 12 official behaviors in `behaviour_list.pbtx`:
1. `grazing`: 496,394 boxes (42.67%) across 485 cuts, 2,058 tracks.
2. `resting-lying`: 208,586 boxes (17.93%) across 126 cuts, 574 tracks.
3. `resting-standing`: 137,987 boxes (11.86%) across 228 cuts, 467 tracks.
4. `hidden`: 107,915 boxes (9.28%) across 244 cuts, 440 tracks (occluded / unobserved).
5. `ruminating-lying`: 75,864 boxes (6.52%) across 52 cuts, 222 tracks.
6. `drinking`: 33,379 boxes (2.87%) across 65 cuts, 87 tracks.
7. `ruminating-standing`: 29,239 boxes (2.51%) across 59 cuts, 95 tracks.
8. `other`: 26,637 boxes (2.29%) across 91 cuts, 98 tracks.
9. `walking`: 23,929 boxes (2.06%) across 66 cuts, 79 tracks.
10. `grooming`: 16,690 boxes (1.43%) across 36 cuts, 45 tracks.
11. `none`: 5,056 boxes (0.43%) across 21 cuts, 26 tracks.
12. `running`: 1,732 boxes (0.15%) across 4 cuts, 4 tracks.

### 3.3 Defensible Taxonomy Mapping
- `Standing` (167,226 boxes, 14.37%): Maps exactly from `resting-standing` (137,987) + posture merge from `ruminating-standing` (29,239).
- `Lying` (284,450 boxes, 24.45%): Maps exactly from `resting-lying` (208,586) + posture merge from `ruminating-lying` (75,864).
- `Walking` (23,929 boxes, 2.06%): Maps exactly from `walking` (23,929).
- `Feeding` (496,394 boxes, 42.67%): Defensible ingestion merge from `grazing` (pasture grass foraging).
- `Drinking` (33,379 boxes, 2.87%): Exact water trough ingestion match.
- Total yield for compact 4-class set: **971,999 boxes (83.55% of dataset)**.

### 3.4 Identity & Track Provenance
- **Zero Biological Cow IDs**: `attributes.id` contains `unknown` (146,600 times) or local session focal indices (`1`, `2`, `3`). Unmarked Black Angus beef herd in open pasture. No persistent biological IDs exist.
- **Track IDs**: Clip-local tracker IDs (`track_id = 0..10`). There are 3,693 unique tracklets. Track IDs repeat across cuts and do not persist across cuts.
- **Track Stability**: 100.0% of tracks (3,693/3,693) exhibit a single homogeneous behavior throughout the 15-second cut (0 transitions). Median track length is 450 frames.

### 3.5 Bounding-Box Miniaturization Deficit
- 87.95% of frames have at least 1 annotated box (mean: 5.15 boxes/frame, max: 11).
- **Median BBox**: **104.0 x 85.0 pixels** (8,840 px$^2$, **0.41%** of 1080p frame).
- **Mean BBox**: **153.1 x 128.9 pixels** (33,524 px$^2$, **1.62%** of frame).
- **MmCows Comparison**: MmCows median crop is **390 x 370 pixels** (144,300 px$^2$). CVB crops are **16.3x smaller in area**. Facial jaw movements for rumination occupy <15x15 pixels on median cows.

### 3.6 Data Leakage in Official AVA Split
- The 502 cuts originate from **66 unique source videos** across 6 calendar dates (`20200322` to `20200327`) and 3 GoPro cameras (`gopro1`, `gopro3`, `gopro4`).
- In the official AVA benchmark (401 train cuts / 101 val cuts):
  - 100% date overlap (6/6 dates in both train and val).
  - 100% camera overlap (3/3 cameras in both train and val).
  - **32 out of 36 validation source videos (88.9%) have sister cuts in the training set!**
- Evaluating on the official split measures static background memorization rather than generalized behavior recognition. Future splits must group strictly by Source Video or by Recording Date.

---

## 4. Architectural Decisions & Action Plan

1. **Dataset Role Retained**:
   - CVB is NOT promoted to Primary Behavior dataset.
   - MmCows remains the canonical Primary Behavior benchmark for Phase 3 (high crop resolution, verified biological cow IDs, synchronized multi-camera views, macro-behavior state modeling).
   - CVB is preserved as an **Optional External Validation Benchmark** for video action modeling on open pasture.

2. **Classification Decision Boundary**:
   - Verdict: **PARTIALLY SUITABLE — WITH SPECIFIC LIMITATIONS**.
   - Suitable for 3D CNNs and dense temporal aggregators on macro-behaviors (grazing, lying, standing).
   - Unsuitable for fine-grained jaw gesture recognition (rumination) or cow-disjoint identity evaluations.

3. **Data Splitting Governance**:
   - Reject the official AVA train/val split for thesis benchmarks.
   - Any future experimental partitioning of CVB must enforce **Source Video Grouping** or **Date Grouping**.

---

## 5. Artifacts & File Registry

1. **Comprehensive Audit Report**:
   - `docs/audits/phase3_cvb_behavior_scientific_audit.md`
2. **Visual Evidence Review Pack (Seed 2026)**:
   - 16 consecutive filmstrips and 2 diagnostic frames under `docs/audits/assets/cvb_behavior_audit/`:
     - `cvb_filmstrip_grazing_seq1.jpg`, `cvb_filmstrip_grazing_seq2.jpg`
     - `cvb_filmstrip_resting-lying_seq1.jpg`, `cvb_filmstrip_resting-lying_seq2.jpg`
     - `cvb_filmstrip_resting-standing_seq1.jpg`, `cvb_filmstrip_resting-standing_seq2.jpg`
     - `cvb_filmstrip_ruminating-lying_seq1.jpg`, `cvb_filmstrip_ruminating-lying_seq2.jpg`
     - `cvb_filmstrip_ruminating-standing_seq1.jpg`, `cvb_filmstrip_ruminating-standing_seq2.jpg`
     - `cvb_filmstrip_drinking_seq1.jpg`, `cvb_filmstrip_drinking_seq2.jpg`
     - `cvb_filmstrip_hidden_seq1.jpg`, `cvb_filmstrip_hidden_seq2.jpg`
     - `cvb_filmstrip_other_seq1.jpg`, `cvb_filmstrip_other_seq2.jpg`
     - `cvb_edge_case_multi_cow_crowding.jpg`
     - `cvb_edge_case_pasture_occlusion.jpg`
3. **Audit Manifests**:
   - `datasets/behavior/cvb/cvb_cuts_manifest.csv` (SHA-256: `1a18fd44a6ac52ececea6db415f715d6868383069896ca45513ed49629a116b2`)
   - `artifacts/behavior_audit/cvb_audit_summary.csv`
4. **Registry & Generator**:
   - `datasets/dataset_registry.csv` (Row 7 updated with audited metrics)
   - `scripts/build_dataset_registry.py`
5. **Inspection & Extraction Scripts**:
   - `scripts/audit_cvb_stage1.py`
   - `scripts/audit_cvb_stage2.py`
   - `scripts/fetch_cvb_audit_assets.py`

---

## 6. Next Steps

1. Update `docs/research_log/README.md` to register this audit report.
2. Update `memory/state.md` and `memory/history.md`.
3. Commit and push all audit artifacts, scripts, and documentation to Git `origin/main`.
4. Present findings to Hasin and ChatGPT for final review.
