# Research Log — 2026-09-20: Dryad BCS Dataset Discrepancy Audit

## 1. Executive Summary
Audited the Dryad Cattle BCS dataset (`Total_sorted_DGE_images.zip`, DOI: `10.5061/dryad.tqjq2bw4s`) to reconcile the discrepancy between legacy project documentation (~5,923 images across classes 2–6) and the physically observed local filesystem (5,940 TIFF files across classes 2–7). Proved that the discrepancy was caused by an artificial omission in `context/preprocess_bcs.py`, which hardcoded `VALID_CLASSES = ['2', '3', '4', '5', '6']` to fit a 5-class target, silently dropping class folder `'7'` (exactly 17 images from `Cow_52`). Class 7 is 100% authentic source data from the Winkler & Boucheron Criollo beef cattle study, which utilizes the 9-point Wagner beef BCS scale. Reconstructed the true biological subject census: the 148 session folders represent 54 unique biological cattle (`Cow_1` to `Cow_55`, with `Cow_39` absent). Verified all 5,940 images as valid, uncorrupted 224x224 RGB-encoded DGE TIFFs with zero cross-cow duplicate leakage. Generated master manifest, cow census, and updated legacy `bcs_index.csv`.

---

## 2. Context & Motivation
Under Phase 3 Step 1 of the Canonical Roadmap, following the ScienceDB and MmCows audits, the remaining discrepancy in the BCS stack was Dryad BCS. Earlier project notes (`context1_master_plan.txt`) cited 5,923 images and 147 cows across classes 2–6, while the physical filesystem inventory showed 5,940 files and folder `'7'`. Furthermore, the root index file `datasets/bcs/bcs_index.csv` was an empty 31-byte placeholder. A forensic audit was needed to determine whether folder `'7'` was corrupted/spurious data or valid benchmark material, reconcile the exact counts, and ensure zero methodological errors before external validation in Phase 3.

---

## 3. Forensic Findings & Data

### 3.1 Reconciliation of 5,923 vs 5,940 Images
Inspecting `context/preprocess_bcs.py` revealed the exact line responsible for the discrepancy:
```python
VALID_CLASSES = ['2', '3', '4', '5', '6']  # Folder '7' omitted!
```
And in `context/context2_bcs.txt`:
```python
DRYAD_LABEL_MAP = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4}
```
The legacy developer designed a 5-class model and discarded folder `'7'` without documenting the truncation.
Summing the actual files on disk by class folder:
- Class 1: 0 files (empty)
- Class 2: 546 images (5 session folders)
- Class 3: 536 images (18 session folders)
- Class 4: 2,212 images (48 session folders)
- Class 5: 1,962 images (52 session folders)
- Class 6: 667 images (24 session folders)
- **Subtotal (Classes 2–6)**: **5,923 images** across **147 session folders** (matches legacy expectation exactly!)
- **Class 7**: **17 images** (1 session folder: `Cow_52`)
- **Total Verified On Disk**: `5,923 + 17 = 5,940 images` across **148 session folders**.

### 3.2 Authenticity of Class 7 & The 1–9 Wagner Beef BCS Scale
- **Source Paper**: *Labeled RGB and depth images for cattle body condition score prediction* (Zachary Winkler & Laura Boucheron, New Mexico State University).
- **Animal Breed**: Criollo cattle (arid-adapted beef cattle breed).
- **BCS Scale**: Beef cattle in the United States are evaluated on the **1–9 Wagner BCS scale** (where 1 = emaciated, 5 = moderate, 9 = obese), in contrast to dairy cattle which use the 1–5 Ferguson scale. This explains why the dataset repository archive contains directories named `1` through `9`.
- **Herd Range**: In the NMSU experimental herd, animals scored between 2 and 7. Folders `1`, `8`, and `9` were distributed in the official archive by the authors but remained empty because no cow was emaciated (score 1) or obese (scores 8 or 9).
- **Cow_52 Longitudinal Provenance**: Class 7 contains 17 images (`68_DGE.tif` through `84_DGE.tif`) belonging to `Cow_52`. Crucially, `Cow_52` also appears in Class 6 as `Cow_52_29` (5 images: `36_DGE.tif` through `40_DGE.tif`), confirming that `Cow_52` is a genuine biological animal recorded at BCS 6 in one scoring session and BCS 7 in another session. Class 7 is 100% authentic source data.

### 3.3 Biological Identity Census: 54 Cattle vs 148 Folders
- The 148 subdirectories across classes 2–7 follow naming patterns like `Cow_1`, `Cow_1_27`, `Cow_1_6006`, `Cow_11_2`, `Cow_11_0018`.
- Regex parsing reveals that the suffixes represent repeated recording sessions/passes over time.
- There are **54 unique biological cattle** in total (`Cow_1` through `Cow_55`, with only `Cow_39` absent from the dataset).
- 47 of the 54 cattle appear in multiple recording sessions over time as their body condition changed (e.g. `Cow_11` appears in 5 sessions spanning BCS 2, BCS 4, and BCS 6).
- **Leakage Warning**: Legacy code treated each folder as an independent "cow" (claiming 147 cows). Any cross-validation or train/test split based on folder names causes severe animal-identity leakage across partitions. Grouping MUST be done by the base animal ID (`Cow_X`).

### 3.4 Image Integrity & Duplication
- **Readability**: All 5,940 files were verified with PIL. 100% are valid, uncorrupted 224x224 RGB-encoded TIFF files containing Depth (R), Grayscale (G), and Edge (B) channels.
- **Duplicates**: 348 duplicate SHA256 hash groups exist, consisting entirely of consecutive frames within the exact same recording session (e.g. frames 116 & 117 of `Cow_11_2`).
- **Cross-Cow Duplicate Leakage**: **0 duplicate hashes cross between different cows.**
- **Cross-Class Duplicate Leakage**: **0 duplicate hashes cross between different classes.**

---

## 4. Architectural Decisions & Action Plan

1. **Retain All 5,940 Images**: No deletion or relabeling. Class 7 is preserved as valid ground truth.
2. **Designation as External Validation Benchmark**: Dryad is confirmed as Secondary External Validation for BCS (multi-modal DGE, beef cattle, 1–9 scale). It is not part of the primary in-domain RGB multi-task model (which uses ScienceDB).
3. **Master Manifest & Updated Index**:
   - Master manifest exported to `datasets/bcs/dryad/manifest.csv` (5,940 rows, 11 fields, SHA256: `80725f2535663f72a661086dfb4329e9cd27c6e7dcb914af7a47184f026bb8bd`).
   - Per-cow census exported to `datasets/bcs/dryad/cow_audit.csv` (54 biological cows).
   - Legacy index `datasets/bcs/bcs_index.csv` updated with all 5,940 rows.

---

## 5. Artifacts & File Registry
- Master Manifest: `datasets/bcs/dryad/manifest.csv` (5,940 rows, 11 fields)
- Cow Census: `datasets/bcs/dryad/cow_audit.csv` (54 biological cows, 14 fields)
- Legacy Index (Updated): `datasets/bcs/bcs_index.csv` (5,940 rows)
- Forensic Report: `datasets/bcs/dryad/audit_report.md`
- Generator Script: `scripts/build_dryad_manifest.py`

---

## 6. Next Steps
1. Re-ID Protocol: Establish evaluation protocols for MultiCamCows2024 / OpenCows2020.
2. Proceed to STEP 2 (Cattle-perception feasibility audit).
