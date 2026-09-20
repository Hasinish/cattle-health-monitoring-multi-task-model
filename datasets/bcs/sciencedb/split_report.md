# ScienceDB Cattle BCS: Repaired Burst-Group Split Report

**Date:** 2026-09-20  
**Dataset Version:** ScienceDB DOI `10.57760/sciencedb.16704`  
**Evaluation Protocol:** Burst-Group-Disjoint / Sequence-Safe (Phase 3 Canonical)  
**Audit Status:** VERIFIED LEAK-FREE & BURST-SAFE  

## 1. Executive Summary & Forensic Context

This report documents the repaired, leak-free partition protocol for the ScienceDB Cattle BCS dataset.
- **Total Verified Images:** 53,566 RGB images (and 53,566 corresponding Pascal VOC XML annotations).
- **Original Parsed Passages:** 5,662 clusters.
- **Repaired Burst Groups:** 5,653 unified burst groups (connected components).
- **Confirmed Overlapping Burst Links Merged:** 9 cross-passage frame overlaps eliminated.
- **Partition Disjointness:** 100% burst-group disjoint across Train, Val, and Test.
  - `train_burst_groups ∩ val_burst_groups = 0`
  - `train_burst_groups ∩ test_burst_groups = 0`
  - `val_burst_groups ∩ test_burst_groups = 0`
- **Exact Duplicate Leakage:** 0 cross-partition duplicates (all byte duplicates locked into same partitions).
- **Burst Leakage Prevention:** 0 confirmed overlapping burst frames cross partitions.

## 2. Correction of Outdated Claims & Forensic Discovery

### The Previous Vulnerability:
The preliminary split (`datasets/bcs/sciencedb/split_report.md` dated 2026-09-20 morning) grouped data by naive author passage IDs (`GS_XXXX`, `YM_XXXX`, `STEREO_blk_XXX`).
An exhaustive perceptual near-duplicate audit (`docs/research_log/2026-09-20_phase3_duplicate_nearduplicate_audit.md`) flagged 88,944 near-duplicate suspect pairs (dHash <= 6).
Forensic analysis proved that passages like `GS_1818` (in val) and `GS_1823` (in train) were **not independent cow visits**; they were overlapping 1-frame-shifted temporal slices of the **exact same continuous video burst** (dHash distance 0, aHash distance 0, normalized pixel MAE = 0.83).

### The Scientific Solution:
We replaced naive passage-level grouping with empirical **connected burst group clustering**:
1. **Exact Content Hash**: SHA-256 byte-level identity.
2. **Perceptual Hash Proximity**: 64-bit dHash <= 2 AND 64-bit aHash <= 2 within the same farm source.
3. **Pixel MAE Verification**: Normalized 64x64 grayscale Mean Absolute Error <= 5.0 (out of 255).
4. **Graph Connected Components**: Disjoint Set Union (Union-Find) transitively clusters all overlapping passages into atomic burst groups.

### Scientific Integrity Note (No Biological Cow ID Fabrication):
> [!IMPORTANT]
> The authors of ScienceDB (Huang et al., 2024) **did not record or publish biological ear tag / RFID cow IDs**.
> Therefore, this split is scientifically designated as **`burst-group-disjoint / sequence-safe`**.
> We do NOT claim biological cow-disjoint evaluation on ScienceDB, nor do we claim zero leakage beyond what empirical checks establish.

## 3. Final Split Statistics

### Burst Groups & Image Totals:

| Split | Burst Groups | % Groups | Total Images | % Images |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | 3958 | 70.0% | 37,045 | 69.2% |
| **Val** | 850 | 15.0% | 8,481 | 15.8% |
| **Test** | 845 | 14.9% | 8,040 | 15.0% |
| **Total** | **5,653** | **100.0%** | **53,566** | **100.0%** |

### BCS Class Distribution Across Splits:

| BCS Class | Overall Count (%) | Train Count (%) | Val Count (%) | Test Count (%) |
| :--- | :--- | :--- | :--- | :--- |
| **3.25** | 7,536 (14.1%) | 5,298 (14.3%) | 1,029 (12.1%) | 1,209 (15.0%) |
| **3.5** | 13,256 (24.7%) | 9,134 (24.7%) | 2,123 (25.0%) | 1,999 (24.9%) |
| **3.75** | 14,255 (26.6%) | 9,954 (26.9%) | 2,069 (24.4%) | 2,232 (27.8%) |
| **4.0** | 12,556 (23.4%) | 8,931 (24.1%) | 1,875 (22.1%) | 1,750 (21.8%) |
| **4.25** | 5,963 (11.1%) | 3,728 (10.1%) | 1,385 (16.3%) | 850 (10.6%) |

### Verification Code:
To re-verify this split at any time, run:
```powershell
python scripts/repair_sciencedb_splits.py --verify-only
```
