# Research Log — 2026-09-20: ScienceDB Cattle BCS Identity Audit & Leakage-Free Split

## 1. Executive Summary
- **Dataset:** Science Data Bank (ScienceDB) Dairy Cow BCS Target Detection Dataset (DOI: `10.57760/sciencedb.16704`).
- **Physical Verification:** 53,566 RGB JPEG images (1024x576) and 53,566 Pascal VOC XML annotation files restored and verified locally in `datasets/bcs/sciencedb_bcs/dataset/`.
- **Key Forensic Discovery:** Disproved the project claim that ScienceDB contains "10,898 unique cows". The legacy identity parser treated 30 FPS video frames of the stereo camera setup as independent biological cows, leading to **94.64% of stereo cow passage sequences being shattered across train, val, and test**.
- **Resolution:** Re-engineered the identity architecture into **5,662 atomic passage clusters** (GS bursts, YM bursts, duplicate-linked clusters, and stereo temporal continuous blocks). Generated a 100% leak-free, passage-disjoint 70/15/15 train/val/test split with zero sequence or duplicate leakage.

---

## 2. Forensic Audit of the "10,898 Cows" Parser Flaw

### Legacy Parser Scheme (`context/preprocess_sciencedb_bcs.py`)
```python
if stem.startswith('GS_') or stem.startswith('YM_'):
    parts = stem.split('_')
    cow_id = f"{parts[0]}_{parts[1]}"
elif stem.startswith('L-i') or stem.startswith('R-i'):
    cow_id = 'i' + stem[3:]  # e.g. L-i1035 -> i1035
```
- Total `GS_` sequences: 3,347
- Total `YM_` sequences: 2,056
- Total `L-i/R-i` unique frame numbers: 5,495
- Claimed total "cows": 3,347 + 2,056 + 5,495 = **10,898**.

### The Mathematical & Visual Evidence of Leakage
1. **Stereo Frame Sequences (`L-i` and `R-i`):**
   - 8,972 total images (4,632 Left, 4,340 Right).
   - In 80.85% of cases, Left and Right cameras for the same frame number have identical BCS classes.
   - Bounding box coordinates translate continuously across consecutive frame numbers (e.g. `L-i1035` box `[91, 189, 536, 502]` -> `L-i1036` box `[69, 127, 456, 456]`).
   - Cows move past the camera in 261 continuous burst sequences of 5 to 30 frames.
   - The legacy parser treated each frame number as an independent cow. In the resulting legacy random split, **247 out of 261 passage blocks (94.64%) were fragmented across train, val, and test**.

2. **Duplicate Video Sequences in `YM`:**
   - Byte-level MD5 hashing across all 53,566 images exposed **19 exact duplicate image groups (38 files)**.
   - `YM_1249` (frames 1-5) and `YM_1264` (frames 6-10) are identical byte-for-byte image frames of the exact same cow passage.
   - `YM_1794` (frames 1-4) and `YM_1905` (frames 1-4) are identical frames labeled `3.75` in sequence 1794 and `4.0` in sequence 1905.
   - In the legacy split, `YM_1249` and `YM_1264` were placed into different splits, leaking identical images between train and test.

3. **Publisher Ground Truth:**
   - Exhaustive scanning of all 53,566 XML annotations confirmed that only Pascal VOC bounding box coordinates and BCS labels are provided.
   - The original publishers (Huang et al., 2024) did NOT record individual ear tag or RFID numbers.
   - True biological cows cannot be distinguished across different recording days. The maximal independent observational unit is the **passage/visit event** (5,662 clusters).

---

## 3. De-Leakage Grouping Methodology

All 53,566 images were partitioned into **5,662 atomic passage clusters**:
1. **`GS_Gansu` (29,833 images):** 3,347 burst sequences (`GS_1` to `GS_3347`).
2. **`YM_Farm2` (14,761 images):** 2,054 sequences. Duplicate sequences `YM_1249` & `YM_1264`, and `YM_1794` & `YM_1905` were merged into unified clusters.
3. **`STEREO_Farm3` (8,972 images):** 261 temporal blocks (`STEREO_blk_001` to `STEREO_blk_261`). Consecutive frames within <= 5 frame intervals (~0.2s) and both Left/Right camera views are locked together.

---

## 4. Stratified Group Partitioning Results

Using deterministic `random.seed(42)` stratified by `(farm_source, primary_bcs)`:

| Split | Passage Clusters | % Clusters | Total Images | % Images |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | 3,963 | 70.0% | 37,126 | 69.3% |
| **Val** | 849 | 15.0% | 8,099 | 15.1% |
| **Test** | 850 | 15.0% | 8,341 | 15.6% |
| **Total** | **5,662** | **100.0%** | **53,566** | **100.0%** |

### BCS Class Distribution Across Splits:
| BCS Class | Overall Count (%) | Train Count (%) | Val Count (%) | Test Count (%) |
| :--- | :--- | :--- | :--- | :--- |
| **3.25** | 7,536 (14.1%) | 5,454 (14.7%) | 1,032 (12.7%) | 1,050 (12.6%) |
| **3.50** | 13,256 (24.7%) | 9,251 (24.9%) | 2,010 (24.8%) | 1,995 (23.9%) |
| **3.75** | 14,255 (26.6%) | 9,908 (26.7%) | 2,207 (27.3%) | 2,140 (25.7%) |
| **4.00** | 12,556 (23.4%) | 8,676 (23.4%) | 2,003 (24.7%) | 1,877 (22.5%) |
| **4.25** | 5,963 (11.1%) | 3,837 (10.3%) | 847 (10.5%) | 1,279 (15.3%) |

---

## 5. Integrity & Verification Assertions
The verification script `python scripts/build_sciencedb_splits.py --verify-only` programmatically enforces:
1. `train_clusters.isdisjoint(val_clusters) == True`
2. `train_clusters.isdisjoint(test_clusters) == True`
3. `val_clusters.isdisjoint(test_clusters) == True`
4. `len(train) + len(val) + len(test) == 53,566`
5. 100% of image paths exist on disk.
6. Zero cross-split duplicate image leakage.

---

## 6. Generated Deliverables
- `datasets/bcs/sciencedb/train.csv` (37,126 rows)
- `datasets/bcs/sciencedb/val.csv` (8,099 rows)
- `datasets/bcs/sciencedb/test.csv` (8,341 rows)
- `datasets/bcs/sciencedb/identity_audit.csv` (5,662 rows)
- `datasets/bcs/sciencedb/split_report.md`
- `scripts/build_sciencedb_splits.py`
- Updated `datasets/bcs/sciencedb_bcs_index.csv` (53,566 rows aligned with leak-free split)
