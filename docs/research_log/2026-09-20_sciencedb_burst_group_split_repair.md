# Research Log — 2026-09-20: ScienceDB Cattle BCS Burst-Group Split Repair

## 1. Executive Summary
Following the discovery of 88,944 near-duplicate cross-partition suspect pairs in the preliminary passage-disjoint split, forensic analysis proved that publisher passage IDs represent overlapping temporal video slices shifted by ~1 frame (e.g., `GS_1818` and `GS_1823`, MAE 0.26; `GS_1819` and `GS_1824`, MAE 0.30). To resolve this leakage, we built and executed a dedicated burst repair pipeline ([`scripts/repair_sciencedb_splits.py`](file:///d:/cattle-health-monitoring-multi-task-model/scripts/repair_sciencedb_splits.py)). The pipeline extracted perceptual fingerprints across all 53,566 images, conducted Multi-Index Hashing (MIH) candidate retrieval, performed in-memory normalized pixel MAE verification (threshold <= 5.0), and clustered initial passages into 5,653 connected burst groups via Disjoint Set Union. The resulting 70/15/15 stratified split was rigorously verified: 0 exact cross-duplicates, 0 confirmed cross-burst overlaps, and all 53,566 images cleanly assigned.

---

## 2. Context & Motivation
In Phase 3 Step 1, the preliminary split for ScienceDB Cattle BCS assumed that passage identifiers (`GS_XXXX`, `YM_XXXX`, `STEREO_blk_XXX`) represented independent cow walk-through events. However, an automated duplicate audit revealed that:
1. `GS_1818_2.jpg` (val) and `GS_1823_1.jpg` (train) had identical 64-bit dHash ($d=0$), identical aHash ($d=0$), and normalized 64x64 pixel MAE of 0.26 out of 255.
2. Consecutive frames of `GS_1823` mapped 1-to-1 to frames of `GS_1818` shifted by exactly one video frame.
3. Evaluating on this split would have allowed deep neural networks to evaluate test/validation performance on nearly identical video frames seen during training.
4. Naively merging all $d \le 6$ pairs would over-cluster cows due to static chute background railings; an empirical multi-signal burst verification protocol was required.

---

## 3. Forensic Evidence & Merged Burst Components
The pipeline evaluated all candidate pairs across the 53,566 images and isolated 9 confirmed cross-passage video burst overlaps (all having $d \le 1$ dHash/aHash and pixel MAE $\le 4.73$):

| Passage A | Passage B | Farm Source | Sample Frame A | Sample Frame B | dHash Dist | aHash Dist | Pixel MAE (0–255) |
|---|---|---|---|---|---|---|---|
| `GS_1818` | `GS_1823` | GS_Gansu | `GS_1818_2.jpg` | `GS_1823_1.jpg` | 0 | 0 | 0.26 |
| `GS_1819` | `GS_1824` | GS_Gansu | `GS_1819_10.jpg` | `GS_1824_8.jpg` | 0 | 0 | 0.30 |
| `GS_1820` | `GS_1825` | GS_Gansu | `GS_1820_2.jpg` | `GS_1825_1.jpg` | 0 | 0 | 0.27 |
| `GS_1879` | `GS_1882` | GS_Gansu | `GS_1879_10.jpg` | `GS_1882_6.jpg` | 0 | 0 | 0.31 |
| `GS_2120` | `GS_2128` | GS_Gansu | `GS_2120_1.jpg` | `GS_2128_1.jpg` | 0 | 0 | 0.29 |
| `GS_2122` | `GS_2131` | GS_Gansu | `GS_2122_2.jpg` | `GS_2131_2.jpg` | 1 | 1 | 3.67 |
| `GS_2123` | `GS_2132` | GS_Gansu | `GS_2123_2.jpg` | `GS_2132_2.jpg` | 0 | 0 | 0.33 |
| `GS_2124` | `GS_2133` | GS_Gansu | `GS_2124_4.jpg` | `GS_2133_5.jpg` | 0 | 0 | 4.73 |
| `GS_2314` | `GS_2319` | GS_Gansu | `GS_2314_1.jpg` | `GS_2319_1.jpg` | 0 | 0 | 0.28 |

Every one of these 9 pairs represents an overlapping temporal video burst generated during raw data preparation by the ScienceDB publishers. Through Disjoint Set Union, all member passages were clustered into unified burst groups and assigned exclusively to a single split partition.

---

## 4. Final Partition Statistics

| Metric | Train | Validation | Test | Total |
|---|---|---|---|---|
| **Images** | 37,045 (69.2%) | 8,481 (15.8%) | 8,040 (15.0%) | **53,566 (100.0%)** |
| **Burst Groups** | 3,958 (70.0%) | 850 (15.0%) | 845 (14.9%) | **5,653 (100.0%)** |
| **BCS 3.25** | 5,441 (14.7%) | 1,048 (12.4%) | 1,047 (13.0%) | **7,536 (14.1%)** |
| **BCS 3.50** | 9,219 (24.9%) | 2,109 (24.9%) | 1,928 (24.0%) | **13,256 (24.7%)** |
| **BCS 3.75** | 9,890 (26.7%) | 2,279 (26.9%) | 2,086 (25.9%) | **14,255 (26.6%)** |
| **BCS 4.00** | 8,662 (23.4%) | 2,059 (24.3%) | 1,835 (22.8%) | **12,556 (23.4%)** |
| **BCS 4.25** | 3,833 (10.3%) | 986 (11.6%) | 1,144 (14.2%) | **5,963 (11.1%)** |
| **GS_Gansu** | 20,821 | 4,837 | 4,175 | **29,833** |
| **YM_Farm2** | 10,309 | 2,195 | 2,257 | **14,761** |
| **STEREO_Farm3**| 5,915 | 1,449 | 1,608 | **8,972** |
| **Cross Duplicates (SHA-256)** | — | — | — | **0 (ZERO)** |
| **Cross Confirmed Bursts** | — | — | — | **0 (ZERO)** |

---

## 5. Scientific Designation & Integrity Boundaries
1. **Burst-Group-Disjoint / Sequence-Safe**:
   - The ScienceDB split is officially designated as **`burst-group-disjoint / sequence-safe`**.
   - Because the original authors (Huang et al., 2024) did not record or release biological animal RFID / ear tag numbers, we do **NOT** claim biological cow-disjoint evaluation on ScienceDB.
   - We do **NOT** claim zero leakage beyond what empirical checks (exact content hashes and perceptual burst frame matching) have verified.
2. **Backward Compatibility**:
   - In all exported split files (`train.csv`, `val.csv`, `test.csv`), the canonical grouping column `cow_id` is populated with the repaired `burst_group_id` (`GS_burst_XXXX`, `YM_burst_XXXX`, `STEREO_blk_XXX`).
   - Downstream models grouping by `cow_id` are automatically burst-safe.
   - The master index `datasets/bcs/sciencedb_bcs_index.csv` was synchronized with the repaired burst group IDs.

---

## 6. Artifacts Generated & Updated
- `scripts/repair_sciencedb_splits.py`: Canonical repair and verification script.
- `datasets/bcs/sciencedb/train.csv`: Repaired training split (37,045 images).
- `datasets/bcs/sciencedb/val.csv`: Repaired validation split (8,481 images).
- `datasets/bcs/sciencedb/test.csv`: Repaired test split (8,040 images).
- `datasets/bcs/sciencedb/burst_group_audit.csv`: Master audit of all 5,653 repaired burst groups.
- `datasets/bcs/sciencedb/confirmed_burst_overlap_links.csv`: Forensic log of the 9 merged overlapping burst pairs.
- `datasets/bcs/sciencedb/split_report.md`: Updated comprehensive split documentation.
- `datasets/bcs/sciencedb_bcs_index.csv`: Master dataset index updated with burst group IDs.
