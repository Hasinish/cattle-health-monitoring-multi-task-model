# RT-DETR Cropped Viewpoint Dataset Split Report

**Date:** 2026-09-23 22:08:34  
**Seed:** 2026  
**Method:** Deterministic Stratified Group K-Fold (`StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=2026)`)  
**Grouping Key:** `duplicate_group_id` (Zero leakage across duplicate/near-duplicate image clusters)  
**Target Proportions:** ~70% Train (14/20 folds), ~15% Val (3/20 folds), ~15% Test (3/20 folds)  
**Total Samples:** 879  
**Total Duplicate Groups:** 879  

---

## 1. Split Distribution & Class Balance

| Split | Total Samples | % of Dataset | Groups | `front` | `side` | `rear` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`train`** | **616** | **70.1%** | 616 | 276 (44.8%) | 155 (25.2%) | 185 (30.0%) |
| **`val`** | **132** | **15.0%** | 132 | 57 (43.2%) | 34 (25.8%) | 41 (31.1%) |
| **`test`** | **131** | **14.9%** | 131 | 59 (45.0%) | 33 (25.2%) | 39 (29.8%) |
| **OVERALL** | **879** | **100.0%** | **879** | **392 (44.6%)** | **222 (25.3%)** | **265 (30.1%)** |

---

## 2. Leakage Protection Verification

- **Train vs. Val Group Overlap:** `0` (Zero shared `duplicate_group_id`)
- **Train vs. Test Group Overlap:** `0` (Zero shared `duplicate_group_id`)
- **Val vs. Test Group Overlap:** `0` (Zero shared `duplicate_group_id`)
- **Quarantined Failure (`rear_0003`):** Verified excluded from all three partitions.

---

## 3. Split Manifest Cryptographic Hashes (SHA-256)

| Split File | Path | Row Count | SHA-256 Checksum |
| :--- | :--- | :---: | :--- |
| `train.csv` | `datasets/viewpoint/self_clean_v1_rtdetr_crop/train.csv` | 616 | `26a8a6924a15dd0d346ab13bdbf3e9746c2216693d2ff2a6c1c785c631ab04af` |
| `val.csv` | `datasets/viewpoint/self_clean_v1_rtdetr_crop/val.csv` | 132 | `07cda3d88c24b40842c9e9e7dfa9e3ed147274b1b4084ec898e36aab1685cea2` |
| `test.csv` | `datasets/viewpoint/self_clean_v1_rtdetr_crop/test.csv` | 131 | `7dd3202dd9ff8421eb6df7541c7170ac13d40d8d3b1f96058edae9f1a33c2a41` |

---

## 4. Evaluation Protocol Constraints

> [!IMPORTANT]
> The `test.csv` partition (131 images across 131 duplicate groups) is strictly FROZEN.
> It must NEVER be used for:
> 1. Learning rate or hyperparameter tuning
> 2. Model architecture selection
> 3. Epoch selection or Early Stopping
> Model checkpoints must be selected exclusively using validation performance (`val.csv`).
