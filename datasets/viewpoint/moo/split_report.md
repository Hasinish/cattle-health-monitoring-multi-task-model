# MOO Synthetic Cattle Viewpoint: Identity-Disjoint Split Report

**Date:** 2026-09-21  
**Dataset:** MOO (Multi-view Oriented Observations, CEA Kalisteo)  
**Task:** Synthetic Viewpoint Pretraining / Representation Learning (Phase 3 Step 2.4 / Step 3)  
**Random Seed:** 2026  
**Split Protocol:** 100% Identity-Disjoint (Cow-Disjoint) Partitioning  

---

## 1. Executive Summary & Forensic Rationale

This report documents the canonical identity-disjoint partition protocol for the MOO synthetic cattle viewpoint dataset.

### Why Official MOO Splits Were Rejected for Viewpoint Evaluation:
1. **Designed for Metric Retrieval, Not Viewpoint Generalization**: The official `/data/splits/` partition files (`train_topside.txt`, `test_topside_gallery.txt`, etc.) were authored exclusively as a closed-set Re-ID gallery/query retrieval benchmark.
2. **100% Identity Overlap**: `train_topside.txt` contains all 1,000 cows. All 500 cows in `test_topside_gallery.txt` and `test_topside_query.txt` are already present in `train_topside.txt` (100% cow identity leakage).
3. **100% Image Overlap**: `train_topside.txt` contains all 128,000 images in the dataset. Testing viewpoint estimation on the official test set evaluates on identical images already seen during training.
4. **Mandatory Identity-Disjoint Split**: To test whether viewpoint representations generalize across unseen anatomical variations, coat patterns, and body geometries, synthetic cows must be strictly partitioned at the identity level.

---

## 2. Dataset Censusing & Filtering Rules

- **Total Available Synthetic Cows:** 1,000 (`cow0` through `cow999`)
- **Total Rendered Images:** 128,000 (exactly 128 images per cow)
- **Usable Directional Classes (8 Classes):** `front`, `front-left`, `left`, `back-left`, `back`, `back-right`, `right`, `front-right`
- **Total Usable Images:** 96,000 (exactly 96 images per cow, 75.0% of dataset)
- **Excluded High-Nadir Images (`top-*`):** 32,000 (exactly 32 images per cow, 25.0% of dataset)
  - `top-front`: 3,997
  - `top-front-left`: 3,985
  - `top-front-right`: 3,980
  - `top-left`: 4,009
  - `top-right`: 4,041
  - `top-back`: 4,002
  - `top-back-left`: 4,004
  - `top-back-right`: 3,982
- **Unmapped / Unknown Labels:** 0

---

## 3. Split Partitioning & Validation Metrics

### Partition Summary:

| Split | Synthetic Cows | % Cows | Total Images | % Images | Images / Cow |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | **800** | 80.0% | **76,800** | 80.0% | 96 |
| **Val** | **100** | 10.0% | **9,600** | 10.0% | 96 |
| **Test** | **100** | 10.0% | **9,600** | 10.0% | 96 |
| **Total** | **1,000** | **100.0%** | **96,000** | **100.0%** | **96** |

### Leakage Verification:
- `train_cows ∩ val_cows`: **0** (PASSED)
- `train_cows ∩ test_cows`: **0** (PASSED)
- `val_cows ∩ test_cows`: **0** (PASSED)
- `train_images ∩ val_images`: **0** (PASSED)
- `train_images ∩ test_images`: **0** (PASSED)
- `val_images ∩ test_images`: **0** (PASSED)
- All 1,000 cow IDs assigned exactly once: **YES** (PASSED)
- All 96,000 usable images assigned exactly once: **YES** (PASSED)

---

## 4. Class Distribution Across Splits

| Class Name | Train Count (%) | Val Count (%) | Test Count (%) | Total Usable (%) |
| :--- | :--- | :--- | :--- | :--- |
| `front` | 9,584 (12.5%) | 1,180 (12.3%) | 1,170 (12.2%) | 11,934 (12.4%) |
| `front-left` | 9,644 (12.6%) | 1,200 (12.5%) | 1,218 (12.7%) | 12,062 (12.6%) |
| `left` | 9,638 (12.5%) | 1,199 (12.5%) | 1,191 (12.4%) | 12,028 (12.5%) |
| `back-left` | 9,508 (12.4%) | 1,206 (12.6%) | 1,210 (12.6%) | 11,924 (12.4%) |
| `back` | 9,681 (12.6%) | 1,192 (12.4%) | 1,207 (12.6%) | 12,080 (12.6%) |
| `back-right` | 9,614 (12.5%) | 1,195 (12.4%) | 1,186 (12.4%) | 11,995 (12.5%) |
| `right` | 9,585 (12.5%) | 1,219 (12.7%) | 1,209 (12.6%) | 12,013 (12.5%) |
| `front-right` | 9,546 (12.4%) | 1,209 (12.6%) | 1,209 (12.6%) | 11,964 (12.5%) |
| **Total** | **76,800** (100.0%) | **9,600** (100.0%) | **9,600** (100.0%) | **96,000** (100.0%) |

---

## 5. Artifacts & Cryptographic Checksums

| File | Rows / Size | SHA-256 Checksum |
| :--- | :--- | :--- |
| `datasets/viewpoint/moo/train.csv` | 76,801 lines (76,800 samples) | `596a1a49c985223202bdf04b6d2b20d6544ab01d7683827fe65b70f6fc521e61` |
| `datasets/viewpoint/moo/val.csv` | 9,601 lines (9,600 samples) | `7995c1e357cc33ccc17c9d70f0a210d1035839bfdd4ff26d7179201fdf42b261` |
| `datasets/viewpoint/moo/test.csv` | 9,601 lines (9,600 samples) | `276603b9589f66d6d66f19140889d2e5ca369e69493815a23b47e13fb0ab4a8d` |
