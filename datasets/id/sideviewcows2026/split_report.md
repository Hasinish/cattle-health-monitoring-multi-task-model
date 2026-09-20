# SideViewCows2026 Canonical Re-ID Protocols & Leakage Audit Report

## 1. Executive Summary
- **Dataset**: SideViewCows2026 (Zenodo DOI: 10.5281/zenodo.21605650)
- **Scientific Role**: **PRIMARY Re-ID Benchmark** (adopted 2026-09-20 under approved MultiCam contingency)
- **Total Images**: 80,260 (100% verified on disk)
- **Total Binary Segmentation Masks**: 80,260 (100% verified on disk; exactly matched 1-to-1)
- **Total Biological Cows**: 110
- **Unique SHA-256 Checksums**: 80,260 (0 exact duplicates in dataset)
- **Status**: **VERIFIED & LEAK-FREE** across all 4 canonical protocols

---

## 2. Subset Breakdown & Recording Grouping
| Subset | Images | Cows | Description | Temporal Span |
|---|---|---|---|---|
| `parlor` | 54,393 | 110 | Fixed camera at milking parlor entrance | Days 0.0 to 156.0 (~5 months) |
| `barn` | 25,260 | 69 | Handheld video recorded in barn | Days 355.4 to 592.4 (~1 to 1.6 years) |
| `snapshots` | 607 | 63 | Unconstrained photos (indoor/outdoor, lying down) | Days 355.3 to 634.4 (~1 to 1.7 years) |

### Recording Session Recovery
- Author warning: *"Frames from the same recording are strongly correlated. Group by individual and recording rather than sampling frames at random."*
- Implementation: Grouped contiguous frames where `dt <= 60s` into discrete recording sessions (`recording_id`).
- Total discrete recording sessions recovered: **3,604** across all 110 cows (average ~33 sessions/cow).

---

## 3. Canonical Evaluation Protocols

### Protocol A: Cross-Setting Domain Shift (`protocol_cross_setting.csv`)
- **Objective**: Benchmark feature representation robustness and retrieval performance under camera and domain shift.
- **Gallery**: Fixed camera `parlor` entrance frames (reference domain).
- **Query 1**: Handheld video in `barn` (Domain shift 1: motion blur, varying angles).
- **Query 2**: Unconstrained `snapshots` (Domain shift 2: extreme angle/posture shift).
- **Distribution**:
  - `gallery` (parlor for 69 multi-setting cows): **36,811** images
  - `query_barn` (barn for 69 cows): **25,260** images
  - `query_snapshots` (snapshots for 63 cows): **607** images
  - `train` (parlor for 41 parlor-only cows): **17,582** images (available for representation learning)

### Protocol B: Longitudinal / Cross-Temporal (`protocol_longitudinal.csv`)
- **Objective**: Benchmark coat pattern persistence over time using verified `time_offset_s`.
- **Distribution**:
  - `train_gallery_early` (first 60% of parlor sessions per cow): **35,433** images
  - `query_parlor_late` (subsequent 40% of parlor sessions per cow): **18,960** images
  - `query_long_barn` (barn video, >200 days later): **25,260** images
  - `query_long_snapshots` (snapshots, >200 days later): **607** images
- **Assertion**: For every cow, all query sessions occur strictly after all gallery sessions (`min_query_time > max_gallery_time`).

### Protocol C: Open-Set / Identity-Disjoint (`protocol_open_set.csv`)
- **Objective**: Benchmark zero-shot metric embedding generalization to completely unseen cow identities.
- **Stratification**: Balanced across subset presence (`multi`: 63, `parlor_barn`: 6, `parlor_only`: 41).
- **Split Breakdown**:
  - `train`: **77 cows** (57,605 images) [44 multi, 4 parlor_barn, 29 parlor_only]
  - `val`: **11 cows** (7,225 images) [6 multi, 1 parlor_barn, 4 parlor_only]
  - `test`: **22 cows** (15,430 images) [13 multi, 1 parlor_barn, 8 parlor_only]
- **Disjointness**: 100% disjoint cow identities (`train ∩ val = ∅`, `train ∩ test = ∅`, `val ∩ test = ∅`).

### Protocol D: Closed-Set Identification (`protocol_closed_set.csv`)
- **Objective**: Standard 110-class metric identification under sequence-safe recording protection.
- **Method**: Chronological recording session partition per cow (70% train, 15% val, 15% test).
- **Distribution**:
  - `train` (in-domain early parlor): **40,745** images
  - `val` (in-domain mid parlor): **7,373** images
  - `test_parlor` (in-domain late parlor): **6,275** images
  - `test_barn` (cross-domain handheld video): **25,260** images
  - `test_snapshots` (cross-domain unconstrained): **607** images
- **Sequence Protection**: Zero adjacent-frame leakage. No video passage crosses split boundaries.

---

## 4. Leakage & Mathematical Assertions
1. **Total Sample Count**: Exactly 80,260 images and 80,260 masks across all protocol files.
2. **Exact Byte Duplicates**: 0 exact cross-partition duplicate pairs (SHA-256 uniqueness = 100%).
3. **Identity Disjointness**: 0 cow overlap between Train, Val, and Test in Protocol C.
4. **Temporal Ordering**: Verified positive time gap in Protocol B and Protocol D.
5. **Mask Alignment**: 100% 1-to-1 filename stem and dimension match.
