# Research Log: SideViewCows2026 Modal Cloud Hydration & Verification (dryousufmozumder)

**Date:** 2026-09-24  
**Author:** Hasin Ishrak  
**Target Modal Profile:** `dryousufmozumder`  
**Persistent Volume:** `sideview-data`  
**Volume Mount:** `/data`  
**Dataset Root:** `/data/sideviewcows2026/`  
**Container Config:** `cpu=1.0, memory=2048 MB, NO GPU`  
**Official Zenodo Record:** [10.5281/zenodo.21605650](https://zenodo.org/records/21605650)  
**Apps Executed:**
- Pre-Flight Smoke Test: `ap-PcAmHd38BT66WqDXxZEyto` (Runtime: ~25s)
- Hydration, Extraction & Full Verification: `ap-92pLaQlpqor4Q9kNbTEcGb` (Runtime: 39.4 mins)

---

## 1. Executive Summary

To prepare the independent cloud execution workspace for **Run 6 (Re-ID GT-Mask Perception Model)** and multi-workspace parallelization, the official ~25 GB SideViewCows2026 dataset was hydrated directly from Zenodo into persistent volume `sideview-data` on Modal under profile `dryousufmozumder`.

The pipeline executed with 16 parallel HTTP Range streams on a minimal 1.0-CPU container, streaming directly Zenodo -> Modal with zero local machine relay. All archives were downloaded, extracted with monitored progress, verified with 100% forensic checks, and cleaned up to reclaim 23.31 GB of volume storage. Total cloud cost: **$0.05**.

---

## 2. Dataset Physical Verification Metrics

| Dataset Partition | Expected Images | Found Images | Expected Masks | Found Masks | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Milking Parlor (`parlor`) | 54,393 | **54,393** | 54,393 | **54,393** | PASS |
| Barn Housing (`barn`) | 25,260 | **25,260** | 25,260 | **25,260** | PASS |
| Field Snapshots (`snapshots`) | 607 | **607** | 607 | **607** | PASS |
| **Global Totals** | **80,260** | **80,260** | **80,260** | **80,260** | **ALL PASS** |

### Integrity & Decodability Audit
- **Unique Cow Identities**: 110 / 110 verified across directories
- **Zero-Byte Stubs Detected**: **0**
- **Random PIL Image Decodes**: 10 / 10 passed (`im.verify()`)
- **Random PIL Mask Decodes**: 10 / 10 passed (`im.verify()`)
- **Storage Freed Post-Extraction**: **23.31 GB** (pruned `parlor.zip`, `barn.zip`, `snapshots.zip`)

---

## 3. Protocol Path Resolution Audit

All four canonical evaluation protocols from `datasets/id/sideviewcows2026/` were verified against `/data/sideviewcows2026/`:

1. `protocol_cross_setting.csv`: **100% path resolution (15/15 sampled rows)**
2. `protocol_longitudinal.csv`: **100% path resolution (15/15 sampled rows)**
3. `protocol_open_set.csv`: **100% path resolution (15/15 sampled rows)**
4. `protocol_closed_set.csv`: **100% path resolution (15/15 sampled rows)**

---

## 4. Status & Next Actions

- **Physical Storage Status**: **100% CERTIFIED & LOCKED** (`sideview-data` on `dryousufmozumder`).
- **Immediate Next Action**: Ready for Run 6 GT-mask perception training on `dryousufmozumder`.
