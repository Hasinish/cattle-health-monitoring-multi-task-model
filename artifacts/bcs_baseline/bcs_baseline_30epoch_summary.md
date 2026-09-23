# Phase 3 ScienceDB RGB BCS Baseline — Smoke Test Report

- **Date**: 2026-09-23 02:20:58 UTC
- **Git Commit**: `UNKNOWN` (UNKNOWN)
- **Encoder**: ImageNet-Pretrained ResNet-18
- **Head Formulation**: `Ordinal BCE (Frank & Hall 2001 Independent Cumulative Logits)`
- **Device**: `NVIDIA L40S`
- **Split Protocol**: `burst-group-disjoint / sequence-safe (5,653 connected burst groups)`
- **Canonical Test Set Status**: `PRESERVED UNTOUCHED (Zero Evaluation in Smoke Mode)`

## Final Canonical Test Set Metrics

| Metric | Value |
| :--- | :---: |
| **Real BCS MAE (Primary)** | **0.1848 BCS units** |
| Class-Index MAE | 0.7391 steps |
| Exact Accuracy (Acc@0) | 41.63% |
| Within-0.25 Tolerance (Acc@1) | 86.74% |
| Balanced Accuracy | 40.41% |
| Macro-F1 Score | 0.4110 |

## Per-Class Performance

| Real BCS Class | Support | Accuracy (%) | Precision | Recall | F1 Score | Real MAE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 3.25 | 1209 | 37.7% | 0.545 | 0.377 | 0.446 | 0.2333 |
| 3.5 | 1999 | 47.5% | 0.446 | 0.475 | 0.460 | 0.1683 |
| 3.75 | 2232 | 33.2% | 0.368 | 0.332 | 0.350 | 0.1873 |
| 4.0 | 1750 | 54.3% | 0.381 | 0.543 | 0.448 | 0.1406 |
| 4.25 | 850 | 29.3% | 0.439 | 0.293 | 0.351 | 0.2388 |

*Smoke test completed successfully in 2023.2 seconds. Canonical test split remained 100% untouched.*
