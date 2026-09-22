# Phase 3 ScienceDB RGB BCS Baseline — Smoke Test Report

- **Date**: 2026-09-22 19:38:48 UTC
- **Git Commit**: `3424fdd0778e739bcbb06dc2bdb079c50b6adb5d` (main)
- **Encoder**: ImageNet-Pretrained ResNet-18
- **Head Formulation**: `Ordinal BCE (Frank & Hall 2001 Independent Cumulative Logits)`
- **Device**: `NVIDIA GeForce GTX 1050 Ti`
- **Split Protocol**: `burst-group-disjoint / sequence-safe (5,653 connected burst groups)`
- **Canonical Test Set Status**: `PRESERVED UNTOUCHED (Zero Evaluation in Smoke Mode)`

## Smoke Validation Subset Performance (Test Set Preserved Untouched)

| Metric | Value |
| :--- | :---: |
| **Real BCS MAE (Primary)** | **0.4200 BCS units** |
| Class-Index MAE | 1.6800 steps |
| Exact Accuracy (Acc@0) | 18.40% |
| Within-0.25 Tolerance (Acc@1) | 50.40% |
| Balanced Accuracy | 18.40% |
| Macro-F1 Score | 0.1827 |

## Per-Class Performance

| Real BCS Class | Support | Accuracy (%) | Precision | Recall | F1 Score | Real MAE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 3.25 | 25 | 0.0% | 0.000 | 0.000 | 0.000 | 0.6700 |
| 3.5 | 25 | 24.0% | 0.429 | 0.240 | 0.308 | 0.2300 |
| 3.75 | 25 | 24.0% | 0.171 | 0.240 | 0.200 | 0.3000 |
| 4.0 | 25 | 32.0% | 0.250 | 0.320 | 0.281 | 0.2600 |
| 4.25 | 25 | 12.0% | 0.130 | 0.120 | 0.125 | 0.6400 |

*Smoke test completed successfully in 32.3 seconds. Canonical test split remained 100% untouched.*
