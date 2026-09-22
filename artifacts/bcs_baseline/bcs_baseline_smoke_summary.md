# Phase 3 ScienceDB RGB BCS Baseline — Smoke Test Report

- **Date**: 2026-09-22 19:29:10 UTC
- **Git Commit**: `848305477e42510d00570e8e8d1a9962a43e203a` (main)
- **Encoder**: ImageNet-Pretrained ResNet-18
- **Head Formulation**: `coral`
- **Device**: `NVIDIA GeForce GTX 1050 Ti`
- **Split Protocol**: `burst-group-disjoint / sequence-safe (5,653 connected burst groups)`

## Test Set Metrics (Real BCS Scale 3.25 to 4.25)

| Metric | Value |
| :--- | :---: |
| **Real BCS MAE (Primary)** | **0.3700 BCS units** |
| Class-Index MAE | 1.4800 steps |
| Exact Accuracy (Acc@0) | 20.00% |
| Within-0.25 Tolerance (Acc@1) | 53.60% |
| Balanced Accuracy | 20.00% |
| Macro-F1 Score | 0.1921 |

## Per-Class Performance

| Real BCS Class | Support | Accuracy (%) | Precision | Recall | F1 Score | Real MAE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 3.25 | 25 | 0.0% | 0.000 | 0.000 | 0.000 | 0.6500 |
| 3.5 | 25 | 20.0% | 0.208 | 0.200 | 0.204 | 0.3200 |
| 3.75 | 25 | 24.0% | 0.130 | 0.240 | 0.169 | 0.2900 |
| 4.0 | 25 | 28.0% | 0.259 | 0.280 | 0.269 | 0.2500 |
| 4.25 | 25 | 28.0% | 0.368 | 0.280 | 0.318 | 0.3400 |

*Smoke test completed successfully in 18.0 seconds.*
