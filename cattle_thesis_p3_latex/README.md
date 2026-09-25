# Unified Cattle Health Monitoring Thesis Manuscript

This directory contains the BRAC University template-derived thesis manuscript for **Multi-Task Deep Learning Framework for Unified Cattle Health and Behavior Monitoring**.

## Manuscript and Evidence Status

- **Completed Thesis Manuscript**: Incorporates the complete single-task and multi-task experimental evaluation across Body Condition Scoring (ScienceDB), Behavior Recognition (CVB and Kaggle Beef), and Individual Cattle Identification (SideViewCows2026 Protocol A).
- **Integrated Multi-Task Configurations**: The completed multi-task experimental suite integrates the Monolithic Hard-Shared MTL Control (E1), Modular Task-Private Adapters (E3), and the PCGrad Optimization Control (E4), all evaluated on matched held-out test populations.
- **Deferred Configurations**: Partial sharing (E2) and dynamic loss weighting via GradNorm (E5) remain explicitly deferred.
- **Scientific Synthesis**: Final scientific writing, cross-chapter numerical calibration, and evidence integration across Chapters 1--6, the Abstract, and appendices are complete.
- **Administrative Front Matter**: Formal administrative items (author roster verification, committee signatures, institutional approval, final dedication/acknowledgments, and external ethics/permission confirmations) remain external submission obligations.

## Build Instructions

Build with TeX Live (pdfLaTeX, Biber, MakeIndex, biblatex, nomencl, PGFPlots, booktabs, tabularx, xurl) and Python 3:

```sh
sh build.sh
```

Printed chapters 1--6 use source filenames `1`, `2`, `3`, `5`, `6`, and `9`. Preserve that mapping. On Overleaf select pdfLaTeX and `main.tex`; `build.sh` defines the canonical compilation sequence.

`WRITING_STATUS.md`, `EVIDENCE_MAP.md`, and `RUBRIC_CHECKLIST.md` track manuscript synthesis, evidence provenance, and rubric coverage. `REFERENCE_VERIFICATION.md` records the verified 51-source bibliography. `BUILD_REPORT.md` records automated compilation checks and visual proof metrics.

No raw datasets, checkpoints, signatures, or font files are committed. `evidence/` contains recorded metric excerpts and arithmetic checks, not new model runs. `main.pdf` is generated locally and ignored by Git. Thesis-writing changes are confined to this directory; experimental sources remain read-only evidence.
