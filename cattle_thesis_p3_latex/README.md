# Phase 3 thesis manuscript

This directory contains the BRACU template-derived Phase 3 thesis for **Multi-Task Deep Learning Framework for Unified Cattle Health and Behavior Monitoring**. Runs 1–6 are fully incorporated; Runs 7–8 and the final integrated synthesis remain intentionally pending.

Build with TeX Live (pdfLaTeX, Biber, MakeIndex, biblatex, nomencl, PGFPlots, booktabs, tabularx, xurl) and Python 3:

```sh
sh build.sh
```

Printed chapters 1--6 use source filenames 1,2,3,5,6,9. Keep that mapping. On Overleaf select pdfLaTeX and main.tex; the build script defines the reference sequence.

`WRITING_STATUS.md`, `EVIDENCE_MAP.md`, and `RUBRIC_CHECKLIST.md` track writing, source evidence, and rubric coverage. `REFERENCE_VERIFICATION.md` records the verified bibliography subset. `BUILD_REPORT.md` records compilation and visual checks.

No raw datasets, checkpoints, signatures, or font files are included. `evidence/` contains recorded metric excerpts and arithmetic checks, not new model runs. `main.pdf` is generated locally and ignored by Git. Thesis-writing changes are confined to this directory; experimental sources remain read-only evidence.
