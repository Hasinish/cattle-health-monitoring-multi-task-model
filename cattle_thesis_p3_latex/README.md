# Phase 3 thesis working draft

All thesis changes belong on `thesis-writing`. `main` is read-only experimental evidence. No force-push or history rewrite.

This is a template-derived working manuscript, not a finished thesis. Abstract, Conclusion and unfinished experiment results have explicit TODOs.

Build with TeX Live (pdfLaTeX, Biber, MakeIndex, biblatex, nomencl, PGFPlots, booktabs, tabularx, xurl) and Python 3:

```sh
sh build.sh
```

Printed chapters 1--6 use source filenames 1,2,3,5,6,9. Keep that mapping. On Overleaf select pdfLaTeX and main.tex; the build script defines the reference sequence.

WRITING_STATUS.md, EVIDENCE_MAP.md and RUBRIC_CHECKLIST.md track writing, source evidence and coverage. REFERENCE_VERIFICATION.md records the existing-reference subset. BUILD_REPORT.md records compilation and visual checks.

No raw datasets, checkpoints, signatures or font files are included. evidence/ contains recorded matrix excerpts and arithmetic checks, not new model runs. main.pdf is a separately supplied generated artifact and ignored by Git.
