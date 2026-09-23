# Build and review report

Reviewed source snapshot: 9baa0c68355d59a039a4c3ce17d71a6c1e16d4e2.
Writing branch: thesis-writing only. Date: 2026-09-23.

PASS: sh build.sh completed using pdfLaTeX, Biber, MakeIndex and repeated pdfLaTeX passes. Output: 48 A4 pages, including front matter, bibliography and evidence appendices.

The validator passed all input paths, preserved chapter mapping 1/2/3/5/6/9, 14 cited bibliography keys, 21 project-evidence IDs and recorded confusion-matrix arithmetic. Twenty-two visible TODO calls remain.

No fatal LaTeX errors, undefined citations/references or overfull boxes occur in the final log. Long List-of-Tables entries were fixed with concise optional captions; full captions retain evidence and qualifications. Inherited standard-report warnings for unused Times/print/index options and the requested 16pt title-size substitution remain documented. Underfull-box warnings are nonfatal.

All 48 pages were rendered at 100 dpi and inspected through four 12-page review sheets. Title/front matter, TOC, nomenclature, mathematics, 14 tables, result plot, bibliography and appendices show no visible clipping or overlaps. Short placeholder pages for unfinished front matter/Conclusion are intentional.

The temporary runtime reset before publication. The sources were reconstructed, updated for the new Re-ID implementation/smoke-test commit, and rebuilt and re-rendered. This report describes that current build, not only the earlier successful build.

This is not a submission-ready thesis. No raw datasets were audited and no model was rerun during writing. Full Re-ID retrieval, enhanced/temporal models, MTL, external evaluation and repeated-run uncertainty remain pending as documented. Behavior fallback use, missing provenance, pose acceptance, external mapping and permissions require resolution.

All deliverable source changes are confined to cattle_thesis_p3_latex/. The source template, P1/P2, samples, experiment scripts, datasets, roadmap and state are untouched.
