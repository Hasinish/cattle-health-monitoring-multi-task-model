# Build and review report

Reviewed source snapshot: `a695a92e7c3b52b165053e182f6d3e3331b0866f`.
Review date: 2026-09-24.

## Automated checks

- `python check_draft.py`: PASS for all input paths, preserved chapter mapping (1/2/3/5/6/9), 51 cited bibliography keys, 34 project-evidence IDs, and recorded confusion-matrix arithmetic.
- Nine visible `TODO` calls remain. They are confined to final administrative/ethics details, the post-MTL Abstract, and the post-MTL Conclusion.
- `sh build.sh` could not be invoked because this Windows environment has no `sh` executable. The same defined sequence was executed directly with MiKTeX: pdfLaTeX, Biber, MakeIndex, and repeated pdfLaTeX passes.
- Final compilation: PASS. `main.pdf` contains 69 A4 pages. The final log has no fatal error, undefined citation/reference, or overfull box.
- The inherited standard-report warning for unused `Times`, `print`, and `index` options remains. Title-size substitution and a few nonfatal underfull-box warnings also remain.

## Visual inspection

Representative front-matter, chapter-opening, methodology, result-table, conclusion, bibliography, and evidence-appendix pages were rendered from the final PDF and inspected. No clipping, overlap, missing table content, or broken figure was found. The compact tables remain legible at page width.

## Evidence boundary

No dataset, model, or experiment was rerun. Runs 1--6 are reported from repository artifacts, while Runs 7--8 and final integrated claims remain explicitly pending. The later Run 2-versus-Run 5 comparison artifact embeds a historical Run 2 support breakdown that disagrees with the original Run 2 artifact. The manuscript therefore uses the original artifact for the canonical 809-sample result and uses the later artifact only for its explicitly matched 780-sample comparison.

All edits made for this writing task are confined to `cattle_thesis_p3_latex/`. No commit or branch was created.
