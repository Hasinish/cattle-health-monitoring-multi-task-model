# Working-copy template adaptations

The original template is untouched. Preserve standard report class, 12pt, A4 150 mm text width, 25 mm top/bottom margins, plain pagination, Roman front matter, Arabic chapters, front-matter order, IEEE biblatex/Biber with sorting=ynt, and chapter filenames 1,2,3,5,6,9.

Necessary changes in the working copy only: fix malformed graphicspath; add typesetting support for tables, mathematics, paths, plot and TODOs; give nomenclature a correct TOC anchor; avoid duplicate bibliography TOC entries; replace example appendices with evidence appendices; fit the existing five-author roster without copying signatures; leave approval explicitly pending. Preserve unused empty chapter_7.tex.

Standard report ignores template options Times/print/index and substitutes the requested 16pt title font size. Preserve that inherited behavior and document warnings rather than silently changing the university font. Confirm any separately mandated Times font before changing it.

No previous thesis, sample paper or original template is modified.

Long figure/table captions have short optional list entries so source IDs and extended qualification text do not overflow the List of Tables/Figures. Full captions remain unchanged.
