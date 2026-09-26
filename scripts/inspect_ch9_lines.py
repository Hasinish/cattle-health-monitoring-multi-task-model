with open("cattle_thesis_p3_latex/chapters/chapter_9.tex", "r", encoding="utf-8") as f:
    lines = f.readlines()

for idx, line in enumerate(lines):
    line_clean = line.strip()
    if not line_clean or line_clean.startswith(r"\section") or line_clean.startswith(r"\subsection") or line_clean.startswith(r"\begin") or line_clean.startswith(r"\end") or line_clean.startswith(r"\textit") or line_clean.startswith(r"\clearpage"):
        continue
    print(f"L{idx+1}: {line_clean[:80]}...")
