with open("cattle_thesis_p3_latex/chapters/chapter_6.tex", "r", encoding="utf-8") as f:
    text = f.read()

import re

# Targets 2, 3
print("=== Target 2 (Baseline RGB Run 1) ===")
m = re.search(r"Baseline RGB.*?(?=\n\n|\Z)", text, re.DOTALL)
if m:
    print(m.group(0)[:200])

print("\n=== Target 3 (matched evaluation Table 5.2) ===")
m = re.search(r".*?temporal foreground modeling.*?(?=\n\n|\Z)", text, re.DOTALL)
if m:
    print(m.group(0)[:200])

print("\n=== Target 9, 10, 11 (E1, E3, E4 bullet points or sections) ===")
m = re.search(r".*?Monolithic Hard Parameter Sharing.*?(?=\n\n|\Z)", text, re.DOTALL)
if m:
    print(m.group(0)[:300])

print("\n=== Target 12 (SideView Protocol A evaluates 69) ===")
m = re.search(r".*?69 biologically distinct cows.*?(?=\n\n|\Z)", text, re.DOTALL)
if m:
    print(m.group(0)[:300])
