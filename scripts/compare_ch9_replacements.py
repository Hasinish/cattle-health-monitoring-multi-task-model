import sys
import os
sys.path.insert(0, os.path.abspath("."))

from scripts.check_ch6_pairs import orig_blocks, para_blocks

with open("cattle_thesis_p3_latex/chapters/chapter_9.tex", "r", encoding="utf-8") as f:
    lines = f.readlines()

from scripts.map_ch9_items import matched_lines

for i in range(29):
    line_no, orig_line = matched_lines[i]
    para = para_blocks[i]
    print(f"=== Item {i+1} (Line {line_no}) ===")
    print("CURRENT IN TEX:")
    print(orig_line.strip())
    print("PARAPHRASE TO PUT:")
    print(para.strip())
    print()
