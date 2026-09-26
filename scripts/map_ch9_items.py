import sys
import os
sys.path.insert(0, os.path.abspath("."))
import re

from scripts.check_ch6_pairs import orig_blocks, para_blocks

with open("cattle_thesis_p3_latex/chapters/chapter_9.tex", "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total orig blocks: {len(orig_blocks)}")

def clean_for_match(text):
    text = re.sub(r'\\textbf\{([^}]+)\}', r'\1', text)
    text = re.sub(r'\\item\s*', '', text)
    text = re.sub(r'\\cite\{[^}]+\}', '', text)
    text = re.sub(r'\$[^$]+\$', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

matched_lines = {}
for i, orig in enumerate(orig_blocks):
    clean_orig = clean_for_match(orig)
    words = clean_orig.split()[:8]
    probe = " ".join(words)
    
    found = False
    for idx, line in enumerate(lines):
        if idx in [v[0]-1 for v in matched_lines.values()]:
            continue
        clean_line = clean_for_match(line)
        if probe.lower() in clean_line.lower() or " ".join(words[:5]).lower() in clean_line.lower():
            matched_lines[i] = (idx + 1, line)
            found = True
            break
    if not found:
        print(f"NOT FOUND: Item {i+1}: {orig[:50]}")
    else:
        print(f"Item {i+1} -> Line {matched_lines[i][0]}: {lines[matched_lines[i][0]-1][:60]}...")
