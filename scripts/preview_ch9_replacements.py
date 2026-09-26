import sys
import os
sys.path.insert(0, os.path.abspath("."))
import re

from scripts.check_ch6_pairs import orig_blocks, para_blocks
from scripts.map_ch9_items import matched_lines

with open("cattle_thesis_p3_latex/chapters/chapter_9.tex", "r", encoding="utf-8") as f:
    orig_lines = f.readlines()

new_lines = list(orig_lines)

# Prepare each replacement
replacements = {}

for i in range(29):
    line_no, old_line = matched_lines[i]
    para = para_blocks[i].strip()
    
    # Check if old line was an \item
    is_item = old_line.strip().startswith(r"\item")
    indent = "    " if is_item else ""
    
    # Process text for LaTeX:
    # 1. escape % that are not already escaped
    # replace unescaped % with \%
    text = re.sub(r'(?<!\\)%', r'\%', para)
    # replace ± with $\pm$
    text = text.replace("±", r"$\pm$")
    # replace en-dash / em-dash if needed, or keep standard
    text = text.replace("–", "--").replace("—", "---")
    
    # If the user included a title like "Title: Body" and it was an item with \textbf{Title}:
    # Let's inspect old_line
    if is_item:
        # Check if old line has \item \textbf{...}: or \item For \textbf{...},
        m_bold_colon = re.match(r'^\s*\\item\s+\\textbf\{([^}]+)\}:\s*(.*)', old_line)
        m_for_bold = re.match(r'^\s*\\item\s+For\s+\\textbf\{([^}]+)\},\s*(.*)', old_line)
        m_the_bold = re.match(r'^\s*\\item\s+The\s+\\textbf\{([^}]+)\},\s*(.*)', old_line)
        
        # Check if para has "Title: Body"
        m_para_colon = re.match(r'^([^:]+):\s*(.*)', text)
        
        if m_bold_colon and m_para_colon:
            title = m_para_colon.group(1).strip()
            body = m_para_colon.group(2).strip()
            # If item 25, preserve \cite{chen2018gradnorm}
            if "GradNorm" in body and "cite" not in body:
                body = body.replace("GradNorm", r"GradNorm~\cite{chen2018gradnorm}")
            final_line = f"{indent}\\item \\textbf{{{title}}}: {body}\n"
        elif m_for_bold:
            # check if para starts with "For ..." or similar
            # If user text starts with "The BCS task used...", we can do:
            # \item The BCS task used... or \item For \textbf{Body Condition Scoring}...
            # Let's check user text
            final_line = f"{indent}\\item {text}\n"
        elif m_the_bold:
            # e.g. The Monolithic Hard-Shared Multi-Task Control (E1)
            # check if starts with "The Monolithic..."
            if text.startswith("The Monolithic Hard-Shared Multi-Task Control (E1)"):
                # keep bold on title
                rest = text[len("The Monolithic Hard-Shared Multi-Task Control (E1)"):].strip()
                final_line = f"{indent}\\item The \\textbf{{Monolithic Hard-Shared Multi-Task Control (E1)}} {rest}\n"
            else:
                final_line = f"{indent}\\item {text}\n"
        else:
            final_line = f"{indent}\\item {text}\n"
    else:
        final_line = f"{text}\n"
        
    replacements[line_no] = final_line

print("All 29 replacements prepared. Let's inspect them:")
for i in range(29):
    lno = matched_lines[i][0]
    print(f"[{i+1}] Line {lno}:")
    print("  OLD:", orig_lines[lno-1].strip()[:90])
    print("  NEW:", replacements[lno].strip()[:90])
