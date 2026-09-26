import sys
import os
sys.path.insert(0, os.path.abspath("."))
import re

from scripts.check_ch6_pairs import orig_blocks, para_blocks
from scripts.map_ch9_items import matched_lines

target_path = "cattle_thesis_p3_latex/chapters/chapter_9.tex"

with open(target_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = list(lines)

for i in range(29):
    line_no, old_line = matched_lines[i]
    para = para_blocks[i].strip()
    
    is_item = old_line.strip().startswith(r"\item")
    indent = "    " if is_item else ""
    
    # Process text for LaTeX:
    text = re.sub(r'(?<!\\)%', r'\%', para)
    text = text.replace("±", r"$\pm$")
    text = text.replace("–", "--").replace("—", "---")
    
    if is_item:
        m_bold_colon = re.match(r'^\s*\\item\s+\\textbf\{([^}]+)\}:\s*(.*)', old_line)
        m_for_bold = re.match(r'^\s*\\item\s+For\s+\\textbf\{([^}]+)\},\s*(.*)', old_line)
        m_the_bold = re.match(r'^\s*\\item\s+The\s+\\textbf\{([^}]+)\},\s*(.*)', old_line)
        m_para_colon = re.match(r'^([^:]+):\s*(.*)', text)
        
        if m_bold_colon and m_para_colon:
            title = m_para_colon.group(1).strip()
            body = m_para_colon.group(2).strip()
            if "GradNorm" in body and "cite" not in body:
                body = body.replace("GradNorm", r"GradNorm~\cite{chen2018gradnorm}")
            final_line = f"{indent}\\item \\textbf{{{title}}}: {body}\n"
        elif m_for_bold:
            final_line = f"{indent}\\item {text}\n"
        elif m_the_bold:
            if text.startswith("The Monolithic Hard-Shared Multi-Task Control (E1)"):
                rest = text[len("The Monolithic Hard-Shared Multi-Task Control (E1)"):].strip()
                final_line = f"{indent}\\item The \\textbf{{Monolithic Hard-Shared Multi-Task Control (E1)}} {rest}\n"
            else:
                final_line = f"{indent}\\item {text}\n"
        else:
            final_line = f"{indent}\\item {text}\n"
    else:
        final_line = f"{text}\n"
        
    new_lines[line_no - 1] = final_line

with open(target_path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print(f"Successfully integrated all 29 paraphrases into {target_path}!")
