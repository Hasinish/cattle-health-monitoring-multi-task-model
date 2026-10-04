import sys
from pathlib import Path
sys.path.insert(0, ".")

from scripts.audit_ai_paragraphs_to_upload import chapter_data

for ch, data in chapter_data.items():
    for sec in data['sections']:
        for p_idx, p in enumerate(sec['paras']):
            lines = p.splitlines()
            formula_lines = [l for l in lines if "[Formula:" in l or "\\begin{equation}" in l or "g_shared" in l or "theta_" in l or "\\mathbf" in l or "\\nabla" in l or "$" in l]
            if formula_lines or "[Formula:" in p:
                print(f"[{ch}] ({sec['title']}) Para {p_idx+1}:")
                for fl in formula_lines:
                    print(f"   -> {fl[:100]}")
                print(f"   Full preview: {p[:150]}...\n")
