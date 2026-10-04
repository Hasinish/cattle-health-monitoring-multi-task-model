import sys
from pathlib import Path
sys.path.insert(0, ".")

import re
import fitz
from scripts.export_paraphrase_docs import clean_latex, parse_latex_sections

WORKSPACE_ROOT = Path("d:/cattle-health-monitoring-multi-task-model")
PDF_PATH = WORKSPACE_ROOT / "T25301094_Final Report-15-79 ai detection.pdf"

# 1. Extract all flagged words / text segments from the Turnitin PDF
doc = fitz.open(str(PDF_PATH))
ai_color = (0.7289999723434448, 0.9100000262260437, 0.9409999847412109)
def color_match(c):
    return c and all(abs(c[i] - ai_color[i]) < 0.05 for i in range(3))

flagged_phrases = []
for page_idx in range(2, len(doc)):
    page = doc[page_idx]
    words = page.get_text("words")
    drawings = page.get_drawings()
    ai_rects = [d["rect"] for d in drawings if color_match(d.get("fill")) or color_match(d.get("color"))]
    if not ai_rects:
        continue
    
    current_phrase = []
    for w in words:
        w_rect = fitz.Rect(w[:4])
        is_flagged = any(w_rect.intersects(r) and (w_rect & r).get_area() / w_rect.get_area() > 0.3 for r in ai_rects)
        if is_flagged:
            current_phrase.append(w[4])
        else:
            if current_phrase:
                if len(current_phrase) >= 5:
                    flagged_phrases.append(" ".join(current_phrase))
                current_phrase = []
    if current_phrase and len(current_phrase) >= 5:
        flagged_phrases.append(" ".join(current_phrase))

print(f"Extracted {len(flagged_phrases)} contiguous AI flagged phrases from Turnitin PDF.")

# Normalize text for fuzzy matching
def norm(t):
    return re.sub(r'[^a-z0-9]', '', t.lower())

flagged_norm = [norm(p) for p in flagged_phrases]

# 2. Check each chapter
chapters_info = [
    ("Chapter 1: Introduction", WORKSPACE_ROOT / "cattle_thesis_p3_latex" / "chapters" / "chapter_1.tex"),
    ("Chapter 2: Literature Review", WORKSPACE_ROOT / "cattle_thesis_p3_latex" / "chapters" / "chapter_2.tex"),
    ("Chapter 3: Requirements & Constraints", WORKSPACE_ROOT / "cattle_thesis_p3_latex" / "chapters" / "chapter_3.tex"),
    ("Chapter 4: Proposed Methodology", WORKSPACE_ROOT / "cattle_thesis_p3_latex" / "chapters" / "chapter_5.tex"),
    ("Chapter 5: Result Analysis", WORKSPACE_ROOT / "cattle_thesis_p3_latex" / "chapters" / "chapter_6.tex"),
    ("Chapter 6: Conclusion", WORKSPACE_ROOT / "cattle_thesis_p3_latex" / "chapters" / "chapter_9.tex"),
]

total_flagged_paras = 0
for ch_name, ch_path in chapters_info:
    content = ch_path.read_text(encoding="utf-8")
    raw_sections = parse_latex_sections(content)
    ch_flagged = 0
    print(f"\n==========================================")
    print(f"{ch_name}")
    print(f"==========================================")
    
    for sec_idx, (lvl, sec_title, sec_body) in enumerate(raw_sections):
        # split into paragraphs
        paras = [clean_latex(p.strip()) for p in sec_body.split("\n\n") if clean_latex(p.strip())]
        for p_idx, p in enumerate(paras):
            # Skip tables, formula-only blocks
            if p.startswith("[Table Reference:") or (p.startswith("[Formula:") and len(p.splitlines()) <= 3):
                continue
            
            p_norm = norm(p)
            if len(p_norm) < 30:
                continue
            
            # Check if any flagged phrase is in this paragraph, or paragraph matches flagged
            # Try 30-char n-grams from flagged phrases
            matched = False
            for fp, fpn in zip(flagged_phrases, flagged_norm):
                # If flagged phrase is at least 30 chars and inside paragraph
                if len(fpn) >= 25 and fpn[:40] in p_norm:
                    matched = True
                    break
                # Or check 6-word window
                fp_words = fp.split()
                if len(fp_words) >= 6:
                    test_sub = norm(" ".join(fp_words[:6]))
                    if test_sub in p_norm:
                        matched = True
                        break
            
            if matched:
                ch_flagged += 1
                total_flagged_paras += 1
                print(f"  [AI-FLAGGED] Sec: '{sec_title}' | Para {p_idx+1}: {p[:75]}...")
    print(f"Total AI-flagged paragraphs in {ch_name}: {ch_flagged}")

print(f"\nGRAND TOTAL AI-FLAGGED PARAGRAPHS ACROSS ALL CHAPTERS: {total_flagged_paras}")
