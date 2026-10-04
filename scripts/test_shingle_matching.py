import sys
from pathlib import Path
sys.path.insert(0, ".")

import re
import fitz
from scripts.export_paraphrase_docs import clean_latex, parse_latex_sections

WORKSPACE_ROOT = Path("d:/cattle-health-monitoring-multi-task-model")
PDF_PATH = WORKSPACE_ROOT / "T25301094_Final Report-15-79 ai detection.pdf"

doc = fitz.open(str(PDF_PATH))
ai_color = (0.7289999723434448, 0.9100000262260437, 0.9409999847412109)
def color_match(c):
    return c and all(abs(c[i] - ai_color[i]) < 0.05 for i in range(3))

# Extract all AI-flagged words and their page numbers
flagged_by_page = {}
all_flagged_words = []

for page_idx in range(2, len(doc)):
    page = doc[page_idx]
    words = page.get_text("words")
    drawings = page.get_drawings()
    ai_rects = [d["rect"] for d in drawings if color_match(d.get("fill")) or color_match(d.get("color"))]
    if not ai_rects:
        continue
    
    page_flagged = []
    for w in words:
        w_rect = fitz.Rect(w[:4])
        # Intersection test
        if any(w_rect.intersects(r) and (w_rect & r).get_area() / w_rect.get_area() > 0.3 for r in ai_rects):
            page_flagged.append(w[4])
            all_flagged_words.append(w[4])
    if page_flagged:
        flagged_by_page[page_idx + 1] = " ".join(page_flagged)

print(f"Total AI-highlighted words found in PDF: {len(all_flagged_words)}")
print(f"Pages with AI highlights: {sorted(flagged_by_page.keys())}")

# Combine all flagged text into a single lowercase search corpus
def clean_norm(s):
    return re.sub(r'[^a-z0-9]', ' ', s.lower())

all_flagged_corpus = " " + clean_norm(" ".join(all_flagged_words)) + " "

def check_para_flagged(para_text):
    words = clean_norm(para_text).split()
    if not words:
        return False
    # Check 5-word shingles from the paragraph
    shingle_size = 5
    if len(words) < shingle_size:
        shingle = " " + " ".join(words) + " "
        return shingle in all_flagged_corpus
    
    match_count = 0
    total_shingles = len(words) - shingle_size + 1
    for i in range(total_shingles):
        shingle = " " + " ".join(words[i:i+shingle_size]) + " "
        if shingle in all_flagged_corpus:
            return True
    return False

chapters = [
    ("Chapter 1: Introduction", WORKSPACE_ROOT / "cattle_thesis_p3_latex" / "chapters" / "chapter_1.tex"),
    ("Chapter 2: Literature Review", WORKSPACE_ROOT / "cattle_thesis_p3_latex" / "chapters" / "chapter_2.tex"),
    ("Chapter 3: Requirements & Constraints", WORKSPACE_ROOT / "cattle_thesis_p3_latex" / "chapters" / "chapter_3.tex"),
    ("Chapter 4: Proposed Methodology", WORKSPACE_ROOT / "cattle_thesis_p3_latex" / "chapters" / "chapter_5.tex"),
    ("Chapter 5: Result Analysis", WORKSPACE_ROOT / "cattle_thesis_p3_latex" / "chapters" / "chapter_6.tex"),
    ("Chapter 6: Conclusion", WORKSPACE_ROOT / "cattle_thesis_p3_latex" / "chapters" / "chapter_9.tex"),
]

total_paras = 0
total_flagged = 0

for ch_title, ch_file in chapters:
    content = ch_file.read_text(encoding="utf-8")
    raw_sections = parse_latex_sections(content)
    ch_p_count = 0
    ch_flagged_count = 0
    print(f"\n=======================================================")
    print(f"{ch_title} ({ch_file.name})")
    print(f"=======================================================")
    for sec_idx, (lvl, sec_title, sec_body) in enumerate(raw_sections):
        # Paragraphs in this section (normalize CRLF and split on blank lines)
        normalized_body = sec_body.replace('\r\n', '\n')
        raw_paras = [clean_latex(p.strip()) for p in re.split(r'\n\s*\n', normalized_body) if clean_latex(p.strip())]
        for p_idx, p in enumerate(raw_paras):
            if p.startswith("[Table Reference:") or (p.startswith("[Formula:") and len(p.splitlines()) <= 3):
                continue
            if len(p.split()) < 5:
                continue
            ch_p_count += 1
            total_paras += 1
            is_flagged = check_para_flagged(p)
            if is_flagged:
                ch_flagged_count += 1
                total_flagged += 1
                print(f"  • [FLAGGED] ({sec_title}) Para {p_idx+1}: {p[:75]}...")
    print(f"Summary for {ch_title}: {ch_flagged_count} flagged / {ch_p_count} total paragraphs")

print(f"\nOVERALL: {total_flagged} flagged paragraphs out of {total_paras} total paragraphs in thesis.")
