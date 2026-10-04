import sys
from pathlib import Path
sys.path.insert(0, ".")

import re
import fitz
from scripts.export_paraphrase_docs import clean_latex

WORKSPACE_ROOT = Path("d:/cattle-health-monitoring-multi-task-model")
PDF_PATH = WORKSPACE_ROOT / "T25301094_Final Report-15-79 ai detection.pdf"

# 1. Extract all AI-highlighted words from Turnitin PDF
doc = fitz.open(str(PDF_PATH))
ai_color = (0.7289999723434448, 0.9100000262260437, 0.9409999847412109)
def color_match(c):
    return c and all(abs(c[i] - ai_color[i]) < 0.05 for i in range(3))

all_flagged_words = []
for page_idx in range(2, len(doc)):
    page = doc[page_idx]
    words = page.get_text("words")
    drawings = page.get_drawings()
    ai_rects = [d["rect"] for d in drawings if color_match(d.get("fill")) or color_match(d.get("color"))]
    if not ai_rects:
        continue
    for w in words:
        w_rect = fitz.Rect(w[:4])
        if any(w_rect.intersects(r) and (w_rect & r).get_area() / w_rect.get_area() > 0.3 for r in ai_rects):
            all_flagged_words.append(w[4])

def clean_norm(s):
    return re.sub(r'[^a-z0-9]', ' ', s.lower())

all_flagged_corpus = " " + clean_norm(" ".join(all_flagged_words)) + " "

def is_para_flagged(para_text):
    words = clean_norm(para_text).split()
    if not words:
        return False
    shingle_size = 5
    if len(words) < shingle_size:
        return (" " + " ".join(words) + " ") in all_flagged_corpus
    
    total_shingles = len(words) - shingle_size + 1
    for i in range(total_shingles):
        shingle = " " + " ".join(words[i:i+shingle_size]) + " "
        if shingle in all_flagged_corpus:
            return True
    return False

def parse_chapter_sections_and_paras(latex_content: str):
    """
    Parses LaTeX content into a list of:
    {
        "title": section_title,
        "paras": [clean_p1, clean_p2, ...]
    }
    Splits on \section, \subsection, \subsubsection, and splits paragraphs BEFORE clean_latex.
    """
    pattern = re.compile(r'(\\section\*?\{([^}]+)\}|\\subsection\*?\{([^}]+)\}|\\subsubsection\*?\{([^}]+)\})')
    matches = list(pattern.finditer(latex_content))
    
    sections = []
    if not matches:
        raw_paras = [p.strip() for p in re.split(r'\n\s*\n', latex_content.replace('\r\n', '\n')) if p.strip()]
        cleaned = [clean_latex(p) for p in raw_paras if clean_latex(p)]
        return [{"title": "Main Content", "paras": cleaned}]
    
    # Check preamble
    if matches[0].start() > 0:
        preamble = latex_content[:matches[0].start()].strip()
        if preamble:
            raw_paras = [p.strip() for p in re.split(r'\n\s*\n', preamble.replace('\r\n', '\n')) if p.strip()]
            cleaned = [clean_latex(p) for p in raw_paras if clean_latex(p)]
            if cleaned:
                sections.append({"title": "Overview", "paras": cleaned})
    
    for i, m in enumerate(matches):
        full_m = m.group(0)
        title = m.group(2) or m.group(3) or m.group(4)
        start_idx = m.end()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(latex_content)
        raw_body = latex_content[start_idx:end_idx].strip()
        
        # Split into paragraphs FIRST
        raw_paras = [p.strip() for p in re.split(r'\n\s*\n', raw_body.replace('\r\n', '\n')) if p.strip()]
        cleaned_paras = []
        for p in raw_paras:
            # Skip pure LaTeX figure/table environments
            if p.startswith(r'\begin{figure') or p.startswith(r'\begin{table'):
                continue
            cp = clean_latex(p)
            # Skip tables, formula-only blocks, or tiny fragments
            if not cp or cp.startswith("[Table Reference:") or (cp.startswith("[Formula:") and len(cp.splitlines()) <= 3):
                continue
            if len(cp.split()) < 4:
                continue
            cleaned_paras.append(cp)
        
        sections.append({"title": title, "paras": cleaned_paras})
    
    return sections

chapters_spec = [
    ("Chapter 1: Introduction", 0, WORKSPACE_ROOT / "cattle_thesis_p3_latex" / "chapters" / "chapter_1.tex"),
    ("Chapter 2: Literature Review", 1111150292, WORKSPACE_ROOT / "cattle_thesis_p3_latex" / "chapters" / "chapter_2.tex"),
    ("Chapter 3: Requirements & Constraints", 521635664, WORKSPACE_ROOT / "cattle_thesis_p3_latex" / "chapters" / "chapter_3.tex"),
    ("Chapter 4: Proposed Methodology", 1997649724, WORKSPACE_ROOT / "cattle_thesis_p3_latex" / "chapters" / "chapter_5.tex"),
    ("Chapter 5: Result Analysis", 1050456210, WORKSPACE_ROOT / "cattle_thesis_p3_latex" / "chapters" / "chapter_6.tex"),
    ("Chapter 6: Conclusion", 612786333, WORKSPACE_ROOT / "cattle_thesis_p3_latex" / "chapters" / "chapter_9.tex"),
]

total_paras_all = 0
total_flagged_all = 0

chapter_data = {}

for ch_title, sheet_id, ch_path in chapters_spec:
    content = ch_path.read_text(encoding="utf-8")
    secs = parse_chapter_sections_and_paras(content)
    
    flagged_secs = []
    ch_total_p = 0
    ch_flagged_p = 0
    
    for s in secs:
        flagged_paras_in_sec = []
        for p in s["paras"]:
            ch_total_p += 1
            total_paras_all += 1
            if is_para_flagged(p):
                ch_flagged_p += 1
                total_flagged_all += 1
                flagged_paras_in_sec.append(p)
        if flagged_paras_in_sec:
            flagged_secs.append({"title": s["title"], "paras": flagged_paras_in_sec})
    
    chapter_data[ch_title] = {
        "sheet_id": sheet_id,
        "sections": flagged_secs,
        "total_paras": ch_total_p,
        "flagged_paras": ch_flagged_p
    }
    
    print(f"\n[{ch_title}]")
    print(f"  Total Paragraphs in Chapter: {ch_total_p}")
    print(f"  AI-Flagged Paragraphs: {ch_flagged_p} ({len(flagged_secs)} sections with AI flags)")
    for fs in flagged_secs:
        print(f"    - Section '{fs['title']}': {len(fs['paras'])} flagged paragraphs")

print(f"\n=======================================================")
print(f"GRAND TOTAL: {total_flagged_all} AI-flagged paragraphs across {total_paras_all} total thesis paragraphs.")
print(f"=======================================================")
