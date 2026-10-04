import fitz
import os

pdf_path = r"D:\cattle-health-monitoring-multi-task-model\T25301094_Final Report-15-79 ai detection.pdf"
doc = fitz.open(pdf_path)

ai_color = (0.7289999723434448, 0.9100000262260437, 0.9409999847412109)

def color_match(c):
    if not c:
        return False
    return all(abs(c[i] - ai_color[i]) < 0.05 for i in range(3))

passages = []

for page_idx in range(2, len(doc)):
    page = doc[page_idx]
    words = page.get_text("words")
    drawings = page.get_drawings()
    ai_rects = [d["rect"] for d in drawings if color_match(d.get("fill")) or color_match(d.get("color"))]

    if not ai_rects:
        continue

    word_flags = []
    for w in words:
        w_rect = fitz.Rect(w[:4])
        is_flagged = any(w_rect.intersects(r) and (w_rect & r).get_area() / w_rect.get_area() > 0.4 for r in ai_rects)
        word_flags.append((w, is_flagged))

    current_seg = []
    unflagged_gap = 0
    for w, flagged in word_flags:
        if flagged:
            current_seg.append(w[4])
            unflagged_gap = 0
        else:
            if current_seg:
                unflagged_gap += 1
                if unflagged_gap > 3:
                    if len(current_seg) >= 8:
                        passages.append({
                            "pdf_page": page_idx + 1,
                            "report_page": page_idx + 14,
                            "text": " ".join(current_seg),
                            "words": len(current_seg)
                        })
                    current_seg = []
                    unflagged_gap = 0
    if current_seg and len(current_seg) >= 8:
        passages.append({
            "pdf_page": page_idx + 1,
            "report_page": page_idx + 14,
            "text": " ".join(current_seg),
            "words": len(current_seg)
        })

def get_chapter(p):
    if 3 <= p <= 10:
        return "Chapter 1: Introduction"
    if 11 <= p <= 25:
        return "Chapter 2: Literature Review"
    if 26 <= p <= 32:
        return "Chapter 3: Requirements & Constraints"
    if 33 <= p <= 43:
        return "Chapter 4: Proposed Methodology"
    if 44 <= p <= 50:
        return "Chapter 5: Result Analysis"
    if 51 <= p <= 67:
        return "Chapter 6: Conclusion & Future Work"
    return "Other"

out_path = r"C:\Users\hasin\.gemini\antigravity-ide\brain\fd2f88b4-425a-4b7e-a033-6a257316d8c8\turnitin_ai_detected_paragraphs.md"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("# Turnitin AI Detection Forensic Audit Report\n\n")
    f.write("**Document Audited:** `T25301094_Final Report-15-79 ai detection.pdf`\n\n")
    f.write("**Submission ID:** `trn:oid:::0:736677945`\n\n")
    f.write("**Overall Turnitin Score:** **33% AI Detected** (85 highlight segments, 6,914 total flagged words)\n\n")
    f.write("---\n\n")
    f.write("## Chapter-Level AI Breakdown\n\n")
    f.write("| Chapter | Flagged Passages | Flagged Words | Primary Flagged Topics |\n")
    f.write("| :--- | :---: | :---: | :--- |\n")
    f.write("| **Chapter 1: Introduction** | 13 | 1,499 | Background, problem statement, research questions |\n")
    f.write("| **Chapter 2: Literature Review** | 15 | 2,764 | Animal pose (Ferguson), shortcut learning, MTL |\n")
    f.write("| **Chapter 3: Requirements & Constraints** | 8 | 502 | Societal/environmental impact, cost table snippets |\n")
    f.write("| **Chapter 4: Proposed Methodology** | 3 | 310 | Adapter zero-init, benchmark framing |\n")
    f.write("| **Chapter 5: Result Analysis** | 8 | 531 | BCS/Behavior discussion, point estimates |\n")
    f.write("| **Chapter 6: Conclusion & Future Work** | 14 | 1,308 | Negative transfer discussion, limitations, future directions |\n")
    f.write("| **Total** | **61** | **6,914** | **33% of Submitted Text** |\n\n")
    f.write("---\n\n")
    f.write("## Line-by-Line Flagged Passages\n\n")

    current_ch = ""
    for i, p in enumerate(passages):
        ch = get_chapter(p["pdf_page"])
        if ch != current_ch:
            current_ch = ch
            f.write(f"### {current_ch}\n\n")
        f.write(f"#### [{i+1:02d}] Turnitin PDF Page {p['pdf_page']} (Thesis Page {p['report_page']}) — {p['words']} words\n\n")
        f.write(f"> {p['text']}\n\n")

print(f"Report successfully generated at {out_path}!")
