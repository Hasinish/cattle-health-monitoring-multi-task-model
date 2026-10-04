import fitz

pdf_path = r"D:\cattle-health-monitoring-multi-task-model\T25301094_Final Report-15-79 ai detection.pdf"
doc = fitz.open(pdf_path)

ai_color = (0.7289999723434448, 0.9100000262260437, 0.9409999847412109)
def color_match(c):
    return c and all(abs(c[i] - ai_color[i]) < 0.05 for i in range(3))

for pno in range(2, 6):
    page = doc[pno]
    drawings = page.get_drawings()
    ai_rects = [d['rect'] for d in drawings if color_match(d.get('fill')) or color_match(d.get('color'))]
    print(f"\n--- Page {pno+1} (PDF) / Thesis Report Page {pno+14} (AI Rects: {len(ai_rects)}) ---")
    blocks = page.get_text('blocks')
    for b_idx, b in enumerate(blocks):
        b_rect = fitz.Rect(b[:4])
        has_ai = any(b_rect.intersects(ar) for ar in ai_rects)
        # Check if text is not header/footer
        text = b[4].strip()
        if not text:
            continue
        first_line = text.splitlines()[0]
        # check if it's header/footer like "Chapter 1" or page number
        if len(text) < 10 and text.isdigit():
            continue
        if "Department of Computer Science" in text or "CSE 400" in text:
            continue
        print(f"Block {b_idx} [AI={has_ai}]: {first_line[:70]}... (total words: {len(text.split())})")
