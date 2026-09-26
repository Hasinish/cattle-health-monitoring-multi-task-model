import sys
import re
sys.path.insert(0, ".")
from scripts.sheet_tool import get_service, SPREADSHEET_ID

def check_exact():
    service = get_service()
    chapters = [
        ("Chapter 1: Introduction", "cattle_thesis_p3_latex/chapters/chapter_1.tex"),
        ("Chapter 2: Literature Review", "cattle_thesis_p3_latex/chapters/chapter_2.tex"),
        ("Chapter 3: Requirements & Constraints", "cattle_thesis_p3_latex/chapters/chapter_3.tex")
    ]

    for title, tex_file in chapters:
        res = service.values().get(spreadsheetId=SPREADSHEET_ID, range=f"'{title}'!A1:D150").execute()
        vals = res.get('values', [])
        
        with open(tex_file, "r", encoding="utf-8") as f:
            tex_content = f.read().lower()

        # normalize whitespace
        tex_norm = " ".join(tex_content.split())

        print(f"\n==================================================")
        print(f"SHEET: {title}")
        print(f"==================================================")

        total_content_rows = 0
        populated_count = 0
        in_latex_count = 0
        not_in_latex = []

        for idx, r in enumerate(vals, 1):
            if len(r) < 2 or not r[1]:
                continue
            b_text = r[1].strip()
            if b_text.startswith("Original (Do Paraphrase") or re.match(r'^\d\.\d\s+', b_text):
                continue

            total_content_rows += 1
            c_text = r[2].strip() if len(r) > 2 and r[2] else ""

            if c_text and not c_text.startswith("LEAVE BLANK"):
                populated_count += 1
                # Take first 35 chars of c_text
                sample = c_text[:35].lower()
                # Clean sample
                sample_clean = " ".join(sample.split())
                if sample_clean in tex_norm:
                    in_latex_count += 1
                else:
                    not_in_latex.append((idx, c_text[:60]))

        print(f"Total Content Paragraphs: {total_content_rows}")
        print(f"Populated Paraphrases (Col C): {populated_count} / {total_content_rows}")
        print(f"Integrated into LaTeX: {in_latex_count} / {populated_count}")
        if not_in_latex:
            print(f"Not found in LaTeX ({len(not_in_latex)}):")
            for r_num, snip in not_in_latex:
                print(f"  Row {r_num}: {snip}...")

if __name__ == "__main__":
    check_exact()
