import sys
import re
sys.path.insert(0, ".")
from scripts.sheet_tool import get_service, SPREADSHEET_ID

def audit_ch123():
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
            tex_content = f.read()

        total_content_rows = 0
        populated_rows = []
        unpopulated_rows = []
        in_latex_count = 0
        not_in_latex = []

        print(f"\n==================================================")
        print(f"SHEET: {title}")
        print(f"FILE : {tex_file}")
        print(f"==================================================")

        for idx, r in enumerate(vals, 1):
            if len(r) < 2 or not r[1]:
                continue
            b_text = r[1].strip()
            # Skip header rows
            if b_text.startswith("Original (Do Paraphrase"):
                continue
            # Skip section header rows like "3.1 Final Specifications..."
            if re.match(r'^\d\.\d\s+', b_text):
                continue

            total_content_rows += 1
            c_text = r[2].strip() if len(r) > 2 and r[2] else ""
            d_text = r[3].strip() if len(r) > 3 and r[3] else ""

            if c_text and not c_text.startswith("LEAVE BLANK"):
                populated_rows.append((idx, b_text, c_text, d_text))
                # Check if in LaTeX
                # Extract first 4-5 words
                words = [w for w in re.findall(r'\b[A-Za-z0-9\-\']+\b', c_text) if len(w) > 3][:6]
                search_str = " ".join(words)
                if search_str and search_str.lower() in tex_content.lower():
                    in_latex_count += 1
                else:
                    not_in_latex.append((idx, search_str, c_text[:70]))
            else:
                unpopulated_rows.append((idx, b_text[:60]))

        print(f"Total Content Paragraphs: {total_content_rows}")
        print(f"Populated Paraphrases (Col C): {len(populated_rows)}")
        print(f"Empty/Unpopulated in Col C: {len(unpopulated_rows)}")
        print(f"Already integrated into LaTeX: {in_latex_count}")
        print(f"NOT integrated into LaTeX: {len(not_in_latex)}")
        
        if not_in_latex:
            print("\n  Rows populated in Sheet but NOT yet in LaTeX:")
            for r_idx, s_str, snippet in not_in_latex:
                print(f"    Row {r_idx}: '{s_str}' -> {snippet}...")

if __name__ == "__main__":
    audit_ch123()
