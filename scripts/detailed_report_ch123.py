import sys
import re
sys.path.insert(0, ".")
from scripts.sheet_tool import get_service, SPREADSHEET_ID

def detailed_report():
    service = get_service()
    chapters = [
        ("Chapter 1: Introduction", "cattle_thesis_p3_latex/chapters/chapter_1.tex"),
        ("Chapter 2: Literature Review", "cattle_thesis_p3_latex/chapters/chapter_2.tex"),
        ("Chapter 3: Requirements & Constraints", "cattle_thesis_p3_latex/chapters/chapter_3.tex")
    ]

    report = {}

    for title, tex_file in chapters:
        res = service.values().get(spreadsheetId=SPREADSHEET_ID, range=f"'{title}'!A1:D150").execute()
        vals = res.get('values', [])
        
        with open(tex_file, "r", encoding="utf-8") as f:
            tex_content = f.read()
        tex_norm = " ".join(tex_content.split()).lower()

        total = 0
        populated = []
        pending_in_latex = []
        already_in_latex = []

        for idx, r in enumerate(vals, 1):
            if len(r) < 2 or not r[1]:
                continue
            b_text = r[1].strip()
            if b_text.startswith("Original (Do Paraphrase") or re.match(r'^\d\.\d\s+', b_text):
                continue

            total += 1
            c_text = r[2].strip() if len(r) > 2 and r[2] else ""

            if c_text and not c_text.startswith("LEAVE BLANK") and not c_text.lower().startswith("no paraphase"):
                # Check if verbatim copy-paste (some rows had copy-paste alert)
                if c_text == b_text:
                    continue
                sample = " ".join(c_text[:35].split()).lower()
                if sample in tex_norm:
                    already_in_latex.append((idx, c_text[:60]))
                else:
                    pending_in_latex.append((idx, b_text[:50], c_text[:60]))
            else:
                pass

        report[title] = {
            "total": total,
            "populated": len(already_in_latex) + len(pending_in_latex),
            "already_in_latex": already_in_latex,
            "pending_in_latex": pending_in_latex
        }

    print("\n======================= SUMMARY =======================")
    for title, data in report.items():
        print(f"\n{title}:")
        print(f"  Total Paragraphs in Sheet: {data['total']}")
        print(f"  Total Paraphrased in Sheet (Col C): {data['populated']}")
        print(f"  Already Integrated into LaTeX: {len(data['already_in_latex'])}")
        print(f"  NEW / PENDING Integration in LaTeX: {len(data['pending_in_latex'])}")
        if data['pending_in_latex']:
            print("  Pending Rows:")
            for r_idx, b_snip, c_snip in data['pending_in_latex']:
                print(f"    • Row {r_idx}: '{c_snip}...'")

if __name__ == "__main__":
    detailed_report()
