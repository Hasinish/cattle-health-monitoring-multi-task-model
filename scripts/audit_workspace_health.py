import os
import sys
import re

def audit_latex():
    print("=== 1. AUDITING LATEX FILES ===")
    chapters_dir = "cattle_thesis_p3_latex/chapters"
    all_ok = True
    for fname in sorted(os.listdir(chapters_dir)):
        if not fname.endswith(".tex"):
            continue
        fpath = os.path.join(chapters_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()

        # Braces balance
        # Exclude escaped \{ and \}
        cleaned = re.sub(r'\\[{}]', '', content)
        open_b = cleaned.count('{')
        close_b = cleaned.count('}')
        if open_b != close_b:
            print(f"[WARN] {fname}: Unbalanced braces: open={open_b}, close={close_b}")
            all_ok = False
        else:
            print(f"[OK] {fname}: Braces perfectly balanced ({open_b})")

        # Check for undefined or malformed cite
        malformed_cites = re.findall(r'\\cite\s*\{[^}]*$', content)
        if malformed_cites:
            print(f"[ERROR] {fname}: Malformed citations found: {malformed_cites}")
            all_ok = False

    return all_ok

def audit_sheets():
    print("\n=== 2. AUDITING GOOGLE SHEETS ===")
    sys.path.insert(0, ".")
    from scripts.sheet_tool import get_service, SPREADSHEET_ID
    service = get_service()
    meta = service.get(spreadsheetId=SPREADSHEET_ID).execute()
    sheets = meta.get('sheets', [])
    print(f"Total sheets found: {len(sheets)}")
    
    expected_sheets = [
        "Chapter 1: Introduction",
        "Chapter 2: Literature Review",
        "Chapter 3: Requirements & Constraints",
        "Chapter 4: Proposed Methodology",
        "Chapter 5: Result Analysis",
        "Chapter 6: Conclusion",
        "Ethics Statement",
        "Abstract",
        "Acknowledgement",
        "Appendix A",
        "Appendix B"
    ]

    for title in expected_sheets:
        sh = next((s for s in sheets if s['properties']['title'] == title), None)
        if not sh:
            print(f"[ERROR] Missing sheet: {title}")
            continue
        sheet_id = sh['properties']['sheetId']
        # Read B and C
        res = service.values().get(spreadsheetId=SPREADSHEET_ID, range=f"'{title}'!A1:D100").execute()
        vals = res.get('values', [])
        print(f"[OK] Sheet '{title}' (ID {sheet_id}): {len(vals)} rows returned")

        # In Ch 4, 5, 6, check if Column B has any unparsed LaTeX tags
        if title in ["Chapter 4: Proposed Methodology", "Chapter 5: Result Analysis", "Chapter 6: Conclusion"]:
            for r_idx, r in enumerate(vals):
                if len(r) > 1 and r[1]:
                    txt = r[1]
                    bad_tags = re.findall(r'\\(?:cite|ref|textbf|textit|section|subsection|evidence)\{', txt)
                    if bad_tags:
                        print(f"  [WARN] Row {r_idx+1} in '{title}' has raw LaTeX tags: {bad_tags}")

if __name__ == "__main__":
    audit_latex()
    audit_sheets()
