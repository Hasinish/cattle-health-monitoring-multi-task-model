import sys
sys.path.insert(0, ".")
from scripts.sheet_tool import get_service, SPREADSHEET_ID

def main():
    service = get_service()
    chapters = [
        "Chapter 4: Proposed Methodology",
        "Chapter 5: Result Analysis",
        "Chapter 6: Conclusion"
    ]
    for title in chapters:
        res = service.values().get(spreadsheetId=SPREADSHEET_ID, range=f"'{title}'!A1:D100").execute()
        vals = res.get('values', [])
        print(f"\n==================== {title} (Rows: {len(vals)}) ====================")
        for idx, r in enumerate(vals, 1):
            if len(r) > 1 and r[1] and r[1] != "Original (Do Paraphrase)":
                b_text = r[1]
                c_text = r[2] if len(r) > 2 else ""
                d_text = r[3] if len(r) > 3 else ""
                print(f"\n--- Row {idx} ---")
                print(f"Col B (Original):\n{b_text}\n")
                print(f"Col C (Paraphrase):\n{c_text if c_text else '<EMPTY>'}\n")
                print(f"Col D (Review):\n{d_text if d_text else '<EMPTY>'}\n")

if __name__ == "__main__":
    main()
