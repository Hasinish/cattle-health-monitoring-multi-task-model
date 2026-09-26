import sys
import json
sys.path.insert(0, ".")
from scripts.sheet_tool import get_service, SPREADSHEET_ID

def main():
    service = get_service()
    chapters = [
        "Chapter 4: Proposed Methodology",
        "Chapter 5: Result Analysis",
        "Chapter 6: Conclusion"
    ]
    report = {}
    for title in chapters:
        res = service.values().get(spreadsheetId=SPREADSHEET_ID, range=f"'{title}'!A1:D100").execute()
        vals = res.get('values', [])
        report[title] = []
        for idx, r in enumerate(vals, 1):
            if len(r) > 1 and r[1] and r[1] != "Original (Do Paraphrase)":
                b_text = r[1]
                c_text = r[2] if len(r) > 2 else ""
                d_text = r[3] if len(r) > 3 else ""
                report[title].append({
                    "row": idx,
                    "orig": b_text,
                    "para": c_text,
                    "rev": d_text
                })

    with open("scratch/ch456_dump.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("DUMP COMPLETED!")

if __name__ == "__main__":
    main()
