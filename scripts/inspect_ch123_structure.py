import sys
sys.path.insert(0, ".")
from scripts.sheet_tool import get_service, SPREADSHEET_ID

service = get_service()
for title in ['Chapter 1: Introduction', 'Chapter 2: Literature Review', 'Chapter 3: Requirements & Constraints']:
    res = service.values().get(spreadsheetId=SPREADSHEET_ID, range=f"'{title}'!A1:D15").execute()
    vals = res.get('values', [])
    print(f"\n=== {title} ===")
    for idx, r in enumerate(vals, 1):
        print(f"Row {idx}: {r}")
