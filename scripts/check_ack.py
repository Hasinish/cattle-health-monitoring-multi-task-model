import sys
sys.path.insert(0, ".")
from scripts.sheet_tool import get_service, SPREADSHEET_ID

service = get_service()
res = service.values().get(spreadsheetId=SPREADSHEET_ID, range="'Acknowledgement'!A1:D30").execute()
vals = res.get('values', [])
for idx, r in enumerate(vals, 1):
    print(f"Row {idx}: {r}")
