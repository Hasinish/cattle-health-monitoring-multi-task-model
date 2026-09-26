import sys
sys.path.insert(0, ".")
from scripts.sheet_tool import get_service, SPREADSHEET_ID

service = get_service()

res = service.values().get(spreadsheetId=SPREADSHEET_ID, range="'Ethics Statement'!A1:D15").execute()
rows = res.get('values', [])
for idx, r in enumerate(rows, 1):
    print(f"Row {idx}: {r}")
