import sys
sys.path.insert(0, ".")
from scripts.sheet_tool import get_service, SPREADSHEET_ID

service = get_service()

sheet_metadata = service.get(spreadsheetId=SPREADSHEET_ID).execute()
sheet_names = [s['properties']['title'] for s in sheet_metadata.get('sheets', [])]
target_sheet = [s for s in sheet_names if "appendix b" in s.lower()][0]
print(f"Target sheet: {target_sheet}")

res = service.values().get(spreadsheetId=SPREADSHEET_ID, range=f"'{target_sheet}'!A1:D40").execute()
rows = res.get('values', [])
for idx, r in enumerate(rows, 1):
    print(f"Row {idx}: {r}")
