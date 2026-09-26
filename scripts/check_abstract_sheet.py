import sys
sys.path.insert(0, ".")
from scripts.sheet_tool import get_service, SPREADSHEET_ID

service = get_service()

sheet_metadata = service.get(spreadsheetId=SPREADSHEET_ID).execute()
sheet_names = [s['properties']['title'] for s in sheet_metadata.get('sheets', [])]
print("All sheets in spreadsheet:")
for s in sheet_names:
    print(" -", s)

target_sheet = [s for s in sheet_names if "abstract" in s.lower()][0]
print(f"\nChecking tab: '{target_sheet}'")
res = service.values().get(spreadsheetId=SPREADSHEET_ID, range=f"'{target_sheet}'!A1:D30").execute()
rows = res.get('values', [])
for idx, r in enumerate(rows, 1):
    print(f"Row {idx}: {r}")
