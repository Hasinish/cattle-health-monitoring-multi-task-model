import sys
sys.path.insert(0, ".")
from scripts.sheet_tool import get_service, SPREADSHEET_ID

service = get_service()

sheet_metadata = service.get(spreadsheetId=SPREADSHEET_ID).execute()
sheet_names = [s['properties']['title'] for s in sheet_metadata.get('sheets', [])]
target_sheet = [s for s in sheet_names if "appendix b" in s.lower()][0]

rows_to_approve = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
data = []
for r in rows_to_approve:
    data.append({
        'range': f"'{target_sheet}'!D{r}",
        'values': [["✅ ঠিক আছে, কোনো সমস্যা নেই!"]]
    })

service.values().batchUpdate(
    spreadsheetId=SPREADSHEET_ID,
    body={'valueInputOption': 'USER_ENTERED', 'data': data}
).execute()

print(f"Successfully updated review approval in tab '{target_sheet}' for rows {rows_to_approve}!")
