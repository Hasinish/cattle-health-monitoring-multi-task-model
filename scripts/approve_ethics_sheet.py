import sys
sys.path.insert(0, ".")
from scripts.sheet_tool import get_service, SPREADSHEET_ID

service = get_service()

data = [
    {'range': "'Ethics Statement'!D2", 'values': [["✅ ঠিক আছে, কোনো সমস্যা নেই!"]]},
    {'range': "'Ethics Statement'!D5", 'values': [["✅ ঠিক আছে, কোনো সমস্যা নেই!"]]},
    {'range': "'Ethics Statement'!D8", 'values': [["✅ ঠিক আছে, কোনো সমস্যা নেই!"]]},
]

service.values().batchUpdate(
    spreadsheetId=SPREADSHEET_ID,
    body={'valueInputOption': 'USER_ENTERED', 'data': data}
).execute()

print("Ethics Statement review approvals updated in Google Sheets!")
