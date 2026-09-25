import sys
import os

sys.path.insert(0, os.path.abspath("."))
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from scripts.sheet_tool import get_service, SPREADSHEET_ID, SHEET_IDS, robust_execute

service = get_service()
tab_name = "Chapter 3: Requirements & Constraints"
sheet_id = SHEET_IDS[tab_name]

print(f"Fetching '{tab_name}'...")
res = robust_execute(lambda: service.values().get(
    spreadsheetId=SPREADSHEET_ID,
    range=f"'{tab_name}'!A1:D100"
).execute())

rows = res.get("values", [])
print(f"Fetched {len(rows)} rows.")

requests = []
fixed_headers = 0

for idx, r in enumerate(rows, start=1):
    row_idx = idx - 1
    col_b = r[1].strip() if len(r) > 1 else ""
    col_c = r[2].strip() if len(r) > 2 else ""
    
    # Check if header row: 'Original (Do Paraphrase...'
    if col_b.startswith("Original (Do Paraphrase"):
        fixed_headers += 1
        
        # 1. Update text of Col C to 'Paraphrased:'
        # 2. Set Green background (RGB: 0, 1, 0), Bold, 10pt font, Middle aligned
        requests.append({
            "updateCells": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": row_idx,
                    "endRowIndex": row_idx + 1,
                    "startColumnIndex": 2,
                    "endColumnIndex": 3
                },
                "rows": [
                    {
                        "values": [
                            {
                                "userEnteredValue": {"stringValue": "Paraphrased:"},
                                "userEnteredFormat": {
                                    "backgroundColor": {"red": 0.0, "green": 1.0, "blue": 0.0},
                                    "textFormat": {
                                        "bold": True,
                                        "fontSize": 10,
                                        "foregroundColor": {"red": 0.0, "green": 0.0, "blue": 0.0}
                                    },
                                    "verticalAlignment": "MIDDLE",
                                    "wrapStrategy": "WRAP"
                                }
                            }
                        ]
                    }
                ],
                "fields": "userEnteredValue,userEnteredFormat(backgroundColor,textFormat,verticalAlignment,wrapStrategy)"
            }
        })

print(f"Found {fixed_headers} 'Paraphrased:' header rows to restore to GREEN.")

if requests:
    print("Applying live format update to Google Sheets...")
    res = robust_execute(lambda: service.batchUpdate(
        spreadsheetId=SPREADSHEET_ID,
        body={"requests": requests}
    ).execute())
    print("SUCCESS! Restored all green header formatting in Chapter 3!")
