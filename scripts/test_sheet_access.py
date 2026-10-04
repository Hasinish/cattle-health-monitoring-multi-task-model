import sys
from pathlib import Path
WORKSPACE_ROOT = Path("d:/cattle-health-monitoring-multi-task-model")
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.sheet_tool import get_service, SPREADSHEET_ID

def main():
    service = get_service()
    meta = service.get(spreadsheetId=SPREADSHEET_ID).execute()
    print("SPREADSHEET TITLE:", meta['properties']['title'])
    print("\nALL AVAILABLE TABS:")
    for s in meta['sheets']:
        props = s['properties']
        print(f"  • {props['title']} (sheetId: {props['sheetId']}, rows: {props['gridProperties']['rowCount']}, cols: {props['gridProperties']['columnCount']})")

if __name__ == "__main__":
    main()
