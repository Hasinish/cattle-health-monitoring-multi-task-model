import sys
from pathlib import Path

WORKSPACE_ROOT = Path("d:/cattle-health-monitoring-multi-task-model")
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.sheet_tool import get_service, SPREADSHEET_ID, SHEET_IDS

def inspect():
    service = get_service()
    meta = service.get(spreadsheetId=SPREADSHEET_ID).execute()
    
    for s in meta['sheets']:
        title = s['properties']['title']
        print(f"\n=======================================================")
        print(f"TAB: {title}")
        print(f"=======================================================")
        
        # Get first 15 rows
        res = service.values().get(spreadsheetId=SPREADSHEET_ID, range=f"'{title}'!A1:G15").execute()
        rows = res.get('values', [])
        for idx, row in enumerate(rows, start=1):
            rendered_cells = []
            for col_idx, cell in enumerate(row):
                col_letter = chr(ord('A') + col_idx)
                snippet = cell.replace('\n', ' ')
                if len(snippet) > 60:
                    snippet = snippet[:57] + "..."
                rendered_cells.append(f"{col_letter}: {snippet}")
            print(f"Row {idx:2d} | " + " | ".join(rendered_cells))

if __name__ == "__main__":
    inspect()
