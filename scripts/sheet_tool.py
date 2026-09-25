"""
High-Speed Google Sheets Turbo CLI for Thesis Paraphrasing Workbench.
Executes batch gets, searches, and atomic updates in 2-3 seconds with robust auto-retry.
"""

import sys
import os
import socket
import time
import argparse
from pathlib import Path

# Safe socket timeout (15s prevents 60s OS hangs while allowing full payload download)
socket.setdefaulttimeout(15.0)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, ".")
from googleapiclient.discovery import build
from scripts.upload_to_google_sheet import get_credentials

SPREADSHEET_ID = "14UIi22gtPx_ogVGBPfhV45rN1R-zTREAQG3Aymcqk0A"
SHEET_IDS = {
    "Chapter 1: Introduction": 0,
    "Chapter 2: Literature Review": 1111150292,
    "Chapter 3: Requirements & Constraints": 521635664
}

def robust_execute(request_callable, max_retries=3):
    for attempt in range(max_retries):
        try:
            return request_callable()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(1.0)

def get_service():
    creds = get_credentials()
    return build("sheets", "v4", credentials=creds, cache_discovery=False).spreadsheets()

def search_text(query):
    service = get_service()
    query_lower = query.lower()
    
    ranges = [f"'{tab}'!A1:D260" for tab in SHEET_IDS.keys()]
    res = robust_execute(lambda: service.values().batchGet(spreadsheetId=SPREADSHEET_ID, ranges=ranges).execute())
    
    results = []
    for tab_name, val_range in zip(SHEET_IDS.keys(), res.get("valueRanges", [])):
        rows = val_range.get("values", [])
        for idx, r in enumerate(rows, start=1):
            for c_idx, cell in enumerate(r):
                if query_lower in cell.lower():
                    col_letter = chr(ord('A') + c_idx)
                    snippet = cell.replace("\n", " ")[:90]
                    results.append((tab_name, idx, col_letter, snippet))
    
    if not results:
        print(f"[!] No matches found for '{query}'.")
        return
    
    print(f"\n[FOUND {len(results)} MATCHES FOR '{query}']:")
    for tab, row, col, snip in results:
        print(f"  • {tab} -> Row {row} (Col {col}): \"{snip}...\"")
    print()

def update_review(tab_name, row_num, feedback_text, highlight_yellow=True):
    service = get_service()
    feedback_text = feedback_text.replace("\\n", "\n")
    sheet_id = SHEET_IDS.get(tab_name)
    if sheet_id is None:
        print(f"[ERROR] Tab '{tab_name}' not recognized!")
        return

    row_idx = row_num - 1
    requests = []
    
    if highlight_yellow:
        requests.append({
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": row_idx,
                    "endRowIndex": row_idx + 1,
                    "startColumnIndex": 2,
                    "endColumnIndex": 3
                },
                "cell": {
                    "userEnteredFormat": {
                        "backgroundColor": {"red": 1.0, "green": 0.98, "blue": 0.8}
                    }
                },
                "fields": "userEnteredFormat.backgroundColor"
            }
        })

    requests.append({
        "repeatCell": {
            "range": {
                "sheetId": sheet_id,
                "startRowIndex": row_idx,
                "endRowIndex": row_idx + 1,
                "startColumnIndex": 3,
                "endColumnIndex": 4
            },
            "cell": {
                "userEnteredFormat": {
                    "wrapStrategy": "WRAP",
                    "verticalAlignment": "TOP",
                    "textFormat": {"fontSize": 10}
                }
            },
            "fields": "userEnteredFormat(wrapStrategy,verticalAlignment,textFormat)"
        }
    })

    # 1. Update text
    robust_execute(lambda: service.values().update(
        spreadsheetId=SPREADSHEET_ID,
        range=f"'{tab_name}'!D{row_num}",
        valueInputOption="USER_ENTERED",
        body={"values": [[feedback_text]]}
    ).execute())

    # 2. Update format
    if requests:
        robust_execute(lambda: service.batchUpdate(
            spreadsheetId=SPREADSHEET_ID,
            body={"requests": requests}
        ).execute())

    print(f"[TURBO SUCCESS] {tab_name} Row {row_num} updated + formatted in 1 shot!")

def get_row(tab_name, row_num):
    service = get_service()
    res = robust_execute(lambda: service.values().get(
        spreadsheetId=SPREADSHEET_ID,
        range=f"'{tab_name}'!A{row_num}:D{row_num}"
    ).execute())
    vals = res.get("values", [[]])[0]
    col_b = vals[1] if len(vals) > 1 else ""
    col_c = vals[2] if len(vals) > 2 else ""
    col_d = vals[3] if len(vals) > 3 else ""
    print(f"\n--- [{tab_name} Row {row_num}] ---")
    print(f"Col B (Original):   {col_b}")
    print(f"Col C (Teammate):   {col_c}")
    print(f"Col D (Feedback):   {col_d}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turbo Google Sheets Review Tool")
    subparsers = parser.add_subparsers(dest="cmd")

    search_p = subparsers.add_parser("find")
    search_p.add_argument("query", help="Text to search")

    row_p = subparsers.add_parser("get")
    row_p.add_argument("--tab", required=True)
    row_p.add_argument("--row", type=int, required=True)

    update_p = subparsers.add_parser("set")
    update_p.add_argument("--tab", required=True)
    update_p.add_argument("--row", type=int, required=True)
    update_p.add_argument("--text", required=True)
    update_p.add_argument("--no-yellow", action="store_true")

    args = parser.parse_args()
    if args.cmd == "find":
        search_text(args.query)
    elif args.cmd == "get":
        get_row(args.tab, args.row)
    elif args.cmd == "set":
        update_review(args.tab, args.row, args.text, highlight_yellow=not args.no_yellow)
    else:
        parser.print_help()
