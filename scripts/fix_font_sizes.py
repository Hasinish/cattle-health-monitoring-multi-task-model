"""
Comprehensive Font and Style Fix for Chapter 1 Tab in Google Docs.
- Sets ALL table text to exactly 11pt.
- Cleans body paragraphs to normal weight (bold: False).
- Keeps 'Original...' bold with Red highlight.
- Keeps 'Paraphrased:' bold with Green highlight.
"""

import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

WORKSPACE_ROOT = Path("d:/cattle-health-monitoring-multi-task-model")
token_path = WORKSPACE_ROOT / "token.json"
creds = Credentials.from_authorized_user_file(str(token_path), ["https://www.googleapis.com/auth/documents"])
docs_service = build("docs", "v1", credentials=creds)
doc_id = "1XrZgw-45ZhicfpWZ1NimfZV_mzDoBJiG440uYRjx8zI"
tab_id = "t.0"

print("=== COMPREHENSIVE FONT SIZE (11PT) & BOLD CLEANUP ===")
doc = docs_service.documents().get(documentId=doc_id, includeTabsContent=True).execute()
t0 = [t for t in doc.get("tabs", []) if t.get("tabProperties", {}).get("tabId") == tab_id][0]
body = t0["documentTab"]["body"]["content"]

update_reqs = []

for el in body:
    if "table" in el:
        tbl = el["table"]
        for r in tbl.get("tableRows", []):
            for c in r.get("tableCells", []):
                for c_el in c.get("content", []):
                    if "paragraph" in c_el:
                        for p_elem in c_el["paragraph"]["elements"]:
                            if "textRun" in p_elem:
                                tr = p_elem["textRun"]
                                text = tr.get("content", "")
                                s_idx = p_elem["startIndex"]
                                e_idx = p_elem["endIndex"]

                                if text.startswith("Original"):
                                    # Bold, Red Highlight, 11pt
                                    update_reqs.append({
                                        "updateTextStyle": {
                                            "range": {"tabId": tab_id, "startIndex": s_idx, "endIndex": e_idx},
                                            "textStyle": {
                                                "bold": True,
                                                "fontSize": {"magnitude": 11, "unit": "PT"},
                                                "backgroundColor": {"color": {"rgbColor": {"red": 1.0, "green": 0.0, "blue": 0.0}}}
                                            },
                                            "fields": "bold,fontSize,backgroundColor"
                                        }
                                    })
                                elif text.startswith("Paraphrased:"):
                                    # Bold, Green Highlight, 11pt
                                    update_reqs.append({
                                        "updateTextStyle": {
                                            "range": {"tabId": tab_id, "startIndex": s_idx, "endIndex": e_idx},
                                            "textStyle": {
                                                "bold": True,
                                                "fontSize": {"magnitude": 11, "unit": "PT"},
                                                "backgroundColor": {"color": {"rgbColor": {"red": 0.0, "green": 1.0, "blue": 0.0}}}
                                            },
                                            "fields": "bold,fontSize,backgroundColor"
                                        }
                                    })
                                else:
                                    # Body paragraphs / empty lines -> Normal unbolded, 11pt
                                    update_reqs.append({
                                        "updateTextStyle": {
                                            "range": {"tabId": tab_id, "startIndex": s_idx, "endIndex": e_idx},
                                            "textStyle": {
                                                "bold": False,
                                                "fontSize": {"magnitude": 11, "unit": "PT"}
                                            },
                                            "fields": "bold,fontSize"
                                        }
                                    })

print(f"[*] Prepared {len(update_reqs)} styling updates.")
CHUNK_SIZE = 500
for i in range(0, len(update_reqs), CHUNK_SIZE):
    chunk = update_reqs[i:i + CHUNK_SIZE]
    docs_service.documents().batchUpdate(documentId=doc_id, body={"requests": chunk}).execute()

print("[SUCCESS] All body text set to normal weight (not bold) and 11pt font size! 🎉")
