"""
Populate Chapter 1 Tab in Google Docs with a 1-Column Multi-Row Table.

Tab: Chapter 1: Introduction (tabId: 't.0')
Document ID: 1XrZgw-45ZhicfpWZ1NimfZV_mzDoBJiG440uYRjx8zI

Table Design:
  1 Column Table
  Row 0: Header Banner
  For each paragraph i:
    Row 2*i + 1: [ORIGINAL + WARNING TAG] Content of paragraph i
    Row 2*i + 2: [EMPTY / PARAPHRASED] Empty row reserved for teammate input
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
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.export_paraphrase_docs import clean_latex, parse_latex_sections

DOCUMENT_ID = "1XrZgw-45ZhicfpWZ1NimfZV_mzDoBJiG440uYRjx8zI"
TAB_ID = "t.0"
SCOPES = ["https://www.googleapis.com/auth/documents"]


def build_chapter1_items():
    ch1_path = WORKSPACE_ROOT / "cattle_thesis_p3_latex" / "chapters" / "chapter_1.tex"
    content = ch1_path.read_text(encoding="utf-8")
    sections = parse_latex_sections(content)

    items = []
    for lvl, title, body in sections:
        paras = [p.strip() for p in body.split("\n\n") if p.strip()]
        for p_idx, p in enumerate(paras):
            t_lower = title.lower()
            
            # Determine appropriate status and warning tag
            if "preliminary investigations" in t_lower or "limitations" in t_lower:
                tag = "🟡 [DO PARAPHRASE — RULE: NEVER USE 'PHASE 2']\n" \
                      "⚠️ WARNING: Describe previous baseline work as 'preliminary monolithic investigations'. NEVER use the words 'Phase 1', 'Phase 2', or 'P2'."
            elif "objective" in t_lower or "research question" in t_lower:
                tag = "🟡 [DO PARAPHRASE WITH CARE — RESEARCH QUESTIONS ARE LOCKED]\n" \
                      "⚠️ NOTE: Preserve the exact core scientific questions (RQ1: Visual Representations, RQ2: Multi-Task Negative Transfer, RQ3: Cross-Setting Robustness)."
            else:
                tag = "🟢 [DO PARAPHRASE — 100% FREEZE-SAFE]\n" \
                      "ℹ️ INFO: Standard academic monograph prose. Preserve all '[cite: key]' tags exactly."

            header_info = f"--- SECTION {title.upper()} (Para {p_idx + 1}/{len(paras)}) ---\n{tag}\n\n"
            full_original = header_info + p

            items.append({
                "section": title,
                "original": full_original,
                "para_text": p
            })
    return items


def main():
    print("=== POPULATING CHAPTER 1 WITH 1-COLUMN MULTI-ROW TABLE ===")
    token_path = WORKSPACE_ROOT / "token.json"
    if not token_path.exists():
        print("ERROR: token.json not found!")
        sys.exit(1)

    creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    docs_service = build("docs", "v1", credentials=creds)

    items = build_chapter1_items()
    print(f"[*] Total paragraphs to insert: {len(items)}")

    # Total rows: 1 header row + 2 rows per paragraph (Original row + Empty Paraphrase row)
    total_rows = 1 + (len(items) * 2)
    print(f"[*] Total table rows to construct: {total_rows} (1 column)")

    # Step 1: Ensure tab t.0 is empty
    doc = docs_service.documents().get(documentId=DOCUMENT_ID, includeTabsContent=True).execute()
    tab_t0 = None
    for t in doc.get("tabs", []):
        if t.get("tabProperties", {}).get("tabId") == TAB_ID:
            tab_t0 = t
            break

    if not tab_t0:
        print(f"ERROR: Tab {TAB_ID} not found in document!")
        sys.exit(1)

    body_content = tab_t0.get("documentTab", {}).get("body", {}).get("content", [])
    if body_content:
        end_idx = body_content[-1].get("endIndex", 1)
        if end_idx > 2:
            print(f"[*] Clearing existing content in tab {TAB_ID} (length: {end_idx})...")
            clear_req = [{
                "deleteContentRange": {
                    "range": {
                        "tabId": TAB_ID,
                        "startIndex": 1,
                        "endIndex": end_idx - 1
                    }
                }
            }]
            docs_service.documents().batchUpdate(documentId=DOCUMENT_ID, body={"requests": clear_req}).execute()
            print("[OK] Tab cleared.")

    # Step 2: Insert 1-column table
    print(f"[*] Inserting empty {total_rows}x1 table into tab {TAB_ID}...")
    insert_table_req = [{
        "insertTable": {
            "rows": total_rows,
            "columns": 1,
            "location": {
                "tabId": TAB_ID,
                "index": 1
            }
        }
    }]
    docs_service.documents().batchUpdate(documentId=DOCUMENT_ID, body={"requests": insert_table_req}).execute()
    print("[OK] Empty table created successfully!")

    # Step 3: Fetch table cell indices
    print("[*] Fetching newly created table indices...")
    doc = docs_service.documents().get(documentId=DOCUMENT_ID, includeTabsContent=True).execute()
    for t in doc.get("tabs", []):
        if t.get("tabProperties", {}).get("tabId") == TAB_ID:
            tab_t0 = t
            break

    table_element = None
    for el in tab_t0.get("documentTab", {}).get("body", {}).get("content", []):
        if "table" in el:
            table_element = el["table"]
            break

    if not table_element:
        print("ERROR: Failed to find table element in tab!")
        sys.exit(1)

    table_rows = table_element.get("tableRows", [])
    print(f"[OK] Located table with {len(table_rows)} rows.")

    # Build row texts: row_idx -> text
    row_texts = {}

    # Header Row 0
    header_text = (
        "📘 CHAPTER 1: INTRODUCTION — PARAPHRASING WORKBENCH\n"
        "========================================================================================\n"
        "• HOW TO USE: Each paragraph has an ORIGINAL row followed by an EMPTY row.\n"
        "• TEAM INPUT: Type your humanized/paraphrased version into the empty row directly below each paragraph.\n"
        "• CITATIONS: Never delete '[cite: key]' tags. Move them naturally with your paraphrased sentences.\n"
        "• WARNINGS: Watch for yellow warnings (e.g. never use 'Phase 2'; keep RQ1-RQ3 meanings exact).\n"
        "========================================================================================"
    )
    row_texts[0] = header_text

    for i, itm in enumerate(items):
        orig_row_idx = 1 + (i * 2)
        empty_row_idx = orig_row_idx + 1

        row_texts[orig_row_idx] = itm["original"]
        # Empty row kept for teammate paraphrased version
        row_texts[empty_row_idx] = (
            f"✍️ [PARAPHRASED VERSION FOR PARA {i+1} — TYPE HERE]:\n"
            f""
        )

    # Step 4: Batch insert text into cells in REVERSE order (from bottom row to top row)
    # Reverse order guarantees that model index shifts do NOT invalidate preceding cell start indices!
    print("[*] Preparing reverse-order text insertion requests...")
    insert_requests = []

    # Sort row indices descending
    for r_idx in range(len(table_rows) - 1, -1, -1):
        if r_idx in row_texts:
            cell = table_rows[r_idx]["tableCells"][0]
            # First element in cell paragraph
            cell_start = cell["content"][0]["paragraph"]["elements"][0]["startIndex"]
            insert_requests.append({
                "insertText": {
                    "location": {
                        "tabId": TAB_ID,
                        "index": cell_start
                    },
                    "text": row_texts[r_idx]
                }
            })

    print(f"[*] Dispatching {len(insert_requests)} text insertion requests to Google Docs API...")
    docs_service.documents().batchUpdate(documentId=DOCUMENT_ID, body={"requests": insert_requests}).execute()
    print("[SUCCESS] Chapter 1 table successfully populated! 🎉")
    print(f" -> URL: https://docs.google.com/document/d/{DOCUMENT_ID}/edit?tab={TAB_ID}")


if __name__ == "__main__":
    main()
