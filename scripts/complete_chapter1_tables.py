"""
Complete Chapter 1 Tab in Google Docs with Exact User 1-Column Table Formatting.

Document ID: 1XrZgw-45ZhicfpWZ1NimfZV_mzDoBJiG440uYRjx8zI
Tab ID: 't.0' (Chapter 1: Introduction)

Formatting Pattern per Paragraph:
  Row 4*i + 0: Original (Do Paraphrase) [Bold, Red Highlight RGB(1,0,0), 11pt]
  Row 4*i + 1: Clean natural English original text [Normal weight, 11pt, zero citations/LaTeX]
  Row 4*i + 2: Paraphrased: [Bold, Green Highlight RGB(0,1,0), 11pt]
  Row 4*i + 3: Empty lines for teammate typing [Normal weight, 11pt]
"""

import sys
import time
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


def get_sections_data():
    ch1 = (WORKSPACE_ROOT / "cattle_thesis_p3_latex" / "chapters" / "chapter_1.tex").read_text(encoding="utf-8")
    raw_sections = parse_latex_sections(ch1)

    sections_data = [
        {
            "heading": "1.1 Background",
            "level": 1,
            "paras": [clean_latex(p.strip()) for p in raw_sections[1][2].split("\n\n") if clean_latex(p.strip())],
            "warning": None
        },
        {
            "heading": "1.2 Rationale of the Study and Motivation",
            "level": 1,
            "paras": [],
            "warning": None
        },
        {
            "heading": "1.2.1 Task-Appropriate Visual Representations",
            "level": 2,
            "paras": [clean_latex(p.strip()) for p in raw_sections[3][2].split("\n\n") if clean_latex(p.strip())],
            "warning": None
        },
        {
            "heading": "1.2.2 Preliminary Investigations and Their Limitations",
            "level": 2,
            "paras": [clean_latex(p.strip()) for p in raw_sections[4][2].split("\n\n") if clean_latex(p.strip())],
            "warning": "Warning: Never use 'Phase 2' (use 'preliminary monolithic baseline investigations')"
        },
        {
            "heading": "1.2.3 Rationale for Jointly Studying the Three Tasks",
            "level": 2,
            "paras": [clean_latex(p.strip()) for p in raw_sections[5][2].split("\n\n") if clean_latex(p.strip())],
            "warning": None
        },
        {
            "heading": "1.3 Problem Statement",
            "level": 1,
            "paras": [clean_latex(p.strip()) for p in raw_sections[6][2].split("\n\n") if clean_latex(p.strip())],
            "warning": None
        },
        {
            "heading": "1.4 Objectives & Research Questions",
            "level": 1,
            "paras": [clean_latex(p.strip()) for p in raw_sections[7][2].split("\n\n") if clean_latex(p.strip())],
            "warning": "Note: Keep RQ1, RQ2, and RQ3 core definitions exact"
        },
        {
            "heading": "1.5 Methodology in Brief",
            "level": 1,
            "paras": [clean_latex(p.strip()) for p in raw_sections[8][2].split("\n\n") if clean_latex(p.strip())],
            "warning": None
        },
        {
            "heading": "1.6 Scopes and Challenges",
            "level": 1,
            "paras": [clean_latex(p.strip()) for p in raw_sections[9][2].split("\n\n") if clean_latex(p.strip())],
            "warning": None
        }
    ]
    return sections_data


def add_section_and_table(docs_service, section_info):
    heading = section_info["heading"]
    lvl = section_info["level"]
    paras = section_info["paras"]
    warning = section_info["warning"]

    print(f"\n[*] Processing Section: {heading} ({len(paras)} paragraphs)...")

    # 1. Append heading text
    insert_heading_req = [{
        "insertText": {
            "text": f"\n\n{heading}\n",
            "endOfSegmentLocation": {"tabId": TAB_ID}
        }
    }]
    docs_service.documents().batchUpdate(documentId=DOCUMENT_ID, body={"requests": insert_heading_req}).execute()

    # Style heading
    doc = docs_service.documents().get(documentId=DOCUMENT_ID, includeTabsContent=True).execute()
    t0 = [t for t in doc.get("tabs", []) if t.get("tabProperties", {}).get("tabId") == TAB_ID][0]
    body_content = t0["documentTab"]["body"]["content"]
    last_p = [el for el in body_content if "paragraph" in el][-1]
    p_start = last_p["startIndex"]
    p_end = last_p["endIndex"]

    font_size = 14 if lvl == 1 else 12
    style_heading_req = [{
        "updateTextStyle": {
            "range": {"tabId": TAB_ID, "startIndex": p_start, "endIndex": p_end},
            "textStyle": {"bold": True, "fontSize": {"magnitude": font_size, "unit": "PT"}},
            "fields": "bold,fontSize"
        }
    }]
    docs_service.documents().batchUpdate(documentId=DOCUMENT_ID, body={"requests": style_heading_req}).execute()

    if not paras:
        print(f"[OK] Section heading '{heading}' inserted (parent section, no direct paragraphs).")
        return

    # 2. Append Table (4 * len(paras) rows)
    total_rows = len(paras) * 4
    insert_tbl_req = [{
        "insertTable": {
            "rows": total_rows,
            "columns": 1,
            "endOfSegmentLocation": {"tabId": TAB_ID}
        }
    }]
    docs_service.documents().batchUpdate(documentId=DOCUMENT_ID, body={"requests": insert_tbl_req}).execute()

    # 3. Fetch newly inserted table
    doc = docs_service.documents().get(documentId=DOCUMENT_ID, includeTabsContent=True).execute()
    t0 = [t for t in doc.get("tabs", []) if t.get("tabProperties", {}).get("tabId") == TAB_ID][0]
    tbl = [el["table"] for el in t0["documentTab"]["body"]["content"] if "table" in el][-1]
    table_rows = tbl.get("tableRows", [])

    row_texts = {}
    for p_i, p_text in enumerate(paras):
        r_orig_hdr = p_i * 4
        r_orig_txt = r_orig_hdr + 1
        r_para_hdr = r_orig_txt + 1
        r_para_txt = r_para_hdr + 1

        if warning:
            row_texts[r_orig_hdr] = f"Original (Do Paraphrase — {warning})\n"
        else:
            row_texts[r_orig_hdr] = "Original (Do Paraphrase)\n"

        row_texts[r_orig_txt] = p_text + "\n"
        row_texts[r_para_hdr] = "Paraphrased:\n"
        row_texts[r_para_txt] = "\n\n\n\n\n"

    # Insert text in reverse row order
    insert_cell_reqs = []
    for r_idx in range(len(table_rows) - 1, -1, -1):
        cell = table_rows[r_idx]["tableCells"][0]
        cell_start = cell["content"][0]["paragraph"]["elements"][0]["startIndex"]
        insert_cell_reqs.append({
            "insertText": {
                "location": {"tabId": TAB_ID, "index": cell_start},
                "text": row_texts[r_idx]
            }
        })

    docs_service.documents().batchUpdate(documentId=DOCUMENT_ID, body={"requests": insert_cell_reqs}).execute()
    print(f"[OK] Section '{heading}' completely built with {len(paras)} paragraph units!")


def standardize_tab_styles(docs_service):
    """Ensure every cell is strictly 11pt, correct bold/highlighting, and zero residual styling."""
    print("\n[*] Applying comprehensive styling (11pt font, headers highlighted, body unbolded)...")
    doc = docs_service.documents().get(documentId=DOCUMENT_ID, includeTabsContent=True).execute()
    t0 = [t for t in doc.get("tabs", []) if t.get("tabProperties", {}).get("tabId") == TAB_ID][0]
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
                                        update_reqs.append({
                                            "updateTextStyle": {
                                                "range": {"tabId": TAB_ID, "startIndex": s_idx, "endIndex": e_idx},
                                                "textStyle": {
                                                    "bold": True,
                                                    "fontSize": {"magnitude": 11, "unit": "PT"},
                                                    "backgroundColor": {"color": {"rgbColor": {"red": 1.0, "green": 0.0, "blue": 0.0}}}
                                                },
                                                "fields": "bold,fontSize,backgroundColor"
                                            }
                                        })
                                    elif text.startswith("Paraphrased:"):
                                        update_reqs.append({
                                            "updateTextStyle": {
                                                "range": {"tabId": TAB_ID, "startIndex": s_idx, "endIndex": e_idx},
                                                "textStyle": {
                                                    "bold": True,
                                                    "fontSize": {"magnitude": 11, "unit": "PT"},
                                                    "backgroundColor": {"color": {"rgbColor": {"red": 0.0, "green": 1.0, "blue": 0.0}}}
                                                },
                                                "fields": "bold,fontSize,backgroundColor"
                                            }
                                        })
                                    else:
                                        update_reqs.append({
                                            "updateTextStyle": {
                                                "range": {"tabId": TAB_ID, "startIndex": s_idx, "endIndex": e_idx},
                                                "textStyle": {
                                                    "bold": False,
                                                    "fontSize": {"magnitude": 11, "unit": "PT"}
                                                },
                                                "fields": "bold,fontSize"
                                            }
                                        })

    print(f"[*] Prepared {len(update_reqs)} styling updates across all tables.")
    chunk_size = 500
    for i in range(0, len(update_reqs), chunk_size):
        chunk = update_reqs[i:i + chunk_size]
        docs_service.documents().batchUpdate(documentId=DOCUMENT_ID, body={"requests": chunk}).execute()
    print("[SUCCESS] All table text standardized to 11pt, clean weights, and colored headers! 🎉")


def main():
    print("=== COMPLETING CHAPTER 1 IN GOOGLE DOCS (EXACT USER TEMPLATE) ===")
    token_path = WORKSPACE_ROOT / "token.json"
    creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    docs_service = build("docs", "v1", credentials=creds)

    # 1. Clear Tab t.0 to build clean from start to finish
    doc = docs_service.documents().get(documentId=DOCUMENT_ID, includeTabsContent=True).execute()
    t0 = [t for t in doc.get("tabs", []) if t.get("tabProperties", {}).get("tabId") == TAB_ID][0]
    body_content = t0["documentTab"]["body"]["content"]
    end_idx = body_content[-1].get("endIndex", 1)

    print(f"[*] Resetting Tab {TAB_ID} (current length: {end_idx})...")
    if end_idx > 2:
        clear_req = [{"deleteContentRange": {"range": {"tabId": TAB_ID, "startIndex": 1, "endIndex": end_idx - 1}}}]
        docs_service.documents().batchUpdate(documentId=DOCUMENT_ID, body={"requests": clear_req}).execute()
        print("[OK] Tab cleared.")

    # 2. Build each section sequentially
    sections = get_sections_data()
    for s_info in sections:
        add_section_and_table(docs_service, s_info)
        time.sleep(0.3)

    # 3. Apply comprehensive 11pt styling and highlights across all tables
    standardize_tab_styles(docs_service)

    print("\n" + "=" * 60)
    print("🏆 ALL 9 SECTIONS OF CHAPTER 1 POPULATED & 100% STANDARDIZED!")
    print(f" -> URL: https://docs.google.com/document/d/{DOCUMENT_ID}/edit?tab={TAB_ID}")
    print("=" * 60)


if __name__ == "__main__":
    main()
