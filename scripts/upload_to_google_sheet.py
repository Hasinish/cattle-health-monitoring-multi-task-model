"""
Upload Freeze-Safe Thesis Chapters to Google Sheets with Side-by-Side Paraphrasing Template.

Layout:
  Col A: Spacer / Row index (40px)
  Col B: Section Heading (Row 1), 'Original (Do Paraphrase)' [Red] (Row 2), Original Text (Row 3, 560px, WRAP)
  Col C: Empty (Row 1), 'Paraphrased:' [Green] (Row 2), Teammate Paraphrase Space (Row 3, 560px, WRAP)
  Row 4: Blank separator row between paragraph blocks.

Chapters Supported:
  - Chapter 1: Introduction (9 sections, 36 paragraphs)
  - Chapter 2: Literature Review (18 sections, 78 paragraphs)
  - Chapter 3: Requirements & Constraints (8 sections, 26 paragraphs)
"""

import sys
import os
import re
import argparse
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

WORKSPACE_ROOT = Path("d:/cattle-health-monitoring-multi-task-model")
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.export_paraphrase_docs import clean_latex, parse_latex_sections

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive.file"
]


def get_credentials():
    token_path = WORKSPACE_ROOT / "token.json"
    creds_path = WORKSPACE_ROOT / "credentials.json"
    creds = None

    needs_auth = False
    if token_path.exists():
        try:
            import json
            with open(token_path) as f:
                data = json.load(f)
            file_scopes = data.get("scopes", [])
            if any(s not in file_scopes for s in SCOPES):
                needs_auth = True
            else:
                creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
                if creds and creds.expired and creds.refresh_token:
                    from google.auth.transport.requests import Request
                    try:
                        creds.refresh(Request())
                        with open(token_path, "w") as token_file:
                            token_file.write(creds.to_json())
                        needs_auth = False
                    except Exception:
                        needs_auth = True
                elif not creds or not creds.valid:
                    needs_auth = True
        except Exception:
            needs_auth = True
    else:
        needs_auth = True

    if needs_auth:
        print("[*] Google Sheets scope needed. Launching browser for one-click approval...")
        flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
        creds = flow.run_local_server(port=0)
        with open(token_path, "w") as token_file:
            token_file.write(creds.to_json())
        print("[OK] New token saved with Sheets permissions!")

    return creds


def get_chapter1_data():
    ch1_file = WORKSPACE_ROOT / "cattle_thesis_p3_latex" / "chapters" / "chapter_1.tex"
    content = ch1_file.read_text(encoding="utf-8")
    raw_sections = parse_latex_sections(content)

    s14_raw = [clean_latex(p.strip()) for p in raw_sections[7][2].split("\n\n") if clean_latex(p.strip())]
    s14_paras = [
        {"text": s14_raw[0], "warning": "Main Thesis Goal: Keep the 3 core tasks intact (BCS, Behavior, Re-ID)"},
        {"text": s14_raw[1], "warning": "Specific Objective 1: Data leakage prevention and cow-grouping protocols"},
        {"text": s14_raw[2], "warning": "Specific Objective 2: Building single-task RGB reference baselines"},
        {"text": s14_raw[3], "warning": "Specific Objective 3: Cattle perception cues (masks, pose, viewpoint, video)"},
        {"text": s14_raw[4], "warning": "Specific Objective 4: Comparing hard sharing vs task-specific modular sharing"},
        {"text": s14_raw[5], "warning": "Specific Objective 5: Testing robustness under external farm domain shifts"},
        {"text": s14_raw[6], "warning": "CRITICAL - THE 3 MAIN RESEARCH QUESTIONS: Do NOT change the 3 questions! Lightly smooth grammar only, but MUST keep exact technical terms: 'cattle-centered representations', 'temporal information', 'task-conditioned sharing', 'hard sharing', 'negative transfer'"},
    ]

    s13_raw = [clean_latex(p.strip()) for p in raw_sections[6][2].split("\n\n") if clean_latex(p.strip())]
    s13_paras = [
        {"text": s13_raw[0], "warning": None},
        {"text": s13_raw[1], "warning": None},
        {"text": s13_raw[2], "warning": None},
        {"text": s13_raw[3], "warning": "The Quoted Question: This is the central thesis question. Do NOT change its meaning, keep the core question in quotes"},
    ]

    sections = [
        {"title": "1.1 Background", "paras": [clean_latex(p.strip()) for p in raw_sections[1][2].split("\n\n") if clean_latex(p.strip())], "warning": None},
        {"title": "1.2 Rationale of the Study and Motivation", "paras": [], "warning": None},
        {"title": "1.2.1 Task-Appropriate Visual Representations", "paras": [clean_latex(p.strip()) for p in raw_sections[3][2].split("\n\n") if clean_latex(p.strip())], "warning": None},
        {"title": "1.2.2 Preliminary Investigations and Their Limitations", "paras": [clean_latex(p.strip()) for p in raw_sections[4][2].split("\n\n") if clean_latex(p.strip())], "warning": "FORBIDDEN PHRASE: NEVER write 'Phase 2'! Write 'preliminary baseline investigations' instead"},
        {"title": "1.2.3 Rationale for Jointly Studying the Three Tasks", "paras": [clean_latex(p.strip()) for p in raw_sections[5][2].split("\n\n") if clean_latex(p.strip())], "warning": None},
        {"title": "1.3 Problem Statement", "paras": s13_paras, "warning": None},
        {"title": "1.4 Objectives & Research Questions", "paras": s14_paras, "warning": None},
        {"title": "1.5 Methodology in Brief", "paras": [clean_latex(p.strip()) for p in raw_sections[8][2].split("\n\n") if clean_latex(p.strip())], "warning": None},
        {"title": "1.6 Scopes and Challenges", "paras": [clean_latex(p.strip()) for p in raw_sections[9][2].split("\n\n") if clean_latex(p.strip())], "warning": "SCOPE BOUNDARY: NEVER say we detect lameness or replace veterinarians. Lameness is excluded from this thesis"},
    ]
    return sections


def get_chapter2_data():
    ch2_file = WORKSPACE_ROOT / "cattle_thesis_p3_latex" / "chapters" / "chapter_2.tex"
    content = ch2_file.read_text(encoding="utf-8")
    raw_sections = parse_latex_sections(content)

    sec_nums = {
        1: ("2.1 Preliminaries", "Theoretical Foundations: Visual representations and transfer learning context"),
        2: ("2.1.1 Visual Representations and Transfer Learning", "Visual Representation: Encoders, pretraining, and cattle-centered cues"),
        3: ("2.1.2 Prediction Tasks and Their Visual Requirements", "Task Taxonomy: Ordered BCS, behavior video, and open-set cow Re-ID"),
        4: ("2.1.3 Multi-Task Learning and Negative Transfer", "Multi-Task Theory: Shared representations, loss weighting, and gradient interference"),
        5: ("2.2 Review of Existing Research", "Literature Review: Integrated precision livestock monitoring"),
        6: ("2.2.1 From Automated Observations to Integrated Livestock Monitoring", "Literature: Evolution from single sensors to vision decision support"),
        7: ("2.2.2 Body Condition Scoring: Anatomy, Image Geometry, and Ordered Prediction", "BCS Literature: Anatomical landmarks (pelvis, loin) and depth/RGB models"),
        8: ("2.2.3 Behavior Recognition: Posture, Motion, and Environmental Context", "Behavior Literature: Posture vs temporal motion and pen context shortcuts"),
        9: ("2.2.4 Individual Identification and Re-ID: From Coat Patterns to Cross-Setting Recognition", "Re-ID Literature: Biometric coat patterns, open-set gallery retrieval"),
        10: ("2.2.5 Localization and Segmentation as Representation Design", "Perception Literature: Foreground segmentation and background noise elimination"),
        11: ("2.2.6 Pose, Anatomy, and Viewpoint", "Structural Priors: DeepLabCut SuperAnimal pose and viewpoint angles"),
        12: ("2.2.7 Temporal Representations for Activity Recognition", "Temporal Modeling: 1D TCN, sequence sampling, and action boundaries"),
        13: ("2.2.8 Shortcut Learning, Domain Shift, and Evaluation", "Shortcut Learning: Geirhos & Xiao background shortcuts and leakage-safe splits"),
        14: ("2.2.9 Multi-Task Learning: Selective Sharing and Negative Transfer", "MTL Interactions: Hard sharing vs task-specific modular pathways"),
        15: ("2.3 Summary of Key Findings", "Synthesis: Identified research gaps in livestock deep learning"),
        16: ("2.3.1 A Common Animal, but Different Information Requirements", "Gap 1: Different tasks looking at the same cow need different features"),
        17: ("2.3.2 Representation Quality Must Be Connected to Robust Evaluation", "Gap 2: Prior papers lacked strict cow-disjoint or burst-group splits"),
        18: ("2.3.3 Selective Sharing as the Basis of the Unified Framework", "Gap 3: Motivation for modular task-conditioned sharing over hard sharing"),
    }

    sections = []
    for i in range(1, 19):
        lvl, title, body = raw_sections[i]
        full_title, note = sec_nums[i]
        raw_paras = [clean_latex(p.strip()) for p in body.split("\n\n") if clean_latex(p.strip())]

        paras_with_notes = []
        for p in raw_paras:
            warn = note
            if "[Formula:" in p:
                warn = "DONT PARAPHRASE THIS — Mathematical Formula (Keep As-Is)"
            elif "[Table Reference:" in p:
                warn = "DONT PARAPHRASE THIS — Table Reference (Omitted from prose)"
            elif "Phase 2" in p:
                warn = "FORBIDDEN PHRASE: NEVER write 'Phase 2'! Write 'prior monolithic multi-task baseline' instead"
            paras_with_notes.append({"text": p, "warning": warn})

        sections.append({
            "title": full_title,
            "paras": paras_with_notes,
            "warning": None
        })
    return sections


def get_chapter3_data():
    ch3_file = WORKSPACE_ROOT / "cattle_thesis_p3_latex" / "chapters" / "chapter_3.tex"
    content = ch3_file.read_text(encoding="utf-8")
    raw_sections = parse_latex_sections(content)

    sec_nums_ch3 = [
        ("3.1 Final Specifications and Requirements", "System Specs: Functional (R1-R4) and non-functional requirements"),
        ("3.2 Societal Impact", "Societal Context: Non-invasive camera monitoring, food security, and veterinary boundaries"),
        ("3.3 Environmental Impact", "Environmental Context: Green AI, GPU compute footprint, and pipeline efficiency"),
        ("3.4 Ethical Issues", "Ethical Responsibility: Absence of biological cow IDs in ScienceDB and honest limitation reporting"),
        ("3.5 Standards and Technical Conventions", "Technical Standards: IEEE, ISO, and anatomical scoring standards"),
        ("3.6 Project Management Plan", "Project Schedule: Research milestone progression stages across tasks"),
        ("3.7 Risk Management", "Scientific Risk Mitigation: Data leakage protection, burst-group splits, and confounding factors"),
        ("3.8 Economic Analysis", "Economic Context: Sensor-based vs computer vision deployment cost-benefit analysis")
    ]

    sections = []
    for i in range(len(raw_sections)):
        lvl, title, body = raw_sections[i]
        full_title, note = sec_nums_ch3[i]
        raw_paras = [clean_latex(p.strip()) for p in body.split("\n\n") if clean_latex(p.strip())]
        paras_with_notes = [{"text": p, "warning": note} for p in raw_paras]

        sections.append({
            "title": full_title,
            "paras": paras_with_notes,
            "warning": None
        })
    return sections


def get_or_create_sheet(sheets_service, spreadsheet_id, target_title):
    meta = sheets_service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    for s in meta.get("sheets", []):
        if s["properties"]["title"] == target_title:
            return s["properties"]["sheetId"]

    # Not found, create new sheet tab
    res = sheets_service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"requests": [{"addSheet": {"properties": {"title": target_title}}}]}
    ).execute()
    new_sheet_id = res["replies"][0]["addSheet"]["properties"]["sheetId"]
    print(f"[+] Created new sheet tab: '{target_title}' (sheetId={new_sheet_id})")
    return new_sheet_id


def populate_sheet_chapter(sheets_service, spreadsheet_id, sheet_id, sheet_title, sections):
    print(f"\n[*] Populating '{sheet_title}' (sheetId={sheet_id}, {len(sections)} sections)...")

    # SAFETY CHECK: Check if teammates have already typed anything in Column C!
    try:
        existing_c = sheets_service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=f"'{sheet_title}'!C1:C1000"
        ).execute().get("values", [])
        user_edits = [row[0] for row in existing_c if row and row[0].strip() and row[0].strip() != "Paraphrased:"]
        if user_edits:
            print(f"[CAUTION] Found {len(user_edits)} existing teammate edits in Column C of '{sheet_title}'!")
            print(f"[BLOCKED] Preserving teammate work. Refusing to run bulk wipe on '{sheet_title}'.")
            return
    except Exception as e:
        print(f"[!] Warning checking existing Column C: {e}")

    # Reset any leftover manual formatting or ghost highlights on the sheet (ONLY IF SAFE)
    try:
        sheets_service.spreadsheets().values().clear(spreadsheetId=spreadsheet_id, range=f"'{sheet_title}'!A1:Z1000", body={}).execute()
        sheets_service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"requests": [{"updateCells": {"range": {"sheetId": sheet_id}, "fields": "userEnteredFormat"}}]}
        ).execute()
    except Exception:
        pass

    # Build 2D grid of values
    rows_values = []
    # Row formatting requests
    format_requests = []

    # Current 0-indexed row in the sheet
    curr_row = 0

    for sec in sections:
        title = sec["title"]
        paras = sec["paras"]
        warning = sec["warning"]

        # 1. Section Title Row
        # Col A: "", Col B: title, Col C: ""
        rows_values.append(["", title, ""])
        # Format Section Title: Bold, 12pt
        format_requests.append({
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": curr_row,
                    "endRowIndex": curr_row + 1,
                    "startColumnIndex": 1,
                    "endColumnIndex": 2
                },
                "cell": {
                    "userEnteredFormat": {
                        "textFormat": {"bold": True, "fontSize": 12},
                        "verticalAlignment": "MIDDLE"
                    }
                },
                "fields": "userEnteredFormat(textFormat,verticalAlignment)"
            }
        })
        curr_row += 1

        if not paras:
            # Empty separator row after parent header
            rows_values.append(["", "", ""])
            curr_row += 1
            continue

        for p_idx, p_item in enumerate(paras):
            if isinstance(p_item, dict):
                p_text = p_item["text"]
                p_warning = p_item.get("warning") or warning
            elif isinstance(p_item, tuple):
                p_text, p_warning = p_item
            else:
                p_text = p_item
                p_warning = warning

            # 2. Header Row: Col B (Original - Red), Col C (Paraphrased - Green)
            orig_header = f"Original (Do Paraphrase — {p_warning})" if p_warning else "Original (Do Paraphrase)"
            rows_values.append(["", orig_header, "Paraphrased:"])

            # Format Header Row
            # Red on Col B
            format_requests.append({
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": curr_row,
                        "endRowIndex": curr_row + 1,
                        "startColumnIndex": 1,
                        "endColumnIndex": 2
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "backgroundColor": {"red": 1.0, "green": 0.0, "blue": 0.0},
                            "textFormat": {"bold": True, "fontSize": 10, "foregroundColor": {"red": 0.0, "green": 0.0, "blue": 0.0}},
                            "verticalAlignment": "MIDDLE"
                        }
                    },
                    "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment)"
                }
            })
            # Green on Col C
            format_requests.append({
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": curr_row,
                        "endRowIndex": curr_row + 1,
                        "startColumnIndex": 2,
                        "endColumnIndex": 3
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "backgroundColor": {"red": 0.0, "green": 1.0, "blue": 0.0},
                            "textFormat": {"bold": True, "fontSize": 10, "foregroundColor": {"red": 0.0, "green": 0.0, "blue": 0.0}},
                            "verticalAlignment": "MIDDLE"
                        }
                    },
                    "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment)"
                }
            })
            curr_row += 1

            # 3. Content Row: Col B (Original text), Col C (empty for teammate typing)
            rows_values.append(["", p_text, ""])

            # Format Content Row: Wrap text, 10pt, top vertical alignment
            format_requests.append({
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": curr_row,
                        "endRowIndex": curr_row + 1,
                        "startColumnIndex": 1,
                        "endColumnIndex": 3
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "wrapStrategy": "WRAP",
                            "textFormat": {"bold": False, "fontSize": 10},
                            "verticalAlignment": "TOP"
                        }
                    },
                    "fields": "userEnteredFormat(wrapStrategy,textFormat,verticalAlignment)"
                }
            })
            curr_row += 1

            # 4. Blank spacer row
            rows_values.append(["", "", ""])
            curr_row += 1

    # Update Values in Sheet
    sheets_service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=f"'{sheet_title}'!A1:C{len(rows_values)}",
        valueInputOption="USER_ENTERED",
        body={"values": rows_values}
    ).execute()
    print(f"[OK] Injected {len(rows_values)} rows into '{sheet_title}'!")

    # Set Column Dimensions: Col A = 40px, Col B = 560px, Col C = 560px + Permanent WRAP
    col_width_reqs = [
        {
            "updateDimensionProperties": {
                "range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": 0, "endIndex": 1},
                "properties": {"pixelSize": 40},
                "fields": "pixelSize"
            }
        },
        {
            "updateDimensionProperties": {
                "range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": 1, "endIndex": 2},
                "properties": {"pixelSize": 560},
                "fields": "pixelSize"
            }
        },
        {
            "updateDimensionProperties": {
                "range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": 2, "endIndex": 3},
                "properties": {"pixelSize": 560},
                "fields": "pixelSize"
            }
        },
        {
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startColumnIndex": 1,
                    "endColumnIndex": 3
                },
                "cell": {
                    "userEnteredFormat": {
                        "wrapStrategy": "WRAP",
                        "verticalAlignment": "TOP"
                    }
                },
                "fields": "userEnteredFormat(wrapStrategy,verticalAlignment)"
            }
        }
    ]

    all_batch_reqs = col_width_reqs + format_requests

    # Execute formatting in chunks of 200
    chunk_size = 200
    for i in range(0, len(all_batch_reqs), chunk_size):
        chunk = all_batch_reqs[i:i + chunk_size]
        sheets_service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"requests": chunk}
        ).execute()

    print(f"[SUCCESS] Formatting, colors, text-wrapping, and column widths applied to '{sheet_title}'!")


def main():
    parser = argparse.ArgumentParser(description="Upload thesis sections to Google Sheets")
    parser.add_argument("--url", type=str, help="Existing Google Sheet URL")
    parser.add_argument("--id", type=str, help="Existing Google Sheet ID")
    parser.add_argument("--chapter", type=str, default="all", choices=["all", "1", "2", "3", "2,3"], help="Chapter(s) to populate")
    args = parser.parse_args()

    creds = get_credentials()
    sheets_service = build("sheets", "v4", credentials=creds)

    spreadsheet_id = None
    if args.id:
        spreadsheet_id = args.id
    elif args.url:
        m = re.search(r'/spreadsheets/d/([a-zA-Z0-9-_]+)', args.url)
        if m:
            spreadsheet_id = m.group(1)
        else:
            spreadsheet_id = args.url
    else:
        spreadsheet_id = "14UIi22gtPx_ogVGBPfhV45rN1R-zTREAQG3Aymcqk0A"

    chapters_to_run = []
    if args.chapter == "all":
        chapters_to_run = ["1", "2", "3"]
    elif args.chapter == "2,3":
        chapters_to_run = ["2", "3"]
    else:
        chapters_to_run = [args.chapter]

    print(f"=== POPULATING CHAPTERS {chapters_to_run} IN GOOGLE SHEET ({spreadsheet_id}) ===")

    if "1" in chapters_to_run:
        ch1_title = "Chapter 1: Introduction"
        ch1_id = get_or_create_sheet(sheets_service, spreadsheet_id, ch1_title)
        ch1_sections = get_chapter1_data()
        populate_sheet_chapter(sheets_service, spreadsheet_id, ch1_id, ch1_title, ch1_sections)

    if "2" in chapters_to_run:
        ch2_title = "Chapter 2: Literature Review"
        ch2_id = get_or_create_sheet(sheets_service, spreadsheet_id, ch2_title)
        ch2_sections = get_chapter2_data()
        populate_sheet_chapter(sheets_service, spreadsheet_id, ch2_id, ch2_title, ch2_sections)

    if "3" in chapters_to_run:
        ch3_title = "Chapter 3: Requirements & Constraints"
        ch3_id = get_or_create_sheet(sheets_service, spreadsheet_id, ch3_title)
        ch3_sections = get_chapter3_data()
        populate_sheet_chapter(sheets_service, spreadsheet_id, ch3_id, ch3_title, ch3_sections)

    print("\n" + "=" * 60)
    print("🏆 ALL REQUESTED CHAPTERS POPULATED IN GOOGLE SHEETS!")
    print(f" -> URL: https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit")
    print("=" * 60)


if __name__ == "__main__":
    main()
