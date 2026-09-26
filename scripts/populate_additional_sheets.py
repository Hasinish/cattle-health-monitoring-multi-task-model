import sys
import os

sys.path.insert(0, ".")
from scripts.sheet_tool import get_service, SPREADSHEET_ID

def get_or_create_sheet(service, spreadsheet_id, title):
    sheet_metadata = service.get(spreadsheetId=spreadsheet_id).execute()
    sheets = sheet_metadata.get('sheets', '')
    for s in sheets:
        if s['properties']['title'] == title:
            print(f"[FOUND] Sheet '{title}' exists with ID {s['properties']['sheetId']}")
            return s['properties']['sheetId']

    # Create sheet
    req = {
        "addSheet": {
            "properties": {
                "title": title,
                "gridProperties": {
                    "rowCount": 200,
                    "columnCount": 10
                }
            }
        }
    }
    res = service.batchUpdate(spreadsheetId=spreadsheet_id, body={"requests": [req]}).execute()
    new_id = res['replies'][0]['addSheet']['properties']['sheetId']
    print(f"[CREATED] Sheet '{title}' created with ID {new_id}")
    return new_id

# 1. Ethics Statement (3 paragraphs)
ethics_paras = [
    "This research was undertaken solely for academic purposes in partial fulfillment of the requirements for the degree of B.Sc. in Computer Science and Engineering. All data, benchmark corpora, and computational resources consulted during this investigation have been fully, accurately, and transparently cited in the bibliography.",
    "The experimental evaluations strictly utilize secondary, non-invasive, and publicly accessible cattle image and video datasets (ScienceDB, CVB, Kaggle Beef, and SideViewCows2026). No live animals were handled, manipulated, or subjected to invasive experimental protocols by the authors. This study concerns automated perception auditing, multi-task deep representation learning, and benchmark evaluation; it does not establish clinical veterinary effectiveness or authorize autonomous medical treatment decisions.",
    "Those who contributed to this study or facilitated access to research resources are gratefully acknowledged in the Acknowledgement section. Consequently, this work conforms to institutional ethical standards and presents no ethical concerns."
]

# 2. Abstract (Broken down into logical parts)
abstract_paras = [
    "Part 1 (Context & Problem): Continuous cattle monitoring requires evaluating body condition, behavior, and individual identity, yet these tasks demand distinct visual cues: static morphology, temporal dynamics, and fine surface biometrics. Consequently, forcing these heterogeneous tasks into generic hard-shared representations can create task interference and negative transfer.",
    "Part 2 (Methodology & Representation Design): This thesis investigates cattle-centered visual representations—incorporating target localization, binary foreground masks, and lightweight temporal modeling—across single-task and multi-task frameworks.",
    "Part 3 (Leakage-Aware Evaluation Protocols): Leakage-aware protocols were established across all domains: burst-group-disjoint evaluation without verified cow identifiers for ScienceDB Body Condition Scoring (BCS); source- and session-disjoint evaluation for combined barn and feedlot behavior sequences; and open-set retrieval across 69 held-out cows for SideView re-identification (Re-ID).",
    "Part 4 (Matched Single-Task Findings): In matched single-task evaluations, cattle-centered representations provided task-dependent benefits: for BCS, localization and masking reduced mean absolute error from 0.1929 to 0.1709 BCS units and raised tolerance accuracy within ±0.25 from 84.95% to 89.40%; for behavior, foreground-guided temporal convolutions increased balanced accuracy from 71.30% to 74.43% despite a slight overall accuracy decrease (88.46% to 87.44%); and under an oracle-mask condition, Re-ID Snapshot-to-Parlor Rank-1 rose from 38.88% to 62.93% (mAP from 27.05% to 40.42%). Because perception components were modified jointly, gains cannot be attributed to segmentation alone.",
    "Part 5 (Multi-Task Negative Transfer): Under multi-task learning, monolithic hard parameter sharing produced outcome-level degradation relative to matched single-task references across all three tasks: BCS MAE degraded to 0.1788, behavior balanced accuracy dropped to 67.30%, and Re-ID Barn Rank-1 fell to 57.38%.",
    "Part 6 (Architectural & Optimization Interventions): Incorporating task-private residual adapters (+3.32% capacity) produced selective behavior changes but did not consistently recover BCS or Re-ID performance. Applying gradient-projected optimization (PCGrad) with matched capacity confirmed regular gradient interference during training (44,177 conflict projections across 16,140 super-steps) and provided selective mitigation, raising behavior balanced accuracy to 69.15% (Macro-F1 0.7114), though dedicated single-task models remained superior.",
    "Part 7 (Conclusion & Engineering Takeaway): These findings demonstrate that cattle-centered representations provide useful but task-dependent benefits. One hard-shared representation was insufficient for these heterogeneous tasks, task-private adapters alone did not resolve degradation, and conflict-aware optimization offered selective rather than uniform recovery. Unified cattle monitoring therefore requires balancing shared representations with task-specific capacity and conflict-aware optimization.",
    "Part 8 (Keywords): Keywords: cattle monitoring; body condition scoring; behavior recognition; cattle re-identification; cattle-centered representation; multi-task learning; negative transfer."
]

# 3. Acknowledgement (5 paragraphs)
acknowledgement_paras = [
    "First and foremost, all praise to the Almighty for granting us the strength, patience, and perseverance to complete this undergraduate thesis.",
    "We would like to express our deepest gratitude and profound appreciation to our honorable supervisor, Dr. Md. Khalilur Rahman, Professor, Department of Computer Science and Engineering, Brac University, for his invaluable guidance, insightful suggestions, and continuous encouragement throughout this research. His profound expertise in computer vision, robotics, and machine intelligence has been instrumental in shaping the methodology and scientific rigor of this work.",
    "We also extend our sincere thanks to our co-supervisor, Mehedi Hasan Emo, Lecturer, Department of Computer Science and Engineering, Brac University, for his constructive feedback and assistance during our research evaluations.",
    "We are deeply grateful to the Department of Computer Science and Engineering at Brac University for providing us with high-performance computing resources, laboratory facilities, and a supportive academic environment. We also express our sincere appreciation to the authors and institutions who created and publicly released the benchmark datasets utilized in this work, whose commitment to open science made this investigation possible.",
    "Finally, we extend our heartfelt gratitude to our parents, families, and friends for their unconditional love, moral support, and endless prayers throughout our academic journey."
]

# 4. Appendix A: Methodology Evidence (6 narrative paragraphs)
appendix_a_paras = [
    "The manuscript was reviewed against repository commit d55ea2df436f504c3de4d83bcf9db845a6b706e3. This is the latest Git commit inspected for writing; individual experiment artifacts may record an earlier or unknown execution revision. No model was rerun for the thesis rewrite.",
    "ScienceDB contains 37,045 training, 8,481 validation, and 8,040 test images in 3,958, 850, and 845 repaired burst groups. This supports sequence-safe evaluation, not a cow-disjoint claim. The Behavior protocol contains 3,785/680/809 Train/Validation/Test samples grouped by 267 source videos or sessions; Walking is available only in CVB. SideView Protocol A learns from 41 cows and evaluates 69 different cows using a 36,811-image parlor gallery, 25,260 barn queries, and 607 snapshot queries.",
    "Runs 4 and 5 use matched controls because their perception pipelines exclude some canonical samples. Run 4 and its Run 1 control share 7,549 test images. Run 5 and its Run 2 control share 780 test sequences. Run 6 uses all Protocol A inputs because SideView releases paired ground-truth masks, but its result is an oracle condition rather than automatic segmentation.",
    "All reported single-task baselines and multi-task models were evaluated from single trained checkpoints; repeated-seed distributions, confidence intervals, and hypothesis tests were precluded by compute constraints. External cross-dataset stress tests on uncalibrated herds remain unexecuted roadmap items. Re-ID evaluations utilized released ground-truth masks rather than an automated upstream segmentation model. Downstream transfer of anatomical pose estimation and viewpoint priors remains unverified. Final administrative permissions, committee details, and ethics/AI disclosures require institutional confirmation.",
    "Descriptive model names are used in the methodology. The identifiers below retain the connection to the results tables and archived experiment records across all completed single-task and multi-task configurations.",
    "The following table groups the supporting records for Chapter 4. Evidence identifiers refer to the versioned register in this appendix. The grouping replaces repeated record identifiers in the methodology prose without changing the underlying sources."
]

# 5. Appendix B: Evidence Boundaries (12 narrative paragraphs)
appendix_b_paras = [
    "The thesis records deterministic split files, executed implementations, machine-readable metrics, and task-specific coverage. Raw datasets, full checkpoints, and cloud volumes are not embedded in the manuscript. Reproducing the results therefore requires the repository revision, the referenced manifests, access to the corresponding data, and the saved model artifacts.",
    "Run 1 records seed 42 and split hashes. Run 5 records seed 2026, retained-sample hashes, and a bit-identical checkpoint reload. Run 6 also records a bit-identical reload and integrity checks for synchronized RGB/mask crops. Multi-task configurations E1, E3, and E4 used versioned implementations, validation-only checkpoint selection, and identical held-out test populations for direct comparison; E4 additionally records deterministic PCGrad RNG seeds and training diagnostics. Some historical artifacts record an unknown Git revision because Git metadata was unavailable in their execution container; the thesis review SHA is not substituted for those missing execution fields.",
    "Independence units. ScienceDB protects connected bursts rather than biological identities. CVB and Beef protect source videos or sessions rather than verified cows. Only SideView Protocol A supports the stated identity-disjoint learning/evaluation comparison.",
    "Matched coverage. Run 4 applies to 7,549 of 8,040 test images, and Run 5 applies to 780 of 809 test sequences. Their corresponding RGB checkpoints were evaluated on exactly those retained samples. Coverage failures remain part of the system limitation.",
    "Combined configurations. Run 4 changes localization, crop, and mask guidance; Run 5 changes representation and temporal aggregation; Run 6 changes crop geometry and adds a GT mask channel. None of these experiments isolates one component's causal effect.",
    "Oracle Re-ID condition. Run 6 uses SideView ground-truth target masks. It does not demonstrate automatic SAM segmentation, and it does not prove that shortcut learning has been eliminated.",
    "Capacity confounding in modular MTL. The modular task-private adapter architecture (E3) added 395,904 trainable parameters (+3.32% capacity over E1). Performance differences cannot be attributed purely to architectural routing in isolation from parameter scale.",
    "Gradient conflict diagnostics. Direct gradient conflict measurements were recorded exclusively during E4 PCGrad training; opposing gradient directions were directly observed across task pairs, but conflict statistics cannot be retroactively assigned to E1 or E3.",
    "Selective mitigation. PCGrad provided selective mitigation on some metrics (notably behavior recognition), but dedicated single-task models remained superior on several primary metrics; negative transfer was not universally eliminated.",
    "Statistical scope. All reported single-task baselines and multi-task models are single-run point estimates. No confidence interval, repeated-seed standard deviation, p-value, or statistical-significance claim is supplied. External cross-dataset generalization to unseen herds remains unverified.",
    "Future experimental research includes executing external cross-herd stress tests (e.g., on Ruchay, Dryad, MmCows, CBVD-5, and BECA) and multi-seed training runs to quantify parameter variance and confidence intervals.",
    "Administrative review must confirm author order, semester, committee details, ethics and AI-assistance requirements, dataset permissions, acknowledgments, and any separate IEEE-format deliverable or presentation/demonstration obligations. No signature, approval, user study, or institutional exemption is invented."
]

def populate_sheet(service, sheet_id, title, paras):
    rows_values = []
    format_requests = []
    curr_row = 0

    for idx, p_text in enumerate(paras, 1):
        # Header row
        orig_header = "Original (Do Paraphrase)"
        para_header = "Paraphrased:"
        rev_header = "রিভিউ ও ফিডব্যাক 📝 (Review Notes)"
        
        rows_values.append(["", orig_header, para_header, rev_header])
        
        # Col B (Red), Col C (Green), Col D (Blue)
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
                        "backgroundColor": {"red": 0.95, "green": 0.25, "blue": 0.25},
                        "textFormat": {"bold": True, "fontSize": 10, "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}},
                        "verticalAlignment": "MIDDLE"
                    }
                },
                "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment)"
            }
        })
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
                        "backgroundColor": {"red": 0.2, "green": 0.75, "blue": 0.35},
                        "textFormat": {"bold": True, "fontSize": 10, "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}},
                        "verticalAlignment": "MIDDLE"
                    }
                },
                "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment)"
            }
        })
        format_requests.append({
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": curr_row,
                    "endRowIndex": curr_row + 1,
                    "startColumnIndex": 3,
                    "endColumnIndex": 4
                },
                "cell": {
                    "userEnteredFormat": {
                        "backgroundColor": {"red": 0.26, "green": 0.52, "blue": 0.96},
                        "textFormat": {"bold": True, "fontSize": 10, "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}},
                        "verticalAlignment": "MIDDLE"
                    }
                },
                "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment)"
            }
        })
        curr_row += 1

        # Content row: Col B = Original, Col C = Empty, Col D = Empty
        rows_values.append(["", p_text, "", ""])
        format_requests.append({
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": curr_row,
                    "endRowIndex": curr_row + 1,
                    "startColumnIndex": 1,
                    "endColumnIndex": 4
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

        # Spacer row
        rows_values.append(["", "", "", ""])
        curr_row += 1

    # Clear existing sheet content
    service.values().clear(spreadsheetId=SPREADSHEET_ID, range=f"'{title}'!A1:Z500").execute()

    # Upload values
    service.values().update(
        spreadsheetId=SPREADSHEET_ID,
        range=f"'{title}'!A1:D{len(rows_values)}",
        valueInputOption="USER_ENTERED",
        body={"values": rows_values}
    ).execute()
    print(f"[OK] Injected {len(rows_values)} rows ({len(paras)} separate paragraphs) into '{title}'")

    # Column dimensions
    dim_requests = [
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
            "updateDimensionProperties": {
                "range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": 3, "endIndex": 4},
                "properties": {"pixelSize": 420},
                "fields": "pixelSize"
            }
        }
    ]

    all_reqs = dim_requests + format_requests
    service.batchUpdate(spreadsheetId=SPREADSHEET_ID, body={"requests": all_reqs}).execute()
    print(f"[SUCCESS] Formatting & column dimensions applied to '{title}'!")

def main():
    service = get_service()

    sheets_to_create = [
        ("Ethics Statement", ethics_paras),
        ("Abstract", abstract_paras),
        ("Acknowledgement", acknowledgement_paras),
        ("Appendix A", appendix_a_paras),
        ("Appendix B", appendix_b_paras)
    ]

    for title, paras in sheets_to_create:
        sheet_id = get_or_create_sheet(service, SPREADSHEET_ID, title)
        populate_sheet(service, sheet_id, title, paras)

    print("\nALL 5 ADDITIONAL SHEETS CREATED AND POPULATED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
