"""
scripts/apply_super_smart_paraphrase_workbench.py
=================================================
Master Paraphrase Workbench Generator with 100% Verbatim Column B Term Extraction.
Guarantees that EVERY single term in '📌 Words/phrases to keep intact:' is an exact,
literal substring that ACTUALLY EXISTS in the original paragraph text in Column B!
Applies native Google Sheets rich text bolding (textFormatRuns) with zero asterisks.
"""

import sys
import re
import socket
import time

socket.setdefaulttimeout(25.0)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, ".")
from scripts.sheet_tool import get_service, SPREADSHEET_ID, SHEET_IDS, robust_execute
from scripts.format_all_chapters_comprehensive import FLAGGED_BY_TAB

CANDIDATE_TERMS = [
    # Multi-word anatomical & task sequences
    "feeding, drinking, standing, lying, and walking",
    "feeding, drinking, standing, lying, walking",
    "Standing, lying, or feeding",
    "standing, lying, and walking",
    "standing, lying, walking",
    "pelvis, loin, spine, and tailhead",
    "pelvis, loin, spine, tailhead",
    "tailhead, vertebrae, loin, and pelvis",
    "tailhead, loin, and pelvis",
    "Body Condition Scoring (BCS)",
    "Body Condition Scoring",
    "body condition scoring",
    "body condition score",
    "body condition",
    "Multi-task learning (MTL)",
    "Multi-task learning",
    "multi-task learning",
    "multi-task deep learning",
    "multi-task model",
    "multi-task framework",
    "multi-task",
    "Individual identification",
    "individual identification",
    "individual cow identification",
    "cow identification",
    "re-identification",
    "Re-identification",
    "Re-ID",
    "re-id",
    "Behavior recognition",
    "behavior recognition",
    "behavior detection",
    "behavior classification",
    "cattle behavior",
    "Precision livestock farming",
    "precision livestock farming",
    "precision livestock husbandry",
    "precision livestock",
    "shortcut learning",
    "Shortcut learning",
    "spurious correlations",
    "spurious correlation",
    "data leakage",
    "Data leakage",
    "leakage resistance",
    "leakage",
    "train-test split",
    "train-test splits",
    "test data",
    "validation data",
    "held-out test data",
    "held-out test set",
    "held-out test",
    "held-out",
    "cow-disjoint",
    "group-disjoint",
    "burst-group",
    "unseen-cow",
    "unseen cow",
    "unseen cows",
    "unseen cattle",
    "memorized backgrounds",
    "stanchion occlusions",
    "stanchion barriers",
    "viewpoint shifts",
    "varying cow appearances",
    
    # Models & Architectures
    "Convolutional neural networks",
    "convolutional neural networks",
    "convolutional network",
    "convolutional networks",
    "deep neural networks",
    "Deep neural networks",
    "neural networks",
    "deep learning",
    "Deep learning",
    "transfer learning",
    "Transfer learning",
    "representation learning",
    "Representation learning",
    "visual representation",
    "visual representations",
    "visual features",
    "Visual features",
    "feature representation",
    "feature representations",
    "skip connections",
    "residual connections",
    "prediction head",
    "prediction heads",
    "shared representations",
    "shared representation",
    "shared model components",
    "shared features",
    "task-specific",
    "task-private",
    "hard sharing",
    "hard parameter sharing",
    "selective sharing",
    "negative transfer",
    "Negative transfer",
    "gradient conflict",
    "gradient interference",
    "conflicting gradients",
    "conflicting information",
    "loss weighting",
    "combined loss",
    
    # Specific tool / model names
    "RT-DETR-L",
    "RT-DETR",
    "SAM 2.1",
    "Segment Anything",
    "SAM",
    "SuperAnimal",
    "DeepLabCut",
    "ResNet-18",
    "ResNet-50",
    "ResNet",
    "EfficientNet",
    "ConvNeXt",
    "Vision Transformer",
    "ViT",
    "TCN",
    "ST-GCN",
    "CBAM",
    "CATR",
    "Cross-Stitch",
    "PCGrad",
    "GradNorm",
    "CORAL",
    "AdamW",
    
    # Datasets
    "ScienceDB",
    "SideViewCows2026",
    "CVB",
    "Kaggle Beef",
    "CBVD-5",
    "CBVD",
    "MmCows",
    "BECA",
    "CattleEyeView",
    "OpenCows2020",
    "ImageNet",
    "COCO",
    "Figshare",
    "Zenodo",
    "GitHub",
    
    # Environment & Perception cues
    "CCTV surveillance footage",
    "surveillance footage",
    "surveillance video",
    "surveillance cameras",
    "surveillance",
    "CCTV",
    "video sequence",
    "video sequences",
    "video frames",
    "repeated frames",
    "temporal correlation",
    "temporal aggregation",
    "temporal information",
    "temporal activity",
    "temporal models",
    "temporal detail",
    "temporal inputs",
    "optical flow",
    "Optical flow",
    "bounding box",
    "Bounding box",
    "localization",
    "Localization",
    "segmentation",
    "Segmentation",
    "foreground masks",
    "foreground mask",
    "soft mask",
    "soft masks",
    "pose estimation",
    "Pose estimation",
    "keypoints",
    "Keypoints",
    "anatomical landmarks",
    "Anatomical landmarks",
    "landmarks",
    "camera viewpoint",
    "Camera viewpoint",
    "viewpoint",
    "side-view",
    "rear-view",
    "top-down",
    "occlusion",
    "Occlusion",
    "stanchion",
    "stanchions",
    "barn / pen",
    "barn",
    "Barn",
    "pen",
    "housing and management",
    "animal welfare",
    "Animal welfare",
    "body shape",
    "morphology",
    "Morphology",
    "posture and movement",
    "posture changes over time",
    "posture",
    "locomotion",
    "coat pattern",
    "coat patterns",
    "coat markings",
    "biometric pattern",
    "subcutaneous fat",
    "body reserves",
    "energy reserves",
    "pelvis",
    "loin",
    "spine",
    "tailhead",
    "feeding",
    "drinking",
    "standing",
    "lying",
    "walking",
    "rumination",
    
    # Systems, Hardware & Governance terms
    "offline feature caching",
    "feature caching",
    "GPU resources",
    "academic compute budgets",
    "GPU compute hours",
    "GPU compute",
    "GPU hours",
    "Modal cloud",
    "Modal",
    "Tesla T4",
    "T4",
    "L40S",
    "local laptop",
    "reproducibility",
    "Reproducibility",
    "deterministic manifests",
    "versioned code",
    "checkpoints",
    "Git versioning",
    "decision support",
    "decision-making",
    "non-invasive",
    
    # Chapter 4 Methodology & Architecture Terms
    "ImageNet-pretrained ResNet-18",
    "ImageNet-initialized ResNet-18",
    "cumulative threshold logits",
    "cumulative ordinal logits",
    "cumulative logits",
    "ordinal binary cross-entropy",
    "binary cross-entropy",
    "cross-entropy loss",
    "categorical cross-entropy",
    "ordinal_bce",
    "binary mask channel",
    "oracle mask channel",
    "binary mask",
    "binary masks",
    "four-channel ResNet-18",
    "target-tracklet",
    "Conv1D",
    "residual Conv1D",
    "width-three Conv1D",
    "temporal convolution",
    "Protocol A",
    "SideView Protocol A",
    "cosine similarity",
    "retrieval embedding",
    "gallery retrieval",
    "Rank-k",
    "mean average precision",
    "mAP",
    "target-centered cropping",
    "biological cow IDs",
    "connected burst groups",
    "burst-group-disjoint",
    "passage-disjoint",
    "sequence-safe",
    "unseen-cow evaluation",
    "monolithic hard-shared control",
    "monolithic control",
    "hard-shared control",
    "hard-shared",
    "hard parameter sharing",
    "modular task-private architecture",
    "task-private residual adapters",
    "residual bottleneck adapter",
    "bottleneck adapter",
    "residual adapters",
    "task-private adapter",
    "task-private adapters",
    "task-private parameters",
    "Identity Initialization",
    "Projecting Conflicting Gradients",
    "gradient conflict",
    "pairwise conflict",
    "pairwise projections",
    "multi-task super-step",
    "super-steps per epoch",
    "super-steps",
    "super-step",
    "interleaved mini-batch",
    "equal task weighting",
    "gradient isolation",
    "held-out test populations",
    "validation checkpoint selection",
    "checkpoint selection",
    "validation loss",
    "Behavior evaluation",
    "overall accuracy",
    "balanced accuracy",
    "Macro-F1",
    "gradient vector",
    "opposing directions",
    "task adapter",
    "zero weights",
    "adapter contribution",
    "batch loss",
    "logit",
    "non-functional requirements",
    "evaluation protocols",
    "evaluation design",
    "Animal-centered evaluation splits",
    "randomly split dataset",
    "random splits",
    "strict evaluation protocols",
    "second research gap",
    "research gap",
    "literature review",
    "proposed methodology",
    "experimental results",
    "design requirements",
    "practical constraints",
    "farm environment",
    "Camera placement",
    "standard farm video equipment",
    "specialized sensors",
    "public research datasets",
    "private farm footage",
    "blurring or exclusion",
    "ethical practice",
    "livestock management",
    "Animal health",
    "visual system",
    "real farm routines",
    "Dataset reuse",
    "dataset registry",
    "non-commercial academic research",
    "Author contributions",
    "Dr. Md. Khalilur Rahman",
    "BRAC University",
    "submission details",
    "scientific and operational risks",
    "limited compute resources",
    "local development time",
    "economic deployment analysis",
    "acquisition conditions",
    "external datasets",
    "model comparisons",
    "monitoring cattle",
    "commercially sensitive information",
    "commercial use",
    "licensing conditions",
    "field deployment",
    "research cost",
    "farm-deployment business case",
    "accelerators",
    "preprocessing steps",
    "storage needs",
    "CSE400",
    "IEEE",
    "ISO"
]

def is_section_title(text):
    t = text.strip()
    if re.match(r'^\d+\.\d+', t) or t.startswith('[Table Reference') or t.startswith('Chapter '):
        return True
    return False

def extract_verbatim_intact_terms(col_b):
    """
    Extracts terms that are 100% GUARANTEED to exist as exact literal substrings inside col_b.
    """
    if not col_b or len(col_b) <= 35:
        return []
    if is_section_title(col_b) or col_b.startswith("Original (") or col_b.startswith("["):
        return []
        
    found_matches = []
    col_b_lower = col_b.lower()
    
    # 1. Author citations (e.g. 'Sani et al.', 'Lee et al.', 'Ferguson et al.')
    citations = re.findall(r'\b[A-Z][a-zA-Z\s\-]+ et al\.', col_b)
    for cite in citations:
        if cite not in found_matches:
            found_matches.append(cite)
            
    # 2. Match candidate dictionary against col_b
    sorted_candidates = sorted(CANDIDATE_TERMS, key=len, reverse=True)
    for cand in sorted_candidates:
        pattern = r'\b' + re.escape(cand.lower()) + r'\b'
        match = re.search(pattern, col_b_lower)
        if match:
            start, end = match.span()
            actual_casing = col_b[start:end]
            if not any(actual_casing.lower() in m.lower() for m in found_matches):
                found_matches.append(actual_casing)
                
    # 3. Capitalized technical acronyms
    acronyms = re.findall(r'\b[A-Z]{2,}(?:-[A-Za-z0-9]+)?\b', col_b)
    for acr in acronyms:
        if acr not in ["AND", "THE", "FOR", "NOT", "BUT", "ALL", "ARE", "WAS", "THIS", "WITH", "THAT"]:
            if not any(acr.lower() in m.lower() for m in found_matches):
                found_matches.append(acr)
                
    # Fallback if no candidate terms matched: extract significant words directly from col_b
    if not found_matches:
        words = re.findall(r'\b[A-Za-z]{6,}\b', col_b)
        stopwords = {"because", "therefore", "instead", "between", "several", "another", "through", "without", "during", "before", "across", "should", "further"}
        candidates = [w for w in words if w.lower() not in stopwords]
        if candidates:
            found_matches = candidates[:3]
        else:
            found_matches = re.findall(r'\b[A-Za-z]{4,}\b', col_b)[:2]

    # Sort matches by order of appearance in original text
    found_matches.sort(key=lambda m: col_b_lower.find(m.lower()))
    
    # Take top 3 to 5 terms
    final_terms = found_matches[:5]
    
    # Strict validation: EVERY term MUST be a substring of col_b!
    for t in final_terms:
        assert t.lower() in col_b_lower, f"CRITICAL BUG: '{t}' not in col_b!"
        
    return final_terms

def parse_markdown_bold(md_text):
    clean_chars = []
    runs = []
    in_bold = False
    utf16_idx = 0
    
    i = 0
    while i < len(md_text):
        if md_text[i:i+2] == '**':
            in_bold = not in_bold
            runs.append({
                "startIndex": utf16_idx,
                "format": {"bold": in_bold}
            })
            i += 2
        else:
            char = md_text[i]
            clean_chars.append(char)
            utf16_units = len(char.encode('utf-16-le')) // 2
            utf16_idx += utf16_units
            i += 1

    clean_text = "".join(clean_chars)
    
    if in_bold:
        runs.append({
            "startIndex": utf16_idx,
            "format": {"bold": False}
        })
        
    filtered_runs = []
    current_bold = False
    for r in runs:
        if r["format"]["bold"] != current_bold:
            filtered_runs.append(r)
            current_bold = r["format"]["bold"]
            
    return clean_text, filtered_runs

def build_markdown_for_row(tab_name, row_num, col_b, col_c):
    flagged_dict = FLAGGED_BY_TAB.get(tab_name, {})
    
    # Extract 100% verbatim terms from col_b
    verbatim_terms = extract_verbatim_intact_terms(col_b)
    intact_str = ", ".join(verbatim_terms) if verbatim_terms else "Task-specific terminology and model names"
    
    # 1. Flagged Row
    if row_num in flagged_dict:
        item = flagged_dict[row_num]
        
        fixes_lines = []
        for idx, f in enumerate(item["fixes"], start=1):
            clean_f = re.sub(r'^Fix\s*\d+:\s*', '', f)
            fixes_lines.append(f"• **Fix {idx}:** {clean_f}")
        fixes_str = "\n".join(fixes_lines)
        
        return (
            f"**📌 Words/phrases to keep intact:**\n• {intact_str}\n\n"
            f"**🛠️ Required Fixes:**\n{fixes_str}"
        )
        
    # 2. Approved or Pending Academic Paragraph
    if col_c and col_c.strip() and col_c.strip() not in ["Paraphrased:", "LEAVE BLANK (Do Not Paraphrase):"]:
        return (
            f"**✅ অর্থ ঠিক রাখা হইছে।**\n\n"
            f"**📌 Words/phrases to keep intact:**\n• {intact_str}"
        )
    else:
        return (
            f"**📌 Words/phrases to keep intact:**\n• {intact_str}"
        )

def process_chapter(service, tab_name, sheet_id):
    print(f"\n========================================================")
    print(f"[*] Upgrading '{tab_name}' (sheetId: {sheet_id})...")
    
    res = robust_execute(lambda: service.values().get(
        spreadsheetId=SPREADSHEET_ID,
        range=f"'{tab_name}'!A1:D350"
    ).execute())
    
    rows = res.get("values", [])
    print(f"[*] Fetched {len(rows)} rows.")
    
    requests = []
    paras_count = 0
    non_paras_count = 0
    titles_count = 0
    headers_count = 0
    flagged_count = 0
    
    for idx, r in enumerate(rows, start=1):
        row_idx = idx - 1
        col_b = r[1].strip() if len(r) > 1 else ""
        col_c = r[2].strip() if len(r) > 2 else ""
        
        # A. Section Title Row
        if is_section_title(col_b):
            titles_count += 1
            # Clear Col D
            requests.append({
                "updateCells": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": row_idx,
                        "endRowIndex": row_idx + 1,
                        "startColumnIndex": 3,
                        "endColumnIndex": 4
                    },
                    "rows": [
                        {
                            "values": [
                                {
                                    "userEnteredValue": {"stringValue": ""},
                                    "textFormatRuns": []
                                }
                            ]
                        }
                    ],
                    "fields": "userEnteredValue,textFormatRuns"
                }
            })
            # Bold Col B
            requests.append({
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": row_idx,
                        "endRowIndex": row_idx + 1,
                        "startColumnIndex": 1,
                        "endColumnIndex": 2
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "textFormat": {"bold": True, "fontSize": 11}
                        }
                    },
                    "fields": "userEnteredFormat.textFormat(bold,fontSize)"
                }
            })
            
        # B. DONT PARAPHRASE / Non-Paragraph Content Rows (e.g. formulas, table references)
        elif any(term in col_b for term in ["DONT PARAPHRASE", "Table Reference:", "[Table Reference", "[Formula:"]):
            non_paras_count += 1
            is_header = col_b.startswith("Original (DONT")
            cell_val = "LEAVE BLANK (Do Not Paraphrase)" if is_header else "🚫 DO NOT PARAPHRASE — Raw formula / table reference (Keep As-Is)"
            
            requests.append({
                "updateCells": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": row_idx,
                        "endRowIndex": row_idx + 1,
                        "startColumnIndex": 3,
                        "endColumnIndex": 4
                    },
                    "rows": [
                        {
                            "values": [
                                {
                                    "userEnteredValue": {"stringValue": cell_val},
                                    "textFormatRuns": [
                                        {"startIndex": 0, "format": {"bold": True}}
                                    ],
                                    "userEnteredFormat": {
                                        "textFormat": {"bold": True, "fontSize": 9},
                                        "backgroundColor": {"red": 0.95, "green": 0.95, "blue": 0.95},
                                        "verticalAlignment": "MIDDLE",
                                        "wrapStrategy": "WRAP"
                                    }
                                }
                            ]
                        }
                    ],
                    "fields": "userEnteredValue,textFormatRuns,userEnteredFormat(textFormat,backgroundColor,verticalAlignment,wrapStrategy)"
                }
            })
            
            # If formula content row, give Col C light grey background
            if not is_header and "[Formula:" in col_b:
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
                                "backgroundColor": {"red": 0.95, "green": 0.95, "blue": 0.95}
                            }
                        },
                        "fields": "userEnteredFormat.backgroundColor"
                    }
                })
            
        # C. Table Header Row ('Original (Do Paraphrase...')
        elif col_b.startswith("Original (Do Paraphrase"):
            headers_count += 1
            requests.append({
                "updateCells": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": row_idx,
                        "endRowIndex": row_idx + 1,
                        "startColumnIndex": 3,
                        "endColumnIndex": 4
                    },
                    "rows": [
                        {
                            "values": [
                                {
                                    "userEnteredValue": {"stringValue": "রিভিউ ও ফিডব্যাক 📝 (Review Notes)"},
                                    "textFormatRuns": [
                                        {"startIndex": 0, "format": {"bold": True}}
                                    ],
                                    "userEnteredFormat": {
                                        "textFormat": {"bold": True, "fontSize": 10},
                                        "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9},
                                        "verticalAlignment": "MIDDLE",
                                        "wrapStrategy": "WRAP"
                                    }
                                }
                            ]
                        }
                    ],
                    "fields": "userEnteredValue,textFormatRuns,userEnteredFormat(textFormat,backgroundColor,verticalAlignment,wrapStrategy)"
                }
            })
            
        # D. Genuine Academic Paragraph Row
        elif len(col_b) > 35:
            paras_count += 1
            md_text = build_markdown_for_row(tab_name, idx, col_b, col_c)
            clean_text, runs = parse_markdown_bold(md_text)
            
            requests.append({
                "updateCells": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": row_idx,
                        "endRowIndex": row_idx + 1,
                        "startColumnIndex": 3,
                        "endColumnIndex": 4
                    },
                    "rows": [
                        {
                            "values": [
                                {
                                    "userEnteredValue": {"stringValue": clean_text},
                                    "textFormatRuns": runs,
                                    "userEnteredFormat": {
                                        "wrapStrategy": "WRAP",
                                        "verticalAlignment": "TOP",
                                        "textFormat": {"fontSize": 10}
                                    }
                                }
                            ]
                        }
                    ],
                    "fields": "userEnteredValue,textFormatRuns,userEnteredFormat(wrapStrategy,verticalAlignment,textFormat)"
                }
            })
            
            # If flagged -> pastel yellow on Col C
            if idx in FLAGGED_BY_TAB.get(tab_name, {}):
                flagged_count += 1
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

    print(f"[*] Prepared {len(requests)} atomic requests: {paras_count} paras, {titles_count} titles, {headers_count} headers, {non_paras_count} non-paras, {flagged_count} flagged.")
    
    if requests:
        start_t = time.time()
        res = robust_execute(lambda: service.batchUpdate(
            spreadsheetId=SPREADSHEET_ID,
            body={"requests": requests}
        ).execute())
        elapsed = time.time() - start_t
        print(f"[SUCCESS] Updated '{tab_name}' in {elapsed:.2f}s!")

def main():
    service = get_service()
    total_start = time.time()
    
    for tab_name, sheet_id in SHEET_IDS.items():
        process_chapter(service, tab_name, sheet_id)
        
    print(f"\n🎉 ALL 4 CHAPTERS SUCCESSFULLY UPGRADED WITH 100% VERBATIM COLUMN B KEYWORDS in {time.time() - total_start:.2f}s!")

if __name__ == "__main__":
    main()
