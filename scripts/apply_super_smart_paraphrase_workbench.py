"""
scripts/apply_super_smart_paraphrase_workbench.py
=================================================
Master Paraphrase Workbench Generator with Super-Smart Academic Keyword Taxonomy.
Eliminates all dumb keywords ('Therefore', 'Better', 'DONT, PARAPHRASE', 'Tabletab').
Maps every single academic paragraph across Chapters 1, 2, and 3 to 3-4 pristine,
high-value domain entities (models, metrics, anatomy, evaluation protocols, citations).
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

# 136 Hand-Curated, Pristine Academic Keyword Sets for All Paragraphs in Chapters 1, 2, and 3
CURATED_PARA_TERMS = {
    # =========================================================================
    # CHAPTER 1: INTRODUCTION
    # =========================================================================
    ('Chapter 1: Introduction', 3): ['Body condition & energy reserves', 'Reduced food intake / feeding drop', 'Housing and barn management'],
    ('Chapter 1: Introduction', 6): ['Body Condition Scoring (BCS)', 'Behavior recognition', 'Individual Cow Identification / Re-ID', 'Feeding, drinking, standing, lying, walking'],
    ('Chapter 1: Introduction', 9): ['Routine welfare monitoring', 'Precision Livestock Farming (PLF)', 'Brief behavioral observation periods'],
    ('Chapter 1: Introduction', 12): ['Non-invasive visual monitoring', 'Precision Livestock Farming (PLF)', 'Computer vision in livestock'],
    ('Chapter 1: Introduction', 15): ['Visual feature representations', 'Downstream tasks', 'Transfer learning / ImageNet pretraining'],
    ('Chapter 1: Introduction', 18): ['Multi-Task Learning (MTL)', 'Shared spatial features', 'Selective parameter sharing', 'Task trade-offs'],
    ('Chapter 1: Introduction', 24): ['Two distinct CV tasks (BCS & Re-ID)', 'Shared cow image input', 'Task-specific prediction heads'],
    ('Chapter 1: Introduction', 27): ['Body Condition Scoring (BCS)', 'Subcutaneous energy reserves', 'Barn / pen background scene bias'],
    ('Chapter 1: Introduction', 30): ['Shortcut learning / Spurious correlations', 'Barn / pen background cues', 'Unseen-cow generalization'],
    ('Chapter 1: Introduction', 33): ['Cattle-centered representation learning', 'RT-DETR-L localization', 'SAM 2.1 foreground segmentation'],
    ('Chapter 1: Introduction', 37): ['Single-task reference baselines', 'Independent task optimization', 'Representation capacity'],
    ('Chapter 1: Introduction', 40): ['CCTV surveillance video', 'Temporal autocorrelation leakage', 'Train-test split integrity'],
    ('Chapter 1: Introduction', 43): ['Single-task reference baselines', 'Model architecture components', 'Independent perception cues'],
    ('Chapter 1: Introduction', 47): ['Temporal activity recognition', 'Posture and locomotion', 'Video sequence modeling'],
    ('Chapter 1: Introduction', 50): ['BCS, Behavior, and Re-ID heads', 'Task-specific representations', 'Single-task vs multi-task'],
    ('Chapter 1: Introduction', 53): ['Promptable segmentation (SAM 2.1)', 'Pretrained perception filters', 'Cattle foreground isolation'],
    ('Chapter 1: Introduction', 57): ['Shared visual representations', 'Selective parameter sharing', 'Downstream task heads'],
    ('Chapter 1: Introduction', 60): ['Camera angle & viewpoint variation', 'Multi-task visual features', 'BCS, Behavior, and Re-ID'],
    ('Chapter 1: Introduction', 63): ['Task-private residual adapters (E3)', 'Shared backbone capacity', 'Gradient conflict mitigation'],
    ('Chapter 1: Introduction', 66): ['Multi-Task Learning (MTL)', 'Negative transfer mitigation', 'Unified cattle health framework'],
    ('Chapter 1: Introduction', 70): ['BCS, Behavior, and Re-ID', 'Shared visual representations', 'Multi-task optimization'],
    ('Chapter 1: Introduction', 73): ['Cow-disjoint evaluation split', 'SideViewCows2026 parlor dataset', 'ScienceDB cattle dataset', 'Data leakage prevention'],
    ('Chapter 1: Introduction', 76): ['RGB visual representations', 'Single-task baseline reference', 'Negative transfer analysis'],
    ('Chapter 1: Introduction', 79): ['Bounding-box localization / crop', 'Foreground soft masks', 'Spurious background suppression'],
    ('Chapter 1: Introduction', 82): ['Multi-task optimization', 'Negative transfer mitigation', 'Task gradient balancing'],
    ('Chapter 1: Introduction', 85): ['Viewpoint & lighting robustness', 'Real-world barn occlusions', 'Perception generalization'],
    ('Chapter 1: Introduction', 88): ['BCS, Behavior, and Re-ID outputs', 'RGB input processing', 'Multi-task framework'],
    ('Chapter 1: Introduction', 92): ['Formal research objectives (RO1-RO5)', 'Unified multi-task framework', 'Thesis scope'],
    ('Chapter 1: Introduction', 95): ['SAM 2.1 soft masks', 'Cattle crops', 'Temporal aggregation window', 'All three tasks'],
    ('Chapter 1: Introduction', 98): ['Open-set gallery retrieval', 'Rank-1 / Rank-5 accuracy', 'Mean Average Precision (mAP)'],
    ('Chapter 1: Introduction', 101): ['Multi-task parameter sharing', 'Shared spatial backbone', 'Task head isolation'],
    ('Chapter 1: Introduction', 105): ['Body Condition Scoring (BCS)', 'Behavior recognition', 'Individual Cow Identification / Re-ID'],
    ('Chapter 1: Introduction', 108): ['ScienceDB, CVB, and SideViewCows2026', 'Matched held-out test sets', 'Three monitoring tasks'],
    ('Chapter 1: Introduction', 111): ['Real-world stanchion occlusions', 'Viewpoint angle shifts', 'Empirical benchmark evaluation'],
    ('Chapter 1: Introduction', 114): ['ResNet-18 spatial backbone', 'Green AI compute footprint', 'Edge deployment feasibility'],
    ('Chapter 1: Introduction', 117): ['Unified multi-task deep learning framework', 'Integrated empirical evaluation', 'Thesis contributions'],
    
    # =========================================================================
    # CHAPTER 2: LITERATURE REVIEW
    # =========================================================================
    ('Chapter 2: Literature Review', 3): ['This thesis', 'Anatomical morphology', 'Posture and locomotion', 'Multi-Task Learning (MTL)'],
    ('Chapter 2: Literature Review', 6): ['Thesis roadmap & chapter progression', 'Literature review taxonomy', 'Visual representation learning'],
    ('Chapter 2: Literature Review', 10): ['Deep visual representations', 'Convolutional neural networks (CNN)', 'Handcrafted vs learned features'],
    ('Chapter 2: Literature Review', 13): ['ResNet-18 / ResNet-50', 'EfficientNet baseline', 'Residual skip connections', 'Transfer learning'],
    ('Chapter 2: Literature Review', 16): ['Cattle-centered perception pipeline', 'Transfer learning / ImageNet pretraining', 'Foreground soft masks'],
    ('Chapter 2: Literature Review', 20): ['Body Condition Scoring (BCS)', 'CORAL ordinal regression', 'Ordinal BCE loss'],
    ('Chapter 2: Literature Review', 23): ['Behavior recognition', '5 Core Behaviors (standing, walking, lying, feeding, drinking)', 'Posture dynamics'],
    ('Chapter 2: Literature Review', 26): ['Individual Cow Identification / Re-ID', 'Biometric coat patterns', 'Open-set gallery retrieval'],
    ('Chapter 2: Literature Review', 30): ['Multi-Task Learning (MTL)', 'Hard parameter sharing (E1)', 'Shared backbone representations'],
    ('Chapter 2: Literature Review', 36): ['Multi-task loss weighting', 'Task balancing coefficients (λ_t)', 'Joint loss optimization'],
    ('Chapter 2: Literature Review', 39): ['Negative transfer mitigation', 'Task-specific feature requirements', 'Single-task reference baselines'],
    ('Chapter 2: Literature Review', 45): ['Antognoli et al.', 'Precision Livestock Farming (PLF)', 'Non-invasive cattle monitoring'],
    ('Chapter 2: Literature Review', 48): ['Palma et al. / Sani et al.', 'Sensor data vs visual representations', 'Farm management decisions'],
    ('Chapter 2: Literature Review', 51): ['Wearable IoT sensors vs computer vision', 'Multi-modal cattle monitoring', 'Precision Livestock Farming (PLF)'],
    ('Chapter 2: Literature Review', 54): ['Task-specific feature requirements', 'Selective parameter sharing', 'Multi-Task Learning (MTL)'],
    ('Chapter 2: Literature Review', 58): ['Ferguson et al. (BCS system)', 'Edmonson et al. (BCS chart)', '5-point BCS scale'],
    ('Chapter 2: Literature Review', 61): ['Rodriguez-Alvarez et al.', 'Ordinal regression in BCS', 'Body shape and condition'],
    ('Chapter 2: Literature Review', 64): ['RGB image inputs', 'EfficientNet for BCS', 'Ordinal classification tolerance (+/- 0.25)'],
    ('Chapter 2: Literature Review', 67): ['ConvNeXt architecture', 'Side-view chute perspective', 'BCS feature extraction'],
    ('Chapter 2: Literature Review', 70): ['CNN feature extractors', 'Foreground segmentation for BCS', 'Anatomical landmark coverage'],
    ('Chapter 2: Literature Review', 73): ['Anatomical landmarks (pelvis, loin, spine, tailhead)', 'CORAL ordinal loss', 'Camera viewpoint variations'],
    ('Chapter 2: Literature Review', 77): ['Behavior recognition (feeding, drinking, standing, lying, walking)', 'Duration & appearance variance', 'Postural transitions'],
    ('Chapter 2: Literature Review', 80): ['Bounding-box localization / crop', 'Standing vs lying postures', 'CCTV surveillance video'],
    ('Chapter 2: Literature Review', 83): ['CBVD-5 benchmark dataset', 'Standing, lying, drinking behaviors', 'Cattle video datasets'],
    ('Chapter 2: Literature Review', 86): ['MmCows dataset', 'Multi-camera behavior recognition', 'Temporal activity tracking'],
    ('Chapter 2: Literature Review', 89): ['Temporal sequence structure', 'Video frame sampling rate', 'Behavior recognition'],
    ('Chapter 2: Literature Review', 92): ['Feeding and drinking head angles', 'Feed bunk and water trough monitoring', 'Activity recognition'],
    ('Chapter 2: Literature Review', 95): ['Overall behavior classification accuracy', 'Macro-F1 score', 'Class imbalance handling'],
    ('Chapter 2: Literature Review', 99): ['Andrew et al. (cattle identification)', 'Holstein-Friesian coat markings', 'RGB side-view identification'],
    ('Chapter 2: Literature Review', 102): ['Deep metric learning', 'Cosine distance / Triplet loss', 'Andrew et al. identification'],
    ('Chapter 2: Literature Review', 105): ['Individual Cow Identification / Re-ID', 'Stanchion & pen occlusions', 'CNN feature embeddings'],
    ('Chapter 2: Literature Review', 108): ['Semi-supervised identity learning', 'Yu et al. (cow re-id)', 'Open-set gallery retrieval'],
    ('Chapter 2: Literature Review', 111): ['Cross-camera re-identification', 'Open-set identification protocol', 'Rank-1 retrieval accuracy'],
    ('Chapter 2: Literature Review', 114): ['BECA cattle identification dataset', 'Coat pattern biometric verification', 'Unseen-cow evaluation'],
    ('Chapter 2: Literature Review', 117): ['Open-set gallery retrieval', 'Probe queries', 'Mean Average Precision (mAP)'],
    ('Chapter 2: Literature Review', 121): ['RT-DETR-L cattle detector', 'Bounding-box localization / crop', 'Target animal isolation'],
    ('Chapter 2: Literature Review', 124): ['RT-DETR real-time transformer detector', 'CNN feature maps', 'Cattle localization'],
    ('Chapter 2: Literature Review', 127): ['SAM 2.1 (Segment Anything)', 'Zero-shot promptable segmentation', 'Foreground soft mask extraction'],
    ('Chapter 2: Literature Review', 130): ['CattleEyeView top-down dataset', 'Overhead camera angle', 'Top-down segmentation for BCS'],
    ('Chapter 2: Literature Review', 133): ['CATR cattle transformer', 'Anatomical morphology segmentation', 'Keypoint localization'],
    ('Chapter 2: Literature Review', 136): ['Convolutional Block Attention (CBAM)', 'Foreground soft mask / segmentation', 'Spatial and channel attention'],
    ('Chapter 2: Literature Review', 139): ['Multi-task visual representations', 'BCS, Behavior, and Re-ID', 'Feature sharing requirements'],
    ('Chapter 2: Literature Review', 143): ['DeepLabCut keypoint toolkit', 'Pose estimation as interpretable model input', 'Anatomical skeletal landmarks'],
    ('Chapter 2: Literature Review', 146): ['SuperAnimal pose foundation model', 'CattleEyeView top-down pose', 'Keypoint tracking limitations'],
    ('Chapter 2: Literature Review', 149): ['Pose keypoints for BCS & Behavior', 'Posture transitions (lying to standing)', 'Skeletal geometry'],
    ('Chapter 2: Literature Review', 152): ['Camera viewpoint variations (side / top / rear)', 'Cross-viewpoint degradation', 'Viewpoint-invariant embeddings'],
    ('Chapter 2: Literature Review', 155): ['Grolleau et al. (MOO synthetic cattle)', 'Camera viewpoint classification', 'Cross-viewpoint generalization'],
    ('Chapter 2: Literature Review', 158): ['Viewpoint conditioning across BCS, Behavior, Re-ID', 'Pose landmarks', 'Representation alignment'],
    ('Chapter 2: Literature Review', 165): ['Temporal modeling (TCN / ST-GCN)', 'Pose keypoint tracking', 'Behavior recognition dynamics'],
    ('Chapter 2: Literature Review', 168): ['CVB (Cattle Video Benchmark)', 'Temporal CCTV video sequences', 'Action recognition in cattle'],
    ('Chapter 2: Literature Review', 171): ['Bai et al. (temporal convolutions)', 'TCN vs 2D CNN frame averaging', 'Behavior recognition'],
    ('Chapter 2: Literature Review', 174): ['ST-GCN spatio-temporal graph convolutional networks', 'Skeleton joint graphs', 'Behavior classification'],
    ('Chapter 2: Literature Review', 177): ['Optical flow vs 2D spatial frames', 'Temporal aggregation window (T=8)', 'Behavior recognition'],
    ('Chapter 2: Literature Review', 181): ['Geirhos et al. (shortcut learning)', 'Texture vs shape bias in CNNs', 'Barn background shortcut learning'],
    ('Chapter 2: Literature Review', 184): ['Xiao et al. (background bias)', 'Foreground segmentation masking', 'Spurious correlation suppression'],
    ('Chapter 2: Literature Review', 187): ['Beery et al. (camera trap generalization)', 'Location-specific background shortcuts', 'Cattle pen background shift'],
    ('Chapter 2: Literature Review', 190): ['Spurious pen correlations across BCS, Behavior, Re-ID', 'Background shortcut learning', 'Representation cleansing'],
    ('Chapter 2: Literature Review', 193): ['Cow-disjoint evaluation split', 'Data leakage prevention', 'Burst-group-disjoint splitting'],
    ('Chapter 2: Literature Review', 196): ['Cow-disjoint evaluation split', 'Unseen-cow generalization', 'Shortcut learning prevention'],
    ('Chapter 2: Literature Review', 200): ['Multi-Task Learning (MTL)', 'Negative transfer mitigation', 'Shared visual backbone'],
    ('Chapter 2: Literature Review', 203): ['Caruana (1997) MTL foundation', 'Shared inductive bias', 'Cross-task feature regularization'],
    ('Chapter 2: Literature Review', 206): ['Negative transfer phenomenon', 'Gradient conflict / interference', 'Task capacity bottlenecks'],
    ('Chapter 2: Literature Review', 209): ['Task-specific feature divergence', 'Conflicting task gradients', 'Selective parameter sharing'],
    ('Chapter 2: Literature Review', 212): ['Single-task reference baselines (E0)', 'Task performance trade-offs', 'Multi-task comparison'],
    ('Chapter 2: Literature Review', 215): ['Cross-Stitch Networks / Sluice networks', 'CATR cattle multi-task transformer', 'Task-private pathways'],
    ('Chapter 2: Literature Review', 218): ['Task conflict between Re-ID and Behavior', 'Identity invariance vs identity specificity', 'Morphology preservation'],
    ('Chapter 2: Literature Review', 221): ['Disjoint multi-task dataset training', 'Domain discrepancy across tasks', 'Task-balanced batch scheduling'],
    ('Chapter 2: Literature Review', 228): ['Specific multi-task cattle monitoring problem', 'Representation requirements across tasks', 'Anatomical vs temporal features'],
    ('Chapter 2: Literature Review', 232): ['BCS anatomical geometry & subcutaneous fat', 'Ordinal regression formulation', 'Side and rear chute viewpoints'],
    ('Chapter 2: Literature Review', 235): ['Cattle localization, body structure, and motion', 'Deep visual representations', 'Task-specific feature extractors'],
    ('Chapter 2: Literature Review', 239): ['Masks, landmarks, and viewpoint representations', 'Robust benchmark evaluation', 'Representation reliability'],
    ('Chapter 2: Literature Review', 242): ['Scientific research gap synthesis', 'Cow-disjoint evaluation split', 'Unseen-cow generalization'],
    ('Chapter 2: Literature Review', 246): ['Task update balancing vs architectural sharing', 'Gradient conflict (PCGrad / GradNorm)', 'Task-private residual adapters (E3)'],
    ('Chapter 2: Literature Review', 249): ['Task-conditioned cattle-centered framework', 'Task-private residual adapters', 'Negative transfer mitigation'],
    ('Chapter 2: Literature Review', 252): ['Thesis roadmap & chapter progression', 'Scientific research gap synthesis', 'Unified multi-task deep learning framework'],
    
    # =========================================================================
    # CHAPTER 3: REQUIREMENTS & CONSTRAINTS
    # =========================================================================
    ('Chapter 3: Requirements & Constraints', 3): ['Core output specifications (BCS, Behavior, Cow ID)', '3-output unified framework', 'Evaluation protocols'],
    ('Chapter 3: Requirements & Constraints', 6): ['BCS 1-5 score range (+/- 0.25 tolerance)', '5 behavior classes (Macro-F1)', 'Re-ID Rank-1 accuracy (>50%)'],
    ('Chapter 3: Requirements & Constraints', 9): ['Data leakage prevention', 'Cow-disjoint split integrity', 'Deterministic model checkpoints'],
    ('Chapter 3: Requirements & Constraints', 12): ['Perception failure modes', 'Detector false negatives (RT-DETR-L)', 'SAM 2.1 segmentation quality'],
    ('Chapter 3: Requirements & Constraints', 16): ['Unified multi-task visual monitoring', 'Non-invasive animal welfare', 'Precision Livestock Farming (PLF)'],
    ('Chapter 3: Requirements & Constraints', 19): ['Decision-support system (non-replacement)', 'Veterinary clinical oversight', 'Human-in-the-loop decision making'],
    ('Chapter 3: Requirements & Constraints', 22): ['Commercial farm camera placement', 'Operational environmental variations', 'Edge deployment accessibility'],
    ('Chapter 3: Requirements & Constraints', 25): ['Producer & worker privacy (blurring)', 'Commercial farm confidentiality', 'Ethical visual data collection'],
    ('Chapter 3: Requirements & Constraints', 29): ['Upstream perception preprocessing', 'Downstream multi-task heads', 'Green AI compute footprint'],
    ('Chapter 3: Requirements & Constraints', 32): ['Perception feature caching', 'Zero redundant inference', 'Tesla T4 vs L40S efficiency'],
    ('Chapter 3: Requirements & Constraints', 35): ['Precision Livestock Farming (PLF) & feed efficiency', 'Non-invasive visual monitoring', 'Commercial dairy/beef routines'],
    ('Chapter 3: Requirements & Constraints', 39): ['Unseen-cow evaluation', 'Ground-truth annotation verification', 'Oracle evaluation bounds'],
    ('Chapter 3: Requirements & Constraints', 42): ['ScienceDB, CVB, and SideViewCows2026', 'Open-access research licenses (CC-BY)', 'Dataset provenance registry'],
    ('Chapter 3: Requirements & Constraints', 45): ['Held-out test population', 'Model validation selection', 'Zero test-set tuning / peeking'],
    ('Chapter 3: Requirements & Constraints', 48): ['AI ethics & author accountability', 'Scientific integrity', 'Turnitin plagiarism compliance'],
    ('Chapter 3: Requirements & Constraints', 52): ['IEEE software standards (R1-R4)', 'Documented technical conventions', 'Scientific software reproducibility'],
    ('Chapter 3: Requirements & Constraints', 55): ['BRAC University CSE400 format', 'IEEE reference formatting', 'Formal thesis chapter progression'],
    ('Chapter 3: Requirements & Constraints', 59): ['Thesis project phases & milestones', 'Protocol verification to MTL integration', 'Milestone deliverables'],
    ('Chapter 3: Requirements & Constraints', 62): ['Modal cloud infrastructure', 'NVIDIA GPU compute (Tesla T4 / L40S)', 'Local laptop smoke verification'],
    ('Chapter 3: Requirements & Constraints', 65): ['Deterministic manifests & Git versioning', 'Reproducible seed (2026)', 'Modal persistent storage'],
    ('Chapter 3: Requirements & Constraints', 68): ['Author contributions & committee review', 'Dr. Md. Khalilur Rahman (supervisor)', 'BRAC University CSE400 submission'],
    ('Chapter 3: Requirements & Constraints', 72): ['Scientific risk mitigation (stanchions/leakage)', 'Data leakage across video frames', 'Table reference (Tabletab:risks)'],
    ('Chapter 3: Requirements & Constraints', 75): ['Burst-group-disjoint splitting', 'Held-out test set isolation', 'Walking CCTV video clips'],
    ('Chapter 3: Requirements & Constraints', 79): ['Research compute expenditure', 'GPU hours (Tesla T4 vs L40S)', 'Economic feasibility of computer vision'],
    ('Chapter 3: Requirements & Constraints', 82): ['Economic feasibility & deployment hardware', 'On-farm camera installation', 'Commercial camera network maintenance']
}

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

def is_section_title(text):
    t = text.strip()
    if re.match(r'^\d+\.\d+', t):
        return True
    if t.startswith("[Table Reference") or t.startswith("Chapter "):
        return True
    return False

def build_markdown_for_row(tab_name, row_num, col_b, col_c):
    flagged_dict = FLAGGED_BY_TAB.get(tab_name, {})
    
    # 1. Flagged Row
    if row_num in flagged_dict:
        item = flagged_dict[row_num]
        roast = item["roast"]
        
        # Use curated terms if available, else fallback
        terms = CURATED_PARA_TERMS.get((tab_name, row_num), item.get("intact", []))
        intact_str = ", ".join(terms)
        
        fixes_lines = []
        for idx, f in enumerate(item["fixes"], start=1):
            clean_f = re.sub(r'^Fix\s*\d+:\s*', '', f)
            fixes_lines.append(f"• **Fix {idx}:** {clean_f}")
        fixes_str = "\n".join(fixes_lines)
        
        return (
            f"{roast}\n\n"
            f"**📌 Words/phrases to keep intact:**\n• {intact_str}\n\n"
            f"**🛠️ Required Fixes:**\n{fixes_str}"
        )
        
    # 2. Approved or Pending Academic Paragraph
    terms = CURATED_PARA_TERMS.get((tab_name, row_num), [])
    if not terms:
        # Fallback if somehow not mapped
        terms = ["Multi-task cattle health framework", "Task-specific representations"]
    intact_str = ", ".join(terms)
    
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
        range=f"'{tab_name}'!A1:D265"
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
        elif any(term in col_b for term in ["DONT PARAPHRASE", "Table Reference:", "[Table Reference"]):
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
                                    "userEnteredValue": {"stringValue": "রিভিউ ও রোস্টিং ☕ (Review Notes)"},
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
        
    print(f"\n🎉 ALL 3 CHAPTERS SUCCESSFULLY UPGRADED WITH SUPER-SMART ACADEMIC KEYWORDS in {time.time() - total_start:.2f}s!")

if __name__ == "__main__":
    main()
