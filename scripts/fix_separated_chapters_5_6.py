import sys
import os

sys.path.insert(0, ".")
from scripts.sheet_tool import get_service, SPREADSHEET_ID

def get_sheet_id_by_title(service, spreadsheet_id, title):
    sheet_metadata = service.get(spreadsheetId=spreadsheet_id).execute()
    sheets = sheet_metadata.get('sheets', '')
    for s in sheets:
        if s['properties']['title'] == title:
            return s['properties']['sheetId']
    raise ValueError(f"Sheet '{title}' not found!")

# Chapter 5: Exactly separated paragraphs
ch5_paras = [
    "This chapter reports the completed single-task and multi-task evaluations, including the monolithic hard-shared control, modular task-private architecture, and PCGrad optimization control. The single-task evaluations first assess whether task-specific cattle-centered representations provide empirical utility over generic RGB baselines on matched populations. The subsequent multi-task evaluations directly examine the central questions of joint training: whether a unified spatial representation produces outcome-level negative transfer, whether architectural modularity mitigates cross-task degradation, whether gradient projection mitigates shared-backbone conflicts, and what training diagnostics reveal about gradient interference during optimization.",
    "The Baseline RGB model (Run 1) provides the canonical benchmark across all 8,040 ScienceDB test images, achieving a physical MAE of 0.1848 BCS units with 86.74% tolerance accuracy (Acc@1). Predictions generally stayed within ±0.25 units, though exact score separation remained challenging. This benchmark is a single point estimate under a burst-group-disjoint, rather than biological cow-disjoint, protocol.",
    "On the matched evaluation (Table 5.2), temporal foreground modeling raised balanced accuracy from 71.30% to 74.43% and reduced test loss from 0.5505 to 0.4430. Overall accuracy dipped slightly from 88.46% to 87.44%, indicating a redistribution of predictive weight rather than a uniform gain across all metrics.",
    "Open-set biometric retrieval was evaluated under SideView Protocol A, querying the fixed 36,811-image parlor gallery with barn CCTV and handheld snapshot imagery across 69 unseen cows disjoint from the 41 training identities.",
    "For Barn-to-Parlor surveillance queries, the Oracle Segmentation-Guided model (Run 6) improved primary retrieval quality (Table 5.5), raising Rank-1 accuracy from 58.64% to 63.90% and mean Average Precision (mAP) from 38.32% to 40.68%. The oracle target-centered configuration improved top-rank retrieval while deeper rank measures remained similar or slightly lower.",
    "Joint training under the Monolithic Hard-Shared MTL Control (E1) degraded the principal held-out BCS outcomes relative to the matched single-task reference (Run 4; Table 5.8). Physical MAE increased from 0.1709 to 0.1788 BCS units, alongside drops in tolerance accuracy and class-balanced measures. Although E1 recorded a slightly lower ordinal cross-entropy loss, this loss reduction did not translate into tighter physical score estimation. This pattern demonstrates outcome-level negative transfer when forcing ordinal morphological regression into a shared backbone alongside behavior and biometric identification.",
    "Architectural and optimization interventions yielded divergent retrieval outcomes. Modular adapters (E3) produced secondary degradation, driving Barn Rank-1 down to 49.08%. While PCGrad (E4) recovered substantial ground relative to E3—lifting Barn Rank-1 back to 54.97% and Snapshot mAP to 33.87%—it remained mixed relative to E1 and well below the single-task reference. Isolated exceptions appeared at deeper retrieval ranks (such as E3 maintaining slightly higher Rank-10 in snapshots), confirming that multi-task feature sharing alters retrieval rank distributions without matching dedicated single-task biometric precision.",
    "These diagnostics provide direct empirical evidence that opposing gradient directions on the shared four-channel ResNet-18 spatial backbone occurred regularly during joint training. Crucially, strict scientific claim boundaries must be maintained. E1 used ordinary shared-gradient accumulation without PCGrad projection, whereas E4 explicitly detected and projected opposing shared-backbone gradient directions. The E4 diagnostics establish that such conflicts occurred frequently during E4 training, but they do not establish identical conflict frequencies or a causal mechanism for E1. These measurements describe optimization dynamics under projection rather than proving that gradient interference alone accounted for observed performance gaps.",
    "Monolithic Hard Parameter Sharing (E1): Naively forcing three heterogeneous agricultural vision tasks into a single four-channel ResNet-18 spatial trunk produced held-out degradation across BCS, Behavior, and principal Re-ID retrieval metrics. This outcome-level negative transfer demonstrates that mutual synergy cannot be assumed when combining coarse body shape estimation (BCS), localized temporal motion (Behavior), and fine-grained biometric surface patterns (Re-ID). The specific internal feature corruption, however, cannot be isolated from E1 alone.",
    "Modular Task-Private Adapters (E3): Introducing task-private residual bottleneck adapters between the shared backbone and each prediction head failed to provide general negative-transfer mitigation. While selected surveillance and loss measures improved, BCS and Re-ID suffered further degradation, and Behavior balanced accuracy declined. Because E3 added 395,904 trainable parameters (+3.32% capacity), its performance differences cannot be attributed purely to architectural routing.",
    "PCGrad Optimization Control (E4): By applying PCGrad projection exclusively to shared backbone gradients under exact parameter parity with E1 (11,926,706 trainable parameters; zero adapters), E4 provided a controlled optimization comparison. E4 achieved selective, metric-dependent mitigation of negative transfer—notably improving Behavior overall accuracy, balanced accuracy, Macro-F1, and CVB surveillance performance, while recovering principal Re-ID and BCS metrics relative to E3. However, E4 remained below dedicated single-task models on several primary metrics.",
    "SideView Protocol A evaluates 69 biologically distinct cows excluded from representation learning. However, all Re-ID models use released dataset ground-truth masks, representing an oracle condition rather than autonomous field segmentation.",
    "External datasets referenced in the broader roadmap (such as Ruchay, Dryad, MmCows, CBVD-5, and BECA) remain unexecuted stress tests; their documentation does not imply cross-domain generalization.",
    "First, cattle-centered visual representations provide tangible utility over generic RGB frames on matched single-task benchmarks, but the nature of this benefit is highly task-dependent."
]

# Chapter 6: Every single paragraph and list item cleanly separated
ch6_paras = [
    # 1. Summary of Findings - Intro
    "This thesis investigated whether cattle-centered visual representations and multi-task learning can support unified monitoring of dairy cattle across three distinct vision tasks: body condition scoring (BCS), behavior recognition, and individual cow re-identification (Re-ID). Across all experimental investigations, rigorous provenance tracking, leakage-aware evaluation protocols, and matched-population controls were enforced to ensure reproducible and reliable empirical conclusions.",
    
    # 2. Single-Task BCS
    "For Body Condition Scoring, evaluating on the matched population of N = 7,549 ScienceDB test images where automatic detection and segmentation succeeded (93.89% perception coverage), localized cropping and binary foreground masks reduced physical error from 0.1929 to 0.1709 Mean Absolute Error (MAE) in BCS units and increased Acc@1 within ±0.25 units from 84.95% to 89.40% relative to the generic RGB baseline. However, exact classification accuracy remained modest (43.57%), and class-balanced metrics showed minor declines (balanced accuracy of 39.70% vs. 40.23%; Macro-F1 of 0.4039 vs. 0.4074).",
    
    # 3. Single-Task Behavior
    "For Behavior Recognition, evaluating on N = 780 matched video sequences (T = 8 frames; 96.42% retained coverage), combining cattle-centered crops, foreground masks, and a lightweight 1D temporal convolutional network increased balanced accuracy from 71.30% to 74.43%, decreased test loss from 0.5505 to 0.4430, and improved minority-class Walking F1 from 0.2174 to 0.2456 relative to the single-frame RGB baseline. However, overall accuracy decreased slightly from 88.46% to 87.44%, and performance redistributed across sources (improving on feedlot camera footage while declining slightly on barn CCTV surveillance).",
    
    # 4. Single-Task Re-ID
    "For Cattle Re-Identification, evaluated under SideView Protocol A across 69 unseen cattle identities, target-centered cropping guided by released ground-truth masks improved first-match retrieval across both query conditions: Barn-to-Parlor Rank-1 increased from 58.64% to 63.90% (mAP from 38.32% to 40.68%), while Snapshot-to-Parlor Rank-1 rose from 38.88% to 62.93% (mAP from 27.05% to 40.42%). This substantial gain under handheld snapshot queries indicates that target-centered isolation is particularly beneficial when handling extreme posture, angle, and illumination shifts; however, because released ground-truth masks were used, this represents an oracle condition rather than an automatic segmentation deployment pipeline.",
    
    # 5. Multi-Task E1 Hard Sharing
    "The Monolithic Hard-Shared Multi-Task Control (E1), which forced all three tasks through a single four-channel ResNet-18 spatial backbone, resulted in outcome-level negative transfer across all three tasks relative to matched single-task references: BCS MAE degraded from 0.1709 to 0.1788; Behavior balanced accuracy dropped from 74.43% to 67.30% (with Walking F1 falling from 0.2456 to 0.0408); and Re-ID Barn Rank-1 fell from 63.90% to 57.38% (mAP falling from 40.68% to 30.37%).",
    
    # 6. RQ1 Conclusion / Attribution Boundary
    "These outcomes demonstrate that cattle-centered visual representations provide tangible utility, but they do not universally benefit all evaluation metrics or eliminate shortcut learning across tasks. Furthermore, because these comparisons evaluated combined perception pipelines (crop plus mask for BCS and Re-ID; crop, mask, and temporal convolution for Behavior), the gains cannot be attributed to segmentation alone.",
    
    # 7. RQ2 Intro
    "The experimental results support task-specific incorporation of temporal dynamics and structural cues rather than forcing heterogeneous livestock monitoring tasks into a uniform input representation:",
    
    # 8. RQ2 BCS Morphology
    "For Body Condition Scoring, static morphology is the primary biological signal. Incorporating localized bounding-box crops and foreground binary masks provided a localized foreground representation while retaining visual morphology relevant to BCS, achieving superior error proximity without requiring temporal aggregation.",
    
    # 9. RQ2 Behavior Dynamics
    "For Behavior Recognition, posture and motion dynamics require temporal context. The combined eight-frame foreground-guided TCN configuration improved balanced accuracy and Walking F1 relative to the matched single-frame reference.",
    
    # 10. RQ2 Re-ID Surface Details
    "For Cow Re-Identification, biometric coat patterns and identity features require fine-grained surface spatial detail. Cropping and mask guidance preserved localized surface geometry; the oracle-mask experiments demonstrated substantial retrieval gains under unconstrained query conditions when reliable foreground isolation is available.",
    
    # 11. RQ2 Pose & Viewpoint Boundary
    "Anatomical pose keypoints and viewpoint priors were systematically audited. Frozen SuperAnimal pose estimation exhibited high missingness (45.60% detector failure rate on behavior sequences, deteriorating to 72.92% failure on recumbent/lying cattle) and lacked critical rear-pelvic landmarks required for BCS; pose was consequently excluded from the primary downstream pipeline. Real-cattle viewpoint classification achieved 86.26% accuracy on its own domain, but was not fused into primary models because cross-domain transfer to downstream farm environments remained unverified, with high camera and source shortcut risks.",
    
    # 12. RQ2 Conclusion
    "Thus, preserving task-specific requirements is best accomplished by tailoring visual cues to the physical demands of each domain—spatial morphology for BCS, temporal sequence modeling for behavior, and surface biometrics for re-identification—rather than imposing a single monolithic cue across all tasks.",
    
    # 13. RQ3 Hard Sharing Negative Transfer
    "Outcome-Level Negative Transfer under Hard Sharing: Monolithic hard parameter sharing (E1) degraded held-out performance across all three task domains relative to matched single-task references: BCS MAE rose by +0.0079 units; Behavior balanced accuracy dropped by -7.13 percentage points (with Walking F1 falling from 0.2456 to 0.0408); and Re-ID Barn Rank-1 dropped by -6.52 percentage points (mAP falling by -10.31 percentage points). These consistent held-out degradations confirm outcome-level negative transfer when forcing disparate cattle vision tasks into a single shared ResNet-18 spatial trunk.",
    
    # 14. Contribution 3
    "Task-Specific Cattle Representation Strategy: Formulated and verified a principled, evidence-led input representation pipeline tailored to the physical demands of each task: static morphology-focused crop and mask for BCS; 8-frame sequential modeling with 1D temporal convolution for behavior; and biometric surface pattern preservation for re-identification. Audited and documented empirical boundaries for anatomical pose estimation and viewpoint priors.",
    
    # 15. Contribution 4
    "Systematic Multi-Task Sharing Benchmark: Conducted a controlled multi-task evaluation comparing monolithic hard parameter sharing (E1), modular task-private residual adapters (E3), and gradient-projected optimization (E4) against matched single-task baselines on identical held-out test sets, providing an empirical benchmark for livestock multi-task vision.",
    
    # 16. Contribution 5
    "Direct Gradient Conflict Diagnostics: Documented direct empirical measurements of shared-parameter gradient interference during multi-task training under PCGrad, recording 44,177 pairwise conflict projections across 16,140 super-steps and demonstrating regular gradient opposition (46.5% to 47.8% frequency) between agricultural monitoring tasks.",
    
    # 17. Contribution 6
    "Calibrated Understanding of Multi-Task Trade-Offs: Provided empirical evidence that multi-task learning does not universally benefit individual agricultural tasks through shared representations, demonstrating that negative transfer is outcome-level and that mitigation strategies provide selective, metric-dependent trade-offs rather than uniform performance gains.",
    
    # 18. Limitation 12
    "Persistence of Negative Transfer: Despite selective improvements from PCGrad, final multi-task models underperformed dedicated single-task references on several primary metrics, indicating that joint multi-task deployment involves deliberate engineering compromises.",
    
    # 19. Future Work 1
    "Repeated-Seed Training and Uncertainty Estimation: Future investigations should conduct multi-seed training runs to compute confidence intervals, variance bounds, and formal statistical tests across single-task and multi-task configurations.",
    
    # 20. Future Work 2
    "End-to-End Automatic Re-ID Segmentation: Integrating an upstream automated segmentation model (such as fine-tuned SAM or YOLO-seg) to replace oracle masks with predicted masks, evaluating real-world degradation under segmentation error.",
    
    # 21. Future Work 3
    "External Multi-Herd Generalization Testing: Evaluating trained models on independent external datasets, including Ruchay2026 and Dryad for BCS, and MmCows or CBVD-5 for behavior recognition, to establish cross-farm generalization.",
    
    # 22. Future Work 4
    "Longitudinal Identity Retrieval: Stress-testing cow re-identification across longitudinal cohorts (such as BECA-L and BECA-D) to assess biometric stability across lactation stages, seasonal coat changes, and physical growth.",
    
    # 23. Future Work 5
    "Isolated Perception Component Ablations: Conducting factorial ablation studies to isolate the individual contributions of bounding-box localization, binary foreground masking, soft probability masking, and temporal sequence length.",
    
    # 24. Future Work 6
    "Robust Cattle Pose Estimation: Fine-tuning animal pose models specifically on recumbent, lying, and rear-view cattle postures to overcome the keypoint missingness that currently prevents pose integration into livestock behavior monitoring.",
    
    # 25. Future Work 7
    "Advanced Multi-Task Optimization Strategies: Investigating partial parameter sharing (such as branched trunks) and adaptive loss-weighting algorithms (such as GradNorm) to further mitigate cross-task gradient tension.",
    
    # 26. Final Conclusion - Framework Setup
    "This thesis established an empirical and methodological framework for multi-task deep learning in precision cattle monitoring, evaluating body condition scoring, behavior recognition, and cow re-identification under controlled single-task and joint multi-task settings.",
    
    # 27. Final Conclusion - Cattle-Centered Representations
    "The empirical evidence demonstrates that cattle-centered visual representations provide substantial, measurable utility for individual livestock monitoring tasks, particularly by improving BCS error tolerance and enhancing cow re-identification under unconstrained camera shifts. However, these benefits are task-dependent and do not eliminate the distinct representational requirements inherent to morphology regression, temporal sequence modeling, and biometric identity matching.",
    
    # 28. Final Conclusion - Multi-Task Trade-Offs & PCGrad
    "When integrated into a unified multi-task framework, naive hard parameter sharing resulted in clear outcome-level negative transfer across all three tasks. Incorporating task-private residual adapters failed to resolve this degradation, while gradient-projected optimization (PCGrad) directly revealed that opposing shared-backbone gradient directions occurred regularly during training. By resolving opposing gradient components, PCGrad provided selective mitigation—notably boosting behavior recognition metrics and authentic barn surveillance performance—yet dedicated single-task models remained superior on several task-specific measures.",
    
    # 29. Final Conclusion - Final Takeaway
    "Ultimately, these findings demonstrate that multi-task learning in agricultural computer vision does not provide a universal performance benefit through shared representations; successful deployment requires balancing shared cattle features with task-specific capacity and conflict-aware optimization."
]

def populate_chapter(service, sheet_id, title, paras):
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
    service.values().clear(spreadsheetId=SPREADSHEET_ID, range=f"'{title}'!A1:Z1000").execute()

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

    # Chapter 5: Result Analysis
    ch5_id = get_sheet_id_by_title(service, SPREADSHEET_ID, "Chapter 5: Result Analysis")
    populate_chapter(service, ch5_id, "Chapter 5: Result Analysis", ch5_paras)

    # Chapter 6: Conclusion
    ch6_id = get_sheet_id_by_title(service, SPREADSHEET_ID, "Chapter 6: Conclusion")
    populate_chapter(service, ch6_id, "Chapter 6: Conclusion", ch6_paras)

    print("\nALL SHEETS RE-POPULATED WITH DISCRETE SEPARATED PARAGRAPHS!")

if __name__ == "__main__":
    main()
