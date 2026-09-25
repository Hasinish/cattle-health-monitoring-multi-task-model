import sys
sys.path.insert(0, ".")
from googleapiclient.discovery import build
from scripts.upload_to_google_sheet import get_credentials

creds = get_credentials()
sheets = build("sheets", "v4", credentials=creds).spreadsheets()
spreadsheet_id = "14UIi22gtPx_ogVGBPfhV45rN1R-zTREAQG3Aymcqk0A"

# Light yellow RGB
LIGHT_YELLOW = {"red": 1.0, "green": 0.98, "blue": 0.8}

ch1_reviews = {
    3: "Circular logic alert 🔄: You wrote that 'reduced food intake will cause a cow to eat less' 💀. Eating less causes it to eat less?! In the original, eating less and resting differently are separate co-occurring symptoms, not a cause that causes itself!",
    9: "'Trivial incidents'?! ☕ A cow drinking water, feeding, or showing early sickness isn't a 'trivial incident' like spilling tea! And 'short observations' refers to brief time duration, not 'modest' personality traits.",
    12: "Why is the first sentence wrapped in quotation marks like a movie review? 🎬 Direct quotes trigger automatic Turnitin plagiarism alerts! Also 'applications... in other livestock applications' repeats the word 'applications' twice in 5 words.",
    15: "'Learned traits' and 'various duties'? 😭 In animal science, 'traits' refers to genetics (like heritability of milk yield), not deep neural network representations! And deep learning models perform machine learning 'tasks', not 9-to-5 office 'duties'!",
    18: "'Sharing allows OCCUPATIONS'?! 💀 Are the cows applying for corporate office jobs on LinkedIn?! In multi-task learning, they are called machine learning 'tasks', never 'occupations'!",
    24: "'Two occupations responsible for the care of the same cow'?! 🧑‍🌾👨‍⚕️ The paper is discussing TWO COMPUTER VISION TASKS (BCS and Re-ID) processing the same image, not a veterinarian and a farmer having a meeting in the barn!",
    27: "1) BCS is body fat/energy reserves, NOT 'sickness' 🤒! Fat cows aren't sick, they're just well-fed! 2) A feed bunk or stall in a barn is not 'geography' 🗺️, it's pen background context.",
    30: "1) 'Shortcut learning' is a famous formal ML concept (Geirhos et al., 2020)—you cannot rename it to 'quick learning' like a mobile app speedrun! 2) 'The cow and its house'?! 🏡 Cows live in barns and pens, they don't have a mortgage!",
    33: "1) 'The target cow is localized through localization' is a tautology (like saying 'water is hydrated through water') 💀. 2) 'job-specific' -> Again with the corporate office jobs! It's 'task-specific'.",
    40: "1) 'Movie'?! 🍿 These cows are standing in barn CCTV surveillance footage, not starring in an IMAX Hollywood movie! 2) The train-test split doesn't 'measure characteristics'—nearby video frames cause data leakage that distorts what the split measures!",
    43: "Copy-paste disaster! 🤦‍♂️ You accidentally pasted the exact same movie paragraph from Row 40 here! Row 43 is supposed to be about adding model components and single-task reference baselines.",
    47: "'Recognition of activity is activity'?! 🤖 The floor here is made out of floor! Say 'Behavior recognition focuses on temporal activity'.",
    53: "1) 'The input is to look into...'?! A scientific thesis has a *contribution*, not an 'input'! 2) 'Quick segmentation'? The paper specifically refers to *promptable segmentation* (SAM).",
    57: "'Teach and send visual representations'?! 📬 Are we mailing neural network features by post office?! The ML term is *learn and share* representations across tasks!",
    63: "'The volume of information to be conveyed' sounds like an internet broadband package 📶. The question is how much parameter capacity should be *shared* between tasks!",
    73: "Factual contradiction! ❌ You claimed authentic IDs are not available at all, but SideViewCows2026 HAS 110 authentic biological cow IDs! Session groups are only used where IDs don't exist (like ScienceDB).",
    79: "'GEOLOCATION'?! 🛰️📍 Are we tracking cows with military satellites on Google Maps?! It's bounding-box *localization* in an image, not GPS coordinates!",
    82: "'Without harming individual actions'? It's without hurting individual ML *tasks* (preventing negative transfer), not preventing a cow from taking an action!",
    92: "Random floating number '1.' in the middle of a narrative paragraph 🔢. Did number 2 get lost in the pasture?",
    95: "Meaning warped: You made it sound like ONLY the behavior model uses masks and localization! All three tasks use masks and crops; only behavior adds temporal aggregation.",
    98: "'Retrieval assignment'?! 📝 Re-ID is an information *retrieval task* in computer vision, not homework assigned by your teacher!",
    101: "'Collected in a collaborative manner'?! 🤝 The paper means multi-task *parameter sharing* in a deep neural network, not farmers holding hands to collect data together!",
    105: "🚨 CRITICAL ALARM: YOU LITERALLY DELETED BCS! You listed 'Behavior Recognition, Individual Cow Identification, and Re-ID' (which is the same thing twice) and completely dropped Body Condition Scoring from the 3 tasks of the thesis!",
    111: "1) 'Barrier'?! In computer vision it's *occlusion* (view blockage), not an obstacle course! 2) 'Validated in the real world' is an unbacked overclaim—we did benchmark experiments, not live farm deployment!",
    117: "'Combining many activities in real life'?! 🏃‍♂️ It's about multi-task deep learning integration, not multitasking your daily chores!"
}

ch3_reviews = {
    3: "Caught red-handed! 🚨 The entire second half ('Because of this, the requirements include...') is copied 100% word-for-word from the original without changing a single letter! Plagiarism detectors will light up like a Christmas tree 🎄!",
    9: "'Not leaking' sounds like a plumbing repair manual, not machine learning data leakage prevention 🚰! And 'Keep big checkpoints' is a casual command that breaks formal academic tone.",
    12: "'Perceptual mistakes'? The vision detector failed to localize a cow, it didn't have a philosophical hallucination 👻! Use 'perception failures'.",
    22: "1) 'Species'?! Cattle are ONE species (*Bos taurus*), you meant *breed* 🐄! 2) 'The agri-environment is also giving access' sounds like the pasture is unlocking Wi-Fi passwords 📶!",
    29: "Architecture flipped upside down! 🔄 You wrote that upstream heavy detectors depend on minor downstream models! In reality, small downstream models depend on heavy upstream segmentation/detection models!",
    39: "'INVISIBLE COW ASSESSMENT' AND 'EVALUATION OF CONDUCT'?! 👻🐮 Are we auditing the moral conduct of ghost cows?! In ML, it's 'unseen-cow evaluation' (cows not seen in training) and 'cattle behavior', not moral discipline!",
    45: "'Measures the number of errors that are removed by perceptual errors'? Word salad! 🥗 Coverage reports the percentage of images successfully segmented vs dropped, not 'errors removing errors'!",
    68: "'Overseers'?! 💀 Dr. Khalilur Rahman is a distinguished university thesis supervisor, not a 19th-century plantation overseer! Use 'supervisors' or 'advisors'.",
    72: "1) 'Appropriate comments about train-test splits'?! The risk is DATA LEAKAGE (correlated frames leaking across splits), not polite conversation! 2) Don't rename the LaTeX table key to 'Tabletab:dangers'!",
    75: "1) 'Ambulation'?! 🚶‍♂️ The 5-class behavior ontology has a specific class called 'Walking'—you can't rename the label to fancy Latin! 2) 'Behavioral films'? These are 10-second CCTV camera clips, not Sundance films!"
}

def apply_reviews_to_tab(tab_name, sheet_id, reviews_dict):
    print(f"\n[*] Applying reviews to '{tab_name}' ({len(reviews_dict)} bad paraphrases)...")
    
    # 1. Update Column D header and values
    res = sheets.values().get(spreadsheetId=spreadsheet_id, range=f"'{tab_name}'!A1:C260").execute()
    rows = res.get("values", [])
    
    col_d_values = []
    format_requests = []
    
    # Set Column D width to 480px
    format_requests.append({
        "updateDimensionProperties": {
            "range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": 3, "endIndex": 4},
            "properties": {"pixelSize": 480},
            "fields": "pixelSize"
        }
    })
    
    # Format Col D cells with permanent WRAP and TOP alignment
    format_requests.append({
        "repeatCell": {
            "range": {
                "sheetId": sheet_id,
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
    
    # Prepare row by row
    for idx, r in enumerate(rows):
        row_num = idx + 1
        col_b = r[1] if len(r) > 1 else ""
        
        # Row 2 is the main table header
        if row_num == 2 or (col_b.startswith("Original") and row_num in [r_idx - 1 for r_idx in reviews_dict]):
            col_d_values.append(["Review Feedback & Meaning Check 🧐"])
            # Header format: Bold, light grey bg
            format_requests.append({
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": idx,
                        "endRowIndex": idx + 1,
                        "startColumnIndex": 3,
                        "endColumnIndex": 4
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "textFormat": {"bold": True, "fontSize": 10},
                            "backgroundColor": {"red": 0.9, "green": 0.9, "blue": 0.9},
                            "verticalAlignment": "MIDDLE"
                        }
                    },
                    "fields": "userEnteredFormat(textFormat,backgroundColor,verticalAlignment)"
                }
            })
        elif row_num in reviews_dict:
            feedback = reviews_dict[row_num]
            col_d_values.append([feedback])
            
            # Highlight bad paraphrase in Col C with LIGHT YELLOW
            format_requests.append({
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": idx,
                        "endRowIndex": idx + 1,
                        "startColumnIndex": 2,
                        "endColumnIndex": 3
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "backgroundColor": LIGHT_YELLOW
                        }
                    },
                    "fields": "userEnteredFormat(backgroundColor)"
                }
            })
            # Format feedback in Col D with soft light red/orange tint or clear
            format_requests.append({
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": idx,
                        "endRowIndex": idx + 1,
                        "startColumnIndex": 3,
                        "endColumnIndex": 4
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "textFormat": {"fontSize": 10, "italic": True}
                        }
                    },
                    "fields": "userEnteredFormat(textFormat)"
                }
            })
        elif len(r) > 2 and r[2] and r[2].strip() and r[2].strip() not in ["Paraphrased:", "LEAVE BLANK (Do Not Paraphrase):"]:
            col_d_values.append(["✅ Accurate & clean meaning preserved."])
        else:
            col_d_values.append([""])

    # Update Column D values
    sheets.values().update(
        spreadsheetId=spreadsheet_id,
        range=f"'{tab_name}'!D1:D{len(col_d_values)}",
        valueInputOption="USER_ENTERED",
        body={"values": col_d_values}
    ).execute()
    print(f"[OK] Injected {len(col_d_values)} rows into Column D of '{tab_name}'!")
    
    # Execute batch formatting
    chunk_size = 100
    for i in range(0, len(format_requests), chunk_size):
        chunk = format_requests[i:i + chunk_size]
        sheets.batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"requests": chunk}
        ).execute()
    print(f"[SUCCESS] Formatting and light yellow highlights applied to '{tab_name}'!")

print("=== APPLYING REVIEWS TO CHAPTER 1 ===")
apply_reviews_to_tab("Chapter 1: Introduction", 0, ch1_reviews)

print("\n=== APPLYING REVIEWS TO CHAPTER 3 ===")
apply_reviews_to_tab("Chapter 3: Requirements & Constraints", 521635664, ch3_reviews)

print("\n🏆 ALL REVIEWS AND LIGHT YELLOW HIGHLIGHTS APPLIED!")
