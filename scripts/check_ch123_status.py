import sys
sys.path.insert(0, ".")
from scripts.sheet_tool import get_service, SPREADSHEET_ID

def check_ch123():
    service = get_service()
    chapters = {
        "Chapter 1: Introduction": "cattle_thesis_p3_latex/chapters/chapter_1.tex",
        "Chapter 2: Literature Review": "cattle_thesis_p3_latex/chapters/chapter_2.tex",
        "Chapter 3: Requirements & Constraints": "cattle_thesis_p3_latex/chapters/chapter_3.tex"
    }

    for title, tex_file in chapters.items():
        res = service.values().get(spreadsheetId=SPREADSHEET_ID, range=f"'{title}'!A1:D120").execute()
        vals = res.get('values', [])
        
        with open(tex_file, "r", encoding="utf-8") as f:
            tex_content = f.read()

        total_paras = 0
        populated_c = 0
        in_latex = 0
        not_in_latex = []

        print(f"\n==================== {title} ====================")
        for idx, r in enumerate(vals, 1):
            if len(r) > 1 and r[1] and r[1] != "Original (Do Paraphrase)":
                total_paras += 1
                b_text = r[1].strip()
                c_text = r[2].strip() if len(r) > 2 and r[2] else ""
                
                if c_text:
                    populated_c += 1
                    # Check if a snippet of c_text (first 30 chars) is in tex_content
                    # Clean out non-alphanumeric for matching
                    words = [w for w in c_text.split() if len(w) > 3][:5]
                    search_phrase = " ".join(words)
                    if search_phrase and search_phrase.lower() in tex_content.lower():
                        in_latex += 1
                    else:
                        not_in_latex.append((idx, c_text[:80]))

        print(f"Total paragraphs in Sheet: {total_paras}")
        print(f"Paraphrased in Column C: {populated_c} / {total_paras}")
        print(f"Integrated into LaTeX: {in_latex} / {populated_c}")
        if not_in_latex:
            print(f"  [PENDING INTEGRATION / NOT FOUND IN LATEX] ({len(not_in_latex)} rows):")
            for r_num, snippet in not_in_latex:
                print(f"    Row {r_num}: {snippet}...")
        else:
            print("  All paraphrases in Column C match LaTeX content!")

if __name__ == "__main__":
    check_ch123()
