import os
import re
import pypdf

samples_dir = "P3 Samples"
all_pdfs = []
for root, dirs, files in os.walk(samples_dir):
    for f in files:
        if f.lower().endswith(".pdf"):
            all_pdfs.append(os.path.join(root, f))

targets = {
    "Dr. Md. Khalilur Rahman": [r"\bkhalilur\b", r"\bkhalil\b"],
    "Mehedi Hasan Emo": [r"\bmehedi\s+hasan\b", r"\bemo\b"],
    "Mollah MD Saif": [r"\bmollah\b", r"\bsaif\b"],
    "Dr. Sadia Hamid Kazi": [r"\bsadia\s+hamid\b", r"\bkazi\b"]
}

print(f"Total PDFs: {len(all_pdfs)}")

hits = {k: [] for k in targets}

for pdf_path in all_pdfs:
    try:
        reader = pypdf.PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            for name, patterns in targets.items():
                for pat in patterns:
                    m = re.search(pat, txt, re.IGNORECASE)
                    if m:
                        hits[name].append({
                            "pdf": pdf_path,
                            "page": i + 1,
                            "match": m.group(0),
                            "snippet": txt[max(0, m.start()-50):min(len(txt), m.end()+50)].replace('\n', ' ')
                        })
                        break
    except Exception as e:
        pass

for name, matches in hits.items():
    print(f"\n==================== {name} (Matches: {len(matches)}) ====================")
    for m in matches:
        print(f"  {m['pdf']} (Page {m['page']}) -> Match: '{m['match']}' | Context: {m['snippet']!r}")
