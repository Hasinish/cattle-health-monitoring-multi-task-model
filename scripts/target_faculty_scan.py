import os
import pypdf

samples_dir = "P3 Samples"
all_pdfs = []
for root, dirs, files in os.walk(samples_dir):
    for f in files:
        if f.lower().endswith(".pdf"):
            all_pdfs.append(os.path.join(root, f))

print(f"Total PDFs found: {len(all_pdfs)}")

target_faculty = [
    "khalilur",
    "emo",
    "saif",
    "sadia",
    "kazi",
    "rabiul"
]

findings = []

for pdf_path in all_pdfs:
    try:
        reader = pypdf.PdfReader(pdf_path)
        # Check pages 1 to 7 (front matter)
        for p_idx in range(min(7, len(reader.pages))):
            page = reader.pages[p_idx]
            text = page.extract_text() or ""
            text_lower = text.lower()
            
            matched = [fac for fac in target_faculty if fac in text_lower]
            if matched and any(k in text_lower for k in ["approval", "examining committee", "declaration"]):
                findings.append({
                    "pdf": pdf_path,
                    "page": p_idx + 1,
                    "matched": matched,
                    "num_images": len(page.images),
                    "text": text[:300]
                })
    except Exception as e:
        print(f"Error on {pdf_path}: {e}")

print(f"\nFound {len(findings)} relevant front matter pages:")
for f in findings:
    print(f"\nPDF: {f['pdf']} | Page: {f['page']}")
    print(f"Matched Faculty: {f['matched']} | Images on page: {f['num_images']}")
    print(f"Snippet: {f['text']!r}")
