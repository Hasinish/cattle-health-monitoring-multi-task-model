import os
import sys
import pypdf

names = [
    "Khalilur",
    "Rahman",
    "Mehedi",
    "Emo",
    "Mollah",
    "Saif",
    "Sadia",
    "Kazi",
    "Rabiul",
    "Alam"
]

def search_pdf(pdf_path):
    matches = []
    try:
        reader = pypdf.PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text:
                continue
            text_lower = text.lower()
            
            # Check supervisor/cosupervisor/HOD specifically
            found_names = []
            if "khalilur" in text_lower:
                found_names.append("Khalilur Rahman")
            if "mehedi" in text_lower or "emo" in text_lower:
                found_names.append("Mehedi Hasan Emo")
            if "mollah" in text_lower or "saif" in text_lower:
                found_names.append("Mollah MD Saif")
            if "sadia" in text_lower or "kazi" in text_lower:
                found_names.append("Sadia Hamid Kazi")
            if "rabiul" in text_lower or "gra" in text_lower:
                found_names.append("Golam Rabiul Alam")
                
            if found_names:
                # Check if approval or signature page
                is_approval = any(w in text_lower for w in ["approval", "examining committee", "declaration", "supervisor", "head of department"])
                matches.append({
                    "page": i + 1,
                    "names": list(set(found_names)),
                    "is_approval": is_approval,
                    "has_images": len(page.images) > 0,
                    "num_images": len(page.images)
                })
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return matches

def main():
    samples_dir = "P3 Samples"
    all_pdfs = []
    for root, dirs, files in os.walk(samples_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                all_pdfs.append(os.path.join(root, f))
                
    print(f"Scanning {len(all_pdfs)} PDF files in '{samples_dir}'...")
    results = {}
    for p in all_pdfs:
        res = search_pdf(p)
        if res:
            results[p] = res
            print(f"\n[MATCH] {p}:")
            for m in res:
                print(f"  Page {m['page']}: Names={m['names']}, Approval={m['is_approval']}, Images={m['num_images']}")

if __name__ == "__main__":
    main()
