import os
import glob
import pypdf

sample_dir = r"D:\cattle-health-monitoring-multi-task-model\P3 Samples"
pdf_files = glob.glob(os.path.join(sample_dir, "*.pdf")) + glob.glob(os.path.join(sample_dir, "fall 2025", "*.pdf"))

print(f"Found {len(pdf_files)} P3 sample PDFs.")

abstracts = {}

for pdf_path in pdf_files[:10]:
    try:
        reader = pypdf.PdfReader(pdf_path)
        # Search the first 10 pages for "Abstract"
        for page_idx in range(min(12, len(reader.pages))):
            text = reader.pages[page_idx].extract_text()
            if "abstract" in text.lower():
                # check if it's the actual abstract section
                lines = text.split("\n")
                abstract_lines = []
                capturing = False
                for line in lines:
                    if "abstract" in line.lower() and len(line.strip()) < 20:
                        capturing = True
                        continue
                    if capturing:
                        if any(stop_word in line.lower() for stop_word in ["keywords:", "key words:", "table of contents", "acknowledgment", "dedication", "\x0c"]):
                            break
                        abstract_lines.append(line)
                
                content = " ".join(abstract_lines).strip()
                if len(content) > 100:
                    words = len(content.split())
                    name = os.path.basename(pdf_path)
                    abstracts[name] = (words, content)
                    break
    except Exception as e:
        pass

print(f"\nSuccessfully extracted {len(abstracts)} abstracts from P3 samples:\n")
for name, (words, text) in abstracts.items():
    print(f"=== {name} ===")
    print(f"Word Count: {words} words")
    print(f"Sample snippet: {text[:250]}...\n")
