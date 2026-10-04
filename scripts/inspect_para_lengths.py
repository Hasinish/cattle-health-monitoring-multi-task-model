import sys
from pathlib import Path
sys.path.insert(0, ".")

from scripts.audit_ai_paragraphs_to_upload import chapter_data

lengths = []
for ch, data in chapter_data.items():
    for sec in data['sections']:
        for p in sec['paras']:
            w = len(p.split())
            lengths.append((w, ch, sec['title'], p))

lengths.sort(key=lambda x: x[0], reverse=True)
print(f"Total paragraphs: {len(lengths)}")
print(f"Max words in a single paragraph: {lengths[0][0]}")
print(f"Paragraphs > 80 words: {sum(1 for w, _, _, _ in lengths if w > 80)}")
print(f"Paragraphs > 100 words: {sum(1 for w, _, _, _ in lengths if w > 100)}")
print(f"Paragraphs > 120 words: {sum(1 for w, _, _, _ in lengths if w > 120)}")

print("\nTop 10 Longest Paragraphs:")
for w, ch, sec, p in lengths[:10]:
    print(f"  • [{w} words] [{ch}] ({sec}):")
    print(f"    \"{p[:100]}...\"\n")
