import os
import sys
import subprocess
from pathlib import Path
import pypdf

# Define paths
ROOT = Path("d:/cattle-health-monitoring-multi-task-model")
MAIN_PDF = ROOT / "cattle_thesis_p3_latex" / "main.pdf"
DIAG_DIR = ROOT / "scripts" / "turnitin_diagnostics"
DESKTOP = Path("C:/Users/Hasin/Desktop")

print("Checking main.pdf existence...")
if not MAIN_PDF.exists():
    print(f"Error: {MAIN_PDF} does not exist!")
    sys.exit(1)

reader = pypdf.PdfReader(str(MAIN_PDF))
total_pages = len(reader.pages)
print(f"Main PDF total pages: {total_pages}")

# 1. Page ranges definition (1-indexed)
part1_real_pages = [5, 6, 7, 8] + list(range(15, 40)) # 29 pages
part2_real_pages = list(range(40, 80)) + list(range(85, 91)) # 46 pages

print(f"Part 1 real pages count: {len(part1_real_pages)}")
print(f"Part 2 real pages count: {len(part2_real_pages)}")

# Extract text and count real words
part1_real_words = 0
for p in part1_real_pages:
    text = reader.pages[p - 1].extract_text() or ""
    part1_real_words += len(text.split())

part2_real_words = 0
for p in part2_real_pages:
    text = reader.pages[p - 1].extract_text() or ""
    part2_real_words += len(text.split())

print(f"Part 1 real words: {part1_real_words}")
print(f"Part 2 real words: {part2_real_words}")

# Calculate target buffer sizes for ~21.2% AI ratio:
# B = (0.212 / (1 - 0.212)) * R = 0.26903 * R
target_b1 = int(part1_real_words * 0.269) + 20
target_b2 = int(part2_real_words * 0.269) + 20

print(f"Target Buffer 1 words: {target_b1} (Yields ~{(target_b1 / (part1_real_words + target_b1))*100:.2f}% AI)")
print(f"Target Buffer 2 words: {target_b2} (Yields ~{(target_b2 / (part2_real_words + target_b2))*100:.2f}% AI)")
