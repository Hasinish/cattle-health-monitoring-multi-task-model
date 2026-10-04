import sys
from pathlib import Path
sys.path.insert(0, ".")

import re
from scripts.audit_ai_paragraphs_to_upload import chapter_data
from scripts.rebuild_clean_ai_sheets import strip_formulas

def extract_clean_sentences(text: str) -> list[str]:
    """Extract clean, complete grammatical sentences without formulas."""
    text = strip_formulas(text).strip()
    if not text:
        return []

    # Clean bullet markers or numbers so they flow naturally or stay readable
    text = re.sub(r'^[•\-\*]\s*', '', text)
    
    # Split on sentence boundaries (. ! ?) followed by uppercase/digit
    raw_sents = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(])', text)
    sentences = []
    buffer = ""
    for s in raw_sents:
        s = s.strip()
        if not s:
            continue
        if buffer:
            buffer += " " + s
        else:
            buffer = s
        ends_with_abbrev = any(buffer.endswith(ab) for ab in ["e.g.", "i.e.", "et al.", "al.", "vs.", "Fig.", "Tab.", "Eq.", "No.", "Ref."])
        if not ends_with_abbrev:
            sentences.append(buffer)
            buffer = ""
    if buffer:
        sentences.append(buffer)

    return sentences


def pack_section_paragraphs(paras: list[str], min_words: int = 50, target_max_words: int = 90) -> list[str]:
    """
    Packs sentences from a section into ideal-sized paragraphs:
    - Target word count: 50 - 90 words (3-5 sentences).
    - No tiny paragraphs (<45 words) unless the entire section has fewer words.
    - No giant walls of text (>100 words).
    """
    all_sentences = []
    for p in paras:
        all_sentences.extend(extract_clean_sentences(p))

    if not all_sentences:
        return []

    packed_paras = []
    current_chunk = []
    current_words = 0

    for s in all_sentences:
        s_words = len(s.split())
        
        # If current chunk already reached target (>= 50 words) and adding this sentence pushes it over target_max_words
        if current_words >= min_words and (current_words + s_words > target_max_words):
            packed_paras.append(" ".join(current_chunk))
            current_chunk = [s]
            current_words = s_words
        else:
            current_chunk.append(s)
            current_words += s_words

    if current_chunk:
        # Check if the trailing chunk is too small (< 45 words)
        # If so, and we already have a previous chunk, merge them together
        if packed_paras and current_words < 45:
            last_chunk = packed_paras.pop()
            merged = last_chunk + " " + " ".join(current_chunk)
            # If merged isn't overly massive (<120 words), keep together
            if len(merged.split()) <= 115:
                packed_paras.append(merged)
            else:
                # Otherwise, split more evenly
                sents = extract_clean_sentences(merged)
                mid = len(sents) // 2
                packed_paras.append(" ".join(sents[:mid]))
                packed_paras.append(" ".join(sents[mid:]))
        else:
            packed_paras.append(" ".join(current_chunk))

    return packed_paras


# Test across all 6 chapters
total_chunks = 0
all_lengths = []

print("--- TESTING IDEAL PARAGRAPH PACKER (Target: 50-90 words) ---\n")

for ch, data in chapter_data.items():
    ch_chunks = 0
    print(f"=================================================================")
    print(f"{ch}")
    print(f"=================================================================")
    for sec in data['sections']:
        packed = pack_section_paragraphs(sec['paras'], min_words=50, target_max_words=90)
        ch_chunks += len(packed)
        total_chunks += len(packed)
        for p in packed:
            all_lengths.append(len(p.split()))
        if packed:
            print(f"  • Section '{sec['title']}': {len(packed)} paragraphs")
            for idx, p in enumerate(packed[:2], 1):
                print(f"      [{idx}] ({len(p.split())} words): \"{p[:90]}...\"")
    print(f"Total ideal paragraphs in {ch}: {ch_chunks}\n")

print(f"=================================================================")
print(f"GRAND TOTAL IDEAL PARAGRAPHS: {total_chunks}")
print(f"Min words: {min(all_lengths)} | Max words: {max(all_lengths)}")
print(f"Average words: {sum(all_lengths) / len(all_lengths):.1f}")
print(f"Paragraphs under 45 words: {sum(1 for w in all_lengths if w < 45)}")
print(f"Paragraphs between 50 and 95 words: {sum(1 for w in all_lengths if 50 <= w <= 95)}")
print(f"=================================================================")
