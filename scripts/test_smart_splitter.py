import sys
from pathlib import Path
sys.path.insert(0, ".")

import re
from scripts.audit_ai_paragraphs_to_upload import chapter_data
from scripts.rebuild_clean_ai_sheets import strip_formulas

def split_large_paragraph(text: str, target_max_words: int = 75) -> list[str]:
    """
    Intelligently splits large paragraphs into smaller, digestible sub-paragraphs:
    1. Splits on numbered items (1., 2., 3.) or bullet points (•)
    2. Splits long narrative prose by sentences into ~50-75 word logical chunks.
    """
    text = strip_formulas(text).strip()
    if not text:
        return []

    words = text.split()
    if len(words) <= target_max_words:
        return [text]

    # Check for numbered items or bullets (e.g., "1. ", "\n2. ", "• ")
    bullet_pattern = re.compile(r'(?:^|\n|\s)(?:(\d+\.\s+[A-Z])|(•\s+[A-Z]))')
    matches = list(bullet_pattern.finditer(text))
    if len(matches) >= 2:
        # Split on bullets/numbered points
        split_points = [m.start() for m in matches]
        chunks = []
        if split_points[0] > 0:
            intro = text[:split_points[0]].strip()
            if intro:
                chunks.append(intro)
        for i in range(len(split_points)):
            start = split_points[i]
            end = split_points[i+1] if i+1 < len(split_points) else len(text)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
        
        # Recursively split any sub-chunk that is still overly long (>90 words)
        final_chunks = []
        for c in chunks:
            if len(c.split()) > target_max_words + 20:
                final_chunks.extend(split_narrative_sentences(c, target_max_words))
            else:
                final_chunks.append(c)
        return final_chunks

    # Otherwise, split narrative prose by sentences
    return split_narrative_sentences(text, target_max_words)


def split_narrative_sentences(text: str, target_max_words: int = 75) -> list[str]:
    # Split by sentence boundaries (e.g. ". ", "! ", "? ")
    # Negative lookbehind for abbreviations like "e.g.", "et al.", "i.e.", "Fig.", "vs."
    raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(])', text)
    
    # Re-combine sentences if split on abbreviations
    sentences = []
    buffer = ""
    for s in raw_sentences:
        if buffer:
            buffer += " " + s
        else:
            buffer = s
        
        # Check if buffer ends with an abbreviation
        ends_with_abbrev = any(buffer.endswith(ab) for ab in ["e.g.", "i.e.", "et al.", "al.", "vs.", "Fig.", "Tab.", "Eq.", "No.", "Ref."])
        if not ends_with_abbrev:
            sentences.append(buffer)
            buffer = ""
    if buffer:
        sentences.append(buffer)

    chunks = []
    current_chunk = []
    current_word_count = 0

    for s in sentences:
        s_words = len(s.split())
        if current_word_count + s_words > target_max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [s]
            current_word_count = s_words
        else:
            current_chunk.append(s)
            current_word_count += s_words

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Test on the top 5 longest paragraphs
from scripts.inspect_para_lengths import lengths

print("--- TESTING SMART PARAGRAPH SPLITTER ---\n")
for w, ch, sec, p in lengths[:4]:
    print(f"=================================================================")
    print(f"ORIGINAL [{w} words] [{ch}] ({sec})")
    print(f"=================================================================")
    sub_paras = split_large_paragraph(p, target_max_words=70)
    print(f"-> SPLIT INTO {len(sub_paras)} DIGESTIBLE PARAGRAPHS:")
    for idx, sp in enumerate(sub_paras, start=1):
        print(f"   [{idx}] ({len(sp.split())} words):")
        print(f"       \"{sp}\"\n")
