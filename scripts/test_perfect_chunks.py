import sys
from pathlib import Path
sys.path.insert(0, ".")

import re
from scripts.audit_ai_paragraphs_to_upload import chapter_data
from scripts.rebuild_clean_ai_sheets import strip_formulas

def clean_sentence_text(text: str) -> list[str]:
    text = strip_formulas(text).strip()
    if not text:
        return []

    # Clean bullet markers, item labels, etc.
    text = re.sub(r'^[•\-\*]\s*', '', text)
    # Convert newline bullets like "\n• " to ". " so sentences separate cleanly
    text = re.sub(r'\n+[•\-\*]\s*', '. ', text)
    text = re.sub(r'\n+\d+\.\s*', '. ', text)
    
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
            # Clean punctuation at end of sentence
            if not buffer.endswith(('.', '!', '?', ':')):
                buffer += '.'
            sentences.append(buffer)
            buffer = ""
    if buffer:
        if not buffer.endswith(('.', '!', '?', ':')):
            buffer += '.'
        sentences.append(buffer)

    return sentences


def create_ideal_paragraphs(paras: list[str], min_words: int = 50, target_max_words: int = 85) -> list[str]:
    """
    Packs sentences into ideal paragraphs matching Hasin's example:
    - Target: ~55 - 85 words (3 - 5 sentences).
    - Hard floor: >= 45 words (unless entire section has fewer).
    - Hard ceiling: <= 105 words.
    """
    all_sents = []
    for p in paras:
        all_sents.extend(clean_sentence_text(p))

    if not all_sents:
        return []

    chunks = []
    curr_chunk = []
    curr_words = 0

    for s in all_sents:
        s_words = len(s.split())
        
        # If we already have >= min_words and adding s exceeds target_max_words, finalize chunk
        if curr_words >= min_words and (curr_words + s_words > target_max_words):
            chunks.append(" ".join(curr_chunk))
            curr_chunk = [s]
            curr_words = s_words
        else:
            curr_chunk.append(s)
            curr_words += s_words

    if curr_chunk:
        # Check if the trailing chunk is too small (< 45 words)
        if chunks and curr_words < 45:
            last = chunks.pop()
            combined_sents = clean_sentence_text(last) + curr_chunk
            total_w = sum(len(s.split()) for s in combined_sents)
            if total_w <= 105:
                chunks.append(" ".join(combined_sents))
            else:
                # Balance into two ~50-65 word chunks
                half_words = total_w // 2
                c1, c2 = [], []
                w1 = 0
                for s in combined_sents:
                    sw = len(s.split())
                    if w1 + sw <= half_words or not c1:
                        c1.append(s)
                        w1 += sw
                    else:
                        c2.append(s)
                chunks.append(" ".join(c1))
                chunks.append(" ".join(c2))
        else:
            chunks.append(" ".join(curr_chunk))

    return chunks


all_lengths = []
total_ideal = 0

for ch, data in chapter_data.items():
    print(f"\n=======================================================")
    print(f"{ch}")
    print(f"=======================================================")
    ch_count = 0
    for sec in data['sections']:
        packed = create_ideal_paragraphs(sec['paras'], min_words=50, target_max_words=85)
        ch_count += len(packed)
        total_ideal += len(packed)
        for p in packed:
            w = len(p.split())
            all_lengths.append(w)
            if w < 40 or w > 105:
                print(f"  [ATTN: {w} words] ({sec['title']}): \"{p[:80]}...\"")
    print(f"Total ideal paragraphs in {ch}: {ch_count}")

print(f"\n=======================================================")
print(f"GRAND TOTAL IDEAL PARAGRAPHS: {total_ideal}")
print(f"Min words: {min(all_lengths)} | Max words: {max(all_lengths)}")
print(f"Average words: {sum(all_lengths) / len(all_lengths):.1f}")
print(f"Paragraphs under 45 words: {sum(1 for w in all_lengths if w < 45)}")
print(f"Paragraphs between 50 and 95 words: {sum(1 for w in all_lengths if 50 <= w <= 95)}")
print(f"=======================================================")
