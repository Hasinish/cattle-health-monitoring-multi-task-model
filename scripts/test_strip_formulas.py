import sys
from pathlib import Path
sys.path.insert(0, ".")

import re
from scripts.audit_ai_paragraphs_to_upload import chapter_data

def strip_formulas(text: str) -> str:
    # 1. Convert specific math phrases that form grammatical sentences
    text = re.sub(r'If\s*\[Formula:\s*g_i\^T\s*g_j\s*<\s*0,?\s*\]\s*', 'If the inner product is negative, ', text)
    text = re.sub(r'If\s*g_i\^T\s*g_j\s*<\s*0\s*,?\s*', 'If the inner product is negative, ', text)
    text = re.sub(r'If\s*g_i\^T\s*g_j\s*≥\s*0\s*,?\s*', 'If the inner product is non-negative, ', text)
    
    # 2. Strip standalone [Formula: ...] blocks completely
    text = re.sub(r'\[Formula:[^\]]*\]', '', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{equation\*?\}.*?\\end\{equation\*?\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\\begin\{align\*?\}.*?\\end\{align\*?\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\\\[.*?\\\]', '', text, flags=re.DOTALL)
    text = re.sub(r'\$\$(.*?)\$\$', '', text, flags=re.DOTALL)
    
    # 3. Clean trailing formula leads like "For N samples, [Formula]"
    text = re.sub(r'For\s+[A-Za-z0-9_]+\s+samples,?\s*$', '', text)
    text = re.sub(r'where\s+epsilon\s*=\s*1e-8.*?\.\s*', '', text)
    
    # 4. Clean residual math variables & symbols
    text = re.sub(r'\\?mathbf\{?0\}?|mathbf0', '0', text)
    text = re.sub(r'\\?mathbf\{?[a-zA-Z0-9_]+\}?', '', text)
    text = re.sub(r'\\?boldsymbol\{?[a-zA-Z0-9_]+\}?', '', text)
    text = re.sub(r'\\?nabla_\{?[a-zA-Z0-9_]+\}?', '', text)
    text = re.sub(r'\\?qquad', ' ', text)
    text = re.sub(r'delta_h_t', 'adapter features', text)
    text = re.sub(r'h_shared', 'shared representations', text)
    text = re.sub(r'h_t', 'task representations', text)
    text = re.sub(r'g_k\s*=\s*w_k\^T\s*h\s*\+\s*b_k', 'threshold logits', text)
    text = re.sub(r'z\s*=\s*h\s*/\s*\|\|h\|\|_2', 'unit-normalized embeddings', text)
    text = re.sub(r'\|\|[a-zA-Z0-9_]+\|\|_2', '', text)
    
    # 5. Fix sentence punctuation
    text = re.sub(r':\s*([,\.\?!;])', r'\1', text)
    text = re.sub(r':\s*\n', '.\n', text)
    text = re.sub(r':\s*$', '.', text)
    text = re.sub(r'\.\s*\.', '.', text)
    
    # 6. Clean extra spaces & newlines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\s*\n\s*', '\n', text)
    return text.strip()

for ch, data in chapter_data.items():
    for sec in data['sections']:
        for p in sec['paras']:
            if '[Formula:' in p:
                print(f"[{ch}] ({sec['title']})")
                print('CLEANED:\n', strip_formulas(p))
                print('=' * 60 + '\n')
