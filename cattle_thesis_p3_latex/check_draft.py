#!/usr/bin/env python3
"""Check draft links, citations and recorded-matrix arithmetic, not experiment validity."""
from pathlib import Path
import json
import re
import sys
ROOT=Path(__file__).resolve().parent

def main() -> int:
    errors=[]
    texts={p:p.read_text(encoding='utf-8') for p in ROOT.rglob('*.tex')}
    keys=set(re.findall(r'@\w+\s*\{\s*([^,\s]+)',(ROOT/'bibliography/references.bib').read_text()))
    used=set()
    for path,text in texts.items():
        for name in re.findall(r'\\(?:input|include)\{([^}]+)\}',text):
            target=ROOT/name
            if not target.suffix: target=target.with_suffix('.tex')
            if not target.is_file(): errors.append(f'Missing input {name} in {path.name}')
        for group in re.findall(r'\\cite\w*(?:\[[^]]*\])*\{([^}]+)\}',text):
            used.update(x.strip() for x in group.split(','))
    if used-keys: errors.append('Undefined bibliography keys: '+', '.join(sorted(used-keys)))
    source='\n'.join(texts.values())
    ids=set(re.findall(r'\| (E\d\d) \|',(ROOT/'EVIDENCE_MAP.md').read_text()))
    for group in re.findall(r'\\evidence\{([^}]+)\}',source):
        for eid in group.split(','):
            if eid.strip() not in ids: errors.append('Unmapped evidence: '+eid)
    mapping=re.findall(r'\\input\{chapters/chapter_(\d+)\.tex\}',(ROOT/'main.tex').read_text())
    if mapping!=['1','2','3','5','6','9']: errors.append('Unexpected chapter mapping '+str(mapping))
    data=json.loads((ROOT/'evidence/baseline_confusion_matrices.json').read_text())
    for name,n_expected,f_expected in [('bcs',8040,.4109745347432794),('behavior',809,.7413)]:
        cm=data[name+'_matrix'];n=sum(map(sum,cm));m=len(cm)
        support=list(map(sum,cm));pred=[sum(row[k] for row in cm) for k in range(m)]
        f1=sum(2*cm[k][k]/(support[k]+pred[k]) for k in range(m))/m
        if n!=n_expected or abs(f1-f_expected)>.00005: errors.append(name+' arithmetic mismatch')
    if (ROOT/'main.log').exists():
        log=(ROOT/'main.log').read_text(errors='replace')
        for pattern in [r'Undefined control sequence',r'Emergency stop',r'Fatal error occurred',r'Citation .* undefined',r'Reference .* undefined',r'There were undefined references']:
            if re.search(pattern,log): errors.append('Build issue: '+pattern)
    if errors:
        print('\n'.join(errors),file=sys.stderr);return 1
    print(f'PASS: inputs, chapter mapping, {len(used)} cited keys, {len(ids)} evidence IDs, matrix arithmetic.')
    todo_count=len(re.findall(r'\\drafttodo\{',source))
    print(f'DRAFT: {todo_count} TODO calls remain; no raw-data or model audit performed.')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
