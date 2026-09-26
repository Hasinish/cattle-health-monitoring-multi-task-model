from pathlib import Path
import re

targets = [
    ("Target 1", "These objectives are expressed"),
    ("Target 2", "Together, these works show two ways"),
    ("Target 3", "accessiblity of the monitoring"),
    ("Target 4", "Agricultural pictures can also"),
    ("Target 5", "Perception preprocessing, training"),
    ("Target 6", "The authors are still in charge"),
    ("Target 7", "The study follows well-known technical rules"),
    ("Target 8", "A real deployment estimate would need")
]

tex_files = list(Path("cattle_thesis_p3_latex").rglob("*.tex"))
for label, t in targets:
    found = False
    for tf in tex_files:
        content = tf.read_text(encoding="utf-8")
        # search case-insensitive, normalized spaces
        norm_content = " ".join(content.split())
        norm_t = " ".join(t.split())
        if norm_t.lower() in norm_content.lower():
            # find line number
            lines = content.splitlines()
            for line_idx, line in enumerate(lines, 1):
                if any(w.lower() in line.lower() for w in t.split()[:4]):
                    print(f"[{label}] Found in {tf} at line ~{line_idx}: {line[:80]}...")
                    found = True
                    break
            if not found:
                print(f"[{label}] Found in {tf} (multi-line)")
                found = True
            break
    if not found:
        print(f"[{label}] NOT FOUND: '{t}'")
