# Rule: Mandatory Research Logging, Manifest Preservation, and State Tracking

## 1. Research Logging Protocol
Whenever you conduct any technical investigation, forensic dataset audit, data leakage diagnosis, hyperparameter search, or architectural pivot:
1. **Never leave findings scattered only in chat or scratch scripts.**
2. **Always document the findings in a dedicated markdown log**:
   - Location: `docs/research_log/YYYY-MM-DD_<topic>.md`
   - Must include:
     - Executive Summary
     - Context & Motivation
     - Forensic Findings / Data (exact numbers, distributions, leakage evidence)
     - Architectural Decisions & Trade-off Analysis
     - File & Artifact Registry (exact relative links)
     - Immediate Next Steps
3. **Always register the new entry** in the index table in `docs/research_log/README.md`.

## 2. Dataset Manifest & Artifact Preservation
1. **Never rely on ephemeral or runtime random splits** without saving a deterministic manifest.
2. Whenever grouping, de-duplicating, or partitioning a dataset:
   - Generate a deterministic CSV manifest with filename, class, animal/source group ID, confidence, evidence, and fold assignment.
   - Save the manifest in `datasets/<task>/<task>_manifest.csv` or `docs/manifests/`.
   - Ensure the manifest CSV is tracked by Git (unignored in `.gitignore`).
   - Save the generation script under `scripts/`.

## 3. Memory Synchronization Rule
1. **Always update `memory/state.md`** whenever a task or goal is completed, something new is learned, or progress is made.
2. **Always update `memory/history.md`** by prepending a clear summary of the session accomplishments.
3. **Always update `memory/index.md`** when new directories or major files (such as research logs or audit reports) are created.

## 4. Git Synchronization Rule
1. Audit reports (`docs/audits/`), research logs (`docs/research_log/`), manifests (`datasets/**/*.csv`), and memory files must be staged and committed so that work is safely preserved and transferable between the local laptop and the BRACU Lab Research PC.
2. Scratchpad directories (`scratch/`) containing temporary clones or gigabyte-scale archives must remain excluded via `.gitignore`.
