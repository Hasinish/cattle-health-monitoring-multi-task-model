# ==============================================================================
# SETUP RESEARCH PC - MASTER DATASET RESTORATION PROTOCOL (DEPRECATED)
# ==============================================================================
# NOTE: The BRACU Lab Research PC (RTX 5090) is disqualified due to environment bloat.
# Heavy compute is dispatched to rotating Modal cloud profiles. Preserved for provenance.

$ErrorActionPreference = "Continue"

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  P3 RESEARCH PC AUTOMATED RESTORATION PROTOCOL       " -ForegroundColor Yellow
Write-Host "======================================================" -ForegroundColor Cyan

$REPO_ROOT = Split-Path -Parent $PSScriptRoot
$DATASETS_DIR = "$REPO_ROOT\datasets"

# 1. Ensure directories exist
Write-Host "`n[STEP 1/3] Initializing directory tree..." -ForegroundColor Green
$folders = @(
    "$DATASETS_DIR\bcs\dryad_bcs",
    "$DATASETS_DIR\bcs\sciencedb_bcs",
    "$DATASETS_DIR\behavior\mmcows",
    "$DATASETS_DIR\behavior\CBVD-5",
    "$DATASETS_DIR\lameness",
    "$DATASETS_DIR\id",
    "$REPO_ROOT\final_models"
)
foreach ($f in $folders) {
    if (-not (Test-Path $f)) {
        New-Item -ItemType Directory -Path $f -Force | Out-Null
        Write-Host "  + Created $f" -ForegroundColor DarkGray
    }
}

# 2. Run Unified Python Pipeline
Write-Host "`n[STEP 2/3] Executing Unified Master Restoration Pipeline..." -ForegroundColor Green
$masterScript = "$REPO_ROOT\scripts\download_all.py"

python $masterScript --all

# 3. Final Verification
Write-Host "`n[STEP 3/3] Verifying Training Readiness..." -ForegroundColor Green
python -c "
from pathlib import Path
root = Path(r'$DATASETS_DIR')
checks = {
    'Lameness CSV': root / 'lameness' / 'lameness_index.csv',
    'Behavior CSV': root / 'behavior' / 'behavior_index.csv',
    'ID CSV': root / 'id' / 'id_index.csv',
    'BCS CSV': root / 'bcs' / 'bcs_index.csv'
}
for name, p in checks.items():
    status = '[READY]' if p.exists() else '[PENDING]'
    print(f'  {status:<10} {name:<15}: {p}')
"

Write-Host "`n======================================================" -ForegroundColor Cyan
Write-Host "  SETUP COMPLETE! YOU CAN NOW LAUNCH MULTI-TASK TRAINING" -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Cyan
