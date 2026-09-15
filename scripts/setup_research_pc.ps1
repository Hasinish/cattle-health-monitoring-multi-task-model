# ==============================================================================
# SETUP RESEARCH PC - DATASET DOWNLOAD & RESTORATION SCRIPT
# ==============================================================================
# Run this script in PowerShell on the Research PC (RTX 5090)
# It creates the directory structure, downloads repos, and guides extraction.

$ErrorActionPreference = "Continue"

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  P3 RESEARCH PC AUTOMATED RESTORATION PROTOCOL       " -ForegroundColor Yellow
Write-Host "======================================================" -ForegroundColor Cyan

# 1. Base Directories
$BASE_DIR = "D:\T25301094 P2"
$DATASETS_DIR = "$BASE_DIR\datasets"

Write-Host "`n[STEP 1/6] Creating directory structure at $BASE_DIR..." -ForegroundColor Green
$folders = @(
    "$DATASETS_DIR\bcs\dryad_bcs",
    "$DATASETS_DIR\bcs\sciencedb_bcs",
    "$DATASETS_DIR\behavior\mmcows",
    "$DATASETS_DIR\behavior\CBVD-5",
    "$DATASETS_DIR\lameness",
    "$DATASETS_DIR\id",
    "$BASE_DIR\final_models"
)
foreach ($f in $folders) {
    if (-not (Test-Path $f)) {
        New-Item -ItemType Directory -Path $f -Force | Out-Null
        Write-Host "  + Created $f" -ForegroundColor DarkGray
    }
}

# 2. Lameness (Git Clone - Instant)
Write-Host "`n[STEP 2/6] Restoring CattleLameness (Git)..." -ForegroundColor Green
$lameTarget = "$DATASETS_DIR\lameness\CattleLameness"
if (-not (Test-Path "$lameTarget\.git")) {
    git clone https://github.com/fahimsohan/CattleLameness $lameTarget
} else {
    Write-Host "  CattleLameness already exists. Skipping clone." -ForegroundColor Yellow
}

# 3. Preprocess Lameness Frames
$lameScript = "d:\cattle-health-monitoring-multi-task-model\context\preprocess_lameness.py"
if (Test-Path $lameScript) {
    Write-Host "  Running lameness frame extraction..." -ForegroundColor Cyan
    python $lameScript
}

# 4. MmCows via Kaggle CLI
Write-Host "`n[STEP 3/6] Checking Kaggle CLI for MmCows..." -ForegroundColor Green
$kaggleJson = "$env:USERPROFILE\.kaggle\kaggle.json"
if (Test-Path $kaggleJson) {
    Write-Host "  Found kaggle.json! Downloading MmCows in background..." -ForegroundColor Cyan
    kaggle datasets download -d hienvuvg/mmcows -p "$DATASETS_DIR\behavior\mmcows" --unzip
    
    $behScript = "d:\cattle-health-monitoring-multi-task-model\context\preprocess_mmcows_behavior.py"
    if (Test-Path $behScript) {
        Write-Host "  Running behavior indexer..." -ForegroundColor Cyan
        python $behScript
    }
} else {
    Write-Host "  WARNING: kaggle.json not found at $kaggleJson!" -ForegroundColor Red
    Write-Host "  Please place your kaggle.json in $env:USERPROFILE\.kaggle\ and rerun," -ForegroundColor Yellow
    Write-Host "  Or download manually from: https://kaggle.com/datasets/hienvuvg/mmcows" -ForegroundColor Yellow
}

# 5. Open URLs for Manual Large Downloads
Write-Host "`n[STEP 4/6] Opening Browser tabs for remaining downloads..." -ForegroundColor Green
Write-Host "  Opening Dryad BCS download..." -ForegroundColor Cyan
Start-Process "https://datadryad.org/dataset/doi:10.5061/dryad.tqjq2bw4s"

Write-Host "  Opening OpenCows2020 download..." -ForegroundColor Cyan
Start-Process "https://datasetninja.com/opencows2020"

Write-Host "  Opening ScienceDB BCS download..." -ForegroundColor Cyan
Start-Process "https://scidb.cn/en/detail?dataSetId=16b8bdaf31ee4c8b9891fc7e9df6e41c"

Write-Host "`n======================================================" -ForegroundColor Cyan
Write-Host "  INSTRUCTIONS FOR UNPACKING DOWNLOADED ZIPS:         " -ForegroundColor Yellow
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "1. Dryad ZIP -> Extract into: D:\T25301094 P2\datasets\bcs\dryad_bcs\"
Write-Host "   Then run: python d:\cattle-health-monitoring-multi-task-model\context\preprocess_bcs.py"
Write-Host "`n2. OpenCows2020 ZIP -> Extract into: D:\T25301094 P2\datasets\id\opencow2020-DatasetNinja\"
Write-Host "   Then run: python d:\cattle-health-monitoring-multi-task-model\context\preprocess_id.py"
Write-Host "`n3. ScienceDB ZIP -> Extract into: D:\T25301094 P2\datasets\bcs\sciencedb_bcs\"
Write-Host "   Then run: python d:\cattle-health-monitoring-multi-task-model\context\preprocess_sciencedb_bcs.py"
Write-Host "======================================================`n" -ForegroundColor Cyan
