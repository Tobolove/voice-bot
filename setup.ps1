# IndexTTS-2 Voice Bot Setup Script for Windows
# ==============================================
# This script will download and set up the IndexTTS-2 model

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  IndexTTS-2 Voice Bot Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$ProjectDir = $PSScriptRoot

# Step 1: Check for git-xet (needed for Hugging Face large files)
Write-Host "[1/5] Checking git-xet..." -ForegroundColor Yellow
$gitXet = Get-Command git-xet -ErrorAction SilentlyContinue
if (-not $gitXet) {
    Write-Host "  Installing git-xet via winget..." -ForegroundColor Gray
    winget install git-xet
}
Write-Host "  git-xet ready!" -ForegroundColor Green

# Step 2: Install Hugging Face CLI
Write-Host ""
Write-Host "[2/5] Installing Hugging Face CLI..." -ForegroundColor Yellow
pip install "huggingface-hub[cli,hf_xet]" --quiet
Write-Host "  HF CLI installed!" -ForegroundColor Green

# Step 3: Clone the Space repository (contains the inference code)
Write-Host ""
Write-Host "[3/5] Cloning IndexTTS-2 Space repository..." -ForegroundColor Yellow
$SpaceDir = Join-Path $ProjectDir "IndexTTS-2-Demo"

if (Test-Path $SpaceDir) {
    Write-Host "  Space already cloned, pulling latest..." -ForegroundColor Gray
    Push-Location $SpaceDir
    git pull
    Pop-Location
} else {
    Write-Host "  Cloning from Hugging Face (this may take a while)..." -ForegroundColor Gray
    git clone https://huggingface.co/spaces/IndexTeam/IndexTTS-2-Demo $SpaceDir
}
Write-Host "  Space cloned!" -ForegroundColor Green

# Step 4: Download the model weights
Write-Host ""
Write-Host "[4/5] Downloading IndexTTS-2 model weights..." -ForegroundColor Yellow
$CheckpointsDir = Join-Path $SpaceDir "checkpoints"

Write-Host "  Downloading model (this may take 10-20 minutes)..." -ForegroundColor Gray
huggingface-cli download IndexTeam/IndexTTS-2 --local-dir $CheckpointsDir

Write-Host "  Model downloaded!" -ForegroundColor Green

# Step 5: Install Python dependencies
Write-Host ""
Write-Host "[5/5] Installing Python dependencies..." -ForegroundColor Yellow
Push-Location $SpaceDir

# Check if requirements.txt exists in the space
if (Test-Path "requirements.txt") {
    pip install -r requirements.txt --quiet
}

Pop-Location

# Also install our app's requirements
pip install -r (Join-Path $ProjectDir "requirements.txt") --quiet
Write-Host "  Dependencies installed!" -ForegroundColor Green

# Create symlink for checkpoints in main project
Write-Host ""
Write-Host "Creating checkpoints link..." -ForegroundColor Yellow
$LocalCheckpoints = Join-Path $ProjectDir "checkpoints"
if (-not (Test-Path $LocalCheckpoints)) {
    # Copy or link checkpoints
    New-Item -ItemType Junction -Path $LocalCheckpoints -Target $CheckpointsDir -ErrorAction SilentlyContinue
    if (-not $?) {
        Write-Host "  Creating copy (junction failed, may need admin)..." -ForegroundColor Gray
        Copy-Item -Path $CheckpointsDir -Destination $LocalCheckpoints -Recurse
    }
}
Write-Host "  Checkpoints linked!" -ForegroundColor Green

# Done!
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To run the Voice Bot:" -ForegroundColor Cyan
Write-Host "  python app.py" -ForegroundColor White
Write-Host ""
Write-Host "Or run the original demo:" -ForegroundColor Cyan
Write-Host "  cd IndexTTS-2-Demo" -ForegroundColor White
Write-Host "  python webui.py" -ForegroundColor White
Write-Host ""
