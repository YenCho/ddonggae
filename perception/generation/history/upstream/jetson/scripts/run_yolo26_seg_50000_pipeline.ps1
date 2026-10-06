$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$Dataset = "datasets\yolo26_seg_50000_ideal_debug"
$RunName = "yolo26n_seg_50000_ideal_debug"
$EmailTo = "jaeyoungi@snu.ac.kr"

Write-Host "Starting YOLO26n segmentation pipeline for 50,000 images..."
Write-Host "Repository: $Repo"
Write-Host "Dataset: $Dataset"
Write-Host "Run name: $RunName"

conda run -n ai_robotics python scripts\run_yolo26_seg_pipeline.py `
  --dataset $Dataset `
  --total 50000 `
  --train_count 40000 `
  --val_count 5000 `
  --test_count 5000 `
  --gen_workers 9 `
  --worker_start_delay 2 `
  --resume_generate `
  --email_to $EmailTo `
  --email_every 1000 `
  --epochs 200 `
  --model yolo26n-seg.pt `
  --batch 32 `
  --device 0 `
  --name $RunName

if ($LASTEXITCODE -ne 0) {
  throw "YOLO26n segmentation pipeline failed with exit code $LASTEXITCODE"
}

Write-Host "YOLO26n segmentation 50,000-image pipeline finished successfully."
