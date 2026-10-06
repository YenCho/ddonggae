param(
    [int]$NumImages = 1000,
    [int]$Workers = 4,
    [string]$SourceDataset = "datasets\meta_v2_smoke_1000",
    [string]$ModelOutputRoot = "datasets\meta_v2_smoke_1000_models"
)

$ErrorActionPreference = "Stop"

$Workspace = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Workspace

Write-Host "Meta V2 smoke generation"
Write-Host "Workspace: $Workspace"
Write-Host "SourceDataset: $SourceDataset"
Write-Host "ModelOutputRoot: $ModelOutputRoot"
Write-Host "NumImages: $NumImages"
Write-Host "Workers: $Workers"

python scripts\run_yolo_parallel.py `
  --output $SourceDataset `
  --num_images $NumImages `
  --workers $Workers `
  --worker_start_delay 2 `
  --width 640 `
  --height 640 `
  --samples 24 `
  --cpu_threads 1 `
  --label_format segment `
  --min_objects 1 `
  --max_objects 8 `
  --scale_min 0.45 `
  --scale_max 2.20 `
  --negative_ratio 0.02 `
  --lens_distortion_prob 0.35 `
  --robot_camera_view `
  --ideal_visibility `
  --ideal_visibility_debug `
  --ideal_visibility_max_attempts 80 `
  --meta_v2 `
  --resume

python scripts\export_meta_v2_model_datasets.py `
  --source_dataset $SourceDataset `
  --output_root $ModelOutputRoot `
  --splits train `
  --copy_mode hardlink `
  --crop_size 224 `
  --reset

Write-Host "Done."
Write-Host "Generated source dataset: $SourceDataset"
Write-Host "Generated model datasets: $ModelOutputRoot"
