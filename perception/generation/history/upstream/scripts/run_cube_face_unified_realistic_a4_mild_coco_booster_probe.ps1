param(
    [int]$Count = 100,
    [string]$Tag = "probe_100_v1",
    [string]$SourceDataset = "",
    [string]$BoosterRoot = "",
    [string]$FruitTextureDir = "datasets\fruit_textures\production_meta_v2_50000_v1_color_filtered_v2",
    [string]$Model = "runs\segment\cube_face_unified_yolo26n_seg_hsv_pruned_coloroutlier_from_last_adamw_lr1e5_ft_v1\weights\best.pt",
    [string]$FruitPrintLabelProfile = "realistic_a4_mild",
    [string]$CameraArtifactProfile = "mild_exposure",
    [string]$FruitTextureAug = "light",
    [ValidateSet("", "apple", "orange", "banana", "pineapple")]
    [string]$ForceFruitClass = "",
    [double]$EasyWeight = 0.50,
    [double]$MidWeight = 0.45,
    [double]$HardWeight = 0.05,
    [double]$ScaleMin = 0.45,
    [double]$ScaleMax = 2.20,
    [int]$MinObjects = 1,
    [int]$MaxObjects = 6,
    [double]$LensDistortionProb = 0.25,
    [double]$CropPad = 0.18,
    [int]$ExportMinObjectPixels = 80,
    [int]$ExportMinFacePixels = 120,
    [double]$ValRatio = 0.90,
    [int]$GenerationWorkers = 4,
    [int]$RenderSamples = 8,
    [int]$Batch = 64,
    [string]$Device = "0",
    [ValidateSet("all", "fruit_for_fruit_truth")]
    [string]$PredictionMode = "all",
    [string]$ReportRoot = "reports\cube_face_unified_eval",
    [switch]$SkipGenerate,
    [switch]$SkipExport,
    [switch]$SkipEval,
    [switch]$KeepGeneratedFaceTextures,
    [switch]$Reset
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $Root

$Python = "C:\Users\user\anaconda3\envs\ai_robotics\python.exe"
$EnvScripts = "C:\Users\user\anaconda3\envs\ai_robotics\Scripts"
$Yolo = Join-Path $EnvScripts "yolo.exe"
$env:PATH = "$EnvScripts;$env:PATH"

if ([string]::IsNullOrWhiteSpace($SourceDataset)) {
    $SourceDataset = "datasets\cube_face_unified_booster_realistic_a4_mild_coco_$($Tag)_source"
}
if ([string]::IsNullOrWhiteSpace($BoosterRoot)) {
    $BoosterRoot = "datasets\cube_face_unified_booster_realistic_a4_mild_coco_$Tag"
}

$LogDir = Join-Path $Root "logs\cube_face_unified_realistic_a4_mild_coco"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $LogDir "run_$Stamp.log"

function Invoke-Logged {
    param([string]$Label, [scriptblock]$Command)
    Write-Host ""
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Label"
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Label" | Out-File -FilePath $LogPath -Append -Encoding utf8
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $script:LASTEXITCODE = 0
    $ErrPath = "$LogPath.stderr.tmp"
    if (Test-Path -LiteralPath $ErrPath) {
        Remove-Item -LiteralPath $ErrPath -Force
    }
    try {
        & $Command 2> $ErrPath | Tee-Object -FilePath $LogPath -Append
    }
    finally {
        if (Test-Path -LiteralPath $ErrPath) {
            Get-Content -LiteralPath $ErrPath | Tee-Object -FilePath $LogPath -Append
            Remove-Item -LiteralPath $ErrPath -Force
        }
        $ErrorActionPreference = $oldErrorActionPreference
    }
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

function Remove-UnderRootIfRequested {
    param([string]$RelativePath)
    if (-not $Reset) {
        return
    }
    $target = Join-Path $Root $RelativePath
    if (-not (Test-Path -LiteralPath $target)) {
        return
    }
    $resolved = (Resolve-Path -LiteralPath $target).Path
    if (-not $resolved.StartsWith($Root, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove path outside project root: $resolved"
    }
    Write-Host "Reset: removing $resolved"
    Remove-Item -LiteralPath $resolved -Recurse -Force
}

Remove-UnderRootIfRequested $SourceDataset
Remove-UnderRootIfRequested $BoosterRoot

if (-not $SkipGenerate) {
    Invoke-Logged "Generate COCO/meta_v2 realistic A4 mild print-label source scenes" {
        $cmdArgs = @(
            "scripts\run_yolo_parallel.py",
            "--output", $SourceDataset,
            "--num_images", "$Count",
            "--workers", "$GenerationWorkers",
            "--worker_start_delay", "2",
            "--width", "640",
            "--height", "640",
            "--samples", "$RenderSamples",
            "--cpu_threads", "1",
            "--fruit_texture_dir", $FruitTextureDir,
            "--fruit_texture_aug", $FruitTextureAug,
            "--fruit_texture_layout", "print_label",
            "--fruit_print_label_profile", $FruitPrintLabelProfile,
            "--camera_artifact_profile", $CameraArtifactProfile,
            "--label_format", "segment",
            "--min_objects", "$MinObjects",
            "--max_objects", "$MaxObjects",
            "--scale_min", "$ScaleMin",
            "--scale_max", "$ScaleMax",
            "--negative_ratio", "0.02",
            "--fruit_visibility_easy_weight", "$EasyWeight",
            "--fruit_visibility_mid_weight", "$MidWeight",
            "--fruit_visibility_hard_weight", "$HardWeight",
            "--lens_distortion_prob", "$LensDistortionProb",
            "--render_device", "auto",
            "--robot_camera_view",
            "--ideal_visibility",
            "--ideal_visibility_max_attempts", "40",
            "--meta_v2",
            "--max_worker_retries", "12",
            "--retry_delay_seconds", "45",
            "--resume"
        )
        if (-not [string]::IsNullOrWhiteSpace($ForceFruitClass)) {
            $cmdArgs += @("--force_fruit_class", $ForceFruitClass)
        }
        if ($KeepGeneratedFaceTextures) {
            $cmdArgs += "--keep_generated_face_textures"
        }
        & $Python @cmdArgs
        <#
        & $Python scripts\run_yolo_parallel.py `
            --output $SourceDataset `
            --num_images $Count `
            --workers $GenerationWorkers `
            --worker_start_delay 2 `
            --width 640 `
            --height 640 `
            --samples $RenderSamples `
            --cpu_threads 1 `
            --fruit_texture_dir $FruitTextureDir `
            --fruit_texture_aug light `
            --fruit_texture_layout print_label `
            --fruit_print_label_profile realistic_a4_mild `
            --camera_artifact_profile mild_exposure `
            --label_format segment `
            --min_objects 1 `
            --max_objects 6 `
            --scale_min 0.45 `
            --scale_max 2.20 `
            --negative_ratio 0.02 `
            --lens_distortion_prob 0.25 `
            --render_device auto `
            --robot_camera_view `
            --ideal_visibility `
            --ideal_visibility_max_attempts 40 `
            --meta_v2 `
            --max_worker_retries 12 `
            --retry_delay_seconds 45 `
            --resume
        #>
    }
}

if (-not $SkipExport) {
    Invoke-Logged "Export source scenes to cube-face unified 224 crops" {
        & $Python scripts\export_meta_v2_cube_face_unified_dataset.py `
            --source_dataset $SourceDataset `
            --output_root $BoosterRoot `
            --source_split train `
            --val_ratio $ValRatio `
            --seed 20260705 `
            --crop_size 224 `
            --crop_pad $CropPad `
            --min_object_pixels $ExportMinObjectPixels `
            --min_face_pixels $ExportMinFacePixels `
            --min_segment_area 8.0 `
            --include_partial_faces `
            --keep_empty_crops `
            --reset
    }
}

if (-not $SkipEval) {
    $DataYaml = Join-Path $BoosterRoot "data.yaml"
    $ValName = "realistic_a4_mild_coco_$Tag"
    Invoke-Logged "YOLO val on realistic A4 mild COCO probe" {
        & $Yolo segment val `
            model=$Model `
            data=$DataYaml `
            imgsz=224 `
            batch=$Batch `
            device=$Device `
            plots=True `
            project=$ReportRoot `
            name=$ValName `
            exist_ok=True
    }
    Invoke-Logged "Crop top-class confusion summary" {
        & $Python scripts\evaluate_cube_face_unified_dataset_confusion.py `
            --dataset $BoosterRoot `
            --model $Model `
            --output (Join-Path $ReportRoot "$($ValName)_top_class") `
            --split val `
            --imgsz 224 `
            --batch $Batch `
            --device $Device `
            --prediction_mode $PredictionMode
    }
}

Write-Host ""
Write-Host "Done. Log: $LogPath"
Write-Host "Source scenes: $SourceDataset"
Write-Host "Unified booster/probe crops: $BoosterRoot"
Write-Host "YOLO val report: $(Join-Path $ReportRoot "realistic_a4_mild_coco_$Tag")"
Write-Host "Top-class report: $(Join-Path $ReportRoot "realistic_a4_mild_coco_$($Tag)_top_class")"
