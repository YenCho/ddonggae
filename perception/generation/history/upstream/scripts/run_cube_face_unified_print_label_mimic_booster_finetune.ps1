param(
    [string]$SourceDataset = "datasets\cube_face_unified_booster_coco_print_mimic_v1_source",
    [string]$BoosterRoot = "datasets\cube_face_unified_booster_coco_print_mimic_v1",
    [string]$MaterializedMixRoot = "datasets\cube_face_unified_finetune_hsv_pruned_plus_coco_print_mimic_v1",
    [string]$BaseDataset = "datasets\cube_face_unified_finetune_meta_v2_50000_plus_hsv_ratio_20000_stronger_pruned_coloroutlier_v1",
    [string]$FruitTextureDir = "datasets\fruit_textures\production_meta_v2_50000_v1_color_filtered_v2",
    [string]$InitModel = "runs\segment\cube_face_unified_yolo26n_seg_hsv_pruned_coloroutlier_from_last_adamw_lr1e5_ft_v1\weights\best.pt",
    [int]$SourceImages = 12000,
    [int]$GenerationWorkers = 8,
    [int]$RenderSamples = 24,
    [int]$BaseSampleCount = 70000,
    [int]$BaseValCount = 6000,
    [int]$BoosterRepeat = 1,
    [int]$Epochs = 120,
    [int]$Batch = 512,
    [int]$Workers = 4,
    [int]$Patience = 35,
    [double]$Lr0 = 0.00001,
    [double]$Lrf = 0.05,
    [string]$Optimizer = "AdamW",
    [string]$Device = "0",
    [string]$Project = "runs\segment",
    [string]$Name = "cube_face_unified_yolo26n_seg_coco_print_mimic_from_hsv_pruned_adamw_lr1e5_ft_v1",
    [string]$EmailTo = "jaeyoungi2006@gmail.com",
    [int]$EmailEvery = 5,
    [switch]$SkipGenerate,
    [switch]$SkipExport,
    [switch]$SkipMaterialize,
    [switch]$SkipTrain,
    [switch]$ChildTrainingWindow
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $Root

$Python = "C:\Users\user\anaconda3\envs\ai_robotics\python.exe"
$EnvScripts = "C:\Users\user\anaconda3\envs\ai_robotics\Scripts"
$env:PATH = "$EnvScripts;$env:PATH"

if (-not $SkipTrain -and -not $ChildTrainingWindow) {
    $ChildArgs = @("-NoExit", "-ExecutionPolicy", "Bypass", "-File", $PSCommandPath)
    foreach ($entry in $PSBoundParameters.GetEnumerator()) {
        if ($entry.Key -eq "ChildTrainingWindow") {
            continue
        }
        $ChildArgs += "-$($entry.Key)"
        if ($entry.Value -is [System.Management.Automation.SwitchParameter]) {
            continue
        }
        $ChildArgs += [string]$entry.Value
    }
    $ChildArgs += "-ChildTrainingWindow"
    Start-Process -FilePath "powershell.exe" -ArgumentList $ChildArgs -WorkingDirectory $Root -WindowStyle Normal
    Write-Host "COCO print-mimic booster fine-tune launched in a new PowerShell window."
    Write-Host "Run name: $Name"
    exit 0
}

$LogDir = Join-Path $Root "logs\cube_face_unified_coco_print_mimic"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $LogDir "run_$Stamp.log"
if ($ChildTrainingWindow) {
    $Host.UI.RawUI.WindowTitle = "Cube-face unified COCO print-mimic: $Name"
}

function Invoke-Logged {
    param([string]$Label, [scriptblock]$Command)
    Write-Host ""
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Label"
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Label" | Out-File -FilePath $LogPath -Append -Encoding utf8
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $script:LASTEXITCODE = 0
    $transcriptStarted = $false
    if ($ChildTrainingWindow) {
        try {
            Start-Transcript -Path $LogPath -Append | Out-Null
            $transcriptStarted = $true
            & $Command
        }
        finally {
            if ($transcriptStarted) {
                Stop-Transcript | Out-Null
            }
            $ErrorActionPreference = $oldErrorActionPreference
        }
    } else {
        try {
            & $Command *>> $LogPath
        }
        finally {
            $ErrorActionPreference = $oldErrorActionPreference
        }
        Get-Content -LiteralPath $LogPath -Tail 40
    }
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

if (-not $SkipGenerate) {
    Invoke-Logged "Generate COCO/meta_v2 print-mimic source scenes" {
        & $Python scripts\run_yolo_parallel.py `
            --output $SourceDataset `
            --num_images $SourceImages `
            --workers $GenerationWorkers `
            --worker_start_delay 2 `
            --width 640 `
            --height 640 `
            --samples $RenderSamples `
            --cpu_threads 1 `
            --fruit_texture_dir $FruitTextureDir `
            --fruit_texture_aug strong `
            --fruit_texture_layout print_mimic `
            --label_format segment `
            --min_objects 1 `
            --max_objects 8 `
            --scale_min 0.45 `
            --scale_max 2.20 `
            --negative_ratio 0.02 `
            --lens_distortion_prob 0.35 `
            --render_device auto `
            --robot_camera_view `
            --ideal_visibility `
            --ideal_visibility_max_attempts 80 `
            --meta_v2 `
            --process_chunk_size 250 `
            --max_worker_retries 30 `
            --retry_delay_seconds 60 `
            --resume
    }
}

if (-not $SkipExport) {
    Invoke-Logged "Export COCO print-mimic scenes to cube-face unified crops" {
        & $Python scripts\export_meta_v2_cube_face_unified_dataset.py `
            --source_dataset $SourceDataset `
            --output_root $BoosterRoot `
            --source_split train `
            --val_ratio 0.10 `
            --seed 20260704 `
            --crop_size 224 `
            --crop_pad 0.18 `
            --min_object_pixels 80 `
            --min_face_pixels 120 `
            --min_segment_area 8.0 `
            --include_partial_faces `
            --keep_empty_crops `
            --reset
    }
    Invoke-Logged "Create booster preview contact sheet" {
        & $Python scripts\make_yolo_bbox_preview.py `
            --dataset $BoosterRoot `
            --split train `
            --count 96 `
            --sheet_count 96 `
            --sheet_columns 8 `
            --sheet_thumb_size 144
    }
}

if (-not $SkipMaterialize) {
    Invoke-Logged "Materialize latest base plus COCO print-mimic booster" {
        & $Python scripts\materialize_cube_face_unified_finetune_mix.py `
            --base_dataset $BaseDataset `
            --booster_dataset $BoosterRoot `
            --output_root $MaterializedMixRoot `
            --base_sample_count $BaseSampleCount `
            --base_val_count $BaseValCount `
            --booster_repeat $BoosterRepeat `
            --mode hardlink `
            --reset
    }
}

if (-not $SkipTrain) {
    if (-not [System.IO.Path]::IsPathRooted($Project)) {
        $Project = Join-Path $Root $Project
    }
    $DataYaml = Join-Path $MaterializedMixRoot "data.yaml"
    Invoke-Logged "Fine-tune cube-face unified model with COCO print-mimic booster" {
        & $Python scripts\train_yolo_with_email.py `
            --task segment `
            --model $InitModel `
            --data $DataYaml `
            --epochs $Epochs `
            --imgsz 224 `
            --batch $Batch `
            --device $Device `
            --workers $Workers `
            --cache False `
            --amp True `
            --deterministic False `
            --patience $Patience `
            --project $Project `
            --name $Name `
            --exist_ok True `
            --email_to $EmailTo `
            --email_every $EmailEvery `
            --email_config config\email_smtp.json `
            "optimizer=$Optimizer" `
            "lr0=$Lr0" `
            "lrf=$Lrf" `
            "cos_lr=True" `
            "warmup_epochs=0.0" `
            "close_mosaic=0" `
            "mosaic=0.0" `
            "copy_paste=0.0" `
            "mixup=0.0"
    }
}

Write-Host ""
Write-Host "Done. Log: $LogPath"
Write-Host "Source scenes: $SourceDataset"
Write-Host "Booster crops: $BoosterRoot"
Write-Host "Mixed dataset: $MaterializedMixRoot"
Write-Host "Run: $Project\$Name"
