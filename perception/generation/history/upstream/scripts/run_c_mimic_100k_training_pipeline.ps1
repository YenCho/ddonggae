param(
    [string]$Python = "C:\Users\user\anaconda3\envs\ai_robotics\python.exe",
    [string]$BaseCData = "datasets\meta_v2_10000_coco_v1_models\c_facecls",
    [string]$MimicRoot = "datasets\c_facecls_2d_mimic_dirty_100k_v1",
    [string]$MixedRoot = "datasets\c_facecls_mixed_meta_v2_10000_plus_mimic_dirty_100k_v1",
    [string]$Project = "runs\meta_v2_c_facecls",
    [string]$RunName = "c_mobilenetv3small_mixed_mimic_dirty_100k_v1",
    [string]$InitCheckpoint = "runs\meta_v2_c_facecls\c_mobilenetv3small_meta_v2_10000\weights\best.pt",
    [int]$TrainPerClass = 15000,
    [int]$Epochs = 40,
    [int]$Batch = 1024,
    [int]$Workers = 8,
    [int]$ImgSz = 128,
    [string]$Device = "0",
    [string]$EmailTo = ""
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$LogDir = Join-Path $Root "logs\c_mimic_100k_training"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Transcript = Join-Path $LogDir "pipeline-$Stamp.log"

function Invoke-Step {
    param(
        [string]$Name,
        [string]$Command
    )
    $start = Get-Date
    Write-Host ""
    Write-Host "[$($start.ToString('yyyy-MM-dd HH:mm:ss'))] START $Name"
    Write-Host $Command
    powershell -NoProfile -ExecutionPolicy Bypass -Command $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
    $end = Get-Date
    $elapsed = New-TimeSpan -Start $start -End $end
    Write-Host "[$($end.ToString('yyyy-MM-dd HH:mm:ss'))] DONE $Name elapsed=$($elapsed.ToString())"
}

Start-Transcript -Path $Transcript -Append | Out-Null
try {
    Write-Host "C mimic 100k training pipeline"
    Write-Host "Root: $Root"
    Write-Host "MimicRoot: $MimicRoot"
    Write-Host "MixedRoot: $MixedRoot"
    Write-Host "Run: $Project\$RunName"
    Write-Host "TrainPerClass: $TrainPerClass with apple/orange boost -> about 100k mimic train images"
    Write-Host "Training: epochs=$Epochs batch=$Batch workers=$Workers imgsz=$ImgSz device=$Device AMP/channels_last enabled"

    $generate = "& `"$Python`" scripts\generate_2d_face_mimic_dataset.py --output_root `"$MimicRoot`" --splits train --train_per_class $TrainPerClass --val_per_class 0 --test_per_class 0 --apple_orange_boost --preview_count 16 --reset"
    Invoke-Step -Name "01_generate_dirty_mimic_100k" -Command $generate

    $merge = "& `"$Python`" scripts\merge_facecls_datasets.py --inputs `"$BaseCData`" `"$MimicRoot`" --output_root `"$MixedRoot`" --splits train --mode hardlink --reset"
    Invoke-Step -Name "02_merge_base_plus_mimic" -Command $merge

    $emailArgs = ""
    if ($EmailTo) {
        $emailArgs = "--email_to `"$EmailTo`" --email_every 5 --email_config config\email_smtp.json"
    }

    $train = "& `"$Python`" scripts\train_face_mobilenetv3.py --data `"$MixedRoot`" --epochs $Epochs --batch $Batch --imgsz $ImgSz --lr 0.00025 --weight_decay 0.0001 --device `"$Device`" --workers $Workers --prefetch_factor 6 --amp --channels_last --init_checkpoint `"$InitCheckpoint`" --project `"$Project`" --name `"$RunName`" $emailArgs"
    Invoke-Step -Name "03_train_c_mimic_finetune" -Command $train

    Write-Host ""
    Write-Host "Pipeline complete."
    Write-Host "Run dir: $Project\$RunName"
    Write-Host "Log: $Transcript"
}
finally {
    Stop-Transcript | Out-Null
}
