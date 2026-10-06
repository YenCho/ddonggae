param(
    [int]$ExperimentId = 3,
    [int]$Seed = 20260914,
    [string]$Python = "C:\Users\user\anaconda3\envs\ai_robotics\python.exe",
    [string]$BaseCData = "datasets\meta_v2_10000_coco_v1_models\c_facecls",
    [string]$BaseCheckpoint = "runs\meta_v2_c_facecls\c_mobilenetv3small_meta_v2_10000\weights\best.pt",
    [string]$OrangeTextureRoot = "datasets\fruit_textures\orange_hard_web_exp_002",
    [int]$HardTrainCountPerClass = 8000,
    [int]$Epochs = 8,
    [int]$Batch = 1024,
    [int]$Workers = 8,
    [string]$Device = "0"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$ExpName = "exp_{0:D3}" -f $ExperimentId
$LoopRoot = "logs\c_orange_target_loop"
$ExpRoot = Join-Path $LoopRoot $ExpName
$MimicRoot = "datasets\c_facecls_apple_orange_balanced_mimic_$ExpName"
$MixedRoot = "datasets\c_facecls_mixed_apple_orange_balanced_$ExpName"
$RunName = "c_mobilenetv3small_apple_orange_balanced_$ExpName"
$RunDir = "runs\meta_v2_c_facecls\$RunName"
$LogPath = Join-Path $ExpRoot "balanced_experiment.log"
$EvalPath = Join-Path $ExpRoot "eval.json"
$Face03EvalPath = Join-Path $ExpRoot "face03_eval.json"
$DonePath = Join-Path $ExpRoot "done.json"

New-Item -ItemType Directory -Force -Path $ExpRoot | Out-Null
Start-Transcript -Path $LogPath -Append | Out-Null
try {
    Write-Host "C apple/orange balanced hard experiment $ExpName"
    Write-Host "Seed: $Seed"
    Write-Host "OrangeTextureRoot: $OrangeTextureRoot"
    Write-Host "MimicRoot: $MimicRoot"
    Write-Host "MixedRoot: $MixedRoot"
    Write-Host "RunDir: $RunDir"

    if (-not (Test-Path (Join-Path $OrangeTextureRoot "orange"))) {
        throw "clean orange texture root not found: $OrangeTextureRoot"
    }

    & "$Python" scripts\generate_2d_face_mimic_dataset.py `
        --output_root "$MimicRoot" `
        --splits train `
        --classes apple orange `
        --train_per_class $HardTrainCountPerClass `
        --val_per_class 0 `
        --test_per_class 0 `
        --extra_orange_texture_dir "$OrangeTextureRoot" `
        --extra_orange_weight 12 `
        --hard_ratio 0.90 `
        --orange_hard_ratio 0.90 `
        --white_margin_hard_ratio 0.88 `
        --orange_green_distractor_prob 0.45 `
        --motion_blur_prob 0.22 `
        --projected_resolution_prob 0.74 `
        --lens_like_distortion_prob 0.35 `
        --preview_count 16 `
        --seed $Seed `
        --reset
    if ($LASTEXITCODE -ne 0) { throw "balanced mimic generation failed: $LASTEXITCODE" }

    & "$Python" scripts\merge_facecls_datasets.py `
        --inputs "$BaseCData" "$MimicRoot" `
        --output_root "$MixedRoot" `
        --splits train `
        --mode hardlink `
        --reset
    if ($LASTEXITCODE -ne 0) { throw "merge failed: $LASTEXITCODE" }

    & "$Python" scripts\train_face_mobilenetv3.py `
        --data "$MixedRoot" `
        --epochs $Epochs `
        --batch $Batch `
        --imgsz 128 `
        --lr 0.00005 `
        --weight_decay 0.0001 `
        --device "$Device" `
        --workers $Workers `
        --prefetch_factor 6 `
        --amp `
        --channels_last `
        --init_checkpoint "$BaseCheckpoint" `
        --project "runs\meta_v2_c_facecls" `
        --name "$RunName"
    if ($LASTEXITCODE -ne 0) { throw "train failed: $LASTEXITCODE" }

    & "$Python" scripts\evaluate_c_target_crop.py `
        --checkpoint "$RunDir\weights\best.pt" `
        --output_json "$EvalPath" `
        --device "$Device"
    if ($LASTEXITCODE -ne 0) { throw "eval failed: $LASTEXITCODE" }

    $face03 = Get-ChildItem -Path $Root -Filter "*cand01_face03_C_output.png" | Select-Object -First 1
    if ($face03) {
        & "$Python" scripts\evaluate_c_target_crop.py `
            --checkpoint "$RunDir\weights\best.pt" `
            --target_image "$($face03.FullName)" `
            --skip_val `
            --output_json "$Face03EvalPath" `
            --device "$Device"
        if ($LASTEXITCODE -ne 0) { throw "face03 eval failed: $LASTEXITCODE" }
    }

    $done = [ordered]@{
        experiment_id = $ExperimentId
        seed = $Seed
        status = "completed"
        completed_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        run_dir = $RunDir
        eval_json = $EvalPath
        face03_eval_json = if ($face03) { $Face03EvalPath } else { $null }
        log = $LogPath
        note = "Balanced apple/orange hard mimic: same generated count per class."
    }
    $done | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $DonePath
}
catch {
    $done = [ordered]@{
        experiment_id = $ExperimentId
        seed = $Seed
        status = "failed"
        failed_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        error = "$_"
        run_dir = $RunDir
        eval_json = $EvalPath
        face03_eval_json = $Face03EvalPath
        log = $LogPath
    }
    $done | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 $DonePath
    throw
}
finally {
    Stop-Transcript | Out-Null
}
