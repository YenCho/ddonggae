param(
    [string]$UnprunedRoot = "datasets\cube_face_unified_finetune_meta_v2_50000_plus_hsv_ratio_20000_stronger_v1",
    [string]$PrunedRoot = "datasets\cube_face_unified_finetune_meta_v2_50000_plus_hsv_ratio_20000_stronger_pruned_coloroutlier_v1",
    [string]$ReaddRoot = "datasets\cube_face_unified_readd_coloroutlier_base_v1",
    [string]$MixRoot = "datasets\cube_face_unified_finetune_hsv_pruned_plus_readd_coloroutlier_base_v1",
    [string]$Model = "jetson\ABC_model\cube_face_unified\preferred_v2\weights\best.pt",
    [string]$ReportRoot = "reports\cube_face_unified_eval\readd_coloroutlier_base_gate_v1",
    [double]$GateRate = 0.05,
    [int]$BaseSampleCount = 70000,
    [int]$BaseValCount = 6000,
    [int]$BoosterRepeat = 1,
    [int]$Epochs = 120,
    [int]$Batch = 512,
    [int]$Workers = 8,
    [string]$Cache = "disk",
    [int]$Patience = 35,
    [double]$Lr0 = 0.00001,
    [double]$Lrf = 0.05,
    [string]$Optimizer = "AdamW",
    [string]$Device = "0",
    [string]$Project = "runs\segment",
    [string]$Name = "cube_face_unified_yolo26n_seg_readd_coloroutlier_base_from_pruned_adamw_lr1e5_ft_v1",
    [string]$EmailTo = "jaeyoungi@snu.ac.kr",
    [int]$EmailEvery = 5,
    [switch]$SkipExtract,
    [switch]$SkipEval,
    [switch]$AutoTrain,
    [switch]$ChildTrainingWindow
)

# Purpose:
# 1) Recover the color-outlier pruned REAL-texture (kind=base) crops from the existing
#    unpruned materialized dataset. No new Blender generation.
# 2) GATE: evaluate the current preferred unified model on the recovered crops.
#    Training is justified only if fruit-to-fruit top-class error >= GateRate.
# 3) If the gate passes (and -AutoTrain is set), materialize pruned+readd mix and fine-tune.

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $Root

$Python = "C:\Users\user\anaconda3\envs\ai_robotics\python.exe"
$EnvScripts = "C:\Users\user\anaconda3\envs\ai_robotics\Scripts"
$env:PATH = "$EnvScripts;$env:PATH"

$LogDir = Join-Path $Root "logs\cube_face_unified_readd_coloroutlier"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$TranscriptPath = Join-Path $LogDir "run_$Stamp.log"
$TrainLogPath = Join-Path $LogDir "train_$Stamp.log"

if ($AutoTrain -and -not $ChildTrainingWindow) {
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
    Write-Host "Readd fine-tune launched in a new PowerShell window (watch live progress there)."
    Write-Host "Run name: $Name"
    Write-Host "Live training log (tail-able): $LogDir\train_<timestamp>.log"
    Write-Host "Full transcript: $LogDir\run_<timestamp>.log"
    exit 0
}

if ($ChildTrainingWindow) {
    $Host.UI.RawUI.WindowTitle = "Cube-face unified readd fine-tune: $Name"
    Start-Transcript -Path $TranscriptPath -Append | Out-Null
}

if (-not $SkipExtract) {
    Write-Host "[1/4] Extract readd candidate (pruned-out base crops, hardlink)"
    & $Python scripts\extract_readd_coloroutlier_base_candidate.py `
        --materialized_root $UnprunedRoot `
        --pruned_root $PrunedRoot `
        --output_root $ReaddRoot `
        --include_kinds base `
        --mode hardlink `
        --reset
    if ($LASTEXITCODE -ne 0) { throw "extract failed" }
}

if (-not $SkipEval) {
    Write-Host "[2/4] Gate evaluation on recovered crops (train + val splits)"
    foreach ($split in @("train", "val")) {
        & $Python scripts\evaluate_cube_face_unified_dataset_confusion.py `
            --dataset $ReaddRoot `
            --model $Model `
            --output (Join-Path $ReportRoot $split) `
            --split $split `
            --prediction_mode fruit_for_fruit_truth `
            --device $Device `
            --batch 64
        if ($LASTEXITCODE -ne 0) { throw "gate eval ($split) failed" }
    }
}

Write-Host "[3/4] Gate decision"
$fruits = @("apple", "orange", "banana", "pineapple")
$fruitTotal = 0
$fruitWrongFruit = 0
$fruitToPlainNone = 0
$offDiag = @{}
foreach ($split in @("train", "val")) {
    $jsonPath = Join-Path $ReportRoot "$split\crop_top_class_confusion.json"
    if (-not (Test-Path $jsonPath)) { throw "missing confusion json: $jsonPath" }
    $matrix = Get-Content -LiteralPath $jsonPath -Raw | ConvertFrom-Json
    foreach ($true_ in $fruits) {
        $row = $matrix.$true_
        if ($null -eq $row) { continue }
        foreach ($pred in $row.PSObject.Properties.Name) {
            $count = [int]$row.$pred
            if ($count -eq 0) { continue }
            $fruitTotal += $count
            if ($pred -eq $true_) { continue }
            if ($fruits -contains $pred) {
                $fruitWrongFruit += $count
                $key = "$true_->$pred"
                if ($offDiag.ContainsKey($key)) { $offDiag[$key] += $count } else { $offDiag[$key] = $count }
            } else {
                $fruitToPlainNone += $count
            }
        }
    }
}
if ($fruitTotal -eq 0) { throw "no fruit crops evaluated" }
$f2fRate = [math]::Round($fruitWrongFruit / $fruitTotal, 4)
$plainRate = [math]::Round($fruitToPlainNone / $fruitTotal, 4)
$gatePassed = $f2fRate -ge $GateRate

$offDiagText = ($offDiag.GetEnumerator() | Sort-Object -Property Value -Descending | ForEach-Object { "$($_.Key): $($_.Value)" }) -join ", "
$decision = [ordered]@{
    model = $Model
    dataset = $ReaddRoot
    fruit_total = $fruitTotal
    fruit_to_fruit_wrong = $fruitWrongFruit
    fruit_to_fruit_rate = $f2fRate
    fruit_to_plain_none = $fruitToPlainNone
    fruit_to_plain_none_rate = $plainRate
    gate_rate = $GateRate
    gate_passed = $gatePassed
    off_diagonal = $offDiagText
    verdict = if ($gatePassed) { "additional fine-tune JUSTIFIED: current model is weak on recovered real-texture crops" } else { "additional fine-tune NOT justified yet: current model already handles recovered crops; keep as probe set" }
}
New-Item -ItemType Directory -Force -Path $ReportRoot | Out-Null
$decision | ConvertTo-Json | Out-File -FilePath (Join-Path $ReportRoot "gate_decision.json") -Encoding utf8
@(
    "# Readd Color-Outlier Base Gate Decision",
    "",
    "- model: ``$Model``",
    "- dataset: ``$ReaddRoot``",
    "- fruit crops evaluated: $fruitTotal",
    "- fruit-to-fruit wrong: $fruitWrongFruit ($f2fRate)",
    "- fruit-to-plain/none: $fruitToPlainNone ($plainRate)",
    "- gate threshold: $GateRate",
    "- gate passed: $gatePassed",
    "- off-diagonal: $offDiagText",
    "",
    "$($decision.verdict)"
) -join "`n" | Out-File -FilePath (Join-Path $ReportRoot "GATE_DECISION.md") -Encoding utf8

Write-Host ""
Write-Host "fruit crops: $fruitTotal, fruit-to-fruit wrong: $fruitWrongFruit ($f2fRate), plain/none: $plainRate"
Write-Host "GATE PASSED: $gatePassed (threshold $GateRate)"
Write-Host "Decision report: $ReportRoot\GATE_DECISION.md"

if (-not $gatePassed) {
    Write-Host "Training skipped: current model accuracy on the recovered set does not justify additional fine-tune."
    exit 0
}

if (-not $AutoTrain) {
    Write-Host ""
    Write-Host "[4/4] Gate passed. Re-run with -AutoTrain to materialize the mix and fine-tune:"
    Write-Host "powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_cube_face_unified_readd_gate_and_finetune.ps1 -SkipExtract -SkipEval -AutoTrain"
    exit 0
}

Write-Host "[4/4] Materialize pruned + readd mix, then fine-tune"
& $Python scripts\materialize_cube_face_unified_finetune_mix.py `
    --base_dataset $PrunedRoot `
    --booster_dataset $ReaddRoot `
    --output_root $MixRoot `
    --base_sample_count $BaseSampleCount `
    --base_val_count $BaseValCount `
    --booster_repeat $BoosterRepeat `
    --mode hardlink `
    --reset
if ($LASTEXITCODE -ne 0) { throw "materialize failed" }

if (-not [System.IO.Path]::IsPathRooted($Project)) {
    $Project = Join-Path $Root $Project
}
$DataYaml = Join-Path $MixRoot "data.yaml"
& $Python scripts\train_yolo_with_email.py `
    --task segment `
    --model $Model `
    --data $DataYaml `
    --epochs $Epochs `
    --imgsz 224 `
    --batch $Batch `
    --device $Device `
    --workers $Workers `
    --cache $Cache `
    --amp True `
    --deterministic False `
    --patience $Patience `
    --project $Project `
    --name $Name `
    --exist_ok True `
    --email_to $EmailTo `
    --email_every $EmailEvery `
    --email_config config\email_smtp.json `
    --log_file $TrainLogPath `
    "optimizer=$Optimizer" `
    "lr0=$Lr0" `
    "lrf=$Lrf" `
    "cos_lr=True" `
    "warmup_epochs=0.0" `
    "close_mosaic=0" `
    "mosaic=0.0" `
    "copy_paste=0.0" `
    "mixup=0.0"
if ($LASTEXITCODE -ne 0) { throw "training failed" }

Write-Host ""
Write-Host "Done."
Write-Host "Readd candidate: $ReaddRoot"
Write-Host "Mixed dataset: $MixRoot"
Write-Host "Run: $Project\$Name"
Write-Host "Training log: $TrainLogPath"
Write-Host "After training: re-run gate eval with the new best.pt, replay pruned val, and re-check the 2 user holdout orange crops before promoting."
if ($ChildTrainingWindow) {
    Stop-Transcript | Out-Null
}
