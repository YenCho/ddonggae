param(
    [string]$Python = "C:\Users\user\anaconda3\envs\ai_robotics\python.exe",
    [string]$BaseCData = "datasets\meta_v2_10000_coco_v1_models\c_facecls",
    [string]$BaseCheckpoint = "runs\meta_v2_c_facecls\c_mobilenetv3small_meta_v2_10000\weights\best.pt",
    [double]$TargetMinProb = 0.50,
    [double]$MinOriginalValAcc = 0.95,
    [int]$MaxExperiments = 50,
    [int]$DownloadCount = 300,
    [int]$OrangeTrainCount = 8000,
    [int]$Epochs = 8,
    [int]$Batch = 1024,
    [int]$Workers = 8,
    [string]$Device = "0",
    [switch]$ForceNext
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$LoopRoot = Join-Path $Root "logs\c_orange_target_loop"
$StatePath = Join-Path $LoopRoot "state.json"
New-Item -ItemType Directory -Force -Path $LoopRoot | Out-Null

function Read-JsonFile([string]$Path) {
    if (-not (Test-Path $Path)) {
        return $null
    }
    return Get-Content -Raw -Encoding UTF8 $Path | ConvertFrom-Json
}

function Write-JsonFile([string]$Path, [object]$Data) {
    $Data | ConvertTo-Json -Depth 12 | Set-Content -Encoding UTF8 $Path
}

function Get-ExpPath([int]$ExperimentId, [string]$Leaf) {
    $expName = "exp_{0:D3}" -f $ExperimentId
    return Join-Path (Join-Path $LoopRoot $expName) $Leaf
}

function Test-ProcessAlive([int]$PidValue) {
    if ($PidValue -le 0) {
        return $false
    }
    try {
        $null = Get-Process -Id $PidValue -ErrorAction Stop
        return $true
    }
    catch {
        return $false
    }
}

function Test-ExperimentSuccess([string]$EvalPath, [double]$TargetThreshold, [double]$ValThreshold) {
    if (-not (Test-Path $EvalPath)) {
        return [ordered]@{
            success = $false
            reason = "eval_json_missing"
        }
    }
    $eval = Read-JsonFile $EvalPath
    $primary = $eval.target.top_label_whitened
    if (-not $primary -or $primary.Count -eq 0) {
        return [ordered]@{
            success = $false
            reason = "top_label_whitened_missing"
        }
    }

    $topClass = [string]$primary[0].class
    $topProb = [double]$primary[0].prob
    $valAcc = $null
    $valOk = $true
    if ($eval.PSObject.Properties.Name -contains "original_val" -and $eval.original_val) {
        $valAcc = [double]$eval.original_val.acc
        $valOk = $valAcc -ge $ValThreshold
    }

    $targetOk = ($topClass -eq "orange" -and $topProb -ge $TargetThreshold)
    return [ordered]@{
        success = ($targetOk -and $valOk)
        reason = if ($targetOk -and $valOk) { "target_orange_and_val_ok" } else { "target_or_val_not_ok" }
        target_top_class = $topClass
        target_top_prob = $topProb
        target_threshold = $TargetThreshold
        original_val_acc = $valAcc
        original_val_threshold = $ValThreshold
    }
}

function Get-NextExperimentId() {
    $existing = @(Get-ChildItem -Path $LoopRoot -Directory -Filter "exp_*" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^exp_(\d+)$' } |
        ForEach-Object { [int]$Matches[1] })
    if ($existing.Count -eq 0) {
        return 1
    }
    return [int](($existing | Measure-Object -Maximum).Maximum + 1)
}

$state = Read-JsonFile $StatePath
if ($state -and $state.status -eq "success" -and -not $ForceNext) {
    Write-Host "C orange target loop already succeeded."
    Write-Host "Experiment: $($state.current_experiment_id)"
    Write-Host "Checkpoint: $($state.success_checkpoint)"
    Write-Host "Eval: $($state.eval_json)"
    exit 0
}

if ($state -and $state.status -eq "running" -and -not $ForceNext) {
    $pidValue = [int]$state.pid
    if (Test-ProcessAlive $pidValue) {
        Write-Host "Experiment exp_$('{0:D3}' -f [int]$state.current_experiment_id) is still running. PID=$pidValue"
        Write-Host "Log: $($state.log)"
        exit 0
    }

    $donePath = [string]$state.done_json
    if (Test-Path $donePath) {
        $done = Read-JsonFile $donePath
        if ($done.status -eq "completed") {
            $result = Test-ExperimentSuccess ([string]$done.eval_json) $TargetMinProb $MinOriginalValAcc
            if ($result.success) {
                $successState = [ordered]@{
                    status = "success"
                    current_experiment_id = [int]$state.current_experiment_id
                    pid = [int]$state.pid
                    seed = [int]$state.seed
                    started_at = [string]$state.started_at
                    completed_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
                    target_success_rule = [string]$state.target_success_rule
                    original_val_rule = [string]$state.original_val_rule
                    done_json = [string]$donePath
                    eval_json = [string]$done.eval_json
                    log = [string]$state.log
                    stdout = [string]$state.stdout
                    stderr = [string]$state.stderr
                    success_checkpoint = (Join-Path ([string]$done.run_dir) "weights\best.pt")
                    success_result = $result
                    parameters = $state.parameters
                }
                Write-JsonFile $StatePath $successState
                Write-Host "SUCCESS: target crop is orange and original validation is OK."
                Write-Host "Checkpoint: $($successState.success_checkpoint)"
                exit 0
            }

            Write-Host "Previous experiment completed but did not pass:"
            Write-Host ($result | ConvertTo-Json -Depth 8)
        }
        else {
            Write-Host "Previous experiment failed: $($done.error)"
        }
    }
    else {
        Write-Host "Previous process ended without done.json; starting next experiment."
    }
}

$nextId = [int](Get-NextExperimentId)
if ($nextId -gt $MaxExperiments) {
    $blocked = [ordered]@{
        status = "stopped"
        stopped_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        reason = "max_experiments_reached"
        max_experiments = $MaxExperiments
    }
    Write-JsonFile $StatePath $blocked
    throw "Reached MaxExperiments=$MaxExperiments without success."
}

$seed = 20260623 + ($nextId * 97)
$expName = "exp_{0:D3}" -f $nextId
$expRoot = Join-Path $LoopRoot $expName
New-Item -ItemType Directory -Force -Path $expRoot | Out-Null

$stdoutPath = Join-Path $expRoot "launcher.out.log"
$stderrPath = Join-Path $expRoot "launcher.err.log"
$scriptPath = Join-Path $Root "scripts\run_c_orange_target_experiment.ps1"

$argList = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$scriptPath`"",
    "-ExperimentId", "$nextId",
    "-Seed", "$seed",
    "-Python", "`"$Python`"",
    "-BaseCData", "`"$BaseCData`"",
    "-BaseCheckpoint", "`"$BaseCheckpoint`"",
    "-DownloadCount", "$DownloadCount",
    "-OrangeTrainCount", "$OrangeTrainCount",
    "-Epochs", "$Epochs",
    "-Batch", "$Batch",
    "-Workers", "$Workers",
    "-Device", "`"$Device`""
)

$proc = Start-Process -FilePath "powershell.exe" `
    -ArgumentList $argList `
    -WorkingDirectory $Root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -PassThru

$newState = [ordered]@{
    status = "running"
    current_experiment_id = $nextId
    pid = $proc.Id
    seed = $seed
    started_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    target_success_rule = "top_label_whitened top1 == orange and prob >= $TargetMinProb"
    original_val_rule = "original_val.acc >= $MinOriginalValAcc"
    done_json = (Get-ExpPath $nextId "done.json")
    eval_json = (Get-ExpPath $nextId "eval.json")
    log = (Get-ExpPath $nextId "experiment.log")
    stdout = $stdoutPath
    stderr = $stderrPath
    parameters = [ordered]@{
        download_count = $DownloadCount
        orange_train_count = $OrangeTrainCount
        epochs = $Epochs
        batch = $Batch
        workers = $Workers
        device = $Device
    }
}
Write-JsonFile $StatePath $newState

Write-Host "Started C orange target experiment $expName"
Write-Host "PID: $($proc.Id)"
Write-Host "Log: $($newState.log)"
