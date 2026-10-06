param(
    [int]$Count = 10000,
    [string]$BoosterRoot = "datasets\cube_face_unified_booster_wholefruit_plainhard_v1",
    [string]$MixRoot = "datasets\cube_face_unified_finetune_wholefruit_plainhard_v1",
    [string]$MaterializedMixRoot = "datasets\cube_face_unified_finetune_wholefruit_plainhard_materialized_v1",
    [string]$BaseDataset = "datasets\meta_v2_50000_cube_face_unified_v1",
    [string]$InitModel = "runs\segment\cube_face_unified_yolo26n_seg_v1\weights\best.pt",
    [string[]]$TextureRoots = @("datasets\fruit_textures\final_fruits36070_original30_color_filtered_v2"),
    [int]$BaseSampleCount = 22000,
    [int]$BaseValCount = 2500,
    [int]$BoosterRepeat = 2,
    [double]$AppleWeight = 1.0,
    [double]$OrangeWeight = 1.0,
    [double]$BananaWeight = 1.0,
    [double]$PineappleWeight = 1.0,
    [double]$PlainRatio = 0.46,
    [double]$BackgroundEmptyRatio = 0.07,
    [int]$Epochs = 120,
    [int]$Batch = 512,
    [int]$Workers = 4,
    [int]$Patience = 30,
    [double]$Lr0 = 0.00005,
    [double]$Lrf = 0.05,
    [string]$Optimizer = "AdamW",
    [double]$WarmupEpochs = 0.0,
    [string]$Device = "0",
    [string]$Project = "runs\segment",
    [string]$Name = "cube_face_unified_yolo26n_seg_wholefruit_plainhard_materialized_ft_v1",
    [string]$EmailTo = "roboticsai891@gmail.com",
    [int]$EmailEvery = 5,
    [int]$EmailIntervalSeconds = 1800,
    [switch]$SkipGenerate,
    [switch]$SkipMaterialize,
    [switch]$UseListMix,
    [switch]$DisableEmailMonitor,
    [switch]$SkipTrain,
    [switch]$ChildTrainingWindow
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
if (-not [System.IO.Path]::IsPathRooted($Project)) {
    $Project = Join-Path $Root $Project
}

$Python = "C:\Users\user\anaconda3\envs\ai_robotics\python.exe"
$EnvScripts = "C:\Users\user\anaconda3\envs\ai_robotics\Scripts"
$env:PATH = "$EnvScripts;$env:PATH"
$Yolo = Join-Path $EnvScripts "yolo.exe"

if (-not $SkipTrain -and -not $ChildTrainingWindow) {
    $ChildArgs = @(
        "-NoExit",
        "-ExecutionPolicy", "Bypass",
        "-File", $PSCommandPath
    )
    foreach ($entry in $PSBoundParameters.GetEnumerator()) {
        if ($entry.Key -eq "ChildTrainingWindow") {
            continue
        }
        $ChildArgs += "-$($entry.Key)"
        if ($entry.Value -is [System.Management.Automation.SwitchParameter]) {
            continue
        }
        if ($entry.Value -is [System.Array]) {
            foreach ($item in $entry.Value) {
                $ChildArgs += [string]$item
            }
        } else {
            $ChildArgs += [string]$entry.Value
        }
    }
    $ChildArgs += "-ChildTrainingWindow"
    Start-Process -FilePath "powershell.exe" -ArgumentList $ChildArgs -WorkingDirectory $Root -WindowStyle Normal
    Write-Host "Training was launched in a new PowerShell window."
    Write-Host "Run name: $Name"
    exit 0
}

$LogDir = Join-Path $Root "logs\cube_face_unified_booster_finetune"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogPath = Join-Path $LogDir "run_$Stamp.log"
if ($ChildTrainingWindow) {
    $Host.UI.RawUI.WindowTitle = "Cube-face unified training: $Name"
}

function Invoke-Logged {
    param([string]$Label, [scriptblock]$Command)
    Write-Host ""
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Label"
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Label" | Out-File -FilePath $LogPath -Append -Encoding utf8
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $transcriptStarted = $false
    try {
        if ($ChildTrainingWindow) {
            Start-Transcript -Path $LogPath -Append | Out-Null
            $transcriptStarted = $true
            & $Command
        } else {
            & $Command *>> $LogPath
        }
        $exitCode = $LASTEXITCODE
    }
    finally {
        if ($transcriptStarted) {
            Stop-Transcript | Out-Null
        }
        $ErrorActionPreference = $oldErrorActionPreference
    }
    if (-not $ChildTrainingWindow) {
        Get-Content -Path $LogPath -Tail 40
    }
    if ($exitCode -ne 0) {
        throw "$Label failed with exit code $exitCode"
    }
}

if (-not $SkipGenerate) {
    $GenerateArgs = @(
        "scripts\make_cube_face_unified_booster.py",
        "--output_root", $BoosterRoot,
        "--base_dataset", $BaseDataset,
        "--texture_roots"
    ) + $TextureRoots + @(
        "--count", $Count,
        "--preview_count", 128,
        "--reset",
        "--make_mix",
        "--mix_output", $MixRoot,
        "--base_sample_count", $BaseSampleCount,
        "--base_val_count", $BaseValCount,
        "--booster_repeat", $BoosterRepeat,
        "--apple_weight", $AppleWeight,
        "--orange_weight", $OrangeWeight,
        "--banana_weight", $BananaWeight,
        "--pineapple_weight", $PineappleWeight,
        "--plain_ratio", $PlainRatio,
        "--background_empty_ratio", $BackgroundEmptyRatio
    )
    Invoke-Logged "Generate whole-fruit/plain-hard booster and mixed fine-tune list" {
        & $Python @GenerateArgs
    }
}

if (-not $UseListMix -and -not $SkipMaterialize) {
    $MaterializedDataYaml = Join-Path $MaterializedMixRoot "data.yaml"
    if (-not (Test-Path $MaterializedDataYaml)) {
        Invoke-Logged "Materialize mixed fine-tune dataset as normal YOLO folders" {
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
}

if (-not $SkipTrain) {
    if ($UseListMix) {
        $DataYaml = Join-Path $MixRoot "data.yaml"
    } else {
        $DataYaml = Join-Path $MaterializedMixRoot "data.yaml"
    }
    $RunDir = Join-Path $Project $Name
    $MonitorProc = $null
    if ($EmailTo -ne "" -and -not $DisableEmailMonitor) {
        $MonitorArgs = @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", (Join-Path $Root "scripts\send_yolo_50000_status_email_loop.ps1"),
            "-Workspace", $Root,
            "-Dataset", $DataYaml,
            "-RunDir", $RunDir,
            "-To", $EmailTo,
            "-Subject", "[30min check] cube-face unified whole-fruit/plain-hard fine-tune",
            "-TaskLabel", "cube-face unified whole-fruit/plain-hard fine-tune",
            "-Epochs", $Epochs,
            "-Config", "config\email_smtp.json",
            "-IntervalSeconds", $EmailIntervalSeconds,
            "-NoAttach",
            "-ProcessPattern", $Name
        )
        $MonitorProc = Start-Process -FilePath "powershell.exe" -ArgumentList $MonitorArgs -WorkingDirectory $Root -WindowStyle Hidden -PassThru
        "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Started status email monitor pid=$($MonitorProc.Id)" | Out-File -FilePath $LogPath -Append -Encoding utf8
    }

    try {
        Invoke-Logged "Fine-tune cube-face unified YOLO model from current best.pt" {
            & $Yolo segment train `
                "model=$InitModel" `
                "data=$DataYaml" `
                "epochs=$Epochs" `
                "imgsz=224" `
                "batch=$Batch" `
                "device=$Device" `
                "workers=$Workers" `
                "cache=False" `
                "amp=True" `
                "deterministic=False" `
                "patience=$Patience" `
                "project=$Project" `
                "name=$Name" `
                "exist_ok=True" `
                "optimizer=$Optimizer" `
                "lr0=$Lr0" `
                "lrf=$Lrf" `
                "cos_lr=True" `
                "warmup_epochs=$WarmupEpochs" `
                "close_mosaic=0" `
                "mosaic=0.0" `
                "copy_paste=0.0" `
                "mixup=0.0"
        }
    }
    finally {
        if ($MonitorProc -and -not $MonitorProc.HasExited) {
            Stop-Process -Id $MonitorProc.Id -Force -ErrorAction SilentlyContinue
            "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Stopped status email monitor pid=$($MonitorProc.Id)" | Out-File -FilePath $LogPath -Append -Encoding utf8
        }
    }
}

Write-Host ""
Write-Host "Done. Log: $LogPath"
