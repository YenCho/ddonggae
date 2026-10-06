param(
    [string]$Workspace = 'C:\Users\user\Documents\Data_Generation_Blender',
    [string]$Dataset = 'datasets\yolo26_seg_100000_ideal_motionblur_aug',
    [string]$RunDir = 'runs\segment\yolo26s_seg_100000_motionblur_finetune_lr5e4',
    [string]$To = 'jaeyoungi@snu.ac.kr',
    [string]$Subject = '[30min check] YOLO26s motion-blur finetune status',
    [string]$TaskLabel = 'YOLO26s motion-blur finetune',
    [int]$Epochs = 30,
    [string]$Config = 'config/email_smtp.json',
    [int]$IntervalSeconds = 1800,
    [switch]$NoAttach = $true,
    [string[]]$ProcessPattern = @('yolo26s_seg_100000_motionblur_finetune_lr5e4', 'yolo26_seg_100000_ideal_motionblur_aug', 'segment train')
)

$ErrorActionPreference = 'Continue'

$LogDir = Join-Path $Workspace 'logs\yolo_training_status_email_loop'

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location -LiteralPath $Workspace

Write-Host "YOLO training status email loop"
Write-Host "Workspace: $Workspace"
Write-Host "Dataset: $Dataset"
Write-Host "RunDir: $RunDir"
Write-Host "Interval: $([Math]::Round($IntervalSeconds / 60, 1)) minutes"
Write-Host "Press Ctrl+C to stop."
Write-Host ""

while ($true) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $log = Join-Path $LogDir "status-email-$stamp.log"
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Running status email script..."

    $args = @(
        'scripts/send_yolo_50000_status_email.py',
        '--to', $To,
        '--subject', $Subject,
        '--task-label', $TaskLabel,
        '--epochs', $Epochs,
        '--config', $Config
    )
    if ($Dataset -ne '') {
        $args += @('--dataset', $Dataset)
    }
    if ($RunDir -ne '') {
        $args += @('--run-dir', $RunDir)
    }
    if ($NoAttach) {
        $args += @('--no-attach')
    }
    foreach ($pattern in $ProcessPattern) {
        if ($pattern -ne '') {
            $args += @('--process-pattern', $pattern)
        }
    }

    python @args *>&1 | Tee-Object -FilePath $log
    $exitCode = $LASTEXITCODE

    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Exit code: $exitCode"
    Write-Host "Log: $log"
    Write-Host "Sleeping 30 minutes..."
    Write-Host ""

    Start-Sleep -Seconds $IntervalSeconds
}
