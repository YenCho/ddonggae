param(
    [string]$Workspace = 'C:\Users\user\Documents\Data_Generation_Blender',
    [string]$StatePath = 'logs\meta_v2_10000_pipeline\pipeline_state.json',
    [string]$To = 'jaeyoungi@snu.ac.kr',
    [string]$Subject = '[30min check] Meta V2 A1/A2/B/C pipeline',
    [string]$Config = 'config/email_smtp.json',
    [int]$IntervalSeconds = 1800,
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Continue'

Set-Location -LiteralPath $Workspace
$StateAbs = if ([System.IO.Path]::IsPathRooted($StatePath)) { $StatePath } else { Join-Path $Workspace $StatePath }
$LogDir = Join-Path $Workspace 'logs\meta_v2_10000_pipeline'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Host "Meta V2 pipeline email monitor"
Write-Host "Workspace: $Workspace"
Write-Host "State: $StateAbs"
Write-Host "Interval: $([Math]::Round($IntervalSeconds / 60, 1)) minutes"
Write-Host "Press Ctrl+C to stop."
Write-Host ""

$sentTerminal = $false

while ($true) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $log = Join-Path $LogDir "pipeline-status-email-$stamp.log"

    if (-not (Test-Path -LiteralPath $StateAbs)) {
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] waiting for state file..."
        Start-Sleep -Seconds ([Math]::Min($IntervalSeconds, 60))
        continue
    }

    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] sending status email..."
    & $Python scripts\send_meta_v2_pipeline_status_email.py `
        --state $StateAbs `
        --to $To `
        --subject $Subject `
        --config $Config *>&1 | Tee-Object -FilePath $log

    $status = ''
    try {
        $state = Get-Content -LiteralPath $StateAbs -Raw | ConvertFrom-Json
        $status = [string]$state.status
    } catch {
        $status = ''
    }

    if ($status -in @('completed', 'failed')) {
        if ($sentTerminal) {
            break
        }
        $sentTerminal = $true
        Write-Host "Terminal status '$status' observed; monitor will exit."
        break
    }

    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] sleeping..."
    Write-Host ""
    Start-Sleep -Seconds $IntervalSeconds
}
