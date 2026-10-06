param(
    [string]$Workspace = 'C:\Users\user\Documents\Data_Generation_Blender',
    [string]$Dataset = 'datasets\yolo26_seg_100000_ideal_debug\_generated_all',
    [string]$To = 'jaeyoungi@snu.ac.kr',
    [int]$IntervalSeconds = 60,
    [int]$IdleArmSeconds = 300,
    [int]$ActiveThresholdSeconds = 20,
    [int]$StallMinutes = 30,
    [string]$Config = 'config/email_smtp.json',
    [string[]]$ProtectedPath = @(
        'datasets\yolo26_seg_100000_ideal_debug',
        'datasets\yolo26n_seg_100000_ideal_debug',
        'runs\yolo26n_seg_100000_ideal_debug',
        'runs\detect',
        'weights'
    )
)

$ErrorActionPreference = 'Continue'

$Python = 'C:\Users\user\anaconda3\envs\ai_robotics\python.exe'
$LogDir = Join-Path $Workspace 'logs\generation_safety_watch'
$LogPath = Join-Path $LogDir 'watch.log'
$StatePath = Join-Path $LogDir 'state.json'

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location -LiteralPath $Workspace

Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class IdleTime {
    [StructLayout(LayoutKind.Sequential)]
    public struct LASTINPUTINFO {
        public uint cbSize;
        public uint dwTime;
    }
    [DllImport("user32.dll")]
    public static extern bool GetLastInputInfo(ref LASTINPUTINFO plii);
    public static uint GetIdleMilliseconds() {
        LASTINPUTINFO lii = new LASTINPUTINFO();
        lii.cbSize = (uint)System.Runtime.InteropServices.Marshal.SizeOf(typeof(LASTINPUTINFO));
        GetLastInputInfo(ref lii);
        return ((uint)Environment.TickCount - lii.dwTime);
    }
}
"@

function Write-WatchLog($Message) {
    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $line = "[$stamp] $Message"
    Write-Host $line
    Add-Content -LiteralPath $LogPath -Value $line -Encoding UTF8
}

function Send-WatchEmail($Subject, $Body) {
    try {
        & $Python scripts\send_progress_email.py --config $Config --to $To --subject $Subject --body $Body | Out-Null
        Write-WatchLog "sent email: $Subject"
    } catch {
        Write-WatchLog "email failed: $($_.Exception.Message)"
    }
}

function Count-Files($Path, $Pattern) {
    if (-not (Test-Path -LiteralPath $Path)) { return 0 }
    return (Get-ChildItem -File -LiteralPath $Path -Filter $Pattern -ErrorAction SilentlyContinue | Measure-Object).Count
}

function Get-GenerationState {
    $datasetPath = if ([System.IO.Path]::IsPathRooted($Dataset)) { $Dataset } else { Join-Path $Workspace $Dataset }
    $imageDir = Join-Path $datasetPath 'images\train'
    $labelDir = Join-Path $datasetPath 'labels\train'
    $imageCount = Count-Files $imageDir '*.jpg'
    $labelCount = Count-Files $labelDir '*.txt'
    $latestImage = Get-ChildItem -File -LiteralPath $imageDir -Filter '*.jpg' -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    return [pscustomobject]@{
        Time = Get-Date
        DatasetPath = $datasetPath
        ImageCount = $imageCount
        LabelCount = $labelCount
        LatestImage = if ($latestImage) { $latestImage.Name } else { '' }
        LatestImageTime = if ($latestImage) { $latestImage.LastWriteTime } else { $null }
    }
}

function Get-GenerationProcesses {
    try {
        $rows = Get-CimInstance Win32_Process | Where-Object {
            $_.Name -in @('python.exe','blender.exe','blenderproc.exe','conda.exe','powershell.exe') -and
            $null -ne $_.CommandLine -and
            $_.CommandLine -match 'run_yolov8_seg|run_yolo_parallel|generate_yolo|blenderproc|run_yolov8_seg_50000_pipeline'
        } | Select-Object ProcessId,Name,CommandLine
        if (-not $rows) { return 'No matching generation processes found.' }
        return ($rows | ForEach-Object { "$($_.Name) pid=$($_.ProcessId)" }) -join "`n"
    } catch {
        return "Process query failed: $($_.Exception.Message)"
    }
}

function Save-State($State) {
    $State | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $StatePath -Encoding UTF8
}

$state = Get-GenerationState
$lastImageCount = $state.ImageCount
$lastLabelCount = $state.LabelCount
$lastProgressTime = Get-Date
$inputArmed = $false
$lastInputAlert = [datetime]::MinValue
$seenProcessIds = @{}
$seenDriveNames = @{}

$SuspiciousProcessRegex = 'robocopy|xcopy|copy-item|compress-archive|7z|7za|winrar|rar\.exe|tar\.exe|zip|scp|sftp|rsync|rclone|winscp|filezilla|gdrive|googledrive|onedrive|dropbox|mega'
$ArchiveExtensions = @('*.zip', '*.7z', '*.rar', '*.tar', '*.gz')

function Normalize-ProtectedPath($Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return (Join-Path $Workspace $Path)
}

function Get-ProtectedPathRegex {
    $parts = @()
    foreach ($path in $ProtectedPath) {
        $resolved = Normalize-ProtectedPath $path
        $parts += [regex]::Escape($resolved)
        $parts += [regex]::Escape($path)
    }
    return ($parts | Where-Object { $_ -ne '' }) -join '|'
}

function Get-SuspiciousCopyProcesses($KnownIds) {
    $protectedRegex = Get-ProtectedPathRegex
    $rows = @()
    try {
        $processes = Get-CimInstance Win32_Process | Where-Object {
            $null -ne $_.CommandLine -and
            $_.Name -notin @('python.exe', 'blender.exe', 'blenderproc.exe', 'conda.exe') -and
            ($_.CommandLine -match $protectedRegex) -and
            ($_.CommandLine -match $SuspiciousProcessRegex)
        }
        foreach ($proc in $processes) {
            if (-not $KnownIds.ContainsKey([string]$proc.ProcessId)) {
                $rows += $proc
                $KnownIds[[string]$proc.ProcessId] = $true
            }
        }
    } catch {
        Write-WatchLog "suspicious process check failed: $($_.Exception.Message)"
    }
    return $rows
}

function Get-RemovableDrives {
    try {
        return Get-CimInstance Win32_LogicalDisk | Where-Object { $_.DriveType -in @(2, 4) } |
            Select-Object DeviceID, VolumeName, DriveType, Size, FreeSpace
    } catch {
        Write-WatchLog "drive check failed: $($_.Exception.Message)"
        return @()
    }
}

function Get-NewArchives {
    $newArchives = @()
    foreach ($path in $ProtectedPath) {
        $resolved = Normalize-ProtectedPath $path
        if (-not (Test-Path -LiteralPath $resolved)) { continue }
        foreach ($pattern in $ArchiveExtensions) {
            $newArchives += Get-ChildItem -File -Path $resolved -Recurse -Filter $pattern -ErrorAction SilentlyContinue |
                Where-Object { $_.LastWriteTime -gt (Get-Date).AddMinutes(-5) }
        }
    }
    return $newArchives
}

Save-State $state
Write-WatchLog "started. dataset=$($state.DatasetPath) images=$($state.ImageCount) labels=$($state.LabelCount)"
foreach ($drive in Get-RemovableDrives) {
    $seenDriveNames[$drive.DeviceID] = $true
}

Send-WatchEmail `
    "[WATCH START] Generation safety watch" `
    ("Generation safety watch started.`n`n" +
     "dataset: $($state.DatasetPath)`n" +
     "images: $($state.ImageCount)`n" +
     "labels: $($state.LabelCount)`n" +
     "idle arm: $IdleArmSeconds sec`n" +
     "protected paths: $($ProtectedPath -join ', ')`n" +
     "interval: $IntervalSeconds sec`n")

while ($true) {
    Start-Sleep -Seconds $IntervalSeconds

    $now = Get-Date
    $state = Get-GenerationState
    $idleSeconds = [math]::Round([IdleTime]::GetIdleMilliseconds() / 1000, 1)

    if ($state.ImageCount -gt $lastImageCount) {
        $lastProgressTime = $now
    }

    if ($state.ImageCount -lt $lastImageCount -or $state.LabelCount -lt $lastLabelCount) {
        Send-WatchEmail `
            "[ALERT] Generated file count decreased" `
            ("Generated file count decreased. Please check for delete or move activity.`n`n" +
             "time: $($now.ToString('yyyy-MM-dd HH:mm:ss'))`n" +
             "dataset: $($state.DatasetPath)`n" +
             "images: $lastImageCount -> $($state.ImageCount)`n" +
             "labels: $lastLabelCount -> $($state.LabelCount)`n`n" +
             "processes:`n$(Get-GenerationProcesses)`n")
    }

    $stallSeconds = ($now - $lastProgressTime).TotalSeconds
    if ($StallMinutes -gt 0 -and $stallSeconds -ge ($StallMinutes * 60)) {
        Send-WatchEmail `
            "[ALERT] Generation stall detected" `
            ("No new image was generated for the configured stall window.`n`n" +
             "stall minutes: $([math]::Round($stallSeconds / 60, 1))`n" +
             "images: $($state.ImageCount)`n" +
             "labels: $($state.LabelCount)`n" +
             "latest image: $($state.LatestImage) $($state.LatestImageTime)`n`n" +
             "processes:`n$(Get-GenerationProcesses)`n")
        $lastProgressTime = $now
    }

    if ($idleSeconds -ge $IdleArmSeconds) {
        $inputArmed = $true
    }
    if ($inputArmed -and $idleSeconds -le $ActiveThresholdSeconds -and (($now - $lastInputAlert).TotalMinutes -ge 10)) {
        Send-WatchEmail `
            "[ACTIVITY DETECTED] Mouse or keyboard input" `
            ("Mouse or keyboard input was detected after the machine had been idle.`n`n" +
             "time: $($now.ToString('yyyy-MM-dd HH:mm:ss'))`n" +
             "idle seconds now: $idleSeconds`n" +
             "dataset images: $($state.ImageCount)`n" +
             "dataset labels: $($state.LabelCount)`n`n" +
             "processes:`n$(Get-GenerationProcesses)`n")
        $lastInputAlert = $now
        $inputArmed = $false
    }

    $newDrives = @()
    foreach ($drive in Get-RemovableDrives) {
        if (-not $seenDriveNames.ContainsKey($drive.DeviceID)) {
            $newDrives += $drive
            $seenDriveNames[$drive.DeviceID] = $true
        }
    }
    if ($newDrives.Count -gt 0) {
        $driveText = ($newDrives | ForEach-Object {
            "$($_.DeviceID) volume=$($_.VolumeName) type=$($_.DriveType) sizeGB=$([math]::Round($_.Size / 1GB, 2))"
        }) -join "`n"
        Send-WatchEmail `
            "[ALERT] Removable or network drive detected" `
            ("A new removable or network drive was detected.`n`n" +
             "$driveText`n`n" +
             "Please check whether data is being copied.`n" +
             "processes:`n$(Get-GenerationProcesses)`n")
    }

    $suspicious = Get-SuspiciousCopyProcesses $seenProcessIds
    if ($suspicious.Count -gt 0) {
        $procText = ($suspicious | ForEach-Object {
            "pid=$($_.ProcessId) name=$($_.Name)`ncmd=$($_.CommandLine)"
        }) -join "`n`n"
        Send-WatchEmail `
            "[ALERT] Suspicious data or pt copy process detected" `
            ("A process command line contains both a protected path and a copy/archive/sync tool pattern.`n`n" +
             "$procText`n")
    }

    $archives = Get-NewArchives
    if ($archives.Count -gt 0) {
        $archiveText = ($archives | Select-Object -First 20 | ForEach-Object {
            "$($_.FullName) sizeMB=$([math]::Round($_.Length / 1MB, 2)) modified=$($_.LastWriteTime)"
        }) -join "`n"
        Send-WatchEmail `
            "[ALERT] Archive created in protected path" `
            ("An archive file was created or modified in a protected path within the last 5 minutes.`n`n$archiveText`n")
    }

    $lastImageCount = $state.ImageCount
    $lastLabelCount = $state.LabelCount
    Save-State $state
    Write-WatchLog "tick images=$($state.ImageCount) labels=$($state.LabelCount) idle=${idleSeconds}s"
}
