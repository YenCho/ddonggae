$ErrorActionPreference = "Continue"

$ProjectRoot = "C:\Users\user\Documents\vscode\Data_Generation_Blender"
$Python = "C:\Users\user\miniconda3\envs\ai_robotics\python.exe"
$Recipient = "jaeyoungi@snu.ac.kr"
$Dataset = Join-Path $ProjectRoot "datasets\yolo_8class_v3_50000"
$RunCandidates = @(
  (Join-Path $ProjectRoot "runs\detect\runs\yolo_train\yolo11n_v3_50000_fast_noamp"),
  (Join-Path $ProjectRoot "runs\yolo_train\yolo11n_v3_50000_fast_noamp")
)
$TotalEpochs = 200
$LogPath = Join-Path $ProjectRoot "runs\training_email_monitor.log"

function Write-MonitorLog($Message) {
  $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  $line = "[$stamp] $Message"
  New-Item -ItemType Directory -Force -Path (Split-Path $LogPath) | Out-Null
  Add-Content -Path $LogPath -Value $line -Encoding UTF8
}

function Format-Duration($Seconds) {
  if ($null -eq $Seconds -or $Seconds -lt 0) { return "계산 중" }
  $span = [TimeSpan]::FromSeconds([double]$Seconds)
  if ($span.TotalHours -ge 1) {
    return ("약 {0}시간 {1}분" -f [int]$span.TotalHours, $span.Minutes)
  }
  return ("약 {0}분" -f [Math]::Max(1, [int][Math]::Round($span.TotalMinutes)))
}

function Get-RunDir {
  foreach ($candidate in $RunCandidates) {
    if (Test-Path $candidate) { return $candidate }
  }
  return $RunCandidates[0]
}

function Get-TrainingProcess {
  try {
    return Get-CimInstance Win32_Process | Where-Object {
      $_.Name -in @("python.exe", "yolo.exe") -and
      $_.CommandLine -match "train_yolo_with_email|yolo detect train|yolo11n_v3_50000"
    } | Select-Object -First 1
  } catch {
    return $null
  }
}

function Get-LatestMetrics($RunDir) {
  $results = Join-Path $RunDir "results.csv"
  if (-not (Test-Path $results)) { return $null }
  try {
    $rows = Import-Csv $results
    if (-not $rows -or $rows.Count -eq 0) { return $null }
    return $rows[-1]
  } catch {
    Write-MonitorLog "Failed to read results.csv: $($_.Exception.Message)"
    return $null
  }
}

function Get-Prop($Object, $Name) {
  if ($null -eq $Object) { return "" }
  $prop = $Object.PSObject.Properties | Where-Object { $_.Name.Trim() -eq $Name } | Select-Object -First 1
  if ($prop) { return $prop.Value }
  return ""
}

try {
  Set-Location $ProjectRoot
  $now = Get-Date
  $runDir = Get-RunDir
  $process = Get-TrainingProcess
  $metrics = Get-LatestMetrics $runDir
  $imageCount = (Get-ChildItem (Join-Path $Dataset "images\train") -Filter *.jpg -ErrorAction SilentlyContinue | Measure-Object).Count
  $labelCount = (Get-ChildItem (Join-Path $Dataset "labels\train") -Filter *.txt -ErrorAction SilentlyContinue | Measure-Object).Count
  $weightsDir = Join-Path $runDir "weights"
  $lastPt = Join-Path $weightsDir "last.pt"
  $bestPt = Join-Path $weightsDir "best.pt"

  $epoch = 0
  $box = ""
  $cls = ""
  $dfl = ""
  $map50 = ""
  $map5095 = ""
  if ($metrics) {
    $epochRaw = Get-Prop $metrics "epoch"
    if ($epochRaw -ne "") {
      $epoch = [int][double]$epochRaw + 1
    }
    $box = Get-Prop $metrics "train/box_loss"
    $cls = Get-Prop $metrics "train/cls_loss"
    $dfl = Get-Prop $metrics "train/dfl_loss"
    $map50 = Get-Prop $metrics "metrics/mAP50(B)"
    $map5095 = Get-Prop $metrics "metrics/mAP50-95(B)"
  }

  $processState = if ($process) { "실행 중 (PID $($process.ProcessId))" } else { "실행 중인 학습 프로세스 없음" }
  $startTime = if ($process) { [Management.ManagementDateTimeConverter]::ToDateTime($process.CreationDate) } else { $null }
  $elapsedSeconds = if ($startTime) { ($now - $startTime).TotalSeconds } else { $null }
  $remainingEpochs = [Math]::Max($TotalEpochs - $epoch, 0)
  $etaSeconds = $null
  if ($epoch -gt 0 -and $elapsedSeconds -gt 0) {
    $etaSeconds = ($elapsedSeconds / $epoch) * $remainingEpochs
  }
  $etaText = Format-Duration $etaSeconds
  $finishText = if ($etaSeconds -ne $null) { ($now.AddSeconds($etaSeconds)).ToString("yyyy-MM-dd HH:mm") } else { "계산 중" }

  $lastPtText = if (Test-Path $lastPt) {
    $item = Get-Item $lastPt
    "있음 ($([Math]::Round($item.Length / 1MB, 1)) MB, 수정 $($item.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')))"
  } else {
    "아직 생성되지 않음"
  }
  $bestPtText = if (Test-Path $bestPt) {
    $item = Get-Item $bestPt
    "있음 ($([Math]::Round($item.Length / 1MB, 1)) MB, 수정 $($item.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')))"
  } else {
    "아직 생성되지 않음"
  }

  $body = @"
YOLO 50000 학습 30분 점검

현재 시각: $($now.ToString("yyyy-MM-dd HH:mm:ss")) Asia/Seoul
프로세스 상태: $processState
데이터셋: images=$imageCount / labels=$labelCount
run dir: $runDir

epoch: $epoch / $TotalEpochs
남은 epoch: $remainingEpochs
예상 남은 시간: $etaText
예상 완료 시각: $finishText

최근 지표:
box_loss: $box
cls_loss: $cls
dfl_loss: $dfl
mAP50: $map50
mAP50-95: $map5095

weights:
last.pt: $lastPtText
best.pt: $bestPtText

이 메일은 Windows 작업 스케줄러의 YOLO50000TrainingEmailMonitor 작업에서 발송되었습니다.
"@

  $subject = "[30분 점검] YOLO 50000 학습 상태"
  $args = @(
    "scripts\send_progress_email.py",
    "--to", $Recipient,
    "--subject", $subject,
    "--body", $body
  )
  if (Test-Path $lastPt) {
    $args += @("--attach", $lastPt)
  }

  & $Python @args
  Write-MonitorLog "sent status email. epoch=$epoch process=$processState lastPt=$(Test-Path $lastPt)"
} catch {
  Write-MonitorLog "monitor failed: $($_.Exception.Message)"
}
