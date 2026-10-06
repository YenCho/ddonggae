param(
    [Parameter(Mandatory = $true)]
    [string]$Video,

    [string]$Python = "C:\Users\jaeyo\anaconda3\envs\yolo\python.exe",
    [string]$Output = "",
    [string]$Device = "cpu",
    [string]$A1Model = "preferred-a1",
    [string]$Model = "preferred-unified",
    [int]$A1ImgSize = 640,
    [int]$ImgSize = 224,
    [double]$A1Conf = 0.25,
    [double]$FaceConf = 0.25,
    [double]$CropPad = 0.18,
    [int]$NFrames = 0,
    [int]$FrameStep = 1,
    [string]$TargetShape = "cube",
    [string]$TargetFruit = "",
    [switch]$NoFrames,
    [switch]$NoInputs,
    [switch]$NoRawCrops,
    [switch]$CopySource,
    [switch]$Pull
)

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if ($Pull) {
    git pull --ff-only
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    if (Get-Command git-lfs -ErrorAction SilentlyContinue) {
        git lfs pull
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
}

if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path "reports" ("unified_video_frame_sort_{0}" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
}

$ArgList = @(
    "jetson\sort_unified_video_frames.py",
    "--video", $Video,
    "--out", $Output,
    "--device", $Device,
    "--a1-model", $A1Model,
    "--model", $Model,
    "--a1-imgsz", "$A1ImgSize",
    "--imgsz", "$ImgSize",
    "--a1-conf", "$A1Conf",
    "--conf", "$FaceConf",
    "--crop-pad", "$CropPad",
    "--n-frames", "$NFrames",
    "--frame-step", "$FrameStep",
    "--target-shape", $TargetShape
)

if (-not [string]::IsNullOrWhiteSpace($TargetFruit)) {
    $ArgList += @("--target-fruit", $TargetFruit)
}
if ($NoFrames) {
    $ArgList += "--no-frames"
}
if ($NoInputs) {
    $ArgList += "--no-inputs"
}
if ($NoRawCrops) {
    $ArgList += "--no-raw-crops"
}
if ($CopySource) {
    $ArgList += "--copy-source"
}

Write-Host "Running: $Python $($ArgList -join ' ')"
& $Python @ArgList
exit $LASTEXITCODE
