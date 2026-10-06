# Model aliases: the primary names are cube-detector (old A1) and
# face-classifier (old cube-face unified), with face-classifier-onnx for ONNX.
# Legacy synonyms preferred-a1 / preferred-a1-onnx / preferred-unified /
# preferred-unified-onnx / latest-unified are still accepted and remain the
# defaults below for compatibility. List all: -ListModels
param(
    [ValidateSet("unified", "abc", "yolo")]
    [string]$Pipeline = "unified",

    [string]$Python = "python",
    [string]$Camera = "0",
    [ValidateSet("auto", "any", "dshow", "v4l2", "gstreamer")]
    [string]$CameraBackend = "auto",
    [string]$Device = "0",
    [string]$Model = "",
    [string]$A1Model = "",
    [int]$ImgSize = 224,
    [int]$A1ImgSize = 640,
    [int]$A2ImgSize = 224,
    [double]$A1Conf = 0.25,
    [double]$FaceConf = 0.25,
    [double]$CropPad = 0.18,
    [string]$TargetShape = "cube",
    [string]$TargetFruit = "apple",
    [switch]$Cpu,
    [switch]$Onnx,
    [switch]$DecisionOverlay,
    [switch]$ShowCInputs,
    [switch]$NoModelOutputPanel,
    [switch]$PrintModelOutput,
    [switch]$PrintTiming,
    [switch]$NoDisplay,
    [int]$MaxFrames = 0,
    [switch]$ListModels
)

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if ($Cpu) {
    $Device = "cpu"
}

if ($ListModels) {
    & $Python "jetson\realtime_seg_cam.py" "--list-model-aliases"
    exit $LASTEXITCODE
}

$ArgList = @(
    "jetson\realtime_seg_cam.py",
    "--pipeline", $Pipeline,
    "--camera", $Camera,
    "--camera-backend", $CameraBackend,
    "--device", $Device,
    "--display-max-width", "1600",
    "--display-max-height", "900"
)

if ($Pipeline -eq "unified") {
    # Defaults stay on the legacy alias names for compatibility; the new
    # primary names (face-classifier / cube-detector) resolve identically.
    if ([string]::IsNullOrWhiteSpace($Model)) {
        $Model = if ($Onnx) { "preferred-unified-onnx" } else { "preferred-unified" }
    }
    if ([string]::IsNullOrWhiteSpace($A1Model)) {
        $A1Model = if ($Onnx) { "preferred-a1-onnx" } else { "preferred-a1" }
    }
    $ArgList += @(
        "--a1-model", $A1Model,
        "--model", $Model,
        "--task", "segment",
        "--imgsz", "$ImgSize",
        "--conf", "$FaceConf",
        "--unified-a1-imgsz", "$A1ImgSize",
        "--unified-a1-conf", "$A1Conf",
        "--unified-crop-pad", "$CropPad",
        "--target-shape", $TargetShape,
        "--target-fruit", $TargetFruit
    )
}
elseif ($Pipeline -eq "abc") {
    $ArgList += @(
        "--runtime-preset", "best_stable_5080",
        "--target-shape", $TargetShape,
        "--target-fruit", $TargetFruit,
        "--imgsz", "960",
        "--a2-imgsz", "$A2ImgSize"
    )
    if ($DecisionOverlay) {
        $ArgList += @("--abc-overlay", "decision")
    }
    if ($ShowCInputs) {
        $ArgList += "--show-c-inputs"
    }
}
else {
    if (-not [string]::IsNullOrWhiteSpace($Model)) {
        $ArgList += @("--model", $Model)
    }
    $ArgList += @(
        "--task", "segment",
        "--imgsz", "$ImgSize"
    )
}

if ($NoModelOutputPanel) {
    $ArgList += "--no-model-output-panel"
}
if ($PrintModelOutput) {
    $ArgList += "--print-model-output"
}
if ($PrintTiming) {
    $ArgList += "--print-timing"
}
if ($NoDisplay) {
    $ArgList += "--no-display"
}
if ($MaxFrames -gt 0) {
    $ArgList += @("--max-frames", "$MaxFrames")
}

Write-Host "Running: $Python $($ArgList -join ' ')"
& $Python @ArgList
exit $LASTEXITCODE
