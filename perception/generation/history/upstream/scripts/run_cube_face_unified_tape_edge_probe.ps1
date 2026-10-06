param(
    [int]$Count = 100,
    [string]$TagPrefix = "tape_edge_probe_$(Get-Date -Format 'MMdd_HHmm')",
    [string]$Model = "runs\segment\cube_face_unified_yolo26n_seg_hsv_pruned_coloroutlier_from_last_adamw_lr1e5_ft_v1\weights\best.pt",
    [string]$FruitTextureDir = "datasets\fruit_textures\production_meta_v2_50000_v1_color_filtered_v2",
    [int]$GenerationWorkers = 6,
    [int]$RenderSamples = 8,
    [int]$Batch = 64,
    [string]$Device = "0",
    [switch]$IncludeAllFruits,
    [switch]$Reset
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $Root

$classes = @("orange", "apple")
if ($IncludeAllFruits) {
    $classes = @("apple", "orange", "banana", "pineapple")
}

foreach ($className in $classes) {
    $tag = "$TagPrefix`_$className"
    Write-Host ""
    Write-Host "=== Tape-edge probe / $className ==="
    $args = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "scripts\run_cube_face_unified_realistic_a4_mild_coco_booster_probe.ps1",
        "-Count", "$Count",
        "-Tag", $tag,
        "-Model", $Model,
        "-FruitTextureDir", $FruitTextureDir,
        "-FruitPrintLabelProfile", "realistic_a4_tape_edge",
        "-CameraArtifactProfile", "webcam_nocolor_mild",
        "-FruitTextureAug", "none",
        "-ForceFruitClass", $className,
        "-EasyWeight", "0.35",
        "-MidWeight", "0.58",
        "-HardWeight", "0.07",
        "-ScaleMin", "0.55",
        "-ScaleMax", "2.15",
        "-LensDistortionProb", "0.25",
        "-CropPad", "0.14",
        "-ExportMinFacePixels", "110",
        "-GenerationWorkers", "$GenerationWorkers",
        "-RenderSamples", "$RenderSamples",
        "-Batch", "$Batch",
        "-Device", $Device,
        "-PredictionMode", "fruit_for_fruit_truth"
    )
    if ($Reset) {
        $args += "-Reset"
    }
    & powershell @args
    if ($LASTEXITCODE -ne 0) {
        throw "Tape-edge probe failed: $tag"
    }
}

Write-Host ""
Write-Host "Tape-edge probes complete. Reports are under reports\cube_face_unified_eval\realistic_a4_mild_coco_$($TagPrefix)_*_top_class"
