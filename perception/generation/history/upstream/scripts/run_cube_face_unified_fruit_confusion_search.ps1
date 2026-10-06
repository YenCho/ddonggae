param(
    [int]$Count = 120,
    [string]$Model = "runs\segment\cube_face_unified_yolo26n_seg_hsv_pruned_coloroutlier_from_last_adamw_lr1e5_ft_v1\weights\best.pt",
    [string]$FruitTextureDir = "datasets\fruit_textures\production_meta_v2_50000_v1_color_filtered_v2",
    [int]$GenerationWorkers = 6,
    [int]$RenderSamples = 8,
    [int]$Batch = 64,
    [string]$Device = "0",
    [string]$TagPrefix = "",
    [string]$AttemptSuffixes = "",
    [switch]$KeepGeneratedFaceTextures,
    [switch]$Reset,
    [switch]$OnlySummarize
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $Root

$Python = "C:\Users\user\anaconda3\envs\ai_robotics\python.exe"
$ReportRoot = "reports\cube_face_unified_eval"

if ([string]::IsNullOrWhiteSpace($TagPrefix)) {
    $TagPrefix = "fcs_$(Get-Date -Format 'MMdd_HHmm')"
}

$attempts = @(
    @{
        suffix = "a"
        print = "realistic_a4_fruit_visible"
        camera = "none"
        aug = "none"
        easy = 0.55
        mid = 0.40
        hard = 0.05
        scaleMin = 0.60
        scaleMax = 1.85
        lens = 0.10
        cropPad = 0.18
        minFace = 120
    },
    @{
        suffix = "b"
        print = "realistic_a4_fruit_visible"
        camera = "webcam_nocolor_mild"
        aug = "none"
        easy = 0.50
        mid = 0.45
        hard = 0.05
        scaleMin = 0.55
        scaleMax = 2.05
        lens = 0.20
        cropPad = 0.18
        minFace = 110
    },
    @{
        suffix = "c"
        print = "realistic_a4_fruit_visible"
        camera = "webcam_nocolor_mild"
        aug = "none"
        easy = 0.30
        mid = 0.60
        hard = 0.10
        scaleMin = 0.48
        scaleMax = 2.25
        lens = 0.35
        cropPad = 0.18
        minFace = 95
    },
    @{
        suffix = "d"
        print = "realistic_a4_fruit_visible"
        camera = "webcam_nocolor_strong"
        aug = "none"
        easy = 0.44
        mid = 0.50
        hard = 0.06
        scaleMin = 0.55
        scaleMax = 2.05
        lens = 0.22
        cropPad = 0.18
        minFace = 105
    },
    @{
        suffix = "e"
        print = "realistic_a4_boundary"
        camera = "webcam_nocolor_mild"
        aug = "none"
        easy = 0.32
        mid = 0.58
        hard = 0.10
        scaleMin = 0.50
        scaleMax = 2.15
        lens = 0.30
        cropPad = 0.18
        minFace = 105
    },
    @{
        suffix = "f"
        print = "realistic_a4_fruit_visible"
        camera = "webcam_nocolor_strong"
        aug = "none"
        easy = 0.28
        mid = 0.62
        hard = 0.10
        scaleMin = 0.48
        scaleMax = 2.30
        lens = 0.42
        cropPad = 0.14
        minFace = 95
    },
    @{
        suffix = "g"
        print = "realistic_a4_boundary"
        camera = "webcam_nocolor_strong"
        aug = "none"
        easy = 0.22
        mid = 0.66
        hard = 0.12
        scaleMin = 0.46
        scaleMax = 2.35
        lens = 0.50
        cropPad = 0.14
        minFace = 90
    },
    @{
        suffix = "h"
        print = "realistic_a4_fruit_visible"
        camera = "webcam_nocolor_mild"
        aug = "none"
        easy = 0.18
        mid = 0.72
        hard = 0.10
        scaleMin = 0.52
        scaleMax = 2.10
        lens = 0.35
        cropPad = 0.10
        minFace = 95
    },
    @{
        suffix = "i"
        print = "realistic_a4_label_offset"
        camera = "none"
        aug = "none"
        easy = 0.40
        mid = 0.54
        hard = 0.06
        scaleMin = 0.55
        scaleMax = 2.05
        lens = 0.18
        cropPad = 0.18
        minFace = 105
    },
    @{
        suffix = "j"
        print = "realistic_a4_label_offset"
        camera = "webcam_nocolor_mild"
        aug = "none"
        easy = 0.34
        mid = 0.58
        hard = 0.08
        scaleMin = 0.52
        scaleMax = 2.15
        lens = 0.30
        cropPad = 0.18
        minFace = 100
    },
    @{
        suffix = "k"
        print = "realistic_a4_label_offset"
        camera = "webcam_nocolor_strong"
        aug = "none"
        easy = 0.32
        mid = 0.60
        hard = 0.08
        scaleMin = 0.52
        scaleMax = 2.15
        lens = 0.32
        cropPad = 0.18
        minFace = 100
    },
    @{
        suffix = "l"
        print = "realistic_a4_label_offset"
        camera = "webcam_nocolor_mild"
        aug = "none"
        easy = 0.24
        mid = 0.68
        hard = 0.08
        scaleMin = 0.50
        scaleMax = 2.25
        lens = 0.40
        cropPad = 0.14
        minFace = 95
    },
    @{
        suffix = "m"
        print = "realistic_a4_label_offset_strong"
        camera = "webcam_nocolor_strong"
        aug = "none"
        easy = 0.24
        mid = 0.66
        hard = 0.10
        scaleMin = 0.50
        scaleMax = 2.25
        lens = 0.42
        cropPad = 0.14
        minFace = 95
    },
    @{
        suffix = "n"
        print = "realistic_a4_label_offset_strong"
        camera = "webcam_nocolor_aggressive"
        aug = "none"
        easy = 0.26
        mid = 0.64
        hard = 0.10
        scaleMin = 0.50
        scaleMax = 2.25
        lens = 0.42
        cropPad = 0.14
        minFace = 100
    },
    @{
        suffix = "o"
        print = "realistic_a4_label_offset_strong"
        camera = "webcam_nocolor_aggressive"
        aug = "none"
        easy = 0.20
        mid = 0.68
        hard = 0.12
        scaleMin = 0.48
        scaleMax = 2.35
        lens = 0.50
        cropPad = 0.10
        minFace = 95
    },
    @{
        suffix = "p"
        print = "realistic_a4_label_offset_strong"
        camera = "webcam_nocolor_aggressive"
        aug = "none"
        easy = 0.16
        mid = 0.70
        hard = 0.14
        scaleMin = 0.46
        scaleMax = 2.40
        lens = 0.58
        cropPad = 0.08
        minFace = 90
    },
    @{
        suffix = "q"
        print = "realistic_a4_label_offset_strong"
        camera = "webcam_nocolor_aggressive"
        aug = "none"
        easy = 0.12
        mid = 0.74
        hard = 0.14
        scaleMin = 0.48
        scaleMax = 2.30
        lens = 0.58
        cropPad = 0.06
        minFace = 92
    },
    @{
        suffix = "r"
        print = "realistic_a4_label_offset_strong"
        camera = "webcam_nocolor_aggressive"
        aug = "none"
        easy = 0.10
        mid = 0.76
        hard = 0.14
        scaleMin = 0.46
        scaleMax = 2.35
        lens = 0.62
        cropPad = 0.04
        minFace = 90
    },
    @{
        suffix = "s"
        print = "realistic_a4_label_offset_strong"
        camera = "webcam_nocolor_aggressive"
        aug = "none"
        easy = 0.08
        mid = 0.78
        hard = 0.14
        scaleMin = 0.46
        scaleMax = 2.40
        lens = 0.66
        cropPad = 0.04
        minFace = 86
    },
    @{
        suffix = "t"
        print = "realistic_a4_label_offset_lqprint"
        camera = "webcam_nocolor_aggressive"
        aug = "none"
        easy = 0.14
        mid = 0.72
        hard = 0.14
        scaleMin = 0.48
        scaleMax = 2.30
        lens = 0.58
        cropPad = 0.08
        minFace = 92
    },
    @{
        suffix = "u"
        print = "realistic_a4_label_offset_lqprint"
        camera = "webcam_nocolor_aggressive"
        aug = "none"
        easy = 0.12
        mid = 0.74
        hard = 0.14
        scaleMin = 0.48
        scaleMax = 2.35
        lens = 0.62
        cropPad = 0.06
        minFace = 90
    },
    @{
        suffix = "v"
        print = "realistic_a4_label_offset_lqprint"
        camera = "webcam_nocolor_aggressive"
        aug = "none"
        easy = 0.10
        mid = 0.76
        hard = 0.14
        scaleMin = 0.46
        scaleMax = 2.40
        lens = 0.66
        cropPad = 0.04
        minFace = 86
    }
)

$attemptSuffixSet = @{}
if (-not [string]::IsNullOrWhiteSpace($AttemptSuffixes)) {
    foreach ($suffix in ($AttemptSuffixes -split ",")) {
        $clean = $suffix.Trim()
        if (-not [string]::IsNullOrWhiteSpace($clean)) {
            $attemptSuffixSet[$clean] = $true
        }
    }
}

$topReports = @()
foreach ($attempt in $attempts) {
    if ($attemptSuffixSet.Count -gt 0 -and -not $attemptSuffixSet.ContainsKey($attempt.suffix)) {
        continue
    }
    $tag = "$TagPrefix`_$($attempt.suffix)"
    $topReport = Join-Path $ReportRoot "realistic_a4_mild_coco_$($tag)_top_class"
    $topReports += $topReport

    if ($OnlySummarize) {
        continue
    }

    Write-Host ""
    Write-Host "=== Running $tag ==="
    Write-Host "print=$($attempt.print), camera=$($attempt.camera), aug=$($attempt.aug)"

    $args = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "scripts\run_cube_face_unified_realistic_a4_mild_coco_booster_probe.ps1",
        "-Count", "$Count",
        "-Tag", $tag,
        "-Model", $Model,
        "-FruitTextureDir", $FruitTextureDir,
        "-FruitPrintLabelProfile", $attempt.print,
        "-CameraArtifactProfile", $attempt.camera,
        "-FruitTextureAug", $attempt.aug,
        "-EasyWeight", "$($attempt.easy)",
        "-MidWeight", "$($attempt.mid)",
        "-HardWeight", "$($attempt.hard)",
        "-ScaleMin", "$($attempt.scaleMin)",
        "-ScaleMax", "$($attempt.scaleMax)",
        "-LensDistortionProb", "$($attempt.lens)",
        "-CropPad", "$($attempt.cropPad)",
        "-ExportMinFacePixels", "$($attempt.minFace)",
        "-GenerationWorkers", "$GenerationWorkers",
        "-RenderSamples", "$RenderSamples",
        "-Batch", "$Batch",
        "-Device", $Device,
        "-PredictionMode", "fruit_for_fruit_truth"
    )
    if ($KeepGeneratedFaceTextures) {
        $args += "-KeepGeneratedFaceTextures"
    }
    if ($Reset) {
        $args += "-Reset"
    }
    & powershell @args
    if ($LASTEXITCODE -ne 0) {
        throw "Attempt failed: $tag"
    }
}

$summaryDir = Join-Path $ReportRoot "$($TagPrefix)_summary"
$existingReports = @()
foreach ($report in $topReports) {
    if (Test-Path -LiteralPath $report) {
        $existingReports += $report
    } else {
        Write-Warning "Missing report, skipped: $report"
    }
}
if ($existingReports.Count -eq 0) {
    throw "No top-class reports found to summarize."
}

Write-Host ""
Write-Host "=== Summarizing fruit-to-fruit confusion ==="
& $Python scripts\summarize_cube_face_fruit_confusion_search.py `
    --reports $existingReports `
    --output $summaryDir `
    --min_fruit_to_fruit_rate 0.05 `
    --max_fruit_to_plain_rate 0.18
if ($LASTEXITCODE -ne 0) {
    throw "Search summary failed."
}

Write-Host ""
Write-Host "Search summary: $summaryDir"
Write-Host "Open: $(Join-Path $summaryDir 'SUMMARY.md')"
Write-Host "Fruit-to-fruit sheet: $(Join-Path $summaryDir 'fruit_to_fruit_error_sheet.jpg')"
