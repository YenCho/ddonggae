param(
    [int]$Count = 80,
    [string]$TagPrefix = "orange_condition_probe_$(Get-Date -Format 'MMdd_HHmm')",
    [string]$Model = "runs\segment\cube_face_unified_yolo26n_seg_hsv_pruned_coloroutlier_from_last_adamw_lr1e5_ft_v1\weights\best.pt",
    [string]$FruitTextureDir = "datasets\fruit_textures\production_meta_v2_50000_v1_color_filtered_v2",
    [int]$GenerationWorkers = 6,
    [int]$RenderSamples = 8,
    [int]$Batch = 64,
    [string]$Device = "0",
    [string]$ConditionNames = "",
    [switch]$IncludeAppleForAllConditions,
    [switch]$Reset
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $Root

$Python = "C:\Users\user\anaconda3\envs\ai_robotics\python.exe"
$ReportRoot = "reports\cube_face_unified_eval"

$conditions = @(
    @{
        name = "easy"
        print = "realistic_a4_fruit_visible"
        camera = "none"
        aug = "none"
        easy = 0.86
        mid = 0.14
        hard = 0.00
        scaleMin = 0.70
        scaleMax = 1.90
        lens = 0.04
        cropPad = 0.18
        minFace = 150
        appleNegative = $false
    },
    @{
        name = "mid"
        print = "realistic_a4_label_offset"
        camera = "webcam_nocolor_mild"
        aug = "none"
        easy = 0.46
        mid = 0.49
        hard = 0.05
        scaleMin = 0.56
        scaleMax = 2.10
        lens = 0.22
        cropPad = 0.16
        minFace = 110
        appleNegative = $false
    },
    @{
        name = "hard_rotation"
        print = "realistic_a4_label_offset_strong"
        camera = "webcam_nocolor_mild"
        aug = "none"
        easy = 0.20
        mid = 0.65
        hard = 0.15
        scaleMin = 0.50
        scaleMax = 2.25
        lens = 0.62
        cropPad = 0.12
        minFace = 95
        appleNegative = $false
    },
    @{
        name = "small_sliver"
        print = "realistic_a4_label_offset_strong"
        camera = "webcam_nocolor_strong"
        aug = "none"
        easy = 0.02
        mid = 0.33
        hard = 0.65
        scaleMin = 0.48
        scaleMax = 2.35
        lens = 0.55
        cropPad = 0.10
        minFace = 45
        appleNegative = $false
    },
    @{
        name = "washed_out"
        print = "realistic_a4_label_offset"
        camera = "mild_exposure"
        aug = "none"
        easy = 0.26
        mid = 0.64
        hard = 0.10
        scaleMin = 0.52
        scaleMax = 2.20
        lens = 0.35
        cropPad = 0.14
        minFace = 100
        appleNegative = $false
    },
    @{
        name = "blur"
        print = "realistic_a4_label_offset"
        camera = "webcam_nocolor_aggressive"
        aug = "none"
        easy = 0.26
        mid = 0.64
        hard = 0.10
        scaleMin = 0.52
        scaleMax = 2.20
        lens = 0.32
        cropPad = 0.14
        minFace = 100
        appleNegative = $false
    },
    @{
        name = "crop_edge"
        print = "realistic_a4_label_offset_strong"
        camera = "webcam_nocolor_mild"
        aug = "none"
        easy = 0.16
        mid = 0.70
        hard = 0.14
        scaleMin = 0.50
        scaleMax = 2.25
        lens = 0.45
        cropPad = 0.02
        minFace = 90
        appleNegative = $false
    },
    @{
        name = "white_face_dominant"
        print = "realistic_a4_label_offset_strong"
        camera = "webcam_nocolor_strong"
        aug = "none"
        easy = 0.00
        mid = 0.15
        hard = 0.85
        scaleMin = 0.48
        scaleMax = 2.35
        lens = 0.55
        cropPad = 0.06
        minFace = 35
        appleNegative = $false
    },
    @{
        name = "apple_orange_boundary"
        print = "realistic_a4_label_offset_strong"
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
        appleNegative = $true
    }
)

$conditionSet = @{}
if (-not [string]::IsNullOrWhiteSpace($ConditionNames)) {
    foreach ($name in ($ConditionNames -split ",")) {
        $clean = $name.Trim()
        if (-not [string]::IsNullOrWhiteSpace($clean)) {
            $conditionSet[$clean] = $true
        }
    }
}

$manifestReports = @()
foreach ($condition in $conditions) {
    if ($conditionSet.Count -gt 0 -and -not $conditionSet.ContainsKey($condition.name)) {
        continue
    }

    $classes = @("orange")
    if ($IncludeAppleForAllConditions -or $condition.appleNegative) {
        $classes += "apple"
    }

    foreach ($className in $classes) {
        $tag = "$TagPrefix`_$($condition.name)_$className"
        Write-Host ""
        Write-Host "=== Probe $($condition.name) / $className ==="
        $args = @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", "scripts\run_cube_face_unified_realistic_a4_mild_coco_booster_probe.ps1",
            "-Count", "$Count",
            "-Tag", $tag,
            "-Model", $Model,
            "-FruitTextureDir", $FruitTextureDir,
            "-FruitPrintLabelProfile", $condition.print,
            "-CameraArtifactProfile", $condition.camera,
            "-FruitTextureAug", $condition.aug,
            "-ForceFruitClass", $className,
            "-EasyWeight", "$($condition.easy)",
            "-MidWeight", "$($condition.mid)",
            "-HardWeight", "$($condition.hard)",
            "-ScaleMin", "$($condition.scaleMin)",
            "-ScaleMax", "$($condition.scaleMax)",
            "-LensDistortionProb", "$($condition.lens)",
            "-CropPad", "$($condition.cropPad)",
            "-ExportMinFacePixels", "$($condition.minFace)",
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
            throw "Probe failed: $tag"
        }

        $manifestReports += [ordered]@{
            condition = $condition.name
            class = $className
            tag = $tag
            source_dataset = "datasets\cube_face_unified_booster_realistic_a4_mild_coco_$($tag)_source"
            probe_dataset = "datasets\cube_face_unified_booster_realistic_a4_mild_coco_$tag"
            report = "reports\cube_face_unified_eval\realistic_a4_mild_coco_$($tag)_top_class"
            print_profile = $condition.print
            camera_profile = $condition.camera
            fruit_texture_aug = $condition.aug
            crop_pad = $condition.cropPad
            min_face_pixels = $condition.minFace
            note = if ($condition.name -eq "white_face_dominant") { "Do not promote hidden/blank-dominant orange crops as orange positives." } else { "" }
        }
    }
}

$summaryRoot = Join-Path $ReportRoot "$($TagPrefix)_summary"
New-Item -ItemType Directory -Force -Path $summaryRoot | Out-Null
$manifestPath = Join-Path $summaryRoot "probe_manifest.json"
@{ reports = $manifestReports } | ConvertTo-Json -Depth 8 | Set-Content -Path $manifestPath -Encoding UTF8

$candidateRoot = "datasets\cube_face_unified_boost_candidates_$TagPrefix"
& $Python scripts\summarize_cube_face_probe_conditions.py `
    --manifest $manifestPath `
    --output $summaryRoot `
    --candidate_output $candidateRoot
if ($LASTEXITCODE -ne 0) {
    throw "Condition summary failed"
}

Write-Host ""
Write-Host "Condition summary: $(Join-Path $summaryRoot 'summary.md')"
Write-Host "Boost candidate dataset: $candidateRoot"
