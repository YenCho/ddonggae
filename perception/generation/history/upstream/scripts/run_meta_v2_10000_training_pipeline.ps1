param(
    [int]$NumImages = 10000,
    [int]$Workers = 8,
    [int]$Samples = 24,
    [string]$SourceDataset = 'datasets\meta_v2_10000_coco_v1',
    [string]$ModelOutputRoot = 'datasets\meta_v2_10000_coco_v1_models',
    [string]$CondaEnv = 'ai_robotics',
    [string]$CondaActivate = "$env:USERPROFILE\anaconda3\Scripts\activate.bat",
    [string]$Device = '0',
    [string]$EmailTo = 'jaeyoungi@snu.ac.kr',
    [string]$EmailConfig = 'config\email_smtp.json',
    [int]$MonitorIntervalSeconds = 1800,
    [int]$GenerationEmailEvery = 1000,
    [int]$TrainEmailEvery = 10,
    [int]$TrainWorkers = 4,
    [int]$TorchWorkers = 2,
    [int]$A1Epochs = 120,
    [int]$A2Epochs = 80,
    [int]$BEpochs = 80,
    [int]$CEpochs = 60,
    [int]$A1Batch = 32,
    [int]$A2Batch = 64,
    [int]$BBatch = 256,
    [int]$CBatch = 256,
    [bool]$MakePreviews = $false,
    [string]$PipelineTag = 'meta_v2_10000',
    [string]$LogDirName = '',
    [string]$FruitTextureDir = 'datasets\fruit_textures\final_fruits36065_original25_fruitseg30_10',
    [ValidateSet('single', 'collage', 'mixed')]
    [string]$FruitTextureLayout = 'single',
    [double]$FruitTextureCollageProb = 0.35,
    [bool]$IdealVisibilityDebug = $false,
    [bool]$KeepGeneratedFaceTextures = $false,
    [int]$ProcessChunkSize = 250,
    [int]$MaxWorkerRetries = 30,
    [int]$RetryDelaySeconds = 60,
    [ValidateSet('auto', 'gpu', 'cpu')]
    [string]$RenderDevice = 'auto',
    [string]$GpuDeviceType = '',
    [ValidateSet('generation', 'export', 'unknown_negatives', 'preview', 'train_a1', 'train_a2', 'train_b', 'train_c', 'done')]
    [string]$StartStage = 'generation',
    [switch]$A2Resume,
    [string]$A2ResumeModel = '',
    [string]$A2Cache = 'disk'
)

$ErrorActionPreference = 'Stop'

$Workspace = Resolve-Path (Join-Path $PSScriptRoot '..')
Set-Location -LiteralPath $Workspace

if ([string]::IsNullOrWhiteSpace($LogDirName)) {
    $LogDirName = "${PipelineTag}_pipeline"
}

$LogDir = Join-Path $Workspace (Join-Path 'logs' $LogDirName)
$CmdDir = Join-Path $LogDir 'cmd'
New-Item -ItemType Directory -Force -Path $LogDir, $CmdDir | Out-Null

$StatePath = Join-Path $LogDir 'pipeline_state.json'
$StartedAt = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

$A1Project = 'runs\meta_v2_a1_objectseg'
$A1Name = "a1_yolo26s_seg_$PipelineTag"
$A2Project = 'runs\meta_v2_a2_faceseg'
$A2Name = "a2_yolo26n_seg_$PipelineTag"
$BProject = 'runs\meta_v2_b_tinyquadnet'
$BName = "b_tinyquadnet_$PipelineTag"
$CProject = 'runs\meta_v2_c_facecls'
$CName = "c_mobilenetv3small_$PipelineTag"

$A1ProjectAbs = Join-Path $Workspace $A1Project
$A2ProjectAbs = Join-Path $Workspace $A2Project
$BProjectAbs = Join-Path $Workspace $BProject
$CProjectAbs = Join-Path $Workspace $CProject

$A1RunDir = Join-Path $A1ProjectAbs $A1Name
$A2RunDir = Join-Path $A2ProjectAbs $A2Name
$BRunDir = Join-Path $BProjectAbs $BName
$CRunDir = Join-Path $CProjectAbs $CName

$script:StageOrder = @('generation', 'export', 'unknown_negatives', 'preview', 'train_a1', 'train_a2', 'train_b', 'train_c', 'done')
$script:StartStageIndex = [Array]::IndexOf($script:StageOrder, $StartStage)
if ($script:StartStageIndex -lt 0) {
    throw "unknown StartStage: $StartStage"
}

$script:State = [ordered]@{
    status = 'starting'
    stage = 'init'
    current_task = ''
    current_run_dir = ''
    source_dataset = (Join-Path $Workspace $SourceDataset)
    model_root = (Join-Path $Workspace $ModelOutputRoot)
    a1_run_dir = $A1RunDir
    a2_run_dir = $A2RunDir
    b_run_dir = $BRunDir
    c_run_dir = $CRunDir
    num_images = $NumImages
    pipeline_tag = $PipelineTag
    fruit_texture_dir = (Join-Path $Workspace $FruitTextureDir)
    fruit_texture_layout = $FruitTextureLayout
    fruit_texture_collage_prob = $FruitTextureCollageProb
    ideal_visibility_debug = $IdealVisibilityDebug
    keep_generated_face_textures = $KeepGeneratedFaceTextures
    process_chunk_size = $ProcessChunkSize
    max_worker_retries = $MaxWorkerRetries
    retry_delay_seconds = $RetryDelaySeconds
    render_device = $RenderDevice
    gpu_device_type = $GpuDeviceType
    start_stage = $StartStage
    a2_resume = $A2Resume
    a2_resume_model = $A2ResumeModel
    a2_cache = $A2Cache
    started_at = $StartedAt
    updated_at = $StartedAt
    last_error = ''
}

function Resolve-WorkspacePathString {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return (Join-Path $Workspace $Path)
}

function Should-RunStage {
    param([string]$Stage)
    $idx = [Array]::IndexOf($script:StageOrder, $Stage)
    if ($idx -lt 0) {
        throw "unknown stage: $Stage"
    }
    return ($idx -ge $script:StartStageIndex)
}

function Skip-Step {
    param(
        [string]$Stage,
        [string]$Task
    )
    Update-State -Status 'running' -Stage "$($Stage)_skipped" -Task "$Task skipped"
}

function Save-State {
    $script:State.updated_at = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $script:State | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $StatePath -Encoding UTF8
}

function Update-State {
    param(
        [string]$Status,
        [string]$Stage,
        [string]$Task,
        [string]$RunDir = '',
        [string]$ErrorMessage = ''
    )
    $script:State.status = $Status
    $script:State.stage = $Stage
    $script:State.current_task = $Task
    $script:State.current_run_dir = $RunDir
    $script:State.last_error = $ErrorMessage
    Save-State
}

function Send-PipelineEmail {
    param([string]$Subject)
    if ($EmailTo -eq '') {
        return
    }
    try {
        python scripts\send_meta_v2_pipeline_status_email.py `
            --state $StatePath `
            --to $EmailTo `
            --subject $Subject `
            --config $EmailConfig | Out-Null
    } catch {
        Write-Host "status email failed: $($_.Exception.Message)"
    }
}

function New-CmdScript {
    param(
        [string]$Name,
        [string]$Command
    )
    $safeName = ($Name -replace '[^A-Za-z0-9_.-]', '_')
    $path = Join-Path $CmdDir "$safeName.cmd"
    $content = @"
@echo off
echo ============================================================
echo Meta V2 pipeline step: $Name
echo Started: %DATE% %TIME%
echo Workspace: $Workspace
echo ============================================================
call "$CondaActivate" $CondaEnv
if errorlevel 1 exit /b %ERRORLEVEL%
cd /d "$Workspace"
$Command
set STEP_EXIT=%ERRORLEVEL%
echo.
echo ============================================================
echo Step $Name finished with exit code %STEP_EXIT%
echo Finished: %DATE% %TIME%
echo ============================================================
exit /b %STEP_EXIT%
"@
    Set-Content -LiteralPath $path -Value $content -Encoding ASCII
    return $path
}

function Invoke-StepWindow {
    param(
        [string]$Name,
        [string]$Stage,
        [string]$Task,
        [string]$Command,
        [string]$RunDir = ''
    )
    Update-State -Status 'running' -Stage $Stage -Task $Task -RunDir $RunDir
    Send-PipelineEmail -Subject "[start] Meta V2 pipeline - $Task"
    $cmdPath = New-CmdScript -Name $Name -Command $Command
    Write-Host ""
    Write-Host "Starting step in new cmd: $Task"
    Write-Host "Command script: $cmdPath"
    $proc = Start-Process -FilePath 'cmd.exe' -ArgumentList @('/c', "`"$cmdPath`"") -WorkingDirectory $Workspace -Wait -PassThru
    if ($proc.ExitCode -ne 0) {
        throw "step failed: $Task exit=$($proc.ExitCode)"
    }
    Update-State -Status 'running' -Stage $Stage -Task "$Task completed" -RunDir $RunDir
    Send-PipelineEmail -Subject "[done] Meta V2 pipeline - $Task"
}

function Start-MonitorWindow {
    if ($EmailTo -eq '') {
        return
    }
    $monitorCmd = @"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\monitor_meta_v2_pipeline_email_loop.ps1 -Workspace "$Workspace" -StatePath "$StatePath" -To "$EmailTo" -Subject "[30min check] Meta V2 A1/A2/B/C pipeline" -Config "$EmailConfig" -IntervalSeconds $MonitorIntervalSeconds -Python python
"@
    $cmdPath = New-CmdScript -Name 'monitor_meta_v2_pipeline_email_loop' -Command $monitorCmd
    Start-Process -FilePath 'cmd.exe' -ArgumentList @('/c', "`"$cmdPath`"") -WorkingDirectory $Workspace | Out-Null
}

try {
    Update-State -Status 'running' -Stage 'init' -Task 'pipeline starting'
    Start-MonitorWindow
    Send-PipelineEmail -Subject "[start] Meta V2 A1/A2/B/C $PipelineTag pipeline"

    $IdealVisibilityDebugFlag = if ($IdealVisibilityDebug) { '--ideal_visibility_debug' } else { '' }
    $KeepGeneratedFaceTexturesFlag = if ($KeepGeneratedFaceTextures) { '--keep_generated_face_textures' } else { '' }
    $GpuDeviceTypeFlag = if ([string]::IsNullOrWhiteSpace($GpuDeviceType)) { '' } else { "--gpu_device_type `"$GpuDeviceType`"" }

    $generationCommand = @"
python scripts\run_yolo_parallel.py --output "$SourceDataset" --num_images $NumImages --workers $Workers --worker_start_delay 2 --width 640 --height 640 --samples $Samples --cpu_threads 1 --fruit_texture_dir "$FruitTextureDir" --fruit_texture_aug strong --fruit_texture_layout $FruitTextureLayout --fruit_texture_collage_prob $FruitTextureCollageProb --label_format segment --min_objects 1 --max_objects 8 --scale_min 0.45 --scale_max 2.20 --negative_ratio 0.02 --lens_distortion_prob 0.35 --render_device $RenderDevice $GpuDeviceTypeFlag --robot_camera_view --ideal_visibility $IdealVisibilityDebugFlag --ideal_visibility_max_attempts 80 --meta_v2 $KeepGeneratedFaceTexturesFlag --process_chunk_size $ProcessChunkSize --max_worker_retries $MaxWorkerRetries --retry_delay_seconds $RetryDelaySeconds --resume --email_to "$EmailTo" --email_every $GenerationEmailEvery --email_config "$EmailConfig"
"@
    if (Should-RunStage 'generation') {
        Invoke-StepWindow -Name '01_generate_meta_v2_10000' -Stage 'generation' -Task "generate $NumImages Meta V2 images" -Command $generationCommand
    } else {
        Skip-Step -Stage 'generation' -Task "generate $NumImages Meta V2 images"
    }

    $exportCommand = @"
python scripts\export_meta_v2_model_datasets.py --source_dataset "$SourceDataset" --output_root "$ModelOutputRoot" --splits train --copy_mode hardlink --crop_size 224 --c_runtime_warp_variants 1 --c_runtime_max_expand 0.18 --c_runtime_max_shift 0.04 --c_runtime_max_corner_jitter 0.03 --reset
"@
    if (Should-RunStage 'export') {
        Invoke-StepWindow -Name '02_export_a1_a2_b_c' -Stage 'export' -Task 'export A1/A2/B/C datasets' -Command $exportCommand
    } else {
        Skip-Step -Stage 'export' -Task 'export A1/A2/B/C datasets'
    }

    $unknownCommand = @"
python scripts\add_facecls_unknown_hard_negatives.py --model_root "$ModelOutputRoot" --split train --count 3000 --crop_size 224 --reset_unknown
"@
    if (Should-RunStage 'unknown_negatives') {
        Invoke-StepWindow -Name '03_add_c_unknown_hard_negatives' -Stage 'unknown_negatives' -Task 'add C unknown hard negatives' -Command $unknownCommand
    } else {
        Skip-Step -Stage 'unknown_negatives' -Task 'add C unknown hard negatives'
    }

    if (-not (Should-RunStage 'preview')) {
        Skip-Step -Stage 'preview' -Task 'preview'
    } elseif ($MakePreviews) {
        $previewCommand = @"
python scripts\make_meta_v2_model_previews.py --model_root "$ModelOutputRoot" --output_dir "reports\$PipelineTag_model_previews" --split train --count 36 --c_per_class 12 --panel_size 220 --cols 6
python scripts\make_meta_v2_io_preview.py --source_dataset "$SourceDataset" --model_root "$ModelOutputRoot" --output "reports\${PipelineTag}_io_preview.jpg" --prefer_occlusion
"@
        Invoke-StepWindow -Name '04_make_previews' -Stage 'preview' -Task 'make A1/A2/B/C previews' -Command $previewCommand
    } else {
        Update-State -Status 'running' -Stage 'preview_skipped' -Task 'preview skipped'
    }

    $a1Command = @"
python scripts\train_yolo_with_email.py --task segment --model yolo26s-seg.pt --data "$ModelOutputRoot\a1_objectseg\data.yaml" --epochs $A1Epochs --imgsz 640 --batch $A1Batch --device $Device --workers $TrainWorkers --cache disk --amp True --deterministic False --patience 30 --project "$A1ProjectAbs" --name "$A1Name" --exist_ok True --email_to "$EmailTo" --email_every $TrainEmailEvery --email_config "$EmailConfig"
"@
    if (Should-RunStage 'train_a1') {
        Invoke-StepWindow -Name '05_train_a1_yolo26s_seg' -Stage 'train_a1' -Task 'train A1 YOLO26s-seg object segmentation' -Command $a1Command -RunDir $A1RunDir
    } else {
        Skip-Step -Stage 'train_a1' -Task 'train A1 YOLO26s-seg object segmentation'
    }

    $A2Model = 'yolo26n-seg.pt'
    $A2ResumeExtra = ''
    if ($A2Resume) {
        if ([string]::IsNullOrWhiteSpace($A2ResumeModel)) {
            $A2Model = Join-Path $A2RunDir 'weights\last.pt'
        } else {
            $A2Model = Resolve-WorkspacePathString $A2ResumeModel
        }
        if (-not (Test-Path -LiteralPath $A2Model)) {
            throw "A2 resume checkpoint not found: $A2Model"
        }
        $A2ResumeExtra = "resume=`"$A2Model`""
    }

    $a2Command = @"
python scripts\train_yolo_with_email.py --task segment --model "$A2Model" --data "$ModelOutputRoot\a2_faceseg\data.yaml" --epochs $A2Epochs --imgsz 224 --batch $A2Batch --device $Device --workers $TrainWorkers --cache $A2Cache --amp True --deterministic False --patience 20 --project "$A2ProjectAbs" --name "$A2Name" --exist_ok True --email_to "$EmailTo" --email_every $TrainEmailEvery --email_config "$EmailConfig" $A2ResumeExtra
"@
    if (Should-RunStage 'train_a2') {
        Invoke-StepWindow -Name '06_train_a2_yolo26n_seg' -Stage 'train_a2' -Task 'train A2 YOLO26n-seg face segmentation' -Command $a2Command -RunDir $A2RunDir
    } else {
        Skip-Step -Stage 'train_a2' -Task 'train A2 YOLO26n-seg face segmentation'
    }

    $bCommand = @"
python scripts\train_tiny_quadnet.py --data "$ModelOutputRoot\b_facequad" --epochs $BEpochs --batch $BBatch --imgsz 128 --lr 0.001 --device $Device --workers $TorchWorkers --project "$BProjectAbs" --name "$BName" --email_to "$EmailTo" --email_every $TrainEmailEvery --email_config "$EmailConfig"
"@
    if (Should-RunStage 'train_b') {
        Invoke-StepWindow -Name '07_train_b_tinyquadnet' -Stage 'train_b' -Task 'train B TinyQuadNet face quad regressor' -Command $bCommand -RunDir $BRunDir
    } else {
        Skip-Step -Stage 'train_b' -Task 'train B TinyQuadNet face quad regressor'
    }

    $cCommand = @"
python scripts\train_face_mobilenetv3.py --data "$ModelOutputRoot\c_facecls" --epochs $CEpochs --batch $CBatch --imgsz 128 --lr 0.0005 --device $Device --workers $TorchWorkers --project "$CProjectAbs" --name "$CName" --email_to "$EmailTo" --email_every $TrainEmailEvery --email_config "$EmailConfig"
"@
    if (Should-RunStage 'train_c') {
        Invoke-StepWindow -Name '08_train_c_mobilenetv3' -Stage 'train_c' -Task 'train C MobileNetV3-Small face classifier' -Command $cCommand -RunDir $CRunDir
    } else {
        Skip-Step -Stage 'train_c' -Task 'train C MobileNetV3-Small face classifier'
    }

    Update-State -Status 'completed' -Stage 'done' -Task 'pipeline completed' -RunDir ''
    Send-PipelineEmail -Subject "[completed] Meta V2 A1/A2/B/C $PipelineTag pipeline"
    Write-Host "Pipeline completed."
} catch {
    Update-State -Status 'failed' -Stage 'failed' -Task 'pipeline failed' -RunDir '' -ErrorMessage $_.Exception.Message
    Send-PipelineEmail -Subject "[failed] Meta V2 A1/A2/B/C $PipelineTag pipeline"
    Write-Host "Pipeline failed: $($_.Exception.Message)"
    throw
}
