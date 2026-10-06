param(
    [int]$NumImages = 50000,
    [int]$Workers = 8,
    [int]$Samples = 24,
    [string]$SourceDataset = 'datasets\meta_v2_50000_coco_texture_v1',
    [string]$ModelOutputRoot = 'datasets\meta_v2_50000_coco_texture_v1_models',
    [string]$FruitTextureDir = 'datasets\fruit_textures\production_meta_v2_50000_v1_color_filtered_v2',
    [string]$CondaEnv = 'ai_robotics',
    [string]$CondaActivate = "$env:USERPROFILE\anaconda3\Scripts\activate.bat",
    [string]$Device = '0',
    [string]$EmailTo = 'jaeyoungi@snu.ac.kr',
    [string]$EmailConfig = 'config\email_smtp.json',
    [int]$MonitorIntervalSeconds = 1800,
    [int]$GenerationEmailEvery = 2500,
    [int]$TrainEmailEvery = 10,
    [int]$TrainWorkers = 4,
    [int]$TorchWorkers = 2,
    [int]$A1Epochs = 140,
    [int]$A2Epochs = 90,
    [int]$BEpochs = 90,
    [int]$CEpochs = 70,
    [int]$A1Batch = 32,
    [int]$A2Batch = 64,
    [int]$BBatch = 256,
    [int]$CBatch = 256,
    [bool]$MakePreviews = $false,
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

& (Join-Path $PSScriptRoot 'run_meta_v2_10000_training_pipeline.ps1') `
    -NumImages $NumImages `
    -Workers $Workers `
    -Samples $Samples `
    -SourceDataset $SourceDataset `
    -ModelOutputRoot $ModelOutputRoot `
    -CondaEnv $CondaEnv `
    -CondaActivate $CondaActivate `
    -Device $Device `
    -EmailTo $EmailTo `
    -EmailConfig $EmailConfig `
    -MonitorIntervalSeconds $MonitorIntervalSeconds `
    -GenerationEmailEvery $GenerationEmailEvery `
    -TrainEmailEvery $TrainEmailEvery `
    -TrainWorkers $TrainWorkers `
    -TorchWorkers $TorchWorkers `
    -A1Epochs $A1Epochs `
    -A2Epochs $A2Epochs `
    -BEpochs $BEpochs `
    -CEpochs $CEpochs `
    -A1Batch $A1Batch `
    -A2Batch $A2Batch `
    -BBatch $BBatch `
    -CBatch $CBatch `
    -MakePreviews $MakePreviews `
    -IdealVisibilityDebug $IdealVisibilityDebug `
    -KeepGeneratedFaceTextures $KeepGeneratedFaceTextures `
    -ProcessChunkSize $ProcessChunkSize `
    -MaxWorkerRetries $MaxWorkerRetries `
    -RetryDelaySeconds $RetryDelaySeconds `
    -RenderDevice $RenderDevice `
    -GpuDeviceType $GpuDeviceType `
    -StartStage $StartStage `
    -A2Resume:$A2Resume `
    -A2ResumeModel $A2ResumeModel `
    -A2Cache $A2Cache `
    -PipelineTag 'meta_v2_50000' `
    -LogDirName 'meta_v2_50000_pipeline' `
    -FruitTextureDir $FruitTextureDir `
    -FruitTextureLayout 'mixed' `
    -FruitTextureCollageProb 0.35
