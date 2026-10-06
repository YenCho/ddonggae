$ErrorActionPreference = "Stop"

Set-Location "C:\Users\user\Documents\vscode\Data_Generation_Blender"

$Dataset = "datasets\yolo_8class_v3_30000"
$TrainName = "yolo11n_v3_30000_fast_noamp"

Write-Host "Starting YOLO 30000 image generation..."
python scripts\run_yolo_parallel.py `
  --num_images 30000 `
  --workers 10 `
  --cpu_threads 1 `
  --output $Dataset `
  --samples 16 `
  --max_objects 5 `
  --scale_min 0.40 `
  --scale_max 2.45 `
  --negative_ratio 0.22 `
  --fruit_texture_dir datasets\fruit_textures\final_fruits36065_original25_fruitseg30_10 `
  --email_to jaeyoungi@snu.ac.kr `
  --email_every 2000 `
  --resume

if ($LASTEXITCODE -ne 0) {
  throw "Generation failed with exit code $LASTEXITCODE"
}

Write-Host "Generation finished. Starting YOLO11n training..."
python scripts\train_yolo_with_email.py `
  --data "C:/Users/user/Documents/vscode/Data_Generation_Blender/datasets/yolo_8class_v3_30000/data.yaml" `
  --model yolo11n.pt `
  --epochs 100 `
  --imgsz 640 `
  --batch 32 `
  --device 0 `
  --workers 4 `
  --cache ram `
  --amp False `
  --deterministic False `
  --patience 20 `
  --project runs/yolo_train `
  --name $TrainName `
  --email_to jaeyoungi@snu.ac.kr `
  --email_every 20 `
  --attach_final

if ($LASTEXITCODE -ne 0) {
  throw "Training failed with exit code $LASTEXITCODE"
}

Write-Host "Pipeline finished successfully."
