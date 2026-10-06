# Backup Manifest

Last backup: 2026-05-16 (Asia/Seoul)

This repository is intended to preserve everything needed to reconstruct the AI robotics YOLO data-generation and training workflow, without committing the bulky generated image datasets.

## Git State

- Local branch: `main`
- Remote: `https://github.com/jaeyoungi2006/Data_Generation_Blender.git`
- Local backup base before this manifest: `6d0f7f6b9c24c2d4a9d9697e20e36e7b012862f4`
- Merged remote school updates from: `origin/master` at `2666d29297b432a8e979ecc4eee63b46a0356e7b`

## Tracked Recovery Assets

- Source scripts: `scripts/`
- Generated OBJ assets: `assets/generated/`
- Example email config: `config/email_smtp.example.json`
- Conda environment export: `environment.yml`
- Main README: `readme.md`
- YOLO dataset metadata:
  - `datasets/yolo_8class_maxobj5_balanced2_10000/data.yaml`
  - `datasets/yolo_8class_maxobj5_balanced2_10000/splits/front5000.yaml`
  - `datasets/yolo_8class_maxobj5_balanced2_10000/splits/front5000_train.txt`
  - `datasets/yolo_8class_maxobj5_balanced2_10000/splits/front5000_val.txt`
- Training metadata:
  - `runs/detect/runs/yolo_train/yolo11n_front5000_fast_b64/args.yaml`
  - `runs/detect/runs/yolo_train/yolo11n_front5000_fast_b64/results.csv`
- Model checkpoints:
  - `weights/best.pt`
  - `weights/yolo11n_front5000_fast_b64_best.pt`

## Checkpoint Integrity

Both tracked checkpoint files are identical.

```text
SHA256: AB44018B768C7CCB0B3F08FD31769337D88EDF15CB567F46800FB45C25A2CBE5
File: weights/best.pt
File: weights/yolo11n_front5000_fast_b64_best.pt
```

## Local Dataset Snapshot

These data files are intentionally ignored by git because they are large, but the current local machine has:

```text
datasets/yolo_8class_maxobj5_balanced2_10000/images/train: 7410
datasets/yolo_8class_maxobj5_balanced2_10000/labels/train: 7410
datasets/fruit_textures/detection_crops_512/apple: 1000
datasets/fruit_textures/detection_crops_512/banana: 1000
datasets/fruit_textures/detection_crops_512/orange: 1000
datasets/fruit_textures/detection_crops_512/pineapple: 1000
datasets/backgrounds/coco2017/val2017: 5000
```

Approximate local bulky data size:

```text
datasets/: 2661.9 MB
runs/:      90.21 MB
weights/:   40.58 MB
```

## Reconstruct From Git

```powershell
git clone https://github.com/jaeyoungi2006/Data_Generation_Blender.git
cd Data_Generation_Blender
conda env create -f environment.yml
conda activate ai_robotics
```

If the environment already exists:

```powershell
conda env update -n ai_robotics -f environment.yml --prune
conda activate ai_robotics
```

Regenerate assets if needed:

```powershell
python scripts\make_polyhedron_objs.py
```

Download/prepare backgrounds and fruit textures using the scripts under `scripts/`, then run one of the pipeline scripts:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_yolo_50000_pipeline.ps1
```

For a smaller continuation from the current 8-class setup, follow `readme.md` and use:

```powershell
python scripts\run_yolo_parallel.py --num_images 10000 --workers 10 --cpu_threads 1 --output datasets\yolo_8class_maxobj5_balanced2_10000 --samples 16 --max_objects 5 --scale_min 0.40 --scale_max 2.45 --negative_ratio 0.22 --fruit_texture_dir datasets\fruit_textures\detection_crops_512 --resume
```

## Verify Model

```powershell
Get-FileHash weights\best.pt -Algorithm SHA256
```

Expected SHA256:

```text
AB44018B768C7CCB0B3F08FD31769337D88EDF15CB567F46800FB45C25A2CBE5
```
