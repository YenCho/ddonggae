# Reproduce the synthetic perception models

This is the runnable recipe for building the perception training inputs from a clean checkout. It creates a new public-data baseline with deterministic seeds; it does not recreate the competition checkpoints byte for byte. The training photographs and original checkpoint used for the July 2026 fine-tune were private, and the exact Blender binary used for the historical weights was not recorded. The renderer now follows the final arena rule: fruit prints on the top and one opposing side pair.

The full BlenderProc generator, dataset exporters, geometry and setup scripts are in [`../generation/`](../generation/). The historical design discussion and measured failures are in [`synthetic-data.md`](synthetic-data.md); the imported upstream experiment log is [`../generation/history/upstream/EXPERIMENTS.md`](../generation/history/upstream/EXPERIMENTS.md).

## 1. Requirements

Use Python 3.11, BlenderProc 2.8.0, Blender Python (`bpy`) 5.0.1, PyTorch 2.10.0, torchvision 0.25.0 and Ultralytics 8.4.54. The pinned host dependencies are in [`../generation/requirements.txt`](../generation/requirements.txt). Install a PyTorch/torchvision pair for your CUDA platform using the official PyTorch wheel index, then install that file. Record the output of `python --version`, `python -m pip freeze`, `blenderproc --version`, and `nvidia-smi` with each run. Use `--device cpu` and `--render_device cpu` when no NVIDIA GPU is available; full generation and training will take substantially longer.

From the repository root in PowerShell:

```powershell
py -3.11 -m venv .venv-perception
.\.venv-perception\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r perception\generation\requirements.txt
python -c "import torch, torchvision, blenderproc, bpy, ultralytics; print(torch.__version__, torchvision.__version__, ultralytics.__version__, bpy.app.version_string)"
```

For a CPU-only install, install the matching CPU PyTorch wheels from `https://download.pytorch.org/whl/cpu` in the same step. Do not install a second torch build afterward. On Linux, use:

```bash
python3.11 -m venv .venv-perception
source .venv-perception/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r perception/generation/requirements.txt
python -c 'import torch, torchvision, blenderproc, bpy, ultralytics; print(torch.__version__, torchvision.__version__, ultralytics.__version__, bpy.app.version_string)'
```

## 2. Build the public assets

Run commands from `perception/generation`; all paths below are relative to that directory. In PowerShell use `Set-Location perception\generation`; in bash use `cd perception/generation`. The fruit texture downloader pins the Fruits-360 Git commit and verifies every downloaded file against its Git blob hash. It writes the license and SHA256 list to `datasets/fruit_textures/public_fruits360_256/assets-manifest.json`. Keep that attribution file with any redistributed texture-derived work; Fruits-360 is CC BY-SA 4.0. The generation commands use PowerShell line continuations and Windows path separators; on Linux replace trailing backticks with `\` and use `/` in paths.

```powershell
Set-Location perception\generation
python prepare_assets.py --out datasets\fruit_textures\public_fruits360_256 --per-class 256
python scripts\make_polyhedron_objs.py --out assets\generated --size 0.08
```

`prepare_assets.py` is resumable and checks the manifest on subsequent runs. The images and generated datasets are intentionally ignored by Git; only their exact preparation recipe and asset hashes are tracked.

## 3. Smoke render, then make 50,000 scenes

First make 120 scenes with the smoke command below. This checks BlenderProc startup, texture lookup, labels, and Meta V2 output before committing to a long render:

```powershell
python scripts\run_yolo_parallel.py `
  --num_images 120 --workers 1 --worker_start_delay 0 `
  --asset_dir assets\generated `
  --fruit_texture_dir datasets\fruit_textures\public_fruits360_256 `
  --background_dir datasets\backgrounds\unused `
  --arena_background_ratio 1.0 --output datasets\smoke120 `
  --width 640 --height 640 --samples 4 --cpu_threads 1 `
  --seed 31000000 --min_objects 1 --max_objects 4 `
  --scale_min 0.40 --scale_max 2.45 --negative_ratio 0.18 `
  --label_format segment --seg_contour_mode largest `
  --seg_contour_epsilon_ratio 0.01 --lighting_mode soft_overhead `
  --fruit_texture_aug strong --fruit_texture_layout mixed `
  --fruit_texture_collage_prob 0.35 --canonical_fruit_textures `
  --allow_mixed_fruit_classes_per_image --allow_multiple_fruit_textures_per_cube `
  --ideal_visibility --ideal_visibility_max_attempts 80 --meta_v2 --render_device cpu
```

Open several images from `datasets/smoke120/images/train/` and compare them with their text labels and JSON files under `_meta/train/`. Confirm that every scene has an image, label and metadata file. Do not scale up if the render reports missing classes, missing assets or a traceback.

For the full dataset, use 50,000 scenes, 16 Cycles samples and the GPU. Start with one worker because each worker starts its own Blender process; increase `--workers` only after confirming available GPU memory and render stability. Keep every option and the seed fixed if resuming:

```powershell
python scripts\run_yolo_parallel.py `
  --num_images 50000 --workers 1 --worker_start_delay 0 `
  --asset_dir assets\generated `
  --fruit_texture_dir datasets\fruit_textures\public_fruits360_256 `
  --background_dir datasets\backgrounds\unused `
  --arena_background_ratio 1.0 --output datasets\public_fruits360_arena_v1 `
  --width 640 --height 640 --samples 16 --cpu_threads 1 `
  --seed 31000000 --min_objects 1 --max_objects 4 `
  --scale_min 0.40 --scale_max 2.45 --negative_ratio 0.18 `
  --label_format segment --seg_contour_mode largest `
  --seg_contour_epsilon_ratio 0.01 --min_object_visible_ratio 0.10 `
  --min_fruit_visible_ratio 0.22 --min_fruit_face_pixels 1800 `
  --single_object_min_projected_area 2600 `
  --single_object_min_fruit_face_pixels 2200 --min_fruit_face_side 36 `
  --fruit_visibility_easy_weight 0.50 --fruit_visibility_mid_weight 0.45 `
  --fruit_visibility_hard_weight 0.05 --hard_min_fruit_visible_ratio 0.10 `
  --hard_min_fruit_face_pixels 900 --lighting_mode soft_overhead `
  --fruit_texture_aug strong --fruit_texture_layout mixed `
  --fruit_texture_collage_prob 0.35 --canonical_fruit_textures `
  --allow_mixed_fruit_classes_per_image --allow_multiple_fruit_textures_per_cube `
  --ideal_visibility --ideal_visibility_max_attempts 80 --meta_v2 --render_device gpu
```

If interrupted, rerun the exact command with `--resume`. The generator skips only complete image/label/metadata triples. Do not change the seed, texture directory, class order or rendering options while resuming.

## 4. Make a shared scene split and export model datasets

Keep all crops from one rendered scene in the same split. This avoids the train/validation leakage that inflated early synthetic scores. The split script sorts scene ids, shuffles with seed `20261006`, and writes a manifest with scene-id checksum and exact counts:

```powershell
python scripts\split_meta_v2_dataset.py `
  --dataset datasets\public_fruits360_arena_v1 `
  --out datasets\public_fruits360_arena_v1\split.json --seed 20261006 `
  --val-ratio 0.10 --test-ratio 0.10
```

Export the A1 and auxiliary training sets using that same manifest:

```powershell
python scripts\export_meta_v2_model_datasets.py `
  --source_dataset datasets\public_fruits360_arena_v1 `
  --output_root datasets\public_fruits360_arena_v1_models `
  --split_manifest datasets\public_fruits360_arena_v1\split.json `
  --splits train --tasks a1 c --copy_mode hardlink --reset

python scripts\export_meta_v2_cube_face_unified_dataset.py `
  --source_dataset datasets\public_fruits360_arena_v1 `
  --output_root datasets\public_fruits360_arena_v1_face `
  --source_split train `
  --split_manifest datasets\public_fruits360_arena_v1\split.json `
  --seed 20261006 --crop_size 224 --crop_pad 0.18 --reset
```

Inspect the exported model datasets' `data.yaml`, `manifest.json`, and `meta_v2_export_audit.json`. The raw renderer writes a legacy compatibility `data.yaml`; do not train from it. A1 must have the four classes `cube_like_object, octahedron, dodecahedron, icosahedron`; face segmentation must have `apple, orange, banana, pineapple, plain` in that order. Both model YAML files must point to separate `images/train`, `images/val`, and `images/test` directories.

## 5. Train from scratch

Training starts from the Ultralytics `yolo26s-seg.yaml` architecture definition with random initialization; it does not download a pretrained checkpoint. Run names are immutable and the entry point refuses to overwrite an existing run:

```powershell
python ..\training\train_segmentation.py `
  --task a1 `
  --data datasets\public_fruits360_arena_v1_models\a1_objectseg\data.yaml `
  --init yolo26s-seg.yaml --epochs 140 --batch 32 --workers 4 `
  --device 0 --seed 31000000 --lr 0.001 `
  --project ..\..\runs\perception --name public_fruits360_arena_v1_a1

python ..\training\train_segmentation.py `
  --task face `
  --data datasets\public_fruits360_arena_v1_face\data.yaml `
  --init yolo26s-seg.yaml --epochs 120 --batch 32 --workers 4 `
  --device 0 --seed 31000000 --lr 0.001 --hsv-h 0 `
  --project ..\..\runs\perception --name public_fruits360_arena_v1_face
```

For the pair classifiers, the `c_facecls` dataset exported above can train apple/orange and banana/pineapple verifiers. Train both from the repository root:

```powershell
python perception\training\train_face_mobilenetv3.py --data perception\generation\datasets\public_fruits360_arena_v1_models\c_facecls --classes apple,orange --epochs 60 --batch 128 --imgsz 128 --lr 0.0005 --weight_decay 0.0001 --workers 4 --device 0 --seed 20261006 --backbone mobilenet_v3_small --project runs\perception --name public_fruits360_arena_v1_pair_apple_orange
python perception\training\train_face_mobilenetv3.py --data perception\generation\datasets\public_fruits360_arena_v1_models\c_facecls --classes banana,pineapple --epochs 60 --batch 128 --imgsz 128 --lr 0.0005 --weight_decay 0.0001 --workers 4 --device 0 --seed 20261006 --backbone mobilenet_v3_small --project runs\perception --name public_fruits360_arena_v1_pair_banana_pineapple
```

These classifiers use the pinned torchvision MobileNetV3-Small ImageNet initialization; torchvision downloads those weights on first use. They train only on `train/` crops, then use the explicit scene-held-out `val/` folders. The classifier writes `weights/best.pt` and `results.csv`. Export a selected verifier with [`../training/101_export_pair_onnx.py`](../training/101_export_pair_onnx.py) and evaluate once on the held-out `test/` crops with [`../training/100_eval_pair_on_val.py`](../training/100_eval_pair_on_val.py). The synthetic holdout measures repeatability on this recipe; field accuracy still requires a separately collected real-camera set.

```powershell
python perception\training\101_export_pair_onnx.py runs\perception\public_fruits360_arena_v1_pair_apple_orange\weights\best.pt --out perception\generation\datasets\verifiers\apple_orange
python perception\training\100_eval_pair_on_val.py --data perception\generation\datasets\public_fruits360_arena_v1_models\c_facecls --classes apple,orange --new runs\perception\public_fruits360_arena_v1_pair_apple_orange\weights\best.pt --split test
python perception\training\101_export_pair_onnx.py runs\perception\public_fruits360_arena_v1_pair_banana_pineapple\weights\best.pt --out perception\generation\datasets\verifiers\banana_pineapple
python perception\training\100_eval_pair_on_val.py --data perception\generation\datasets\public_fruits360_arena_v1_models\c_facecls --classes banana,pineapple --new runs\perception\public_fruits360_arena_v1_pair_banana_pineapple\weights\best.pt --split test
```

Every run directory contains `recipe.json`; save the asset manifest, split manifest, pip freeze, GPU details, and BlenderProc version next to the checkpoints. Report the test split once after model selection. Do not select checkpoints by repeated test-set evaluation.

With Ultralytics installed, evaluate the selected segmentation checkpoints on the held-out test scenes:

```powershell
yolo segment val model=runs\perception\public_fruits360_arena_v1_a1\weights\best.pt data=perception\generation\datasets\public_fruits360_arena_v1_models\a1_objectseg\data.yaml imgsz=640 split=test device=0
yolo segment val model=runs\perception\public_fruits360_arena_v1_face\weights\best.pt data=perception\generation\datasets\public_fruits360_arena_v1_face\data.yaml imgsz=224 split=test device=0
```

## What this reproduces

The scripts, class contracts, random seeds, split ids, public input assets, and hyperparameters are pinned so another person can run the same pipeline. GPU architecture, driver, Blender binary, image decode libraries and floating point kernels can change rendered pixels or final weight bytes. Therefore the expected result is the same dataset and training procedure, not a byte-identical `.pt`. The original 2026 competition models additionally used private/undistributed renders, real arena photographs and earlier initial checkpoints; those are documented as historical results and are not inputs to this public baseline.
