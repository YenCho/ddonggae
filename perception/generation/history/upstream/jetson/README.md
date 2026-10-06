# Jetson Orin Nano Runtime Guide

기준일: 2026-07-04

이 폴더는 Jetson Orin Nano에서 cube/fruit runtime preview를 실행하기 위한 폴더입니다. 학습 데이터 생성과 전체 실험 설명은 루트 [readme.md](../readme.md), 상세 reference는 [readme_specific.md](../readme_specific.md), 실험 색인은 [EXPERIMENTS.md](../EXPERIMENTS.md)를 보세요.

## 한 줄 요약

Jetson에서 가장 먼저 확인할 runtime:

```bash
cd ~/Data_Generation_Blender
git lfs install
git lfs pull
python3 jetson/realtime_seg_cam.py --list-model-aliases

chmod +x jetson/run_webcam_preview.sh
./jetson/run_webcam_preview.sh \
  --pipeline unified \
  --camera 0 \
  --camera-backend v4l2 \
  --device 0 \
  --target-shape cube \
  --target-fruit apple \
  --print-model-output
```

처음에는 `--pipeline unified`를 사용하세요. ABC cascade는 fallback/debug 용도입니다.

## 필요한 Weights

fresh clone에서는 Git LFS를 먼저 실행합니다.

```bash
git lfs install
git lfs pull
```

alias 확인:

```bash
python3 jetson/realtime_seg_cam.py --list-model-aliases
```

unified runtime에 필요한 weight:

| Alias | Path | 역할 |
| --- | --- | --- |
| `cube-detector` (legacy: `preferred-a1`) | `jetson/ABC_model/meta_v2_a1_objectseg/a1_yolo26s_seg_meta_v2_50000/weights/best.pt` | full-frame object segmentation (Cube Detector) |
| `face-classifier` (legacy: `preferred-unified`) | `jetson/ABC_model/cube_face_unified/preferred_v2/weights/best.pt` | cube-crop visible face segmentation/classification (Face Classifier) |

새 primary alias 이름 `cube-detector` / `face-classifier`(ONNX: `cube-detector-onnx` / `face-classifier-onnx`)가 코드에 추가되었습니다. legacy alias(`preferred-a1`, `preferred-a1-onnx`, `preferred-unified`, `preferred-unified-onnx`, `latest-unified`)도 그대로 동작하며 같은 weight로 resolve됩니다.

선택 사항인 ABC fallback weights:

```text
jetson/ABC_model/meta_v2_a2_faceseg/a2_yolo26n_seg_meta_v2_50000/weights/best.pt
jetson/ABC_model/meta_v2_b_tinyquadnet/b_tinyquadnet_meta_v2_50000/weights/best.pt
jetson/ABC_model/meta_v2_c_facecls/c_mobilenetv3small_meta_v2_50000_runtimewarp_ft/weights/best.pt
```

## Jetson에서 쓰는 모델 계보

```text
A1 object segmentation:
  base model: yolo26s-seg.pt
  train data: datasets/meta_v2_50000_coco_texture_v1_models/a1_objectseg/data.yaml
  role: full frame에서 cube_like_object와 non-cube shape을 찾음

cube-face unified preferred_v2:
  base lineage: yolo26n-seg.pt -> cube_face_unified_yolo26n_seg_v1 -> whole-fruit/plain-hard balanced v2 -> verified-fruit balanced AdamW -> flat-icon boundary AdamW -> HSV stronger booster -> color-outlier pruned AdamW
  train data v1: datasets/meta_v2_50000_cube_face_unified_v1/data.yaml
  fine-tune data: datasets/cube_face_unified_finetune_meta_v2_50000_plus_hsv_ratio_20000_stronger_pruned_coloroutlier_v1/data.yaml
  source run: runs/segment/cube_face_unified_yolo26n_seg_hsv_pruned_coloroutlier_from_last_adamw_lr1e5_ft_v1/weights/best.pt
  bundled checkpoint: best epoch 24, copied into preferred_v2/weights/best.pt
  bundled sha256: 6EC2F572475F5649D262F46E886A3149FE6FA6ED72E36BF00B7AA40686FD9153
  role: A1 cube_like_object crop 안에서만 실행하고 visible face mask/class를 출력
```

과거 fallback 구조:

```text
ABC cascade:
  A1 yolo26s-seg.pt object segmentation
  A2 yolo26n-seg.pt visible face segmentation
  B custom TinyQuadNet quad recovery
  C MobileNetV3-Small face classifier
```

old object-only YOLO line은 visible cube face를 판단하지 못하므로 runtime에서는 deprecated입니다.

## 환경 설치

Jetson은 설치된 JetPack/CUDA 버전에 맞는 PyTorch build를 써야 합니다. NVIDIA Jetson PyTorch를 먼저 설치한 뒤 project package를 설치하세요.

기본 package:

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs python3-pip python3-venv v4l-utils python3-opencv
```

clone:

```bash
git clone https://github.com/jaeyoungi2006/Data_Generation_Blender.git
cd Data_Generation_Blender
git lfs install
git lfs pull
```

virtual environment 예시:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ultralytics
python -m pip install -r jetson/requirements.txt
```

CUDA 확인:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

OpenCV import가 깨지면 Jetson에서는 system OpenCV를 우선 사용합니다.

```bash
python -m pip uninstall -y opencv-python opencv-python-headless
sudo apt-get install -y python3-opencv
```

## 첫 실행

USB/V4L2 camera:

```bash
./jetson/run_webcam_preview.sh \
  --pipeline unified \
  --camera 0 \
  --camera-backend v4l2 \
  --device 0
```

headless timing:

```bash
./jetson/run_webcam_preview.sh \
  --pipeline unified \
  --camera 0 \
  --camera-backend v4l2 \
  --device 0 \
  --no-display \
  --print-timing \
  --max-frames 300
```

CPU fallback:

```bash
./jetson/run_webcam_preview.sh \
  --pipeline unified \
  --camera 0 \
  --camera-backend v4l2 \
  --cpu
```

CSI camera 예시:

```bash
./jetson/run_webcam_preview.sh \
  --pipeline unified \
  --camera-backend gstreamer \
  --camera 'nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1' \
  --device 0
```

## 웹캠 Preview

직접 실행 명령:

```bash
python3 jetson/realtime_seg_cam.py \
  --pipeline unified \
  --camera 0 \
  --camera-backend v4l2 \
  --device 0 \
  --a1-model preferred-a1 \
  --model preferred-unified \
  --unified-a1-imgsz 640 \
  --imgsz 224 \
  --unified-a1-conf 0.25 \
  --conf 0.25 \
  --target-shape cube \
  --target-fruit apple \
  --print-model-output
```

자주 쓰는 옵션:

| Option | 의미 | wrapper 기본값 |
| --- | --- | --- |
| `--pipeline unified` | A1 full-frame object model + cube-crop unified face model | yes |
| `--camera` | camera index, video path, GStreamer pipeline | `0` |
| `--camera-backend` | `v4l2`, `gstreamer`, `auto` 등 | shell wrapper는 `v4l2` |
| `--device` | CUDA device `0` 또는 `cpu` | `0` |
| `--a1-model` | A1 path 또는 alias | `preferred-a1` |
| `--model` | cube-face unified path 또는 alias | `preferred-unified` |
| `--unified-a1-imgsz` | A1 full-frame image size | `640` |
| `--imgsz` | unified face crop size | `224` |
| `--unified-a1-conf` | A1 confidence | `0.25` |
| `--conf` | unified face confidence | `0.25` |
| `--print-model-output` | per-frame raw output과 decision 출력 | off |
| `--print-timing` | timing/FPS bottleneck 출력 | off |

## 정상 실행 기준

첫 정상 실행에서는 다음이 보여야 합니다.

- A1 object output에 `cube_like_object` 또는 non-cube shape이 표시됩니다.
- A1 `cube_like_object`마다 crop이 생성됩니다. 단, 원본 frame 기준 너무 작은 cube는 `cube_too_far`로 표시되고 unified face model에는 들어가지 않습니다.
- 각 cube crop 내부 visible face에 대해 unified face class/confidence가 표시됩니다.
- object overlay에 `fruit_cube:orange`, `plain_cube`, `cube_too_far`, `cube_like_unresolved`, non-cube shape label 같은 final identity가 표시됩니다.
- `--print-model-output`을 켜거나 model-output panel을 켜면 오른쪽 panel에 A1 objects, cube crops, decisions, face detections가 frame마다 표시됩니다.

## Overlay 의미

| Overlay label | 의미 | Pickup implication |
| --- | --- | --- |
| `fruit_cube:apple` / `fruit_cube:orange` / `fruit_cube:banana` / `fruit_cube:pineapple` | reliable fruit face가 하나 이상 검출됨 | `--target-fruit`와 일치할 때만 `pickup`, 아니면 `avoid` 또는 `inspect` |
| `plain_cube` | 현재 frame에서 reliable plain face **2개 이상**, fruit/unknown face 없음 (규칙상 fruit cube는 항상 fruit면이 위로 오게 배치되므로, 2면이 모두 blank면 fruit cube가 숨을 수 없음) | `--target-shape cube`일 때만 `pickup`. robot actuation은 반복 plain evidence를 요구하는 것이 안전 |
| `cube_too_far` | A1은 cube-like object를 봤지만 원본 frame 기준 bbox가 너무 작음 | unified face model을 실행하지 않고 다시 접근/inspect |
| `cube_like_unresolved` | Cube Detector는 cube-like object를 봤지만 reliable face evidence가 없거나, unknown 또는 **저신뢰 fruit face(`--fruit-min-conf 0.45` 미만)**가 있어 안전하게 분류 불가 | 다시 inspect. plain으로 취급 금지 |
| `blank_only_cube_ambiguous` | reliable plain face가 **1개뿐** (2개 이상 보이면 `plain_cube`로 확정) | 저각도에서 위쪽 fruit면이 프레임 밖일 수 있으므로 두 번째 면을 다시 inspect |
| `conflicting_fruit_cube` | 한 cube에서 fruit class 여러 개가 나옴 | 자동 pickup 금지, inspect |
| `octahedron`, `dodecahedron`, `icosahedron` | A1 non-cube shape | `--target-shape`와 일치할 때만 `pickup`, 아니면 skip/inspect |

현재 코드는 `decide_cube()` 내부 temporal vote를 하지 않습니다. overlay는 per-frame decision입니다.

## Target Options

`--target-shape`은 Task1 shape target입니다.

```bash
--target-shape cube
--target-shape octahedron
--target-shape dodecahedron
--target-shape icosahedron
```

`--target-fruit`은 Task2 fruit target입니다.

```bash
--target-fruit apple
--target-fruit orange
--target-fruit banana
--target-fruit pineapple
```

decision에 반영되는 방식:

- visible fruit face가 `--target-fruit`와 일치하면 `decide_cube()`가 `action=pickup`, expected score 20을 반환합니다.
- visible fruit face가 target fruit가 아니면 `action=avoid`를 반환합니다.
- plain cube가 검출되고 `--target-shape cube`이면 `action=pickup`, expected score 10을 반환합니다.
- non-cube shape이 `--target-shape`와 일치하면 `decide_shape()`가 `action=pickup`을 반환합니다.

## 현재 Runtime 판단 로직 요약

```text
A1 non-cube shape
  -> decide_shape()

A1 cube_like_object
  -> crop
  -> cube-face unified model
  -> FaceEvidence
  -> decide_cube()
```

규칙:

- `cube_like_object`만으로는 unresolved이며 plain이 아닙니다.
- `cube_too_far`는 face crop을 만들지 않으므로 unified output이 없습니다.
- fruit face evidence는 plain evidence보다 우선합니다.
- non-target fruit는 plain이 아니며 Task1 cube로 집으면 안 됩니다.
- plain face 1개는 ambiguous, **2개 이상이면 plain_cube**입니다 (fruit cube는 항상 fruit면이 위로 오므로 2 blank면이면 fruit cube 불가).
- runtime threshold는 [runtime_policy.py](./runtime_policy.py)에서 관리합니다. `CUBE_TOO_FAR_*`는 작은 cube gate입니다.

## ABC Cascade Fallback

기존 staged path를 debug하거나 C classifier crop을 확인할 때 ABC를 사용합니다.

```bash
./jetson/run_webcam_preview.sh \
  --pipeline abc \
  --camera 0 \
  --camera-backend v4l2 \
  --device 0 \
  --a1-imgsz 640 \
  --a2-imgsz 224 \
  --decision-overlay \
  --show-c-inputs \
  --print-timing
```

ABC 역할:

| Stage | 역할 |
| --- | --- |
| A1 | full-frame object segmentation |
| A2 | cube crop 안 visible face segmentation |
| B | visible mask -> 4-corner quad |
| C | perspective-warp face crop -> fruit/plain/unknown class |

ABC는 설명 가능하고 geometry debugging에 유용하기 때문에 유지합니다. 하지만 per-face runtime overhead가 커서 Jetson 첫 추천 구조는 아닙니다.

## TensorRT Export

TensorRT engine은 실행할 Jetson 장비에서 export하는 것이 안전합니다. engine은 hardware/software 조합에 의존합니다.

A1 export:

```bash
yolo export \
  model=jetson/ABC_model/meta_v2_a1_objectseg/a1_yolo26s_seg_meta_v2_50000/weights/best.pt \
  format=engine \
  imgsz=640 \
  half=True \
  device=0
```

Unified face model export:

```bash
yolo export \
  model=jetson/ABC_model/cube_face_unified/preferred_v2/weights/best.pt \
  format=engine \
  imgsz=224 \
  half=True \
  device=0
```

Ultralytics는 기본적으로 `.pt` 옆에 `.engine`을 씁니다. 현재 built-in alias는 `.pt` path를 가리키므로 engine test는 explicit path를 넘깁니다.

```bash
python3 jetson/realtime_seg_cam.py \
  --pipeline unified \
  --camera 0 \
  --camera-backend v4l2 \
  --device 0 \
  --a1-model jetson/ABC_model/meta_v2_a1_objectseg/a1_yolo26s_seg_meta_v2_50000/weights/best.engine \
  --model jetson/ABC_model/cube_face_unified/preferred_v2/weights/best.engine \
  --unified-a1-imgsz 640 \
  --imgsz 224
```

`.pt` fallback은 alias를 다시 사용하면 됩니다.

```bash
./jetson/run_webcam_preview.sh --pipeline unified --a1-model preferred-a1 --model preferred-unified
```

## 예상 성능

Jetson Orin Nano FPS는 JetPack, PyTorch/TensorRT build, camera backend, display, power mode에 따라 달라집니다. RTX 5080 benchmark 숫자와 Jetson live preview를 직접 비교하면 안 됩니다.

PC-side reference:

| 항목 | 장치 | 조건 | 결과 | Report |
| --- | --- | --- | ---: | --- |
| A1 TensorRT + unified PyTorch benchmark | RTX 5080 dev PC | benchmark images, display off, mean 4.05 objects / 11.65 faces | mean `47.45 FPS`, median `52.00 FPS` | `reports/cube_face_unified_eval/unified_runtime_benchmark_a1engine_ptface.json` |
| Webcam fair compare, unified | captured benchmark run reported `cpu` | same 30s webcam capture | `3.83 FPS` | `reports/share/fair_30s_compare_20260702_155104/summary.md` |
| Webcam fair compare, ABC | captured benchmark run reported `cpu` | same 30s webcam capture | `1.29 FPS` | `reports/share/fair_30s_compare_20260702_155104/summary.md` |

Jetson에서는 먼저 `.pt`로 correctness를 맞추고, 이후 TensorRT engine을 테스트하세요.

## 문제 해결

| 증상 | 확인할 것 |
| --- | --- |
| model file missing | `git lfs pull` 실행 후 `python3 jetson/realtime_seg_cam.py --list-model-aliases` |
| camera does not open | `--camera 1`, `--camera-backend v4l2`, `v4l2-ctl --list-devices` 확인 |
| CUDA unavailable | JetPack-compatible PyTorch 확인 후 CUDA check 실행 |
| OpenCV import error | pip OpenCV 제거 후 `python3-opencv` 사용 |
| unified model이 full frame에서 이상한 것을 잡음 | `preferred-unified`를 직접 `--pipeline yolo`로 돌린 것이 아닌지 확인. 반드시 `--pipeline unified` 사용 |
| FPS가 낮음 | `--no-display --print-timing --max-frames 300`으로 camera/display/model cost 분리 |
| plain cube가 너무 쉽게 뜸 | 현재 overlay는 per-frame입니다. robot actuation 전 repeated evidence를 요구하세요. |

## 관련 문서

| 문서 | 역할 |
| --- | --- |
| [../readme.md](../readme.md) | project overview, quick start, current architecture |
| [../readme_specific.md](../readme_specific.md) | detailed technical reference and historical notes |
| [../EXPERIMENTS.md](../EXPERIMENTS.md) | canonical experiment status index |
| [OPTIMIZATION_EXPERIMENTS.md](./OPTIMIZATION_EXPERIMENTS.md) | ABC runtime optimization log |
