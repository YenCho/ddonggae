# 기술 Reference 및 과거 기록

기준일: 2026-07-04

이 문서는 구현 기준의 상세 reference입니다. 처음 실행하는 법은 [readme.md](./readme.md)를 먼저 보고, 실험의 preferred/fallback/rejected 상태는 [EXPERIMENTS.md](./EXPERIMENTS.md)를 기준으로 보세요.

용어: 문서 기준 **Cube Detector** = 구 `A1`(full-frame object seg), **Face Classifier** = 구 `cube-face unified`. 코드 alias에는 새 이름 `cube-detector` / `face-classifier`가 추가되었고, 경로와 legacy alias(`preferred-a1`, `preferred-unified`, `ABC_model/`, `a1_*`)는 호환성 때문에 그대로 동작합니다. 매핑 표는 [readme.md 용어 정리](./readme.md) 참조.

## 현재 Runtime 구조

현재 runtime 추천 구조는 `A1 object segmentation + cube-face unified model`입니다.

```text
camera frame
  -> A1 object segmentation
       classes: cube_like_object, octahedron, dodecahedron, icosahedron
  -> if A1 class is non-cube shape:
       decide_shape()
  -> if A1 class is cube_like_object:
       crop with padding
       run cube-face unified model on that crop only
       convert unified detections to FaceEvidence
       decide_cube()
```

unified face model은 crop model입니다. full-frame detector로 해석하면 안 됩니다. full-frame detector 역할은 A1이 맡습니다.

## 현재 판단 로직

현재 decision code는 [jetson/abc_inference.py](./jetson/abc_inference.py)에 있습니다.

- `decide_shape(class_name, target_shape=...)`
- `decide_cube(faces, target_shape=..., target_fruit=...)`

현재 webcam unified path는 [jetson/realtime_seg_cam.py](./jetson/realtime_seg_cam.py)에 있습니다.

- `process_unified_crop_frame()`
- `draw_result_at_offset()`
- `resolve_model_path()`

### Object Routing 흐름

| A1 결과 | 다음 단계 | 판단 경로 |
| --- | --- | --- |
| `octahedron` | unified crop 없음 | `decide_shape()` |
| `dodecahedron` | unified crop 없음 | `decide_shape()` |
| `icosahedron` | unified crop 없음 | `decide_shape()` |
| `cube_like_object` | crop -> unified model | `decide_cube()` |

### Cube 판단 표

| FaceEvidence 요약 | Identity | Action | 이유 |
| --- | --- | --- | --- |
| fruit class 1종, `--target-fruit`와 일치 | `fruit_cube:<fruit>` | `pickup` | target fruit face가 보임. rulebook set-2 object는 20점 |
| fruit class 1종, `--target-fruit`와 불일치 | `fruit_cube:<fruit>` | `avoid` | non-target fruit가 보임. 잘못 집으면 20점의 2배 penalty |
| fruit class 1종, target 미설정 | `fruit_cube:<fruit>` | `inspect` | fruit cube는 식별했지만 pickup target이 없음 |
| fruit class 여러 종 | `conflicting_fruit_cube` | `inspect` | 한 cube에서 여러 fruit class가 나오면 규칙상 충돌 |
| reliable face 없음 | `cube_like_unresolved` | `inspect` | cube-like object에 아직 reliable face evidence가 없음 |
| unknown/reject face 존재 | `cube_like_unresolved` | `inspect` | cube를 안전하게 분류할 수 없음 |
| plain face 1개만 존재 | `blank_only_cube_ambiguous` | `inspect` | 저각도에서 위쪽 fruit면이 프레임 밖일 수 있음 |
| plain face 2개 이상, fruit/unknown 없음 | `plain_cube` | `--target-shape cube`이면 `pickup`, 아니면 `skip`/`inspect` | 규칙상 fruit면이 위로 배치되어 2 blank면은 fruit cube 불가 |

중요 정책:

- A1 `cube_like_object`만으로는 절대 `plain_cube`가 되면 안 됩니다.
- fruit evidence는 plain evidence보다 우선합니다.
- visible non-target fruit face가 있으면 plain이 아니므로 Task1 cube로 집으면 안 됩니다.
- 현재 코드의 `plain_cube`는 per-frame identity입니다. 실제 robot pickup은 higher-level tracker에서 여러 frame/view의 반복 plain evidence를 요구해야 안전합니다.

### Temporal Vote / Track Memory

2026-07-03 기준:

- `decide_cube()` 내부에는 temporal vote가 없습니다.
- `realtime_seg_cam.py --print-model-output`은 per-frame model output을 출력합니다.
- unified preview path에는 top1-top2 class-margin gate가 없습니다.
- robot이 더 안전하게 actuation하려면 `decide_cube()` 바깥의 tracker/caller에서 repeated evidence를 구현해야 합니다.

권장 future gate:

```text
same tracked object
  AND at least N good frames/views with only plain faces
  AND no fruit face ever observed for that track
  AND no unknown/conflicting face in the recent window
  -> promote plain cube candidate
```

## FaceEvidence 구조

`FaceEvidence` dataclass:

| 필드 | 타입 | unified mode의 현재 source | 설명 |
| --- | --- | --- | --- |
| `face_index` | `int` | crop result 내부 순서 | crop/object마다 다시 index가 붙습니다. |
| `kind` | `str` | class mapping | `fruit_face`, `plain_face`, `unknown_face` |
| `label` | `str` | unified class name | `apple`, `orange`, `banana`, `pineapple`, `plain` |
| `confidence` | `float` | YOLO detection confidence | class confidence로 사용합니다. |
| `detector_confidence` | `float` | `confidence`와 동일 | ABC는 detector/classifier confidence가 분리될 수 있습니다. |
| `box_xyxy` | `list[float]` | frame 좌표로 offset된 face detection box | drawing과 geometry helper에 사용합니다. |
| `quad_xy` | `list[list[float]]` | unified mode에서는 box corner | ABC B/pose path는 refined quad를 만들 수 있습니다. |
| `visible_pixels` | `int` | polygon이 있으면 mask area, 없으면 box area | quality/visibility proxy입니다. |
| `raw_quad_xy` | `list[list[float]]` | unified mode에서는 `quad_xy`와 동일 | pre-refinement 저장용입니다. |
| `segments_xy` | `list[list[list[float]]]` | YOLO segmentation polygon | 없으면 box polygon fallback을 사용합니다. |
| `source` | `str` | `A1+Unified` | ABC는 `A2+B+C`를 사용합니다. |

관련 dataclass:

| Dataclass | 역할 |
| --- | --- |
| `ObjectEvidence` | A1 object, face evidence, final decision을 묶습니다. |
| `ObjectDecision` | final identity/action/reason/visible face count/expected score를 담습니다. |
| `FrameOutput` | 한 frame의 object evidence list입니다. |

## Threshold와 튜닝 기준

아래 값은 코드 기본값입니다. 최종 robot policy는 synthetic validation만으로 정하면 안 되고 real camera validation으로 조정해야 합니다.

| Setting | Default | File / option | 적용 대상 |
| --- | ---: | --- | --- |
| A1 object confidence, unified path | `0.25` | `jetson/realtime_seg_cam.py --unified-a1-conf` | full-frame A1 |
| Unified face confidence | `0.25` | `jetson/realtime_seg_cam.py --conf` | cube crop face model |
| Fruit acceptance floor (guard) | `0.45` | `jetson/realtime_seg_cam.py --fruit-min-conf` | 이 값 미만의 fruit face는 unknown 처리 → cube가 fruit로 확정되지 않고 `cube_like_unresolved`(inspect). blank/sliver 저신뢰 fruit 오탐 방어. 0이면 비활성 |
| Unified crop padding | `0.18` | `jetson/realtime_seg_cam.py --unified-crop-pad` | A1 box 주변 crop |
| Unified face image size | `224` | `jetson/realtime_seg_cam.py --imgsz` | crop inference |
| Unified A1 image size | `640` | `jetson/realtime_seg_cam.py --unified-a1-imgsz` | full-frame A1 inference |
| Preview overlap suppression | `0.6` | `jetson/realtime_seg_cam.py --overlap` | 겹친 lower-confidence box 제거 |
| ABC A1 confidence | `0.25` | `jetson/abc_inference.py` | ABC full-frame object gate |
| ABC A2 confidence | `0.20` | `jetson/abc_inference.py --a2-conf` | face segmentation gate |
| ABC C confidence | `0.55` | `jetson/abc_inference.py --c-conf` | 이 값보다 낮으면 C result는 unknown |
| ABC min face pixels | `120` | `jetson/abc_inference.py` | A2 face filter |
| ABC min face-object overlap | `0.45` | `jetson/realtime_seg_cam.py --min-face-object-overlap` | object mask 밖 face reject |
| ABC C min visible ratio | `0.40` | `jetson/abc_inference.py` | C crop quality |

현재 있는/없는 gate:

- **fruit acceptance floor 있음** (2026-07-08 추가): `decide_cube(fruit_min_conf=0.45)` — 저신뢰 fruit face를 unknown으로 처리해 blank/sliver 오탐이 cube를 fruit로 뒤집지 못하게 함. recorded holdout에서 정답 orange 100% 유지(min conf 0.463) + blank 오탐 ~88% 제거(max conf 0.496) 기준으로 정한 데이터 기반 값. 실측 튜닝 대상.
- unified top1-top2 margin gate 없음 (FaceEvidence가 winning-class conf만 보유 — margin gate는 top2 plumbing 필요)
- `decide_cube()` 내부 unified temporal vote 없음
- unified explicit `unknown` class 없음 (low-confidence/no-detection + 위 fruit floor로 reject)

## 현재 Dataset 계보

### Meta V2

Meta V2는 다음 model/export의 기반이 되는 합성 dataset 계열입니다.

- A1 object segmentation
- A2 visible face segmentation
- B quad recovery sidecar
- C face classifier crops
- cube-face unified crop dataset

중요 metadata:

```text
object visible/full mask
face visible/full mask
face quad_xy / quad_xy_raw
corner visibility
occluder object ids
camera intrinsics/extrinsics
scene/object transform
```

### Cube-Face Unified Dataset

base export script:

```text
scripts/export_meta_v2_cube_face_unified_dataset.py
```

현재 주요 dataset path:

```text
datasets/meta_v2_50000_cube_face_unified_v1/data.yaml
datasets/cube_face_unified_finetune_wholefruit_plainhard_materialized_v2_balanced/data.yaml
datasets/cube_face_unified_finetune_verified_fruit_balanced_materialized_v1/data.yaml
```

Unified class names:

```text
0 apple
1 orange
2 banana
3 pineapple
4 plain
```

### Booster Dataset Scripts

```text
scripts/make_cube_face_unified_booster.py
scripts/materialize_cube_face_unified_finetune_mix.py
scripts/run_cube_face_unified_booster_finetune.ps1
scripts/build_ai_fruit_booster_from_sheets.py
scripts/build_ai_fruit_booster_pool.py
scripts/build_ai_fruit_runtime_degraded_booster.py
scripts/build_fruit_texture_training_candidate.py
```

Booster training rule:

- 새 dataset folder를 만듭니다.
- 새 run name을 사용합니다.
- production weights를 덮어쓰지 않습니다.
- 자동 promotion은 하지 않습니다.

## 현재 Weight 경로

### Preferred Runtime

```text
jetson/ABC_model/meta_v2_a1_objectseg/a1_yolo26s_seg_meta_v2_50000/weights/best.pt
jetson/ABC_model/cube_face_unified/preferred_v2/weights/best.pt
```

### Preferred Unified Training Run

```text
runs/segment/cube_face_unified_yolo26n_seg_readd_coloroutlier_base_from_pruned_adamw_lr1e5_ft_v1/weights/best.pt
```

### ABC Fallback Weights

```text
jetson/ABC_model/meta_v2_a2_faceseg/a2_yolo26n_seg_meta_v2_50000/weights/best.pt
jetson/ABC_model/meta_v2_b_tinyquadnet/b_tinyquadnet_meta_v2_50000/weights/best.pt
jetson/ABC_model/meta_v2_c_facecls/c_mobilenetv3small_meta_v2_50000_runtimewarp_ft/weights/best.pt
```

### Aliases

`jetson/realtime_seg_cam.py`에 정의되어 있습니다.

| Alias | 의미 |
| --- | --- |
| `cube-detector` | Cube Detector(구 A1) full-frame object segmentation weight의 새 primary alias (ONNX: `cube-detector-onnx`) |
| `face-classifier` | Face Classifier(구 cube-face unified) weight의 새 primary alias (ONNX: `face-classifier-onnx`) |
| `preferred-a1` | legacy — `cube-detector`와 동일한 weight로 resolve |
| `preferred-unified` | legacy — `face-classifier`와 동일한 weight로 resolve (`preferred-unified-onnx` = `face-classifier-onnx`) |
| `latest-unified` | legacy/실험용 alias입니다. 이름만 보고 production으로 취급하면 안 됩니다. |

## Texture 정책

| Source | 상태 | 허용 사용처 | 금지 사용처 | 설명 |
| --- | --- | --- | --- | --- |
| production texture mix | current | Meta V2 base generation | 조용한 ratio 변경 | canonical policy는 `reports/texture_policy_meta_v2_50000_20260623.md` |
| whole-fruit/plain-hard booster | previous preferred fine-tune source | conservative materialized v2 fine-tune | direct overwrite | plain false positive를 줄이되 fruit confidence를 과하게 깎지 않기 위한 source |
| verified fruit balanced booster | previous preferred fine-tune source | class-balanced fruit confidence correction | direct overwrite | obvious orange/banana/pineapple을 apple로 오인식하는 위험을 줄이기 위한 검증 과일 source |
| flat-icon boundary booster | previous preferred fine-tune source | webcam-style printed icon class-boundary correction | direct overwrite | 실제 orange cube crop이 apple/orange 사이에서 흔들린 문제를 mimic한 source |
| HSV stronger + color-outlier pruned booster | previous preferred fine-tune source | color-boundary correction with rejected texture pruning | direct overwrite | 색 변화 robustness를 늘리되 apple/orange/pineapple/banana 색 outlier texture가 학습을 오염시키지 않도록 제거한 source |
| readd color-outlier base | current preferred fine-tune source | pruning 과잉 제거 복구 (실물 텍스처 kind=base 13,465장) | boost(HSV 인위 변형) 크롭 재사용 | pruning이 hue 경계를 과하게 좁힌 것을 복구한 source. 게이트(f2f ≥5%) 통과 후에만 학습 |
| 사용자 orange crops + 실물 녹화 세트 | evaluation-only holdout | 평가/진단/회귀 확인 | **모든 형태의 training 절대 금지** | `reports/unified_recorded_input_frame_sort_20260706_165440` 포함. holdout 오염 시 실물 성능 판정 기준이 사라짐 |
| web cutout sets | rejected/cleanup required | probing과 error examples | direct training | manual cleanup 없이는 너무 noisy |
| AI whole-fruit sets | candidate | low-ratio booster experiments | automatic production replacement | preview와 real camera probe 필요 |
| AI printed/full-square sets | candidate | targeted confidence booster | automatic production replacement | probe는 강하지만 아직 promotion 아님 |

Fruit rules:

- apple은 red apple
- orange는 orange-colored whole orange
- banana는 yellow, unpeeled
- pineapple은 exposed yellow flesh가 없는 brown/green whole pineapple
- 현재 competition 가정에서는 slice/cut fruit를 쓰지 않습니다.

## Validation 정책

모델을 promotion하기 전 확인 순서:

1. `preferred-unified`와 비교합니다.
2. materialized validation metrics를 확인합니다.
3. known apple/orange/plain hard cases를 돌립니다.
4. noisy example을 production에 섞지 않고 harvested/AI candidates를 probe합니다.
5. webcam 또는 robot-camera replay를 실행합니다.
6. report path와 decision을 [EXPERIMENTS.md](./EXPERIMENTS.md)에 기록합니다.

2026-07-02 apple-bias check에서는 clean synthetic/AI set에서 일반적인 apple collapse는 확인되지 않았습니다. non-apple -> apple 비율이 높았던 곳은 주로 noisy web cutout, 특히 pineapple/orange web 후보였습니다. 따라서 class quantity를 무작정 바꾸기보다 noisy web data cleanup/rejection이 우선입니다.

## 현재 알려진 약점

- 단일 frame `plain_cube`는 track memory 없이는 actuation 기준으로 너무 낙관적입니다.
- unified model에는 explicit `unknown` class가 없습니다.
- 여러 whole-fruit probe에서 banana confidence가 낮습니다.
- real/curated domain에서 pineapple이 banana로 혼동될 수 있습니다.
- current preferred(readd best epoch 83)는 mix validation 기준으로 mAP는 안정적이지만, 실물 printed-icon 구성은 미해결입니다: 사용자 orange hard-case 2장 중 1장 여전히 `apple`(0.604), 실물 녹화 세트 orange 인식 65.1% (목표 80%).
- 원인은 확정됨: 모델의 인식 절벽이 아이콘 면적 11~20%에 있고 실물 인쇄 아이콘은 8~10% 구간임 (composition gap, `reports/cube_face_unified_eval/why_recorded_set_fails_20260708/analysis.md`). 다음 booster는 printed-label composition(면적 6~16%) 방향이어야 합니다.
- blank/sliver 크롭에서 저신뢰(0.23~0.39) banana 오탐이 늘었습니다. runtime ambiguity guard 임계(≥0.4) 확인이 필요합니다.
- unified quad는 현재 box-derived입니다. ABC는 더 풍부한 quad/pose logic을 갖습니다.
- web cutout은 manual cleanup 또는 더 강한 semantic filtering이 필요합니다.

## 빠른 실행 명령

### Unified 웹캠 Preview

```powershell
powershell -ExecutionPolicy Bypass -File jetson\run_webcam_preview.ps1 `
  -Pipeline unified `
  -Camera 0 `
  -Device 0 `
  -TargetShape cube `
  -TargetFruit apple `
  -PrintModelOutput
```

### Jetson Preview

```bash
./jetson/run_webcam_preview.sh \
  --pipeline unified \
  --camera 0 \
  --camera-backend v4l2 \
  --device 0 \
  --target-shape cube \
  --target-fruit apple \
  --print-model-output
```

### 전체 재현 명령: Unified Dataset Export

```powershell
python scripts\export_meta_v2_cube_face_unified_dataset.py `
  --source_dataset datasets\meta_v2_50000_coco_texture_v1 `
  --output_root datasets\meta_v2_50000_cube_face_unified_v1 `
  --source_split train `
  --val_ratio 0.10 `
  --seed 20260628 `
  --crop_size 224 `
  --crop_pad 0.18 `
  --min_object_pixels 80 `
  --min_face_pixels 120
```

### 전체 재현 명령: Booster Fine-Tune

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_cube_face_unified_booster_finetune.ps1 `
  -SkipGenerate `
  -Epochs 120 `
  -Batch 512 `
  -Workers 4 `
  -Patience 30 `
  -EmailIntervalSeconds 1800
```

## 과거 기록 Archive

과거 기록 안내: 이 섹션들은 context 보존용입니다. `preferred`로 명시된 경우를 제외하면 현재 runtime recommendation이 아닙니다.

### 2026-07-04

현재 preferred 변경:

- `cube_face_unified_yolo26n_seg_hsv_pruned_coloroutlier_from_last_adamw_lr1e5_ft_v1` validation-best checkpoint를 Jetson unified preferred로 올렸습니다.
- 시작 checkpoint는 `cube_face_unified_yolo26n_seg_hsv_ratio_20000_stronger_from_color_shift_last_adamw_lr1e5_ft_v1/weights/last.pt`입니다.
- 학습 data는 `datasets/cube_face_unified_finetune_meta_v2_50000_plus_hsv_ratio_20000_stronger_pruned_coloroutlier_v1/data.yaml`입니다.
- dataset은 HSV stronger materialized set에서 rejected color-outlier fruit texture를 사용한 cube crop을 제거한 것입니다. train `70000 -> 51139`, val `7000 -> 5085`로 pruning되었습니다.
- 제거량은 pineapple `6197`, apple `5395`, orange `5009`, banana `4175` source-object 기준입니다.
- 학습 설정은 AdamW, `lr0=1e-5`, `lrf=0.05`, `warmup_epochs=0`, `batch=512`, `workers=0`, `imgsz=224`, `patience=30`입니다.
- 160 epoch 설정으로 시작했고 EarlyStopping으로 epoch 55에서 종료되었습니다. best는 mask mAP50-95 기준 epoch 24입니다.
- best epoch 24 validation row: all box P/R/mAP50/mAP50-95 `0.940 / 0.918 / 0.972 / 0.925`, all mask P/R/mAP50/mAP50-95 `0.940 / 0.918 / 0.972 / 0.897`.
- 재생성한 validation 결과는 all mask P/R/mAP50/mAP50-95 `0.941 / 0.918 / 0.972 / 0.896`입니다. orange mask P/R/mAP50/mAP50-95는 `0.932 / 0.890 / 0.958 / 0.898`입니다.
- 사용자가 보낸 orange webcam crop 2장은 학습에는 쓰지 않고 평가에만 사용했습니다. 최신 `best.pt`는 한 장을 `orange 0.935`, 한 장을 `apple 0.521`로 예측했습니다.
- Jetson bundled path `jetson/ABC_model/cube_face_unified/preferred_v2/weights/best.pt`로 run `best.pt`를 복사했고 SHA256은 `6EC2F572475F5649D262F46E886A3149FE6FA6ED72E36BF00B7AA40686FD9153`입니다.
- 판단: 최신 Jetson runtime 후보로는 갱신하지만, orange printed-style boundary는 완전 해결이 아니므로 다음 booster 설계에서 계속 우선순위로 둡니다.

### 2026-07-03

현재 preferred 변경:

- `cube_face_unified_yolo26n_seg_verified_plus_flat_icon_boundary_adamw_lr3e5_ft_v1` validation-best checkpoint를 Jetson unified preferred로 올렸습니다.
- 시작 checkpoint는 `cube_face_unified_yolo26n_seg_verified_fruit_balanced_from_last_adamw_lr5e5_ft_v1/weights/best.pt`입니다.
- 학습 data는 `datasets/cube_face_unified_finetune_verified_plus_flat_icon_boundary_materialized_v2/data.yaml`입니다.
- 학습 설정은 AdamW, `lr0=3e-5`, `lrf=0.05`, `warmup_epochs=0`, `cos_lr=True`, `batch=512`, `workers=4`, `imgsz=224`입니다.
- 로컬 산출물 기준 `epochs=100` 설정 run은 `results.csv`에 epoch 76까지 기록되었고, validation-best는 epoch 46입니다.
- best epoch 46 validation row: all box P/R/mAP50/mAP50-95 `0.941 / 0.919 / 0.975 / 0.933`, all mask P/R/mAP50/mAP50-95 `0.941 / 0.919 / 0.974 / 0.901`.
- 사용자가 보낸 orange webcam crop 2장은 epoch 51 snapshot에서 fruit top label이 모두 `orange`로 바뀌었습니다.
- Jetson bundled path `jetson/ABC_model/cube_face_unified/preferred_v2/weights/best.pt`로 run `best.pt`를 optimizer-strip하여 복사했고 SHA256은 `D7814CCA03A88FF0DCFD91903104025EC6982AC5C422AA0CF98DFB66361AA727`입니다.

이전 preferred 변경:

- `cube_face_unified_yolo26n_seg_verified_fruit_balanced_from_last_adamw_lr5e5_ft_v1`을 Jetson unified preferred로 올렸습니다.
- 시작 checkpoint는 `cube_face_unified_yolo26n_seg_wholefruit_plainhard_balanced_ft_v2/weights/last.pt`입니다.
- 학습 data는 `datasets/cube_face_unified_finetune_verified_fruit_balanced_materialized_v1/data.yaml`입니다.
- 학습 설정은 AdamW, `lr0=5e-5`, `lrf=0.05`, `warmup_epochs=0`, `batch=512`, `workers=4`, `imgsz=224`입니다.
- 120 epoch 설정으로 시작했고 EarlyStopping에 의해 102 epoch에서 종료, best epoch는 72입니다.
- best.pt validation: all box P/R/mAP50/mAP50-95 `0.919 / 0.916 / 0.968 / 0.923`, all mask P/R/mAP50/mAP50-95 `0.921 / 0.914 / 0.968 / 0.896`.
- class별 mask mAP50-95: apple `0.898`, orange `0.908`, banana `0.899`, pineapple `0.911`, plain `0.863`.
- Jetson bundled path `jetson/ABC_model/cube_face_unified/preferred_v2/weights/best.pt`로 복사했고 SHA256은 `59FB77A3E8C06B8F63ABD0A237CCE1F94DAF924043993C1E3D01E4E0B4C1AACC`입니다.

### 2026-07-02

과거 기록 안내: 이 섹션은 context 보존용입니다.

- unified mode가 ABC preview처럼 per-frame model output과 final identity를 보여줄 수 있도록 webcam preview를 보강했습니다.
- `preferred-a1`, `preferred-unified` alias를 확인했습니다.
- fair webcam comparison 실행: captured benchmark run reported as `cpu` 조건에서 current unified `3.83 FPS`, old ABC `1.29 FPS`.
- fruit texture probe와 non-apple -> apple confusion을 확인했습니다. clean AI set에서는 broad apple collapse가 없었고, noisy web cutout에서 집중되었습니다.
- documentation structure를 current reference와 historical archive로 분리했습니다.

관련 report:

```text
reports/share/fair_30s_compare_20260702_155104/summary.md
reports/cube_face_unified_eval/fruit_texture_candidate_probe_20260702/report.md
reports/cube_face_unified_eval/fruit_texture_web_cutout_probe_20260702/report.md
```

### 2026-06-30

과거 기록 안내: 이 섹션은 context 보존용이며 현재 runtime recommendation이 아닙니다.

- AI whole-fruit와 printed/full-square booster candidates를 만들고 probe했습니다.
- AI printed/full-square는 clean probe confidence가 강했지만 production으로 promotion하지 않았습니다.
- AI/web candidates는 `datasets/fruit_textures/other/*` 아래 분리 상태로 유지합니다.

### 2026-06-29

과거 기록 안내: 이 날짜의 preferred 결과는 balanced v2입니다.

- `wholefruit_plainhard_materialized_ft_v1`은 plain hard negative를 개선했지만 fruit confidence를 과하게 낮췄습니다.
- `wholefruit_plainhard_balanced_ft_v2`는 plain 개선을 유지하면서 apple confidence를 회복했습니다.
- v2가 preferred cube-face unified checkpoint가 되었습니다.

핵심 report:

```text
reports/cube_face_unified_eval/balanced_v2_20260629/summary.md
```

### 2026-06-28

과거 기록 안내: 이 섹션은 context 보존용이며 현재 runtime recommendation이 아닙니다.

- RTX 5080 기준 4 cube-like object/about-12 visible-face benchmark에서 ABC runtime 20+ FPS 최적화를 진행했습니다.
- 중요한 교훈: image size는 YOLO stride 요구사항을 지켜야 합니다. 예를 들어 `208`처럼 stride-compatible하지 않은 값은 warning과 함께 자동 조정될 수 있으므로, warning message는 실험 유효성 판단에 반드시 포함해야 합니다.
- ABC는 accepted behavior gate 위에서 약 `20-21 FPS` fallback zone에 도달했지만, 더 공격적인 22+ FPS variant는 behavior를 훼손했습니다.
- 더 큰 결론은 ABC micro-optimization보다 A2+B+C를 cube-crop unified model로 접는 구조 변화가 더 유망하다는 점입니다.

Log:

```text
jetson/OPTIMIZATION_EXPERIMENTS.md
reports/abc_runtime_optimization/
```

### 2026-06-23

과거 기록 안내: 이 섹션은 context 보존용이며 단독으로 현재 runtime recommendation이 아닙니다.

- Meta V2가 visible face, full face, quads, occlusion metadata를 저장하는 main synthetic data 방향이 되었습니다.
- uncontrolled web texture noise를 피하기 위해 production texture policy를 고정했습니다.
- 프로젝트는 object-only recognition에서 face-evidence-based cube identity로 이동했습니다.

Report:

```text
reports/texture_policy_meta_v2_50000_20260623.md
```

### 2026-05-20

과거 기록 안내: 이 섹션은 context 보존용이며 Meta V2 이후 deprecated입니다.

- 초기 8-class object-only YOLO 실험은 banana, orange, pineapple, apple, cube, octahedron, dodecahedron, icosahedron object label을 직접 사용했습니다.
- object detection proof-of-concept로는 유용했지만, rulebook 기준에서는 fruit identity가 visible cube face에 달려 있으므로 충분하지 않았습니다.

관련 report:

```text
reports/initial_model_may14_summary.md
reports/yolo_8class_v3_50000_may17_assets/summary_yolo_8class_v3_50000_may17.md
```
