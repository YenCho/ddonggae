# A1 crop jitter + context booster v1

## 목적

실제 webcam runtime에서는 A1이 검출한 `cube_like_obj` bbox가 매 프레임 완전히 같은 위치와 크기로 들어오지 않는다. Unified face model은 그 crop을 `224x224`로 resize해서 판단하므로, 작은 bbox 흔들림과 주변 물체/배경 침범만으로도 face class boundary가 흔들릴 수 있다.

이 booster는 그 요인만 분리해서 추가한다. 새 cube geometry, flat icon, 임의 도형 렌더링은 만들지 않는다.

## 생성 원칙

- Source는 기존 materialized crop dataset만 사용한다.
- 원본 이미지와 segmentation label을 같은 affine matrix로 같이 변환한다.
- crop이 밀리거나 작아져 생기는 가장자리 영역은 다른 기존 crop에서 가져온 흐린 context pixel로 채운다.
- 새로운 semantic label이나 새로운 과일 모양은 만들지 않는다.
- 기존 boosted set 위에 하나의 요인만 추가한다.

## 산출물

- Booster: `datasets/cube_face_unified_booster_a1_crop_jitter_context_v1`
- Mixed train set: `datasets/cube_face_unified_finetune_meta_v2_50000_plus_hsv_ratio_20000_plus_a1_crop_jitter_context_v1`
- Preview: `datasets/cube_face_unified_booster_a1_crop_jitter_context_v1/previews/a1_crop_jitter_context_contact_sheet.jpg`

## 구성

- Source base: `datasets/cube_face_unified_finetune_meta_v2_50000_plus_hsv_ratio_20000_stronger_v1`
- Booster train/val: `10000 / 1000`
- Mixed train/val: `80000 / 8000`
- Primary class balance in booster train: `apple 2000`, `orange 2000`, `banana 2000`, `pineapple 2000`, `plain 2000`
- Case mix:
  - normal crop jitter: `55%`
  - border/context crop jitter: `35%`
  - tight crop: `10%`

## 재생성 명령

```powershell
cd C:\Users\user\Documents\Data_Generation_Blender
$py = 'C:\Users\user\anaconda3\envs\ai_robotics\python.exe'

& $py scripts\make_cube_face_unified_a1_crop_jitter_booster.py `
  --source_root datasets\cube_face_unified_finetune_meta_v2_50000_plus_hsv_ratio_20000_stronger_v1 `
  --output_root datasets\cube_face_unified_booster_a1_crop_jitter_context_v1 `
  --train_count 10000 `
  --val_count 1000 `
  --preview_count 160 `
  --reset

& $py scripts\materialize_cube_face_unified_finetune_mix.py `
  --base_dataset datasets\cube_face_unified_finetune_meta_v2_50000_plus_hsv_ratio_20000_stronger_v1 `
  --booster_dataset datasets\cube_face_unified_booster_a1_crop_jitter_context_v1 `
  --output_root datasets\cube_face_unified_finetune_meta_v2_50000_plus_hsv_ratio_20000_plus_a1_crop_jitter_context_v1 `
  --base_sample_count 70000 `
  --base_val_count 7000 `
  --booster_repeat 1 `
  --mode hardlink `
  --seed 20260703 `
  --reset
```

## 학습 실행 예시

최종 학습은 사용자가 직접 실행한다. 이전 color-shift 학습에서 이어가려면 `last.pt`를 init으로 둔다.

```powershell
cd C:\Users\user\Documents\Data_Generation_Blender

powershell -ExecutionPolicy Bypass -File scripts\run_cube_face_unified_booster_finetune.ps1 `
  -SkipGenerate `
  -SkipMaterialize `
  -MaterializedMixRoot datasets\cube_face_unified_finetune_meta_v2_50000_plus_hsv_ratio_20000_plus_a1_crop_jitter_context_v1 `
  -BaseDataset datasets\cube_face_unified_finetune_meta_v2_50000_plus_hsv_ratio_20000_stronger_v1 `
  -BoosterRoot datasets\cube_face_unified_booster_a1_crop_jitter_context_v1 `
  -InitModel runs\segment\cube_face_unified_yolo26n_seg_meta_v2_color_shift_from_nongeom_adamw_lr1e5_ft_v1\weights\last.pt `
  -Epochs 100 `
  -Patience 30 `
  -Batch 512 `
  -Workers 4 `
  -Optimizer AdamW `
  -Lr0 0.00001 `
  -Lrf 0.05 `
  -WarmupEpochs 0 `
  -EmailTo roboticsai891@gmail.com `
  -EmailIntervalSeconds 1800 `
  -Name cube_face_unified_yolo26n_seg_hsv_plus_a1_crop_jitter_from_color_shift_last_adamw_lr1e5_ft_v1
```

## 다음 판단 기준

이 booster는 mAP만 보지 말고 실제 webcam replay에서 class flip이 줄었는지를 같이 봐야 한다. 특히 같은 orange cube crop에서 `orange/apple`이 반반으로 흔들리던 probe를 고정 평가셋으로 두고, 학습 전후의 top-1 flip 비율과 class confidence margin을 비교한다.
