# Sparse printed-icon booster 레시피 탐색 (v1~v4, 2026-07-08)

목표: recorded 실물 세트(orange 인식 65.1%, 목표 80%+)의 실패 조건인 "흰 면 위 sparse 인쇄 아이콘 + webcam 열화"를 **holdout 없이** Blender로 재현하는 booster 레시피 확정. 각 버전은 100 scene 생성 → 224 crop export → 현재 preferred(readd best epoch 83)로 mAP/혼동 측정.

정책 준수: recorded 세트/사용자 사진은 어떤 형태로도 사용하지 않음. 소스는 학습용 텍스처(`production_meta_v2_50000_v1_color_filtered_v2`) + Blender 생성만.

## 반복 결과

| 버전 | 변경 | fruit top-class 오류 | 주요 클래스 정확도 | mask mAP50/50-95 | 판정 |
|---|---|---:|---|---|---|
| v1 | patch 선형비 0.38~0.62 (sparse 면적만) | 6.4% | 전 클래스 89~97% | 0.975 / 0.922 | 너무 쉬움 — 면적만으론 부족 |
| v2 | + `realistic_webcam_hard` 열화, patch 0.30~0.52, MaxObjects 4 | 10.6% | apple 86 / orange 85% | 0.939 / 0.874 | webcam 열화가 핵심 축 확인 |
| v3 | + 인쇄 그래픽 평탄화(bilateral+k-means posterize, 75%) | 12.1% | apple 82 / pineapple 79% | 0.948 / 0.888 | 소폭 기여 |
| **v4** | **patch 0.24~0.40 (recorded 아이콘/face 기하 일치)** | **17.8%** | **apple 71.9 / pineapple 75.0 / orange 76.5%** | 0.947 / 0.902 (orange box mAP50 0.900, R 0.776; apple R 0.741) | **채택 — recorded 난이도(65%)에 근접** |

보조 관찰:
- v4에서 true orange의 47%(16/34)가 orange conf < 0.80 → recorded의 low-margin 상태 재현.
- apple→orange 6건 등 apple/orange 경계 혼동 방향도 재현.
- 오류 시트 육안 확인: 흰 cube 면 + 소형 인쇄 + webcam 소프트 — recorded 구성과 동일 regime, 현실적 범위 내.
- 커버리지 측정치(p50 0.206)는 webcam 색 캐스트가 흰 종이를 "유색"으로 오염시켜 과대측정됨. 구성 판단은 육안 + 기하 계산 기준.

## 확정 레시피 (`realistic_a4_sparse_icon` profile)

- `--fruit_texture_layout print_label --fruit_print_label_profile realistic_a4_sparse_icon`
- patch 선형비(face 대비): light 0.24~0.40 → 아이콘 면적 ≈ face의 6~16%
- 인쇄 평탄화: bilateral + k-means(4~6색) posterize 75% 확률 (`flatten_patch_to_print_icon`)
- `--camera_artifact_profile realistic_webcam_hard` (WB/노출/감마/노이즈/저해상 리샘플/vignette/JPEG)
- 밝은 종이/면(realistic_a4_mild 계열), 전 fruit 클래스 동일 스타일, plain/negative 유지
- MaxObjects 4, 나머지는 기존 probe 컨벤션

## 산출물

- probe 데이터셋: `datasets/cube_face_unified_booster_realistic_a4_mild_coco_sparse_icon_probe_100_v{1..4}`
- 평가 리포트: `reports/cube_face_unified_eval/realistic_a4_mild_coco_sparse_icon_probe_100_v{1..4}[_top_class]`
- 코드: `scripts/generate_yolo_coco_composite.py`의 `realistic_a4_sparse_icon` profile + `flatten_patch_to_print_icon`

## 10k 본 세트 결과 (2026-07-08)

확정 레시피로 10,000 scene 생성 (8 worker 시 Blender ACCESS_VIOLATION 크래시 → 4 worker resume으로 완료):

- 데이터셋: `datasets/cube_face_unified_booster_realistic_a4_mild_coco_sparse_icon_booster_10k_v1` — train 12,495 / val 1,427 crops (클래스 균형: val 기준 apple 291 / orange 284 / banana 297 / pineapple 321 / plain 1,195)
- 현재 preferred(readd best epoch 83)의 val 성능 — probe v4 난이도가 스케일에서도 유지:

| 지표 | 값 (mix val 대비) |
|---|---|
| fruit top-class 오류 | 239/1,427 = 16.7% |
| apple 정확도 | 72.9% (최다 오답 orange 40 — 목표 경계 재현) |
| orange / pineapple | 77.8% / 77.6% |
| plain | 99.6% (blank→fruit 오탐 없음) |
| all mask mAP50 / mAP50-95 | 0.924 / 0.863 (mix val 0.970 / 0.894) |
| orange mask mAP50 | 0.889 (mix val 0.959) |
| true orange low-margin (conf<0.8) | 41.5% |

## 학습 전 게이트 (why 분석과 동일)

1. booster 학습 후 면적 스윕 프로브 8% 구간 orange 12.5% → 90%+
2. recorded 세트 평가-only ≥ 80%
3. mix/pruned val 회귀 없음, blank fruit 오탐 비증가
