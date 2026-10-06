# HSV pruned color-outlier unified validation

기준일: 2026-07-04

## 대상 checkpoint

```text
runs/segment/cube_face_unified_yolo26n_seg_hsv_pruned_coloroutlier_from_last_adamw_lr1e5_ft_v1/weights/best.pt
```

이 checkpoint는 `jetson/ABC_model/cube_face_unified/preferred_v2/weights/best.pt`로 복사되어 `preferred-unified` alias의 최신값이 되었습니다.

## 학습 의도

- HSV stronger booster로 색 경계 robustness를 늘립니다.
- 단, fruit texture 중 과하게 색공간을 침범한 outlier를 사용한 cube crop은 제거합니다.
- 기존 production/base dataset을 직접 수정하지 않고 pruned materialized dataset을 별도 폴더로 사용합니다.

## Dataset

```text
datasets/cube_face_unified_finetune_meta_v2_50000_plus_hsv_ratio_20000_stronger_pruned_coloroutlier_v1/data.yaml
```

- source materialized: train `70000`, val `7000`
- pruned kept: train `51139`, val `5085`
- removed: train `18861`, val `1915`
- removed by source-object class: pineapple `6197`, apple `5395`, orange `5009`, banana `4175`

## 학습 결과

`results.csv` 기준 best epoch는 mask mAP50-95 기준 epoch 24입니다. EarlyStopping은 epoch 55에서 멈췄습니다.

| Metric | Value |
| --- | ---: |
| all box P/R/mAP50/mAP50-95 | `0.940 / 0.918 / 0.972 / 0.925` |
| all mask P/R/mAP50/mAP50-95 | `0.940 / 0.918 / 0.972 / 0.897` |
| last epoch all mask mAP50/mAP50-95 | `0.972 / 0.895` |

재생성한 validation 결과:

| Class | Box P/R/mAP50/mAP50-95 | Mask P/R/mAP50/mAP50-95 |
| --- | --- | --- |
| all | `0.941 / 0.917 / 0.972 / 0.926` | `0.941 / 0.918 / 0.972 / 0.896` |
| apple | `0.931 / 0.922 / 0.975 / 0.935` | `0.931 / 0.922 / 0.975 / 0.906` |
| orange | `0.931 / 0.890 / 0.958 / 0.920` | `0.932 / 0.890 / 0.958 / 0.898` |
| banana | `0.951 / 0.920 / 0.971 / 0.926` | `0.951 / 0.920 / 0.971 / 0.902` |
| pineapple | `0.922 / 0.917 / 0.970 / 0.926` | `0.922 / 0.917 / 0.971 / 0.904` |
| plain | `0.971 / 0.938 / 0.988 / 0.926` | `0.970 / 0.937 / 0.987 / 0.872` |

Confusion matrix:

- `confusion_matrix.png`
- `confusion_matrix_normalized.png`

## Orange hard-case probe

사용자가 보낸 orange cube crop 2장은 학습에 사용하지 않고 평가에만 사용했습니다.

| Model | Image 1 fruit top | Image 2 fruit top |
| --- | --- | --- |
| previous init last | `orange 0.552` | `orange 0.935` |
| current pruned best | `apple 0.521` | `orange 0.935` |

판단: validation mAP와 confusion matrix는 안정적이지만, 실제 orange printed-style hard-case는 아직 완전히 해결되지 않았습니다. 최신 Jetson runtime checkpoint로는 갱신하되, 다음 booster는 orange/apple printed-style boundary를 실제 A1 crop 형태로 더 정교하게 보강해야 합니다.
