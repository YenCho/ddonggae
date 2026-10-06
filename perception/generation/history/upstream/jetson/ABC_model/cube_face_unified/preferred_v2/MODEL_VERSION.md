# cube-face unified preferred_v2 (Face Classifier)

기준일: 2026-07-08 (sparse-icon 120-epoch 본 학습 완료·승격)

이 폴더의 `weights/best.pt`는 Jetson runtime에서 `face-classifier`(구 `preferred-unified`) alias로 로드되는 현재 preferred Face Classifier입니다. 파일명은 `best.pt` 유지.

현재 bundled weight는 **sparse printed-icon booster mix로 120 epoch fine-tune한 정식 run의 best (epoch 47, patience 35로 82에서 조기종료)** 입니다. 앞서 승격했던 30-epoch 프로브와 통계적으로 동급이며(아래 비교), 완전 수렴된 정식 run이라 이것으로 최종 확정했습니다.

## Source

```text
runs/segment/cube_face_unified_yolo26n_seg_sparse_icon_from_readd_adamw_lr1e5_ft_v1/weights/best.pt
```

## Lineage

```text
... -> hsv_pruned_coloroutlier -> readd_coloroutlier_base (epoch 83)
  -> sparse_icon_from_readd probe30 (epoch 16, 중간 승격)
  -> sparse_icon_from_readd 120ep 본학습 (best epoch 47)  <- 현재
```

## Training Summary

- Init: readd_coloroutlier_base best (직전 preferred)
- Data: `datasets/cube_face_unified_finetune_readd_plus_sparse_icon_10k_v1` — train 77,099 = readd mix 64,604 + sparse-icon booster 12,495 (`realistic_a4_sparse_icon`: patch 0.24~0.40 + posterize + webcam_hard, 10k scenes) / val 7,893
- AdamW lr0=1e-5, lrf=0.05, cos_lr, warmup 0, batch 512, imgsz 224, workers 4, cache=disk, epochs 120 (patience 35 → 82에서 stop, best 47), 7.1h
- ONNX 재수출 완료 (opset 20, imgsz 224). `best.engine`은 Jetson에서 재빌드 필요.

## Validation / Hard-Set Snapshot

mix val (best epoch 47): all mask mAP50/mAP50-95 `0.967 / 0.894`, box `0.968 / 0.924`. orange mask mAP50 0.953.

직전 preferred(readd) 및 중간 probe30과의 hard-set 비교:

| 지표 | readd best | probe30 (ep16) | **120ep (ep47, 현재)** |
| --- | --- | --- | --- |
| recorded 실물 orange (평가 전용, 86 crops) | 65.1% | 96.5% | **95.3%** |
| orange-apple margin p10 | -0.385 | +0.121 | **+0.111** |
| arena v3 probe apple / orange | 68.8 / 69.7% | 75.0 / 75.8% | **81.2 / 78.8%** |
| mix val mask mAP50-95 | 0.894 | 0.893 | **0.894** |
| blank 저신뢰 fruit 오탐 | 14/25 | 16/25 | **17/25** — runtime ambiguity guard(≥0.4) 필수 |

probe30과 120ep은 recorded 1 crop 차이(노이즈)로 사실상 동급이며, 120ep이 arena에서 소폭 우세. 목표(recorded 80%)를 두 모델 모두 크게 초과.

리포트: `reports/cube_face_unified_eval/recorded_hardset_120ep_20260708/`(작성 예정), `arena_v3_probe_100_v1_top_class_120ep/`, `sparse_icon_probe_iterations_20260708/summary.md`

## File Hash

```text
SHA256 1203602222BDBB3A353844E57BD75AFCB6088280C4FEDF9583B902BA907BAC3E weights/best.pt
SHA256 934B5DAF36F3C9E68FB823AFBD6DC735A1C6646D8F40BCBA3D366D528F24C444 weights/best.onnx
```
