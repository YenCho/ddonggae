# Sparse-icon 120ep 본 학습 — hard-set 최종 비교 (2026-07-08)

`cube_face_unified_yolo26n_seg_sparse_icon_from_readd_adamw_lr1e5_ft_v1` (120 epoch 설정, patience 35로 82 stop, best epoch 47)를 30ep 프로브 및 직전 readd preferred와 비교.

세트: recorded 실물 orange cube 녹화 visible-icon 86 crops (평가 전용 holdout, 학습 미사용) + arena v3 probe 192 crops + mix val.

| 지표 | readd best | probe30 (ep16) | **120ep (ep47)** |
| --- | --- | --- | --- |
| recorded orange 인식 | 65.1% (56/86) | 96.5% (83/86) | **95.3% (82/86)** |
| orange-apple margin p10 | -0.385 | +0.121 | +0.111 |
| arena apple / orange | 68.8 / 69.7% | 75.0 / 75.8% | **81.2 / 78.8%** |
| arena top-class 오류 | — | 29/~170 | 26/~170 |
| mix val mask mAP50-95 | 0.894 | 0.893 | 0.894 |
| blank 저신뢰 fruit 오탐 | 14/25 | 16/25 | 17/25 |

## 판정

- probe30과 120ep은 recorded 1 crop 차이(노이즈)로 **통계적 동급**. 둘 다 목표(80%)를 15%p 초과.
- 120ep이 arena 합성 probe에서 소폭 우세(apple +6, orange +3), 완전 수렴된 정식 run이므로 **120ep best(epoch 47)를 최종 배포**로 확정.
- 잔여: blank/sliver 저신뢰 fruit 오탐(0.2~0.4대)은 학습으로 완전 제거되지 않음 → Jetson runtime ambiguity guard(fruit conf ≥ 0.4) 필수. Jetson에서 `best.engine` 재빌드 필요.

산출물: `arena_v3_probe_100_v1_top_class_120ep/` (arena 혼동), MODEL_VERSION.md (해시/계보).
