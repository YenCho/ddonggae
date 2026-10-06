# A1 (Cube Detector) 저LR 추가 fine-tune — held-out 평가 (2026-07-09)

## 배경

사용자 요청으로 현재 A1 detector(`a1_yolo26s_seg_meta_v2_50000` best)를 저LR로 추가 fine-tune. A1 데이터셋은 `val=train`이라 학습셋 지표(box mAP50 0.994)는 fit만 반영하고 일반화를 못 본다. 그래서 mimic arena로 **held-out 평가셋**을 만들어(neither model trained on it) 두 모델을 공정 비교.

- 학습: init=현재 A1 best, AdamW lr0=1e-5 lrf=0.05, imgsz 640, batch 32, epochs 30 (완주), 원본과 동일 aug
- held-out: `datasets/arena_v3_a1_eval_v1` (60 arena scene, 573 물체 = cube_like_object 320 / octahedron 88 / dodecahedron 75 / icosahedron 90), YOLO-seg 라벨

## 결과

| 지표 | val=train (fit) | | arena held-out (일반화) | |
| --- | --- | --- | --- | --- |
| | current | lowlr | **current** | lowlr |
| box mAP50 | 0.9941 | 0.9943 | **0.831** | 0.806 |
| box mAP50-95 | 0.9784 | 0.9792 | **0.729** | 0.714 |
| mask mAP50 | 0.9942 | 0.9943 | **0.821** | 0.795 |
| mask mAP50-95 | 0.9537 | 0.9542 | **0.558** | 0.540 |

- val=train: 저LR이 +0.0002~0.0008 (노이즈, 학습셋 미세 암기)
- **held-out: 저LR이 4개 지표 모두 1.5~2.5pp 하락** (box mAP50 −2.5pp)

## 결론

near-ceiling 모델에 대한 저LR 추가 학습은 학습셋 fit을 아주 조금 올리는 대신 **안 본 분포로의 일반화를 미세하게 악화**시켰다(전형적 과적합). → **원본 A1 유지, 저LR 모델은 `rejected`.**

방법론적 성과: `val=train` 세팅에서는 이 결론을 절대 볼 수 없었다. mimic arena held-out 평가(`scripts/build_arena_a1_eval.py` + `scripts/eval_a1_models_compare.py`)가 A1의 실제 일반화를 재는 재사용 가능한 도구로 확보됨.

원자료: `a1_eval_comparison.json`
