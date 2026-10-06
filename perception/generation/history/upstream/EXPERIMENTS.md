# 실험 색인

기준일: 2026-07-08

이 문서는 실험 상태를 찾기 위한 기준 색인입니다. 원본 report, generated artifact, historical log는 지우지 않고 각 경로에 보존합니다. README 계열 문서는 이 파일을 기준으로 preferred/fallback/rejected 상태를 요약합니다.

용어: 문서에서 **Cube Detector**는 구 `A1`(full-frame object seg), **Face Classifier**는 구 `cube-face unified`(cube crop 면 분류)를 뜻합니다. 코드/경로/alias(`preferred-a1`, `ABC_model/` 등)는 호환성 때문에 옛 이름을 유지합니다 — 매핑은 [readme.md 용어 정리](./readme.md) 참조. 모델 발전 타임라인도 readme.md에 있습니다.

## Status 용어

| Status | 의미 |
| --- | --- |
| `preferred` | 현재 runtime 또는 training에서 기본 추천값입니다. |
| `current` | 유지보수 중인 현재 구성 요소입니다. 반드시 선택된 checkpoint라는 뜻은 아닙니다. |
| `fallback` | debug, 비교, backup 용도로 유용합니다. |
| `candidate` | 가능성은 있지만 승격되지 않았습니다. validation이 더 필요합니다. |
| `rejected` | production/runtime/training에 직접 쓰면 안 됩니다. 근거 보존용입니다. |
| `deprecated` | 더 나은 구조로 대체되었습니다. |
| `archive` | 과거 기록입니다. |

## 현재 Preferred Runtime

| 실험 | status | runtime 사용 여부 | training 사용 여부 | 핵심 지표 | 판단 | 실패/주의 이유 | report path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cube Detector + Face Classifier sparse-icon best | `preferred` | yes | current checkpoint source | mix val all mask mAP50/mAP50-95 `0.967 / 0.894`; box `0.968 / 0.924` (120ep, best epoch 47); **실물 recorded orange 65.1%→95.3%** (목표 80% 초과); arena v3 probe apple/orange 68.8/69.7→81.2/78.8% | 2026-07-08 기본 runtime checkpoint로 Jetson bundle 갱신 (probe30과 동급, 정식 120ep 확정) | blank/sliver 저신뢰 fruit 오탐 17/25 — runtime ambiguity guard(≥0.4) 필수 | [model version](./jetson/ABC_model/cube_face_unified/preferred_v2/MODEL_VERSION.md) |
| Face Classifier readd color-outlier base best | `fallback` | no | sparse-icon run의 init checkpoint | mix val mask mAP50/mAP50-95 `0.970 / 0.894` (best epoch 83); 실물 orange 61.6%→65.1% | 직전 preferred(2026-07-07) — sparse-icon 계보 source로 보존 | 실물 printed-icon 구성 미달이 sparse-icon 사이클의 동기 | [model version](./jetson/ABC_model/cube_face_unified/preferred_v2/MODEL_VERSION.md) |
| A1 + cube-face unified HSV pruned color-outlier best | `fallback` | no | readd run의 init checkpoint | pruned val all mask mAP50/mAP50-95 `0.972 / 0.897`; box `0.972 / 0.925`; orange mask recall `0.890` | 직전 preferred(2026-07-04) — readd line의 source 계보로 보존 | validation은 안정적이나 실물 orange printed-style boundary 미해결이 readd 사이클의 동기였음 | [model version](./jetson/ABC_model/cube_face_unified/preferred_v2/MODEL_VERSION.md) |
| A1 + cube-face unified flat-icon boundary best | `fallback` | no | checkpoint source | expanded val mask mAP50/mAP50-95 `0.974 / 0.901`; box mAP50/mAP50-95 `0.975 / 0.933`; flat icon boundary 개선 실험 | HSV/pruned line의 이전 source 계보로 보존 | 색 outlier pruning 전 계열이며 최신 Jetson bundle은 아님 | [model version history](./jetson/ABC_model/cube_face_unified/preferred_v2/MODEL_VERSION.md) |
| A1 + cube-face unified balanced v2 | `fallback` | no | fine-tune baseline/checkpoint source | v2 mask mAP50-95 `0.87483`; plain recall `0.89644`; apple recall baseline 수준 회복 | flat-icon boundary fine-tune의 계보상 source로 보존 | orange recall은 original baseline보다 낮음 | [balanced v2 summary](./reports/cube_face_unified_eval/balanced_v2_20260629/summary.md) |
| ABC cascade | `fallback` | optional | no | 4-object/about-12-face benchmark에서 약 `20-21 FPS`, `98.26%` face-label match | debugging, geometry comparison, regression check용으로 유지 | unified보다 runtime overhead가 큼 | [optimization log](./jetson/OPTIMIZATION_EXPERIMENTS.md) |

Preferred runtime weights:

```text
preferred-a1:
  jetson/ABC_model/meta_v2_a1_objectseg/a1_yolo26s_seg_meta_v2_50000/weights/best.pt

preferred-unified:
  jetson/ABC_model/cube_face_unified/preferred_v2/weights/best.pt
  runs/segment/cube_face_unified_yolo26n_seg_readd_coloroutlier_base_from_pruned_adamw_lr1e5_ft_v1/weights/best.pt
```

## 모델 실험

| 실험 | status | runtime 사용 여부 | training 사용 여부 | 핵심 지표 | 판단 | 실패/주의 이유 | report path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 초기 8-class object-only YOLO | `deprecated` | no | no | synthetic object detection은 강해졌지만 face-level identity가 없음 | face-aware Meta V2/ABC/unified 방향으로 대체 | visible fruit/plain face를 결정할 수 없음 | [initial summary](./reports/initial_model_may14_summary.md) |
| Meta V2 dataset | `current` | indirect | yes | object mask, face mask, quad, occlusion metadata 저장 | A1/A2/B/C와 unified export의 기반 | 없음 | [technical reference](./readme_specific.md) |
| A1 object segmentation (Cube Detector) | `current` | yes | yes | val=train box mAP50 `0.9941` / mAP50-95 `0.9784`; **arena held-out(안 본 분포) box mAP50 `0.831` / mask mAP50 `0.821`** | 첫 단계로 유지 (현재 배포) | val=train이라 학습셋 지표는 과대. 실제 일반화는 arena held-out으로 측정 | [Jetson guide](./jetson/README.md) |
| A1 low-LR 추가 fine-tune (`a1_..._50000_lowlr_ft_v1`) | `rejected` | no | 실험 | val=train box mAP50 0.9941→0.9943 (+0.0002, 노이즈); **arena held-out box mAP50 0.831→0.806 (−2.5pp), 4개 지표 모두 하락** | 승격 안 함 — 원본 A1 유지 | near-ceiling 모델에 저LR 추가학습(AdamW lr0=1e-5, 30ep)은 val=train을 미세하게 더 암기하되 안 본 분포 일반화는 오히려 저하(미세 과적합). held-out 평가가 없었으면 놓쳤을 결론 | [A1 held-out 비교](./reports/a1_eval_compare_v1/) |
| A2 visible face segmentation | `fallback` | ABC only | 기본값 아님 | visibility debugging에 유용 | ABC cascade용으로 유지 | multi-stage runtime이 느림 | [Jetson guide](./jetson/README.md) |
| B TinyQuadNet | `fallback` | ABC only | 기본값 아님 | face quadrilateral 복원/보정 | geometry 비교용으로 유지 | 추가 CPU/GPU 작업 필요 | [Jetson guide](./jetson/README.md) |
| C MobileNetV3-Small classifier | `fallback` | ABC only | 기본값 아님 | C orange hard case를 anchor pair exp006에서 회복 | ABC 비교용으로 유지 | warp/classifier stage overhead | [C log](./reports/c_orange_hard_case_experiment_log.md) |
| Cube-face unified v1 | `archive` | no | v2 source checkpoint | 초기 validation에서 synthetic mask mAP50-95 약 `0.897` | lineage baseline | v2보다 plain hard-negative 대응이 약함 | [cube-face eval](./reports/cube_face_unified_eval/summary.md) |
| Cube-face unified balanced v2 | `fallback` | no | source checkpoint | baseline/v1/v2 비교에서 aggregate fitness 최고였음 | flat-icon boundary run의 source 계보로 보존 | orange recall 개선 필요 | [balanced v2 summary](./reports/cube_face_unified_eval/balanced_v2_20260629/summary.md) |
| `verified_fruit_balanced_from_last_adamw_lr5e5_ft_v1` | `fallback` | no | source checkpoint | verified fruit balanced val에서 all mask mAP50-95 `0.896` | flat-icon boundary run의 init checkpoint | 실제 orange icon webcam crop 중 하나가 apple로 흔들림 | [technical reference](./readme_specific.md) |
| `verified_plus_flat_icon_boundary_adamw_lr3e5_ft_v1` best | `fallback` | no | source checkpoint | expanded val all mask mAP50/mAP50-95 `0.974 / 0.901`; flat printed icon boundary line | HSV/pruned line의 이전 source 계보로 보존 | 최신 Jetson bundle은 아님 | [technical reference](./readme_specific.md) |
| `hsv_pruned_coloroutlier_from_last_adamw_lr1e5_ft_v1` best | `fallback` | no | readd run의 init checkpoint | pruned val all mask mAP50/mAP50-95 `0.972 / 0.897`; orange mask recall `0.890`; user orange crop 2장 중 1장만 orange | readd line의 source 계보로 보존 (직전 preferred) | color-outlier pruning이 hue 경계를 좁혀 실물 텍스처 대응이 약해진 것이 readd 사이클의 동기 | [원인 리뷰](./reports/cube_face_unified_eval/hard_set_cause_review_20260705.md) |
| `readd_coloroutlier_base_from_pruned_adamw_lr1e5_ft_v1` best | `fallback` | no | sparse-icon run의 init checkpoint | mix val all mask mAP50/mAP50-95 `0.970 / 0.894` (best epoch 83); 실물 녹화 orange 61.6%→65.1% | sparse-icon 계보 source로 보존 (직전 preferred) | 실물 printed-icon 구성 미달 → sparse-icon booster로 해결 | [원인 분석](./reports/cube_face_unified_eval/why_recorded_set_fails_20260708/analysis.md) |
| `sparse_icon_from_readd_adamw_lr1e5_ft_v1` best (120ep) | `preferred` | yes | current checkpoint source | mix val mask mAP50/mAP50-95 `0.967 / 0.894` (best epoch 47, patience stop @82); **실물 recorded orange 95.3%** (readd 65.1%); arena probe apple/orange 81.2/78.8% | 2026-07-08 Jetson `face-classifier`로 갱신 | blank 저신뢰 fruit 오탐 17/25 — runtime guard 필수 | [model version](./jetson/ABC_model/cube_face_unified/preferred_v2/MODEL_VERSION.md) |

## Runtime FPS 실험

| 실험 | status | runtime 사용 여부 | training 사용 여부 | 지표 | 장치 | 모델 형식 | 입력 source | display | object/face 조건 | 판단 | report path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original-ish ABC sequential | `archive` | no | no | 약 `5 FPS` | RTX 5080 dev PC | PyTorch path | 4-object benchmark | off | about 12 faces | 너무 느림 | [optimization log](./jetson/OPTIMIZATION_EXPERIMENTS.md) |
| Batched ABC runtime | `archive` | no | no | 약 `8.7 FPS` | RTX 5080 dev PC | PyTorch path | 4-object benchmark | off | about 12 faces | batching만으로 부족 | [optimization log](./jetson/OPTIMIZATION_EXPERIMENTS.md) |
| ABC `best_stable_5080` | `fallback` | optional | no | 약 `20-21 FPS`, `98.26%` face-label match | RTX 5080 dev PC | optimized mixed path | 4-object/about-12-face benchmark | off | 4 cube-like objects/about 12 visible faces | 최고 안정 ABC fallback | [optimization log](./jetson/OPTIMIZATION_EXPERIMENTS.md) |
| A2 TensorRT/22 FPS ABC attempts | `rejected` | no | no | `22 FPS` 이상 가능했지만 behavior match가 accepted gate 아래로 하락 | RTX 5080 dev PC | TensorRT-heavy ABC variants | 4-object/about-12-face benchmark | off | about 12 faces | behavior 변경 때문에 승격 금지 | [optimization log](./jetson/OPTIMIZATION_EXPERIMENTS.md) |
| A1 TensorRT + unified PyTorch | `candidate` | performance test yes | no | mean `47.45 FPS`, median `52.00 FPS` | RTX 5080 dev PC | A1 `.engine`, face `.pt` | benchmark images | off | mean 4.05 objects / 11.65 faces | unified architecture에 충분한 speed headroom 확인 | [runtime json](./reports/cube_face_unified_eval/unified_runtime_benchmark_a1engine_ptface.json) |
| Webcam fair compare, unified vs ABC | `current` | preview 비교용 yes | no | unified `3.83 FPS`, ABC `1.29 FPS` on captured run reported as `cpu` | dev PC capture run | PyTorch preview path | same 30s webcam capture | preview output generated | sample 8 A1 objects / 5 cube crops / 14 faces | 같은 captured condition에서 unified preview가 더 빠름 | [fair compare summary](./reports/share/fair_30s_compare_20260702_155104/summary.md) |
| Unified fixed-capture stage timing after policy alignment | `current` | preview/regression yes | no | processed `3.83 FPS`; A1 mean `168.7ms`; unified 5-crop batch mean `85.0ms`; per cube `17.0ms` | dev PC CPU capture run | PyTorch `.pt` | same 30s webcam capture | off | 5 cube crops/frame | A1 is the primary CPU bottleneck; runtime crop/resize policy now matches training export | [policy/bottleneck report](./reports/cube_face_unified_eval/unified_runtime_policy_bottleneck_20260703.md) |

## Cube-Face Unified 실험

| 실험 | status | runtime 사용 여부 | training 사용 여부 | 핵심 지표 | 판단 | 실패/주의 이유 | report path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cube_face_unified_yolo26n_seg_v1` | `archive` | no | source baseline | A1 crop -> visible face mask/class 구조 가능성 확인 | v2로 대체 | plain hard negative가 충분히 강하지 않음 | [cube-face eval](./reports/cube_face_unified_eval/summary.md) |
| `wholefruit_plainhard_materialized_ft_v1` | `fallback` | no | 기본값 아님 | plain recall 크게 개선 | plain false positive만 최우선일 때 alternate로 보존 | fruit recall/confidence를 과하게 깎음; small apple confidence collapse | [v1 summary](./reports/cube_face_unified_eval/wholefruit_plainhard_20260629/summary.md) |
| `wholefruit_plainhard_balanced_ft_v2` | `fallback` | no | source checkpoint (계보) | mask mAP50-95 `0.87483`; apple mask recall `0.88531`; plain mask recall `0.89644` | 이후 verified/flat-icon/HSV/pruned/readd 계보로 대체됨 | orange recall은 original baseline보다 낮음 | [v2 summary](./reports/cube_face_unified_eval/balanced_v2_20260629/summary.md) |
| `ai_wholefruit_balanced_ft_v1` | `candidate` | no | 기본값 아님 | aggregate improvement는 작고 confidence tradeoff 존재 | 승격 안 함 | real camera validation과 class-specific hard-case check 필요 | [AI summary](./reports/cube_face_unified_eval/ai_wholefruit_balanced_20260630/summary.md) |
| `ai_printed_fullsquare_ft_v1` | `candidate` | no | future low-ratio booster 가능 | printed/full-square style probe confidence 강함 | candidate only | production runtime으로 검증되지 않음 | [technical reference](./readme_specific.md) |

## Readd Color-Outlier Base 사이클 (2026-07-05 ~ 07-08)

fruit-to-fruit 혼동(특히 orange↔apple)의 원인 분석부터 readd fine-tune, 실물 세트 검증까지의 기록 색인.

| 단계 | status | 내용 | 핵심 결과 | report path |
| --- | --- | --- | --- | --- |
| 원인 리뷰 | `archive` | pruning이 hue 경계를 좁힘 + 224 crop에서 껍질 텍스처 소실 진단 | holdout orange 1번이 pruning 전 orange 0.552 → 후 apple 0.521 역전 | [hard_set_cause_review](./reports/cube_face_unified_eval/hard_set_cause_review_20260705.md), [fruit_confusion_search](./reports/cube_face_unified_eval/fruit_confusion_search_20260705.md) |
| readd 추출/게이트 | `current` | pruning 제거분 중 kind=base 13,465장 hardlink 복구, 게이트 5% 기준 | 직전 preferred의 f2f 오류 6.16% → 게이트 PASS | [gate decision](./reports/cube_face_unified_eval/readd_coloroutlier_base_gate_v1/GATE_DECISION.md) |
| readd fine-tune | `preferred` | pruned 51,139 + readd 13,465 mix, AdamW lr0=1e-5, 120ep(patience 35, stop@118, best 83) | mix val mask mAP50/mAP50-95 `0.970 / 0.894` | [model version](./jetson/ABC_model/cube_face_unified/preferred_v2/MODEL_VERSION.md) |
| 잔여 오류 유형 분석 | `archive` | 게이트 val에서 안 고쳐지는 114장 분석 (baseline 오류의 88%) | 잔여 오류 = 그림자/blur/sliver/유사 외형 tail; true-conf 중앙값 0.000 | [persistent 분석](./reports/cube_face_unified_eval/persistent_misclassification_analysis_20260707/analysis.md) |
| 실물 녹화 세트 비교 | `current` | 실물 orange cube 녹화 86 visible-icon 크롭, old vs readd best | orange 인식 61.6% → 65.1% (+4/−1); blank 저신뢰 banana 오탐 6→14 | [recorded 비교](./reports/cube_face_unified_eval/recorded_hardset_old_vs_readd_final_20260708/summary.md) |
| 실패 원인 확정 프로브 | `current` | 아이콘 면적 스윕(합성) + 실패 크롭 확대 재예측 + 학습 분포 비교 | 인식 절벽이 면적 11~20%에 위치, recorded 세트는 8~10%; 확대 시 25/30 정답 전환 → composition gap 확정 | [why 분석](./reports/cube_face_unified_eval/why_recorded_set_fails_20260708/analysis.md) |

## Fruit Texture 실험

| 실험 | status | runtime 사용 여부 | training 사용 여부 | 핵심 지표 | 판단 | 실패/주의 이유 | report path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| production texture policy | `current` | indirect | yes | lab 65%, original filtered 25%, fruitseg30 10% policy | production base 안정 유지 | 없음 | [texture policy](./reports/texture_policy_meta_v2_50000_20260623.md) |
| `ai_wholefruit_balanced` probe | `candidate` | no | low-ratio booster 가능 | top1 `0.950`, strong correct `0.428` | low-confidence signal로 유용하지만 direct production 아님 | banana confidence가 낮은 case 다수 | [candidate probe](./reports/cube_face_unified_eval/fruit_texture_candidate_probe_20260702/report.md) |
| `ai_printed_fullsquare` probe | `candidate` | no | low-ratio booster 가능 | top1 `0.978`, strong correct `0.937` | 가장 깨끗한 booster 후보 | runtime/real-camera validation 필요 | [candidate probe](./reports/cube_face_unified_eval/fruit_texture_candidate_probe_20260702/report.md) |
| `combined_curated` probe | `candidate` | no | cleanup 이후만 가능 | top1 `0.840`, strong correct `0.586` | diversity는 있으나 cleanup 필요 | domain/noise gap | [candidate probe](./reports/cube_face_unified_eval/fruit_texture_candidate_probe_20260702/report.md) |
| web cutout sets | `rejected` | no | no | web_v1/web_v3의 wrong/noisy rate 높음; non-apple -> apple confusion도 주로 여기 집중 | `datasets/fruit_textures/other/web_*` 아래 격리 | slices, packaging, bad masks, wrong objects, noisy backgrounds | [web probe](./reports/cube_face_unified_eval/fruit_texture_web_cutout_probe_20260702/report.md) |

## C Classifier 실험

| 실험 | status | runtime 사용 여부 | training 사용 여부 | 핵심 지표 | 판단 | 실패/주의 이유 | report path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C orange hard-case anchor pair exp006 | `fallback` | ABC only | 기본값 아님 | orange hard case `0.999999`, apple hard case `0.9972`, original val acc `0.9854` | 최고 C fallback checkpoint | preferred unified runtime에는 포함되지 않음 | [C log](./reports/c_orange_hard_case_experiment_log.md) |

## Data Policy 실험

| 실험 | status | runtime 사용 여부 | training 사용 여부 | 핵심 지표 | 판단 | 실패/주의 이유 | report path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO cache cleanup | `current` | indirect | yes | 큰 `.npy` cache 압박 제거; booster fine-tune은 `cache=False` 유지 | generated cache로 disk를 채우지 않음 | 없음 | [technical reference](./readme_specific.md) |
| hardlink materialization | `current` | no | yes | Windows + Ultralytics duplicate list stall 회피 | 큰 mix는 materialized folder dataset 사용 | 없음 | [technical reference](./readme_specific.md) |
| booster trigger gate | `current` | no | yes | quantity/diversity/accuracy gate 통과 시에만 training | evidence가 강할 때만 booster 학습 | 없음 | [technical reference](./readme_specific.md) |
| unified runtime crop policy alignment | `current` | yes | no | A1 crop uses `max(width,height) * 0.18`, then `224x224` square resize before unified face model | runtime/test now follows cube-face unified export input policy | old raw rectangular crop path could trigger a training/runtime preprocessing mismatch | [policy/bottleneck report](./reports/cube_face_unified_eval/unified_runtime_policy_bottleneck_20260703.md) |
| **evaluation-only holdout 정책** | `current` | 평가 전용 | **학습 사용 절대 금지** | 사용자 제공 orange webcam crop 2장 + 실물 orange cube 녹화 세트(`reports/unified_recorded_input_frame_sort_20260706_165440`) 전체 | 이 세트들은 원본/크롭/증강/합성 재료 어떤 형태로도 training에 넣지 않는다. booster는 학습용 텍스처/Blender 생성만 사용 | holdout 오염 시 실물 성능 판정 기준 자체가 사라짐 | [why 분석](./reports/cube_face_unified_eval/why_recorded_set_fails_20260708/analysis.md) |

## Rejected / Deprecated 실험

| 실험 | status | 보존 이유 | 사용 금지 | 대체안 |
| --- | --- | --- | --- | --- |
| Initial object-only runtime | `deprecated` | historical baseline | face-aware cube decision | A1 + unified |
| Direct web cutout training | `rejected` | noisy web data가 무엇을 망가뜨리는지 보여줌 | production training | cleaned/validated booster sets |
| A2 TensorRT 22 FPS ABC variants below behavior gate | `rejected` | speed/accuracy tradeoff 기록 | final ABC runtime | `best_stable_5080` fallback 또는 unified |
| v1 plain-heavy as default | default로는 `rejected`, special case로는 `fallback` | plain false positive만 문제일 때 유용 | default runtime | balanced v2 |
| `latest-unified` alias as production | `candidate` only | experiment convenience | production default | `preferred-unified` |

## 다음 Candidate

| 후보 | status | 기대 효과 | 승격 전 필요한 validation |
| --- | --- | --- | --- |
| **printed-label composition booster (`realistic_a4_sparse_icon`)** | `preferred` (완료·배포) | patch 0.24~0.40 + posterize + `realistic_webcam_hard` 10k → readd에서 120ep fine-tune. **실물 recorded orange 65.1%→95.3%** (목표 80% 초과), arena 68.8/69.7→81.2/78.8%, mix val 회귀 없음 ([레시피/반복](./reports/cube_face_unified_eval/sparse_icon_probe_iterations_20260708/summary.md)). holdout 미사용 | 배포 완료. 잔여: blank 저신뢰 fruit 오탐(runtime guard로 방어), Jetson engine 재빌드 |
| Track-memory plain gate | `candidate` | hidden-fruit -> plain_cube pickup risk 감소 | moving camera sequence replay |
| banana hard-case booster | `candidate` | 낮은 banana confidence 개선 | class-balanced validation과 real camera probe |
| pineapple-vs-banana targeted booster | `candidate` | pineapple -> banana drift 감소 | real/curated probe와 confusion matrix |
| explicit unknown/reject policy for unified | `candidate` | background/plain false positive 감소 | background crop set과 threshold sweep |
| mask-based cube fitting from unified masks | `candidate` | ABC B/C cost 없이 face geometry 개선 | ABC quads, pose refine와 비교 |
