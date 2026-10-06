# Data Generation Blender

## 이 문서의 역할

이 파일은 프로젝트의 상세 실험 기록입니다. 처음 실행하려면 [readme.md](./readme.md)를 먼저 보고, 어떤 실험이 어디에 있는지 찾으려면 [EXPERIMENTS.md](./EXPERIMENTS.md)를 보세요.

현재 preferred runtime 구성은 `A1 object segmentation + cube-face unified v2`입니다.

```text
jetson/ABC_model/cube_face_unified/preferred_v2/weights/best.pt
runs/segment/cube_face_unified_yolo26n_seg_wholefruit_plainhard_balanced_ft_v2/weights/best.pt
```

이 문서는 기존 실험 기록을 삭제하지 않고 보존합니다. 최신 요약, 실행법, 보고서 색인은 루트 README와 EXPERIMENTS를 기준으로 정리했습니다.

## 2026-06-29 최신 상태: whole-fruit/plain-hard booster로 unified cube-face 모델 보강

이 섹션은 2026-06-28의 A2+B+C 통합 cube-face 모델 실험 위에 추가한 후속 보강입니다. 기존 Meta V2, visibility 정책, A1/A2/B/C fallback 구조, 그리고 50k unified baseline은 삭제하지 않습니다.

### 문제 정의

통합 cube-face 모델은 A1 crop 안에서 보이는 face를 한 번에 segment/classify하는 구조입니다. 이 구조에서 실제 runtime 약점으로 보이는 부분은 다음입니다.

```text
1. 대회 과일 이미지는 orange slice나 잘린 과일 단면이 아니라 whole fruit 사진에 가깝다.
2. 실제 C/face 입력에는 완벽히 정면인 face만 들어오지 않는다.
3. 흰 cube 면, 조명 받은 cube 면, 그림자 진 cube 면, cube edge/corner, 테이블/벽 일부가 face처럼 들어올 수 있다.
4. 특히 나무 바닥/따뜻한 조명 때문에 plain face가 연노랑/연주황처럼 보일 수 있고, 이게 apple/orange로 튀면 위험하다.
```

따라서 새 Blender 렌더링이나 실제 촬영 데이터 없이, 기존 texture 자산과 2D 합성으로 통합 face model을 보강하는 booster를 만들었습니다.

### 추가된 스크립트

```text
scripts/make_cube_face_unified_booster.py
scripts/run_cube_face_unified_booster_finetune.ps1
```

`make_cube_face_unified_booster.py`는 224x224 cube crop 형태의 YOLO segmentation dataset을 생성합니다. 출력 class는 기존 unified model과 동일합니다.

```text
0 apple
1 orange
2 banana
3 pineapple
4 plain
```

### 생성 정책

```text
whole-fruit face reinforcement:
  apple/orange/banana/pineapple을 whole fruit texture 중심으로 다시 노출
  orange slice, cut fruit, half fruit 같은 파일명은 제외
  apple/orange 비율을 균형 있게 유지

plain hard negative:
  흰 cube face
  따뜻한 연노랑/연주황 plain face
  강한 조명/그림자/blur/JPEG/noise가 들어간 plain face
  cube edge/corner처럼 실제 runtime crop에서 들어올 수 있는 plain face

background-empty negative:
  face처럼 보이는 배경 일부가 들어왔을 때 fruit로 튀지 않도록 소량의 empty-label crop 포함
```

기본 texture source는 더 보수적으로 다음 하나만 사용합니다.

```text
datasets/fruit_textures/final_fruits36070_original30
```

파일명에 아래 단어가 들어간 이미지는 texture 후보에서 제외합니다.

```text
slice, sliced, cut, half, halves, wedge, segment, section, piece, pieces
```

이 필터는 완벽한 semantic filter는 아니지만, "대회 이미지는 fruit slice가 아니다"라는 현재 가정을 코드 수준에서 직접 반영합니다.

### 생성 결과

```text
datasets/cube_face_unified_booster_wholefruit_plainhard_v1/
  images/train
  images/val
  labels/train
  labels/val
  previews/booster_contact_sheet.jpg
  data.yaml
  manifest.json

datasets/cube_face_unified_finetune_wholefruit_plainhard_v1/
  train.txt
  val.txt
  data.yaml
  manifest.json

datasets/cube_face_unified_finetune_wholefruit_plainhard_materialized_v1/
  images/train
  images/val
  labels/train
  labels/val
  data.yaml
  manifest.json
```

| item | count |
| --- | ---: |
| total booster images | 10,000 |
| train images | 8,946 |
| val images | 1,054 |
| apple instances | 1,302 |
| orange instances | 1,341 |
| banana instances | 1,042 |
| pineapple instances | 954 |
| plain instances | 17,340 |
| mixed train list | 39,892 |
| mixed val list | 3,554 |

mixed fine-tune list는 base unified train 22,000장에 booster train 8,946장을 2회 넣어 booster를 의도적으로 oversampling합니다.

2026-06-29 실행 중 확인한 문제:

```text
Windows + Ultralytics label scan에서 큰 train.txt와 duplicate path oversampling 조합이 멈추는 현상이 있었다.
그래서 실제 학습은 train.txt list 대신 hardlink로 materialize한 일반 YOLO folder dataset을 기본값으로 사용한다.
이미지/라벨 내용은 동일하고, 저장공간은 hardlink라 크게 늘지 않는다.
```

materialized mix 생성 스크립트:

```text
scripts/materialize_cube_face_unified_finetune_mix.py
```

### fine-tune 실행

```powershell
cd C:\Users\user\Documents\Data_Generation_Blender
powershell -ExecutionPolicy Bypass -File scripts\run_cube_face_unified_booster_finetune.ps1 -SkipGenerate -Epochs 40 -Batch 384 -Workers 0 -EmailIntervalSeconds 1800
```

학습 설정:

```text
model: runs/segment/cube_face_unified_yolo26n_seg_v1/weights/best.pt
data: datasets/cube_face_unified_finetune_wholefruit_plainhard_materialized_v1/data.yaml
epochs: 40
imgsz: 224
batch: 384
workers: 0
cache: False
amp: True
mosaic/mixup/copy_paste: 0
email: 별도 monitor가 30분마다 results.csv를 읽어 roboticsai891@gmail.com으로 발송
```

`cache=False`는 의도입니다. 이전에 `.npy`/`.cache`가 85GB 가까이 쌓였으므로 이번 보강 fine-tune에서는 cache를 다시 만들지 않습니다.

학습 실행 방식은 direct `yolo segment train`입니다. `train_yolo_with_email.py`로 YOLO stdout을 한 번 더 감싸면 carriage-return progress 출력이 Windows pipe에서 멈추는 현상이 재현되어, 학습 프로세스와 메일 모니터를 분리했습니다. 단, YOLO가 생성하는 `labels/*.cache` metadata 파일은 이미지 `.npy` 캐시가 아니며 label scan 단축용으로 유지해도 됩니다.

2026-06-29 13:28 KST v1 실행 확인:

```text
run dir:
  runs/segment/cube_face_unified_yolo26n_seg_wholefruit_plainhard_materialized_ft_v1

process:
  direct yolo segment train 실행 중
  send_yolo_50000_status_email_loop.ps1 monitor 실행 중

scanner:
  train 39,892 images / 1,294 backgrounds / 0 corrupt 통과
  val 3,554 images / 88 backgrounds / 0 corrupt 통과

epoch 1 results.csv:
  train/box_loss: 0.56284
  train/seg_loss: 0.79874
  train/cls_loss: 0.38738
  Box precision: 0.91361
  Box recall: 0.90578
  Box mAP50: 0.96571
  Box mAP50-95: 0.90014
  Mask precision: 0.91382
  Mask recall: 0.90362
  Mask mAP50: 0.96442
  Mask mAP50-95: 0.87748
```

2026-06-29 14:30 KST v1 평가와 v2 분기:

```text
v1 result:
  run: runs/segment/cube_face_unified_yolo26n_seg_wholefruit_plainhard_materialized_ft_v1
  status: EarlyStopping, 13/40 epochs completed
  best epoch: 1

overall:
  baseline Mask mAP50-95: 0.85846
  v1 Mask mAP50-95: 0.87723
  delta: +0.01877

plain class:
  baseline plain Mask recall: 0.793
  v1 plain Mask recall: 0.935
  baseline plain Mask mAP50: 0.871
  v1 plain Mask mAP50: 0.976

tradeoff:
  apple Mask recall: 0.910 -> 0.883
  orange Mask recall: 0.904 -> 0.869
  small apple hard case confidence: 0.883 -> 0.112
```

판단:

```text
v1은 plain hard negative 억제에는 성공했지만 fruit recall/confidence를 과하게 깎았다.
대회 과일 이미지는 slice가 아니라 whole fruit라는 현재 가정은 유지한다.
다만 booster oversampling이 plain 쪽으로 너무 강하게 걸렸다고 보고, balanced v2를 바로 시작했다.
```

v2 설정:

```text
dataset:
  datasets/cube_face_unified_finetune_wholefruit_plainhard_materialized_v2_balanced

mix:
  base train sample: 30,000
  booster repeat: 1
  train images: 38,946
  val images: 3,554

run:
  runs/segment/cube_face_unified_yolo26n_seg_wholefruit_plainhard_balanced_ft_v2

goal:
  v1의 plain 개선을 최대한 유지하면서 apple/orange recall과 confidence를 회복한다.
```

보고서:

```text
reports/cube_face_unified_eval/wholefruit_plainhard_20260629/summary.md
reports/cube_face_unified_eval/wholefruit_plainhard_20260629/val_compare_summary.json
reports/cube_face_unified_eval/wholefruit_plainhard_20260629/hard_cases_baseline/report.json
reports/cube_face_unified_eval/wholefruit_plainhard_20260629/hard_cases_fine_tuned/report.json
```

2026-06-29 15:35 KST v2 완료 평가:

```text
v2 result:
  run: runs/segment/cube_face_unified_yolo26n_seg_wholefruit_plainhard_balanced_ft_v2
  status: EarlyStopping, 13/30 epochs completed
  best epoch: 1

same v2 materialized validation set:
  baseline Mask mAP50-95: 0.85499
  v1 plain-heavy Mask mAP50-95: 0.87114
  v2 balanced Mask mAP50-95: 0.87483

plain class:
  baseline plain Mask recall: 0.78489
  v1 plain-heavy plain Mask recall: 0.93464
  v2 balanced plain Mask recall: 0.89644
  baseline plain Mask mAP50: 0.86913
  v1 plain-heavy plain Mask mAP50: 0.97118
  v2 balanced plain Mask mAP50: 0.96560

fruit recovery:
  apple Mask recall: baseline 0.88449, v1 0.86845, v2 0.88531
  orange Mask recall: baseline 0.90065, v1 0.87320, v2 0.88338
  small whole-apple hard case confidence: baseline 0.883, v1 0.112, v2 0.872
```

판단:

```text
v1은 plain false positive 억제만 보면 가장 강하지만 fruit confidence를 너무 깎는다.
v2는 plain 개선을 대부분 유지하면서 apple recall/confidence를 baseline 수준으로 회복했다.
orange recall은 baseline보다 낮지만 v1보다는 개선되었고, 전체 Mask mAP50-95와 fitness는 셋 중 가장 좋다.
따라서 runtime 테스트의 기본 cube-face unified checkpoint는 v2 balanced로 지정한다.
```

preferred runtime checkpoint:

```text
runs/segment/cube_face_unified_yolo26n_seg_wholefruit_plainhard_balanced_ft_v2/weights/best.pt
```

평가 보고서:

```text
reports/cube_face_unified_eval/balanced_v2_20260629/summary.md
reports/cube_face_unified_eval/balanced_v2_20260629/val_compare_summary.json
reports/cube_face_unified_eval/balanced_v2_20260629/hard_cases_baseline/report.json
reports/cube_face_unified_eval/balanced_v2_20260629/hard_cases_v1_plainheavy/report.json
reports/cube_face_unified_eval/balanced_v2_20260629/hard_cases_v2_balanced/report.json
```

### 기대 효과와 한계

기대 효과:

```text
orange slice에 맞춰진 잘못된 bias를 줄이고 whole-orange/whole-apple 경계로 다시 이동
plain face가 연노랑/연주황 조명일 때 fruit로 튀는 문제 완화
runtime에서 crop에 cube edge/corner나 배경 일부가 들어와도 plain/background 쪽으로 억제
```

한계:

```text
2D booster는 실제 3D occlusion과 카메라 왜곡을 완전히 대체하지 않는다.
texture 파일명 필터만으로 모든 non-whole-fruit 이미지를 완벽히 제거할 수는 없다.
plain hard negative를 너무 많이 넣으면 fruit recall이 줄 수 있으므로 fine-tune 후 validation과 runtime preview 확인이 필요하다.
이번 booster는 unified cube-face model용이며, A1 object segmentation 자체를 개선하지 않는다.
```

다음 확인 항목:

```text
1. fine-tune 완료 후 기존 validation crop과 booster validation crop에서 class별 mAP 확인
2. 사용자가 보낸 apple/orange 실패 crop에 대한 before/after 추론 비교
3. 실제 webcam runtime에서 raw per-frame 출력 확인
4. plain hard negative에서 apple/orange false positive가 줄었는지 측정
5. 개선되면 jetson portable export에 새 unified model weight 추가
```

## 2026-06-28 최신 상태: A2+B+C 통합 cube-face 모델 실험 준비

이 섹션은 2026-06-23까지의 Meta V2, A1/A2/B/C, C hard-case, texture, visibility 정책을 삭제하거나 대체하는 것이 아닙니다. 기존 실험과 판단은 아래 섹션들에 그대로 보존하고, 여기에는 2026-06-28에 추가된 새 런타임 최적화 분기와 학습 데이터 추출 결과를 기록합니다.

### 새 분기를 추가한 이유

기존 A1/A2/B/C 구조는 역할 분리가 명확하고 디버깅이 쉽지만, 런타임에서는 다음 단계들이 연속으로 실행됩니다.

```text
A1 YOLO segmentation
  -> object crop
  -> A2 YOLO face segmentation
  -> B TinyQuadNet quad recovery
  -> perspective warp
  -> C MobileNetV3-Small classifier
  -> cube fitting / temporal vote
```

최근 5080 PC 기준 최적화 실험에서 안정적인 동작은 약 21 FPS 수준까지 올라왔지만, 4개 cube-like object와 약 12개 face가 보이는 benchmark에서 22 FPS 이상의 안정 목표를 만족하려면 A2/B/C를 합치는 구조가 더 근본적인 개선 후보로 보입니다.

따라서 새 실험 분기는 다음처럼 둡니다.

```text
A1:
  full camera frame에서 cube_like_object / octahedron / dodecahedron / icosahedron 검출

Unified cube-face model:
  A1의 cube_like_object crop 하나를 입력으로 받음
  보이는 face들을 segmentation instance로 찾음
  각 face를 apple / orange / banana / pineapple / plain으로 분류
  quad는 sidecar supervision 또는 후처리 cube fitting으로 보정
```

즉, 기존의 A2+B+C를 완전히 버리는 것이 아니라, 기존 Meta V2 정답을 재조합해서 "A2+B+C 통합 후보"를 새로 학습해보는 실험입니다. 실패해도 기존 A1/A2/B/C 구조와 실험 기록은 그대로 남습니다.

### 기존 데이터만으로 가능한지 평가

2026-06-28 기준으로 기존 `datasets/meta_v2_50000_coco_texture_v1`만 사용해서 새 unified dataset을 만들 수 있는지 확인했습니다.

결론:

```text
1차 실험은 새 Blender 렌더링 없이 가능하다.
```

근거:

```text
_meta/train/*.json 50,000개 존재
각 object에 visible/full mask, visible bbox, visible segments 존재
각 cube face에 visible_mask, visible_segments, quad_xy, quad_xy_raw, corner_visibility 존재
face_kind, fruit_class, quality, occluder_object_ids 존재
scene_snapshot에는 camera, object 3D transform, mesh geometry도 저장됨
```

특히 occlusion이 있는 경우에도 다음 정보가 분리되어 있습니다.

```text
visible_segments:
  실제 이미지에서 보이는 face 영역만 표현

quad_xy:
  occlusion이 없었다면 face가 가져야 하는 4점 quad

corner_visibility:
  각 꼭짓점이 실제로 보이는지 여부

quality:
  good / partial / bad 기준
```

따라서 새 unified model은 YOLO segmentation label로 visible face mask/class를 학습하고, `quads/*.json` sidecar를 통해 추후 keypoint head, quad regression, cube fitting 보정 실험을 이어갈 수 있습니다.

### 새 dataset export 결과

스크립트:

```text
scripts/export_meta_v2_cube_face_unified_dataset.py
```

출력:

```text
datasets/meta_v2_50000_cube_face_unified_v1/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt
  quads/train/*.json
  quads/val/*.json
  data.yaml
  manifest.json
```

YOLO segmentation class:

```text
0 apple
1 orange
2 banana
3 pineapple
4 plain
```

수량:

| item | count |
| --- | ---: |
| source meta files | 50,000 |
| train cube crops | 84,731 |
| val cube crops | 9,479 |
| total cube crops | 94,210 |
| apple face instances | 29,267 |
| orange face instances | 26,345 |
| banana face instances | 30,353 |
| pineapple face instances | 29,622 |
| plain face instances | 120,820 |
| empty-label hard-negative crops | 149 |
| skipped bad-quality faces | 27,323 |
| skipped tiny objects | 292 |
| skipped empty face segments | 7 |

preview:

```text
reports/cube_face_unified_v1_preview/contact_sheet.jpg
```

preview 기준으로 face polygon은 cube crop 안의 실제 visible face 위에 정상적으로 얹힙니다.

### cache 정리

학습 전에 저장공간 확보를 위해 YOLO disk cache와 numpy image cache를 삭제했습니다.

삭제 범위:

```text
datasets/**/*.npy
datasets/**/*.cache
runs/**/*.npy
runs/**/*.cache
```

결과:

| item | count |
| --- | ---: |
| deleted cache files | 173,508 |
| freed space | about 84.94 GB |
| remaining cache files under datasets/runs | 0 |

원본 `.jpg`, `.txt`, `.json`, mask PNG, model weight는 삭제하지 않았습니다.

### 학습 명령

현재 1차 baseline은 YOLO26n-seg를 사용합니다. 이 모델은 keypoint를 직접 내지 않으므로, 첫 실험의 목표는 "cube crop에서 face class segmentation을 한 번에 뽑을 수 있는지"입니다. quad/keypoint는 sidecar JSON을 남겨둔 상태에서 후속 실험으로 붙입니다.

```powershell
cd C:\Users\user\Documents\Data_Generation_Blender
yolo segment train data=C:\Users\user\Documents\Data_Generation_Blender\datasets\meta_v2_50000_cube_face_unified_v1\data.yaml model=yolo26n-seg.pt epochs=120 imgsz=224 batch=256 device=0 workers=2 cache=False amp=True name=cube_face_unified_yolo26n_seg_v1
```

주의:

```text
cache=False를 유지한다. 방금 85GB가량 cache를 지웠으므로 다시 disk cache를 만들지 않는다.
imgsz는 224로 둔다. 208처럼 stride 32 배수가 아닌 값은 내부에서 자동 보정될 수 있으므로 실험 해석을 흐린다.
Windows에서 workers=8은 batch 256과 함께 PyTorch shared file mapping 1455 오류를 만들 수 있어, 실제 시작값은 workers=2를 사용한다.
첫 실험은 A2+B+C를 모두 대체한다는 결론이 아니라, 대체 가능한지 확인하는 baseline이다.
```

### 예상 장단점

장점:

```text
B quad model, C classifier, C warp를 대부분 제거할 수 있다.
face별 작은 crop 여러 번 추론하는 비용이 줄어든다.
cube crop 하나에서 여러 face를 동시에 처리한다.
plain/fruit 판단이 segmentation 단계에서 같이 일어나므로 plain false positive를 더 직접적으로 제어할 수 있다.
```

단점과 위험:

```text
YOLO segmentation head만으로는 4 corner keypoint를 직접 보장하지 않는다.
face mask가 잘 나와도 quad 안정성은 cube fitting 후처리에 의존한다.
plain face instance가 과일 face보다 많아 class imbalance가 있다.
unknown class는 이번 segmentation dataset에 직접 넣지 않았다.
runtime에서 A1 crop이 흔들리면 unified model 입력 분포도 같이 흔들린다.
```

이번 dataset에서 `unknown`을 뺀 이유:

```text
unknown은 자연스러운 cube face class가 아니라 잘못 잡힌 crop, 배경, 애매한 crop을 억제하기 위한 C classifier hard negative에 가깝다.
segmentation 모델에 unknown face instance를 직접 넣으면 "무엇을 segment해야 하는지"가 불명확해질 수 있다.
대신 empty-label hard-negative crop과 plain face 강화를 먼저 사용한다.
```

### 다음 TODO

1. `cube_face_unified_yolo26n_seg_v1` 학습
2. val preview에서 apple/orange/plain 혼동 확인
3. 기존 A2+B+C runtime 결과와 unified 모델 결과를 같은 cube crop 기준으로 비교
4. face-label match, plain false positive, missed face, occluded face 성능 측정
5. 성능이 충분하면 `jetson` runtime에 A1 + unified face model 경로 추가
6. quad가 불안정하면 `quads/*.json`으로 keypoint/quad head 또는 lightweight quad refinement 추가
7. 성공하면 기존 A2/B/C는 삭제하지 않고 fallback runtime path로 남김

## 2026-06-23 최신 상태: 50k Meta V2 학습 준비

이 파일은 `readme.md`의 상세판입니다. 짧은 README는 전체 흐름과 현재 실행 경로를 보기 위한 문서이고, 이 파일은 이전에 고려했던 전략 분기, 폐기한 방향, 실험 결과, 보류한 TODO까지 최대한 보존하는 기록용 문서입니다.

이 파일은 GitHub에 push됐던 `readme.md` 버전들을 기준으로 다시 모은 최종 상세판입니다. 특히 아래 항목은 삭제된 것이 아니라 이 문서 안에 보존되어야 하는 핵심 정책입니다.

```text
visibility 지표
fruit label fallback
ideal_visibility gate
distortion / frame margin policy
metadata 설계
texture source ratio
arena booster 정책
Task 1 / Task 2 분리 논의
Option 3 visibility-aware pipeline
Meta V2 전환 이유
```

통합 기준으로 본 주요 GitHub README 버전:

| commit | 날짜 | README에서 가져온 핵심 |
| --- | --- | --- |
| `228c8dd` | 2026-05-15 | 초기 BlenderProc/YOLO 데이터 생성 방향 |
| `088436b` | 2026-05-15 | 생성/학습/메일 pipeline |
| `f4924e4` | 2026-05-15 | 50k 생성 목표 |
| `2666d29` | 2026-05-16 | 배경, 조명, 카메라, negative, 라벨 품질 설계 |
| `553605d` | 2026-05-17 | segmentation 품질 제어, visibility terms |
| `78e16ba` | 2026-05-17 | `ideal_visibility` gate |
| `8b0b5f6` | 2026-05-17 | `Visibility 지표`, `Fruit 라벨 정책`, `Frame 및 distortion 정책` |
| `b8af34c` | 2026-05-19 | fruit texture augmentation |
| `2d710ec` | 2026-05-19 | fruit texture consistency per cube |
| `b9086c4` | 2026-05-19 | COCO2017 background 기본화 |
| `4b187f7` | 2026-05-20 | collage/mixed fruit texture layout |
| `d5a8b4f` | 2026-05-20 | 재현 가능한 YOLO26 fruit pipeline |
| `625e0fd` | 2026-05-30 | SUN-111/SUN-168 arena booster |
| `0a34cdb` | 2026-06-22 | visibility-aware Option 3 방향 |
| `9f65b7e` | 2026-06-22 | face dataset exporter 방향 |
| `d6679c1` | 2026-06-22 | bbox crop 기반 C exporter와 상세 README 본문 |
| `879a686` | 2026-06-23 | Meta V2, A1/A2/B/C, C 최종화 |
| `5d2b44b` | 2026-06-23 | 50k 학습 계획과 texture policy |

즉, 이 문서는 특정 시점의 실험 하나만 기록하는 문서가 아니라, 위 README 흐름을 한 파일로 합친 detailed reference입니다.

현재 50k 학습은 다음 방향으로 준비했습니다.

```text
기존 100k 8-class dataset 재사용 방향은 폐기.
Meta V2 metadata를 새로 생성하는 방향으로 고정.
배경은 우선 COCO 고정.
SUN-111/SUN-168 arena booster는 Meta V2 안정화 후 다시 추가.
A1/A2/B/C를 모두 새 데이터에서 다시 학습.
```

50k 실행 스크립트:

```powershell
conda activate ai_robotics
cd C:\Users\user\Documents\Data_Generation_Blender

powershell -ExecutionPolicy Bypass -File scripts\run_meta_v2_50000_training_pipeline.ps1
```

50k 기본 출력:

```text
source dataset: datasets/meta_v2_50000_coco_texture_v1
model datasets: datasets/meta_v2_50000_coco_texture_v1_models
texture pack:   datasets/fruit_textures/production_meta_v2_50000_v1
log dir:        logs/meta_v2_50000_pipeline
```

50k 학습 순서:

```text
1. Meta V2 COCO image 50,000장 생성
2. A1/A2/B/C dataset export
3. C unknown hard negative 추가
4. A1 YOLO26s-seg 학습
5. A2 YOLO26n-seg 학습
6. B TinyQuadNet 학습
7. C MobileNetV3-Small 학습
```

## 2026-06-23 texture 정리 정책

과일 texture는 C 모델 hard-case 실험 결과를 반영해서 다음처럼 정리했습니다.

```text
production texture:
  datasets/fruit_textures/production_meta_v2_50000_v1

base texture:
  datasets/fruit_textures/final_fruits36065_original25_fruitseg30_10

archived experiment textures:
  datasets/fruit_textures/_archive_experimental_20260623_c_hardcase
```

기존 source ratio는 유지했습니다.

| class | count |
| --- | ---: |
| apple | 1600 |
| orange | 1600 |
| banana | 1600 |
| pineapple | 1600 |

각 class 내부 source 비율도 유지했습니다.

| source type | count per class | ratio |
| --- | ---: | ---: |
| lab | 1040 | 65% |
| original_filtered | 400 | 25% |
| fruitseg30 | 160 | 10% |

웹에서 받은 orange hard texture는 production texture에 직접 섞지 않았습니다.

```text
1. orange_hard_web_exp_001에는 실제 orange가 아닌 이미지가 섞였다.
2. broad 2D mimic이나 orange-only 보강은 C의 apple/orange balance를 무너뜨릴 수 있다.
3. 이번 50k는 texture source 비율을 바꾸지 않고, render-time augmentation으로 다양성을 늘리는 편이 안전하다.
```

따라서 50k generation은 다음 설정을 사용합니다.

```text
--fruit_texture_aug strong
--fruit_texture_layout mixed
--fruit_texture_collage_prob 0.35
```

## 2026-06-23 C 모델 최종화 요약

C 모델은 MobileNetV3-Small classifier입니다.

```text
input:
  B가 복원한 face quad 기준 rectified face crop

output:
  apple / orange / banana / pineapple / plain / unknown
```

최종 로컬 weight:

```text
weights/c_mobilenetv3small_final.pt
```

GitHub에는 `*.pt`가 올라가지 않으므로, 대신 최종 모델 metadata만 push합니다.

```text
weights/c_mobilenetv3small_final.json
```

선택된 실험:

```text
runs/meta_v2_c_facecls/c_mobilenetv3small_anchor_pair_exp006
```

핵심 결과:

| case | variant | result |
| --- | --- | ---: |
| orange hard case | top_label_whitened | orange 0.999999 |
| orange hard case | remove_top_30_resize | orange 1.000000 |
| apple target case | top_label_whitened | apple 0.9972 |
| apple target case | remove_top_30_resize | apple 0.9989 |

원본 C validation:

| metric | result |
| --- | ---: |
| overall val acc | 0.9854 |
| apple val acc | 0.9789 |
| orange val acc | 0.9492 |
| apple -> orange val error | 5 / 475 |
| orange -> apple val error | 10 / 433 |

중요한 결론:

```text
orange hard case 하나만 보고 orange 데이터를 많이 넣으면 apple이 무너질 수 있다.
apple/orange를 generic하게 같은 수량으로 증강해도 orange hard case가 다시 깨질 수 있다.
실패한 실제 crop에 가까운 anchor pair를 균형 있게 넣는 방식이 현재 가장 안전하다.
blur가 심한 crop은 softmax confidence만 믿지 말고 quality gate + temporal vote가 필요하다.
```

## 2026-06-22 현재 전략 분기: visibility-aware Option 3

이 섹션은 2026-06-22 기준으로 새로 정리한 현재 의사결정입니다. 2026-05-30의 `arena booster` 구현 상태는 보존 기준점으로 삼고, 그 위에서 실제 경기 운영에 맞는 visibility-aware 인식 파이프라인을 단계적으로 붙이는 방향입니다.

현재 깨끗한 기준점은 다음과 같습니다.

```text
commit: 625e0fd 2026-05-30 11:24:24 +0900 Add arena booster dataset generation
branch backup: backup/arena-booster-20260530-clean
tag backup: backup-arena-booster-20260530-clean
source zip: backups/arena_booster_20260530_clean_source_20260622_163828.zip
git bundle: backups/arena_booster_20260530_clean_git_20260622_163828.bundle
```

### 핵심 결론

Task 1용 모델과 Task 2용 모델을 완전히 분리하는 방식은 현재 기준에서 우선순위가 낮습니다. 과일 사진이 붙은 상자도 물리적으로는 cube이고, 과일 사진 면이 보이지 않는 순간에는 plain cube와 구분할 수 없습니다. 따라서 모델을 Task 단위로 쪼개기보다는, 역할 단위로 나누어 다음 순서로 판단합니다.

```text
Camera frame
  -> 기존 8-class YOLO-seg를 proposal model로 사용
  -> cube / banana / orange / pineapple / apple 결과를 cube-like candidate로 취급
  -> 보이는 fruit face만 별도로 찾음
  -> 보이는 face crop만 warp/classify
  -> 여러 frame과 viewpoint에서 evidence를 누적
  -> pickup / inspect-again / skip / avoid 결정
```

가장 중요한 규칙은 다음입니다.

```text
숨겨진 과일은 분류하지 않는다.
cube-like object는 plain cube 확정이 아니다.
과일 cube는 Task 1에서도 cube처럼 보일 수 있다.
Task 1 cube pickup은 한 view만 보고 확정하지 않는다.
보이는 fruit face crop만 fruit class evidence로 사용한다.
```

### 실제 경기장 28개 object 해석

룰북 기준으로 경기장에는 총 28개 object가 동시에 놓이는 것으로 보는 것이 맞습니다.

```text
기본 도형 object: 4종 x 각 4개 = 16개
과일 cube object: 4종 x 각 3개 = 12개
총 28개
```

다만 학습 이미지 한 장에 항상 28개가 전부 보이게 만드는 것은 바람직하지 않습니다. 실제 로봇 카메라는 낮고 시야가 제한되어 있으므로, 한 frame에서 보이는 것은 보통 전체 arena의 일부입니다.

따라서 향후 arena booster는 다음 철학으로 가는 것이 좋습니다.

```text
world scene에는 28개 object를 실제처럼 배치한다.
object끼리는 룰북처럼 서로 붙지 않게 한다.
벽 접촉 배치는 허용한다.
camera는 낮은 robot view로 arena 일부만 보게 한다.
학습 frame에는 가까운 object 1~5개, 중거리 object 일부, 멀리 작은 object 일부가 섞이게 한다.
```

즉, "28개를 한 이미지에 모두 때려 넣는 top-view dataset"이 아니라, "28개가 깔린 arena를 로봇 시점으로 훑은 partial-view dataset"이 목표입니다.

### 현재 선택한 구현 순서: Option 3

현재 선택은 Option 3입니다.

```text
Option 3:
  기존 8-class YOLO를 당장 버리지 않는다.
  기존 YOLO 결과를 final truth가 아니라 proposal/evidence로 재해석한다.
  A2 fruit_face_patch, B0 face warp, C face classifier를 먼저 만든다.
  A1 cube_like body model은 나중에 필요할 때 학습한다.
```

역할별 의미는 다음과 같습니다.

```text
Model A, 현재:
  기존 8-class YOLO26 segmentation.
  full-frame에서 가장 무거운 proposal model.
  banana/orange/pineapple/apple/cube는 runtime에서 cube-like candidate로 묶어 해석한다.
  octahedron/dodecahedron/icosahedron은 Task 1 shape evidence로 직접 사용할 수 있다.

Model A2:
  fruit_face_patch detector.
  과일 사진이 실제로 보이는 면 영역만 segmentation한다.
  처음에는 crop-only 또는 full-frame 소형 실험 둘 다 가능하지만, 최종적으로는 cube-like candidate crop 안에서만 돌리는 것이 안전하다.

Model B:
  cube face quad finder.
  당장 새 model로 시작하지 않는다.
  MVP에서는 A2 fruit_face_patch mask에서 contour/minAreaRect/approxPolyDP로 quad를 근사한다.
  plain face까지 안정적으로 찾을 필요가 생기면 crop-only pose/keypoint model로 확장한다.

Model C:
  warped face classifier.
  입력은 perspective-warp된 160x160 또는 224x224 face crop.
  출력 class는 apple / orange / banana / pineapple / plain / unknown.
```

### A2는 왜 모든 면이 아니라 과일면만 먼저 찾는가

현재 A2의 1차 목표는 모든 cube face가 아니라 `fruit_face_patch`만 찾는 것입니다.

```text
A2 1차 범위:
  입력: Model A가 찾은 cube-like crop
  출력: 과일 사진이 실제로 보이는 면의 segmentation mask

A2가 지금 하지 않는 일:
  plain white face 찾기
  cube의 모든 visible face quad 찾기
  plain cube 확정하기
```

이 결정을 한 이유는 데이터와 일정 때문입니다. 현재 100k dataset metadata에는 `visible_face_segments`가 이미 들어 있어 과일면 detector dataset은 바로 만들 수 있습니다. 반면 모든 cube face를 학습하려면 plain face와 fruit cube의 흰 면까지 포함한 `cube_faces[].quad_xy` ground truth가 필요하지만, 현재 100k metadata에는 이 정보가 없습니다.

따라서 현 단계에서는 다음 순서가 가장 안전합니다.

```text
1. A2는 fruit face patch 전용으로 만든다.
2. A2 mask에서 contour/minAreaRect/approxPolyDP로 B0 quad를 근사한다.
3. C classifier로 보이는 과일면만 apple/orange/banana/pineapple/unknown으로 판단한다.
4. fruit face가 없는 cube-like object는 plain cube가 아니라 unresolved로 둔다.
5. 실제 테스트에서 Task1 cube 판단이 병목이면 그때 모든 face finder를 B1으로 추가한다.
```

즉, "모든 면을 찾는 기능"은 A2에 억지로 넣지 않고, 추후 `Model B1: crop-only cube face quad finder`의 역할로 남깁니다.

### fruit_face_patch의 정확한 의미

`fruit_face_patch`는 과일 cube 전체가 아닙니다. 카메라에서 실제로 보이는 과일 사진 면의 픽셀 영역만 의미합니다.

```text
기존 8-class apple/orange/banana/pineapple label:
  과일 사진이 붙은 cube object 전체 mask

fruit_face_patch:
  cube 전체 중 과일 사진이 실제로 보이는 면 영역만
  흰 cube 면은 포함하지 않음
  카메라 뒤쪽에 가려진 과일면도 포함하지 않음
```

예를 들어 fruit cube가 흰 면만 보이는 각도라면 object는 cube-like candidate지만 `fruit_face_patch`는 없습니다. 이 경우 해당 object를 plain cube로 확정하면 안 되고, "unresolved cube-like"로 두고 다른 viewpoint에서 다시 봐야 합니다.

### 현재 100k dataset metadata로 가능한 것

`datasets/yolo26_seg_100000_ideal_debug`의 `_meta`를 확인한 결과, fruit face 학습에 필요한 정보는 이미 충분히 들어 있습니다.

집계 결과는 다음과 같습니다.

```text
train json: 90000
val json:   5000
test json:  5000

train fruit objects: 80453
  with visible_face_segments: 80453
  with visible_face_bbox:     80453
  with face_textures:         80453

val fruit objects: 4418
  with visible_face_segments: 4418

test fruit objects: 4461
  with visible_face_segments: 4461
```

각 fruit object에는 대체로 다음 정보가 있습니다.

```text
class
object_name
fruit_visibility_pass
visible_face_pixels
visible_face_bbox
visible_face_bbox_width
visible_face_bbox_height
visible_face_segments
face_textures
fruit_visible_ratio
object_visible_ratio
```

따라서 지금 100k dataset만으로 바로 가능한 일은 다음입니다.

```text
A2 fruit_face_patch segmentation dataset 생성
  visible_face_segments -> YOLO segmentation label class 0

C fruit classifier crop dataset 생성
  visible_face_segments 또는 visible_face_bbox -> crop/approx warp
  face_textures/class -> apple/orange/banana/pineapple label

B0 no-model warp MVP
  visible_face_segments -> contour -> minAreaRect/approxPolyDP -> perspective-ish warp
```

### 현재 100k metadata로 부족한 것

현재 100k metadata에는 모든 cube 면의 정확한 4-corner quad가 없습니다.

없는 정보는 다음과 같습니다.

```text
cube_faces
quad_xy
quad_xy_norm
plain face quad
fruit cube의 흰 면 quad
plain cube의 visible face quad
각 cube face의 face_kind / visible_ratio / quality
```

따라서 현재 dataset만으로는 "plain cube face를 정확한 GT quad로 학습"하기 어렵습니다. fruit face는 mask가 있으므로 근사 warp가 가능하지만, plain face는 별도 export가 필요합니다.

향후 generator에 추가해야 할 metadata 예시는 다음입니다.

```json
{
  "cube_faces": [
    {
      "object_name": "apple_000",
      "face_index": 0,
      "face_kind": "fruit",
      "fruit_class": "apple",
      "quad_xy": [[120, 130], [220, 126], [230, 230], [128, 236]],
      "quad_xy_norm": [[0.1875, 0.2031], [0.3438, 0.1969], [0.3594, 0.3594], [0.2000, 0.3688]],
      "visible_pixels": 2345,
      "visible_ratio": 0.73,
      "quality": "good"
    }
  ]
}
```

이 정보가 들어가면 Model B crop-only quad finder와 Model C의 plain/unknown 학습을 훨씬 안정적으로 만들 수 있습니다.

### 왜 전체 100k를 새로 만들지 않는가

대회까지 약 30일 남은 현재 시점에서는 100k 전체를 새로 만드는 것은 권장하지 않습니다.

기존 100k의 장점은 다음입니다.

```text
이미 object segmentation 다양성이 충분하다.
visibility gate 기반으로 label 품질이 검증되어 있다.
fruit face visible segment가 이미 metadata에 있다.
기존 YOLO proposal model 학습/검증 흐름과 호환된다.
```

새로 생성해야 하는 것은 전체 대체 dataset이 아니라 부족한 부분만 보강하는 targeted booster입니다.

```text
재사용:
  datasets/yolo26_seg_100000_ideal_debug
  datasets/arena_sun111_sun168_booster_v1, 존재하는 경우

새로 생성할 가능성이 높은 booster:
  28-object arena partial-view booster
  robot camera low-view booster
  SUN-111 floor / SUN-168 low fence booster
  cube face quad metadata booster
  plain face / unknown face classifier booster
```

### 권장 개발 순서

현재부터의 권장 순서는 다음입니다.

```text
1. 기존 8-class YOLO를 proposal model로 유지한다.
2. 100k metadata에서 fruit_face_patch YOLO dataset을 만든다.
3. 100k metadata에서 warped fruit face classifier dataset을 만든다.
4. 2D mimic face classifier dataset을 만든다.
5. MobileNetV3-Small 기반 C classifier를 먼저 학습한다.
6. 기존 YOLO + B0 contour/minAreaRect warp + C classifier로 MVP inference를 만든다.
7. 실제 카메라 frame 또는 preview에서 failure case를 모은다.
8. failure case가 plain face/hidden fruit cube라면 generator에 cube_faces quad export를 추가한다.
9. 그 다음 crop-only B quad finder를 학습한다.
10. 마지막에 필요하면 A1 cube_like object-body model을 별도 학습한다.
```

### 2026-06-22 추가된 초기 유틸리티

위 방향에 맞춰 기존 코드를 제거하지 않고, 100k metadata를 재사용하는 초기 dataset export 스크립트를 추가했습니다.

```text
scripts/make_fruit_face_patch_dataset.py
  목적: A2 fruit_face_patch YOLO-seg dataset 생성
  기본 모드: object_crop
  입력: 기존 images/labels/_meta
  출력: cube-like crop image + fruit_face_patch label
  positive: fruit face가 보이는 fruit cube crop
  negative: plain cube crop 또는 fruit face가 기준 미달인 cube-like crop

scripts/export_warped_face_dataset.py
  현재 기본값: --warp_method bbox
  주의: 현재 100k metadata의 visible_face_segments는 과일 사진이 보이는 픽셀 mask이지,
        cube face의 실제 4-corner quad가 아니다.
        따라서 approx_quad/minAreaRect warp는 preview에서 면을 정사각형으로 안정적으로 펴지 못했다.
        classifier C의 1차 학습에는 bbox 기반 square crop을 사용하고,
        true perspective warp는 generator가 cube_faces[].quad_xy를 export한 뒤 다시 진행한다.
  목적: C warped face classifier dataset 생성
  입력: 기존 images/_meta visible_face_segments
  출력: apple/orange/banana/pineapple/plain/unknown 폴더 구조
  현재 채움: apple/orange/banana/pineapple, 그리고 기준 미달 crop 기반 unknown
  현재 한계: plain face는 cube face GT가 없어 아직 자동으로 채우지 않음
```

예시 실행:

```powershell
C:\Users\user\anaconda3\envs\ai_robotics\python.exe scripts\make_fruit_face_patch_dataset.py `
  --source_dataset datasets\yolo26_seg_100000_ideal_debug `
  --output_root datasets\fruit_face_patch_a2_v1 `
  --splits train val test `
  --crop_mode object_crop `
  --copy_mode hardlink `
  --min_face_pixels 900 `
  --min_face_side 20

C:\Users\user\anaconda3\envs\ai_robotics\python.exe scripts\export_warped_face_dataset.py `
  --source_dataset datasets\yolo26_seg_100000_ideal_debug `
  --output_root datasets\warped_face_cls_v1 `
  --splits train val test `
  --crop_size 224 `
  --min_face_pixels 900 `
  --min_face_side 20 `
  --warp_method bbox
```

두 스크립트 모두 OpenCV를 사용하므로 `ai_robotics` 환경의 Python으로 실행하는 것을 기준으로 합니다.

### 세부 TODO

#### Dataset TODO

```text
[ ] scripts/make_fruit_face_patch_dataset.py
    입력: datasets/yolo26_seg_100000_ideal_debug
    출력: fruit_face_patch YOLO-seg dataset
    class: 0 fruit_face_patch
    source: _meta/*/visible_face_segments
    옵션: min_face_pixels, min_face_side, copy/symlink, preview

[ ] scripts/export_warped_face_dataset.py
    입력: images + _meta visible_face_segments
    출력: face classifier dataset
    classes: apple, orange, banana, pineapple, plain, unknown
    1차 구현: fruit classes만 확실히 만들고, unknown은 bad crop/blur/tiny crop으로 구성
    crop: bbox square crop 기본
    experimental warp: approx_quad/minAreaRect, 단 true quad GT가 없어 preview 검수 필수
    audit CSV: source image, object_name, class, bbox, segment point count, quality

[ ] scripts/generate_2d_face_mimic_dataset.py
    fruit texture pool에서 2D face crop 대량 생성
    print-like color jitter, blur, JPEG, gamma, crop jitter, partial occlusion 포함
    plain과 unknown class 포함

[ ] scripts/audit_visibility_datasets.py
    fruit_face_patch label 수
    class별 warped crop 수
    tiny/blur/unknown 비율
    apple/orange balance
    깨진 이미지 검사
```

#### Model TODO

```text
[ ] 기존 8-class YOLO best.pt를 proposal model로 고정해 baseline inference 작성
[ ] A2 fruit_face_patch YOLO-seg 학습
[ ] C MobileNetV3-Small classifier 학습
[ ] C confusion matrix 확인
[ ] apple/orange confusion hard set 따로 확인
[ ] unknown threshold와 confidence margin 설계
[ ] B0 no-model quad extraction 구현
[ ] B0 warp 품질 preview sheet 생성
```

#### Generator TODO

```text
[ ] generate_yolo_coco_composite.py에 cube face GT export 추가
[ ] 각 cube object의 6개 face 중 visible face 판정
[ ] face_kind: fruit / plain
[ ] fruit_class 기록
[ ] quad_xy, quad_xy_norm 기록
[ ] visible_pixels, visible_ratio, quality 기록
[ ] heavily occluded/truncated face는 quality=bad 처리
[ ] 28-object arena scene 생성 옵션 검토
[ ] world에는 28개 object를 배치하되 robot camera partial-view로 render
```

#### Runtime TODO

```text
[ ] robot_vision/visibility_pipeline.py 작성
[ ] 기존 YOLO output을 cube_like candidate로 재해석
[ ] cube-like crop 생성
[ ] A2 또는 B0로 fruit face patch 추출
[ ] face warp
[ ] C classifier 적용
[ ] track별 evidence memory 유지
[ ] pickup / inspect-again / skip / avoid decision API 작성
```

#### Decision Logic TODO

```text
[ ] Task2 target fruit:
    target fruit face가 좋은 quality로 반복 검출되면 pickup
    confidence >= threshold
    top1 - top2 margin >= threshold

[ ] Task1 non-cube shape:
    octahedron/dodecahedron/icosahedron은 기존 YOLO shape evidence 중심으로 pickup

[ ] Task1 cube:
    cube-like만 보이면 unresolved
    fruit face가 target/non-target으로 보이면 plain cube가 아님
    여러 viewpoint에서 좋은 plain evidence가 반복될 때만 plain cube candidate
    불확실하면 inspect-again 또는 skip
```

#### Validation TODO

```text
[ ] synthetic validation은 pipeline sanity check 용도로만 사용
[ ] 실제 robot camera frame 100~300장 수집
[ ] fruit face crop label부터 먼저 수작업 검증
[ ] C classifier real crop confusion matrix 작성
[ ] end-to-end pickup decision replay test 작성
[ ] 흔들림/motion blur frame 별도 hard set 구성
[ ] 28-object arena clutter preview 확인
```

#### Deployment TODO

```text
[ ] C classifier ONNX export
[ ] C classifier TensorRT 가능성 확인
[ ] 기존 YOLO TensorRT export 문제 재확인
[ ] per-frame latency 측정
[ ] top-K cube-like crop만 후속 모델 실행
[ ] unresolved track은 매 frame이 아니라 N frame마다 재검사
[ ] final threshold를 실제 카메라 validation으로 튜닝
```

### 지금 당장 하지 않을 일

아래 항목은 나중으로 미룹니다.

```text
새 100k 전체 재생성
Task 1 전용 shape-only model을 먼저 학습
Task 2 전용 fruit-only full-frame model을 먼저 학습
full-frame YOLO-pose로 face quad를 바로 찾기
overlapping label이 있는 5-class Model A를 바로 최종 구조로 확정
cube_like를 plain cube로 즉시 확정하는 runtime 정책
```

5-class Model A는 나쁜 아이디어가 아니라, 실험 순서상 뒤로 미룹니다. `cube_like_object`와 `fruit_face_patch`가 같은 위치에서 겹치는 label 구조이므로 Ultralytics YOLO-seg 학습에서 간섭이 생길 수 있습니다. 먼저 기존 8-class YOLO를 proposal로 쓰고, A2/C/B0를 붙여 실제 성능을 확인한 뒤에 A1 또는 통합 A를 실험하는 것이 안전합니다.

BlenderProc와 COCO 배경 이미지를 이용해 AI 로봇 챌린지용 8-class YOLO 합성 데이터를 생성하고 학습하는 프로젝트입니다. 현재 기준 문서는 이 `readme.md` 하나이며, 기존 README와 백업 매니페스트의 핵심 내용을 현재 파일 구조에 맞춰 통합했습니다.

현재 주력 파이프라인은 **YOLO segmentation 데이터 생성**입니다. bbox 라벨도 만들 수 있지만, 품질 검증과 최종 학습 기준은 segmentation 라벨입니다.

작업 환경은 Windows PowerShell과 `conda activate ai_robotics` 기준입니다.

## 문서 빠른 길잡이

처음 복구하거나 새 PC에서 실행할 때는 아래 순서로 보면 됩니다.

```text
설치 및 복구
  -> 필수 로컬 데이터
  -> YOLO26n segmentation end-to-end pipeline
  -> Preview 생성
  -> 품질 체크리스트
  -> 진행 상황 확인
```

설계 의도를 확인할 때는 `전체 설계 의도`, `데이터 설계 판단`, `라벨 정책`, `Visibility 지표`, `Metadata 설계`를 먼저 보면 됩니다. 실제 명령만 빠르게 확인하려면 `100장 품질 검증 생성`, `50,000장 생성`, `YOLO26n segmentation end-to-end pipeline` 섹션을 보면 됩니다.

## 2026-05-20 현재 최종 기준 요약

현재 최종 학습용 생성 파이프라인은 **YOLO26n segmentation + ideal geometry visibility gate + 최종 color render** 구조입니다. 비싼 Blender color render를 먼저 돌리지 않고, 단순 mesh polygon을 카메라에 투영해서 object visibility와 fruit face visibility를 먼저 계산합니다. 이 단계에서 target tier와 actual tier가 맞고, object/fruit visibility 기준을 통과한 배치만 최종 이미지로 렌더합니다.

현재 기준에서 가장 중요한 운영 결론은 아래입니다.

```text
branch: main
model: yolo26n-seg.pt
label format: segment
image size: 640x640
samples: 16
min/max objects: 1~4
negative ratio: 0.18
background: mostly COCO 2017, arena-style background about 5%
lighting: soft_overhead
fruit texture pool: datasets/fruit_textures/final_fruits36065_original25_fruitseg30_10
fruit texture source mix per class: 1040 Fruits-360 + 400 crawled/original + 160 FruitSeg30
FruitSeg30 apple source: Apple_Gala only, Apple_Golden excluded
fruit texture augmentation: light
fruit texture layout: mixed single/collage, collage probability 0.35
fruit classes per image: mixed allowed
fruit textures per cube: multiple same-class textures allowed
fruit tiers: easy 50%, mid 45%, hard 5%
min fruit visible ratio: 0.22
min fruit face pixels: 1800
single-object min fruit face pixels: 2200
hard-tier fallback floor: fruit_vis 0.10, fruit_face_pixels 900
train augmentation: low_aug
```

검수 이미지는 두 종류를 사용합니다.

```text
ideal_visibility_debug/train/
  검은 배경 위에 ideal geometry visibility 계산 결과를 표시합니다.

visibility_preview/train/
  최종 합성 사진 위에 YOLO label, fruit_vis, obj_vis, OK/fallback을 표시합니다.
```

`scripts/run_yolo26_seg_pipeline.py`는 생성, split, audit, preview, train, predict, side-by-side 생성을 묶는 현재 권장 end-to-end entry point입니다. 생성 품질 검사는 `scripts/audit_fruit_generation.py`가 담당합니다. 기본적으로 한 이미지 안의 mixed fruit class 허용 여부, cube별 texture metadata, face texture metadata 존재 여부를 확인합니다.

## 실험 발전 기록

지금 설정은 한 번에 나온 것이 아니라, confusion matrix와 preview에서 보인 실패 패턴을 줄이면서 정한 값입니다. 정확한 run 산출물 전체는 `runs/`와 `datasets/` 아래 대형 파일이라 Git에 올리지 않지만, 모델을 발전시킨 판단은 아래처럼 남깁니다.

| 단계 | 관찰된 confusion matrix 패턴 | 판단 | 다음 조치 |
|---|---|---|---|
| 초기 legacy YOLO/YOLO11 계열 bbox/seg 실험 | cube/polyhedron은 대각선이 비교적 강하지만 fruit cube class는 off-diagonal이 큼 | 과일상자 segmentation 자체보다 fruit label 분리가 병목 | 기본 모델을 `yolo26n-seg.pt`로 올리고 segmentation 중심으로 전환 |
| 과일 texture 변화가 거의 없는 50,000장 실험 | fruit box mask는 잘 잡지만 apple/orange, banana/pineapple, orange/apple 혼동이 큼 | 물체 검출은 배우지만 과일 사진 class cue가 좁음 | source texture 다양화와 face texture augmentation 도입 |
| `fruit_texture_aug strong` + 강한 색 변화 | fruit class off-diagonal이 줄지 않거나 일부 악화 | 색 변화가 class 고유 색까지 흔들어 라벨 의미를 흐림 | `fruit_texture_aug light`로 낮춤 |
| train 기본 augmentation | YOLO 학습 augmentation이 synthetic texture 색을 추가로 흔들어 fruit confusion 유지 | 생성 단계에서 이미 충분히 흔들었으므로 학습 증강은 낮게 유지 | `--train_low_aug`, mosaic/erasing/auto_augment 제거 |
| hard visibility 비율 높음 | fruit face가 너무 작거나 비스듬한 샘플이 fruit class 학습을 어렵게 만듦 | hard sample은 필요하지만 주력 분포가 되면 안 됨 | hard tier를 5%로 낮추고 hard 하한을 별도 지정 |
| 한 cube 안에 다른 fruit가 섞이는 preview 발견 | `_meta` class와 실제 렌더 texture가 불일치하는 치명적 실패 | 이 상태의 학습은 poison data | face texture/material metadata audit, class-path 검증, unique generated texture 파일 경로로 수정 |
| 한 texture만 반복 | 같은 사진을 외우는 위험, 실제 인쇄물/촬영 변동 부족 | 같은 fruit class 안에서 다양한 texture를 보여야 함 | cube face별 같은 class 다른 texture 허용 |
| 작은 collage texture | 여러 과일을 붙였지만 과일 하나하나가 너무 작아 class cue가 약함 | collage는 좋지만 patch scale이 더 커야 함 | collage patch ratio를 light 기준 0.52~0.70으로 상향 |
| 최종 texture pool | Fruit360만으로는 studio domain이 강하고, crawling만으로는 품질 흔들림 | 고정된 final pool을 만들어 재현성과 다양성을 같이 확보 | `final_fruits36065_original25_fruitseg30_10` 사용 |

현재 목표는 “YOLO 하나가 모든 fruit class를 완벽히 분류”가 아니라, 먼저 fruit box segmentation과 fruit class label이 깨끗한 synthetic dataset을 만들고, 그 위에서 YOLO26n-seg가 fruit class off-diagonal을 얼마나 줄이는지 보는 것입니다. 그래도 fruit class confusion이 남으면 그때는 YOLO seg mask crop 위에 별도 lightweight classifier를 올리는 2-stage 구조를 검토합니다.

## 현재 진행 상태

마지막 정리 기준: 2026-05-20 (Asia/Seoul)

- Git 원격 저장소: `https://github.com/jaeyoungi2006/Data_Generation_Blender.git`
- 로컬 브랜치: `main`
- 주요 소스 코드: `scripts/`
- 생성용 기본 OBJ asset: `assets/generated/`
- SMTP 예시 설정: `config/email_smtp.example.json`
- Conda 환경 export: `environment.yml`
- 복구용 YOLO 데이터셋 메타데이터:
  - `datasets/yolo_8class_maxobj5_balanced2_10000/data.yaml`
  - `datasets/yolo_8class_maxobj5_balanced2_10000/splits/front5000.yaml`
  - `datasets/yolo_8class_maxobj5_balanced2_10000/splits/front5000_train.txt`
  - `datasets/yolo_8class_maxobj5_balanced2_10000/splits/front5000_val.txt`
- 학습 기록 메타데이터:
  - `runs/detect/runs/yolo_train/yolo11n_front5000_fast_b64/args.yaml`
  - `runs/detect/runs/yolo_train/yolo11n_front5000_fast_b64/results.csv`
- 추적 중인 legacy checkpoint:
  - `weights/best.pt`
  - `weights/yolo11n_front5000_fast_b64_best.pt`

두 checkpoint는 같은 파일이며, 현재 YOLO26n segmentation 파이프라인의 권장 checkpoint는 새로 학습한 `runs/yolo26_seg_train/<run_name>/weights/best.pt`를 사용합니다. 대형 학습 산출물은 Git에 올리지 않습니다.

```text
SHA256: AB44018B768C7CCB0B3F08FD31769337D88EDF15CB567F46800FB45C25A2CBE5
```

대형 데이터셋, 학습 run 산출물, 다운로드 캐시, 루트의 YOLO 기본 모델 파일(`yolo11n.pt`, `yolo26n.pt`, `yolo26n-seg.pt`)은 `.gitignore` 기준으로 Git에 올리지 않습니다. 데이터셋은 로컬에서 재생성하는 구조입니다.

## 클래스 정의

`data.yaml`의 class 순서는 아래와 같습니다.

```text
0 banana
1 orange
2 pineapple
3 apple
4 cube
5 octahedron
6 dodecahedron
7 icosahedron
```

과일 class는 실제 과일 mesh가 아닙니다. 흰색 cube의 일부 면에 과일 사진 texture를 붙인 object입니다. 대회 당일 촬영한 과일 사진을 cube에 붙여 합성 데이터를 만드는 상황을 가정한 구조입니다.

## 전체 설계 의도

이 프로젝트는 단순히 예쁜 synthetic image를 만드는 것이 아니라, 실제 카메라에서 들어올 수 있는 불완전한 장면을 견디는 YOLO 모델을 만들기 위해 설계했습니다. 핵심 목표는 다음과 같습니다.

```text
1. 대회 환경의 class 정의를 그대로 반영한다.
2. 실제 촬영 데이터가 적어도 재생성 가능한 합성 데이터로 빠르게 보강한다.
3. 합성 이미지는 다양하게 흔들되, 라벨은 최종 렌더에서 실제 보이는 픽셀만 사용한다.
4. 과일 texture가 보이지 않는 cube를 과일로 학습시키지 않는다.
5. 작은 object, 프레임 밖 object, 과도한 occlusion, 얇은 sliver label을 줄인다.
6. 흰색 물체와 배경에서 생기는 false positive를 negative sample로 억제한다.
7. 생성 결과를 preview와 metadata로 사람이 빠르게 검수할 수 있게 한다.
8. 대형 데이터는 Git에 넣지 않고, 코드와 설정만으로 재생성 가능하게 남긴다.
```

설계의 큰 방향은 “이미지는 다양하고 지저분하게, 라벨은 보수적이고 깨끗하게”입니다. 카메라 노이즈, 조명, blur, JPEG artifact, 배경 변화는 일부러 강하게 넣지만, 학습 target은 ideal geometry visibility map 또는 BlenderProc `instance_segmap` 기반으로 계산해서 사람이 그린 bbox보다 일관되게 만듭니다. 최종 학습용 생성은 ideal geometry visibility gate를 우선 사용합니다.

## 데이터 설계 판단

과일은 실제 3D mesh 대신 cube 위의 texture로 구현했습니다. 이유는 대회에서 실제 물체가 “과일 사진이 붙은 흰색 cube”에 가깝고, 모델이 과일의 3D 형상보다 cube 면에 붙은 2D 과일 이미지를 보고 분류해야 하기 때문입니다.

fruit cube에는 3개 면에 과일 texture overlay를 둡니다. 한 cube의 fruit class는 하나로 고정하며, 예를 들어 apple cube면 3개 면 모두 apple texture만 사용합니다. 현재 기본 생성은 한 이미지 안에 서로 다른 fruit class의 cube가 같이 나올 수 있고, 한 cube의 3개 면도 같은 class 안에서 서로 다른 source texture를 쓸 수 있습니다. 필요하면 `--single_fruit_class_per_image`, `--single_fruit_texture_per_cube`로 예전처럼 강하게 묶을 수 있습니다. 이 overlay는 별도 helper object로 segmentation에 잡히며, 이를 통해 “cube 전체가 보였는지”와 “과일 사진 면이 실제로 보였는지”를 따로 계산합니다.

과일 texture는 class별 색상 score로 1차 필터링합니다. `collect_fruit_textures()`는 apple, banana, orange, pineapple 각각의 대표 색상 조건을 통과하는 이미지를 class별로 캐시하고, banana는 작은 cube에서 pineapple처럼 보이는 초록 다발 texture를 기본 후보에서 제외합니다. JPG/JPEG뿐 아니라 PNG/WebP도 읽으며, alpha가 있는 누끼 texture는 흰 배경 위에 합성해서 사용합니다. texture 파일의 선언 class가 cube class와 다르면 생성 단계에서 즉시 실패시키고, `_meta/*.json`에는 face별 texture 경로와 class를 기록합니다. 이는 잘못된 crop이나 색 정보가 약한 이미지를 줄이고, 한 cube 안에 서로 다른 fruit class가 섞이는 것을 막기 위한 장치입니다.

선택된 texture는 Blender file texture로 직접 로드하고, face별 UV transform으로 회전/확대/축소를 줍니다. `--fruit_texture_aug light`는 회전 ±22도, 배율 0.758x~1.32x이고, `strong`은 회전 ±45도, 배율 0.588x~1.70x입니다. `--fruit_texture_layout collage`를 켜면 같은 class 안에서 작은 과일 사진 여러 장을 흰 배경에 붙인 generated face texture를 만들고, `mixed`를 쓰면 `--fruit_texture_collage_prob` 확률로 single/collage를 섞습니다.

흰색 polyhedron class는 PLA 출력물처럼 보이도록 off-white 색상과 roughness를 랜덤화합니다. 완전한 순백색 하나만 쓰면 모델이 색상에 과하게 붙을 수 있어서, 조명 아래에서 흔들리는 흰색 계열을 학습하도록 설계했습니다.

## 장면 생성 설계

object 수는 기본적으로 positive scene에서 `--min_objects`부터 `--max_objects` 사이로 샘플링합니다. 현재 최종 학습 권장값은 `1~4`입니다. 너무 많은 물체를 넣으면 occlusion과 작은 라벨이 늘고, 너무 적으면 실제 난이도를 충분히 만들지 못합니다.

object 위치는 완전히 균일하게 뿌리지 않습니다. 일부는 기존 object 근처에 배치해 겹침과 근접 상황을 만들지만, 중심 간 최소 거리를 둬서 모든 라벨이 무의미하게 겹치지는 않게 합니다.

fruit visibility tier는 과일 면이 항상 정면으로 잘 보이는 데이터만 생기지 않도록 만든 장치입니다.

```text
easy: 과일 면이 크게 보이는 장면
mid:  과일 면이 부분적으로 보이는 장면
hard: 과일 면이 작거나 비스듬히 보이는 장면
```

구현상 `easy`는 작은 tilt, `mid`는 더 큰 tilt, `hard`는 완전 random rotation에 가깝습니다. scale도 tier별로 조정합니다. easy는 조금 크게, hard는 조금 작게 만들어 실제 난이도 분포를 섞습니다.

생성 단계에서 각 object transform은 여러 번 시도합니다. 과일 object는 최대 300번, 일반 shape는 최대 120번 후보를 평가합니다. 후보 평가는 다음 항목을 같이 봅니다.

```text
frame 안쪽 margin을 지키는지
projected object area가 충분한지
fruit face 예상 픽셀 수가 충분한지
fruit visibility tier 범위에 들어오는지
```

완벽한 후보를 찾지 못하면 가장 점수가 좋은 후보를 사용합니다. 이렇게 하면 생성이 멈추지 않으면서도 품질 기준에 가까운 장면을 계속 만들 수 있습니다.

이 후보 배치 후에는 `--ideal_visibility` 단계가 한 번 더 있습니다. 이 단계는 실제 color render를 돌리기 전에 ideal geometry만으로 occlusion을 계산합니다. 각 mesh face와 fruit face helper를 카메라에 투영하고, 간단한 depth buffer로 앞에 보이는 instance map을 만든 뒤 다음 조건을 검사합니다.

```text
target fruit tier == actual fruit tier
obj_vis > 0.10
fruit_vis >= 0.10
visible fruit face pixels >= 300
visible fruit face bbox width/height >= 10 px
```

조건을 만족하지 못하면 color render 없이 배치를 다시 샘플링합니다. 기본 `--ideal_visibility_max_attempts`는 80입니다. 80번 안에 완벽한 배치를 찾지 못하면 마지막 후보를 사용하고 로그에 fallback을 남깁니다.

## 카메라와 렌더링 설계

카메라는 640x640 기본 해상도에서 lens, 위치, target을 매번 바꿉니다. lens는 넓은 화각부터 망원 느낌까지 흔들고, 카메라 위치는 tabletop/close-up 사이 느낌이 섞이도록 제한된 범위에서 샘플링합니다.

조명은 world color, key light, fill light, optional under light, optional ring light를 랜덤화합니다. 실제 촬영에서는 조명 방향과 색온도가 일정하지 않기 때문에 warm/cool 색을 섞고, area/point/sun light를 섞습니다.

배경은 COCO 이미지를 crop해서 사용합니다. 배경 자체에도 blur, gain/bias, 약한 texture noise를 넣습니다. 이는 물체가 항상 깨끗한 studio background 위에만 놓이는 문제를 피하기 위한 것입니다.

렌더된 foreground는 COCO background와 합성합니다. object mask는 `soften_mask()`로 약간 부드럽게 만들어 배경과 foreground 사이의 지나치게 날카로운 synthetic edge를 완화합니다.

가짜 그림자도 일부 장면에 추가합니다. 그림자는 mask를 이동하고 blur한 뒤 배경을 어둡게 하는 방식입니다. 물리적으로 완벽한 그림자보다, 모델이 바닥/배경의 어두운 접촉부에 익숙해지게 하는 목적이 큽니다.

## 카메라 artifact 설계

최종 이미지에는 실제 카메라에서 흔히 생기는 artifact를 일부러 넣습니다.

```text
channel gain 변화
밝기 gain/bias 변화
gamma 변화
Gaussian noise
Gaussian blur
수평/수직 motion blur
저해상도 down-up sampling
vignetting
JPEG compression
너무 밝거나 너무 어두운 이미지 보정
```

이 artifact는 이미지에만 적용하고, 라벨은 artifact 전에 계산된 segmentation 결과를 기준으로 저장합니다. 즉, 학습 이미지는 더 현실적으로 흔들리지만 라벨 좌표는 렌더 결과와 일관되게 유지됩니다.

## Lens distortion 설계

lens distortion은 무작정 적용하지 않습니다. distortion map이 원본 이미지 밖을 참조하지 않는 파라미터를 최대 80번 찾고, 안전한 파라미터를 찾은 경우에만 이미지, instance segmentation, background를 같은 map으로 remap합니다.

이 방식의 목적은 두 가지입니다.

```text
distortion으로 인한 카메라 다양성은 유지한다.
distortion 후 생기는 검은 테두리나 crop/rescale로 segmentation label이 틀어지는 문제는 피한다.
```

안전한 distortion을 찾지 못하면 해당 이미지에는 distortion을 적용하지 않습니다. 품질을 깨면서까지 증강을 넣지 않는 쪽을 선택한 설계입니다.

## 저장소 구조

```text
Data_Generation_Blender/
  assets/generated/                 # cube, octahedron, dodecahedron, icosahedron OBJ
  config/email_smtp.example.json    # SMTP 설정 예시
  datasets/                         # 대형 데이터와 일부 추적 메타데이터
  runs/                             # 학습 결과와 일부 추적 메타데이터
  scripts/                          # 데이터 생성, 텍스처 준비, preview, 학습 보조 스크립트
  weights/                          # Git에 추적 중인 작은 최종 checkpoint
  environment.yml                   # ai_robotics conda 환경
  main.py                           # 다면체 OBJ 재생성 entry point
  readme.md                         # 현재 기준 문서
```

## 핵심 파이프라인

가장 중요한 생성 흐름은 아래입니다.

```text
scripts/run_yolo_parallel.py
  -> 여러 BlenderProc worker 병렬 실행
  -> worker별 image id chunk 생성
  -> scripts/generate_yolo_coco_composite.py 호출
  -> images/train, labels/train, _meta/train 생성

scripts/make_yolo_bbox_preview.py
  -> YOLO bbox/segment 라벨을 이미지 위에 그려 preview 생성
  -> --debug_visibility 사용 시 fruit/object visibility 지표 표시

scripts/run_yolo26_seg_pipeline.py
  -> 생성, split, audit, preview, 학습, 예측, side-by-side 검수를 한 번에 실행
  -> 현재 YOLO26n segmentation 실험의 권장 entry point
```

`scripts/run_yolo_30000_pipeline.ps1`과 `scripts/run_yolo_50000_pipeline.ps1`은 생성과 학습을 이어서 실행하는 PowerShell wrapper입니다. 현재 README의 권장 segmentation 명령이 최신 기준이며, PowerShell wrapper는 필요 시 경로와 학습 task를 확인해서 사용합니다.

## 주요 스크립트

| 파일 | 역할 |
|---|---|
| `scripts/generate_yolo_coco_composite.py` | BlenderProc 단일/범위 이미지 생성, COCO 배경 합성, YOLO bbox/segment 라벨 생성 |
| `scripts/run_yolo_parallel.py` | 여러 BlenderProc worker를 병렬로 실행하는 주력 생성 launcher |
| `scripts/run_yolo26_seg_pipeline.py` | 생성부터 학습/예측/side-by-side 검수까지 묶은 YOLO segmentation end-to-end pipeline |
| `scripts/run_yolo26_seg_1000_pipeline.ps1` | 1,000장 segmentation smoke/품질 실험 wrapper |
| `scripts/run_yolo26_seg_50000_pipeline.ps1` | 50,000장 segmentation 전체 실험 wrapper |
| `scripts/audit_fruit_generation.py` | fruit class, texture, metadata 일관성 검수 |
| `scripts/make_yolo_bbox_preview.py` | bbox/segmentation preview와 visibility debug 이미지 생성 |
| `scripts/make_polyhedron_objs.py` | cube를 제외한 다면체 OBJ asset 생성 |
| `main.py` | `make_polyhedron_objs.generate_all()`을 호출하는 간단한 entry point |
| `scripts/prepare_kaggle_fruits360_textures.py` | Fruits-360 원본을 class별 texture source로 정리 |
| `scripts/prepare_fruitseg30_textures.py` | FruitSeg30 image/mask pair를 class별 RGBA 누끼 texture source로 정리 |
| `scripts/download_generic_fruit_textures.py` | 일반 과일 texture 후보 다운로드 |
| `scripts/crawl_fruit_images_fast.py` | 웹 과일 이미지 수집 |
| `scripts/extract_fruit_crops.py` | detection source에서 과일 crop 추출 |
| `scripts/clean_fruit_textures_by_color.py` | 색상 기준으로 과일 crop 정리 |
| `scripts/make_mixed_fruit_textures.py` | Fruits-360, crawled/original, FruitSeg30을 섞어 최종 고정 texture pool 생성 |
| `scripts/make_texture_contact_sheets.py` | texture 검수용 contact sheet 생성 |
| `scripts/download_coco128_backgrounds.py` | COCO128 배경 다운로드 |
| `scripts/download_coco2017_backgrounds.py` | COCO 2017 배경 다운로드 |
| `scripts/send_progress_email.py` | SMTP 진행 알림 발송 |
| `scripts/monitor_yolo_training_email.ps1` | 장시간 YOLO 학습 상태를 주기적으로 이메일 알림 |
| `scripts/send_yolo_50000_status_email.py` | 50,000장 실험 상태 요약 이메일 생성 |
| `scripts/send_yolo_50000_status_email_loop.ps1` | 50,000장 실험 상태 이메일 반복 발송 |
| `scripts/train_yolo_with_email.py` | YOLO detection 학습 wrapper와 이메일 알림 |

## 설치 및 복구

새 머신에서 복구할 때:

```powershell
git clone https://github.com/jaeyoungi2006/Data_Generation_Blender.git
cd Data_Generation_Blender
conda env create -f environment.yml
conda activate ai_robotics
```

이미 환경이 있으면:

```powershell
conda env update -n ai_robotics -f environment.yml --prune
conda activate ai_robotics
```

PowerShell에서 프로젝트 루트로 이동합니다.

```powershell
cd (프로젝트 경로)
```

`blenderproc`가 PATH에서 잡히지 않으면 conda 환경의 `Scripts` 경로를 앞에 붙입니다.

```powershell
$env:PATH = "C:\Users\USER\anaconda3\envs\ai_robotics\Scripts;C:\Users\USER\anaconda3\envs\ai_robotics;$env:PATH"
```

## 필수 로컬 데이터

아래 대형 데이터는 Git에 올리지 않습니다. 재생성이나 다운로드로 채웁니다.

```text
assets/generated/
datasets/backgrounds/coco2017/
datasets/fruit_textures/_sources/fruits360_clean_256/
datasets/fruit_textures/_sources/original_color_filtered_512/
datasets/fruit_textures/_sources/fruitseg30_cutouts_512/
datasets/fruit_textures/final_fruits36065_original25_fruitseg30_10/
```

가장 빠른 복구 방법은 기존 작업 PC나 백업에서 아래 두 폴더를 그대로 복사하는 것입니다.

```text
datasets/backgrounds/coco2017/
datasets/fruit_textures/final_fruits36065_original25_fruitseg30_10/
```

복사할 수 없으면 아래 순서대로 다시 만듭니다. 이 순서를 README에 남기는 이유는 `.gitignore`가 `datasets/`, `runs/`, `*.pt`를 기본적으로 제외하기 때문입니다. Git clone만으로는 대형 배경, 과일 texture pool, 학습 run이 복원되지 않습니다.

### 1. 기본 asset 생성

다면체 OBJ asset을 다시 만들려면:

```powershell
python scripts\make_polyhedron_objs.py --out assets/generated --size 0.08
```

또는:

```powershell
python main.py --out assets/generated --size 0.08
```

### 2. COCO 2017 배경 다운로드

기본 이미지 생성 배경은 `datasets/backgrounds/coco2017`입니다. `val2017`만 있어도 동작하지만, 최종 학습용은 `train2017`까지 받아 배경 pool을 크게 쓰는 편이 좋습니다.

COCO 2017 배경을 받으려면:

```powershell
python scripts\download_coco2017_backgrounds.py --split val2017 --output datasets\backgrounds\coco2017
python scripts\download_coco2017_backgrounds.py --split train2017 --output datasets\backgrounds\coco2017
```

COCO128 배경을 받으려면:

```powershell
python scripts\download_coco128_backgrounds.py --output datasets\backgrounds\coco128
```

### 3. Fruits-360 source texture 준비

Fruits-360 원본은 Kaggle `moltean/fruits` 데이터셋을 사용합니다. Kaggle CLI가 설정되어 있으면 아래처럼 받을 수 있습니다.

```powershell
mkdir datasets\fruit_textures\_raw\kaggle_fruits360_download
kaggle datasets download -d moltean/fruits -p datasets\fruit_textures\_raw\kaggle_fruits360_download --unzip
```

다운로드 후 class별 clean source를 만듭니다.

```powershell
python scripts\prepare_kaggle_fruits360_textures.py --clear
```

기본 입력 경로는 아래입니다.

```text
datasets/fruit_textures/_raw/kaggle_fruits360_download/fruits-360_100x100/fruits-360
```

### 4. Crawled/original source texture 준비

현재 실험과 가장 같은 결과를 원하면 기존 `datasets/fruit_textures/_sources/original_color_filtered_512/` 폴더를 백업에서 복사합니다. 이 폴더는 Git에 올리지 않습니다.

복사본이 없으면 Wikimedia 기반 후보로 대체 source를 만들 수 있습니다. 이 경우 웹 검색 결과가 바뀔 수 있으므로 완전히 같은 texture pool은 아닙니다.

```powershell
python scripts\download_generic_fruit_textures.py `
  --output datasets\fruit_textures\_sources\original_color_filtered_512 `
  --per_class 400 `
  --clear
```

대체 source를 만들었으면 반드시 contact sheet를 보고 잘못 들어온 과일이나 지나치게 헷갈리는 이미지를 제거합니다.

### 5. FruitSeg30 누끼 source 준비

FruitSeg30은 Mendeley public API에서 selected class만 내려받고 image/mask를 RGBA cutout으로 변환합니다. 현재 apple source는 `Apple_Gala`만 사용하고, `Apple_Golden Delicious`는 제외합니다.

```powershell
python scripts\prepare_fruitseg30_textures.py --clear --size 512
```

생성 결과:

```text
datasets/fruit_textures/_sources/fruitseg30_cutouts_512/
```

2026-05-20 기준 로컬 count:

```text
apple: 65
banana: 82
orange: 83
pineapple: 65
```

### 6. 최종 고정 texture pool 생성

생성기는 raw source를 매번 뒤지지 않고, 아래 final pool만 읽도록 운영합니다.

```powershell
python scripts\make_mixed_fruit_textures.py --clear
```

기본 출력:

```text
datasets/fruit_textures/final_fruits36065_original25_fruitseg30_10/
```

클래스별 구성:

```text
1040 Fruits-360 clean
400 crawled/original filtered
160 FruitSeg30 cutout
total 1600 textures per class
```

texture pool preview:

```powershell
python scripts\make_texture_contact_sheets.py `
  --root datasets\fruit_textures\final_fruits36065_original25_fruitseg30_10 `
  --count 40 `
  --thumb_size 104 `
  --columns 8
```

확인해야 할 preview:

```text
datasets/fruit_textures/final_fruits36065_original25_fruitseg30_10/_preview_apple.jpg
datasets/fruit_textures/final_fruits36065_original25_fruitseg30_10/_preview_banana.jpg
datasets/fruit_textures/final_fruits36065_original25_fruitseg30_10/_preview_orange.jpg
datasets/fruit_textures/final_fruits36065_original25_fruitseg30_10/_preview_pineapple.jpg
```

## 라벨 정책

현재 권장 생성 경로에서는 라벨과 visibility를 `--ideal_visibility`가 만든 ideal instance map 기준으로 계산합니다. 즉, 실제 color render 전에 단순 geometry projection으로 보이는 object/fruit face 영역을 결정하고, 통과한 배치만 최종 이미지로 렌더합니다. `--ideal_visibility`를 끄면 기존처럼 BlenderProc `render_segmap()` 결과를 사용할 수 있지만, 최종 학습 데이터 생성은 ideal gate 사용을 기준으로 합니다.

어느 경로든 라벨은 카메라에서 실제로 보이는 픽셀만 사용합니다. 뒤쪽에 가려진 object 영역은 라벨에 포함하지 않습니다.

이 판단은 매우 중요합니다. 합성 단계에서는 object의 원래 3D 모양과 위치를 알고 있지만, 학습 데이터로는 카메라에 실제로 보이는 부분만 정답이어야 합니다. 가려진 영역까지 라벨에 넣으면 모델은 이미지에 없는 물체 영역을 맞추도록 학습하게 됩니다.

같은 class의 object가 붙어 있어도 instance가 다르면 서로 다른 segmentation 라벨로 나뉩니다. `--seg_contour_mode largest` 기준에서는 object마다 가장 큰 외곽 contour 하나만 YOLO segment 라벨로 씁니다.

segmentation 라벨에도 occlusion 필터를 적용합니다. `--min_object_visible_ratio 0.10` 이하인 object는 없는 object처럼 취급해서 라벨과 metadata에서 제외합니다. 또한 기본 `--max_covered_ratio 0.80`이므로 visible object ratio가 `0.20`보다 낮은 후보 배치는 ideal gate에서 다시 샘플링됩니다. 아주 얇은 sliver 라벨을 학습시키지 않기 위한 기준입니다.

contour 단순화는 OpenCV `approxPolyDP`를 사용합니다.

```text
--seg_contour_epsilon_ratio 0.01
```

이는 면적 오차 1.0%를 보장한다는 뜻이 아니라, contour perimeter의 1.0% 정도를 경계 거리 오차로 허용한다는 뜻입니다. cube나 fruit cube의 bevel, pixel stair-step, distortion remap 때문에 생긴 자잘한 contour 점을 줄이기 위한 값입니다.

YOLO segment label은 polygon point가 너무 많아지면 파일 크기와 학습 안정성 측면에서 부담이 됩니다. 반대로 너무 단순화하면 cube bevel이나 과일 면의 경계가 무너집니다. `0.01`은 polygon point 수를 더 줄여 학습 라벨을 단순하게 유지하기 위한 기본값입니다.

## Visibility 지표

preview와 `_meta/train/*.json`에는 visibility 지표가 들어갑니다.

```text
obj_vis =
  ideal 또는 segmap 기준으로 보이는 object mask 픽셀 수
  / 화면 밖과 occlusion을 반영한 full projected object 면적

fruit_vis =
  ideal 또는 segmap 기준으로 보이는 fruit face mask 픽셀 수
  / 화면 밖과 occlusion을 반영한 full projected cube 면적
```

`obj_vis`는 object 자체가 얼마나 보이는지를 봅니다. `fruit_vis`는 fruit cube에서 과일 사진 면이 얼마나 보이는지를 봅니다.

비율만으로는 충분하지 않습니다. 이미지 전체에서 과일 사진 면이 너무 작으면 학습 신호가 약하기 때문에, 절대 픽셀 수 기준도 같이 사용합니다.

## Fruit 라벨 정책

fruit cube는 아래 기준을 통과하면 과일 class로 라벨링됩니다.

```text
fruit_vis >= 0.10
visible fruit face pixels >= 300
visible fruit face bbox width >= 10 px
visible fruit face bbox height >= 10 px
target visibility tier == actual visibility tier
```

기준을 통과하지 못한 fruit cube는 과일 class로 학습시키지 않고 `cube`로 fallback됩니다. preview에서는 아래처럼 표시됩니다.

```text
cube  from=banana  fallback
```

fallback의 의미는 “처음에는 banana texture가 붙은 cube였지만, 최종 렌더에서 banana 사진 면이 충분히 보이지 않아 banana 라벨 대신 cube 라벨로 쓴다”입니다.

이 정책이 필요한 이유는 과일 texture가 거의 보이지 않는 장면을 과일 class로 학습시키면 모델이 흰 cube의 모양이나 배경 힌트만 보고 과일 class를 예측하게 될 수 있기 때문입니다. 그런 샘플은 과일 분류 학습에는 해롭지만, cube 검출 학습에는 여전히 유용하므로 버리지 않고 `cube`로 돌립니다.

`_meta/train/*.json`에는 `source_class`, `fallback_class`, `fruit_visibility_pass`, `visible_face_pixels`, `visible_face_bbox`, `visible_face_segments` 같은 정보가 남습니다. 그래서 preview에서 fallback 이유를 눈으로 확인할 수 있고, 나중에 threshold를 바꿀 때도 원인을 추적할 수 있습니다.

## 생성 정책

라벨 단계에서 너무 많은 object를 버리는 것보다, 생성 단계에서 애초에 너무 작거나 잘리는 후보를 피하는 방향으로 설계했습니다.

현재 object transform 샘플링은 다음 조건을 우선 만족하려고 합니다.

```text
projected object area >= 900 px
fruit estimated face pixels >= 300 px
object projection stays inside frame margin
fruit visibility tier target is satisfied when possible
```

`fruit estimated face pixels`는 아래처럼 계산합니다.

```text
projected_object_area * fruit_photo_cube_ratio
```

즉 fruit cube 전체가 커도 과일 사진 면이 너무 작게 보이면 좋은 후보로 보지 않습니다.

negative sample은 흰색 cylinder/sphere distractor와 COCO 배경으로 구성됩니다. 이 물체들은 `category_id=0`인 background로 취급되어 라벨에 들어가지 않습니다. 목적은 모델이 “흰색 물체가 보이면 무조건 cube/polyhedron”이라고 외우는 것을 막는 것입니다.

negative는 `--negative_ratio`와 seed/image_id로 결정됩니다. 같은 seed와 image id를 쓰면 negative 여부가 재현되므로, 중간에 끊겼다가 `--resume`으로 이어도 데이터셋 성격이 크게 흔들리지 않습니다.

## Frame 및 distortion 정책

이전에는 lens distortion 후 이미지 바깥 무효 영역을 crop하고 다시 640x640으로 키우는 방식이 있었습니다. 현재 코드는 그 방식 대신, distortion map이 원본 640x640 바깥을 참조하지 않는 경우에만 distortion을 적용합니다.

즉 distortion 강도 범위는 유지하되, 결과 픽셀이 원본 이미지 밖을 참조하는 distortion 파라미터는 버립니다. 안전한 파라미터를 찾지 못하면 해당 이미지에는 distortion을 적용하지 않습니다.

object 자체가 프레임 밖으로 걸리는 문제도 따로 막습니다. object transform 샘플링 시 projected vertices가 화면 안쪽 margin에 들어오는 후보를 고릅니다. 현재 margin은 `0.08`입니다.

이 margin은 edge-touch label을 줄이기 위한 장치입니다. 물체가 가장자리에서 잘리면 segmentation 경계가 이미지 테두리와 붙고, 실제 object 크기와 라벨 크기 해석이 애매해집니다. 최근 품질 체크에서 `edge-touch labels = 0`을 목표로 둔 이유도 여기에 있습니다.

## Metadata 설계

각 이미지마다 `_meta/train/{image_id}.json`을 저장합니다. 이 metadata는 단순 로그가 아니라 품질 기준을 자동으로 검증하기 위한 설계 산출물입니다.

주요 필드:

```text
image_id
negative
label_format
fruit_visibility_tier
label_objects
fruit_objects
```

`label_objects`에는 실제 YOLO 라벨로 저장된 object의 class, visible pixel 수, projected area, object visible ratio, fallback 정보가 들어갑니다. `fruit_objects`에는 과일 texture 면이 실제로 얼마나 보였는지, pass/fallback 판단이 왜 내려졌는지 들어갑니다.

이 metadata 덕분에 preview만 보는 수동 검수와, threshold 위반 샘플을 찾는 자동 검수를 같이 할 수 있습니다. README의 품질 체크리스트가 단순 권장사항이 아니라 `_meta`를 기반으로 확인 가능한 기준이 되도록 설계했습니다.

## 권장 기본값

현재 추천 기본값은 코드와 맞춰 아래와 같습니다.

| option | value | 이유 |
|---|---:|---|
| `--label_format` | `segment` | 현재 주력은 YOLO segmentation |
| `--seg_contour_mode` | `largest` | instance별 주 외곽만 사용 |
| `--seg_contour_epsilon_ratio` | `0.01` | bevel/픽셀 계단 contour 완화 |
| `--min_projected_area` | `900` | 너무 작은 object 생성 감소 |
| `--min_fruit_face_pixels` | `1800` | 과일 사진 면의 절대 정보량 보장 |
| `--single_object_scale_min` | `0.82` | object가 1개뿐일 때 너무 작은 fruit cube가 일반 cube처럼 보이는 문제 방지 |
| `--single_object_min_projected_area` | `2600` | 단일 object scene의 화면 내 최소 크기 상향 |
| `--single_object_min_fruit_face_pixels` | `2200` | 단일 fruit cube의 과일 사진 면 정보량 보장 |
| `--min_fruit_visible_ratio` | `0.22` | fruit label을 남길 때 필요한 과일 면 비율 |
| `--hard_min_fruit_visible_ratio` | `0.10` | hard tier에서만 허용하는 낮은 fruit visibility 하한 |
| `--hard_min_fruit_face_pixels` | `900` | hard tier에서만 허용하는 낮은 fruit face pixel 하한 |
| `--min_fruit_face_side` | `36` | 얇은 fruit face 라벨 방지 |
| `--min_object_visible_ratio` | `0.10` | obj_vis 0.10 이하 object는 없는 object로 취급 |
| `--max_covered_ratio` | `0.80` | ideal gate에서 과한 occlusion 배치 재샘플링 |
| `--negative_ratio` | `0.18` | 흰색/배경 false positive 방지 |
| `--samples` | `16` | 속도와 품질 균형 |
| `--min_objects` | `1` | empty positive scene 방지 |
| `--max_objects` | `4` | 과한 occlusion과 메모리 사용 방지 |
| `--scale_min` | `0.40` | 작은 object 다양성 유지 |
| `--scale_max` | `2.45` | 큰 object 다양성 유지 |
| `--lighting_mode` | `soft_overhead` | 한쪽 강한 광원보다 overhead soft + fill 중심 |
| `--arena_background_ratio` | `0.05` | 대부분 COCO 배경, 경기장형 배경은 5%만 섞음 |
| `--fruit_texture_dir` | `datasets/fruit_textures/final_fruits36065_original25_fruitseg30_10` | Fruits-360 65%, crawled/original 25%, FruitSeg30 10% 고정 pool |
| `--fruit_texture_aug` | `light` | class 색 의미를 보존하면서 face별 회전/확대/축소 |
| `--fruit_texture_layout` | `mixed` | single face texture와 same-class collage texture를 섞음 |
| `--fruit_texture_collage_prob` | `0.35` | `mixed` layout에서 face별 collage 사용 확률 |
| `--allow_mixed_fruit_classes_per_image` | enabled | 한 이미지 안에 서로 다른 fruit cube class 허용 |
| `--allow_multiple_fruit_textures_per_cube` | enabled | 한 cube의 3개 fruit face에 같은 class 내 다른 source texture 허용 |
| `--train_low_aug` | enabled | YOLO 학습 augmentation이 fruit class 색을 다시 흔드는 것을 줄임 |
| `--ideal_visibility` | enabled | render 전에 ideal geometry로 배치 검증 |
| `--ideal_visibility_max_attempts` | `80` | 좋은 배치를 찾는 최대 시도 횟수 |

scale은 더 이상 균등분포로 뽑지 않습니다. 작은 object set은 유지하되 과하게 자주 나오지 않도록 triangular distribution을 사용합니다. 여러 object가 있는 scene에서는 작은 object를 허용하고, object가 1개뿐인 scene에서는 scale 하한과 projected area 하한을 올려 fruit cube와 plain cube가 너무 쉽게 헷갈리는 샘플을 줄입니다.

과일 class는 조금 더 자주 나오도록 가중치가 있습니다.

```text
shape classes: 1.00
banana:        1.55
orange:        1.35
pineapple:     1.55
apple:         1.50
```

fruit visibility tier는 아래 비율로 섞습니다.

```text
easy 50%: fruit_photo_cube_ratio 0.50 이상
mid  45%: fruit_photo_cube_ratio 0.20~0.50
hard  5%: fruit_photo_cube_ratio 0.10~0.20
```

## 100장 품질 검증 생성

새 설정을 검증할 때는 100장을 새 폴더에 생성한 뒤 preview와 자동 지표를 함께 확인합니다.

```powershell
python scripts\run_yolo_parallel.py `
  --num_images 100 `
  --workers 8 `
  --cpu_threads 1 `
  --output datasets\yolo_8class_seg_quality100 `
  --width 640 `
  --height 640 `
  --samples 16 `
  --min_objects 1 `
  --max_objects 4 `
  --scale_min 0.40 `
  --scale_max 2.45 `
  --negative_ratio 0.18 `
  --fruit_texture_dir datasets\fruit_textures\final_fruits36065_original25_fruitseg30_10 `
  --background_dir datasets\backgrounds\coco2017 `
  --arena_background_ratio 0.05 `
  --label_format segment `
  --seg_contour_mode largest `
  --seg_contour_epsilon_ratio 0.01 `
  --min_object_visible_ratio 0.10 `
  --min_fruit_visible_ratio 0.22 `
  --min_fruit_face_pixels 1800 `
  --single_object_scale_min 0.82 `
  --single_object_min_projected_area 2600 `
  --single_object_min_fruit_face_pixels 2200 `
  --min_fruit_face_side 36 `
  --fruit_visibility_easy_weight 0.50 `
  --fruit_visibility_mid_weight 0.45 `
  --fruit_visibility_hard_weight 0.05 `
  --hard_min_fruit_visible_ratio 0.10 `
  --hard_min_fruit_face_pixels 900 `
  --lighting_mode soft_overhead `
  --fruit_texture_aug light `
  --fruit_texture_layout mixed `
  --fruit_texture_collage_prob 0.35 `
  --allow_mixed_fruit_classes_per_image `
  --allow_multiple_fruit_textures_per_cube `
  --canonical_fruit_textures `
  --ideal_visibility `
  --ideal_visibility_debug `
  --ideal_visibility_max_attempts 80
```

중간에 끊겼으면 같은 명령에 `--resume`을 추가합니다.

출력 구조는 아래와 같습니다.

```text
datasets/yolo_8class_seg_quality100/
  data.yaml
  images/train/*.jpg
  labels/train/*.txt
  _meta/train/*.json
  ideal_visibility_debug/train/*.jpg
  _chunks/logs/*.log
```

`--ideal_visibility_debug`는 검은 배경 위에 ideal geometry 기준 `full silhouettes`, `visible objects`, `visible fruit faces`를 저장합니다. 최종 합성 사진 위에서 라벨을 보는 preview는 아래 `make_yolo_bbox_preview.py --debug_visibility`가 담당합니다.

## Preview 생성

일반 segmentation preview:

```powershell
python scripts\make_yolo_bbox_preview.py `
  --dataset datasets\yolo_8class_seg_quality100 `
  --count 100
```

visibility debug preview:

```powershell
python scripts\make_yolo_bbox_preview.py `
  --dataset datasets\yolo_8class_seg_quality100 `
  --count 100 `
  --debug_visibility
```

debug preview 결과:

```text
datasets/yolo_8class_seg_quality100/visibility_preview/train/
```

caption panel에는 아래 정보가 표시됩니다.

```text
#번호  class  tier=easy/mid/hard  fruit_vis=...  obj_vis=...  OK/fallback
```

과일 face highlight는 training 이미지에는 들어가지 않고 preview에서만 표시됩니다.

preview는 이 프로젝트의 핵심 검증 도구입니다. synthetic data는 숫자 지표만 보면 좋아 보여도, 실제로는 과일 면이 너무 작거나, segmentation contour가 깨지거나, background와 object 경계가 이상할 수 있습니다. 그래서 새 설정은 바로 대량 생성하지 않고 100장 preview로 먼저 봅니다.

`--debug_visibility`는 과일 면을 시각적으로 highlight해서 “왜 과일로 남았는지 / 왜 cube fallback이 됐는지”를 빠르게 확인하기 위한 모드입니다. 이 highlight는 preview 이미지에만 들어가며 학습 이미지에는 절대 들어가지 않습니다.

## 품질 체크리스트

100장 생성 후 아래 항목을 확인합니다.

```text
images / labels / metas 개수 일치
ideal visibility debug 100장 생성
negative_ratio와 empty label 개수 대략 일치
edge-touch labels = 0
projected object area < 900 라벨 = 0
fruit label visible_face_pixels < 1800 = 0, except hard tier floor 900
fruit label fruit_vis < 0.22 = 0, except hard tier floor 0.10
fruit label target tier != actual tier = 0
obj_vis <= 0.10 라벨 = 0
worker err log에 Traceback/Error/Exception 없음
visibility preview 100장 생성
```

최근 검증 세트 `datasets/ideal_gate_100_test` 기준 결과:

```text
images / labels / metas: 100 / 100 / 100
ideal debug / preview debug: 100 / 100
empty non-negative labels: 0
label/meta mismatch: 0
obj_vis <= 0.10 labels: 0
fruit tier mismatch labels: 0
worker errors: 0
worker 1, 320x320 smoke timing:
  total 173.63s
  color render total 141.70s
  ideal geometry + debug + Python overhead 31.93s
  non-render overhead about 0.32s/image
```

품질 검증은 “완벽한 class 균등 분포”보다 “라벨 오류를 만들지 않는 것”을 우선합니다. class 분포는 100장 단위에서는 흔들릴 수 있지만, edge-touch, 너무 작은 라벨, 과일 면 부족, 과한 occlusion 같은 오류는 대량 생성 전에 반드시 잡아야 합니다.

## 병렬 생성 및 재시작 설계

`scripts/run_yolo_parallel.py`는 image id를 worker 수만큼 round-robin chunk로 나눕니다. 각 worker는 `_chunks/worker_###.txt`에 자기 id 목록을 받고, stdout/stderr는 `_chunks/logs/worker_###.*.log`에 따로 남깁니다.

이 구조를 둔 이유는 대량 생성 중 일부 worker가 실패했을 때 원인을 분리해서 보기 쉽고, 이미 생성된 이미지를 유지한 채 이어서 돌리기 쉽기 때문입니다.

`--resume`은 이미지와 라벨 파일이 모두 존재하는 id를 완료로 봅니다.

```text
images/train/000123.jpg exists
labels/train/000123.txt exists
=> image_id 123은 skip
```

metadata만 있고 이미지/라벨이 없거나, 이미지 하나만 있는 경우는 완료로 보지 않습니다. 생성 데이터셋의 최소 단위는 image와 label의 쌍이기 때문입니다.

진행률은 output 폴더의 image 개수를 기준으로 계산합니다. 이메일 알림은 생성 실패를 막지 않도록 non-blocking으로 처리합니다. SMTP 문제가 생겨도 데이터 생성 자체는 계속 진행하는 설계입니다.

## 50,000장 생성

최종 dataset은 50,000장 목표로 생성합니다. GPU/CPU 안정성이 우선이면 `--workers 10`부터 시작하고, 시스템이 안정적이면 16까지 올릴 수 있습니다.

```powershell
python scripts\run_yolo_parallel.py `
  --num_images 50000 `
  --workers 16 `
  --cpu_threads 1 `
  --output datasets\yolo_8class_seg_v5_50000 `
  --samples 16 `
  --min_objects 1 `
  --max_objects 4 `
  --scale_min 0.40 `
  --scale_max 2.45 `
  --negative_ratio 0.18 `
  --fruit_texture_dir datasets\fruit_textures\final_fruits36065_original25_fruitseg30_10 `
  --background_dir datasets\backgrounds\coco2017 `
  --arena_background_ratio 0.05 `
  --label_format segment `
  --seg_contour_mode largest `
  --seg_contour_epsilon_ratio 0.01 `
  --min_object_visible_ratio 0.10 `
  --min_fruit_visible_ratio 0.22 `
  --min_fruit_face_pixels 1800 `
  --single_object_min_projected_area 2600 `
  --single_object_min_fruit_face_pixels 2200 `
  --min_fruit_face_side 36 `
  --fruit_visibility_easy_weight 0.50 `
  --fruit_visibility_mid_weight 0.45 `
  --fruit_visibility_hard_weight 0.05 `
  --hard_min_fruit_visible_ratio 0.10 `
  --hard_min_fruit_face_pixels 900 `
  --lighting_mode soft_overhead `
  --fruit_texture_aug light `
  --fruit_texture_layout mixed `
  --fruit_texture_collage_prob 0.35 `
  --allow_mixed_fruit_classes_per_image `
  --allow_multiple_fruit_textures_per_cube `
  --canonical_fruit_textures `
  --ideal_visibility `
  --ideal_visibility_max_attempts 80 `
  --email_to jaeyoungi@snu.ac.kr `
  --email_every 2000
```

재시작:

```powershell
python scripts\run_yolo_parallel.py `
  --num_images 50000 `
  --workers 16 `
  --cpu_threads 1 `
  --output datasets\yolo_8class_seg_v5_50000 `
  --samples 16 `
  --min_objects 1 `
  --max_objects 4 `
  --scale_min 0.40 `
  --scale_max 2.45 `
  --negative_ratio 0.18 `
  --fruit_texture_dir datasets\fruit_textures\final_fruits36065_original25_fruitseg30_10 `
  --background_dir datasets\backgrounds\coco2017 `
  --arena_background_ratio 0.05 `
  --label_format segment `
  --seg_contour_mode largest `
  --seg_contour_epsilon_ratio 0.01 `
  --min_object_visible_ratio 0.10 `
  --min_fruit_visible_ratio 0.22 `
  --min_fruit_face_pixels 1800 `
  --single_object_min_projected_area 2600 `
  --single_object_min_fruit_face_pixels 2200 `
  --min_fruit_face_side 36 `
  --fruit_visibility_easy_weight 0.50 `
  --fruit_visibility_mid_weight 0.45 `
  --fruit_visibility_hard_weight 0.05 `
  --hard_min_fruit_visible_ratio 0.10 `
  --hard_min_fruit_face_pixels 900 `
  --lighting_mode soft_overhead `
  --fruit_texture_aug light `
  --fruit_texture_layout mixed `
  --fruit_texture_collage_prob 0.35 `
  --allow_mixed_fruit_classes_per_image `
  --allow_multiple_fruit_textures_per_cube `
  --canonical_fruit_textures `
  --ideal_visibility `
  --ideal_visibility_max_attempts 80 `
  --email_to jaeyoungi@snu.ac.kr `
  --email_every 2000 `
  --resume
```

50,000장 생성 후 preview:

```powershell
python scripts\make_yolo_bbox_preview.py `
  --dataset datasets\yolo_8class_seg_v5_50000 `
  --count 100 `
  --debug_visibility
```

## YOLO segmentation 학습

segmentation dataset은 `yolo segment train`으로 학습합니다.

```powershell
yolo segment train `
  model=yolo26n-seg.pt `
  data=(프로젝트 경로)/datasets/yolo26_seg_5000_best_fruit_pool_v1/data.yaml `
  epochs=150 `
  imgsz=640 `
  batch=16 `
  device=0 `
  workers=4 `
  cache=disk `
  amp=False `
  deterministic=False `
  patience=30 `
  project=runs/yolo26_seg_train `
  name=yolo26n_seg_5000_best_fruit_pool_v1 `
  hsv_h=0.003 `
  hsv_s=0.15 `
  hsv_v=0.12 `
  mosaic=0 `
  close_mosaic=0 `
  erasing=0 `
  auto_augment=None `
  exist_ok=True
```

보통은 이 standalone 학습 명령보다 `scripts/run_yolo26_seg_pipeline.py`를 쓰는 편이 낫습니다. pipeline은 split, audit, preview, train, predict, side-by-side까지 같이 처리합니다.

학습 설계상 synthetic validation만으로 최종 성능을 판단하지 않습니다. synthetic validation은 생성 파이프라인이 망가지지 않았는지 보는 용도에 가깝고, 최종 판단은 실제 카메라 이미지 100~300장으로 별도 검증하는 것이 좋습니다. 합성 데이터와 실제 촬영 데이터 사이에는 조명, 인쇄 품질, 렌즈, 배경, 손떨림, object scale의 domain gap이 남기 때문입니다.

## 이메일 알림

이메일 설정은 `config/email_smtp.json` 또는 환경 변수로 제공합니다. 이 파일은 비밀번호가 들어갈 수 있으므로 Git에 올리지 않습니다.

예시:

```json
{
  "host": "smtp.gmail.com",
  "port": 587,
  "user": "your_email@gmail.com",
  "password": "app_password",
  "from": "your_email@gmail.com",
  "tls": true
}
```

이메일 테스트:

```powershell
python scripts\send_progress_email.py `
  --to jaeyoungi@snu.ac.kr `
  --subject "Data Generation Blender test" `
  --body "SMTP test"
```

생성 중 이메일 옵션:

```text
--email_to jaeyoungi@snu.ac.kr
--email_every 2000
```

메일 전송에 실패해도 데이터 생성 자체는 계속 진행되도록 처리되어 있습니다.

## 진행 상황 확인

생성 개수 확인:

```powershell
$ds='datasets\yolo_8class_seg_v5_50000'; `
$img=(Get-ChildItem "$ds\images\train" -Filter *.jpg).Count; `
$lbl=(Get-ChildItem "$ds\labels\train" -Filter *.txt).Count; `
"images=$img labels=$lbl"
```

관련 프로세스 확인:

```powershell
Get-CimInstance Win32_Process |
  Where-Object {
    $_.Name -in @('python.exe','blenderproc.exe','blender.exe') -and
    $_.CommandLine -match 'run_yolo_parallel|generate_yolo_coco_composite|blenderproc'
  } |
  Select-Object ProcessId,Name,CommandLine
```

GPU 상태:

```powershell
nvidia-smi
```

## Git 관리 기준

Git에 올리는 파일:

```text
source code
README
environment.yml
config/email_smtp.example.json
assets/generated/*.obj
필요한 작은 dataset metadata
필요한 training metadata
최종 checkpoint 일부
```

Git에 올리지 않는 파일:

```text
datasets/ 아래 대형 image/label 산출물
runs/ 아래 대형 학습 산출물
config/email_smtp.json
다운로드 cache
루트의 YOLO 기본 모델 파일 (*.pt)
로컬 scratch note
```

현재 `.gitignore`는 `datasets/`, `runs/`, `*.pt`를 기본적으로 무시합니다. 이미 추적 중인 복구 메타데이터와 checkpoint는 예외적으로 Git에 남아 있습니다.

Git 설계 기준은 “대형 산출물을 저장소에 넣지 않고도 같은 실험을 다시 만들 수 있어야 한다”입니다. 그래서 source code, environment, 작은 metadata, checkpoint hash, 실행 명령, quality checklist를 남기고, 수 GB 규모의 image dataset은 로컬 재생성을 전제로 둡니다.

README는 현재 프로젝트의 단일 기준 문서입니다. 새로운 threshold나 pipeline 판단을 바꾸면 코드만 바꾸지 말고, 왜 바꿨는지 이 문서의 설계 섹션에도 함께 남깁니다.

## Arena SUN-111/SUN-168 Booster Dataset

Use this booster dataset for arena-specific fine-tuning. It is not a replacement for the existing 100k synthetic dataset. It is intended for fine-tuning from an existing best checkpoint, or for mixed fine-tuning with oversampling so the arena signal is not washed out.

Arena assumptions:

```text
floor: SUN-111 beige/yellow wood family with subtle procedural plywood grain
wall/fence: SUN-168 matte beige paint family
camera: low robot-camera view with slight downward pitch
background: 100% synthetic arena floor/wall, no COCO background
fruit sampling inside booster only: apple 30%, orange 30%, banana 20%, pineapple 20%
scene profiles: normal / SUN color shift / motion blur / close-up / plain cube hard negative
```

Recommended 3,000-image booster generation:

```powershell
python scripts\run_yolo_parallel.py `
  --output datasets\arena_sun111_sun168_booster_v1 `
  --num_images 3000 `
  --workers 4 `
  --worker_start_delay 2 `
  --width 640 `
  --height 640 `
  --samples 24 `
  --cpu_threads 1 `
  --label_format segment `
  --min_objects 1 `
  --max_objects 4 `
  --scale_min 0.45 `
  --scale_max 2.20 `
  --negative_ratio 0 `
  --arena_booster_mode `
  --arena_floor_material sun111_wood `
  --arena_wall_material sun168_beige `
  --arena_full_background_ratio 1.0 `
  --robot_camera_view `
  --wall_contact_ratio 0.25 `
  --corner_scene_ratio 0.10 `
  --motion_blur_hard_negative_ratio 0.10 `
  --plain_cube_hard_negative_ratio 0.05 `
  --apple_orange_boost `
  --fruit_texture_aug light `
  --fruit_texture_layout mixed `
  --fruit_texture_collage_prob 0.25 `
  --ideal_visibility `
  --ideal_visibility_debug `
  --ideal_visibility_max_attempts 80 `
  --resume
```

Preview sheets:

```powershell
python scripts\make_arena_booster_preview_sheets.py `
  --dataset datasets\arena_sun111_sun168_booster_v1 `
  --count 40
```

Mixed fine-tuning dataset example:

```powershell
python scripts\make_mixed_finetune_dataset.py `
  --base_dataset datasets\yolo26_seg_100000_ideal_debug `
  --booster_dataset datasets\arena_sun111_sun168_booster_v1 `
  --output datasets\mixed_ideal12000_arena3000_finetune_v1 `
  --base_count 12000 `
  --booster_count 3000 `
  --copy_mode copy `
  --reset
```

This creates a 15,000-image fine-tune dataset with a 20% booster ratio, which is much stronger than simply concatenating 100k + 3k.

## YOLO26n segmentation end-to-end pipeline

범용 파이프라인은 `scripts/run_yolo26_seg_pipeline.py`입니다. 이 스크립트는 한 번에 다음을 수행합니다.

1. ideal visibility gate와 ideal debug를 켠 synthetic segmentation data 생성
2. `train/val/test` split 생성 및 `data.yaml` 작성
3. fruit cube texture/class 섞임 audit 실행
4. split별 `visibility_preview` debug 이미지 생성
5. `yolo segment train` 실행 (`yolo26n-seg.pt` 기본 weight 사용)
6. test split prediction 실행
7. `visibility_preview/test`와 prediction을 붙인 side-by-side 이미지 생성

1000장 기본 실험:

```powershell
conda run -n ai_robotics python scripts\run_yolo26_seg_pipeline.py `
  --dataset datasets\yolo26_seg_1000_ideal_debug `
  --total 1000 `
  --gen_workers 4 `
  --email_to jaeyoungi@snu.ac.kr `
  --email_every 1000 `
  --epochs 100 `
  --model yolo26n-seg.pt `
  --batch 16 `
  --device 0
```

같은 1,000장 실험은 PowerShell wrapper로도 실행할 수 있습니다. 이 wrapper는 repo-relative 경로를 사용하므로 저장소 위치가 바뀌어도 그대로 동작합니다.

```powershell
cd (프로젝트 경로)
powershell -ExecutionPolicy Bypass -File scripts\run_yolo26_seg_1000_pipeline.ps1
```

현재 wrapper 기준값은 `train_count=800`, `val_count=100`, `test_count=100`, `gen_workers=12`, `email_every=200`, `batch=32`입니다. 위의 직접 실행 예시는 필요한 값만 보여주는 짧은 형태이므로, 재현성을 우선하면 wrapper 또는 wrapper와 같은 인자를 직접 지정하는 방식을 사용합니다.

50,000장 전체 실험용 PowerShell wrapper도 있습니다. 다만 최종 권장 설정은 계속 바뀌었기 때문에, 재현성을 우선하면 아래 Python 명령처럼 모든 핵심 인자를 명시하는 방식을 권장합니다. wrapper를 사용할 때도 이 섹션의 인자와 맞는지 먼저 확인합니다.

```powershell
cd (프로젝트 경로)
powershell -ExecutionPolicy Bypass -File scripts\run_yolo26_seg_50000_pipeline.ps1
```

현재 50,000장 wrapper 기준값은 `train_count=40000`, `val_count=5000`, `test_count=5000`, `gen_workers=9`, `worker_start_delay=2`, `resume_generate`, `batch=32`, `epochs=200`입니다. 장시간 실행 중 중단되면 같은 wrapper를 다시 실행해도 `--resume_generate` 기준으로 이미 생성된 image/label pair를 건너뜁니다.

현재 best 설정을 50,000장으로 직접 쓰면 아래와 같습니다.

```powershell
conda run -n ai_robotics python scripts\run_yolo26_seg_pipeline.py `
  --dataset datasets\yolo26_seg_50000_ideal_debug `
  --total 50000 `
  --train_count 40000 `
  --val_count 5000 `
  --test_count 5000 `
  --gen_workers 9 `
  --worker_start_delay 2 `
  --resume_generate `
  --fruit_texture_dir datasets\fruit_textures\final_fruits36065_original25_fruitseg30_10 `
  --background_dir datasets\backgrounds\coco2017 `
  --arena_background_ratio 0.05 `
  --fruit_texture_aug light `
  --fruit_texture_layout mixed `
  --fruit_texture_collage_prob 0.35 `
  --allow_mixed_fruit_classes_per_image `
  --allow_multiple_fruit_textures_per_cube `
  --canonical_fruit_textures `
  --lighting_mode soft_overhead `
  --fruit_visibility_easy_weight 0.50 `
  --fruit_visibility_mid_weight 0.45 `
  --fruit_visibility_hard_weight 0.05 `
  --fruit_class_weight_scale 1.20 `
  --min_fruit_visible_ratio 0.22 `
  --min_fruit_face_pixels 1800 `
  --single_object_min_projected_area 2600 `
  --single_object_min_fruit_face_pixels 2200 `
  --min_fruit_face_side 36 `
  --hard_min_fruit_visible_ratio 0.10 `
  --hard_min_fruit_face_pixels 900 `
  --train_low_aug `
  --preview_count 200 `
  --email_to jaeyoungi@snu.ac.kr `
  --email_every 1000 `
  --epochs 200 `
  --model yolo26n-seg.pt `
  --batch 32 `
  --device 0 `
  --name yolo26n_seg_50000_ideal_debug
```

## 현재 권장 5,000장 생성 + 학습 명령

현재까지의 실험을 모두 고려한 5,000장 권장 명령입니다. 새 texture pool, COCO 2017 배경, arena 5%, soft overhead lighting, light texture augmentation, mixed collage, hard 5%, low train augmentation, YOLO26n-seg를 한 번에 사용합니다.

```powershell
conda run -n ai_robotics python scripts\run_yolo26_seg_pipeline.py `
  --dataset datasets\yolo26_seg_5000_best_fruit_pool_v1 `
  --total 5000 `
  --train_count 4000 `
  --val_count 500 `
  --test_count 500 `
  --gen_workers 8 `
  --worker_start_delay 2 `
  --resume_generate `
  --fruit_texture_dir datasets\fruit_textures\final_fruits36065_original25_fruitseg30_10 `
  --background_dir datasets\backgrounds\coco2017 `
  --arena_background_ratio 0.05 `
  --fruit_texture_aug light `
  --fruit_texture_layout mixed `
  --fruit_texture_collage_prob 0.35 `
  --allow_mixed_fruit_classes_per_image `
  --allow_multiple_fruit_textures_per_cube `
  --canonical_fruit_textures `
  --lighting_mode soft_overhead `
  --fruit_visibility_easy_weight 0.50 `
  --fruit_visibility_mid_weight 0.45 `
  --fruit_visibility_hard_weight 0.05 `
  --fruit_class_weight_scale 1.20 `
  --min_fruit_visible_ratio 0.22 `
  --min_fruit_face_pixels 1800 `
  --single_object_min_projected_area 2600 `
  --single_object_min_fruit_face_pixels 2200 `
  --min_fruit_face_side 36 `
  --hard_min_fruit_visible_ratio 0.10 `
  --hard_min_fruit_face_pixels 900 `
  --train_low_aug `
  --preview_count 200 `
  --preview_sheet_count 100 `
  --epochs 150 `
  --model yolo26n-seg.pt `
  --batch 16 `
  --device 0 `
  --train_workers 4 `
  --name yolo26n_seg_5000_best_fruit_pool_v1 `
  --email_to jaeyoungi@snu.ac.kr `
  --email_every 500
```

`--gen_workers 8`은 안정성과 속도의 균형값입니다. PC가 안정적이면 10~12까지 올릴 수 있고, Blender worker가 죽으면 4~6으로 낮춥니다. 중단되면 같은 명령을 다시 실행하면 `--resume_generate` 기준으로 이미 생성된 image/label pair를 건너뜁니다.

주요 출력:

```text
datasets/<name>/images/{train,val,test}/
datasets/<name>/labels/{train,val,test}/
datasets/<name>/_meta/{train,val,test}/
datasets/<name>/ideal_visibility_debug/{train,val,test}/
datasets/<name>/visibility_preview/{train,val,test}/
datasets/<name>/predictions/test/
datasets/<name>/side_by_side/test/
runs/yolo26_seg_train/<run_name>/weights/best.pt
```

중간 단계부터 재개할 때는 `--skip_generate`, `--skip_split`, `--skip_preview`, `--skip_train`, `--skip_predict`, `--skip_side_by_side`를 조합해서 사용합니다.

예를 들어 이미 dataset과 split이 만들어져 있고 학습부터 다시 시작하려면 아래처럼 앞 단계를 건너뜁니다.

```powershell
conda run -n ai_robotics python scripts\run_yolo26_seg_pipeline.py `
  --dataset datasets\yolo26_seg_50000_ideal_debug `
  --total 50000 `
  --train_count 40000 `
  --val_count 5000 `
  --test_count 5000 `
  --skip_generate `
  --skip_split `
  --skip_preview `
  --model yolo26n-seg.pt `
  --epochs 200 `
  --batch 32 `
  --device 0 `
  --name yolo26n_seg_50000_ideal_debug
```

이메일 알림은 생성 단계에서 `--email_to`가 있을 때만 켜집니다. 시작 시점에 한 번, `--email_every 1000` 기준으로 생성 이미지가 1000장 늘어날 때마다 한 번, 완료/오류 시점에 한 번 진행 상황을 보냅니다.

