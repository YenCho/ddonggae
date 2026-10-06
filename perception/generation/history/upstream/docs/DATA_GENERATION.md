# 데이터 생성 로직 (완전판)

기준일: 2026-07-08

이 문서는 **"학습 데이터를 왜/어떻게 이렇게 만들었나"** 를 설명하는 현행(living) 기준 문서입니다. 2026-07-02 문서 재작성 때 삭제됐던 데이터 생성 로직 전체를 [docs/history/readme_specific_full_20260702.md](./history/readme_specific_full_20260702.md)(삭제 직전 완전판, 105KB)에서 복원해 현재 기준으로 재정리했습니다. 아카이브는 원문 그대로 보존되고, 이 문서가 앞으로 유지보수되는 기준입니다.

- 프로젝트 개요/런타임 구조: [readme.md](../readme.md)
- 실험 상태 색인: [EXPERIMENTS.md](../EXPERIMENTS.md) / 모델 계보 서사: [HISTORY.md](../HISTORY.md)
- 학습/배포 기술 reference: [readme_specific.md](../readme_specific.md)

표기 원칙: 여기 나오는 CLI 옵션은 2026-07-08 기준 `scripts/run_yolo_parallel.py`, `scripts/generate_yolo_coco_composite.py`에 실제로 존재하는지 확인했습니다. 옵션은 유지되지만 예시 명령의 출력 경로/클래스 구성이 옛 세대(8-class)인 경우 **"구버전 기준"** 으로 표시하고 삭제하지 않습니다.

## 1. 철학과 원칙

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

설계의 큰 방향은 **"이미지는 다양하고 지저분하게, 라벨은 보수적이고 깨끗하게"** 입니다. 카메라 노이즈, 조명, blur, JPEG artifact, 배경 변화는 일부러 강하게 넣지만, 학습 target은 ideal geometry visibility map 또는 BlenderProc `instance_segmap` 기반으로 계산해서 사람이 그린 bbox보다 일관되게 만듭니다. 최종 학습용 생성은 ideal geometry visibility gate를 우선 사용합니다.

가장 중요한 운영 규칙(visibility 철학 원문):

```text
숨겨진 과일은 분류하지 않는다.
cube-like object는 plain cube 확정이 아니다.
과일 cube는 Task 1에서도 cube처럼 보일 수 있다.
Task 1 cube pickup은 한 view만 보고 확정하지 않는다.
보이는 fruit face crop만 fruit class evidence로 사용한다.
```

즉 **보이는 픽셀만 라벨링하고, fruit face가 보이지 않으면 fruit로 학습시키지 않습니다.** 이 원칙은 데이터 생성(fruit 라벨 fallback)과 런타임 판단(`cube_like_unresolved`) 양쪽에 똑같이 적용됩니다.

### 데이터 설계 판단

과일은 실제 3D mesh 대신 cube 위의 texture로 구현했습니다. 대회에서 실제 물체가 "과일 사진이 붙은 흰색 cube"에 가깝고, 모델이 과일의 3D 형상보다 cube 면에 붙은 2D 과일 이미지를 보고 분류해야 하기 때문입니다.

- fruit cube에는 **3개 면에만** 과일 texture overlay를 둡니다. 한 cube의 fruit class는 하나로 고정합니다(apple cube면 3개 면 모두 apple texture).
- 기본 생성은 한 이미지 안에 서로 다른 fruit class의 cube가 같이 나올 수 있고, 한 cube의 3개 면도 같은 class 안에서 서로 다른 source texture를 쓸 수 있습니다. 필요하면 `--single_fruit_class_per_image`, `--single_fruit_texture_per_cube`로 예전처럼 강하게 묶을 수 있습니다.
- 이 overlay는 별도 helper object로 segmentation에 잡히며, 이를 통해 "cube 전체가 보였는지"와 "과일 사진 면이 실제로 보였는지"를 따로 계산합니다.
- 과일 texture는 class별 색상 score로 1차 필터링합니다. `collect_fruit_textures()`는 class별 대표 색상 조건을 통과하는 이미지를 캐시하고, banana는 작은 cube에서 pineapple처럼 보이는 초록 다발 texture를 기본 후보에서 제외합니다. alpha가 있는 누끼 texture는 흰 배경 위에 합성해서 사용합니다.
- texture 파일의 선언 class가 cube class와 다르면 생성 단계에서 **즉시 실패**시키고, `_meta/*.json`에 face별 texture 경로와 class를 기록합니다. 한 cube 안에 다른 fruit class가 섞이는 것(poison data)을 코드 수준에서 막는 장치입니다.
- 흰색 polyhedron은 PLA 출력물처럼 보이도록 off-white 색상과 roughness를 랜덤화합니다. 완전한 순백색 하나만 쓰면 모델이 색상에 과하게 붙을 수 있기 때문입니다.

## 2. Visibility 정책 (수치 기준)

### 2.1 지표 정의

preview와 `_meta/train/*.json`에는 visibility 지표가 들어갑니다.

```text
obj_vis =
  ideal 또는 segmap 기준으로 보이는 object mask 픽셀 수
  / 화면 밖과 occlusion을 반영한 full projected object 면적

fruit_vis =
  ideal 또는 segmap 기준으로 보이는 fruit face mask 픽셀 수
  / 화면 밖과 occlusion을 반영한 full projected cube 면적
```

`obj_vis`는 object 자체가 얼마나 보이는지, `fruit_vis`는 fruit cube에서 과일 사진 면이 얼마나 보이는지를 봅니다. 비율만으로는 충분하지 않습니다 — 이미지 전체에서 과일 사진 면이 너무 작으면 학습 신호가 약하기 때문에 **절대 픽셀 수 기준**도 같이 사용합니다.

### 2.2 Fruit 라벨 기준과 fallback

fruit cube는 아래 기준을 **모두** 통과해야 과일 class로 라벨링됩니다.

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

fallback의 의미는 "처음에는 banana texture가 붙은 cube였지만, 최종 렌더에서 banana 사진 면이 충분히 보이지 않아 banana 라벨 대신 cube 라벨로 쓴다"입니다.

이 정책이 필요한 이유: 과일 texture가 거의 보이지 않는 장면을 과일 class로 학습시키면 모델이 흰 cube의 모양이나 배경 힌트만 보고 과일 class를 예측하게 됩니다. 그런 샘플은 과일 분류 학습에는 해롭지만 cube 검출 학습에는 여전히 유용하므로 버리지 않고 `cube`로 돌립니다.

`_meta/train/*.json`에는 `source_class`, `fallback_class`, `fruit_visibility_pass`, `visible_face_pixels`, `visible_face_bbox`, `visible_face_segments`가 남아, preview에서 fallback 이유를 눈으로 확인할 수 있고 threshold를 바꿀 때도 원인을 추적할 수 있습니다.

### 2.3 Visibility tier 정의와 비율

fruit visibility tier는 과일 면이 항상 정면으로 잘 보이는 데이터만 생기지 않도록 만든 장치입니다.

```text
easy: 과일 면이 크게 보이는 장면
mid:  과일 면이 부분적으로 보이는 장면
hard: 과일 면이 작거나 비스듬히 보이는 장면
```

구현상 `easy`는 작은 tilt, `mid`는 더 큰 tilt, `hard`는 완전 random rotation에 가깝습니다. scale도 tier별로 조정합니다(easy는 조금 크게, hard는 조금 작게).

tier 혼합 비율과 `fruit_photo_cube_ratio`(projected object 면적 대비 과일 사진 면 비율) 기준:

```text
easy 50%: fruit_photo_cube_ratio 0.50 이상
mid  45%: fruit_photo_cube_ratio 0.20~0.50
hard  5%: fruit_photo_cube_ratio 0.10~0.20
```

hard tier에는 별도의 완화된 하한을 둡니다: `fruit_vis 0.10`, `fruit_face_pixels 900` (일반 기준은 각각 0.22 / 1800). hard sample은 필요하지만 주력 분포가 되면 안 된다는 실험 결론(§8) 때문에 5%로 억제합니다.

### 2.4 Ideal visibility gate

후보 배치 후에는 `--ideal_visibility` 단계가 한 번 더 있습니다. 비싼 Blender color render를 돌리기 전에 ideal geometry만으로 occlusion을 계산합니다. 각 mesh face와 fruit face helper를 카메라에 투영하고, 간단한 depth buffer로 앞에 보이는 instance map을 만든 뒤 다음 조건을 검사합니다.

```text
target fruit tier == actual fruit tier
obj_vis > 0.10
fruit_vis >= 0.10
visible fruit face pixels >= 300
visible fruit face bbox width/height >= 10 px
```

조건을 만족하지 못하면 color render 없이 배치를 다시 샘플링합니다. 기본 `--ideal_visibility_max_attempts`는 80입니다. 80번 안에 완벽한 배치를 찾지 못하면 마지막 후보를 사용하고 로그에 fallback을 남깁니다.

### 2.5 라벨 정책 (보이는 픽셀만)

- 어느 경로든(ideal gate 또는 `render_segmap()`) **라벨은 카메라에서 실제로 보이는 픽셀만** 사용합니다. 뒤쪽에 가려진 object 영역은 라벨에 포함하지 않습니다. 합성 단계에서는 object의 원래 3D 모양과 위치를 알고 있지만, 가려진 영역까지 라벨에 넣으면 모델은 이미지에 없는 물체 영역을 맞추도록 학습하게 됩니다.
- 같은 class의 object가 붙어 있어도 instance가 다르면 서로 다른 segmentation 라벨로 나뉩니다. `--seg_contour_mode largest` 기준에서는 object마다 가장 큰 외곽 contour 하나만 YOLO segment 라벨로 씁니다.
- occlusion 필터: `--min_object_visible_ratio 0.10` 이하인 object는 없는 object처럼 취급해서 라벨과 metadata에서 제외합니다. 또한 기본 `--max_covered_ratio 0.80`이므로 visible ratio가 `0.20`보다 낮은 후보 배치는 ideal gate에서 다시 샘플링됩니다. 아주 얇은 sliver 라벨을 학습시키지 않기 위한 기준입니다.
- contour 단순화는 OpenCV `approxPolyDP`, `--seg_contour_epsilon_ratio 0.01`을 사용합니다. 면적 오차 1%가 아니라 contour perimeter의 1% 정도를 경계 거리 오차로 허용한다는 뜻입니다. cube bevel, pixel stair-step, distortion remap 때문에 생긴 자잘한 contour 점을 줄이면서도 과일 면 경계가 무너지지 않는 균형값입니다.

## 3. 장면 / 카메라 / 렌더링 설계

### 3.1 장면 생성 설계

- object 수는 positive scene에서 `--min_objects`부터 `--max_objects` 사이로 샘플링합니다. 최종 학습 권장값은 `1~4`입니다. 너무 많으면 occlusion과 작은 라벨이 늘고, 너무 적으면 실제 난이도를 충분히 만들지 못합니다.
- object 위치는 완전히 균일하게 뿌리지 않습니다. 일부는 기존 object 근처에 배치해 겹침/근접 상황을 만들되, 중심 간 최소 거리를 둬서 모든 라벨이 무의미하게 겹치지는 않게 합니다.
- 각 object transform은 여러 번 시도합니다. **과일 object는 최대 300번, 일반 shape는 최대 120번** 후보를 평가하며, 후보 평가는 다음을 같이 봅니다.

```text
frame 안쪽 margin을 지키는지
projected object area가 충분한지
fruit face 예상 픽셀 수가 충분한지
fruit visibility tier 범위에 들어오는지
```

- 완벽한 후보를 찾지 못하면 가장 점수가 좋은 후보를 사용합니다. 생성이 멈추지 않으면서도 품질 기준에 가까운 장면을 계속 만들기 위한 설계입니다.
- `fruit estimated face pixels`는 `projected_object_area * fruit_photo_cube_ratio`로 계산합니다. fruit cube 전체가 커도 과일 사진 면이 너무 작게 보이면 좋은 후보로 보지 않습니다.
- 생성 단계 우선 조건: `projected object area >= 900 px`, `fruit estimated face pixels >= 300 px`, frame margin 준수, tier target 만족.
- negative sample은 흰색 cylinder/sphere distractor와 COCO 배경으로 구성됩니다. `category_id=0` background로 취급되어 라벨에 들어가지 않습니다. 목적은 "흰색 물체가 보이면 무조건 cube/polyhedron"이라고 외우는 것을 막는 것입니다.
- negative 여부는 `--negative_ratio`와 seed/image_id로 결정됩니다. 같은 seed와 image id를 쓰면 negative 여부가 재현되므로, 중간에 끊겼다가 `--resume`으로 이어도 데이터셋 성격이 크게 흔들리지 않습니다.

### 3.2 카메라와 렌더링 설계

- 카메라는 640x640 기본 해상도에서 lens, 위치, target을 매번 바꿉니다. lens는 넓은 화각부터 망원 느낌까지 흔들고, 카메라 위치는 tabletop/close-up 사이 느낌이 섞이도록 제한된 범위에서 샘플링합니다.
- 조명은 world color, key light, fill light, optional under light, optional ring light를 랜덤화합니다. warm/cool 색을 섞고, area/point/sun light를 섞습니다.
- 배경은 COCO 이미지를 crop해서 사용하며, 배경 자체에도 blur, gain/bias, 약한 texture noise를 넣습니다. 물체가 항상 깨끗한 studio background 위에만 놓이는 문제를 피하기 위함입니다.
- 렌더된 foreground는 COCO background와 합성합니다. object mask는 `soften_mask()`로 약간 부드럽게 만들어 지나치게 날카로운 synthetic edge를 완화합니다.
- 가짜 그림자도 일부 장면에 추가합니다(mask를 이동/blur 후 배경을 어둡게). 물리적으로 완벽한 그림자보다, 모델이 바닥/배경의 어두운 접촉부에 익숙해지게 하는 목적입니다.

### 3.3 카메라 artifact 설계

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

이 artifact는 **이미지에만 적용하고, 라벨은 artifact 전에 계산된 segmentation 결과를 기준으로 저장합니다.** 학습 이미지는 현실적으로 흔들리지만 라벨 좌표는 렌더 결과와 일관되게 유지됩니다. (현행 추가: `--camera_artifact_profile`로 강도 프로파일 선택 가능 — §10.1 참조.)

### 3.4 Lens distortion 설계

lens distortion은 무작정 적용하지 않습니다. distortion map이 원본 이미지 밖을 참조하지 않는 파라미터를 **최대 80번** 찾고, 안전한 파라미터를 찾은 경우에만 이미지, instance segmentation, background를 같은 map으로 remap합니다.

```text
distortion으로 인한 카메라 다양성은 유지한다.
distortion 후 생기는 검은 테두리나 crop/rescale로 segmentation label이 틀어지는 문제는 피한다.
```

안전한 distortion을 찾지 못하면 해당 이미지에는 distortion을 적용하지 않습니다. 품질을 깨면서까지 증강을 넣지 않는 쪽을 선택한 설계입니다. 적용 확률은 `--lens_distortion_prob`(기본 0.35)로 제어합니다.

이전에는 distortion 후 무효 영역을 crop하고 다시 640x640으로 키우는 방식이 있었지만(구버전), 현재 코드는 그 방식 대신 "안전 파라미터만 채택" 방식을 씁니다.

### 3.5 Frame margin 정책

object 자체가 프레임 밖으로 걸리는 문제도 따로 막습니다. object transform 샘플링 시 projected vertices가 화면 안쪽 margin에 들어오는 후보를 고릅니다. 현재 margin은 `0.08`입니다.

이 margin은 edge-touch label을 줄이기 위한 장치입니다. 물체가 가장자리에서 잘리면 segmentation 경계가 이미지 테두리와 붙고, 실제 object 크기와 라벨 크기 해석이 애매해집니다. 품질 체크리스트에서 `edge-touch labels = 0`을 목표로 두는 이유입니다.

## 4. Metadata 스키마

### 4.1 Meta V1 (`_meta/train/*.json`)

각 이미지마다 `_meta/train/{image_id}.json`을 저장합니다. 단순 로그가 아니라 품질 기준을 자동으로 검증하기 위한 설계 산출물입니다.

주요 필드:

```text
image_id
negative
label_format
fruit_visibility_tier
label_objects
fruit_objects
```

- `label_objects`: 실제 YOLO 라벨로 저장된 object의 class, visible pixel 수, projected area, object visible ratio, fallback 정보.
- `fruit_objects`: 과일 texture 면이 실제로 얼마나 보였는지, pass/fallback 판단 근거 (`fruit_visibility_pass`, `visible_face_pixels`, `visible_face_bbox`, `visible_face_segments`, `face_textures`, `fruit_visible_ratio`, `object_visible_ratio` 등).

이 metadata 덕분에 preview 수동 검수와 threshold 위반 샘플 자동 검수를 같이 할 수 있습니다. §7.4 품질 체크리스트가 단순 권장사항이 아니라 `_meta` 기반으로 확인 가능한 기준이 되도록 설계했습니다.

### 4.2 Meta V2 (`--meta_v2`, 현행 기준)

Meta V2는 A1/A2/B/C와 cube-face unified export의 기반이 되는 확장 스키마입니다(`--meta_v2` 옵션으로 생성). 저장되는 핵심 필드 7종:

```text
1. object visible/full mask        (보이는 영역과 가려지기 전 전체 영역 분리)
2. face visible/full mask          (cube face 단위 mask + visible_segments)
3. face quad_xy / quad_xy_raw      (occlusion이 없었다면 face가 가졌을 4점 quad / refinement 전 원본)
4. corner visibility               (face 꼭짓점 4개 각각의 실제 가시 여부)
5. occluder object ids             (누가 이 face를 가렸는지)
6. camera intrinsics/extrinsics    (카메라 내부/외부 파라미터)
7. scene/object transform          (scene_snapshot: object 3D transform, mesh geometry)
```

face 단위 항목에는 `face_kind`(fruit/plain), `fruit_class`, `visible_pixels`, `visible_ratio`, `quality`(good/partial/bad)가 같이 붙습니다. occlusion이 있어도 다음 정보가 분리되어 있는 것이 핵심입니다.

```text
visible_segments:   실제 이미지에서 보이는 face 영역만 표현
quad_xy:            occlusion이 없었다면 face가 가져야 하는 4점 quad
corner_visibility:  각 꼭짓점이 실제로 보이는지 여부
```

cube face 항목 예시(JSON):

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

이 스키마 덕분에 새 Blender 렌더링 없이도 `datasets/meta_v2_50000_coco_texture_v1`의 `_meta`만 재조합해 unified cube-face dataset(94,210 crops)을 export할 수 있었습니다. quad/keypoint 후속 실험은 `quads/*.json` sidecar로 이어집니다.

## 5. 권장 기본값 (생성 CLI 옵션과 이유)

현재 추천 기본값은 코드와 맞춰 아래와 같습니다. 주의: 아래는 **권장값**이며, 일부는 코드 기본값과 다릅니다(예: `--min_fruit_face_pixels` 코드 기본값 300, 권장 1800 / `--min_fruit_visible_ratio` 코드 기본값 0.10, 권장 0.22). 재현 시 명령에 명시해야 합니다. 모든 옵션은 2026-07-08 기준 코드에 존재함을 확인했습니다.

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
| `--fruit_class_weight_scale` | `1.20` | 과일 class 등장 가중치 전체 배율 |
| `--fruit_visibility_easy_weight` | `0.50` | easy tier 50% |
| `--fruit_visibility_mid_weight` | `0.45` | mid tier 45% |
| `--fruit_visibility_hard_weight` | `0.05` | hard tier 5% |
| `--allow_mixed_fruit_classes_per_image` | enabled | 한 이미지 안에 서로 다른 fruit cube class 허용 |
| `--allow_multiple_fruit_textures_per_cube` | enabled | 한 cube의 3개 fruit face에 같은 class 내 다른 source texture 허용 |
| `--canonical_fruit_textures` | enabled | 검증된 canonical texture 경로만 사용 |
| `--train_low_aug` | enabled | YOLO 학습 augmentation이 fruit class 색을 다시 흔드는 것을 줄임 (`run_yolo26_seg_pipeline.py` 옵션) |
| `--ideal_visibility` | enabled | render 전에 ideal geometry로 배치 검증 |
| `--ideal_visibility_max_attempts` | `80` | 좋은 배치를 찾는 최대 시도 횟수 |

### Scale 분포: triangular distribution

scale은 균등분포로 뽑지 않습니다. 작은 object set은 유지하되 과하게 자주 나오지 않도록 **triangular distribution**(`random.triangular(lower, scale_max, mode)`)을 사용합니다. 여러 object가 있는 scene에서는 작은 object를 허용하고, object가 1개뿐인 scene에서는 scale 하한(`--single_object_scale_min 0.82`)과 projected area 하한을 올려 fruit cube와 plain cube가 너무 쉽게 헷갈리는 샘플을 줄입니다.

### Fruit class 가중치

과일 class는 조금 더 자주 나오도록 가중치가 있습니다 (`generate_yolo_coco_composite.py`의 `FRUIT_CLASS_WEIGHTS`, `--fruit_class_weight_scale`로 배율 조정):

```text
shape classes: 1.00
banana:        1.55
orange:        1.35
pineapple:     1.55
apple:         1.50
```

## 6. 텍스처 파이프라인

### 6.1 소스 준비 6단계

아래 대형 데이터는 Git에 올리지 않으므로 재생성/다운로드로 채웁니다. 가장 빠른 복구는 백업에서 `datasets/backgrounds/coco2017/`과 `datasets/fruit_textures/final_fruits36065_original25_fruitseg30_10/`을 그대로 복사하는 것입니다. 복사할 수 없으면 아래 순서대로 다시 만듭니다.

```text
assets/generated/
datasets/backgrounds/coco2017/
datasets/fruit_textures/_sources/fruits360_clean_256/
datasets/fruit_textures/_sources/original_color_filtered_512/
datasets/fruit_textures/_sources/fruitseg30_cutouts_512/
datasets/fruit_textures/final_fruits36065_original25_fruitseg30_10/
```

**1단계 — 기본 asset 생성** (다면체 OBJ):

```powershell
python scripts\make_polyhedron_objs.py --out assets/generated --size 0.08
```

**2단계 — COCO 2017 배경 다운로드** (`val2017`만 있어도 동작하지만, 최종 학습용은 `train2017`까지 받아 배경 pool을 크게):

```powershell
python scripts\download_coco2017_backgrounds.py --split val2017 --output datasets\backgrounds\coco2017
python scripts\download_coco2017_backgrounds.py --split train2017 --output datasets\backgrounds\coco2017
```

**3단계 — Fruits-360 source 준비** (Kaggle `moltean/fruits`):

```powershell
mkdir datasets\fruit_textures\_raw\kaggle_fruits360_download
kaggle datasets download -d moltean/fruits -p datasets\fruit_textures\_raw\kaggle_fruits360_download --unzip
python scripts\prepare_kaggle_fruits360_textures.py --clear
```

**4단계 — Crawled/original source 준비**: 기존 실험과 같은 결과를 원하면 `datasets/fruit_textures/_sources/original_color_filtered_512/`를 백업에서 복사합니다. 복사본이 없으면 Wikimedia 기반 대체 source를 만들 수 있으나, 웹 검색 결과가 바뀌므로 완전히 같은 pool은 아닙니다. 대체 source를 만들었으면 반드시 contact sheet로 잘못 들어온 과일을 제거합니다.

```powershell
python scripts\download_generic_fruit_textures.py `
  --output datasets\fruit_textures\_sources\original_color_filtered_512 `
  --per_class 400 `
  --clear
```

**5단계 — FruitSeg30 누끼 source 준비**: Mendeley public API에서 selected class만 내려받아 image/mask를 RGBA cutout으로 변환합니다. **apple source는 `Apple_Gala`만 사용하고 `Apple_Golden Delicious`는 제외**합니다.

```powershell
python scripts\prepare_fruitseg30_textures.py --clear --size 512
```

2026-05-20 기준 로컬 count: apple 65 / banana 82 / orange 83 / pineapple 65.

**6단계 — 최종 고정 texture pool 생성**: 생성기는 raw source를 매번 뒤지지 않고 final pool만 읽습니다.

```powershell
python scripts\make_mixed_fruit_textures.py --clear
```

출력: `datasets/fruit_textures/final_fruits36065_original25_fruitseg30_10/`

pool preview 확인:

```powershell
python scripts\make_texture_contact_sheets.py `
  --root datasets\fruit_textures\final_fruits36065_original25_fruitseg30_10 `
  --count 40 `
  --thumb_size 104 `
  --columns 8
```

`_preview_apple.jpg` / `_preview_banana.jpg` / `_preview_orange.jpg` / `_preview_pineapple.jpg`를 눈으로 검수합니다.

### 6.2 Production 비율 정책

class별 1,600장, 소스 비율은 **lab(Fruits-360) 65% / crawled·original 25% / FruitSeg30 10%** 로 고정합니다.

| source type | count per class | ratio |
| --- | ---: | ---: |
| lab (Fruits-360 clean) | 1040 | 65% |
| original_filtered (crawled) | 400 | 25% |
| fruitseg30 (누끼) | 160 | 10% |

apple/orange/banana/pineapple 각 1,600장. Fruits-360만으로는 studio domain이 강하고, crawling만으로는 품질이 흔들리므로 고정 final pool로 재현성과 다양성을 같이 확보합니다(§8 실험 발전 기록 참조).

Meta V2 50k 세대의 texture 정리(2026-06-23):

```text
production texture:  datasets/fruit_textures/production_meta_v2_50000_v1
base texture:        datasets/fruit_textures/final_fruits36065_original25_fruitseg30_10
archived experiment: datasets/fruit_textures/_archive_experimental_20260623_c_hardcase
```

웹에서 받은 orange hard texture는 production에 직접 섞지 않았습니다. ① `orange_hard_web_exp_001`에는 실제 orange가 아닌 이미지가 섞였고, ② orange-only 보강은 apple/orange balance를 무너뜨릴 수 있으며, ③ texture source 비율을 바꾸지 않고 render-time augmentation으로 다양성을 늘리는 편이 안전하기 때문입니다. ratio 변경은 report 없이 금지입니다 (canonical policy: `reports/texture_policy_meta_v2_50000_20260623.md`).

### 6.3 Texture 증강 수치

- `--fruit_texture_aug light`: 회전 ±22도, 배율 0.758x~1.32x
- `--fruit_texture_aug strong`: 회전 ±45도, 배율 0.588x~1.70x (구버전에서 사용하다 light로 하향 — §8 참조. 단 Meta V2 50k 세대는 `strong` + `mixed` + collage 0.35를 사용했음)
- `--fruit_texture_layout collage`: 같은 class 안에서 작은 과일 사진 여러 장을 흰 배경에 붙인 generated face texture 생성. collage patch ratio는 light 기준 0.52~0.70 (작은 patch는 class cue가 약해 상향 조정된 값)
- `--fruit_texture_layout mixed`: `--fruit_texture_collage_prob` 확률로 single/collage 혼합
- `print_label` layout과 프로파일 계열은 §10.1 참조 (현행 추가)

## 7. 운영 절차

### 7.1 설치 및 복구

```powershell
git clone https://github.com/jaeyoungi2006/Data_Generation_Blender.git
cd Data_Generation_Blender
conda env create -f environment.yml
conda activate ai_robotics
```

이미 환경이 있으면 `conda env update -n ai_robotics -f environment.yml --prune`. `blenderproc`가 PATH에 없으면:

```powershell
$env:PATH = "C:\Users\USER\anaconda3\envs\ai_robotics\Scripts;C:\Users\USER\anaconda3\envs\ai_robotics;$env:PATH"
```

### 7.2 100장 품질 검증 워크플로우

새 설정은 바로 대량 생성하지 않고 **100장을 새 폴더에 생성한 뒤 preview와 자동 지표를 함께 확인**합니다. synthetic data는 숫자 지표만 보면 좋아 보여도 실제로는 과일 면이 너무 작거나, contour가 깨지거나, 경계가 이상할 수 있기 때문입니다.

<details>
<summary>재현 명령 전문 — 100장 품질 검증 생성 (구버전 기준 출력 경로, 옵션은 현행 유효)</summary>

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

</details>

출력 구조:

```text
datasets/yolo_8class_seg_quality100/
  data.yaml
  images/train/*.jpg
  labels/train/*.txt
  _meta/train/*.json
  ideal_visibility_debug/train/*.jpg
  _chunks/logs/*.log
```

검수 이미지는 두 종류입니다.

```text
ideal_visibility_debug/train/
  검은 배경 위에 ideal geometry visibility 계산 결과를 표시 (full silhouettes / visible objects / visible fruit faces)

visibility_preview/train/
  최종 합성 사진 위에 YOLO label, fruit_vis, obj_vis, OK/fallback을 표시
```

### 7.3 Preview 생성

```powershell
# 일반 segmentation preview
python scripts\make_yolo_bbox_preview.py `
  --dataset datasets\yolo_8class_seg_quality100 `
  --count 100

# visibility debug preview
python scripts\make_yolo_bbox_preview.py `
  --dataset datasets\yolo_8class_seg_quality100 `
  --count 100 `
  --debug_visibility
```

debug preview는 `visibility_preview/train/`에 생성되고, caption panel에 다음이 표시됩니다.

```text
#번호  class  tier=easy/mid/hard  fruit_vis=...  obj_vis=...  OK/fallback
```

preview는 이 프로젝트의 핵심 검증 도구입니다. `--debug_visibility`는 과일 면을 시각적으로 highlight해서 "왜 과일로 남았는지 / 왜 cube fallback이 됐는지"를 빠르게 확인하는 모드이며, **highlight는 preview 이미지에만 들어가고 학습 이미지에는 절대 들어가지 않습니다.**

### 7.4 품질 체크리스트

100장 생성 후 확인 항목:

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

검증 세트 `datasets/ideal_gate_100_test` 기준 결과 (기록):

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

품질 검증은 "완벽한 class 균등 분포"보다 **"라벨 오류를 만들지 않는 것"** 을 우선합니다. class 분포는 100장 단위에서 흔들릴 수 있지만, edge-touch, 너무 작은 라벨, 과일 면 부족, 과한 occlusion 같은 오류는 대량 생성 전에 반드시 잡아야 합니다. 생성 품질 자동 검사는 `scripts/audit_fruit_generation.py`가 담당합니다(mixed fruit class 허용 여부, cube별 texture metadata, face texture metadata 존재 확인).

### 7.5 병렬 생성 및 재시작 설계

`scripts/run_yolo_parallel.py`는 image id를 worker 수만큼 **round-robin chunk**로 나눕니다. 각 worker는 `_chunks/worker_###.txt`에 자기 id 목록을 받고, stdout/stderr는 `_chunks/logs/worker_###.*.log`에 따로 남습니다. 대량 생성 중 일부 worker가 실패했을 때 원인을 분리해서 보기 쉽고, 이미 생성된 이미지를 유지한 채 이어서 돌리기 쉽게 하기 위한 구조입니다.

`--resume`의 완료 판정 규칙: **이미지와 라벨 파일이 모두 존재하는 id만 완료**로 봅니다.

```text
images/train/000123.jpg exists
labels/train/000123.txt exists
=> image_id 123은 skip
```

metadata만 있고 이미지/라벨이 없거나, 이미지 하나만 있는 경우는 완료로 보지 않습니다. 생성 데이터셋의 최소 단위는 image와 label의 쌍이기 때문입니다.

진행률은 output 폴더의 image 개수 기준으로 계산합니다. 이메일 알림은 생성 실패를 막지 않도록 non-blocking으로 처리합니다 — SMTP 문제가 생겨도 데이터 생성은 계속 진행됩니다. worker가 죽으면 `--max_worker_retries 5`, `--retry_delay_seconds 30` 기준으로 재시도하며, `--process_chunk_size`로 짧은 수명의 BlenderProc 프로세스 여러 개로 쪼갤 수도 있습니다(장시간 실행 안정화).

### 7.6 50,000장 생성

최종 dataset은 50,000장 목표로 생성합니다. GPU/CPU 안정성이 우선이면 `--workers 10`부터 시작하고, 시스템이 안정적이면 16까지 올릴 수 있습니다.

<details>
<summary>재현 명령 전문 — 50,000장 생성 (구버전 기준 8-class 출력 경로/이메일 주소, 옵션은 현행 유효)</summary>

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

재시작은 **같은 명령 전체에 `--resume`만 추가**해서 다시 실행합니다 (같은 seed/설정이어야 negative 여부 등이 재현됨).

생성 후 preview:

```powershell
python scripts\make_yolo_bbox_preview.py `
  --dataset datasets\yolo_8class_seg_v5_50000 `
  --count 100 `
  --debug_visibility
```

</details>

현행(Meta V2 세대) 50k 생성은 위 명령 대신 wrapper를 사용합니다 — 생성부터 A1/A2/B/C 학습까지 7단계를 묶은 것:

```powershell
conda activate ai_robotics
cd C:\Users\user\Documents\Data_Generation_Blender
powershell -ExecutionPolicy Bypass -File scripts\run_meta_v2_50000_training_pipeline.ps1
```

```text
1. Meta V2 COCO image 50,000장 생성      (source: datasets/meta_v2_50000_coco_texture_v1)
2. A1/A2/B/C dataset export               (datasets/meta_v2_50000_coco_texture_v1_models)
3. C unknown hard negative 추가
4. A1 YOLO26s-seg 학습                    (texture pack: datasets/fruit_textures/production_meta_v2_50000_v1)
5. A2 YOLO26n-seg 학습
6. B TinyQuadNet 학습                     (log: logs/meta_v2_50000_pipeline)
7. C MobileNetV3-Small 학습
```

50k generation의 texture 설정(2026-06-23 확정): `--fruit_texture_aug strong`, `--fruit_texture_layout mixed`, `--fruit_texture_collage_prob 0.35`.

참고: 8-class 시절 end-to-end 실험(생성→split→audit→preview→train→predict→side-by-side)은 `scripts/run_yolo26_seg_pipeline.py`가 담당했습니다(구버전 기준이지만 스크립트/옵션 유지). wrapper 기준값은 1,000장 `run_yolo26_seg_1000_pipeline.ps1`(train 800/val 100/test 100), 50,000장 `run_yolo26_seg_50000_pipeline.ps1`(train 40000/val 5000/test 5000, `--resume_generate`). 중간 단계 재개는 `--skip_generate --skip_split --skip_preview --skip_train --skip_predict --skip_side_by_side` 조합으로 합니다.

### 7.7 이메일 알림 설정

이메일 설정은 `config/email_smtp.json` 또는 환경 변수로 제공합니다. 비밀번호가 들어가므로 **Git에 올리지 않습니다** (예시는 `config/email_smtp.example.json`).

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

테스트와 생성 중 옵션:

```powershell
python scripts\send_progress_email.py `
  --to jaeyoungi@snu.ac.kr `
  --subject "Data Generation Blender test" `
  --body "SMTP test"
```

```text
--email_to <주소>
--email_every 2000     # 생성 이미지 2000장마다 1회, 시작/완료/오류 시 각 1회
```

메일 전송에 실패해도 데이터 생성은 계속 진행됩니다. 장시간 YOLO 학습 모니터링은 `scripts/monitor_yolo_training_email.ps1`, `scripts/send_yolo_50000_status_email_loop.ps1`(30분마다 results.csv 요약 발송)를 사용합니다. 학습 프로세스 자체를 메일 wrapper로 감싸지 않는 이유: `train_yolo_with_email.py`로 YOLO stdout을 감싸면 carriage-return progress 출력이 Windows pipe에서 멈추는 현상이 재현되어, **학습 프로세스와 메일 모니터를 분리**했습니다.

### 7.8 진행 상황 확인

```powershell
# 생성 개수
$ds='datasets\yolo_8class_seg_v5_50000'; `
$img=(Get-ChildItem "$ds\images\train" -Filter *.jpg).Count; `
$lbl=(Get-ChildItem "$ds\labels\train" -Filter *.txt).Count; `
"images=$img labels=$lbl"

# 관련 프로세스
Get-CimInstance Win32_Process |
  Where-Object {
    $_.Name -in @('python.exe','blenderproc.exe','blender.exe') -and
    $_.CommandLine -match 'run_yolo_parallel|generate_yolo_coco_composite|blenderproc'
  } |
  Select-Object ProcessId,Name,CommandLine

# GPU
nvidia-smi
```

## 8. 실험 발전 기록 (관찰 → 판단 → 조치)

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
| (이후 세대) 목표 재정의 | YOLO 하나로 모든 fruit class를 완벽히 분류하기 어려움 | 먼저 fruit box seg + 깨끗한 class label dataset을 만들고, confusion이 남으면 mask crop 위 lightweight classifier를 올리는 2-stage 검토 | Meta V2 → ABC cascade → Face Classifier(unified)로 발전 (모델 계보는 [HISTORY.md](../HISTORY.md), 중복 기술 생략) |

## 9. Git 관리 기준

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
config/email_smtp.json          (비밀번호 포함 가능)
다운로드 cache
루트의 YOLO 기본 모델 파일 (*.pt)
로컬 scratch note
```

현재 `.gitignore`는 `datasets/`, `runs/`, `*.pt`를 기본적으로 무시합니다. 이미 추적 중인 복구 메타데이터와 checkpoint는 예외적으로 Git에 남아 있습니다. (현행 보충: runtime weights는 `jetson/ABC_model/**`에서 **Git LFS**로 추적 — [readme.md](../readme.md) 참조.)

Git 설계 기준은 **"대형 산출물을 저장소에 넣지 않고도 같은 실험을 다시 만들 수 있어야 한다"** 입니다. 그래서 source code, environment, 작은 metadata, checkpoint hash, 실행 명령, quality checklist를 남기고, 수 GB 규모의 image dataset은 로컬 재생성을 전제로 둡니다. 새로운 threshold나 pipeline 판단을 바꾸면 코드만 바꾸지 말고, 왜 바꿨는지 이 문서의 설계 섹션에도 함께 남깁니다.

## 10. 현행 추가사항 (2026-07 기준)

아카이브 복원 범위 밖에서, 2026-07 시점에 데이터 생성 로직에 추가된 것들입니다.

### 10.1 print_label 프로파일 계열

`--fruit_texture_layout`에 `print_label`이 추가됐습니다: 과일 사진을 face 전체에 채우는 대신 **흰 종이/면 위에 인쇄된 라벨처럼** 붙입니다. 실물 경기 물체가 "흰 cube 면 위 소형 인쇄 아이콘"이라는 관찰을 반영한 것입니다. 프로파일은 `--fruit_print_label_profile`로 선택합니다 (2026-07-08 코드 기준 전체 목록):

```text
legacy
realistic_a4_mild
realistic_a4_sparse_icon
realistic_a4_hard
realistic_a4_boundary
realistic_a4_fruit_visible
realistic_a4_label_offset
realistic_a4_label_offset_strong
realistic_a4_label_offset_lqprint
realistic_a4_tape_edge
realistic_a4_tape_edge_visible
realistic_a4_orange_icon_cluster
realistic_a4_orange_icon_cluster_bright
```

최신이자 가장 중요한 프로파일은 **`realistic_a4_sparse_icon`** 입니다 (2026-07-08 확정, [레시피 반복 기록](../reports/cube_face_unified_eval/sparse_icon_probe_iterations_20260708/summary.md)).

- 목적: 실물 recorded 세트의 실패 regime — **흰 면 위 face 면적의 6~16%짜리 sparse 인쇄 아이콘 + webcam 열화** — 를 holdout 없이 Blender로 재현하는 booster.
- patch 선형비(face 대비): light 기준 **0.24~0.40** → 아이콘 면적 ≈ face의 6~16%.
- 인쇄 평탄화: bilateral + k-means(4~6색) **posterize를 75% 확률**로 적용 (`flatten_patch_to_print_icon`).
- 카메라 열화: `--camera_artifact_profile realistic_webcam_hard` (WB/노출/감마/노이즈/저해상 리샘플/vignette/JPEG) 조합이 난이도의 핵심 축.
- 사용 예: `--fruit_texture_layout print_label --fruit_print_label_profile realistic_a4_sparse_icon --camera_artifact_profile realistic_webcam_hard`
- 10k booster: `datasets/cube_face_unified_booster_realistic_a4_mild_coco_sparse_icon_booster_10k_v1` — 현재 preferred 모델의 fruit top-class 오류 16.7%로 recorded 난이도를 재현 (학습/게이트 진행 상태는 [EXPERIMENTS.md](../EXPERIMENTS.md)).

`--camera_artifact_profile` 선택지(§3.3 artifact의 프로파일화): `default / none / mild_exposure / realistic_webcam_hard / realistic_webcam_boundary / webcam_nocolor_mild / webcam_nocolor_strong / webcam_nocolor_aggressive`.

### 10.2 Arena SUN-111/SUN-168 booster

실제 경기장 색(SUN-111 우드 floor / SUN-168 베이지 fence)을 재현하는 booster 생성 모드입니다. **기존 base dataset의 대체가 아니라**, best checkpoint에서의 fine-tune 또는 oversampling mixed fine-tune 전용입니다.

Arena 가정:

```text
floor: SUN-111 beige/yellow wood family with subtle procedural plywood grain
wall/fence: SUN-168 matte beige paint family
camera: low robot-camera view with slight downward pitch
background: 100% synthetic arena floor/wall, no COCO background
fruit sampling inside booster only: apple 30%, orange 30%, banana 20%, pineapple 20%
scene profiles: normal / SUN color shift / motion blur / close-up / plain cube hard negative
```

<details>
<summary>재현 명령 전문 — 3,000장 arena booster 생성 + preview + mixed fine-tune dataset</summary>

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

```powershell
python scripts\make_arena_booster_preview_sheets.py `
  --dataset datasets\arena_sun111_sun168_booster_v1 `
  --count 40
```

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

mixed dataset은 15,000장에 booster 20% 비율로, 100k + 3k 단순 concat보다 arena 신호가 훨씬 강합니다.

</details>

향후 arena booster의 철학(28-object partial view):

```text
world scene에는 28개 object를 실제처럼 배치한다.
object끼리는 룰북처럼 서로 붙지 않게 한다.
벽 접촉 배치는 허용한다.
camera는 낮은 robot view로 arena 일부만 보게 한다.
학습 frame에는 가까운 object 1~5개, 중거리 일부, 멀리 작은 object 일부가 섞이게 한다.
```

"28개를 한 이미지에 모두 넣는 top-view dataset"이 아니라, **"28개가 깔린 arena를 로봇 시점으로 훑은 partial-view dataset"** 이 목표입니다.

### 10.3 3D mimic arena 렌더러

경기장 최종 공지/실물 사진 기준의 4m x 4m 경기장을 통째로 재현하는 standalone 렌더러가 추가됐습니다.

- 스크립트: [scripts/render_mimic_arena_scene.py](../scripts/render_mimic_arena_scene.py) — SUN-111 우드 바닥(플레이트 이음선 포함), SUN-168 fence 290mm, 알루미늄 코너 플레이트, high-bay LED 4개, 보관함/출발 구역, 28개 object 배치를 [docs/arena_final_environment.md](./arena_final_environment.md) 사양대로 렌더링합니다.
- 환경 사양의 단일 기준 문서: [docs/arena_final_environment.md](./arena_final_environment.md) (공지/실물 사진이 룰북과 다르면 공지/실물 우선).
- 실행 예:

```powershell
C:\Users\user\anaconda3\envs\ai_robotics\Scripts\blenderproc.exe run `
  scripts\render_mimic_arena_scene.py `
  --output reports\arena_mimic_preview_v1 --num_images 10
```

---

변경 이력: 2026-07-08 아카이브(`docs/history/readme_specific_full_20260702.md`)에서 데이터 생성 로직 전체 복원 + 현행 추가사항(§10) 작성.




