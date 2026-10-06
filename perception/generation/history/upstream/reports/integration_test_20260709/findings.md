# End-to-end 통합 검증 결과 (2026-07-09)

실제 런타임 함수 `process_unified_crop_frame`(Cube Detector → crop → Face Classifier → guard 붙은 `decide_cube`)를 arena 프레임에 그대로 호출. 모델은 **신규 alias `cube-detector`/`face-classifier`**로 로딩(rename 검증 겸).

## ✅ 배선/코드: 검증됨 (버그 없음)

- 24개 와이드 프레임 + 8개 approach 프레임 전부 **0 런타임 에러**로 처리.
- 신규 alias 정상 resolve, guard 파라미터(`fruit_min_conf=0.45`) 정상 전달.
- `cube_too_far` 게이팅 정상(와이드 프레임에서 98건 정확히 게이팅), shape 탐지 정상(octahedron/icosahedron conf 0.70~0.95).
- alias rename·guard·decide_cube 변경으로 인한 통합 버그 없음.

## ⚠️ 배포 관점 발견 (컴포넌트 테스트가 못 잡던 것)

**실제 로봇-접근(pickup) 프레임에서 fruit 큐브가 fruit_cube로 식별되지 않고 대부분 `inspect`(unresolved/ambiguous)로 빠짐.** 원인 2가지 (approach 프레임 정밀 추적):

1. **`cube_too_far` short-side 게이트(80px)가 접근 시나리오에 빡셈.** 예: arena_000 큰 큐브 short_side=78px, area=6318px² → area는 통과(≥6000)하나 short_side가 2px 모자라 too_far. 코앞 큐브가 거부됨.
2. **근접·하향 시점에서 Face Classifier가 거의 발화 안 함.** 게이트 통과한 큰 큐브(short_side 126/111px)도 face model 출력 `[]`. 로봇 카메라(높이 ~0.2m, 거리 ~0.5m)가 큐브를 내려다보면 **plain 윗면이 지배적**이고, 인쇄면은 ㄷ배치로 옆에 있어 foreshorten → 판별 실패.
   - 대조: arena face-crop probe(GT bbox 기준 깨끗한 224 crop, 이상적 각도)는 orange 79%. 즉 **모델은 정상, 문제는 실제 접근 시점의 crop 구성/각도**.

## 해석

- shipped 코드는 정상. 발견된 것은 **런타임 threshold + 로봇 카메라 각도**의 배포 튜닝 이슈로, 프로젝트 정책상 원래 실제 로봇 카메라로 검증할 대상.
- 컴포넌트 지표(recorded 95.3%, arena probe 79%)는 "깨끗한 crop 기준 모델 성능"이고, **로봇이 실제로 pickup하려면 (a) too_far short-side 완화 검토, (b) 인쇄면이 카메라를 향하도록 접근 각도/자세 설계**가 필요.

## 권장 (실물 카메라 검증 시)

1. `CUBE_TOO_FAR_MIN_SHORT_SIDE_PIXELS`(현 80) 실측 하향 튜닝 — 접근 거리에서 인쇄면이 읽히는 최소 크기 기준으로.
2. 접근 자세: 인쇄면(옆면)이 카메라 정면을 향하도록. 윗면만 보이는 각도 회피.
3. 런타임 crop이 학습 crop 분포와 일치하는지 실물에서 재확인.

## 산출물

- 재사용: 렌더러에 `--approach N` 모드 추가 (fruit 큐브 근접 pickup 뷰; `scripts/render_mimic_arena_scene.py`)
- 근거 프레임: `annotated_*.jpg`(와이드, too_far 게이팅), `approach_*.jpg`(근접, cube_too_far/빈 face 출력)
