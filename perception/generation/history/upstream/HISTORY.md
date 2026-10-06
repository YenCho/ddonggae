# 프로젝트 역사와 과거 문서 보관소

기준일: 2026-07-08. 이 문서는 두 가지를 담습니다: ① 모델/데이터가 발전해온 서사, ② 과거 문서 정리 과정에서 삭제됐던 내용의 복원 위치.

## 1. 발전 서사 (왜 지금 구조가 되었나)

| 시기 | 일어난 일 | 남긴 교훈 |
| --- | --- | --- |
| 2026-05-14~17 | Blender 합성 데이터 파이프라인 + 8-class object-only YOLO 시작 | 물체는 찾아도 "보이는 면의 과일"을 판단 못함 |
| 2026-05-17~20 | confusion matrix 기반 반복 개선 (yolo26n-seg 전환, texture aug 완화, poison data 발견·audit, texture pool 확정) | "관찰→판단→조치" 기록 문화 정착 (원문: 아카이브 §실험 발전 기록) |
| 2026-05-30 | 실제 경기장 색(SUN-111 우드 floor / SUN-168 베이지 fence) arena booster | booster는 base 대체가 아닌 oversampling fine-tune 전용 |
| 2026-06-22~23 | 경기장 28개 물체(도형 4종×4 + 과일 큐브 4종×3) 해석 확정, visibility-aware Meta V2 + ABC cascade(A1→A2→B→C) + 50k 학습 계획 | robot partial-view 전제; 보이는 픽셀만 라벨링(fruit_vis ≥ 0.10 등 수치 기준) |
| 2026-06-28~29 | A2+B+C를 **Face Classifier(unified)** 하나로 통합, balanced v2 | 다단계 overhead 제거 (ABC 20 FPS → unified 47 FPS 여유) |
| 2026-07-02~04 | verified fruit → flat-icon boundary → HSV stronger + color-outlier pruned 승격 계보 | 실물 webcam 인쇄 아이콘 경계 문제와의 싸움 시작 |
| 2026-07-05~07 | pruning 과잉 진단 → readd base 복구 학습 (**현재 preferred**, best epoch 83) | 게이트 통과 후에만 학습; 실물 세트 61.6→65.1% |
| 2026-07-08 | 실패 원인 = sparse printed-icon composition gap 실증, `realistic_a4_sparse_icon` booster 10k + 학습 진행 | holdout(실물 세트/사용자 사진)은 영구 평가 전용 |

모델 계보 한 줄: `yolo26n-seg → unified v1 → balanced v2 → verified_fruit → flat_icon_boundary → hsv_stronger → hsv_pruned_coloroutlier → readd_coloroutlier_base(현재) → (sparse_icon 진행 중)`

## 2. 과거 문서 복원 보관소

2026-07-02 문서 재작성(`a642db5`) 때 readme_specific.md가 2,500줄→419줄로 축약되며 많은 지식이 삭제됐습니다. **삭제 전 완전판을 그대로 복원해 보관합니다:**

- **`docs/history/readme_specific_full_20260702.md`** — 삭제 직전 완전판 (105KB). 아래 내용의 원문 전체가 여기 있음.
- **`docs/history/readme_full_20260623.md`** — 2026-06-23 시점 readme.

> **2026-07-08 복원 완료**: 아래 표의 **데이터 생성 관련 내용 전부** (visibility 철학/수치, 장면·카메라·distortion 설계, Meta V2 스키마, 권장 기본값 표, 텍스처 파이프라인, 100장 검증/병렬 생성/`--resume`/50k 명령, 실험 발전 기록, Git 기준, arena booster)는 이제 **현행 문서 [docs/DATA_GENERATION.md](./docs/DATA_GENERATION.md)로 복원되어 유지보수됩니다.** 아카이브는 원문 그대로(verbatim) 계속 보관하며, 앞으로의 수정은 DATA_GENERATION.md에만 반영합니다.

### 복원된 내용 찾아보기 (완전판 안의 위치)

| 잃었던 내용 | 요약 | 원문 위치 |
| --- | --- | --- |
| 실험 발전 기록 | confusion 관찰→판단→조치 10행 내러티브 (aug 완화, poison data audit 등) | 완전판 §실험 발전 기록 (L1264~) |
| 50k Meta V2 학습 계획 | `run_meta_v2_50000_training_pipeline.ps1` 7단계 학습 순서, 출력 경로 | 완전판 + `git show 5d2b44b` |
| Visibility 정책 수치 | `obj_vis`/`fruit_vis` 수식, fruit 라벨 기준(`fruit_vis>=0.10`, ≥300px, bbox≥10px), "안 보이면 학습 금지" 철학 | 완전판 §visibility 정책 |
| 권장 기본값 표 | 생성 CLI ~35개 옵션 값+이유, class 가중치, tier 비율 | 완전판 (L1809~) |
| 데이터 생성 운영 문서 | 설치/복구, 텍스처 소스 준비 6단계, 품질 검증, 병렬 생성/`--resume` 설계, 50k 생성 명령, Git 관리 기준 | 완전판 (L1507~2500) |
| 장면/카메라/distortion 설계 | 왜 이렇게 흔들었는가의 근거 | 완전판 (L1358~1441, 1780~) |
| Option 3 전략 분기 (ABC 탄생 논리) | 경기장 28개 물체 해석, Task 분리 기각 근거, 10단계 개발 순서, backup 기준점 | 완전판 (L735~1206) |
| Arena SUN-111/168 booster | 경기장 색 가정과 사용 지침 | 완전판 + `git show 625e0fd:readme.md` |
| GitHub README 버전 통합표 | 18개 commit별 정책 유래 매핑 | `git show 5d2b44b` |
| C 모델 최종화/booster 상세 | anchor pair exp006, whole-fruit booster 실행 전문 | 완전판 + `reports/c_orange_hard_case_experiment_log.md` |

### 기타 보존 문서

- `BACKUP_MANIFEST.md` — 2026-05-16 구세대(8-class) 백업 스냅샷 (커밋/SHA256/재구성 절차). 수정하지 않고 보존.
- `jetson/OPTIMIZATION_EXPERIMENTS.md` — ABC 최적화 실험 E000~E042 전체 로그 (최종 preset `best_stable_5080`, 20-21 FPS / 98.26%).
- `reports/**` — 모든 실험 원본 리포트 (EXPERIMENTS.md가 색인).
