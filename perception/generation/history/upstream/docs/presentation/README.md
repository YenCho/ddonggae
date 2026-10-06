# 발표 슬라이드 (docs/presentation)

`slides.tex` — Beamer 16:9, 시각 위주(PPT 스타일) 16장. 5월말 이후 정리분:
ABC Cascade(단계 I/O + runtime 흐름)와 한계 → Unified 전환(input/output·FPS) →
최종 object 분류 + `decide_cube` 결정 트리(큐브 identity class 의미) →
Fine-tuning(+α) → mimic arena → 정직한 검증 → 요약.

## 컴파일

**한글이라 kotex이 필요합니다. pdfLaTeX로 컴파일하세요.**

### Overleaf (권장, 가장 간단)
1. `docs/presentation` 폴더 전체(`slides.tex` + `img/`)를 업로드
2. Menu → Compiler: **pdfLaTeX**
3. Recompile

### 로컬 (TeX Live 전체 설치 필요)
```
cd docs/presentation
pdflatex slides.tex
pdflatex slides.tex   # 목차/참조 안정화 위해 2회
```
- 필요한 패키지: `kotex`(한글, 나눔폰트), `tikz`, `booktabs`, `graphicx` — 전부 TeX Live 표준.

## 이미지 (슬라이드에 번들됨)
- `img/ex_abc.png` — ABC cascade 4단계 실제 I/O (A1→A2→B→C, ABC 학습 데이터셋, 같은 큐브 관통)
- `img/ex_a1.png` — Cube Detector 실제 I/O (프레임 → 물체+마스크)
- `img/ex_face.png` — Face Classifier 실제 I/O (crop → class+마스크, 3 클래스)
- `img/ex_fail.png` — 실물 오렌지 큐브 before(apple)/after(orange)
- `img/ex_cliff.png` — 인식 절벽: 같은 오렌지, 면적만 다름 (실제 크롭)
- `img/arena_real_floor.jpg` / `img/arena_mimic.jpg` — 실물 경기장 vs Blender 재현

I/O 예시 이미지는 `scratchpad/gen_io_examples.py`, `gen_io_v2.py`로 실제 모델 추론해서 생성.
추가 이미지는 `docs/reference_photos/`, `reports/integration_test_20260709/` 참고.

## 참고
- 로컬에 LaTeX 컴파일러가 없어 자동 컴파일 검증은 못 했으나, 환경 균형·매크로 인자·이미지 경로·참조를 수동 점검했습니다.
- 내용/디자인 수정 요청 주시면 반영합니다 (슬라이드 추가·순서 변경·수치 갱신 등).
