# Data Generation Notes

이 파일은 짧은 안내용 pointer입니다. `jetson/` 폴더는 Jetson Orin Nano runtime 실행을 위한 폴더이며, 전체 training history를 보관하는 위치가 아닙니다.

데이터 생성과 학습 관련 내용은 아래 root 문서를 보세요.

| 문서 | 역할 |
| --- | --- |
| [../readme.md](../readme.md) | 현재 구조, 빠른 시작, 판단 로직 |
| [../readme_specific.md](../readme_specific.md) | 기술 reference, threshold, dataset 계보, 과거 기록 |
| [../EXPERIMENTS.md](../EXPERIMENTS.md) | 실험 상태 기준 색인 |

Jetson runtime 주요 파일:

```text
jetson/realtime_seg_cam.py
jetson/abc_inference.py
jetson/run_webcam_preview.sh
jetson/run_webcam_preview.ps1
jetson/ABC_model/
```

새 모델을 학습한 뒤 Jetson에서 테스트하려면 검증된 runtime candidate만 `jetson/ABC_model/` 아래 새 explicit folder로 복사하세요. preferred weights를 조용히 덮어쓰면 안 됩니다.
