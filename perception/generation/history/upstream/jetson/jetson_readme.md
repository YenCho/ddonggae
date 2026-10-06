# Jetson Runtime Readme

이 파일은 예전 링크 호환을 위해 남겨둔 짧은 안내 문서입니다.

Jetson Orin Nano runtime의 기준 문서는 [README.md](./README.md)입니다. 해당 문서에는 다음 내용이 들어 있습니다.

- 필요한 weights와 model aliases
- 첫 webcam 실행
- overlay 의미
- target options
- ABC cascade fallback
- Jetson에서 TensorRT export하는 방법

빠른 실행:

```bash
./jetson/run_webcam_preview.sh \
  --pipeline unified \
  --camera 0 \
  --camera-backend v4l2 \
  --device 0 \
  --target-shape cube \
  --target-fruit apple \
  --print-model-output
```

주의: `preferred-unified`를 full frame에 직접 실행하지 마세요. 반드시 `--pipeline unified`를 사용해 A1이 `cube_like_object`를 먼저 crop하게 해야 합니다.
