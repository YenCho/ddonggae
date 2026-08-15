#!/usr/bin/env python3
"""이진 검증기 학습본(.pt) → 런타임용 ONNX 변환.

런타임 계약(`_fieldlib.PairVerifier`)에 맞춘다:
  · 입력 1×3×128×128 float32, **배치 축 동적** (프레임당 1-forward 배치)
  · 전처리는 호출부 담당 (BGR읽기→RGB→/255→ImageNet 정규화)
  · 출력은 로짓 N×n_cls — 호출부가 softmax
  · 클래스 순서는 체크포인트 `classes` 키를 따르며, 산출 디렉터리 구조는
    기존 검증기와 동일하게 `<name>/weights/best.onnx` + `<name>/classes.json`

사용:
  python3 scripts/dev/real_validation/101_export_pair_onnx.py \
      logs/real_validation/bp_train/bp_real_v1/weights/best.pt \
      --out data/yolo/weights/topfruit_v3ft_release/verifiers/pair_banana_pineapple_real_v1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=128)
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()

    from train_face_mobilenetv3 import make_model  # noqa: E402

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes = list(ck.get("classes", []))
    if not classes:
        raise SystemExit("체크포인트에 classes 키가 없다 — 런타임이 순서를 못 정한다")
    state = ck.get("model", ck.get("state_dict"))
    model, model_name, _ = make_model(len(classes), pretrained=False)
    model.load_state_dict(state)
    model.eval()

    wdir = args.out / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    onnx_path = wdir / "best.onnx"
    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz)
    torch.onnx.export(
        model, dummy, str(onnx_path), opset_version=args.opset,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})

    (args.out / "classes.json").write_text(json.dumps(
        {"classes": classes, "input": [1, 3, args.imgsz, args.imgsz],
         "preprocess": "BGR read -> RGB -> /255 -> ImageNet mean/std",
         "source_checkpoint": str(args.checkpoint), "arch": model_name},
        ensure_ascii=False, indent=1))

    # 등가성 확인 — torch 와 onnxruntime 출력이 갈리면 런타임에서 조용히 틀린다
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 3, args.imgsz, args.imgsz)).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    got = sess.run(None, {"input": x})[0]
    err = float(np.abs(ref - got).max())
    print(f"클래스 {classes} / arch {model_name}")
    print(f"ONNX: {onnx_path}")
    print(f"torch vs onnxruntime 최대 오차 {err:.2e} "
          f"({'OK' if err < 1e-4 else '⚠ 확인 필요'})")


if __name__ == "__main__":
    main()
