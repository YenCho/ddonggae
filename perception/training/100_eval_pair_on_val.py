#!/usr/bin/env python3
"""이진 검증기 평가 — **실기 도메인 val(center scan, 셀 GT)** 에서 신구 비교.

왜 별도 스크립트인가: `scripts/train_face_mobilenetv3.py` 는 `train/` 폴더만
읽고 거기서 `--val_fraction` 만큼 잘라 검증한다(263~273행). 즉 우리가 따로
만든 `val/`(15:45 center scan · 셀 GT 확정본)은 학습 중에 **쓰이지 않는다**.
그 내부 val 은 frames/photo 도메인이고 3fps 연속 프레임이 양쪽에 들어가
누수까지 있어 acc 1.0 이 쉽게 나온다 — 실기 성능의 근거가 못 된다.

이 스크립트는 실기 도메인 val 에서만 잰다:
  · 신규 학습본 (.pt, MobileNetV3-Small)
  · 현행 런타임 검증기 (.onnx, `_fieldlib.PairVerifier` 와 동일 전처리)

사용:
  python3 scripts/dev/real_validation/100_eval_pair_on_val.py \
      --data data/yolo/pair_real_20260723 --classes banana,pineapple \
      --new logs/real_validation/bp_train/bp_real_v1/weights/best.pt \
      --old data/yolo/weights/topfruit_v3ft_release/verifiers/pair_banana_pineapple_v2_1/weights/best.onnx
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_split(root: Path, split: str, classes):
    items = []
    for ci, c in enumerate(classes):
        d = root / split / c
        for pat in ("*.png", "*.jpg"):
            for p in sorted(d.glob(pat)):
                items.append((p, ci, c))
    return items


def to_tensor(path, imgsz=128):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((x - MEAN) / STD).transpose(2, 0, 1)


def report(name, y_true, y_pred, conf, classes):
    n = len(y_true)
    acc = float(np.mean(np.asarray(y_true) == np.asarray(y_pred))) if n else 0.0
    print(f"\n=== {name} ===  n={n}  정확도 {acc*100:.1f}%")
    print(f"{'':12s} " + " ".join(f"pred_{c:<9s}" for c in classes))
    for ci, c in enumerate(classes):
        row = [sum(1 for t, p in zip(y_true, y_pred) if t == ci and p == cj)
               for cj in range(len(classes))]
        print(f"true_{c:<8s} " + " ".join(f"{v:>14d}" for v in row))
    hi = [i for i, cf in enumerate(conf) if cf >= 0.80]
    if hi:
        hacc = float(np.mean([y_true[i] == y_pred[i] for i in hi]))
        print(f"  런타임 교체 임계(conf>=0.80) 적용분: {len(hi)}/{n}건, "
              f"그중 정확도 {hacc*100:.1f}%")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--classes", required=True, help="예: banana,pineapple")
    ap.add_argument("--new", type=Path, default=None, help="신규 학습본 .pt")
    ap.add_argument("--old", type=Path, default=None, help="현행 런타임 .onnx")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    classes = args.classes.split(",")
    items = load_split(args.data, args.split, classes)
    if not items:
        raise SystemExit(f"{args.split} 비어 있음")
    print(f"평가셋: {args.split}  {dict(Counter(c for _, _, c in items))}")
    X = np.stack([to_tensor(p) for p, _, _ in items])
    y = [ci for _, ci, _ in items]

    results = {}
    if args.old and args.old.exists():
        import onnxruntime as ort
        sess = ort.InferenceSession(str(args.old), providers=["CPUExecutionProvider"])
        logits = sess.run(None, {sess.get_inputs()[0].name: X.astype(np.float32)})[0]
        e = np.exp(logits - logits.max(1, keepdims=True))
        pr = e / e.sum(1, keepdims=True)
        # ONNX 출력 클래스 순서는 체크포인트 classes 키 기준 — BP 는 (banana, pineapple)
        pred, conf = pr.argmax(1).tolist(), pr.max(1).tolist()
        results["old"] = report(f"현행 {args.old.parent.parent.name}", y, pred, conf, classes)

    if args.new and args.new.exists():
        import torch
        ck = torch.load(args.new, map_location="cpu", weights_only=False)
        ck_classes = list(ck.get("classes", classes))
        from train_face_mobilenetv3 import make_model  # noqa
        model, _, _ = make_model(len(ck_classes), pretrained=False)
        model.load_state_dict(ck["model"] if "model" in ck else ck["state_dict"])
        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(X))
            pr = torch.softmax(out, 1).numpy()
        # 학습본의 클래스 순서가 평가 순서와 다를 수 있어 매핑
        idx = [ck_classes.index(c) for c in classes]
        pr = pr[:, idx]
        pred, conf = pr.argmax(1).tolist(), pr.max(1).tolist()
        results["new"] = report(f"신규 {args.new.parent.parent.name}", y, pred, conf, classes)

    if len(results) == 2:
        d = (results["new"] - results["old"]) * 100
        print(f"\n>>> 신규 - 현행 = {d:+.1f}%p "
              f"(n={len(items)} — 이 크기에서 1건 = {100/len(items):.1f}%p)")


if __name__ == "__main__":
    main()
