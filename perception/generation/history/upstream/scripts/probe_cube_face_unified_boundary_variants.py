from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


CLASS_NAMES = ["apple", "orange", "banana", "pineapple", "plain"]
FRUIT = set(CLASS_NAMES[:4])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe cube-face unified class-boundary stability under mild runtime-like variants."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--match-iou", type=float, default=0.25)
    parser.add_argument("--batch", type=int, default=192)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-images", type=int, default=0, help="0 means all images in split.")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--sheet-limit", type=int, default=80)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def read_yolo_labels(path: Path, size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        cls_id = int(float(parts[0]))
        coords = [float(v) for v in parts[1:]]
        pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= size
        pts[:, 1] *= size
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        rows.append(
            {
                "idx": idx,
                "cls_id": cls_id,
                "cls": CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id),
                "box": [float(x1), float(y1), float(x2), float(y2)],
            }
        )
    return rows


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-9)


def adjust_bgr(image: np.ndarray, *, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
    return np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def shift_hsv(image: np.ndarray, *, dh: int = 0, sat: float = 1.0, val: float = 1.0) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + dh) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * val, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return image.copy()
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return decoded if decoded is not None else image.copy()


def variants(image: np.ndarray) -> dict[str, np.ndarray]:
    warm = image.astype(np.float32)
    warm[..., 2] *= 1.08
    warm[..., 1] *= 1.02
    warm[..., 0] *= 0.96
    warm = np.clip(warm, 0, 255).astype(np.uint8)
    return {
        "clean": image,
        "warm_wb_plus": warm,
        "red_orange_hue": shift_hsv(image, dh=-4, sat=1.08, val=1.02),
        "low_sat_warm": shift_hsv(warm, dh=0, sat=0.78, val=1.02),
        "dim_warm": adjust_bgr(warm, alpha=0.88, beta=-6),
        "jpeg82": jpeg_roundtrip(image, 82),
        "soft_focus": cv2.GaussianBlur(image, (3, 3), 0.55),
        "contrast_plus": adjust_bgr(image, alpha=1.12, beta=-8),
    }


def collect_images(dataset: Path, split: str, max_images: int, seed: int) -> list[Path]:
    image_dir = dataset / "images" / split
    paths = sorted([p for p in image_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if max_images and len(paths) > max_images:
        rng = random.Random(seed)
        paths = sorted(rng.sample(paths, max_images))
    return paths


def classify_transition(gt: str, pred: str | None) -> str:
    if pred is None:
        return f"{gt}->missed"
    if pred == gt:
        return f"{gt}->correct"
    return f"{gt}->{pred}"


def draw_sample(image: np.ndarray, gt: dict[str, Any], pred: dict[str, Any] | None, title: str) -> np.ndarray:
    out = image.copy()
    gx1, gy1, gx2, gy2 = [int(round(v)) for v in gt["box"]]
    cv2.rectangle(out, (gx1, gy1), (gx2, gy2), (40, 220, 40), 1, cv2.LINE_AA)
    if pred is not None:
        px1, py1, px2, py2 = [int(round(v)) for v in pred["box"]]
        cv2.rectangle(out, (px1, py1), (px2, py2), (35, 75, 235), 1, cv2.LINE_AA)
        label = f"{gt['cls']}->{pred['cls']} {pred['conf']:.2f}"
    else:
        label = f"{gt['cls']}->miss"
    cv2.putText(out, title[:22], (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(out, title[:22], (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, label[:28], (4, out.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(out, label[:28], (4, out.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def write_sheet(path: Path, images: list[np.ndarray], cell: int = 128) -> None:
    if not images:
        return
    cols = min(8, len(images))
    rows = int(math.ceil(len(images) / cols))
    sheet = np.full((rows * cell, cols * cell, 3), 242, dtype=np.uint8)
    for idx, img in enumerate(images):
        resized = cv2.resize(img, (cell, cell), interpolation=cv2.INTER_AREA)
        y = (idx // cols) * cell
        x = (idx % cols) * cell
        sheet[y : y + cell, x : x + cell] = resized
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)


def main() -> None:
    args = parse_args()
    args.model = resolve(args.model)
    args.dataset = resolve(args.dataset)
    args.output = resolve(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    sheet_dir = args.output / "sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(args.dataset, args.split, args.max_images, args.seed)
    model = YOLO(str(args.model), task="segment")

    records: list[dict[str, Any]] = []
    pending_images: list[np.ndarray] = []
    pending_meta: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal pending_images, pending_meta
        if not pending_images:
            return
        results = model.predict(
            source=pending_images,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
            batch=args.batch,
        )
        for meta, img, result in zip(pending_meta, pending_images, results):
            preds: list[dict[str, Any]] = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                clss = result.boxes.cls.detach().cpu().numpy().astype(int)
                confs = result.boxes.conf.detach().cpu().numpy()
                for box, cls_id, conf in zip(boxes, clss, confs):
                    cls_name = CLASS_NAMES[int(cls_id)] if 0 <= int(cls_id) < len(CLASS_NAMES) else str(int(cls_id))
                    preds.append({"cls": cls_name, "cls_id": int(cls_id), "box": [float(v) for v in box], "conf": float(conf)})
            used: set[int] = set()
            for gt in meta["labels"]:
                best_idx = -1
                best_iou = 0.0
                for pred_idx, pred in enumerate(preds):
                    if pred_idx in used:
                        continue
                    iou = box_iou(gt["box"], pred["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = pred_idx
                pred = preds[best_idx] if best_idx >= 0 and best_iou >= args.match_iou else None
                if best_idx >= 0 and best_iou >= args.match_iou:
                    used.add(best_idx)
                pred_cls = pred["cls"] if pred else None
                records.append(
                    {
                        "stem": meta["stem"],
                        "variant": meta["variant"],
                        "gt": gt["cls"],
                        "pred": pred_cls or "missed",
                        "pred_conf": pred["conf"] if pred else 0.0,
                        "match_iou": best_iou if pred else 0.0,
                        "transition": classify_transition(gt["cls"], pred_cls),
                    }
                )
                key = classify_transition(gt["cls"], pred_cls)
                if key in samples and len(samples[key]) < args.sheet_limit:
                    samples[key].append(draw_sample(img, gt, pred, f"{meta['variant']} {meta['stem']}"))
        pending_images = []
        pending_meta = []

    sample_keys = [
        "orange->apple",
        "apple->orange",
        "banana->pineapple",
        "pineapple->banana",
        "fruit->plain",
        "plain->fruit",
        "non_apple_fruit->apple",
    ]
    samples: dict[str, list[np.ndarray]] = {key: [] for key in sample_keys}

    for img_path in images:
        label_path = args.dataset / "labels" / args.split / f"{img_path.stem}.txt"
        labels = read_yolo_labels(label_path, args.imgsz)
        if not labels:
            continue
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        if image.shape[:2] != (args.imgsz, args.imgsz):
            image = cv2.resize(image, (args.imgsz, args.imgsz), interpolation=cv2.INTER_AREA)
        for name, var_img in variants(image).items():
            pending_images.append(var_img)
            pending_meta.append({"stem": img_path.stem, "variant": name, "labels": labels})
            if len(pending_images) >= args.batch:
                flush()
    flush()

    by_variant: dict[str, Counter[str]] = defaultdict(Counter)
    aggregate_by_variant: dict[str, Counter[str]] = defaultdict(Counter)
    per_class: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for row in records:
        by_variant[row["variant"]][row["transition"]] += 1
        gt = row["gt"]
        pred = row["pred"]
        if gt in FRUIT and pred == "plain":
            aggregate_by_variant[row["variant"]]["fruit->plain"] += 1
        if gt == "plain" and pred in FRUIT:
            aggregate_by_variant[row["variant"]]["plain->fruit"] += 1
        if gt in FRUIT and pred == "apple" and gt != "apple":
            aggregate_by_variant[row["variant"]]["non_apple_fruit->apple"] += 1
        status = "correct" if row["gt"] == row["pred"] else ("missed" if row["pred"] == "missed" else "wrong")
        per_class[row["variant"]][row["gt"]][status] += 1

    summary: dict[str, Any] = {
        "model": str(args.model),
        "dataset": str(args.dataset),
        "split": args.split,
        "images": len(images),
        "records": len(records),
        "conf": args.conf,
        "match_iou": args.match_iou,
        "variants": {},
    }
    for variant, counts in sorted(by_variant.items()):
        class_rows = {}
        for cls_name in CLASS_NAMES:
            c = per_class[variant][cls_name]
            total = c["correct"] + c["wrong"] + c["missed"]
            class_rows[cls_name] = {
                "total": int(total),
                "correct": int(c["correct"]),
                "wrong": int(c["wrong"]),
                "missed": int(c["missed"]),
                "acc": float(c["correct"] / total) if total else 0.0,
            }
        summary["variants"][variant] = {
            "transitions": dict(counts.most_common()),
            "aggregate_transitions": dict(aggregate_by_variant[variant].most_common()),
            "class": class_rows,
        }

    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "variant", "gt", "pred", "pred_conf", "match_iou", "transition"])
        writer.writeheader()
        writer.writerows(records)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Cube-Face Unified Boundary Variant Probe",
        "",
        "Policy: evaluation-only. Existing validation images are transformed in memory; no training data is modified.",
        "",
        f"- Model: `{args.model}`",
        f"- Dataset: `{args.dataset}`",
        f"- Split/images: `{args.split}` / `{len(images)}`",
        f"- Runtime confidence: `{args.conf}`",
        f"- Match IoU: `{args.match_iou}`",
        "",
        "## Boundary Counts",
        "",
        "| variant | orange->apple | apple->orange | banana->pineapple | pineapple->banana | fruit->plain | plain->fruit | non-apple fruit->apple |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in sorted(by_variant):
        c = by_variant[variant]
        agg = aggregate_by_variant[variant]
        lines.append(
            f"| {variant} | {c['orange->apple']} | {c['apple->orange']} | {c['banana->pineapple']} | "
            f"{c['pineapple->banana']} | {agg['fruit->plain']} | {agg['plain->fruit']} | {agg['non_apple_fruit->apple']} |"
        )
    lines.extend(["", "## Per-Class Accuracy", ""])
    for variant in sorted(summary["variants"]):
        lines.append(f"### {variant}")
        lines.append("| class | total | correct | wrong | missed | acc |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for cls_name in CLASS_NAMES:
            row = summary["variants"][variant]["class"][cls_name]
            lines.append(
                f"| {cls_name} | {row['total']} | {row['correct']} | {row['wrong']} | {row['missed']} | {row['acc']:.2%} |"
            )
        lines.append("")
    lines.extend(["## Files", "", f"- CSV: `{args.output / 'predictions.csv'}`", f"- Summary JSON: `{args.output / 'summary.json'}`"])
    for key, imgs in samples.items():
        sheet = sheet_dir / f"{key.replace('->', '_to_')}.jpg"
        write_sheet(sheet, imgs)
        if imgs:
            lines.append(f"- `{key}` sheet: `{sheet}`")
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"images": len(images), "records": len(records), "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
