from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark A1 + cube-face unified runtime on fixed Meta V2 samples.")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/meta_v2_50000_coco_texture_v1"))
    parser.add_argument("--output", type=Path, default=Path("reports/cube_face_unified_eval/unified_runtime_benchmark.json"))
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--cube-count", type=int, default=4)
    parser.add_argument("--face-count", type=int, default=12)
    parser.add_argument("--meta-scan-limit", type=int, default=20000)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--device", default="0")
    parser.add_argument("--a1-imgsz", type=int, default=640)
    parser.add_argument("--face-imgsz", type=int, default=224)
    parser.add_argument("--a1-conf", type=float, default=0.25)
    parser.add_argument("--face-conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--crop-pad", type=float, default=0.18)
    parser.add_argument(
        "--a1-model",
        type=Path,
        default=Path("reports/abc_runtime_optimization/artifacts/a1_yolo26s_seg_meta_v2_50000_fp32_dynamic_b1.engine"),
    )
    parser.add_argument(
        "--face-model",
        type=Path,
        default=Path("runs/segment/cube_face_unified_yolo26n_seg_v1/weights/best.pt"),
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * p))))
    return float(values[idx])


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "p90": percentile(values, 0.9),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def choose_samples(dataset: Path, cube_count: int, face_count: int, count: int, limit: int) -> list[str]:
    meta_dir = dataset / "_meta" / "train"
    rows: list[tuple[int, int, int, str]] = []
    for idx, path in enumerate(sorted(meta_dir.glob("*.json"))):
        if idx >= limit:
            break
        data = json.loads(path.read_text(encoding="utf-8"))
        meta_v2 = data.get("meta_v2") or {}
        cubes = 0
        good_faces = 0
        visible_pixels = 0
        for obj in meta_v2.get("objects") or []:
            if int(obj.get("a1_class_id", -1)) != 0:
                continue
            cubes += 1
            for face in obj.get("faces") or []:
                pixels = int(face.get("visible_pixels") or 0)
                if face.get("quad_xy") and pixels >= 120:
                    good_faces += 1
                    visible_pixels += pixels
        if cubes == cube_count and good_faces >= max(1, face_count - 1):
            rows.append((abs(good_faces - face_count), -visible_pixels, good_faces, path.stem))
    rows.sort()
    if len(rows) < count:
        raise RuntimeError(f"not enough samples: found {len(rows)}, need {count}")
    return [row[3] for row in rows[:count]]


def load_frames(dataset: Path, stems: list[str]) -> list[tuple[str, np.ndarray]]:
    image_dir = dataset / "images" / "train"
    frames: list[tuple[str, np.ndarray]] = []
    for stem in stems:
        path = image_dir / f"{stem}.jpg"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image: {path}")
        frames.append((stem, image))
    return frames


def expand_box(box: np.ndarray, width: int, height: int, pad: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(1.0, x2 - x1 + 1.0)
    bh = max(1.0, y2 - y1 + 1.0)
    crop_pad = max(bw, bh) * float(pad)
    x1 -= crop_pad
    x2 += crop_pad
    y1 -= crop_pad
    y2 += crop_pad
    return (
        int(max(0, np.floor(x1))),
        int(max(0, np.floor(y1))),
        int(min(width - 1, np.ceil(x2))),
        int(min(height - 1, np.ceil(y2))),
    )


def process_frame(
    frame: np.ndarray,
    a1: YOLO,
    face_model: YOLO,
    args: argparse.Namespace,
) -> dict[str, Any]:
    height, width = frame.shape[:2]
    t0 = time.perf_counter()
    a1_result = a1.predict(
        source=frame,
        imgsz=args.a1_imgsz,
        conf=args.a1_conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
    )[0]
    sync()
    t1 = time.perf_counter()

    crops: list[np.ndarray] = []
    if a1_result.boxes is not None:
        boxes = a1_result.boxes.xyxy.detach().cpu().numpy()
        classes = a1_result.boxes.cls.detach().cpu().numpy().astype(int)
        for box, cls_id in zip(boxes, classes):
            if int(cls_id) != 0:
                continue
            x1, y1, x2, y2 = expand_box(box, width, height, args.crop_pad)
            crop = frame[y1 : y2 + 1, x1 : x2 + 1]
            if crop.size:
                crops.append(cv2.resize(crop, (int(args.face_imgsz), int(args.face_imgsz)), interpolation=cv2.INTER_AREA))
    t2 = time.perf_counter()

    face_count = 0
    class_counts: dict[str, int] = {}
    if crops:
        face_results = face_model.predict(
            source=crops,
            imgsz=args.face_imgsz,
            conf=args.face_conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )
        sync()
        for result in face_results:
            if result.boxes is None:
                continue
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            face_count += int(len(classes))
            for cls_id in classes:
                name = str(face_model.names.get(int(cls_id), cls_id))
                class_counts[name] = class_counts.get(name, 0) + 1
    else:
        sync()
    t3 = time.perf_counter()

    return {
        "objects": len(crops),
        "faces": face_count,
        "class_counts": class_counts,
        "timing_ms": {
            "a1": (t1 - t0) * 1000.0,
            "crop": (t2 - t1) * 1000.0,
            "face_unified": (t3 - t2) * 1000.0,
            "total": (t3 - t0) * 1000.0,
        },
    }


def main() -> None:
    args = parse_args()
    args.dataset = resolve(args.dataset)
    args.output = resolve(args.output)
    args.a1_model = resolve(args.a1_model)
    args.face_model = resolve(args.face_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    stems = choose_samples(args.dataset, args.cube_count, args.face_count, args.sample_count, args.meta_scan_limit)
    frames = load_frames(args.dataset, stems)
    a1 = YOLO(str(args.a1_model), task="segment")
    face_model = YOLO(str(args.face_model), task="segment")

    for idx in range(args.warmup):
        process_frame(frames[idx % len(frames)][1], a1, face_model, args)

    frame_results: list[dict[str, Any]] = []
    fps_values: list[float] = []
    timing: dict[str, list[float]] = {"a1": [], "crop": [], "face_unified": [], "total": []}
    object_counts: list[int] = []
    face_counts: list[int] = []
    class_counts_total: dict[str, int] = {}

    for _repeat in range(args.repeats):
        for stem, frame in frames:
            result = process_frame(frame, a1, face_model, args)
            total_ms = result["timing_ms"]["total"]
            fps_values.append(1000.0 / max(total_ms, 1e-9))
            object_counts.append(int(result["objects"]))
            face_counts.append(int(result["faces"]))
            for key in timing:
                timing[key].append(float(result["timing_ms"][key]))
            for key, value in result["class_counts"].items():
                class_counts_total[key] = class_counts_total.get(key, 0) + int(value)
            frame_results.append({"stem": stem, **result})

    report = {
        "config": {
            "dataset": str(args.dataset),
            "a1_model": str(args.a1_model),
            "face_model": str(args.face_model),
            "sample_count": args.sample_count,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "a1_imgsz": args.a1_imgsz,
            "face_imgsz": args.face_imgsz,
            "a1_conf": args.a1_conf,
            "face_conf": args.face_conf,
            "crop_pad": args.crop_pad,
        },
        "stems": stems,
        "summary": {
            "fps": summarize(fps_values),
            "timing_ms": {key: summarize(values) for key, values in timing.items()},
            "objects_per_frame": summarize([float(v) for v in object_counts]),
            "faces_per_frame": summarize([float(v) for v in face_counts]),
            "class_counts": class_counts_total,
        },
        "frames": frame_results,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
