from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]
JETSON_ROOT = REPO_ROOT / "jetson"
sys.path.insert(0, str(JETSON_ROOT))

import realtime_seg_cam as rt  # noqa: E402
from abc_inference import (  # noqa: E402
    ACTION_COLORS,
    FRUIT_CLASSES,
    FaceEvidence,
    ObjectDecision,
    decide_cube,
)

rt.cv2 = cv2
rt.ACTION_COLORS = ACTION_COLORS
rt.FRUIT_CLASSES = FRUIT_CLASSES
rt.FaceEvidence = FaceEvidence
rt.ObjectDecision = ObjectDecision
rt.decide_cube = decide_cube


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark A1 + cube-face unified preview on a fixed video.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--a1-model", type=Path, required=True)
    parser.add_argument("--face-model", type=Path, required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--device", default="0")
    parser.add_argument("--a1-imgsz", type=int, default=640)
    parser.add_argument("--face-imgsz", type=int, default=224)
    parser.add_argument("--a1-conf", type=float, default=0.25)
    parser.add_argument("--face-conf", type=float, default=0.25)
    parser.add_argument("--crop-pad", type=float, default=0.18)
    parser.add_argument("--overlap", type=float, default=0.6)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--warmup-frames", type=int, default=6)
    parser.add_argument("--sample-frame", type=int, default=30)
    parser.add_argument("--target-shape", default="cube")
    parser.add_argument("--target-fruit", default="apple")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * p))
    return float(ordered[max(0, min(len(ordered) - 1, idx))])


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "p90": percentile(values, 0.9),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def iter_frames(video: Path, max_frames: int = 0):
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"failed to open video: {video}")
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield index, frame
            index += 1
            if max_frames and index >= max_frames:
                break
    finally:
        cap.release()


def write_banner(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    banner_h = 48
    out = np.full((image.shape[0] + banner_h, image.shape[1], 3), (30, 30, 30), dtype=np.uint8)
    out[banner_h:, :] = image
    cv2.putText(out, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 245, 0), 2, cv2.LINE_AA)
    cv2.putText(out, subtitle, (12, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 235, 235), 1, cv2.LINE_AA)
    return out


def main() -> None:
    args = parse_args()
    args.video = resolve(args.video)
    args.output = resolve(args.output)
    args.a1_model = resolve(args.a1_model)
    args.face_model = resolve(args.face_model)
    args.output.mkdir(parents=True, exist_ok=True)

    a1 = YOLO(str(args.a1_model), task="segment")
    face = YOLO(str(args.face_model), task="segment")
    runtime_args = argparse.Namespace(
        device=args.device,
        unified_a1_conf=args.a1_conf,
        unified_a1_imgsz=args.a1_imgsz,
        unified_crop_pad=args.crop_pad,
        conf=args.face_conf,
        imgsz=args.face_imgsz,
        overlap=args.overlap,
        unified_mask_alpha=0.52,
        target_shape=args.target_shape,
        target_fruit=args.target_fruit,
    )

    warmup = []
    for index, frame in iter_frames(args.video, args.warmup_frames):
        warmup.append(frame)
    for frame in warmup:
        rt.process_unified_crop_frame(frame, a1, face, runtime_args)

    total_ms: list[float] = []
    stage_ms: dict[str, list[float]] = {}
    object_counts: list[int] = []
    face_counts: list[int] = []
    sample: np.ndarray | None = None
    sample_lines: list[str] = []
    frames = 0
    wall_start = time.perf_counter()
    for index, frame in iter_frames(args.video, args.max_frames):
        t0 = time.perf_counter()
        annotated, lines, timings = rt.process_unified_crop_frame(frame, a1, face, runtime_args)
        elapsed = (time.perf_counter() - t0) * 1000.0
        total_ms.append(elapsed)
        for key, value in timings.items():
            stage_ms.setdefault(key, []).append(float(value))
        for line in lines:
            if line.startswith("A1 objects="):
                parts = line.replace("=", " ").split()
                try:
                    object_counts.append(int(parts[2]))
                    face_counts.append(int(parts[-1]))
                except Exception:
                    pass
                break
        if index == args.sample_frame:
            sample = annotated.copy()
            sample_lines = list(lines)
        frames += 1
    wall_seconds = time.perf_counter() - wall_start
    process_seconds = sum(total_ms) / 1000.0
    fps = frames / max(process_seconds, 1e-9)

    if sample is None:
        for _, frame in iter_frames(args.video, 1):
            sample, sample_lines, _ = rt.process_unified_crop_frame(frame, a1, face, runtime_args)
            break
    sample_path = args.output / "sample.jpg"
    if sample is not None:
        banner = write_banner(sample, f"{args.label or args.face_model.name}: {fps:.2f} processed FPS", str(args.face_model))
        cv2.imwrite(str(sample_path), banner)

    report: dict[str, Any] = {
        "label": args.label,
        "video": str(args.video),
        "a1_model": str(args.a1_model),
        "face_model": str(args.face_model),
        "frames": frames,
        "wall_seconds": wall_seconds,
        "process_seconds": process_seconds,
        "processed_fps": fps,
        "per_frame_ms": stats(total_ms),
        "stage_ms": {key: stats(values) for key, values in stage_ms.items()},
        "objects_per_frame": stats([float(v) for v in object_counts]),
        "faces_per_frame": stats([float(v) for v in face_counts]),
        "sample_image": str(sample_path) if sample is not None else "",
        "sample_lines": sample_lines[:24],
        "config": {
            "device": args.device,
            "a1_imgsz": args.a1_imgsz,
            "face_imgsz": args.face_imgsz,
            "a1_conf": args.a1_conf,
            "face_conf": args.face_conf,
            "crop_pad": args.crop_pad,
            "overlap": args.overlap,
            "warmup_frames": args.warmup_frames,
        },
    }
    (args.output / "benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Cube-Face Unified Video Benchmark",
        "",
        f"- Label: `{args.label}`",
        f"- Video: `{args.video}`",
        f"- A1 model: `{args.a1_model}`",
        f"- Face model: `{args.face_model}`",
        f"- Frames: `{frames}`",
        f"- Processed FPS: `{fps:.2f}`",
        "",
        "| stage | mean ms | median ms | p90 ms |",
        "|---|---:|---:|---:|",
        f"| total | {report['per_frame_ms']['mean']:.1f} | {report['per_frame_ms']['median']:.1f} | {report['per_frame_ms']['p90']:.1f} |",
    ]
    for key in ["a1_ms", "crop_ms", "face_ms", "total_ms"]:
        if key in report["stage_ms"]:
            s = report["stage_ms"][key]
            lines.append(f"| {key} | {s['mean']:.1f} | {s['median']:.1f} | {s['p90']:.1f} |")
    lines.extend(["", "## Sample Lines", ""])
    lines.extend(f"- {line}" for line in sample_lines[:16])
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["label", "frames", "processed_fps", "per_frame_ms", "stage_ms"]}, indent=2))


if __name__ == "__main__":
    main()
