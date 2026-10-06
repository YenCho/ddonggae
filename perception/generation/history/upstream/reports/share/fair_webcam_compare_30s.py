from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
JETSON_ROOT = REPO_ROOT / "jetson"
sys.path.insert(0, str(JETSON_ROOT))

import realtime_seg_cam as rt  # noqa: E402
from abc_inference import ABCPipeline, append_c_input_panel, draw_frame_output, frame_summary  # noqa: E402
from ultralytics import YOLO  # noqa: E402


ABC_ROOT = REPO_ROOT.parent / "ABC_model"
ABC_A1 = JETSON_ROOT / "ABC_model" / "meta_v2_a1_objectseg" / "a1_yolo26s_seg_meta_v2_50000" / "weights" / "best.pt"
ABC_A2 = ABC_ROOT / "meta_v2_a2_faceseg" / "a2_yolo26n_seg_meta_v2_50000" / "weights" / "best.pt"
ABC_B = ABC_ROOT / "meta_v2_b_tinyquadnet" / "b_tinyquadnet_meta_v2_50000" / "weights" / "best.pt"
ABC_C = ABC_ROOT / "meta_v2_c_facecls" / "c_mobilenetv3small_meta_v2_50000_runtimewarp_ft" / "weights" / "best.pt"
ABC_C_ONNX = REPO_ROOT / "reports" / "abc_runtime_optimization" / "artifacts" / "c_mobilenetv3small_runtimewarp_warmplain_ft.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fair 30s webcam benchmark for unified vs ABC cascade.")
    parser.add_argument("--camera", default="0")
    parser.add_argument("--camera-backend", default="dshow", choices=("auto", "any", "dshow", "v4l2", "gstreamer"))
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--sample-second", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-shape", default="cube")
    parser.add_argument("--target-fruit", default="apple")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "reports" / "share")
    return parser.parse_args()


def open_camera(camera: str, backend: str) -> cv2.VideoCapture:
    source: int | str = int(camera) if str(camera).isdigit() else camera
    if backend == "dshow":
        return cv2.VideoCapture(source, cv2.CAP_DSHOW)
    if backend == "v4l2":
        return cv2.VideoCapture(source, cv2.CAP_V4L2)
    if backend == "gstreamer":
        return cv2.VideoCapture(str(camera), cv2.CAP_GSTREAMER)
    if backend == "any":
        return cv2.VideoCapture(source, cv2.CAP_ANY)
    return cv2.VideoCapture(source)


def capture_video(args: argparse.Namespace, out_dir: Path) -> tuple[Path, Path, int, float, int]:
    cap = open_camera(args.camera, args.camera_backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open camera={args.camera} backend={args.camera_backend}")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or args.width)
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or args.height)
    nominal_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if nominal_fps <= 1.0:
        nominal_fps = 30.0

    video_path = out_dir / "capture_30s_mjpg.avi"
    sample_path = out_dir / "capture_5s_raw.jpg"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        nominal_fps,
        (actual_w, actual_h),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"failed to open video writer: {video_path}")

    print(f"[capture] camera={args.camera} backend={args.camera_backend} size={actual_w}x{actual_h} nominal_fps={nominal_fps:.2f}", flush=True)
    start = time.perf_counter()
    sample_frame: np.ndarray | None = None
    sample_index = 0
    frames = 0
    while True:
        ok, frame = cap.read()
        now = time.perf_counter()
        elapsed = now - start
        if not ok:
            print("[capture] camera read failed; stopping early", flush=True)
            break
        writer.write(frame)
        if sample_frame is None and elapsed >= args.sample_second:
            sample_frame = frame.copy()
            sample_index = frames
        frames += 1
        if elapsed >= args.duration:
            break
        if frames % 90 == 0:
            print(f"[capture] {elapsed:.1f}s frames={frames}", flush=True)

    cap.release()
    writer.release()
    elapsed = max(time.perf_counter() - start, 1e-6)
    if sample_frame is None:
        sample_frame = frame.copy()
        sample_index = max(0, frames - 1)
    cv2.imwrite(str(sample_path), sample_frame)
    print(f"[capture] done elapsed={elapsed:.2f}s frames={frames} capture_fps={frames / elapsed:.2f} sample_index={sample_index}", flush=True)
    return video_path, sample_path, frames, elapsed, sample_index


def read_first_frame(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read first frame from {video_path}")
    return frame


def iter_video_frames(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield index, frame
            index += 1
    finally:
        cap.release()


def put_banner(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    banner_h = 48
    out = np.full((image.shape[0] + banner_h, image.shape[1], 3), (28, 28, 28), dtype=np.uint8)
    out[banner_h:, :] = image
    cv2.putText(out, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(out, subtitle, (12, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (235, 235, 235), 1, cv2.LINE_AA)
    return out


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    width = max(1, int(round(image.shape[1] * height / image.shape[0])))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def timing_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def benchmark_unified(video_path: Path, sample_index: int, args: argparse.Namespace) -> dict[str, Any]:
    print("[unified] loading models", flush=True)
    a1_model = YOLO(str(rt.resolve_model_path("preferred-a1")), task="segment")
    face_model = YOLO(str(rt.resolve_model_path("preferred-unified", unified_default=True)), task="segment")
    unified_args = argparse.Namespace(
        device=args.device,
        unified_a1_conf=0.25,
        unified_a1_imgsz=640,
        unified_crop_pad=0.18,
        conf=0.25,
        imgsz=224,
        overlap=0.6,
        unified_mask_alpha=0.52,
        target_shape=args.target_shape,
        target_fruit=args.target_fruit,
    )
    first = read_first_frame(video_path)
    for _ in range(2):
        rt.process_unified_crop_frame(first, a1_model, face_model, unified_args)

    print("[unified] benchmarking fixed 30s capture", flush=True)
    total_times: list[float] = []
    stage_times: dict[str, list[float]] = {}
    sample_image: np.ndarray | None = None
    sample_lines: list[str] = []
    count = 0
    start = time.perf_counter()
    for index, frame in iter_video_frames(video_path):
        t0 = time.perf_counter()
        annotated, lines, timings = rt.process_unified_crop_frame(frame, a1_model, face_model, unified_args)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_times.append(elapsed_ms)
        for key, value in timings.items():
            stage_times.setdefault(key, []).append(float(value))
        if index == sample_index:
            sample_image = annotated.copy()
            sample_lines = list(lines)
        count += 1
        if count % 50 == 0:
            print(f"[unified] processed={count} running_fps={count / max(time.perf_counter() - start, 1e-6):.2f}", flush=True)

    process_seconds = sum(total_times) / 1000.0
    fps = count / max(process_seconds, 1e-6)
    print(f"[unified] done frames={count} fps={fps:.2f}", flush=True)
    return {
        "name": "current_unified",
        "frames": count,
        "fps": fps,
        "process_seconds": process_seconds,
        "per_frame_ms": timing_stats(total_times),
        "stage_ms": {key: timing_stats(values) for key, values in stage_times.items()},
        "sample_image": sample_image,
        "sample_lines": sample_lines,
        "config": {
            "device": args.device,
            "a1_imgsz": 640,
            "face_imgsz": 224,
            "a1_conf": 0.25,
            "face_conf": 0.25,
            "crop_pad": 0.18,
        },
    }


def benchmark_abc(video_path: Path, sample_index: int, args: argparse.Namespace) -> dict[str, Any]:
    print("[abc] loading models", flush=True)
    pipeline = ABCPipeline(
        a1_model=ABC_A1,
        a2_model=ABC_A2,
        b_model=ABC_B,
        c_model=ABC_C,
        c_onnx_model=ABC_C_ONNX if ABC_C_ONNX.exists() else None,
        c_onnx_provider="cpu",
        device=args.device,
        a1_conf=0.25,
        a2_conf=0.20,
        c_conf=0.55,
        imgsz=960,
        a2_imgsz=224,
        min_face_pixels=120,
        min_face_object_overlap=0.45,
        refine_quads="pose",
        blacken_c_occluded_face_area=True,
        c_occlusion_visible_ratio=0.92,
        c_max_black_fraction=0.60,
        c_warp_size=112,
        target_shape=args.target_shape,
        target_fruit=args.target_fruit,
        enable_timing=True,
        keep_c_inputs=True,
        use_a1_object_mask=False,
        batch_b_across_objects=True,
        batch_c_across_objects=True,
    )
    first = read_first_frame(video_path)
    for _ in range(2):
        output = pipeline.process_frame(first)
        annotated = draw_frame_output(first, output)
        append_c_input_panel(annotated, pipeline.last_c_inputs, panel_width=360, tile_size=132)

    print("[abc] benchmarking fixed 30s capture", flush=True)
    total_times: list[float] = []
    stage_times: dict[str, list[float]] = {}
    sample_image: np.ndarray | None = None
    sample_lines: list[str] = []
    count = 0
    start = time.perf_counter()
    for index, frame in iter_video_frames(video_path):
        t0 = time.perf_counter()
        output = pipeline.process_frame(frame)
        annotated = draw_frame_output(frame, output)
        annotated = append_c_input_panel(annotated, pipeline.last_c_inputs, panel_width=360, tile_size=132)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_times.append(elapsed_ms)
        for key, value in pipeline.last_timing.items():
            if isinstance(value, (int, float)):
                stage_times.setdefault(key, []).append(float(value))
        if index == sample_index:
            sample_image = annotated.copy()
            sample_lines = frame_summary(output)
        count += 1
        if count % 50 == 0:
            print(f"[abc] processed={count} running_fps={count / max(time.perf_counter() - start, 1e-6):.2f}", flush=True)

    process_seconds = sum(total_times) / 1000.0
    fps = count / max(process_seconds, 1e-6)
    print(f"[abc] done frames={count} fps={fps:.2f}", flush=True)
    return {
        "name": "old_abc_cascade",
        "frames": count,
        "fps": fps,
        "process_seconds": process_seconds,
        "per_frame_ms": timing_stats(total_times),
        "stage_ms": {key: timing_stats(values) for key, values in stage_times.items()},
        "sample_image": sample_image,
        "sample_lines": sample_lines,
        "config": {
            "device": args.device,
            "a1_imgsz": 960,
            "a2_imgsz": 224,
            "a1_conf": 0.25,
            "a2_conf": 0.20,
            "c_conf": 0.55,
            "c_runtime": "onnx:cpu" if ABC_C_ONNX.exists() else "torch",
            "c_warp_size": 112,
            "use_a1_object_mask": False,
            "batch_b": True,
            "batch_c": True,
            "refine_quads": "pose",
        },
    }


def save_outputs(
    out_dir: Path,
    capture_info: dict[str, Any],
    unified: dict[str, Any],
    abc: dict[str, Any],
) -> None:
    unified_img = unified.pop("sample_image")
    abc_img = abc.pop("sample_image")
    if unified_img is None or abc_img is None:
        raise RuntimeError("sample frame was not benchmarked")

    unified_path = out_dir / "current_unified_5s.jpg"
    abc_path = out_dir / "old_abc_cascade_5s_with_warp.jpg"
    side_path = out_dir / "side_by_side_5s_same_capture_30s_avg.jpg"
    json_path = out_dir / "benchmark.json"
    md_path = out_dir / "summary.md"

    unified_banner = put_banner(
        unified_img,
        f"CURRENT UNIFIED  30s AVG FPS: {unified['fps']:.2f}",
        "same 30s captured frames; face masks classified directly inside A1 crop; no warp stage",
    )
    abc_banner = put_banner(
        abc_img,
        f"OLD ABC CASCADE  30s AVG FPS: {abc['fps']:.2f}",
        "same 30s captured frames; right panel shows C classifier perspective-warp inputs",
    )
    height = max(unified_banner.shape[0], abc_banner.shape[0])
    unified_banner = resize_to_height(unified_banner, height)
    abc_banner = resize_to_height(abc_banner, height)
    spacer = np.full((height, 8, 3), (60, 60, 60), dtype=np.uint8)
    side_by_side = np.hstack([unified_banner, spacer, abc_banner])

    cv2.imwrite(str(unified_path), unified_banner)
    cv2.imwrite(str(abc_path), abc_banner)
    cv2.imwrite(str(side_path), side_by_side)

    data = {
        "capture": capture_info,
        "unified": unified,
        "abc": abc,
    }
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def stage_line(result: dict[str, Any]) -> str:
        stages = result.get("stage_ms") or {}
        keys = ["a1_ms", "a2_ms", "b_ms", "c_ms", "face_ms", "total_ms"]
        parts = []
        for key in keys:
            if key in stages:
                parts.append(f"{key}={stages[key]['mean']:.1f}ms")
        return ", ".join(parts) if parts else "-"

    md = [
        "# Fair Webcam FPS Compare",
        "",
        f"- Capture frames: {capture_info['frames']}",
        f"- Capture elapsed: {capture_info['elapsed_seconds']:.2f}s",
        f"- Capture FPS: {capture_info['capture_fps']:.2f}",
        f"- Sample frame: {capture_info['sample_second']:.1f}s, index {capture_info['sample_index']}",
        f"- Device: {capture_info['device']}",
        "",
        "| Pipeline | 30s avg FPS | Mean frame ms | Notes |",
        "|---|---:|---:|---|",
        f"| Current unified | {unified['fps']:.2f} | {unified['per_frame_ms']['mean']:.1f} | A1 crop -> unified face seg/class, no warp |",
        f"| Old ABC cascade | {abc['fps']:.2f} | {abc['per_frame_ms']['mean']:.1f} | A1/A2/B/C, C warp panel shown in sample |",
        "",
        "## Stage Means",
        "",
        f"- Current unified: {stage_line(unified)}",
        f"- Old ABC cascade: {stage_line(abc)}",
        "",
        "## Files",
        "",
        f"- Side by side: `{side_path}`",
        f"- Unified: `{unified_path}`",
        f"- ABC cascade: `{abc_path}`",
        f"- Raw benchmark JSON: `{json_path}`",
        "",
        "## Unified Sample Lines",
        "",
    ]
    md.extend(f"- {line}" for line in unified.get("sample_lines", [])[:12])
    md.extend(["", "## ABC Sample Lines", ""])
    md.extend(f"- {line}" for line in abc.get("sample_lines", [])[:12])
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[done] side_by_side={side_path}", flush=True)
    print(f"[done] summary={md_path}", flush=True)


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / f"fair_30s_compare_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[start] output={out_dir}", flush=True)

    video_path, sample_path, frames, elapsed, sample_index = capture_video(args, out_dir)
    capture_info = {
        "video": str(video_path),
        "sample_raw": str(sample_path),
        "frames": frames,
        "elapsed_seconds": elapsed,
        "capture_fps": frames / max(elapsed, 1e-6),
        "sample_second": args.sample_second,
        "sample_index": sample_index,
        "device": args.device,
        "camera": args.camera,
        "camera_backend": args.camera_backend,
        "width": args.width,
        "height": args.height,
    }

    unified = benchmark_unified(video_path, sample_index, args)
    abc = benchmark_abc(video_path, sample_index, args)
    save_outputs(out_dir, capture_info, unified, abc)


if __name__ == "__main__":
    main()
