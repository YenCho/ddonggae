from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from realtime_seg_cam import (
    detection_summaries,
    draw_confidence_brightness,
    has_segmentation_masks,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_MODEL_EXTENSIONS = {".pt", ".onnx", ".engine"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO segmentation on every image in a folder with .pt, .onnx, or .engine."
    )
    parser.add_argument(
        "--model",
        default="best.engine",
        help="Path to the YOLO segmentation model. Supports .pt, .onnx, and Ultralytics .engine.",
    )
    parser.add_argument(
        "--source",
        default="image",
        help="Folder that contains input images.",
    )
    parser.add_argument(
        "--output",
        default="runs/predict_seg_engine",
        help="Folder to save annotated images.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size. TensorRT engines may require the exported size.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device for .pt models, for example '0' or 'cpu'. .engine needs GPU/TensorRT.",
    )
    parser.add_argument(
        "--task",
        default="segment",
        choices=("detect", "segment"),
        help="Force the model task. Use segment for segmentation ONNX/engine models.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.8,
        help=(
            "Hide the lower-confidence detection when two boxes overlap by "
            "this ratio or more. Set 1.0 to disable."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print class id, class name, and confidence for each kept detection.",
    )
    return parser.parse_args()


def collect_images(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_model(model_path: Path, task: str) -> YOLO:
    suffix = model_path.suffix.lower()
    if suffix not in SUPPORTED_MODEL_EXTENSIONS:
        raise ValueError(
            f"Unsupported model type: {model_path.suffix}. Use .pt, .onnx, or .engine."
        )

    if suffix == ".engine":
        print("Using TensorRT engine. This requires a compatible NVIDIA GPU/TensorRT setup.")
    elif suffix == ".onnx":
        print("Using ONNX model. If masks are missing, check that it was exported from a segmentation model.")

    try:
        return YOLO(str(model_path), task=task)
    except Exception as exc:
        if suffix == ".engine":
            raise RuntimeError(
                "Could not load the TensorRT .engine file. Make sure it was exported "
                "by Ultralytics on a compatible environment and that TensorRT/CUDA are available."
            ) from exc
        raise


def predict_image(model: YOLO, frame, args: argparse.Namespace):
    predict_kwargs = {
        "source": frame,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "verbose": False,
    }

    if args.device is not None:
        predict_kwargs["device"] = args.device

    try:
        return model.predict(**predict_kwargs)
    except Exception as exc:
        if Path(args.model).suffix.lower() == ".engine":
            raise RuntimeError(
                "TensorRT inference failed. If this engine was exported with a fixed image "
                "size, run again with the same --imgsz value used during export."
            ) from exc
        raise


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    source_dir = Path(args.source)
    output_dir = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path.resolve()}")
    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder not found: {source_dir.resolve()}")

    image_paths = collect_images(source_dir)
    if not image_paths:
        raise FileNotFoundError(f"No image files found in: {source_dir.resolve()}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(model_path, args.task)

    print(f"Model: {model_path}")
    print(f"Model names: {model.names}")
    print(f"Found {len(image_paths)} image(s).")
    print(f"Saving results to: {output_dir}")

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Skipped unreadable image: {image_path.name}")
            continue

        results = predict_image(model, frame, args)
        if args.task == "segment" and not has_segmentation_masks(results[0]):
            print(
                f"Warning: no segmentation mask returned for {image_path.name}. "
                "If boxes appear without masks, re-export from a segmentation .pt model."
            )

        annotated_frame = draw_confidence_brightness(
            frame,
            results[0],
            overlap_threshold=args.overlap,
        )

        save_path = output_dir / image_path.name
        cv2.imwrite(str(save_path), annotated_frame)
        print(f"Saved: {save_path}")

        if args.debug:
            summaries = detection_summaries(
                results[0],
                overlap_threshold=args.overlap,
            )
            if summaries:
                for summary in summaries:
                    print(f"  {summary}")
            else:
                print("  No detections.")

    print("Done.")


if __name__ == "__main__":
    main()
