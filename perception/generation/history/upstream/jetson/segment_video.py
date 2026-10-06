from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from abc_runtime_presets import add_runtime_preset_arg, apply_runtime_preset
from realtime_seg_cam import draw_confidence_brightness, has_segmentation_masks


SUPPORTED_MODEL_EXTENSIONS = {".pt", ".onnx", ".engine"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO segmentation on every frame of a video and save an annotated mp4."
    )
    parser.add_argument(
        "--pipeline",
        choices=("yolo", "abc"),
        default="yolo",
        help="Use the original single-YOLO output or the A1/A2/B/C cascade.",
    )
    add_runtime_preset_arg(parser)
    parser.add_argument(
        "--model",
        default="best.pt",
        help="Path to the YOLO segmentation model. Supports .pt, .onnx, and .engine.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Input video path, for example input.mp4.",
    )
    parser.add_argument(
        "--output",
        default="runs/predict_seg_video/output.mp4",
        help="Output mp4 path.",
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
        help="Inference image size.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example '0' or 'cpu'. .engine needs GPU/TensorRT.",
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
        "--max-frames",
        type=int,
        default=0,
        help="Limit processed frames for testing. 0 means process the whole video.",
    )
    parser.add_argument(
        "--a1-model",
        default=None,
        help="ABC mode: path to A1 object segmentation weight.",
    )
    parser.add_argument(
        "--a2-model",
        default=None,
        help="ABC mode: path to A2 visible face segmentation weight.",
    )
    parser.add_argument(
        "--b-model",
        default=None,
        help="ABC mode: path to B TinyQuadNet weight.",
    )
    parser.add_argument(
        "--c-model",
        default=None,
        help="ABC mode: path to C face classifier weight. Defaults to the 50000-run best.pt.",
    )
    parser.add_argument(
        "--a2-conf",
        type=float,
        default=0.20,
        help="ABC mode: A2 face confidence threshold.",
    )
    parser.add_argument(
        "--c-conf",
        type=float,
        default=0.55,
        help="ABC mode: minimum C classifier confidence before marking a face unknown.",
    )
    parser.add_argument(
        "--a2-imgsz",
        type=int,
        default=224,
        help="ABC mode: A2 image size. Use 0 to reuse --imgsz.",
    )
    parser.add_argument(
        "--target-shape",
        default="",
        choices=("", "cube", "octahedron", "dodecahedron", "icosahedron"),
        help="ABC mode: rulebook set-1 target shape for pickup decisions.",
    )
    parser.add_argument(
        "--target-fruit",
        default="",
        choices=("", "apple", "orange", "banana", "pineapple"),
        help="ABC mode: rulebook set-2 target fruit for pickup decisions.",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="ABC mode: save frame decisions to output path with .jsonl suffix.",
    )
    parser.add_argument(
        "--abc-overlay",
        choices=("raw", "decision"),
        default="raw",
        help="ABC mode: output video overlay. raw shows per-frame A1/A2/C outputs; decision shows rulebook action.",
    )
    parser.add_argument(
        "--min-face-object-overlap",
        type=float,
        default=0.45,
        help="ABC mode: reject A2 face candidates if this fraction is not inside the A1 object mask.",
    )
    parser.add_argument(
        "--min-face-pixels",
        type=int,
        default=120,
        help="ABC mode: minimum A2 visible-face mask pixels, matching the dataset exporter default.",
    )
    parser.add_argument(
        "--b-refine-passes",
        type=int,
        choices=(1, 2),
        default=1,
        help="ABC mode: run B once by default; use 2 for experimental full-quad bbox bootstrap.",
    )
    parser.add_argument(
        "--refine-quads",
        choices=("none", "pose", "pose_fruit", "pose_fast_iou"),
        default="pose",
        help="ABC mode: refine B quads with projective cube geometry before C classification.",
    )
    parser.add_argument(
        "--no-blacken-c-occlusion",
        dest="blacken_c_occlusion",
        action="store_false",
        default=True,
        help="ABC mode: do not blacken inferred occluded regions inside the C face warp.",
    )
    parser.add_argument(
        "--c-occlusion-visible-ratio",
        type=float,
        default=0.92,
        help="ABC mode: blacken hidden C face area when visible/full quad ratio is below this value.",
    )
    parser.add_argument(
        "--c-max-black-fraction",
        type=float,
        default=0.60,
        help="ABC mode: mark a C crop unknown if blackened pixels exceed this fraction.",
    )
    parser.add_argument("--c-min-quad-area", type=float, default=400.0, help="ABC mode: C quad area filter.")
    parser.add_argument("--c-min-quad-side", type=float, default=10.0, help="ABC mode: C quad side filter.")
    parser.add_argument("--c-max-quad-aspect", type=float, default=14.0, help="ABC mode: C quad aspect filter.")
    parser.add_argument("--c-min-visible-pixels", type=int, default=400, help="ABC mode: C visible-pixel filter.")
    parser.add_argument("--c-min-visible-ratio", type=float, default=0.40, help="ABC mode: C visible/full face ratio filter.")
    parser.add_argument("--c-onnx-model", default="", help="ABC mode: optional ONNX Runtime model for C classifier.")
    parser.add_argument(
        "--c-onnx-provider",
        choices=("cuda", "tensorrt", "cpu"),
        default="cuda",
        help="ABC mode: ONNX Runtime provider preference for --c-onnx-model.",
    )
    parser.add_argument(
        "--c-warp-size",
        type=int,
        default=224,
        help="ABC mode: perspective-warped C crop size before C model resize.",
    )
    parser.add_argument(
        "--no-a1-object-mask",
        dest="use_a1_object_mask",
        action="store_false",
        default=True,
        help="ABC mode: skip A1 mask extraction and use object boxes only for A2 crops/overlap filtering.",
    )
    parser.add_argument("--batch-b-across-objects", action="store_true", help="ABC mode: batch all B face-quad calls once per frame.")
    parser.add_argument("--batch-c-across-objects", action="store_true", help="ABC mode: batch all C face-classifier calls once per frame.")
    parser.add_argument(
        "--c-max-visible-to-full-ratio",
        type=float,
        default=1.15,
        help="ABC mode: C inconsistent visible/full ratio filter.",
    )
    return parser.parse_args()


def load_model(model_path: Path, task: str) -> YOLO:
    suffix = model_path.suffix.lower()
    if suffix not in SUPPORTED_MODEL_EXTENSIONS:
        raise ValueError(
            f"Unsupported model type: {model_path.suffix}. Use .pt, .onnx, or .engine."
        )

    if suffix == ".engine":
        print("Using TensorRT engine. This requires a compatible NVIDIA GPU/TensorRT setup.")
    elif suffix == ".onnx":
        print("Using ONNX model.")

    return YOLO(str(model_path), task=task)


def main() -> None:
    args = parse_args()
    apply_runtime_preset(args)

    if args.pipeline == "abc":
        run_abc_video(args)
        return

    model_path = Path(args.model)
    source_path = Path(args.source)
    output_path = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path.resolve()}")
    if not source_path.exists():
        raise FileNotFoundError(f"Video file not found: {source_path.resolve()}")

    model = load_model(model_path, args.task)

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source_path.resolve()}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path.resolve()}")

    print(f"Model: {model_path}")
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    print(f"Video: {width}x{height}, {fps:.2f} FPS, {total_frames} frame(s)")

    frame_index = 0
    warned_no_masks = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )

            if args.task == "segment" and not warned_no_masks:
                if results and not has_segmentation_masks(results[0]):
                    print(
                        "Warning: segmentation masks are missing. "
                        "Check that the model was exported from a YOLO segmentation model."
                    )
                    warned_no_masks = True

            annotated_frame = draw_confidence_brightness(
                frame,
                results[0],
                overlap_threshold=args.overlap,
            )
            writer.write(annotated_frame)

            frame_index += 1
            if frame_index % 30 == 0:
                if total_frames > 0:
                    print(f"Processed {frame_index}/{total_frames} frame(s)")
                else:
                    print(f"Processed {frame_index} frame(s)")

            if args.max_frames > 0 and frame_index >= args.max_frames:
                break
    finally:
        cap.release()
        writer.release()

    print(f"Done. Saved: {output_path}")


def run_abc_video(args: argparse.Namespace) -> None:
    from abc_inference import ABCPipeline, draw_frame_output, draw_raw_model_output

    source_path = Path(args.source)
    output_path = Path(args.output)

    if not source_path.exists():
        raise FileNotFoundError(f"Video file not found: {source_path.resolve()}")

    pipeline = ABCPipeline(
        a1_model=args.a1_model,
        a2_model=args.a2_model,
        b_model=args.b_model,
        c_model=args.c_model,
        c_onnx_model=args.c_onnx_model or None,
        c_onnx_provider=args.c_onnx_provider,
        device=args.device,
        a1_conf=args.conf,
        a2_conf=args.a2_conf,
        c_conf=args.c_conf,
        imgsz=args.imgsz,
        a2_imgsz=args.imgsz if args.a2_imgsz == 0 else args.a2_imgsz,
        b_refine_passes=args.b_refine_passes,
        min_face_pixels=args.min_face_pixels,
        min_face_object_overlap=args.min_face_object_overlap,
        refine_quads=args.refine_quads,
        blacken_c_occluded_face_area=args.blacken_c_occlusion,
        c_occlusion_visible_ratio=args.c_occlusion_visible_ratio,
        c_min_quad_area=args.c_min_quad_area,
        c_min_quad_side=args.c_min_quad_side,
        c_max_quad_aspect=args.c_max_quad_aspect,
        c_min_visible_pixels=args.c_min_visible_pixels,
        c_min_visible_ratio=args.c_min_visible_ratio,
        c_warp_size=args.c_warp_size,
        c_max_visible_to_full_ratio=args.c_max_visible_to_full_ratio,
        c_max_black_fraction=args.c_max_black_fraction,
        target_shape=args.target_shape,
        target_fruit=args.target_fruit,
        use_a1_object_mask=args.use_a1_object_mask,
        batch_b_across_objects=args.batch_b_across_objects,
        batch_c_across_objects=args.batch_c_across_objects,
    )

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source_path.resolve()}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path.resolve()}")

    jsonl_file = None
    jsonl_path = output_path.with_suffix(".jsonl")
    if args.save_json:
        jsonl_file = jsonl_path.open("w", encoding="utf-8")

    print(f"ABC A1: {pipeline.a1_path}")
    print(f"ABC A2: {pipeline.a2_path}")
    print(f"ABC B:  {pipeline.b_path}")
    print(f"ABC C:  {pipeline.c_path}")
    print(f"ABC quad refine: {pipeline.refine_quads}")
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    if args.save_json:
        print(f"JSONL: {jsonl_path}")
    print(f"Video: {width}x{height}, {fps:.2f} FPS, {total_frames} frame(s)")

    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            output = pipeline.process_frame(frame)
            if args.abc_overlay == "decision":
                annotated_frame = draw_frame_output(frame, output)
            else:
                annotated_frame = draw_raw_model_output(frame, output)
            writer.write(annotated_frame)

            if jsonl_file is not None:
                payload = {"frame_index": frame_index, **output.to_dict()}
                jsonl_file.write(json.dumps(payload, ensure_ascii=False))
                jsonl_file.write("\n")

            frame_index += 1
            if frame_index % 30 == 0:
                if total_frames > 0:
                    print(f"Processed {frame_index}/{total_frames} frame(s)")
                else:
                    print(f"Processed {frame_index} frame(s)")

            if args.max_frames > 0 and frame_index >= args.max_frames:
                break
    finally:
        cap.release()
        writer.release()
        if jsonl_file is not None:
            jsonl_file.close()

    print(f"Done. Saved: {output_path}")


if __name__ == "__main__":
    main()
