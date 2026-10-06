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
from abc_runtime_presets import add_runtime_preset_arg, apply_runtime_preset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO segmentation on every image in a folder."
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
        help="Path to the YOLO segmentation model.",
    )
    parser.add_argument(
        "--source",
        default="image",
        help="Folder that contains input images.",
    )
    parser.add_argument(
        "--output",
        default="runs/predict_seg_images",
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
        help="Inference image size.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example '0' for CUDA GPU or 'cpu'.",
    )
    parser.add_argument(
        "--task",
        default="segment",
        choices=("detect", "segment"),
        help="Force the model task. Use segment for segmentation ONNX/engine models.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print class id, class name, and confidence for each detection.",
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
        help="ABC mode: save one JSON decision file beside each annotated image.",
    )
    parser.add_argument(
        "--image-layout",
        choices=("stages", "final", "both"),
        default="both",
        help="ABC mode: save final overlay, model-by-model panels, or both.",
    )
    parser.add_argument(
        "--abc-overlay",
        choices=("raw", "decision"),
        default="raw",
        help="ABC mode: final overlay style. raw shows per-frame A1/A2/C outputs; decision shows rulebook action.",
    )
    parser.add_argument(
        "--stage-panel-width",
        type=int,
        default=520,
        help="ABC mode: width of each model output panel in the stages image.",
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


def collect_images(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> None:
    args = parse_args()
    apply_runtime_preset(args)

    if args.pipeline == "abc":
        run_abc_images(args)
        return

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
    model = YOLO(str(model_path), task=args.task)

    print(f"Model names: {model.names}")
    print(f"Found {len(image_paths)} image(s).")
    print(f"Saving results to: {output_dir}")

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Skipped unreadable image: {image_path.name}")
            continue

        results = model.predict(
            source=frame,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )

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


def run_abc_images(args: argparse.Namespace) -> None:
    from abc_inference import (
        ABCPipeline,
        draw_frame_output,
        draw_model_stage_grid,
        draw_raw_model_output,
        raw_model_summary,
    )

    source_dir = Path(args.source)
    output_dir = Path(args.output)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder not found: {source_dir.resolve()}")

    image_paths = collect_images(source_dir)
    if not image_paths:
        raise FileNotFoundError(f"No image files found in: {source_dir.resolve()}")

    output_dir.mkdir(parents=True, exist_ok=True)
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

    print(f"ABC A1: {pipeline.a1_path}")
    print(f"ABC A2: {pipeline.a2_path}")
    print(f"ABC B:  {pipeline.b_path}")
    print(f"ABC C:  {pipeline.c_path}")
    print(f"ABC quad refine: {pipeline.refine_quads}")
    print(f"Found {len(image_paths)} image(s).")
    print(f"Saving results to: {output_dir}")

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Skipped unreadable image: {image_path.name}")
            continue

        output = pipeline.process_frame(frame)
        if args.abc_overlay == "decision":
            final_frame = draw_frame_output(frame, output)
        else:
            final_frame = draw_raw_model_output(frame, output)
        stages_frame = None
        if args.image_layout in ("stages", "both"):
            stages_frame = draw_model_stage_grid(
                frame,
                output,
                panel_width=args.stage_panel_width,
            )
        annotated_frame = stages_frame if args.image_layout == "stages" else final_frame
        save_path = output_dir / image_path.name
        cv2.imwrite(str(save_path), annotated_frame)
        print(f"Saved: {save_path}")

        if args.image_layout == "both" and stages_frame is not None:
            stages_dir = output_dir / "model_stages"
            stages_dir.mkdir(parents=True, exist_ok=True)
            stages_path = stages_dir / image_path.name
            cv2.imwrite(str(stages_path), stages_frame)
            print(f"Saved model stages: {stages_path}")

        if args.save_json:
            json_path = output_dir / f"{image_path.stem}.json"
            json_path.write_text(output.to_json(indent=2), encoding="utf-8")

        if args.debug:
            summaries = raw_model_summary(output)
            if summaries:
                for summary in summaries:
                    print(f"  {summary}")
            else:
                print("  No ABC objects.")

    print("Done.")


if __name__ == "__main__":
    main()
