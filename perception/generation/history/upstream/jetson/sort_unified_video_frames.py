from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the current A1 crop -> unified cube-face model on a video and "
            "save annotated frames/crops grouped by the final fruit decision."
        )
    )
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument(
        "--out",
        default="",
        help="Output report folder. Default: reports/unified_video_frame_sort_<timestamp>.",
    )
    parser.add_argument(
        "--a1-model",
        default="preferred-a1",
        help=(
            "Cube Detector (A1) model alias or path, e.g. cube-detector "
            "(legacy: preferred-a1)."
        ),
    )
    parser.add_argument(
        "--model",
        default="preferred-unified",
        help=(
            "Face Classifier (unified cube-face) model alias or path, "
            "e.g. face-classifier (legacy: preferred-unified)."
        ),
    )
    parser.add_argument("--device", default="cpu", help="Ultralytics device, e.g. cpu or 0.")
    parser.add_argument("--a1-imgsz", type=int, default=640)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--a1-conf", type=float, default=0.25)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--fruit-min-conf", type=float, default=0.45,
                        help="decide_cube fruit acceptance floor; weak fruit face -> inspect. 0 disables.")
    parser.add_argument("--crop-pad", type=float, default=0.18)
    parser.add_argument("--overlap", type=float, default=0.6)
    parser.add_argument("--mask-alpha", type=float, default=0.28)
    parser.add_argument("--target-shape", default="cube")
    parser.add_argument("--target-fruit", default=None)
    parser.add_argument(
        "--n-frames",
        type=int,
        default=0,
        help="Maximum number of processed frames. 0 means the whole video.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Process every Nth frame.",
    )
    parser.add_argument("--no-frames", action="store_true", help="Do not save full annotated frames.")
    parser.add_argument("--no-inputs", action="store_true", help="Do not save 224x224 unified inputs.")
    parser.add_argument("--no-raw-crops", action="store_true", help="Do not save raw A1 crops.")
    parser.add_argument("--copy-source", action="store_true", help="Copy the source video into the report.")
    return parser.parse_args()


def safe_name(text: object) -> str:
    return "".join(
        char if char.isalnum() or char in "._-" else "_"
        for char in str(text)
    ).strip("_") or "unknown"


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def open_video_unicode(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if capture.isOpened():
        return capture

    # Some OpenCV builds on Windows still stumble on non-ASCII paths.
    temp_path = Path(tempfile.gettempdir()) / f"abc_video_source{path.suffix or '.mp4'}"
    shutil.copy2(path, temp_path)
    capture = cv2.VideoCapture(str(temp_path))
    if capture.isOpened():
        return capture
    raise RuntimeError(f"Could not open video: {path}")


def resolve_model_text(model_text: str, *, unified_default: bool = False) -> str:
    text = str(model_text or "").strip()
    if unified_default and text in {"", "best.pt"}:
        text = "preferred-unified"

    # Primary alias names: cube-detector (old A1) / face-classifier (old
    # cube-face unified). Legacy names are added below and keep working.
    aliases = {
        "cube-detector": [
            Path("jetson") / "ABC_model" / "meta_v2_a1_objectseg" / "a1_yolo26s_seg_meta_v2_50000" / "weights" / "best.pt",
            Path("ABC_model") / "meta_v2_a1_objectseg" / "a1_yolo26s_seg_meta_v2_50000" / "weights" / "best.pt",
        ],
        "cube-detector-onnx": [
            Path("jetson") / "ABC_model" / "meta_v2_a1_objectseg" / "a1_yolo26s_seg_meta_v2_50000" / "weights" / "best.onnx",
            Path("ABC_model") / "meta_v2_a1_objectseg" / "a1_yolo26s_seg_meta_v2_50000" / "weights" / "best.onnx",
        ],
        "face-classifier": [
            Path("jetson") / "ABC_model" / "cube_face_unified" / "preferred_v2" / "weights" / "best.pt",
            Path("runs") / "segment" / "cube_face_unified_yolo26n_seg_hsv_pruned_coloroutlier_from_last_adamw_lr1e5_ft_v1" / "weights" / "best.pt",
        ],
        "face-classifier-onnx": [
            Path("jetson") / "ABC_model" / "cube_face_unified" / "preferred_v2" / "weights" / "best.onnx",
        ],
        "latest-unified": [
            Path("jetson") / "ABC_model" / "cube_face_unified" / "latest_experiment" / "weights" / "best.pt",
            Path("runs") / "segment" / "cube_face_unified_yolo26n_seg_hsv_pruned_coloroutlier_from_last_adamw_lr1e5_ft_v1" / "weights" / "best.pt",
        ],
    }
    # Legacy alias names kept for backward compatibility (same paths).
    aliases["preferred-a1"] = aliases["cube-detector"]
    aliases["preferred-a1-onnx"] = aliases["cube-detector-onnx"]
    aliases["preferred-unified"] = aliases["face-classifier"]
    aliases["preferred-unified-onnx"] = aliases["face-classifier-onnx"]

    if text in aliases:
        candidates = aliases[text]
    else:
        raw = Path(text).expanduser()
        candidates = [raw]
        if not raw.is_absolute():
            candidates.extend([Path.cwd() / raw, SCRIPT_DIR / raw, REPO_ROOT / raw])

    checked: list[Path] = []
    for candidate in candidates:
        checked.append(candidate)
        if candidate.exists():
            return str(candidate)

    checked_text = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(f"Model file not found for '{model_text}'. Checked:\n{checked_text}")


def bootstrap_runtime_helpers():
    sys.path.insert(0, str(SCRIPT_DIR))
    import realtime_seg_cam as rt
    from runtime_policy import (
        CUBE_TOO_FAR_COLOR_BGR,
        CUBE_TOO_FAR_IDENTITY,
        cube_is_too_far,
        cube_too_far_reason,
    )
    from abc_inference import (
        ACTION_COLORS,
        FRUIT_CLASSES,
        FaceEvidence,
        ObjectDecision,
        decide_cube,
    )

    rt.ACTION_COLORS = ACTION_COLORS
    rt.FRUIT_CLASSES = FRUIT_CLASSES
    rt.FaceEvidence = FaceEvidence
    rt.ObjectDecision = ObjectDecision
    rt.cv2 = cv2
    return rt, decide_cube, CUBE_TOO_FAR_COLOR_BGR, CUBE_TOO_FAR_IDENTITY, cube_is_too_far, cube_too_far_reason


def output_root_from_args(out_text: str) -> Path:
    if out_text:
        return Path(out_text)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("reports") / f"unified_video_frame_sort_{timestamp}"


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)

    (
        rt,
        decide_cube,
        cube_too_far_color,
        cube_too_far_identity,
        cube_is_too_far_fn,
        cube_too_far_reason_fn,
    ) = bootstrap_runtime_helpers()

    output_root = output_root_from_args(args.out)
    output_root.mkdir(parents=True, exist_ok=True)

    source_video = Path(args.video).expanduser()
    if not source_video.exists():
        raise FileNotFoundError(f"Video file not found: {source_video}")

    if args.copy_source:
        shutil.copy2(source_video, output_root / f"source{source_video.suffix or '.mp4'}")

    a1_model_path = resolve_model_text(args.a1_model)
    unified_model_path = resolve_model_text(args.model, unified_default=True)
    print(f"A1 model: {a1_model_path}")
    print(f"Unified model: {unified_model_path}")
    print(f"Video: {source_video}")
    print(f"Output: {output_root}")

    a1_model = YOLO(a1_model_path)
    unified_model = YOLO(unified_model_path)

    cap = open_video_unicode(source_video)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"Input video info: fps={fps:.3f}, frames={total_frames}")

    frame_csv_path = output_root / "frame_results.csv"
    object_csv_path = output_root / "object_results.csv"
    face_csv_path = output_root / "face_results.csv"
    summary: Counter[str] = Counter()

    with (
        frame_csv_path.open("w", newline="", encoding="utf-8") as frame_file,
        object_csv_path.open("w", newline="", encoding="utf-8") as object_file,
        face_csv_path.open("w", newline="", encoding="utf-8") as face_file,
    ):
        frame_writer = csv.writer(frame_file)
        object_writer = csv.writer(object_file)
        face_writer = csv.writer(face_file)
        frame_writer.writerow([
            "frame_index",
            "timestamp_s",
            "primary_category",
            "primary_identity",
            "primary_confidence",
            "cube_count",
        ])
        object_writer.writerow([
            "frame_index",
            "timestamp_s",
            "object_index",
            "category",
            "identity",
            "decision_confidence",
            "a1_confidence",
            "a1_box_xyxy",
            "crop_box_xyxy",
            "face_count",
            "face_counts",
            "faces",
        ])
        face_writer.writerow([
            "frame_index",
            "timestamp_s",
            "object_index",
            "label",
            "confidence",
            "box_xyxy",
            "visible_pixels",
        ])

        frame_index = -1
        processed = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1

            if frame_index % max(1, int(args.frame_step)) != 0:
                continue
            if args.n_frames > 0 and processed >= args.n_frames:
                break

            timestamp_s = frame_index / max(fps, 1e-6)
            height, width = frame.shape[:2]
            annotated = frame.copy()
            objects: list[tuple[str, str, float]] = []

            a1_result = a1_model.predict(
                source=frame,
                imgsz=args.a1_imgsz,
                conf=args.a1_conf,
                device=args.device,
                verbose=False,
            )[0]

            if a1_result.boxes is not None and len(a1_result.boxes) > 0:
                boxes = a1_result.boxes.xyxy.cpu().numpy()
                classes = a1_result.boxes.cls.cpu().numpy().astype(int)
                confidences = a1_result.boxes.conf.cpu().numpy()
                order = np.argsort(confidences)[::-1]

                crops: list[np.ndarray] = []
                crop_metas: list[dict[str, object]] = []
                for det_index in order:
                    class_name = rt.get_class_name(a1_result.names, int(classes[det_index]))
                    if class_name != "cube_like_object":
                        continue

                    box = boxes[det_index]
                    object_box = tuple(int(round(v)) for v in box)
                    too_far, size_metrics = cube_is_too_far_fn(object_box, frame.shape[:2])
                    if too_far:
                        identity = str(cube_too_far_identity)
                        category = safe_name(identity)
                        confidence = float(confidences[det_index])
                        objects.append((category, identity, confidence))
                        obj_x1, obj_y1, obj_x2, obj_y2 = object_box
                        cv2.rectangle(annotated, (obj_x1, obj_y1), (obj_x2, obj_y2), cube_too_far_color, 3)
                        rt.draw_small_label(
                            annotated,
                            f"{identity} {confidence:.2f}",
                            (obj_x1, max(0, obj_y1 - 22)),
                            cube_too_far_color,
                            scale=0.58,
                        )
                        object_writer.writerow([
                            frame_index,
                            f"{timestamp_s:.4f}",
                            len(objects) - 1,
                            category,
                            identity,
                            f"{confidence:.6f}",
                            f"{confidence:.6f}",
                            json.dumps(list(object_box), ensure_ascii=False),
                            "",
                            0,
                            "{}",
                            cube_too_far_reason_fn(size_metrics),
                        ])
                        continue

                    crop_box = rt.expand_xyxy_for_crop(box, width, height, args.crop_pad)
                    if crop_box is None:
                        continue

                    x1, y1, x2, y2 = crop_box
                    raw_crop = frame[y1 : y2 + 1, x1 : x2 + 1]
                    if raw_crop.size == 0:
                        continue

                    face_input = cv2.resize(
                        raw_crop,
                        (int(args.imgsz), int(args.imgsz)),
                        interpolation=cv2.INTER_AREA,
                    )
                    crops.append(face_input)
                    crop_metas.append({
                        "object_box": object_box,
                        "crop_box": crop_box,
                        "raw_crop": raw_crop,
                        "face_input": face_input,
                        "a1_confidence": float(confidences[det_index]),
                    })

                if crops:
                    face_results = unified_model.predict(
                        source=crops,
                        imgsz=args.imgsz,
                        conf=args.conf,
                        device=args.device,
                        verbose=False,
                    )
                    for object_index, (face_result, meta) in enumerate(zip(face_results, crop_metas)):
                        crop_box = meta["crop_box"]
                        raw_crop = meta["raw_crop"]
                        face_input = meta["face_input"]
                        object_box = meta["object_box"]
                        a1_confidence = float(meta["a1_confidence"])
                        x1, y1, _x2, _y2 = crop_box
                        crop_h, crop_w = raw_crop.shape[:2]
                        result_h, result_w = rt.result_frame_shape(face_result)
                        if result_h <= 0 or result_w <= 0:
                            result_h = result_w = int(args.imgsz)

                        _lines, faces = rt.draw_result_at_offset(
                            annotated,
                            face_result,
                            offset=(x1, y1),
                            coord_scale=(crop_w / max(result_w, 1), crop_h / max(result_h, 1)),
                            prefix=f"o{object_index} ",
                            overlap_threshold=args.overlap,
                            mask_alpha=args.mask_alpha,
                        )
                        decision = decide_cube(
                            faces,
                            target_shape=args.target_shape,
                            target_fruit=args.target_fruit,
                            fruit_min_conf=args.fruit_min_conf,
                        )
                        decision_label, decision_confidence = rt.decision_label_with_confidence(
                            decision,
                            faces,
                            object_confidence=a1_confidence,
                        )
                        identity = str(decision.identity)
                        category = identity.split(":", 1)[1] if identity.startswith("fruit_cube:") else identity
                        category = safe_name(category)
                        objects.append((category, identity, decision_confidence))

                        obj_x1, obj_y1, obj_x2, obj_y2 = object_box
                        color = rt.decision_color(decision)
                        cv2.rectangle(annotated, (obj_x1, obj_y1), (obj_x2, obj_y2), color, 3)
                        rt.draw_small_label(
                            annotated,
                            decision_label,
                            (obj_x1, max(0, obj_y1 - 22)),
                            color,
                            scale=0.58,
                        )

                        base_name = (
                            f"frame_{frame_index:05d}_t{timestamp_s:07.2f}_"
                            f"o{object_index}_{safe_name(identity)}_conf{decision_confidence:.2f}"
                        )
                        if not args.no_inputs:
                            imwrite_unicode(
                                output_root / "unified_inputs_by_decision" / category / f"{base_name}_input{args.imgsz}.jpg",
                                face_input,
                            )
                        if not args.no_raw_crops:
                            imwrite_unicode(
                                output_root / "raw_crops_by_decision" / category / f"{base_name}_rawcrop.jpg",
                                raw_crop,
                            )

                        face_counts = Counter(face.label for face in faces)
                        face_text = ";".join(f"{face.label}:{face.confidence:.3f}" for face in faces)
                        object_writer.writerow([
                            frame_index,
                            f"{timestamp_s:.4f}",
                            object_index,
                            category,
                            identity,
                            f"{decision_confidence:.6f}",
                            f"{a1_confidence:.6f}",
                            json.dumps(list(object_box), ensure_ascii=False),
                            json.dumps(list(crop_box), ensure_ascii=False),
                            len(faces),
                            json.dumps(dict(face_counts), ensure_ascii=False),
                            face_text,
                        ])
                        for face in faces:
                            face_writer.writerow([
                                frame_index,
                                f"{timestamp_s:.4f}",
                                object_index,
                                face.label,
                                f"{face.confidence:.6f}",
                                json.dumps(face.box_xyxy, ensure_ascii=False),
                                int(face.visible_pixels),
                            ])

            if objects:
                primary_category, primary_identity, primary_confidence = objects[0]
            else:
                primary_category, primary_identity, primary_confidence = "no_cube", "no_cube", 0.0

            summary[primary_category] += 1
            frame_writer.writerow([
                frame_index,
                f"{timestamp_s:.4f}",
                primary_category,
                primary_identity,
                f"{primary_confidence:.6f}",
                len(objects),
            ])

            if not args.no_frames:
                frame_name = f"frame_{frame_index:05d}_t{timestamp_s:07.2f}_{safe_name(primary_identity)}.jpg"
                imwrite_unicode(output_root / "frames_by_decision" / primary_category / frame_name, annotated)

            processed += 1
            if processed % 25 == 0:
                print(f"Processed {processed} frame(s)...")

    cap.release()

    summary_json = {
        "video": str(source_video),
        "output": str(output_root),
        "processed_frames": processed,
        "frame_step": int(args.frame_step),
        "n_frames_limit": int(args.n_frames),
        "fps": fps,
        "counts": dict(summary),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Done.")
    print(json.dumps(summary_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
