from __future__ import annotations

import argparse
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np

from abc_runtime_presets import add_runtime_preset_arg, apply_runtime_preset

try:
    from .runtime_policy import (
        CUBE_TOO_FAR_COLOR_BGR,
        CUBE_TOO_FAR_IDENTITY,
        cube_is_too_far,
        cube_too_far_reason,
    )
except ImportError:
    from runtime_policy import (
        CUBE_TOO_FAR_COLOR_BGR,
        CUBE_TOO_FAR_IDENTITY,
        cube_is_too_far,
        cube_too_far_reason,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Primary alias names follow the documented model names (readme.md 용어 사전):
# "cube-detector" = Cube Detector (old A1) and "face-classifier" = Face
# Classifier (old cube-face unified). Legacy alias names are registered below
# and keep working for backward compatibility.
MODEL_ALIASES: dict[str, list[Path]] = {
    "cube-detector": [
        SCRIPT_DIR / "ABC_model" / "meta_v2_a1_objectseg" / "a1_yolo26s_seg_meta_v2_50000" / "weights" / "best.pt",
        REPO_ROOT / "ABC_model" / "meta_v2_a1_objectseg" / "a1_yolo26s_seg_meta_v2_50000" / "weights" / "best.pt",
        REPO_ROOT.parent / "ABC_model" / "meta_v2_a1_objectseg" / "a1_yolo26s_seg_meta_v2_50000" / "weights" / "best.pt",
    ],
    "cube-detector-onnx": [
        SCRIPT_DIR / "ABC_model" / "meta_v2_a1_objectseg" / "a1_yolo26s_seg_meta_v2_50000" / "weights" / "best.onnx",
        REPO_ROOT / "ABC_model" / "meta_v2_a1_objectseg" / "a1_yolo26s_seg_meta_v2_50000" / "weights" / "best.onnx",
        REPO_ROOT.parent / "ABC_model" / "meta_v2_a1_objectseg" / "a1_yolo26s_seg_meta_v2_50000" / "weights" / "best.onnx",
    ],
    "face-classifier": [
        SCRIPT_DIR / "ABC_model" / "cube_face_unified" / "preferred_v2" / "weights" / "best.pt",
        REPO_ROOT / "runs" / "segment" / "cube_face_unified_yolo26n_seg_hsv_pruned_coloroutlier_from_last_adamw_lr1e5_ft_v1" / "weights" / "best.pt",
    ],
    "face-classifier-onnx": [
        SCRIPT_DIR / "ABC_model" / "cube_face_unified" / "preferred_v2" / "weights" / "best.onnx",
    ],
    "latest-unified": [
        SCRIPT_DIR / "ABC_model" / "cube_face_unified" / "latest_experiment" / "weights" / "best.pt",
        REPO_ROOT / "runs" / "segment" / "cube_face_unified_yolo26n_seg_hsv_pruned_coloroutlier_from_last_adamw_lr1e5_ft_v1" / "weights" / "best.pt",
    ],
}

# Legacy alias name -> primary alias name. Both spellings resolve to the same
# weight paths; legacy names stay accepted for backward compatibility.
LEGACY_MODEL_ALIASES: dict[str, str] = {
    "preferred-a1": "cube-detector",
    "preferred-a1-onnx": "cube-detector-onnx",
    "preferred-unified": "face-classifier",
    "preferred-unified-onnx": "face-classifier-onnx",
}
for _legacy_alias, _primary_alias in LEGACY_MODEL_ALIASES.items():
    MODEL_ALIASES[_legacy_alias] = MODEL_ALIASES[_primary_alias]

# Accepted legacy aliases that have no renamed primary counterpart.
LEGACY_ONLY_ALIASES: frozenset[str] = frozenset({"latest-unified"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO segmentation in real time from a webcam."
    )
    parser.add_argument(
        "--pipeline",
        choices=("yolo", "abc", "unified"),
        default="yolo",
        help=(
            "Use the original single-YOLO output, the A1/A2/B/C cascade, "
            "or the cube-face unified YOLO preview."
        ),
    )
    add_runtime_preset_arg(parser)
    parser.add_argument(
        "--model",
        default="best.pt",
        help=(
            "Path to the YOLO segmentation model. In --pipeline unified, "
            "the default best.pt is treated as the face-classifier alias "
            "(legacy name: preferred-unified)."
        ),
    )
    parser.add_argument(
        "--list-model-aliases",
        action="store_true",
        help=(
            "Print built-in model aliases such as cube-detector and "
            "face-classifier (legacy: preferred-a1, preferred-unified) and exit."
        ),
    )
    parser.add_argument(
        "--camera",
        default="0",
        help="Camera index, video path, or GStreamer pipeline. Try 1 or 2 if camera 0 does not open.",
    )
    parser.add_argument(
        "--camera-backend",
        choices=("auto", "any", "dshow", "v4l2", "gstreamer"),
        default="auto",
        help="Camera backend. Use v4l2 for USB cameras on Jetson/Linux, gstreamer for CSI pipelines.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--fruit-min-conf",
        type=float,
        default=0.45,
        help=(
            "decide_cube fruit acceptance floor: a unified fruit face below this confidence "
            "is treated as unknown (cube -> inspect) instead of a fruit vote, so weak "
            "detections on blank/sliver crops cannot flip a cube to a fruit identity. "
            "Set 0 to disable. Provisional; tune with real robot-camera footage."
        ),
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
        "--width",
        type=int,
        default=1280,
        help="Requested camera width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Requested camera height.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the annotated webcam output to runs/realtime_seg_cam.mp4.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not call cv2.imshow(). Useful for SSH/headless CPU or Jetson tests.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames. 0 means run until q/ESC/Ctrl+C.",
    )
    parser.add_argument(
        "--print-timing",
        action="store_true",
        help="ABC/unified mode: print per-frame timing so FPS bottlenecks are visible.",
    )
    parser.add_argument(
        "--print-model-output",
        action="store_true",
        help="ABC/unified mode: print raw model outputs every frame, without temporal voting.",
    )
    parser.add_argument(
        "--model-output-every",
        type=int,
        default=1,
        help="Print raw model output every N frames when --print-model-output is enabled.",
    )
    parser.add_argument(
        "--abc-overlay",
        choices=("raw", "decision"),
        default="raw",
        help="ABC mode: show per-frame raw model outputs or the rulebook decision overlay.",
    )
    parser.add_argument(
        "--timing-every",
        type=int,
        default=30,
        help="Print timing every N frames when --print-timing is enabled.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.6,
        help=(
            "Hide the lower-confidence detection when two boxes overlap by "
            "this ratio or more. Set 1.0 to disable."
        ),
    )
    parser.add_argument(
        "--a1-model",
        default=None,
        help=(
            "ABC/unified mode: path or alias for the Cube Detector (A1) object "
            "segmentation weight, e.g. cube-detector (legacy: preferred-a1)."
        ),
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
        "--show-c-inputs",
        action="store_true",
        help="ABC mode: show warped face crops sent to the C classifier beside the webcam view.",
    )
    parser.add_argument(
        "--no-model-output-panel",
        dest="show_model_output_panel",
        action="store_false",
        default=True,
        help="ABC/unified mode: hide the right-side raw per-frame model output text panel.",
    )
    parser.add_argument(
        "--model-output-panel-width",
        type=int,
        default=420,
        help="ABC/unified mode: side panel width for raw per-frame model output text.",
    )
    parser.add_argument(
        "--unified-mask-alpha",
        type=float,
        default=0.52,
        help="Unified mode: segmentation mask overlay opacity.",
    )
    parser.add_argument(
        "--unified-a1-imgsz",
        type=int,
        default=640,
        help="Unified mode: A1 full-frame image size before cube crop extraction.",
    )
    parser.add_argument(
        "--unified-a1-conf",
        type=float,
        default=0.25,
        help="Unified mode: A1 confidence threshold. --conf is used for the face model inside A1 crops.",
    )
    parser.add_argument(
        "--unified-crop-pad",
        type=float,
        default=0.18,
        help="Unified mode: padding ratio around each A1 cube_like_object box before face-model inference.",
    )
    parser.add_argument(
        "--c-input-panel-width",
        type=int,
        default=360,
        help="ABC mode: side panel width for C classifier inputs.",
    )
    parser.add_argument(
        "--c-input-tile-size",
        type=int,
        default=132,
        help="ABC mode: tile size for each C classifier input crop.",
    )
    parser.add_argument(
        "--min-face-object-overlap",
        type=float,
        default=0.45,
        help="ABC mode: reject A2 face candidates if this fraction is not inside the A1 object mask.",
    )
    parser.add_argument(
        "--no-a1-object-mask",
        dest="use_a1_object_mask",
        action="store_false",
        default=True,
        help="ABC mode: skip A1 mask extraction and use object boxes only for A2 crops/overlap filtering.",
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
        "--display-scale",
        type=float,
        default=0.0,
        help="Scale displayed webcam window. 0 means auto-fit to --display-max-width/height.",
    )
    parser.add_argument(
        "--display-max-width",
        type=int,
        default=1280,
        help="Maximum displayed window width when --display-scale is 0.",
    )
    parser.add_argument(
        "--display-max-height",
        type=int,
        default=900,
        help="Maximum displayed window height when --display-scale is 0.",
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
    parser.add_argument("--trace-torch-models", action="store_true", help="ABC mode: TorchScript trace B/C models at startup.")
    parser.add_argument("--yolo-half", action="store_true", help="ABC mode: run A1/A2 YOLO inference with FP16 on CUDA.")
    parser.add_argument("--batch-b-across-objects", action="store_true", help="ABC mode: batch all B face-quad calls once per frame.")
    parser.add_argument("--batch-c-across-objects", action="store_true", help="ABC mode: batch all C face-classifier calls once per frame.")
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
        help="ABC mode: perspective-warped C crop size before C model resize. Try 128 for speed.",
    )
    parser.add_argument(
        "--c-max-visible-to-full-ratio",
        type=float,
        default=1.15,
        help="ABC mode: C inconsistent visible/full ratio filter.",
    )
    return parser.parse_args()


def print_model_aliases() -> None:
    print("Model aliases:")
    for alias, candidates in MODEL_ALIASES.items():
        if alias in LEGACY_MODEL_ALIASES:
            label = f"{alias} (legacy; same as {LEGACY_MODEL_ALIASES[alias]})"
        elif alias in LEGACY_ONLY_ALIASES:
            label = f"{alias} (legacy)"
        else:
            label = alias
        print(f"  {label}")
        for candidate in candidates:
            exists = "OK" if candidate.exists() else "missing"
            print(f"    [{exists}] {candidate}")


def resolve_model_path(model_text: str, *, unified_default: bool = False) -> Path:
    text = str(model_text or "").strip()
    if unified_default and text in {"", "best.pt"}:
        text = "preferred-unified"

    if text in MODEL_ALIASES:
        candidates = MODEL_ALIASES[text]
    else:
        raw = Path(text).expanduser()
        candidates = [raw]
        if not raw.is_absolute():
            candidates.extend([
                Path.cwd() / raw,
                SCRIPT_DIR / raw,
                REPO_ROOT / raw,
            ])

    checked: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in checked:
            checked.append(resolved)
        if resolved.exists():
            return resolved

    checked_text = "\n".join(f"  - {path}" for path in checked)
    alias_hint = ""
    if text in MODEL_ALIASES:
        alias_hint = "\nIf this is a fresh clone, run `git lfs pull` from the repository root."
    raise FileNotFoundError(
        f"Model file not found for '{model_text}'. Checked:\n{checked_text}{alias_hint}"
    )


def camera_source(value: str):
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return text


def camera_backend_flag(name: str) -> int:
    if name == "any":
        return cv2.CAP_ANY
    if name == "dshow":
        return cv2.CAP_DSHOW
    if name == "v4l2":
        return cv2.CAP_V4L2
    if name == "gstreamer":
        return cv2.CAP_GSTREAMER
    if platform.system().lower().startswith("win"):
        return cv2.CAP_DSHOW
    return cv2.CAP_V4L2


def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    source = camera_source(args.camera)
    backend_name = args.camera_backend
    if backend_name == "auto":
        backend = camera_backend_flag("auto")
    else:
        backend = camera_backend_flag(backend_name)

    cap = cv2.VideoCapture(source, backend)
    if not cap.isOpened() and backend_name == "auto":
        cap.release()
        cap = cv2.VideoCapture(source, cv2.CAP_ANY)
    return cap


def draw_confidence_brightness(
    frame: np.ndarray,
    result,
    overlap_threshold: float = 0.6,
    mask_alpha: float | None = None,
) -> np.ndarray:
    annotated = frame.copy()
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return annotated

    names = result.names
    confidences = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    xyxy_boxes = boxes.xyxy.cpu().numpy().astype(int)

    mask_polygons = get_mask_polygons(result)
    mask_data = None
    if result.masks is not None and result.masks.data is not None:
        mask_data = result.masks.data.cpu().numpy()

    visible_indices = filter_overlapping_boxes(xyxy_boxes, confidences, overlap_threshold)

    for index in visible_indices:
        confidence = confidences[index]
        class_id = classes[index]
        xyxy = xyxy_boxes[index]
        confidence = float(confidence)
        brightness = int(np.clip(80 + confidence * 175, 80, 255))
        base_color = class_color(class_id)
        color = scale_color(base_color, brightness / 255.0)

        mask_area = get_mask_area(
            frame.shape[:2],
            index,
            mask_polygons,
            mask_data,
        )
        if mask_area is not None:
            if mask_alpha is None:
                alpha = float(np.clip(0.25 + confidence * 0.55, 0.25, 0.8))
            else:
                alpha = float(np.clip(mask_alpha, 0.0, 0.9))
            bright_layer = np.full_like(annotated, color)
            annotated[mask_area] = cv2.addWeighted(
                annotated[mask_area],
                1.0 - alpha,
                bright_layer[mask_area],
                alpha,
                0,
            )

            contours, _ = cv2.findContours(
                mask_area.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(annotated, contours, -1, color, 2)

        x1, y1, x2, y2 = xyxy
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        class_name = get_class_name(names, class_id)
        label = f"{class_name} {confidence:.2f}"
        text_size, baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2,
        )
        label_y = max(y1, text_size[1] + baseline + 4)
        cv2.rectangle(
            annotated,
            (x1, label_y - text_size[1] - baseline - 4),
            (x1 + text_size[0] + 8, label_y),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 4, label_y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated


def get_mask_polygons(result) -> list[np.ndarray]:
    if result.masks is None:
        return []

    polygons = getattr(result.masks, "xy", None)
    if polygons is None:
        return []

    return [
        polygon.astype(np.int32)
        for polygon in polygons
        if polygon is not None and len(polygon) >= 3
    ]


def get_mask_area(
    frame_shape: tuple[int, int],
    index: int,
    mask_polygons: list[np.ndarray],
    mask_data: np.ndarray | None,
) -> np.ndarray | None:
    height, width = frame_shape

    if index < len(mask_polygons):
        mask_area = np.zeros((height, width), dtype=np.uint8)
        polygon = mask_polygons[index].copy()
        polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
        cv2.fillPoly(mask_area, [polygon], 1)
        return mask_area.astype(bool)

    if mask_data is None or index >= len(mask_data):
        return None

    mask = mask_data[index]
    if mask.shape[:2] != frame_shape:
        mask = cv2.resize(
            mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

    return mask > 0.5


def has_segmentation_masks(result) -> bool:
    return result.masks is not None and result.masks.data is not None


def filter_overlapping_boxes(
    boxes: np.ndarray,
    confidences: np.ndarray,
    overlap_threshold: float,
) -> list[int]:
    if overlap_threshold >= 1.0 or len(boxes) <= 1:
        return list(range(len(boxes)))

    sorted_indices = np.argsort(confidences)[::-1]
    kept: list[int] = []

    for index in sorted_indices:
        if all(
            bbox_overlap_ratio(boxes[index], boxes[kept_index]) < overlap_threshold
            for kept_index in kept
        ):
            kept.append(int(index))

    return kept


def bbox_overlap_ratio(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    smaller_area = min(area_a, area_b)

    if smaller_area == 0:
        return 0.0

    return float(inter_area / smaller_area)


def get_class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def detection_summaries(result, overlap_threshold: float = 0.6) -> list[str]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    names = result.names
    confidences = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    xyxy_boxes = boxes.xyxy.cpu().numpy().astype(int)
    visible_indices = filter_overlapping_boxes(xyxy_boxes, confidences, overlap_threshold)

    return [
        f"id={class_id}, name={get_class_name(names, class_id)}, conf={confidence:.3f}"
        for class_id, confidence in zip(classes[visible_indices], confidences[visible_indices])
    ]


def result_frame_shape(result) -> tuple[int, int]:
    shape = getattr(result, "orig_shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[0]), int(shape[1])
    return 0, 0


def unified_detection_summaries(
    result,
    overlap_threshold: float = 0.6,
    max_items: int = 18,
) -> list[str]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return ["No detections."]

    names = result.names
    confidences = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    xyxy_boxes = boxes.xyxy.cpu().numpy().astype(int)
    visible_indices = filter_overlapping_boxes(xyxy_boxes, confidences, overlap_threshold)

    labels = [get_class_name(names, int(classes[index])) for index in visible_indices]
    counts = Counter(labels)
    count_text = ", ".join(
        f"{label}={count}" for label, count in sorted(counts.items())
    )
    lines = [f"detections: total={len(visible_indices)} {count_text}".strip()]

    frame_shape = result_frame_shape(result)
    mask_polygons = get_mask_polygons(result)
    mask_data = None
    if result.masks is not None and result.masks.data is not None:
        mask_data = result.masks.data.cpu().numpy()

    for rank, index in enumerate(visible_indices[:max_items]):
        class_id = int(classes[index])
        label = get_class_name(names, class_id)
        confidence = float(confidences[index])
        x1, y1, x2, y2 = [int(v) for v in xyxy_boxes[index]]
        mask_pixels = 0
        if frame_shape[0] > 0 and frame_shape[1] > 0:
            mask_area = get_mask_area(frame_shape, index, mask_polygons, mask_data)
            if mask_area is not None:
                mask_pixels = int(mask_area.sum())
        lines.append(
            f"#{rank} {label} conf={confidence:.3f} mask={mask_pixels} "
            f"box=[{x1},{y1},{x2},{y2}]"
        )

    remaining = len(visible_indices) - max_items
    if remaining > 0:
        lines.append(f"... {remaining} more detections")
    return lines


def panel_wrapped_lines(lines: list[str], max_chars: int) -> list[str]:
    wrapped: list[str] = []
    for line in lines:
        text = str(line)
        while len(text) > max_chars:
            split_at = text.rfind(" ", 0, max_chars)
            if split_at <= 0:
                split_at = max_chars
            wrapped.append(text[:split_at])
            text = text[split_at:].lstrip()
        wrapped.append(text)
    return wrapped


def draw_text_panel(title: str, lines: list[str], width: int, height: int) -> np.ndarray:
    width = max(220, int(width))
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width - 1, 44), (36, 36, 36), -1)
    cv2.putText(
        panel,
        title,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    max_chars = max(18, int((width - 24) / 8.5))
    y = 66
    for line in panel_wrapped_lines(lines, max_chars):
        if y > height - 14:
            cv2.putText(
                panel,
                "...",
                (12, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (80, 80, 80),
                1,
                cv2.LINE_AA,
            )
            break
        cv2.putText(
            panel,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (35, 35, 35),
            1,
            cv2.LINE_AA,
        )
        y += 20
    return panel


def append_unified_output_panel(
    frame: np.ndarray,
    result,
    *,
    panel_width: int,
    overlap_threshold: float,
) -> np.ndarray:
    lines = unified_detection_summaries(result, overlap_threshold=overlap_threshold)
    panel = draw_text_panel("Unified model output", lines, panel_width, frame.shape[0])
    return np.hstack([frame, panel])


def append_unified_crop_output_panel(
    frame: np.ndarray,
    lines: list[str],
    *,
    panel_width: int,
) -> np.ndarray:
    panel = draw_text_panel("A1 crop -> unified", lines, panel_width, frame.shape[0])
    return np.hstack([frame, panel])


def class_color(class_id: int) -> tuple[int, int, int]:
    palette = [
        (40, 220, 255),
        (80, 180, 60),
        (255, 120, 40),
        (70, 70, 255),
        (220, 80, 220),
        (255, 220, 40),
        (80, 255, 160),
        (200, 160, 255),
    ]
    return palette[class_id % len(palette)]


def face_color_for_label(label: str) -> tuple[int, int, int]:
    palette = {
        "apple": (40, 40, 230),
        "orange": (0, 150, 255),
        "banana": (40, 230, 255),
        "pineapple": (40, 180, 120),
        "plain": (230, 230, 230),
        "unknown": (200, 160, 255),
    }
    return palette.get(str(label).lower(), (255, 255, 255))


def decision_color(decision: ObjectDecision) -> tuple[int, int, int]:
    if decision.identity.startswith("fruit_cube:"):
        return face_color_for_label(decision.identity.split(":", 1)[1])
    if decision.identity == "plain_cube":
        return face_color_for_label("plain")
    return ACTION_COLORS.get(decision.action, (200, 160, 255))


def decision_confidence(
    decision: ObjectDecision,
    faces: list[FaceEvidence],
    *,
    object_confidence: float,
) -> float:
    identity = str(decision.identity)
    if identity.startswith("fruit_cube:"):
        fruit = identity.split(":", 1)[1].lower()
        scores = [float(face.confidence) for face in faces if face.label.lower() == fruit]
        if scores:
            return float(np.mean(scores))
    if identity == "conflicting_fruit_cube":
        scores = [
            float(face.confidence)
            for face in faces
            if face.label.lower() in FRUIT_CLASSES
        ]
        if scores:
            return float(max(scores))
    if identity in {"plain_cube", "blank_only_cube_ambiguous"}:
        scores = [float(face.confidence) for face in faces if face.label.lower() == "plain"]
        if scores:
            return float(np.mean(scores))
    if faces:
        return float(max(float(face.confidence) for face in faces))
    return float(object_confidence)


def decision_label_with_confidence(
    decision: ObjectDecision,
    faces: list[FaceEvidence],
    *,
    object_confidence: float,
) -> tuple[str, float]:
    score = decision_confidence(decision, faces, object_confidence=object_confidence)
    return f"{decision.identity} {score:.2f}", score


def scale_color(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(int(np.clip(channel * factor, 0, 255)) for channel in color)


def expand_xyxy_for_crop(
    box: np.ndarray,
    width: int,
    height: int,
    pad_ratio: float,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(1.0, x2 - x1 + 1.0)
    bh = max(1.0, y2 - y1 + 1.0)
    pad = max(bw, bh) * float(pad_ratio)
    x1 = max(0, int(np.floor(x1 - pad)))
    y1 = max(0, int(np.floor(y1 - pad)))
    x2 = min(width - 1, int(np.ceil(x2 + pad)))
    y2 = min(height - 1, int(np.ceil(y2 + pad)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def draw_small_label(
    image: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    color: tuple[int, int, int],
    *,
    scale: float = 0.52,
) -> None:
    x, y = anchor
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(th + baseline + 4, y)
    x = max(0, min(x, max(0, image.shape[1] - tw - 8)))
    cv2.rectangle(image, (x, y - th - baseline - 4), (x + tw + 8, y), color, -1)
    text_color = (0, 0, 0) if sum(color) > 360 else (255, 255, 255)
    cv2.putText(image, text, (x + 4, y - baseline - 2), font, scale, text_color, thickness, cv2.LINE_AA)


def offset_polygon(
    polygon: np.ndarray,
    offset: tuple[int, int],
    frame_shape: tuple[int, int],
    scale: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    height, width = frame_shape
    shifted = polygon.astype(np.float32).copy()
    shifted[:, 0] *= float(scale[0])
    shifted[:, 1] *= float(scale[1])
    shifted[:, 0] += int(offset[0])
    shifted[:, 1] += int(offset[1])
    shifted[:, 0] = np.clip(shifted[:, 0], 0, width - 1)
    shifted[:, 1] = np.clip(shifted[:, 1], 0, height - 1)
    return np.rint(shifted).astype(np.int32)


def draw_result_at_offset(
    image: np.ndarray,
    result,
    *,
    offset: tuple[int, int] = (0, 0),
    coord_scale: tuple[float, float] = (1.0, 1.0),
    prefix: str = "",
    overlap_threshold: float = 0.6,
    mask_alpha: float = 0.52,
) -> tuple[list[str], list[FaceEvidence]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return [], []

    names = result.names
    confidences = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    xyxy_boxes = boxes.xyxy.cpu().numpy().astype(int)
    visible_indices = filter_overlapping_boxes(xyxy_boxes, confidences, overlap_threshold)
    mask_polygons = get_mask_polygons(result)

    overlay = image.copy()
    lines: list[str] = []
    faces: list[FaceEvidence] = []
    ox, oy = offset
    sx, sy = float(coord_scale[0]), float(coord_scale[1])
    for index in visible_indices:
        confidence = float(confidences[index])
        class_id = int(classes[index])
        class_name = get_class_name(names, class_id)
        color = scale_color(face_color_for_label(class_name), float(np.clip(0.7 + confidence * 0.3, 0.7, 1.0)))
        segment_xy: list[list[float]]
        visible_pixels = 0

        if index < len(mask_polygons):
            polygon = offset_polygon(mask_polygons[index], offset, image.shape[:2], scale=coord_scale)
            cv2.fillPoly(overlay, [polygon], color)
            cv2.polylines(image, [polygon], True, color, 2, cv2.LINE_AA)
            segment_xy = [[float(x), float(y)] for x, y in polygon.reshape(-1, 2)]
            mask_area = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask_area, [polygon], 1)
            visible_pixels = int(mask_area.sum())
        else:
            segment_xy = []

        x1, y1, x2, y2 = [int(v) for v in xyxy_boxes[index]]
        p1 = (int(round(x1 * sx + ox)), int(round(y1 * sy + oy)))
        p2 = (int(round(x2 * sx + ox)), int(round(y2 * sy + oy)))
        p1 = (max(0, min(image.shape[1] - 1, p1[0])), max(0, min(image.shape[0] - 1, p1[1])))
        p2 = (max(0, min(image.shape[1] - 1, p2[0])), max(0, min(image.shape[0] - 1, p2[1])))
        if not segment_xy:
            segment_xy = [
                [float(p1[0]), float(p1[1])],
                [float(p2[0]), float(p1[1])],
                [float(p2[0]), float(p2[1])],
                [float(p1[0]), float(p2[1])],
            ]
            visible_pixels = max(0, p2[0] - p1[0]) * max(0, p2[1] - p1[1])
        quad_xy = [
            [float(p1[0]), float(p1[1])],
            [float(p2[0]), float(p1[1])],
            [float(p2[0]), float(p2[1])],
            [float(p1[0]), float(p2[1])],
        ]
        if class_name.lower() in FRUIT_CLASSES:
            kind = "fruit_face"
        elif class_name.lower() == "plain":
            kind = "plain_face"
        else:
            kind = "unknown_face"
        faces.append(
            FaceEvidence(
                face_index=len(faces),
                kind=kind,
                label=class_name,
                confidence=confidence,
                detector_confidence=confidence,
                box_xyxy=[float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])],
                quad_xy=quad_xy,
                visible_pixels=int(visible_pixels),
                raw_quad_xy=quad_xy,
                segments_xy=[segment_xy],
                source="A1+Unified",
            )
        )
        lines.append(f"{prefix}{class_name} conf={confidence:.3f} box=[{p1[0]},{p1[1]},{p2[0]},{p2[1]}]")

    cv2.addWeighted(overlay, float(np.clip(mask_alpha, 0.0, 0.9)), image, 1.0 - float(np.clip(mask_alpha, 0.0, 0.9)), 0, dst=image)
    return lines, faces


def process_unified_crop_frame(
    frame: np.ndarray,
    a1_model: YOLO,
    face_model: YOLO,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[str], dict[str, float]]:
    height, width = frame.shape[:2]
    annotated = frame.copy()
    timings: dict[str, float] = {}

    total_start = time.perf_counter()
    t0 = time.perf_counter()
    a1_result = a1_model.predict(
        source=frame,
        conf=args.unified_a1_conf,
        imgsz=args.unified_a1_imgsz,
        device=args.device,
        verbose=False,
    )[0]
    timings["a1_ms"] = (time.perf_counter() - t0) * 1000.0

    t_crop = time.perf_counter()
    crops: list[np.ndarray] = []
    crop_boxes: list[tuple[int, int, int, int]] = []
    crop_original_shapes: list[tuple[int, int]] = []
    crop_object_boxes: list[tuple[int, int, int, int]] = []
    crop_object_confidences: list[float] = []
    object_lines: list[str] = []
    a1_object_count = 0
    too_far_count = 0
    if a1_result.boxes is not None and len(a1_result.boxes) > 0:
        boxes = a1_result.boxes.xyxy.cpu().numpy()
        classes = a1_result.boxes.cls.cpu().numpy().astype(int)
        confidences = a1_result.boxes.conf.cpu().numpy()
        order = np.argsort(confidences)[::-1]
        a1_polygons = get_mask_polygons(a1_result)
        a1_overlay = annotated.copy()
        a1_draw_jobs: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int, int], str]] = []

        for det_index in order:
            class_id = int(classes[det_index])
            class_name = get_class_name(a1_result.names, class_id)
            confidence = float(confidences[det_index])
            a1_object_count += 1
            box = boxes[det_index]
            bx1, by1, bx2, by2 = [int(round(v)) for v in box]
            bx1 = max(0, min(width - 1, bx1))
            by1 = max(0, min(height - 1, by1))
            bx2 = max(0, min(width - 1, bx2))
            by2 = max(0, min(height - 1, by2))
            too_far = False
            too_far_reason = ""
            if class_id == 0:
                too_far, size_metrics = cube_is_too_far((bx1, by1, bx2, by2), annotated.shape[:2])
                if too_far:
                    too_far_reason = cube_too_far_reason(size_metrics)
            object_color = (
                CUBE_TOO_FAR_COLOR_BGR
                if too_far
                else ((255, 180, 40) if class_id == 0 else class_color(class_id + 4))
            )

            if det_index < len(a1_polygons):
                polygon = offset_polygon(a1_polygons[det_index], (0, 0), annotated.shape[:2])
                cv2.fillPoly(a1_overlay, [polygon], object_color)
                cv2.polylines(annotated, [polygon], True, object_color, 2, cv2.LINE_AA)
            if class_id != 0 or too_far:
                a1_draw_jobs.append(
                    (
                        (bx1, by1),
                        (bx2, by2),
                        object_color,
                        f"{CUBE_TOO_FAR_IDENTITY if too_far else class_name} {confidence:.2f}",
                    )
                )

            if class_id != 0:
                object_lines.append(
                    f"A1 {class_name} conf={confidence:.3f} box=[{bx1},{by1},{bx2},{by2}]"
                )
                continue
            if too_far:
                too_far_count += 1
                object_lines.append(
                    f"A1 {CUBE_TOO_FAR_IDENTITY} conf={confidence:.3f} "
                    f"box=[{bx1},{by1},{bx2},{by2}] {too_far_reason}; unified skipped"
                )
                continue

            crop_box = expand_xyxy_for_crop(box, width, height, args.unified_crop_pad)
            if crop_box is None:
                continue
            x1, y1, x2, y2 = crop_box
            crop = frame[y1 : y2 + 1, x1 : x2 + 1]
            if crop.size == 0:
                continue

            crop_h, crop_w = crop.shape[:2]
            face_input_size = max(1, int(args.imgsz))
            face_input = cv2.resize(crop, (face_input_size, face_input_size), interpolation=cv2.INTER_AREA)
            obj_index = len(crops)
            crops.append(face_input)
            crop_boxes.append(crop_box)
            crop_original_shapes.append((crop_h, crop_w))
            crop_object_boxes.append((bx1, by1, bx2, by2))
            crop_object_confidences.append(confidence)
            object_lines.append(
                f"o{obj_index} A1={class_name} conf={confidence:.3f} crop=[{x1},{y1},{x2},{y2}] "
                f"face_input={face_input_size}x{face_input_size}"
            )

        cv2.addWeighted(a1_overlay, 0.16, annotated, 0.84, 0, dst=annotated)
        for p1, p2, color, label in a1_draw_jobs:
            cv2.rectangle(annotated, p1, p2, color, 2)
            draw_small_label(annotated, label, p1, color)

    timings["crop_ms"] = (time.perf_counter() - t_crop) * 1000.0

    face_lines: list[str] = []
    decision_lines: list[str] = []
    fruit_cube_count = 0
    t_face = time.perf_counter()
    if crops:
        face_results = face_model.predict(
            source=crops,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )
        for obj_index, (crop_box, original_shape, object_box, object_confidence, face_result) in enumerate(
            zip(crop_boxes, crop_original_shapes, crop_object_boxes, crop_object_confidences, face_results)
        ):
            x1, y1, _x2, _y2 = crop_box
            crop_h, crop_w = original_shape
            result_h, result_w = result_frame_shape(face_result)
            if result_h <= 0 or result_w <= 0:
                result_h = result_w = max(1, int(args.imgsz))
            coord_scale = (
                float(crop_w) / max(float(result_w), 1.0),
                float(crop_h) / max(float(result_h), 1.0),
            )
            lines, faces = draw_result_at_offset(
                annotated,
                face_result,
                offset=(x1, y1),
                coord_scale=coord_scale,
                prefix=f"o{obj_index} ",
                overlap_threshold=args.overlap,
                mask_alpha=args.unified_mask_alpha,
            )
            decision = decide_cube(
                faces,
                target_shape=args.target_shape,
                target_fruit=args.target_fruit,
                fruit_min_conf=args.fruit_min_conf,
            )
            decision_label, decision_conf = decision_label_with_confidence(
                decision,
                faces,
                object_confidence=object_confidence,
            )
            if decision.identity.startswith("fruit_cube:"):
                fruit_cube_count += 1
            fruit_summary = decision.fruit_faces if decision.fruit_faces else {}
            decision_text = (
                f"o{obj_index} decision={decision.identity} action={decision.action} "
                f"decision_conf={decision_conf:.3f} a1_conf={object_confidence:.3f} "
                f"faces={decision.visible_faces} blank={decision.blank_faces} "
                f"fruit={fruit_summary} unknown={decision.unknown_faces}"
            )
            decision_lines.append(decision_text)
            obj_x1, obj_y1, obj_x2, obj_y2 = object_box
            color = decision_color(decision)
            cv2.rectangle(annotated, (obj_x1, obj_y1), (obj_x2, obj_y2), color, 3)
            draw_small_label(
                annotated,
                decision_label,
                (obj_x1, max(0, obj_y1 - 22)),
                color,
                scale=0.58,
            )
            if lines:
                face_lines.extend(lines)
            else:
                face_lines.append(f"o{obj_index} no unified face detections")
    timings["face_ms"] = (time.perf_counter() - t_face) * 1000.0
    timings["total_ms"] = (time.perf_counter() - total_start) * 1000.0

    if not object_lines:
        object_lines = ["No A1 objects detected."]
    count_line = (
        f"A1 objects={a1_object_count} cube crops={len(crops)} "
        f"too_far={too_far_count} fruit cubes={fruit_cube_count} "
        f"unified faces={sum(1 for line in face_lines if ' conf=' in line)}"
    )
    panel_lines = [count_line] + object_lines + decision_lines + face_lines
    return annotated, panel_lines, timings


def resize_for_display(frame: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    scale = float(args.display_scale)
    if scale <= 0:
        height, width = frame.shape[:2]
        max_width = max(1, int(args.display_max_width))
        max_height = max(1, int(args.display_max_height))
        scale = min(1.0, max_width / max(width, 1), max_height / max(height, 1))
    if scale >= 0.999:
        return frame
    new_width = max(1, int(round(frame.shape[1] * scale)))
    new_height = max(1, int(round(frame.shape[0] * scale)))
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def main() -> None:
    args = parse_args()
    apply_runtime_preset(args)

    if args.list_model_aliases:
        print_model_aliases()
        return

    global ACTION_COLORS, FRUIT_CLASSES, FaceEvidence, ObjectDecision, YOLO, cv2, decide_cube
    import cv2 as cv2_module
    from ultralytics import YOLO as ultralytics_yolo
    from abc_inference import (
        ACTION_COLORS as action_colors,
        FRUIT_CLASSES as fruit_classes,
        FaceEvidence as face_evidence,
        ObjectDecision as object_decision,
        decide_cube as decide_cube_fn,
    )

    ACTION_COLORS = action_colors
    FRUIT_CLASSES = fruit_classes
    FaceEvidence = face_evidence
    ObjectDecision = object_decision
    decide_cube = decide_cube_fn
    cv2 = cv2_module
    YOLO = ultralytics_yolo

    model = None
    model_path = None
    unified_a1 = None
    unified_a1_path = None
    pipeline = None
    if args.pipeline == "abc":
        from abc_inference import (
            ABCPipeline,
            append_c_input_panel,
            append_raw_model_output_panel,
            draw_frame_output,
            draw_raw_model_output,
            raw_model_summary,
        )

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
            enable_timing=args.print_timing,
            trace_torch_models=args.trace_torch_models,
            yolo_half=args.yolo_half,
            keep_c_inputs=args.show_c_inputs,
            use_a1_object_mask=args.use_a1_object_mask,
            batch_b_across_objects=args.batch_b_across_objects,
            batch_c_across_objects=args.batch_c_across_objects,
        )
    elif args.pipeline == "unified":
        unified_a1_path = resolve_model_path(args.a1_model or "preferred-a1")
        model_path = resolve_model_path(args.model, unified_default=True)
        unified_a1 = YOLO(str(unified_a1_path), task="segment")
        model = YOLO(str(model_path), task=args.task)
    else:
        model_path = resolve_model_path(args.model)
        model = YOLO(str(model_path), task=args.task)

    cap = open_camera(args)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {args.camera}. Try --camera 1/2, --camera-backend v4l2, or a GStreamer pipeline."
        )

    writer = None
    if args.save:
        output_path = Path("runs") / "realtime_seg_cam.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1:
            fps = 30

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_width = frame_width
        if args.pipeline in {"abc", "unified"} and args.show_model_output_panel:
            output_width += max(1, int(args.model_output_panel_width))
        if args.pipeline == "abc" and args.show_c_inputs:
            output_width += max(1, int(args.c_input_panel_width))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (output_width, frame_height),
        )
        print(f"Saving video to: {output_path}")

    if args.pipeline == "abc":
        print(f"ABC A1: {pipeline.a1_path}")
        print(f"ABC A2: {pipeline.a2_path}")
        print(f"ABC B:  {pipeline.b_path}")
        print(f"ABC C:  {pipeline.c_path}")
        print(f"ABC quad refine: {pipeline.refine_quads}")
        print("ABC webcam started.")
    elif args.pipeline == "unified":
        print(f"Unified A1 model: {unified_a1_path}")
        print(f"Unified face model: {model_path}")
        names = model.names.values() if isinstance(model.names, dict) else model.names
        print(f"Unified classes: {', '.join(str(name) for name in names)}")
        print("Unified crop webcam preview started: A1 full frame -> cube crop -> unified face model.")
    else:
        print(f"YOLO model: {model_path}")
        print("YOLO segmentation webcam started.")
    print("Press 'q' or ESC to quit.")

    prev_time = time.perf_counter()
    warned_no_masks = False
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read a frame from the camera.")
                break

            if args.pipeline == "abc":
                output = pipeline.process_frame(frame)
                if args.abc_overlay == "decision":
                    annotated_frame = draw_frame_output(frame, output)
                else:
                    annotated_frame = draw_raw_model_output(frame, output)
                if args.show_c_inputs:
                    annotated_frame = append_c_input_panel(
                        annotated_frame,
                        pipeline.last_c_inputs,
                        panel_width=args.c_input_panel_width,
                        tile_size=args.c_input_tile_size,
                    )
                if args.show_model_output_panel:
                    annotated_frame = append_raw_model_output_panel(
                        annotated_frame,
                        output,
                        panel_width=args.model_output_panel_width,
                    )
            elif args.pipeline == "unified":
                annotated_frame, unified_lines, unified_timing = process_unified_crop_frame(
                    frame,
                    unified_a1,
                    model,
                    args,
                )
                if args.show_model_output_panel:
                    annotated_frame = append_unified_crop_output_panel(
                        annotated_frame,
                        unified_lines,
                        panel_width=args.model_output_panel_width,
                    )
            else:
                results = model.predict(
                    source=frame,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    device=args.device,
                    verbose=False,
                )
                result = results[0]

                if args.task == "segment" and not warned_no_masks:
                    if results and not has_segmentation_masks(result):
                        print(
                            "Warning: segmentation masks are missing. "
                            "Check that the model was exported from a YOLO segmentation model "
                            "and try forcing --task segment."
                        )
                        warned_no_masks = True

                annotated_frame = draw_confidence_brightness(
                    frame,
                    result,
                    overlap_threshold=args.overlap,
                    mask_alpha=None,
                )

            now = time.perf_counter()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            fps_y = 65 if args.pipeline == "abc" else 35
            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f}",
                (15, fps_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(annotated_frame)

            frame_index += 1
            if args.pipeline == "abc" and args.print_model_output:
                every = max(1, int(args.model_output_every))
                if frame_index == 1 or frame_index % every == 0:
                    print(f"[frame {frame_index}] raw model output fps={fps:.2f}")
                    summaries = raw_model_summary(output)
                    if summaries:
                        for summary in summaries:
                            print(f"  {summary}")
                    else:
                        print("  no A1 objects")
            if args.pipeline == "unified" and args.print_model_output:
                every = max(1, int(args.model_output_every))
                if frame_index == 1 or frame_index % every == 0:
                    print(f"[frame {frame_index}] unified crop model output fps={fps:.2f}")
                    for summary in unified_lines:
                        print(f"  {summary}")
            if args.pipeline == "abc" and args.print_timing:
                every = max(1, int(args.timing_every))
                if frame_index == 1 or frame_index % every == 0:
                    print(f"[frame {frame_index}] fps={fps:.2f} {pipeline.timing_summary()}")
            if args.pipeline == "unified" and args.print_timing:
                every = max(1, int(args.timing_every))
                if frame_index == 1 or frame_index % every == 0:
                    speed_text = " ".join(
                        f"{key}={value:.1f}ms" for key, value in unified_timing.items()
                    )
                    print(f"[frame {frame_index}] fps={fps:.2f} {speed_text}".rstrip())

            if not args.no_display:
                if args.pipeline == "abc":
                    window_title = "ABC Webcam"
                elif args.pipeline == "unified":
                    window_title = "Unified Cube-Face Webcam"
                else:
                    window_title = "YOLO Segmentation Webcam"
                cv2.imshow(window_title, resize_for_display(annotated_frame, args))

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
            if args.max_frames > 0 and frame_index >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
