from __future__ import annotations

import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_ABC_ROOT = PROJECT_ROOT / "ABC_model"

A1_DEFAULT = (
    DEFAULT_ABC_ROOT
    / "meta_v2_a1_objectseg"
    / "a1_yolo26s_seg_meta_v2_50000"
    / "weights"
    / "best.pt"
)
A2_DEFAULT = (
    DEFAULT_ABC_ROOT
    / "meta_v2_a2_faceseg"
    / "a2_yolo26n_seg_meta_v2_50000"
    / "weights"
    / "best.pt"
)
B_DEFAULT = (
    DEFAULT_ABC_ROOT
    / "meta_v2_b_tinyquadnet"
    / "b_tinyquadnet_meta_v2_50000"
    / "weights"
    / "best.pt"
)
C_DEFAULT = (
    DEFAULT_ABC_ROOT
    / "meta_v2_c_facecls"
    / "c_mobilenetv3small_meta_v2_50000_runtimewarp_ft"
    / "weights"
    / "best.pt"
)

FRUIT_CLASSES = ("apple", "orange", "banana", "pineapple")
C_CLASSES = (*FRUIT_CLASSES, "plain", "unknown")
CUBE_LIKE_CLASSES = {"cube_like_object", "cube", *FRUIT_CLASSES, "fruit_cube"}
POLYHEDRON_CLASSES = {"octahedron", "dodecahedron", "icosahedron"}
PLAIN_LABELS = {"plain", "blank", "plain_face", "blank_face"}

# Minimum confidence for a unified fruit face to count as reliable fruit evidence in
# decide_cube(). Below this, the fruit face is treated as unknown so a weak detection on
# a blank/sliver crop cannot flip the cube to a fruit identity. Data-driven default:
# on the recorded orange-cube holdout it retains 100% of true-orange faces (min conf 0.463)
# while rejecting ~88% of blank-crop false positives (max conf 0.496). Provisional — the
# final value should be confirmed against real robot-camera footage.
FRUIT_MIN_CONF_DEFAULT = 0.45

_NVIDIA_DLL_HANDLES: list[Any] = []

ACTION_COLORS = {
    "pickup": (40, 220, 80),
    "inspect": (40, 210, 255),
    "skip": (180, 180, 180),
    "avoid": (40, 40, 255),
    "unknown": (200, 160, 255),
}


@dataclass
class FaceEvidence:
    face_index: int
    kind: str
    label: str
    confidence: float
    detector_confidence: float
    box_xyxy: list[float]
    quad_xy: list[list[float]]
    visible_pixels: int
    raw_quad_xy: list[list[float]] = field(default_factory=list)
    segments_xy: list[list[list[float]]] = field(default_factory=list)
    source: str = "A2+B+C"


@dataclass
class ObjectDecision:
    identity: str
    action: str
    reason: str
    visible_faces: int
    blank_faces: int
    fruit_faces: dict[str, int]
    unknown_faces: int
    expected_score: int = 0


@dataclass
class ObjectEvidence:
    object_index: int
    class_name: str
    confidence: float
    box_xyxy: list[float]
    segments_xy: list[list[list[float]]] = field(default_factory=list)
    faces: list[FaceEvidence] = field(default_factory=list)
    decision: ObjectDecision | None = None


@dataclass
class FrameOutput:
    objects: list[ObjectEvidence]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


def resolve_existing_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()

    candidates = [
        Path.cwd() / path,
        PROJECT_ROOT / path,
        REPO_ROOT / path,
        path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / path).resolve()


def path_or_default(path_text: str | None, default: Path) -> Path:
    if path_text:
        return resolve_existing_path(Path(path_text))
    return default


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} model file not found: {path.resolve()}\n"
            "If this is a fresh clone, run `git lfs pull` from the repository root "
            "or pass the model path explicitly."
        )
    return path


def get_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def expand_xyxy(
    box: np.ndarray | list[float],
    width: int,
    height: int,
    pad_ratio: float,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(1.0, x2 - x1 + 1.0)
    bh = max(1.0, y2 - y1 + 1.0)
    pad = max(bw, bh) * float(pad_ratio)
    x1 = max(0, int(math.floor(x1 - pad)))
    y1 = max(0, int(math.floor(y1 - pad)))
    x2 = min(width - 1, int(math.ceil(x2 + pad)))
    y2 = min(height - 1, int(math.ceil(y2 + pad)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def mask_to_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask.astype(bool))
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_from_points(points: np.ndarray | list[list[float]]) -> list[float] | None:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] != 2 or not np.isfinite(pts).all():
        return None
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return [float(x1), float(y1), float(x2), float(y2)]


def result_mask(result: Any, index: int, shape: tuple[int, int]) -> np.ndarray | None:
    height, width = shape
    if result.masks is None:
        return None

    polygons = getattr(result.masks, "xy", None)
    if polygons is not None and index < len(polygons):
        polygon = polygons[index]
        if polygon is not None and len(polygon) >= 3:
            mask = np.zeros((height, width), dtype=np.uint8)
            pts = polygon.astype(np.int32)
            pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
            cv2.fillPoly(mask, [pts], 1)
            return mask.astype(bool)

    if result.masks.data is None or index >= len(result.masks.data):
        return None
    mask = result.masks.data[index].cpu().numpy()
    if mask.shape[:2] != shape:
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask > 0.5


def mask_to_polygons_xy(
    mask: np.ndarray | None,
    *,
    min_area: float = 25.0,
    epsilon_ratio: float = 0.002,
) -> list[list[list[float]]]:
    if mask is None or not np.any(mask):
        return []
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[list[float]]] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < min_area:
            continue
        perimeter = cv2.arcLength(contour.astype(np.float32), True)
        approx = cv2.approxPolyDP(
            contour.astype(np.float32),
            max(0.5, perimeter * epsilon_ratio),
            True,
        )
        points = approx.reshape(-1, 2)
        if len(points) >= 3:
            polygons.append([[float(x), float(y)] for x, y in points])
    return polygons


def transform_polygons_xy(
    polygons: list[list[list[float]]],
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> list[list[list[float]]]:
    transformed: list[list[list[float]]] = []
    for polygon in polygons:
        transformed.append(
            [
                [float(x) * scale_x + offset_x, float(y) * scale_y + offset_y]
                for x, y in polygon
            ]
        )
    return transformed


def order_quad_points(points: np.ndarray | list[list[float]]) -> np.ndarray | None:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        return None
    if np.unique(np.round(pts, 3), axis=0).shape[0] != 4:
        return None
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -start, axis=0)
    if abs(cv2.contourArea(ordered.reshape(-1, 1, 2))) <= 0:
        return None
    return ordered.astype(np.float32)


def quad_metrics(points: np.ndarray | list[list[float]]) -> dict[str, float] | None:
    pts = order_quad_points(points)
    if pts is None:
        return None
    return ordered_quad_metrics(pts)


def ordered_quad_metrics(pts: np.ndarray) -> dict[str, float]:
    sides = [
        float(np.linalg.norm(pts[(idx + 1) % 4] - pts[idx]))
        for idx in range(4)
    ]
    min_side = min(sides) if sides else 0.0
    max_side = max(sides) if sides else 0.0
    return {
        "area": float(abs(cv2.contourArea(pts.reshape(-1, 1, 2)))),
        "min_side": min_side,
        "max_side": max_side,
        "aspect": float(max_side / max(min_side, 1e-6)),
    }


def quad_passes(
    points: np.ndarray | list[list[float]],
    min_area: float,
    min_side: float,
    max_aspect: float,
) -> np.ndarray | None:
    pts = order_quad_points(points)
    if pts is None:
        return None
    metrics = ordered_quad_metrics(pts)
    if metrics["area"] < min_area:
        return None
    if metrics["min_side"] < min_side:
        return None
    if metrics["aspect"] > max_aspect:
        return None
    return pts


def quad_from_mask(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 1:
        return None
    rect = cv2.minAreaRect(contour)
    return order_quad_points(cv2.boxPoints(rect))


def warp_quad(image: np.ndarray, quad: np.ndarray, crop_size: int) -> np.ndarray | None:
    src = order_quad_points(quad)
    if src is None:
        return None
    dst = np.asarray(
        [[0, 0], [crop_size - 1, 0], [crop_size - 1, crop_size - 1], [0, crop_size - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (crop_size, crop_size), flags=cv2.INTER_LINEAR)


def mask_from_polygons_xy(
    polygons: list[list[list[float]]],
    shape: tuple[int, int] | tuple[int, int, int],
) -> np.ndarray | None:
    height, width = shape[:2]
    if not polygons:
        return None
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        pts = np.asarray(polygon, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2 or not np.isfinite(pts).all():
            continue
        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [np.rint(pts).astype(np.int32).reshape(-1, 1, 2)], 1)
    return mask.astype(bool) if np.any(mask) else None


def full_quad_mask(
    shape: tuple[int, int] | tuple[int, int, int],
    quad: np.ndarray | list[list[float]],
) -> np.ndarray | None:
    height, width = shape[:2]
    pts = order_quad_points(quad)
    if pts is None:
        return None
    mask = np.zeros((height, width), dtype=np.uint8)
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    cv2.fillPoly(mask, [np.rint(pts).astype(np.int32).reshape(-1, 1, 2)], 1)
    return mask.astype(bool)


def blacken_hidden_face_area(
    image: np.ndarray,
    visible_face_mask: np.ndarray | None,
    quad: np.ndarray | list[list[float]],
    *,
    fill_value: int = 0,
    visible_dilate: int = 3,
) -> np.ndarray:
    if visible_face_mask is None:
        return image
    face_full = full_quad_mask(image.shape, quad)
    if face_full is None or not np.any(face_full):
        return image
    visible = visible_face_mask.astype(bool)
    if visible_dilate > 0:
        kernel = np.ones((visible_dilate, visible_dilate), dtype=np.uint8)
        visible = cv2.dilate(visible.astype(np.uint8), kernel, iterations=1).astype(bool)
    hidden = face_full & (~visible)
    if not np.any(hidden):
        return image
    out = image.copy()
    out[hidden] = int(np.clip(fill_value, 0, 255))
    return out


def c_warp_source_image(
    frame: np.ndarray,
    face: "FaceEvidence",
    quad: np.ndarray,
    *,
    blacken_occluded_face_area: bool = True,
    occlusion_visible_ratio: float = 0.92,
    fill_value: int = 0,
    visible_dilate: int = 3,
) -> tuple[np.ndarray, bool, float, float, int, int]:
    visible_mask = mask_from_polygons_xy(face.segments_xy, frame.shape)
    full_mask = full_quad_mask(frame.shape, quad)
    if visible_mask is None or full_mask is None or not np.any(full_mask):
        return frame, False, 1.0, 1.0, int(face.visible_pixels), int(full_mask.sum()) if full_mask is not None else 0
    visible_in_full = int(np.logical_and(visible_mask, full_mask).sum())
    visible_pixels = int(visible_mask.sum())
    full_pixels = int(full_mask.sum())
    visible_in_full_ratio = visible_in_full / max(full_pixels, 1)
    visible_to_full_ratio = visible_pixels / max(full_pixels, 1)
    if not blacken_occluded_face_area or visible_in_full_ratio >= float(occlusion_visible_ratio):
        return frame, False, float(visible_in_full_ratio), float(visible_to_full_ratio), visible_pixels, full_pixels
    source = blacken_hidden_face_area(
        frame,
        visible_mask,
        quad,
        fill_value=fill_value,
        visible_dilate=visible_dilate,
    )
    return source, True, float(visible_in_full_ratio), float(visible_to_full_ratio), visible_pixels, full_pixels


def local_face_visibility_stats(
    frame_shape: tuple[int, int] | tuple[int, int, int],
    face: "FaceEvidence",
    quad: np.ndarray,
) -> tuple[float, float, int, int]:
    height, width = frame_shape[:2]
    full_quad = order_quad_points(quad)
    if full_quad is None:
        return 1.0, 1.0, int(face.visible_pixels), 0

    points = [full_quad]
    for polygon in face.segments_xy or []:
        pts = np.asarray(polygon, dtype=np.float32)
        if pts.ndim == 2 and pts.shape[0] >= 3 and pts.shape[1] == 2 and np.isfinite(pts).all():
            points.append(pts)
    all_points = np.concatenate(points, axis=0)
    x1 = max(0, int(np.floor(float(all_points[:, 0].min()))) - 2)
    y1 = max(0, int(np.floor(float(all_points[:, 1].min()))) - 2)
    x2 = min(width, int(np.ceil(float(all_points[:, 0].max()))) + 3)
    y2 = min(height, int(np.ceil(float(all_points[:, 1].max()))) + 3)
    if x2 <= x1 or y2 <= y1:
        return 1.0, 1.0, int(face.visible_pixels), 0

    roi_shape = (y2 - y1, x2 - x1)
    offset = np.asarray([x1, y1], dtype=np.float32)
    full_mask = np.zeros(roi_shape, dtype=np.uint8)
    shifted_quad = full_quad - offset
    cv2.fillPoly(full_mask, [np.rint(shifted_quad).astype(np.int32).reshape(-1, 1, 2)], 1)
    if not np.any(full_mask):
        return 1.0, 1.0, int(face.visible_pixels), 0

    visible_mask = np.zeros(roi_shape, dtype=np.uint8)
    for polygon in face.segments_xy or []:
        pts = np.asarray(polygon, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2 or not np.isfinite(pts).all():
            continue
        shifted = pts - offset
        shifted[:, 0] = np.clip(shifted[:, 0], 0, roi_shape[1] - 1)
        shifted[:, 1] = np.clip(shifted[:, 1], 0, roi_shape[0] - 1)
        cv2.fillPoly(visible_mask, [np.rint(shifted).astype(np.int32).reshape(-1, 1, 2)], 1)
    if not np.any(visible_mask):
        return 1.0, 1.0, int(face.visible_pixels), int(full_mask.sum())

    visible_in_full = int(np.logical_and(visible_mask.astype(bool), full_mask.astype(bool)).sum())
    visible_pixels = int(visible_mask.sum())
    full_pixels = int(full_mask.sum())
    return (
        float(visible_in_full / max(full_pixels, 1)),
        float(visible_pixels / max(full_pixels, 1)),
        visible_pixels,
        full_pixels,
    )


def visible_crop_mask_from_segments(
    segments_xy: list[list[list[float]]],
    quad: np.ndarray,
    crop_size: int,
    *,
    visible_dilate: int = 3,
) -> np.ndarray | None:
    src = order_quad_points(quad)
    if src is None or not segments_xy:
        return None
    dst = np.asarray(
        [[0, 0], [crop_size - 1, 0], [crop_size - 1, crop_size - 1], [0, crop_size - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
    for polygon in segments_xy:
        pts = np.asarray(polygon, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2 or not np.isfinite(pts).all():
            continue
        warped = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        warped[:, 0] = np.clip(warped[:, 0], 0, crop_size - 1)
        warped[:, 1] = np.clip(warped[:, 1], 0, crop_size - 1)
        cv2.fillPoly(mask, [np.rint(warped).astype(np.int32).reshape(-1, 1, 2)], 1)
    if not np.any(mask):
        return None
    if visible_dilate > 0:
        kernel = np.ones((visible_dilate, visible_dilate), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask.astype(bool)


def c_filter_reject_reason(
    quad: np.ndarray,
    *,
    visible_pixels: int,
    full_pixels: int,
    visible_ratio: float,
    visible_to_full_ratio: float,
    black_fraction: float,
    min_face_pixels: int,
    c_min_quad_area: float,
    c_min_quad_side: float,
    c_max_quad_aspect: float,
    c_min_visible_pixels: int,
    c_min_visible_ratio: float,
    c_max_visible_to_full_ratio: float,
    c_max_black_fraction: float,
) -> str:
    if quad_passes(quad, c_min_quad_area, c_min_quad_side, c_max_quad_aspect) is None:
        return "bad_quad"
    if full_pixels and full_pixels < int(min_face_pixels):
        return "tiny_full_face"
    if int(visible_pixels) < max(int(min_face_pixels), int(c_min_visible_pixels)):
        return "too_few_visible_pixels"
    if visible_to_full_ratio > float(c_max_visible_to_full_ratio):
        return "inconsistent_visible_full"
    if visible_ratio < float(c_min_visible_ratio):
        return "low_visible_ratio"
    if black_fraction > float(c_max_black_fraction):
        return "too_much_black"
    return ""


def choose_torch_device(device_arg: str | None):
    import torch

    if not device_arg:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if str(device_arg).lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{str(device_arg).split(',')[0]}")


def choose_yolo_device(device_arg: str | None) -> str | None:
    import torch

    if not device_arg:
        return None
    text = str(device_arg).strip().lower()
    if text == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    return str(device_arg)


def configure_torch_cuda_fast_path() -> None:
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def torch_load(path: Path, device: Any) -> Any:
    import pathlib
    import sys
    import types

    import torch

    if "pathlib._local" not in sys.modules:
        compat = types.ModuleType("pathlib._local")
        compat.Path = pathlib.Path
        compat.PosixPath = pathlib.PosixPath
        compat.WindowsPath = pathlib.WindowsPath
        compat.PurePath = pathlib.PurePath
        compat.PurePosixPath = pathlib.PurePosixPath
        compat.PureWindowsPath = pathlib.PureWindowsPath
        sys.modules["pathlib._local"] = compat

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def add_nvidia_cuda_dll_directories() -> None:
    if not hasattr(os, "add_dll_directory"):
        return
    try:
        import nvidia
    except Exception:
        return
    for root_text in getattr(nvidia, "__path__", []):
        root = Path(root_text).resolve()
        for bin_dir in sorted(root.glob("*/bin")):
            if not bin_dir.exists():
                continue
            text = str(bin_dir)
            if text not in os.environ.get("PATH", ""):
                os.environ["PATH"] = text + os.pathsep + os.environ.get("PATH", "")
            try:
                _NVIDIA_DLL_HANDLES.append(os.add_dll_directory(text))
            except OSError:
                pass


def make_tiny_quadnet():
    import torch
    from torch import nn

    class TinyQuadNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.SiLU(inplace=True),
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.SiLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.SiLU(inplace=True),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.SiLU(inplace=True),
                nn.Conv2d(128, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.SiLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 128),
                nn.SiLU(inplace=True),
                nn.Dropout(0.05),
                nn.Linear(128, 8),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.head(self.features(x))

    return TinyQuadNet()


def make_mini_classifier(num_classes: int):
    from torch import nn

    class MiniClassifier(nn.Module):
        def __init__(self, n: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 24, 3, stride=2, padding=1),
                nn.BatchNorm2d(24),
                nn.SiLU(inplace=True),
                nn.Conv2d(24, 48, 3, stride=2, padding=1),
                nn.BatchNorm2d(48),
                nn.SiLU(inplace=True),
                nn.Conv2d(48, 96, 3, stride=2, padding=1),
                nn.BatchNorm2d(96),
                nn.SiLU(inplace=True),
                nn.Conv2d(96, 160, 3, stride=2, padding=1),
                nn.BatchNorm2d(160),
                nn.SiLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(160, n),
            )

        def forward(self, x):
            return self.net(x)

    return MiniClassifier(num_classes)


def make_c_model(num_classes: int):
    from torch import nn

    try:
        from torchvision.models import mobilenet_v3_small

        model = mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    except Exception:
        return make_mini_classifier(num_classes)


class ABCPipeline:
    def __init__(
        self,
        *,
        a1_model: str | Path | None = None,
        a2_model: str | Path | None = None,
        b_model: str | Path | None = None,
        c_model: str | Path | None = None,
        c_onnx_model: str | Path | None = None,
        c_onnx_provider: str = "cuda",
        device: str | None = None,
        b_device: str | None = None,
        torch_cuda_fast_path: bool = False,
        a1_conf: float = 0.25,
        a2_conf: float = 0.20,
        c_conf: float = 0.55,
        imgsz: int = 640,
        a2_imgsz: int | None = 224,
        a2_crop_pad: float = 0.18,
        b_crop_pad: float = 0.12,
        b_imgsz: int = 128,
        b_refine_passes: int = 1,
        b_min_quad_area: float = 80.0,
        b_min_quad_side: float = 4.0,
        b_max_quad_aspect: float = 24.0,
        c_imgsz: int = 128,
        c_warp_size: int = 224,
        min_face_pixels: int = 120,
        min_face_object_overlap: float = 0.45,
        refine_quads: str = "pose",
        blacken_c_occluded_face_area: bool = True,
        c_occlusion_visible_ratio: float = 0.92,
        c_min_quad_area: float = 400.0,
        c_min_quad_side: float = 10.0,
        c_max_quad_aspect: float = 14.0,
        c_min_visible_pixels: int = 400,
        c_min_visible_ratio: float = 0.40,
        c_max_visible_to_full_ratio: float = 1.15,
        c_max_black_fraction: float = 0.60,
        target_shape: str = "",
        target_fruit: str = "",
        enable_timing: bool = False,
        trace_torch_models: bool = False,
        yolo_half: bool = False,
        keep_c_inputs: bool = False,
        use_a1_object_mask: bool = True,
        classify_fruit_faces_only: bool = False,
        plain_face_fastpath_min_conf: float = 0.0,
        batch_b_across_objects: bool = False,
        batch_c_across_objects: bool = False,
        final_face_filter_mode: str = "full",
    ) -> None:
        self.a1_path = require_file(path_or_default(str(a1_model) if a1_model else None, A1_DEFAULT), "A1")
        self.a2_path = require_file(path_or_default(str(a2_model) if a2_model else None, A2_DEFAULT), "A2")
        self.b_path = require_file(path_or_default(str(b_model) if b_model else None, B_DEFAULT), "B")
        self.c_path = require_file(path_or_default(str(c_model) if c_model else None, C_DEFAULT), "C")
        self.c_onnx_path = Path(c_onnx_model) if c_onnx_model else None
        self.yolo_device = choose_yolo_device(device)
        self.torch_device = choose_torch_device(device)
        self.b_device = choose_torch_device(b_device) if b_device else self.torch_device
        self.torch_cuda_fast_path = bool(torch_cuda_fast_path)
        if self.torch_cuda_fast_path:
            configure_torch_cuda_fast_path()
        self.yolo_half = bool(yolo_half and str(self.yolo_device).lower() != "cpu")
        self.a1_conf = float(a1_conf)
        self.a2_conf = float(a2_conf)
        self.c_conf = float(c_conf)
        self.imgsz = int(imgsz)
        self.a2_imgsz = int(a2_imgsz or 224)
        if self.a2_imgsz % 32 != 0:
            raise ValueError(f"a2_imgsz must be a multiple of 32, got {self.a2_imgsz}")
        self.a2_crop_pad = float(a2_crop_pad)
        self.b_crop_pad = float(b_crop_pad)
        self.b_imgsz = int(b_imgsz)
        self.b_refine_passes = max(1, min(2, int(b_refine_passes)))
        self.b_min_quad_area = float(b_min_quad_area)
        self.b_min_quad_side = float(b_min_quad_side)
        self.b_max_quad_aspect = float(b_max_quad_aspect)
        self.c_imgsz = int(c_imgsz)
        self.c_warp_size = int(c_warp_size)
        self.min_face_pixels = int(min_face_pixels)
        self.min_face_object_overlap = float(min_face_object_overlap)
        self.refine_quads = refine_quads.strip().lower()
        if self.refine_quads not in {"none", "pose", "pose_fruit", "pose_fast_iou"}:
            raise ValueError("--refine-quads must be one of: none, pose, pose_fruit, pose_fast_iou")
        self.blacken_c_occluded_face_area = bool(blacken_c_occluded_face_area)
        self.c_occlusion_visible_ratio = float(c_occlusion_visible_ratio)
        self.c_min_quad_area = float(c_min_quad_area)
        self.c_min_quad_side = float(c_min_quad_side)
        self.c_max_quad_aspect = float(c_max_quad_aspect)
        self.c_min_visible_pixels = int(c_min_visible_pixels)
        self.c_min_visible_ratio = float(c_min_visible_ratio)
        self.c_max_visible_to_full_ratio = float(c_max_visible_to_full_ratio)
        self.c_max_black_fraction = float(c_max_black_fraction)
        self.target_shape = target_shape.strip().lower()
        self.target_fruit = target_fruit.strip().lower()
        self.trace_torch_models = bool(trace_torch_models)
        self.keep_c_inputs = bool(keep_c_inputs)
        self.use_a1_object_mask = bool(use_a1_object_mask)
        self.classify_fruit_faces_only = bool(classify_fruit_faces_only)
        self.plain_face_fastpath_min_conf = float(plain_face_fastpath_min_conf)
        self.batch_b_across_objects = bool(batch_b_across_objects)
        self.batch_c_across_objects = bool(batch_c_across_objects)
        self.final_face_filter_mode = str(final_face_filter_mode or "full").strip().lower()
        if self.final_face_filter_mode not in {"full", "reindex"}:
            raise ValueError("--final-face-filter-mode must be one of: full, reindex")
        self.c_onnx_provider = str(c_onnx_provider or "cuda").strip().lower()
        self.last_c_inputs: list[dict[str, Any]] = []
        self.enable_timing = bool(enable_timing)
        self.last_timing: dict[str, float | int | str] = {}

        self.a1 = YOLO(str(self.a1_path), task="segment")
        self.a2 = YOLO(str(self.a2_path), task="segment")
        self.b = self._load_b()
        self.c, self.c_classes = self._load_c()
        self.c_onnx_session = self._load_c_onnx()
        self.c_onnx_input_name = self.c_onnx_session.get_inputs()[0].name if self.c_onnx_session is not None else ""
        self.c_onnx_active_providers = (
            ",".join(self.c_onnx_session.get_providers()) if self.c_onnx_session is not None else ""
        )
        if self.trace_torch_models:
            self._trace_b_c_models()

    def _reset_timing(self) -> None:
        if self.enable_timing:
            self.last_timing = {
                "device_yolo": str(self.yolo_device),
                "device_torch": str(self.torch_device),
                "device_b": str(self.b_device),
                "yolo_half": int(self.yolo_half),
                "torch_cuda_fast_path": int(self.torch_cuda_fast_path),
                "batch_b_across_objects": int(self.batch_b_across_objects),
                "batch_c_across_objects": int(self.batch_c_across_objects),
                "final_face_filter_mode": self.final_face_filter_mode,
                "c_runtime": f"onnx:{self.c_onnx_active_providers}" if self.c_onnx_session is not None else "torch",
            }
        else:
            self.last_timing = {}

    def _add_timing(self, key: str, start: float) -> None:
        if not self.enable_timing:
            return
        self.last_timing[key] = float(self.last_timing.get(key, 0.0)) + (time.perf_counter() - start) * 1000.0

    def _set_timing(self, key: str, value: float | int | str) -> None:
        if self.enable_timing:
            self.last_timing[key] = value

    def timing_summary(self) -> str:
        if not self.last_timing:
            return ""
        keys = [
            "total_ms",
            "a1_ms",
            "a2_ms",
            "b_ms",
            "c_ms",
            "c_source_ms",
            "c_warp_ms",
            "c_filter_ms",
            "c_preprocess_ms",
            "a1_post_ms",
            "a2_prepare_ms",
            "a2_post_ms",
            "a2_face_parse_ms",
            "a2_face_build_ms",
            "a2_face_filter_ms",
            "a2_face_dedupe_ms",
            "a2_refine_sane_ms",
            "a2_refine_ms",
            "c_crop_loop_ms",
            "c_label_apply_ms",
            "c_skipped_plain_faces",
            "a1_objects",
            "a2_inputs",
            "b_faces",
            "c_faces",
        ]
        parts = []
        for key in keys:
            if key not in self.last_timing:
                continue
            value = self.last_timing[key]
            if isinstance(value, float):
                parts.append(f"{key}={value:.1f}")
            else:
                parts.append(f"{key}={value}")
        return " ".join(parts)

    def _load_b(self):
        model = make_tiny_quadnet().to(self.b_device)
        checkpoint = torch_load(self.b_path, self.b_device)
        state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state)
        model.eval()
        return model

    def _load_c(self):
        model = make_c_model(len(C_CLASSES)).to(self.torch_device)
        checkpoint = torch_load(self.c_path, self.torch_device)
        state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        classes = tuple(checkpoint.get("classes", C_CLASSES)) if isinstance(checkpoint, dict) else C_CLASSES
        model.load_state_dict(state)
        model.eval()
        return model, classes

    def _load_c_onnx(self):
        if self.c_onnx_path is None:
            return None
        if not self.c_onnx_path.exists():
            raise FileNotFoundError(f"C ONNX model not found: {self.c_onnx_path}")
        add_nvidia_cuda_dll_directories()
        import onnxruntime as ort

        if self.c_onnx_provider == "tensorrt":
            providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        elif self.c_onnx_provider == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = set(ort.get_available_providers())
        providers = [provider for provider in providers if provider in available]
        if not providers:
            providers = ["CPUExecutionProvider"]
        return ort.InferenceSession(str(self.c_onnx_path), providers=providers)

    def _trace_b_c_models(self) -> None:
        import torch

        with torch.inference_mode():
            b_dummy = torch.zeros(1, 1, self.b_imgsz, self.b_imgsz, device=self.b_device)
            c_dummy = torch.zeros(1, 3, self.c_imgsz, self.c_imgsz, device=self.torch_device)
            self.b = torch.jit.trace(self.b, b_dummy, strict=False).eval()
            self.c = torch.jit.trace(self.c, c_dummy, strict=False).eval()

    def process_frame(self, frame: np.ndarray) -> FrameOutput:
        self._reset_timing()
        total_start = time.perf_counter()
        self.last_c_inputs = []
        t0 = time.perf_counter()
        a1_results = self.a1.predict(
            source=frame,
            conf=self.a1_conf,
            imgsz=self.imgsz,
            device=self.yolo_device,
            half=self.yolo_half,
            verbose=False,
        )
        self._add_timing("a1_ms", t0)
        objects: list[ObjectEvidence] = []
        if not a1_results:
            self._add_timing("total_ms", total_start)
            return FrameOutput(objects)

        t_post = time.perf_counter()
        result = a1_results[0]
        if result.boxes is None or len(result.boxes) == 0:
            self._add_timing("a1_post_ms", t_post)
            self._add_timing("total_ms", total_start)
            return FrameOutput(objects)

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        cube_jobs: list[tuple[ObjectEvidence, np.ndarray, np.ndarray | None]] = []

        for index in np.argsort(confidences)[::-1]:
            class_name = get_name(result.names, int(classes[index])).lower()
            box = boxes[index].astype(float)
            object_mask = result_mask(result, int(index), frame.shape[:2]) if self.use_a1_object_mask else None
            box_segments = [
                [
                    [float(box[0]), float(box[1])],
                    [float(box[2]), float(box[1])],
                    [float(box[2]), float(box[3])],
                    [float(box[0]), float(box[3])],
                ]
            ]
            obj = ObjectEvidence(
                object_index=len(objects),
                class_name=class_name,
                confidence=float(confidences[index]),
                box_xyxy=[float(v) for v in box],
                segments_xy=mask_to_polygons_xy(object_mask, min_area=80.0) if object_mask is not None else box_segments,
            )
            if class_name in CUBE_LIKE_CLASSES:
                cube_jobs.append((obj, box, object_mask if self.use_a1_object_mask else None))
            else:
                obj.decision = decide_shape(
                    class_name,
                    target_shape=self.target_shape,
                )
            objects.append(obj)

        self._set_timing("a1_objects", len(objects))
        self._add_timing("a1_post_ms", t_post)
        if cube_jobs:
            self._process_cube_candidates_batch(frame, cube_jobs)

        self._add_timing("total_ms", total_start)
        return FrameOutput(objects)

    def _process_cube_candidate(
        self,
        frame: np.ndarray,
        box: np.ndarray,
        object_index: int,
        object_mask: np.ndarray | None,
    ) -> list[FaceEvidence]:
        prepared = self._prepare_cube_candidate(frame, box, object_mask)
        if prepared is None:
            return []
        t0 = time.perf_counter()
        a2_results = self.a2.predict(
            source=prepared["a2_input"],
            conf=self.a2_conf,
            imgsz=self.a2_imgsz,
            device=self.yolo_device,
            half=self.yolo_half,
            verbose=False,
        )
        self._add_timing("a2_ms", t0)
        if not a2_results or a2_results[0].boxes is None or len(a2_results[0].boxes) == 0:
            return []
        return self._process_prepared_cube_candidate_result(
            frame,
            prepared,
            object_index,
            [float(v) for v in box],
            a2_results[0],
        )

    def _prepare_cube_candidate(
        self,
        frame: np.ndarray,
        box: np.ndarray,
        object_mask: np.ndarray | None,
    ) -> dict[str, Any] | None:
        height, width = frame.shape[:2]
        crop_box = expand_xyxy(box, width, height, self.a2_crop_pad)
        if crop_box is None:
            return None
        x1, y1, x2, y2 = crop_box
        crop = frame[y1 : y2 + 1, x1 : x2 + 1]
        if crop.size == 0:
            return None
        a2_input = cv2.resize(
            crop,
            (self.a2_imgsz, self.a2_imgsz),
            interpolation=cv2.INTER_AREA,
        )
        object_mask_a2 = None
        if object_mask is not None and np.any(object_mask):
            object_mask_crop = object_mask[y1 : y2 + 1, x1 : x2 + 1].astype(np.uint8)
            if object_mask_crop.size:
                object_mask_a2 = cv2.resize(
                    object_mask_crop,
                    (self.a2_imgsz, self.a2_imgsz),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                kernel = np.ones((5, 5), dtype=np.uint8)
                object_mask_a2 = cv2.dilate(
                    object_mask_a2.astype(np.uint8),
                    kernel,
                    iterations=1,
                ).astype(bool)

        return {
            "crop_box": crop_box,
            "crop": crop,
            "a2_input": a2_input,
            "object_mask_a2": object_mask_a2,
        }

    def _process_cube_candidates_batch(
        self,
        frame: np.ndarray,
        cube_jobs: list[tuple[ObjectEvidence, np.ndarray, np.ndarray | None]],
    ) -> None:
        t_prepare = time.perf_counter()
        prepared_jobs: list[tuple[ObjectEvidence, np.ndarray, dict[str, Any]]] = []
        for obj, box, object_mask in cube_jobs:
            prepared = self._prepare_cube_candidate(frame, box, object_mask)
            if prepared is not None:
                prepared_jobs.append((obj, box, prepared))
        self._set_timing("a2_inputs", len(prepared_jobs))
        self._add_timing("a2_prepare_ms", t_prepare)

        if prepared_jobs:
            t0 = time.perf_counter()
            a2_results = self.a2.predict(
                source=[job[2]["a2_input"] for job in prepared_jobs],
                conf=self.a2_conf,
                imgsz=self.a2_imgsz,
                device=self.yolo_device,
                half=self.yolo_half,
                verbose=False,
                batch=max(1, len(prepared_jobs)),
            )
            self._add_timing("a2_ms", t0)
        else:
            a2_results = []

        t_post = time.perf_counter()
        if self.batch_b_across_objects:
            self._process_cube_candidates_batch_frame_b(frame, prepared_jobs, a2_results)
            self._add_timing("a2_post_ms", t_post)
            return

        for result_index, (obj, box, prepared) in enumerate(prepared_jobs):
            faces: list[FaceEvidence] = []
            if result_index < len(a2_results) and a2_results[result_index].boxes is not None and len(a2_results[result_index].boxes):
                faces = self._process_prepared_cube_candidate_result(
                    frame,
                    prepared,
                    obj.object_index,
                    [float(v) for v in box],
                    a2_results[result_index],
                )
            obj.faces = self._final_filter_faces(faces, obj.box_xyxy)
            obj.decision = decide_cube(
                obj.faces,
                target_shape=self.target_shape,
                target_fruit=self.target_fruit,
            )

        for obj, _box, _mask in cube_jobs:
            if obj.decision is None:
                obj.decision = decide_cube(
                    obj.faces,
                    target_shape=self.target_shape,
                    target_fruit=self.target_fruit,
                )

        self._add_timing("a2_post_ms", t_post)

    def _process_cube_candidates_batch_frame_b(
        self,
        frame: np.ndarray,
        prepared_jobs: list[tuple[ObjectEvidence, np.ndarray, dict[str, Any]]],
        a2_results: Any,
    ) -> None:
        contexts: list[dict[str, Any]] = []
        b_items: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
        b_map: list[tuple[int, int]] = []

        for result_index, (obj, box, prepared) in enumerate(prepared_jobs):
            context = {
                "obj": obj,
                "box": box,
                "object_box_xyxy": [float(v) for v in box],
                "candidates": [],
            }
            contexts.append(context)
            if result_index >= len(a2_results):
                continue
            result = a2_results[result_index]
            if result.boxes is None or len(result.boxes) == 0:
                continue

            x1, y1, _x2, _y2 = prepared["crop_box"]
            crop = prepared["crop"]
            a2_input = prepared["a2_input"]
            object_mask_a2 = prepared["object_mask_a2"]
            crop_h, crop_w = a2_input.shape[:2]
            scale_xy = (
                crop.shape[1] / max(a2_input.shape[1], 1),
                crop.shape[0] / max(a2_input.shape[0], 1),
            )
            context["origin"] = (x1, y1)
            context["scale_xy"] = scale_xy
            candidates = self._parse_face_candidates(result, crop_h, crop_w, object_mask_a2)
            context["candidates"] = candidates
            for candidate_index, candidate in enumerate(candidates):
                b_items.append((candidate["face_mask"], candidate["b_box"]))
                b_map.append((len(contexts) - 1, candidate_index))

        quads = self._predict_quads(b_items)
        self._set_timing("b_faces", int(self.last_timing.get("b_faces", 0)) + len(b_items))
        grouped_quads: list[list[np.ndarray | None]] = [
            [None] * len(context["candidates"]) for context in contexts
        ]
        for quad, (context_index, candidate_index) in zip(quads, b_map):
            grouped_quads[context_index][candidate_index] = quad

        global_c_entries: list[dict[str, Any]] = []
        global_c_batch_indices: list[int] = []
        global_c_batch_crops: list[np.ndarray] = []
        global_c_skipped_plain_faces = 0

        for context, quads_for_context in zip(contexts, grouped_quads):
            obj = context["obj"]
            candidates = context["candidates"]
            if candidates:
                faces = self._build_faces_from_candidates(
                    candidates,
                    quads_for_context,
                    context["scale_xy"],
                    context["origin"],
                )
                faces = self._refine_faces(
                    frame,
                    faces,
                    context["object_box_xyxy"],
                )
                if self.batch_c_across_objects:
                    entries, batch_indices, batch_crops, skipped_plain_faces = self._prepare_c_entries(frame, faces)
                    offset = len(global_c_entries)
                    context["c_entries"] = entries
                    global_c_entries.extend(entries)
                    global_c_batch_indices.extend(offset + idx for idx in batch_indices)
                    global_c_batch_crops.extend(batch_crops)
                    global_c_skipped_plain_faces += skipped_plain_faces
                    faces = []
                else:
                    faces = self._classify_faces_for_object(frame, faces, int(obj.object_index))
            else:
                faces = []
                if self.batch_c_across_objects:
                    context["c_entries"] = []
            obj.faces = self._final_filter_faces(faces, obj.box_xyxy)
            obj.decision = decide_cube(
                obj.faces,
                target_shape=self.target_shape,
                target_fruit=self.target_fruit,
            )

        if self.batch_c_across_objects:
            self._set_timing(
                "c_skipped_plain_faces",
                int(self.last_timing.get("c_skipped_plain_faces", 0)) + global_c_skipped_plain_faces,
            )
            self._set_timing("c_faces", int(self.last_timing.get("c_faces", 0)) + len(global_c_batch_crops))
            for entry_index, (label, confidence) in zip(global_c_batch_indices, self._classify_faces(global_c_batch_crops)):
                if confidence < self.c_conf:
                    label = "unknown"
                global_c_entries[entry_index]["label"] = label
                global_c_entries[entry_index]["confidence"] = float(confidence)

            for context in contexts:
                obj = context["obj"]
                faces = self._finalize_c_entries(context.get("c_entries", []), int(obj.object_index))
                obj.faces = self._final_filter_faces(faces, obj.box_xyxy)
                obj.decision = decide_cube(
                    obj.faces,
                    target_shape=self.target_shape,
                    target_fruit=self.target_fruit,
                )

    def _final_filter_faces(
        self,
        faces: list[FaceEvidence],
        object_box_xyxy: list[float],
    ) -> list[FaceEvidence]:
        if self.final_face_filter_mode == "reindex":
            return reindex_faces(faces)
        return filter_cube_faces(faces, object_box_xyxy)

    def _process_prepared_cube_candidate_result(
        self,
        frame: np.ndarray,
        prepared: dict[str, Any],
        object_index: int,
        object_box_xyxy: list[float],
        result: Any,
    ) -> list[FaceEvidence]:
        x1, y1, _x2, _y2 = prepared["crop_box"]
        crop = prepared["crop"]
        a2_input = prepared["a2_input"]
        object_mask_a2 = prepared["object_mask_a2"]
        scale_x = crop.shape[1] / max(a2_input.shape[1], 1)
        scale_y = crop.shape[0] / max(a2_input.shape[0], 1)
        return self._process_faces(
            frame,
            a2_input,
            crop,
            (x1, y1),
            (scale_x, scale_y),
            object_index,
            object_box_xyxy,
            object_mask_a2,
            result,
        )

    def _process_faces(
        self,
        frame: np.ndarray,
        a2_input: np.ndarray,
        original_crop: np.ndarray,
        origin: tuple[int, int],
        scale_xy: tuple[float, float],
        object_index: int,
        object_box_xyxy: list[float],
        object_mask_a2: np.ndarray | None,
        result: Any,
    ) -> list[FaceEvidence]:
        crop_h, crop_w = a2_input.shape[:2]
        face_candidates = self._parse_face_candidates(result, crop_h, crop_w, object_mask_a2)

        quads = self._predict_quads([(item["face_mask"], item["b_box"]) for item in face_candidates])
        self._set_timing("b_faces", int(self.last_timing.get("b_faces", 0)) + len(face_candidates))

        faces = self._build_faces_from_candidates(face_candidates, quads, scale_xy, origin)
        return self._refine_and_classify_faces(frame, faces, object_index, object_box_xyxy)

    def _parse_face_candidates(
        self,
        result: Any,
        crop_h: int,
        crop_w: int,
        object_mask_a2: np.ndarray | None,
    ) -> list[dict[str, Any]]:
        t_parse = time.perf_counter()
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        face_candidates: list[dict[str, Any]] = []
        for index in np.argsort(confidences)[::-1]:
            kind = get_name(result.names, int(classes[index])).lower()
            face_mask = result_mask(result, int(index), (crop_h, crop_w))
            if face_mask is None:
                box = boxes[index]
                face_mask = np.zeros((crop_h, crop_w), dtype=bool)
                face_box = expand_xyxy(box, crop_w, crop_h, 0.0)
                if face_box is not None:
                    fx1, fy1, fx2, fy2 = face_box
                    face_mask[fy1 : fy2 + 1, fx1 : fx2 + 1] = True
            visible_pixels = int(face_mask.sum())
            if visible_pixels < self.min_face_pixels:
                continue
            if object_mask_a2 is not None and np.any(object_mask_a2):
                overlap_pixels = int(np.logical_and(face_mask, object_mask_a2).sum())
                overlap_ratio = overlap_pixels / max(visible_pixels, 1)
                if overlap_ratio < self.min_face_object_overlap:
                    continue

            face_box = mask_to_box(face_mask)
            if face_box is None:
                continue
            b_box = expand_xyxy(face_box, crop_w, crop_h, self.b_crop_pad)
            if b_box is None:
                continue

            face_candidates.append(
                {
                    "kind": kind,
                    "face_mask": face_mask,
                    "face_box": face_box,
                    "b_box": b_box,
                    "confidence": float(confidences[index]),
                    "visible_pixels": visible_pixels,
                }
            )
        self._add_timing("a2_face_parse_ms", t_parse)
        return face_candidates

    def _build_faces_from_candidates(
        self,
        face_candidates: list[dict[str, Any]],
        quads: list[np.ndarray | None],
        scale_xy: tuple[float, float],
        origin: tuple[int, int],
    ) -> list[FaceEvidence]:
        faces: list[FaceEvidence] = []
        scale_x, scale_y = scale_xy
        t_build = time.perf_counter()
        for item, quad in zip(face_candidates, quads):
            if quad is None:
                continue
            kind = item["kind"]
            face_mask = item["face_mask"]
            face_box = item["face_box"]
            visible_pixels = item["visible_pixels"]
            quad_original = quad.copy()
            quad_original[:, 0] *= scale_x
            quad_original[:, 1] *= scale_y

            ox, oy = origin
            quad_full = (quad_original + np.asarray([ox, oy], dtype=np.float32)).tolist()
            box_full = [
                float(face_box[0] * scale_x + ox),
                float(face_box[1] * scale_y + oy),
                float(face_box[2] * scale_x + ox),
                float(face_box[3] * scale_y + oy),
            ]
            face_segments = transform_polygons_xy(
                mask_to_polygons_xy(face_mask, min_area=8.0),
                scale_x=scale_x,
                scale_y=scale_y,
                offset_x=ox,
                offset_y=oy,
            )
            faces.append(
                FaceEvidence(
                    face_index=len(faces),
                    kind=kind,
                    label="unknown",
                    confidence=0.0,
                    detector_confidence=float(item["confidence"]),
                    box_xyxy=box_full,
                    quad_xy=[[float(x), float(y)] for x, y in quad_full],
                    visible_pixels=visible_pixels,
                    raw_quad_xy=[[float(x), float(y)] for x, y in quad_full],
                    segments_xy=face_segments,
                    source="A2+B",
                )
            )
        self._add_timing("a2_face_build_ms", t_build)
        return faces

    def _refine_and_classify_faces(
        self,
        frame: np.ndarray,
        faces: list[FaceEvidence],
        object_index: int,
        object_box_xyxy: list[float],
    ) -> list[FaceEvidence]:
        faces = self._refine_faces(frame, faces, object_box_xyxy)
        return self._classify_faces_for_object(frame, faces, object_index)

    def _refine_faces(
        self,
        frame: np.ndarray,
        faces: list[FaceEvidence],
        object_box_xyxy: list[float],
    ) -> list[FaceEvidence]:
        t_filter = time.perf_counter()
        t_dedupe = time.perf_counter()
        faces = filter_cube_faces(faces, object_box_xyxy, require_sane=False)
        self._add_timing("a2_face_dedupe_ms", t_dedupe)
        do_pose_refine = self.refine_quads in {"pose", "pose_fast_iou"} or (
            self.refine_quads == "pose_fruit"
            and any(face.kind.lower() not in PLAIN_LABELS for face in faces)
        )
        if do_pose_refine:
            t_sane = time.perf_counter()
            refine_shared_edges = cube_refine_shared_edges_if_sane(faces, object_box_xyxy)
            self._add_timing("a2_refine_sane_ms", t_sane)
            if refine_shared_edges:
                t_refine = time.perf_counter()
                faces = refine_cube_quads_by_pose(
                    faces,
                    object_box_xyxy,
                    frame.shape[:2],
                    shared_edges=refine_shared_edges,
                    check_segment_iou=self.refine_quads != "pose_fast_iou",
                )
                self._add_timing("a2_refine_ms", t_refine)
        self._add_timing("a2_face_filter_ms", t_filter)
        return faces

    def _classify_faces_for_object(
        self,
        frame: np.ndarray,
        faces: list[FaceEvidence],
        object_index: int,
    ) -> list[FaceEvidence]:
        classified_entries, c_batch_indices, c_batch_crops, c_skipped_plain_faces = self._prepare_c_entries(frame, faces)
        self._set_timing("c_skipped_plain_faces", int(self.last_timing.get("c_skipped_plain_faces", 0)) + c_skipped_plain_faces)
        self._set_timing("c_faces", int(self.last_timing.get("c_faces", 0)) + len(c_batch_crops))
        for entry_index, (label, confidence) in zip(c_batch_indices, self._classify_faces(c_batch_crops)):
            if confidence < self.c_conf:
                label = "unknown"
            classified_entries[entry_index]["label"] = label
            classified_entries[entry_index]["confidence"] = float(confidence)
        return self._finalize_c_entries(classified_entries, object_index)

    def _prepare_c_entries(
        self,
        frame: np.ndarray,
        faces: list[FaceEvidence],
    ) -> tuple[list[dict[str, Any]], list[int], list[np.ndarray], int]:
        classified_entries: list[dict[str, Any]] = []
        c_batch_crops: list[np.ndarray] = []
        c_batch_indices: list[int] = []
        c_skipped_plain_faces = 0
        t_crops = time.perf_counter()
        for face in faces:
            if (
                self.classify_fruit_faces_only
                and face.kind.lower() in PLAIN_LABELS
                and float(face.detector_confidence) >= self.plain_face_fastpath_min_conf
            ):
                classified_entries.append(
                    {
                        "face": face,
                        "face_crop": np.empty((0, 0, 3), dtype=frame.dtype),
                        "c_crop_info": {
                            "image": None,
                            "occlusion_blackened": False,
                            "visible_ratio": 1.0,
                            "visible_to_full_ratio": 1.0,
                            "visible_pixels": int(face.visible_pixels),
                            "full_pixels": 0,
                            "black_fraction": 0.0,
                            "reject_reason": "",
                        },
                        "reject_reason": "",
                        "used_projective": False,
                        "raw_crop_info": None,
                        "label": "plain",
                        "confidence": float(face.detector_confidence),
                    }
                )
                c_skipped_plain_faces += 1
                continue

            quad_full = np.asarray(face.quad_xy, dtype=np.float32)
            c_crop_info = self._build_c_crop(frame, face, quad_full)
            if c_crop_info is None:
                continue

            used_projective = bool(
                self.refine_quads in {"pose", "pose_fruit", "pose_fast_iou"}
                and face.raw_quad_xy
                and np.asarray(face.raw_quad_xy, dtype=np.float32).shape == quad_full.shape
                and not np.allclose(np.asarray(face.raw_quad_xy, dtype=np.float32), quad_full, atol=0.5)
            )
            raw_crop_info = None
            if c_crop_info["reject_reason"] and used_projective:
                raw_quad = np.asarray(face.raw_quad_xy, dtype=np.float32)
                raw_crop_info = self._build_c_crop(frame, face, raw_quad)
                if raw_crop_info is not None and not raw_crop_info["reject_reason"]:
                    face.quad_xy = [[float(x), float(y)] for x, y in raw_quad.tolist()]
                    clear_face_geometry_cache(face)
                    quad_full = raw_quad
                    c_crop_info = raw_crop_info
                    used_projective = False

            face_crop = c_crop_info["image"]
            if face_crop is None or face_crop.size == 0:
                continue

            c_reject_reason = str(c_crop_info["reject_reason"])
            entry = {
                "face": face,
                "face_crop": face_crop,
                "c_crop_info": c_crop_info,
                "reject_reason": c_reject_reason,
                "used_projective": used_projective,
                "raw_crop_info": raw_crop_info,
                "label": "unknown",
                "confidence": 0.0,
            }
            if c_reject_reason:
                pass
            else:
                c_batch_indices.append(len(classified_entries))
                c_batch_crops.append(face_crop)
            classified_entries.append(entry)
        self._add_timing("c_crop_loop_ms", t_crops)
        return classified_entries, c_batch_indices, c_batch_crops, c_skipped_plain_faces

    def _finalize_c_entries(
        self,
        classified_entries: list[dict[str, Any]],
        object_index: int,
    ) -> list[FaceEvidence]:
        t_label = time.perf_counter()
        classified_faces: list[FaceEvidence] = []
        for entry in classified_entries:
            face = entry["face"]
            face_crop = entry["face_crop"]
            c_crop_info = entry["c_crop_info"]
            c_reject_reason = entry["reject_reason"]
            used_projective = entry["used_projective"]
            raw_crop_info = entry["raw_crop_info"]
            label = entry["label"]
            confidence = float(entry["confidence"])
            face.label = str(label)
            face.confidence = confidence
            if used_projective:
                face.source = "A2+B+projective+C"
            else:
                face.source = "A2+B+C"
            classified_faces.append(face)
            if self.keep_c_inputs:
                self.last_c_inputs.append(
                    {
                        "object_index": int(object_index),
                        "face_index": int(len(classified_faces) - 1),
                        "kind": face.kind,
                        "label": label,
                        "confidence": float(confidence),
                        "occlusion_blackened": bool(c_crop_info["occlusion_blackened"]),
                        "visible_ratio": float(c_crop_info["visible_ratio"]),
                        "visible_to_full_ratio": float(c_crop_info["visible_to_full_ratio"]),
                        "visible_pixels": int(c_crop_info["visible_pixels"]),
                        "full_pixels": int(c_crop_info["full_pixels"]),
                        "black_fraction": float(c_crop_info["black_fraction"]),
                        "reject_reason": c_reject_reason,
                        "projective_fallback": bool(raw_crop_info is not None and raw_crop_info is c_crop_info),
                        "image": face_crop.copy(),
                    }
                )

        self._add_timing("c_label_apply_ms", t_label)
        return reindex_faces(classified_faces)

    def _build_c_crop(
        self,
        frame: np.ndarray,
        face: FaceEvidence,
        quad_full: np.ndarray,
    ) -> dict[str, Any] | None:
        t_source = time.perf_counter()
        visible_ratio, visible_to_full_ratio, c_visible_pixels, c_full_pixels = local_face_visibility_stats(
            frame.shape,
            face,
            quad_full,
        )
        occlusion_blackened = bool(
            self.blacken_c_occluded_face_area
            and visible_ratio < float(self.c_occlusion_visible_ratio)
            and c_full_pixels > 0
        )
        self._add_timing("c_source_ms", t_source)
        t_warp = time.perf_counter()
        face_crop = warp_quad(frame, quad_full, self.c_warp_size)
        if face_crop is not None and face_crop.size and occlusion_blackened:
            visible_crop_mask = visible_crop_mask_from_segments(face.segments_xy, quad_full, self.c_warp_size)
            if visible_crop_mask is not None:
                hidden = ~visible_crop_mask
                if np.any(hidden):
                    face_crop = face_crop.copy()
                    face_crop[hidden] = 0
                else:
                    occlusion_blackened = False
            else:
                occlusion_blackened = False
        self._add_timing("c_warp_ms", t_warp)
        if face_crop is None or face_crop.size == 0:
            return None
        t_filter = time.perf_counter()
        black_fraction = float(
            ((face_crop[:, :, 0] < 8) & (face_crop[:, :, 1] < 8) & (face_crop[:, :, 2] < 8)).mean()
        )
        c_reject_reason = c_filter_reject_reason(
            quad_full,
            visible_pixels=c_visible_pixels,
            full_pixels=c_full_pixels,
            visible_ratio=visible_ratio,
            visible_to_full_ratio=visible_to_full_ratio,
            black_fraction=black_fraction,
            min_face_pixels=self.min_face_pixels,
            c_min_quad_area=self.c_min_quad_area,
            c_min_quad_side=self.c_min_quad_side,
            c_max_quad_aspect=self.c_max_quad_aspect,
            c_min_visible_pixels=self.c_min_visible_pixels,
            c_min_visible_ratio=self.c_min_visible_ratio,
            c_max_visible_to_full_ratio=self.c_max_visible_to_full_ratio,
            c_max_black_fraction=self.c_max_black_fraction,
        )
        self._add_timing("c_filter_ms", t_filter)
        return {
            "image": face_crop,
            "occlusion_blackened": bool(occlusion_blackened),
            "visible_ratio": float(visible_ratio),
            "visible_to_full_ratio": float(visible_to_full_ratio),
            "visible_pixels": int(c_visible_pixels),
            "full_pixels": int(c_full_pixels),
            "black_fraction": float(black_fraction),
            "reject_reason": c_reject_reason,
        }

    def _predict_quad(
        self,
        face_mask: np.ndarray,
        b_box: tuple[int, int, int, int],
    ) -> np.ndarray | None:
        first_quad = self._predict_quad_once(face_mask, b_box)
        if first_quad is not None:
            if self.b_refine_passes >= 2:
                predicted_box = bbox_from_points(first_quad)
                if predicted_box is not None:
                    height, width = face_mask.shape[:2]
                    second_box = expand_xyxy(predicted_box, width, height, self.b_crop_pad)
                    if second_box is not None and bbox_iou(list(map(float, b_box)), list(map(float, second_box))) < 0.98:
                        second_quad = self._predict_quad_once(face_mask, second_box)
                        if second_quad is not None:
                            return second_quad
            return first_quad

        fallback = quad_from_mask(face_mask)
        return (
            quad_passes(
                fallback,
                min_area=self.b_min_quad_area,
                min_side=self.b_min_quad_side,
                max_aspect=self.b_max_quad_aspect,
            )
            if fallback is not None
            else None
        )

    def _predict_quad_once(
        self,
        face_mask: np.ndarray,
        b_box: tuple[int, int, int, int],
    ) -> np.ndarray | None:
        import torch

        x1, y1, x2, y2 = b_box
        mask_crop = face_mask[y1 : y2 + 1, x1 : x2 + 1].astype(np.uint8) * 255
        if mask_crop.size == 0:
            return None
        mask_input = cv2.resize(
            mask_crop,
            (self.b_imgsz, self.b_imgsz),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32) / 255.0
        tensor = torch.from_numpy(mask_input[None, None, :, :]).to(self.b_device)
        with torch.no_grad():
            pred = self.b(tensor).detach().cpu().numpy().reshape(4, 2)

        box_w = max(1.0, float(x2 - x1 + 1))
        box_h = max(1.0, float(y2 - y1 + 1))
        quad = np.column_stack([x1 + pred[:, 0] * box_w, y1 + pred[:, 1] * box_h]).astype(np.float32)
        quad = quad_passes(
            quad,
            min_area=self.b_min_quad_area,
            min_side=self.b_min_quad_side,
            max_aspect=self.b_max_quad_aspect,
        )
        if quad is not None:
            return quad
        return None

    def _predict_quads(
        self,
        items: list[tuple[np.ndarray, tuple[int, int, int, int]]],
    ) -> list[np.ndarray | None]:
        if not items:
            return []

        first_quads = self._predict_quads_once(items)
        output: list[np.ndarray | None] = []
        second_pass_items: list[tuple[int, np.ndarray, tuple[int, int, int, int]]] = []

        for idx, ((face_mask, b_box), first_quad) in enumerate(zip(items, first_quads)):
            if first_quad is None:
                fallback = quad_from_mask(face_mask)
                output.append(
                    quad_passes(
                        fallback,
                        min_area=self.b_min_quad_area,
                        min_side=self.b_min_quad_side,
                        max_aspect=self.b_max_quad_aspect,
                    )
                    if fallback is not None
                    else None
                )
                continue
            output.append(first_quad)
            if self.b_refine_passes >= 2:
                predicted_box = bbox_from_points(first_quad)
                if predicted_box is not None:
                    height, width = face_mask.shape[:2]
                    second_box = expand_xyxy(predicted_box, width, height, self.b_crop_pad)
                    if second_box is not None and bbox_iou(list(map(float, b_box)), list(map(float, second_box))) < 0.98:
                        second_pass_items.append((idx, face_mask, second_box))

        if second_pass_items:
            second_quads = self._predict_quads_once([(mask, box) for _idx, mask, box in second_pass_items])
            for (idx, _mask, _box), second_quad in zip(second_pass_items, second_quads):
                if second_quad is not None:
                    output[idx] = second_quad

        return output

    def _predict_quads_once(
        self,
        items: list[tuple[np.ndarray, tuple[int, int, int, int]]],
    ) -> list[np.ndarray | None]:
        import torch

        outputs: list[np.ndarray | None] = [None] * len(items)
        tensors: list[np.ndarray] = []
        metas: list[tuple[int, int, int, int, int]] = []
        for idx, (face_mask, b_box) in enumerate(items):
            x1, y1, x2, y2 = b_box
            mask_crop = face_mask[y1 : y2 + 1, x1 : x2 + 1].astype(np.uint8) * 255
            if mask_crop.size == 0:
                continue
            mask_input = cv2.resize(
                mask_crop,
                (self.b_imgsz, self.b_imgsz),
                interpolation=cv2.INTER_AREA,
            ).astype(np.float32) / 255.0
            tensors.append(mask_input)
            metas.append((idx, x1, y1, x2, y2))

        if not tensors:
            return outputs

        t0 = time.perf_counter()
        tensor = torch.from_numpy(np.stack(tensors, axis=0)[:, None, :, :]).to(self.b_device)
        with torch.inference_mode():
            preds = self.b(tensor).detach().cpu().numpy().reshape(-1, 4, 2)
        self._add_timing("b_ms", t0)

        for pred, (idx, x1, y1, x2, y2) in zip(preds, metas):
            box_w = max(1.0, float(x2 - x1 + 1))
            box_h = max(1.0, float(y2 - y1 + 1))
            quad = np.column_stack([x1 + pred[:, 0] * box_w, y1 + pred[:, 1] * box_h]).astype(np.float32)
            outputs[idx] = quad_passes(
                quad,
                min_area=self.b_min_quad_area,
                min_side=self.b_min_quad_side,
                max_aspect=self.b_max_quad_aspect,
            )
        return outputs

    def _classify_face(self, face_crop: np.ndarray) -> tuple[str, float]:
        if self.c_onnx_session is not None:
            return self._classify_faces([face_crop])[0]
        import torch
        import torch.nn.functional as F

        image = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.c_imgsz, self.c_imgsz), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        tensor = torch.from_numpy(image.transpose(2, 0, 1)[None, :, :, :]).to(self.torch_device)
        with torch.no_grad():
            probs = F.softmax(self.c(tensor), dim=1)[0].detach().cpu().numpy()
        class_id = int(np.argmax(probs))
        return str(self.c_classes[class_id]), float(probs[class_id])

    def _classify_faces(self, face_crops: list[np.ndarray]) -> list[tuple[str, float]]:
        import torch
        import torch.nn.functional as F

        if not face_crops:
            return []
        t_preprocess = time.perf_counter()
        images = []
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        for face_crop in face_crops:
            image = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.c_imgsz, self.c_imgsz), interpolation=cv2.INTER_AREA)
            image = image.astype(np.float32) / 255.0
            images.append(((image - mean) / std).transpose(2, 0, 1))
        self._add_timing("c_preprocess_ms", t_preprocess)

        t0 = time.perf_counter()
        batch = np.stack(images, axis=0).astype(np.float32, copy=False)
        if self.c_onnx_session is not None:
            logits = self.c_onnx_session.run(None, {self.c_onnx_input_name: batch})[0]
            logits = logits - np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        else:
            tensor = torch.from_numpy(batch).to(self.torch_device)
            with torch.inference_mode():
                probs = F.softmax(self.c(tensor), dim=1).detach().cpu().numpy()
        self._add_timing("c_ms", t0)
        class_ids = np.argmax(probs, axis=1).astype(int)
        return [(str(self.c_classes[class_id]), float(probs[row, class_id])) for row, class_id in enumerate(class_ids)]


def decide_shape(class_name: str, *, target_shape: str = "") -> ObjectDecision:
    if target_shape and class_name == target_shape:
        return ObjectDecision(
            identity=class_name,
            action="pickup",
            reason=f"target shape visible; rulebook set-1 object is worth 10 points",
            visible_faces=0,
            blank_faces=0,
            fruit_faces={},
            unknown_faces=0,
            expected_score=10,
        )
    action = "skip" if target_shape else "inspect"
    reason = "non-target shape" if target_shape else "shape detected; configure --target-shape for pickup decisions"
    return ObjectDecision(
        identity=class_name,
        action=action,
        reason=reason,
        visible_faces=0,
        blank_faces=0,
        fruit_faces={},
        unknown_faces=0,
    )


def filter_cube_faces(
    faces: list[FaceEvidence],
    object_box_xyxy: list[float],
    *,
    duplicate_iou_threshold: float = 0.35,
    duplicate_center_threshold: float = 0.18,
    max_faces: int = 3,
    require_sane: bool = True,
) -> list[FaceEvidence]:
    if len(faces) <= 1:
        return faces

    ranked = sorted(faces, key=face_rank_score, reverse=True)
    kept: list[FaceEvidence] = []
    object_diag = bbox_diag(object_box_xyxy)
    for face in ranked:
        if any(
            faces_are_duplicates(
                face,
                kept_face,
                object_diag=object_diag,
                iou_threshold=duplicate_iou_threshold,
                center_threshold=duplicate_center_threshold,
            )
            for kept_face in kept
        ):
            continue
        kept.append(face)

    if len(kept) > max_faces:
        kept = choose_geometrically_spread_faces(kept, object_box_xyxy, max_faces)

    if require_sane and len(kept) == max_faces and not cube_face_group_is_sane(kept, object_box_xyxy):
        kept = choose_geometrically_spread_faces(kept, object_box_xyxy, max_faces - 1)

    return reindex_faces(kept)


def cube_refine_input_is_sane(
    faces: list[FaceEvidence],
    object_box_xyxy: list[float],
) -> bool:
    return bool(cube_refine_shared_edges_if_sane(faces, object_box_xyxy))


def cube_refine_shared_edges_if_sane(
    faces: list[FaceEvidence],
    object_box_xyxy: list[float],
) -> list[SharedEdge]:
    if len(faces) < 2:
        return []
    object_diag = bbox_diag(object_box_xyxy)
    top_faces = faces[:3]
    if len(top_faces) >= 3 and not cube_face_group_is_sane(top_faces, object_box_xyxy):
        return []
    shared_edges = find_shared_edges(top_faces, object_diag)
    required_edges = max(1, min(len(top_faces) - 1, 2))
    return shared_edges if len(shared_edges) >= required_edges else []


def face_rank_score(face: FaceEvidence) -> float:
    label_weight = 1.15 if face.label.lower() in FRUIT_CLASSES else 1.0
    return (
        float(face.detector_confidence) * 0.55
        + float(face.confidence) * 0.35
        + min(float(face.visible_pixels) / 60000.0, 1.0) * 0.10
    ) * label_weight


def faces_are_duplicates(
    face_a: FaceEvidence,
    face_b: FaceEvidence,
    *,
    object_diag: float,
    iou_threshold: float,
    center_threshold: float,
) -> bool:
    box_overlap = bbox_iou(face_a.box_xyxy, face_b.box_xyxy)
    if box_overlap >= max(iou_threshold, 0.45):
        return True
    center_dist = float(np.linalg.norm(face_center(face_a) - face_center(face_b)))
    if center_dist <= object_diag * center_threshold:
        return True
    if box_overlap > 0.0 and face_polygon_iou(face_a, face_b) >= iou_threshold:
        return True
    return False


def choose_geometrically_spread_faces(
    faces: list[FaceEvidence],
    object_box_xyxy: list[float],
    max_faces: int,
) -> list[FaceEvidence]:
    import itertools

    best_combo: tuple[FaceEvidence, ...] | None = None
    best_score = -1e9
    for combo in itertools.combinations(faces, max_faces):
        score = face_group_geometry_score(combo, object_box_xyxy)
        if score > best_score:
            best_score = score
            best_combo = combo
    return list(best_combo) if best_combo is not None else faces[:max_faces]


def cube_face_group_is_sane(
    faces: list[FaceEvidence],
    object_box_xyxy: list[float],
) -> bool:
    if len(faces) < 3:
        return True

    centers = np.asarray([face_center(face) for face in faces], dtype=np.float32)
    x1, y1, x2, y2 = [float(v) for v in object_box_xyxy]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    normalized = np.column_stack([(centers[:, 0] - x1) / width, (centers[:, 1] - y1) / height])

    if np.any(normalized < -0.12) or np.any(normalized > 1.12):
        return False

    spread_x = float(normalized[:, 0].max() - normalized[:, 0].min())
    spread_y = float(normalized[:, 1].max() - normalized[:, 1].min())
    if spread_x < 0.18 or spread_y < 0.12:
        return False

    triangle_area = abs(
        float(
            cv2.contourArea(
                normalized.astype(np.float32).reshape(-1, 1, 2),
            )
        )
    )
    if triangle_area < 0.012:
        return False

    object_diag = bbox_diag(object_box_xyxy)
    close_edge_pairs = 0
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            box_overlap = bbox_iou(faces[i].box_xyxy, faces[j].box_xyxy)
            if box_overlap >= 0.55:
                return False
            if box_overlap > 0.0 and face_polygon_iou(faces[i], faces[j]) >= 0.30:
                return False
            center_dist = float(np.linalg.norm(centers[i] - centers[j])) / max(object_diag, 1.0)
            if center_dist < 0.14:
                return False
            edge_distance = face_edge_distance(faces[i], faces[j])
            if edge_distance <= object_diag * 0.09:
                close_edge_pairs += 1

    return close_edge_pairs >= 2


def face_group_geometry_score(
    faces: tuple[FaceEvidence, ...],
    object_box_xyxy: list[float],
) -> float:
    centers = np.asarray([face_center(face) for face in faces], dtype=np.float32)
    object_diag = bbox_diag(object_box_xyxy)
    pair_dist = 0.0
    overlap_penalty = 0.0
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            pair_dist += float(np.linalg.norm(centers[i] - centers[j])) / max(object_diag, 1.0)
            overlap_penalty += face_polygon_iou(faces[i], faces[j])

    x1, y1, x2, y2 = [float(v) for v in object_box_xyxy]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    normalized = np.column_stack([(centers[:, 0] - x1) / width, (centers[:, 1] - y1) / height])
    spread_x = float(normalized[:, 0].max() - normalized[:, 0].min())
    spread_y = float(normalized[:, 1].max() - normalized[:, 1].min())
    area_bonus = sum(min(face_area(face) / max(width * height, 1.0), 0.5) for face in faces)
    confidence_bonus = sum(face_rank_score(face) for face in faces) / max(len(faces), 1)
    return pair_dist + spread_x + spread_y + area_bonus + confidence_bonus - overlap_penalty * 3.0


def reindex_faces(faces: list[FaceEvidence]) -> list[FaceEvidence]:
    out: list[FaceEvidence] = []
    for index, face in enumerate(faces):
        face.face_index = index
        out.append(face)
    return out


def face_center(face: FaceEvidence) -> np.ndarray:
    cached = getattr(face, "_center_xy", None)
    if cached is not None:
        return cached
    quad = np.asarray(face.quad_xy, dtype=np.float32)
    if quad.shape == (4, 2) and np.isfinite(quad).all():
        center = quad.mean(axis=0)
        setattr(face, "_center_xy", center)
        return center
    x1, y1, x2, y2 = [float(v) for v in face.box_xyxy]
    center = np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
    setattr(face, "_center_xy", center)
    return center


def face_area(face: FaceEvidence) -> float:
    cached = getattr(face, "_area_xy", None)
    if cached is not None:
        return float(cached)
    quad = np.asarray(face.quad_xy, dtype=np.float32)
    if quad.shape == (4, 2) and np.isfinite(quad).all():
        area = float(abs(cv2.contourArea(quad.reshape(-1, 1, 2))))
        setattr(face, "_area_xy", area)
        return area
    x1, y1, x2, y2 = [float(v) for v in face.box_xyxy]
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    setattr(face, "_area_xy", area)
    return area


def bbox_diag(box: list[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    return float(math.hypot(max(0.0, x2 - x1), max(0.0, y2 - y1)))


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else float(inter / union)


def polygon_iou(
    polygons_a: list[list[list[float]]],
    polygons_b: list[list[list[float]]],
) -> float:
    contours_a = polygon_contours_unclipped(polygons_a)
    contours_b = polygon_contours_unclipped(polygons_b)
    return polygon_iou_from_contours(contours_a, contours_b)


def polygon_iou_from_contours(contours_a: list[np.ndarray], contours_b: list[np.ndarray]) -> float:
    if not contours_a or not contours_b:
        return 0.0
    all_points = np.vstack([c.reshape(-1, 2) for c in contours_a] + [c.reshape(-1, 2) for c in contours_b])
    min_xy = np.floor(all_points.min(axis=0)).astype(np.int32)
    max_xy = np.ceil(all_points.max(axis=0)).astype(np.int32)
    width = int(max(1, max_xy[0] - min_xy[0] + 3))
    height = int(max(1, max_xy[1] - min_xy[1] + 3))
    if width * height > 2_000_000:
        return 0.0
    offset = min_xy - 1
    mask_a = np.zeros((height, width), dtype=np.uint8)
    mask_b = np.zeros((height, width), dtype=np.uint8)
    shifted_a = [(contour.reshape(-1, 2) - offset).astype(np.int32).reshape(-1, 1, 2) for contour in contours_a]
    shifted_b = [(contour.reshape(-1, 2) - offset).astype(np.int32).reshape(-1, 1, 2) for contour in contours_b]
    cv2.fillPoly(mask_a, shifted_a, 1)
    cv2.fillPoly(mask_b, shifted_b, 1)
    inter = int(cv2.countNonZero(cv2.bitwise_and(mask_a, mask_b)))
    union = int(cv2.countNonZero(cv2.bitwise_or(mask_a, mask_b)))
    return 0.0 if union <= 0 else float(inter / union)


def polygon_contours_unclipped(polygons: list[list[list[float]]]) -> list[np.ndarray]:
    contours: list[np.ndarray] = []
    for polygon in polygons:
        pts = np.asarray(polygon, dtype=np.float32)
        if pts.ndim == 2 and pts.shape[0] >= 3 and pts.shape[1] == 2 and np.isfinite(pts).all():
            contours.append(np.rint(pts).astype(np.int32).reshape(-1, 1, 2))
    return contours


def face_segments_contours(face: FaceEvidence) -> list[np.ndarray]:
    cached = getattr(face, "_segments_contours_xy", None)
    if cached is None:
        cached = polygon_contours_unclipped(face.segments_xy)
        setattr(face, "_segments_contours_xy", cached)
    return cached


def face_polygon_iou(face_a: FaceEvidence, face_b: FaceEvidence) -> float:
    cache = getattr(face_a, "_polygon_iou_cache", None)
    if cache is None:
        cache = {}
        setattr(face_a, "_polygon_iou_cache", cache)
    key = id(face_b)
    if key in cache:
        return float(cache[key])
    value = polygon_iou_from_contours(face_segments_contours(face_a), face_segments_contours(face_b))
    cache[key] = value
    reverse_cache = getattr(face_b, "_polygon_iou_cache", None)
    if reverse_cache is None:
        reverse_cache = {}
        setattr(face_b, "_polygon_iou_cache", reverse_cache)
    reverse_cache[id(face_a)] = value
    return value


def quad_contour_unclipped(quad: np.ndarray) -> np.ndarray | None:
    pts = np.asarray(quad, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2 or not np.isfinite(pts).all():
        return None
    return np.rint(pts).astype(np.int32).reshape(-1, 1, 2)


def face_edge_distance(face_a: FaceEvidence, face_b: FaceEvidence) -> float:
    points_a = face_edge_points(face_a)
    points_b = face_edge_points(face_b)
    if len(points_a) == 0 or len(points_b) == 0:
        return float("inf")
    distances = np.linalg.norm(points_a[:, None, :] - points_b[None, :, :], axis=2)
    return float(distances.min())


def face_edge_points(face: FaceEvidence) -> np.ndarray:
    cached = getattr(face, "_edge_points_xy", None)
    if cached is not None:
        return cached
    contours = face_segments_contours(face)
    if contours:
        points = np.vstack([contour.reshape(-1, 2).astype(np.float32) for contour in contours])
        setattr(face, "_edge_points_xy", points)
        return points
    quad = np.asarray(face.quad_xy, dtype=np.float32)
    if quad.shape == (4, 2) and np.isfinite(quad).all():
        setattr(face, "_edge_points_xy", quad)
        return quad
    x1, y1, x2, y2 = [float(v) for v in face.box_xyxy]
    points = np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    setattr(face, "_edge_points_xy", points)
    return points


def clear_face_geometry_cache(face: FaceEvidence) -> None:
    for name in ("_center_xy", "_area_xy", "_edge_points_xy", "_polygon_iou_cache"):
        if hasattr(face, name):
            delattr(face, name)


def refine_cube_quads_by_pose(
    faces: list[FaceEvidence],
    object_box_xyxy: list[float],
    image_shape: tuple[int, int],
    *,
    max_mean_error_ratio: float = 0.08,
    max_point_move_ratio: float = 0.18,
    shared_edges: list[SharedEdge] | None = None,
    check_segment_iou: bool = True,
) -> list[FaceEvidence]:
    return refine_cube_quads_by_projective_geometry(
        faces,
        object_box_xyxy,
        image_shape,
        max_mean_error_ratio=max_mean_error_ratio,
        max_point_move_ratio=max_point_move_ratio,
        shared_edges=shared_edges,
        check_segment_iou=check_segment_iou,
    )


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


@dataclass
class SharedEdge:
    face_a: int
    edge_a: int
    face_b: int
    edge_b: int
    reverse: bool
    distance: float


def refine_cube_quads_by_projective_geometry(
    faces: list[FaceEvidence],
    object_box_xyxy: list[float],
    image_shape: tuple[int, int],
    *,
    max_mean_error_ratio: float = 0.08,
    max_point_move_ratio: float = 0.18,
    shared_edges: list[SharedEdge] | None = None,
    check_segment_iou: bool = True,
) -> list[FaceEvidence]:
    if len(faces) < 2:
        return faces

    ordered_faces = reindex_faces(faces[:3])
    raw_quads = [order_quad_points(face.quad_xy) for face in ordered_faces]
    if any(quad is None for quad in raw_quads):
        return faces
    quads = [quad.copy() for quad in raw_quads if quad is not None]

    height, width = image_shape
    object_diag = max(bbox_diag(object_box_xyxy), 1.0)
    if shared_edges is None:
        shared_edges = find_shared_edges(ordered_faces, object_diag)
    if len(shared_edges) < max(1, len(ordered_faces) - 1):
        return faces

    vertex_groups, edge_groups = build_cube_projective_groups(len(ordered_faces), shared_edges)
    quads = snap_shared_vertices(quads, ordered_faces, vertex_groups)
    edge_models = estimate_edge_family_models(quads, edge_groups, object_box_xyxy)
    if len(edge_models) < 3 and len(ordered_faces) >= 2:
        return faces

    refined_quads = intersect_projective_edge_lines(quads, edge_groups, edge_models)
    if refined_quads is None:
        return faces
    refined_quads = snap_shared_vertices(refined_quads, ordered_faces, vertex_groups)

    if not projective_refinement_passes(
        raw_quads=[quad for quad in raw_quads if quad is not None],
        refined_quads=refined_quads,
        faces=ordered_faces,
        object_box_xyxy=object_box_xyxy,
        image_shape=(height, width),
        max_mean_error_ratio=max_mean_error_ratio,
        max_point_move_ratio=max_point_move_ratio,
        check_segment_iou=check_segment_iou,
    ):
        return faces

    refined_faces: list[FaceEvidence] = []
    for face, refined_quad in zip(ordered_faces, refined_quads):
        passed = quad_passes(refined_quad, min_area=30.0, min_side=3.0, max_aspect=30.0)
        if passed is None:
            return faces
        if not face.raw_quad_xy:
            face.raw_quad_xy = [[float(x), float(y)] for x, y in face.quad_xy]
        face.quad_xy = [[float(x), float(y)] for x, y in passed.tolist()]
        refined_faces.append(face)
    return reindex_faces(refined_faces)


def face_edge_global_id(face_index: int, edge_index: int) -> int:
    return face_index * 4 + edge_index % 4


def face_vertex_global_id(face_index: int, vertex_index: int) -> int:
    return face_index * 4 + vertex_index % 4


def find_shared_edges(faces: list[FaceEvidence], object_diag: float) -> list[SharedEdge]:
    shared_edges: list[SharedEdge] = []
    for face_a in range(len(faces)):
        for face_b in range(face_a + 1, len(faces)):
            quad_a = order_quad_points(faces[face_a].quad_xy)
            quad_b = order_quad_points(faces[face_b].quad_xy)
            if quad_a is None or quad_b is None:
                continue
            best: SharedEdge | None = None
            for edge_a in range(4):
                a0 = quad_a[edge_a]
                a1 = quad_a[(edge_a + 1) % 4]
                for edge_b in range(4):
                    b0 = quad_b[edge_b]
                    b1 = quad_b[(edge_b + 1) % 4]
                    same = float(np.linalg.norm(a0 - b0) + np.linalg.norm(a1 - b1)) * 0.5
                    reverse = float(np.linalg.norm(a0 - b1) + np.linalg.norm(a1 - b0)) * 0.5
                    distance = min(same, reverse)
                    candidate = SharedEdge(
                        face_a=face_a,
                        edge_a=edge_a,
                        face_b=face_b,
                        edge_b=edge_b,
                        reverse=reverse < same,
                        distance=distance,
                    )
                    if best is None or candidate.distance < best.distance:
                        best = candidate
            if best is not None and best.distance <= object_diag * 0.16:
                shared_edges.append(best)

    shared_edges.sort(key=lambda edge: edge.distance)
    selected: list[SharedEdge] = []
    pair_keys: set[tuple[int, int]] = set()
    for edge in shared_edges:
        pair_key = (edge.face_a, edge.face_b)
        if pair_key in pair_keys:
            continue
        selected.append(edge)
        pair_keys.add(pair_key)
    return selected


def build_cube_projective_groups(
    face_count: int,
    shared_edges: list[SharedEdge],
) -> tuple[DisjointSet, DisjointSet]:
    vertex_groups = DisjointSet(face_count * 4)
    edge_groups = DisjointSet(face_count * 4)
    for face_index in range(face_count):
        for edge_index in range(4):
            edge_groups.union(
                face_edge_global_id(face_index, edge_index),
                face_edge_global_id(face_index, edge_index + 2),
            )

    for edge in shared_edges:
        a0 = face_vertex_global_id(edge.face_a, edge.edge_a)
        a1 = face_vertex_global_id(edge.face_a, edge.edge_a + 1)
        if edge.reverse:
            b0 = face_vertex_global_id(edge.face_b, edge.edge_b + 1)
            b1 = face_vertex_global_id(edge.face_b, edge.edge_b)
        else:
            b0 = face_vertex_global_id(edge.face_b, edge.edge_b)
            b1 = face_vertex_global_id(edge.face_b, edge.edge_b + 1)
        vertex_groups.union(a0, b0)
        vertex_groups.union(a1, b1)
        edge_groups.union(
            face_edge_global_id(edge.face_a, edge.edge_a),
            face_edge_global_id(edge.face_b, edge.edge_b),
        )
    return vertex_groups, edge_groups


def snap_shared_vertices(
    quads: list[np.ndarray],
    faces: list[FaceEvidence],
    vertex_groups: DisjointSet,
) -> list[np.ndarray]:
    sums: dict[int, np.ndarray] = {}
    weights: dict[int, float] = {}
    for face_index, (quad, face) in enumerate(zip(quads, faces)):
        weight = max(0.05, float(face.detector_confidence))
        for vertex_index in range(4):
            root = vertex_groups.find(face_vertex_global_id(face_index, vertex_index))
            sums[root] = sums.get(root, np.zeros(2, dtype=np.float32)) + quad[vertex_index] * weight
            weights[root] = weights.get(root, 0.0) + weight

    snapped = [quad.copy() for quad in quads]
    for face_index, quad in enumerate(snapped):
        for vertex_index in range(4):
            root = vertex_groups.find(face_vertex_global_id(face_index, vertex_index))
            quad[vertex_index] = sums[root] / max(weights[root], 1e-6)
    return snapped


def estimate_edge_family_models(
    quads: list[np.ndarray],
    edge_groups: DisjointSet,
    object_box_xyxy: list[float],
) -> dict[int, dict[str, Any]]:
    families: dict[int, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for face_index, quad in enumerate(quads):
        for edge_index in range(4):
            root = edge_groups.find(face_edge_global_id(face_index, edge_index))
            families[root].append((quad[edge_index], quad[(edge_index + 1) % 4]))

    models: dict[int, dict[str, Any]] = {}
    object_center = np.asarray(
        [
            (float(object_box_xyxy[0]) + float(object_box_xyxy[2])) * 0.5,
            (float(object_box_xyxy[1]) + float(object_box_xyxy[3])) * 0.5,
        ],
        dtype=np.float32,
    )
    object_diag = max(bbox_diag(object_box_xyxy), 1.0)
    for root, segments in families.items():
        lines = [line_from_points(p0, p1) for p0, p1 in segments]
        lines = [line for line in lines if line is not None]
        if not lines:
            continue
        vp = least_squares_intersection(lines)
        if vp is not None and np.linalg.norm(vp - object_center) <= object_diag * 80.0:
            models[root] = {"type": "vanishing", "point": vp}
        else:
            models[root] = {"type": "parallel", "direction": average_segment_direction(segments)}
    return models


def intersect_projective_edge_lines(
    quads: list[np.ndarray],
    edge_groups: DisjointSet,
    edge_models: dict[int, dict[str, Any]],
) -> list[np.ndarray] | None:
    refined: list[np.ndarray] = []
    for face_index, quad in enumerate(quads):
        edge_lines: list[np.ndarray] = []
        for edge_index in range(4):
            root = edge_groups.find(face_edge_global_id(face_index, edge_index))
            model = edge_models.get(root)
            if model is None:
                line = line_from_points(quad[edge_index], quad[(edge_index + 1) % 4])
            else:
                mid = (quad[edge_index] + quad[(edge_index + 1) % 4]) * 0.5
                line = projective_edge_line(mid, model)
            if line is None:
                return None
            edge_lines.append(line)

        new_quad = np.zeros((4, 2), dtype=np.float32)
        for vertex_index in range(4):
            point = intersect_lines(
                edge_lines[(vertex_index - 1) % 4],
                edge_lines[vertex_index],
            )
            if point is None or not np.isfinite(point).all():
                return None
            new_quad[vertex_index] = point
        ordered = order_quad_points(new_quad)
        if ordered is None:
            return None
        refined.append(ordered)
    return refined


def line_from_points(point_a: np.ndarray, point_b: np.ndarray) -> np.ndarray | None:
    p0 = np.asarray([float(point_a[0]), float(point_a[1]), 1.0], dtype=np.float64)
    p1 = np.asarray([float(point_b[0]), float(point_b[1]), 1.0], dtype=np.float64)
    line = np.cross(p0, p1)
    norm = math.hypot(float(line[0]), float(line[1]))
    if norm < 1e-6:
        return None
    return (line / norm).astype(np.float64)


def least_squares_intersection(lines: list[np.ndarray]) -> np.ndarray | None:
    if len(lines) < 2:
        return None
    a = np.asarray([[line[0], line[1]] for line in lines], dtype=np.float64)
    b = -np.asarray([line[2] for line in lines], dtype=np.float64)
    if np.linalg.matrix_rank(a) < 2:
        return None
    try:
        point, *_ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return point.astype(np.float32)


def average_segment_direction(segments: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    direction_sum = np.zeros(2, dtype=np.float64)
    reference: np.ndarray | None = None
    for point_a, point_b in segments:
        direction = np.asarray(point_b - point_a, dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            continue
        direction /= norm
        if reference is None:
            reference = direction.copy()
        elif float(np.dot(direction, reference)) < 0:
            direction *= -1.0
        direction_sum += direction
    norm = np.linalg.norm(direction_sum)
    if norm < 1e-6:
        return np.asarray([1.0, 0.0], dtype=np.float32)
    return (direction_sum / norm).astype(np.float32)


def projective_edge_line(midpoint: np.ndarray, model: dict[str, Any]) -> np.ndarray | None:
    if model["type"] == "vanishing":
        vp = np.asarray(model["point"], dtype=np.float32)
        if np.linalg.norm(vp - midpoint) > 1e-4:
            return line_from_points(vp, midpoint)
    direction = np.asarray(model.get("direction", [1.0, 0.0]), dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return None
    direction /= norm
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    c = -float(np.dot(normal, midpoint.astype(np.float64)))
    return np.asarray([normal[0], normal[1], c], dtype=np.float64)


def intersect_lines(line_a: np.ndarray, line_b: np.ndarray) -> np.ndarray | None:
    point = np.cross(line_a, line_b)
    if abs(float(point[2])) < 1e-6:
        return None
    point = point[:2] / point[2]
    return point.astype(np.float32)


def projective_refinement_passes(
    raw_quads: list[np.ndarray],
    refined_quads: list[np.ndarray],
    faces: list[FaceEvidence],
    object_box_xyxy: list[float],
    image_shape: tuple[int, int],
    *,
    max_mean_error_ratio: float,
    max_point_move_ratio: float,
    check_segment_iou: bool = True,
) -> bool:
    object_diag = max(bbox_diag(object_box_xyxy), 1.0)
    moves: list[float] = []
    iou_checks: list[tuple[np.ndarray, np.ndarray, list[np.ndarray]]] = []
    height, width = image_shape
    for raw_quad, refined_quad, face in zip(raw_quads, refined_quads, faces):
        if quad_passes(refined_quad, min_area=30.0, min_side=3.0, max_aspect=30.0) is None:
            return False
        if np.any(refined_quad[:, 0] < -width * 0.05) or np.any(refined_quad[:, 0] > width * 1.05):
            return False
        if np.any(refined_quad[:, 1] < -height * 0.05) or np.any(refined_quad[:, 1] > height * 1.05):
            return False
        moves.extend(np.linalg.norm(refined_quad - raw_quad, axis=1).tolist())
        if check_segment_iou:
            segment_contours = face_segments_contours(face)
            if segment_contours:
                iou_checks.append((raw_quad, refined_quad, segment_contours))
    if not moves:
        return False
    mean_error_ratio = float(np.mean(moves)) / object_diag
    max_move_ratio = float(np.max(moves)) / object_diag
    if mean_error_ratio > max_mean_error_ratio or max_move_ratio > max_point_move_ratio:
        return False
    for raw_quad, refined_quad, segment_contours in iou_checks:
        raw_contour = quad_contour_unclipped(raw_quad)
        refined_contour = quad_contour_unclipped(refined_quad)
        if raw_contour is None or refined_contour is None:
            return False
        raw_iou = polygon_iou_from_contours([raw_contour], segment_contours)
        refined_iou = polygon_iou_from_contours([refined_contour], segment_contours)
        if raw_iou > 0.05 and refined_iou < raw_iou * 0.55:
            return False
    return True


def decide_cube(
    faces: list[FaceEvidence],
    *,
    target_shape: str = "",
    target_fruit: str = "",
    fruit_min_conf: float = FRUIT_MIN_CONF_DEFAULT,
) -> ObjectDecision:
    fruit_counter: Counter[str] = Counter()
    blank_faces = 0
    unknown_faces = 0
    weak_fruit_faces = 0

    for face in faces:
        label = face.label.lower()
        if label in FRUIT_CLASSES:
            if float(face.confidence) >= fruit_min_conf:
                fruit_counter[label] += 1
            else:
                # Low-confidence fruit face is unreliable evidence. It must not flip
                # the cube to a confident fruit identity (a wrong set-2 pickup costs
                # double), and it must not let the cube be called plain_cube either.
                # Treat it as unknown so the cube stays cube_like_unresolved (inspect).
                unknown_faces += 1
                weak_fruit_faces += 1
        elif label in PLAIN_LABELS:
            blank_faces += 1
        else:
            unknown_faces += 1

    visible_faces = blank_faces + sum(fruit_counter.values()) + unknown_faces
    fruit_faces = dict(fruit_counter)

    if fruit_counter:
        if len(fruit_counter) > 1:
            return ObjectDecision(
                identity="conflicting_fruit_cube",
                action="inspect",
                reason="multiple fruit classes on one cube conflict with the rulebook same-fruit-face constraint",
                visible_faces=visible_faces,
                blank_faces=blank_faces,
                fruit_faces=fruit_faces,
                unknown_faces=unknown_faces,
            )
        fruit = next(iter(fruit_counter))
        if target_fruit:
            if fruit == target_fruit:
                return ObjectDecision(
                    identity=f"fruit_cube:{fruit}",
                    action="pickup",
                    reason="target fruit face is visible; rulebook set-2 object is worth 20 points",
                    visible_faces=visible_faces,
                    blank_faces=blank_faces,
                    fruit_faces=fruit_faces,
                    unknown_faces=unknown_faces,
                    expected_score=20,
                )
            return ObjectDecision(
                identity=f"fruit_cube:{fruit}",
                action="avoid",
                reason="non-target fruit is visible; wrong set-2 pickup costs double its 20 point value",
                visible_faces=visible_faces,
                blank_faces=blank_faces,
                fruit_faces=fruit_faces,
                unknown_faces=unknown_faces,
            )
        return ObjectDecision(
            identity=f"fruit_cube:{fruit}",
            action="inspect",
            reason="fruit cube identified; configure --target-fruit for pickup decisions",
            visible_faces=visible_faces,
            blank_faces=blank_faces,
            fruit_faces=fruit_faces,
            unknown_faces=unknown_faces,
        )

    if visible_faces == 0:
        return ObjectDecision(
            identity="cube_like_unresolved",
            action="inspect",
            reason="cube-like object has no reliable face evidence yet",
            visible_faces=0,
            blank_faces=0,
            fruit_faces={},
            unknown_faces=0,
        )

    if unknown_faces:
        if weak_fruit_faces:
            reason = (
                f"a fruit face below the acceptance threshold ({fruit_min_conf:.2f}) was ignored as "
                "unreliable, so the cube cannot be safely classified; inspect again for a clearer view"
            )
        else:
            reason = "at least one visible face is unknown, so the cube cannot be safely classified"
        return ObjectDecision(
            identity="cube_like_unresolved",
            action="inspect",
            reason=reason,
            visible_faces=visible_faces,
            blank_faces=blank_faces,
            fruit_faces={},
            unknown_faces=unknown_faces,
        )

    if blank_faces >= 2:
        # Rulebook update: a fruit cube's three printed faces are arranged in a
        # "ㄷ" band and the cube is always placed with a fruit face pointing up,
        # so any camera that resolves two or more faces of a fruit cube must see
        # at least one fruit face. Two confidently blank faces with no fruit/unknown
        # evidence therefore cannot be a fruit cube hiding its icons -> plain cube.
        action = "pickup" if target_shape == "cube" else ("skip" if target_shape else "inspect")
        expected_score = 10 if action == "pickup" else 0
        reason = (
            "two or more visible faces are confidently blank with no fruit/unknown face evidence; "
            "the rulebook always orients a fruit cube with a fruit face upward, so two blank faces "
            "cannot belong to a fruit cube"
        )
        if action == "pickup":
            reason += "; target set-1 cube is worth 10 points"
        elif target_shape and target_shape != "cube":
            reason += "; current target shape is not cube"
        else:
            reason += "; configure --target-shape cube to enable pickup"
        return ObjectDecision(
            identity="plain_cube",
            action=action,
            reason=reason,
            visible_faces=visible_faces,
            blank_faces=blank_faces,
            fruit_faces={},
            unknown_faces=0,
            expected_score=expected_score,
        )

    return ObjectDecision(
        identity="blank_only_cube_ambiguous",
        action="inspect",
        reason=(
            "only one blank face is visible; a single low-angle view can still belong to a fruit cube "
            "whose upward fruit face is out of frame, so inspect for a second face before deciding"
        ),
        visible_faces=visible_faces,
        blank_faces=blank_faces,
        fruit_faces={},
        unknown_faces=0,
    )


def draw_frame_output(frame: np.ndarray, output: FrameOutput) -> np.ndarray:
    annotated = frame.copy()
    for obj in output.objects:
        label, color = concise_object_label_and_color(obj)
        draw_polygons(
            annotated,
            obj.segments_xy,
            color,
            alpha=0.12,
            outline_color=color,
            thickness=2,
        )

        for face in obj.faces:
            if face.label.lower() in FRUIT_CLASSES:
                face_color = face_color_for_label(face.label)
                draw_polygons(
                    annotated,
                    face.segments_xy,
                    face_color,
                    alpha=0.48,
                    outline_color=face_color,
                    thickness=3,
                )
            else:
                draw_polygons(
                    annotated,
                    face.segments_xy,
                    (255, 255, 255),
                    alpha=0.08,
                    outline_color=(255, 255, 255),
                    thickness=1,
                )

        x1, y1, x2, y2 = [int(round(v)) for v in obj.box_xyxy]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        draw_label(annotated, label, (x1, max(0, y1 - 4)), color)

    return annotated


def draw_raw_model_output(frame: np.ndarray, output: FrameOutput) -> np.ndarray:
    annotated = frame.copy()
    for obj in output.objects:
        object_color = class_color_for_object(obj.class_name)
        draw_polygons(
            annotated,
            obj.segments_xy,
            object_color,
            alpha=0.10,
            outline_color=object_color,
            thickness=2,
        )
        x1, y1, x2, y2 = [int(round(v)) for v in obj.box_xyxy]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), object_color, 2)
        draw_label(
            annotated,
            f"A1 o{obj.object_index} {obj.class_name} {obj.confidence:.2f}",
            (x1, max(0, y1 - 4)),
            object_color,
            scale=0.52,
        )

        for face in obj.faces:
            face_color = face_color_for_label(face.label)
            draw_polygons(
                annotated,
                face.segments_xy,
                face_color,
                alpha=0.28 if face.label.lower() in FRUIT_CLASSES else 0.10,
                outline_color=face_color,
                thickness=2,
            )
            quad = np.asarray(face.quad_xy, dtype=np.int32)
            if quad.shape == (4, 2):
                cv2.polylines(annotated, [quad.reshape(-1, 1, 2)], True, face_color, 2)
                fx, fy = int(quad[:, 0].min()), int(quad[:, 1].min())
            else:
                fx, fy, _x2, _y2 = [int(round(v)) for v in face.box_xyxy]

            label = (
                f"f{face.face_index} A2 {face.detector_confidence:.2f} "
                f"C {face.label} {face.confidence:.2f}"
            )
            draw_label(
                annotated,
                label,
                (fx, max(0, fy - 4)),
                face_color,
                scale=0.47,
            )
    if not output.objects:
        draw_center_notice(annotated, "ABC raw: no objects")
    return annotated


def append_c_input_panel(
    frame: np.ndarray,
    c_inputs: list[dict[str, Any]],
    *,
    panel_width: int = 360,
    tile_size: int = 132,
) -> np.ndarray:
    panel = draw_c_input_panel(
        c_inputs,
        height=frame.shape[0],
        width=panel_width,
        tile_size=tile_size,
    )
    return np.hstack([frame, panel])


def append_raw_model_output_panel(
    frame: np.ndarray,
    output: FrameOutput,
    *,
    panel_width: int = 420,
) -> np.ndarray:
    panel = draw_raw_model_output_panel(
        output,
        height=frame.shape[0],
        width=panel_width,
    )
    return np.hstack([frame, panel])


def draw_raw_model_output_panel(
    output: FrameOutput,
    *,
    height: int,
    width: int = 420,
) -> np.ndarray:
    panel = np.full((height, width, 3), (245, 245, 245), dtype=np.uint8)
    header_h = 42
    panel[:header_h, :] = (28, 28, 28)
    cv2.putText(
        panel,
        "Raw model output",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    y = header_h + 28
    lines = raw_model_summary(output)
    if not lines:
        draw_wrapped_text(panel, "No A1 objects detected.", 14, y, width - 28)
        return panel

    for line in lines:
        y = draw_wrapped_text(panel, line, 14, y, width - 28, scale=0.48)
        y += 10
        if y > height - 30:
            draw_wrapped_text(panel, "...", 14, y, width - 28, scale=0.48)
            break
    return panel


def draw_c_input_panel(
    c_inputs: list[dict[str, Any]],
    *,
    height: int,
    width: int = 360,
    tile_size: int = 132,
) -> np.ndarray:
    panel = np.full((height, width, 3), (245, 245, 245), dtype=np.uint8)
    header_h = 42
    panel[:header_h, :] = (28, 28, 28)
    cv2.putText(
        panel,
        "C model input",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if not c_inputs:
        draw_wrapped_text(panel, "No face crop this frame.", 14, header_h + 32, width - 28)
        return panel

    margin = 12
    label_h = 22
    gap = 12
    columns = max(1, (width - margin) // max(tile_size + gap, 1))
    tile_w = min(tile_size, max(32, (width - margin * 2 - gap * (columns - 1)) // columns))
    tile_h = tile_w
    y0 = header_h + 14

    for idx, item in enumerate(c_inputs):
        row = idx // columns
        col = idx % columns
        x = margin + col * (tile_w + gap)
        y = y0 + row * (tile_h + label_h + gap)
        if y + tile_h + label_h > height - margin:
            draw_wrapped_text(panel, f"+{len(c_inputs) - idx} more", margin, max(y, height - 22), width - 28)
            break

        image = item.get("image")
        if not isinstance(image, np.ndarray) or image.size == 0:
            continue
        crop = cv2.resize(image, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        panel[y + label_h : y + label_h + tile_h, x : x + tile_w] = crop
        cv2.rectangle(
            panel,
            (x, y + label_h),
            (x + tile_w - 1, y + label_h + tile_h - 1),
            face_color_for_label(str(item.get("label", "unknown"))),
            2,
        )
        label = (
            f"o{item.get('object_index', '?')}/f{item.get('face_index', '?')} "
            f"{item.get('label', 'unknown')} {float(item.get('confidence', 0.0)):.2f}"
        )
        if item.get("occlusion_blackened"):
            label += " occ"
        if item.get("reject_reason"):
            label += f" {item.get('reject_reason')}"
        cv2.putText(
            panel,
            label[:28],
            (x, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )
    return panel


def draw_model_stage_grid(
    frame: np.ndarray,
    output: FrameOutput,
    *,
    panel_width: int = 520,
) -> np.ndarray:
    panels = [
        stage_panel(draw_a1_output(frame, output), "A1 object segmentation", panel_width),
        stage_panel(draw_a2_output(frame, output), "A2 visible face segmentation", panel_width),
        stage_panel(draw_b_output(frame, output), "B visible mask -> quad", panel_width),
        stage_panel(draw_c_output(frame, output), "C warped face classifier", panel_width),
        stage_panel(draw_raw_model_output(frame, output), "Raw per-frame model output", panel_width),
        stage_panel(draw_frame_output(frame, output), "Rulebook decision overlay", panel_width),
    ]
    panels.append(draw_raw_model_text_panel(output, panels[0].shape[1], panels[0].shape[0]))
    return stack_panels(panels, columns=2)


def draw_a1_output(frame: np.ndarray, output: FrameOutput) -> np.ndarray:
    annotated = frame.copy()
    for obj in output.objects:
        x1, y1, x2, y2 = [int(round(v)) for v in obj.box_xyxy]
        color = (80, 220, 80) if obj.class_name in CUBE_LIKE_CLASSES else (255, 180, 60)
        draw_polygons(
            annotated,
            obj.segments_xy,
            color,
            alpha=0.22,
            outline_color=color,
            thickness=2,
        )
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        draw_label(
            annotated,
            f"A1 {obj.object_index}:{obj.class_name} {obj.confidence:.2f}",
            (x1, max(0, y1 - 4)),
            color,
            scale=0.55,
        )
    if not output.objects:
        draw_center_notice(annotated, "A1: no objects")
    return annotated


def draw_a2_output(frame: np.ndarray, output: FrameOutput) -> np.ndarray:
    annotated = frame.copy()
    face_count = 0
    for obj in output.objects:
        for face in obj.faces:
            face_count += 1
            color = (40, 70, 255) if "fruit" in face.kind else (235, 235, 235)
            draw_polygons(
                annotated,
                face.segments_xy,
                color,
                alpha=0.34 if "fruit" in face.kind else 0.12,
                outline_color=color,
                thickness=2,
            )
            x1, y1, x2, y2 = [int(round(v)) for v in face.box_xyxy]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            draw_label(
                annotated,
                f"A2 obj{obj.object_index} {face.kind}",
                (x1, max(0, y1 - 4)),
                color,
                scale=0.5,
            )
    if face_count == 0:
        draw_center_notice(annotated, "A2: no visible faces")
    return annotated


def draw_b_output(frame: np.ndarray, output: FrameOutput) -> np.ndarray:
    annotated = frame.copy()
    face_count = 0
    for obj in output.objects:
        for face in obj.faces:
            face_count += 1
            quad = np.asarray(face.quad_xy, dtype=np.int32)
            if quad.shape != (4, 2):
                continue
            raw_quad = np.asarray(face.raw_quad_xy, dtype=np.int32) if face.raw_quad_xy else None
            if raw_quad is not None and raw_quad.shape == (4, 2) and not np.array_equal(raw_quad, quad):
                cv2.polylines(annotated, [raw_quad.reshape(-1, 1, 2)], True, (120, 120, 120), 1)
            color = (255, 255, 255)
            cv2.polylines(annotated, [quad.reshape(-1, 1, 2)], True, color, 3)
            for corner_index, (x, y) in enumerate(quad):
                cv2.circle(annotated, (int(x), int(y)), 5, (0, 255, 255), -1)
                cv2.putText(
                    annotated,
                    str(corner_index),
                    (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
            fx, fy = int(quad[:, 0].min()), int(quad[:, 1].min())
            quad_mode = "geom" if "projective" in face.source else "raw"
            draw_label(
                annotated,
                f"B obj{obj.object_index} face{face.face_index} {quad_mode}",
                (fx, max(0, fy - 4)),
                color,
                scale=0.5,
            )
    if face_count == 0:
        draw_center_notice(annotated, "B: no quads")
    return annotated


def draw_c_output(frame: np.ndarray, output: FrameOutput) -> np.ndarray:
    annotated = frame.copy()
    face_count = 0
    for obj in output.objects:
        for face in obj.faces:
            face_count += 1
            quad = np.asarray(face.quad_xy, dtype=np.int32)
            if quad.shape == (4, 2):
                color = face_color_for_label(face.label)
                cv2.polylines(annotated, [quad.reshape(-1, 1, 2)], True, color, 3)
                fx, fy = int(quad[:, 0].min()), int(quad[:, 1].min())
            else:
                color = face_color_for_label(face.label)
                x1, y1, _x2, _y2 = [int(round(v)) for v in face.box_xyxy]
                fx, fy = x1, y1
            draw_label(
                annotated,
                f"C {face.label} {face.confidence:.2f}",
                (fx, max(0, fy - 4)),
                color,
                scale=0.55,
            )
    if face_count == 0:
        draw_center_notice(annotated, "C: no face crops")
    return annotated


def stage_panel(image: np.ndarray, title: str, panel_width: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = float(panel_width) / max(width, 1)
    panel_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
    header_h = 38
    panel = np.zeros((panel_height + header_h, panel_width, 3), dtype=np.uint8)
    panel[:header_h, :] = (28, 28, 28)
    panel[header_h:, :] = resized
    cv2.putText(
        panel,
        title,
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def draw_decision_text_panel(output: FrameOutput, width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), (245, 245, 245), dtype=np.uint8)
    panel[:38, :] = (28, 28, 28)
    cv2.putText(
        panel,
        "Decision details",
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    y = 62
    if not output.objects:
        draw_wrapped_text(panel, "No objects detected.", 18, y, width - 36)
        return panel
    for obj in output.objects:
        decision = obj.decision
        if decision is None:
            lines = [f"obj{obj.object_index}: {obj.class_name} conf={obj.confidence:.2f}"]
        else:
            fruit = ", ".join(f"{k}:{v}" for k, v in decision.fruit_faces.items()) or "-"
            lines = [
                f"obj{obj.object_index}: {decision.action} / {decision.identity}",
                f"A1={obj.class_name} conf={obj.confidence:.2f} score={decision.expected_score}",
                (
                    f"faces visible={decision.visible_faces} blank={decision.blank_faces} "
                    f"fruit={fruit} unknown={decision.unknown_faces}"
                ),
                f"reason: {decision.reason}",
            ]
        for line in lines:
            y = draw_wrapped_text(panel, line, 18, y, width - 36)
            if y > height - 30:
                draw_wrapped_text(panel, "...", 18, y, width - 36)
                return panel
        y += 12
    return panel


def draw_raw_model_text_panel(output: FrameOutput, width: int, height: int) -> np.ndarray:
    panel = np.full((height, width, 3), (245, 245, 245), dtype=np.uint8)
    panel[:38, :] = (28, 28, 28)
    cv2.putText(
        panel,
        "Raw model outputs",
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    y = 62
    lines = raw_model_summary(output)
    if not lines:
        lines = ["No A1 objects detected."]
    for line in lines:
        y = draw_wrapped_text(panel, line, 18, y, width - 36)
        y += 4
        if y > height - 30:
            draw_wrapped_text(panel, "...", 18, y, width - 36)
            return panel
    return panel


def stack_panels(panels: list[np.ndarray], columns: int) -> np.ndarray:
    if not panels:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    width = max(panel.shape[1] for panel in panels)
    height = max(panel.shape[0] for panel in panels)
    normalized = [pad_panel(panel, width, height) for panel in panels]
    rows = []
    for start in range(0, len(normalized), columns):
        row = normalized[start : start + columns]
        while len(row) < columns:
            row.append(np.full((height, width, 3), 245, dtype=np.uint8))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def pad_panel(panel: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.full((height, width, 3), 245, dtype=np.uint8)
    out[: panel.shape[0], : panel.shape[1]] = panel
    return out


def draw_wrapped_text(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    max_width: int,
    *,
    scale: float = 0.52,
    color: tuple[int, int, int] = (25, 25, 25),
) -> int:
    words = text.split()
    line = ""
    line_height = 21
    for word in words:
        trial = word if not line else f"{line} {word}"
        (tw, _th), _baseline = cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        if tw > max_width and line:
            cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
            y += line_height
            line = word
        else:
            line = trial
    if line:
        cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += line_height
    return y


def draw_center_notice(image: np.ndarray, text: str) -> None:
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    x = max(8, (image.shape[1] - tw) // 2)
    y = max(th + baseline + 8, image.shape[0] // 2)
    cv2.rectangle(image, (x - 8, y - th - baseline - 8), (x + tw + 8, y + 6), (30, 30, 30), -1)
    cv2.putText(image, text, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)


def build_object_label(obj: ObjectEvidence) -> str:
    decision = obj.decision
    if decision is None:
        return f"{obj.object_index}:{obj.class_name} {obj.confidence:.2f}"
    return (
        f"{obj.object_index}:{obj.class_name} {obj.confidence:.2f} "
        f"{decision.action}/{decision.identity} faces={decision.visible_faces}"
    )


def concise_object_label_and_color(obj: ObjectEvidence) -> tuple[str, tuple[int, int, int]]:
    if obj.decision is not None:
        identity = obj.decision.identity
        if identity.startswith("fruit_cube:"):
            return identity, face_color_for_label(identity.split(":", 1)[1])
        if identity == "plain_cube":
            return identity, (235, 235, 235)
        if identity in {"conflicting_fruit_cube", "cube_like_unresolved", "blank_only_cube_ambiguous"}:
            return identity, ACTION_COLORS.get(obj.decision.action, (200, 160, 255))
        return identity, class_color_for_object(identity)
    return f"{obj.class_name} ({obj.confidence:.2f})", class_color_for_object(obj.class_name)


def class_color_for_object(class_name: str) -> tuple[int, int, int]:
    palette = {
        "cube_like_object": (60, 60, 255),
        "cube": (60, 60, 255),
        "octahedron": (255, 170, 40),
        "dodecahedron": (80, 220, 120),
        "icosahedron": (220, 100, 220),
    }
    return palette.get(class_name.lower(), (80, 220, 255))


def draw_polygons(
    image: np.ndarray,
    polygons: list[list[list[float]]],
    fill_color: tuple[int, int, int],
    *,
    alpha: float,
    outline_color: tuple[int, int, int] | None = None,
    thickness: int = 2,
) -> None:
    contours = polygon_contours(polygons, image.shape[:2])
    if not contours:
        return
    if alpha > 0:
        overlay = image.copy()
        cv2.fillPoly(overlay, contours, fill_color)
        cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, dst=image)
    if outline_color is not None and thickness > 0:
        cv2.polylines(image, contours, True, outline_color, thickness, cv2.LINE_AA)


def polygon_contours(
    polygons: list[list[list[float]]],
    shape: tuple[int, int],
) -> list[np.ndarray]:
    height, width = shape
    contours: list[np.ndarray] = []
    for polygon in polygons:
        pts = np.asarray(polygon, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
            continue
        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
        contours.append(np.rint(pts).astype(np.int32).reshape(-1, 1, 2))
    return contours


def face_color_for_label(label: str) -> tuple[int, int, int]:
    palette = {
        "apple": (50, 60, 230),
        "orange": (0, 150, 255),
        "banana": (40, 230, 255),
        "pineapple": (40, 180, 120),
        "plain": (230, 230, 230),
        "unknown": (200, 160, 255),
    }
    return palette.get(label.lower(), (255, 255, 255))


def draw_label(
    image: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    color: tuple[int, int, int],
    *,
    scale: float = 0.55,
) -> None:
    x, y = anchor
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(th + baseline + 2, y)
    x = max(0, min(x, max(0, image.shape[1] - tw - 8)))
    cv2.rectangle(image, (x, y - th - baseline - 4), (x + tw + 8, y), color, -1)
    text_color = (0, 0, 0) if sum(color) > 360 else (255, 255, 255)
    cv2.putText(image, text, (x + 4, y - baseline - 2), font, scale, text_color, thickness, cv2.LINE_AA)


def draw_summary(image: np.ndarray, output: FrameOutput) -> None:
    counts = defaultdict(int)
    for obj in output.objects:
        action = obj.decision.action if obj.decision else "unknown"
        counts[action] += 1
    if not counts:
        text = "ABC: no objects"
    else:
        text = "ABC: " + ", ".join(f"{key}={counts[key]}" for key in sorted(counts))
    draw_label(image, text, (12, 28), (40, 40, 40), scale=0.6)


def frame_summary(output: FrameOutput) -> list[str]:
    lines = []
    for obj in output.objects:
        if obj.decision is None:
            lines.append(f"{obj.object_index}: {obj.class_name} {obj.confidence:.3f}")
            continue
        decision = obj.decision
        fruit_text = ",".join(f"{k}:{v}" for k, v in decision.fruit_faces.items()) or "-"
        lines.append(
            f"{obj.object_index}: {obj.class_name} conf={obj.confidence:.3f} "
            f"action={decision.action} identity={decision.identity} "
            f"faces={decision.visible_faces} blank={decision.blank_faces} "
            f"fruit={fruit_text} unknown={decision.unknown_faces} reason={decision.reason}"
        )
    return lines


def raw_model_summary(output: FrameOutput) -> list[str]:
    lines: list[str] = []
    for obj in output.objects:
        if obj.faces:
            face_text = "; ".join(
                (
                    f"f{face.face_index} A2={face.detector_confidence:.3f} "
                    f"C={face.label}:{face.confidence:.3f} src={face.source}"
                )
                for face in obj.faces
            )
        else:
            face_text = "faces=-"
        lines.append(
            f"obj{obj.object_index} A1={obj.class_name}:{obj.confidence:.3f} "
            f"box={[round(v, 1) for v in obj.box_xyxy]} {face_text}"
        )
    return lines
