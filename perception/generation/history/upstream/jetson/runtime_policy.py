from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


# A1 cube-size guard for the unified pipeline.
#
# Cube-like objects below this size are marked as cube_too_far and are not sent
# to the unified face model. The effective area threshold is the larger of the
# absolute pixel threshold and the frame-relative threshold.
CUBE_TOO_FAR_MIN_BBOX_AREA_PIXELS = 6000
CUBE_TOO_FAR_MIN_BBOX_AREA_RATIO = 0.003
CUBE_TOO_FAR_MIN_SHORT_SIDE_PIXELS = 80
CUBE_TOO_FAR_IDENTITY = "cube_too_far"
CUBE_TOO_FAR_COLOR_BGR = (200, 120, 255)


@dataclass(frozen=True)
class CubeSizeMetrics:
    bbox_width: float
    bbox_height: float
    bbox_area: float
    frame_area: float
    area_ratio: float
    short_side: float
    min_required_area: float


def cube_size_metrics(
    box_xyxy: Sequence[int | float],
    frame_shape: Sequence[int],
) -> CubeSizeMetrics:
    if len(box_xyxy) < 4:
        raise ValueError("box_xyxy must contain at least four values")
    frame_height = max(1, int(frame_shape[0]))
    frame_width = max(1, int(frame_shape[1]))
    x1, y1, x2, y2 = [float(value) for value in box_xyxy[:4]]
    bbox_width = max(1.0, x2 - x1 + 1.0)
    bbox_height = max(1.0, y2 - y1 + 1.0)
    bbox_area = bbox_width * bbox_height
    frame_area = float(frame_width * frame_height)
    min_required_area = max(
        float(CUBE_TOO_FAR_MIN_BBOX_AREA_PIXELS),
        frame_area * float(CUBE_TOO_FAR_MIN_BBOX_AREA_RATIO),
    )
    return CubeSizeMetrics(
        bbox_width=bbox_width,
        bbox_height=bbox_height,
        bbox_area=bbox_area,
        frame_area=frame_area,
        area_ratio=bbox_area / max(frame_area, 1.0),
        short_side=min(bbox_width, bbox_height),
        min_required_area=min_required_area,
    )


def cube_is_too_far(
    box_xyxy: Sequence[int | float],
    frame_shape: Sequence[int],
) -> tuple[bool, CubeSizeMetrics]:
    metrics = cube_size_metrics(box_xyxy, frame_shape)
    too_far = (
        metrics.bbox_area < metrics.min_required_area
        or metrics.short_side < float(CUBE_TOO_FAR_MIN_SHORT_SIDE_PIXELS)
    )
    return too_far, metrics


def cube_too_far_reason(metrics: CubeSizeMetrics) -> str:
    return (
        f"bbox={metrics.bbox_width:.0f}x{metrics.bbox_height:.0f}, "
        f"area={metrics.bbox_area:.0f}, area_ratio={metrics.area_ratio:.4f}, "
        f"min_area={metrics.min_required_area:.0f}, "
        f"min_short_side={CUBE_TOO_FAR_MIN_SHORT_SIDE_PIXELS}"
    )
