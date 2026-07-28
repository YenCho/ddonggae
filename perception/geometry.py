"""Camera-based fine alignment for grasping (replaces the oracle CONFIRM).

The real robot cannot cheat-grip: after the street approach it must measure
the target with the near camera (54 deg tilt D435) and servo onto it. This
module holds the PURE math/vision parts so they run identically on the real
robot and in Isaac Sim:

  1) ``pixel_to_ground``     - back-project a pixel through the floor plane
                               (the object's bottom edge touches the floor,
                               so its ground point is exact, no depth needed)
  2) ``find_object_blob``    - segment the non-floor blob nearest to the
                               expected position (grid prior)
  3) ``measure_object``      - full pipeline: image -> robot-frame object
                               center (forward, left) + pixel diagnostics
  4) ``align_command``       - robot-frame measurement -> (dyaw, creep)
                               move_relative arguments for the grasp

Frames: robot base frame with x RIGHT, y FORWARD, z UP (matches
``calibrate_stitch_homography``). ``align_command`` converts to the
move_relative convention (dx forward, dy left, dyaw CCW).

Validation in sim: ``sim/isaacsim/scripts/validate_camera_grasp_align.py``
subscribes to /camera_54/rgb + /camera_54/camera_info and compares the
measurement against /sim/ground_truth_pose + /sim/objects_state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraMount:
    """Near-camera mount relative to base center, tilt pitching down.

    2026-07-18 depth 바닥평면 실측 (89_mastup_topcam_calib.py, 평면 RMS 0.9mm):
    height 0.3143 / tilt 53.2 — 7/16 그리퍼·마스트 재조립 때 브래킷이 구 공칭
    (0.2680/54.0) 대비 ~4.6cm 위로 이동. 이 기본값은 **마스트 다운** 기준
    (파지/접근이 일어나는 상태).

    ⚠ 정정 (2026-07-20): 하단캠은 섀시 고정이 아니라 **마스트 동승**이다.
    7/18의 "마스트 무관 확인"은 마스트를 올리지 않은 채 label 만 mastup 으로
    찍힌 무효 측정이었다. 실측: 마스트 업 = height 0.4600 / tilt 53.24
    (줄자 30.8→45.9cm = 스트로크 14.89cm 동승, depth 적합 RMS 1.7mm).
    마스트 업 계산에는 `fieldlib.NEAR_MOUNT_UP` 을 쓸 것.
    forward_m 은 평면 피팅으로 측정 불가라 구 값 유지 — 30cm 자 검증 필요.
    """

    forward_m: float = 0.1874
    height_m: float = 0.3143
    tilt_deg: float = 53.2
    # [2026-07-23 신규 — 조작자 승인] 좌우(lateral) 장착 오프셋 [m], **+가 로봇
    # 우측**. 종전 모델은 카메라가 로봇 중심선 위에 있다고 가정해 cam_pos[0]
    # 을 0.0 으로 하드코딩했고, 그래서 좌우 편향을 표현할 방법 자체가 없었다.
    #
    # 근거 (01:10~01:20 실기 로그 분석):
    #   · 접근 사진 13장 전부에서 그리퍼 중심선이 근접캠 광축보다 +194~+257px
    #     (평균 +222px, fx≈1397 → 시야각 9.0°) **우측**에 있다. 즉 광축이
    #     그리퍼 중심선의 왼쪽에 있다.
    #   · 같은 구간 파지 실측 x_right = +0.003~+0.071 m (평균 +0.054, σ0.024)
    #     로 6회 전부 부호가 같은 고정 편향 — 위 픽셀 관측과 정합한다.
    #
    # ⚠ 기본값 0.0 을 유지한다. 픽셀 관측만으로는 **lateral 병진**(거리 무관
    #   상수 오차)과 **카메라 yaw 틀어짐**(거리 비례 오차)을 분리할 수 없고,
    #   위 6샘플은 거리 범위가 0.325~0.375m 뿐이라 둘을 가르지 못한다.
    #   실측으로 확정하기 전에 값을 넣으면 파지 정렬을 반대로 틀 수 있다.
    #   측정 절차와 판별법은 perception/docs/stitching-and-calibration.md 참조.
    # ⚠ roll(-1.0~-2.5°) 미모델링은 여전히 남아 있다 (fieldlib.py 주석).
    lateral_m: float = 0.0


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_camera_info(cls, k_row_major: list[float]) -> "Intrinsics":
        """Build from a sensor_msgs/CameraInfo ``k`` (3x3 row-major)."""
        return cls(fx=float(k_row_major[0]), fy=float(k_row_major[4]),
                   cx=float(k_row_major[2]), cy=float(k_row_major[5]))


def camera_rotation(tilt_deg: float) -> np.ndarray:
    """Rows = camera axes in the robot frame (see calibrate_stitch_homography).

    Camera looks along -Z with +Y up, pitched down by ``tilt_deg`` from the
    robot's +y (forward) axis.
    """
    t = math.radians(tilt_deg)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.sin(t), math.cos(t)],
        [0.0, -math.cos(t), math.sin(t)],
    ])


def pixel_to_ground(u: float, v: float, intr: Intrinsics, mount: CameraMount) -> tuple[float, float]:
    """Back-project pixel (u, v) onto the floor plane (z=0 in robot frame).

    Returns (x_right, y_forward) of the ground point in the robot frame.
    Raises ValueError for pixels at/above the horizon (no floor hit).
    """
    rot = camera_rotation(mount.tilt_deg)
    # Direction in CAMERA coords: +x right, +y up, view along -z.
    d_cam = np.array([(u - intr.cx) / intr.fx, -(v - intr.cy) / intr.fy, -1.0])
    d_rob = rot.T @ d_cam
    if d_rob[2] >= -1e-6:
        raise ValueError(f"pixel ({u:.1f},{v:.1f}) does not intersect the floor")
    # [2026-07-23] 0.0 하드코딩 → mount.lateral_m (좌우 장착 오프셋).
    cam_pos = np.array([mount.lateral_m, mount.forward_m, mount.height_m])
    s = -cam_pos[2] / d_rob[2]
    p = cam_pos + s * d_rob
    return float(p[0]), float(p[1])


def project_ground(x_right: float, y_forward: float, intr: Intrinsics, mount: CameraMount) -> tuple[float, float]:
    """Forward-project a floor point to pixel coordinates (test/expectation)."""
    rot = camera_rotation(mount.tilt_deg)
    # [2026-07-23] 0.0 하드코딩 → mount.lateral_m (좌우 장착 오프셋).
    cam_pos = np.array([mount.lateral_m, mount.forward_m, mount.height_m])
    p_cam = rot @ (np.array([x_right, y_forward, 0.0]) - cam_pos)
    if p_cam[2] >= -1e-6:
        raise ValueError("point is behind the camera")
    u = intr.cx + intr.fx * (p_cam[0] / -p_cam[2])
    v = intr.cy - intr.fy * (p_cam[1] / -p_cam[2])
    return float(u), float(v)


def horizon_row(intr: Intrinsics, mount: CameraMount, margin_px: int = 12) -> int:
    """First image row that can see the floor (rows above look at walls/sky).

    A ray ``a`` radians above the optical axis is horizontal when
    a == tilt, i.e. at v = cy - fy*tan(tilt). At 54 deg this is far above
    the image (whole frame sees floor); at 19 deg it cuts the top rows.
    """
    return max(0, int(intr.cy - intr.fy * math.tan(math.radians(mount.tilt_deg))) + margin_px)


def find_object_blob(
    rgb: np.ndarray,
    expected_uv: tuple[float, float] | None = None,
    floor_delta: float = 28.0,
    min_pixels: int = 60,
    mask_above_row: int = 0,
) -> dict | None:
    """Segment non-floor blobs; return the one nearest ``expected_uv``.

    The floor is assumed to dominate the image bottom rows; its reference
    color is the median there. Non-floor = pixels whose channel-max deviation
    from that reference exceeds ``floor_delta``. Connected components are
    labeled with a simple BFS (numpy-only so it runs on the Jetson without
    OpenCV/scipy at this stage; the real pipeline may swap in cv2 later).

    Returns {"centroid_uv", "bottom_uv", "pixels"} or None.
    """
    h, w, _ = rgb.shape
    img = rgb.astype(np.int16)
    floor_ref = np.median(img[int(h * 0.85):, :, :].reshape(-1, 3), axis=0)
    mask = np.abs(img - floor_ref).max(axis=2) > floor_delta
    if mask_above_row > 0:
        mask[:mask_above_row, :] = False  # above-horizon region: walls, not floor

    visited = np.zeros_like(mask, dtype=bool)
    best = None
    ys, xs = np.nonzero(mask)
    order = np.argsort(ys)[::-1]  # grow from the bottom (closest) first
    for idx in order:
        sy, sx = int(ys[idx]), int(xs[idx])
        if visited[sy, sx]:
            continue
        stack = [(sy, sx)]
        visited[sy, sx] = True
        blob_y, blob_x = [], []
        while stack:
            cy, cx = stack.pop()
            blob_y.append(cy)
            blob_x.append(cx)
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(blob_y) < min_pixels:
            continue
        by = np.asarray(blob_y)
        bx = np.asarray(blob_x)
        centroid = (float(bx.mean()), float(by.mean()))
        bottom_row = by.max()
        bottom_cols = bx[by >= bottom_row - 1]
        cand = {
            "centroid_uv": centroid,
            "bottom_uv": (float(bottom_cols.mean()), float(bottom_row)),
            "pixels": int(len(blob_y)),
        }
        if expected_uv is None:
            if best is None or cand["pixels"] > best["pixels"]:
                best = cand
        else:
            dist = math.hypot(centroid[0] - expected_uv[0], centroid[1] - expected_uv[1])
            if best is None or dist < best.get("_dist", 1e9):
                cand["_dist"] = dist
                best = cand
    if best is not None:
        best.pop("_dist", None)
    return best


def measure_object(
    rgb: np.ndarray,
    intr: Intrinsics,
    mount: CameraMount,
    expected_ground: tuple[float, float] | None = None,
    half_extent_m: float = 0.04,
) -> dict | None:
    """Image -> robot-frame object CENTER estimate.

    The blob's bottom edge is the front face's floor contact line; the object
    center sits ``half_extent_m`` beyond it along the viewing ray direction.
    ``expected_ground`` (x_right, y_forward) is the grid prior used to pick
    the right blob when several objects are visible.
    """
    expected_uv = None
    if expected_ground is not None:
        try:
            expected_uv = project_ground(expected_ground[0], expected_ground[1], intr, mount)
        except ValueError:
            expected_uv = None
    blob = find_object_blob(rgb, expected_uv=expected_uv,
                            mask_above_row=horizon_row(intr, mount))
    if blob is None:
        return None
    gx, gy = pixel_to_ground(blob["bottom_uv"][0], blob["bottom_uv"][1], intr, mount)
    # Push from the front contact line to the body center.
    norm = math.hypot(gx, gy)
    if norm > 1e-6:
        gx += half_extent_m * gx / norm
        gy += half_extent_m * gy / norm
    return {"x_right": gx, "y_forward": gy, **blob}


def align_command(x_right: float, y_forward: float, grip_forward_m: float = 0.145) -> tuple[float, float]:
    """Robot-frame object center -> (dyaw, creep_forward) for move_relative.

    dyaw is CCW-positive (an object to the LEFT gives dyaw > 0); creep is the
    forward distance after the turn that puts the object at the grip point.
    """
    dyaw = math.atan2(-x_right, y_forward)
    creep = math.hypot(x_right, y_forward) - grip_forward_m
    return dyaw, creep
