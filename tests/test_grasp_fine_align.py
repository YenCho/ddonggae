"""Unit tests for the camera-based grasp fine alignment (pure math parts)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]

from geometry import (  # noqa: E402
    CameraMount,
    Intrinsics,
    align_command,
    find_object_blob,
    measure_object,
    pixel_to_ground,
    project_ground,
)

W, H = 640, 480
INTR = Intrinsics(fx=1.88 / 1.95 * W, fy=1.88 / 1.95 * W, cx=W / 2, cy=H / 2)
MOUNT = CameraMount()  # near camera defaults (2026-07-18 실측: fwd 0.1874, h 0.3143, tilt 53.2)


def test_project_backproject_roundtrip():
    for gx, gy in ((0.0, 0.45), (0.1, 0.6), (-0.15, 0.35), (0.05, 1.2)):
        u, v = project_ground(gx, gy, INTR, MOUNT)
        bx, by = pixel_to_ground(u, v, INTR, MOUNT)
        assert math.hypot(bx - gx, by - gy) < 1e-9


def test_principal_ray_hits_expected_forward_distance():
    # The optical axis pitched tilt_deg down from height_m hits the floor at
    # forward = mount_fwd + h/tan(tilt).
    gx, gy = pixel_to_ground(INTR.cx, INTR.cy, INTR, MOUNT)
    assert abs(gx) < 1e-9
    assert gy == pytest.approx(
        MOUNT.forward_m + MOUNT.height_m / math.tan(math.radians(MOUNT.tilt_deg)), abs=1e-9)


def test_horizon_pixel_rejected():
    # At 54 deg tilt the whole frame sees floor - even the top row.
    gx, gy = pixel_to_ground(INTR.cx, 0.0, INTR, MOUNT)
    assert gy > 0.5
    # A pixel far above the optical axis maps above the horizon -> guarded.
    with pytest.raises(ValueError):
        pixel_to_ground(INTR.cx, -2000.0, INTR, MOUNT)


def _synthetic_scene(obj_ground, size_px=42):
    """Wood floor + one bright object blob whose bottom edge sits at the
    object's ground-contact pixel. At 54 deg tilt the whole frame sees floor
    (no wall band) - matches the real near-camera framing during a grasp."""
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[:, :] = (184, 134, 77)  # plywood tone
    u, v = project_ground(obj_ground[0], obj_ground[1], INTR, MOUNT)
    u, v = int(u), int(v)
    rgb[max(0, v - size_px):v + 1, max(0, u - size_px // 2):u + size_px // 2] = (240, 240, 235)
    return rgb


def test_blob_detection_and_measurement():
    truth = (0.06, 0.52)
    rgb = _synthetic_scene(truth)
    blob = find_object_blob(rgb)
    assert blob is not None and blob["pixels"] > 200
    meas = measure_object(rgb, INTR, MOUNT, expected_ground=truth, half_extent_m=0.0)
    assert meas is not None
    # bottom edge of the synthetic blob IS the ground contact -> few-mm accuracy
    assert math.hypot(meas["x_right"] - truth[0], meas["y_forward"] - truth[1]) < 0.02


def test_expected_prior_picks_correct_blob():
    truth = (-0.10, 0.50)
    decoy = (0.15, 0.75)
    rgb = _synthetic_scene(truth)
    du, dv = project_ground(decoy[0], decoy[1], INTR, MOUNT)
    du, dv = int(du), int(dv)
    rgb[dv - 60:dv + 1, du - 30:du + 30] = (20, 20, 20)  # bigger decoy
    meas = measure_object(rgb, INTR, MOUNT, expected_ground=truth, half_extent_m=0.0)
    assert meas is not None
    assert abs(meas["x_right"] - truth[0]) < 0.03
    assert abs(meas["y_forward"] - truth[1]) < 0.03


def test_align_command_signs_and_creep():
    # object dead ahead at 0.5 m: no turn, creep to grip point
    dyaw, creep = align_command(0.0, 0.5, grip_forward_m=0.145)
    assert dyaw == pytest.approx(0.0)
    assert creep == pytest.approx(0.355)
    # object to the LEFT (x_right < 0) -> CCW positive turn
    dyaw, _ = align_command(-0.1, 0.5)
    assert dyaw > 0.0
    # object to the RIGHT -> CW negative turn
    dyaw, _ = align_command(0.1, 0.5)
    assert dyaw < 0.0
