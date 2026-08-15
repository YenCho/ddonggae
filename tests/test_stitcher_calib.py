"""fieldlib.Stitcher 기하 계약 테스트.

핵심 계약 두 가지:
  1. 캘리브 파일이 없으면 기존 평행이동 모델과 **비트 단위로 동일**해야 한다
     (저장 전에는 인식 파이프라인이 절대 바뀌지 않는다).
  2. `to_source()` 는 어떤 변환을 쓰든 **정확한 역변환**이어야 한다 —
     3D 역투영이 원본 캠 프레임에서만 유효하다는 계약이 여기에 걸려 있다.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

import fieldlib as fl  # noqa: E402

W, H = 640, 480


def legacy_stitch(mast, top, near):
    """2026-07-19 이전 구현 (camera_stitch_node 조성 재현)."""
    p = fl.STITCH_PARAMS[mast]
    x = p["x_offset"]
    left, right = max(0, x), min(W, x + W)
    bottom_left = left - x
    top_keep_h = H - min(p["crop_bottom_from_top"], H - 1) - p["blend_overlap"]
    crop_top = p["crop_top_from_bottom"]
    top_a = top[:, left:right]
    near_a = near[:, bottom_left:bottom_left + (right - left)]

    def to_source(u, v):
        if v < top_keep_h:
            return "top", u + left, v
        return "near", u + bottom_left, (v - top_keep_h) + crop_top

    return np.vstack((top_a[:top_keep_h], near_a[crop_top:])), to_source


@pytest.fixture()
def frames():
    rng = np.random.default_rng(20260720)
    return (rng.integers(0, 255, (H, W, 3), dtype=np.uint8),
            rng.integers(0, 255, (H, W, 3), dtype=np.uint8))


@pytest.mark.parametrize("mast", ["up", "down"])
def test_no_calib_matches_legacy_bit_exact(mast, frames, monkeypatch):
    """캘리브 파일이 없으면 레거시 출력과 완전히 같아야 한다."""
    monkeypatch.setattr(fl, "load_stitch_calib", lambda m: {})
    top, near = frames
    st = fl.Stitcher(mast)
    old, _ = legacy_stitch(mast, top, near)
    assert st.calib_source.startswith("STITCH_PARAMS")
    assert np.array_equal(st.stitch(top, near), old)


@pytest.mark.parametrize("mast", ["up", "down"])
def test_no_calib_to_source_matches_legacy(mast, frames, monkeypatch):
    monkeypatch.setattr(fl, "load_stitch_calib", lambda m: {})
    top, near = frames
    st = fl.Stitcher(mast)
    _, legacy_to_source = legacy_stitch(mast, top, near)
    for v in (0, 10, st.seam - 1, st.seam, st.seam + 1, st.out_h - 1):
        for u in (0, 1, 300, st.right - st.left - 1):
            cam_a, ua, va = legacy_to_source(u, v)
            cam_b, ub, vb = st.to_source(u, v)
            assert cam_a == cam_b
            assert ua == pytest.approx(ub, abs=1e-9)
            assert va == pytest.approx(vb, abs=1e-9)


@pytest.mark.parametrize("mast", ["up", "down"])
def test_warp_path_equals_fast_path(mast, frames, monkeypatch):
    """평행이동일 때 warp 경로와 슬라이싱 경로가 같은 픽셀을 내야 한다."""
    monkeypatch.setattr(fl, "load_stitch_calib", lambda m: {})
    top, near = frames
    fast = fl.Stitcher(mast)
    slow = fl.Stitcher(mast)
    slow._pure_shift = False
    assert np.array_equal(fast.stitch(top, near), slow.stitch(top, near))


def _similarity(scale, rot_deg, dx, dy):
    th = math.radians(rot_deg)
    return np.array([[scale * math.cos(th), -scale * math.sin(th), dx],
                     [scale * math.sin(th), scale * math.cos(th), dy],
                     [0.0, 0.0, 1.0]])


@pytest.mark.parametrize("A", [
    _similarity(1.0, 0.0, 8.0, 410.0),        # 순수 평행이동
    _similarity(1.02, 0.6, 5.0, 405.0),       # 배율+회전 (실측 0.53° 상당)
    _similarity(0.97, -1.4, -6.0, 418.0),
])
def test_to_source_is_exact_inverse(A, frames, monkeypatch):
    """to_source ∘ from_source 왕복이 항등이어야 한다 (3D 역투영 계약)."""
    monkeypatch.setattr(fl, "load_stitch_calib", lambda m: {})
    st = fl.Stitcher("up", calib={"A": A.tolist(), "seam": 464,
                                  "left": 8, "right": 640, "note": "t"})
    for v in (st.seam, st.seam + 37, st.out_h - 1):
        for u in (0, 200, 500):
            cam, su, sv = st.to_source(u, v)
            assert cam == "near"
            bu, bv = st.from_source(cam, su, sv)
            assert bu == pytest.approx(u, abs=1e-6)
            assert bv == pytest.approx(v, abs=1e-6)


def test_calibrator_fit_recovers_known_transform():
    """알려진 변환으로 만든 대응점에서 그 변환을 되찾아야 한다."""
    from stitch_calibrator import decompose, fit_A, residuals

    truth = _similarity(1.03, 0.75, 6.0, 407.0)
    rng = np.random.default_rng(7)
    near_pts = rng.uniform([20, 20], [W - 20, H - 20], size=(6, 2))
    pts = []
    for nu, nv in near_pts:
        q = truth @ np.array([nu, nv, 1.0])
        pts.append({"near": [float(nu), float(nv)],
                    "top": [float(q[0] / q[2]), float(q[1] / q[2])]})

    A, _ = fit_A(pts, "similarity")
    _, rms = residuals(A, pts)
    assert rms < 1e-3
    d = decompose(A)
    assert d["scale_x"] == pytest.approx(1.03, abs=1e-4)
    assert d["rot_deg"] == pytest.approx(0.75, abs=1e-3)

    # 한 점만 주면 평행이동밖에 못 정한다 — 지금 증상의 근본 원인
    one, _ = fit_A(pts[:1], "translate")
    _, rms1 = residuals(one, pts)
    assert rms1 > 3.0, "1점 적합이 전체 대응을 맞추면 안 된다 (배율/회전 미구속)"
    assert fit_A(pts[:1], "similarity")[0] is None


# ---- 해상도 재스케일 (2026-07-21 RGB FHD 전환) ----

def _khat(cam, w, h):
    return fl.crop_model_k(cam, w, h)


def test_rescale_preserves_ground_correspondence(monkeypatch):
    """640 캘리브 A 가 맺어준 near↔top 대응은 어느 해상도로 가도 유지돼야 한다:
    A_new · S_near · p == S_top · A · p  (구성상 항등이지만 seam/left/right 및
    행렬 조립 경로 전체를 통과시키는 계약 테스트)."""
    monkeypatch.setattr(fl, "load_stitch_calib", lambda m: {})
    A = _similarity(1.0176, 0.54, -9.07, 419.05)          # 실제 up.json 상당
    calib = {"A": A.tolist(), "seam": 464, "left": 8, "right": 640, "note": "t"}
    st0 = fl.Stitcher("up", calib=dict(calib))
    for tw, th in ((1280, 720), (1920, 1080)):
        intr = {c: _khat(c, tw, th) for c in ("top", "near")}
        st1 = fl.Stitcher("up", w=tw, h=th, calib=dict(calib), intr=intr)
        s_top = fl.pixel_scale_matrix(fl.CALIB_REF_K["top"], intr["top"])
        s_near = fl.pixel_scale_matrix(fl.CALIB_REF_K["near"], intr["near"])
        for u, v in ((40.0, 10.0), (320.0, 35.0), (600.0, 60.0)):
            p = np.array([u, v, 1.0])
            lhs = st1.A @ (s_near @ p)
            rhs = s_top @ (st0.A @ p)
            assert np.allclose(lhs / lhs[2], rhs / rhs[2], atol=1e-9)
        # seam 은 top 행 스케일을 따라야 한다
        exp_seam = s_top[1, 1] * 464 + s_top[1, 2]
        assert st1.seam == pytest.approx(exp_seam, abs=1.0)


def test_rescale_to_source_exact_inverse(monkeypatch):
    monkeypatch.setattr(fl, "load_stitch_calib", lambda m: {})
    A = _similarity(1.0104, 0.478, -5.81, 451.85)         # down.json 상당
    calib = {"A": A.tolist(), "seam": 464, "left": 8, "right": 640, "note": "t"}
    intr = {c: _khat(c, 1920, 1080) for c in ("top", "near")}
    st = fl.Stitcher("down", w=1920, h=1080, calib=calib, intr=intr)
    for v in (st.seam, st.seam + 111, st.out_h - 1):
        for u in (0, 700, 1500):
            cam, su, sv = st.to_source(float(u), float(v))
            bu, bv = st.from_source(cam, su, sv)
            assert bu == pytest.approx(u, abs=1e-6)
            assert bv == pytest.approx(v, abs=1e-6)


def test_rescale_legacy_no_calib(monkeypatch):
    """캘리브 파일이 없어도(레거시 평행이동) FHD 재스케일이 성립해야 한다."""
    monkeypatch.setattr(fl, "load_stitch_calib", lambda m: {})
    st = fl.Stitcher("up", w=1920, h=1080)                # intr 없음 → 크롭모델 폴백
    assert st.w == 1920 and 900 < st.seam < 1080
    assert st.out_h > st.seam
    top = np.zeros((1080, 1920, 3), np.uint8)
    near = np.zeros((1080, 1920, 3), np.uint8)
    out = st.stitch(top, near)
    assert out.shape == (st.out_h, st.right - st.left, 3)


def test_crop_model_k_matches_measured_fx_ratio():
    """크롭 모델의 자기일관성: 640→1920 배율이 2.25, 주점 이동이 크롭 오프셋."""
    for cam in ("top", "near"):
        fx0, fy0, cx0, cy0 = fl.CALIB_REF_K[cam]
        fx1, fy1, cx1, cy1 = fl.crop_model_k(cam, 1920, 1080)
        assert fx1 / fx0 == pytest.approx(2.25, abs=1e-9)
        assert cx1 == pytest.approx(cx0 * 2.25 + 240.0, abs=1e-9)
        assert cy1 == pytest.approx(cy0 * 2.25, abs=1e-9)


def test_rescale_rejects_unknown_ref_res(monkeypatch):
    # 크롭 모델이 모르는 기준 해상도(4:3 비640)는 여전히 거부
    monkeypatch.setattr(fl, "load_stitch_calib", lambda m: {})
    calib = {"A": np.eye(3).tolist(), "seam": 500, "left": 0, "right": 800,
             "res": [800, 600], "note": "t"}
    with pytest.raises(ValueError):
        fl.Stitcher("up", w=1920, h=1080, calib=calib)


def test_rescale_between_16_9_refs(monkeypatch):
    # 2026-07-21 일반화: FHD 네이티브 캘리브 시대 — 16:9 기준끼리(및 →640 역방향)
    # 크롭 모델 K 쌍으로 변환한다. 720p→1080p 는 순수 ×1.5 배율.
    monkeypatch.setattr(fl, "load_stitch_calib", lambda m: {})
    calib = {"A": np.eye(3).tolist(), "seam": 700, "left": 0, "right": 1280,
             "res": [1280, 720], "note": "t"}
    st = fl.Stitcher("up", w=1920, h=1080, calib=calib)
    assert st.seam == 1050
    assert (st.left, st.right) == (0, 1920)
