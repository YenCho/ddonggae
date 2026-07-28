#!/usr/bin/env python3
"""상/근접 캠 스티치 캘리브레이터 — 브라우저 UI.

왜 필요한가
-----------
지금까지의 스티치 파라미터는 **대응점 한 개**로 뽑았다
(`logs/real_validation/stitch_sweep/*/calib_result.json` 의 `correspondence`).
점 하나는 평행이동(dx, dy)밖에 못 정한다 — 두 캠의 **배율(시야각)** 과 **롤 회전**
차이는 전혀 구속하지 못하므로, 가운데는 맞고 **시야 양 끝이 벌어지는** 지금 증상이
그대로 나온다.

이 도구는 대응점을 여러 개(양 끝 포함) 잡아 배율·회전까지 포함한 변환을 적합시키고,
잔차를 픽셀 단위로 보여준다. 결과는 `data/calibration/stitch/<mast>.json` 에 저장되고
`_fieldlib.Stitcher` 가 자동으로 읽는다. 파일이 없으면 기존 평행이동 모델 그대로다
(비트 단위 동일 — 저장 전에는 파이프라인이 바뀌지 않는다).

기하 모델
---------
`A` = **near 픽셀 → top 프레임(아래로 연장) 픽셀** 3x3 변환.
스티치 (u,v) 는 top 프레임 (u+left, v) 이므로 v>=seam 이면 원본 near = A⁻¹·(u+left, v).
`to_source()` 가 이 역변환을 정확히 수행하므로 3D 역투영 계약(원본 캠 프레임)이 유지된다.

사용
----
    # 실기 (카메라 노드가 떠 있어야 함)
    python3 scripts/dev/field_ops/stitch_calibrator.py --mast up
    # 저장된 페어로 오프라인 (경기장 시간 아껴서 나중에)
    python3 scripts/dev/field_ops/stitch_calibrator.py --mast up \
        --pair-dir logs/field_ops/20260720_032304_up/scan

    → http://<jetson>:8099/   (SSH 로 들어와도 브라우저로 접속 가능)

절차는 `docs/hardware/camera_stitch_calibration.md` 참조.
"""
import argparse
import io
import json
import math
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fieldlib as fl  # noqa: E402

REPO_ROOT = fl.REPO_ROOT
TOPICS = {"top": "/camera_19/rgb", "near": "/camera_54/rgb"}
# 카메라 노드 이름 (real_competition_bridge.launch.py: namespace=camera/<n>, name=<n>)
CAM_NODES = {"top": "/camera/top/top", "near": "/camera/bottom/bottom"}
PHOTO_PARAMS = ("enable_auto_exposure", "exposure", "gain",
                "enable_auto_white_balance", "white_balance",
                "power_line_frequency")


# =========================================================================
# 프레임 소스
# =========================================================================
class OfflineSource:
    """저장된 top/near 페어 디렉터리. shotNN_top.png / pair_NN_top.png 둘 다."""

    PAT = re.compile(r"^(.*?)_?(top)\.png$")

    def __init__(self, path: Path):
        self.dir = Path(path)
        tops = sorted(p for p in self.dir.glob("*top.png") if "depth" not in p.name)
        self.pairs = []
        for t in tops:
            n = t.parent / t.name.replace("top.png", "near.png")
            if not n.exists():
                n = t.parent / t.name.replace("top.png", "bottom.png")
            if n.exists():
                self.pairs.append((t, n))
        if not self.pairs:
            raise SystemExit(f"{self.dir} 에서 top/near 페어를 못 찾음")
        self.idx = 0
        self.live = False

    def info(self):
        t, _ = self.pairs[self.idx]
        return f"오프라인 {self.dir.name} — {self.idx + 1}/{len(self.pairs)} ({t.name})"

    def get(self):
        t, n = self.pairs[self.idx]
        return cv2.imread(str(t)), cv2.imread(str(n))

    def step(self, d):
        self.idx = (self.idx + d) % len(self.pairs)

    frozen = None                     # 오프라인은 원래 정지 화면 — freeze 무의미

    def freeze(self, on):
        return False


class LiveSource:
    """ROS 토픽 구독 (rclpy). 카메라 노드가 떠 있어야 한다."""

    def __init__(self):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image

        rclpy.init()
        self.node = Node("stitch_calibrator")
        self.last = {"top": None, "near": None}
        self.frozen = None            # (top, near) 래치 — freeze 모드
        self.lock = threading.Lock()

        def cb(key):
            def _f(msg):
                ch = 4 if msg.encoding == "rgba8" else 3
                a = np.frombuffer(msg.data, np.uint8).reshape(
                    msg.height, msg.width, ch)[:, :, :3]
                bgr = a if msg.encoding.startswith("bgr") else a[:, :, ::-1]
                with self.lock:
                    self.last[key] = np.ascontiguousarray(bgr)
            return _f

        for k, t in TOPICS.items():
            self.node.create_subscription(Image, t, cb(k), qos_profile_sensor_data)
        self._stop = False
        threading.Thread(target=self._spin, daemon=True).start()
        self.live = True

    def _spin(self):
        import rclpy
        while not self._stop:
            rclpy.spin_once(self.node, timeout_sec=0.1)

    def info(self):
        with self.lock:
            ok = sum(v is not None for v in self.last.values())
            fz = self.frozen is not None
        return (f"라이브 {TOPICS['top']} / {TOPICS['near']} — 수신 {ok}/2"
                + (" · ❄ FREEZE" if fz else ""))

    def get(self):
        with self.lock:
            if self.frozen is not None:
                return self.frozen
            return self.last["top"], self.last["near"]

    def freeze(self, on: bool) -> bool:
        """현재 프레임 래치. 대응점 top/near 클릭이 같은 순간의 프레임 위에서 이뤄진다."""
        with self.lock:
            if on and all(v is not None for v in self.last.values()):
                self.frozen = (self.last["top"].copy(), self.last["near"].copy())
            else:
                self.frozen = None
            return self.frozen is not None

    def step(self, d):
        pass


# =========================================================================
# 적합 (near -> top 프레임)
# =========================================================================
def fit_A(points, model):
    """points: [{'top':[u,v], 'near':[u,v]}]. 반환 (A 3x3, 사유) 또는 (None, 사유)."""
    if len(points) < 1:
        return None, "대응점이 없다"
    src = np.array([p["near"] for p in points], np.float32).reshape(-1, 1, 2)
    dst = np.array([p["top"] for p in points], np.float32).reshape(-1, 1, 2)
    n = len(points)
    need = {"translate": 1, "similarity": 2, "affine": 3, "homography": 4}[model]
    if n < need:
        return None, f"{model} 은 대응점 {need}개 이상 필요 (현재 {n})"

    s2 = src.reshape(-1, 2).astype(np.float64)
    d2 = dst.reshape(-1, 2).astype(np.float64)

    # affine/homography 는 v(세로) 방향 퍼짐이 없으면 발산한다. 겹침 띠가 70행쯤
    # 밖에 안 되므로 바닥 가장자리를 따라 한 줄로 찍으면 반드시 이 상황이 된다.
    # 1px 클릭잡음·8점 시뮬 기준 양끝 예측오차:
    #   v퍼짐   5px    20px   40px   60px
    #   affine  1.9    1.8    1.8    1.8   ← 사실상 무관
    #   homog   330    83     46     6.0   ← 발산
    # similarity 는 u 폭(≈570px)이 회전을 구속하므로 v퍼짐과 무관하게 ~1.1px.
    v_span = float(s2[:, 1].max() - s2[:, 1].min())
    if model == "homography" and v_span < 40.0:
        return None, (f"대응점이 세로로 {v_span:.0f}px 안에 몰려 있다 — homography 는 "
                      f"발산한다(40px 이상 필요). similarity 를 쓸 것 ★")
    if model == "affine" and v_span < 15.0:
        return None, (f"대응점이 세로로 {v_span:.0f}px 안에 몰려 있다 — affine 이 "
                      f"불안정하다(15px 이상 필요). similarity 를 쓸 것 ★")

    if model == "translate":
        d = (d2 - s2).mean(axis=0)
        A = np.eye(3)
        A[0, 2], A[1, 2] = float(d[0]), float(d[1])
        return A, f"평행이동 {n}점 평균"

    if model == "similarity":
        # 복소수 최소제곱: y = c*x + t (c 가 배율×회전). 점이 적어도 안전하다.
        # 점이 넉넉하면 RANSAC 으로 이상점을 걸러 다시 잡는다 (자동매칭 대비).
        if n >= 6:
            M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                 ransacReprojThreshold=3.0)
            if M is not None:
                k = int(inl.sum()) if inl is not None else n
                return np.vstack([M, [0, 0, 1]]), f"배율+회전+이동 {n}점 (RANSAC 인라이어 {k})"
        x = s2[:, 0] + 1j * s2[:, 1]
        y = d2[:, 0] + 1j * d2[:, 1]
        xc, yc = x - x.mean(), y - y.mean()
        den = float((np.abs(xc) ** 2).sum())
        if den < 1e-12:
            return None, "대응점이 한 자리에 몰려 있다 — 떨어뜨려 찍을 것"
        c = complex((np.conj(xc) * yc).sum() / den)
        t = complex(y.mean() - c * x.mean())
        A = np.array([[c.real, -c.imag, t.real],
                      [c.imag, c.real, t.imag],
                      [0.0, 0.0, 1.0]])
        return A, f"배율+회전+이동 {n}점 최소제곱"

    if model == "affine":
        if n >= 6:
            M, inl = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC,
                                          ransacReprojThreshold=3.0)
            if M is not None:
                k = int(inl.sum()) if inl is not None else n
                return np.vstack([M, [0, 0, 1]]), f"아핀 {n}점 (RANSAC 인라이어 {k})"
        X = np.hstack([s2, np.ones((n, 1))])
        M, *_ = np.linalg.lstsq(X, d2, rcond=None)     # (3,2)
        return np.vstack([M.T, [0, 0, 1]]), f"아핀 {n}점 최소제곱"

    H, inl = cv2.findHomography(src, dst,
                                cv2.RANSAC if n >= 6 else 0, 3.0)
    if H is None:
        return None, "homography 적합 실패 — 점이 한 직선 위에 있지 않은지 확인"
    k = int(inl.sum()) if inl is not None else n
    return H, f"호모그래피 {n}점 (인라이어 {k})"


def residuals(A, points):
    """각 대응점의 잔차(px)와 RMS."""
    if A is None or not points:
        return [], 0.0
    out = []
    for p in points:
        q = A @ np.array([p["near"][0], p["near"][1], 1.0])
        q = q[:2] / q[2]
        out.append(float(math.hypot(q[0] - p["top"][0], q[1] - p["top"][1])))
    rms = float(math.sqrt(sum(r * r for r in out) / len(out)))
    return out, rms


def quality(A, points, thresh: float = 3.0):
    """인라이어/이상점을 갈라 본 적합 품질.

    RANSAC(임계 3px)이 이상점을 버리고 A 를 잡는데, 전체 점에 대한 RMS 하나만
    보여주면 버려진 점의 오차가 섞여 들어가 실제보다 나쁘게 보인다. 실제로
    `down.json` 은 헤드라인 RMS 3.70px 이지만 인라이어 4점만 보면 0.93px 이다.
    합격선(≤1.5px)은 **인라이어 RMS** 에 대해 판정해야 한다.
    """
    res, rms_all = residuals(A, points)
    if not res:
        return {"rms_all": 0.0, "rms_in": 0.0, "n_in": 0, "n_out": 0, "res": []}
    inl = [r for r in res if r <= thresh]
    rms_in = math.sqrt(sum(r * r for r in inl) / len(inl)) if inl else rms_all
    return {"rms_all": rms_all, "rms_in": rms_in,
            "n_in": len(inl), "n_out": len(res) - len(inl), "res": res}


def decompose(A):
    """A 의 배율/회전/이동 요약 (사람이 읽는 용도)."""
    sx = math.hypot(A[0, 0], A[1, 0])
    sy = math.hypot(A[0, 1], A[1, 1])
    rot = math.degrees(math.atan2(A[1, 0], A[0, 0]))
    persp = float(max(abs(A[2, 0]), abs(A[2, 1])))
    return {"scale_x": sx, "scale_y": sy, "rot_deg": rot,
            "dx": float(A[0, 2]), "dy": float(A[1, 2]), "persp": persp}


def auto_match(top, near, A0, band):
    """겹침 띠에서 특징점 자동 매칭 → 대응점 후보. 바닥이 밋밋하면 적게 나온다."""
    if top is None or near is None:
        return [], "프레임 없음"
    v0, v1 = band
    if v1 - v0 < 8:
        return [], "겹침 띠가 너무 얇다"
    g_t = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
    g_n = cv2.cvtColor(near, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    g_t, g_n = clahe.apply(g_t), clahe.apply(g_n)
    # near 를 A0 로 top 프레임에 올려 같은 띠를 본다
    h_t, w_t = g_t.shape
    warp = cv2.warpPerspective(g_n, A0, (w_t, max(h_t, v1 + 1)))
    t_band = g_t[v0:v1]
    n_band = warp[v0:v1]
    det = cv2.AKAZE_create()
    k1, d1 = det.detectAndCompute(t_band, None)
    k2, d2 = det.detectAndCompute(n_band, None)
    if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
        return [], "겹침 띠에 특징점이 부족 (바닥이 밋밋 — 무늬 있는 타깃을 놓고 다시)"
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(d1, d2), key=lambda m: m.distance)[:60]
    A0i = np.linalg.inv(A0)
    pts = []
    for m in matches:
        tu, tv = k1[m.queryIdx].pt
        nu, nv = k2[m.trainIdx].pt
        tv, nv = tv + v0, nv + v0            # 띠 오프셋 복원
        q = A0i @ np.array([nu, nv, 1.0])    # top 프레임 → 원본 near
        pts.append({"top": [float(tu), float(tv)],
                    "near": [float(q[0] / q[2]), float(q[1] / q[2])], "src": "auto"})
    return pts, f"AKAZE 매칭 {len(pts)}쌍 (겹침 띠 v{v0}~{v1})"


def edge_check(top, near, A, band):
    """겹침 띠를 좌/중/우 3등분해 각 구간의 남은 어긋남을 위상상관으로 측정.

    세 값이 서로 다르면 **평행이동으로는 못 잡는 배율/회전 오차**가 남아 있다는 뜻 —
    '가운데는 맞는데 양 끝이 안 맞는' 증상의 정량 지표.
    """
    if top is None or near is None or A is None:
        return []
    v0, v1 = band
    if v1 - v0 < 8:
        return []
    h_t, w_t = top.shape[:2]
    g_t = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY).astype(np.float32)
    warp = cv2.warpPerspective(cv2.cvtColor(near, cv2.COLOR_BGR2GRAY), A,
                               (w_t, max(h_t, v1 + 1))).astype(np.float32)
    out = []
    thirds = [("좌", 0, w_t // 3), ("중", w_t // 3, 2 * w_t // 3), ("우", 2 * w_t // 3, w_t)]
    for label, c0, c1 in thirds:
        a = g_t[v0:v1, c0:c1]
        b = warp[v0:v1, c0:c1]
        if a.shape != b.shape or a.size == 0 or b.std() < 1e-3 or a.std() < 1e-3:
            out.append({"zone": label, "dx": None, "dy": None, "conf": 0.0})
            continue
        win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
        (dx, dy), conf = cv2.phaseCorrelate(a, b, win)
        out.append({"zone": label, "dx": float(dx), "dy": float(dy), "conf": float(conf)})
    return out


# =========================================================================
# 상태
# =========================================================================
class State:
    def __init__(self, src, mast):
        self.src = src
        self.mast = mast
        self.lock = threading.Lock()
        self.points = []
        self.model = "similarity"
        self.msg = "대응점을 찍고 [적합]을 누르세요."
        self.res = None                       # (w, h) — 실프레임 도착 시 확정
        self.nudge = {"dx": 0.0, "dy": 0.0, "scale": 1.0, "rot": 0.0}
        self._init_calib(fl.CALIB_REF_RES)    # 프레임 오기 전 임시(640 기준)

    def _init_calib(self, res):
        """캘리브를 res 픽셀 좌표계로 로드 (기존 640 캘리브는 Stitcher 가 자동 재스케일)."""
        w, h = res
        try:
            st = fl.Stitcher(self.mast, w=w, h=h)
        except ValueError:                    # 저장본 좌표계에서 변환 불가 → 레거시로 시작
            st = fl.Stitcher(self.mast, w=w, h=h, calib={})
        self.res = (w, h)
        self.base_A = st.A.copy()
        self.A = st.A.copy()
        self.seam = st.seam
        self.left, self.right = st.left, st.right
        self.calib_note = st.calib_source

    def ensure_res(self, top):
        """프레임 해상도가 상태 좌표계와 다르면 전환한다. 점은 좌표계가 달라 무효."""
        if top is None:
            return
        res = (top.shape[1], top.shape[0])
        if res == self.res:
            return
        had = len(self.points)
        self._init_calib(res)
        self.points = []
        self.nudge = {"dx": 0.0, "dy": 0.0, "scale": 1.0, "rot": 0.0}
        self.msg = (f"해상도 {res[0]}x{res[1]} 감지 — 캘리브 좌표계 전환"
                    + (f", 이전 해상도의 점 {had}개 비움" if had else ""))

    # ---- 겹침 띠 (top 프레임 기준) ----
    def band(self, top, near):
        if top is None or near is None:
            return (0, 0)
        h_t = top.shape[0]
        c = np.array([[0, 0, 1], [near.shape[1], 0, 1]], np.float64).T
        m = self.A @ c
        v_near_top = float((m[1] / m[2]).min())
        v0 = max(0, int(math.floor(v_near_top)))
        return (v0, h_t) if h_t > v0 else (0, 0)

    def effective_A(self):
        """적합 A 에 수동 미세조정(nudge)을 얹은 최종 변환."""
        n = self.nudge
        cx, cy = self.res[0] / 2.0, self.res[1] / 2.0   # 회전/배율 피벗 = 프레임 중앙
        th = math.radians(n["rot"])
        s = n["scale"]
        R = np.array([[s * math.cos(th), -s * math.sin(th), 0],
                      [s * math.sin(th), s * math.cos(th), 0],
                      [0, 0, 1.0]])
        T1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
        T2 = np.array([[1, 0, cx + n["dx"]], [0, 1, cy + n["dy"]], [0, 0, 1.0]])
        return T2 @ R @ T1 @ self.A

    def stitcher(self):
        w, h = self.res
        A = self.effective_A()
        left, right = coverage_bounds(A, w, h)
        return fl.Stitcher(self.mast, w=w, h=h, calib={
            "A": A.tolist(), "seam": self.seam,
            "left": left, "right": right, "res": [w, h],
            "note": "calibrator-preview"})


def coverage_bounds(A, w, h):
    """near(w×h)가 A 로 top 프레임을 덮는 실제 가로 범위 → 스티치 left/right.

    구 640 캘리브에서 상속되던 4:3 크롭을 버리고 항상 실커버리지 전체 폭을
    쓴다 (2026-07-21). affine/homography 모두 가장자리 극값은 모서리에서 난다.
    """
    A = np.asarray(A, np.float64)

    def tx(x, y):
        p = A @ np.array([x, y, 1.0])
        return p[0] / p[2]

    left = max(0, int(math.ceil(max(tx(0, 0), tx(0, h)))))
    right = min(w, int(math.floor(min(tx(w, 0), tx(w, h)))))
    return (left, right) if right > left else (0, w)


# =========================================================================
# 렌더링
# =========================================================================
def render_seam(state, top, near, view, band_pad=40, zoom=1.0):
    """겹침 띠 점검 뷰."""
    if top is None or near is None:
        img = np.zeros((240, 640, 3), np.uint8)
        cv2.putText(img, "no frame", (20, 130), 0, 1.0, (0, 0, 255), 2)
        return img
    A = state.effective_A()
    h_t, w_t = top.shape[:2]
    v0, v1 = state.band(top, near)
    if v1 <= v0:
        v0, v1 = max(0, state.seam - band_pad), min(h_t, state.seam + band_pad)
    warp = cv2.warpPerspective(near, A, (w_t, max(h_t, v1 + 1)))
    a, b = top[v0:v1], warp[v0:v1]
    if view == "diff":
        out = cv2.absdiff(a, b)
        out = cv2.applyColorMap(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_INFERNO)
    elif view == "blink":
        out = a if int(time.time() * 2) % 2 == 0 else b
        out = out.copy()
    elif view == "checker":
        out = a.copy()
        cs = 32
        for y in range(0, out.shape[0], cs):
            for x in range(0, out.shape[1], cs):
                if ((x // cs) + (y // cs)) % 2:
                    out[y:y + cs, x:x + cs] = b[y:y + cs, x:x + cs]
    elif view == "overlay":
        out = cv2.addWeighted(a, 0.5, b, 0.5, 0)
    else:  # edges — 경계선만 겹쳐 보기 (양 끝 어긋남이 가장 잘 보인다)
        ea = cv2.Canny(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), 60, 160)
        eb = cv2.Canny(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), 60, 160)
        out = np.zeros_like(a)
        out[:, :, 2] = ea      # top = 빨강
        out[:, :, 1] = eb      # near = 초록  (겹치면 노랑)
    # 좌/중/우 구획선
    for x in (out.shape[1] // 3, 2 * out.shape[1] // 3):
        cv2.line(out, (x, 0), (x, out.shape[0]), (255, 255, 255), 1)
    if zoom != 1.0:
        out = cv2.resize(out, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    return out


def render_stitch(state, top, near, mark_seam=True):
    if top is None or near is None:
        return np.zeros((240, 640, 3), np.uint8)
    out = state.stitcher().stitch(top, near).copy()
    if mark_seam:
        cv2.line(out, (0, state.seam), (out.shape[1], state.seam), (0, 0, 255), 1)
    return out


def jpg(img, quality=80):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


# =========================================================================
# 포토메트리 (노출/게인/화이트밸런스 잠금)
# =========================================================================
def photo_apply(values, dry=False):
    """두 카메라에 **동일한** 고정값을 적용. auto 를 먼저 끄고 수동값을 넣는다."""
    log = []
    order = ["enable_auto_exposure", "enable_auto_white_balance",
             "exposure", "gain", "white_balance", "power_line_frequency"]
    for cam, node in CAM_NODES.items():
        for key in order:
            if key not in values:
                continue
            v = values[key]
            v = "true" if v is True else "false" if v is False else str(v)
            cmd = ["ros2", "param", "set", node, f"rgb_camera.{key}", v]
            if dry:
                log.append(f"[dry] {' '.join(cmd)}")
                continue
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                ok = r.returncode == 0 and "successful" in r.stdout.lower()
                log.append(f"{cam}.{key}={v} {'ok' if ok else (r.stdout + r.stderr).strip()[:60]}")
            except Exception as e:  # noqa: BLE001
                log.append(f"{cam}.{key}={v} 실패: {e}")
    return log


def photo_read():
    out = {}
    for cam, node in CAM_NODES.items():
        d = {}
        for key in PHOTO_PARAMS:
            try:
                r = subprocess.run(["ros2", "param", "get", node, f"rgb_camera.{key}"],
                                   capture_output=True, text=True, timeout=6)
                d[key] = r.stdout.strip().split(":")[-1].strip() if r.returncode == 0 else "?"
            except Exception:  # noqa: BLE001
                d[key] = "?"
        out[cam] = d
    return out


def photo_stats(top, near, band):
    """두 캠의 밝기/색 통계 + 겹침 띠에서의 차이. WB/노출이 갈렸는지 객관 지표."""
    if top is None or near is None:
        return {}
    v0, v1 = band

    def st(img, rows=None):
        a = img if rows is None else img[rows[0]:rows[1]]
        if a.size == 0:
            return None
        b, g, r = [float(a[:, :, i].mean()) for i in range(3)]
        y = 0.114 * b + 0.587 * g + 0.299 * r
        return {"b": b, "g": g, "r": r, "y": y,
                "rg": r / max(g, 1e-6), "bg": b / max(g, 1e-6)}

    band_t = st(top, (v0, v1)) if v1 > v0 else None
    out = {"top": st(top), "near": st(near), "band_top": band_t}
    if band_t:
        out["band_near"] = st(near, (0, max(1, v1 - v0)))
        bn = out["band_near"]
        out["delta"] = {"y": band_t["y"] - bn["y"],
                        "rg": band_t["rg"] - bn["rg"],
                        "bg": band_t["bg"] - bn["bg"]}
    return out


# =========================================================================
# HTTP
# =========================================================================
PAGE = r"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>스티치 캘리브레이터</title><style>
body{font-family:'Noto Sans KR',sans-serif;margin:10px;background:#f6f6f6;color:#222;font-size:14px}
h1{font-size:19px;margin:0 0 8px}
.wrap{display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap}
.col{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px}
.pick{position:relative;display:inline-block}
.pick img{display:block;border:2px solid #bbb;cursor:crosshair;width:420px}
.pick.act img{border-color:#c0392b}
.lbl{font-weight:bold;margin:4px 0}
#mag{position:fixed;width:180px;height:180px;border:2px solid #333;background:#000;
     display:none;pointer-events:none;z-index:99;image-rendering:pixelated}
table{border-collapse:collapse;font-size:12px}
td,th{border:1px solid #ddd;padding:2px 6px}
button{font-size:13px;padding:5px 10px;margin:2px;border-radius:5px;border:1px solid #888;
       background:#2c3e50;color:#fff;cursor:pointer}
button.s{background:#fff;color:#222}
.bad{color:#c0392b;font-weight:bold}.good{color:#27ae60;font-weight:bold}
.hint{color:#666;font-size:12px}
input[type=range]{width:150px;vertical-align:middle}
code{background:#eee;padding:1px 4px;border-radius:3px}
</style></head><body>
<h1>🔧 스티치 캘리브레이터 <span id="src" class="hint"></span></h1>
<div id="msg" class="hint"></div>
<div class="wrap">

<div class="col">
  <div class="lbl">① 대응점 찍기 — 두 그림에서 <b>같은 물리 점</b>을 차례로 클릭</div>
  <div class="hint">정확도가 전부다. 확대경을 보며 모서리 꼭짓점을 찍을 것.
  <b>반드시 좌·중·우 골고루</b> — 가운데만 찍으면 지금과 똑같이 양 끝이 벌어진다.</div>
  <div>
    <div class="pick" id="pt"><div class="lbl">TOP (상단캠)</div>
      <img id="itop" src="/img/top.jpg"></div>
    <div class="pick" id="pn"><div class="lbl">NEAR (근접캠)</div>
      <img id="inear" src="/img/near.jpg"></div>
  </div>
  <div id="pend" class="hint"></div>
  <div>
    <button class="s" id="bfreeze" onclick="post('/api/freeze',{on:!(S&&S.frozen)})">❄ freeze</button>
    <button class="s" onclick="post('/api/points',{op:'undo'})">↩ 마지막 삭제</button>
    <button class="s" onclick="post('/api/points',{op:'clear'})">전체 지우기</button>
    <button class="s" onclick="post('/api/auto',{})">🔍 자동 매칭 시도</button>
    <button class="s" onclick="post('/api/step',{d:1})">다음 페어 ▶</button>
  </div>
</div>

<div class="col">
  <div class="lbl">② 적합</div>
  <select id="model">
    <option value="translate">translate — 이동만 (1점, 기존 방식)</option>
    <option value="similarity" selected>similarity — 이동+배율+회전 (2점~) ★권장</option>
    <option value="affine">affine — +비등방 배율 (3점~)</option>
    <option value="homography">homography — +원근 (4점~)</option>
  </select>
  <button onclick="post('/api/fit',{model:document.getElementById('model').value})">적합</button>
  <button class="s" onclick="post('/api/reset',{})">되돌리기</button>
  <div id="fitinfo"></div>
  <table id="ptab"></table>

  <div class="lbl" style="margin-top:10px">③ 양 끝 점검 (위상상관 잔여 어긋남)</div>
  <div class="hint">좌·중·우 값이 <b>서로 다르면</b> 배율/회전이 아직 안 맞은 것.
  전부 비슷하고 0에 가까우면 정렬 완료.</div>
  <table id="etab"></table>

  <div class="lbl" style="margin-top:10px">④ 수동 미세조정</div>
  <div>dx <input type="range" id="n_dx" min="-40" max="40" step="0.5" value="0"><span id="v_dx">0</span></div>
  <div>dy <input type="range" id="n_dy" min="-40" max="40" step="0.5" value="0"><span id="v_dy">0</span></div>
  <div>배율 <input type="range" id="n_scale" min="0.9" max="1.1" step="0.001" value="1"><span id="v_scale">1</span></div>
  <div>회전° <input type="range" id="n_rot" min="-5" max="5" step="0.05" value="0"><span id="v_rot">0</span></div>

  <div class="lbl" style="margin-top:10px">⑤ 저장</div>
  <button onclick="post('/api/save',{})">💾 <span id="savepath"></span> 에 저장</button>
  <div class="hint">저장하면 <code>_fieldlib.Stitcher</code>가 다음 실행부터 자동으로 읽는다.
  저장 전에는 파이프라인이 전혀 바뀌지 않는다.</div>
</div>

<div class="col">
  <div class="lbl">겹침 띠 점검</div>
  <select id="view" onchange="refresh()">
    <option value="edges" selected>edges — 경계선(빨강=top, 초록=near, 노랑=일치)</option>
    <option value="diff">diff — 차영상</option>
    <option value="blink">blink — 교대 점멸</option>
    <option value="checker">checker — 체커보드</option>
    <option value="overlay">overlay — 반투명</option>
  </select>
  <div><img id="seam" style="width:430px;border:1px solid #999"></div>
  <div class="lbl" style="margin-top:8px">스티치 결과 (빨간선=이음매)</div>
  <div><img id="stitch" style="width:300px;border:1px solid #999"></div>
</div>

<div class="col">
  <div class="lbl">⑥ 노출·화이트밸런스 잠금</div>
  <div class="hint">두 캠에 <b>같은 고정값</b>을 넣는다. auto 를 끄지 않으면 두 캠이
  따로 수렴해 이음매에서 색·밝기가 갈린다.</div>
  <div>노출 <input type="range" id="p_exposure" min="20" max="800" step="5" value="156"><span id="v_exposure">156</span></div>
  <div>게인 <input type="range" id="p_gain" min="0" max="128" step="1" value="64"><span id="v_gain">64</span></div>
  <div>색온도 <input type="range" id="p_white_balance" min="2800" max="6500" step="50" value="4600"><span id="v_white_balance">4600</span></div>
  <div>전원주파수
    <select id="p_power_line_frequency">
      <option value="0">0 = 없음</option><option value="1">1 = 50Hz</option>
      <option value="2" selected>2 = 60Hz</option></select>
    <span class="hint">체육관 형광등 깜빡임(밴딩) 방지</span></div>
  <button onclick="applyPhoto()">두 캠에 적용 + auto 끄기</button>
  <button class="s" onclick="post('/api/photo_read',{})">현재값 읽기</button>
  <div id="photo" class="hint"></div>
  <div class="lbl" style="margin-top:8px">측광 비교</div>
  <table id="stab"></table>
</div>
</div>

<canvas id="mag" width="180" height="180"></canvas>
<script>
let S={}, pending=null;
function post(u,b){fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify(b)}).then(r=>r.json()).then(j=>{apply(j);});}
function fmt(x,n){return x===null||x===undefined?'—':(+x).toFixed(n===undefined?2:n);}

function apply(j){
  const wasFrozen=!!(S&&S.frozen);
  S=j;
  const bf=document.getElementById('bfreeze');
  if(bf){bf.textContent=j.frozen?'▶ 라이브 재개':'❄ freeze';
         bf.style.background=j.frozen?'#07c':''; bf.style.color=j.frozen?'#fff':'';}
  if(wasFrozen!==!!j.frozen) forceRefresh();   // 토글 직후 한 번만 이미지 갱신
  document.getElementById('src').textContent=j.source||'';
  document.getElementById('msg').textContent=j.msg||'';
  document.getElementById('savepath').textContent=j.save_path||'';
  const d=j.decomp||{};
  document.getElementById('fitinfo').innerHTML =
    `<b>RMS 잔차 <span class="${j.rms<1.5?'good':'bad'}">${fmt(j.rms)} px</span></b> · `
    + `배율 ${fmt(d.scale_x,4)}/${fmt(d.scale_y,4)} · 회전 ${fmt(d.rot_deg,3)}° · `
    + `이동 (${fmt(d.dx,1)}, ${fmt(d.dy,1)})`;
  let t='<tr><th>#</th><th>top</th><th>near</th><th>잔차px</th><th></th></tr>';
  (j.points||[]).forEach((p,i)=>{ const r=j.res[i];
    t+=`<tr><td>${i+1}</td><td>${fmt(p.top[0],1)},${fmt(p.top[1],1)}</td>`
      +`<td>${fmt(p.near[0],1)},${fmt(p.near[1],1)}</td>`
      +`<td class="${r>3?'bad':''}">${fmt(r)}</td>`
      +`<td><button class="s" onclick="post('/api/points',{op:'del',i:${i}})">✕</button></td></tr>`;});
  document.getElementById('ptab').innerHTML=t;
  let e='<tr><th>구간</th><th>dx px</th><th>dy px</th><th>신뢰</th></tr>';
  (j.edges||[]).forEach(z=>{ const m=Math.max(Math.abs(z.dx||0),Math.abs(z.dy||0));
    e+=`<tr><td>${z.zone}</td><td class="${m>2?'bad':'good'}">${fmt(z.dx)}</td>`
      +`<td class="${m>2?'bad':'good'}">${fmt(z.dy)}</td><td>${fmt(z.conf)}</td></tr>`;});
  document.getElementById('etab').innerHTML=e;
  const s=j.photo_stats||{};
  let g='<tr><th></th><th>Y</th><th>R/G</th><th>B/G</th></tr>';
  ['top','near'].forEach(k=>{const v=s[k]; if(v) g+=`<tr><td>${k}</td><td>${fmt(v.y,1)}</td>`
    +`<td>${fmt(v.rg,3)}</td><td>${fmt(v.bg,3)}</td></tr>`;});
  if(s.delta){const D=s.delta; const bad=Math.abs(D.y)>6||Math.abs(D.rg)>0.05||Math.abs(D.bg)>0.05;
    g+=`<tr><td><b>이음매차</b></td><td class="${bad?'bad':'good'}">${fmt(D.y,1)}</td>`
      +`<td class="${bad?'bad':'good'}">${fmt(D.rg,3)}</td><td class="${bad?'bad':'good'}">${fmt(D.bg,3)}</td></tr>`;}
  document.getElementById('stab').innerHTML=g;
  if(j.photo_read) document.getElementById('photo').textContent=JSON.stringify(j.photo_read);
  if(j.log) document.getElementById('photo').textContent=j.log.join(' | ');
  document.getElementById('pend').textContent = pending
    ? `TOP (${pending[0].toFixed(1)}, ${pending[1].toFixed(1)}) 찍음 → 이제 NEAR 에서 같은 점을 클릭`
    : '다음: TOP 에서 점을 클릭';
  document.getElementById('pt').className='pick'+(pending?'':' act');
  document.getElementById('pn').className='pick'+(pending?' act':'');
}

function nat(img,ev){const r=img.getBoundingClientRect();
  return [(ev.clientX-r.left)*img.naturalWidth/r.width,
          (ev.clientY-r.top)*img.naturalHeight/r.height];}

['itop','inear'].forEach(id=>{const img=document.getElementById(id);
  img.addEventListener('click',ev=>{const p=nat(img,ev);
    if(id==='itop'){pending=p; apply(S);}
    else if(pending){post('/api/points',{op:'add',top:pending,near:p}); pending=null;}
    else {document.getElementById('msg').textContent='TOP 부터 클릭하세요.';}});
  // 확대경
  const mag=document.getElementById('mag'), mx=mag.getContext('2d');
  img.addEventListener('mousemove',ev=>{const p=nat(img,ev); const Z=6, R=15;
    mag.style.display='block';
    mag.style.left=Math.min(window.innerWidth-190,ev.clientX+20)+'px';
    mag.style.top=Math.max(4,ev.clientY-190)+'px';
    // 700ms 리로드 도중(img 디코드 전)엔 drawImage 가 빈 화면을 그린다 —
    // 로딩 중이면 직전 내용을 유지해 검은 화면을 막는다.
    if(!img.complete||!img.naturalWidth) return;
    mx.imageSmoothingEnabled=false; mx.clearRect(0,0,180,180);
    mx.drawImage(img,p[0]-R,p[1]-R,2*R,2*R,0,0,180,180);
    mx.strokeStyle='#0f0'; mx.beginPath(); mx.moveTo(90,0); mx.lineTo(90,180);
    mx.moveTo(0,90); mx.lineTo(180,90); mx.stroke();});
  img.addEventListener('mouseleave',()=>{mag.style.display='none';});});

['dx','dy','scale','rot'].forEach(k=>{const el=document.getElementById('n_'+k);
  el.addEventListener('input',()=>{document.getElementById('v_'+k).textContent=el.value;
    post('/api/nudge',{dx:+n_dx.value,dy:+n_dy.value,scale:+n_scale.value,rot:+n_rot.value});});});
['exposure','gain','white_balance'].forEach(k=>{const el=document.getElementById('p_'+k);
  el.addEventListener('input',()=>{document.getElementById('v_'+k).textContent=el.value;});});
function applyPhoto(){post('/api/photo',{exposure:+p_exposure.value,gain:+p_gain.value,
  white_balance:+p_white_balance.value,
  power_line_frequency:+p_power_line_frequency.value});}

function forceRefresh(){const t=Date.now();
  document.getElementById('itop').src='/img/top.jpg?t='+t;
  document.getElementById('inear').src='/img/near.jpg?t='+t;
  refreshPreviews();}
function refreshPreviews(){const t=Date.now(), v=document.getElementById('view').value;
  document.getElementById('seam').src='/img/seam.jpg?view='+v+'&t='+t;
  document.getElementById('stitch').src='/img/stitch.jpg?t='+t;}
function refresh(){
  // freeze 중엔 클릭 대상(top/near)은 리로드하지 않는다 — 확대경 블랙아웃 방지.
  // seam/stitch 미리보기는 적합·nudge 반영을 위해 계속 갱신 (프레임은 서버에서 래치됨).
  if(S&&S.frozen){refreshPreviews(); return;}
  forceRefresh();}
setInterval(refresh,700);
setInterval(()=>fetch('/api/state').then(r=>r.json()).then(apply),1000);
fetch('/api/state').then(r=>r.json()).then(apply);
</script></body></html>"""


def make_handler(state):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj):
            self._send(200, "application/json", json.dumps(obj).encode())

        def _snapshot(self, extra=None):
            with state.lock:
                top, near = state.src.get()
                state.ensure_res(top)
                band = state.band(top, near)
                A = state.effective_A()
                res, rms = residuals(A, state.points)
                out = {
                    "source": state.src.info(),
                    "msg": state.msg,
                    "points": state.points,
                    "res": res, "rms": rms,
                    "decomp": decompose(A),
                    "edges": edge_check(top, near, A, band),
                    "photo_stats": photo_stats(top, near, band),
                    "save_path": str(fl.stitch_calib_path(state.mast)
                                     .relative_to(REPO_ROOT)),
                    "band": list(band),
                    "frozen": state.src.frozen is not None,
                }
                if extra:
                    out.update(extra)
                return out

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                return self._send(200, "text/html; charset=utf-8", PAGE.encode())
            if self.path.startswith("/api/state"):
                return self._json(self._snapshot())
            if self.path.startswith("/img/"):
                with state.lock:
                    top, near = state.src.get()
                    state.ensure_res(top)
                    which = self.path.split("/")[-1].split(".")[0].split("?")[0]
                    if which == "top":
                        img = top
                    elif which == "near":
                        img = near
                    elif which == "stitch":
                        img = render_stitch(state, top, near)
                    else:
                        view = "edges"
                        if "view=" in self.path:
                            view = self.path.split("view=")[1].split("&")[0]
                        img = render_seam(state, top, near, view)
                if img is None:
                    img = np.zeros((240, 320, 3), np.uint8)
                return self._send(200, "image/jpeg", jpg(img))
            return self._send(404, "text/plain", b"nope")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            path = self.path
            extra = None
            with state.lock:
                state.ensure_res(state.src.get()[0])
                if path.endswith("/points"):
                    op = body.get("op")
                    if op == "add":
                        state.points.append({"top": body["top"], "near": body["near"],
                                             "src": "manual"})
                        state.msg = f"대응점 {len(state.points)}개"
                    elif op == "undo" and state.points:
                        state.points.pop()
                    elif op == "del":
                        i = int(body.get("i", -1))
                        if 0 <= i < len(state.points):
                            state.points.pop(i)
                    elif op == "clear":
                        state.points = []
                elif path.endswith("/fit"):
                    model = body.get("model", "similarity")
                    A, why = fit_A(state.points, model)
                    if A is None:
                        state.msg = f"적합 실패 — {why}"
                    else:
                        state.A = A
                        state.model = model
                        state.nudge = {"dx": 0.0, "dy": 0.0, "scale": 1.0, "rot": 0.0}
                        q = quality(A, state.points)
                        state.msg = (f"적합 완료 — {why}, 인라이어 RMS "
                                     f"{q['rms_in']:.2f}px ({q['n_in']}점)")
                        if q["n_out"]:
                            state.msg += (f" · 이상점 {q['n_out']}점 제외"
                                          f"(전체 RMS {q['rms_all']:.2f}px) — "
                                          f"잔차 큰 점은 지우고 다시 적합할 것")
                elif path.endswith("/auto"):
                    top, near = state.src.get()
                    pts, why = auto_match(top, near, state.effective_A(),
                                          state.band(top, near))
                    state.points.extend(pts)
                    state.msg = f"{why} — 잔차 큰 점은 지우고 적합하세요."
                elif path.endswith("/nudge"):
                    state.nudge = {k: float(body.get(k, state.nudge[k]))
                                   for k in state.nudge}
                elif path.endswith("/reset"):
                    state.A = state.base_A.copy()
                    state.nudge = {"dx": 0.0, "dy": 0.0, "scale": 1.0, "rot": 0.0}
                    state.msg = "기존(레거시) 파라미터로 되돌림"
                elif path.endswith("/step"):
                    state.src.step(int(body.get("d", 1)))
                elif path.endswith("/freeze"):
                    on = state.src.freeze(bool(body.get("on")))
                    state.msg = ("❄ FREEZE — 화면 고정. 클릭·분석 전부 이 프레임 기준"
                                 if on else "라이브 재개")
                elif path.endswith("/save"):
                    extra = {"msg": save_calib(state)}
                    state.msg = extra["msg"]
                elif path.endswith("/photo"):
                    vals = dict(body)
                    vals["enable_auto_exposure"] = False
                    vals["enable_auto_white_balance"] = False
                    extra = {"log": photo_apply(vals)}
                elif path.endswith("/photo_read"):
                    extra = {"photo_read": photo_read()}
            return self._json(self._snapshot(extra))

    return H


def save_calib(state) -> str:
    A = state.effective_A()
    q = quality(A, state.points)
    res, rms = q["res"], q["rms_all"]
    # left/right 는 항상 A 의 실커버리지에서 계산 — 구캘리브 크롭을 상속하지 않는다
    left, right = coverage_bounds(A, *state.res)
    st = fl.Stitcher(state.mast, w=state.res[0], h=state.res[1],
                     calib={"A": A.tolist(), "seam": state.seam,
                            "left": left, "right": right,
                            "res": list(state.res), "note": "tmp"})
    path = fl.stitch_calib_path(state.mast)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        bak = path.with_suffix(f".json.bak_{datetime.now():%Y%m%d_%H%M%S}")
        bak.write_text(path.read_text())
    data = {
        "version": 2,
        # 이 캘리브의 픽셀 좌표계 — 다른 해상도로 돌 때 Stitcher 가 재스케일 근거로 쓴다.
        # state 좌표계(A/seam/left/right 가 사는 곳)가 유일한 기준이다 — 프레임에서
        # 다시 읽으면 좌표계와 어긋난 res 가 저장될 수 있다.
        "res": list(state.res),
        "mast": state.mast,
        "model": state.model,
        "A": A.tolist(),
        "seam": state.seam,
        "left": left,
        "right": right,
        "out_h": st.out_h,
        "points": state.points,
        "residual_px": {"rms": rms, "per_point": res, "max": max(res) if res else None,
                        "rms_inlier": q["rms_in"], "n_inlier": q["n_in"],
                        "n_outlier": q["n_out"], "inlier_thresh_px": 3.0},
        "decomposition": decompose(A),
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": state.src.info(),
        "note": (f"stitch_calibrator {state.model} {len(state.points)}점 "
                 f"인라이어 RMS {q['rms_in']:.2f}px ({q['n_in']}점, 이상점 {q['n_out']}점)"),
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return (f"저장 완료 {path.relative_to(REPO_ROOT)} "
            f"(인라이어 RMS {q['rms_in']:.2f}px, {q['n_in']}/{len(state.points)}점)")


def report(state, limit=0) -> int:
    """브라우저 없이 정렬 상태만 진단한다 (여러 페어 집계).

    좌/중/우 값이 서로 다르면 배율·회전 오차가 남은 것 — 평행이동으로는 못 잡는다.
    """
    src = state.src
    n_all = len(getattr(src, "pairs", [1]))
    n = min(limit, n_all) if limit else n_all
    zones = {"좌": [], "중": [], "우": []}
    dY, dRG = [], []
    for i in range(n):
        src.idx = i if hasattr(src, "pairs") else 0
        top, near = src.get()
        if top is None or near is None:
            continue
        state.ensure_res(top)
        band = state.band(top, near)
        for z in edge_check(top, near, state.A, band):
            if z["dx"] is not None and z["conf"] and z["conf"] > 0.25:
                zones[z["zone"]].append((z["dx"], z["dy"]))
        s = photo_stats(top, near, band)
        if s.get("delta"):
            dY.append(s["delta"]["y"])
            dRG.append(s["delta"]["rg"])

    print(f"\n=== 스티치 정렬 진단  mast={state.mast} ===")
    print(f"소스: {src.info()}  (페어 {n}개, 위상상관 conf>0.25 만 집계)")
    print(f"현재 캘리브: {state.calib_note}")
    print(f"겹침 띠(top 프레임 행): {state.band(*src.get())}")
    print("\n  구간   n     dx 중앙      dy 중앙")
    med = {}
    for z, v in zones.items():
        if not v:
            print(f"  {z}   신뢰 가능한 샘플 없음 (무늬 없는 바닥일 수 있음)")
            continue
        a = np.array(v)
        med[z] = (float(np.median(a[:, 0])), float(np.median(a[:, 1])))
        print(f"  {z}  {len(v):4d}  {med[z][0]:+8.2f}px  {med[z][1]:+8.2f}px")
    bad = False
    if len(med) >= 2:
        spread_x = max(m[0] for m in med.values()) - min(m[0] for m in med.values())
        spread_y = max(m[1] for m in med.values()) - min(m[1] for m in med.values())
        rot = math.degrees(math.atan2(spread_y, float(state.res[0])))
        print(f"\n  좌↔우 편차: dx {spread_x:.2f}px, dy {spread_y:.2f}px "
              f"→ 회전 오차 환산 약 {rot:.2f}°")
        bad = spread_x > 3.0 or spread_y > 3.0
        print("  " + ("⚠ 평행이동만으로는 못 맞춘다 — similarity 이상으로 재캘리브 필요"
                      if bad else "✓ 좌우 편차 작음 — 평행이동 모델로 충분"))
    if dY:
        print(f"\n  이음매 밝기차 Y: 중앙 {np.median(dY):+.1f} "
              f"(범위 {min(dY):+.1f}~{max(dY):+.1f}), R/G 차 중앙 {np.median(dRG):+.3f}")
        if abs(float(np.median(dY))) > 6:
            print("  ⚠ 두 캠의 노출/화이트밸런스가 갈렸다 — auto 를 끄고 동일 고정값으로")
            bad = True
        else:
            print("  ✓ 측광 일치")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mast", choices=("up", "down", "mid"), default="up",
                    help="마스트 자세 — 캘리브는 자세마다 따로 잡아야 한다")
    ap.add_argument("--pair-dir", default="",
                    help="오프라인: top/near png 페어가 있는 디렉터리 "
                         "(예: logs/field_ops/<run>/scan)")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--report", action="store_true",
                    help="브라우저 없이 정렬/측광 진단만 출력하고 종료 "
                         "(어긋남이 크면 exit 1)")
    ap.add_argument("--limit", type=int, default=0,
                    help="--report 에서 볼 페어 수 (0=전부)")
    args = ap.parse_args()

    if args.pair_dir:
        src = OfflineSource(Path(args.pair_dir))
    else:
        try:
            src = LiveSource()
        except Exception as e:  # noqa: BLE001
            raise SystemExit(
                f"라이브 소스 실패({e}).\n"
                "카메라 노드를 먼저 띄우거나(arena-bringup), "
                "--pair-dir 로 저장된 페어를 쓰세요.")

    state = State(src, args.mast)
    if args.report:
        raise SystemExit(report(state, args.limit))
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"[stitch_calibrator] mast={args.mast}  {src.info()}")
    print(f"  현재 캘리브: {state.calib_note}")
    print(f"  저장 위치  : {fl.stitch_calib_path(args.mast).relative_to(REPO_ROOT)}")
    print(f"  →  http://<이 기기 IP>:{args.port}/   (Ctrl-C 종료)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n종료")


if __name__ == "__main__":
    main()
