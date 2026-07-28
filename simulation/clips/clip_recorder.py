#!/usr/bin/env python3
"""Multi-camera video recorder for Isaac Sim competition runs.

Records several simultaneous cinematic views of a match while the real mission
code (``mission/match_runner.py``) drives the simulated robot, and encodes each
view straight to MP4.

Design notes
------------
* **Frames are piped to ffmpeg, never written as PNG sequences.** A 20-minute
  run at 4 views x 1080p would be tens of gigabytes of intermediate PNGs. One
  ffmpeg subprocess per camera consumes raw frames on stdin and writes an MP4,
  so peak disk use is the finished video.

* **The mission code is not modified and not even aware of this.** Recording
  hangs off the simulation bridge's update loop. Stage boundaries are recovered
  afterwards from the runner's own timestamped stdout — see ``cut_clips.py``.
  A ``frames.jsonl`` sidecar records wall-clock time, sim time and robot pose
  per frame so the two can be aligned exactly.

* **Rigs are declarative.** Each rig gives its own resolution, field of view and
  a pose function evaluated every frame, so a chase camera is a few lines rather
  than a special case.

Requires Isaac Sim (``omni.replicator.core``) at runtime. The geometry helpers
and the ffmpeg pipe are importable and testable without it — run
``python3 clip_recorder.py --selftest``.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

# --------------------------------------------------------------------------
# Arena constants (metres, map frame — origin at arena centre)
# --------------------------------------------------------------------------
ARENA_HALF = 2.0          # the arena is 4 m x 4 m
TOPDOWN_MARGIN = 1.06     # 6 % breathing room so the walls are not flush
STAGING_XY = (-1.40, -1.40)   # storage staging point, matches match_runner
STORAGE_CORNER_YAW = math.radians(-135.0)


# --------------------------------------------------------------------------
# Camera maths (pure, no Isaac dependency)
# --------------------------------------------------------------------------
def fov_to_aperture(fov_deg: float, focal_length: float) -> float:
    """USD aperture that yields ``fov_deg`` at the given focal length."""
    return 2.0 * focal_length * math.tan(math.radians(fov_deg) * 0.5)


def aperture_to_fov(aperture: float, focal_length: float) -> float:
    return math.degrees(2.0 * math.atan(aperture / (2.0 * focal_length)))


def topdown_height(half_extent: float, vfov_deg: float) -> float:
    """Camera height that makes ``half_extent`` exactly fill half the frame."""
    return half_extent / math.tan(math.radians(vfov_deg) * 0.5)


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """Row-major 4x4 camera-to-world matrix, USD convention.

    USD cameras look down **-Z** with +Y up, so the basis is built as
    ``z = normalize(eye - target)``, ``x = normalize(up x z)``, ``y = z x x``.
    Returns a list of 4 rows of 4 floats, ready for ``Gf.Matrix4d(*rows)``.
    """
    ex, ey, ez = eye
    tx, ty, tz = target
    zx, zy, zz = ex - tx, ey - ty, ez - tz
    n = math.sqrt(zx * zx + zy * zy + zz * zz)
    if n < 1e-9:
        raise ValueError("look_at: eye and target coincide")
    zx, zy, zz = zx / n, zy / n, zz / n

    ux, uy, uz = up
    xx = uy * zz - uz * zy
    xy = uz * zx - ux * zz
    xz = ux * zy - uy * zx
    n = math.sqrt(xx * xx + xy * xy + xz * xz)
    if n < 1e-6:
        # Looking straight down (or up): `up` is parallel to the view axis, so
        # pick a stable fallback that keeps +Y of the image pointing at world +Y.
        ux, uy, uz = 0.0, 1.0, 0.0
        xx = uy * zz - uz * zy
        xy = uz * zx - ux * zz
        xz = ux * zy - uy * zx
        n = math.sqrt(xx * xx + xy * xy + xz * xz)
    xx, xy, xz = xx / n, xy / n, xz / n

    yx = zy * xz - zz * xy
    yy = zz * xx - zx * xz
    yz = zx * xy - zy * xx

    return [[xx, xy, xz, 0.0],
            [yx, yy, yz, 0.0],
            [zx, zy, zz, 0.0],
            [ex, ey, ez, 1.0]]


# --------------------------------------------------------------------------
# Rig definitions
# --------------------------------------------------------------------------
@dataclass
class Rig:
    """One camera view.

    ``pose_fn(pose) -> (eye, target)`` is evaluated every captured frame, where
    ``pose`` is the robot's ``(x, y, yaw)``. Static rigs ignore it.
    """
    name: str
    width: int
    height: int
    hfov_deg: float
    vfov_deg: float
    pose_fn: Callable[[tuple[float, float, float]], tuple[Sequence[float], Sequence[float]]]
    focal_length: float = 24.0
    caption: str = ""


def _chase(pose):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    # 1.8 m behind, 1.25 m up, aimed slightly ahead of the robot so the
    # destination is in frame rather than the robot filling it.
    eye = (x - 1.8 * c, y - 1.8 * s, 1.25)
    target = (x + 0.5 * c, y + 0.5 * s, 0.15)
    return eye, target


def _pov(pose):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    # Approximates the camera mast: slightly ahead of centre, mast height,
    # pitched down so the near ground and the far wall are both in frame.
    eye = (x + 0.16 * c, y + 0.16 * s, 0.62)
    target = (x + 1.30 * c, y + 1.30 * s, 0.02)
    return eye, target


def _topdown(_pose):
    h = topdown_height(ARENA_HALF * TOPDOWN_MARGIN, 47.2)
    return (0.0, 0.0, h), (0.0, 0.0, 0.0)


def _drift(_pose):
    # Low three-quarter angle outside the storage corner, framing the staging
    # point. Deliberately static: the mecanum yaw-while-translating reads much
    # more clearly against a fixed horizon than from a camera that follows.
    return (-3.55, -0.75, 0.80), (STAGING_XY[0] - 0.15, STAGING_XY[1] - 0.15, 0.18)


def default_rigs() -> list[Rig]:
    """The four views. Widths pair up: chase|pov and topdown|drift make 1920."""
    return [
        Rig("topdown", 1024, 1024, 47.2, 47.2, _topdown,
            caption="arena — the 42-point grid"),
        Rig("chase", 1110, 1080, 60.2, 58.9, _chase, focal_length=18.0,
            caption="chase"),
        # Robot POV: a single virtual camera with roughly double the vertical
        # field of view, standing in for the two physically stitched cameras.
        # Horizontal 68.8 deg matches the real D435; vertical 85 deg is the
        # doubled span. This is a visualisation, not the stitching pipeline.
        Rig("pov", 810, 1080, 68.8, 85.0, _pov,
            caption="robot view (simulated wide FOV — the real robot stitches two cameras)"),
        Rig("drift", 1920, 1080, 50.0, 29.9, _drift,
            caption="mecanum drift — translating and slewing to -135 deg at once"),
    ]


# --------------------------------------------------------------------------
# ffmpeg pipe
# --------------------------------------------------------------------------
def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:  # bundled binary, no system install and no sudo needed
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "no ffmpeg found — install it, or `pip install imageio-ffmpeg`") from exc


class VideoPipe:
    """One ffmpeg subprocess consuming raw RGB frames on stdin."""

    def __init__(self, path: Path, width: int, height: int, fps: int,
                 crf: int = 20, preset: str = "veryfast"):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            [find_ffmpeg(), "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
             "-an", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
             # yuv420p + even dimensions keeps the result playable everywhere
             "-pix_fmt", "yuv420p", str(path)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE)
        self.frames = 0

    def write(self, rgb) -> None:
        self.proc.stdin.write(rgb.tobytes())
        self.frames += 1

    def close(self) -> None:
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        self.proc.wait(timeout=120)


# --------------------------------------------------------------------------
# Recorder
# --------------------------------------------------------------------------
@dataclass
class ClipRecorder:
    """Attach to the sim bridge and record every rig at a fixed frame rate.

    Usage from inside the Isaac process::

        rec = ClipRecorder(out_dir=Path("logs/clips/run1"), fps=15)
        rec.setup(stage)
        ...
        rec.update(sim_time=..., wall_time=..., pose=(x, y, yaw), state="SCAN")
        ...
        rec.close()
    """
    out_dir: Path
    fps: int = 15
    rigs: list[Rig] = field(default_factory=default_rigs)
    root_prim: str = "/World/ClipCams"

    _pipes: dict = field(default_factory=dict, init=False)
    _annotators: dict = field(default_factory=dict, init=False)
    _cams: dict = field(default_factory=dict, init=False)
    _next_t: float = field(default=0.0, init=False)
    _index: int = field(default=0, init=False)
    _sidecar = None
    _stage = None

    # -- setup -------------------------------------------------------------
    def setup(self, stage) -> None:
        import omni.replicator.core as rep
        from pxr import Gf, UsdGeom

        self._stage = stage
        self.out_dir.mkdir(parents=True, exist_ok=True)

        for rig in self.rigs:
            path = f"{self.root_prim}/{rig.name}"
            cam = UsdGeom.Camera.Define(stage, path)
            cam.CreateFocalLengthAttr(rig.focal_length)
            cam.CreateHorizontalApertureAttr(
                fov_to_aperture(rig.hfov_deg, rig.focal_length))
            cam.CreateVerticalApertureAttr(
                fov_to_aperture(rig.vfov_deg, rig.focal_length))
            # Wide clipping range: the top-down sits ~4.9 m up, the POV ~0.1 m
            # from the nearest floor it can see.
            cam.CreateClippingRangeAttr(Gf.Vec2f(0.02, 60.0))
            self._cams[rig.name] = cam

            rp = rep.create.render_product(path, (rig.width, rig.height),
                                           name=f"clip_{rig.name}", force_new=True)
            ann = rep.AnnotatorRegistry.get_annotator("rgb")
            ann.attach([rp])
            self._annotators[rig.name] = ann
            self._pipes[rig.name] = VideoPipe(
                self.out_dir / f"{rig.name}.mp4", rig.width, rig.height, self.fps)

        self._sidecar = (self.out_dir / "frames.jsonl").open("w", encoding="utf-8")
        (self.out_dir / "rigs.json").write_text(json.dumps(
            [{"name": r.name, "width": r.width, "height": r.height,
              "hfov_deg": r.hfov_deg, "vfov_deg": r.vfov_deg,
              "caption": r.caption} for r in self.rigs],
            indent=1), encoding="utf-8")

    # -- per-tick ----------------------------------------------------------
    def _place(self, rig: Rig, pose) -> None:
        from pxr import Gf, UsdGeom
        eye, target = rig.pose_fn(pose)
        rows = look_at(eye, target)
        xf = UsdGeom.Xformable(self._cams[rig.name].GetPrim())
        ops = [op for op in xf.GetOrderedXformOps()
               if op.GetOpType() == UsdGeom.XformOp.TypeTransform]
        op = ops[0] if ops else xf.AddTransformOp()
        op.Set(Gf.Matrix4d(*[c for row in rows for c in row]))

    def update(self, sim_time: float, wall_time: float,
               pose: tuple[float, float, float], state: str = "") -> None:
        """Call once per simulation tick; throttles itself to ``fps``."""
        if self._sidecar is None:
            return
        if sim_time < self._next_t:
            # Still reposition, so the camera is already correct on the tick we
            # do capture — otherwise fast motion shows a one-frame lag.
            for rig in self.rigs:
                self._place(rig, pose)
            return
        self._next_t = sim_time + 1.0 / self.fps

        for rig in self.rigs:
            self._place(rig, pose)
            data = self._annotators[rig.name].get_data()
            if data is None or getattr(data, "size", 0) == 0:
                continue  # render product not ready yet; skip this frame
            self._pipes[rig.name].write(data[..., :3])

        self._sidecar.write(json.dumps({
            "i": self._index, "sim_time": round(sim_time, 4),
            "wall_time": round(wall_time, 4),
            "pose": [round(v, 4) for v in pose], "state": state}) + "\n")
        self._sidecar.flush()
        self._index += 1

    # -- teardown ----------------------------------------------------------
    def close(self) -> None:
        for name, pipe in self._pipes.items():
            try:
                pipe.close()
                print(f"[clips] {name}.mp4  {pipe.frames} frames "
                      f"({pipe.frames / max(self.fps, 1):.1f}s)")
            except Exception as exc:  # pragma: no cover
                print(f"[clips] {name}: {exc}")
        if self._sidecar:
            self._sidecar.close()
            self._sidecar = None


# --------------------------------------------------------------------------
# Self-test — everything that does not need Isaac
# --------------------------------------------------------------------------
def _selftest() -> int:
    import numpy as np
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        print(f"  {'ok  ' if cond else 'FAIL'} {label} {detail}")
        ok = ok and cond

    print("geometry")
    h = topdown_height(ARENA_HALF * TOPDOWN_MARGIN, 47.2)
    check("top-down height frames the arena", 4.5 < h < 5.2, f"h={h:.3f} m")
    span = 2.0 * h * math.tan(math.radians(47.2) * 0.5)
    check("visible span covers 4 m + margin", 4.15 < span < 4.35, f"{span:.3f} m")

    a = fov_to_aperture(85.0, 24.0)
    check("aperture round-trips", abs(aperture_to_fov(a, 24.0) - 85.0) < 1e-6)

    hf = aperture_to_fov(fov_to_aperture(85.0, 24.0) * 810 / 1080, 24.0)
    check("POV horizontal FOV matches the D435 (~69 deg)", 67.5 < hf < 70.0,
          f"{hf:.2f} deg")

    print("look_at")
    rows = look_at((0, 0, 5), (0, 0, 0))
    check("straight-down is well conditioned", all(
        all(math.isfinite(c) for c in row) for row in rows))
    check("translation row is the eye", rows[3][:3] == [0, 0, 5])
    zc = rows[2][:3]
    check("view axis points from target to eye", abs(zc[2] - 1.0) < 1e-9, f"z={zc}")

    eye, tgt = _chase((1.0, 0.0, 0.0))
    check("chase sits behind the robot", eye[0] < 1.0 and tgt[0] > 1.0,
          f"eye.x={eye[0]:.2f} target.x={tgt[0]:.2f}")

    print("ffmpeg pipe")
    try:
        exe = find_ffmpeg()
        check("ffmpeg located", bool(exe), exe.split("/")[-1])
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "probe.mp4"
            pipe = VideoPipe(out, 320, 240, 10)
            for i in range(10):
                frame = np.full((240, 320, 3), i * 25, dtype=np.uint8)
                pipe.write(frame)
            pipe.close()
            check("encoded a playable file", out.exists() and out.stat().st_size > 500,
                  f"{out.stat().st_size if out.exists() else 0} bytes")
    except Exception as exc:
        check("ffmpeg pipe", False, str(exc))

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print(__doc__)
    print("This module is imported by the Isaac scene script; "
          "run with --selftest to verify the parts that do not need Isaac.")
