#!/usr/bin/env python3
"""Calibrate Isaac Sim arena lighting to the real arena low-light photometry.

The real arena's competition-evening light level was measured on 2026-07-20 and
frozen in data/calibration/photometry/arena_20260720_212531.json (floor-band
Y_top 107.63 / Y_near 113.82, p99 152/167, saturation 0%). This tool drives the
sim scene's runtime lighting hook until the sim cameras report the same band
statistics, so perception rehearsals in KINEMATIC sim see field-equivalent
brightness.

Prerequisite: the mecanum competition scene is running and publishing
/camera_19/rgb + /camera_54/rgb (scripts/dev/run_isaac_mecanum_scene.sh
--headless --drive-mode kinematic). This tool needs rclpy -> `source
/opt/ros/humble/setup.bash` first, then run with system python3, NOT Isaac's
python.sh (bare python3 without the ROS env has no rclpy).

  # sweep (key, fill) until targets hit, then save the preset
  python3 sim/isaacsim/scripts/lowlight_lighting_sweep.py --stamp 20260721_1200

  # after a sim reboot: verify current lighting only, no changes
  python3 sim/isaacsim/scripts/lowlight_lighting_sweep.py --measure-only

Lighting contract (create_mecanum_competition_scene.py _lighting_callback):
publish std_msgs/String JSON {"key": <intensity>, "fill": <intensity>,
"color": [r, g, b]} on /sim/lighting -> /Looks/KeyLight (DistantLight,
authored default 450.0) and /Looks/FillLight (DomeLight, authored default
180.0, color applies to fill only).

Search strategy (why "grid over fill ratio, then bisection on key"):
pre-tonemap the rendered band radiance is linear in each light's intensity,
Y_cam ~= a_cam*key + b_cam*fill, and the tonemap is monotone. With the ratio
fill/key fixed, band Y is therefore a monotone 1-D function of key ->
bisection converges. The ratio is what splits the two cameras: the -45 deg
DistantLight and the ambient DomeLight weight the far-floor band (top cam)
and the near-floor band (near cam) differently, so a coarse ratio grid first
picks the ratio whose measured Y_near/Y_top balance (key-invariant at fixed
ratio) best matches the real 113.82/107.63 ≈ 1.058, then bisection on key
sets the absolute level. If the sweep ends close on mean Y but off on dY,
rerun with a denser --fill-ratios.

Exit codes: 0 pass, 1 targets unreachable (best-effort preset written),
2 infra failure (no frames at all / frame stream stalled mid-sweep /
wrong camera resolution). A mid-sweep stall still writes the best-effort
preset+samples for inspection, but exits 2: restart the sim, don't retune.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_JSON_DEFAULT = REPO_ROOT / "perception" / "calibration" / "photometry" / "arena_20260720_212531.json"
PRESET_OUT_DEFAULT = REPO_ROOT / "perception" / "calibration" / "photometry" / "sim_lowlight_preset.json"
LOG_ROOT = REPO_ROOT / "logs" / "sim_validation"

# Authored scene defaults — keep in sync with create_mecanum_competition_scene.py
# ("/Looks/KeyLight" intensity 450.0, "/Looks/FillLight" intensity 180.0).
SCENE_KEY_DEFAULT = 450.0
SCENE_FILL_DEFAULT = 180.0

# Fallbacks if the real calibration json is missing (values copied from
# arena_20260720_212531.json "measured").
FALLBACK_Y_TOP = 107.63
FALLBACK_Y_NEAR = 113.82
FALLBACK_BAND_ROWS = (419, 480)


# ------------------------------------------------------------------ metrics
# Replicated verbatim from scripts/dev/field_ops/photometry_tune.py
# measure()/stats() (lines ~100-140): band mean Y = 0.299R+0.587G+0.114B of
# per-channel band means, sat = %% of pixels with per-pixel Y >= 250, p99 =
# 99th percentile of per-pixel Y. We deliberately do NOT import
# photometry_tune.measure: its band rows come from the live stitch homography
# (fl.Stitcher — currently (426, 480) with today's up.json), while the real
# targets were measured over the band recorded in the target json
# ((419, 480)). Pinning the band to the json keeps target and measurement
# commensurable, and keeps this sim tool independent of the field_ops
# calibration files.
def band_stats(img_rgb: np.ndarray, rows: tuple[int, int]) -> dict:
    a = img_rgb[rows[0]:rows[1]].astype(float)
    r, g, b = (a[:, :, i].mean() for i in range(3))
    y_map = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    return {"y": 0.299 * r + 0.587 * g + 0.114 * b,
            "sat": float((y_map >= 250).mean() * 100),
            "p99": float(np.percentile(y_map, 99))}


def measure_pair(top: np.ndarray, near: np.ndarray, band: tuple[int, int]) -> dict:
    """Band stats per camera. Same band composition as photometry_tune.measure:
    top rows [v0, h), near rows [0, h - v0)."""
    v0, h = band
    # Band rows are defined in a 480-tall frame; a taller image would silently
    # measure mid-frame instead of the floor band, so require exact height.
    if top.shape[0] != h or near.shape[0] != h:
        print(f"image height {top.shape[0]}/{near.shape[0]} != {h} — "
              "sim cameras must publish 640x480 like the real bridge",
              file=sys.stderr)
        sys.exit(2)  # infra/config failure, not a lighting one
    t = band_stats(top, (v0, h))
    n = band_stats(near, (0, h - v0))
    return {"y_top": t["y"], "y_near": n["y"], "dY": t["y"] - n["y"],
            "sat_top_pct": t["sat"], "sat_near_pct": n["sat"],
            "p99_top": t["p99"], "p99_near": n["p99"]}


def measure_avg(pairs: list[tuple[np.ndarray, np.ndarray]], band: tuple[int, int]) -> dict:
    """Average stats across N fresh frame pairs (render noise smoothing)."""
    ms = [measure_pair(t, n, band) for t, n in pairs]
    out = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
    out["frames"] = len(ms)
    return out


def verdict(m: dict, t: dict) -> list[str]:
    bad = []
    for cam, key, target in (("top", "y_top", t["y_top"]), ("near", "y_near", t["y_near"])):
        if abs(m[key] - target) > t["tol_y"]:
            bad.append(f"Y({cam}) {m[key]:.1f} — target {target:.2f}±{t['tol_y']}")
    for cam, key in (("top", "p99_top"), ("near", "p99_near")):
        if not (t["p99_lo"] <= m[key] <= t["p99_hi"]):
            bad.append(f"p99({cam}) {m[key]:.0f} — target [{t['p99_lo']:.0f}, {t['p99_hi']:.0f}]")
    sat = max(m["sat_top_pct"], m["sat_near_pct"])
    if sat > t["max_sat"]:
        bad.append(f"saturation {sat:.2f}% > {t['max_sat']}%")
    return bad


def score(m: dict, t: dict) -> float:
    """Closeness metric for best-effort selection. Y error dominates;
    saturation over the cap is disqualifying-heavy; p99 drift is a soft penalty."""
    e = max(abs(m["y_top"] - t["y_top"]), abs(m["y_near"] - t["y_near"]))
    sat = max(m["sat_top_pct"], m["sat_near_pct"])
    if sat > t["max_sat"]:
        e += 100.0 + sat
    for p in (m["p99_top"], m["p99_near"]):
        if p < t["p99_lo"]:
            e += (t["p99_lo"] - p) * 0.1
        elif p > t["p99_hi"]:
            e += (p - t["p99_hi"]) * 0.1
    return e


def fmt(m: dict) -> str:
    return (f"Y top {m['y_top']:5.1f} / near {m['y_near']:5.1f}  dY {m['dY']:+5.1f}  "
            f"sat {m['sat_top_pct']:.1f}/{m['sat_near_pct']:.1f}%  "
            f"p99 {m['p99_top']:.0f}/{m['p99_near']:.0f}")


# ------------------------------------------------------------------ ROS I/O
def decode_image(msg) -> np.ndarray | None:
    """sensor_msgs/Image -> RGB uint8 HxWx3. Handles bgr8 and rgb8, honors step."""
    if msg.encoding not in ("bgr8", "rgb8"):
        return None
    row = int(msg.step) if msg.step else msg.width * 3
    buf = np.frombuffer(msg.data, np.uint8)
    if buf.size < msg.height * row:
        return None
    img = buf[: msg.height * row].reshape(msg.height, row)[:, : msg.width * 3]
    img = img.reshape(msg.height, msg.width, 3)
    return img[:, :, ::-1].copy() if msg.encoding == "bgr8" else img.copy()


class SimPhotometry:
    """Persistent node: /sim/lighting publisher + fresh-frame grabber.

    "Fresh" = received after (publish time + settle). The sim render pipeline
    lags light edits by a few frames, so per sample we wait >= 1 s of settle
    AND count frames arriving only after that cutoff.
    """

    def __init__(self, args):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
        from std_msgs.msg import String

        self.rclpy = rclpy
        self.String = String
        rclpy.init()
        self.node = Node("lowlight_lighting_sweep")
        self._cutoff = 0.0
        self._fresh: dict[str, list] = {"top": [], "near": []}
        self.node.create_subscription(Image, args.top_topic, self._cb("top"),
                                      qos_profile_sensor_data)
        self.node.create_subscription(Image, args.near_topic, self._cb("near"),
                                      qos_profile_sensor_data)
        self.pub = self.node.create_publisher(String, args.lighting_topic, 10)

    def _cb(self, name):
        def _inner(msg):
            if time.monotonic() < self._cutoff:
                return
            img = decode_image(msg)
            if img is not None:
                self._fresh[name].append(img)
        return _inner

    def wait_subscriber(self, timeout=10.0) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.pub.get_subscription_count() > 0:
                return True
            self.rclpy.spin_once(self.node, timeout_sec=0.2)
        return False

    def set_lighting(self, key: float, fill: float, color: tuple[float, float, float]):
        msg = self.String()
        msg.data = json.dumps({"key": round(float(key), 2), "fill": round(float(fill), 2),
                               "color": [round(float(c), 4) for c in color]})
        self.pub.publish(msg)

    def grab_fresh(self, n: int, settle: float, timeout: float):
        """N fresh pairs after `settle`, or None on timeout."""
        self._fresh = {"top": [], "near": []}
        self._cutoff = time.monotonic() + settle
        deadline = self._cutoff + timeout
        while (time.monotonic() < deadline
               and (len(self._fresh["top"]) < n or len(self._fresh["near"]) < n)):
            self.rclpy.spin_once(self.node, timeout_sec=0.2)
        if len(self._fresh["top"]) < n or len(self._fresh["near"]) < n:
            return None
        return list(zip(self._fresh["top"][-n:], self._fresh["near"][-n:]))

    def shutdown(self):
        self.node.destroy_node()
        self.rclpy.shutdown()


# ------------------------------------------------------------------ artifacts
def rel(p: Path) -> str:
    """Repo-relative for display/preset fields; absolute if outside the repo
    (out-of-repo --real-json / --preset-out must not crash the final write)."""
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def write_samples(run_dir: Path, targets: dict, samples: list[dict]):
    """Rewritten after every sample so a crash loses nothing."""
    (run_dir / "samples.json").write_text(json.dumps(
        {"targets": targets, "samples": samples}, indent=2, ensure_ascii=False))
    lines = ["| # | phase | key | fill | Y_top | Y_near | dY | p99 t/n | sat t/n % | score | pass |",
             "|---|-------|-----|------|-------|--------|----|---------|-----------|-------|------|"]
    for s in samples:
        m = s["measured"]
        lines.append(
            f"| {s['idx']} | {s['phase']} | {s['key']:.0f} | {s['fill']:.0f} "
            f"| {m['y_top']:.1f} | {m['y_near']:.1f} | {m['dY']:+.1f} "
            f"| {m['p99_top']:.0f}/{m['p99_near']:.0f} "
            f"| {m['sat_top_pct']:.1f}/{m['sat_near_pct']:.1f} "
            f"| {s['score']:.2f} | {'✓' if s['pass'] else '✗'} |")
    hdr = (f"# lighting sweep — targets Y_top {targets['y_top']:.2f} / "
           f"Y_near {targets['y_near']:.2f} (±{targets['tol_y']}), "
           f"p99 [{targets['p99_lo']:.0f},{targets['p99_hi']:.0f}], "
           f"sat ≤ {targets['max_sat']}%, band rows {targets['band_rows']}\n")
    (run_dir / "samples.md").write_text(hdr + "\n" + "\n".join(lines) + "\n")


def write_preset(path: Path, stamp: str, label: str, targets: dict, best: dict,
                 measured: dict, passed: bool, bad: list[str], n_samples: int,
                 run_dir: Path, real_json: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "created": stamp,
        "label": label,
        "source_real_json": rel(real_json) if real_json.is_file() else None,
        "targets": targets,
        "scene_defaults": {"key": SCENE_KEY_DEFAULT, "fill": SCENE_FILL_DEFAULT},
        "applied": {"key": best["key"], "fill": best["fill"], "color": best["color"]},
        "measured": measured,
        "pass": passed,
        "verdict": bad or ["pass"],
        "iterations": n_samples,
        "samples_log": rel(run_dir),
    }, indent=2, ensure_ascii=False))


# ------------------------------------------------------------------ sweep
def run_sweep(args, sp: SimPhotometry, targets: dict, run_dir: Path) -> int:
    band = tuple(targets["band_rows"])
    color = tuple(float(c) for c in args.color.split(","))
    samples: list[dict] = []
    target_mean = (targets["y_top"] + targets["y_near"]) / 2.0
    # Frame stall (infra, exit 2) vs iteration budget exhausted (targets
    # unreachable, exit 1) — take_sample returns None for both, so the infra
    # case is flagged here for finish() to classify.
    infra = {"stalled": False}

    def take_sample(phase: str, key: float, fill: float) -> dict | None:
        if len(samples) >= args.max_iters:
            return None
        sp.set_lighting(key, fill, color)
        pairs = sp.grab_fresh(args.frames, args.settle, args.frame_timeout)
        if pairs is None:
            infra["stalled"] = True  # frame stream died mid-sweep
            return None
        m = measure_avg(pairs, band)
        s = {"idx": len(samples) + 1, "phase": phase, "key": float(key),
             "fill": float(fill), "color": list(color), "measured": m,
             "score": score(m, targets), "pass": not verdict(m, targets)}
        samples.append(s)
        write_samples(run_dir, targets, samples)
        print(f"  [{s['idx']:2d}] {phase:9s} key {key:6.1f} fill {fill:6.1f}  "
              f"{fmt(m)}  score {s['score']:.2f}{'  ✓' if s['pass'] else ''}")
        return s

    def mean_y(s):
        return (s["measured"]["y_top"] + s["measured"]["y_near"]) / 2.0

    def finish() -> int:
        if not samples:
            print("no samples collected — sim not rendering? (exit 2)")
            return 2
        # Prefer passing samples: a failing sample can out-score a passing one
        # (soft p99 penalty vs a legitimate ~tol Y error), and the winner is
        # what gets applied to the scene.
        passing = [s for s in samples if s["pass"]]
        best = min(passing or samples, key=lambda s: s["score"])
        bad = verdict(best["measured"], targets)
        measured = best["measured"]
        # Re-apply the winner if a later sample was the last one published,
        # and take one verification measurement at the final setting.
        last = samples[-1]
        if (best["key"], best["fill"]) != (last["key"], last["fill"]):
            sp.set_lighting(best["key"], best["fill"], color)
            pairs = sp.grab_fresh(args.frames, args.settle, args.frame_timeout)
            if pairs is not None:
                infra["stalled"] = False  # stream came back — not a dead pipeline
                measured = measure_avg(pairs, band)
                bad = verdict(measured, targets)
        passed = not bad
        write_preset(Path(args.preset_out), args.stamp, args.label, targets, best,
                     measured, passed, bad, len(samples), run_dir, Path(args.real_json))
        print(f"\n[final] key {best['key']:.1f} fill {best['fill']:.1f}  {fmt(measured)}")
        print(("  ✗ " + "\n  ✗ ".join(bad)) if bad else "  ✓ pass")
        print(f"preset: {rel(Path(args.preset_out))}")
        print(f"samples: {rel(run_dir)}/samples.md")
        if infra["stalled"]:
            print("frame stream stalled mid-sweep — preset above is best-effort; "
                  "restart the sim before trusting it (exit 2)")
            return 2
        return 0 if passed else 1

    if not sp.wait_subscriber():
        print(f"⚠ no subscriber on {args.lighting_topic} — is the sim scene up? "
              "Proceeding anyway (late discovery is possible).")

    # -- stage 1: coarse grid (ratio picks the top/near balance, see module doc)
    grid_keys = [float(x) for x in args.grid_keys.split(",")]
    ratios = [float(x) for x in args.fill_ratios.split(",")]
    print(f"[grid] keys {grid_keys} × fill ratios {ratios} "
          f"(targets Y {targets['y_top']:.1f}/{targets['y_near']:.1f} ±{targets['tol_y']})")
    grid_by_ratio: dict[float, list[dict]] = {r: [] for r in ratios}
    for r in ratios:
        for k in grid_keys:
            s = take_sample("grid", k, k * r)
            if s is None:
                return finish()  # zero-sample / stall / budget: finish() classifies
            if s["pass"]:
                return finish()
            grid_by_ratio[r].append(s)

    # Ratio selection must be brightness-invariant: absolute level is fixed
    # later by bisection, but the ratio is frozen here. Under the linear model
    # Y_near/Y_top is independent of key at fixed fill ratio, so pick the
    # ratio whose measured near/top balance best matches the real one
    # (113.82/107.63 ≈ 1.058). Samples outside mean Y 40..220 are excluded —
    # the tonemap's crushed/clipped ends distort the balance.
    target_ratio = targets["y_near"] / max(targets["y_top"], 1e-6)

    def near_top(s):
        return s["measured"]["y_near"] / max(s["measured"]["y_top"], 1e-6)

    def balance_err(r):
        usable = [s for s in grid_by_ratio[r] if 40.0 <= mean_y(s) <= 220.0]
        pool = usable or grid_by_ratio[r]
        return abs(float(np.mean([near_top(s) for s in pool])) - target_ratio)

    ratio = min(ratios, key=balance_err)
    at_ratio = sorted(grid_by_ratio[ratio], key=lambda s: s["key"])
    print(f"[bisect] fill ratio fixed at {ratio:.3f} (near/top balance err "
          f"{balance_err(ratio):.3f} vs target {target_ratio:.3f}), "
          f"target mean Y {target_mean:.1f}")

    # -- stage 2: bracket the target mean on key (mean Y monotone in key)
    lo = max((s for s in at_ratio if mean_y(s) <= target_mean),
             key=lambda s: s["key"], default=None)
    hi = min((s for s in at_ratio if mean_y(s) > target_mean),
             key=lambda s: s["key"], default=None)
    for _ in range(4):  # extend bracket if the grid didn't straddle the target
        if lo is not None and hi is not None:
            break
        k = (min(s["key"] for s in at_ratio) / 2.0 if lo is None
             else max(s["key"] for s in at_ratio) * 2.0)
        s = take_sample("bracket", k, k * ratio)
        if s is None:
            return finish()
        if s["pass"]:
            return finish()
        at_ratio.append(s)
        lo = max((x for x in at_ratio if mean_y(x) <= target_mean),
                 key=lambda x: x["key"], default=None)
        hi = min((x for x in at_ratio if mean_y(x) > target_mean),
                 key=lambda x: x["key"], default=None)
    if lo is None or hi is None:
        print("  ⚠ could not bracket target mean Y — reporting closest sample")
        return finish()

    # -- stage 3: bisection on key
    k_lo, k_hi = lo["key"], hi["key"]
    while len(samples) < args.max_iters and (k_hi - k_lo) > args.key_resolution:
        k = (k_lo + k_hi) / 2.0
        s = take_sample("bisect", k, k * ratio)
        if s is None:
            return finish()
        if s["pass"]:
            return finish()
        if mean_y(s) <= target_mean:
            k_lo = k
        else:
            k_hi = k
    return finish()


def run_measure_only(args, sp: SimPhotometry, targets: dict, run_dir: Path) -> int:
    band = tuple(targets["band_rows"])
    pairs = sp.grab_fresh(args.frames, args.settle, args.frame_timeout)
    if pairs is None:
        print(f"frame timeout on {args.top_topic} / {args.near_topic} — sim bridge up? (exit 2)")
        return 2
    m = measure_avg(pairs, band)
    bad = verdict(m, targets)
    print(f"[measure-only] {fmt(m)}")
    print(("  ✗ " + "\n  ✗ ".join(bad)) if bad else "  ✓ pass")
    (run_dir / "measure_only.json").write_text(json.dumps(
        {"created": args.stamp, "targets": targets, "measured": m,
         "pass": not bad, "verdict": bad or ["pass"]}, indent=2, ensure_ascii=False))
    print(f"log: {rel(run_dir)}/measure_only.json")
    return 0 if not bad else 1


# ------------------------------------------------------------------ main
def load_targets(args) -> dict:
    y_top, y_near, band = FALLBACK_Y_TOP, FALLBACK_Y_NEAR, FALLBACK_BAND_ROWS
    real_json = Path(args.real_json)
    if real_json.is_file():
        meas = json.loads(real_json.read_text()).get("measured", {})
        y_top = float(meas.get("y_top", y_top))
        y_near = float(meas.get("y_near", y_near))
        band = tuple(int(v) for v in meas.get("band_rows", band))
    else:
        print(f"⚠ real calibration json missing ({real_json}) — using fallback "
              f"constants Y {y_top}/{y_near}, band {band}")
    if args.target_y_top is not None:
        y_top = args.target_y_top
    if args.target_y_near is not None:
        y_near = args.target_y_near
    return {"y_top": y_top, "y_near": y_near, "tol_y": args.tol_y,
            "p99_lo": args.p99_lo, "p99_hi": args.p99_hi,
            "max_sat": args.max_sat, "band_rows": list(band)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--measure-only", action="store_true",
                    help="no sweep: report current stats vs targets (post-boot check)")
    ap.add_argument("--real-json", default=str(REAL_JSON_DEFAULT),
                    help="real arena photometry json providing targets + band rows")
    ap.add_argument("--target-y-top", type=float, default=None,
                    help="override target Y (top cam band); default from --real-json")
    ap.add_argument("--target-y-near", type=float, default=None,
                    help="override target Y (near cam band); default from --real-json")
    ap.add_argument("--tol-y", type=float, default=3.0, help="per-camera |ΔY| tolerance")
    ap.add_argument("--p99-lo", type=float, default=140.0)
    ap.add_argument("--p99-hi", type=float, default=180.0)
    ap.add_argument("--max-sat", type=float, default=0.0,
                    help="band saturation cap %% (real arena measured 0%%)")
    ap.add_argument("--top-topic", default="/camera_19/rgb")
    ap.add_argument("--near-topic", default="/camera_54/rgb")
    ap.add_argument("--lighting-topic", default="/sim/lighting")
    ap.add_argument("--grid-keys", default="60,120,240,450",
                    help="coarse-grid KeyLight intensities (450 = authored default)")
    ap.add_argument("--fill-ratios", default="0.2,0.4,0.7",
                    help="coarse-grid fill/key ratios (0.4 = authored default 180/450)")
    ap.add_argument("--color", default="1,1,1", help="FillLight color r,g,b")
    ap.add_argument("--key-resolution", type=float, default=4.0,
                    help="stop bisection when the key bracket is this narrow")
    ap.add_argument("--frames", type=int, default=3,
                    help="fresh frame pairs to average per sample")
    ap.add_argument("--settle", type=float, default=1.5,
                    help="render settle after a lighting change s (min 1.0 enforced)")
    ap.add_argument("--frame-timeout", type=float, default=15.0)
    ap.add_argument("--max-iters", type=int, default=25,
                    help="hard cap on measured samples (grid+bracket+bisect)")
    ap.add_argument("--label", default=None,
                    help="run label -> logs/sim_validation/lighting_sweep_<label>/ (default: stamp)")
    ap.add_argument("--stamp", default=None,
                    help="trusted timestamp string from the orchestrator "
                         "(sim hosts may lack clock trust); default: local now")
    ap.add_argument("--preset-out", default=str(PRESET_OUT_DEFAULT))
    args = ap.parse_args()

    args.settle = max(args.settle, 1.0)  # render latency floor (task contract)
    args.stamp = args.stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    args.label = args.label or args.stamp
    targets = load_targets(args)
    run_dir = LOG_ROOT / f"lighting_sweep_{args.label}"
    run_dir.mkdir(parents=True, exist_ok=True)

    sp = SimPhotometry(args)
    try:
        if args.measure_only:
            rc = run_measure_only(args, sp, targets, run_dir)
        else:
            rc = run_sweep(args, sp, targets, run_dir)
    finally:
        sp.shutdown()
    sys.exit(rc)


if __name__ == "__main__":
    main()
