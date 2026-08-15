#!/usr/bin/env python3
"""Sim replica of the 2026-07-18 real street-dataset capture (88번).

Drives the mecanum sim robot through multiple scan points, and at every point
does a stop-rotate-shoot sweep saving BOTH cameras' RGB + aligned depth plus
the FULL pose record per shot:
  - arena pose  = lidar wall_range localizer output (/arena_lightweight/status,
    the quantity under test) with its localization meta (reason/latency/...)
  - gt pose     = /sim/ground_truth_pose (only available in sim - this is the
    point of replicating the field capture here)
Ground-truth object placement is dumped from /sim/objects_state so the offline
grid-snap / class analysis (71 / 82 / 87) runs against known truth.

Motion uses /base/move_relative anchored on the ARENA pose (never GT), so the
drive chain exercises the same localization the real robot used.

Prereqs (3 terminals):
  1) bash scripts/dev/run_isaac_mecanum_scene.sh --headless --drive-mode kinematic \
         [--camera-z-offset 0.20]   # mast UP variant (assumed +20 cm)
  2) arena node against sim lidar (see RUNBOOK_sim_street_capture.md)
  3) this script (plain host python3 with ROS humble sourced)

Output dir: logs/sim_validation/street_dataset_sim_<stamp>[_<label>]/
  shot_<i>_top.png / _near.png          (RGB, 71-compatible names)
  shot_<i>_top_rgb.png / _near_rgb.png  (hardlink aliases, 78/81-compatible)
  shot_<i>_top_depth.png / _near_depth.png (uint16 mm)
  top_K.txt / near_K.txt, report.json, placement_gt.json, summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import rclpy
from PIL import Image as PILImage
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_BASE = REPO_ROOT / "logs" / "sim_validation"

# Default scan points: the real capture's 3x3 quadrant/center pattern.
DEFAULT_POINTS = "0.75,0.75;0,0.75;-0.75,0.75;-0.75,0;0,0;0.75,0;0.75,-0.75;0,-0.75;-0.75,-0.75"

FRUITS = {"apple", "banana", "orange", "pineapple"}


def decode_rgb(msg) -> np.ndarray:
    ch = 4 if msg.encoding == "rgba8" else 3
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, ch)[:, :, :3]
    return arr[:, :, ::-1].copy() if msg.encoding == "bgr8" else arr.copy()


def decode_depth_mm(msg) -> np.ndarray | None:
    if msg.encoding == "16UC1":
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).copy()
    if msg.encoding == "32FC1":
        d = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(d * 1000.0, 0, 65535).astype(np.uint16)
    return None


def save_depth_png(path: Path, depth_mm: np.ndarray) -> None:
    PILImage.fromarray(depth_mm, mode="I;16").save(path)


class SimStreetCapture:
    CAMS = {"top": "camera_19", "near": "camera_54"}

    def __init__(self):
        self.node = rclpy.create_node("street_dataset_sim_capture")
        self.status = None
        self.gt = None
        self.gt_seq = 0
        self.objects = None
        self.move_result = None
        self.frames: dict[str, Image | None] = {}
        self.frame_seq: dict[str, int] = {}
        self.k: dict[str, list[float]] = {}
        for role, cam in self.CAMS.items():
            for kind in ("rgb", "depth"):
                key = f"{role}_{kind}"
                self.node.create_subscription(
                    Image, f"/{cam}/{kind}", self._frame_cb(key), qos_profile_sensor_data)
            self.node.create_subscription(
                CameraInfo, f"/{cam}/camera_info",
                lambda m, r=role: self.k.setdefault(r, list(m.k)), 10)
        self.node.create_subscription(
            String, "/arena_lightweight/status",
            lambda m: setattr(self, "status", json.loads(m.data)), 10)
        self.node.create_subscription(String, "/sim/ground_truth_pose", self._on_gt, 10)
        self.node.create_subscription(
            String, "/sim/objects_state",
            lambda m: setattr(self, "objects", json.loads(m.data)), 10)
        self.node.create_subscription(
            String, "/base/move_result",
            lambda m: setattr(self, "move_result", json.loads(m.data)), 10)
        self.move_pub = self.node.create_publisher(String, "/base/move_relative", 10)
        self.pose_pub = self.node.create_publisher(String, "/arena_lightweight/pose", 10)

    def _frame_cb(self, key):
        def cb(msg):
            self.frames[key] = msg
            self.frame_seq[key] = self.frame_seq.get(key, 0) + 1
        return cb

    def _on_gt(self, msg):
        self.gt = json.loads(msg.data)
        self.gt_seq += 1
        self.gt_last_wall = time.monotonic()

    def scene_alive(self, stale_after=8.0) -> bool:
        """The sim publishes GT continuously; a stalled GT stream means the
        scene process died (frames/poses would be stale garbage from here)."""
        self.spin_for(0.3)
        return (time.monotonic() - getattr(self, "gt_last_wall", 0.0)) < stale_after

    def spin_for(self, sec: float):
        end = time.monotonic() + sec
        while time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.05)

    # -- pose accessors -------------------------------------------------
    def arena_pose(self):
        if not self.status or not self.status.get("pose"):
            return None
        p = self.status["pose"]
        return float(p["x"]), float(p["y"]), float(p["yaw"])

    def gt_pose(self):
        if self.gt is None:
            return None
        return float(self.gt["x"]), float(self.gt["y"]), float(self.gt["yaw"])

    def localization_meta(self):
        if not self.status:
            return None
        return self.status.get("localization")

    # -- motion (arena-pose anchored) -----------------------------------
    def move_relative(self, dx, dy, dyaw=0.0, max_v=None, timeout=40.0):
        self.move_result = None
        payload = {"dx": float(dx), "dy": float(dy), "dyaw": float(dyaw)}
        if max_v:
            payload["max_v"] = float(max_v)
        m = String()
        m.data = json.dumps(payload)
        self.move_pub.publish(m)
        deadline = time.monotonic() + timeout
        while self.move_result is None and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
        return self.move_result or {"ok": False, "error": "timeout"}

    def face_plus_y(self):
        """Rotate to the strategy's fixed heading yaw=+90 (facing +y_world)."""
        pose = self.arena_pose()
        if pose is None:
            return
        err = math.atan2(math.sin(math.pi / 2 - pose[2]), math.cos(math.pi / 2 - pose[2]))
        if abs(err) > 0.03:
            self.move_relative(0.0, 0.0, err)

    def goto(self, gx, gy, tol=0.04, max_iter=6, max_v=0.5):
        """Iterative arena-pose-feedback goto, yaw held at +90 like the
        street strategy (fwd=+y_world, left=-x_world)."""
        self.face_plus_y()
        for _ in range(max_iter):
            self.spin_for(0.3)
            pose = self.arena_pose()
            if pose is None:
                return False
            dxw, dyw = gx - pose[0], gy - pose[1]
            if math.hypot(dxw, dyw) < tol:
                return True
            self.move_relative(dyw, -dxw, max_v=max_v)
        return math.hypot(gx - self.arena_pose()[0], gy - self.arena_pose()[1]) < 2 * tol

    # -- capture --------------------------------------------------------
    def fresh_frames(self, timeout=8.0):
        want = [f"{r}_{k}" for r in self.CAMS for k in ("rgb", "depth")]
        s0 = {w: self.frame_seq.get(w, 0) for w in want}
        deadline = time.monotonic() + timeout
        while (any(self.frame_seq.get(w, 0) < s0[w] + 2 for w in want)
               and time.monotonic() < deadline):
            rclpy.spin_once(self.node, timeout_sec=0.05)
        return {w: self.frames.get(w) for w in want}

    def capture(self, out_dir: Path, idx: int) -> list[str]:
        saved = []
        frames = self.fresh_frames()
        for role in self.CAMS:
            rgb_msg = frames.get(f"{role}_rgb")
            if rgb_msg is not None:
                main = out_dir / f"shot_{idx}_{role}.png"  # 71-compatible name
                PILImage.fromarray(decode_rgb(rgb_msg)).save(main)
                alias = out_dir / f"shot_{idx}_{role}_rgb.png"  # 78/81-compatible
                if not alias.exists():
                    alias.hardlink_to(main)
                saved.append(f"{role}_rgb")
            depth_msg = frames.get(f"{role}_depth")
            if depth_msg is not None:
                d = decode_depth_mm(depth_msg)
                if d is not None:
                    save_depth_png(out_dir / f"shot_{idx}_{role}_depth.png", d)
                    saved.append(f"{role}_depth")
        return saved

    # -- ground truth placement -----------------------------------------
    def placement_gt(self) -> dict:
        objs = []
        truth_parts = []
        for o in (self.objects or {}).get("objects", []):
            x, y = float(o["x"]), float(o["y"])
            gx_cm = min(range(50, 351, 50), key=lambda g: abs(g - (x + 2.0) * 100))
            gy_cm = min(range(100, 351, 50), key=lambda g: abs(g - (y + 2.0) * 100))
            prim = o.get("prim", "")
            parts = prim.split("/")[-1].split("_")
            cls = parts[-2] if parts[-1].isdigit() else parts[-1]
            objs.append({"prim": prim, "class": cls, "map_xy": [round(x, 3), round(y, 3)],
                         "official_cm": [gx_cm, gy_cm],
                         "snap_err_m": round(math.hypot(x - (gx_cm / 100 - 2.0),
                                                        y - (gy_cm / 100 - 2.0)), 3)})
            truth_cls = "fruit" if cls in FRUITS else cls
            truth_parts.append(f"{truth_cls}:{gx_cm},{gy_cm}")
        return {"objects": objs, "truth_arg": ";".join(truth_parts)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--points", default=DEFAULT_POINTS,
                    help='scan points "x,y;x,y;..." in map frame')
    ap.add_argument("--shots-per-point", type=int, default=8)
    ap.add_argument("--continuous-frames", type=int, default=0,
                    help="if >0, also capture N frames during one continuous "
                         "rotation at each point (blur/latency study)")
    ap.add_argument("--spin-rad-s", type=float, default=0.8)
    ap.add_argument("--label", default="", help="suffix for the output dir, e.g. mastdown")
    ap.add_argument("--seed-pose", default="gt",
                    help='arena pose seed: "gt" (default, seed from sim ground '
                         'truth), "x,y,yaw", or "none"')
    ap.add_argument("--max-v", type=float, default=0.5)
    args = ap.parse_args()

    points = []
    for tok in args.points.split(";"):
        tok = tok.strip()
        if tok:
            x, y = tok.split(",")
            points.append((float(x), float(y)))

    rclpy.init()
    cap = SimStreetCapture()
    print("== preflight: topics ==", flush=True)
    cap.spin_for(3.0)
    missing = []
    if cap.gt is None:
        missing.append("/sim/ground_truth_pose (scene not running?)")
    if cap.objects is None:
        missing.append("/sim/objects_state")
    seed = None
    if args.seed_pose == "gt" and cap.gt is not None:
        seed = cap.gt_pose()
    elif args.seed_pose not in ("", "none", "gt"):
        seed = tuple(float(v) for v in args.seed_pose.split(","))
    if seed is not None:
        m = String()
        m.data = json.dumps({"x": seed[0], "y": seed[1], "yaw": seed[2]})
        cap.pose_pub.publish(m)
        print(f"pose seeded: {tuple(round(v, 3) for v in seed)}", flush=True)
        cap.spin_for(1.5)
    if cap.status is None:
        missing.append("/arena_lightweight/status (arena node not running?)")
    deadline = time.monotonic() + 20.0
    while len(cap.k) < 2 and time.monotonic() < deadline:
        rclpy.spin_once(cap.node, timeout_sec=0.2)
    if len(cap.k) < 2:
        missing.append("camera_info x2 (scene launched with --no-cameras?)")
    frames = cap.fresh_frames(timeout=10.0)
    for w, f in frames.items():
        if f is None:
            missing.append(f"camera frame {w}")
    if missing:
        print(json.dumps({"ok": False, "missing": missing}, ensure_ascii=False))
        return 1

    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"street_dataset_sim_{stamp}" + (f"_{args.label}" if args.label else "")
    out_dir = OUT_BASE / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "top_K.txt").write_text(" ".join(str(v) for v in cap.k["top"]) + "\n")
    (out_dir / "near_K.txt").write_text(" ".join(str(v) for v in cap.k["near"]) + "\n")
    placement = cap.placement_gt()
    (out_dir / "placement_gt.json").write_text(
        json.dumps(placement, indent=2, ensure_ascii=False))
    print(f"out: {out_dir}  points={len(points)} shots/pt={args.shots_per_point}", flush=True)

    report = []
    pose_errs = []
    idx = 0

    def record(idx, pi, goal, phase, saved):
        pose = cap.arena_pose()
        gt = cap.gt_pose()
        err = None
        if pose and gt:
            dyaw = math.atan2(math.sin(pose[2] - gt[2]), math.cos(pose[2] - gt[2]))
            err = {"xy_m": round(math.hypot(pose[0] - gt[0], pose[1] - gt[1]), 4),
                   "yaw_rad": round(dyaw, 4)}
            pose_errs.append(err)
        entry = {"shot": idx, "point": pi, "point_goal": list(goal), "phase": phase,
                 "pose": list(pose) if pose else None,
                 "gt_pose": list(gt) if gt else None,
                 "pose_err": err,
                 "localization": cap.localization_meta(),
                 "time_wall": time.time(), "saved": saved}
        report.append(entry)
        return entry

    def abort_if_scene_dead():
        if not cap.scene_alive():
            (out_dir / "report.json").write_text(json.dumps(report, indent=2))
            (out_dir / "ABORTED").write_text("sim scene died (GT stream stalled)\n")
            print(json.dumps({"ok": False, "error": "scene died mid-capture",
                              "shots_done": idx, "out": str(out_dir)}))
            raise SystemExit(3)

    t0 = time.monotonic()
    for pi, (gx, gy) in enumerate(points):
        abort_if_scene_dead()
        ok = cap.goto(gx, gy, max_v=args.max_v)
        print(f"[point {pi}] goto ({gx:+.2f},{gy:+.2f}) ok={ok}", flush=True)
        cap.face_plus_y()
        step = 2.0 * math.pi / args.shots_per_point
        for k in range(args.shots_per_point):
            abort_if_scene_dead()
            if k:
                cap.move_relative(0.0, 0.0, step)
            cap.spin_for(0.5)  # settle + let localizer converge
            saved = cap.capture(out_dir, idx)
            e = record(idx, pi, (gx, gy), "discrete", saved)
            print(f"  shot {idx}: saved={len(saved)}/4 pose_err={e['pose_err']}", flush=True)
            idx += 1
        if args.continuous_frames > 0:
            # one continuous revolution via short move_relative arcs, grabbing
            # frames between arcs (sim has no direct cmd_vel spin helper here)
            arc = 2.0 * math.pi / args.continuous_frames
            for _ in range(args.continuous_frames):
                cap.move_relative(0.0, 0.0, arc, max_v=args.spin_rad_s)
                saved = cap.capture(out_dir, idx)
                record(idx, pi, (gx, gy), "continuous", saved)
                idx += 1
        cap.face_plus_y()

    xy = [e["xy_m"] for e in pose_errs]
    yaw = [abs(e["yaw_rad"]) for e in pose_errs]
    summary = {
        "shots": idx, "points": len(points), "duration_s": round(time.monotonic() - t0, 1),
        "pose_err_xy_m": {"mean": round(float(np.mean(xy)), 4),
                          "max": round(float(np.max(xy)), 4)} if xy else None,
        "pose_err_yaw_rad": {"mean": round(float(np.mean(yaw)), 4),
                             "max": round(float(np.max(yaw)), 4)} if yaw else None,
        "truth_arg": placement["truth_arg"],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"ok": True, "out": str(out_dir), **summary}, ensure_ascii=False),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
