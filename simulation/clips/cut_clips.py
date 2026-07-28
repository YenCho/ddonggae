#!/usr/bin/env python3
"""Cut recorded match footage into per-stage clips with burned-in subtitles.

Takes the multi-camera MP4s produced by ``clip_recorder.py`` plus the match
runner's own ``report.json``, and emits one short MP4 per stage of the run:

    01_mast_up_and_center_scan   mast raises, robot drives to centre, 28-cell scan
    02_minigoal_approach         route planning and the drive to the target cell
    03_final_approach_and_grasp  depth re-measure, target re-verify, grasp
    04_return_to_storage         carry, the -135 deg mecanum drift, deposit

Alignment
---------
``report.json`` carries the runner's own timestamped log (``[HH:MM:SS] ...``)
and per-stage durations; ``frames.jsonl`` carries wall-clock time per recorded
frame. Matching one against the other gives an exact video timestamp for every
stage boundary without touching the mission code.

If the log patterns do not match (they are Korean strings and may drift), the
cutter falls back to accumulating the durations in ``report["stages"]``. Run
with ``--list`` first to see what it resolved before spending encode time.

Usage
-----
    python3 cut_clips.py --run logs/clips/run1 --report logs/field_ops/<run>/report.json
    python3 cut_clips.py --run logs/clips/run1 --report ... --list
"""
from __future__ import annotations

import argparse
import bisect
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clip_recorder import find_ffmpeg  # noqa: E402

LOG_TS = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\]\s*(.*)$")

# Stage -> (title, subtitle). Subtitles are burned in; keep them short enough to
# read at a glance and honest about what the viewer is looking at.
CLIPS = [
    {
        "key": "01_mast_up_and_center_scan",
        "title": "1 — Mast up, centre scan",
        "subtitle": "Camera mast raises, the robot drives to the arena centre and scans all 42 grid cells",
        "stages": ["MAST_UP", "GOTO_CENTER", "SCAN"],
        "rigs": ["topdown", "pov"],
    },
    {
        "key": "02_minigoal_approach",
        "title": "2 — Mini-goal approach",
        "subtitle": "Target selected, route planned along the 25 cm streets between grid points",
        "phases": ["route"],
        "rigs": ["chase", "topdown"],
    },
    {
        "key": "03_final_approach_and_grasp",
        "title": "3 — Final approach and grasp",
        "subtitle": "Depth re-measurement, target re-verification, then the grasp",
        "phases": ["face", "approach", "grasp"],
        "rigs": ["chase", "pov"],
    },
    {
        "key": "04_return_to_storage",
        "title": "4 — Return to storage",
        "subtitle": "Carrying to the bin — the robot translates and slews to -135 deg simultaneously",
        "phases": ["carry"],
        "rigs": ["drift", "topdown"],
    },
]


def parse_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def log_timeline(report: dict) -> list[tuple[float, str]]:
    """[(seconds_since_midnight, message)] from the runner's own log."""
    out = []
    for line in report.get("log", []):
        m = LOG_TS.match(line)
        if m:
            h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
            out.append((h * 3600 + mi * 60 + s, m.group(4)))
    return out


def frame_index(run_dir: Path) -> tuple[list[float], float]:
    """([wall_time per frame], fps) from the recorder sidecar."""
    times = []
    for line in (run_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            times.append(json.loads(line)["wall_time"])
    if len(times) < 2:
        raise SystemExit("frames.jsonl has too few frames to align")
    fps = (len(times) - 1) / max(times[-1] - times[0], 1e-6)
    return times, fps


def wall_to_video_t(times: list[float], t_wall: float, fps: float) -> float:
    """Video timestamp (s) for a wall-clock time, clamped into range."""
    i = bisect.bisect_left(times, t_wall)
    i = max(0, min(i, len(times) - 1))
    return i / fps


def stage_spans(report: dict) -> dict[str, tuple[float, float]]:
    """Stage -> (start, end) in seconds relative to the run start.

    Uses the sequential durations in ``report["stages"]``, which is the one
    field guaranteed present regardless of log wording.
    """
    spans, t = {}, 0.0
    for st in report.get("stages", []):
        name, dur = st.get("stage", "?"), float(st.get("sec") or 0.0)
        spans[name] = (t, t + dur)
        t += dur
    return spans


def cycle_spans(report: dict) -> list[dict]:
    """Per-collection-cycle phase spans, relative to the start of COLLECT."""
    out, t = [], 0.0
    for cyc in report.get("cycles", []):
        phases, spans = cyc.get("phases", {}), {}
        for name in ("route", "face", "approach", "grasp", "carry"):
            d = float(phases.get(name) or 0.0)
            if d > 0:
                spans[name] = (t, t + d)
                t += d
        out.append({"index": cyc.get("index"), "identity": cyc.get("identity"),
                    "ok": cyc.get("ok"), "spans": spans})
        # cycles are contiguous; total_sec absorbs any unaccounted remainder
        total = float(cyc.get("total_sec") or 0.0)
        acc = sum(e - s for s, e in spans.values())
        if total > acc:
            t += total - acc
    return out


def resolve(report: dict, run_dir: Path):
    """Work out an absolute video-time window for every configured clip."""
    times, fps = frame_index(run_dir)
    t0_wall = times[0]

    # Anchor run-relative time to wall-clock using the first log timestamp.
    tl = log_timeline(report)
    run_start_wall = t0_wall
    if tl:
        # frames.jsonl wall_time is time.time(); the log is seconds-since-midnight.
        first_log_sod = tl[0][0]
        midnight = datetime.fromtimestamp(t0_wall).replace(
            hour=0, minute=0, second=0, microsecond=0).timestamp()
        run_start_wall = midnight + first_log_sod

    st = stage_spans(report)
    cyc = cycle_spans(report)
    collect_start = st.get("COLLECT", (0.0, 0.0))[0]

    resolved = []
    for spec in CLIPS:
        windows = []
        if "stages" in spec:
            picked = [st[n] for n in spec["stages"] if n in st]
            if picked:
                windows.append((min(s for s, _ in picked), max(e for _, e in picked)))
        if "phases" in spec:
            for c in cyc:
                picked = [c["spans"][p] for p in spec["phases"] if p in c["spans"]]
                if picked:
                    windows.append((collect_start + min(s for s, _ in picked),
                                    collect_start + max(e for _, e in picked)))
        for k, (rel_s, rel_e) in enumerate(windows):
            if rel_e - rel_s < 0.4:
                continue
            w_s, w_e = run_start_wall + rel_s, run_start_wall + rel_e
            v_s = wall_to_video_t(times, w_s, fps)
            v_e = wall_to_video_t(times, w_e, fps)
            # A window outside the recorded footage clamps to the last frame and
            # would silently produce a half-second stub. Flag it instead.
            covered = times[0] <= w_s and w_e <= times[-1]
            shrunk = (v_e - v_s) < 0.5 * (rel_e - rel_s) - 0.5
            resolved.append({
                "key": spec["key"] if len(windows) == 1 else f"{spec['key']}_{k + 1}",
                "title": spec["title"], "subtitle": spec["subtitle"],
                "rigs": spec["rigs"], "video_start": v_s, "video_end": v_e,
                "real_seconds": rel_e - rel_s,
                "warning": None if (covered and not shrunk) else (
                    "window falls outside the recorded footage — the recording "
                    "probably started late, stopped early, or belongs to a "
                    "different run than this report.json"),
            })

    span_report = max((c["video_end"] for c in resolved), default=0.0)
    if resolved and span_report >= (times[-1] - times[0]) - 1.0:
        print(f"WARNING: report spans past the end of the footage "
              f"({times[-1] - times[0]:.0f}s recorded). Check that --run and "
              f"--report are from the same run.\n", file=sys.stderr)
    return resolved, fps


def esc(text: str) -> str:
    """Escape for ffmpeg drawtext."""
    return (text.replace("\\", r"\\\\").replace(":", r"\:")
                .replace("'", r"\'").replace(",", r"\,").replace("%", r"\%"))


def cut(run_dir: Path, out_dir: Path, clip: dict, rig: str, dry: bool) -> None:
    src = run_dir / f"{rig}.mp4"
    if not src.exists():
        print(f"    skip {rig}: {src.name} not recorded")
        return
    dst = out_dir / f"{clip['key']}__{rig}.mp4"
    dur = max(clip["video_end"] - clip["video_start"], 0.5)

    title, sub = esc(clip["title"]), esc(clip["subtitle"])
    timing = esc(f"real robot: {clip['real_seconds']:.1f}s")
    # Bottom-anchored caption block with a translucent backing box, plus the
    # measured real-robot duration in the corner so the viewer knows the number
    # is the robot's and not the simulator's playback speed.
    vf = (
        f"drawtext=text='{title}':fontcolor=white:fontsize=h/22:x=(w-tw)/2:"
        f"y=h-th-h/14:box=1:boxcolor=black@0.55:boxborderw=14,"
        f"drawtext=text='{sub}':fontcolor=white@0.88:fontsize=h/38:x=(w-tw)/2:"
        f"y=h-th-h/30:box=1:boxcolor=black@0.45:boxborderw=10,"
        f"drawtext=text='{timing}':fontcolor=yellow:fontsize=h/40:"
        f"x=w-tw-h/40:y=h/40:box=1:boxcolor=black@0.5:boxborderw=8"
    )
    cmd = [find_ffmpeg(), "-y", "-loglevel", "error",
           "-ss", f"{clip['video_start']:.3f}", "-i", str(src), "-t", f"{dur:.3f}",
           "-vf", vf, "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-pix_fmt", "yuv420p", str(dst)]
    if dry:
        print(f"    would write {dst.name}  ({dur:.1f}s)")
        return
    subprocess.run(cmd, check=True)
    print(f"    {dst.name}  ({dur:.1f}s)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path,
                    help="recorder output directory (contains *.mp4 + frames.jsonl)")
    ap.add_argument("--report", required=True, type=Path,
                    help="report.json written by mission/match_runner.py")
    ap.add_argument("--out", type=Path, default=None, help="default: <run>/clips")
    ap.add_argument("--list", action="store_true",
                    help="resolve and print the windows without encoding")
    args = ap.parse_args()

    report = parse_report(args.report)
    clips, fps = resolve(report, args.run)
    out_dir = args.out or (args.run / "clips")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"recorded at {fps:.2f} fps; resolved {len(clips)} clip windows\n")
    for c in clips:
        print(f"  {c['key']}")
        print(f"    video {c['video_start']:.1f}s -> {c['video_end']:.1f}s "
              f"| real robot {c['real_seconds']:.1f}s | rigs: {', '.join(c['rigs'])}")
        for rig in c["rigs"]:
            cut(args.run, out_dir, c, rig, dry=args.list)
        print()

    if not args.list:
        (out_dir / "clips.json").write_text(json.dumps(clips, indent=1), encoding="utf-8")
        print(f"wrote {out_dir}/clips.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
