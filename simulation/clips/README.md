# Match clip recording

Records cinematic video of a simulated match from several camera angles at once,
then cuts it into one short clip per stage with burned-in captions. Used to
produce driving-explainer clips. None are published with this repository.

The mission code is untouched and unaware of any of this.

## The four clips

| Clip | Covers | Cameras |
|---|---|---|
| `01_mast_up_and_center_scan` | mast raises → drive to arena centre → 42-cell scan | top-down, robot POV |
| `02_minigoal_approach` | target selection, street routing, drive to the cell | chase, top-down |
| `03_final_approach_and_grasp` | depth re-measurement, target re-verification, grasp | chase, robot POV |
| `04_return_to_storage` | carry to the bin, the −135° corner approach, deposit | drift cam, top-down |

## The camera rigs

| Rig | Resolution | FOV (H×V) | Behaviour |
|---|---|---|---|
| `topdown` | 1024×1024 | 47.2° × 47.2° | Static at 4.85 m. **Square on purpose** — a 4 m arena fills a square frame exactly, with 6 % margin. In 16:9 it would float in a letterbox. |
| `chase` | 1110×1080 | 60.2° × 58.9° | 1.8 m behind the robot, 1.25 m up, aimed 0.5 m ahead so the destination stays in frame. |
| `pov` | 810×1080 | 68.8° × 85.0° | Robot's-eye view from the mast. |
| `drift` | 1920×1080 | 50.0° × 29.9° | Static low three-quarter angle outside the storage corner. |

`chase` + `pov` side by side is exactly 1920×1080, as is `topdown` + `drift`.

### About the POV camera

It is **one virtual camera with roughly double the vertical field of view**, not
a reproduction of the real perception path. The real robot stitches two
RealSense D435 streams with a homography warp; simulating that would look worse
and prove nothing. The horizontal 68.8° is the genuine D435 figure and the
vertical 85° is the doubled span, so the framing is representative even though
the mechanism is not. Any published clip using this view should say so — the
caption in `cut_clips.py` already does.

## Usage

Record — from inside the Isaac process, alongside the existing bridge:

```python
from simulation.clips.clip_recorder import ClipRecorder

rec = ClipRecorder(out_dir=Path("logs/clips/run1"), fps=15)
rec.setup(stage)
# ... every tick:
rec.update(sim_time=sim_t, wall_time=time.time(), pose=(x, y, yaw), state=match_state)
# ... at the end:
rec.close()
```

15 fps is deliberate. Four render products at 1080p is the expensive part, and
these clips are explanatory rather than action footage.

Cut — after the run, against the runner's own report:

```bash
# always dry-run first: it prints the resolved windows without encoding
python3 simulation/clips/cut_clips.py \
    --run logs/clips/run1 \
    --report logs/field_ops/<run>/report.json --list

python3 simulation/clips/cut_clips.py \
    --run logs/clips/run1 --report logs/field_ops/<run>/report.json
```

## How alignment works

`report.json` contains the runner's own timestamped log and its per-stage and
per-cycle durations. `frames.jsonl`, written by the recorder, contains
wall-clock time, sim time and robot pose for every captured frame. Matching one
against the other yields an exact video timestamp for each stage boundary — no
markers, no instrumentation, no changes to the mission code.

If a resolved window falls outside the recorded footage the cutter says so
rather than silently emitting a half-second stub; that almost always means
`--run` and `--report` are from different runs.

Captions carry the **real robot's** measured duration for the stage, not the
simulator's playback time, so a viewer is never misled about how fast the robot
actually moved. That distinction matters here: the sim mover is
proportional-only and runs 2–3× slower than the robot (see
[`../README.md`](../README.md)), so playback time is not a robot timing and is
never captioned as one.

## Testing without Isaac

```bash
python3 simulation/clips/clip_recorder.py --selftest
```

Covers the camera geometry (top-down framing, aperture/FOV round-trip, the POV
horizontal FOV landing on the D435's 69°, look-at conditioning including the
straight-down degenerate case) and encodes a short probe video through the real
ffmpeg pipe.
