# Isaac Sim — kinematic surrogate

> **Policy.** This simulation is a **kinematic surrogate for perception and mission logic.**
> It is not a physics validator. Every claim this project makes about traction, roller slip,
> grasp force, motor current, PID behaviour or timing margin was measured on the real robot.
> The simulator's job is to let the *same* mission code, the *same* perception chain and the
> *same* ROS 2 topic contract run end-to-end against a known ground truth, thousands of times,
> without a charged battery.

The decision is recorded as [ADR-0004](../docs/adr/0004-simulation-is-kinematic-only.md).

---

## What we actually used it for

Mostly: **planning the match and spending the three minutes well.** The scoring
rules make this an optimisation problem before it is a robotics problem.

A perfect round is 7 objects — 4 from the first target set (40 points) and 3
from the second (60 points). Three minutes divided by seven is a **budget of
about 25 seconds per object**, scan and deposit included. And a wrong pickup is
penalised at *double* the object's value (−40 for a second-set mistake), so
"grab it if unsure" is negative expected value. Those two numbers drove almost
every behavioural decision in `mission/match_runner.py`.

The simulator is where those decisions were tried, because it runs the real
mission code against known ground truth as often as we like. Concretely, it is
where we worked out:

- **Where the time actually goes.** Every run writes a per-object phase
  breakdown — route / face / approach / grasp / carry — so a cycle that blew the
  25 s budget could be attributed rather than guessed at.
- **That rotation was the expensive thing.** Excess turning was collapsing the
  localiser, so cardinal snapping was replaced with a 4-axis
  minimum-residual alignment (±15°, 45° max) and short, final and highway legs
  stopped rotating at all.
- **That "settling" was costing whole seconds per move.** Relaxing the goal
  tolerance to 0.25 m, adding a 1.0 s stall early-exit, and skipping
  micro-rotations below 0.10 rad removed a deadband-parking → firmware-settling
  grind that had been eating ~8 s in a single reverse.
- **The route policy.** The arena is a 42-point grid on 50 cm spacing with 28
  points occupied; the gap between neighbours is 42 cm against a 40 cm robot, so
  cutting between columns is not survivable. The band below y = 100 cm is always
  empty, which makes it a free highway between the storage bin and the start
  zone. Approaching each column from below keeps the path behind the robot clear.
- **Target ordering and quotas** — first/second set quotas, fruit-first
  sequencing, and refusing non-target objects outright.

None of this needed physics. It needed the real state machine, a real map, real
detections and a clock — which is exactly what a kinematic surrogate provides,
at a rate no amount of battery-swapping on the real robot could match.

The honest caveat: **sim wall-clock timings are not the robot's timings.** Runs
come out roughly 2–3× slower. The *ratios and the ordering* transfer — which is
what the decisions above rested on — but the absolute seconds do not. Real
per-stage timings live in `docs/07-results-and-lessons.md`.

This is a limitation of the mover, not something inherent to kinematic
simulation, and it is worth fixing. Two causes, both in
`scripts/create_mecanum_competition_scene.py`:

- `move_speed_cap` is taken from the mission layer's `max_v`, but moves that
  don't specify one fall back to `MOVE_RELATIVE_MAX_LINEAR_MPS = 0.18` m/s,
  below the real firmware's 0.233 m/s default.
- More importantly, `_move_relative_twist()` is **pure proportional control**
  (`v = 5.0 × error`, clamped). Velocity decays to zero as the goal is
  approached, so every move ends in a long crawl to the 0.01 m tolerance. The
  real firmware runs a **trapezoidal profile** instead — accelerate at
  12.0 rad/s², cruise, decelerate — which covers the same distance far faster.

Porting the firmware's trapezoid into the sim mover, and taking the speed caps
from the robot's own configuration, is what would bring sim wall-clock into
agreement with the real robot without any post-hoc time scaling.

---

## What it is

`simulation/scripts/create_mecanum_competition_scene.py` composes the arena stage
(`simulation/usd/stadium_competition_grid42.usd`) with the mecanum robot URDF
(`hardware/cad/urdf/robot_mk3_mecanum_sim_80mm.urdf`) and then runs a ROS 2 bridge that
**mirrors the real robot's topic contract exactly** — same topic names, same JSON payload
vocabulary, same command semantics. It imports the robot's own
`robot_hardware.mecanum_control.MecanumKinematics`, so the sim exercises the identical
inverse/forward kinematics the firmware bridge runs.

The consequence that matters: the competition state machine
(`mission/match_runner.py`) drives the simulator and the real robot with **no code
branch and no sim flag**. If a run works in sim, the same bytes ran.

### The bridge contract

| Direction | Topic | Type / payload |
|---|---|---|
| sub | `/cmd_vel_direct` > `/cmd_vel` > `/cmd_vel_smoothed` > `/cmd_vel_nav` | `geometry_msgs/Twist`, 0.5 s staleness priority mux (same order as `cmd_vel_mux_node`) |
| sub | `/base/move_relative` | `std_msgs/String` JSON `{dx, dy, dyaw, max_v}` |
| sub | `/gripper/command`, `/lift/command` | `std_msgs/String` (`OPEN`/`CLOSE`/`SET_DEG`, `LIFT_TO_TOP`/`LIFT_TO_BOTTOM`/`LIFT_STATUS?`) |
| sub | `/safety/e_stop`, `/sim/lighting` | e-stop; runtime lighting hook `{key, fill, color}` |
| pub | `/odom/wheel` (+ `/chassis/odom`), `/joint_states`, `/motor/state` | odometry and wheel telemetry, `drive_type=mecanum` |
| pub | `/base/move_result` | JSON result for each relative move |
| pub | `/laser_scan` | re-stamped from the Isaac PhysX lidar |
| pub | `/imu/d435i` | synthetic BMI055 gyro/accel (noise density, bias, range from the datasheet) |
| pub | `/camera_19/{rgb,depth,camera_info}`, `/camera_54/{rgb,depth,camera_info}` | top and near cameras, **640×480** |
| pub | `/gripper/state`, `/gripper/grasp`, `/lift/state` | real bridge vocabulary, incl. `{"state": "held"\|"empty"\|"checking"\|"unknown"}` |
| pub | `/sim/ground_truth_pose`, `/sim/objects_state` | ground truth — sim only, never consumed by mission code |
| pub | `/clock`, TF `odom → base_link` | |

Both cameras ride a **runtime-animated camera mast**: `/lift/command` interpolates between
the four measured mount pairs (down/up × top/near), on wall time, with the same
~7 s raise / ~6 s lower durations as the real lift, and publishes `/lift/state` with the
real payload shape. Mounts are kept digit-identical with `perception/fieldlib.py`:

| mast | camera | forward (m) | height (m) | tilt (deg) |
|---|---|---|---|---|
| down | top | 0.198 | 0.3497 | 18.44 |
| down | near | 0.1874 | 0.3143 | 53.20 |
| up | top | 0.198 | 0.4986 | 19.39 |
| up | near | 0.1874 | 0.4600 | 53.24 |

Note that raising the mast is *not* a pure height offset — the top bracket sags by
+0.95 deg when extended. Modelling UP as "DOWN plus height" silently biases every
back-projected object range.

### The arena

`generate_competition_stadium_layout.py --layout grid42 --seed 14 --target-size-m 0.08
--random-yaw` places 28 objects (4 cubes × 4 polyhedra classes, 3 each of apple / orange /
banana / pineapple faces) on 28 of the 42 official 50 cm grid points, applies random yaw,
and enforces the arena's face rule (fruit on the front, top and back; blank on the left, right
and bottom).
It emits a JSON inventory that is the ground truth for every parity measurement, and
`logs`-free ground-truth text that `mission/match_runner.py --gt-file` reads directly.

---

## What it is good for

Concretely, these are things the surrogate found or proved that hardware time could not
have afforded:

- **Perception parity on a known layout.** `field_parity_analyze.py` replays the *real*
  recognition chain (stitch → A1 at imgsz 896 → 224 BGR crops → face vote → depth
  back-projection → 42-cell snap → cell vote) over sim captures. It does not reimplement
  the chain; it subclasses `E2ERunner` from `mission/match_runner.py` and calls the real
  `_process_shots_batch` / `_finalize_scan`. Single-source or it is not parity.
- **Mission-logic regression.** Full 3-minute runs against ground truth: scan → approach →
  measure → grasp → transport → place → retreat, with per-cycle timings and a
  GT-vs-detection confusion table.
- **Contract regression.** The 2026-07-20 field bug where the runner read
  `grasp_state` instead of `state` from `/gripper/grasp` is exactly the class of bug the
  surrogate catches, because the sim publishes the real key.
- **Negative-path verification.** The mis-identification guard (`verify_target`) was
  exercised on *both* branches in sim: agreement (apple 0.96 → grasp → store) and a
  confirmed contradiction (scan said apple, re-check said pineapple 0.58 → skip), which is
  what prevents a 2× mis-pick penalty.
- **Geometry and routing.** Street-router clearance, storage-zone pin layout, and the
  scan-saturation argument (28 of 28 cells covered in a 24 s spin scan) are pure geometry
  and transfer.
- **Photometry rehearsal.** Renderer lighting was swept until the sim cameras matched the
  measured arena band statistics, so the perception chain rehearses at field-equivalent
  brightness rather than studio brightness. See [docs/field-parity.md](docs/field-parity.md).

## What it is not good for

- **Anything involving contact.** Mecanum roller slip, wheel scale error, floor friction,
  gripper closing force, whether an icosahedron rolls out of the fingers. `--drive-mode
  physics` exists and articulates the four wheels with Kaya-style 45° passive rollers, but
  it was **never used as evidence**. At physics rates the real-time factor drops and CPU
  contention makes the localizer run under conditions the robot never sees, so a physics
  result is neither a pass nor a fail — it is a different experiment.
- **Grasp success.** Grasping in sim is a *cheat grip*: on `CLOSE` the nearest object inside
  a hard-coded zone (0.095–0.245 m ahead of base centre, |lateral| ≤ 0.06 m) is attached to
  the robot. A miss logs a diagnostic in the robot frame. This validates *aiming*, never
  *holding*. The real grasp check is sensorless and measured on hardware — settled finger
  gap 3.0–3.2° empty vs 14.1° holding an icosahedron, threshold 8.0°.
- **Timing budgets.** Sim `move_relative` is a P-controller capped at 0.18 m/s
  (`MOVE_RELATIVE_MAX_LINEAR_MPS`), while the real firmware runs its own trapezoidal
  profile. Measured sim moves ran a consistent **2.4–2.9× slower** than the field. The
  instrumentation (`move_stats`) is the transferable part; the wall-clock numbers are not.
- **Face identity confidence.** Sim cube textures are cleaner than printed paper under
  arena light, so face-level accuracy in sim is *optimistic*. Trust the surrogate to the
  A1 four-class level (cube / octahedron / dodecahedron / icosahedron); do not read sim
  fruit accuracy as a field prediction. This is stated again, with numbers, in the parity
  study.

---

## Running it

Requires Isaac Sim 5.1 with its bundled ROS 2 bridge. Three terminals.

```bash
# T1 — the scene (headless, kinematic, mast down, 4 h session)
scripts/run_isaac_mecanum_scene.sh --headless --drive-mode kinematic --mast down --run-seconds 14400

# T2 — the real arena localizer, unmodified
source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 run arena_lightweight_control arena_control_node --ros-args \
  -p controller_type:=mecanum -p localization_mode:=wall_range \
  -p localization_reseed_yaw_step_rad:=0.12 -p localization_reseed_xy_step_m:=0.24 \
  -p imu_topic:=/imu/d435i

# T3 — re-apply the low-light preset (it is not persisted in the USD), then run a match
ros2 topic pub --once /sim/lighting std_msgs/msg/String \
  "{data: '{\"key\": 331.875, \"fill\": 232.3125}'}"
python3 mission/match_runner.py --gt-file <ground-truth.txt> --max-objects 3 \
  --pair off --speed-profile normal --votes-k 1 --fruit-k 1 --yes
```

Two operational gotchas, both learned the hard way:

1. `match_runner.py` needs `rclpy` **and** torch/ultralytics in the *same* interpreter.
   A ROS-sourced system python usually has the first and not the second; use an
   environment that has both.
2. Before re-running a match, the robot must be within 0.35 m of START
   (1.8, −1.8, 90°) — that is the seed gate. Sessions where the robot was manually
   teleported and reset repeatedly accumulated pose residue and produced a run of
   all-failed grasps that had nothing to do with the code under test. If a run looks
   inexplicably bad, reboot the simulator before believing it.

Perception-only parity (no mission run):

```bash
python3 simulation/scripts/street_dataset_sim_capture.py   # drives scan points, saves raw pairs + depth + poses
python3 simulation/scripts/field_parity_analyze.py --capture-dir <dir> --mast up --out <dir>
```

`street_dataset_sim_capture.py` always saves the **raw top/near pair** alongside the
stitched frame, exactly as the field capture tools do — a stitched-only archive cannot be
re-diagnosed.

---

## Files

| Path | Role |
|---|---|
| `scripts/create_mecanum_competition_scene.py` | Scene composer + ROS 2 bridge. The whole surrogate. |
| `scripts/generate_competition_stadium_layout.py` | Arena generator: official cm→map transform, 42 grid points, face rule, JSON inventory. |
| `scripts/street_dataset_sim_capture.py` | Multi-point rotate-and-shoot capture (raw pairs + depth + localizer pose + GT pose per shot). |
| `scripts/field_parity_analyze.py` | Offline replay of the real recognition chain over sim captures. |
| `scripts/lowlight_lighting_sweep.py` | Drives sim lighting to the measured arena band statistics; writes the preset. |
| `scripts/mecanum_motion_selftest.py` | Asserts forward/strafe/rotate signs against ground truth. Exits 0/1 with a JSON verdict. |
| `usd/stadium.usd` | Base stage: 4×4 m floor, 30 cm walls, source object prims. |
| `usd/stadium_competition_grid42.usd` | The arena used in every 2026-07 rehearsal (seed 14, 28 objects). |

`create_mecanum_competition_scene.py` imports `create_mobile_manipulator_scene.py` as a
library for camera specs, publisher wiring and collision-primitive rendering; both must be
in the same directory.

---

## Renderer settings

Three Isaac render defaults invalidate small-object perception experiments, and all three
produce plausible-looking output while doing it. Set them in the Isaac process before any
capture:

```python
carb.settings.get_settings().set("/rtx/post/histogram/enabled", False)  # no auto-exposure
carb.settings.get_settings().set("/rtx/post/tonemap/op", 1)             # linear, no filmic shoulder
carb.settings.get_settings().set("/rtx/post/dlss/execMode", 3)          # DLAA: native res, no upscale
```

With auto-exposure on, a lighting sweep converges on nothing; with DLSS upscaling on, a
640×480 camera renders internally at 320×240 and top-camera detection range collapses from
3.0 m to 1.9 m. The measurements behind each value are in
[docs/field-parity.md](docs/field-parity.md#2-the-three-renderer-traps).
