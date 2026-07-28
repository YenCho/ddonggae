# 08 — Reproducibility

What you can take from this repository and expect to work, what you have to measure again on
your own robot and in your own room, and what is ours alone and will never run anywhere else.

This page is deliberately blunt. A robot that wins a 3-minute match does so on about forty
numbers, and most of them are properties of *one* chassis, *one* pair of camera brackets and
*one* lighting rig. The code is portable. The constants mostly are not, and copying them will
give you a stack that builds, launches, logs cleanly and misjudges every object position.

Every number on this page is a measurement of *this* robot, taken on the date and in the room
named beside it. Read each one as a measurement with a provenance, never as a constant.

Related pages, so this one does not repeat them:
[`04-getting-started.md`](04-getting-started.md) (install and build),
[`07-results-and-lessons.md`](07-results-and-lessons.md) (the measurements themselves),
[`06-troubleshooting.md`](06-troubleshooting.md) (what each symptom means),
[`../CONTRIBUTING.md`](../CONTRIBUTING.md) (the translate-never-delete comment policy that
kept the provenance of these constants alive).

---

## 0. The pinned environment

| Component | Version | How it is verified here |
|---|---|---|
| OS (workstation and robot) | **Ubuntu 22.04** | [`04-getting-started.md`](04-getting-started.md) §1; the Jetson runs the JetPack 6 / L4T flavour of it |
| ROS 2 | **Humble Hawksbill** | The only supported distro. `AGENTS`-level rule in the source repo, restated in [`../CONTRIBUTING.md`](../CONTRIBUTING.md); Jazzy is opt-in only and was never run |
| Python | **3.10** | Ubuntu 22.04 system Python; the ROS packages are `ament_python` against it |
| Build | `colcon` + `--base-paths hardware/ros2 navigation/ros2 third_party` | `../CONTRIBUTING.md`, `conftest.py` |
| RPLIDAR driver | `sllidar_ros2` **1.0.1**, vendored | `third_party/sllidar_ros2/package.xml:5` — vendored precisely so the revision is pinned |
| librealsense (RSUSB build) | **v2.58.2** | `scripts/build_realsense_rsusb.sh:4`. The script also symlinks `librealsense2.so.2.58` → `.so.2.57` (`:46-47`) so the apt-installed `realsense2_camera` links against the userspace build |
| Ultralytics | **8.4.54** training host / **8.4.21** Jetson | Recorded in the private repo's `docs/project/yolo_models.md`; quoted in `04-getting-started.md:103`. No divergence was ever reproduced, but it stays on the suspect list when two machines disagree |
| PyTorch | **2.5.1+cu121** (training host) | Private repo `docs/hardware/school_gpu_server_access.md`. Not verifiable from this tree |
| numpy | **2.4.6** (training host) | Same source. The robot's version is not recorded |
| onnxruntime | version **not recorded anywhere** | Used only by the two pair verifiers, which shipped disabled (`--pair off` in every recorded run) |
| BlenderProc / bpy | **2.8.0** / **5.0.1** | [`../perception/docs/synthetic-data.md`](../perception/docs/synthetic-data.md) §"Reproducibility caveat". The Blender binary version BlenderProc downloaded is **unrecorded** — run `blenderproc run` once and note it before claiming bit-level reproducibility |
| Isaac Sim | **6.0** | Optional, `simulation/` only |

**There is no lockfile.** No `requirements.txt`, no `poetry.lock`, no pinned conda export in
this tree, and `.github/workflows/` is empty — nothing in this repository verifies a version
automatically. The table above is the whole of what we can honestly pin.

The two things that *are* automatically checkable, and that you should run first on any new
machine:

```bash
python3 -m pytest tests/ -q --ignore=tests/test_controllers.py   # 47 tests, no ROS needed
python3 mission/match_runner.py --offline                        # mission logic, no ROS, no weights
```

`conftest.py` puts `perception/`, `perception/tools/` and the three package directories on
`sys.path`, so both run in a bare checkout. `tests/test_controllers.py` is the only test that
imports `rclpy`.

---

## 1. What transfers

These are the parts that are *about the problem*, not about our hardware. They are the reason
the repository is public.

### 1.1 The algorithms

| What | Where | Why it transfers |
|---|---|---|
| Closed-form wall-range localisation against a known rectangle | `navigation/ros2/arena_lightweight_control/arena_lightweight_control/map_localization.py` | Four divides per beam; no map lookup, no particle filter. Any arena that really is an empty convex rectangle gets the same ~20× speed-up. See [`../navigation/docs/localization.md`](../navigation/docs/localization.md) |
| Penalising short returns instead of skipping them | same file, `score_candidate` | Correct **only** where nothing can rise above the scan plane (ours: LiDAR at 0.32 m, tallest object 8 cm). The legacy skip branch is kept in the code with its reasoning, because the opposite is right elsewhere |
| Gyro feed-forward to break the square's 90° degeneracy | `.../arena_control_node.py:1201-1245` | The insight — *the degeneracy needs a sensor, not a better filter* — transfers to any symmetric room. The axis vector does not (§2.5) |
| Mecanum holonomic controller, goal latch, speed profiles | `.../controllers.py` | Standard kinematics plus a hysteretic latch; gains need retuning, structure does not. [`../navigation/docs/control-and-routing.md`](../navigation/docs/control-and-routing.md) |
| Street router (axis-aligned corridors between object rows) | same doc, §"The street router" | A layout strategy for a gridded arena, not a tuned constant |
| Grid snap + per-cell K-vote, asymmetric fruit vote | `perception/fieldlib.py`, `mission/match_runner.py:1343` | Turns detection into classification over a known finite set. The *rule* (`K=2` strong fruit faces beat a blank majority) is derived from a competition rule about where the blank face sits, and is re-derivable for any equivalent rule |
| Depth-median ranging at the contact pixel, with ground back-projection as a logged fallback | `mission/match_runner.py:250`, `:1511-1520` | Shape-agnostic by construction. See §4 and [`07-results-and-lessons.md`](07-results-and-lessons.md) §4.3 |
| Sensorless grasp verification by settled position gap | `hardware/ros2/robot_hardware/robot_hardware/gripper_bridge_node.py:579-619` | The *method* — find the signal your actuator already produces, take a median after the settling curve flattens — transfers. The threshold does not (§2.7) |
| Exact-inverse stitch (`to_source`) | `perception/fieldlib.py:375` | Inference on the stitched frame, back-projection always via the original camera pixels. This is a design contract, and it is what makes an arbitrary 3×3 stitch safe to use |

### 1.2 The ROS graph and the wire contracts

The node/topic graph, the JSON-over-`std_msgs/String` payload schemas, the QoS choices and
the two serial grammars are documented exhaustively in
[`03-ros2-interfaces.md`](03-ros2-interfaces.md), and the reasoning for JSON over a custom
interface package is in [`adr/0002-json-over-std-msgs-instead-of-custom-messages.md`](adr/0002-json-over-std-msgs-instead-of-custom-messages.md).
Two structural rules are worth copying verbatim:

- **One process owns one tty.** Opening the OpenRB-150's port toggles DTR, which reboots the
  board and wipes the RAM-held mast home. See [`../hardware/docs/gripper-and-mast.md`](../hardware/docs/gripper-and-mast.md).
- **Firmware owns the motion primitives** ([`adr/0003-firmware-owns-motion-primitives.md`](adr/0003-firmware-owns-motion-primitives.md)) —
  wheel PID and synchronised trapezoidal relative moves run on the Arduino, so a host stall
  cannot turn into a runaway.

### 1.3 The mission logic

`mission/match_runner.py` is a script, not a node, and `--offline` exercises the argument
parser, the ground-truth layout parser, the corridor/street router and the two-set candidate
filter with no ROS, no weights and no robot. That means the *decision* layer — scan point
choice, target ordering, the fail-closed placement gates, the 6-attempt measure-with-retry,
the re-verify-before-grasping rule — is inspectable and testable on a laptop. The rationale
for each choice is in [`../mission/docs/match-strategy.md`](../mission/docs/match-strategy.md).

### 1.4 The training recipe

Zero hand-labelled images ([`adr/0005-synthetic-only-training-data.md`](adr/0005-synthetic-only-training-data.md)).
What transfers is the whole method, documented in
[`../perception/docs/synthetic-data.md`](../perception/docs/synthetic-data.md):

- analytic object geometry, hybrid render, a camera-artefact stack applied to pixels only so
  labels stay clean;
- the three visibility gates and the metadata schema that makes a rejected render auditable;
- the class rebalancing (plain subsampled from 54.6 % to 40.0 %; apple capped to match orange
  *exactly*, because apple↔orange was the failure axis) and the 30 %-of-crops in-place
  degradation recipe;
- **per-machine render seeds.** Five retrained face models all lost on the real robot because
  every render machine used the same default seed, so only ~22.8 k of 53.7 k scenes were
  unique and duplicates leaked across the train/val split;
- the two runtime contracts, which are part of the model and not implementation details:
  face inference at `imgsz=224` (`perception/fieldlib.py:68`) and **BGR** numpy input.

The scripts are in `perception/training/`. The texture pool and the rendered bundles are
*not* redistributed for licensing reasons — the recipe, ratios, colour gates and a clean
Wikimedia rebuild path are published instead.

### 1.5 The negative results

Four ideas that were built, measured and lost —
[`07-results-and-lessons.md`](07-results-and-lessons.md) §4. They transfer as knowledge and
they are cheap to inherit: the pair-verifier gate that was more accurate in isolation and
worse in the system, the distance gate that was hiding a two-line inference bug, ground-contact
ranging, and the `val = train` number. If you only reuse one thing from this repository,
reuse the habit of publishing these.

---

## 2. What must be re-measured on your robot and at your venue

Every item below is a number in this tree that is a property of our machine. Each has the
tool that produced it.

| # | Quantity | Value here | Tool |
|---|---|---|---|
| 2.1 | Stitch calibration, per mast height | `up`: affine, 6 pts, 0.78 px RMS · `down`: similarity, 1.13 px | `perception/tools/stitch_calibrator.py` |
| 2.2 | Venue photometry | exposure 200 / gain 64 / WB 4600 K (stored sweep), 320 / 64 / 3500 K (launch defaults) | `perception/tools/photometry_tune.py` |
| 2.3 | Encoder CPR | 1320 | `hardware/bringup_tools/20_encoder_cpr_calib.py` |
| 2.4 | Wheel radius and contact geometry | r 0.0388 m · half_length 0.108 · half_width 0.090 | `hardware/bringup_tools/50…53_*.py` |
| 2.5 | IMU yaw axis projection | `[0.0, −0.586, −0.810]` | rotation test, §2.5 |
| 2.6 | Camera mount offsets, both mast heights | 4 measured pairs, plane-fit RMS 0.9–2.7 mm | depth floor-plane fit — **tool not shipped**, §2.6 |
| 2.7 | Gripper open/closed angles, grasp gap threshold | 214.37° / 36.74°, gap ≥ 8.0° | DYNAMIXEL Wizard + `/gripper/grasp` telemetry |

### 2.1 Stitch calibration — once per robot, per mast height

`perception/calibration/stitch/{up,down}.json` is a 3×3 near→top pixel transform fitted at
1920×1080 for *our* two cameras on *our* mast. It encodes the relative pose of two brackets.
There is nothing generic in it.

```bash
python3 perception/tools/stitch_calibrator.py --mast up      # browser UI on :8099
python3 perception/tools/stitch_calibrator.py --mast up --report   # CLI diagnosis, exit 1 if bad
python3 perception/tools/stitch_calibrator.py --mast up \
        --pair-dir logs/field_ops/<run>/scan                 # offline, from saved pairs
```

`--mast` takes `up`, `down` or `mid`; **calibrate each mast position separately** — the two
shipped fits differ in model (affine vs similarity) and in `out_h` (2044 vs 2137). The `mid`
entry in `perception/fieldlib.py:47` is an explicit midpoint *interpolation*, flagged in the
source as an estimate and not a fit.

Two lessons are baked into the tool and will bite you the same way:

- The original parameters came from a **single** correspondence point. One point fixes
  translation and constrains neither scale (field of view) nor roll — the classic "centre
  lines up, both edges splay apart" symptom, measured here as 0.53° of roll error. Put your
  points across the **full width** of the overlap band.
- The overlap band is only ~70 rows tall (mast up), which is too little vertical spread to
  constrain a homography: fitting one made the corners diverge by 330 px. `fit_A` therefore
  refuses affine/homography when the point spread is insufficient and falls back to a
  similarity fit. Use a **flat** target — the two cameras sit at different heights, so nothing
  raised off the floor can be aligned by any 2D transform.

Calibration is resolution-independent (`A' = S_top · A · S_near⁻¹`,
`perception/fieldlib.py:311-329`), so a fit taken at one resolution rescales; it does not
survive a change of camera, bracket or mast.

Stitch geometry is **not** venue-dependent. You do not need to redo it at a competition — only
photometry.

### 2.2 Venue photometry — every venue, every lighting change

Auto exposure and auto white balance are **off** on both cameras in every configuration. Two
free-running cameras converge to different exposures and the seam splits in brightness, which
is a perception failure and not a cosmetic one.

```bash
ros2 topic pub --once /lift/command std_msgs/String "{data: LIFT_TO_TOP}"   # ~7 s
python3 perception/tools/photometry_tune.py --check    # ~8 s diagnosis
python3 perception/tools/photometry_tune.py --sweep    # ~1 min; prints the launch arguments
python3 perception/tools/photometry_tune.py --restore  # back to lab defaults
```

Acceptance gates the tool enforces: floor luma **Y 100–140**, **|ΔY| ≤ 6** across the seam,
**|ΔR/G| and |ΔB/G| ≤ 0.05**, saturation ≤ **8 %**.

Four things the source records that you should inherit:

- **Measure with the mast up.** The mast-down overlap band is 29 rows of extreme lens edge
  where vignetting and specular reflection dominate; measured ΔY 28 down vs 6 up.
- **The residual colour difference is a sensor-unit difference** between the D435 (top) and
  the D435i (bottom) and runtime white balance does not correct it: sweeping WB 3200→6000 K
  moved ΔR/G by 0.004. Match brightness, accept the chroma offset, treat ΔR/G and ΔB/G as
  informational. (`--match-wb` is deprecated and survives only for compatibility.)
- **`power_line_frequency: 2`** (60 Hz) prevents fluorescent banding. Set it to your mains.
- The stored `perception/calibration/photometry/arena.json` honestly records `"pass": false`
  (seam ΔY −6.19 against a ±6 criterion). **Re-run the tool and overwrite the whole file**
  rather than hand-editing `applied` and leaving stale `measured` values attached to it.

Exposure 300 was *rejected* on 2026-07-20 for burning the X-strokes off an arena floor marker
under that day's lighting, while the sweep stored the same evening still lists it as a candidate
at floor luma Y 138.8. What transfers is the acceptance gate, not the exposure number: this is a
venue measurement and not a robot constant.

> **Packaging defect, stated rather than hidden.** `perception/tools/photometry_tune.py:52`
> computes `REPO_ROOT = Path(__file__).resolve().parents[3]`, which was correct at its old
> location (`scripts/dev/field_ops/`) and now resolves to the **parent of the repository**.
> Its `OUT_DIR` (`:62`) and log paths therefore land outside the checkout. `perception/fieldlib.py:27`
> uses `parents[1]` and is correct. Fix the constant before you rely on the tool's output
> paths; the measurement logic itself is unaffected.

### 2.3 Encoder CPR

```bash
python3 hardware/bringup_tools/20_encoder_cpr_calib.py --revs 5
```

Robot on blocks (wheels in the air), **ROS bridge stopped** — this script talks to the
firmware over serial directly. Tape a reference mark on each wheel, zero the counter, turn it
exactly N revolutions by hand, read the ticks. The script prints per-wheel CPR, the deviation
from the mean, and the two places to write the result.

Ours is **1320**, and it is a specification that happened to be confirmed by measurement:
`JGB37-520 (12 V, 333 rpm): 11 PPR hall × 4 (quadrature decode) × 1:30 gearbox = 1320`
(`hardware/firmware/arduino_mecanum/mecanum_encoder_control.ino:120-122`, measured 2026-07-14).
It must be written in **two** places — `real.yaml` `mecanum_bridge_node.encoder_cpr` and the
firmware's `ENCODER_CPR`, followed by a re-flash — or host and firmware will disagree about
what a rad/s is.

### 2.4 Wheel geometry and separation — *effective*, not physical

This is the item most likely to be copied by mistake, because the values look like they could
be measured with a ruler. They cannot.

```yaml
# hardware/ros2/robot_bringup/config/real.yaml — mecanum_bridge_node
wheel_radius_m: 0.0388     # physical 0.040 (80 mm wheel); 0.034 gave 112-116 % actual travel (LiDAR, 2026-07-15)
half_length_m: 0.108       # (L+W) = 0.198 effective; physical 0.275 made a 90 deg command turn 158 deg
half_width_m:  0.090
```

Both are **effective** values that absorb mecanum roller slip, floor friction and tyre
compliance. The Korean comments say so explicitly and tell you to re-measure them whenever the
wheels or the floor surface change. The diff-drive branch carries its own, separate pair —
`wheel_radius_m: 0.044`, `wheel_separation_m: 0.25` — which belongs to the earlier two-wheel
build.

```bash
python3 hardware/bringup_tools/50_move_accuracy.py       # closed-loop moves + tape measure
python3 hardware/bringup_tools/51_move_accuracy_lidar.py # same, LiDAR localisation as ground truth
python3 hardware/bringup_tools/52_single_wall_verify.py --wall front 0,1  # one wall + gyro, no arena
python3 hardware/bringup_tools/53_max_speed.py --axis x   # drive envelope / saturation point
```

Scripts 50–53 need the **ROS bridge running** (the opposite of 20). `52` exists because it
replaces the whole-arena test with a single wall plus the gyro, which is what you can actually
get in a corridor the night before. `53` finds the saturation point by ramping commanded speed
and measuring true ground speed as the slope of wall distance over time — our envelope
(1.22 m/s at PWM saturation, 0.9 m/s operational, 0.6 m/s strafe ≈ 90 % roller transfer,
2.9 rad/s rotation) is a property of this drivetrain and this floor.

While you are here, re-check the **LiDAR mount yaw** too:

```bash
python3 hardware/bringup_tools/40_lidar_orientation_check.py --topic /laser_scan
```

The launch default is `base_scan_yaw = 3.14159265359` (LiDAR 0° facing backwards, correct for
the earlier MK3 mount); the mecanum build ran **0.0**. Guessing wrong mirrors your map.

### 2.5 The IMU yaw axis projection

```python
# navigation/ros2/arena_lightweight_control/launch/lightweight_real.launch.py:171
"imu_yaw_axis": [0.0, -0.586, -0.810],
```

The IMU is inside the **bottom D435i, which is tilted 53–54° down**, so body-yaw rotation
splits across the sensor's y and z axes with the sign inverted by the mount orientation:
`−0.586 ≈ −sin 54°`, `−0.810 ≈ −cos 54°`, measured by rotating the robot on 2026-07-15. The
node default is `[0, 0, 1]` (a body-frame IMU, which is what the simulator provides) and the
node normalises whatever it is given (`arena_control_node.py:392-394`).

There is no dedicated calibration script in this release. The procedure that produced the
vector, and that you can repeat:

1. Park the robot, note the wall-range yaw.
2. Command a known in-place rotation (`hardware/bringup_tools/53_max_speed.py --axis yaw`
   records integrated gyro `dyaw`, odometry `dyaw`, IMU message rate and instantaneous gyro
   peak per step; its `DEFAULT_YAW_AXIS` constant holds the same measured vector, so a
   candidate axis can be substituted and re-run).
3. Compare integrated gyro yaw against the localiser's yaw over several turns in both
   directions. Solve for the projection that makes them agree in **magnitude and sign**.

Get this wrong and the failure is not graceful degradation: the yaw prior integrates
*backwards* and actively drives the estimate toward a symmetric 90° flip, which is how a robot
confidently drops an object in the wrong corner. Re-measure it before anything else if you
change the camera mount. Background:
[`../navigation/docs/localization.md`](../navigation/docs/localization.md).

### 2.6 Camera mount offsets — four of them, one per camera per mast height

```python
# perception/fieldlib.py:41-46
TOP_MOUNT_DOWN  = dict(forward_m=0.198,  height_m=0.3497, tilt_deg=18.44)  # plane fit RMS 2.4 mm, tape 35.0/35.1
NEAR_MOUNT_DOWN = dict(forward_m=0.1874, height_m=0.3143, tilt_deg=53.20)  # RMS 0.9 mm, tape 30.5/30.8
TOP_MOUNT_UP    = dict(forward_m=0.198,  height_m=0.4986, tilt_deg=19.39)  # RMS 2.7 mm, tape 50.0/49.7
NEAR_MOUNT_UP   = dict(forward_m=0.1874, height_m=0.4600, tilt_deg=53.24)  # RMS 1.7 mm (79 % inliers), tape 45.9
```

Every one of these came from fitting the floor plane in the aligned depth image and was
cross-checked with a tape measure. Four corrections recorded in those comments are worth
carrying over as warnings, because each was believed for days:

- **Both cameras ride the mast.** The 2026-07-18 conclusion that the lower camera was
  chassis-fixed came from a capture taken with the mast still down but the file labelled
  `mastup`. Lift stroke is 148.9 mm; the tape deltas are 14.6 / 15.1 cm.
- **The top camera's tilt increases by +0.95° when the mast rises** — the upper bracket sags.
  Adding height alone is wrong. The lower bracket does not sag (53.20 → 53.24).
- **`forward_m` has never been measured by any calibration.** A plane fit cannot observe it.
  It is a tape-measure value and is flagged as such in the source.
- **Roll (−1.0° to −2.5°) is not modelled** by `pixel_to_ground` at all, so a few cm of lateral
  error is possible at the frame edges.

The old nominal values (bottom camera 0.268 m / 54.0°, pre-2026-07-16 reassembly) over-estimated
forward distance by ~17 %, which showed up as grasps landing 2–5 cm short. That is the size of
error you should expect from *not* re-measuring.

**The plane-fitting tool is not in this release.** It lived in the private repo as
`scripts/dev/real_validation/89_mastup_topcam_calib.py` and is referenced by name in
`perception/geometry.py`. What is published is the procedure and the acceptance bar: capture
aligned depth of an empty floor at each mast position, fit a plane (RANSAC, report inlier
fraction), solve height and tilt from the plane's offset and normal, and reject a fit whose
residual RMS is worse than ~3 mm. Then cross-check with a tape measure — every value above has
one, and the tape is what caught the mislabelled `mastup` capture.

Sensitivity, from the source: the depth back-projection scan uses only `tilt` and `forward`, and
a 0.5° tilt error is a few mm at 1 m. The mount numbers matter most in the final approach, not
in the scan.

### 2.7 Gripper angles and the grasp position-gap threshold

```c
// hardware/firmware/openrb_gripper_mast/openrb_gripper.ino:16-19
const float CLOSED_DEG    = 36.74;   // re-measured 2026-07-16 after reassembly, DYNAMIXEL Wizard
const float FULL_OPEN_DEG = 214.37;
```

Mirrored in `real.yaml` (`gripper_bridge_node.closed_deg` / `open_deg`) and used by the mission
code. Re-measure with DYNAMIXEL Wizard after any reassembly: drive the jaws closed until they
just meet, read the angle; drive to full open, read the angle. Two traps:

- An earlier calibration — **closed 70° / open 267°** — is still quoted in the project's own
  older notes. It predates the 2026-07-16 reassembly and is stale.
- `gripper_bridge_node` **declares** defaults of `open_deg: 270.95` / `closed_deg: 187.97`
  (`gripper_bridge_node.py:62-63`), MK2-era values that are only harmless because `real.yaml`
  always overrides them. Launch the node without the parameter file and it commands angles that
  do not correspond to this gripper.

The grasp threshold is a **two-population measurement**, not a tuning knob:

| Condition | Settled position gap | Settled current |
|---|---:|---:|
| Empty hand | **3.0–3.2°** | 107–113 raw |
| Holding an icosahedron | **14.1°** | 120 raw (saturated) |

`grasp_pos_gap_deg: 8.0` sits mid-way, ~5° of margin either side. Current is useless as a
discriminator — an empty close saturates too — and is demoted to a "the joint is actually
loaded" gate at `grasp_current_raw_min: 60`.

To re-measure on your gripper: issue `CLOSE` on an empty hand ~10 times and on a held object
~10 times, and read `pos_gap_deg` from `/gripper/grasp` (or `grasp_pos_gap_deg` on
`/gripper/state`). Take the two populations, put the threshold between them, and confirm you
have margin on both sides. Then re-measure the **settling curve** — ours goes dead-flat at
~577 ms empty / ~537 ms held, which is why `grasp_check_delay_sec: 0.6` and
`grasp_check_window_sec: 0.25` give a ≈0.85 s verdict. If your servo or profile differs, those
two numbers move with it, and the whole window must still fit inside the firmware's 3 s
auto-open watchdog.

Stated weakness, unchanged: the threshold was fitted on one object family. A jaw gap under
~8° of arc — a thin plate, a wire — reads as `empty` while actually held.
[`../hardware/docs/gripper-and-mast.md`](../hardware/docs/gripper-and-mast.md) has the rest.

### 2.8 And also, briefly

- **Motor PID and feedforward** (`speed_kp/ki`, `min_pwm`, `feedforward_slope`) — measured on
  the floor under load, not on blocks; `hardware/bringup_tools/30_pid_step_tune.py` and
  `31_pid_floor_verify.py`. The shipped CSV in `hardware/calibration/motor_pwm/` is ours.
- **`position_tolerance_rad: 0.30`** — deliberately loose. It went 0.035 → 0.08 → 0.20 → 0.30
  across four field sessions because tight tolerances turned into 5 s move timeouts under
  carrying load, costing 36–74 s per run. Re-derive it against *your* grasp tolerance.
- **`GRIP_FORWARD_M = 0.115`** (`perception/fieldlib.py`) — the forward creep at which the
  front plate contacts the object. Adjust in ±0.01 m steps if grasps fail.
- **The map** (`navigation/ros2/arena_lightweight_control/maps/stadium.yaml`) and everything
  derived from it: wall bounds, the 42-cell grid, `START_POSE`, `STORAGE_RECT_MAP`,
  `CENTER_SCAN_XY`, `STORAGE_CORNER_YAW = −135°`. These are the venue's rules, not ours.

---

## 3. What will never transfer

Not "should be re-measured" — **cannot be used at all**, by construction.

### 3.1 TensorRT `.engine` files

The Jetson loaded `a1_yolo26s_seg_896_fp16.engine` when one sat next to the `.pt` and fell back
to the `.pt` otherwise. A serialised engine is bound to one TensorRT version, one CUDA version
and one GPU architecture; an engine built on our JetPack will not load on yours, and a mismatch
usually fails loudly rather than silently, which is the one mercy here.

**None are distributed**, and the export script that built ours is not part of this release
either. Either export your own on the target device from
`perception/models/a1_objectseg/best.pt`, or just run the `.pt` — the pipeline works both ways,
only slower. If you do export, re-tune `A1_BATCH_CHUNK` (8) and `FACE_BATCH_CHUNK` (32) at the
same time: a single 65-shot batch already exhausted the Orin Nano's 8 GB unified memory
(NvMap error 12 → CUDA allocator assert), and raising the input size costs memory in the same
budget.

### 3.2 USB device paths

| Constant | Value here | Where |
|---|---|---|
| Gripper/mast board | `/dev/serial/by-path/platform-3610000.usb-usb-0:2.3.3:1.0` | `real.yaml` fallback list, `real_competition_bridge.launch.py:146-147` |
| Arduino (motors) | `/dev/serial/by-id/usb-Arduino__www.arduino.cc__Arduino_14101-if00` | `real.yaml:3-4` |
| Arduino fallback | `/dev/serial/by-path/platform-3610000.usb-usb-0:2.3.1:1.0` | `real.yaml` `serial_port_candidates` |
| LiDAR | `/dev/rplidar` | udev rule, `third_party/sllidar_ros2/scripts/rplidar.rules` |

A `by-path` string encodes the Orin Nano's USB topology (`platform-3610000.usb`) **and the
physical port the cable is in** — move the cable and it changes. A `by-id` string encodes the
board's own USB serial number, which is unique to that board. `real.yaml` carries a
`serial_port_candidates` list precisely because no single fixed path was reliable. Find yours
with `ls -l /dev/serial/by-id /dev/serial/by-path` and override.

### 3.3 Camera serial numbers

`_030422070364` (top D435) and `_112322074553` (near D435i) appear in
`real_competition_bridge.launch.py:152-153` and are drawn as concrete examples in
[`03-ros2-interfaces.md`](03-ros2-interfaces.md). **They identify our two physical cameras.**
They are printed in the documentation so the graph is readable, not so they can be copied.
Enumerate yours with `rs-enumerate-devices` and pass them as launch arguments.

Which camera is top and which is near is not interchangeable either: the top is a D435, the
near is a D435i, and the IMU used by the localiser lives in the **near** one (§2.5).

### 3.4 The 2026-07-24 fine-tuned face weight

The weight that ran in the finals was fine-tuned on real arena photographs captured hours
before the final matches: 409 train + 45 val real crops and 108 `plain` hard negatives mixed
with 6,215 synthetic scenes, best epoch 32, real-val mask mAP50 **0.755**. **It is not
distributed** — it is a model of one venue's printed objects under one venue's lights, and it
would tell you nothing about your own.

What `scripts/fetch_models.sh` installs is `topfruit_face_s_v3ft`, the fully
synthetic-trained model, and every face-model number quoted in this repository (43 ok / 6 wrong
/ 1 missed on the real 70-crop holdout) belongs to *that* weight. The fine-tune's number is not
comparable to it — it was measured on a real-image validation split, not the 70-crop set.

The fine-tuning **recipe** is published (`perception/training/train_face_ft.py`), so the path
is reproducible even though the artefact is not.

### 3.5 Two more, for completeness

- **The fruit texture pool and the rendered dataset bundles.** 6,400 images assembled from
  sources with unresolved licensing (Kaggle Fruits-360 CC BY-SA, an undocumented FruitSeg30
  slice, and a search-engine scrape). Not redistributable. The recipe and a clean Wikimedia
  rebuild path are published instead —
  [`../perception/docs/synthetic-data.md`](../perception/docs/synthetic-data.md).
- **`perception/eval/face_model_compare.py` as shipped.** It has the same relocation defect as
  the photometry tool (`REPO_ROOT = parents[3]`, `:28`) and its `A1_WEIGHTS` still points at the
  private repo's `data/yolo/weights/cube_face/a1_objectseg/best.pt`, while its datasets live
  under `logs/real_validation/`, which is not part of this release. The comparison *method* is
  readable and reusable; the script will not run unmodified.

---

## 4. Claim → where it was measured → should you expect the same number?

"Same number" means: on a different robot in a different room, built from this repository.

| Claim | Where it was measured | Same number? |
|---|---|---|
| Generic scan matcher costs 345–449 ms per solve | Real robot, 2026-07-04 | **No** — implementation- and CPU-specific. The *ratio* to a closed-form matcher is the transferable part |
| `wall_range` localiser 20.35 ms (synthetic 1080-beam), 8–10 ms yaw-locked global | `map_localization.py`, offline | **Order of magnitude, yes** — it is ~30 float ops per beam in pure Python. Absolute value scales with your CPU |
| On-robot localisation latency: median 28.2 ms, p95 185 ms, max 300 ms (n = 2132) | Three real-arena sessions, Jetson, YOLO running concurrently | **No** — the tail is GIL contention with your perception load |
| Localisation error 2 cm (65 cm before the short-return fix) | Synthetic reference scan | **Yes, if** your arena really is an empty rectangle and nothing rises above the scan plane. Otherwise the fix itself is wrong for you |
| Pose jitter 1.5–2 cm while rotating in place | Street-dataset A/B, 2026-07-18 | Probably similar in kind; the useful part is that it is well inside a ±25 cm grid snap |
| Drive envelope 1.22 / 0.9 / 0.6 m/s, 2.9 rad/s | `53_max_speed.py` sweeps | **No** — drivetrain, floor and battery state. Re-measure (§2.4) |
| Effective wheel radius 0.0388 m, (L+W) 0.198 m | LiDAR-referenced move accuracy, 2026-07-15 | **No** — these are slip-absorbing effective values, not geometry |
| Encoder CPR 1320 | Hand-turn measurement, 2026-07-14 | **Yes if** you use the same JGB37-520 (11 PPR × 4 × 30). Otherwise no |
| IMU yaw axis `[0, −0.586, −0.810]` | Rotation test, 2026-07-15 | **No** — it is `−sin/−cos` of *our* 54° camera tilt. Wrong values flip the pose |
| Stitch inlier RMS 0.78 px (up) / 1.13 px (down) | `stitch_calibrator.py`, 2026-07-21 | **No** — but ≤ ~1.3 px is a reasonable bar for a 6-point affine fit on a flat target |
| Seam ΔY +12.4 → +0.8 px after calibration | 22 fresh live pairs | **No** as a value; **yes** as an acceptance criterion (|ΔY| ≤ 6) |
| Photometry exposure 200–320 / gain 64 / WB 3500–4600 K | Venue sweeps, 2026-07-20/21 | **No, and do not copy** — venue lighting only. Re-sweep in your own room (§2.2) |
| Camera mount plane fits 0.9–2.7 mm RMS | Depth floor-plane fits, 2026-07-18/20 | **No** as values; **yes** as the quality bar a good fit should clear |
| Ground-contact ranging over-estimates by +27…64 mm at z = 4–7 cm | Cross-validation vs depth, 2026-07-21 | **Yes in mechanism and sign** (a tangent in the air back-projects long). Magnitude depends on object height and camera tilt |
| Depth ranging fixed 0/4 icosahedron grasps | Real arena, 2026-07-20/21 | Directionally yes; the specific counts are ours |
| A1 detector: box mAP50 **0.9941** on `val = train` | Ultralytics validation on the training split | **Yes, and that is the point** — you will get a number like this too, and it will mean nothing |
| A1 detector: box mAP50 0.831 / mask mAP50-95 0.558 on a held-out arena set | 60 scenes, 573 objects, 2026-07-08 | **No** — depends on your render recipe and your arena. Build your own held-out set |
| Face model: 43 ok / 6 wrong / 1 missed on 70 real crops | Hand-checked real arena crops | **No** — depends entirely on your printed objects, and it is the synthetic-trained weight measured before any real-photo fine-tune (§3.4) |
| `imgsz=640` collapses face confidence from ~0.9 to 0.1–0.4 | Runtime measurement | **Yes** — it is a contract violation, not a tuning effect. Same for RGB-instead-of-BGR flipping orange↔plain and apple↔pineapple |
| Batched inference 3.86× faster (5 crops: 10.9 vs 42.2 ms) | RTX 5080 | **Directionally yes** on any accelerator; the factor varies with batch size and device |
| 2-stage pipeline 2.97× faster than the 4-stage cascade | Identical 30 s CPU capture, 2026-07-02 | **Directionally yes** — it is four models collapsed into two |
| Asymmetric fruit vote, K = 2: 9 fruit cubes vs 1 for plain majority | Offline re-tally, 5 sessions, 361 detections | **Rule yes, counts no.** The rule follows from the blank-face-on-bottom competition rule |
| Pair verifiers OFF is best (85.7 % vs 74.3 %) | 780-configuration sweep, 175 real face patches | **No** as a number; the *method* (evaluate the gate, not the classifier) is the transferable result — and it contradicted our own earlier 155-crop study, which is also published |
| Grasp gap: empty 3.0–3.2°, held 14.1°, threshold 8.0° | Hardware measurement, 2026-07-17 | **No** — jaw geometry and object size. The two-population method transfers exactly |
| Grasp settling 577 ms empty / 537 ms held → 0.85 s check | Settling-curve capture | **No** — servo and profile specific |
| Full-frame YOLO 43 ms ≈ 23 FPS on the robot | Jetson Orin Nano, stitched frame | **No.** Note also that three reports named `jetson_runtime_*` were actually produced on a Windows RTX 5080 box; there is **no** clean Jetson benchmark in this repository |
| Best rehearsal: 28 cells scanned, 23 correct, 0 ghosts, 0 missed, 180.4 s | Run `20260721_155127`, real arena | **No** — but "zero ghosts, zero missed, five identity errors" is the shape of result this design produces: detection and localisation solved, identity not |
| Sim ↔ real parity: localisation 2.7–2.9 cm, 100 % depth ranging in a full sim run | Isaac kinematic surrogate, 2026-07-21 | **Kinematics yes, physics never** — physics was deliberately never validated in simulation ([`adr/0004`](adr/0004-simulation-is-kinematic-only.md)) |
| Final score 150 (60 + 90), 1st of 16 | SNU AI ROBOT CHALLENGE 2026 | Not a claim about this repository. Two of the four matches lost points to a crashed node and to the clock, not to perception |

---

## Where to go next

| You want to | Read |
|---|---|
| Install and build | [`04-getting-started.md`](04-getting-started.md) |
| See every measurement with its provenance | [`07-results-and-lessons.md`](07-results-and-lessons.md) |
| Diagnose a symptom | [`06-troubleshooting.md`](06-troubleshooting.md) |
| Re-render the training data | [`../perception/docs/synthetic-data.md`](../perception/docs/synthetic-data.md) |
| Rebuild the robot | [`../hardware/README.md`](../hardware/README.md) |
