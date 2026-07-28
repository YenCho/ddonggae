# 04 — Getting Started

Install, build, and run the stack: on a Linux workstation for the offline parts, and on a
Jetson Orin Nano for the real robot.

Read [`docs/08-reproducibility.md`](08-reproducibility.md) alongside this page. Several things
in this repository are **hardware- and venue-specific by construction** and cannot be copied
into a different setup — §9 below lists them explicitly. If you skip that section you will
get a stack that builds, launches, and then misjudges every object position.

---

## 1. Supported platform

| | Workstation (offline) | Robot (onboard) |
|---|---|---|
| OS | Ubuntu 22.04 | Ubuntu 22.04 (JetPack 6, L4T) |
| ROS 2 | Humble | Humble |
| Python | 3.10 | 3.10 |
| Compute | any x86-64 + NVIDIA GPU (optional) | Jetson Orin Nano 8 GB |
| What you can do | build, run the unit tests, run the 2-stage perception pipeline on saved images, train, Isaac Sim | everything, plus drive the real robot |

ROS 2 Humble is the only supported distribution. The launch scripts accept `jazzy` as an
explicit opt-in argument but nothing here was run on Jazzy.

---

## 2. System packages

```bash
sudo apt update
sudo apt install -y \
  git-lfs \
  python3-colcon-common-extensions \
  python3-pip \
  python3-serial \
  ros-humble-realsense2-camera \
  ros-humble-tf2-ros \
  ros-humble-vision-msgs
```

`ros-humble-realsense2-camera` is only needed on the robot. Everything else is needed on
both machines.

### LiDAR device symlink

The bringup opens the RPLIDAR at `/dev/rplidar`
(`hardware/ros2/robot_bringup/launch/real_competition_bridge.launch.py:192`). That name comes
from a udev rule shipped with the vendored driver:

```bash
sudo cp third_party/sllidar_ros2/scripts/rplidar.rules /etc/udev/rules.d/
sudo service udev reload && sudo service udev restart
# replug the LiDAR, then:
ls -l /dev/rplidar
```

### Two RealSense cameras on one Jetson

Two D435-class cameras on a single Orin Nano do not reliably enumerate on the stock kernel
UVC backend. The competition robot ran the **RSUSB** (userspace libusb) backend for the
bottom camera. Build it once:

```bash
scripts/build_realsense_rsusb.sh          # librealsense v2.58.2, FORCE_RSUSB_BACKEND=ON
```

It installs to `~/.local/opt/librealsense-rsusb-2.58.2/`. `scripts/run_real_competition_bridge.sh`
picks it up automatically when present; set `USE_REALSENSE_RSUSB=false` to disable.

---

## 3. Clone the repository

Install `git-lfs` **before** cloning — the simulation USD scenes, the STL visual meshes and the
map PGM are LFS objects.

```bash
git lfs install
git clone <this-repo-url> ddonggae
cd ddonggae
git lfs pull
```

Check that no LFS object is still a ~130-byte pointer file:

```bash
find simulation/usd hardware/cad/meshes -type f -size -1k -print
```

---

## 4. Python dependencies

```bash
pip install --user numpy opencv-python ultralytics onnxruntime pyserial pillow
```

| Package | Used by | Note |
|---|---|---|
| `numpy` | everything | |
| `opencv-python` | stitching, calibration tools, crops | headless hosts can use `opencv-python-headless` |
| `ultralytics` | both YOLO stages | **8.4.54** on the training host, **8.4.21** on the Jetson. No divergence was ever reproduced between them, but keep it on the list of suspects when comparing results across machines. |
| `onnxruntime` | the two pair verifiers (`perception/models/verifiers/*.onnx`) | if it is missing, the pair route degrades to `off` automatically |
| `pyserial` | motor / gripper bridges | also available as the apt package `python3-serial` |
| `pillow` | depth PNG I/O (`uint16` mm, PIL mode `I;16`) | |

`torch` / `torchvision` are only needed for **training** (`perception/training/`), not for
running a match — `ultralytics` pulls in a suitable torch when you install it.

If a pinned `requirements.txt` is present at the repository root, prefer
`pip install -r requirements.txt` over the loose install above.

---

## 5. Model weights

The weights are **not in git**. They are published as GitHub Release assets so that a plain
clone stays small, and because they are Ultralytics YOLO derivatives with different licensing
terms from this repository (see `NOTICE`).

```bash
scripts/fetch_models.sh
```

This downloads the published weights into `perception/models/` and verifies them against the published
`SHA256SUMS`:

| Path | Role | Size |
|---|---|---|
| `perception/models/a1_objectseg/best.pt` | stage A1 — full-frame object segmentation (`cube_like_object`, `octahedron`, `dodecahedron`, `icosahedron`) | 23.4 MB |
| `perception/models/unified_face/best.pt` | stage 2 — cube face identity (`apple`, `orange`, `banana`, `pineapple`, `plain`) | 23.3 MB |
| `perception/models/verifiers/pair_apple_orange_v2.onnx` | binary apple-vs-orange second opinion | 6.1 MB |
| `perception/models/verifiers/pair_banana_pineapple_v2.onnx` | binary banana-vs-pineapple second opinion | 6.1 MB |

Override the source with `GH_REPO=<owner/repo>` and `RELEASE=<tag>`.

Two runtime contracts are wired into the code and must not be "cleaned up". Both were found
by losing accuracy to them:

- **The face model must run at `imgsz=224`** (`perception/fieldlib.py:68`). It was trained at
  224. Passing `imgsz=640` collapses face confidence from ~0.9 to 0.1–0.4 and mis-classifies
  fruit faces. The safest call is to not pass `imgsz` at all and let Ultralytics use the
  checkpoint's training value. Stage A1 is the opposite: it was trained at 640 and runs at
  **896** on the stitched frame (`perception/fieldlib.py:67`), because the stitched frame is
  ~890 px tall and 640 would downscale it by 0.72x.
- **The face model must receive BGR numpy arrays** — the `cv2.imread` convention that
  Ultralytics assumes. Feeding a PIL-read RGB array flips `orange`↔`plain` and
  `apple`↔`pineapple` in measurements. If you are on a PIL path, convert with
  `arr[:, :, ::-1]`.

The face model started from the synthetic-render training set described in
[`perception/docs/synthetic-data.md`](../perception/docs/synthetic-data.md) and was fine-tuned
on real arena photographs on 2026-07-24.

---

## 6. Build the ROS 2 workspace

The ROS 2 packages are **not** in a single `src/` directory. They live next to the subsystem
they belong to:

| Package | Location | Type |
|---|---|---|
| `robot_hardware` | `hardware/ros2/robot_hardware` | ament_python |
| `robot_bringup` | `hardware/ros2/robot_bringup` | ament_python |
| `arena_lightweight_control` | `navigation/ros2/arena_lightweight_control` | ament_python |
| `sllidar_ros2` | `third_party/sllidar_ros2` | ament_cmake |

`colcon` discovers packages recursively under every directory given to `--base-paths`, so one
invocation covers all four:

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install \
  --base-paths hardware/ros2 navigation/ros2 third_party
source install/setup.bash
```

Verify before building, if you want to see what will be picked up:

```bash
colcon list --base-paths hardware/ros2 navigation/ros2 third_party
# arena_lightweight_control  navigation/ros2/arena_lightweight_control  (ros.ament_python)
# robot_bringup              hardware/ros2/robot_bringup                (ros.ament_python)
# robot_hardware             hardware/ros2/robot_hardware               (ros.ament_python)
# sllidar_ros2               third_party/sllidar_ros2                   (ros.ament_cmake)
```

A clean build of all four takes about 11 s on a desktop; `sllidar_ros2` is the only C++
package and emits compiler warnings from the vendored Slamtec SDK — those are expected.

To avoid typing `--base-paths` every time, put a `colcon_defaults.yaml` at the repository
root:

```yaml
build:
  base-paths: [hardware/ros2, navigation/ros2, third_party]
  symlink-install: true
```

### `rosdep` does not work here

`hardware/ros2/robot_bringup/package.xml` still declares `exec_depend` entries for five
packages that were part of the original private workspace and are **not** in this public
release: `example_nav2`, `robot_description`, `robot_perception`, `robot_semantic_mapping`,
`robot_task_planner`. `colcon build` is unaffected (they are runtime, not build, dependencies
and nothing in the shipped launch files instantiates them), but `rosdep install --from-paths`
will fail on them. Install the apt dependencies from §2 by hand instead.

---

## 7. Run the tests

The unit tests are the parts of the stack that need no robot and no GPU: mecanum kinematics
and odometry, wall-range localisation, the goal-latch controllers, the stitch geometry
contract, the grasp fine-alignment maths, and the web UI handlers.

```bash
source install/setup.bash                       # puts the ament_python packages on PYTHONPATH
PYTHONPATH="$PWD/perception:$PYTHONPATH" python3 -m pytest tests/ -q
```

`perception/` has to be added by hand because `fieldlib.py` and `geometry.py` are plain
modules, not a ROS package — they are deliberately importable without an ROS environment so
that the perception contract can be tested and reused off-robot.

---

## 8. Run the bringup

Everything below assumes the robot: motors powered, kill switch on, LiDAR and both cameras
plugged in, and the robot physically placed in the start zone.

### 8.1 One launch for the whole stack

`lightweight_real.launch.py` includes the hardware bridge by default
(`launch_bridge:=true`), so a single launch brings up cameras, LiDAR + scan watchdog, the
`cmd_vel` mux, the motor bridge, the gripper/mast bridge, and the arena localisation and
control node.

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch arena_lightweight_control lightweight_real.launch.py \
  launch_bridge:=true \
  launch_cameras:=true \
  drive_type:=mecanum \
  controller_type:=mecanum \
  base_scan_yaw:=0.0 \
  imu_topic:=/imu/data \
  enable_web_ui:=true \
  launch_python_ui:=false \
  map_yaml:=$PWD/navigation/ros2/arena_lightweight_control/maps/stadium.yaml
```

Four of those arguments are not optional in practice:

- **`drive_type` / `controller_type`** default to `diff` / `diff_drive`. The competition robot
  is mecanum. Setting only one of the two silently gives you a mismatched pair
  (a diff-drive bridge fed by a mecanum controller, or the reverse).
- **`base_scan_yaw`** is the LiDAR mount yaw. The launch default is `3.14159265359`
  (`real_competition_bridge.launch.py:210`), which was correct for the earlier MK3 mount; the
  field automation used **`0.0`** for the mecanum build
  (`perception/fieldlib.py`, `BridgeManager.launch_stack`). Measure yours with
  `hardware/bringup_tools/40_lidar_orientation_check.py` before trusting either.
- **`map_yaml`** must be passed explicitly. Its default still resolves through
  `FindPackageShare("example_nav2")`
  (`navigation/ros2/arena_lightweight_control/launch/lightweight_real.launch.py:39-44`), and
  that package is not part of this release; the map moved to
  `navigation/ros2/arena_lightweight_control/maps/`. Without the override the launch aborts.

The web UI is then at `http://<robot-ip>:18765` — a map view with click-to-goal, used by the
operator for setup and debugging, never during a scored run.

### 8.2 The operator wrapper

`scripts/run_lightweight_arena_control.sh` and `scripts/run_real_competition_bridge.sh` are
the wrappers that were actually typed in the field. On top of the launch they:

- clean up orphaned bridge processes before starting (a stale `motor_bridge_node` sharing the
  Arduino's tty with a fresh `mecanum_bridge_node` corrupts the firmware's command parser — a
  2026-07-18 field incident that needed a physical USB replug to recover);
- select the RSUSB librealsense prefix if it was built;
- scrub `AMENT_PREFIX_PATH` / `PYTHONPATH` of other ROS distributions before sourcing.

They read the same knobs as environment variables (`DRIVE_TYPE`, `BASE_SCAN_YAW`,
`LAUNCH_CAMERAS`, `IMU_TOPIC`, `UI_PORT`, `GRIPPER_SERIAL_PORT`, …). Check the paths inside
them against your own checkout before use.

### 8.3 Check that the stack is alive

```bash
python3 mission/field_autopilot.py check
```

This subscribes for 8 s and prints a pass/fail line per contract topic: LiDAR scan, arena
status, wheel odometry, IMU, all four camera streams, gripper/mast board liveness, and the
localisation lock. Exit code `0` means everything required is present, `2` means something
required is missing. `field_autopilot.py up` does the same but launches the stack first and
seeds the start pose; `field_autopilot.py down` stops everything and reports leftovers.

---

## 9. Run a match

`mission/match_runner.py` is the competition state machine: seed pose → raise mast → drive to
the scan point → 12-way stitched scan → snap detections onto the 42-cell grid → for each target
{route, approach, measure, grasp, carry, place} → repeat until the object budget or the
3-minute clock runs out.

Start conservatively:

```bash
# offline: argument, GT-parsing and router self-test, no ROS needed
python3 mission/match_runner.py --offline

# first real run: pick exactly one object, do not place it
python3 mission/match_runner.py \
  --gt-text "apple:150,200;plain:250,300;octahedron:100,150" \
  --max-objects 1 --no-place --speed-profile safe

# full run in competition mode
python3 mission/match_runner.py \
  --target-shape icosahedron --target-fruit apple \
  --order fruit-first --speed-profile normal
```

Notable arguments:

| Argument | Meaning |
|---|---|
| `--gt-text` / `--gt-file` | the operator types the true object layout in official cm coordinates (`cls:x,y;…`). The runner prints a correct/wrong/ghost/missed table right after the scan, so a bad scan is caught **before** the robot drives anywhere. |
| `--target-shape` / `--target-fruit` | competition mode: set 1 is a polyhedron shape (10 pts x4), set 2 is a fruit face (20 pts x3). Announced on the day. |
| `--order` | `nearest` \| `fruit-first` \| `shape-first` |
| `--speed-profile` | `safe` (0.5 / 0.4 / 0.3 m/s drive/approach/place), `normal` (0.8 / 0.6 / 0.4), `fast` (0.95 / 0.7 / 0.45 — 95 % of the measured ~1.0 m/s ceiling) |
| `--pair on\|off\|ao\|bp` | which binary pair verifiers give a second opinion on fruit faces. Default `bp`: banana/pineapple only. Both finals ran `bp`; qualifier 1 ran `on` and qualifier 2 `ao`. Auto-off if the ONNX files are missing |
| `--range-mode depth\|ground` | `depth` (default) measures the target from the aligned depth image; `ground` back-projects the silhouette's bottom edge through the floor plane. `ground` over-estimates range by 27–64 mm for objects whose visible face is 4–7 cm above the floor — that bias cost 0/4 icosahedron grasps on 2026-07-20 and is why `depth` is the default. |
| `--dry-run` | run scan and inference, log motion/gripper commands instead of sending them |

Outputs land in `logs/field_ops/<timestamp>_e2e/`: `report.json` (per-cycle phase timings,
measurement attempts, grasp verdicts), `grid_map.json`, `gt.json`, the raw top/near RGB+depth
pairs, and the annotated stitched debug frames.

---

## 10. What cannot be reproduced as-is

This is the honest part. Four classes of artefact in this repository are tied to *our*
hardware and *our* venue, and copying the numbers will actively hurt you.

### 10.1 TensorRT engines — not shipped, not portable

The Jetson ran a TensorRT engine for stage A1
(`a1_yolo26s_seg_896_fp16.engine`) when one was present next to the `.pt`, and fell back to
the `.pt` otherwise. Engines are serialised against one specific TensorRT version, CUDA
version, and GPU architecture; an engine built on our JetPack will not load on yours. None
are distributed. Either build your own on the target device from
`perception/models/a1_objectseg/best.pt`, or just run the `.pt` — the pipeline works either
way, only slower.

### 10.2 Device paths and camera serials — machine-specific

| Constant | Value | Where |
|---|---|---|
| Top camera serial | `_030422070364` | `real_competition_bridge.launch.py:152` |
| Bottom camera serial | `_112322074553` | `real_competition_bridge.launch.py:153` |
| Gripper/mast board | `/dev/serial/by-id/usb-ROBOTIS_OpenRB-150*-if00` (was by-path until the board swap on 07-23) | `~~/dev/serial/by-path/platform-3610000.usb-usb-0:2.3.3:1.0` | `real_competition_bridge.launch.py:146-147` |
| Arduino (motors) | `/dev/serial/by-id/usb-Arduino__www.arduino.cc__Arduino_14101-if00` | `hardware/ros2/robot_bringup/config/real.yaml:3-4` |
| LiDAR | `/dev/rplidar` | udev symlink, §2 |

The serial numbers identify *our* two cameras. The `by-path` string encodes the Orin Nano's
USB topology (`platform-3610000.usb`) **and the physical port the board is plugged into** —
move the cable and it changes. `real.yaml` carries a `serial_port_candidates` list precisely
because a single fixed path was not reliable. Find yours with
`rs-enumerate-devices` and `ls -l /dev/serial/by-id /dev/serial/by-path`, then override the
launch arguments.

### 10.3 Stitch calibration — must be re-measured on your robot

`perception/calibration/stitch/{up,down}.json` hold a 3x3 near→top pixel transform fitted at
1920x1080 for two specific cameras on one specific mast, at two specific mast heights
(inlier RMS 0.78 px for `up`, 1.13 px for `down`). It encodes the *relative pose of your two
camera brackets*. There is nothing generic about it.

Re-fit it with the browser calibrator:

```bash
python3 perception/tools/stitch_calibrator.py --mast up     # then open http://<robot>:8099/
```

Two lessons are baked into that tool. The original parameters were fitted from **one**
correspondence point, which fixes translation only and constrains neither scale (field of
view) nor roll — the classic "centre lines up, both edges splay apart" symptom, measured here
as a 0.53° roll error. And the overlap band is only ~70 rows tall, which is too little
vertical spread to constrain a homography: fitting one made the corners diverge by 330 px, so
`fit_A` now refuses affine/homography when the point spread is insufficient and falls back to
a similarity fit. Put your correspondence points across the **full width** of the band, and
use a flat target — the two cameras sit at different heights, so anything raised off the
floor cannot be aligned by any 2D transform.

### 10.4 Venue photometry — must be re-measured at your venue

Auto-exposure must be **locked**, because two free-running cameras converge to different
exposures and split the seam. The right value depends entirely on the room's lighting.

| Setting | Value | Where |
|---|---|---|
| Exposure | 320 | `real_competition_bridge.launch.py:176` |
| Gain | 64 | `real_competition_bridge.launch.py:177` |
| White balance | 6300.0 K | `real_competition_bridge.launch.py:171` |
| Recorded arena sweep, 2026-07-20 | exposure 200 / gain 64 / WB 4600 K, `pass: false` | `perception/calibration/photometry/arena.json` |

Re-measure at your venue, with the mast **up** (measuring with the mast down inflates the
seam metrics several-fold — the down-position overlap band is 29 rows of extreme lens edge,
where vignetting and specular reflection dominate: ΔY 28 down vs ΔY 6 up):

```bash
ros2 topic pub --once /lift/command std_msgs/String "{data: LIFT_TO_TOP}"
python3 perception/tools/photometry_tune.py --check    # ~8 s diagnosis
python3 perception/tools/photometry_tune.py --sweep    # ~1 min, prints the launch arguments
```

Acceptance: floor luma Y in 100–140, |ΔY| ≤ 6 across the seam, |ΔR/G| ≤ 0.05, |ΔB/G| ≤ 0.05,
and no blown-out white objects. One caveat we measured and you will hit too: the residual
colour difference between the two cameras is **sensor-to-sensor** (D435 vs D435i) and white
balance does not fix it — sweeping WB from 3200 K to 6000 K moved ΔR/G by 0.004. Treat the
colour deltas as informational.

### 10.5 Camera mount geometry

`TOP_MOUNT_*` / `NEAR_MOUNT_*` in `perception/fieldlib.py:41-52` are measured mount poses
(forward offset, height, tilt) for three mast positions, fitted against depth floor-plane
measurements and cross-checked with a tape measure. Two honest caveats are recorded in the
source and repeated here: `forward_m` has never been measured by any calibration (plane
fitting cannot observe it), and mount roll (−1.0° to −2.5°) is not modelled by
`pixel_to_ground` at all, so lateral error of a few cm is possible at the frame edges.

---

## Where to go next

| You want to | Read |
|---|---|
| Understand what the robot is answering to | [`01-competition-and-rules.md`](01-competition-and-rules.md) |
| Understand how the pieces fit | [`02-architecture.md`](02-architecture.md) |
| Know the exact topics and message contracts | [`03-ros2-interfaces.md`](03-ros2-interfaces.md) |
| Run a competition day end to end | [`05-field-runbook.md`](05-field-runbook.md) |
| Fix something that is broken | [`06-troubleshooting.md`](06-troubleshooting.md) |
| See what actually happened, including what failed | [`07-results-and-lessons.md`](07-results-and-lessons.md) |
