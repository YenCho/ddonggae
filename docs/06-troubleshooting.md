# 06 — Troubleshooting

Symptom → likely cause → what to do. Every entry here is something that actually happened to
this robot between 2026-06 and 2026-07-24, on the bench or in the arena. Where a fix is
already in the code, the file and line are cited so you can see the guard rather than trust
this page.

The competition-day short version lives in [`05-field-runbook.md`](05-field-runbook.md); this
page is the long version with the reasoning.

---

## 0. Sixty-second triage

Run this first. It is faster than guessing.

```bash
python3 mission/field_autopilot.py check       # 8 s listen, pass/fail per contract topic
```

Exit `0` = every required stream is present. Exit `2` = something required is missing, and
the failing line names it. Then find the symptom below.

| Symptom | First guess | Action |
|---|---|---|
| Every move command times out, odometry never changes | Battery switch off | Switch on, re-run `check`. Confirm with `field_autopilot.py up --battery-probe` (3 cm probe move) |
| Motors stutter at low speed | Two motor bridges on one tty | `pkill -f mecanum_bridge_node`, relaunch. `field_autopilot.py up` does this automatically |
| No camera topics | RSUSB backend / USB enumeration | `field_autopilot.py down` then `up`. If it persists, physically replug |
| No localisation output | Pose never seeded, or robot not where you told it | Re-place in the start zone facing north, re-run `up` (re-seeds) |
| `LIFT_*` does nothing | OpenRB unpowered or home not captured | Check the kill switch first. Then `check`, then power-cycle the board **with the mast at the bottom** |
| Scan finds ghosts or misses cells | Perception, not navigation | Compare against the GT table the runner prints; capture with `field_autopilot.py record` and analyse offline |

---

## 1. Power and motion

### Everything times out and the robot does not move

**Cause.** The battery kill switch is off. The motor bridge still reports "connected" —
the Arduino is powered over USB from the Jetson, so the serial link stays healthy while the
motor rail is dead. That combination is deliberately misleading.

**Action.** `battery_off_signature()` in `perception/fieldlib.py` encodes the test: command a
3 cm move, and if it fails or times out *while wheel odometry stays within 5 mm*, the battery
is off. `field_autopilot.py up --battery-probe` runs it for you.

### The motors stutter, or the firmware stops accepting commands

**Cause.** Two processes are writing to the same Arduino tty. The classic version is an
orphaned diff-drive `motor_bridge_node` left over from a previous launch, sharing the port
with a fresh `mecanum_bridge_node`. Interleaved writes corrupt the firmware's line parser.
On 2026-07-18 this required a physical USB replug to recover.

**Action.** Two guards exist:

- both bridges open the port with `serial.Serial(..., exclusive=True)`
  (`hardware/ros2/robot_hardware/robot_hardware/mecanum_bridge_node.py:252`,
  `motor_bridge_node.py:350`), so the second opener now fails loudly instead of corrupting the
  stream;
- `scripts/run_real_competition_bridge.sh` kills known stale bridge processes before
  launching, and aborts if any survive.

If you launch by hand with `ros2 launch`, you skip the second guard. Check with
`pgrep -af mecanum_bridge_node` before starting.

### A single relative move takes seconds longer than the profile predicts

**Cause.** `move_relative` without an explicit `max_v` runs the firmware's *position* mode,
which is capped by `position_max_rad_s: 6.0`
(`hardware/ros2/robot_bringup/config/real.yaml:91`) — about 0.233 m/s, regardless of the
speed profile. On 2026-07-20 a −0.2 m reverse after a grasp took 8 s where the profile
predicted 1.1 s; the timing matched the firmware's own move deadline
(`profile x2 + 5 s ≈ 7.2 s`), i.e. the move failed to settle and timed out rather than being
slow. Note that a −0.35 m move in the same run took 1.74 s, so distance was not the variable.

**Action.** Pass `max_v` on moves that matter. `match_runner.py` records per-move requested
vs. actual duration in `report.json` (`move_stats`) — check it before blaming the controller.

### Teleop or an external `/cmd_vel` publisher is ignored

**Cause.** `cmd_vel_mux_node` is a strict-priority mux:
`/cmd_vel_direct` > `/cmd_vel` > `/cmd_vel_nav` → `/cmd_vel_motor` at 30 Hz, each source
expiring after 0.4 s. The arena node publishes zeros on `/cmd_vel_direct` even while idle,
which used to mask `/cmd_vel` permanently (2026-07-17).

**Action.** Already fixed: the mux selects the highest-priority **non-zero** source and only
falls back to a zero command if every live source is zero
(`hardware/ros2/robot_bringup/robot_bringup/cmd_vel_mux_node.py:80-104`). If teleop is still
ignored, something is publishing non-zero on `/cmd_vel_direct` — `ros2 topic echo` it.

### Motion stops on its own after half a second

**Cause.** By design. `command_timeout_sec: 0.5` (`real.yaml:74`): the bridge zeroes the
wheels if no new command arrives. A publisher running slower than 2 Hz will produce visible
stutter-stop-stutter.

---

## 2. Serial devices

### The bridge cannot find the Arduino or the OpenRB

**Cause.** `by-id` and `by-path` names are not stable across ports, boards, or hosts. The
`by-path` string encodes the physical USB port on the Orin Nano.

**Action.** `real.yaml` carries a candidate list for exactly this reason — the bridge tries
each in turn (`hardware/ros2/robot_bringup/config/real.yaml:3-4` for the Arduino, `:111-112`
for the OpenRB, ending with a bare `/dev/ttyACM0` / `/dev/ttyACM1` fallback). Add your own
path to the list rather than hard-coding one. Reconnection is retried every
`serial_reconnect_interval_sec: 1.0`.

### The robot reboots the Arduino mid-run and then rejects every move

**Cause.** A USB reconnect resets the Arduino. Its RAM-held PID and geometry configuration
returns to firmware defaults, and any in-flight move can never report completion — so
`move_active` sticks and every subsequent move is rejected.

**Action.** Already handled: the bridge watches for the firmware's `READY` banner, fails the
in-flight move explicitly, and re-pushes the configuration
(`mecanum_bridge_node.py:513-524`). If you see `firmware rebooted mid-move` in the log, the
recovery worked — but the cable or the power rail is the real problem.

### Opening the gripper board's tty directly

**Do not.** This is the single most damaging manual action available on this robot.

Opening the OpenRB's serial port from a second process asserts DTR, which **reboots the
board**. Three things happen at once: the gripper force-opens (dropping whatever it holds),
the mast's captured home position — which lives in RAM — is lost, and a subsequent
`LIFT_TO_TOP` then drives toward a position it can no longer bound, into the hard stop.

`gripper_bridge_node` therefore owns the port exclusively (`:219`) and passes `LIFT_*`
commands through to the same board (`:527`), so the mast is driven over `/lift/command` and
never by a second process. This was the fix on 2026-07-19 and it is a hard rule.

```bash
# correct
ros2 topic pub --once /lift/command std_msgs/String "{data: LIFT_TO_TOP}"
# wrong — reboots the board
python3 -c "import serial; serial.Serial('/dev/ttyACM1', 115200)"
```

---

## 3. LiDAR and localisation

### Scans drop out for a few hundred milliseconds and the controller lurches

**Cause.** USB scan dropouts. Downstream consumers see a gap and either stall or act on a
stale pose.

**Action.** `laser_scan_watchdog_node` sits between the driver and everything else: it
subscribes to `/scan_raw`, republishes on `/laser_scan` (plus the `/scan` alias) at a steady
15 Hz, and bridges short gaps by holding the last scan for a bounded time. The relevant knobs
are `raw_fresh_timeout_sec`, `hold_stale_scan_sec`, `restamp_scans` and
`republish_fresh_scan`; the bringup launch sets them explicitly
(`real_competition_bridge.launch.py:199-206`). Note that the launch default and the operator
wrapper disagree on `hold_stale_sec` (`0.0` vs `0.15`) — decide deliberately which you want.

The driver itself is launched with `respawn=True, respawn_delay=2.0`
(`real_competition_bridge.launch.py:250-251`), so a driver crash self-heals in ~2 s.

### The map is mirrored, or obstacles appear behind the robot

**Cause.** LiDAR mount yaw. The launch default is `base_scan_yaw:=3.14159265359` (the sensor's
0° facing the robot's rear, correct for the earlier MK3 mount); the mecanum build ran `0.0`.
A reassembly that rotates the scanner 180° silently inverts the whole world model.

**Action.** `python3 hardware/bringup_tools/40_lidar_orientation_check.py`, then set
`base_scan_yaw` to match. Never guess.

### The pose is 90° off and the robot drives confidently to the wrong wall

**Cause.** The arena is a **square**. Wall-range matching alone has a genuine four-fold yaw
ambiguity, and a bad reseed can settle in the wrong quadrant. In one field run this put the
robot at the wrong start heading and it stored objects in the wrong place.

**Action.** Pass an IMU topic. `arena_control_node` integrates the bottom D435i's gyro-z into
a continuous yaw estimate and uses it as a prior; **once yaw is fixed, the wall-range solution
in a square arena is unique**, and the coarse+refine search becomes a single-yaw global x/y
solve (~200 candidates). Leaving `imu_topic` empty removes the prior and reopens the flip.

```bash
ros2 launch arena_lightweight_control lightweight_real.launch.py imu_topic:=/imu/data ...
```

There is also an online gyro-bias EMA and a 5 s yaw trim while stationary, so a slowly
drifting gyro does not need manual re-zeroing.

### Localisation is slow and the yaw twitches

**Cause.** Full known-map scan matching. Measured at **345–449 ms** per solve on 2026-07-04,
which is long enough for the pose to be stale by the time the controller uses it.

**Action.** Use the default `localization_mode:=wall_range` with a reduced beam count
(`localization_max_beams:=24`). On a synthetic 1080-beam scan the wall localiser measures
**~20.35 ms**. Scan processing is additionally throttled to 10 Hz with QoS depth 1.

### The solver locks onto a far-away pose and x becomes unobservable

**Cause.** A subtle scoring bug worth knowing about even if you write your own localiser.
Beams that return *shorter* than the map predicts were being discarded as "occluded". In a
local window that is harmless; in a global search it lets a distant wrong candidate beat the
true pose because it simply throws away the evidence against it. The synthetic case produced
a 65 cm error.

**Action.** Fixed by *penalising* short returns rather than dropping them
(`penalize_short_returns`). The justification is physical: the scan plane sits at 0.32 m and
the objects are 8 cm tall, so nothing on the arena floor can occlude a wall — a short return
is real evidence, not a missing measurement. Verified at 2 cm / 10 ms on synthetic data.

### The pose snaps back to an old value after the operator resets it

**Cause.** A scan-matching callback that started before the reset finishes afterwards and
overwrites the new pose.

**Action.** Fixed with a `pose_revision` guard (2026-07-04): a result computed against a stale
revision is discarded.

### The robot oscillates near the goal

**Cause.** cm-scale pose jitter combined with a tight yaw tolerance keeps re-triggering the
final-yaw correction.

**Action.** The shipped tolerances are deliberately loose: `xy_tolerance_m: 0.15`,
`yaw_tolerance_rad: 0.40`, with a goal position latch and a wider release tolerance
(`xy_release_tolerance_m: 0.25`, 3 consecutive samples) forming a hysteresis band. Do not
tighten these to "improve accuracy" — measured pose jitter during an in-place rotation is
1.5–2 cm, which is well inside the ±25 cm grid snap that actually matters. Mecanum makes the
final approach gentler because yaw need not change, but it does not remove the jitter.

---

## 4. Cameras: stitch seam and photometry

### The stitched frame lines up in the middle and splays apart at both edges

**Cause.** The calibration was fitted from a **single** correspondence point. One point
determines translation only; it constrains neither scale (field of view) nor roll. Phase
correlation across 81 saved pairs measured the residual as left `dx +13.3 / dy −10.5`, centre
`+11.9 / −11.7`, right `+7.7 / −16.4` — a left↔right disagreement equal to about **0.53° of
roll**.

**Action.** Re-calibrate with several points spread across the **full width** of the overlap
band: `python3 perception/tools/stitch_calibrator.py --mast up`, then open
`http://<robot>:8099/`. Current inlier RMS is 0.78 px (`up`, affine, 6 points) and 1.13 px
(`down`, similarity). Two traps:

- **Use a flat target.** The two cameras are at different heights, so anything raised off the
  floor has parallax that no 2D transform can remove.
- **Do not force a homography.** The overlap band is only ~70 rows tall — not enough vertical
  spread to constrain one. Fitting a homography made the corners diverge by 330 px, while a
  similarity fit stayed at 1.1 px because the full horizontal extent constrains roll on its
  own. `fit_A` now rejects affine/homography when the point spread is insufficient.

### The two halves of the stitch have visibly different brightness or colour

**Cause, part 1 — auto exposure.** If auto-exposure is not locked, the two cameras converge
independently and the seam splits. Measured on 2026-07-20 with the mast up: ΔY **+12.4** at
the seam.

**Cause, part 2 — a type bug.** The launch locked white balance by passing the string
`"4600"`. The `realsense2_camera` parameter is declared as a *double*, so an integer-looking
string was rejected and **the lock silently never applied**. It has to be `"4600.0"`.

**Action.** Lock exposure and white balance explicitly
(`real_competition_bridge.launch.py:171-177`: `camera_auto_exposure:=false`, plus exposure,
gain and a WB value written with a decimal point). Then re-run the venue sweep — see
[`04-getting-started.md` §10.4](04-getting-started.md#104-venue-photometry--must-be-re-measured-at-your-venue).

Do not chase the residual **colour** difference. It is sensor-to-sensor (D435 vs D435i) and
white balance does not touch it: sweeping WB from 3200 K to 6000 K changed ΔR/G by 0.004.

### The seam metrics look terrible and nothing you change helps

**Cause.** You are measuring with the mast **down**. In the down position the overlap band is
29 rows at the extreme edge of both lenses, where vignetting and specular reflection dominate.
Measured 2026-07-20: ΔY **28** with the mast down vs **6** with it up, same lighting.

**Action.** Raise the mast (`LIFT_TO_TOP`) before measuring photometry. The scan — which is
what the stitch is for — happens with the mast up anyway.

### Aligned depth is empty at the left and right edges of the frame

**Cause.** A 4:3 depth profile. The D435's stereo imagers are natively 16:9; asking for
640x480 crops them horizontally and drops the depth H-FOV from ~87° to ~65°, which is
*narrower than the 69° RGB FOV*. Everything near the left and right edge of an FHD colour
frame then has no depth at all, and objects there cannot be located.

**Action.** Keep depth on a 16:9 profile. The shipped default is `848x480x15`
(`real_competition_bridge.launch.py:166`) — full FOV and Intel's recommended resolution. The
reasoning is written out at `real_competition_bridge.launch.py:158-164`.

### RealSense refuses to start the depth stream

**Cause.** A depth profile the sensor cannot produce. The D435 depth stream tops out at
1280x720; asking for 1080p fails negotiation. Aligned depth is resampled to the colour
resolution automatically, so there is no reason to ask.

**Action.** `camera_rgb_profile:=1920x1080x15 camera_depth_profile:=848x480x15`.

### One camera does not enumerate at all on the Jetson

**Cause.** Two D435-class cameras on one Orin Nano under the stock kernel UVC backend.

**Action.** Build the userspace RSUSB backend (`scripts/build_realsense_rsusb.sh`) — the
bottom camera runs on it (`use_rsusb_backend=True`,
`real_competition_bridge.launch.py:235-246`). If it still does not appear, replug; if the
*OpenRB* is the device that vanished, check the kill switch before anything else (that was
the whole cause on 2026-07-21).

---

## 5. Perception

### Face confidence collapsed from ~0.9 to 0.1–0.4 and fruits are mislabelled

**Cause.** The face model is being run at the wrong input size. It was trained at
`imgsz=224`; forcing 640 destroys it.

**Action.** `FACE_IMGSZ = 224` (`perception/fieldlib.py:68`). The most robust fix is to pass
no `imgsz` at all and let Ultralytics use the checkpoint's training value. Stage A1 is the
opposite — trained at 640, run at **896** on the stitched frame
(`perception/fieldlib.py:67`), because the stitch is ~890 px tall and 640 would downscale it
by 0.72x.

This one masqueraded as a hardware difference for a while: "it gets 95 % on my colleague's
machine" turned out to mean "my colleague did not pass `imgsz`".

### `orange` reads as `plain`, or `apple` reads as `pineapple`

**Cause.** RGB input. Ultralytics' numpy path assumes the `cv2.imread` convention, i.e.
**BGR**. A PIL-read RGB array flips exactly these two pairs — both are colour-dominated
decisions.

**Action.** Convert with `arr[:, :, ::-1]` before `predict()`. If you are debugging a
classification regression, check this before touching the weights.

### Face inference is unexpectedly slow

**Cause.** Crops are being predicted one at a time.

**Action.** Batch all crops from a frame into a single `predict()` call. Measured with 5
crops: **10.9 ms batched vs 42.2 ms looped — 3.86x**.

### A cube at the edge of the frame is classified `plain`

**Cause.** The cube is cut off by the frame border, so its fruit face is partly or wholly
outside the image and only side faces remain — and by the arena's construction, side faces
*are* plain. The model is not wrong about what it can see.

**Status.** A known limitation: there is no border-contact detection, so a clipped cube is
judged on the faces that remain visible. If you add it, suppress the identity decision for
border-touching detections rather than forcing a class.

### The scan sees ghost cells, or misses cells it should see

**Action.** Do not debug this in the middle of a run. `match_runner.py` prints a
correct / wrong / ghost / missed table against the GT layout immediately after the scan,
before the robot drives anywhere — stop there. Then switch to data capture,
`field_autopilot.py record --mast down --gt-text "..."`, which saves raw top/near RGB+depth
pairs, the stitched YOLO overlay and a rosbag, and analyse offline.

Historical note: a doubling artefact that looked like a localisation problem turned out to be
depth under-estimation scattered along the line of sight. In-place rotation pose jitter was
only 1.5–2 cm, well inside the ±25 cm snap. The fix was grid-first snapping plus depth
back-projection plus a ≥3-vote requirement, which took an A/B set to 11 cells with 0 duplicates
and 0 misses. Nearby objects raise the wall-fit residual (`loc_score`, correlation −0.73) but
do **not** bias the pose.

### The model is right in the lab and wrong at the venue

**Cause.** Train/deploy domain gap. This cost us the qualifiers: the apple printed on the
arena objects was a **yellowish, under-ripe apple**, not the red one announced beforehand.
Every training image came from Blender, where the apple was red. The face model mis-predicted
it, and no amount of runtime tuning recovers a colour prior that was never trained.

**Action.** Photograph the *actual* objects at the *actual* venue, and fine-tune. That is
what we did between the qualifiers and the finals, on 2026-07-24: the face model that ran in
the finals was fine-tuned on real arena photographs. The fine-tuning recipe is
`perception/training/train_face_ft.py`.

### The robot approaches the object but stops short or overshoots

**Cause.** Range measured from the silhouette's bottom edge (`--range-mode ground`) assumes
the lowest visible pixel touches the floor. For round objects — and for any object whose
visible face is above the floor — it does not. Numerical cross-check: at a true floor point,
`ground` and `depth` agree to 0.0 mm; at 4 cm and 7 cm above the floor, `ground` over-estimates
by **+27–32 mm** and **+53–64 mm**. That bias produced unnecessary defensive hops, and after
the hop the contact point was outside the frame, so re-measurement failed every time — 0/4
icosahedron grasps on 2026-07-20.

**Action.** `--range-mode depth` is the default. Watch the `[approach]` log line: a high rate
of `src=ground` fallback means depth is returning holes (check alignment and the depth
profile, §4), not that `ground` is fine.

---

## 6. Gripper and mast

### The gripper node died mid-match — fixed the morning of the finals

**What happened.** At 05:12 on 2026-07-24 the bridge died on a `termios.error(5, 'Input/output
error')` raised out of `reset_input_buffer` during a USB re-enumeration. The handler caught
`OSError` and `serial.SerialException`; `termios.error` is neither, so it escaped `rclpy.spin()`
and ended the process, and nothing restarted it. A whole practice run collected zero objects.

**The three layers that went in before the finals:**

- The bridge now catches `termios.error` alongside `OSError`/`SerialException` at every serial
  call site — open, close, write, the `STATUS?` request and the read — and reconnects instead of
  dying (`hardware/ros2/robot_hardware/robot_hardware/gripper_bridge_node.py`, `SERIAL_ERRORS`).
- The node is launched with `respawn=True, respawn_delay=2.0`, matching the LiDAR driver on the
  same launch file.
- `field_autopilot.py up` treats the board as required and runs `ensure_gripper_alive()` before
  giving up: clear orphan duplicates, wait out the launch respawn, kick the launch child, start a
  standalone as a last resort.

**What that did *not* fix, and what actually happened in Final 1.** The process stayed alive all
match; the OpenRB simply stopped replying to the grasp check. Since 07-23 the runner only aborts a
grasp on an *active* `empty` verdict — silence is logged as `grasp_unconfirmed` and the cycle
continues, which is the right trade against an intermittently flaky board and exactly the rule that
carried an empty hand across the arena. The open item is a positive liveness requirement on the
verdict itself, not another restart mechanism.
- On shutdown the node re-opens the gripper when `open_on_shutdown` is enabled, which would
  drop a held object. `hardware/ros2/robot_bringup/config/real.yaml:117` sets it to `false`,
  so that path was not armed for the competition — but check it before you enable it.

**Action if you are building on this.** Wrap `rclpy.spin()` so that only clean shutdowns exit,
add `respawn=True` to the gripper node in the launch, and make the mission layer treat a
gripper-state timeout as a recoverable state rather than assuming the bridge is alive. None of
those three changes are in the code as it stands.

### `LIFT_*` does nothing

**Causes, in the order they actually occurred.**

1. **The kill switch is off**, so the OpenRB is unpowered and does not enumerate at all. This
   was the entire cause on 2026-07-21 and it looks exactly like a dead board.
2. **The board's firmware is not running** — no boot banner on the serial link. Re-flash
   (`hardware/bringup_tools/`) and confirm `READY`.
3. **Home was never captured.** Mast home lives in the board's RAM and is captured on the
   first lift command after boot. Any DTR reset (see §2) loses it. Confirm `home_set` in
   `/lift/state` before commanding the mast.
4. **You are in a mode that never calls it.** `match_runner.py --gt-map` deliberately does not
   raise the mast, because that mode skips the scan. Pass `--mast-up` if you want it raised
   anyway. The code logs "not a fault" for this case.

### Mast motion is reported as finished too late, or too early

**Cause.** Polling `LIFT_STATUS` alone over-estimated the raise as 13.5 s against a measured
~7 s.

**Action.** `lift_wait_idle(timeout, assume_done_s)` completes on whichever comes first: an
observed `MOVING == 0`, or the elapsed-time budget (7 s up, 6 s down), and returns the reason
it used. Non-blocking callers pass only the remaining budget measured from the command
timestamp, so parallel driving time is not counted twice.

### A held object is dropped when placing, or the fingers hit the storage wall

**Cause.** Commanding a full `OPEN` at the storage box makes the fingers collide with the box
wall. Commanding an absolute angle is also wrong: when the gripper holds an object the fingers
stop at the object's width, not at `closed_deg`, so the same absolute target produces a
different opening per object.

**Action.** Release **relative to the measured present position**: `--release-ticks 600` in competition, 450 by default (≈52.8° and 39.6°; an earlier build used 400) from
`present_deg` as reported on `/gripper/state`, tunable with `--release-ticks`. No firmware
change is needed — the bridge still sends `SET_DEG`.

### Grasp success is always reported as failed

**Cause (historical, fixed).** The `/gripper/grasp` payload key is `state`, and the mission
code read `grasp_state`, so it always saw an empty string and judged every grasp a failure —
including successful ones. `perception/fieldlib.py` now reads both keys for compatibility.

**Cause (configuration).** Grasp detection is sensorless — there is no force or tactile sensor
on this gripper. It compares the settled finger position against the commanded close. The
threshold has a measured basis (2026-07-17): an empty close settles at a gap of **3.0–3.2°**,
a gripped icosahedron at **14.1°**, so `grasp_pos_gap_deg: 8.0` sits mid-way with ~5° of
margin on both sides. Motor current is *not* usable as the primary signal — empty reads
107–113 raw and gripped reads 120 raw, both saturated — so it is only a secondary check
(`grasp_current_raw_min: 60`). The verdict is taken in a 0.25 s window 0.6 s after the close,
because the position curve goes dead flat at ~577 ms empty / ~537 ms gripped
(`hardware/ros2/robot_bringup/config/real.yaml:121-133`).

Objects much thinner than 8° of finger travel will not be detected. Re-measure `open_deg` /
`closed_deg` for your own gripper (`:119-120`) after any mechanical change.

### The grasp is consistently a centimetre short or long

**Action.** `GRIP_FORWARD_M = 0.115` (`perception/fieldlib.py:78`) is the measured forward
creep at which the gripper's front plate contacts the object. It is a single source of truth
shared by the mission code. Adjust it in ±0.01 steps; do not scatter local offsets.

---

## 7. Match-level

### The 3-minute clock ran out mid-grasp — it cost us the last object of Final 2

**What happened.** In Final 2 the timeout hit while the robot was grasping the last object.
The run was correct, just not finished.

**Action if you are building on this.** The speed profiles exist for this
(`safe` / `normal` / `fast`, at 0.5 / 0.8 / 0.95 m/s drive speed), and the shipped procedure
is to escalate: one `safe` cycle to prove the scan table and the grasp, then a full `normal`
run, then a full `fast` run, and adopt the fastest profile that keeps the success rate while
finishing inside 180 s. What is *missing* is a time-aware policy — the state machine has no
notion of "do not start a cycle that cannot finish". Compare `summary.total_sec` and the
per-cycle `route` / `approach` / `grasp` / `carry` phase times in `report.json` when tuning.

### The robot collects an object that is not the target

**Cause.** Target selection uses the scan's identity. If the class is confidently wrong, the
wrong object gets collected.

**Action.** `--target-class` re-verifies the object immediately before the close: A1 + face
(+ pair verifiers) are re-run at the final measurement, and only a **definite contradiction**
causes a skip — a polyhedron class mismatch, "not a cube", or a different fruit face at
conf ≥ 0.3. `plain`, no detection, and a weak fruit face all pass, because the arena's objects
show plain on every side face, so treating "undecidable" as a mismatch would discard genuine
targets.

**Watch out for the silent fallback.** If no cell is confidently the target class,
`pick_target` quietly reverts to nearest-first collection. If collecting a non-target costs
points under your rules, that fallback is a liability — make it explicit.

---

## 8. When none of this helps: capture, do not guess

```bash
python3 mission/field_autopilot.py record --mast down --gt-text "apple:150,200;..."
# drive the robot around with the UI or teleop, Ctrl-C to stop
python3 mission/field_autopilot.py down
```

`logs/field_ops/<timestamp>_*/` then holds everything needed for offline analysis:

| Artefact | Contents |
|---|---|
| `verdict.json` | the automatic pass/fail from `up` / `check` |
| `frame_<n>_{top,near}_{rgb,depth}.png` | raw pairs — depth is `uint16` mm, PIL mode `I;16` |
| `yolo_debug_<n>.png` | stitched frame with A1 + face overlays, confidences and face votes |
| `frames.jsonl` | per-frame pose, localisation latency, detections |
| `bag/` | rosbag of scan, arena status, motor, odometry, IMU, gripper, lift, `cmd_vel` |
| `gt.json` | the layout the operator typed in |

Always save the **raw top/near pair**, never only the stitched image. Almost every perception
question we could not answer afterwards was a case where only the stitch had been kept.
