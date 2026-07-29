# Gripper and camera mast

One OpenRB-150, one Dynamixel bus, two XC330 servos: a parallel gripper (ID 0) and the
camera-mast lift (ID 12). This document covers the gripper's safe arc and current limits,
how a grasp is verified with **no force sensor**, how the mast homes with **no limit
switch**, and the architectural rule that keeps both alive: exactly one process may open
that serial port.

Wire-level command tables are in
[`wiring-and-firmware.md`](wiring-and-firmware.md#9-openrb-150-serial-protocol-115200-usb-cdc).

---

## 1. The gripper (XC330, Dynamixel ID 0)

### 1.1 Calibration

```c
// openrb_gripper.ino:16-19
// Re-measured 2026-07-16 after the gripper was reassembled (DYNAMIXEL Wizard):
// closed 36.74 deg, fully open 214.37 deg.
const float CLOSED_DEG   = 36.74;
const float FULL_OPEN_DEG = 214.37;
```

| Quantity | Value | Source |
|---|---:|---|
| Closed | **36.74°** | Firmware constant; mirrored in `real.yaml` `gripper_bridge_node.closed_deg` |
| Fully open | **214.37°** | Firmware constant; mirrored in `real.yaml` `open_deg` |
| Safe arc travel | 177.63° | `FULL_OPEN_DEG − CLOSED_DEG` |
| Reported jaw width | 0 – 70 mm, linear in the arc ratio | `gripper_bridge_node.py:484-493` |

> **An earlier calibration is superseded.** Closed **70°** / open **267°** dates from
> 2026-06-28, before the 2026-07-16 gripper reassembly. The firmware, `real.yaml` and the
> mission code all agree on **36.74 / 214.37**; if you find 70/267 in older material, it is
> stale.

A second trap: `gripper_bridge_node` **declares** defaults of `open_deg: 270.95` /
`closed_deg: 187.97` (`gripper_bridge_node.py:62-63`) — MK2-era values that are only correct
because `real.yaml` always overrides them. Launch the node without the parameter file and
it will command angles that do not correspond to this gripper.

### 1.2 Why every move is clamped to an arc

The XC330 runs in **extended (multi-turn) position** coordinates, so a raw goal position can
legally be many turns away from the mechanical range. `clampDeg()` forces every target onto
the `[CLOSED_DEG, FULL_OPEN_DEG]` arc, and `targetExtendedDeg()`
(`openrb_gripper.ino:162-168`) converts that safe angle into an extended-coordinate goal
**relative to the present position**, so the servo never takes the long way round through the
mechanical stop.

### 1.3 Current-based position mode and the fault latch

| Constant | Value | Meaning |
|---|---:|---|
| `MAX_GOAL_CURRENT_RAW` | 120 | Goal-current ceiling, also written into the servo's `CURRENT_LIMIT` at boot |
| `MIN_GOAL_CURRENT_RAW` | 10 | Floor for `GRIP` / `SET_CURRENT` |
| `DEFAULT_GRIP_CURRENT_RAW` / `DEFAULT_OPEN_CURRENT_RAW` | 120 / 120 | Closing and opening effort |
| `CURRENT_FAULT_RAW` | 140 | Sustained current above this latches a fault |
| `CURRENT_FAULT_HOLD_MS` | 250 | How long it must persist |
| `PROFILE_VELOCITY_RAW` / `PROFILE_ACCELERATION_RAW` | 40 / 10 | Motion profile |
| `REQUIRE_CURRENT_BASED_POSITION` | `true` | If the servo refuses current-based position mode, the firmware reports `ERR_CURRENT_BASED_POSITION_REQUIRED` and stays not-ready rather than silently falling back to plain position mode and crushing the object |
| `ENABLE_DXL_POWER_ON_BOOT` | `false` | The servo bus is not powered from the sketch at boot |

A latched fault (overcurrent, or any non-zero `HARDWARE_ERROR_STATUS`) torques the servo off
and blocks further motion until `CLEAR_FAULT`.

One negative result worth recording: on 2026-07-17 the closing profile was raised to
velocity 150 / acceleration 40 to try to close faster. **It changed the closing curve not at
all** — in current-based position mode the closing speed is bounded by the goal current, not
the profile. The values were reverted. Faster closing requires raising the current ceiling,
which trades directly against how hard the jaws squeeze.

### 1.4 The 3-second watchdog is a dead-man switch

```c
// openrb_gripper.ino:1009-1015
if (millis() - last_cmd_time > WATCHDOG_MS) {   // WATCHDOG_MS = 3000
  if (!fault_latched) openGripper();
  PC_SERIAL.println("WATCHDOG_OPEN");
}
```

Any received line resets the timer — including the `STATUS?` poll that
`gripper_bridge_node` sends every 0.5 s. So while the bridge is alive the watchdog never
fires; if the bridge (or the whole ROS stack) dies while carrying an object, **the gripper
opens within 3 seconds and drops it.** That is deliberate: an un-commanded servo holding
120 raw of current into a stalled jaw is worse. But it means gripper-side software failures
are visible as "the robot suddenly let go", not as a frozen hand.

The gripper bridge died at 05:12 on the morning of the finals (an uncaught `termios.error`), and by the time we played it had three defences: the exception is caught and the port reconnects, the launch respawns the node after 2 s, and bring-up refuses a green light until the board answers. Final 1 was the failure those do not cover — the process alive, the board silent. In that match the robot reached the object and it never
grasped. Losing the bridge is the one case this watchdog is built for.

---

## 2. What "CLOSE" actually does

```
/gripper/command "CLOSE"
  → gripper_bridge_node.send_deg(closed_deg)       # 36.74
  → serial "SET_DEG 36.74"
  → firmware moveToDeg(36.74, GRIP_CURRENT_RAW=120)
  → node.start_grasp_check()                        # see §3
```

`SET_DEG` within 1.0° of `closed_deg` is also treated as a grasp attempt, because the
approach code issues precision closes rather than a bare `CLOSE`
(`gripper_bridge_node.py:514-522`). `OPEN`, `STOP` and any `LIFT_*` command cancel a
pending grasp check.

The bridge de-duplicates identical commands issued within 0.2 s, so repeated `CLOSE`
publications do not restart the check.

---

## 3. Sensorless grasp detection

There is no force sensor, no tactile pad and no object-presence switch. The robot still has
to answer "am I actually holding something?" before it drives 2 m to the storage box.

### 3.1 Why current does not work

The obvious signal is motor current. It fails, and the measurement says why
(`real.yaml`, `gripper_bridge_node.py:65-82`, both measured 2026-07-17):

| Condition | Settled current | Settled position gap, `abs(present − closed)` |
|---|---:|---:|
| Empty hand | 107–113 raw | **3.0–3.2°** |
| Holding an icosahedron | 120 raw (saturated) | **14.1°** |

In current-based position mode with a goal current of 120, an empty close saturates too —
the jaws simply push against each other. 113 vs 120 is not a discriminator.

### 3.2 The signal that does work

The **settled position gap**. An empty gripper closes until the fingers meet, landing near
`closed_deg`. With an object between the jaws it stops short by the object's width. The
threshold sits in the middle of the two measured populations:

```yaml
grasp_pos_gap_deg: 8.0        # 3.2 and 14.1 are ~5 deg either side
grasp_current_raw_min: 60     # current demoted to a "is the joint actually loaded" gate
```

Decision (`gripper_bridge_node.py:579-619`): `held` if `median(gap) ≥ 8.0` **and**
`median(|current|) ≥ 60`, else `empty`.

### 3.3 Timing

The settling curve was measured rather than guessed: after `CLOSE` from the open position,
position goes dead-flat at **~577 ms empty** and **~537 ms holding an object**. So:

| Parameter | Value | Rationale |
|---|---:|---|
| `grasp_check_delay_sec` | 0.6 | Just past the slower (empty) settling time |
| `grasp_check_window_sec` | 0.25 | Two samples at the node's 0.2 s tick |
| Total decision time | **≈ 0.85 s** | Down from 1.3 s — 0.5 s saved on every pick |

The whole window fits comfortably inside the firmware's 3 s auto-open watchdog, so the
verdict is always taken while the object is still held. During the window the node bypasses
its own `STATUS?` throttle so it gets fresh telemetry every tick, then takes the **median**
of the samples (not the last one) before latching.

### 3.4 The ROS contract

`/gripper/grasp` (`std_msgs/String`, JSON):

```json
{"state": "held", "pos_gap_deg": 14.12, "current_raw": 120, "samples": 2, "stamp": 1.7e9}
```

`state` ∈ `unknown` / `checking` / `held` / `empty`. The same fields are mirrored into
`/gripper/state` as `grasp_state`, `grasp_checked`, `grasp_pos_gap_deg`, `grasp_current_raw`.

### 3.5 Where it is weakest

Stated plainly: the threshold was fitted on one object family. A jaw-width of under
about 8° of arc — a thin plate, a wire — would read as `empty` while actually held. The
in-code note says as much ("re-verify when gripping thin objects"). A second failure mode is
a partial grasp that slips after the window closes; the latch is never re-evaluated during
transport.

---

## 4. The camera mast (XC330, Dynamixel ID 12)

Both RealSense cameras ride the mast. Raising it lifts the whole stereo pair, which is why
the mast height is not a cosmetic setting — it changes the perception calibration (§4.4).

### 4.0 Why there is a mast at all

The mast is raised for exactly one stage of the match, the centre scan, and it exists
because of a viewing-angle problem the rulebook creates. A Set 2 object is a white cube
carrying a fruit photograph on three of its six faces: the **top**, and one **opposing pair
of sides**. A Set 1 cube is the same white cube with nothing on it.

The geometry that matters is which of those three faces you can count on seeing. The two
fruit sides are opposite each other, so from any given direction you see one of them only
about half the time — a cube set down with its blank pair facing you is, from chassis height,
indistinguishable from a Set 1 cube. The top face has no such problem: it is visible from
every azimuth. But at chassis height it is presented nearly edge-on, foreshortening to a
sliver a few pixels tall at the 1–2 m ranges the scan works at, and the face classifier has
nothing to classify. Mast down, cube identity is therefore a coin flip on yaw, and no amount
of model quality fixes it.

Adding 148.9 mm of camera height at the scan point steepens the look-down angle onto every
cell enough that the top faces present real area — turning the one orientation-independent
fruit face from unusable into the primary evidence. It buys a second thing for free: at a
worst-case scan radius of 2.15 m, the higher vantage means the objects in the near rows
occlude the far rows much less, so a single 12-shot spin can see most of the 42 cells.

The price is paid in two places, and both are handled elsewhere in this document: a second
full calibration set, because the lift is not a pure height offset (§4.4), and ~7 s up plus
~6 s down out of a 180 s budget, which the match runner hides under the drive to the centre
and under batched inference respectively (§4.3). Everything after the scan runs mast-down.

The perception-side version of this argument, with the measured fruit-face hit rates that
justify the asymmetric vote, is in
[`perception/docs/grid-voting.md`](../../perception/docs/grid-voting.md).

### 4.1 Homing without a limit switch

There is no limit switch and no absolute encoder across the full stroke. The scheme
(`openrb_gripper.ino:48-63`, `:439-481`):

1. The **first** lift initialization in each Dynamixel power session captures the present
   multi-turn tick as **floor home (0)** and emits `LIFT_HOME_SET <raw>`.
2. Raising the mast **decreases** the tick count (`LIFT_RAISE_SIGN = -1`).
3. Every `LIFT_TO_*` command computes an **absolute** goal `home + delta`, reading the
   present position first. Because targets are absolute rather than relative strokes, any
   sequence of bottom / mid / top is safe — there is no accumulated drift.
4. `LIFT_TORQUE_OFF` (for hand positioning) clears `lift_ready` but **keeps** the home,
   because the multi-turn counter keeps tracking while the bus stays powered.
   `DXL_POWER_OFF` / `DXL_POWER_CYCLE` clear `lift_home_set` — power loss resets the
   multi-turn counter, so the home must be re-captured.
5. `LIFT_SET_HOME` re-declares the present position as home, for when the robot was powered
   on with the mast already raised.

### 4.2 The measured stroke and the top margin

| Constant | Value | Provenance |
|---|---:|---|
| `LIFT_STROKE_TICKS` | 31641 | Measured 2026-07-17: floor `POS_RAW` 33122 ↔ top `POS_RAW` 1481. About 7.7 turns |
| `LIFT_TOP_MARGIN_TICKS` | 800 | ≈ 0.2 turn of deliberate clearance. `LIFT_TO_TOP` stops **short** of the hard stop |
| `LIFT_MAX_MOVE_TICKS` | 40000 | Typo guard — any commanded travel beyond this is rejected with `ERR_LIFT_MOVE_TOO_FAR` |
| `LIFT_GOAL_CURRENT_RAW` | 600 | XC330 is 1 mA/LSB; stall is ~1.5 A, so this leaves margin |
| `LIFT_PROFILE_VELOCITY_RAW` | 400 | 0.229 rpm/LSB ≈ 92 rpm, near the velocity limit — safe because the current headroom is large |
| `LIFT_PROFILE_ACCEL_RAW` | 60 | — |

The 800-tick margin exists because driving a torque-limited multi-turn servo into a hard
stop is exactly how you lose the home reference (it stalls, the controller keeps commanding,
and any later "home" you capture is wrong).

In physical units the stroke is **148.9 mm**, cross-checked two ways
(`perception/fieldlib.py:36-46`):

| Method | Bottom → top | Delta |
|---|---|---:|
| Depth floor-plane fit, top camera | 0.3497 m → 0.4986 m | 148.9 mm |
| Depth floor-plane fit, near camera | 0.3143 m → 0.4600 m | 145.7 mm |
| Tape measure, top camera | 35.1 cm → 49.7 cm | 146 mm |
| Tape measure, near camera | 30.8 cm → 45.9 cm | 151 mm |

### 4.3 Commands and timing

`LIFT_TO_TOP`, `LIFT_TO_MID` (exactly half the top target), `LIFT_TO_BOTTOM`,
`LIFT_MOVE <ticks>`, `LIFT_SET_HOME`, `LIFT_STOP`, `LIFT_TORQUE_OFF|ON`, `LIFT_STATUS?`.
Full reply grammar in
[`wiring-and-firmware.md §9.2`](wiring-and-firmware.md#92-camera-mast-commands-dynamixel-id-12).

Measured travel times, used by the match runner as an upper bound: **~7 s to raise, ~6 s to
lower** (`mission/match_runner.py:173-175`). Completion is detected as
`LIFT_STATUS.MOVING == 0` **or** the elapsed-time bound, whichever comes first — polling
alone over-estimated the time and left the robot waiting on a mast that had already arrived
(`perception/fieldlib.py`, `lift_wait_idle`).

The match runner overlaps mast motion with driving and inference wherever it can, but it
**must** synchronize before any step that depends on camera calibration
(`stage_mast_wait()`).

### 4.4 Mast height is a perception parameter

Because both cameras ride the mast, there are three calibrated mount sets, not one
(`perception/fieldlib.py:36-52`):

| Mast | Top camera (forward / height / tilt) | Near camera |
|---|---|---|
| down | 0.198 m / 0.3497 m / 18.44° | 0.1874 m / 0.3143 m / 53.20° |
| up | 0.198 m / 0.4986 m / **19.39°** | 0.1874 m / 0.4600 m / 53.24° |
| mid | 0.198 m / 0.4242 m / 18.92° (interpolated) | 0.1874 m / 0.3872 m / 53.22° (interpolated) |

Note the trap recorded in that file: raising the mast does **not** just add height. The top
camera's tilt increases by 0.95° because the upper bracket sags under its own moment; the
near camera's tilt is unchanged, because the sag is local to the top bracket. Adding height
alone to a "down" calibration is explicitly called out as forbidden. The `mid` row is
interpolated between the down and up sets rather than fitted in the arena; the two fitted
rows are the ones to trust.

The stitch calibration is likewise per-mast-height
(`perception/calibration/stitch/{up,down}.json`).

---

## 5. One process owns the tty

**The gripper and the camera mast share one OpenRB-150 on one USB serial port.**

Opening that port from a second process asserts DTR, which reboots the OpenRB. On reboot:

- `lift_home_set` goes false — the mast's home is gone. A subsequent `LIFT_TO_TOP` computes
  its goal from a *newly captured* home at whatever height the mast happens to be, and can
  drive straight into the hard stop.
- `setupDynamixel()` finishes by calling `openGripper()` — **the gripper force-opens**,
  mid-grasp, dropping whatever it was carrying.

So the architecture is: `gripper_bridge_node` is the **sole owner** of the port. It opens it
with `exclusive=True`, multiplexes gripper commands and mast commands onto it, and exposes
the mast as a pass-through topic:

```
/gripper/command  ──┐
                    ├─→ gripper_bridge_node ─→ (one tty) ─→ OpenRB-150 ─→ Dynamixel bus
/lift/command     ──┘                                                     ├─ ID 0  gripper
                                                                          └─ ID 12 mast
/gripper/state, /gripper/grasp, /lift/state  ←──────────────────────────┘
```

Practical rules that follow:

1. **Never** run a serial monitor, `screen`, `arduino-cli monitor`, or a one-off Python
   script against the OpenRB while the bridge is running. Publish to `/lift/command` or
   `/gripper/command` instead.
2. To flash the OpenRB, stop the bridge first — the reboot is then intentional, and the
   mast home is re-captured on the next lift command.
3. If the bridge is restarted with the mast raised, issue `LIFT_SET_HOME` only if the mast
   is actually at the floor. Otherwise send `LIFT_TO_BOTTOM` first and let it re-home.
4. The mission-side helper library states the same rule in one line: *"lift/gripper — all
   through the bridge; never open the tty directly."*

---

## 6. Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `/gripper/state` shows `width_mm: null`, `error: true` | The servo was not found on the bus (`STATUS READY 0`) | Check the Dynamixel bus power and ID; send `PING`, then `SCAN` |
| Gripper opens by itself every ~3 s | Nothing is talking to the OpenRB — the watchdog is firing | The bridge died or lost the port; check `/gripper/state.serial_error` |
| `ERR_DYNAMIXEL_NOT_READY` on every command | A fault is latched (overcurrent ≥ 140 raw for 250 ms, or a hardware error) | Clear the mechanical jam, then `CLEAR_FAULT` and `TORQUE_ON` |
| Grasp always reads `empty` on a real object | The object is thinner than the 8° gap threshold, or the gripper's closed reference has drifted | Re-measure `CLOSED_DEG` with DYNAMIXEL Wizard; re-fit `grasp_pos_gap_deg` between the empty and held populations |
| Grasp reads `held` with nothing in the jaws | `closed_deg` is set larger than the true closed angle, so an empty close leaves a false gap | Same recalibration |
| `LIFT_TO_TOP` stalls / grinds | The home was captured somewhere other than the floor (usually after a stray tty open) | `LIFT_TO_BOTTOM`, verify the mast is down, then `LIFT_SET_HOME` |
| `ERR_LIFT_MOVE_TOO_FAR` | Commanded travel exceeded 40000 ticks — almost always a typo or a lost home | Re-home before moving |
| Detections land systematically short or long | The perception mount set does not match the actual mast height | Match `--mast` / the mount set to the real mast position (§4.4) |
