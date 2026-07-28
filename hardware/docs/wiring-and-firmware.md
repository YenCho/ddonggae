# Wiring and firmware

Everything between the battery and the ROS graph: pin map, motor driver, encoder wiring,
the complete serial protocol for both microcontrollers, the two-level PID cascade, and the
motion primitives the firmware owns.

---

## 1. Which sketches are the flashed competition firmware

| Board | Sketch | Status |
|---|---|---|
| Arduino UNO | `hardware/firmware/arduino_mecanum/mecanum_encoder_control.ino` | **Flashed for the competition.** Last modified 2026-07-17. Its serial grammar matches `mecanum_control.py`'s parsers and `mecanum_bridge_node.push_firmware_config()` exactly. |
| OpenRB-150 | `hardware/firmware/openrb_gripper_mast/openrb_gripper.ino` | **Flashed for the competition.** Last modified 2026-07-17. Every string `gripper_bridge_node` writes or parses is implemented here. |

Three earlier Arduino sketches from the project history (`mecanum_pwm_control` open-loop,
`encoder_pwm_test` differential-drive, `sketch_may21a` bring-up demo) are **not** published:
they are superseded and their pin maps contradict the shipped wiring.

Flashing:

```bash
bash hardware/bringup_tools/00_flash_firmware.sh          # arduino-cli, fqbn arduino:avr:uno
```

Stop `mecanum_bridge_node` first — it holds the port with `exclusive=True` (see §8.4). The
OpenRB sketch is uploaded from the Arduino IDE / `arduino-cli` with the OpenRB-150 board
package and needs the ROBOTIS `Dynamixel2Arduino` library from the Library Manager (the
project vendored it; this repository does not).

---

## 2. Boards, buses, and who owns what

| Bus | Device | Baud | Owner process |
|---|---|---|---|
| USB CDC → Arduino UNO | 4 × motor, 4 × encoder | 115200, 8N1, line-oriented | `mecanum_bridge_node` (exclusive lock) |
| USB CDC → OpenRB-150 | gripper + mast | 115200 | `gripper_bridge_node` (**sole owner** — see [gripper-and-mast.md](gripper-and-mast.md#5-one-process-owns-the-tty)) |
| OpenRB `Serial1` → Dynamixel | XC330 ID 0 (gripper), ID 12 (mast) | 1 000 000, Protocol 2.0 | OpenRB firmware |
| USB → RPLIDAR A2M12 | LiDAR | 256000 | `sllidar_node` |
| USB3 → 2 × RealSense | RGB + depth (+ IMU on the D435i) | — | `realsense2_camera` |

Body frame convention, identical in firmware, `mecanum_control.py` and the URDF generator:
**+x forward, +y left, +yaw counter-clockwise**. Wheel order is always
`(front_left, front_right, rear_left, rear_right)`.

---

## 3. Arduino UNO pin map

The firmware's own `pins` command is the authority; this table is its output
(`mecanum_encoder_control.ino:617-628`).

| Wheel | Driver channel | DIR | PWM | Encoder A | Encoder B |
|---|---|---:|---:|---:|---:|
| front_left (FL) | MDD10A #1 M1 | D4 | D5 | D8 | D9 |
| front_right (FR) | MDD10A #1 M2 | D7 | D6 | A0 | A1 |
| rear_left (RL) | MDD10A #2 M1 | D2 | D3 | A2 | A3 |
| rear_right (RR) | MDD10A #2 M2 | D12 | D10 | **D11** (see §4) | A5 |

| Pin | Use |
|---|---|
| D0, D1 | USB serial to the Jetson — connect nothing |
| D13 | Spare, but it drives the on-board LED; not recommended for an encoder input |
| A4 | Reported dead (stuck low) on 2026-07-14 and masked out of the pin-change interrupt |

All four PWM pins are hardware PWM (Timer0: D5/D6, Timer2: D3, Timer1: D10). Encoders are
decoded with pin-change interrupts and a quadrature state table
(`mecanum_encoder_control.ino:226-278`), so no external-interrupt pins are needed — which is
why D2 and D3 are free for the rear-left motor.

---

## 4. The rear-right encoder A pin

**Sources in the repository disagree about this one pin. Verify it against your own board.**

| Source | Says |
|---|---|
| `mecanum_encoder_control.ino:18` (file header comment) | `Rear-right encoder: A = A4, B = A5` |
| `mecanum_encoder_control.ino:116` (the constant that actually compiles) | `const uint8_t ENC_RR_A = 11;  // PB3 (was A4 — dead line)` |
| `mecanum_encoder_control.ino:622` (`pins` output) | `encA=D11 encB=A5`, `dead=A4` |
| `mecanum_encoder_control.ino:985` (interrupt mask) | `PCMSK1 = ... // A0-A3, A5 (A4 dead — masked out)` |
| Original wiring guide (`docs/hardware/mecanum_wiring.md`, Korean) | RR encoder A = D11; A4 marked *do not use* |
| `ros2/robot_bringup/config/real.yaml` comment | "RR judged after the A4 broken line was **repaired**" |

The code and the wiring guide agree on **D11**; the file-header comment still says A4, and
the `real.yaml` comment implies the A4 line was repaired at some point. **Trust the `pins`
output on your own board**, and treat the header comment as stale.

Related, and equally load-bearing: the rear-right encoder module has a **dead A output**, so
the firmware ships with single-channel counting enabled by default
(`mecanum_encoder_control.ino:186-192`):

```c
volatile bool rrSingleChannel = true;   // default ON until the encoder is replaced
volatile int8_t rrFbRawDir = 2;
```

In this mode every rear-right B edge adds ±2 ticks (so the effective CPR stays 1320), with
the sign taken from the last commanded PWM direction. It loses accuracy only around
direction reversals, and it is part of why the position tolerance had to be relaxed
(§7.3). Turn it off with `rr1ch 0` once the encoder is replaced. This command exists in the
parser (`:890-899`) but is missing from the firmware's own help text — an undocumented
command in the shipped build.

---

## 5. Motor driver wiring (Cytron MDD10A × 2)

Each MDD10A carries two channels; each channel takes **PWM (speed) + DIR (direction)**,
sign-magnitude, with no EN/STBY pin. 10 A continuous / 30 A peak per channel, 5–30 V motor
supply, 3.3 V/5 V-tolerant logic.

| MDD10A terminal | Connect to |
|---|---|
| VM+ / VM− | Battery +12 V / GND (both boards branched in parallel) |
| M1A / M1B | Motor 1 (FL or RL) |
| M2A / M2B | Motor 2 (FR or RR) |
| PWM1 / DIR1, PWM2 / DIR2 | Arduino pins per §3 |
| GND (signal) | **Common with the Arduino GND** |

Notes carried over from the original wiring guide:

- Motor lead polarity (`M?A` / `M?B` swapped) only reverses that wheel. Do not rewire —
  fix it in software with the `sign` command (§6.2).
- MDD10A accepts PWM up to 20 kHz, so the UNO's default 490/980 Hz is fine.
- The on-board M1A/M1B/M2A/M2B test buttons let you verify power and motor wiring
  **before** the Arduino is connected. Do this first.
- **Common ground is mandatory**: battery GND ↔ both MDD10A GND ↔ Arduino GND ↔ Jetson
  (via USB). A missing common ground was the suspected cause of a 2026-06-26 motor fault.
- Never power motors from the Arduino 5 V rail.

The firmware also supports the older inverted-PWM driver style (forward = DIR HIGH,
`analogWrite(255 - duty)`) as a runtime switch, because the project ran on such a board
while waiting for the MDD10A units:

```
mstyle 1 1 1 1      # MDD10A sign-magnitude (the shipped default)
mstyle 0 1 0 1      # the legacy diff-drive driver's mixed left/right convention
```

Using style 0 with an MDD10A makes reverse duty the complement of the command — full
reverse produces duty 0. That bug is documented in-line at
`mecanum_encoder_control.ino:324-334`.

---

## 6. Encoder wiring and polarity

- Each encoder: A, B, VCC, GND. **VCC comes from the Arduino 5 V rail**, not from motor power.
- A0–A5 are used as digital inputs with internal pull-ups, so open-collector encoder
  outputs work unmodified.
- Swapping A and B only inverts the count sign — fix it in software.

### 6.1 Counts per revolution

`ENCODER_CPR = 1320` (`mecanum_encoder_control.ino:120-122`): JGB37-520, 11 PPR hall × 4×
quadrature decode × 1:30 gearbox. Bench-confirmed on 2026-07-14 from the no-load full-PWM
tick rate (~1330 measured). Re-measure with `bringup_tools/20_encoder_cpr_calib.py` if you
use a different gearmotor.

### 6.2 Polarity, fixed in software

Bench calibration on 2026-07-14 found that all four motors were wired so that positive PWM
drives the wheel backwards, and that the FR and RR encoders had A/B swapped. Rather than
rewire, the polarity lives in two arrays that both the firmware defaults and
`real.yaml` carry:

```
motor_signs:   [-1, -1, -1, -1]
encoder_signs: [ 1, -1,  1, -1]
```

`10_wheel_selftest.py` determines these for you and prints both the firmware `sign` line and
the YAML snippet.

---

## 7. The control cascade

Everything below runs **on the Arduino**, not on the Jetson.

### 7.1 Inner loop — per-wheel velocity PID at 50 Hz

`PID_INTERVAL_MS = 20`. Wheel speed is reconstructed from tick deltas, then
(`mecanum_encoder_control.ino:477-499`):

```
feedforward = min_pwm + ff_slope * |target|
output      = feedforward * sign(target)
            + kp * error + ki * ∫error dt + kd * d(error)/dt
output      = clamp(output, -max_pwm, +max_pwm)
```

The integral is clamped to `±integral_limit`, and targets below 0.01 rad/s reset the loop
and command zero (so a stopped wheel does not wind up).

Shipped gains and where they came from:

| Parameter | Value | Provenance |
|---|---:|---|
| `speed_kp` | 10.0 | 2026-07-14 block step tuning on JGB37-520, re-tuned 2026-07-15 under floor load |
| `speed_ki` | 8.0 | No-load gains (`ki=5`) let the front-left wheel sag −18 % at low speed on the floor |
| `speed_kd` | 0.0 | Not needed |
| `min_pwm` | 10.0 | Feedforward offset to break stiction |
| `max_pwm` | 250.0 | — |
| `feedforward_slope` | 6.9 | Fitted from an open-loop PWM/speed sweep |
| `integral_limit` | 20.0 | Raised with `ki` for the same floor-load reason |

Acceptance after re-tuning: steady-state error < ±6 % at 3, 6 and 9 rad/s forward and
reverse, rise time from standstill < 0.6 s — with the rear-right wheel in single-channel
mode.

### 7.2 Outer loop — position, with a synchronized trapezoid

A `d` (or `pw`) command starts a position move (`:401-449`, `:517-609`):

1. Convert the body displacement to four wheel-angle deltas with the same inverse
   kinematics used for velocity.
2. Build a trapezoidal profile for the **longest-travel wheel** using `position_max_rad_s`
   and `position_accel_rad_s2` (collapsing to a triangle if the move is short), then scale
   the other three wheels proportionally — so all four start and finish together.
3. Every 20 ms, sample the profile, and command
   `velocity_target = profile_velocity + posKp·position_error + posKd·d(error)/dt`
   into the inner loop.
4. After the profile ends, if any wheel is still outside `position_tolerance_rad`, floor its
   correction speed at `position_min_rad_s` so stiction cannot park it just under the PWM
   dead band. Once all four wheels are inside tolerance for `position_hold_ms`, the move is
   done.

Two guards worth stealing:

- **Runaway clamp.** The corrective term is clamped to `1.15 × position_max_rad_s`, not to
  the global wheel limit. Comment at `:585-590`: after `MAX_WHEEL_RAD_S` was raised 16 → 30,
  an unbounded `posKp` correction actually did exceed the caller's `max_v`, producing a
  terminal deceleration spike, wheel slip and objects being knocked over.
- **Deadline, not a fixed timeout.** `deadline = 2 × profile_time + 5 s` (`:446-448`).
  Raised from +1.5 s → +2.5 s → +5.0 s: on 2026-07-19 a 0.3–0.4 m approach hop timed out in
  the arena, the caller proceeded anyway, and the gripper closed on empty air.

### 7.3 Position tolerance: the number we relaxed three times

| Date | `position_tolerance_rad` | Wheel-surface error | Why |
|---|---:|---:|---|
| initial | 0.035 | ≈ 1.2 mm | Bench value |
| 2026-07-15 | 0.08 | ≈ 3 mm | Floor load + rear-right single-channel settling noise made strafes and turns fail to hold → timeouts |
| 2026-07-16 | 0.20 | ≈ 7.8 mm (≈ 2.2° of yaw) | Small moves and turns chattered outside 0.08 on stiction; 3 of 8 turns in an arena end-to-end run hit the 4.9 s timeout |
| 2026-07-20 | **0.30** (+ `position_hold_ms` 100 → 50) | ≈ 11.7 mm | Under carry load (reversing with an object, approaching and backing off the storage box) half of all single moves burned the deadline, costing 36–74 s per run. Storage still succeeded 5/5, so effective precision was unchanged — only waiting time was recovered |

The honest reading: this is not a precise base. It is a base whose imprecision (≈ 12 mm) is
comfortably inside the grasp lateral tolerance (±55 mm), and we spent the difference on
time.

### 7.4 Kinematics

X-configuration mecanum, identical in `mecanum_encoder_control.ino:358-364` and
`ros2/robot_hardware/robot_hardware/mecanum_control.py:69-79`:

```
FL = (vx − vy − k·wz) / r        k = half_length + half_width
FR = (vx + vy + k·wz) / r        r = wheel_radius
RL = (vx + vy − k·wz) / r
RR = (vx − vy + k·wz) / r
```

`r` and `k` are **fitted** values (0.0388 m and 0.198 m), not ruler measurements — see
[`../README.md`](../README.md#34-understand-which-constants-are-effective-not-physical).

Speed ceilings: firmware `MAX_WHEEL_RAD_S = 30.0` rad/s (raised from 16 on 2026-07-17 —
16 rad/s was 0.62 m/s, an artificial cap far below the motor's 34.9 rad/s no-load speed);
the effective limit is the bridge's `max_wheel_rad_s: 26.0`, with body caps
`max_linear_x_mps: 0.9`, `max_linear_y_mps: 0.6`, `max_angular_rps: 2.0`.

---

## 8. Arduino serial protocol (115200 8N1, one command per line)

### 8.1 Commands

| Command | Meaning |
|---|---|
| `m <vx_mps> <vy_mps> <wz_rad_s>` | Body velocity (velocity mode). Replies `OK m` |
| `w <fl> <fr> <rl> <rr>` | Per-wheel angular velocity targets, rad/s. Replies `OK w` |
| `d <dx_m> <dy_m> <dyaw_rad>` | **Relative body displacement** (position mode). Replies `OK move`, then `DONE,move,...` or `ERR move timeout` |
| `pw <fl> <fr> <rl> <rr>` | Relative wheel position deltas in rad (position mode) |
| `p <fl> <fr> <rl> <rr>` | Direct signed PWM, −255..255; disables the closed loop until the next `m`/`w`/`d` |
| `pid <kp> <ki> <kd> <min_pwm> <max_pwm> <ff_slope> <int_limit>` | Velocity loop tuning (shared by all wheels) |
| `ppid <kp> <kd> <max_rad_s> <accel> <tol_rad> <hold_ms> <min_rad_s>` | Position loop tuning |
| `geom <wheel_radius_m> <half_length_m> <half_width_m>` | Kinematics geometry used by `m` and `d` |
| `sign <m_fl> <m_fr> <m_rl> <m_rr> <e_fl> <e_fr> <e_rl> <e_rr>` | Per-wheel motor and encoder polarity (±1) |
| `mstyle <fl> <fr> <rl> <rr>` | PWM encoding style per wheel (0 = DIR_HIGH_PWM_LOW, 1 = PWM_HIGH_DIR_LOW) |
| `rr1ch <0\|1>` | Rear-right single-channel encoder mode (**undocumented in the firmware's own help**) |
| `z` | Zero the encoder counters |
| `stop` | Stop all motors, abort any position move |
| `stream <0\|1>` | Enable/disable periodic `STATE` lines |
| `pins` | Print the pin map and current `mstyle` |
| `test [pwm] [ms]` | Drive each wheel forward then reverse in turn, printing `STEP_TICKS` |
| `?` | Help (also printed at boot, preceded by the `READY` banner) |

### 8.2 Telemetry

```
STATE,<ms>,<fl_ticks>,<fr_ticks>,<rl_ticks>,<rr_ticks>,<fl_pwm>,<fr_pwm>,<rl_pwm>,<rr_pwm>,<mode>
```

at 10 Hz when streaming (`REPORT_INTERVAL_MS = 100`), where `mode` is 0 = idle/PWM,
1 = velocity, 2 = position. Tick values already have `encoder_signs` applied, so they are
positive for physical forward rotation on every wheel.

Other lines: `READY mecanum_encoder_control` (boot banner), `OK ...`, `ERR ...`,
`WARN velocity timeout`, `DONE,move,<fl_err>,<fr_err>,<rl_err>,<rr_err>` (residual error per
wheel, rad), `STEP,...` / `STEP_TICKS,...` during `test`, `PINS ...` / `MSTYLE ...`.

Safety: velocity mode auto-stops if no new command arrives within
`VELOCITY_TIMEOUT_MS = 500` and prints `WARN velocity timeout`. Position mode is bounded by
its deadline (§7.2).

### 8.3 The ROS contract on top of it

`mecanum_bridge_node` (`ros2/robot_hardware/robot_hardware/mecanum_bridge_node.py`):

| ROS | Direction | Serial |
|---|---|---|
| `/cmd_vel_motor` (`geometry_msgs/Twist`) | in | `m <vx> <vy> <wz>` at up to 20 Hz, resent at least every 0.2 s |
| `/base/move_relative` (`std_msgs/String`, JSON `{dx, dy, dyaw, max_v?}`) | in | `d <dx> <dy> <dyaw>`, preceded by a fresh `ppid` if `max_v` changes this move's cruise cap |
| `/base/move_result` (JSON `{ok, reason, wheel_errors_rad, elapsed_sec}`) | out | from `DONE,move,...` / `ERR move timeout` |
| `/safety/e_stop` (`Bool`) | in | `stop` |
| `/chassis/odom`, `/joint_states`, `/motor/state` | out | from the `STATE` stream |

This is what makes a Nav2-free mission possible: the planner asks for "20 cm forward" or
"45° left" and gets a completion event with per-wheel residuals, instead of running its own
controller.

Note that the primitive is genuinely holonomic: `d 0.30 0.20 -0.79` is one move in which the
robot translates **and** yaws simultaneously, because all three components go through the
same inverse kinematics and produce one synchronized trapezoid. A caller that wants a
mecanum "drift" — slewing to a heading while driving — sends it as one goal rather than as a
rotate followed by a strafe.

### 8.4 Five serial-layer bugs, and their fixes

These cost real field time and are the most transferable part of this document.

1. **The UNO's 64-byte RX buffer overflows right after boot.** Immediately after `READY` the
   sketch spends tens of milliseconds printing ~500 characters of help text, during which
   `loop()` never drains RX. Blasting the six-command configuration burst truncated the
   `pid` line and lost `stream 1` — so odometry silently stopped. Fix: 300 ms settle plus
   60 ms between lines (`mecanum_bridge_node.py:310-317`).
2. **Non-blocking `readline()` returns half lines.** With `timeout=0`, pySerial happily
   returns a fragment. Fragmented `STATE` lines are invisible, but one fragmented
   `DONE,move` left `move_active` latched forever and every later position move was
   rejected. Fix: accumulate raw bytes and dispatch only complete newline-terminated lines
   (`:475-496`).
3. **`termios.error` is not a subclass of `OSError`.** During a USB reconnect race,
   `reset_input_buffer()` raised it and killed the whole node on the field (2026-07-17). It
   is now caught explicitly at every serial call site.
4. **A rebooting Arduino silently loses its tuning.** All firmware configuration lives in
   RAM. The bridge watches for the `READY` banner and re-pushes geometry, both PID sets,
   signs and `stream 1`, and additionally fails any in-flight move with
   `firmware rebooted mid-move` — otherwise `move_active` sticks (`:513-524`).
5. **Two processes opened the same UNO.** On 2026-07-18 a stray differential-drive
   `motor_bridge_node` from an orphaned launch interleaved commands into the firmware
   parser and corrupted it until the USB was replugged. The port is now opened with
   `exclusive=True` (`:240-253`).

---

## 9. OpenRB-150 serial protocol (115200, USB CDC)

Full command set as implemented in `openrb_gripper.ino:802-1002`. Semantics, calibration
values and the grasp-detection logic are in [`gripper-and-mast.md`](gripper-and-mast.md);
this is the wire-level reference.

### 9.1 Gripper commands (Dynamixel ID 0)

| Command | Effect | Reply |
|---|---|---|
| `OPEN` | Move to `FULL_OPEN_DEG` at `OPEN_CURRENT_RAW` | `OK_MOVE_DEG ...` |
| `CLOSE` | Move to `CLOSED_DEG` at `GRIP_CURRENT_RAW` | `OK_MOVE_DEG ...` |
| `SET_DEG <deg>` | Move to an absolute angle, clamped to the safe arc | `OK_MOVE_DEG <deg> TARGET_DEG <ext> RAW <raw> CURRENT_RAW <cur>` |
| `SET_RATIO <0..1>` | Same, as a fraction of the safe arc | as above |
| `GRIP <deg> <current_raw>` | Set the goal current *and* move | as above |
| `SET_CURRENT <raw>` | Set the goal current only (clamped 10..120) | `OK_SET_CURRENT CURRENT_RAW <n>` |
| `STATUS?` | One-shot telemetry line | `STATUS READY <0\|1> POS_RAW .. POS_DEG .. POS_RATIO .. CURRENT_RAW .. VOLTAGE_RAW .. HW_ERROR .. TORQUE .. FAULT .. CURRENT_LIMIT_RAW .. ACTUAL_CURRENT_LIMIT_RAW .. SAFE_ARC_WRAP .. DXL_POWER .. AUTO_TORQUE_OFF_MS 0 ID .. BAUD .. PROTOCOL ..` |
| `STOP` | Torque off | `OK_STOP` / `ERR_STOP` |
| `TORQUE_ON` | Torque on (refused while a fault is latched) | `OK_TORQUE_ON` / `ERR_FAULT_LATCHED` |
| `REBOOT` | Dynamixel reboot + full reconfigure | `OK_REBOOT` / `ERR_REBOOT` |
| `CLEAR_FAULT` | Clear the latched overcurrent / hardware-error fault | `OK_CLEAR_FAULT` |
| `PING` | Ping ID 0, reconfigure if it was not ready | `OK_PING ... MODEL <n>` / `ERR_PING ...` |
| `SCAN [max_id]` | Sweep protocols 1–2 × 5 baud rates × IDs 0..max | `SCAN_FOUND ...`, `SCAN_DONE COUNT <n>` |
| `SET_DXL <id> [baud] [protocol]` | Retarget the gripper servo | `OK_SET_DXL ...` |
| `DXL_POWER_ON` / `DXL_POWER_OFF` / `DXL_POWER_CYCLE` | Servo bus power (no-op unless the board defines `BDPIN_DXL_PWR_EN`); power-off also invalidates the mast home | `OK_DXL_POWER_*` |

Unknown input returns `ERR_UNKNOWN_COMMAND`.

### 9.2 Camera-mast commands (Dynamixel ID 12)

| Command | Effect | Reply |
|---|---|---|
| `LIFT_TO_TOP` | Absolute move to `stroke − top_margin` above home | `OK_LIFT_TO_TOP HOME <h> <present> -> <goal>` |
| `LIFT_TO_MID` | Absolute move to exactly half of the `LIFT_TO_TOP` target (`openrb_gripper.ino:552-555`) | `OK_LIFT_TO_MID ...` |
| `LIFT_TO_BOTTOM` | Absolute return to home | `OK_LIFT_TO_BOTTOM ...` |
| `LIFT_MOVE <ticks>` | Relative move, magnitude capped at `LIFT_MAX_MOVE_TICKS` (40000) | `OK_LIFT_MOVE <present> -> <goal>` / `ERR_LIFT_MOVE_TOO_FAR` |
| `LIFT_SET_HOME` | Re-declare the present position as floor home | `OK_LIFT_SET_HOME <raw>` |
| `LIFT_STOP` | Freeze at the present position | `OK_LIFT_STOP <raw>` |
| `LIFT_TORQUE_OFF` / `LIFT_TORQUE_ON` | Release / re-engage (multi-turn tracking survives torque-off, not power-off) | `OK_LIFT_TORQUE_*` |
| `LIFT_STATUS?` | Telemetry | `LIFT_STATUS READY .. POS_RAW .. CURRENT_RAW .. MOVING .. HW_ERROR .. TORQUE .. HOME_SET .. HOME_RAW .. STROKE ..` |

Also emitted unsolicited: `LIFT_HOME_SET <raw>` the first time the lift initializes in a
Dynamixel power session, and `WATCHDOG_OPEN` whenever the 3 s gripper watchdog fires.

### 9.3 The ROS contract

| ROS topic | Direction | Serial |
|---|---|---|
| `/gripper/command` (`String`: `OPEN`, `CLOSE`, `STOP`, `SET_DEG <deg>`, `LIFT_*`) | in | corresponding command |
| `/lift/command` (`String`: `LIFT_*` only) | in | pass-through |
| `/gripper/state` (JSON) | out | parsed `STATUS` + node-side bookkeeping |
| `/lift/state` (JSON) | out | parsed `LIFT_STATUS` / `LIFT_HOME_SET` |
| `/gripper/grasp` (JSON `{state, pos_gap_deg, current_raw, samples, stamp}`) | out | derived — see [gripper-and-mast.md](gripper-and-mast.md#3-sensorless-grasp-detection) |

---

## 10. Post-assembly verification sequence

Translated from `firmware/arduino_mecanum/README.md`, with the pin table corrected.

1. `pins` — compare the printed map against your wiring.
2. `test 100 800` — wheels must fire in FL → FR → RL → RR order.
   - Wrong order → driver channels are swapped.
   - `STEP_TICKS` = 0 → that encoder is not wired (or its A line is dead).
   - Negative ticks on a forward step → that encoder's A/B are swapped; fix with `sign`.
   - Wheel spins backwards → fix that wheel's motor sign with `sign`.
3. `geom <r> <half_l> <half_w>` with your measured values, then `m 0.1 0 0` — all four
   wheels must rotate forward.
4. `m 0 0.1 0` — the robot must strafe **left** (FL and RR backward, FR and RL forward). If
   it moves diagonally, the mecanum wheels are not in X configuration.
5. `d 0.2 0 0` — 20 cm forward. Measure it with a tape and check the `DONE,move` residuals.

Then continue with the numbered chain in [`../bringup_tools/`](../bringup_tools/) to fit the
effective geometry and the PID gains.
