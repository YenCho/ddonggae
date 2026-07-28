# ADR-0003 — Firmware owns the motion primitives

**Status:** Accepted (2026-07-14, with the mecanum conversion). Shipped.

## Context

Grasping needs centimetre-scale relative moves: "advance 0.34 m", "yaw −45°", "creep 3 cm".
The first generation did this the obvious way — a host-side PI loop in the ROS node,
streaming raw PWM (`p <l> <r>`) to the Arduino. That loop shares a Python GIL with YOLOv8
inference and two 1080p camera callbacks, and rides a 115200 baud USB serial link. Its
period is whatever the Jetson has left over. A position loop closed under those conditions
is not a position loop.

## Decision

Move both loops into the Arduino UNO firmware
(`hardware/firmware/arduino_mecanum/mecanum_encoder_control.ino`):

- `m <vx> <vy> <wz>` — body velocity, per-wheel velocity PID with feedforward.
- `d <dx> <dy> <dyaw>` — a **synchronised trapezoidal position profile**
  (`startPositionMove`, line 401): the longest-travel wheel sets the timebase and the other
  three are scaled proportionally so all four start and stop together. Replies `OK move`,
  then `DONE,move,<4 wheel errors>` when settled, or `ERR move timeout`.

The host exposes this as `/base/move_relative` (JSON) → `/base/move_result` and does not
close any loop itself. All tuning — geometry, velocity PID, position PID, per-wheel motor
and encoder polarity, and even the driver's PWM encoding style — is pushed over serial at
connect and **re-pushed whenever the firmware's `READY` banner appears**, so a brownout or
USB re-enumeration self-heals instead of silently reverting to default gains.

## Consequences

**Benefits.** Move accuracy is independent of host load — inference spikes cannot deform a
trajectory. Bring-up needs no toolchain: `sign` and `mstyle` fix reversed wiring and a
different motor driver at runtime, so a wiring mistake costs a serial line, not a reflash.
The mission planner gets a clean request/ack primitive it can await.

**Costs.**

- **The primitive is blind.** The firmware knows nothing about localisation, the map or
  obstacles; a live `cmd_vel` has to preempt an in-flight move. Any safety reasoning must
  happen before the command is sent.
- **Deadlines live on the wrong side of the link, and mis-tuning them lost objects.** The
  move timeout is firmware-side. At 2.5 s, real 0.3–0.4 m approach moves under floor load
  timed out mid-travel; the runner proceeded to close the gripper on empty air. Raised to
  5.0 s on 2026-07-19 — the reasoning is preserved in the source comment at line 444.
- **Remote state can desynchronise.** On 2026-07-15 a fragmented `DONE,move` line (non-
  blocking `readline()` returning half a line) left `move_active` stuck true, and **every
  subsequent position move was rejected** for the rest of the session. The bridge now
  accumulates bytes and dispatches only complete lines.
- **The config burst has to respect an 8-bit MCU.** Right after boot the UNO spends tens of
  milliseconds printing ~500 characters of help text without draining RX; blasting the
  six-command config burst overflowed its 64-byte buffer, truncated the `pid` line and lost
  `stream 1`, and odometry silently stopped. Fix: 300 ms settle plus 60 ms inter-line pacing
  (`mecanum_bridge_node.py:277-317`).

Every one of those failures is a consequence of the split. They were still cheaper than a
position loop at the mercy of the GIL.
