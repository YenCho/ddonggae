# Mecanum Encoder Control

Closed-loop mecanum firmware for 4 motors and 4 encoders. It replaced the
open-loop `mecanum_pwm_control` sketch and supports both a wheel **velocity**
PID and a **position** PID (cascaded, with a synchronised trapezoidal profile).

This is the sketch that was flashed for the competition.

Wiring reference: `hardware/docs/wiring-and-firmware.md`.
The ROS side is `mecanum_bridge_node` in `hardware/ros2/robot_hardware/`.

## Pin map (Arduino UNO)

| Wheel | DIR | PWM | ENC A | ENC B |
|---|---:|---:|---:|---:|
| front_left | D4 | D5 | D8 | D9 |
| front_right | D7 | D6 | A0 | A1 |
| rear_left | D2 | D3 | A2 | A3 |
| rear_right | D12 | D10 | A4 | A5 |

- D0/D1 are reserved for USB serial. D11 and D13 are spare.
- The two front motor lines (D4–D7) and the front_left encoder (D8/D9) are
  unchanged from the earlier differential-drive robot, which is why they look
  out of pattern.
- All four PWM pins are hardware PWM (Timer0: D5/D6, Timer2: D3, Timer1: D10).
- Encoders are decoded with pin-change interrupts plus a quadrature state
  table, so they do **not** need the external interrupt pins (D2/D3). That is
  what makes four encoders fit on an UNO at all.

> **Open item.** The rear-right encoder pin assignment is inconsistent between
> the sketch header comment (A4), the constant in the code (D11, annotated
> "was A4 — dead line"), and `real.yaml` (which notes "RR is A4, broken line
> repaired"). Confirm against the physical robot before trusting the table above
> for that wheel.

## Serial protocol (115200 baud)

```text
m <vx_mps> <vy_mps> <wz_rad_s>          body velocity (velocity mode)
w <fl> <fr> <rl> <rr>                   wheel rad/s (velocity mode)
d <dx_m> <dy_m> <dyaw_rad>              relative displacement (position mode)
pw <fl_rad> <fr_rad> <rl_rad> <rr_rad>  wheel position deltas (position mode)
p <fl> <fr> <rl> <rr>                   direct signed PWM, -255..255
pid <kp> <ki> <kd> <min_pwm> <max_pwm> <ff_slope> <int_limit>
ppid <kp> <kd> <max_rad_s> <accel> <tol_rad> <hold_ms> <min_rad_s>
geom <wheel_radius_m> <half_length_m> <half_width_m>
sign <m_fl> <m_fr> <m_rl> <m_rr> <e_fl> <e_fr> <e_rl> <e_rr>
z / stop / stream <0|1> / pins / test [pwm] [ms] / ?
```

Coordinate frame: `+x` forward, `+y` left, `+wz` counter-clockwise (ROS REP-103).

- `d` is acknowledged immediately with `OK move`, then on completion emits
  `DONE,move,<fl_err>,<fr_err>,<rl_err>,<rr_err>` — the residual error per wheel
  in radians. On overrun it emits `ERR move timeout`.
- **Velocity mode stops automatically after 500 ms without a new command.**
  Position mode is protected by a profile-time-based deadline instead.
- `sign` flips motor and encoder polarity per wheel at runtime, so polarity
  mistakes are fixed without reflashing or rewiring.
- STATE line, 10 Hz:
  `STATE,<ms>,<fl_t>,<fr_t>,<rl_t>,<rr_t>,<fl_pwm>,<fr_pwm>,<rl_pwm>,<rr_pwm>,<mode>`
  where mode is 0 = idle/pwm, 1 = velocity, 2 = position.

## How position mode works

- **Outer loop:** per-wheel position error → target velocity (`posKp`, `posKd`),
  with trapezoidal-profile feedforward. All four wheels are made to start and
  finish together by taking the longest-travelling wheel as the reference and
  scaling the other three proportionally.
- **Inner loop:** the existing velocity PID, including its `min_pwm`
  feedforward, unchanged.
- If residual error after the profile ends exceeds `tol_rad`, the target
  velocity is floored at `min_rad_s`. This exists to prevent a specific failure
  we hit on the differential-drive robot: near the PWM deadband a wheel would
  simply stall a few millimetres short and never converge (the "final-yaw
  stall").
- `tol_rad` defaults to 0.08 rad in the firmware; the competition config set it to 0.30 rad ≈ 11.7 mm at the 0.0388 m effective wheel radius.

## Bring-up verification order (straight after assembly)

1. `pins` — print the pin map and check it against the actual wiring.
2. `test 100 800` — drive each wheel forward and back in turn. Use the
   `STEP_TICKS` output to confirm every encoder is alive and correctly signed.
   - If ticks are not positive when the wheel turns forward, invert that
     wheel's **encoder** sign.
   - If the wheel turns the wrong way, invert that wheel's **motor** sign.
   Both via the `sign` command.
3. `geom 0.034 0.15 0.125` (substitute your measured values), then `m 0.1 0 0` —
   all four wheels should drive forward.
4. `m 0 0.1 0` — strafe left. Correct behaviour is FL and RR reversing while
   FR and RL drive forward.
5. `d 0.2 0 0` — a 20 cm position move; check the residual error reported by
   `DONE,move`.
