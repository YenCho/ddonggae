# Mecanum bring-up scripts

Run these in numeric order on a freshly built chassis. Each one ends by printing
the values you should write into `hardware/ros2/robot_bringup/config/real.yaml`
or push into the firmware — that is the point of the whole chain: you finish it
holding a calibrated robot, not a pile of notes.

| Step | Command | Preconditions |
|---|---|---|
| 0. Flash firmware | `bash hardware/bringup_tools/00_flash_firmware.sh` | USB connected, ROS bridge **stopped** |
| 1. Wheels / encoders / signs | `python3 hardware/bringup_tools/10_wheel_selftest.py` | Robot **on blocks**, bridge stopped |
| 2. Measure encoder CPR | `python3 hardware/bringup_tools/20_encoder_cpr_calib.py` | On blocks, bridge stopped |
| 3. Tune wheel PID | `python3 hardware/bringup_tools/30_pid_step_tune.py` | On blocks, bridge stopped |
| 3b. Verify PID on the floor | `python3 hardware/bringup_tools/31_pid_floor_verify.py` | On the floor, bridge stopped |
| 4. Determine LiDAR orientation | `python3 hardware/bringup_tools/40_lidar_orientation_check.py` | LiDAR driver **running** |
| 5. Motion accuracy (odometry) | `python3 hardware/bringup_tools/50_move_accuracy.py` | On the floor, ROS bridge **running** |
| 5b. Motion accuracy (LiDAR-referenced) | `python3 hardware/bringup_tools/51_move_accuracy_lidar.py` | On the floor, bridge running |
| 5c. Single-wall pose check | `python3 hardware/bringup_tools/52_single_wall_verify.py` | Facing a flat wall, bridge running |
| 5d. Maximum speed | `python3 hardware/bringup_tools/53_max_speed.py` | Clear floor, bridge running |

## The one thing that trips people up

**Steps 0–3 talk to the firmware over the serial port directly, so
`mecanum_bridge_node` must NOT be running** — two processes cannot own the same
tty. Steps 4 onward are the opposite: they need the ROS bridge up.

If a script hangs waiting for a serial reply, this is almost always why.

## Notes

- Steps 1–3 want the robot **on blocks** with the wheels free. Step 1 reverses
  each wheel individually; on the floor it will simply drive away.
- Step 1 is where you fix polarity. The firmware's `sign` command flips motor
  and encoder direction per wheel without reflashing, so correct it there rather
  than rewiring.
- Step 3b exists because PID gains tuned with unloaded wheels are optimistic.
  Verify them under real floor friction before trusting them.
- Step 4 matters more than it looks: a LiDAR mounted 180° out produces a map
  that is entirely self-consistent and completely wrong.

See `hardware/docs/wiring-and-firmware.md` for the pin map and the full serial
protocol, and `docs/05-field-runbook.md` for the competition-day procedure.
