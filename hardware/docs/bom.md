# Bill of materials — TEMPLATE

> ⚠️ **This is a template. Every "Approx cost" cell says `TODO` and must be filled in by the
> team.** The component list below is derived from the firmware, the ROS configuration, the
> URDF and the project's hardware notes, so the *parts* are accurate; the *prices* are not
> recorded anywhere in the repository and were never captured. Fill in currency, unit price
> and purchase date before treating this as a real BOM.
>
> Fields marked **(unverified)** are parts we can identify by family but not by exact model
> number from any file in the repository. Confirm against the physical robot before ordering.

Quantities are per robot. Cost column: enter unit price, and note the currency.

---

## Compute and control

| Component | Part | Qty | Notes | Approx cost |
|---|---|---:|---|---|
| Onboard computer | NVIDIA Jetson Orin Nano Developer Kit | 1 | Runs the entire ROS 2 Humble stack and both YOLO stages. All three microcontrollers and both cameras hang off its USB. Exact module variant (4 GB / 8 GB) **(unverified)** | TODO |
| Motion controller | Arduino UNO (R3) | 1 | 4 encoders, 50 Hz velocity PID, cascaded position loop. Firmware `hardware/firmware/arduino_mecanum/` | TODO |
| Servo controller | ROBOTIS OpenRB-150 | 1 | Gripper + camera mast on one Dynamixel bus. Firmware `hardware/firmware/openrb_gripper_mast/` | TODO |
| microSD / storage | — | 1 | For the Jetson **(unverified — size and type not recorded)** | TODO |

## Drivetrain

| Component | Part | Qty | Notes | Approx cost |
|---|---|---:|---|---|
| Gearmotor with encoder | JGB37-520, 12 V, 333 rpm, 1:30 gearbox, integrated quadrature hall encoder | 4 | 1320 counts/rev at the output shaft (11 PPR × 4 decode × 30). **All four must have encoders** — the position loop is per-wheel. No-load speed measured ≈ 34.9 rad/s at the wheel | TODO |
| Motor driver | Cytron MDD10A dual-channel | 2 | 2 channels each, 10 A continuous / 30 A peak per channel, 5–30 V motor supply, PWM + DIR sign-magnitude inputs, no EN/STBY pin. On-board test buttons are useful for pre-Arduino verification | TODO |
| Mecanum wheels | 80 mm diameter mecanum set (2 left-hand + 2 right-hand) | 4 | Must be mounted in **X configuration**. Effective contact radius fitted at 0.0388 m. A 68 mm set was also procured as a spare/alternative — the URDF generator supports both | TODO |
| Wheel hubs / couplers | 6 mm D-shaft hub to match the JGB37-520 output **(unverified)** | 4 | — | TODO |

## Sensors

| Component | Part | Qty | Notes | Approx cost |
|---|---|---:|---|---|
| Camera (top) | Intel RealSense D435 | 1 | Far/top view of the stitched pair. Rides the mast | TODO |
| Camera (near) | Intel RealSense D435i | 1 | Near/down view. **Its IMU is the robot's gyro** (`enable_bottom_imu`, remapped to `/imu/data`) — the localizer needs it to disambiguate the square arena's 90° symmetry | TODO |
| LiDAR | Slamtec RPLIDAR A2M12 | 1 | 256000 baud. Mounted at `base_link` z = 0.262 m with yaw = π (0° points to the robot's rear) | TODO |
| USB 3 cables | USB-C to USB-A/C, RealSense-grade | 2 | Two D435-class cameras on one Jetson need a custom `librealsense` RSUSB build — see `scripts/build_realsense_rsusb.sh` | TODO |

## Manipulator

| Component | Part | Qty | Notes | Approx cost |
|---|---|---:|---|---|
| Gripper servo | ROBOTIS Dynamixel XC330 (ID 0) | 1 | Current-based position mode, current limit 120 raw. Exact variant (e.g. XC330-M288 vs T288) **(unverified)** | TODO |
| Mast lift servo | ROBOTIS Dynamixel XC330 (ID 12) | 1 | Multi-turn, 31641-tick (148.9 mm) stroke, ~7.7 turns. Exact variant **(unverified)** | TODO |
| Dynamixel cables | 3-pin JST (X-series) | 2+ | Daisy-chained: OpenRB → ID 0 → ID 12 | TODO |
| Parallel gripper mechanism | Custom — jaws, linkage, servo horn | 1 set | 0–70 mm reported jaw span. Source CAD is not published | TODO |
| Camera mast + lift | Custom — vertical rail/lead-screw assembly carrying both cameras | 1 set | 148.9 mm measured stroke; the top bracket sags ~0.95° when raised, which the perception calibration models | TODO |

## Power and wiring

| Component | Part | Qty | Notes | Approx cost |
|---|---|---:|---|---|
| Battery | 12 V pack | 1 | Chemistry and capacity **not recorded anywhere in the repository** — fill in from the physical robot | TODO |
| Jetson power | Regulator / dedicated supply for the Jetson **(unverified)** | 1 | The Jetson is not powered from the 12 V motor rail in any documented way | TODO |
| Main power switch / fuse | Kill switch on the 12 V rail | 1 | Rating and fusing **(unverified — not documented)**. Operationally it is load-bearing: the Jetson must boot with this switch **off**, or the OpenRB-150 does not enumerate reliably ([wiring-and-firmware.md](wiring-and-firmware.md#boot-the-jetson-with-the-12-v-kill-switch-off)) | TODO |
| Power distribution | Battery → 2 × MDD10A VM in parallel; common ground bus | 1 set | **Common ground is mandatory**: battery ↔ both MDD10A ↔ Arduino ↔ Jetson (via USB). A missing common ground was the suspected cause of a 2026-06-26 motor fault. Check branch-wire gauge and connector rating against the stall current | TODO |
| Motor / power connectors | XT60 or equivalent **(unverified)** | — | — | TODO |
| Encoder wiring | 4-conductor per motor (A, B, 5 V, GND) | 4 | Encoder VCC comes from the **Arduino 5 V rail**, never from motor power | TODO |
| USB cables | Arduino UNO (USB-B), OpenRB-150 (USB **(unverified connector)**), RPLIDAR | 3 | — | TODO |
| Powered USB hub | Self-powered, for the UNO and the OpenRB-150 | 1 | Not optional in practice: both boards dropped their ttys intermittently when plugged straight into the Orin Nano, and were noticeably more stable behind the hub ([wiring-and-firmware.md §2](wiring-and-firmware.md#2-boards-buses-and-who-owns-what)). Its port topology is what the default `by-path` device paths encode | TODO |

## Structure

| Component | Part | Qty | Notes | Approx cost |
|---|---|---:|---|---|
| Lower deck plate | 190 × 250 × 6 mm | 1 | Height 0.173 m above ground (URDF collision box) | TODO |
| Middle deck plate | 190 × 210 × 6 mm | 1 | Height 0.258 m | TODO |
| Upper deck plate | 190 × 170 × 6 mm | 1 | Height 0.333 m | TODO |
| Front / side panels | see URDF collision boxes | — | `hardware/cad/urdf/robot_mk3_mecanum_sim_80mm.urdf` | TODO |
| Standoffs, brackets, fasteners | M3 assortment **(unverified)** | — | — | TODO |
| Motor mounts | JGB37-520 brackets | 4 | — | TODO |

Modelled chassis mass is 2.4 kg (simulation value in the URDF, not a weighed figure).

## Tools and consumables for bring-up

| Item | Why | Approx cost |
|---|---|---|
| Blocks / stand to lift the wheels clear of the floor | Bring-up steps 10–30 must run with the wheels free | TODO |
| Tape measure and masking tape | Effective wheel radius and rotation geometry are fitted against measured travel | TODO |
| `arduino-cli` with the `arduino:avr` core | `hardware/bringup_tools/00_flash_firmware.sh` | free |
| ROBOTIS DYNAMIXEL Wizard 2.0 | Used to measure the gripper's closed/open angles (36.74° / 214.37°, 2026-07-16) | free |

---

## Notes for whoever fills this in

1. Add a currency and a date column — component prices for this class of parts moved
   noticeably during 2026.
2. Record the **exact** XC330 variant and the battery specification from the physical robot;
   they are the two most consequential unknowns in this list.
3. The Fusion 360 source CAD is not published. `hardware/cad/meshes/` and the URDF
   carry the geometry the code actually uses; the mechanical parts would have to be
   remodelled from those and from the dimensions in this list.
4. If you substitute the motor driver, check §5 of
   [`wiring-and-firmware.md`](wiring-and-firmware.md#5-motor-driver-wiring-cytron-mdd10a--2)
   first: the firmware supports both sign-magnitude and inverted-PWM drivers, but only via
   the runtime `mstyle` command, and the shipped defaults assume MDD10A.
