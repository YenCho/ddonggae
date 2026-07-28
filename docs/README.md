# Documentation index

Every document in this repository, with one line on what it is for. Start at
[`../README.md`](../README.md) if you have not read it; start at
[`07-results-and-lessons.md`](07-results-and-lessons.md) if you only read one page.

---

## Numbered guides — read in order, or by need

| Document | Purpose |
|---|---|
| [`01-competition-and-rules.md`](01-competition-and-rules.md) | The constraints every design decision answers to: arena, 42-point grid, both object sets, the scoring table with its 2× mispick penalty, and a rule → code → file map. |
| [`02-architecture.md`](02-architecture.md) | The four layers (firmware / device bridges / arena control / mission script), why the boundaries are where they are, the match state machine, and why there are no custom ROS messages. |
| [`03-ros2-interfaces.md`](03-ros2-interfaces.md) | The exhaustive sensor → bridge → actuator graph: every node, topic, message type, QoS profile, JSON payload schema and serial line. Written from the code. |
| [`04-getting-started.md`](04-getting-started.md) | Install, build, fetch weights, run the tests, launch the stack, run a match — plus §10, the four classes of artefact that are ours and do not transfer. |
| [`05-field-runbook.md`](05-field-runbook.md) | Competition-day procedure as operated: two-phase bringup, health checks, pose seeding, the match invocation, symptom → action, shutdown. |
| [`06-troubleshooting.md`](06-troubleshooting.md) | Symptom → first hypothesis → action, and the traps that cost the most time (wrong `imgsz`, RGB/BGR, duplicate bridges, a lost mast home, silently no-op speed profiles). |
| [`07-results-and-lessons.md`](07-results-and-lessons.md) | The record: the score, every measured number with where it was measured, the three competition failures, four negative results, ten transferable lessons. |
| [`08-reproducibility.md`](08-reproducibility.md) | What can and cannot be reproduced from this repository, and what has to be re-measured on your own robot and at your own venue. |
| [`CREDITS.md`](CREDITS.md) | The seven team members, their GitHub accounts, how the work split, and the AI tooling disclosure. |

## Architecture decision records

| Record | Decision |
|---|---|
| [`adr/README.md`](adr/README.md) | Index and format note — every record ends in costs, including the ones paid publicly. |
| [`adr/0001-drop-nav2.md`](adr/0001-drop-nav2.md) | Drop Nav2 for a hand-written wall-range localiser, and what that gives up. |
| [`adr/0002-json-over-std-msgs-instead-of-custom-messages.md`](adr/0002-json-over-std-msgs-instead-of-custom-messages.md) | JSON on `std_msgs/String` instead of an interface package — and the misread key that made every successful grasp score as a failure. |
| [`adr/0003-firmware-owns-motion-primitives.md`](adr/0003-firmware-owns-motion-primitives.md) | Wheel PID and synchronised trapezoidal moves live in the Arduino, not the host. |
| [`adr/0004-simulation-is-kinematic-only.md`](adr/0004-simulation-is-kinematic-only.md) | Isaac Sim is a perception and mission-logic surrogate; physics was validated on the robot. |
| [`adr/0005-synthetic-only-training-data.md`](adr/0005-synthetic-only-training-data.md) | Zero hand-labelled images. It survived a rule change four days out and lost the qualifiers to an under-ripe apple. |

---

## Subsystem documentation

### Hardware

| Document | Purpose |
|---|---|
| [`../hardware/README.md`](../hardware/README.md) | What the robot is, the three-microcontroller rules, the rebuild order, and which constants are *fitted* rather than physical. |
| [`../hardware/docs/bom.md`](../hardware/docs/bom.md) | Bill of materials, quantities and substitution notes. |
| [`../hardware/docs/wiring-and-firmware.md`](../hardware/docs/wiring-and-firmware.md) | Pin map, motor drivers, the complete serial grammar of both boards, the PID cascade, and five serial-layer bugs fixed in the field. |
| [`../hardware/docs/gripper-and-mast.md`](../hardware/docs/gripper-and-mast.md) | Sensorless grasp detection by settled position gap, limit-switch-free mast homing, and the one-process-owns-the-tty rule. |
| [`../hardware/bringup_tools/README.md`](../hardware/bringup_tools/README.md) | The numbered 00..53 bring-up chain (original Korean). |
| [`../hardware/firmware/arduino_mecanum/README.md`](../hardware/firmware/arduino_mecanum/README.md) | Original Korean firmware notes; its pin table is stale — see `wiring-and-firmware.md`. |

### Perception

| Document | Purpose |
|---|---|
| [`../perception/README.md`](../perception/README.md) | The two-stage pipeline in one picture, the two non-negotiable runtime contracts, what ships, and three negative results. |
| [`../perception/docs/pipeline.md`](../perception/docs/pipeline.md) | Stage by stage, every threshold with the observation that set it, and measured timings. |
| [`../perception/docs/synthetic-data.md`](../perception/docs/synthetic-data.md) | How the training set was rendered in Blender, and the three failures that shaped it — including the shared-seed leak. |
| [`../perception/docs/models.md`](../perception/docs/models.md) | Weight standard, inference rules, the Jetson `.engine`/`.pt` fallback, and the deprecation list. |
| [`../perception/docs/stitching-and-calibration.md`](../perception/docs/stitching-and-calibration.md) | The exact-inverse dual-camera stitch, resolution-independent calibration, and the venue photometry lock. |
| [`../perception/docs/grid-voting.md`](../perception/docs/grid-voting.md) | Snapping detections to the 42-cell grid and the asymmetric fruit vote. |

### Navigation

| Document | Purpose |
|---|---|
| [`../navigation/README.md`](../navigation/README.md) | Why Nav2 was deleted, the measurement that decided it, the before/after component table, and what you give up. |
| [`../navigation/docs/localization.md`](../navigation/docs/localization.md) | The wall-range matcher, the IMU yaw prior, the square arena's 4-fold symmetry, and latency handling. |
| [`../navigation/docs/control-and-routing.md`](../navigation/docs/control-and-routing.md) | Goal latch, mecanum controller, speed profiles, and the street router. |

### Mission and simulation

| Document | Purpose |
|---|---|
| [`../mission/README.md`](../mission/README.md) | How to run a match, the full CLI reference, what each stage does, the outputs, and the exit codes. |
| [`../mission/docs/match-strategy.md`](../mission/docs/match-strategy.md) | Why the scan happens where it does, why ranging comes from depth, the storage geometry, and what we would change. |
| [`../simulation/README.md`](../simulation/README.md) | The Isaac Sim kinematic surrogate: its topic contract, what it is good for, and what it must never be trusted for. |
| [`../simulation/docs/field-parity.md`](../simulation/docs/field-parity.md) | The sim-to-field parity study, including the three renderer traps that silently invalidate results. |

### Repository meta

| Document | Purpose |
|---|---|
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | How to build, how to test, the translate-never-delete comment policy, and English-only for new documentation. |
| [`../NOTICE`](../NOTICE) | Third-party components, their licences, and what is deliberately not redistributed. |
| [`../CITATION.cff`](../CITATION.cff) | Citation metadata. |
| [`../media/README.md`](../media/README.md) | What team-supplied media goes where; each subdirectory names its expected files. |
