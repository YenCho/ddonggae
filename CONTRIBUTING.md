# Contributing

This is the public release of a robot that competed once, on 2026-07-22..24. It is a **frozen
record**, not an actively developed product: the mission stack here was recovered from the
competition Jetson, so the code is the code that ran, and the value of the repository is that the
two match. Contributions are welcome where they make that record more
usable — fixes to documentation, portability patches, missing measurements — and are not welcome
where they rewrite history.

## Build

The ROS 2 packages are not under a single `src/`; they sit next to the subsystem they belong to.

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --base-paths hardware/ros2 navigation/ros2 third_party
source install/setup.bash
```

ROS 2 **Humble** is the only supported distribution. Do not add Jazzy sourcing to a script or launch
file unless the caller opts in explicitly (`ros_distro:=jazzy` / `--ros-distro jazzy`).

## Test

```bash
python3 -m pytest tests/ -q                              # full suite, needs ROS 2 sourced
python3 -m pytest tests/ -q --ignore=tests/test_controllers.py   # 47 tests, no ROS needed
python3 mission/match_runner.py --offline                # mission-logic self-test, no ROS, no weights
```

`conftest.py` puts `perception/` and the three package directories on `sys.path`, so the tests run
without an installed workspace — except `tests/test_controllers.py`, which imports `rclpy`.

Any change to the router, the target-selection logic, the localiser or the controllers must keep
both the pytest suite and `--offline` green. If you fix a field bug, add the regression case: the
28-object router regression in `--offline` exists because a detour branch once computed a waypoint
and never checked it, driving a carry route through six obstacle cells.

## Comments: translate, never delete

Most of the source is commented in Korean, and those comments are frequently the **only** record of
why a constant has the value it has — which day it was measured, against what reference, and what
broke before it. For example, `perception/fieldlib.py:41-54` records not just the four camera mount
poses but their fit residuals, their tape-measure cross-checks, and the fact that the earlier
"bottom camera is chassis-fixed" conclusion was an invalid measurement.

So:

- **Do not delete a Korean comment.** If you translate it, keep every number, date and caveat.
- **Do not "clean up" a constant** whose comment explains a field observation. `FACE_IMGSZ = 224`
  and BGR input are contracts, not preferences, and both fail silently when violated.
- If you change a value, say in the comment *when* and *against what measurement* — that is the
  convention the rest of the file follows.

## Documentation

- **New documentation is English only.** The Korean originals stay where they are, in the private
  development repository; this repository is the English record.
- Prefer a number and its provenance over an adjective. Cite files as
  `perception/fieldlib.py:67`, relative to the repository root.
- If you cannot verify something, write "not measured" or leave it out. Do not estimate.
- Publish negative results plainly. Several pages here exist only to record something that did not
  work; that is deliberate.

## What does not belong in this tree

- **Model weights.** They are Ultralytics YOLO derivatives (AGPL-3.0) and ship as GitHub Release
  assets — `scripts/fetch_models.sh` fetches them. `.gitignore` already excludes `*.pt`, `*.onnx`
  and `*.engine`.
- **Datasets, captures and raw run logs.** Large artefacts live outside the repository.
- **Absolute personal paths.** Compute paths from the repository root
  (`Path(__file__).resolve().parents[...]`, or `REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"`),
  and use an environment-variable override for anything outside it.
- **Anything that identifies a non-team member.** Check photographs before adding them under
  `media/`.

## Pull requests

One topic per pull request. Say what you measured, on what hardware, and what would falsify it.
A PR that changes a tuned constant without a measurement will be asked for one.
