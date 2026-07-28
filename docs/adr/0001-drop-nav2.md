# ADR-0001 — Drop Nav2

**Status:** Accepted (2026-07-04). Shipped in the competition build.

## Context

The task is a 4 m × 4 m empty square with a known map, a 3-minute round, and a Jetson Orin
Nano that is simultaneously running two 1080p camera streams and YOLOv8-seg inference. We
started on the standard stack: Nav2 + AMCL + `robot_localization`, with an RViz-driven
workflow. In that arena a full scan match cost **345–449 ms**. The pose estimate jumped
between updates, and the controller bent the path chasing it. Nav2 also drags in costmaps,
planner/controller/behaviour servers, lifecycle management and a full TF tree — all of it
solving problems (unknown maps, arbitrary obstacle fields, kinematic planning around
clutter) that this arena does not have.

## Decision

Replace the entire navigation stack with one rclpy node,
`navigation/ros2/arena_lightweight_control/arena_lightweight_control/arena_control_node.py`.

- Localise with `WallRangeLocalizer` (`map_localization.py:330`): the arena is a rectangle,
  so the expected range of a beam is a closed-form ray/rectangle intersection. 16–24
  downsampled beams, clamped residuals, trimmed mean (worst 30 % dropped), ~200 candidates
  per scan.
- Feed gyro-z forward into yaw between scans so yaw is *known*, which makes the square's
  90°-degenerate x/y solution unique and allows a **global** re-solve every scan instead of
  a local window that can slide onto a neighbouring optimum.
- Emit `Twist` on `/cmd_vel_direct` from a rotate-then-drive goal controller (mecanum
  variant on competition day). No `map → odom` TF, no AMCL, no particle filter, no costmap,
  no planner, no RViz.

## Consequences

**Benefits.** Localisation runs in **8–10 ms offline, ~41 ms on the Jetson** against a 50 ms
budget. The hot path is pure Python stdlib — no numpy — so it is trivially testable: 34 unit
tests cover the matchers, both controllers and the goal latch, and they run without ROS. The
operator console is one embedded HTML string served by `http.server`. The whole navigation
stack is one file you can read in an afternoon.

**Costs, paid in full.**

- **No planner and no obstacle avoidance.** Routing between the grid and the storage zone is
  hand-written ("street router"). Its last-resort `street_forced` mode was measured in
  simulation producing 11 grid-cell intrusions at 0.000 m clearance when all 28 cells are
  occupied. Nav2 would have handled that case.
- **It only works in a known rectangle.** The matcher's speed comes from exploiting the wall
  geometry analytically. Nothing here transfers to an unknown or non-convex map.
- **It requires an IMU.** Without gyro yaw the square's symmetry returns and the node falls
  back to a fragile ±0.12 m / ±0.14 rad local window.
- **Hand-written matchers have hand-written bugs.** Short lidar returns were originally
  skipped as occlusions, which let a wrong candidate win a global search by discarding every
  beam that disagreed with it — a synthetic 65 cm error. Nothing can occlude a wall at a
  0.32 m scan plane above 8 cm objects, so short returns are now penalised instead (2 cm).
- **No TF tree** means camera and gripper geometry live as measured constants in
  `perception/fieldlib.py`, duplicated by hand into the simulator.
- **Nothing off-the-shelf composes with it.** No Nav2 plugin, RViz panel or third-party
  planner can be dropped in later.
