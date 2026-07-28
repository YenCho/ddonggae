# Control and routing

> Paths are relative to the repository root. `ALC/` abbreviates
> `navigation/ros2/arena_lightweight_control/arena_lightweight_control/`.

Once the robot knows where it is ([`localization.md`](localization.md)), everything else is: turn a
goal into a `Twist`, and choose goals that do not drive through objects. Those two jobs live in
different places on purpose.

| layer | file | role |
|---|---|---|
| controller | `ALC/controllers.py` | goal `Pose2D` → `(vx, vy, ω)`, 20 Hz, no knowledge of objects |
| goal handling | `ALC/arena_control_node.py:890-988, 1044-1128` | latch, obstacle stop, idle silence, publishing |
| router | `mission/match_runner.py:284-431` | pure functions: object list + start + end → waypoint list |

The router is deliberately **not** in the ROS node. It has no ROS imports, it is exercised offline by
`match_runner.py --offline`, and every routing decision it makes was replayed against the real
ground-truth object layout before it ever ran on the robot.

---

## The direct path: `/cmd_vel_direct`

There is no action server, no `NavigateToPose`, no feedback protocol. A goal is a JSON string:

```bash
ros2 topic pub --once /arena_lightweight/goal std_msgs/String \
  '{data: "{\"x\": 1.20, \"y\": -0.75}"}'
```

`goal_command_callback` (`ALC/arena_control_node.py:1053-1063`) parses it, `set_goal` stores it, and
the 20 Hz `control_tick` (`:890-988`) turns it into a `geometry_msgs/Twist` on `/cmd_vel_direct`.
Progress is observed by reading `/arena_lightweight/status` (20 Hz JSON in the competition launch: pose, phase, match score,
latency) and comparing the pose against the goal — that is exactly what
`FieldNode.goto_xy_goal()` does in `perception/fieldlib.py:618-663`.

`/cmd_vel_direct` is the **highest-priority input of `cmd_vel_mux`**, which has an important
consequence the node has to handle explicitly.

### Idle silence

If the node published a zero `Twist` forever while idle, `/cmd_vel_direct` would permanently
outrank the teleop input on `/cmd_vel` and the robot would become undrivable by hand. So after a goal
finishes, the node publishes zeros for only `idle_cmd_publish_sec = 0.6 s` — long enough to guarantee
a brake — and then goes **silent**, letting the mux time out and fall through to lower-priority
sources (`ALC/arena_control_node.py:966-975`). Found on 2026-07-19 by not being able to teleop the
robot after a goal.

### Scan timeout

If no scan has been processed within `scan_timeout_sec = 0.5 s`, the controller is not run at all and
the phase becomes `scan_timeout` with a zero command (`:935-936`). Driving on a stale pose is worse
than not driving.

---

## The goal message accepts an optional `yaw`

```jsonc
{"x": 1.20, "y": -0.75}                 // position only
{"x": 1.20, "y": -0.75, "yaw": -2.356}  // position AND heading
```

`yaw` is resolved by `resolve_goal_yaw()` (`ALC/arena_control_node.py:1355-1364`):

| request | `default_goal_yaw_rad` | result |
|---|---|---|
| `yaw` present | — | that value |
| `yaw` absent | `NaN` (the default) | `None` — no final heading; the robot stops on whatever heading it has |
| `yaw` absent | finite | that value |

`default_goal_yaw_rad` defaults to `NaN` deliberately. An earlier build defaulted it to +90° ("always
settle facing north"), which meant every goal ended with a pointless in-place rotation. Rotation is
the single most expensive thing this chassis does — both in seconds and in localisation quality — so
the default became "do not rotate unless asked" on 2026-07-19 (`:72-76`).

**What `yaw` buys you on a mecanum chassis is simultaneity.** `MecanumController.compute_command`
(`ALC/controllers.py:126-171`) computes the holonomic translation *and*, in the same command, a yaw
rate from `goal_yaw_command()` — so `vx`, `vy` and `ω` are all non-zero at once and the robot slews
toward the target heading **while** it translates toward the target position. The phase is called
`holonomic_drive`. A diff-drive robot cannot do this; that is the entire reason for the wheel swap.

The simultaneity is available to any caller that wants it, and the mission layer uses it
selectively. `FieldNode.goto_xy_goal()` (`perception/fieldlib.py:637`) publishes
`{"x": …, "y": …}` only — position goals deliberately carry no final heading — while the storage
approach sets the corner heading explicitly, with `do_rotate(STORAGE_CORNER_YAW)` and a strafing
`do_move(fwd, left)` (`mission/match_runner.py:1854, 1896-1900`).

---

## The mecanum controller

```
distance = |goal − pose|

if distance ≤ xy_tolerance (0.15 m):
    if goal.yaw is None or |yaw_error| ≤ yaw_tolerance (0.40 rad):  → reached
    else                                                            → phase final_yaw (rotate in place)

else:                                                               → phase holonomic_drive
    rotate (dx, dy) into the body frame
    speed = min(max_linear_mps, linear_gain · distance) · near_goal_scale
    (vx, vy) = unit(body vector) · speed
    ω = goal_yaw_command(pose, goal) · near_goal_scale
```

| parameter | default | rationale |
|---|---:|---|
| `xy_tolerance_m` | 0.15 | deliberately loose — see the latch below |
| `yaw_tolerance_rad` | 0.40 | ≈23°; tighter than this triggers dead-zone grinding |
| `max_linear_mps` | 0.36 | node-side cap; the mission overrides it per speed profile |
| `min_linear_mps` | 0.075 | diff-drive only (`MecanumController` does not apply a floor) |
| `max_angular_rps` | 0.85 | |
| `linear_gain` | 1.425 | proportional gain on distance |
| `angular_gain` | 1.7 | proportional gain on yaw error |
| `drive_angular_deadband_rad` | 0.08 | below this yaw error, `ω = 0` exactly |
| `drive_angular_gain_scale` | 0.45 | while driving, yaw correction runs at 45 % gain |
| `obstacle_stop_distance_m` | 0.22 | front-sector LiDAR minimum (±55°) |

**The yaw deadband is load-bearing.** Localisation yaw jitters by a fraction of a degree scan to
scan. Without a deadband that jitter becomes a continuous stream of micro-`ω` commands, and on the
real chassis that is the dominant cause of a "straight" drive curving. Both controllers apply the
same deadband, and `MecanumController.goal_yaw_command` (`ALC/controllers.py:187-196`) documents why.

**Obstacle stop is direction-aware.** `DiffDriveController` stops dead on a front obstacle. The
mecanum version only treats a front obstacle as blocking when the commanded motion actually has a
forward component (`robot_x > 1e-3`, `ALC/controllers.py:160-165`) — strafing sideways or backing
away past an obstacle is one of the main reasons to fit mecanum wheels, and refusing to move at all
would throw that away.

`DiffDriveController` is still shipped and still tested. It is the rotate-to-goal → drive →
final-yaw state machine the robot used before the wheel swap, and it shares
`GoalControllerConfig` with the mecanum controller — which is why converting the whole stack to
holonomic drive was one launch argument (`DRIVE_TYPE=mecanum`, which flips the motor bridge and the
goal controller together).

### Near-goal slowdown

`near_goal_speed_scale()` (`ALC/controllers.py:214-218`) multiplies **both** the linear and angular
command by `near_goal_speed_scale = 0.5` once inside `near_goal_slow_radius_m = 0.10 m`. It is a
step, not a ramp — the proportional term `linear_gain · distance` is already the ramp; this is an
extra margin for the last 10 cm.

There is a real cost to it, and we measured it. Halved commands near the goal can fall **below the
motor firmware's PWM dead zone**, at which point the robot parks 0.17–0.35 m short of the goal and
sits there. On 2026-07-21 this burned the full 25 s goal timeout three times in one match (75 s+ of a
180 s round). The fixes were made in the caller rather than by shrinking the dead band:

- distance-adaptive timeouts instead of a flat 25 s (`mission/match_runner.py:613`);
- **stall detection** — under 3 cm of progress in 1.0 s (after a 0.8 s grace period for the
  acceleration ramp) aborts the goal early and hands back to the caller
  (`perception/fieldlib.py:645-661`);
- a residual under 0.35 m is simply **accepted** — the last few centimetres are absorbed by the next
  stage (visual approach measurement, placement gate, or the next leg). "Correct it while driving"
  beats "grind to a stop" on a 3-minute clock (`mission/match_runner.py:617-625`).

### The related `final_yaw` dead-zone finding

At a yaw error of ~0.24 rad the controller commands `ω = 0.205 rad/s`, which reached the motors as
PWM ±55 — and produced **no wheel motion at all**. We considered forcing a floor
(`final_yaw_min_angular_rps`), and the parameter exists, but the default is `0.0`: the chosen policy
is to widen `yaw_tolerance_rad` to 0.40 and simply not attempt corrections that small. Fighting a
firmware dead zone from the top of the stack is the wrong place to fight it.

---

## The goal-position latch

Pose estimates jitter by 1.5–2 cm. With a 15 cm tolerance, a robot sitting 14.5 cm from its goal will
flip between `reached` and `drive` on jitter alone, producing a twitch every few hundred
milliseconds. The latch removes that failure mode entirely.

```python
update_goal_position_latch(latched, release_count, distance,
                           enter_tol=0.15, release_tol=0.25, release_threshold=3)
```
(`ALC/arena_control_node.py:1334-1352`)

- `distance ≤ 0.15 m` → **latched**, release counter reset.
- Once latched, the *control goal* is replaced by `Pose2D(pose.x, pose.y, goal.yaw)`
  (`goal_for_position_latch`, `:1330-1331`) — the position error is defined to be zero, so the drive
  phase cannot re-open. Only the yaw component can still act.
- Releasing requires **3 consecutive** ticks beyond `0.25 m`. One outlier reading cannot un-latch.
- Any new goal or pose seed clears the latch (`:1044-1049`, `:1093-1099`).

This is ordinary Schmitt-trigger hysteresis, but the asymmetry matters: entering is instant (stop
promptly), leaving is slow (do not be fooled). `xy_release_tolerance_m` is clamped to be at least
`xy_tolerance_m` at construction (`:218-221`) so the two can never be configured inconsistently.

The tolerances are loose on purpose, and the reasoning (2026-07-05) is worth stating: on a chassis
where the last centimetre costs seconds of dead-zone grinding, and where a downstream visual
alignment step exists anyway, "close enough, keep moving" wins a timed match. Do not tighten these
without also removing the dead zone.

### The pose gate before placing

The latch protects against jitter. It does not protect against a *confidently wrong* pose. On
2026-07-21 the localiser mirrored and the robot dropped an object in the START corner while believing
it was at the storage staging point. The mitigation in the mission layer
(`mission/match_runner.py:1861-1893`) checks, before every placement, that (a) the status topic is
fresher than 1 s and (b) the pose is within 0.45 m of the staging point; on failure it re-drives once,
and on second failure it **fails closed** — opens the gripper where it stands (0 points, no penalty)
rather than risk a mis-placement. The primary defence against the mirror itself is upstream: reduce
total in-place rotation, see [`localization.md`](localization.md#the-imu-yaw-prior).

---

## Speed profiles

`mission/match_runner.py:100-105`:

| profile | cruise | approach | place |
|---|---:|---:|---:|
| `safe` | 0.50 | 0.40 | 0.30 |
| `normal` | 0.80 | 0.60 | 0.40 |
| `fast` | 0.95 | 0.70 | 0.45 |

Units are m/s. `fast` is ~95 % of the measured chassis limit of ~1.0 m/s (the wheel budget, 26.0 rad/s × r 0.0388); the final clamps live in
`real.yaml` (0.9 m/s linear) and in the Arduino firmware.

`cruise` applies to router legs, `approach` to the final visual approach to an object, `place` to
storage-box manoeuvres. In-place rotation uses the firmware's *position* profile and cannot be set
per-move — it is tuned through `position_max_rad_s` in `real.yaml`, and raising the rotation wheel cap
via `ROTATE_MAX_V = 0.525` (0.35 was an intermediate value, raised ×1.5 on 07-22; from the 6.0 rad/s ≈ 0.233 m/s default) took a single 90° turn from 2–9 s
down to roughly 1.8 rad/s of body rate.

### The bug that made the profiles a no-op

For weeks the profiles visibly changed the approach and placement phases but had **no effect on
cruising**. The chain of causes, confirmed on 2026-07-20:

1. `do_goto` drives via an arena goal, which is closed-loop inside the node — it carries no `max_v`,
   so the node's own `max_linear_mps` governed cruise speed. Correct diagnosis, but the prescribed
   fix (`ros2 param set`) did nothing.
2. `arena_control_node` built `GoalControllerConfig` **once in `__init__`** and never re-read
   parameters.
3. `GoalControllerConfig` is a `@dataclass(frozen=True)`, so even assigning a field would have raised.

The fix is `add_on_set_parameters_callback` plus `dataclasses.replace` to swap the whole frozen
config atomically, for 11 live-tunable fields (`ALC/arena_control_node.py:461-501`); the mission
runner now pushes `max_linear_mps` at setup (`mission/match_runner.py:1930-1945`) and the node logs
`controller config updated (live)` when it lands.

Verification, 25 timed centre-approach runs: before the fix, 9.4–9.7 s regardless of profile
(0.33 m/s); after, `fast` 5.65–6.13 s (0.51–0.56 m/s) and `safe` 7.31 s (0.43 m/s). Excluded as
causes by measurement: the diff-drive `motor_bridge_node` cap (not on the mecanum path) and
`cmd_vel_mux` (no clamp).

---

## The street router

### The geometry

The rulebook places objects only on a 50 cm grid — 42 cells inside the 4 m × 4 m arena. Objects are
8 cm cubes (4 cm half-width). The robot's half-width is 9.5 cm; its **circumscribed** radius is 16 cm.

The lines running exactly halfway between grid columns and rows are 25 cm-wide corridors that contain
no objects **by construction**. We called them streets:

```python
STREET_XS_M = tuple((x - 200) / 100.0 for x in range(25, 376, 50))   # -1.75 … +1.75
STREET_YS_M = tuple((y - 200) / 100.0 for y in range(75, 376, 50))   # -1.25 … +1.75
HIGHWAY_Y_M = -1.40          # the line the robot actually ran; the rulebook places no object at y < 100 cm
```
(`mission/match_runner.py:326-328`)

Diagonal motion is forbidden on a street route — a diagonal passes directly over grid points, which is
where the objects are.

### Why the naive inflation radius declared every street blocked

Corridor checking inflates each known object by

```python
inflation_for(votes) = CORRIDOR_BASE_M(0.10) + ROBOT_RADIUS_M(0.16) + VOTE_UNCERT_M(0.05)/votes
```
(`mission/match_runner.py:284`) — **0.26 m to 0.31 m** depending on detection confidence (fewer
votes ⇒ more uncertainty ⇒ more inflation).

A street's half-width is 0.25 m. `0.31 > 0.25`. Therefore every perfectly legal street was declared
blocked, always.

The radius was not wrong, it was the wrong radius for the situation. 0.16 m is the robot's
**circumscribed** radius, which is the correct figure when the direction of travel is arbitrary — the
robot can present any of its corners to the obstacle. On a street the travel axis is **fixed** and
axis-aligned, so the relevant figure is the robot's **half-width**:

```
0.095 (robot half-width) + 0.04 (object half-width) + 0.065 (pose error margin) = 0.20
STREET_CLEAR_M = 0.20   ≤   0.25 (street half-width)          # match_runner.py:329-332
```

`corridor_free(..., clearance=STREET_CLEAR_M)` (`mission/match_runner.py:298-311`) overrides the
vote-based inflation for street legs only. Every other leg still uses the conservative circumscribed radius.

Axis alignment is therefore a safety precondition, not a style choice — which is why legs are aligned
before they are driven (below), and why the tolerance is 15°: a 15° misalignment sweeps
`0.095·cos15° + 0.129·sin15° ≈ 0.125 m`, plus the 4 cm object half-width gives 0.165 m, still inside
0.20 m.

### The 2026-07-20 bug: waypoints that were never checked

This is the failure we most want other teams to read.

`plan_route()` had a last-resort branch that, when the straight line was blocked, computed a
perpendicular detour waypoint 0.5 m to the side of the blocking cell. It computed it correctly. It
then **returned it without ever checking whether the resulting three-leg path was clear**.

With 28 objects on the grid, the direct corridor and both axis-aligned L-shapes are blocked almost
every time — so essentially *every* carry leg fell into that unverified branch. Replaying the real
06:23 carry route showed the path passing **6 obstacle cells inside their inflation radius, with a
minimum clearance of 4.8 cm**. The robot had been driving over objects.

The fix has four parts (`mission/match_runner.py:350-431`):

1. **Every candidate is verified end to end** with `path_free()` (`:313-320`), which walks each
   consecutive waypoint pair through `corridor_free()`. A candidate that does not pass is not
   returned. This includes the detour branch, which now also tries **both** perpendicular directions.
2. **The street grid becomes the primary route** for a crowded arena, with the correct
   `STREET_CLEAR_M` so that legal streets stop being rejected.
3. **`street_forced`** as the last resort (below).
4. **Explicit `blocked`** as the final state instead of silently pretending a route exists.

The resulting ladder, in order:

| mode | what it is |
|---|---|
| `direct` | straight line, corridor clear — a single diagonal sprint |
| `L_x_first` / `L_y_first` | one axis-aligned corner, both orders tried, both fully verified |
| `street` | nearest street from start → axis-aligned L on the street grid → target; x-first and y-first both tried, shortest verified one wins |
| `detour` | verified perpendicular 0.5 m waypoint (`DETOUR_CLEAR_M`, clamped to `ARENA_CLAMP_M = 1.9 m`), both signs tried |
| `street_forced` | **no** collision-free route exists — follow the street grid anyway |
| `blocked` | not even a street route exists — drive the straight line and say so |

### `street_forced`

When nothing verifies, the honest options are "give up" or "drive anyway". Giving up costs the whole
object; driving the *straight* line is the worst possible choice, because a diagonal passes over grid
points and grid points are exactly where objects are.

`street_forced` re-plans on the street grid **with the obstacle list emptied**
(`plan_route_street(p0, p1, [])`, `:424`). The route is not verified — it is chosen on the structural
argument that objects live on grid points and streets are the gaps between them, so following the
street grid is always at least as good as a diagonal. It is reported as a warning in the run log and
recorded in `report.json`, so a `street_forced` leg is never silently indistinguishable from a clean
one.

### Verification

Replayed offline over the real 28-cell ground-truth layout, all target combinations:

- **0 / 28 intrusions**, minimum object clearance **0.250 m** (22 routes via street, 6 direct)
- street-centre tracking: 5 mm deviation along x, but **up to 12.6 cm along y**

That y number is an honest structural warning we did not fully resolve: the node's
`xy_tolerance_m = 0.15` is *larger* than the street's 11.5 cm of spare clearance (0.25 − 0.135), so a
goal declared "reached" at the edge of tolerance can be tighter against an object than the router's
own margin assumes. It worked, but the margin came from the tolerance being reached in practice, not
from the geometry guaranteeing it.

### Leg alignment before driving

`align_leg_yaw()` (`mission/match_runner.py:668-701`) runs before each non-final street leg:

- **Minimum-residual alignment, not cardinal snapping.** The robot is square and holonomic, so any of
  its four faces may lead. The code computes the offset from the *nearest* of the four axes
  (`off_axis = (off + π/4) mod (π/2) − π/4`, giving a residual in [−45°, +45°)) and rotates only by
  that residual — at most 45°, often 0.
- **`ALIGN_TOL_DEG = 15`**: within 15°, do not rotate at all.
- **Skip inside the southern safety band.** If both endpoints are below `HIGHWAY_FREE_Y_M = −1.40 m`,
  no alignment is done — the rulebook places no objects below y = 100 cm, the southernmost object
  silhouette reaches ~96 cm, and the robot's 16 cm circumscribed radius clears it below y = 80 cm.
  −1.40 m (official y = 60 cm) is the conservative margin. Any leg finishing entirely inside that band
  may take any heading.
- **Skip short legs** (< 0.35 m) and the **final** leg — a grasp or placement rotation always follows
  it, so aligning first is a double rotation. One measured instance: a −90° alignment immediately
  followed by a +48° re-rotation, 9 s wasted.

The v1 of this function snapped the *body* to the travel direction, permitting rotations up to 180°.
It produced 2–4 large rotations per collection cycle, and **both** runs that used it ended in
localiser collapse (mirror convergence, object dropped in the START corner), while the control run
with no rotation logic at all was clean. In-place rotation smears the LiDAR sweep and injects mecanum
slip into the odometry; degrade the yaw estimate past ±45° in a square arena and the wall-range
solution mirrors. Rotation is not free on this robot, and the routing layer is where you pay for it.

`do_rotate` (`mission/match_runner.py:638-666`) reflects the same economics: skip below 0.10 rad
(a 9° rotation once cost 2.9 s of firmware settling), verify the residual afterwards and re-rotate at
most once, and accept anything within 15°.

### What actually drove the matches

Everything above describes `plan_route()` — the corridor router. It is worth reading, because it is
where the street idea was worked out and because it is still the carry-path planner and the
`--nav legacy` fallback. But it is **not** what drove the four competition matches. From 07-22 the
runner drove the base itself, through `navigation/street_nav.py`, and every match ran `--nav street`.

The difference is closed loop versus open loop. `plan_route()` hands waypoints to the arena node and
lets it follow them. `StreetNavigator` sends the arena node a `STOP`, takes `/cmd_vel_direct` for
itself, and runs its own 20 Hz loop against the pose feed. Concretely:

- **The heading is locked north.** The gripper sticks 26 cm past the lidar centre, so a robot facing
  along the corridor sweeps 26 cm sideways where only 21 cm is free; facing north it sweeps 9.5 cm.
  So the robot travels sideways and forwards but does not turn — the only rotations in a cycle are
  ±45° at the mini-goal.
- **The approach geometry is a mini-goal**, one grid diagonal (0.354 m) south-east of the object, on
  the street. The robot stops there and turns 45° to face the object, instead of driving to a
  standoff behind it.
- **Cross-track error is corrected continuously** (P gain 2.5, cap 0.45 m/s), and a lane excursion is
  not a failure: forward progress goes to zero and only the lateral term runs until the robot is back
  under 6 cm. Recorded in the four matches: worst excursion 0.164 m, at most two recoveries a leg.
- **The highway is `HIGHWAY_Y_M = -1.40`**, not the `-1.25` of the router constant.
- **Carry is a drift**, not a rotate-then-drive: the translation to staging and the −135° turn are
  commanded together.

Reaching the central scan point is the one place a diagonal is allowed: `drive_free()` cuts straight
across the free band below the object rows, because that band provably has no grid points in it.
Inside the field the rule stands — never cross the grid diagonally when a street runs where you want
to go.
