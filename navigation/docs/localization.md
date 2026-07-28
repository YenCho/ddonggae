# Localization — matching wall ranges against a known square

> Paths are relative to the repository root. `ALC/` abbreviates
> `navigation/ros2/arena_lightweight_control/arena_lightweight_control/`.

The robot knows where it is by comparing each LiDAR beam against the range that beam *would* have if
the arena were an empty rectangle — because it is. No particle filter, no EKF, no TF tree. One
`Pose2D(x, y, yaw)` in map frame, updated at ~13 Hz (the lidar measures 13.1 Hz), dead-reckoned forward by a gyro in between.

Two matchers ship in `ALC/map_localization.py`:

| class | what it does | active? |
|---|---|---|
| `WallRangeLocalizer` (`:330`) | closed-form expected wall range per beam; local, global and global-coarse solves | **yes** — `localization_mode=wall_range` is the default everywhere |
| `KnownMapLocalizer` (`:206`) | generic correlative matcher over a Dijkstra distance field built from the PGM | no — kept as the readable, portable reference implementation and as a fallback for non-`wall_range` modes |

---

## The observation model

`OccupancyMap.inner_wall_bounds()` (`ALC/map_localization.py:123-159`) reduces the whole 250×250-cell
PGM to four numbers: the inner face of each wall. It does this by taking occupied cells in a central
band (to ignore corners) and picking the innermost left/right/bottom/top extremes.

Given that rectangle, the expected range of a beam pointing along the unit vector
`(cos θ, sin θ)` from `(x, y)` is a ray/AABB intersection —
`expected_wall_range_from_unit()`, `ALC/map_localization.py:744-771`:

```python
if cos_theta >  eps: best = min(best, (xmax - x) / cos_theta)
elif cos_theta < -eps: best = min(best, (xmin - x) / cos_theta)
if sin_theta >  eps: best = min(best, (ymax - y) / sin_theta)
elif sin_theta < -eps: best = min(best, (ymin - y) / sin_theta)
```

Four divides, two comparisons. That is the entire sensor model. There is no ray casting and no map
lookup in the hot loop, which is why a 200-candidate global search costs single-digit milliseconds
in pure Python.

### Scoring (`score_candidate`, `ALC/map_localization.py:706-739`)

1. Beams are downsampled to at most `max_beams` (24 for tracking, 16 for global reseeds) by even
   index striding — `wall_beams()`, `:677-704`.
2. Each residual `measured − expected` is clamped to `max_residual_m = 0.35`, so one wild beam
   cannot dominate.
3. Residuals are sorted and the worst `trim_fraction = 30 %` are dropped. The score is the mean of
   what remains — a trimmed mean, not a sum of squares.
4. Fewer than `min_beams = 8` usable residuals ⇒ `inf` (match fails, pose is not applied).
5. In local search only, a small `prior_penalty = 0.004` per normalised unit of deviation from the
   prior breaks ties toward continuity.

A match is accepted only if its score is ≤ `wall_localization_max_score = 0.22` (i.e. mean trimmed
residual ≤ 22 cm).

### The short-return bug — a negative result worth publishing

The original implementation treated a beam that came back much shorter than the wall
(`residual < −0.22 m`) as an occlusion and **skipped** it. That is the standard, sensible thing to do
in a normal environment. Here it was actively harmful.

In a global search, a badly-wrong candidate can *discard every beam that disagrees with it* — it
scores only on the handful of beams that happen to agree, and wins. In one synthetic test this
produced a **65 cm** error with a better score than the truth, with `x` effectively unobservable.

The fix is one flag. The LiDAR plane is at 0.32 m and the tallest object in the rulebook is 8 cm, so
**nothing in this arena can occlude a wall** — every short return is evidence *against* the
candidate, not missing data. `penalize_short_returns = True`
(`ALC/arena_control_node.py:243-246`) makes the residual count normally. Synthetic error went from
65 cm to **2 cm**, at ~10 ms.

The legacy branch is still in the code (`:722-731`) with the reasoning in a comment, because the
opposite choice is correct in any arena where objects can rise above the scan plane.

---

## The IMU yaw prior

### Why a square arena needs one

A square's wall-range observation is **90°-degenerate**. Rotate the true pose by 90° about the arena
centre and you get a pose that explains the same scan exactly as well. In practice the estimate does
not sit at the degeneracy — it *jumps* into a neighbouring quadrant when the local search window
falls behind a fast rotation, and then confidently stays there. Two real runs on 2026-07-21 at 16:1x
suffered exactly this: the localiser mirrored, and the robot dropped an object in the START corner
believing it was the storage corner.

The resolution is to stop estimating yaw from the scan.

### Gyro-z integration, fed forward

`imu_yaw_callback` (`ALC/arena_control_node.py:1201-1245`) runs on every `/imu/data` message and does
three things:

```python
wz_raw = axis · gyro                       # project onto the body-up axis
if abs(wz_raw - bias) < 0.02:              # indistinguishable from rest?
    bias = 0.98 * bias + 0.02 * wz_raw     # online zero-rate estimate (EMA)
dyaw = (wz_raw - bias) * dt
self.pose = Pose2D(pose.x, pose.y, wrap(pose.yaw + dyaw))   # FEED FORWARD
```

The key word is *feed forward*: the node does not merely keep a yaw estimate to compare against, it
**advances `pose.yaw` itself** between scans. By the time a scan arrives, its prior yaw is already
current, and the matcher's job shrinks to correcting a small residual instead of chasing a turn.

Only deltas between scan applications are consumed, so the absolute gyro offset is irrelevant and
slow bias drift (a few deg/min on a BMI055-class part) does not accumulate into the estimate within a
3-minute match. Unlike wheel odometry on mecanum wheels, a gyro does not slip.

The zero-rate EMA only updates when the reading is already close to the current bias estimate — that
gate is what stops a genuine slow turn from being absorbed into the bias.

### The gyro axis is not z

On the real robot the IMU lives inside the **bottom D435I, which is tilted 54°**. Body-yaw rotation
therefore splits across the sensor's y and z axes, with the sign inverted by the mount orientation.
The measured projection is:

```python
"imu_yaw_axis": [0.0, -0.586, -0.810]   # navigation/ros2/.../launch/lightweight_real.launch.py:171
```

`-0.586 ≈ -cos 54°`, `-0.810 ≈ -sin 54°`, measured by rotating the robot on 2026-07-15. The node
default is `[0, 0, 1]` (a body-frame IMU, which is what the simulator provides) and it normalises
whatever it is given (`ALC/arena_control_node.py:392-394`).

If this axis is wrong the yaw prior integrates *backwards*, which does not merely degrade the
estimate — it actively drives it toward a symmetric flip. If you rebuild this robot with a different
camera mount, remeasure it before anything else.

### The yaw gate

Two priors, two gates (`ALC/arena_control_node.py:537-600`):

| prior | gate | escape hatch |
|---|---|---|
| IMU, fresh (`< 0.5 s` old) | ±45° (`imu_yaw_gate_rad = π/4`) | 60 consecutive rejects |
| wheel odom (IMU absent) | ±0.6 rad (`odom_yaw_gate_rad`) | 12 consecutive rejects |

45° is not arbitrary: it is the radius of the degeneracy basin. Any yaw more than 45° from the
propagated prior is closer to a *neighbouring* 90° optimum than to the true one, so it is a flip by
definition and is rejected. Because the seed guarantees the true yaw at t=0 and the gyro guarantees
the prior stays true, no legitimate scan-to-scan correction can ever exceed 45°.

The escape hatch exists because the *prior itself* can be corrupted. Wheel odometry on a slipping
mecanum chassis drifts fast; without a hatch, one bad odom yaw would make the gate reject every
correct match forever. With a fresh IMU the hatch is essentially dead code — which is why the gate is
skipped entirely on the IMU path (`:582`): with feed-forward, the yaw-locked global solve returns
exactly the prior yaw, so there is nothing to gate.

There is a second, independent ±45° check on the **global reseed** result
(`maybe_reseed_wall_match`, `ALC/arena_control_node.py:825-848`). A whole-arena re-solve is precisely
the operation most likely to land on a neighbouring 90° optimum, so its output is rejected if it
disagrees with the propagated prior by more than 45°, and the local result is kept instead. The
rejection is visible in the status topic as `reseed=rejected,yaw_jump>45deg`.

---

## The three solve modes

`match_scan()` (`ALC/arena_control_node.py:699-786`) picks one:

**Yaw-locked global (the competition path).** When the IMU is fresh, yaw is *known*, and a known yaw
makes the square's x/y solution **unique**. So the matcher does a full-arena grid search on a single
yaw every scan: `xy_step = 0.3 m` over the interior ⇒ roughly 200 candidates, 16 beams each
(`:753-764`). There is no local window to fall behind and no reseed guard to trap a diverged
estimate. This directly fixed a rehearsal failure in which the pose was lost by 1.3 m for 20 s
because *both* the tracking window (±0.12 m) and the reseed guard were smaller than the error.

**Local window + periodic reseed (the IMU-less fallback).** ±0.12 m / ±0.14 rad around the prior at
0.04 m / 0.07 rad steps, then a half-step refinement (`match_scan`, `ALC/map_localization.py:359-445`).
Every `localization_reseed_period_sec = 2.5 s`, or immediately whenever the local score exceeds the
acceptance threshold, a full global re-solve runs and is adopted only if it beats the local score by
`localization_reseed_score_margin = 0.035` **and** passes the ±45° yaw-jump check.

**Yaw trim.** Gyro bias (~0.05 deg/s after calibration) still walks a few degrees over a 3-minute
match. So when the robot is **stationary** (`|wz| < 0.03 rad/s`) and 5 s have passed, the node drops
out of the yaw-locked path for exactly one scan and runs a normal yaw-searched match to re-anchor
the yaw (`ALC/arena_control_node.py:743-746`). Stationary is the right moment: no motion blur across
the sweep, and no cost to the mission.

---

## Reseeding and pose seeding

**Operator seed (mandatory).** Before every run the START corner is published on
`/arena_lightweight/pose`. `set_pose()` (`:1093-1100`) hard-sets the pose, clears the last match,
clears the goal latch, and bumps `pose_revision`.

**The revision counter.** Scan matching runs in a different callback than the seed. A match that
started before the seed and finished after it would silently overwrite the operator's correction.
`pose_revision` is captured when the scan arrives and compared before applying
(`:514`, `:538-542`); a stale match is discarded. There is a second monotonicity guard,
`last_localization_apply_started`, so out-of-order completions cannot go backwards.

**Match-time yaw catch-up.** A match takes ~30–80 ms, during which the robot may still be turning.
Applying the match's yaw verbatim would rewind the turn. Instead the applied yaw is
`result.pose.yaw + (imu_yaw_now − imu_yaw_at_scan_capture)` (`:603-607`).

**Automatic reseed** is the global re-solve described above. There is no kidnapped-robot recovery:
if the arena is symmetric and the prior is confidently wrong, nothing in this design will notice. Our
mitigation was procedural (seed carefully, minimise in-place rotation) plus a pose sanity gate in the
mission layer before dropping an object — see
[`control-and-routing.md`](control-and-routing.md#the-pose-gate-before-placing).

---

## Latency

| stage | measurement |
|---|---|
| generic distance-field matcher (what we replaced) | 345–449 ms |
| wall-range local match, 1080-beam synthetic scan | ~20.35 ms |
| yaw-locked global solve, offline | 8–10 ms |
| on-robot, Jetson Orin Nano, YOLO running concurrently | median 28.2 ms, mean 38.9 ms, p95 185 ms, max 300 ms (n = 2132) |
| warning threshold | 50 ms |

The on-robot figures come from `loc_latency_ms` recorded per capture frame in three real-arena
sessions on 2026-07-18 (`mission/field_autopilot.py:640`, sourced from the node's own
`MatchResult.latency_ms`). The tail is CPU contention, not algorithmic: the same process runs the
perception pipeline, and CPython holds one GIL.

Two mitigations, both worth copying:

- **`depth=1` on the scan subscription** (`ALC/arena_control_node.py:352-360`). When matching falls
  behind, stale scans must be *dropped*, not queued. A backlog of five scans at 300 ms each once put
  the pose estimate seconds into the past — with a queue, falling behind is unrecoverable; without
  one, it is merely a lower update rate.
- **A 12.5 Hz software throttle** (`:505-511`). The simulator's PhysX LiDAR published at render rate
  (30–60 Hz) rather than the real sensor's 10 Hz, and back-to-back matching saturated the process:
  6 ms of compute stretched to 300 ms and starved the status and IMU callbacks. The throttle caps
  processing at the real sensor rate regardless of publisher behaviour.

There is a watchdog: exceeding `localization_warn_latency_ms = 50.0` logs a throttled warning
(`:627-632`), and the value is exported in the status JSON so the field tooling records it per frame.

---

## Accuracy

| context | error |
|---|---|
| synthetic reference scan, after the short-return fix | 2 cm |
| Isaac kinematic parity study vs. ground truth, 2026-07-18 | 2.7–2.9 cm |
| real arena, pose jitter while rotating in place | 1.5–2 cm |

The rotating-in-place figure matters more than it looks. During the 2026-07-18 investigation of
duplicated grid detections, localisation was the prime suspect; measuring 1.5–2 cm of jitter inside a
±25 cm grid snap **exonerated it** and moved the investigation to depth under-estimation, which was
the real cause. Object proximity does raise the wall-fit residual (`loc_score` correlates −0.73 with
clearance) but does not bias the pose — objects are below the scan plane, so they add noise to the
score, not to the geometry.

---

## Why this beat AMCL for this arena

- **AMCL's strength is our non-problem.** A particle filter earns its cost by representing multi-modal
  belief in an unknown-start, cluttered, partially-observable map. Our start pose is given, the map is
  four line segments, and every beam sees a wall. We were paying for machinery whose output was
  always a single tight mode.
- **The generic matcher's cost is dominated by generality.** A distance-field correlative matcher
  must be able to represent any occupancy geometry, so it pays a grid lookup per beam per candidate.
  Substituting the closed-form rectangle intersection removed that entirely and bought a 17–22×
  speed-up for ~90 lines of code.
- **Latency is a control problem, not a stats problem.** At 345–449 ms the pose was three LiDAR
  periods stale and the drive controller was steering on history — the visible symptom was bent
  straight lines, not a large reported error. Dropping to ~28 ms fixed the driving.
- **The degeneracy needed a sensor, not a better filter.** No amount of particle-filter tuning
  resolves a genuine 90° symmetry in a square room. A €0 gyro already inside the depth camera did.
  Recognising that the fix belonged in the sensing layer, not the estimator, was the actual insight.
- **We could read all of it.** 1004 lines of dependency-free Python that we could instrument,
  unit-test offline and reason about at 02:00 in a competition venue. Every failure above was
  diagnosed from a log line and a unit test, not from a parameter sweep over a black box.

The honest counterweight is in [`../README.md`](../README.md#what-you-give-up): every one of these
advantages is purchased with an assumption that this arena happens to satisfy.
