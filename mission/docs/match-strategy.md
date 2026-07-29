# Match strategy

Why the robot does what it does in a 3-minute round. Every number here is either a rulebook
constraint, a constant in `mission/match_runner.py` / `perception/fieldlib.py`, or a measurement
from a dated field run. Where a decision came out of a failure, the failure is named.

---

## 1. The rules are the whole design

| Rule | Consequence we exploited |
|---|---|
| Objects sit only on **42 fixed grid points**, 50 cm apart (official x ∈ 50…350, y ∈ 100…350 cm); 28 are occupied | Localisation of objects becomes **classification**, not continuous estimation |
| The lowest object row is y = 100 cm | The strip below is provably empty for the whole match — a free "bottom highway" between START and STORAGE |
| Set 1: polyhedra, 10 pts. Set 2: fruit-photo cubes, 20 pts | Value density differs 2:1; ordering matters when time runs out |
| **Mispick = −2× the object's base points** | A wrong fruit cube costs −40. Expected value says: when uncertain, *do not pick* |
| 180 s per match; max 100 pts = 7 objects | ~25 s per object including the scan and the drop |
| Ties broken by completion time | Speed is a tiebreaker, not a scoring term |

The 4 m × 4 m arena, the 40 × 40 cm START (bottom right) and STORAGE (bottom left) zones, and the
robot's own 40 cm footprint set everything else.

## 2. Perception as classification: the 42-cell snap

A detection does not have to be accurate — it has to be closer to the right grid point than to any
other. Adjacent points are 50 cm apart, so the snap tolerance is ±25 cm, which is far larger than
either our localisation error or our ranging error.

`fieldlib.snap_cell()` snaps every detection to the nearest of the 42 points and returns the
residual; `match_runner.py` rejects anything with `snap_err > MAX_SNAP_ERR_M = 0.30 m`
(`match_runner.py:82`). The gate is deliberately loose: it exists to throw out garbage geometry
(a detection on a wall, a bad depth pixel), not to filter measurement noise. Offline the snap was
7/7 correct with a 7.5 cm mean residual (2026-07-17 analysis).

Cell identity is then decided by **voting**, not by a single frame — see §5.

## 3. Scanning: one centre point, mast up, eight headings

### Why scan at all, and why from the centre

The robot drives to `(0.25, 0.25)` map = official `(225, 225) cm` and spins in place, taking 12 shots
at 30° steps. Two properties of that point (`fieldlib.py:86`):

* It is the **centre of the object field**. Objects span x 50–350 (centre 200) and y 100–350
  (centre 225). The earlier scan point `(0.25, −0.25)` sat 50 cm south, which pushed the northern
  row (y = 350) far enough away that it kept getting demoted as a distant observation. Changed
  2026-07-21.
* It sits on a **street intersection**, 35 cm diagonally from its four nearest grid points, rather
  than on a grid point where the arena's symmetry would put objects at identical ranges.

Worst-case distance from the scan point to any of the 42 cells is **2.15 m** (the far corners at
(50, 100) and (50, 350)) — under the 2.5 m `FAR_PRESENCE_M` threshold at which observations used to
be downgraded to presence-only.

### The 4-point plan that we did not ship

A 2026-07-17 analysis computed the worst-case distance from the nearest scan point to any grid cell
as a function of the number of scan points:

| Scan points | Worst-case distance |
|---|---|
| 1 (centre) | 2.12 m |
| 2 | 1.80 m |
| 3 | 1.63 m |
| **4 (quadrant centres)** | **0.98 m** |
| 6 (2 × 3) | 0.98 m — no improvement |

Four is both the minimum and the saturation point: the binding constraint is the four corner cells,
and quadrant-centre points already cover them. The shipped runner nonetheless scans from a single
point, because each extra scan point costs a drive plus a full 12-shot spin out of a 180 s budget,
and because FHD stitching plus depth ranging made the far corners workable in practice. We never ran
the 4-point tour on the real robot, so treat the table as analysis, not as a rejected measurement.

### Why the mast goes up

Both cameras ride the mast (measured stroke 14.89 cm; tape cross-check 35.1 → 49.7 cm top camera,
30.8 → 45.9 cm near camera). Raised, the top camera sits at 0.4986 m with an 18–19° down-tilt, which
turns the scan into a near-top-down view of the whole grid — objects occlude each other far less,
and the fruit photo, which is printed on the **top** face of a cube, is visible at all.

The cost is that mast-up geometry is a different calibration set (`TOP_MOUNT_UP` / `NEAR_MOUNT_UP`,
`perception/calibration/stitch/up.json`) and the mast takes ~7 s up and ~6 s down. Both are hidden:
the raise overlaps highway leg 2, the lower overlaps batch inference.

### The structural blind spot, and the pre-shot

The four cells diagonally adjacent to the scan point are only 0.35 m away. On 2026-07-21, at
mast-up (14:32) and again at mast-mid (17:12), **all 8 spin shots missed them** — at that range and
angle A1 confidence collapses to 0.05–0.11, below the 0.25 scan threshold.

Lowering the global threshold would have bought ghosts everywhere. Instead the runner takes one
dedicated **mast-down** shot at the end of highway leg 1 (`_take_preshot`, `match_runner.py:938`),
where those four cells are 1.8–2.3 m away and seen through the best-calibrated mount. Those four
cells are assigned exclusively to that shot — the pre-shot only votes for them (`only_cells`), the
spin scan's votes for them are **rejected** (`exclude_cells`), and because they only get one shot
their vote requirement is relaxed to 1 (`_finalize_scan`, `match_runner.py:1343`).

## 4. Driving: streets, not diagonals

Objects only occupy grid points, so the lines midway between them are 25 cm-wide corridors that are
free by construction. Robot half-width 0.095 m + object half-width 0.04 m = 0.135 m fits with margin;
`STREET_CLEAR_M = 0.20` is the clearance used for axis-aligned street driving. A diagonal dash, by
contrast, passes directly over grid points.

`plan_route()` (`match_runner.py:377`) tries in order: direct diagonal → axis-aligned L (both
orders) → the street grid → a perpendicular detour — and **validates every candidate over its whole
length** with `path_free()`. If nothing is clear it forces the street grid rather than the diagonal.

That validation is not decoration. On 2026-07-20 the detour branch computed a waypoint and never
checked it: a carry route ran through 6 obstacle cells with 4.8 cm minimum clearance. A secondary
cause was that the general inflation radius (0.31 m, based on the robot's *circumscribed* radius)
exceeded the street half-width of 0.25 m, so even a perfectly good street was always reported
blocked. After the fix, all target combinations on a real 28-object layout produced 0 incursions
with a 0.250 m minimum object distance (22 street routes, 6 direct).

### Rotation is a failure cause, not just a cost

The square arena's `wall_range` localiser is 90°-symmetric, so a yaw error beyond ±45° can
mirror-lock. In-place rotation smears the LiDAR scan and adds mecanum slip on top.

Two runs on 2026-07-21 (16:13, 16:18) used cardinal yaw snapping, which produced 2–4 rotations of
90/180° per cycle. Both ended in localiser collapse — one of them dropped an object in the START
corner while believing it was at the storage staging point. The control run (14:32, same cameras, no
rotation logic) was incident-free.

The fix exploits the fact that a square mecanum robot can drive on any of its four faces.
`align_leg_yaw()` (`match_runner.py:668`) aligns to the *nearest* of `yaw + k·90°`, so it rotates at
most 45°, and skips entirely when: the offset is within ±15°, the leg is shorter than 0.35 m, it is
the final leg (a FACE or corner rotation follows immediately anyway), or the whole leg lies inside
the object-free highway band. The 15° tolerance is justified geometrically — sweep half-width
0.095·cos15° + 0.129·sin15° ≈ 0.125 m, plus 0.04 m object half-width = 0.165 m ≤ 0.20 m street
clearance.

## 5. Deciding what a cell is

Votes accumulate per cell across all shots. The code defaults are `votes ≥ 3` and
`CELL_FRUIT_K = 2`, but **every match ran at 1 and 1** — `run_match_day.sh` fixes them there,
because a 12-shot spin does not see most cells twelve times, it sees most of them once (see
`perception/docs/grid-voting.md`). Identity is then decided **asymmetrically**, because the two
stages are not equally trustworthy about the same thing:

1. **Fruit first.** Count face-model votes with `face_conf ≥ 0.5`. If the top fruit has at least
   `fruit_k` such votes, that is the identity.
2. **Ties.** If the top two fruits are within 1 vote, re-decide on the **sum of face confidences**,
   and only accept if the winner leads by ≥ 0.10; otherwise the cell is held as
   `conflicting_fruit` and never picked. This rule exists because of a specific loss: on 2026-07-21
   at 17:12 a misread `pineapple` at 0.773 tied a correct `apple` at 0.924, the cell went to
   `conflicting_fruit`, the A1 fallback then demoted it to `plain`, and a real apple target was
   thrown away.
3. **Otherwise**, fall back to an A1-confidence-weighted tally, with all fruit labels collapsed to
   `plain` (a cube whose top face was never resolved is, for target purposes, a plain cube).

Optional pair verifiers (apple-vs-orange, banana-vs-pineapple) run on the fruit-face patches: above
`PAIR_CONF = 0.80` they replace the label, below it they strip the face of its strong vote.
An honest caveat carried from the perception evaluation: on a 175-patch real-image set the verifier
measured *worse* than face-only (85.7 % with the verifier off vs 74.3 % for the best gated
configuration). That verdict held for the *synthetic-trained* verifiers. On 2026-07-24 the same sweep was run
again over the retrained pair — face fine-tune, AO off, BP real — and it won at 43/44 cells, so
the default became `--pair bp` and both finals ran with the banana/pineapple verifier live. See
`perception/docs/models.md`.

## 6. Choosing targets: no fallback, by design

In competition mode (`--target-shape` + `--target-fruit`), `match_candidates()`
(`match_runner.py:454`) returns only cells that:

* have a voted identity equal to one of the two announced targets,
* are not flagged `conflicting_fruit`,
* belong to a set whose quota (4 shapes, 3 fruits) is not yet filled.

There is **no non-target fallback anywhere**. A mispick costs 2× base points, so picking a
non-target is strictly worse than standing still. "Uncertain ⇒ do not pick" is encoded, not
aspirational. (The rehearsal-only `--target-class` flag *does* fall back to nearest-any, which is
why it is barred from being combined with the competition flags.)

The default is **nearest to the robot**, and the 2026-07-17 analysis argued for shortest-job-first
instead: with one gripper and one depot, every object is an independent round trip, so total
distance is order-independent and the right objective is *count within 180 s*, which favours
objects nearest **storage**. On a real 10-object map that ordering completed 9/10 inside 180 s.

Neither is what played the finals. Every match ran `--order fruit-first` (take the 20-point set
first), and on the last day that got replaced outright by the **consistency factor**: rank every
candidate by `p · value / T_exp`, where `p` is the probability the cell really is the target and
`T_exp` is the expected time including the drive back to storage. That objective *is* the
shortest-job-first idea, arrived at from the other direction — and because `T_exp` counts the
whole round trip, it prefers objects near storage on its own. Where `p` comes from is §6.5.

## 7. Ranging: depth, not the ground plane

This is the single most consequential change of the last week, made on 2026-07-21.

**The old method** back-projected the lowest pixel of the object's silhouette onto the floor plane,
which assumes that pixel is the ground contact point. For near-spherical classes — icosahedron,
dodecahedron — the lowest silhouette point is a tangent *in the air*, so range came out
systematically long.

**The bias, quantified.** A numerical cross-check of both projections agrees exactly (0.0 mm) for a
point on the floor, and diverges for a point on the object's front face: ground over-reads by
**+27–32 mm at z = 4 cm** and **+53–64 mm at z = 7 cm**. On the real robot on 2026-07-20 the same
effect measured **+5–10 cm on 3/3 icosahedra**.

**The failure chain it caused.** Over-reading pushed measurements past the 0.50 m defensive-hop
threshold → the robot hopped → the old hop landed at ~0.25 m from the object → at that range the
contact point is below the near camera's field of view (mount 0.187 m forward, 0.314 m high, tilted
53.2°, 42° vertical FoV) → re-measurement failed **5/5** → **0/4 icosahedron grasps**.

**The fix** (`measure_target`, `match_runner.py:1509`) reads the median depth in a patch at the
contact pixel, sampled 3 px *above* it (scaled by image width / 640, because aligned depth is
upsampled to colour resolution — at FHD a fixed 3 px radius covers a third of the intended solid
angle and falls into upsampling holes). Valid range 0.1–4.5 m; one retry with a doubled patch on a
hole. This is shape-independent: it measures distance to the object's front face for every class
alike, and the gripper offset absorbs the difference. The ground-plane value is still computed and
logged as `y_ground_ref` in every measurement so the bias keeps being measured, and it is used as a
fallback only when the depth pixel is invalid — recorded as `src: "ground"` so the two are never
confused in the report.

`--range-mode ground` still exists to reproduce the old behaviour.

## 8. Measure with retry: 6 attempts

`measure_with_retry()` (`match_runner.py:1660`) retries a failed measurement up to
`MEASURE_TRIES = 6` times, waiting 0.35 s between attempts. Most measurement failures are transient
and frame-related — motion blur right after a move, a stale frame, a pose estimate that has not
converged — so simply asking again with a fresh frame usually works.

The important part of this change is *where* it applies. Previously the post-hop re-measurement had
exactly one attempt, i.e. no insurance at the most fragile moment in the whole cycle: immediately
after a move. First measurement and post-hop re-measurement now share one code path.

Cost is bounded and only paid on cells that are failing anyway: ~0.6 s per failed attempt, so an
approach that normally takes 2.5–2.8 s takes 3.4–4.9 s when all six attempts fail. The successful
attempt number is written to `report.cycles[].measure.attempt` / `.hop_attempt`. The source comment
is explicit that **6 is an initial estimate, not a validated value** — if the attempt distribution
turns out to be all 1–3, it should drop to 3.

## 9. The defensive hop

If a measurement says the object is more than `FAST_HOP_THRESHOLD_M = 0.50 m` away, the runner does
not trust it enough to run the single-shot approach. It advances part of the way, re-measures, and
only then commits.

Where it lands matters. The original "advance half the distance" rule landed at ~0.25 m, where the
near camera can no longer see the object's contact point — the 5/5 re-measurement failure above.
`HOP_LAND_M = 0.40` instead leaves a fixed 0.40 m gap, chosen as the median of the 0.36–0.51 m band
in which measurements actually succeeded on the real robot.

## 10. Re-verifying the target immediately before closing

Scanning decides identity from 2 m away, top-down. Then the robot drives 2 m and closes a gripper on
whatever is there. `verify_target()` (`match_runner.py:1604`) runs one last A1 + face (+ pair)
inference on the final approach frame — the last moment anything is recognisable, since after the
final advance the object sits ~11 cm in front of the gripper.

The rule that makes it useful is that **only a definite contradiction skips the cell**:

| Reading at close range | Decision | Why |
|---|---|---|
| Target is a polyhedron, A1 re-reads it as anything else | **skip** | Definite contradiction |
| Target is a fruit, A1 says "not a cube" | **skip** | Definite contradiction |
| Target is a fruit, face model says a *different* fruit with conf ≥ 0.30 | **skip** | Definite contradiction |
| Face model says `plain` | **pass** | Undecidable — two of a fruit cube's four side azimuths are blank by rule, so one `plain` reading is a 1:2 likelihood ratio, not a refutation |
| No detection at all, or a weak fruit face below the vote gate | **pass** | Undecidable — the one fruit face guaranteed to be facing you is the *top* one, which a near, low view may not see |
| Face inference throws | **pass** | Trust the scan identity |

Treating "plain" as a mismatch would throw away real targets on nearly every cycle. The asymmetry is
the whole point: the close-range view can *refute* an identity but cannot *confirm* one, so it is
only allowed to veto.

It fires in practice. In the Isaac Sim rehearsal run `20260721_155553` a cycle was correctly
aborted with `face re-read pineapple(0.58) ≠ target apple — contradiction`, spending 15.7 s instead
of −40 points. (Kinematic surrogate run, not the real robot — the timing is not a hardware
measurement.)

## 11. Storage: a bowling-pin layout entered from one fixed angle

The rule is top-down containment: seen from above, an object must lie inside the 40 × 40 cm storage
box (which has 10 mm × 3 mm lips). Stacking is legal.

**One approach angle for everything.** The robot always faces the bottom-left corner at
`STORAGE_CORNER_YAW = −135°`. Entering each slot at its own `atan2` heading was tried on 2026-07-20
and made the gripper meet the box lip obliquely.

**Pins, not slots.** Pin 1 is at official (30, 30) cm — the deepest point the *robot base* can reach
given the gripper's length. It is a robot target, not an object position: the object ends up
`GRIP_FORWARD_M = 0.115 m` further towards the corner. Rows step 12 cm along the corner diagonal and
13 cm laterally. `storage_pins()` (`match_runner.py:125`) computes the predicted object centre for
each pin and **rejects any pin whose object would fall outside [4.5, 35.5] cm** — the rule is enforced
geometrically at design time rather than discovered on the field:

| Pin | Robot target (official cm) | Predicted object centre (cm) |
|---|---|---|
| 1 | (30.0, 30.0) | (21.9, 21.9) |
| 2 | (33.9, 43.1) | (25.8, 34.9) |
| 3 | (43.1, 33.9) | (34.9, 25.8) |
| 4 | (20.8, 39.2) | (12.7, 31.1) |
| 5 | (39.2, 20.8) | (31.1, 12.7) |
| — | rejected: 3 further pins | (29.6, 48.0), (38.8, 38.8), (48.0, 29.6) — outside the box |

Five usable pins; from the 6th object onward pins are reused, which is legal because stacking is.

**Release is relative.** Dropping opens the fingers by `--release-ticks 600` ≈ 52.7° *from where
they currently are*. A full open (214°) hits the storage wall, and an absolute angle is meaningless
while the fingers are stopped on an object of unknown width. Then retreat 0.35 m and re-close.

### Two fail-closed gates

Both came from specific disasters, and both prefer scoring zero to scoring negative or corrupting
the next cycle.

* **Placement gate.** On 2026-07-21 at 16:18 the localiser was *confidently wrong* and the robot
  released an object in the START corner believing it was at staging. The gate now checks status
  freshness (< 1.0 s) and deviation from the staging point (< 0.45 m), retries the move once, and if
  it still fails, **releases where it stands** — outside the box, worth 0 points but no penalty —
  rather than carrying the object into the next cycle, where the approach `OPEN` would drop it
  anywhere. Note the honest limit written in the source: a confidently-wrong mirror lock is not
  caught here; reducing total rotation (§4) is the primary defence.
* **Stall gate.** A firmware move timeout during the storage advance means the robot is pushing
  against a wall or an already-placed object. Re-advancing would just grind another 8 s, so the
  runner retreats 0.10 m, drops, and flags the placement `place_stall: true` as suspect.

## 12. Every run scores itself

`stage_report()` writes `summary.score_est` (20 pts per placed fruit, 10 per placed shape),
`placed_by_cls` against quota, per-cycle phase timings, the measured total against the hard 180 s
budget with an explicit overrun flag, and a `PASS` / `PARTIAL` / `FAIL` verdict. When a ground-truth
layout was entered, the scan-vs-truth confusion table (ok / wrong / ghosts / missed) is printed
**before collection starts** — a bad scan is caught before the robot commits to picking anything.

---

## 13. What we would do differently

We won 1st of 16 (qualifier 1: 60 pts at 10 % weight; qualifier 2: 60 pts at 90 %; finals scored as
the sum of two matches, 60 + 90 = 150). We also lost points in three distinct ways, and each one
points at a real gap.

### Failure 1 — the under-ripe apple (qualifiers)

The apple printed on the arena cubes was a **yellowish, under-ripe apple**, not the red one shown
beforehand. The face model — trained entirely on Blender-synthesised data, with zero hand-labelled
real images — mis-predicted it. A textbook train/deploy colour domain gap.

What worked in the end: on 2026-07-24 we photographed the actual arena objects and fine-tuned the
face model on them before the finals.

What we would build instead of relying on a same-day rescue:

* **Colour-domain randomisation as a first-class axis** in the synthetic generator — hue/ripeness
  jitter on the fruit textures, not just lighting and pose. The cost of a wider distribution is a
  point or two of validation accuracy; the cost of the gap was a match.
* **A 10-minute on-site capture-and-adapt loop as a planned step**, not an emergency: the field
  tooling (`field_autopilot.py record`) already captures raw pairs and crops. Making "shoot the real
  objects, fine-tune, A/B against the previous weight with `perception/eval/face_model_compare.py`"
  a scheduled 30 minutes would have removed the surprise entirely.
* **Value-aware hedging in the strategy layer.** When the fruit stage is the known weak point, and
  a mispick costs 2×, `--order shape-first` banks the certain 40 points before touching the risky
  20-pointers. We shipped the lever and did not have a rule for pulling it.

### Failure 2 — the gripper node crashed (final 1)

The robot navigated correctly, reached the object, and could not close on it. That match was 90
points had it grasped.

The runner does check the gripper: the health check probes the OpenRB board for real liveness (not
just topic existence), and `gripper_alive()` re-checks telemetry before the collection loop starts.
Both are **startup** checks. There is nothing watching the bridge *during* the match, no supervisor
to restart it, and no rule that says "if `/gripper/state` has gone stale for N seconds, stop driving
and recover". The state machine kept executing cycles against a dead actuator.

What we would build:

* **A liveness watchdog inside the cycle**, checked at the same place the grasp result is read: if
  telemetry is stale, attempt one bridge restart (the bridge owns the tty; the runner already knows
  never to open it directly) and re-home before retrying.
* **A supervised bridge process** — the bring-up already kills duplicate bridges; it should also
  respawn a dead one.
* **Escalation instead of repetition.** The grasp path retries `OPEN → +0.02 m → CLOSE` exactly
  once and then skips the cell. That is the correct response to a *missed* grasp and the wrong
  response to a *dead gripper*; the two are distinguishable from telemetry and were not
  distinguished.

### Failure 3 — ran out of time (final 2)

The 3-minute timeout hit while the robot was grasping the last object.

The runner is **not time-aware**. It compares total time against the 180 s budget in the report —
after the fact. Nothing in `stage_collect()` or `pick_target()` knows how much time is left, so it
will happily start a cycle it cannot finish, and a cycle it cannot finish scores exactly zero.

What we would build:

* **A deadline-feasibility check before each cycle.** Route length, standoff, grasp and carry are
  all predictable within a few seconds — `report.move_stats` and per-phase timings from every
  rehearsal already contain the distribution. Don't start a cycle whose predicted round trip exceeds
  the time remaining; spend the remainder on a closer object instead.
* **Shortest-job-first ordering**, which the 2026-07-17 analysis showed is the right objective for a
  single-depot, capacity-1 problem, and which we never implemented — the shipped runner takes the
  nearest object *to the robot*, not the one with the cheapest round trip to storage.
* **A per-cycle time box.** The measurement retry budget (6 attempts) and the rotation costs are
  reasonable at t = 30 s and indefensible at t = 165 s. The policy should tighten as the clock runs
  down.
* **Attack the known waste first.** The two largest measured time sinks were both structural, not
  algorithmic: in-place rotations (2–9 s each before the rotation-minimisation rework) and
  firmware settle grind on short moves (a −0.2 m reverse that took ~8 s against a 1.1 s prediction,
  caused by an unspecified `max_v` falling back to 0.233 m/s). Both were found by instrumenting
  every move with elapsed-vs-expected and reading the outliers. That instrumentation should have
  existed from the first field day, not the second-to-last.

### What we would keep

The parts that earned their complexity: the 42-cell snap that turns localisation into
classification; refusing to pick anything uncertain; depth-based ranging; measuring the ground-plane
value anyway so the bias stays visible; the pre-grasp veto that can only refute, never confirm; the
fail-closed placement gate; and writing a self-scoring `report.json` after every stage, which is the
only reason most of the numbers in this document exist.
