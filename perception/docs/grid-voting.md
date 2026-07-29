# The Grid, the Scan, and the Vote

The arena rule that objects only ever sit on 42 fixed grid points is the single most
exploitable fact in the whole competition. It converts object localisation from a
continuous estimation problem — where every millimetre of camera, depth and pose error
lands in the answer — into a **classification over 42 labels**, where all of that error is
thrown away as long as it stays under 25 cm. Everything in this document is downstream of
that one observation.

This document covers the decision layer: the grid, the rotating scan that feeds it, how
observations accumulate into per-cell votes, how those votes resolve into an identity, and
the last-moment veto before the gripper closes. The perception stages that *produce* the
observations (stitch, A1, face model, pair verifiers, depth back-projection) are documented
in [`pipeline.md`](pipeline.md); the routing and control consequences are in
[`../../navigation/docs/control-and-routing.md`](../../navigation/docs/control-and-routing.md);
the rulebook-level strategy is in [`../../mission/docs/match-strategy.md`](../../mission/docs/match-strategy.md).
Cross-references are used instead of repetition.

Authoritative code: `perception/fieldlib.py` (grid, snap, face vote, GT comparison) and
`mission/match_runner.py` (scan, accumulation, finalisation, re-verification).

**Contents**

1. [The grid, and why quantising is the biggest single win](#1-the-grid-and-why-quantising-is-the-biggest-single-win)
2. [The scan: one point, eight headings, mast up](#2-the-scan-one-point-eight-headings-mast-up)
3. [What accumulates: the shape of a vote](#3-what-accumulates-the-shape-of-a-vote)
4. [Finalisation: the decision ladder](#4-finalisation-the-decision-ladder)
5. [The thresholds, and what the field actually ran](#5-the-thresholds-and-what-the-field-actually-ran)
6. [Votes leak into the router: inflation and the naive radius](#6-votes-leak-into-the-router-inflation-and-the-naive-radius)
7. [Pre-grasp re-verification: refute only, never confirm](#7-pre-grasp-re-verification-refute-only-never-confirm)
8. [Measured: the sim parity scans](#8-measured-the-sim-parity-scans)
9. [What we would change](#9-what-we-would-change)

---

## 1. The grid, and why quantising is the biggest single win

### The grid

```python
# perception/fieldlib.py:32
GRID_XS_CM = tuple(range(50, 351, 50))    # 50, 100, ... 350   (7 columns)
GRID_YS_CM = tuple(range(100, 351, 50))   # 100, 150, ... 350  (6 rows)
```

7 × 6 = **42 candidate positions** on a 50 cm pitch, in the rulebook's official
centimetre frame whose origin is the arena's bottom-left corner. The runner works in the
map frame (metres, origin at arena centre); the two are one translation apart
(`official_cm_to_map` / `map_to_official_cm`, `perception/fieldlib.py:174`–`179`).

Note what the y range implies: the lowest object row is y = 100 cm, so the strip below it is
provably empty for the entire match. The mission layer spends that fact on a free "bottom
highway" between START and STORAGE — see [`match-strategy.md`](../../mission/docs/match-strategy.md).

### The snap

```python
# perception/fieldlib.py:182
def snap_cell(map_x, map_y):
    cx, cy = map_to_official_cm(map_x, map_y)
    gx = min(GRID_XS_CM, key=lambda g: abs(g - cx))
    gy = min(GRID_YS_CM, key=lambda g: abs(g - cy))
    return (gx, gy), math.hypot(cx - gx, cy - gy) / 100.0
```

Nearest grid value independently per axis, plus the Euclidean residual in metres. Because
the axes are independent, each grid point owns a 50 × 50 cm Voronoi **square**: inscribed
radius 25 cm, circumscribed radius 35.4 cm.

### Why this is the biggest accuracy win in the system

Every error source in the perception chain adds into one number, the estimated map position
of the object:

| Source | Typical magnitude |
|---|---|
| Depth median at the contact pixel | ~1–3 cm |
| Camera mount fit (plane-fit RMS, four measured pairs) | 0.9–2.7 mm |
| Unmeasured mount roll (−1.0° to −2.5°), worst at frame edges | a few cm |
| `forward_m`, never measured by any calibration | unknown, small |
| LiDAR pose estimate feeding `to_map()` | a few cm |

Continuous localisation would have to carry all of that into the grasp. Snapping *discards*
it: an estimate is correct as long as it is closer to the true grid point than to any
neighbour, so the entire budget above only has to stay under **25 cm**, which it does by
roughly a factor of three. Quantisation is a free error-correcting code, and the arena
handed it to us.

The measured residuals from three recorded 8-shot rehearsal scans (§8) make the margin concrete:

| Run | Voting observations | Mean residual | Max residual |
|---|---:|---:|---:|
| `20260721_154536` | 27 | 5.9 cm | 14.2 cm |
| `20260721_155127` | 37 | 7.5 cm | 15.6 cm |
| `20260721_155553` | 35 | 8.8 cm | 20.7 cm |

An offline check on real-arena data reached the same conclusion earlier: 7/7 correct snaps
at a 7.5 cm mean residual (2026-07-17).

### The snap-error gate

```python
# mission/match_runner.py:1198
if snap_err > MAX_SNAP_ERR_M:          # MAX_SNAP_ERR_M = 0.30, match_runner.py:82
    det["why"] = "snap_err 초과"
    continue
```

The gate is at 0.30 m, deliberately **larger** than the 0.25 m decision boundary and
smaller than the 0.354 m corner of the Voronoi square — it accepts 95.1 % of the square's
area. That choice is not a compromise between the two, it reflects what the gate is *for*:
it exists to throw away geometry that is not an arena object at all (a detection on a wall,
a depth pixel that hit a hole and back-projected to nowhere), not to filter measurement
noise. Filtering noise is the vote's job.

The recorded rejections back this up. Across the three scans the gate fired exactly three
times, at residuals of **0.67, 0.80 and 0.84 m** — every one an order of magnitude past
anything a real object could produce. Not a single marginal case was ever rejected. A
tighter gate would have bought nothing and started costing real detections.

---

## 2. The scan: one point, eight headings, mast up

`stage_scan()` (`mission/match_runner.py:995`) drives to a single point, spins, and shoots.

**Where.** `CENTER_SCAN_XY = (0.25, 0.25)` map = official (225, 225) cm
(`perception/fieldlib.py:86`). Two properties, both recorded in the source comment:

- It is the true centre of the *object field* (x spans 50–350, centre 200; y spans 100–350,
  centre 225) — not the centre of the arena. The previous point (0.25, −0.25) sat 50 cm
  south, which pushed the northern row at y = 350 far enough away that it was repeatedly
  demoted as a distant observation. Changed 2026-07-21.
- It sits on a street intersection, exactly `hypot(25, 25)` = **35.4 cm** diagonally from its
  four nearest grid points, rather than on a grid point where the arena's symmetry would put
  many objects at identical ranges and identical appearances.

Worst-case distance from there to any of the 42 cells is `hypot(175, 125)` = **2.15 m**, to
the far corner cells at (50, 100) and (50, 350).

**How many shots.** `SCAN_SHOTS = 8` is the code default, but every competition match ran
**12** (`run_match_day.sh --scan-shots 12`), i.e. 30° steps through
a full turn, clockwise (counter-clockwise hit firmware settle timeouts). Between steps the
runner waits `SCAN_SETTLE_S = 0.35 s` to kill motion blur — reduced from 0.6 s on 2026-07-20
after that value looked like pure overhead, and exposed as `--scan-settle`. A
`--scan-mode continuous` alternative shoots while rotating without stopping and lets the
count fall out of the rotation duration.

**Mast up.** Both cameras ride the lifting mast. Raised, the top camera sits at 0.4986 m
with a 19.39° down-tilt, which makes the scan a much more top-down view: objects occlude each
other far less, and — decisively — the cube's **top** face presents real area. That face
matters because it is the only one of the three photographed faces visible from *every*
azimuth; the other two are an opposing pair of sides, so a mast-down scan sees a photograph
from only half the directions around a cube and sees it at grazing incidence from most of
those. The cost is a second calibration set (`TOP_MOUNT_UP` / `NEAR_MOUNT_UP`,
`perception/calibration/stitch/up.json`) and ~7 s up / ~6 s down, both hidden under other
work: the raise overlaps the drive to the centre, and the lower is issued *before* inference
so it overlaps the batched forward passes (`mission/match_runner.py:1037`).

**The structural blind spot.** The four cells diagonally adjacent to the scan point are only
0.354 m away, and at that range the near camera's extreme top-down view collapses A1
confidence to 0.05–0.11 — below the 0.25 scan threshold. On 2026-07-21 this was verified
twice: at mast-up (14:32) and again at mast-mid (17:12), **all 8 spin shots missed all four
cells**. Lowering the global threshold would have manufactured ghosts everywhere else, so
instead those four cells are handed to a dedicated **pre-shot**:

```python
# mission/match_runner.py:477
def preshot_cells() -> set:
    ox = fl.CENTER_SCAN_XY[0] * 100 + 200      # 225
    oy = fl.CENTER_SCAN_XY[1] * 100 + 200      # 225
    xs = (int(ox // 50) * 50, int(ox // 50) * 50 + 50)   # (200, 250)
    ys = (int(oy // 50) * 50, int(oy // 50) * 50 + 50)   # (200, 250)
    return {(x, y) for x in xs for y in ys}
```

One mast-**down** frame taken at the end of highway leg 1 (`_take_preshot`,
`mission/match_runner.py:938`), where those four cells are 1.8–2.3 m away and seen through
the best-calibrated mount pair. The ownership is exclusive in both directions, enforced
inside the shared detection loop:

```python
# mission/match_runner.py:1203
if only_cells is not None and cell not in only_cells:
    det["why"] = "프리샷 전담 외 셀"          # pre-shot may vote only for its four cells
    continue
if exclude_cells is not None and cell in exclude_cells:
    det["why"] = "프리샷 전담 셀 (초근접 스핀표 불신)"   # spin votes for them are rejected
    continue
```

and, because those cells get exactly one frame, their vote requirement is relaxed to 1 in
finalisation (§4). If the pre-shot capture fails, the runner disables it and the four cells
fall back to spin votes — degraded, but not broken.

---

## 3. What accumulates: the shape of a vote

A vote is **per detection**, not per shot. One stitched frame can vote for the same cell
twice, because the stitch is two cameras and each half back-projects independently through
its own intrinsics and its own mount (`fl.Stitcher.to_source`, see
[`pipeline.md` §7](pipeline.md#7-depth-back-projection-to-a-3d-target)). §5 has a real
example where that mattered.

```python
# mission/match_runner.py:1272
self.cell_votes[tuple(det["cell"])].append(
    {"identity": det["identity"],   # A1 class, or the face-vote result for cubes
     "a1": det["cls"],              # raw A1 class
     "a1_conf": det["conf"],
     "face_conf": det.get("face_conf"),   # None for non-cubes
     "cam": det["cam"],             # "top" | "near" — which physical camera
     "range_m": det["range_m"],
     "shot": shot["shot"]})
```

`identity` is where the two stages meet. For the three polyhedron classes A1's label *is*
the identity and there is no face stage. For `cube_like_object` the crop goes to the face
model and `identity` is replaced by the winner of `face_vote()` — one of
`apple / orange / banana / pineapple / plain` — with its confidence stored separately in
`face_conf`. The distinction matters at finalisation, because the cell rule trusts the two
fields differently.

Three properties of this accumulation are worth stating explicitly:

- **The scan threshold is deliberately low.** `A1_SCAN_CONF = 0.25`
  (`mission/match_runner.py:80`), with the Korean comment "스캔: 저문턱 (셀 투표 누적이
  거름)" — *low threshold on the scan; the accumulated cell vote is the filter, not the
  detector*. The face model runs even lower, at `conf = 0.1`. Selection is pushed as far
  downstream as possible, where there is more evidence to select with.
- **Distance no longer demotes.** `FAR_PRESENCE_M = 2.5` survives in
  `perception/fieldlib.py:74`, and the demotion it controlled was removed on 2026-07-19
  (`mission/match_runner.py:1197`). Distant observations now vote normally. The consequence
  is that `self.presence` is **never incremented anywhere in the runner** — the
  presence-only branches in `_finalize_scan`, `_render_grid_png` and `build_obstacles` are
  reachable only by loading an older `grid_map.json` through `--map-file`
  (`mission/match_runner.py:2495`). Published as dead-but-loaded code rather than quietly
  removed, because a saved map from before 07-19 still parses.
- **Everything is batched.** All twelve shots go through A1 as one chunked batch, then every
  cube crop from every shot goes through the face model as a second batch, then every fruit
  face patch goes through each pair verifier route as a third. Voting happens after all
  three (`_process_shots_batch`, `mission/match_runner.py:1111`). Measured stage cost for one
  8-shot rehearsal scan producing 18 crops: stitch 0.064 s, A1 0.137 s, geometry+snap 0.087 s,
  face+pair 0.080 s.

---

## 4. Finalisation: the decision ladder

`_finalize_scan()` (`mission/match_runner.py:1343`) walks every cell that collected any
votes and applies four rules in order.

### Rule 0 — enough votes at all

```python
# mission/match_runner.py:1349
if len(vs) < (1 if cell in pre else self.votes_k):
    continue
```

Below the threshold the cell is not confirmed and simply does not exist for the rest of the
match: not a target, and not an obstacle either. Pre-shot cells need 1.

### Rule 1 — strong fruit faces win

```python
# mission/match_runner.py:1351
fruit_hits = Counter(v["identity"] for v in vs
                     if v["identity"] in fl.FRUITS
                     and (v["face_conf"] or 0) >= fl.CELL_FRUIT_CONF)   # 0.5
```

Only observations that are a fruit **and** carry a face confidence ≥ `CELL_FRUIT_CONF = 0.5`
count. If the leading fruit has at least `fruit_k` such hits, it wins outright — regardless
of how many `plain` observations the cell also collected, and regardless of their
confidence.

This is the cell-level half of the asymmetry. The other half is inside a single crop
(`face_vote()`, `perception/fieldlib.py:465`, documented in
[`pipeline.md` §5](pipeline.md#5-face-level-vote-inside-one-crop)). Both exist for the same
rulebook reason: a fruit cube carries its three printed faces on a vertical ring — front, top,
back — leaving left, right and bottom blank. Two of the four side azimuths are therefore
*legitimately* blank on a real fruit cube, and the top face only presents usable area from an
elevated viewpoint. Most observations of a fruit cube are consequently `plain` observations:
fruit faces were measured at only **10–24 %** of all face observations across a scan. A
symmetric majority rule would label almost every fruit cube `plain`.

The asymmetry is not free, and the direction of the risk is exactly the direction the
thresholds are tuned against: a single false fruit face can capture a cell that a dozen
honest `plain` observations agree about. That is what `fruit_k` is for, and §5 shows what
happens when it is set to 1.

### Rule 2 — the tie, re-decided on summed confidence

```python
# mission/match_runner.py:1359
if len(top) > 1 and (top[0][1] - top[1][1]) < 2:
    conf_sum = defaultdict(float)      # sum face_conf per fruit over qualifying votes
    ...
    if ranked[0][1] - ranked[1][1] >= FRUIT_TIE_CONF_MARGIN:   # 0.10
        identity = ranked[0][0]
    else:
        conflict = "conflicting_fruit"
```

Note the condition: the tie-break fires whenever the top two fruits are within **less than
two votes**, so a 1–1 tie *and* a 2–1 lead both get re-decided on summed face confidence.
Only genuine ambiguity — a confidence lead under 0.10 — is held as `conflicting_fruit`.

The rule was written against a specific loss, and the Korean comment records it in full. On
2026-07-21 at 17:12 a misread `pineapple` at 0.773 tied a correctly-read `apple` at 0.924.
Under the old rule the cell became `conflicting_fruit`, `identity` stayed `None`, the Rule 3
fallback then collapsed *both* fruit labels to `plain` — and a true apple cell was confirmed
as a plain cube and never picked. One false positive did not merely add noise; it laundered
a 20-point target into a non-target.

A held cell is still confirmed (Rule 3 gives it an identity) but is flagged, and
`match_candidates()` excludes any flagged cell from target selection
(`mission/match_runner.py:467`) — uncertain means *do not pick*, because a mispick costs
twice the object's base points.

### Rule 3 — A1-confidence-weighted fallback

```python
# mission/match_runner.py:1379
tally = defaultdict(float)
for v in vs:
    lbl = v["identity"] if v["identity"] in fl.POLYHEDRA else (
        "plain" if v["identity"] in fl.FRUITS | {"plain"} else v["identity"])
    tally[lbl] += v["a1_conf"]
identity = max(tally.items(), key=lambda kv: kv[1])[0]
```

If no fruit identity survived, every fruit label is collapsed to `plain` and the cell is
decided by **summed A1 confidence** over shapes and `plain`. The collapse is the honest
answer: a cube whose top face was never resolved is, for target purposes, a plain cube.
Weighting by A1 confidence rather than counting votes means a single crisp 0.97 octahedron
outranks two hesitant 0.3 detections.

One latent case is visible in that expression. If the face model returns **zero** boxes for a
crop even at `conf = 0.1`, `face_vote()` returns `None`, `identity` stays as the raw A1 label
`cube_like_object`, and the `else` branch keeps it — so a cell can in principle be confirmed
with the identity `cube_like_object`, which is not a member of `fl.CLASSES`. It would then
never match a target and would count as `wrong` against ground truth. It does not occur in
any recorded run (the face model always returns something at 0.1), but it is a real path.
Collapsing `cube_like_object` to `plain` alongside the fruits would be the one-word fix.

### Output

Confirmed cells are written to `grid_map.json` (reloadable with `--skip-scan --map-file`),
rendered to `grid_map.png` for a human, and — when a ground-truth layout was supplied —
compared cell by cell with `compare_with_gt()` (`perception/fieldlib.py:150`) into
`ok / wrong / ghosts / missed`, printed **before** collection starts. A bad scan is caught
before the robot commits to picking anything.

---

## 5. The thresholds, and what the field actually ran

### The checked-in defaults

| Constant | Value | Where | Role |
|---|---:|---|---|
| `CELL_VOTES_MIN` | 3 | `perception/fieldlib.py:73` | Observations before a cell may be confirmed at all |
| `CELL_FRUIT_CONF` | 0.5 | `perception/fieldlib.py:70` | What counts as a *strong* fruit observation |
| `CELL_FRUIT_K` | 2 | `perception/fieldlib.py:71` | Strong fruit hits needed to override the blank majority |
| `FRUIT_TIE_CONF_MARGIN` | 0.10 | `mission/match_runner.py:83` | Confidence-sum lead needed to break a fruit tie |
| `MAX_SNAP_ERR_M` | 0.30 | `mission/match_runner.py:82` | Snap residual above which an observation is discarded |

Both K values are overridable at the command line and both are resolved once at
construction (`mission/match_runner.py:499`):

```python
self.votes_k = args.votes_k or fl.CELL_VOTES_MIN     # --votes-k
self.fruit_k = args.fruit_k or fl.CELL_FRUIT_K       # --fruit-k
```

`CELL_FRUIT_K = 2` is a measured choice. On five real arena sessions (361 detections):

| Rule | Fruit cubes correctly identified |
|---|---:|
| Plain majority vote | 1 |
| Asymmetric, K = 1, conf ≥ 0.5 | 10 — but one octahedron was flipped to pineapple by a single false positive |
| **Asymmetric, K = 2, conf ≥ 0.5** | **9**, with all polyhedra intact |

One correct fruit traded for the removal of a whole class of catastrophic error, in a game
where a mispick is worth −2×.

### What the last real field runs used

> **This is the important part of this section.** Every recorded end-to-end run on
> 2026-07-21 — the last full runs before the competition — passed
> **`--votes-k 1 --fruit-k 1`**, not the checked-in `3` and `2`. `report.summary.effective_params`
> records it in each one. If you reproduce a published scan number with the defaults, you
> will not get the published number.

The reason is in the recorded vote distributions. Cell vote counts from the two clean
8-shot scans:

| Run | Cells with 1 vote | 2 votes | 3+ votes | Confirmed cells |
|---|---:|---:|---:|---:|
| `20260721_154536` | 17 | 5 | **0** | 22 |
| `20260721_155127` | 21 | 5 | **2** | 28 |

With `votes_k = 3`, run `155127` would have confirmed **2 cells out of 28** and run `154536`
**zero**. The spin does not see most cells once per shot; it sees most cells *once, total*. The
default was written for an imagined observation density the scan never delivered, and under
a 180 s budget there was no time to add scan points to produce it (the 4-point tour analysed
in [`match-strategy.md` §3](../../mission/docs/match-strategy.md#3-scanning-one-centre-point-mast-up-eight-headings)
was never run on hardware).

### What `fruit_k = 1` cost, measured

Cell (150, 200) in run `20260721_155553`, ground truth **octahedron**. Two observations, both
from the *same* stitched frame (shot 7), one per camera:

| Camera | A1 class | A1 conf | face identity | face conf | snap residual |
|---|---|---:|---|---:|---:|
| top | `octahedron` | 0.974 | — | — | 0.059 m |
| near | `cube_like_object` | 0.641 | `banana` | 0.664 | 0.127 m |

Result: `{'identity': 'banana', 'votes': 2, 'fruit_hits': {'banana': 1}}`. With
`fruit_k = 1`, one fruit face at 0.664 outranked an octahedron read at 0.974 and the cell
became a banana. With the checked-in `fruit_k = 2` the fruit branch would have been
rejected, Rule 3 would have summed A1 confidence (0.974 octahedron vs 0.641 collapsed to
plain) and the cell would have been correct.

That is precisely the failure the K = 2 sweep predicted a year of hindsight earlier —
"one octahedron was flipped to pineapple by a single false positive" — reproduced verbatim
except for the fruit. The lowered thresholds bought coverage (28 confirmed cells instead of
2) and paid for it in exactly this coin. Both halves of that trade are real; we shipped the
coverage because a cell that is never confirmed scores zero with certainty.

---

## 6. Votes leak into the router: inflation and the naive radius

Confirmed cells are not only targets — they are the obstacle map. `build_obstacles()`
(`mission/match_runner.py:1462`) turns every confirmed, uncollected cell into an obstacle
carrying its **vote count**, and the router inflates each obstacle by an amount that shrinks
as the evidence grows:

```python
# mission/match_runner.py:283
def inflation_for(votes):
    return CORRIDOR_BASE_M + ROBOT_RADIUS_M + VOTE_UNCERT_M * (1.0 / max(1, votes))
    #      0.10             + 0.16           + 0.05 / votes    →  0.26 … 0.31 m
```

A cell seen once is treated as less precisely located than a cell seen five times, which is
the correct instinct. The trap is that `ROBOT_RADIUS_M = 0.16` is the robot's
**circumscribed** radius — right for driving in an arbitrary direction, wrong for the street
grid, whose corridors are 25 cm half-width. `0.31 > 0.25`, so with the naive radius **every
legal street was reported blocked**, and the router fell through to detours that were worse
in every way. The fix passes an explicit clearance for axis-aligned street legs:

```python
# mission/match_runner.py:331
# 일반 inflation_for(0.31, 외접반경 기준)를 쓰면 정상 street 도 전부 막힘 판정.
STREET_CLEAR_M = 0.20    # robot half-width 0.095 + object half-width 0.04 + 0.065 pose margin
```

`corridor_free(..., clearance=STREET_CLEAR_M)` overrides the vote-based inflation for street
legs only; every other leg still uses the conservative circumscribed value. The full story,
including the 2026-07-20 bug where a detour waypoint was computed and never validated (a
carry route passed 6 obstacle cells with 4.8 cm minimum clearance), is in
[`control-and-routing.md`](../../navigation/docs/control-and-routing.md#why-the-naive-inflation-radius-declared-every-street-blocked).

One coupling worth flagging: lowering `votes_k` to 1 does not only add targets, it adds
**obstacles at their maximum inflation radius** (0.31 m, since `votes = 1` maximises the
uncertainty term). Coverage and route freedom pull in opposite directions through the same
constant, and nothing in the runner models that trade-off explicitly.

---

## 7. Pre-grasp re-verification: refute only, never confirm

The scan decides identity from up to 2 m away, top-down, with the mast up. The robot then
drives up to 2 m, lowers the mast, and closes a gripper on whatever is actually there. Any
error in between — a wrong cell, an object nudged, a face misread — is about to become a
−2× mispick. `verify_target()` (`mission/match_runner.py:1604`) spends one last A1 + face
(+ pair) inference on the final approach frame, the last moment anything is still
recognisable: after the final advance the object sits ~11 cm in front of the gripper.

The rule that makes it useful is that **only a confirmed contradiction skips the cell.**

| Reading at close range | Decision | Why |
|---|---|---|
| Target is a polyhedron, A1 re-reads it as anything else | **skip** | Definite contradiction |
| Target is a fruit, A1 says it is not a cube | **skip** | Definite contradiction |
| Target is a fruit, face model reads a *different* fruit at conf ≥ `FACE_FRUIT_MIN` (0.30) | **skip** | Definite contradiction |
| Target is `plain`, face model reads any fruit at conf ≥ 0.30 | **skip** | Definite contradiction |
| Face model reads `plain` | **pass** | Undecidable |
| No face detection at all, or a fruit face below 0.30 | **pass** | Undecidable |
| Face inference raises, or no crop is available | **pass** | Trust the scan identity |

The asymmetry is forced by the same rulebook fact that motivated asymmetric fruit voting —
though the docstring that originally justified it got the fact wrong, and the correction is
worth reading because the conclusion survived it unchanged. Translated from the correction
note in `mission/match_runner.py:3576-3585` (2026-07-23):

> ⚠ The previous comment's premise was wrong. Discarded: *"by the layout rule a cube's side
> faces are always plain and the fruit face is only on top."* The actual rule is **top 1 +
> one opposing pair of sides = fruit**, **bottom 1 + the other opposing pair = plain**. So
> **two of the four side azimuths do carry a photograph**, and any two adjacent sides are
> always one fruit and one plain — which means a corner (45°) view is guaranteed one fruit
> face. That is why close-range re-verification works as well as it does.
>
> The conclusion (`plain` observed → undecidable → pass) is unaffected: even for a fruit
> cube, two of the four side azimuths are legitimately plain, so a single `plain` observation
> carries a likelihood ratio of only 1:2 against a genuine plain cube. That is not enough to
> refute on its own.

So the reason a low mast-down `plain` cannot veto is *not* that a fruit cube's sides are
always blank — it is that they are blank half the time by design. Treating `plain` as a
mismatch would veto real targets on roughly half of all approaches, which is strictly worse
than not verifying at all. The close-range view can **refute** an identity — an actively
detected *different* fruit is a real contradiction — but it cannot **confirm** one, so it is
only granted a veto.

The verification runs the identical contract as the scan — same 0.18 crop pad, same
`imgsz = 224`, same `face_vote`, same pair routing and the same `PAIR_CONF = 0.80` label
replacement — so a pass here means the same thing it meant during the scan.

It fires in practice. In run `20260721_155553` a cycle was aborted with
`face re-read pineapple(0.58) ≠ target apple — contradiction`, spending 15.7 s instead of
−40 points. (Kinematic surrogate run — the timing is not a hardware measurement.)

---

## 8. Measured: the sim parity scans

Three complete 8-shot rehearsal scans against the same 28-object ground-truth layout
(`gt_seed14.txt`: 12 polyhedra — 4 each of octahedron / dodecahedron / icosahedron — plus
12 fruit cubes — 3 each of apple / orange / banana / pineapple — plus 4 plain cubes, on the
42-point grid),
run in the Isaac Sim kinematic parity environment on 2026-07-21, all with
`--votes-k 1 --fruit-k 1 --pair off --range-mode depth`, mast up, 8 step shots. These are
**not** real-arena scans; the surrogate is kinematic only (see
[`../../docs/adr/0004-simulation-is-kinematic-only.md`](../../docs/adr/0004-simulation-is-kinematic-only.md)),
and it uses the same models and the same code path as the robot but not the same photons.

| Run | Cells confirmed | Correct | Mis-identified | Phantom | Missed |
|---|---:|---:|---:|---:|---:|
| `20260721_154536` | 22 | 19 | 3 | **0** | 6 |
| `20260721_155127` | 28 | 23 | 5 | **0** | 0 |
| `20260721_155553` | 25 | 20 | 5 | **0** | 3 |

The headline result is the one that is easy to skim past: across three scans and 75
confirmed cells, **zero phantoms**. Not one detection was ever placed at a grid point that
had no object. Combined with the residual statistics in §1 — mean 5.9–8.8 cm, max 20.7 cm,
against a 25 cm decision boundary — this says the position half of the problem was
effectively solved by the grid.

Every one of the 13 mis-identifications sat at the **correct** cell. They are identity
errors, not localisation errors:

| Run | Cell | Predicted | Truth |
|---|---|---|---|
| `154536` | (350, 200) | apple | orange |
| `154536` | (50, 150) | apple | orange |
| `154536` | (150, 100) | apple | pineapple |
| `155127` | (350, 200) | apple | orange |
| `155127` | (150, 300) | apple | orange |
| `155127` | (50, 150) | plain | orange |
| `155127` | (100, 350) | apple | pineapple |
| `155127` | (150, 100) | apple | pineapple |
| `155553` | (350, 200) | apple | orange |
| `155553` | (150, 300) | apple | orange |
| `155553` | (100, 350) | plain | pineapple |
| `155553` | (150, 100) | apple | pineapple |
| `155553` | (150, 200) | banana | octahedron |

Two patterns, both of which went on to matter:

- **Ten of thirteen errors read a non-apple fruit as `apple`** — six orange → apple and four
  pineapple → apple.
  This is the apple/orange confusion the binary pair verifier was built to fix and
  measurably failed to fix (see
  [`pipeline.md` §6](pipeline.md#6-binary-pair-verification--shipped-and-disabled): a sweep
  over 771 accumulated field crops recorded the AO gate performing "swap orange → apple" 48–49
  times, the opposite of its intended direction). It is also, in substance, the failure that
  cost us the qualifiers, where the arena's printed apple turned out to be a yellowish
  under-ripe apple rather than the announced red one and the face model — trained on Blender
  renders with zero hand-labelled real images — mis-read it. A colour-domain gap in the
  synthetic generator shows up here as a systematic pull towards `apple`, in simulation,
  three days before it showed up on the scoreboard. We had the evidence and did not read it
  that way.
- **One error was not a face error**: (150, 200), octahedron → banana, dissected in §5. It is
  the `fruit_k = 1` failure mode, not a model failure.

Two `plain` outcomes (orange and pineapple cells read as plain) are the opposite kind of
error and cost far less: a fruit that reads as plain is simply not picked, worth 0 rather
than −40.

For an end-to-end reading of these runs — timing, grasps, placements — see
[`match-strategy.md`](../../mission/docs/match-strategy.md).

---

## 9. What we would change

- **Make `votes_k` adaptive, not constant.** The measured vote distribution (21 of 28 cells
  seen exactly once) means a fixed threshold of 3 is a mass deletion and a threshold of 1 is
  no filter at all. The information the runner already has — how many shots covered the
  cell's bearing, how far it was, whether it fell in the near blind zone — is enough to set a
  per-cell expectation and require agreement relative to *opportunity* rather than an
  absolute count.
- **Keep `fruit_k = 2` and buy the coverage elsewhere.** The trade in §5 was forced by a
  scan that produced too few observations, not by the vote rule. A second scan point, or
  simply more shots at the same point, would have restored the margin the K = 2 sweep
  measured. The 2026-07-17 analysis showed 4 quadrant-centre scan points cut the worst-case
  object distance from 2.15 m to 0.98 m; we never ran it on hardware because of the 180 s
  budget.
- **Weight votes by expected reliability.** Every vote currently counts the same whether it
  came from a 2.1 m top-camera detection at the frame edge or a 0.7 m near-camera detection
  at the centre. `range_m`, `cam` and `snap_err_m` are all recorded in the vote record and
  none of them is used in the decision.
- **Collapse `cube_like_object` to `plain`** in the Rule 3 fallback (§4).
- **Model the coverage/clearance coupling.** Lowering `votes_k` simultaneously adds targets
  and adds maximally-inflated obstacles (§6). One knob should not silently move two
  objectives in opposite directions.
- **Read the sim confusion table as a domain-gap alarm.** Ten of thirteen scan errors pulled
  towards `apple` three days before an under-ripe apple cost us a qualifier match. The
  numbers were in `report.json`; nobody asked what direction the errors pointed.
