# Stitching and Calibration

Two RealSense cameras, one virtual frame. This document is the reference for the two
calibrations that make that frame usable: the **geometric** stitch (`near → top`, one 3×3
matrix per mast height, venue-independent) and the **photometric** lock (exposure / gain /
white balance, venue-dependent and re-measured on site).

The runtime contract the stitch has to satisfy — inference on the stitched frame, 3D
back-projection always in the *source* camera — is described in
[`pipeline.md` §1](pipeline.md#1-capture-and-stitch) and §7 and is not repeated here.
The operator procedures are in
[`../../docs/04-getting-started.md` §10.3–10.4](../../docs/04-getting-started.md#103-stitch-calibration--must-be-re-measured-on-your-robot),
and the symptom-first version is in
[`../../docs/06-troubleshooting.md` §4](../../docs/06-troubleshooting.md#4-cameras-stitch-seam-and-photometry).
This page is the *why* and the *format*.

**Contents**

1. [Why two cameras](#1-why-two-cameras)
2. [What the calibration is, and what it is per](#2-what-the-calibration-is-and-what-it-is-per)
3. [The `mid` height, and why it is not here](#3-the-mid-height-and-why-it-is-not-here)
4. [File format](#4-file-format)
5. [How the calibrator fits `A`](#5-how-the-calibrator-fits-a)
6. [The measured FHD result](#6-the-measured-fhd-result)
7. [Why one point was not enough](#7-why-one-point-was-not-enough)
8. [Resolution independence](#8-resolution-independence)
9. [Photometry: why it is locked](#9-photometry-why-it-is-locked)
10. [When to re-calibrate](#10-when-to-re-calibrate)

---

## 1. Why two cameras

A D435 colour stream has a vertical field of view of about 42°. The robot has to see, in the
same 3-minute match, a cube 3.5 m away at the far wall *and* a cube 25 cm in front of the
gripper. One 42° camera cannot do both: aim it at the horizon and the floor immediately in
front of the robot falls out of the bottom of the frame; aim it down and the far half of the
arena disappears.

So the robot carries two, on the same lifting mast, at fixed and very different tilts:
a **top** camera at ~19° below horizontal for search, and a **near** camera at ~53° for the
final approach and grasp alignment. The mount table is in
[`pipeline.md` §7](pipeline.md#7-depth-back-projection-to-a-3d-target).

Projecting those mounts onto the floor (mount pose from `perception/fieldlib.py:41`–`52`,
intrinsics from the crop model at 1920×1080, `perception/fieldlib.py:235`) gives the coverage
each camera actually has. These are derived numbers, not tape-measure ones:

| Mast | Camera | Vertical rays (below horizontal) | Floor coverage ahead of the wheel axis |
|---|---|---|---|
| down | top | −2.97° … 39.85° | 0.62 m → far wall |
| down | near | 31.56° … 74.84° | 0.27 m … 0.70 m |
| up | top | −2.02° … 40.80° | 0.78 m → far wall |
| up | near | 31.60° … 74.88° | 0.31 m … 0.94 m |

Two things follow.

- **The top camera alone is blind inside 0.78 m** with the mast up — which is exactly the
  band the gripper works in. The near camera exists to cover it.
- The two coverages **overlap** by about 15 cm of floor (0.78–0.94 m, mast up). That strip is
  the only place a single 2D transform can join the two images, and it is where every
  correspondence point has to be picked. The field runbook independently quotes the same
  strip as "77.5–92 cm ahead of the robot, ±40 cm laterally", which is where the tuner's
  floor-brightness statistics are measured.

The stitch seam is placed inside that strip: `seam = 1044` of a 2044-row stitched frame
corresponds to floor distance **0.82 m** ahead. Everything above row 1044 is top-camera
pixels, everything below is warped near-camera pixels.

One consequence is worth stating plainly because no calibration can fix it: **the alignment
is exact on the floor plane only.** The two cameras are ~4 cm apart vertically and tilted
34° apart, so anything with height that straddles the seam has parallax and will ghost. That
is why the calibration target must be flat, and why the seam is placed at 0.82 m rather than
somewhere an 8 cm cube is likely to sit during a scan.

Both cameras ride the mast (a 2026-07-18 conclusion that the lower one was chassis-mounted
was a mis-labelled measurement — see
[`../../hardware/docs/gripper-and-mast.md` §4.4](../../hardware/docs/gripper-and-mast.md#44-mast-height-is-a-perception-parameter)),
so the relative pose of the two cameras — and therefore the stitch — changes with mast height.

---

## 2. What the calibration is, and what it is per

The stored model is a single 3×3 homography `A` mapping **near pixels into the top frame
extended downward** (`perception/fieldlib.py:252`). `seam`, `left`, `right` and `out_h` say
how the composite frame is cut from it, and `to_source()` (`perception/fieldlib.py:375`) is
its exact inverse, which is what keeps depth back-projection valid.

The calibration is **per mast height and per camera pair**, and it is **not** per venue:

| Varies with | Stitch geometry (`up.json` / `down.json`) | Photometry (`arena.json`) |
|---|---|---|
| Mast height | **yes** — separate file per height | yes — measure with the mast **up** |
| Camera brackets touched / camera swapped | **yes** — re-fit | no |
| RGB resolution | no — rescaled analytically (§8) | no |
| Room lighting | no | **yes** — re-measure at every venue |

Both shipped heights are load-bearing at runtime: the 12-shot centre scan uses `up`
(`mission/match_runner.py:1061`), the mast-down pre-shot that owns the four near cells uses
`down` (`mission/match_runner.py:1049`), and the pre-grasp re-verification also uses `down`
(`mission/match_runner.py:1544`).

If a calibration file is missing, `Stitcher` falls back to the legacy translation-only
parameters in `STITCH_PARAMS` (`perception/fieldlib.py:57`) and reproduces the pre-2026-07-19
pipeline **bit for bit** — `tests/test_stitcher_calib.py` asserts exactly that, for both the
stitched pixels and `to_source()`. Deleting a calibration file is therefore a safe rollback
that requires no code change.

---

## 3. The `mid` height, and why it is not here

`perception/fieldlib.py` defines `TOP_MOUNT_MID` / `NEAR_MOUNT_MID` (an interpolation of the
up/down fits, flagged in the source as an estimate), the calibrator accepts `--mast mid`
(`perception/tools/stitch_calibrator.py:1026`), and the match runner accepts
`--scan-mast mid`.

**No `mid.json` is shipped, and `mid` was never used in competition.**

The reason is not policy, it is the file itself: the `mid.json` that once existed was a
**byte-identical copy of `up.json`** — same MD5, same six correspondence points, same
`created` timestamp, and `"mast": "up"` still inside it. No `mid` fit was ever performed.
Shipping it would have published a calibration that silently claims to be something it is not.

The code handles the absence correctly and loudly. `--scan-mast mid` is gated on the file
existing and exits with an explicit error telling you to raise the mast to `LIFT_TO_MID` and
run the calibrator (`mission/match_runner.py:2406`–`2410`). `photometry_tune.py` does not
offer `mid` at all — `--mast` is `("up", "down")` (`perception/tools/photometry_tune.py:418`).
Every competition run used `--scan-mast up`, the default.

---

## 4. File format

`perception/calibration/stitch/{up,down}.json`, written by `save_calib()`
(`perception/tools/stitch_calibrator.py:921`). Every field:

| Field | Meaning |
|---|---|
| `version` | `2` — the `A`-matrix format. Version 1 was the translation-only parameter set. |
| `res` | **The pixel coordinate system this file lives in**, e.g. `[1920, 1080]`. The single source of truth for §8's rescale. Written from the calibrator's own state, deliberately *not* re-read from the frame, so it can never disagree with `A`. |
| `mast` | `up` or `down`. Informational — the filename is what the loader keys on. |
| `model` | Which family was fitted: `translate` / `similarity` / `affine` / `homography`. |
| `A` | 3×3 near→top-frame transform, row-major. The only field the geometry depends on. |
| `seam` | Row in the top frame where top pixels stop and warped near pixels start. |
| `left`, `right` | Horizontal crop of the composite, in top-frame columns. Always recomputed from `A`'s real coverage (`coverage_bounds`, `perception/tools/stitch_calibrator.py:425`), never inherited from an older file — that inheritance is what left the FHD stitch cropped to a 4:3 sub-window before 2026-07-21. |
| `out_h` | Total stitched height = the lowest top-frame row the warped near image reaches. Recomputed if the calibration is rescaled to another resolution. |
| `points` | Every correspondence used, as `{"top": [u,v], "near": [u,v], "src": "manual"｜"auto"}`. Kept so a fit can be audited or re-run offline without the robot. |
| `residual_px.per_point` | Reprojection error of each point under the fitted `A`, in pixels. |
| `residual_px.rms` | RMS over **all** points, including RANSAC outliers. Not the acceptance metric. |
| `residual_px.rms_inlier`, `n_inlier`, `n_outlier`, `inlier_thresh_px` | The acceptance metric and its population: RMS over points within `inlier_thresh_px` (3.0) of the fit. See §5. |
| `decomposition` | Human-readable `scale_x`, `scale_y`, `rot_deg`, `dx`, `dy`, `persp` of `A`. Never read by the runtime — it exists so a human can sanity-check that a fit is a plausible camera-to-camera relationship and not a numerical accident. |
| `created`, `source` | Timestamp and where the frames came from (live topics, or an offline pair directory; `❄ FREEZE` if the live view was latched). |
| `note` | One-line summary auto-generated at save time. |

Saving backs the previous file up to `<mast>.json.bak_<timestamp>` first.

---

## 5. How the calibrator fits `A`

`perception/tools/stitch_calibrator.py` is a small HTTP server (default port 8099) that
serves a browser UI over either live camera topics or a directory of saved `*_top.png` /
`*_near.png` pairs. Working offline from saved pairs is a first-class mode — arena time is
scarce, and the fit does not need the robot.

**1. Correspondence points.** You click a physical point in the top image and then the same
physical point in the near image. A magnifier follows the cursor; a freeze button latches one
live frame so clicking, fitting and analysis all refer to the same pixels. Minimum 4 points,
recommended 6–8, spread across the **full width** of the overlap band. `auto_match()`
(`:284`) can propose candidates by AKAZE-matching the two images inside the band after
CLAHE, but an arena floor is usually too featureless for it to help much, and its output is
tagged `"src": "auto"` so a later reader can tell.

**2. The fit** (`fit_A`, `:168`). Four model families, each with a minimum point count:

| Model | Min points | Degrees of freedom captured |
|---|---:|---|
| `translate` | 1 | dx, dy |
| `similarity` | 2 | + uniform scale, roll |
| `affine` | 3 | + anisotropic scale, shear |
| `homography` | 4 | + perspective |

With ≥6 points the fit is RANSAC (`estimateAffinePartial2D` / `estimateAffine2D` /
`findHomography`, reprojection threshold 3.0 px); below that it falls back to a closed form —
a complex-number least squares for `similarity` (`y = c·x + t`, where `c` carries scale and
rotation together) and `np.linalg.lstsq` for `affine` — so the tool stays usable with four
points.

**3. The conditioning guard.** This is the part that matters most, and it is the result of a
simulation study recorded in the source (`:180`–`:196`). The overlap band is short in the
vertical direction, so correspondence points are inevitably squashed into a narrow range of
`v`. With 1 px of click noise and 8 points, the predicted error at the frame edges is:

| Vertical spread of the points | 5 px | 20 px | 40 px | 60 px |
|---|---:|---:|---:|---:|
| `similarity` | 1.2 | 1.1 | 1.1 | 1.2 |
| `affine` | 1.9 | 1.8 | 1.8 | 1.8 |
| `homography` | **330** | **83** | **46** | 6.0 |

A homography needs vertical spread to observe its perspective terms; without it the fit is
unconstrained and the corners diverge by hundreds of pixels while the residual at the clicked
points still looks fine. `similarity` is immune because the ~1900 px of *horizontal* extent
constrains roll on its own. So `fit_A` **refuses** to fit a homography below 40 px of vertical
spread and an affine below 15 px, and says which model to use instead. This is a hard refusal,
not a warning — a diverging homography is not detectable from the residual you are looking at.

**4. Inliers versus outliers** (`quality`, `:257`). RANSAC discards points at >3 px, but the
headline RMS is computed over *all* points, so a couple of mis-clicks drag it up and make a
good fit look bad. The docstring records the case that motivated the split: a `down`
calibration read 3.70 px overall and 0.93 px over its 4 inliers. **The acceptance threshold of
≤ 1.5 px is judged on the inlier RMS.** The UI colours the number green below 1.5 px, reports
`n_outlier` separately, and tells you what an outlier usually means: a mis-click, or a point
on an object with height rather than on the floor.

**5. Edge check** (`edge_check`, `:319`). Independently of the point fit, the band is split
into left / centre / right thirds and each third is phase-correlated between the top image and
the warped near image. Three *different* residual shifts mean scale or roll error is left over
— the thing a translation can never fix. Acceptance is all three within ±2 px (the
`--report` mode judges ≤3 px and sets the exit code). This is the metric that diagnosed the
original problem (§7), and it is the one to trust, because it uses every pixel in the band
rather than the handful you clicked.

---

## 6. The measured FHD result

Both shipped files were fitted natively at 1920×1080 on 2026-07-21, from live topics.

| | `up.json` | `down.json` |
|---|---|---|
| Model | affine, 6 points | similarity, 12 points |
| **Inlier RMS** | **0.78 px** (6 of 6 inliers) | **1.13 px** (5 of 12 inliers) |
| RMS over all points | 0.78 px | 5.39 px |
| Worst point | 1.26 px | 11.94 px |
| `scale_x` / `scale_y` | 1.00852 / 0.99212 | 1.02307 / 1.02307 |
| Rotation | 0.479° | 0.996° |
| `dx` / `dy` | −19.26 / +956.71 px | −32.20 / +998.35 px |
| `seam` / `out_h` | 1044 / 2044 | 1044 / 2137 |
| `left` / `right` | 0 / 1908 | 0 / 1912 |

Notes an implementer should not skip:

- **`up` is the good one.** 0.78 px inlier RMS with zero outliers, worst point 1.26 px, and
  the `left`/`right` bounds recovered to the full real coverage (0–1908 of 1920) rather than
  the inherited 4:3 crop.
- **`down` passes on inliers and is honestly weak overall.** 7 of its 12 points were rejected
  at the 3 px threshold, two of them at ~10–12 px. Read the point list and the reason is
  visible: several entries are near-duplicate clicks on the same marker across frames. Per §5
  the correct response is to delete the high-residual points and re-fit. `down` is used only
  for the mast-down pre-shot and the pre-grasp re-check, both at short range, which is why it
  survived as fitted. If you re-fit anything, re-fit this one.
- **Why `affine` for `up` and not `homography`.** The six `up` points span 33.7…60.7 px in
  near-image `v` — a spread of **27 px**. That clears the affine guard (≥15 px) and is
  rejected by the homography guard (≥40 px), exactly as designed. The `down` points span
  18.0 px, which is why `similarity` was used there.
- The fitted rotation of 0.479° for `up` is an independent confirmation of the roll error that
  phase correlation measured as ≈0.53° before any of this existed (§7). Two different methods,
  same number.
- `scale_x ≠ scale_y` for `up` (1.0085 vs 0.9921) is the anisotropy that motivated affine over
  similarity; it is small but it is 3 px across a 1908 px frame.

---

## 7. Why one point was not enough

Every stitch parameter before 2026-07-20 came from **one** correspondence — the front-face
marker of an 8 cm cube. One point determines translation and nothing else: it constrains
neither the scale difference (the two cameras do not have identical effective fields of view)
nor the relative roll. The result is the classic symptom: the seam matches in the middle and
splays apart at both edges.

Phase correlation over 81 saved pairs put a number on it:

| Mast | Left | Centre | Right |
|---|---|---|---|
| up | dx +13.30 / dy −10.54 px | +11.87 / −11.69 | **+7.74 / −16.42** |
| down | +0.07 / −0.02 | +0.64 / +8.17 | −1.63 / +4.77 |

The 5.6 px of dx and 5.9 px of dy disagreement between the left and right thirds of the `up`
band is ≈0.53° of roll. No amount of nudging `x_offset` removes it.

There was a second error in the same measurement, unrelated to geometry: the original target
was a cube face, i.e. a point **8 cm off the floor**. The two cameras are at different heights,
so an elevated point has parallax and cannot be aligned by any single 2D transform — including
the correct one. Use flat markers on the floor.

`tests/test_stitcher_calib.py::test_calibrator_fit_recovers_known_transform` encodes both
halves of this lesson as an assertion: it synthesises points from a known
similarity transform, checks the fit recovers scale to 1e-4 and rotation to 1e-3, and then
asserts that a **1-point translation fit leaves residuals > 3 px** on those same points.

---

## 8. Resolution independence

A stitch calibration must not be tied to the resolution it happened to be measured at. The
project changed RGB resolution twice (640×480 → 1280×720 → 1920×1080) and re-calibrating on
the arena floor each time was not an acceptable cost.

`Stitcher` therefore rescales the stored `A` analytically
(`perception/fieldlib.py:310`–`329`):

```
A' = S_top · A · S_near⁻¹        where   S = K_new · K_ref⁻¹
```

`S` is a pure affine pixel-to-pixel map, and it is *exact* — not an approximation — because
the two resolutions come from the same lens through a crop-and-scale relationship. `seam`,
`left` and `right` are carried through the top camera's `S`; `out_h` is recomputed from
scratch.

The crop model behind `S` is a measurement, recorded at `perception/fieldlib.py:210`–`217`
and `:235`: the RealSense 4:3 640×480 colour mode is not a downscale of the 16:9 sensor, it
is a **centre 1440×1080 crop scaled by 2.25**. The evidence is the intrinsics themselves —
the near camera measures `fx = 605` at 640 wide, which is a 1440-wide crop of a ~1380 px FHD
focal length (`1380 · 640/1440 ≈ 605`, the reasoning as written in the source comment);
a plain downscale of the full 1920-wide frame would have given `fx ≈ 460` instead.
`crop_model_k()` encodes that, and
`test_crop_model_k_matches_measured_fx_ratio` asserts the 2.25 factor and the 240 px principal
point offset.

Three practical points:

- **Live `camera_info` wins.** The crop model is only the fallback. `make_stitcher()`
  (`mission/match_runner.py:786`) reads real intrinsics from each camera's `camera_info` and
  passes them as `intr`; it drops to the crop model only if a `camera_info` has not arrived
  (`mission/match_runner.py:797`–`802`).
- **Any 16:9 pair works in either direction** (generalised 2026-07-21). An FHD-native
  calibration can be driven at 640×480, a 720p calibration at 1080p, and the 640→FHD path is
  bit-identical to the original implementation. A reference resolution that is neither 640×480
  nor 16:9 is **rejected with an error** rather than silently approximated — see
  `test_rescale_rejects_unknown_ref_res`.
- **The invariant that is actually tested** is not the matrix but the correspondence:
  `test_rescale_preserves_ground_correspondence` asserts `A' · S_near · p == S_top · A · p`
  for sample points at 1280×720 and 1920×1080, and `test_rescale_to_source_exact_inverse`
  asserts `to_source ∘ from_source == identity` after rescaling. The exact-inverse property is
  what the depth back-projection depends on, so it is tested at every resolution.

Because both shipped files are already FHD-native, the robot does no rescaling at all —
`res` is `[1920, 1080]` and the live streams run at 1920×1080×15.

---

## 9. Photometry: why it is locked

### The lock

Auto-exposure and auto white balance are **off** on both cameras, and fixed values are pushed
in at bringup. Two independent reasons:

1. **Two free-running cameras do not converge to the same exposure.** They see different parts
   of the scene at different tilts. Measured 2026-07-20 with the mast up: the seam split by
   **ΔY +12.4**, top brighter. Since the stitched frame is a single image to the detector, a
   split seam means a cube's appearance depends on which side of row 1044 it lands on.
2. **Auto-exposure makes detection non-reproducible.** The face model reads printed fruit
   icons that cover 8–10 % of a cube face (see [`../README.md`](../README.md)); its behaviour
   depends on contrast and colour. With auto-exposure on, the same cube at the same distance
   produces different pixels depending on what else is in frame, so a lab result does not
   transfer to the venue and a venue result is not reproducible five minutes later. Locking
   turns photometry into a *calibration* — one measured number, recorded in a file, re-measured
   when the room changes — instead of a hidden variable.

A related trap, worth repeating because it cost a day: the launch file locked white balance by
passing the **string** `"4600"`, while `realsense2_camera` declares that parameter as a
`double`. The set was rejected and the lock silently never applied. It has to be `"4600.0"`.

### What the tuner measures

`perception/tools/photometry_tune.py` is the ~1-minute CLI version of the calibrator's
photometry panel. `measure()` (`:100`) builds a `Stitcher` for the current mast and
resolution, computes the overlap band from `A` (the same band the geometry uses), and reduces
it to channel means over the **whole band** — deliberately not per-point, because point-wise
sampling at the extreme lens edge inflates colour differences with vignetting and local
texture (measured 2026-07-20).

`verdict()` (`:143`) then applies:

| Criterion | Limit | Constant | Rationale |
|---|---|---|---|
| Floor luma `Y`, each camera | 100 … 140 | `Y_LO`, `Y_HI` (`:65`) | Bright enough for icon contrast, dark enough that white objects keep detail. |
| Seam luma difference `|ΔY|` | ≤ 6 | `DY_MAX` (`:66`) | The seam must not be visible to the detector. |
| Saturation, either camera | ≤ 8 % of pixels at `Y ≥ 250` | `--max-sat` (`:416`) | A blown-out white cube loses the printed face entirely. Measured in the lab on 2026-07-20: at exposure 300 the strokes of the printed `X` marker disappeared. |
| Band left/right luma split | ≤ 25 | `verdict` (`:167`) | A large left-right split means a wall or a specular highlight has entered the band, not that exposure is wrong. The fix is to move the robot, and the message says so. |
| `\|ΔR/G\|`, `\|ΔB/G\|` | ≤ 0.05 **informational**; > 0.2 fails | `DC_MAX` (`:66`) | See below. |

**Colour difference is reported, not chased.** A 2026-07-20 sweep from 3200 K to 6000 K —
with the applied value verified back out of the parameter server — moved `ΔR/G` by **0.004**.
The residual colour difference between the two cameras is a sensor-to-sensor property
(D435 vs D435i) that runtime white balance does not touch, and it only affects the colour of a
crop that straddles the seam. The `--match-wb` mode that tried to correct it is deprecated and
prints its own obituary before running. Above 0.2, though, `verdict` fails hard — a gap that
large is not sensor variation, it is auto white balance still being alive.

**Measure with the mast up.** With the mast down the overlap band collapses to a thin strip
(29 rows at the 640-wide reference scale) at the extreme edge of both lenses, where vignetting and specular reflection dominate: ΔY **28** down versus
**6** up in the same lighting. The scan the stitch exists for happens with the mast up anyway.

**Selection rule** (`cmd_sweep`, `:269`): sweep the exposure candidates (default
`120,160,200,240,300`), drop any whose saturation exceeds `--max-sat`, and among the rest pick
the one whose mid-band luma is closest to `--target-y` (default 115). Then re-apply, re-measure
and re-judge — the sweep never trusts the value it just chose. Every run writes a contact sheet
of the seam strip at each exposure plus the final stitched frame with the seam drawn in red,
because the numeric criteria do not tell you whether a printed fruit face is legible and your
eyes do.

`--apply last` re-applies the stored values in ~15 s after a stack restart (the lightweight
stack cannot take exposure launch arguments, so a restart otherwise reverts to the lab
default), and `--restore` returns to the lab default of exposure 156 / gain 64 / WB 4600 K
(`DEFAULTS`, `:63`).

### The shipped record

`perception/calibration/photometry/arena.json` is one such sweep, from 2026-07-20 21:25,
renamed from the tool's timestamped output (`arena_20260720_212531.json`):

```jsonc
"applied":  { "exposure": 200, "gain": 64,
              "white_balance_top": 4600, "white_balance_near": 4600,
              "power_line_frequency": 2 },     // 2 = 60 Hz, anti-banding
"measured": { "y_top": 107.6, "y_near": 113.8, "dY": -6.19,
              "dRG": 0.145, "dBG": 0.062,
              "sat_top_pct": 0.0, "sat_near_pct": 0.0 },
"pass": false
```

It records `pass: false`, and the reason is a single criterion missed by 0.19: `|ΔY| = 6.19`
against a limit of 6.0. Both luma values are inside 100–140, saturation is zero, and the
colour deltas are the informational sensor-pair offset. The candidate table in the same file
shows what the sweep saw — exposure 120 → Y 87.0, 160 → 95.2, 200 → 109.6, 240 → 122.0,
300 → 138.8, all at 0 % saturation — and the tool chose 200 because it was closest to the
default target of 115. The file's `note` also records a correction: the `--match-wb` result of
4200 K was thrown out and 4600 K written instead, once white balance was proven not to affect
the inter-camera colour difference.

Photometry is the one calibration in this repository that belongs to a room rather than to a
robot (§10). The values above are the ones measured in that room, on that date, by that
sweep; they are the starting point, not a constant. Run the sweep in your own venue and
overwrite the whole file.

---

## 10. When to re-calibrate

**Geometry** — re-fit `up.json` and `down.json` when you touch a camera bracket, disassemble
or reassemble the mast, swap a camera, or change the RGB resolution to something the crop
model cannot express. Not when you change venue: lighting does not move brackets.

**Photometry** — re-measure at every venue, and twice on competition day (morning and just
before the match). It is the only calibration in this repository that depends on the room.

**Downstream check after either.** If the stitched height changes, `A1_IMGSZ_STITCHED = 896`
(`perception/fieldlib.py:67`) is stated in native-scale terms and must be re-checked with it —
see [`pipeline.md` §2](pipeline.md#2-a1--full-frame-object-segmentation).
Then run `python3 -m pytest tests/test_stitcher_calib.py -q`, which is what protects you from
a bad edit: it asserts the legacy fallback stays bit-exact and that `to_source` remains an
exact inverse at every resolution.

**Two path notes.** The tools' docstrings quote their older development locations
(`scripts/dev/field_ops/...`, `data/calibration/...`); the shipped locations are
`perception/tools/` and `perception/calibration/`, and `stitch_calibrator.py` writes to the
correct place because it resolves the path through `fieldlib`. `photometry_tune.py` does not:
its `REPO_ROOT` is `Path(__file__).resolve().parents[3]`
(`perception/tools/photometry_tune.py:52`), which was right when the file lived three levels
down under `scripts/dev/field_ops/` and now resolves one level **above** the repository. Its
sweep outputs (`OUT_DIR`, `:62`) and image artifacts (`artifact_dir`, `:182`) therefore land
outside the checkout. Change it to `parents[2]` — or use `fl.REPO_ROOT`, as
`stitch_calibrator.py` does — before running a venue sweep.
