# The Two-Stage Perception Pipeline, Stage by Stage

This is the runtime path that produced the arena map in competition: two stitched RGB
frames in, a 42-cell grid of object identities out. Every threshold below is quoted with
the reason it has the value it has — most of those reasons were recorded only as Korean
comments in the source, and are translated here.

The authoritative implementations are `perception/fieldlib.py` (shared contract) and
`mission/match_runner.py` (the match state machine that calls it).

**Contents**

1. [Capture and stitch](#1-capture-and-stitch)
2. [A1 — full-frame object segmentation](#2-a1--full-frame-object-segmentation)
3. [Distance gating and crop preparation](#3-distance-gating-and-crop-preparation)
4. [Face segmentation on cube crops (batched)](#4-face-segmentation-on-cube-crops-batched)
5. [Face-level vote inside one crop](#5-face-level-vote-inside-one-crop)
6. [Binary pair verification — shipped, and disabled](#6-binary-pair-verification--shipped-and-disabled)
7. [Depth back-projection to a 3D target](#7-depth-back-projection-to-a-3d-target)
8. [Grid snap and per-cell K-vote](#8-grid-snap-and-per-cell-k-vote)
9. [Measured cost](#9-measured-cost)

---

## 1. Capture and stitch

**In:** two RealSense RGB streams — an upper camera and a lower (near) camera, both riding
the same lifting mast. **Out:** one tall stitched RGB frame, plus the raw top/near pair and
the aligned depth pair, all archived separately.

Inference always runs on the stitched frame. 3D back-projection *never* does — it always
returns to the original camera's pixels first (§7). That is why the stitch model is a
single 3×3 homography `A` mapping **near pixels into the top frame extended downward**,
with an exact inverse:

```python
# perception/fieldlib.py:375
def to_source(self, u, v):
    if v < self.seam:
        return "top", u + self.left, v
    p = self.A_inv @ np.array([u + self.left, v, 1.0])
    return "near", float(p[0] / p[2]), float(p[1] / p[2])
```

Two properties follow from writing it this way:

- The legacy translation-only stitch is a special case, `A = [[1,0,x_offset],[0,1,seam−crop_top],[0,0,1]]`,
  so a missing calibration file degrades to exactly the old behaviour
  (`perception/fieldlib.py:57`).
- When `A` is an integer translation the stitch is done by array slicing, not
  `warpPerspective` — same pixels, no interpolation cost (`perception/fieldlib.py:335`).

Calibration is resolution-independent. The stored calibration is rescaled to whatever
resolution the cameras are actually publishing with `A' = S_top · A · S_near⁻¹`, where
`S = K_new · K_ref⁻¹` (`perception/fieldlib.py:311`–`329`). The comment at
`perception/fieldlib.py:210` records the measurement that justifies the crop model: the
RealSense 4:3 640×480 mode is a centre 1440×1080 crop of the 16:9 sensor scaled by 2.25,
proven by `fx = 605 = 1380·640/1440` — a pure downscale would have given `fx = 460`.

Shipped calibrations (`perception/calibration/stitch/`):

| Mast | Model | Points | Inlier RMS | `seam` | `out_h` | Resolution |
|---|---|---:|---:|---:|---:|---|
| `up` | affine | 6 | 0.78 px (0 outliers) | 1044 | 2044 | 1920×1080 native |
| `down` | similarity | 12 | 1.13 px (5 inliers, 7 outliers) | 1044 | 2137 | 1920×1080 native |

A `mid` height exists in code as the midpoint bootstrap of up/down and is explicitly
flagged as an estimate, not a fit (`perception/fieldlib.py:47`).

**Rule:** the raw top/near pair is always saved next to the stitched image
(`mission/match_runner.py:1306`). A failure that is only visible in the stitched frame
cannot be diagnosed later without the source pixels.

---

## 2. A1 — full-frame object segmentation

**In:** the stitched frame, converted to BGR. **Out:** instance masks and boxes over four
classes — `cube_like_object`, `octahedron`, `dodecahedron`, `icosahedron` — with confidence.

```python
# mission/match_runner.py:1148
results = _chunked_predict(self.models[0], bgr_l, A1_BATCH_CHUNK,
                           imgsz=fl.A1_IMGSZ_STITCHED, conf=A1_SCAN_CONF, verbose=False)
```

| Constant | Value | Where | Why |
|---|---:|---|---|
| `A1_IMGSZ_STITCHED` | 896 | `perception/fieldlib.py:67` | Inference size for the stitched frame, which is always taller than the model's training size — ~890 px at 640×480 input, where running A1 at 640 would downscale by 0.72× and lose small-object recall. A1 itself was trained at 640. |
| `A1_SCAN_CONF` | 0.25 | `mission/match_runner.py:80` | Deliberately low during the scan — cell-level vote accumulation (§8) is the filter, not the detector threshold. |
| `A1_APPROACH_CONF` | 0.35 | `mission/match_runner.py:81` | Higher when approaching a single known target, where a false positive is expensive. |
| `A1_BATCH_CHUNK` | 8 | `mission/match_runner.py:180` | All eight scan shots go through the model as one chunked batch. |

Non-cube polyhedra are finished here. Their A1 label *is* their identity; they never touch
the face model.

On a Jetson, a TensorRT FP16 engine built from the same weights is preferred and `.pt` is
the fallback. Engines are device- and TensorRT-version-bound and are deliberately **not**
distributed; build your own.

---

## 3. Distance gating and crop preparation

**In:** each `cube_like_object` box plus the frame. **Out:** one crop per cube worth
looking at.

### The crop

```python
# mission/match_runner.py:1213
pad = int(max(x2 - x1, y2 - y1) * 0.18)
crop = bgr[max(0, y1 - pad):min(h, y2 + pad),
           max(0, x1 - pad):min(w, x2 + pad)]
```

The 0.18 pad is not a taste decision: it is the pad the *training* crops were exported
with. The face model was trained on crops cut this way from the same A1 boxes, so runtime
must cut them the same way.

### The train/serve skew bug

The export pipeline did this:

```
A1 box → expand by max(w, h) × 0.18 → crop → cv2.resize(crop, 224×224, INTER_AREA) → train
```

— a **plain square resize**, which distorts aspect ratio. The runtime originally handed
the raw rectangular crop to Ultralytics, which **letterboxes** it: pads to square with grey
bars and preserves aspect ratio. Same nominal `imgsz`, different pixels, and the model had
never seen a letterboxed crop. This was found and fixed on 2026-07-03; the fixed runtime
resizes the crop to `imgsz × imgsz` with `INTER_AREA` before predicting, exactly matching
the export.

> **The two runtimes differ here.** The field runner does *not* do the square resize —
> `mission/match_runner.py:1213`–`1228` passes the padded rectangular crop straight to
> `predict(imgsz=224)` and lets Ultralytics letterbox it. The square-resize fix lives in the
> offline reference runtime, not in the robot code. Both paths were used to produce results
> in this project; when reproducing a number, check which one produced it. If you are
> building on this, match the export: square `INTER_AREA` resize.

### The size gate

A cube too far away cannot have its printed face read at all, and asking anyway produces
low-confidence noise. The reference runtime therefore refuses to call the face model when:

```
bbox_area < max(6000 px², 0.003 × frame_area)   or   short_side < 80 px
```

and emits the pseudo-class `cube_too_far` instead.

The value of that gate was measured on 361 real detections from five arena sessions, and
the conclusion **reversed** once the `imgsz=224` / BGR bugs (see [`../README.md`](../README.md))
were fixed:

| `short_side` threshold | ≈ distance | `cube_too_far` | `unresolved` | fruit faces read | clusters |
|---:|---:|---:|---:|---:|---:|
| 45 px (default) | 1.09 m | 113 | **4** | **27** | 25 |
| 70 px | 0.70 m | 202 | 0 | 16 | 24 |
| 100 px | 0.49 m | 244 | 0 | 8 | 24 |
| 130 px | 0.38 m | 256 | 0 | 5 | 24 |

Before the inference fix, the same sweep showed 69 unresolved detections at 45 px, and the
recommendation had been to raise the threshold to 100 px. Afterwards, the "noise" the gate
was cleaning up was gone — it had been a bug artefact — and raising the threshold only
destroyed good identifications (fruit faces read: 27 → 5). The corrected recommendation is
**no gate, or the 45 px default**, used for its original purpose only: honestly labelling a
cube that is genuinely too far to read.

The match runner shipped here carries **no size gate at all**. Its filters are the depth
validity window (0.1 m < d < 4.5 m, `mission/match_runner.py:1186`) and the grid-snap error
gate (`MAX_SNAP_ERR_M = 0.30`, `mission/match_runner.py:82`). The `FAR_PRESENCE_M = 2.5`
constant survives in `perception/fieldlib.py:74`, but the far-demotion it controlled was
removed on 2026-07-19 — distant observations now vote normally
(`mission/match_runner.py:1196`).

---

## 4. Face segmentation on cube crops (batched)

**In:** *all* the cube crops of *all* the shots in the scan, as one list of BGR arrays.
**Out:** per-visible-face mask, class and confidence over
`apple / orange / banana / pineapple / plain`.

```python
# mission/match_runner.py:1226
face_res = _chunked_predict(self.models[1], crop_list, FACE_BATCH_CHUNK,
                            imgsz=fl.FACE_IMGSZ, conf=0.1, verbose=False)
```

| Constant | Value | Where | Why |
|---|---:|---|---|
| `FACE_IMGSZ` | 224 | `perception/fieldlib.py:68` | The training size. `imgsz=640` collapses confidence from ~0.9 to 0.1–0.4 (`perception/fieldlib.py:68` comment: "training size fixed — violating it collapses conf"). |
| `FACE_BATCH_CHUNK` | 32 | `mission/match_runner.py:181` | One batched forward per chunk. 5 crops: 10.9 ms batched vs 42.2 ms looped, **3.86×** (measured, RTX 5080). |
| face `conf` | 0.10 | `mission/match_runner.py:1228` | Very low on purpose: the detector's job here is recall. Selection happens in the vote (§5) and the cell vote (§8), which have their own, higher thresholds. |

Input must be **BGR**. `mission/match_runner.py:1142` builds the BGR frame once per shot
with `np.ascontiguousarray(stitched[:, :, ::-1])` and everything downstream — crops, warped
verifier patches — inherits it.

Before the mission starts, both models are preloaded and warmed up on a dummy frame in a
background thread (`mission/match_runner.py:205`–`236`), so the first real inference of the
match does not pay CUDA/TensorRT initialisation inside the 180 s budget.

---

## 5. Face-level vote inside one crop

**In:** all face detections belonging to one cube crop. **Out:** one provisional identity.

```python
# perception/fieldlib.py:465
def face_vote(faces, face_fruit_min=FACE_FRUIT_MIN):
    if not faces:
        return None, None
    fruit = [(n, c) for n, c in faces if n in FRUITS and c >= face_fruit_min]
    return max(fruit, key=lambda t: t[1]) if fruit else max(faces, key=lambda t: t[1])
```

This is deliberately **asymmetric**: any fruit face at `conf ≥ FACE_FRUIT_MIN = 0.30`
beats a plain face of *any* confidence.

The justification is a competition rule, not a heuristic. A fruit cube carries its printed
faces on a vertical ring — front, top, back — with the single blank face fixed to the
**bottom**. Left, right and bottom are blank. So any viewpoint that resolves the cube sees
two fruit faces and one plain face, and a symmetric "most confident face wins" rule would
label fruit cubes plain most of the time. Measured over a scan, fruit faces occupy only
10–24 % of all face observations.

A separate runtime guard (`fruit_min_conf = 0.45`, added 2026-07-08) re-counts weak fruit
detections as *unknown* rather than ignoring them. The threshold is data-derived from a
recorded orange holdout: 0.45 keeps 100 % of true-orange faces (lowest true-orange
confidence 0.463) while rejecting ~88 % of blank-crop false positives (highest such
confidence 0.496).

---

## 6. Binary pair verification — shipped, and disabled

**In:** a face mask polygon. **Out:** either a replacement label, or the loss of that
face's right to cast a strong vote.

The two confusion pairs that actually hurt are apple↔orange and banana↔pineapple. Each gets
a dedicated binary MobileNetV3-Small second opinion, run on CPU through ONNX Runtime.

The patch contract is mandatory and is the whole reason the verifier works
(`perception/fieldlib.py:438`):

```
face mask polygon
  → cv2.minAreaRect → cv2.boxPoints
  → 224×224 warpPerspective   (the face, rectified — 0.239 ms/face)
  → resize 128
  → BGR→RGB → /255 → ImageNet normalise
```

Feeding a whole resized crop instead of the rectified quad produced an absurd 29.00
apple/orange vote ratio against a ground truth of 0.50. The verifier only makes sense on a
rectified face.

Routing and gating (`perception/fieldlib.py:400`, `mission/match_runner.py:1247`; the
verifier weights are resolved by name in `perception/fieldlib.py:405`–`423`, preferring
`pair_banana_pineapple_v2_1` over `v2`):

- apple/orange → AO verifier, banana/pineapple → BP verifier; one forward per route per
  frame (0.95 ms at batch 1 — faster than CUDA at these batch sizes, where dispatch
  overhead dominates).
- `PAIR_CONF = 0.80` (`perception/fieldlib.py:72`): at or above, the label is replaced; below,
  the face keeps its label but loses its strong-vote right.
- Expected end-to-end cost: −1.4 % FPS.

### Two measurements disagree

Two studies disagree. Both are published.

| Study | Setting | Verdict |
|---|---|---|
| Offline, 155 real crops | ABC-cascade path | Verifiers **help**: 61 % → 64 % per-cube accuracy |
| Robot-side, 780-combination sweep on 175 real face patches | Full-frame runtime | Verifiers **off** is best: 85.7 %, vs 74.3 % for AO v2_anchor + asymmetric gate and 71.4 % for AO v2 + asymmetric. The BP gate flipped *correct* bananas to pineapple, errors 3 → 9–11. |

Supporting the sceptical view: a 2026-07-24 sweep over 771 accumulated field crops recorded
the AO gate performing "swap orange → apple" 48 times (old AO) and 49 times (new AO) — the
opposite of the direction it was built to fix, and precisely the failure mode that cost
points in the qualifiers.

`--pair` defaults to `on` (`mission/match_runner.py:2366`) and falls back to face-only if
the weights fail to load. **Every recorded run passed `pair: off`.**

---

## 7. Depth back-projection to a 3D target

**In:** a detection in stitched-frame pixels. **Out:** a point in the robot frame, then the
map frame.

The stitched frame is a virtual image; it has no camera model. So the first thing that
happens is a return to the source camera:

```python
cam, u_src, v_src = stitcher.to_source(u, v)     # perception/fieldlib.py:375
d = robust_depth_median(depth_map[cam], u_src, v_src)   # mission/match_runner.py:250
x, y = pixel_depth_to_robot_xy(u_src, v_src, d, intr[cam], mounts[cam])  # :267
```

`robust_depth_median` takes the median of a small patch around the pixel (radius scaled
with resolution, ~3 px at 640 width) because aligned depth is holey at object edges.

### Why depth, and not the ground plane

Until 2026-07-21 the range came from `pixel_to_ground` — back-projecting the object's
bottom pixel onto the floor plane, which needs no depth at all
(`perception/geometry.py:84`). It assumes the visible bottom edge is the floor contact
line. That assumption holds for a cube and breaks for a polyhedron resting on a vertex or
edge: cross-validation measured a **+27…64 mm bias at z = 4…7 cm**, and on 2026-07-20 it
produced **0 successful grasps out of 4** on icosahedra.

Depth replaced it. Depth is the measured distance to the contact pixel and is
shape-agnostic. Ground back-projection survives only as the fallback for invalid depth
pixels, tagged with a `src` field so the two can be told apart in the logs
(`mission/match_runner.py:1511`–`1520`). `--range-mode` still accepts `ground` for
comparison, and in that mode the depth reading is recorded anyway as `depth_ref_m` for
cross-validation (`mission/match_runner.py:1183`).

Camera mounts are measured, not nominal — four pairs of them, one per mast position, each
fitted to a depth ground-plane and cross-checked with a tape measure
(`perception/fieldlib.py:35`–`54`):

| Mount | forward (m) | height (m) | tilt (°) | Plane-fit RMS |
|---|---:|---:|---:|---:|
| top, mast down | 0.198 | 0.3497 | 18.44 | 2.4 mm |
| near, mast down | 0.1874 | 0.3143 | 53.20 | 0.9 mm |
| top, mast up | 0.198 | 0.4986 | 19.39 | 2.7 mm |
| near, mast up | 0.1874 | 0.4600 | 53.24 | 1.7 mm (79 % inliers) |

Two corrections recorded in those comments are worth repeating because they were both
believed for days:

- The top camera's tilt increases by +0.95° when the mast is raised — the upper bracket
  sags. Adding height alone is wrong.
- The lower camera is **not** chassis-mounted; it rides the mast too. The 2026-07-18
  conclusion that it was fixed came from a measurement taken with the mast still down but
  the file labelled `mastup`.

Still unmeasured, and stated as such in the source: `forward_m` (a plane fit cannot
constrain it) and roll (−1.0° to −2.5°, not modelled by `pixel_to_ground`, worth a few cm
of lateral error at the frame edges).

---

## 8. Grid snap and per-cell K-vote

The arena has 42 candidate positions on a 50 cm pitch (x ∈ 50…350 cm, y ∈ 100…350 cm).
Every observation is converted to map coordinates and snapped to the nearest one
(`perception/fieldlib.py:182`); a snap error above `MAX_SNAP_ERR_M = 0.30` discards the
observation rather than placing it in the wrong cell.

Votes accumulate per cell across the whole 12-shot scan, and are resolved in
`mission/match_runner.py:1343`:

| Constant | Value | Meaning |
|---|---:|---|
| `CELL_VOTES_MIN` | 3 | Total observations before a cell may be confirmed at all |
| `CELL_FRUIT_CONF` | 0.5 | What counts as a *strong* fruit observation |
| `CELL_FRUIT_K` | 2 | How many strong fruit observations override the blank majority |

`K = 2` rather than 1 is a measured choice. On five real arena sessions (361 detections):

| Rule | Fruit cubes correctly identified (of 42 cells) |
|---|---:|
| Plain majority vote | **1** |
| Asymmetric, K = 1, conf ≥ 0.5 | 10 — but one octahedron was flipped to pineapple by a single false positive |
| **Asymmetric, K = 2, conf ≥ 0.5** | **9**, with all polyhedra intact |

One extra rule handles ties. If the top two fruit classes are within one vote of each
other, the cell is re-decided on summed face confidence, and only held as
`conflicting_fruit` if even that is close (`FRUIT_TIE_CONF_MARGIN = 0.10`). This was added
after a real run on 2026-07-21 in which a misread `pineapple 0.773` tied a correct
`apple 0.924`, the cell was held, the A1 fallback then demoted the fruit votes to `plain`,
and a true apple cell was confirmed as plain.

If no fruit identity is confirmed, the cell falls back to an A1-confidence-weighted tally
over shapes and `plain`.

Under time pressure the K values are overridable, and were overridden: the successful
2026-07-21 runs used `votes_k=1 / fruit_k=1` to fit inside the 180 s budget.

Finally, when a target class is specified, the pipeline **re-verifies immediately before
grasping** (`mission/match_runner.py:1594`–`1641`): a fresh crop at the same 0.18 pad, the
same batched face call, the same pair route. Only a *confirmed contradiction* aborts the
pick — a `plain` or empty result is allowed through, because from the side a fruit cube can
legitimately show no fruit face.

---

## 9. Measured cost

Per-stage timings from one recorded 8-shot rehearsal scan producing 18 cube crops
(run `20260721_155127`; **this was an Isaac Sim field-parity run, not the real arena**, and
the report does not record the compute device):

| Stage | Time for 8 frames |
|---|---:|
| Stitch | 0.06 s |
| A1 (batched) | 0.14 s |
| Geometry: depth back-projection, map transform, grid snap | 0.09 s |
| Face stage (batched, 18 crops; pair verifier off in this run) | 0.08 s |

PNG archiving of every shot runs on a background thread so it never blocks the mission
path (`mission/match_runner.py:1288`).

Offline comparisons, all on non-Jetson hardware:

| Measurement | Value | Device |
|---|---:|---|
| Unified 2-stage, 30 s fixed capture | 3.83 FPS (259.9 ms/frame: A1 153.5 + face 100.7) | dev-box CPU |
| Four-stage "ABC" cascade, same capture | 1.29 FPS (713.7 ms/frame) | dev-box CPU |
| Unified with a TensorRT A1 | mean 47.45 / median 52.00 FPS | RTX 5080 |
| ABC cascade, best stable | 20.12 mean / 20.46 median FPS | RTX 5080 |
| Face model, 5 crops batched vs looped | 10.9 ms vs 42.2 ms (3.86×) | RTX 5080 |
| Verifier, ONNX Runtime CPU, batch 1 | 0.95 ms (CUDA: 2.9 ms) | dev-box CPU |
| Verifier patch warp | 0.239 ms/face | dev-box CPU |

**No Jetson Orin Nano benchmark exists.** Three report files named `jetson_runtime_*` were
in fact produced on a Windows RTX 5080 box. Do not read 47 FPS as on-robot performance.

---

## Why two stages and not one

The first model, in May 2026, was a single 8-class YOLO11n detector. It localised objects
well and could not say which fruit face was showing, because a fruit cube seen from the
side *is* a plain cube. It was abandoned.

The replacement was a four-stage cascade: A1 object segmentation, A2 visible-face
segmentation on a 0.18-padded 224 crop, B a hand-written 5-conv network recovering the
amodal 4-corner face quad from a possibly-occluded mask, and C a MobileNetV3-Small
classifying the rectified patch. It was optimised across 43 logged experiments
(E000–E042), from 4.98 FPS sequential to 8.71 batched to 20.12 mean FPS, while holding a
98.26 % face-label behaviour gate; the variants that reached 22 FPS all fell below that
gate and were rejected. Most of the gain came from frame-level batching, not model swaps
(B: 3.92 → 1.24 ms, C: 6.10 → 3.41 ms).

The written conclusion of that optimisation round was structural: **collapse A2 + B + C
into one cube-crop segmentation model.** That is the pipeline documented above. On one
identical 30 s capture it was 2.97× faster than the cascade it replaced, and on a 155-crop
real benchmark the resurrected cascade scored 55 % (naive) → 61 % (domain-aware K-vote) →
64 % (with verifiers) against 83 % for the single unified system.
