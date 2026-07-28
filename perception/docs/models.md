# The Model Standard — Rules for Using the Weights

Four trained models ran on the robot. What each one *is* — architecture, training set,
measured accuracy, hashes — is in
[`../models/MODEL_CARD.md`](../models/MODEL_CARD.md). Where each one sits in the runtime is in
[`pipeline.md`](pipeline.md). How the training data was made is in
[`synthetic-data.md`](synthetic-data.md).

This document is the third thing: **the rules for calling them.** Every rule below is a
contract that fails *silently* — no exception, no warning, just worse numbers. Each cost this
team at least a day. They are written here as rules with their evidence, so that anyone
reusing these weights does not have to rediscover them.

**Contents**

1. [The five rules, in one table](#1-the-five-rules-in-one-table)
2. [Which weights are canonical](#2-which-weights-are-canonical)
3. [Rule 1 — input size](#3-rule-1--input-size-face-224-a1-896)
4. [Rule 2 — BGR, not RGB](#4-rule-2--bgr-not-rgb)
5. [Rule 3 — one batched `predict()` per frame](#5-rule-3--one-batched-predict-per-frame)
6. [Rule 4 — the crop contract](#6-rule-4--the-crop-contract-018-pad--plain-square-resize)
7. [Rule 5 — class order is an interface](#7-rule-5--class-order-is-an-interface)
8. [Deprecated weights](#8-deprecated-weights-do-not-use)
9. [TensorRT on Jetson](#9-tensorrt-on-jetson-engine-preferred-pt-fallback)
10. [A 20-line preflight](#10-a-20-line-preflight)

---

## 1. The five rules, in one table

| # | Rule | Applies to | Symptom if broken |
|---|---|---|---|
| 1 | `imgsz=224` for the face model; `imgsz=896` for A1 on the stitched frame | both YOLO models | Face confidence collapses from ~0.9 to 0.1–0.4; A1 loses small-object recall |
| 2 | Input must be a **BGR** numpy array | both YOLO models | `orange` ↔ `plain` and `apple` ↔ `pineapple` flip |
| 3 | One batched `predict()` per frame, never a loop | face model mainly | 3.86× slower (10.9 ms → 42.2 ms for 5 crops) |
| 4 | Expand the A1 box by `0.18 × max(w, h)`, then **plain square resize** to 224 | face model | Train–serve skew: the model sees pixel geometry it was never trained on |
| 5 | Class index order is fixed | face model, A1, engines | Silent, total mislabelling |

Rules 1 and 2 were found together on 2026-07-18 while chasing an accuracy regression that had
been blamed on hardware for days. The tell was a teammate reporting ~95 % accuracy on the same
weights and the same images: their script simply never passed `imgsz`, so Ultralytics used the
checkpoint's training value and got it right by omission.

---

## 2. Which weights are canonical

The weights are not in git (Ultralytics derivatives, AGPL-3.0 — see [`../../NOTICE`](../../NOTICE)).
`scripts/fetch_models.sh` downloads them from the GitHub Release into `perception/models/` and
verifies SHA256 against the published `SHA256SUMS`. Verify the hashes; a partially-downloaded
weight loads without complaint and scores badly.

Runtime resolves exactly two paths, and nothing else is a supported source:

```python
# perception/fieldlib.py:65
A1_WEIGHTS   = REPO_ROOT / "perception" / "models" / "a1_objectseg"  / "best.pt"
FACE_WEIGHTS = REPO_ROOT / "perception" / "models" / "unified_face"  / "best.pt"
```

Every consumer goes through those constants — the match program
(`mission/match_runner.py:212`–`213`), the field autopilot
(`mission/field_autopilot.py:186`–`187`), the offline A/B scorer
(`perception/eval/face_model_compare.py:92`, `:178`) and the simulation parity analyser
(`simulation/scripts/field_parity_analyze.py:398`). If you swap a weight, swap the file; do not
add a second load path.

The pair verifiers resolve by *name*, not path, and accept two directory layouts — the flat
`verifiers/<name>.onnx` used in this repository and the original
`<name>/weights/best.onnx` training-tree layout (`perception/fieldlib.py:409`–`427`). They prefer
`pair_banana_pineapple_v2_1` over `v2`; the file shipped here as
`pair_banana_pineapple_v2.onnx` is in fact the v2_1 weight (see
[`../README.md`](../README.md) for that naming caveat).

Two face weights were carried at this same path during development: `preferred_v2`
(YOLO26n-seg, 6,499,101 B, `6ec2f572…`) and `topfruit_face_s_v3ft` (YOLO26s-seg,
23,291,542 B, `3d2d4dba…`), and the s-model beats the n-model on every column of the
70-crop real holdout. `3d2d4dba…` is the weight this repository ships and the one every rule below
was measured against; the face model's real arena photograph fine-tune (2026-07-24) is
published as `perception/training/train_face_ft.py`. Whether the pair verifiers earn their
place alongside it is a separate question with measurements on both sides —
[`pipeline.md` §6](pipeline.md) and
[`../../docs/07-results-and-lessons.md`](../../docs/07-results-and-lessons.md).

Both YOLO models are loaded and warmed up on a dummy frame in a background thread *before* the
match clock starts (`mission/match_runner.py:205`). CUDA/TensorRT initialisation is several
seconds; paying it inside a 180 s match is a wasted object. Warm-up failure is logged but not
fatal.

---

## 3. Rule 1 — input size: face 224, A1 896

```python
# perception/fieldlib.py:67
A1_IMGSZ_STITCHED = 896   # stitched frame is ~890 px tall (640 would downscale it 0.72×)
FACE_IMGSZ        = 224   # training size, fixed — violating it collapses confidence
```

### The face model: 224, or omit `imgsz` entirely

The face model was trained at 224 and only at 224. Passing `imgsz=640` at inference **collapses
confidence from ~0.9 to 0.1–0.4** and produces a stream of fruit-face misclassifications. This
is not a soft degradation you can compensate for with thresholds — the confidences that the
whole voting stack is calibrated against (`FACE_FRUIT_MIN = 0.30`, `CELL_FRUIT_CONF = 0.5`)
land in the wrong regime.

The safest call is to **pass no `imgsz` at all**: Ultralytics then reads the value from the
checkpoint's `train_args` and is correct by construction. The code here passes `fl.FACE_IMGSZ`
explicitly instead, because an explicit constant with a comment is easier to defend in review
than an omission.

### A1: 896, explicitly, on the stitched frame

A1 is the opposite case. It was **trained at 640**, so omitting `imgsz` gives you 640 — and
that is wrong at runtime, because A1 does not see a 640-px camera frame. It sees the *stitched*
frame, which at the 640×480 camera mode was 632×890 px. Running the tall stitch at 640
downscales it by 0.72×, and the objects that matter — an 8 cm cube three metres away — are
exactly the ones that fall below the detector's size floor when you do that.

So A1's runtime size is a property of the **stitch**, not of the weight. If you change the
stitch calibration or the camera resolution, re-derive it.

`A1_IMGSZ_STITCHED = 896` is derived for the 632×890 stitch. The stitch calibration shipped
here is FHD-native (`out_h` 2044, `perception/calibration/stitch/up.json`), so if you run at
that resolution, re-derive the input size from the stitch height the same way.

### The letterbox that 896 actually produces

`imgsz=896` on a 632×890 image does **not** give you an 896×896 tensor. Ultralytics letterboxes
to the long side and rounds the short side up to a multiple of 32, i.e. **896×640 (H×W)**. That
detail is invisible for `.pt` inference and load-bearing for TensorRT (§9), where the shape is
frozen into the engine. The export tool in the original tree measured it rather than assuming
it, by calling `model.predictor.pre_transform([dummy])[0].shape[:2]` after one throwaway
predict and aborting if the result differed from the export shape.

---

## 4. Rule 2 — BGR, not RGB

Ultralytics' numpy input path follows the `cv2.imread` convention: **channel order is BGR**. It
does not inspect, and cannot detect, what you actually handed it.

Feeding an RGB array — which is what you get from PIL, from `sensor_msgs/Image` with
`encoding: rgb8`, or from any pipeline that already converted — flips exactly two pairs, both of
them colour-dominated decisions:

```
orange ↔ plain
apple  ↔ pineapple
```

That was measured on real crops, not reasoned about. It is also the single most convincing
symptom to check first when a classification regression appears without a weight change.

The runtime converts once per shot, at the top of the frame path, and everything downstream —
crops, verifier patches — inherits the conversion:

```python
# mission/match_runner.py:1143
bgr_l.append(np.ascontiguousarray(stitched[:, :, ::-1]))
```

> **A live trap in this repository.** The optional in-process detector in the navigation node,
> `navigation/ros2/arena_lightweight_control/arena_lightweight_control/internal_perception.py:112`–`117`,
> converts `bgr8` messages **to RGB** and passes `rgb8` through untouched, then predicts at a
> default `imgsz` of 416 (`arena_control_node.py:63`). It breaks both Rule 1 and Rule 2. It is a
> coarse obstacle-presence detector, disabled by default
> (`enable_internal_image_processing = False`, `detector_model_path = ""`), and it is **not** a
> place to plug the face weight in. If you enable it, fix the channel order first.

---

## 5. Rule 3 — one batched `predict()` per frame

This is a performance contract rather than a correctness one, but it is the difference between
a scan that fits in the match budget and one that does not.

| Workload | Batched | Looped | Ratio | Device |
|---|---:|---:|---:|---|
| Face model, 5 cube crops | 10.9 ms | 42.2 ms | **3.86×** | RTX 5080 |

The reason the gain is that large at this scale is that each crop is 224×224 — the per-call
Python, preprocessing and CUDA-dispatch overhead dominates the actual convolution work. The
same argument is why the pair verifiers run on the ONNX Runtime **CPU** provider: at batch 1
they take 0.95 ms on CPU versus 2.9 ms on CUDA.

The runtime therefore accumulates crops across *all twelve shots of a scan* and submits them
together, in chunks:

```python
# mission/match_runner.py:180
A1_BATCH_CHUNK   = 8    # all eight scan shots as one A1 batch
FACE_BATCH_CHUNK = 32   # all cube crops of the scan, 32 at a time
```

`_chunked_predict` (`mission/match_runner.py:184`) is the only inference entry point; A1 is
called at `mission/match_runner.py:1149` and the face model at `:1227`. Crop preparation itself
is free by comparison — 1.3 ms mean per frame for 5 crops, measured over 294 frames on a CPU
box.

One caveat that only appears on Jetson: a TensorRT engine exported with a *static* batch raises
an `AssertionError` mentioning "input size" and "max model size" when handed a batch. The
lightweight map node catches exactly that string pair, warns once, and falls back to one crop
per inference. Prefer fixing the engine (§9) over relying on the fallback — it gives up the
3.86×.

---

## 6. Rule 4 — the crop contract: 0.18 pad → plain square resize

The face model does not see cubes. It sees crops, and the crop is part of the model's input
specification just as much as `imgsz` is. The **export** pipeline that produced the training
crops did this:

```
A1 box
  → pad = max(box_w, box_h) × 0.18, applied to all four sides, clipped to the frame
  → crop from the original image
  → cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)      # PLAIN SQUARE
```

Two properties matter. The pad is computed from `max(w, h)`, so it is **isotropic** — a tall
narrow box and a wide flat box of the same longest side get the same absolute margin. And the
final resize is a **plain square resize**, which *distorts aspect ratio*: a 90×140 crop is
squashed horizontally and stretched vertically into 224×224. That distortion is not a bug in
the export; it is simply what the model was trained on, so it is what the model expects.

Runtime reproduces the pad exactly:

```python
# mission/match_runner.py:1213
pad = int(max(x2 - x1, y2 - y1) * 0.18)
crop = bgr[max(0, y1 - pad):min(h, y2 + pad),
           max(0, x1 - pad):min(w, x2 + pad)]
```

and `mission/field_autopilot.py:831` exposes the same `--crop-pad 0.18` for recording.

### The train–serve skew bug

Before 2026-07-03, the runtime handed that **raw rectangular crop** straight to
`predict(imgsz=224)`. Ultralytics then applied its own preprocessing: **letterbox** — scale the
long side to 224, pad the short side with grey bars, aspect ratio preserved.

Same weight, same nominal `imgsz`, completely different pixels. Every object in the letterboxed
crop is smaller than the model expects, sits at a different position, and is surrounded by grey
that never appeared in training. The model had never seen a letterboxed crop in its life. The
audit that found it is blunt about the cause: the training export policy and the runtime policy
had simply never been diffed against each other.

The fix is one line — resize the crop to `imgsz × imgsz` with `INTER_AREA` before predicting —
and the correct order of operations is:

```
A1 box → pad 0.18 → crop → resize 224×224 INTER_AREA → predict → scale masks/boxes back
```

> **Two crop paths live in this repository.** The match runner does **not** do the
> square resize; `mission/match_runner.py:1213`–`1228` passes the padded rectangular crop to
> `predict(imgsz=224)` and lets Ultralytics letterbox it. The fix lives in the offline reference
> runtime, not in the robot code. Both paths produced published numbers in this project, so
> check which one produced a figure before comparing against it. If you are building on this:
> match the export, resize the square.

---

## 7. Rule 5 — class order is an interface

```
A1    0 cube_like_object   1 octahedron   2 dodecahedron   3 icosahedron
face  0 apple   1 orange   2 banana   3 pineapple   4 plain
```

Downstream code indexes and name-matches against these. `FRUITS` and `POLYHEDRA` in
`perception/fieldlib.py:29`–`31` are name sets, the grid vote counts by name, and the GT parser
accepts exactly these strings. A reordering — which is easy to introduce by retraining from a
differently-sorted dataset directory, or by an export that rebuilds the metadata — produces no
error anywhere. It produces a robot that confidently collects the wrong object.

The TensorRT export tool treats this as a hard gate rather than a check: it compares
`engine.names` against the expected dictionary and exits with a dedicated status code on
mismatch, before any accuracy comparison runs. Its comment is worth translating verbatim:
*"the class names in the engine metadata must be exactly identical to the `.pt`. If even one is
shifted, the entire pipeline is silently mislabelled."*

---

## 8. Deprecated weights (do not use)

| Weight | Status |
|---|---|
| `best.pt` at the old repository root | **Deprecated.** Single 8-class segmentation model |
| `data/yolo/weights/best.pt` | **Deprecated.** Same lineage |
| `data/yolo/weights/seg1000.pt` | **Deprecated.** Same lineage |

These are the pre-2026-07-17 single-stage models: one YOLO detector with all eight object
classes in one head. They are not distributed with this repository and are **not** a
lighter-weight alternative to the two-stage pipeline. They were abandoned for a structural
reason, not a performance one: a fruit cube seen from the side *is* a plain cube, so a
single-stage detector that must commit to `apple` or `plain` from a full-frame view cannot be
right — the information is not in the pixels it is given. The two-stage split exists precisely
so that the identity decision is made on a crop that is large enough to carry it, and can be
deferred across a multi-shot scan when it is not. The longer history, including the four-stage
cascade that came between them, is in [`pipeline.md` §"Why two stages and not one"](pipeline.md).

**Three files in this repository still reference `seg1000.pt`:**

```
hardware/ros2/robot_bringup/launch/real_bringup.launch.py:12
hardware/ros2/robot_bringup/launch/match.launch.py:11
hardware/ros2/robot_bringup/config/real.yaml:161      (semantic_mapper_node)
```

They are left in place because they document the legacy Nav2-era stack honestly, and because
`real.yaml` also carries live configuration for nodes that *are* used. None of them is on the
competition path — the bridge that ran in competition is
`real_competition_bridge.launch.py`, started by `scripts/run_real_competition_bridge.sh`, and
perception is driven by `mission/match_runner.py`, which reads only the two `fieldlib`
constants. If you launch `real_bringup.launch.py` expecting the competition perception stack,
you will get a node pointed at a weight that this repository does not ship.

---

## 9. TensorRT on Jetson: `.engine` preferred, `.pt` fallback

On the Jetson Orin Nano, a TensorRT FP16 engine built from the same weights is loaded in
preference to the `.pt`, with the `.pt` as the fallback when no engine is present. The path
convention is identical — same directory, same stem, extension swapped:

```
perception/models/a1_objectseg/best.pt      →   perception/models/a1_objectseg/best.engine
```

The A1 engine built for the 632×890 stitch was `a1_yolo26s_seg_896_fp16.engine`, 24,888,452 B.

### Engines are not distributed, and cannot be

A TensorRT engine is a *serialised, compiled plan*, not a portable model file. It is bound to:

- **the GPU architecture** it was built on — an Orin engine will not load on a desktop RTX card,
  or vice versa;
- **the TensorRT and CUDA versions** present at build time — a JetPack upgrade invalidates it;
- **the exported input shape**, which for this pipeline is frozen at 896×640 (§3), so a change
  to the stitch calibration or camera resolution invalidates it even on the same machine;
- **the batch profile** it was optimised with.

There is a fifth, softer version skew worth recording: the models were trained under
Ultralytics 8.4.54 while the Jetson had 8.4.21 installed. No divergence was ever reproduced,
but when comparing numbers between machines, put the library version on the suspect list.

So the release ships `.pt` only, and **you build your own engine on the machine that will run
it.** The export cannot be a plain `yolo export`; four things have to be right.

| Requirement | Value | Why |
|---|---|---|
| Input shape | `imgsz=[896, 640]` (H, W) | The runtime letterboxes the 632×890 stitch to 896×640. Exporting a square 896 makes the engine pad to 896×896 at inference, which changes every object's relative scale and therefore changes the detections. |
| Batch | `dynamic=True`, `batch=8` | The scan submits its twelve shots in chunks of eight; the approach path submits one image. A static batch-1 engine breaks batched inference (§5) — which is exactly why the finals fell back to the `.pt`: the engine on the Jetson was a batch-1 build and did not pass the runtime's name-convention check. |
| Precision | `half=True` (FP16), FP32 as the fallback | FP16 is the speed win, but its effect on confidence and masks has to be *measured*, not assumed. |
| Workspace | 2 GiB default | Fits Orin's free memory during a build. Drop to 1 GiB with `batch=4` if the build runs out. |

Concretely, the export call that was used:

```python
YOLO("best.pt").export(format="engine", imgsz=[896, 640], dynamic=True,
                       batch=8, half=True, device=0, workspace=2.0,
                       simplify=True, verbose=False)
```

### The parity gate

Building the engine is the easy half. The tool that produced these engines refused to accept
one until it had been compared against the `.pt` **on real saved stitch frames** — synthetic
images being useless for a quantisation parity check — with these criteria:

| Check | Tolerance | Rationale |
|---|---|---|
| Class names | exact dict equality | §7 |
| Detection count per frame | exact | A dropped detection is a missed object |
| Class per detection | exact | — |
| Confidence per detection | ±0.06 | FP16 quantisation headroom |
| Bottom-contact pixel | ≤ 4.0 px | This pixel back-projects to the arena position |
| **Resulting grid cell** | **exact match** | The cell *is* the pickup target; one differing cell fails the build |

The comparison also re-verified the letterbox contract by measurement before exporting, and
backed up any existing engine with a timestamped suffix before overwriting. The tool itself
(`scripts/dev/export_a1_tensorrt.py` in the original tree) is not part of this repository
because it reads from the private field-log layout, but the contract above is the whole of it.

---

## 10. A 20-line preflight

If you are wiring these weights into something new, this is the shortest check that all five
rules hold. It needs one real stitched frame and one cube crop.

```python
import cv2, numpy as np
from ultralytics import YOLO
import perception.fieldlib as fl

a1   = YOLO(str(fl.A1_WEIGHTS))
face = YOLO(str(fl.FACE_WEIGHTS))

# Rule 5 — class order
assert list(a1.names.values())[:1] == ["cube_like_object"]
assert list(face.names.values()) == ["apple", "orange", "banana", "pineapple", "plain"]

frame_bgr = cv2.imread("shot0_stitched.png")          # Rule 2 — cv2 gives BGR
r = a1.predict(frame_bgr, imgsz=fl.A1_IMGSZ_STITCHED, conf=0.25, verbose=False)[0]

x1, y1, x2, y2 = [int(v) for v in r.boxes[0].xyxy[0]]  # a cube_like_object box
pad  = int(max(x2 - x1, y2 - y1) * 0.18)               # Rule 4 — isotropic pad
h, w = frame_bgr.shape[:2]
crop = frame_bgr[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]
crop = cv2.resize(crop, (fl.FACE_IMGSZ, fl.FACE_IMGSZ), interpolation=cv2.INTER_AREA)

f = face.predict([crop], imgsz=fl.FACE_IMGSZ, conf=0.1, verbose=False)[0]  # Rules 1, 3
print([(f.names[int(b.cls)], round(float(b.conf), 3)) for b in (f.boxes or [])])
```

A healthy result on a clearly-visible fruit face is a top confidence around **0.8–0.98**. If
everything comes back in the **0.1–0.4** band, you have broken Rule 1. If the *labels* look
systematically swapped within `orange`/`plain` or `apple`/`pineapple`, you have broken Rule 2.
Symptom-first versions of these are in
[`../../docs/06-troubleshooting.md` §5](../../docs/06-troubleshooting.md).
