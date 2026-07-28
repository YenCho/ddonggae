# Model card

Four trained models make up the perception stack. None of them are stored in this
git repository — run `scripts/fetch_models.sh` to download them from the GitHub
Release.

> **License** — these weights are derivatives of Ultralytics YOLO and are
> therefore **AGPL-3.0**, not MIT. The MIT license in this repository covers its
> own source and documentation. See `NOTICE`.

## The runtime set

| Stage | Runtime status |
|---|---|
| **A1** object segmenter | every scan shot |
| **Face** identity | every cube crop; fine-tuned on **real arena photographs**, 2026-07-24 |
| **BP** banana/pineapple verifier | pair stage, `v2_1` — the bunch retrain (§3–4) |
| **AO** apple/orange verifier | **switched off for the finals** |

The pair stage is a single toggle in the code (`--pair`, default `on`,
`mission/match_runner.py:2366`): it loads both heads and routes each fruit face
to the matching one. For the finals the apple/orange head was not used, and the
reason is worth recording — the AO verifier existed specifically to
counteract the face model's apple over-calling, and an apple mis-read is what
cost the qualifying round. It was dropped in favour of fine-tuning the face model
on real photographs instead. The two choices were never measured against each
other.

## Shared runtime contracts

These two rules are not tuning knobs. Violating either silently destroys
accuracy, and both were discovered the hard way in the field.

1. **The face model must run at `imgsz=224`.** Passing 640 collapses confidence
   from roughly 0.9 to 0.1–0.4. 224 is the size the crops were exported at.
2. **The face model must receive BGR numpy input.** Feeding RGB flips
   orange ↔ plain and apple ↔ pineapple.

A third rule is a performance contract rather than a correctness one: put every
crop of a frame into **one batched `predict()` call**. Five crops batched take
10.9 ms; the same five looped take 42.2 ms — a 3.86× difference.

## 1. A1 — object segmenter

| | |
|---|---|
| Architecture | YOLO26s-seg |
| Task | Full-frame instance segmentation |
| Classes | `cube_like_object`, `octahedron`, `dodecahedron`, `icosahedron` |
| Input | Stitched RGB frame, `imgsz=896` at runtime (trained at 640) |
| File | `a1_objectseg/best.pt` (23,371,293 B) |
| sha256 | `08ffa8c4932eb0186dd3fa721dca3229edabcc79572c850eed2731a3f87c2289` |
| Training data | Fully synthetic (Blender). See `perception/docs/synthetic-data.md`. |

`imgsz=896` matches the native height of the stitched frame; running it at 640
downscales by 0.72× and loses small-object recall.

**Measured accuracy.** The training run reported box mAP50 0.9941 / mask
mAP50-95 0.9537 — but that run used **val = train**, so those numbers measure
nothing. On a purpose-built held-out arena set (60 scenes, 573 objects) the same
model scores **box mAP50 0.831 / mask mAP50-95 0.558**. That is the honest
number. A follow-up low-learning-rate fine-tune improved validation by +0.0002
and lost 1.5–2.5 pp on every held-out metric, so it was rejected.

Non-cube polyhedra terminate here and never enter the face stage.

## 2. Unified face — face identity segmenter

| | |
|---|---|
| Architecture | YOLO26s-seg |
| Task | Per-visible-face segmentation and identity on a cube crop |
| Classes | `apple` (0), `orange` (1), `banana` (2), `pineapple` (3), `plain` (4) |
| Input | 224×224 **BGR** crop, `imgsz=224` |
| File (finals) | `unified_face_ft_20260724/best.pt` (23,306,518 B) |
| sha256 | `e83a031f6bb61db39ae7a2b94cc579de8f32bbb1b5c4e0a8beca60166b55ddef` |
| File (qualifiers) | `unified_face/best.pt` (23,291,542 B) |
| sha256 | `3d2d4dba14c6bf709b49e5654f537cbad05db931a6b3a454eb0c71e534734927` |
| Training data | Synthetic renders. The finals weight adds an overnight fine-tune (2026-07-24 00:49, 57 epochs) on photographs of the real arena objects — scan crops, driving frames and phone photos — relabelled by an operator: about 74 corrections and 60 drops. |

**Which one ran.** The qualifiers ran `unified_face`; both finals ran
`unified_face_ft_20260724`, which is the runtime default (`fieldlib._FACE_DEFAULT`,
overridable with `FACE_WEIGHTS=`). The fine-tune lifted fruit-face accuracy from
83.3 % to 89.2 % on the two-run replay set (92.5 % on the strong faces that
actually carry a vote), mostly by cutting `apple → orange` misreads from 10 to 2.

Crops are produced by expanding the A1 box by `0.18 × max(w, h)` and resizing to
a **plain square 224×224** — deliberately *not* letterboxed, because that is
exactly how the training crops were exported. The letterbox/square mismatch was
a real train–serve skew bug.

**Known failure mode.** Colour-domain shift on apple. The training pool covered
red apples; the qualifying round used a yellowish under-ripe apple and the model
mis-predicted it. This is the single most instructive failure in the project —
see `docs/07-results-and-lessons.md`.

## 3–4. Pair verifiers — binary second opinions

All verifiers are MobileNetV3-Small, 128×128 input, ONNX opset 17 with dynamic
batch, run on the ONNX Runtime **CPU** execution provider — CPU is deliberate,
because for a single small batch the CUDA dispatch overhead exceeds the compute.

Each takes a perspective-warped patch of one face and gives a binary opinion on
a confusable pair.

| Asset | Role | sha256 |
|---|---|---|
| `verifiers/pair_apple_orange_v2_anchor.onnx` | AO, anchor fine-tuned — the stronger of the two | `b5677af6ce34b970b01c99e69af0dc0bb70e1a3f4980741351446c3d461d5e4a` |
| `verifiers/pair_apple_orange_v2.onnx` | AO, render-only (superseded) | `8dbca32c517ca72bdfb9734c1ab833928b4e9e5f5b8ed2b1ff3436dd907ab1eb` |
| `verifiers/pair_banana_pineapple_real_v1.onnx` | **BP that played the finals** — retrained on real photographs (2026-07-23) | `178bb65796f423d0f890cb7c09868efd8c90b77ee058ab2f9f9422f83b8f5bc1` |
| `verifiers/pair_apple_orange_real_20260724.onnx` | AO, retrained on real photographs; **loaded but its route is off** under `--pair bp` | `92508a7289d3459462961480fa73b97e14974291c9d9feb2ff9d622b03c2c218` |
| `verifiers/pair_banana_pineapple_v2.onnx` | BP `v2_1`, render-trained — **do not use**: it flipped a correct `banana` to `pineapple` at confidence 1.00, nine times across two runs | `630114efaaeb3f02338cf7d6262f738503dbb24a59032c2487dadbc95a2de23a` |

> The file the robot repo ships as `pair_banana_pineapple_v2` is **byte-identical**
> to the model the perception side called `v2_1`. Only the name differs.

### Input contract (both heads)

Not a tuning knob — violating it collapses accuracy:

```
face mask polygon → cv2.minAreaRect → boxPoints
                  → warpPerspective to 224×224
                  → resize to 128×128
                  → BGR→RGB → /255 → ImageNet normalise
```

**Do not simply resize the crop.** The perspective warp is required, because the
patches were trained that way.

### The asymmetric apple/orange gate

The face model over-calls apple, so the gate is deliberately lopsided — cheap to
flip apple → orange, expensive to keep apple:

```python
FLIP_THR       = 0.55   # apple -> orange: low bar
KEEP_APPLE_THR = 0.88   # keeping apple: high bar
```

Anything that clears neither threshold is *unresolved* and loses its strong vote.

**Measured (real arena, held-out split not used in training).** Against the
render-only AO: orange faces **9/15 → 12/15** with apple held at 7/7. The
apple/orange vote ratio (ground truth 0.50) moved 1.13/1.36 → 0.94/1.06, and
0.88/0.94 once the asymmetric gate was applied.

### The banana-bunch fix

Banana *bunches* were read as pineapple: the seam shadows between bananas look
like pineapple striping, and 92 % of the texture pool was single bananas, so
bunches were simply out of distribution. Retraining with synthesised bunch
patches plus real bunch cutouts took the flip rate on a 1,051-patch bunch
holdout from **25.02 % → 0.00 %**, pineapple regression 0.50 % → 0.40 % (none),
and a real bunch cube went from pineapple 0.758 to banana 0.996+, 4/4 correct.

### Two honest caveats

**Capacity was never the bottleneck.** An EfficientNet-B0 ablation tied with
MobileNetV3-Small + anchor fine-tuning, was *worse* without it, and ran 2.5×
slower. The limiting factor was data, not model size.

**The verifier stage may not have been a net win.** Across a large configuration
sweep it measured *worse* overall than the face model alone. Treat it as an
experiment. See `docs/07-results-and-lessons.md`.

> **Which apple/orange weight, and whether it runs.** The perception side's final
> runtime guide (2026-07-20) designated `pair_apple_orange_v2_anchor` as the
> better of the two AO weights, while `perception/fieldlib.py:428` loads
> `pair_apple_orange_v2`. Neither ran in the finals — the AO stage was not used,
> for the reason given at the top of this card. Both files are published because
> the measurements above are real and the asymmetric-gate idea is reusable. The
> BP head did run.
