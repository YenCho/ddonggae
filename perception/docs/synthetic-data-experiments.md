# Synthetic perception experiments and measured failures

This is the historical evidence log for why the renderer and training recipe changed. It is kept separate from the runnable clean-checkout workflow in [`synthetic-data-reproduction.md`](synthetic-data-reproduction.md).

## 7. Honest result 1 — the hue war

**Symptom.** Real printed **orange** faces were confidently read as **apple**. In the
2026-07-18 round, five newly trained face models scored **0.858–0.9024 mask mAP50-95** on
synthetic validation and **all five lost** to the incumbent on the robot's 22-cell arena
benchmark (incumbent 21/22 with one wrong confirmation; challengers 16–18/22 with 4–6).
Orange was the failure axis: incumbent 3/4, challengers 0–2/4.

Three independent causes were found. Each was fixed **in the data**, not in a threshold.

### 7.1 The renderer bug: clamp before gain

Apple textures are hue-jittered and then clamped back into a red band so an apple never
drifts into the orange band. The clamp ran **before** the per-channel BGR gains — and the
gains then shifted the hue right back out. Measured over 200 strong-jitter trials:

| | Apple hue leaking into the orange band |
|---|---:|
| Clamp before colour operations | **50 %** |
| Clamp moved to the final step | **0.0 %** |

Fixed in commit `86defe7`, along with an apple texture gate (`red ≥ 0.08`) and an increase
of the orange class weight from 1.35 to 1.60.

A direct consequence for training: once the renderer clamps hue, **hue augmentation during
training must be disabled** (`hsv_h = 0.0`), or it re-introduces exactly the leakage the
clamp removes.

### 7.2 Realism asymmetry: an attractor made of photographs

Only **apple** had photographic cutout textures. The other classes were dominated by
studio-lab imagery. The model learned an unintended shortcut — *photo-realistic ⇒ apple* —
which fires on real camera input by construction.

The fix was to rebuild the texture pool so realism was not class-correlated. The v3 pack:

| Class | Cutout-v2 textures |
|---|---:|
| **orange** | **4,643** (the largest of any class) |
| apple | 2,131 |
| pineapple | 1,866 |
| banana | 1,440 (legacy lab set) |

### 7.3 The one that hid the other two: a shared random seed

Rendering was distributed across four machines. Every machine used **the same default seed
with the same image ids** — so the machines rendered **the same scenes**. Of 53,700 round-1
scenes, only about **22,800 were unique**, and the duplicates landed on *both sides* of the
train/validation split.

That is why five models could score 0.858–0.9024 on synthetic validation and lose on the
robot. The validation set was partly the training set. The hue leak and the realism
asymmetry were both invisible behind it.

Fixed by assigning a distinct seed per machine:

| Machine | GPU | Scenes | Seed |
|---|---|---:|---:|
| local | RTX 5080 | 3,700 | 31,000,000 |
| school cluster | A6000 (PBS) | 8,100 | 32,000,000 |
| spare 1 | 2080 Ti | 3,800 | 33,000,000 |
| spare 2 | 5070 Ti | 14,400 | 20,260,513 |

30,000 v3 scenes, four machines, four seeds. The crop exporter also gained a
`--crop_prefix` flag: without it, identical metadata stems from different render sets
silently collide and crops are dropped (~3 k lost in round 1).

**The lesson we would tell anyone doing distributed synthetic generation:** count your
unique scenes. Do not count your files.

### 7.4 Did it work?

The first model trained on the corrected v3 render, `topfruit_face_s_v3ft`, was the first
new model to beat the incumbent on the real 70-crop holdout:

| Model | Fruit correct | Wrong-confirm | Missed | Plain correct | False fruit | orange | pineapple | apple | banana |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| incumbent (`preferred_v2`) | 35 | 13 | 2 | 15 | 5 | 8/15 | 13/21 | 7/7 | 7/7 |
| **`s_v3ft` (shipped here)** | **43** | **6** | 1 | 17 | **3** | 10/15 | **19/21** | 7/7 | 7/7 |

Five of the six residual errors are still orange → apple, but at confidence 0.50–0.86
instead of the incumbent's confident 0.93–0.98 — i.e. they moved into the range a verifier
can catch.

### 7.5 The same lesson twice: banana bunches

A real cube printed with a *bunch* of bananas was read as **pineapple at 0.758**. Cause:
92 % of the v3 banana texture pool was single bananas, so the dark shadows between adjacent
bananas in a bunch were out of distribution. Fixed in data — 5,000 synthetic bunch
composites (2–4 filtered single-banana cutouts overlapped 30–52 % with explicit blurred
seam shadows) plus 5,000 real cutout patches, giving 15,029 banana patches:

| | Before | After |
|---|---:|---:|
| Bunch-holdout flip rate (1,051 unseen patches) | 25.02 % | **0.00 %** |
| Pineapple regression | 0.50 % | 0.40 % (none) |
| Real bunch cube | pineapple 0.758 | banana ≥ 0.996, 4/4 correct |

A related capacity ablation is worth recording as a negative: swapping MobileNetV3-Small
for EfficientNet-B0 gave 6/15 real oranges render-only (worse), 12/15 with anchors (an
exact tie) and ran 2.5× slower. **The bottleneck was data, not capacity.**

### 7.6 And then it happened again, at the competition

In the qualifiers the apple printed on the arena objects was a **yellowish, under-ripe
apple**, not the red one that had been announced beforehand. The face model read it as
orange at 0.95. Same failure family as §7.1 — a colour domain gap between what was
rendered and what was printed — arriving from the opposite direction, and this time with
no rendering time left to fix it.

The response was the largest use of real photographs in this project (an earlier anchor fine-tune on 07-19 had already used 64 real crops), and the only one that reached a competition weight for the face model — in this project's training:
crops harvested from one end-to-end run on 2026-07-23 (the 15:45 match on the final objects), its own driving recorder frames and phone photos, mixed with the synthetic set and
fine-tuned overnight. The recipe is published as `perception/training/train_face_ft.py`,
and its comments record the reasoning:

```python
# Start from the model CURRENTLY DEPLOYED on the robot. Keep the existing training
# contract, but drop hsv_h from 0.015 to 0.005: hue jitter smears the apple/orange
# boundary, and that boundary is the axis this misclassification sits on
# (an under-ripe apple read as orange 0.95).
```

Mixing real with synthetic is itself non-trivial — 121,597 synthetic instances against 807
real ones means the real data is 0.7 % of the signal and disappears, while dropping the
synthetic data collapses `pineapple` and `plain` through catastrophic forgetting. The
published `105_build_mix_dataset.py` handles it by capping synthetic instances per class
with a greedy pass that prefers images containing rare classes, and by listing each real
image N times in the training manifest instead of duplicating files.

---

## 8. Honest result 2 — `val = train` is not a number

The A1 object segmenter was trained with its validation split *equal to* its training set.
It read **0.9941 box mAP50**. That number means nothing about generalisation, and it was
quoted internally for weeks.

A held-out evaluation set was built from a 3D replica of the real arena — 60 scenes, 573
objects (`cube_like_object` 320 / octahedron 88 / dodecahedron 75 / icosahedron 90),
rendered with the mimic-arena script and never used for training. The same weights:

| Metric | `val = train` (fit) | Held-out arena (generalisation) |
|---|---:|---:|
| box mAP50 | 0.9941 | **0.831** |
| box mAP50-95 | 0.9784 | **0.729** |
| mask mAP50 | 0.9942 | **0.821** |
| mask mAP50-95 | 0.9537 | **0.558** |

The same experiment then rejected a change that looked like an improvement. A low-LR extra
fine-tune (AdamW, `lr0 = 1e-5`, 30 epochs, run to completion):

- on `val = train`: **+0.0002 to +0.0008** — noise, or memorisation of the training set;
- on the held-out arena set: **−1.5 to −2.5 pp on all four metrics** (box mAP50 −2.5 pp).

Textbook overfitting, and **invisible in the `val = train` setting**. The original A1
weight was kept and the fine-tune marked `rejected`. That weight, `08ffa8c4…`, landed on
2026-07-02 and was never changed again — it is the weight in
`perception/models/a1_objectseg/best.pt`.

---

## 9. Honest result 3 — the recognition cliff

**Symptom.** A recorded set of real orange cubes was recognised only **65.1 %** of the time
(56/86 crops) by a model with a 0.894 mask mAP50-95 on synthetic validation.

**Diagnosis.** It was not colour. It was **composition**. A sweep over synthetic icon area
found a *cliff*: recognition collapses when the printed icon covers roughly **11–16 %** of
the face. Real arena icons cover **8–10 %** — below the cliff. The training data covered
about **30 %** — dense icons, well above it. And the decisive test: simply **enlarging** the
failing crops turned **25 of 30** of them correct. The model could read the icons. It had
never been shown small ones.

*(One internal index records the cliff band as 11–20 %; the presentation figure and the
booster recipe both use 11–16 %.)*

**Fix.** A booster targeted precisely at the cliff, using **no real imagery at all** — a
strict policy in this project, since the recorded set was the evaluation holdout. Four
iterations, each 100 scenes, each scored against the current model:

| Version | Change | Fruit top-class error | Verdict |
|---|---|---:|---|
| v1 | Icon area 0.38–0.62 linear (area only) | 6.4 % | Too easy — area alone is not enough |
| v2 | + `realistic_webcam_hard` degradation, area 0.30–0.52 | 10.6 % | Capture degradation is the key second axis |
| v3 | + print flattening (bilateral + k-means posterise, 75 %) | 12.1 % | Minor contribution |
| **v4** | **Area 0.24–0.40, matching the recorded icon geometry** | **17.8 %** | **Adopted — reproduces the real 65 % difficulty** |

The v4 recipe (`realistic_a4_sparse_icon`): printed patch covering **6–16 %** of the face,
posterised print flattening, webcam-grade degradation (white balance, exposure, gamma,
noise, low-res resample, vignette, JPEG). 10,000 scenes were rendered with it and used for
a low-LR fine-tune (120 epochs, best epoch 47).

| Metric | Base (`readd`) | Sparse-icon fine-tune |
|---|---:|---:|
| Recorded real-orange recognition | 65.1 % (56/86) | **95.3 % (82/86)** |
| orange−apple margin, p10 | −0.385 | **+0.111** |
| Arena probe apple / orange | 68.8 % / 69.7 % | **81.2 % / 78.8 %** |
| Mixed-val mask mAP50-95 | 0.894 | 0.894 (no regression) |

One regression remained: weak (0.2–0.4) fruit detections on blank crops, 17 of 25. It was
handled at runtime rather than in data, with the `fruit_min_conf = 0.45` guard described in
[`pipeline.md`](pipeline.md) §5 — a threshold derived from the same holdout, keeping 100 %
of true-orange faces while rejecting ~88 % of blank-crop false positives.

**This model is not the one that ran on the robot.** The sparse-icon fine-tune is the last
face model promoted here; the robot ran an earlier face weight throughout. The July
retraining round was therefore benchmarked against a checkpoint that was not the best
available, so some of the round-1 rejection verdicts were measured against a weaker baseline
than they should have been. Read those verdicts with that in mind.

---

## 10. The datasets

The rendered datasets are **not published**. Together they run to tens of
gigabytes, and hosting that is not something this project can maintain. What is
published is the recipe: every rule that decides a scene, a label and a rejection
is written out above, the seeds and counts are recorded per bundle below, and the
training entry points that consumed the output are in `perception/training/`.
Regenerating is the intended path — the renderer settings that matter are in §4,
§5 and §7, and the visibility gates in §6.

The table below is therefore a record of what was rendered and what each bundle
was for, not a download index.

| Bundle | Size | Contents |
|---|---|---|
| `meta_v2_50000_coco_texture_v1` | 50,000 scenes (40 k train / 5 k val / 5 k test) | The flagship render, full meta_v2 metadata. Every shipped model traces to it. |
| `meta_v2_topfruit_v3_{local,sp1,sp2,server}` | 30,000 scenes | The July v3 render — corrected fruit-face rule, hue-clamp-last, **per-machine seeds**. Bundled per machine so seed provenance stays visible. |
| `meta_v2_50000_cube_face_unified_v1` | 94,210 crops @ 224 px | Cube-face crops re-derived from the 50 k render with no re-rendering (pad 0.18, seed 20260628). Empty crops deliberately kept as hard negatives. |
| `face_crops_phase5` | 41,715 crops (37,621 / 4,094) | The final face training set. See below. |
| `face_crops_full` (round 1) | 47,081 / 4,957 crops | **ARCHIVE ONLY — DEFECTIVE.** The duplicate-seed set. Published deliberately: it is the counterexample that makes the seed-hygiene lesson concrete. Every round-1 validation number computed on it is inflated. |
| verifier patch sets | 12,667–15,029 per class | 224 px mask-quad perspective warps, balanced 50:50, 30 % capture-degraded. **Exclude the v1 sets** — contaminated by pre-hue-fix renders. |
| `banana_bunch_patches` | 5,000 + 1,051 holdout | The bunch composites and the holdout that measured 25.02 % → 0.00 %. |
| `arena_v3_a1_eval_v1` | 60 scenes / 573 objects | The held-out A1 evaluation set from §8. The only honest generalisation number A1 has. |

`face_crops_phase5` rebalancing is worth spelling out, because raw synthetic instance
counts are always wrong:

- Raw: apple 16,070 / orange 15,920 / banana 16,337 / pineapple 16,449 / **plain 77,914
  (54.6 %)**.
- Plain subsampled to 43,094 (**40.0 %**); apple capped at 15,920 to match orange *exactly*
  — since apple↔orange was the failure axis, an apple majority is a thumb on the scale.
- **30.0 % of training crops (11,305) were degraded in place**: gamma 1.2–2.2, contrast
  0.55–0.85, warm white balance (R ×1.03–1.15, B ×0.85–0.97), Gaussian noise σ 3–9,
  JPEG q 35–60, blur k3/k5.

### What is *not* redistributed

**The fruit texture pool is not published.** It is 6,400 images: Kaggle Fruits-360 at 65 %
(CC BY-SA — share-alike may propagate to derived renders), a 10 % FruitSeg30 slice whose
licence is not documented anywhere in the source repo, and a 25 % slice produced by
scraping a search engine, which is not redistributable. The COCO 2017 backgrounds are under
Flickr terms.

What *is* published is the **recipe**: the preparation scripts, the fixed 65 / 25 / 10
ratio (1,040 / 400 / 160 images per class, 1,600 per class total), the HSV colour-gate
thresholds, and the contact sheets. `download_generic_fruit_textures.py` (Wikimedia
Commons) is the sanctioned clean-rebuild path; the search-engine scraper is deliberately
not released.

This licensing question is unresolved and may encumber the *rendered* bundles above, not
only the texture pack. If you intend to build on these datasets commercially, establish
the Fruits-360 share-alike and FruitSeg30 provenance questions first.

### Reproducibility caveat

BlenderProc downloads and manages its own Blender binary, and **no log, document or
lockfile in either repository records which Blender version was used**. The environment
pins `blenderproc==2.8.0` and `bpy==5.0.1`. Run `blenderproc run` once and record the
printed version before claiming bit-level reproducibility.
