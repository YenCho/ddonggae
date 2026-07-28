# ADR-0005 — Synthetic-only training data

**Status:** Accepted, then amended twice. Synthetic-only trained everything through qualifier 1;
qualifier 2 already deployed a real-photo retrained apple/orange verifier, and both finals ran a
real-photo face weight. Synthetic-only held through the
qualifiers; the face weight deployed in the finals was fine-tuned on real arena photographs
on 2026-07-24.

## Context

Eight object identities (four polyhedra, four printed fruit faces) had to be segmented and
identified with no labelled dataset in existence, by a small team, on a schedule where the
arena's own face rule was clarified on 2026-07-18 — four days before the first round.
Hand-labelling instance-segmentation masks for tens of thousands of images was not
available to us, and any labelled set would have been invalidated by the rule change.

## Decision

**Zero hand-labelled training images.** All data is rendered by one BlenderProc script
(~4,365 lines). Geometry is analytic — the dodecahedron is constructed as the exact dual of
the icosahedron — and every mesh is scaled to an 8 cm maximum extent to match the real
objects. Fruit cubes are a primitive cube with textured overlay planes 0.45 mm proud of each
face, carrying their **own segmentation category id**, so the renderer reports *printed-face*
visibility independently of *cube* visibility. Labels come from the rendered segmap, never
from a human. Cycles renders only the objects; they are composited onto COCO2017 or
procedural arena backgrounds, then distortion, gain, gamma, noise, vignette and JPEG
artefacts are applied **to pixels only, never to labels**.

## Consequences

**Benefits.**

- Pixel-exact masks, and a label the real world cannot provide: whether a printed face is
  visible enough to be worth classifying. Below the visibility gate a fruit cube is demoted
  to the generic `cube` class rather than dropped — which is what makes the runtime
  two-stage design possible at all.
- Class balance is a parameter, not a hope (apple crops capped to exactly match orange).
- The 2026-07-18 face-rule change was a re-render, not a re-labelling project.
- Diagnosed failures get fixed **in data**. A real banana-bunch face flipped to pineapple at
  0.758 confidence because 92 % of the banana texture pool was single bananas; injecting
  5,000 synthetic bunch composites plus 5,000 real cutouts took the bunch-holdout flip rate
  from 25.02 % to 0.00 %. A renderer bug leaking apple hue into the orange band was measured
  at 50 % leakage before the fix and 0.0 % after, over 200 jitter trials.

**Costs.**

- **The domain gap is the failure mode, and it cost us the qualifiers.** The apple printed
  on the real arena objects was a yellowish, under-ripe apple — not the red one announced
  beforehand. The face model, trained on red apples, mis-predicted it. A textbook
  train/deploy colour-domain gap.
- **Synthetic validation scores are not evidence.** Five round-1 models scored 0.858–0.9024
  mask mAP50-95 on synthetic validation and **all five lost to the incumbent on the real
  robot**; the cause was every render machine using the same default seed, leaving only
  ~22.8k of 53.7k scenes unique with train/val leakage. Separately, the A1 detector's
  0.9941 box mAP50 was a `val = train` artefact: on a purpose-built held-out set (60 scenes,
  573 objects) it is 0.831.
- **You only discover what you thought to render.** Recognition collapses when a printed
  icon occupies 11–16 % of a face; real icons occupy 8–10 %; the training data covered
  ~30 %. A sparse-icon booster took recorded real-orange recognition from 65.1 % to 95.3 % —
  and that model was never deployed.
- **It did not finish the job.** Before the finals the deployed face weight was fine-tuned on
  real arena photographs taken on 2026-07-24. Synthetic-only was the right call for building
  the system; a few hundred real photographs at the venue was the right call for winning
  with it.
