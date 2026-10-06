# Cube-Face Unified Whole-Fruit / Plain-Hard Fine-Tune Evaluation

Date: 2026-06-29

## Runs

- Baseline: `runs/segment/cube_face_unified_yolo26n_seg_v1/weights/best.pt`
- Fine-tuned v1: `runs/segment/cube_face_unified_yolo26n_seg_wholefruit_plainhard_materialized_ft_v1/weights/best.pt`
- Validation data: `datasets/cube_face_unified_finetune_wholefruit_plainhard_materialized_v1/data.yaml`

The v1 fine-tune stopped by early stopping after 13 epochs. The best checkpoint was epoch 1.

## Overall Validation

| metric | baseline | fine-tuned v1 | delta |
| --- | ---: | ---: | ---: |
| Box precision | 0.91877 | 0.91413 | -0.00464 |
| Box recall | 0.88441 | 0.90573 | +0.02132 |
| Box mAP50 | 0.94943 | 0.96575 | +0.01633 |
| Box mAP50-95 | 0.87536 | 0.89998 | +0.02462 |
| Mask precision | 0.91780 | 0.91516 | -0.00264 |
| Mask recall | 0.88538 | 0.90235 | +0.01698 |
| Mask mAP50 | 0.94778 | 0.96448 | +0.01670 |
| Mask mAP50-95 | 0.85846 | 0.87723 | +0.01877 |

## Class-Level Reading

The fine-tune achieved the intended plain hard-negative effect:

- plain mask recall: `0.793 -> 0.935`
- plain mask mAP50: `0.871 -> 0.976`
- plain mask mAP50-95: `0.736 -> 0.840`

However, the plain-heavy booster also reduced fruit recall:

- apple mask recall: `0.910 -> 0.883`
- orange mask recall: `0.904 -> 0.869`
- banana mask recall: `0.909 -> 0.907`
- pineapple mask recall: `0.910 -> 0.918`

Interpretation: v1 is better at suppressing plain false positives, but the booster pressure is too plain-heavy. It improves aggregate mAP because plain is large and frequent, while some fruit recall is traded away.

## Hard-Case Direct Crop Check

Outputs:

- `hard_cases_baseline/report.json`
- `hard_cases_fine_tuned/report.json`

Representative observations:

- Small apple hard case: baseline predicted `apple 0.883`; v1 predicted `apple 0.112`. The class stayed correct, but confidence fell too far.
- Orange-slice screenshot: baseline predicted mostly `apple 0.839`; v1 still predicted `apple 0.719`. This is not a primary target anymore because the current assumption is that competition fruit images are whole-fruit, not slices.
- Blurred fruit screenshot: baseline predicted `apple 0.680`; v1 predicted `apple 0.575`. Confidence is lower but still not explicitly uncertainty-aware.
- Full runtime screenshot is not a valid direct C crop; direct-crop results on it should not be used as a C-model metric.

## Decision

Keep v1 as a useful checkpoint, but do not treat it as final. The next concrete attempt should reduce plain over-pressure while preserving the whole-fruit/plain-hard idea:

- use a larger base sample from the 50k cube-face dataset
- use the booster only once instead of twice
- train a balanced v2 from the original baseline checkpoint

Proposed v2 mix:

```text
base train sample: 30,000
booster train repeat: 1x
expected train size: about 38,946
```

Goal for v2:

- keep most of the v1 plain improvement
- recover apple/orange recall and confidence
- avoid optimizing for fruit slices unless later real competition images contradict the whole-fruit assumption
