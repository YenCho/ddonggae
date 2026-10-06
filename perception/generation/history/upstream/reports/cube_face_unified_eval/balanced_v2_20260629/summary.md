# Cube-Face Unified Balanced v2 Evaluation

Date: 2026-06-29

## Runs

- Baseline: `runs/segment/cube_face_unified_yolo26n_seg_v1/weights/best.pt`
- v1 plain-heavy fine-tune: `runs/segment/cube_face_unified_yolo26n_seg_wholefruit_plainhard_materialized_ft_v1/weights/best.pt`
- v2 balanced fine-tune: `runs/segment/cube_face_unified_yolo26n_seg_wholefruit_plainhard_balanced_ft_v2/weights/best.pt`
- Validation data: `datasets/cube_face_unified_finetune_wholefruit_plainhard_materialized_v2_balanced/data.yaml`

v2 stopped by EarlyStopping after 13 epochs. The best checkpoint was epoch 1.

## Overall Validation

| metric | baseline | v1 plain-heavy | v2 balanced |
| --- | ---: | ---: | ---: |
| Box precision | 0.91758 | 0.91551 | 0.91927 |
| Box recall | 0.87655 | 0.89842 | 0.88037 |
| Box mAP50 | 0.94608 | 0.96057 | 0.95969 |
| Box mAP50-95 | 0.87291 | 0.89589 | 0.89509 |
| Mask precision | 0.92018 | 0.91473 | 0.91880 |
| Mask recall | 0.87503 | 0.89722 | 0.87904 |
| Mask mAP50 | 0.94415 | 0.95777 | 0.95756 |
| Mask mAP50-95 | 0.85499 | 0.87114 | 0.87483 |
| fitness | 1.72790 | 1.76704 | 1.76993 |

## Key Class Tradeoffs

| class metric | baseline | v1 plain-heavy | v2 balanced |
| --- | ---: | ---: | ---: |
| apple mask recall | 0.88449 | 0.86845 | 0.88531 |
| orange mask recall | 0.90065 | 0.87320 | 0.88338 |
| plain mask recall | 0.78489 | 0.93464 | 0.89644 |
| plain mask mAP50 | 0.86913 | 0.97118 | 0.96560 |
| plain mask mAP50-95 | 0.73555 | 0.83914 | 0.82430 |

Interpretation:

- v1 is the strongest plain suppressor, but it over-pressures fruit recall and small-apple confidence.
- v2 keeps most of the plain hard-negative improvement while recovering apple recall to baseline level.
- v2 orange recall is still below baseline, but it is better than v1 and the aggregate mask mAP50-95/fitness is best among the three.

## Hard-Case Direct Crop Check

Top prediction per hard crop:

| hard case | baseline | v1 plain-heavy | v2 balanced |
| --- | --- | --- | --- |
| small whole apple | apple 0.883 | apple 0.112 | apple 0.872 |
| orange-slice screenshot | apple 0.839 | apple 0.719 | apple 0.943 |
| blurred fruit screenshot | apple 0.680 | apple 0.575 | apple 0.763 |
| full runtime screenshot | no face | pineapple 0.086 | no face |

The orange-slice screenshot remains a failure case, but it is not the current primary target because the competition face images are assumed to be whole-fruit photos rather than cut/slice photos.

The important recovery is the small whole-apple crop: v1 collapsed confidence from `0.883` to `0.112`, while v2 restored it to `0.872`.

## Decision

Use v2 balanced as the preferred cube-face unified checkpoint for runtime testing:

```text
runs/segment/cube_face_unified_yolo26n_seg_wholefruit_plainhard_balanced_ft_v2/weights/best.pt
```

Keep v1 as an alternate checkpoint when plain false positives are the only priority, but do not use it as the default runtime checkpoint.

Next practical step:

```text
Run ABC/unified runtime tests with v2 balanced and compare real camera false positives against the previous baseline.
```
