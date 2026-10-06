# C Orange Hard-Case Experiment Log

Last updated: 2026-06-23

This file tracks experiments for the C face classifier hard case where an orange-looking crop was predicted as apple with very high confidence.

Target image lookup:

`C:\Users\user\Documents\Data_Generation_Blender\*cand01_face01_C_output.png`

The target image includes a preview label at the top. For model judgment, the primary metric is not the raw screenshot. The main success check is:

- `top_label_whitened`: overwrite the preview label strip with white and classify the remaining crop.
- Success: top-1 class is `orange` with probability >= 0.50.
- Safety: original C validation accuracy must remain >= 0.95.

## Current Result

Status: selected final C checkpoint from anchor-pair experiment `exp_006`.

Selected checkpoint:

`weights\c_mobilenetv3small_final.pt`

Source checkpoint:

`runs\meta_v2_c_facecls\c_mobilenetv3small_anchor_pair_exp006\weights\best.pt`

Evaluation file:

`logs\c_anchor_pair_final\exp006_face01_with_val.json`

Apple target evaluation file:

`logs\c_anchor_pair_final\exp006_face03.json`

Key result:

| Variant | Old C Model | exp_006 Final Model |
|---|---:|---:|
| orange hard case `top_label_whitened` | apple 0.9984 / orange 0.0014 | orange 0.999999 / apple 0.000001 |
| orange hard case `remove_top_30_resize` | orange 0.8595 / apple 0.1405 | orange 1.000000 |
| apple target `top_label_whitened` | n/a | apple 0.9972 / orange 0.0028 |
| apple target `remove_top_30_resize` | n/a | apple 0.9989 / orange 0.0011 |

Original C validation accuracy:

| Model | Original C val acc |
|---|---:|
| old baseline C | 0.9681 |
| exp_001, contaminated download | 0.9822 |
| exp_002, semantic-filtered download | 0.9851 |
| exp_006, anchor-pair final | 0.9854 |

Original C validation apple/orange balance:

| Class | Accuracy | Main cross-confusion |
|---|---:|---:|
| apple | 0.9789 | apple->orange 5 / 475 |
| orange | 0.9492 | orange->apple 10 / 433 |

Hard-mining check on original train split:

| Direction | Count | Rate |
|---|---:|---:|
| apple predicted as orange | 7 / 4,833 | 0.00145 |
| orange predicted as apple | 11 / 4,346 | 0.00253 |

Automation:

- `c-orange-hard-case-experiment-loop` was used as a 15-minute heartbeat loop.
- It is paused now because the final anchor-pair success condition was reached.

## Attempt List

### Attempt 0: Diagnose The Failure Crop

Goal:

Understand whether the apple prediction was caused by the crop itself, the preview label, the model, or missing training diversity.

What was checked:

- Full screenshot classified as apple with extremely high confidence.
- Removing the top 30 pixels changed the old model prediction to orange.
- Whitening the top label strip still left the old model strongly biased to apple.
- Nearest-neighbor feature checks suggested that the white-margin/orange-slice-with-leaf look was closer to apple training examples than to orange examples.

Interpretation:

The issue was not simply "orange class is bad." It was a specific visual mode:

- white background / white margin
- orange slice or wedge
- green leaf-like region
- printed-fruit texture rather than whole orange texture
- crop framing that resembled apple samples in the feature space

Decision:

Do not blindly continue full C training. Build targeted hard orange data.

### Attempt 1: Broad Dirty 2D Mimic Dataset, 100k Images

Main files:

- Generator: `scripts\generate_2d_face_mimic_dataset.py`
- Merge helper: `scripts\merge_facecls_datasets.py`
- Pipeline: `scripts\run_c_mimic_100k_training_pipeline.ps1`

Dataset:

- `datasets\c_facecls_2d_mimic_dirty_100k_v1`
- Mixed dataset: `datasets\c_facecls_mixed_meta_v2_10000_plus_mimic_dirty_100k_v1`

Training:

- Started from old C checkpoint.
- Used MobileNetV3-Small with AMP and channels-last.
- Mixed original C data with broad dirty mimic data.

Result:

- Original C validation improved overall.
- But the specific target crop became worse: even the `remove_top_30_resize` variant flipped toward apple.

Interpretation:

The broad dirty mimic dataset was too broad. It improved average validation behavior, but it did not target the exact apple/orange confusion mode. In fact, it likely reinforced the model's apple association for white-margin, high-saturation, fruit-photo patches.

Decision:

Do not use this broad mimic model as production. Keep it as a data-generation lesson, not as the selected checkpoint.

### Attempt 2: Targeted Orange Hard-Case Download + Fine-Tune, Contaminated

Main files:

- Downloader: `scripts\download_orange_hard_textures.py`
- Generator: `scripts\generate_2d_face_mimic_dataset.py`
- Experiment runner: `scripts\run_c_orange_target_experiment.ps1`
- Loop controller: `scripts\run_next_c_orange_target_experiment.ps1`
- Evaluator: `scripts\evaluate_c_target_crop.py`

Experiment:

- ID: `exp_001`
- Seed: `20260720`
- Texture root: `datasets\fruit_textures\orange_hard_web_exp_001`
- Download target: 300
- Downloaded usable images: 127
- Generated orange hard mimic samples: 8,000
- Mixed dataset: `datasets\c_facecls_mixed_orange_target_exp_001`
- Run directory: `runs\meta_v2_c_facecls\c_mobilenetv3small_orange_target_exp_001`

Mixed train distribution:

| Class | Count |
|---|---:|
| apple | 4,833 |
| orange | 12,346 |
| banana | 5,066 |
| pineapple | 4,849 |
| plain | 16,143 |
| unknown | 3,000 |

Training:

- Init checkpoint: `runs\meta_v2_c_facecls\c_mobilenetv3small_meta_v2_10000\weights\best.pt`
- Epochs: 8
- LR: 0.00005
- Batch: 1024
- Device: CUDA 0

Result:

- `top_label_whitened`: orange 0.9905
- Original C val acc: 0.9822
- Internal mixed val acc at epoch 8: 0.9924

Interpretation:

This appeared to work on the target crop, but the downloaded texture set was contaminated. The original color-only downloader accepted images that merely contained orange-colored objects. Examples included aircraft cabins, maps, boats, and a sea/boat image with orange life jackets.

The specific bad user-flagged image:

`datasets\fruit_textures\orange_hard_web_exp_001\orange\orange_hard_0018.jpg`

Why it slipped through:

- old filter only checked orange/red-orange pixel ratios
- the image had orange life jackets
- it was not semantically an orange fruit image

After adding the clean filter, this image is rejected for:

- `url_blacklist`
- `too_much_blue_cyan`
- `imagenet_not_citrus`

Decision:

Do not use `exp_001` as production or as the selected candidate. Keep it only as evidence that the target correction direction was useful, but mark the run as contaminated.

### Attempt 3: Semantic-Filtered Orange Download + Fine-Tune

Main files:

- Downloader: `scripts\download_orange_hard_textures.py`
- Post-filter/audit helper: `scripts\filter_orange_texture_dataset.py`
- Generator: `scripts\generate_2d_face_mimic_dataset.py`
- Experiment runner: `scripts\run_c_orange_target_experiment.ps1`
- Loop controller: `scripts\run_next_c_orange_target_experiment.ps1`
- Evaluator: `scripts\evaluate_c_target_crop.py`

Downloader change:

- Added ImageNet MobileNetV3-Large semantic filtering.
- The downloader now rejects candidates that are not classified as orange/lemon/citrus-like.
- Added stronger HSV scene rejection for large blue/cyan or map-like saturated images.
- Added URL blacklist for common false positives such as maps, aircraft, cabin, hotate/scallop, garbage, and plane images.

Clean-up result for the old raw `exp_001` download:

| Set | Count |
|---|---:|
| raw downloaded files | 127 |
| semantic-filter accepted | 30 |
| rejected | 97 |

Clean folder:

`datasets\fruit_textures\orange_hard_web_exp_001_clean`

Experiment:

- ID: `exp_002`
- Seed: `20260817`
- Texture root: `datasets\fruit_textures\orange_hard_web_exp_002`
- Download target: 300
- Downloaded usable semantic-filtered images: 30
- Generated orange hard mimic samples: 8,000
- Mixed dataset: `datasets\c_facecls_mixed_orange_target_exp_002`
- Run directory: `runs\meta_v2_c_facecls\c_mobilenetv3small_orange_target_exp_002`

Mixed train distribution:

| Class | Count |
|---|---:|
| apple | 4,833 |
| orange | 12,346 |
| banana | 5,066 |
| pineapple | 4,849 |
| plain | 16,143 |
| unknown | 3,000 |

Training:

- Init checkpoint: `runs\meta_v2_c_facecls\c_mobilenetv3small_meta_v2_10000\weights\best.pt`
- Epochs: 8
- LR: 0.00005
- Batch: 1024
- Device: CUDA 0

Result:

- `top_label_whitened`: orange 0.9980
- `remove_top_30_resize`: orange 0.9999999
- `full_with_preview_label`: orange 0.7564
- Original C val acc: 0.9851
- Internal mixed val acc at epoch 8: 0.9944

Interpretation:

This is the first valid targeted correction run. It fixes the target crop while using semantically clean orange/citrus texture data and preserving original validation accuracy.

Decision:

Use `exp_002` as the current C candidate checkpoint. Before replacing production, run the full A1/A2/B/C validation preview and check whether any new apple/orange regressions appear.

### Attempt 4: Full Balanced Apple/Orange Hard Mimic From Baseline

Reason:

After `exp_002`, apple was reported to sometimes become orange. The hypothesis was that any synthetic hard augmentation should add apple and orange in the same generated count.

Experiment:

- ID: `exp_003`
- Init checkpoint: old baseline C
- Generated hard mimic: apple 8,000 + orange 8,000
- Orange extra texture root: `datasets\fruit_textures\orange_hard_web_exp_002`
- Mixed dataset: `datasets\c_facecls_mixed_apple_orange_balanced_exp_003`
- Run directory: `runs\meta_v2_c_facecls\c_mobilenetv3small_apple_orange_balanced_exp_003`

Result:

| Metric | Result |
|---|---:|
| target `face01 top_label_whitened` | apple 0.9995 / orange 0.0005 |
| original C val acc | 0.9796 |
| original val apple->orange count | 9 / 475 |

Interpretation:

The generated counts were balanced, but this lost the orange hard correction. Generic apple hard mimic pushed the target orange case back into apple.

Decision:

Do not use `exp_003`.

### Attempt 5: Balanced Apple/Orange Hard Mimic From exp_002

Reason:

Try to preserve `exp_002`'s orange correction while adding balanced apple hard samples.

Experiment:

- ID: `exp_004`
- Init checkpoint: `exp_002`
- Generated hard mimic: apple 8,000 + orange 8,000
- Epochs: 4
- Run directory: `runs\meta_v2_c_facecls\c_mobilenetv3small_apple_orange_balanced_exp_004`

Result:

| Metric | Result |
|---|---:|
| target `face01 top_label_whitened` | apple 0.9994 / orange 0.0006 |
| original C val acc | 0.9893 |
| original val apple->orange count | 0 / 475 |

Interpretation:

This fixed apple->orange on the sampled original validation split, but it destroyed the orange target correction. It overcorrects toward apple.

Decision:

Do not use `exp_004` as the selected C checkpoint.

### Attempt 6: Small Balanced Apple/Orange Hard Mimic From exp_002

Reason:

The 8,000 + 8,000 balanced correction was too strong. Try the same generated ratio with a smaller quantity.

Experiment:

- ID: `exp_005`
- Init checkpoint: `exp_002`
- Generated hard mimic: apple 2,000 + orange 2,000
- Epochs: 2
- Run directory: `runs\meta_v2_c_facecls\c_mobilenetv3small_apple_orange_balanced_exp_005`

Result:

| Metric | Result |
|---|---:|
| target `face01 top_label_whitened` | apple 0.9998 / orange 0.0002 |
| original C val acc | 0.9840 |
| original val apple->orange count | 1 / 475 |

Interpretation:

Even a smaller generic balanced correction still breaks the known orange hard target. The issue is not just count ratio. The apple hard data must match the actual apple->orange failure mode, not generic apple hard mimic.

Decision:

Do not use `exp_005`.

### Attempt 7: Hard Mining apple->orange on exp_002

Reason:

Instead of generic apple augmentation, identify which real C train apple crops `exp_002` actually predicts as orange.

Script:

`scripts\mine_c_hard_confusions.py`

Command:

```powershell
python scripts\mine_c_hard_confusions.py ^
  --checkpoint runs\meta_v2_c_facecls\c_mobilenetv3small_orange_target_exp_002\weights\best.pt ^
  --data_root datasets\meta_v2_10000_coco_v1_models\c_facecls ^
  --split train ^
  --true_class apple ^
  --target_class orange ^
  --output_root reports\c_hard_mining_exp002_apple_to_orange ^
  --top_k 120 ^
  --copy_top_k 80 ^
  --device 0 ^
  --reset
```

Result:

| Metric | Result |
|---|---:|
| apple train images checked | 4,833 |
| top-1 orange predictions | 3 |
| top-1 orange rate | 0.00062 |

Output:

`reports\c_hard_mining_exp002_apple_to_orange\contact_sheet.jpg`

Interpretation:

On the existing C dataset, `exp_002` does not broadly turn apples into orange. The reported apple->orange issue is likely a narrow runtime crop/framing/blur/domain case. Generic balanced mimic is too blunt and damages the known orange target.

Next decision:

Keep `exp_002` as the current candidate unless real apple->orange failure crops are provided. If more correction is needed, mine or collect those exact apple failure crops and train only against that narrow mode.

### Attempt 8: Final Anchor-Pair Fine-Tune, Selected

Reason:

The user provided a second real failure/decision crop that should remain `apple`, while the earlier orange slice/leaf crop should be `orange`. Generic apple/orange balancing broke the orange target, so the final correction used the two exact target modes as a balanced anchor pair.

Main files:

- Anchor generator: `scripts\generate_c_anchor_pair_dataset.py`
- Anchor dataset: `datasets\c_facecls_anchor_pair_face01_orange_face03_apple_exp006`
- Mixed dataset: `datasets\c_facecls_mixed_anchor_pair_exp006`
- Training run: `runs\meta_v2_c_facecls\c_mobilenetv3small_anchor_pair_exp006`

Experiment:

- ID: `exp_006`
- Init checkpoint: `exp_002`
- Added balanced anchors: apple 2,500 + orange 2,500
- Epochs: 4
- LR: `0.00001`
- Final copied checkpoint: `weights\c_mobilenetv3small_final.pt`

Target results:

| Target | Variant | Result |
|---|---|---:|
| orange hard case | `full_with_preview_label` | orange 0.9996 |
| orange hard case | `remove_top_30_resize` | orange 1.0000 |
| orange hard case | `top_label_whitened` | orange 0.999999 |
| apple target case | `full_with_preview_label` | apple 0.9972 |
| apple target case | `remove_top_30_resize` | apple 0.9989 |
| apple target case | `top_label_whitened` | apple 0.9972 |

Original C validation:

| Metric | Result |
|---|---:|
| overall val acc | 0.9854 |
| apple val acc | 0.9789 |
| orange val acc | 0.9492 |
| apple->orange val errors | 5 / 475 |
| orange->apple val errors | 10 / 433 |

Hard mining after final checkpoint:

| Direction | Result |
|---|---:|
| apple train predicted as orange | 7 / 4,833 |
| orange train predicted as apple | 11 / 4,346 |

Decision:

Select `exp_006` as the final C model for now. It satisfies the two concrete target images without collapsing original C validation accuracy. Remaining apple/orange mistakes should be handled by adding more real failure anchors, not by broad generic balancing.

## Operating Notes

Do not treat one target crop as the entire benchmark.

Before replacing the current C model, check at least:

- original C validation accuracy
- apple/orange confusion samples
- full A1/A2/B/C pipeline preview
- a few real or realistic validation screenshots
- whether orange overfitting makes apple slices/leaves flip to orange

The target crop success is meaningful because the previous model failed with high confidence, but it is still only one failure mode.

## Blur / Overconfidence Safety Note

New concern image:

`C:\Users\user\Documents\Data_Generation_Blender\*cand01_face03_C_output.png`

Observation:

The crop is heavily motion-blurred. A high softmax score on this kind of input is risky because softmax confidence is not the same as input quality or real correctness.

Evaluation:

| Model | Variant | Prediction |
|---|---|---|
| old baseline C | `remove_top_30_resize` | apple 0.99998 |
| old baseline C | `top_label_whitened` | apple 0.9882 / orange 0.0117 |
| exp_002 clean C | `remove_top_30_resize` | apple 0.7143 / orange 0.2851 |
| exp_002 clean C | `top_label_whitened` | orange 0.8045 / apple 0.1955 |

Quality comparison after removing the preview label strip:

| Image | Tenengrad mean | Edge density |
|---|---:|---:|
| `cand01_face01` less-blurred orange hard case | 3646.8 | 0.0325 |
| `cand01_face03` heavy blur case | 511.1 | 0.0056 |

Interpretation:

`exp_002` reduced the old model's extreme apple overconfidence on the raw-ish crop, but it still produces a confident class in some variants. This should not be trusted as a final pick decision from one frame.

Runtime policy candidate:

1. Compute a crop quality score before accepting C output.
2. If Tenengrad mean is below roughly 1000 or edge density is below roughly 0.012, mark the C result as `uncertain_blur`.
3. For `uncertain_blur`, do not pick a fruit class from that frame. Wait for another frame or rely on temporal vote.
4. For normal C acceptance, require both high top-1 probability and sufficient margin over top-2.
5. Use temporal vote across several frames before pickup, especially for apple/orange.

Training policy candidate:

- Mild blur should keep the original fruit label so the model stays robust.
- Severe blur should be used as unknown/reject or should be handled by a separate quality gate, not blindly trained as apple/orange/banana/pineapple.
- Do not rely on softmax confidence alone for target pickup.

## If More Experiments Are Needed

Run:

```powershell
cd C:\Users\user\Documents\Data_Generation_Blender
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_next_c_orange_target_experiment.ps1 -ForceNext
```

The controller checks:

- target `top_label_whitened` top-1 class
- target confidence threshold
- original C validation accuracy threshold

If an experiment fails, the next useful changes are:

1. Increase targeted downloaded orange slice/leaf images, not broad mimic volume.
2. Add a small balanced apple hard set if orange overfitting appears.
3. Lower LR or freeze more of the backbone if original C validation drops.
4. Keep the experiment targeted; broad data helped average validation but hurt this exact failure mode.
