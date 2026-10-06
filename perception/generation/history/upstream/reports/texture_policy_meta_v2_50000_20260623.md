# Meta V2 50k Texture Policy

Date: 2026-06-23

This note records the texture decision for the Meta V2 50k generation run.

## Production Texture Directory

Local production texture pack:

```text
datasets/fruit_textures/production_meta_v2_50000_v1
```

Base texture pack:

```text
datasets/fruit_textures/final_fruits36065_original25_fruitseg30_10
```

Experimental web-orange folders were moved to:

```text
datasets/fruit_textures/_archive_experimental_20260623_c_hardcase
```

## Preserved Ratios

The production pack preserves the existing class balance:

| class | count |
| --- | ---: |
| apple | 1600 |
| orange | 1600 |
| banana | 1600 |
| pineapple | 1600 |

The production pack also preserves the source-type ratio inside every class:

| source type | count per class | ratio |
| --- | ---: | ---: |
| lab | 1040 | 65% |
| original_filtered | 400 | 25% |
| fruitseg30 | 160 | 10% |

## Why Web Orange Textures Are Not Mixed Directly

The C classifier hard-case experiments showed that broad or weakly filtered web-orange images can contaminate the orange class. They can also overcorrect apple/orange confusion.

Therefore, the 50k run keeps the proven source ratio and increases visual variation only at render time:

```text
--fruit_texture_aug strong
--fruit_texture_layout mixed
--fruit_texture_collage_prob 0.35
```

This keeps apple/orange/banana/pineapple sampling balanced while making fruit-face appearance more diverse.
