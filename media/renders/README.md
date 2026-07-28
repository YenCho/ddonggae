# 3D renders

Rendered stills of the robot, supplied by the team. **24 frames**: a turntable at
60° steps in four configurations. Delivered as PNG, stored here as WebP — 9.3 MB
of near-identical grey-background renders compresses to 0.9 MB with no visible
loss, and this repository deliberately has no LFS.

| Mast | View | Files |
|---|---|---|
| up | iso | `mast-up_iso_00.webp` … `_05.webp` |
| up | side | `mast-up_side_00.webp` … `_05.webp` |
| down | iso | `mast-down_iso_00.webp` … `_05.webp` |
| down | side | `mast-down_side_00.webp` … `_05.webp` |

`00` is the front of the robot (gripper toward the camera) and the index
increases with the turntable, 60° per step. If the files arrive named
differently, rename them to this scheme rather than editing the consumers —
everything below is generated from these names.

## What gets built from them

| Output | Made from | Used by |
|---|---|---|
| `robot-hero.webp` | one iso frame, mast up | root `README.md` header |
| `robot-overview.webp` | one iso frame, mast down | `hardware/README.md`, root `README.md` |
| `mast-up-down.webp` | the two side frames at the same angle, side by side | `hardware/README.md` |
| `robot-turntable.webp` | the six iso frames, looping | root `README.md` |

The mast comparison is the one that earns its place: two mast heights means two
camera calibrations, and the perception documents refer to that difference
constantly.

## No interactive viewer

There is no `.glb` export and no GitHub Pages model viewer. GitHub strips scripts
and canvases from rendered Markdown, so a live 3D viewer could never have run
inside a README; the looping turntable is what a reader actually sees.
