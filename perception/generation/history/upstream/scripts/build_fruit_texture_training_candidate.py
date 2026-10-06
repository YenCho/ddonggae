#!/usr/bin/env python3
"""Build a training-ready fruit texture candidate root.

The default candidate combines the curated/open local pool with the balanced AI
whole-fruit booster pool at a low ratio. Files are hardlinked by default to
avoid duplicating thousands of images.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


CLASSES = ("apple", "orange", "banana", "pineapple")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curated_root", type=Path, default=Path("datasets/fruit_textures/other/combined_curated_v1"))
    parser.add_argument(
        "--ai_root",
        type=Path,
        default=Path("datasets/fruit_textures/other/ai_wholefruit_booster_balanced_candidate_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/fruit_textures/other/training_candidate_curated_ai10_v1"),
    )
    parser.add_argument("--copy_mode", choices=("copy", "hardlink", "symlink"), default="hardlink")
    parser.add_argument("--ai_fraction", type=float, default=0.10, help="Target AI fraction of final per-class pool.")
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--clear", action="store_true")
    return parser.parse_args()


def clear_dir(path: Path) -> None:
    if not path.exists():
        return
    for child in sorted(path.iterdir()):
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def iter_images(root: Path, cls: str) -> list[Path]:
    class_dir = root / cls
    if not class_dir.exists():
        return []
    return sorted(path for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)


def link_or_copy(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    if mode == "symlink":
        try:
            dst.symlink_to(src.resolve())
            return "symlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy_fallback"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy_fallback"


def write_contact_sheet(paths: list[Path], output: Path, thumb: int = 112, cols: int = 8) -> None:
    thumbs: list[np.ndarray] = []
    for path in paths[:64]:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        thumbs.append(cv2.resize(img, (thumb, thumb), interpolation=cv2.INTER_AREA))
    if not thumbs:
        return
    rows = int(np.ceil(len(thumbs) / cols))
    canvas = np.full((rows * thumb, cols * thumb, 3), 245, dtype=np.uint8)
    for idx, img in enumerate(thumbs):
        row, col = divmod(idx, cols)
        canvas[row * thumb : (row + 1) * thumb, col * thumb : (col + 1) * thumb] = img
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 94])


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    curated_root = args.curated_root.resolve()
    ai_root = args.ai_root.resolve()
    rng = random.Random(args.seed)

    output.mkdir(parents=True, exist_ok=True)
    if args.clear:
        clear_dir(output)
    (output / "previews").mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, str | int | float]] = []
    summary: dict[str, dict[str, int | float]] = {}
    for cls in CLASSES:
        class_out = output / cls
        class_out.mkdir(parents=True, exist_ok=True)

        curated = iter_images(curated_root, cls)
        ai = iter_images(ai_root, cls)
        rng.shuffle(curated)
        rng.shuffle(ai)

        # ai / (curated + ai) ~= requested fraction.
        ai_quota = int(round((len(curated) * args.ai_fraction) / max(1e-9, 1.0 - args.ai_fraction)))
        selected_ai = ai[: min(ai_quota, len(ai))]

        selected: list[tuple[str, Path]] = [("curated", path) for path in curated] + [
            ("ai_balanced", path) for path in selected_ai
        ]
        rng.shuffle(selected)

        counts = {"curated": 0, "ai_balanced": 0}
        for idx, (source_type, src) in enumerate(selected):
            dst = class_out / f"{cls}_{idx:05d}_{source_type}{src.suffix.lower()}"
            op = link_or_copy(src, dst, args.copy_mode)
            counts[source_type] += 1
            manifest.append(
                {
                    "class": cls,
                    "source_type": source_type,
                    "source": str(src),
                    "output": str(dst.relative_to(output)),
                    "operation": op,
                }
            )

        total = counts["curated"] + counts["ai_balanced"]
        summary[cls] = {
            "curated": counts["curated"],
            "ai_balanced": counts["ai_balanced"],
            "total": total,
            "ai_fraction_actual": round(counts["ai_balanced"] / max(1, total), 4),
        }
        write_contact_sheet(sorted(class_out.glob("*")), output / "previews" / f"{cls}_contact_sheet.jpg")

    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "source_type", "source", "output", "operation"])
        writer.writeheader()
        writer.writerows(manifest)

    summary_data = {
        "root": str(output),
        "curated_root": str(curated_root),
        "ai_root": str(ai_root),
        "copy_mode_requested": args.copy_mode,
        "ai_fraction_target": args.ai_fraction,
        "classes": summary,
        "usage_note": "Training candidate texture root. Use for an experiment, then validate apple/orange/plain hard cases before promotion.",
    }
    (output / "summary.json").write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    print(json.dumps(summary_data, indent=2))


if __name__ == "__main__":
    main()
