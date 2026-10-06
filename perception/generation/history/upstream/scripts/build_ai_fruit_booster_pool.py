#!/usr/bin/env python3
"""Build a balanced AI whole-fruit booster pool from reviewed AI sets."""

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


DEFAULT_SOURCES = (
    ("v1_clean", "datasets/fruit_textures/other/ai_wholefruit_booster_v1", 64),
    ("v2_camera", "datasets/fruit_textures/other/ai_wholefruit_booster_v2_camera_variation", 64),
    ("v3_softprint", "datasets/fruit_textures/other/ai_wholefruit_booster_v3_lowres_softprint", 64),
    ("v4_smallmargin", "datasets/fruit_textures/other/ai_wholefruit_booster_v4_small_margin", 64),
    ("v5_runtime_degraded", "datasets/fruit_textures/other/ai_wholefruit_booster_v5_runtime_degraded", 96),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/fruit_textures/other/ai_wholefruit_booster_balanced_candidate_v1"),
    )
    parser.add_argument("--copy_mode", choices=("copy", "hardlink", "symlink"), default="hardlink")
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
    cv2.imwrite(str(output), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 94])


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.clear:
        clear_dir(output)
    (output / "previews").mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    manifest: list[dict[str, str | int]] = []
    summary: dict[str, dict[str, int]] = {}

    for cls in CLASSES:
        class_out = output / cls
        class_out.mkdir(parents=True, exist_ok=True)
        class_counts: dict[str, int] = {}
        class_total = 0
        for source_label, source_root_text, quota in DEFAULT_SOURCES:
            source_root = Path(source_root_text).resolve()
            paths = iter_images(source_root, cls)
            rng.shuffle(paths)
            selected = paths[: min(quota, len(paths))]
            class_counts[source_label] = len(selected)
            for idx, src in enumerate(selected):
                dst_name = f"{cls}_{source_label}_{idx:04d}{src.suffix.lower()}"
                dst = class_out / dst_name
                op = link_or_copy(src, dst, args.copy_mode)
                manifest.append(
                    {
                        "class": cls,
                        "source_label": source_label,
                        "source": str(src),
                        "output": str(dst.relative_to(output)),
                        "operation": op,
                    }
                )
                class_total += 1
        class_counts["total"] = class_total
        summary[cls] = class_counts
        write_contact_sheet(sorted(class_out.glob("*")), output / "previews" / f"{cls}_contact_sheet.jpg")

    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "source_label", "source", "output", "operation"])
        writer.writeheader()
        writer.writerows(manifest)

    summary_data = {
        "root": str(output),
        "copy_mode_requested": args.copy_mode,
        "source_plan": [
            {"label": label, "root": root, "quota_per_class": quota}
            for label, root, quota in DEFAULT_SOURCES
        ],
        "classes": summary,
        "usage_note": "AI-only candidate pool. Mix into real/curated fruit textures at low ratio only; validate hard cases before promotion.",
    }
    (output / "summary.json").write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    print(json.dumps(summary_data, indent=2))


if __name__ == "__main__":
    main()
