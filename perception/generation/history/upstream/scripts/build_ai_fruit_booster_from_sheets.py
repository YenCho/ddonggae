#!/usr/bin/env python3
"""Build an AI-generated whole-fruit texture booster from contact sheets.

The expected input is one or more 4x4 contact sheets per class on a flat
magenta (#ff00ff) chroma-key background. The script crops each cell, removes
the chroma key into an alpha matte, saves both RGBA cutouts and white-background
training textures, then writes a manifest and preview sheets.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


CLASSES = ("apple", "orange", "banana", "pineapple")


@dataclass
class Sample:
    cls: str
    sheet: str
    row: int
    col: int
    index: int
    cutout_path: Path
    white_path: Path
    mask_area: int
    bbox: tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("datasets/fruit_textures/other/ai_wholefruit_booster_v1"),
        help="Output dataset root containing source_sheets/ by default.",
    )
    parser.add_argument("--sheet_dir", type=Path, default=None)
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--pad_ratio", type=float, default=0.10)
    parser.add_argument("--min_mask_area", type=int, default=2000)
    parser.add_argument("--augment_per_crop", type=int, default=3)
    parser.add_argument("--fit_ratio_min", type=float, default=0.90)
    parser.add_argument("--fit_ratio_max", type=float, default=0.90)
    parser.add_argument("--extract_mode", choices=("grid", "components"), default="grid")
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--clear", action="store_true")
    return parser.parse_args()


def ensure_clean_dir(path: Path, clear: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not clear:
        return
    for child in path.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            for p in sorted(child.rglob("*"), reverse=True):
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    p.rmdir()
            child.rmdir()


def chroma_mask_bgr(img: np.ndarray, keep_largest: bool = True) -> np.ndarray:
    """Return foreground mask for a magenta chroma-key image."""
    b, g, r = cv2.split(img)
    bg = (r > 165) & (b > 165) & (g < 115) & ((r.astype(np.int16) - g.astype(np.int16)) > 70)
    mask = (~bg).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    if not keep_largest:
        keep_all = np.zeros_like(mask)
        for label in range(1, num):
            if stats[label, cv2.CC_STAT_AREA] >= 250:
                keep_all[labels == label] = 255
        return keep_all
    keep = np.zeros_like(mask)
    areas = [(label, stats[label, cv2.CC_STAT_AREA]) for label in range(1, num)]
    largest_label, largest_area = max(areas, key=lambda item: item[1])
    if largest_area <= 0:
        return keep
    keep[labels == largest_label] = 255

    # Preserve tiny nearby parts like apple stems only when they touch the main
    # fruit region after a small dilation. This drops stray fragments that some
    # generated contact sheets place between cells.
    touch_kernel = np.ones((13, 13), np.uint8)
    expanded_main = cv2.dilate(keep, touch_kernel, iterations=1)
    main_cx, main_cy = centroids[largest_label]
    for label, area in areas:
        if label == largest_label or area < 80 or area > largest_area * 0.18:
            continue
        component = labels == label
        cx, cy = centroids[label]
        if abs(cx - main_cx) > img.shape[1] * 0.35 or abs(cy - main_cy) > img.shape[0] * 0.35:
            continue
        if np.any(expanded_main[component] > 0):
            keep[component] = 255
    return keep


def crop_masked_region_to_rgba(
    img: np.ndarray,
    mask: np.ndarray,
    pad_ratio: float,
    min_mask_area: int,
) -> tuple[np.ndarray, int, tuple[int, int, int, int]] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) < min_mask_area:
        return None

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    w, h = x1 - x0, y1 - y0
    pad = int(max(w, h) * pad_ratio)
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(img.shape[1], x1 + pad)
    y1 = min(img.shape[0], y1 + pad)

    crop = img[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1]
    alpha = cv2.GaussianBlur(crop_mask, (3, 3), 0)
    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    return rgba, int((crop_mask > 0).sum()), (x0, y0, x1, y1)


def crop_to_square_rgba(cell: np.ndarray, pad_ratio: float, min_mask_area: int) -> tuple[np.ndarray, int, tuple[int, int, int, int]] | None:
    mask = chroma_mask_bgr(cell, keep_largest=True)
    return crop_masked_region_to_rgba(cell, mask, pad_ratio, min_mask_area)


def resize_on_canvas(
    rgba: np.ndarray,
    size: int,
    fit_ratio: float,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> tuple[np.ndarray, np.ndarray]:
    h, w = rgba.shape[:2]
    scale = min(size * fit_ratio / max(w, 1), size * fit_ratio / max(h, 1))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_AREA)

    rgba_canvas = np.zeros((size, size, 4), dtype=np.uint8)
    white_canvas = np.full((size, size, 3), bg_color, dtype=np.uint8)
    x = (size - nw) // 2
    y = (size - nh) // 2
    rgba_canvas[y : y + nh, x : x + nw] = resized

    alpha = resized[:, :, 3:4].astype(np.float32) / 255.0
    rgb = resized[:, :, :3].astype(np.float32)
    base = white_canvas[y : y + nh, x : x + nw].astype(np.float32)
    white_canvas[y : y + nh, x : x + nw] = (rgb * alpha + base * (1.0 - alpha)).astype(np.uint8)
    return rgba_canvas, white_canvas


def jitter_white_image(img: np.ndarray, rng: random.Random) -> np.ndarray:
    out = img.copy()
    angle = rng.uniform(-8.0, 8.0)
    scale = rng.uniform(0.88, 1.06)
    tx = rng.uniform(-14, 14)
    ty = rng.uniform(-14, 14)
    h, w = out.shape[:2]
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    mat[0, 2] += tx
    mat[1, 2] += ty
    out = cv2.warpAffine(out, mat, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

    alpha = rng.uniform(0.88, 1.12)
    beta = rng.uniform(-10, 10)
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    if rng.random() < 0.45:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), rng.uniform(0.25, 1.0))
    if rng.random() < 0.35:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), rng.randint(62, 88)]
        ok, buf = cv2.imencode(".jpg", out, encode_param)
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out


def write_contact_sheet(image_paths: list[Path], out_path: Path, thumb: int = 112, cols: int = 8) -> None:
    if not image_paths:
        return
    imgs = []
    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        imgs.append(cv2.resize(img, (thumb, thumb), interpolation=cv2.INTER_AREA))
    if not imgs:
        return
    rows = int(np.ceil(len(imgs) / cols))
    canvas = np.full((rows * thumb, cols * thumb, 3), 245, dtype=np.uint8)
    for i, img in enumerate(imgs):
        r, c = divmod(i, cols)
        canvas[r * thumb : (r + 1) * thumb, c * thumb : (c + 1) * thumb] = img
    cv2.imwrite(str(out_path), canvas)


def process_sheet(sheet_path: Path, cls: str, args: argparse.Namespace, rng: random.Random) -> list[Sample]:
    img = cv2.imread(str(sheet_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read sheet: {sheet_path}")
    h, w = img.shape[:2]
    cutout_dir = args.root / "cutouts_512" / cls
    white_dir = args.root / "white_512" / cls
    class_dir = args.root / cls
    for d in (cutout_dir, white_dir, class_dir):
        d.mkdir(parents=True, exist_ok=True)

    samples: list[Sample] = []
    sheet_stem = sheet_path.stem
    crop_items: list[tuple[int, int, int, tuple[np.ndarray, int, tuple[int, int, int, int]]]] = []
    if args.extract_mode == "components":
        full_mask = chroma_mask_bgr(img, keep_largest=False)
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(full_mask, connectivity=8)
        components: list[tuple[float, float, int, tuple[np.ndarray, int, tuple[int, int, int, int]]]] = []
        for label in range(1, num):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < args.min_mask_area:
                continue
            comp_mask = np.zeros_like(full_mask)
            comp_mask[labels == label] = 255
            cropped = crop_masked_region_to_rgba(img, comp_mask, args.pad_ratio, args.min_mask_area)
            if cropped is None:
                continue
            cx, cy = centroids[label]
            components.append((cy, cx, label, cropped))
        components.sort(key=lambda item: (item[0], item[1]))
        for idx, (_, _, _, cropped) in enumerate(components):
            crop_items.append((idx // args.grid, idx % args.grid, idx, cropped))
    else:
        cell_h = h // args.grid
        cell_w = w // args.grid
        idx = 0
        for row in range(args.grid):
            for col in range(args.grid):
                cell = img[row * cell_h : (row + 1) * cell_h, col * cell_w : (col + 1) * cell_w]
                cropped = crop_to_square_rgba(cell, args.pad_ratio, args.min_mask_area)
                if cropped is None:
                    continue
                crop_items.append((row, col, idx, cropped))
                idx += 1

    for row, col, idx, cropped in crop_items:
        rgba, mask_area, bbox = cropped
        fit_ratio = rng.uniform(args.fit_ratio_min, args.fit_ratio_max)
        rgba_canvas, white = resize_on_canvas(rgba, args.size, fit_ratio)
        name = f"ai_{cls}_{sheet_stem}_r{row:02d}_c{col:02d}"
        cutout_path = cutout_dir / f"{name}.png"
        white_path = white_dir / f"{name}.jpg"
        train_path = class_dir / f"{name}.jpg"
        cv2.imwrite(str(cutout_path), rgba_canvas)
        cv2.imwrite(str(white_path), white, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        cv2.imwrite(str(train_path), white, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        samples.append(Sample(cls, sheet_path.name, row, col, idx, cutout_path, white_path, mask_area, bbox))

        for aug_idx in range(args.augment_per_crop):
            aug = jitter_white_image(white, rng)
            aug_name = f"{name}_aug{aug_idx:02d}.jpg"
            cv2.imwrite(str(white_dir / aug_name), aug, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            cv2.imwrite(str(class_dir / aug_name), aug, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return samples


def main() -> None:
    args = parse_args()
    args.root = args.root.resolve()
    sheet_dir = (args.sheet_dir or (args.root / "source_sheets")).resolve()

    for sub in ("cutouts_512", "white_512", "previews"):
        ensure_clean_dir(args.root / sub, args.clear)
    for cls in CLASSES:
        ensure_clean_dir(args.root / cls, args.clear)

    rng = random.Random(args.seed)
    all_samples: list[Sample] = []
    summary: dict[str, int] = {}
    for cls in CLASSES:
        sheets = sorted(sheet_dir.glob(f"{cls}_sheet_*.png"))
        if not sheets:
            summary[cls] = 0
            continue
        count_before = len(all_samples)
        for sheet in sheets:
            all_samples.extend(process_sheet(sheet, cls, args, rng))
        summary[cls] = len(all_samples) - count_before

    manifest_path = args.root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class", "sheet", "row", "col", "index", "cutout_path", "white_path", "mask_area", "bbox"],
        )
        writer.writeheader()
        for sample in all_samples:
            writer.writerow(
                {
                    "class": sample.cls,
                    "sheet": sample.sheet,
                    "row": sample.row,
                    "col": sample.col,
                    "index": sample.index,
                    "cutout_path": str(sample.cutout_path.relative_to(args.root)),
                    "white_path": str(sample.white_path.relative_to(args.root)),
                    "mask_area": sample.mask_area,
                    "bbox": list(sample.bbox),
                }
            )

    expanded_counts = {
        cls: len(list((args.root / cls).glob("*.jpg")))
        for cls in CLASSES
    }
    summary_data = {
        "root": str(args.root),
        "source_sheets": str(sheet_dir),
        "base_crops_per_class": summary,
        "expanded_white_jpg_per_class": expanded_counts,
        "augment_per_crop": args.augment_per_crop,
        "fit_ratio_min": args.fit_ratio_min,
        "fit_ratio_max": args.fit_ratio_max,
        "extract_mode": args.extract_mode,
        "policy": {
            "apple": "whole red apple only, unpeeled, no slices",
            "orange": "whole orange-colored orange only, unpeeled, no slices",
            "banana": "whole yellow banana only, unpeeled, no exposed flesh",
            "pineapple": "whole brown/green pineapple exterior only, no exposed yellow flesh",
        },
        "usage_note": "Keep this as an AI-generated booster. Mix at a low ratio and validate against real/hard cases before promoting.",
    }
    (args.root / "summary.json").write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

    preview_dir = args.root / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for cls in CLASSES:
        paths = sorted((args.root / cls).glob("*.jpg"))[:64]
        write_contact_sheet(paths, preview_dir / f"{cls}_contact_sheet.jpg")

    print(json.dumps(summary_data, indent=2))


if __name__ == "__main__":
    main()
