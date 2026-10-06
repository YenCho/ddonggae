#!/usr/bin/env python3
"""Build a runtime-degraded booster from existing AI whole-fruit textures.

This script does not create new semantic content. It takes already-reviewed
AI whole-fruit boosters and produces harder versions that resemble C-model
runtime crops: lower effective resolution, JPEG artifacts, blur, exposure
variation, mild color-temperature shifts, and slight off-center placement.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


CLASSES = ("apple", "orange", "banana", "pineapple")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_roots",
        nargs="+",
        type=Path,
        default=[
            Path("datasets/fruit_textures/other/ai_wholefruit_booster_v1"),
            Path("datasets/fruit_textures/other/ai_wholefruit_booster_v2_camera_variation"),
            Path("datasets/fruit_textures/other/ai_wholefruit_booster_v3_lowres_softprint"),
            Path("datasets/fruit_textures/other/ai_wholefruit_booster_v4_small_margin"),
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/fruit_textures/other/ai_wholefruit_booster_v5_runtime_degraded"),
    )
    parser.add_argument("--variants_per_source", type=int, default=2)
    parser.add_argument("--max_sources_per_class", type=int, default=160)
    parser.add_argument("--image_size", type=int, default=512)
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


def read_image(path: Path, size: int) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def make_motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    length = max(3, int(length) | 1)
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    mat = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, mat, (length, length))
    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel /= kernel_sum
    return kernel


def transform(img: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict[str, float | int | str]]:
    h, w = img.shape[:2]
    out = img.copy()
    meta: dict[str, float | int | str] = {}

    # Simulate the fruit texture becoming small in a C crop, then being scaled
    # back for the classifier.
    if rng.random() < 0.45:
        fit = rng.uniform(0.45, 0.92)
        small = int(round(w * fit))
        resized = cv2.resize(out, (small, small), interpolation=cv2.INTER_AREA)
        canvas = np.full_like(out, 255)
        dx = rng.randint(-(w - small) // 3, (w - small) // 3) if w > small else 0
        dy = rng.randint(-(h - small) // 3, (h - small) // 3) if h > small else 0
        x = (w - small) // 2 + dx
        y = (h - small) // 2 + dy
        canvas[y : y + small, x : x + small] = resized
        out = canvas
        meta["fit"] = round(fit, 3)
    else:
        meta["fit"] = 1.0

    # Low effective camera/warp resolution.
    low = rng.choice([64, 80, 96, 112, 128, 160, 192, 224, 256])
    low_img = cv2.resize(out, (low, low), interpolation=cv2.INTER_AREA)
    out = cv2.resize(low_img, (w, h), interpolation=cv2.INTER_LINEAR)
    meta["low_res"] = low

    # Small affine drift that mimics imperfect face crop/warp while keeping the
    # image front-facing enough for C.
    angle = rng.uniform(-5.0, 5.0)
    scale = rng.uniform(0.94, 1.04)
    tx = rng.uniform(-10, 10)
    ty = rng.uniform(-10, 10)
    mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    mat[0, 2] += tx
    mat[1, 2] += ty
    out = cv2.warpAffine(out, mat, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    meta["angle"] = round(angle, 3)
    meta["scale"] = round(scale, 3)

    # Exposure, contrast, and color temperature shifts.
    alpha = rng.uniform(0.75, 1.22)
    beta = rng.uniform(-20, 20)
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    meta["contrast"] = round(alpha, 3)
    meta["brightness"] = round(beta, 3)

    b, g, r = cv2.split(out.astype(np.float32))
    temp = rng.uniform(-0.10, 0.10)
    r *= 1.0 + temp
    b *= 1.0 - temp
    out = cv2.merge([b, g, r])
    out = np.clip(out, 0, 255).astype(np.uint8)
    meta["temp_shift"] = round(temp, 3)

    # Blur family.
    blur_choice = rng.random()
    if blur_choice < 0.35:
        k = rng.choice([3, 5, 7])
        sigma = rng.uniform(0.4, 1.5)
        out = cv2.GaussianBlur(out, (k, k), sigma)
        meta["blur"] = f"gaussian_{k}_{sigma:.2f}"
    elif blur_choice < 0.62:
        length = rng.choice([5, 7, 9, 11])
        angle_blur = rng.uniform(0, 180)
        out = cv2.filter2D(out, -1, make_motion_kernel(length, angle_blur))
        meta["blur"] = f"motion_{length}_{angle_blur:.1f}"
    else:
        meta["blur"] = "none"

    # Sensor noise is intentionally mild.
    if rng.random() < 0.35:
        sigma = rng.uniform(1.5, 5.5)
        noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0, sigma, out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        meta["noise_sigma"] = round(sigma, 3)
    else:
        meta["noise_sigma"] = 0

    # JPEG artifacts last.
    quality = rng.randint(45, 86)
    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if ok:
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    meta["jpeg_quality"] = quality
    return out, meta


def write_contact_sheet(paths: list[Path], out_path: Path, thumb: int = 112, cols: int = 8) -> None:
    imgs = []
    for path in paths[:64]:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        imgs.append(cv2.resize(img, (thumb, thumb), interpolation=cv2.INTER_AREA))
    if not imgs:
        return
    rows = int(np.ceil(len(imgs) / cols))
    canvas = np.full((rows * thumb, cols * thumb, 3), 245, dtype=np.uint8)
    for idx, img in enumerate(imgs):
        row, col = divmod(idx, cols)
        canvas[row * thumb : (row + 1) * thumb, col * thumb : (col + 1) * thumb] = img
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def collect_sources(source_roots: list[Path], cls: str) -> list[Path]:
    paths: list[Path] = []
    for root in source_roots:
        class_dir = root / cls
        if class_dir.exists():
            paths.extend(sorted(class_dir.glob("*.jpg")))
    return paths


def main() -> None:
    args = parse_args()
    args.output = args.output.resolve()
    args.source_roots = [root.resolve() for root in args.source_roots]
    rng = random.Random(args.seed)

    args.output.mkdir(parents=True, exist_ok=True)
    if args.clear:
        clear_dir(args.output)
    (args.output / "previews").mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str | int | float]] = []
    summary: dict[str, dict[str, int]] = {}
    for cls in CLASSES:
        out_dir = args.output / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        source_paths = collect_sources(args.source_roots, cls)
        rng.shuffle(source_paths)
        selected = source_paths[: args.max_sources_per_class]
        output_count = 0
        for src_idx, src in enumerate(selected):
            img = read_image(src, args.image_size)
            if img is None:
                continue
            for variant_idx in range(args.variants_per_source):
                out, meta = transform(img, rng)
                name = f"ai_runtime_{cls}_{src_idx:04d}_v{variant_idx:02d}.jpg"
                dst = out_dir / name
                cv2.imwrite(str(dst), out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                row: dict[str, str | int | float] = {
                    "class": cls,
                    "source": str(src),
                    "output": str(dst.relative_to(args.output)),
                    "variant": variant_idx,
                }
                row.update(meta)
                manifest_rows.append(row)
                output_count += 1
        summary[cls] = {
            "source_candidates": len(source_paths),
            "selected_sources": len(selected),
            "outputs": output_count,
        }
        write_contact_sheet(sorted(out_dir.glob("*.jpg")), args.output / "previews" / f"{cls}_contact_sheet.jpg")

    fieldnames = sorted({key for row in manifest_rows for key in row.keys()})
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary_data = {
        "root": str(args.output),
        "source_roots": [str(root) for root in args.source_roots],
        "classes": summary,
        "variants_per_source": args.variants_per_source,
        "max_sources_per_class": args.max_sources_per_class,
        "image_size": args.image_size,
        "policy": "Derived from already-reviewed AI whole-fruit boosters; runtime degradation only, no new semantics.",
        "usage_note": "Use as a low-ratio hard booster for C/unified face classification; validate before promotion.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    print(json.dumps(summary_data, indent=2))


if __name__ == "__main__":
    main()
