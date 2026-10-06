from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


CLASS_NAMES = ["apple", "orange", "banana", "pineapple", "plain"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
DEFAULT_SEVERITY_WEIGHTS = {"easy": 0.50, "normal": 0.35, "hard": 0.15}

# OpenCV hue is 0-179, so dh=5 is roughly a 10 degree hue move.
# Keep H bounded; use S/V to carry most of the camera/lighting drift.
HSV_VARIANTS = {
    "hsv_easy_warm": {"severity": "easy", "dh": -2, "sat": 1.07, "val": 1.06},
    "hsv_easy_cool": {"severity": "easy", "dh": 2, "sat": 0.92, "val": 0.94},
    "hsv_easy_dim": {"severity": "easy", "dh": -2, "sat": 0.92, "val": 0.90},
    "hsv_easy_bright": {"severity": "easy", "dh": 2, "sat": 1.08, "val": 1.08},
    "hsv_normal_red_orange": {"severity": "normal", "dh": -4, "sat": 1.18, "val": 1.10},
    "hsv_normal_low_sat_warm": {"severity": "normal", "dh": -4, "sat": 0.74, "val": 1.08},
    "hsv_normal_cool_dim": {"severity": "normal", "dh": 4, "sat": 0.78, "val": 0.82},
    "hsv_normal_contrast_dim": {"severity": "normal", "dh": 3, "sat": 1.20, "val": 0.86},
    "hsv_hard_red_orange": {"severity": "hard", "dh": -7, "sat": 1.28, "val": 1.16},
    "hsv_hard_warm_low_sat": {"severity": "hard", "dh": -7, "sat": 0.60, "val": 1.14},
    "hsv_hard_cool_dim": {"severity": "hard", "dh": 7, "sat": 0.62, "val": 0.72},
    "hsv_hard_dark_warm": {"severity": "hard", "dh": -6, "sat": 0.72, "val": 0.68},
}

CLASS_VARIANT_POOLS = {
    "orange": {
        "easy": ["hsv_easy_warm", "hsv_easy_bright", "hsv_easy_dim"],
        "normal": ["hsv_normal_red_orange", "hsv_normal_low_sat_warm", "hsv_normal_contrast_dim"],
        "hard": ["hsv_hard_red_orange", "hsv_hard_warm_low_sat", "hsv_hard_dark_warm"],
    },
    "banana_pineapple": {
        "easy": ["hsv_easy_warm", "hsv_easy_cool", "hsv_easy_dim"],
        "normal": ["hsv_normal_low_sat_warm", "hsv_normal_cool_dim", "hsv_normal_contrast_dim"],
        "hard": ["hsv_hard_warm_low_sat", "hsv_hard_cool_dim", "hsv_hard_dark_warm"],
    },
    "apple": {
        "easy": ["hsv_easy_warm", "hsv_easy_cool", "hsv_easy_bright"],
        "normal": ["hsv_normal_red_orange", "hsv_normal_contrast_dim", "hsv_normal_cool_dim"],
        "hard": ["hsv_hard_red_orange", "hsv_hard_dark_warm", "hsv_hard_cool_dim"],
    },
    "plain": {
        "easy": ["hsv_easy_warm", "hsv_easy_cool", "hsv_easy_dim"],
        "normal": ["hsv_normal_low_sat_warm", "hsv_normal_cool_dim"],
        "hard": ["hsv_hard_cool_dim", "hsv_hard_dark_warm"],
    },
}


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def label_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx, part in enumerate(parts):
        if part == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    raise ValueError(f"image path does not contain images directory: {image_path}")


def collect_images(root: Path, split: str) -> list[Path]:
    image_dir = root / "images" / split
    return sorted(
        p for p in image_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS and label_for_image(p).exists()
    )


def read_label_classes(label_path: Path) -> set[str]:
    classes: set[str] = set()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        cls_id = int(float(parts[0]))
        if 0 <= cls_id < len(CLASS_NAMES):
            classes.add(CLASS_NAMES[cls_id])
    return classes


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, value) for value in weights.values())
    if total <= 0.0:
        raise ValueError("at least one severity ratio must be positive")
    return {key: max(0.0, value) / total for key, value in weights.items()}


def choose_variant(classes: set[str], rng: random.Random, severity_weights: dict[str, float]) -> str:
    severities = list(severity_weights.keys())
    weights = [severity_weights[key] for key in severities]
    severity = rng.choices(severities, weights=weights, k=1)[0]
    if "orange" in classes:
        group = "orange"
    elif "banana" in classes or "pineapple" in classes:
        group = "banana_pineapple"
    elif "apple" in classes:
        group = "apple"
    else:
        group = "plain"
    return rng.choice(CLASS_VARIANT_POOLS[group][severity])


def shift_hsv(image: np.ndarray, *, dh: int = 0, sat: float = 1.0, val: float = 1.0) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + dh) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * val, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_variant(image: np.ndarray, variant: str) -> np.ndarray:
    params = HSV_VARIANTS.get(variant)
    if params is None:
        raise ValueError(f"unknown variant: {variant}")
    return shift_hsv(image, dh=params["dh"], sat=params["sat"], val=params["val"])


def link_or_copy_label(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy_fallback"


def write_data_yaml(root: Path) -> None:
    lines = [
        f"path: {root.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for idx, name in enumerate(CLASS_NAMES):
        lines.append(f"  {idx}: {name}")
    (root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_contact_sheet(records: list[dict], output: Path, max_items: int, cell: int = 144) -> None:
    if not records:
        return
    chosen = records[:]
    random.Random(19).shuffle(chosen)
    chosen = chosen[:max_items]
    cols = 8
    rows = int(np.ceil(len(chosen) / cols))
    header = 22
    sheet = np.full((rows * (cell + header), cols * cell, 3), 245, dtype=np.uint8)
    for idx, rec in enumerate(chosen):
        image = cv2.imread(rec["image"], cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.resize(image, (cell, cell), interpolation=cv2.INTER_AREA)
        row, col = divmod(idx, cols)
        x = col * cell
        y = row * (cell + header)
        sheet[y + header : y + header + cell, x : x + cell] = image
        text = f"{rec['variant']} {'/'.join(rec['classes'])}"[:30]
        cv2.putText(sheet, text, (x + 3, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def write_records_csv(records: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["split", "variant", "severity", "dh", "sat", "val", "classes", "source", "image", "label"]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "split": rec["split"],
                    "variant": rec["variant"],
                    "severity": rec["severity"],
                    "dh": rec["dh"],
                    "sat": rec["sat"],
                    "val": rec["val"],
                    "classes": "|".join(rec["classes"]),
                    "source": rec["source"],
                    "image": rec["image"],
                    "label": rec["label"],
                }
            )


def materialize_split(
    source_root: Path,
    output_root: Path,
    split: str,
    count: int,
    rng: random.Random,
    label_mode: str,
    severity_weights: dict[str, float],
) -> tuple[list[dict], Counter]:
    images = collect_images(source_root, split)
    if not images:
        raise RuntimeError(f"no source images for split={split}: {source_root}")
    sample = rng.sample(images, min(count, len(images)))
    records: list[dict] = []
    stats: Counter = Counter()
    for idx, src_image in enumerate(tqdm(sample, desc=f"color shift {split}", unit="img")):
        src_label = label_for_image(src_image)
        classes = sorted(read_label_classes(src_label))
        variant = choose_variant(set(classes), rng, severity_weights)
        variant_params = HSV_VARIANTS[variant]
        image = cv2.imread(str(src_image), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image: {src_image}")
        boosted = apply_variant(image, variant)
        stem = f"{idx:06d}_{variant}_{src_image.stem}"
        dst_image = output_root / "images" / split / f"{stem}.jpg"
        dst_label = output_root / "labels" / split / f"{stem}.txt"
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst_image), boosted, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        link_or_copy_label(src_label, dst_label, label_mode)
        stats[f"{split}_{variant}"] += 1
        stats[f"{split}_{variant_params['severity']}"] += 1
        for class_name in classes:
            stats[f"{split}_{class_name}"] += 1
        records.append(
            {
                "split": split,
                "source": str(src_image),
                "image": str(dst_image),
                "label": str(dst_label),
                "variant": variant,
                "severity": variant_params["severity"],
                "dh": variant_params["dh"],
                "sat": variant_params["sat"],
                "val": variant_params["val"],
                "classes": classes,
            }
        )
    return records, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a color/lighting booster from existing Meta V2 cube-face crops.")
    parser.add_argument("--source_root", type=Path, default=Path("datasets/meta_v2_50000_cube_face_unified_v1"))
    parser.add_argument("--output_root", type=Path, default=Path("datasets/cube_face_unified_booster_meta_v2_color_shift_v1"))
    parser.add_argument("--train_count", type=int, default=12000)
    parser.add_argument("--val_count", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--preview_count", type=int, default=160)
    parser.add_argument("--label_mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--easy_ratio", type=float, default=DEFAULT_SEVERITY_WEIGHTS["easy"])
    parser.add_argument("--normal_ratio", type=float, default=DEFAULT_SEVERITY_WEIGHTS["normal"])
    parser.add_argument("--hard_ratio", type=float, default=DEFAULT_SEVERITY_WEIGHTS["hard"])
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reset:
        reset_dir(args.output_root)
    for split in ["train", "val"]:
        (args.output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    severity_weights = normalize_weights(
        {"easy": args.easy_ratio, "normal": args.normal_ratio, "hard": args.hard_ratio}
    )
    all_records: list[dict] = []
    stats: Counter = Counter()
    for split, count in [("train", args.train_count), ("val", args.val_count)]:
        records, split_stats = materialize_split(
            args.source_root, args.output_root, split, count, rng, args.label_mode, severity_weights
        )
        all_records.extend(records)
        stats.update(split_stats)

    write_data_yaml(args.output_root)
    write_records_csv(all_records, args.output_root / "records.csv")
    preview = args.output_root / "previews" / "color_shift_contact_sheet.jpg"
    make_contact_sheet(all_records, preview, args.preview_count)
    manifest = {
        "task": "cube_face_unified_meta_v2_color_shift_booster",
        "policy": "Derived only from existing Meta V2 cube-face unified crops. No geometric/icon synthetic cube is generated.",
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "train_count": args.train_count,
        "val_count": args.val_count,
        "class_names": CLASS_NAMES,
        "severity_weights": severity_weights,
        "hsv_variants": HSV_VARIANTS,
        "variant_policy": {
            "orange": "HSV-only weak-to-hard red/orange color drift while preserving original crop geometry and labels",
            "banana_pineapple": "HSV-only warm/cool/dim variation without synthetic geometry or severe blur",
            "plain": "HSV-only lighting/color-temperature variation to probe fruit-vs-plain robustness",
        },
        "args": vars(args) | {"source_root": str(args.source_root), "output_root": str(args.output_root)},
        "stats": dict(stats),
        "preview": str(preview),
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
