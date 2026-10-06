from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = ["apple", "orange", "banana", "pineapple", "plain"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

EXPOSURE_PROFILES = {
    "moderate": {
        "mild_global": {"gain": 1.16, "bias": 4.0, "gamma": 0.98, "desat_high": 0.05, "glare": False},
        "bright_global": {"gain": 1.28, "bias": 8.0, "gamma": 0.96, "desat_high": 0.10, "glare": False},
        "soft_top_glare": {"gain": 1.20, "bias": 6.0, "gamma": 0.97, "desat_high": 0.08, "glare": True},
        "warm_bright": {
            "gain": 1.24,
            "bias": 7.0,
            "gamma": 0.96,
            "desat_high": 0.10,
            "glare": False,
            "warm": True,
        },
    },
    "strong": {
        "strong_global": {"gain": 1.55, "bias": 18.0, "gamma": 0.92, "desat_high": 0.20, "glare": False},
        "clip_high": {"gain": 1.85, "bias": 30.0, "gamma": 0.88, "desat_high": 0.42, "glare": False},
        "top_glare": {"gain": 1.42, "bias": 14.0, "gamma": 0.94, "desat_high": 0.25, "glare": True},
        "warm_clip": {"gain": 1.68, "bias": 22.0, "gamma": 0.90, "desat_high": 0.35, "glare": False, "warm": True},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 100-image evaluation-only exposure stress probe from the original cube-face unified dataset."
    )
    parser.add_argument("--source", type=Path, default=Path("datasets/meta_v2_50000_cube_face_unified_v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--preview-count", type=int, default=80)
    parser.add_argument("--profile", choices=sorted(EXPOSURE_PROFILES), default="strong")
    return parser.parse_args()


def label_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx, part in enumerate(parts):
        if part == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    raise ValueError(f"image path does not contain images directory: {image_path}")


def read_label_classes(label_path: Path) -> list[str]:
    classes: list[str] = []
    if not label_path.exists():
        return classes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        cls_id = int(float(parts[0]))
        if 0 <= cls_id < len(CLASS_NAMES):
            classes.append(CLASS_NAMES[cls_id])
    return sorted(set(classes), key=CLASS_NAMES.index)


def collect_images(root: Path, split: str) -> list[Path]:
    image_dir = root / "images" / split
    return sorted(
        p for p in image_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS and label_for_image(p).exists()
    )


def choose_balanced(images: list[Path], count: int, rng: random.Random) -> list[Path]:
    by_class: dict[str, list[Path]] = defaultdict(list)
    for path in images:
        classes = read_label_classes(label_for_image(path))
        for cls in classes or ["plain"]:
            by_class[cls].append(path)

    chosen: list[Path] = []
    used: set[Path] = set()
    per_class_target = max(1, count // len(CLASS_NAMES))
    for cls in CLASS_NAMES:
        pool = by_class.get(cls, [])[:]
        rng.shuffle(pool)
        for path in pool:
            if path in used:
                continue
            chosen.append(path)
            used.add(path)
            if sum(1 for p in chosen if cls in read_label_classes(label_for_image(p))) >= per_class_target:
                break

    leftovers = [p for p in images if p not in used]
    rng.shuffle(leftovers)
    for path in leftovers:
        if len(chosen) >= count:
            break
        chosen.append(path)
        used.add(path)
    return chosen[:count]


def apply_exposure(image: np.ndarray, variant: str, rng: random.Random, profile: str) -> np.ndarray:
    params = EXPOSURE_PROFILES[profile][variant]
    out = image.astype(np.float32)

    if params.get("warm"):
        out[..., 2] *= 1.05
        out[..., 1] *= 1.01
        out[..., 0] *= 0.96

    out = out * float(params["gain"]) + float(params["bias"])
    out = np.clip(out, 0, 255) / 255.0
    out = np.power(out, float(params["gamma"]))
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    high = hsv[..., 2] > 215
    if np.any(high):
        hsv[..., 1][high] *= 1.0 - float(params["desat_high"])
        hsv[..., 2][high] = np.clip(hsv[..., 2][high] * 1.04 + 8.0, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if params.get("glare"):
        h, w = out.shape[:2]
        cx = int(w * rng.uniform(0.40, 0.66))
        cy = int(h * rng.uniform(0.14, 0.30))
        rx = int(w * rng.uniform(0.20, 0.38))
        ry = int(h * rng.uniform(0.045, 0.10))
        yy, xx = np.mgrid[0:h, 0:w]
        mask = ((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2
        alpha = np.clip(1.0 - mask, 0, 1) ** 1.8
        glare_range = (0.16, 0.30) if profile == "moderate" else (0.28, 0.48)
        alpha *= rng.uniform(*glare_range)
        white = np.full_like(out, 255)
        out = (out.astype(np.float32) * (1 - alpha[..., None]) + white.astype(np.float32) * alpha[..., None]).astype(
            np.uint8
        )

    # Tiny JPEG round-trip to mimic webcam capture after highlight clipping.
    quality = rng.randint(82, 95) if profile == "moderate" else rng.randint(70, 92)
    ok, encoded = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if ok:
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is not None:
            out = decoded
    return out


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_data_yaml(root: Path) -> None:
    lines = [
        f"path: {root.resolve().as_posix()}",
        "train: images/val",
        "val: images/val",
        "names:",
    ]
    for idx, name in enumerate(CLASS_NAMES):
        lines.append(f"  {idx}: {name}")
    (root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_contact_sheet(records: list[dict], output: Path, max_items: int, cell: int = 160) -> None:
    chosen = records[:max_items]
    if not chosen:
        return
    cols = 5
    rows = int(np.ceil(len(chosen) / cols))
    header = 42
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
        text1 = f"{rec['variant']} {'/'.join(rec['classes'])}"[:24]
        text2 = Path(rec["source"]).name[:24]
        cv2.putText(sheet, text1, (x + 3, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(sheet, text2, (x + 3, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (70, 70, 70), 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    reset_dir(args.output)
    image_out = args.output / "images" / "val"
    label_out = args.output / "labels" / "val"
    preview_out = args.output / "preview"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    preview_out.mkdir(parents=True, exist_ok=True)

    source_images = collect_images(args.source, args.split)
    selected = choose_balanced(source_images, args.count, rng)

    records: list[dict] = []
    variant_names = list(EXPOSURE_PROFILES[args.profile])
    for idx, src in enumerate(selected):
        variant = variant_names[idx % len(variant_names)]
        # Shuffle variant assignment while keeping counts close.
        variant = rng.choice(variant_names) if idx >= len(variant_names) else variant
        label = label_for_image(src)
        classes = read_label_classes(label)
        image = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if image is None:
            continue
        exposed = apply_exposure(image, variant, rng, args.profile)
        stem = f"{idx:04d}_{variant}_{src.stem}"
        dst_img = image_out / f"{stem}.jpg"
        dst_label = label_out / f"{stem}.txt"
        cv2.imwrite(str(dst_img), exposed, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        shutil.copy2(label, dst_label)
        records.append(
            {
                "idx": idx,
                "variant": variant,
                "classes": classes,
                "source": str(src),
                "image": str(dst_img),
                "label": str(dst_label),
            }
        )

    write_data_yaml(args.output)
    (args.output / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "variant", "classes", "source", "image", "label"])
        writer.writeheader()
        for rec in records:
            row = dict(rec)
            row["classes"] = "|".join(row["classes"])
            writer.writerow(row)

    make_contact_sheet(records, preview_out / f"{args.profile}_exposure_contact_sheet.jpg", args.preview_count)
    class_counts = Counter(cls for rec in records for cls in rec["classes"])
    variant_counts = Counter(rec["variant"] for rec in records)
    summary = {
        "source": str(args.source),
        "output": str(args.output),
        "split": args.split,
        "profile": args.profile,
        "count": len(records),
        "class_counts": dict(class_counts),
        "variant_counts": dict(variant_counts),
        "policy": f"evaluation-only {args.profile} exposure stress set; source labels copied unchanged",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "README.md").write_text(
        f"# {args.profile.title()} Exposure Probe\n\n"
        "Evaluation-only 100-image probe derived from the original meta_v2 cube-face unified validation split.\n"
        f"Images are copied through {args.profile} exposure / highlight / optional top glare transforms; labels are unchanged.\n"
        "Do not train on this set unless it is later rebuilt through the full generation policy and manually previewed.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
