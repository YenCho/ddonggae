import argparse
import csv
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


CLASSES = ["apple", "banana", "orange", "pineapple"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def iter_images(root, class_name):
    class_dir = Path(root) / class_name
    if not class_dir.exists():
        return []
    return sorted(path for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)


def read_image_with_alpha(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        alpha = np.full(image.shape[:2] + (1,), 255, dtype=np.uint8)
        image = np.concatenate([image, alpha], axis=2)
    return image


def composite_on_background(image, background):
    bgr = image[..., :3].astype(np.float32)
    alpha = image[..., 3:4].astype(np.float32) / 255.0
    bg = np.full_like(bgr, background, dtype=np.float32)
    return np.clip(bgr * alpha + bg * (1.0 - alpha), 0, 255).astype(np.uint8)


def square_resize(image, size, background):
    bgr = composite_on_background(image, background)
    h, w = bgr.shape[:2]
    scale = (size * 0.92) / max(h, w)
    resized = cv2.resize(
        bgr,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
    )

    canvas = np.full((size, size, 3), background, dtype=np.uint8)
    y0 = (size - resized.shape[0]) // 2
    x0 = (size - resized.shape[1]) // 2
    canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    return canvas


def choose_paths(source_sets, total, rng):
    selected = []
    available = [item for item in source_sets if item["paths"]]
    requested_non_lab = 0
    for item in available:
        if item["label"] == "lab":
            continue
        count = int(round(total * item["ratio"]))
        requested_non_lab += count
        selected.extend((item["label"], path) for path in rng.choices(item["paths"], k=count))

    lab = next((item for item in available if item["label"] == "lab"), None)
    lab_count = max(0, total - requested_non_lab)
    if lab and lab["paths"]:
        selected.extend(("lab", path) for path in rng.choices(lab["paths"], k=lab_count))

    fallback = []
    for item in available:
        fallback.extend((item["label"], path) for path in item["paths"])
    while len(selected) < total:
        if not fallback:
            break
        selected.append(rng.choice(fallback))

    rng.shuffle(selected)
    return selected[:total]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lab", type=str, default="datasets/fruit_textures/_sources/fruits360_clean_256")
    parser.add_argument("--real", type=str, default="datasets/fruit_textures/_sources/original_color_filtered_512")
    parser.add_argument("--fruitseg30", type=str, default="datasets/fruit_textures/_sources/fruitseg30_cutouts_512")
    parser.add_argument("--output", type=str, default="datasets/fruit_textures/final_fruits36065_original25_fruitseg30_10")
    parser.add_argument("--total_per_class", type=int, default=1600)
    parser.add_argument("--real_ratio", type=float, default=0.25)
    parser.add_argument("--fruitseg30_ratio", type=float, default=0.10)
    parser.add_argument("--real_label", type=str, default="original_filtered")
    parser.add_argument("--fruitseg30_label", type=str, default="fruitseg30")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    lab_root = Path(args.lab)
    real_root = Path(args.real)
    fruitseg30_root = Path(args.fruitseg30)
    output = Path(args.output)

    if args.clear and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    for root in [real_root, fruitseg30_root, output]:
        for class_name in CLASSES:
            (root / class_name).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    rows = []
    for class_name in CLASSES:
        class_out = output / class_name
        if class_out.exists():
            shutil.rmtree(class_out)
        class_out.mkdir(parents=True, exist_ok=True)

        lab_paths = iter_images(lab_root, class_name)
        real_paths = iter_images(real_root, class_name)
        fruitseg30_paths = iter_images(fruitseg30_root, class_name)
        selected = choose_paths(
            [
                {"label": "lab", "paths": lab_paths, "ratio": 0.0},
                {"label": args.real_label, "paths": real_paths, "ratio": args.real_ratio},
                {"label": args.fruitseg30_label, "paths": fruitseg30_paths, "ratio": args.fruitseg30_ratio},
            ],
            args.total_per_class,
            rng,
        )

        written = {"lab": 0, args.real_label: 0, args.fruitseg30_label: 0, "failed": 0}
        for idx, (source_type, source_path) in enumerate(selected):
            image = read_image_with_alpha(source_path)
            if image is None:
                written["failed"] += 1
                continue
            background = rng.randint(238, 250)
            square = square_resize(image, args.size, background)
            target = class_out / f"{class_name}_{idx:04d}_{source_type}.jpg"
            cv2.imwrite(str(target), square, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
            written[source_type] += 1
            rows.append({
                "class": class_name,
                "output": str(target),
                "source_type": source_type,
                "source": str(source_path),
            })

        print(
            f"{class_name}: {written['lab']} lab, "
            f"{written[args.real_label]} {args.real_label}, "
            f"{written[args.fruitseg30_label]} {args.fruitseg30_label}, "
            f"{written['failed']} failed -> {class_out}"
        )

    with open(output / "_manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "output", "source_type", "source"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"manifest: {output / '_manifest.csv'}")
    print(f"real cutout input folder: {real_root}")
    print(f"FruitSeg30 cutout input folder: {fruitseg30_root}")


if __name__ == "__main__":
    main()
