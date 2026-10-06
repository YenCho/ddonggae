import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np


CLASS_SOURCES = {
    "apple": [
        "Apple Braeburn 1",
        "Apple Crimson Snow 1",
        "Apple Pink Lady 1",
        "Apple Red 1",
        "Apple Red 2",
        "Apple Red 3",
        "Apple Red Delicious 1",
        "Apple Red Yellow 1",
        "Apple Red Yellow 2",
    ],
    "banana": [
        "Banana 1",
        "Banana 3",
        "Banana 4",
        "Banana Lady Finger 1",
    ],
    "orange": [
        "Orange 1",
        "Orange 2",
        "Orange 3",
        "orange 4",
    ],
    "pineapple": [
        "Pineapple 1",
        "Pineapple Mini 1",
    ],
}


def hsv_ratios(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    non_white = ~((s < 35) & (v > 190))
    ratios = {
        "red": (((h <= 12) | (h >= 168)) & (s > 40) & (v > 45)),
        "apple_warm": ((((h <= 12) | (h >= 168)) | ((h >= 13) & (h <= 28))) & (s > 35) & (v > 45)),
        "yellow": ((h >= 18) & (h <= 42) & (s > 35) & (v > 55)),
        "orange": ((h >= 4) & (h <= 25) & (s > 55) & (v > 50)),
        "green": ((h >= 35) & (h <= 92) & (s > 25) & (v > 35)),
        "brown": ((h >= 8) & (h <= 26) & (s > 30) & (v > 25) & (v < 150)),
        "non_white": non_white,
    }
    return {name: float(mask.mean()) for name, mask in ratios.items()}


def class_score(class_name, image):
    ratios = hsv_ratios(image)
    red = ratios["red"]
    apple_warm = ratios["apple_warm"]
    yellow = ratios["yellow"]
    orange = ratios["orange"]
    green = ratios["green"]
    brown = ratios["brown"]
    non_white = ratios["non_white"]

    if class_name == "apple":
        score = apple_warm + 0.25 * green - 0.35 * yellow
        passed = apple_warm >= 0.06 and red >= 0.025 and yellow < 0.32 and non_white >= 0.08
    elif class_name == "banana":
        score = yellow + 0.20 * green - 0.45 * red
        passed = yellow >= 0.07 and red < 0.10 and orange < 0.24 and non_white >= 0.08
    elif class_name == "orange":
        score = orange - 0.35 * green - 0.30 * yellow
        passed = orange >= 0.12 and green < 0.10 and non_white >= 0.08
    elif class_name == "pineapple":
        exterior = green + 0.65 * brown + 0.35 * yellow
        score = exterior - 0.45 * red
        passed = exterior >= 0.08 and green >= 0.025 and red < 0.12 and non_white >= 0.08
    else:
        raise ValueError(class_name)
    return score, passed, ratios


def resize_square(image, size):
    if size <= 0:
        return image
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)


def iter_source_images(source, class_names, splits):
    for split in splits:
        split_dir = source / split
        if not split_dir.exists():
            continue
        for class_dir_name in class_names:
            class_dir = split_dir / class_dir_name
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.glob("*.jp*g")):
                yield split, class_dir_name, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="datasets/fruit_textures/_raw/kaggle_fruits360_download/fruits-360_100x100/fruits-360")
    parser.add_argument("--output", type=str, default="datasets/fruit_textures/_sources/fruits360_clean_256")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--max_per_class", type=int, default=1200)
    parser.add_argument("--splits", nargs="+", default=["Training", "Test"])
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    if args.clear and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    debug_rows = []
    for class_name, source_classes in CLASS_SOURCES.items():
        class_output = output / class_name
        if class_output.exists():
            shutil.rmtree(class_output)
        class_output.mkdir(parents=True, exist_ok=True)

        candidates = []
        for split, source_class, path in iter_source_images(source, source_classes, args.splits):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            score, passed, ratios = class_score(class_name, image)
            candidates.append({
                "split": split,
                "source_class": source_class,
                "path": path,
                "score": score,
                "passed": passed,
                "ratios": ratios,
            })

        candidates.sort(key=lambda item: item["score"], reverse=True)
        kept = [item for item in candidates if item["passed"]][:args.max_per_class]
        if not kept:
            kept = candidates[:args.max_per_class]

        for idx, item in enumerate(kept):
            image = cv2.imread(str(item["path"]), cv2.IMREAD_COLOR)
            image = resize_square(image, args.size)
            target = class_output / f"{class_name}_{idx:04d}.jpg"
            cv2.imwrite(str(target), image, [int(cv2.IMWRITE_JPEG_QUALITY), 94])

        kept_paths = {item["path"] for item in kept}
        print(f"{class_name}: kept {len(kept)} / {len(candidates)}")
        for item in candidates:
            row = {
                "class": class_name,
                "source_class": item["source_class"],
                "split": item["split"],
                "source": str(item["path"]),
                "kept": item["path"] in kept_paths,
                "score": f"{item['score']:.6f}",
            }
            row.update({k: f"{v:.6f}" for k, v in item["ratios"].items()})
            debug_rows.append(row)

    if debug_rows:
        fields = ["class", "source_class", "split", "source", "kept", "score", "red", "apple_warm", "yellow", "orange", "green", "brown", "non_white"]
        with open(output / "_selection_debug.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(debug_rows)


if __name__ == "__main__":
    main()
