import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np


CLASSES = ["apple", "banana", "orange", "pineapple"]


def hsv_masks(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    valid = (s > 35) & (v > 45)
    masks = {
        "red": (((h <= 10) | (h >= 168)) & (s > 45) & (v > 45)),
        "apple_red": ((((h <= 12) | (h >= 168)) | ((h >= 13) & (h <= 26))) & (s > 40) & (v > 45)),
        "yellow": ((h >= 18) & (h <= 42) & (s > 35) & (v > 58)),
        "orange": ((h >= 4) & (h <= 24) & (s > 55) & (v > 50)),
        "green": ((h >= 35) & (h <= 88) & (s > 30) & (v > 45)),
        "brown": ((h >= 8) & (h <= 26) & (s > 30) & (v > 25) & (v < 150)),
        "dark": (v < 38),
        "valid": valid,
    }
    return {name: float(mask.mean()) for name, mask in masks.items()}


def score_image(class_name, image):
    ratios = hsv_masks(image)
    dark = ratios["dark"]
    valid = ratios["valid"]
    red = ratios["red"]
    apple_red = ratios["apple_red"]
    yellow = ratios["yellow"]
    orange = ratios["orange"]
    green = ratios["green"]
    brown = ratios["brown"]

    if class_name == "apple":
        target = apple_red + 0.45 * green
        penalty = 0.50 * yellow + 0.35 * orange + 0.25 * dark
        passed = (
            target >= 0.18
            and red >= 0.08
            and apple_red >= 0.09
            and valid >= 0.18
            and yellow < 0.25
            and orange < 0.22
        )
    elif class_name == "banana":
        target = yellow
        penalty = 0.80 * red + 0.25 * dark
        passed = target >= 0.18 and green < 0.22 and red < 0.10 and orange < 0.24
    elif class_name == "orange":
        target = orange
        penalty = 0.50 * green + 0.35 * yellow + 0.25 * dark
        passed = orange >= 0.30 and green < 0.12 and yellow < 0.32
    elif class_name == "pineapple":
        target = green + 0.65 * brown + 0.35 * yellow
        penalty = 1.35 * red + 0.20 * dark
        passed = target >= 0.12 and green >= 0.04 and red < 0.08
    else:
        raise ValueError(class_name)

    score = target - penalty
    return score, passed, ratios


def resize_max_side(image, max_side):
    h, w = image.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return image
    scale = max_side / side
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def collect_candidates(source, class_name):
    candidates = []
    for path in sorted((source / class_name).glob("*.jpg")):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        if min(image.shape[:2]) < 48:
            continue
        score, passed, ratios = score_image(class_name, image)
        candidates.append({
            "path": path,
            "score": score,
            "passed": passed,
            "ratios": ratios,
        })
    return candidates


def write_debug_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["class", "source", "kept", "score", "red", "apple_red", "yellow", "orange", "green", "brown", "dark", "valid"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="datasets/fruit_textures/detection_crops")
    parser.add_argument("--output", type=str, default="datasets/fruit_textures/detection_crops_clean_512")
    parser.add_argument("--max_per_class", type=int, default=800)
    parser.add_argument("--max_side", type=int, default=512)
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    if args.clear and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    debug_rows = []
    for class_name in CLASSES:
        class_out = output / class_name
        if class_out.exists():
            shutil.rmtree(class_out)
        class_out.mkdir(parents=True, exist_ok=True)

        candidates = collect_candidates(source, class_name)
        candidates.sort(key=lambda item: item["score"], reverse=True)
        kept = [item for item in candidates if item["passed"]][:args.max_per_class]

        for idx, item in enumerate(kept):
            image = cv2.imread(str(item["path"]), cv2.IMREAD_COLOR)
            image = resize_max_side(image, args.max_side)
            target = class_out / f"{class_name}_{idx:04d}.jpg"
            cv2.imwrite(str(target), image, [int(cv2.IMWRITE_JPEG_QUALITY), 94])

        kept_paths = {item["path"] for item in kept}
        for item in candidates:
            ratios = item["ratios"]
            debug_rows.append({
                "class": class_name,
                "source": str(item["path"]),
                "kept": item["path"] in kept_paths,
                "score": f"{item['score']:.6f}",
                "red": f"{ratios['red']:.6f}",
                "apple_red": f"{ratios['apple_red']:.6f}",
                "yellow": f"{ratios['yellow']:.6f}",
                "orange": f"{ratios['orange']:.6f}",
                "green": f"{ratios['green']:.6f}",
                "brown": f"{ratios['brown']:.6f}",
                "dark": f"{ratios['dark']:.6f}",
                "valid": f"{ratios['valid']:.6f}",
            })

        print(f"{class_name}: kept={len(kept)} candidates={len(candidates)}")

    write_debug_csv(debug_rows, output / "_debug_scores.csv")
    total = sum(len(list((output / class_name).glob("*.jpg"))) for class_name in CLASSES)
    print(f"total: {total}")
    print(f"output: {output}")


if __name__ == "__main__":
    main()
