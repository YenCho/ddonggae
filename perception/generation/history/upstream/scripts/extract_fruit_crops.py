import argparse
import shutil
from pathlib import Path

import cv2


SOURCE_ID_TO_NAME = {
    0: "apple",
    1: "banana",
    3: "orange",
    4: "pineapple",
}


def crop_from_yolo(image, label, pad_ratio):
    h, w = image.shape[:2]
    cls, xc, yc, bw, bh = label
    x1 = int((xc - bw / 2) * w)
    y1 = int((yc - bh / 2) * h)
    x2 = int((xc + bw / 2) * w)
    y2 = int((yc + bh / 2) * h)
    pad = int(max(x2 - x1, y2 - y1) * pad_ratio)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def read_labels(path):
    labels = []
    if not path.exists():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        if cls not in SOURCE_ID_TO_NAME:
            continue
        labels.append((cls, *[float(v) for v in parts[1:]]))
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="datasets/fruit_textures/lightly_fruits_detection")
    parser.add_argument("--output", type=str, default="datasets/fruit_textures/detection_crops")
    parser.add_argument("--max_per_class", type=int, default=1000)
    parser.add_argument("--min_size", type=int, default=48)
    parser.add_argument("--pad_ratio", type=float, default=0.18)
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    if args.clear and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    counts = {name: len(list((output / name).glob("*.jpg"))) if (output / name).exists() else 0 for name in SOURCE_ID_TO_NAME.values()}
    for name in counts:
        (output / name).mkdir(parents=True, exist_ok=True)

    for split in ["train", "valid", "test"]:
        image_dir = source / split / "images"
        label_dir = source / split / "labels"
        for image_path in sorted(image_dir.glob("*.*")):
            labels = read_labels(label_dir / f"{image_path.stem}.txt")
            if not labels:
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            for label in labels:
                class_name = SOURCE_ID_TO_NAME[label[0]]
                if counts[class_name] >= args.max_per_class:
                    continue
                crop = crop_from_yolo(image, label, args.pad_ratio)
                if crop is None or min(crop.shape[:2]) < args.min_size:
                    continue
                target = output / class_name / f"{class_name}_{counts[class_name]:04d}.jpg"
                cv2.imwrite(str(target), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
                counts[class_name] += 1
        if all(v >= args.max_per_class for v in counts.values()):
            break

    total = sum(counts.values())
    for name, count in counts.items():
        print(f"{name}: {count}")
    print(f"total: {total}")
    print(f"output: {output}")


if __name__ == "__main__":
    main()
