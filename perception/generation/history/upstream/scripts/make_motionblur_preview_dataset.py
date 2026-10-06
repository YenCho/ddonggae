import argparse
import math
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


NAMES = {
    0: "banana",
    1: "orange",
    2: "pineapple",
    3: "apple",
    4: "cube",
    5: "octahedron",
    6: "dodecahedron",
    7: "icosahedron",
}


def motion_kernel(size: int, angle_deg: float) -> np.ndarray:
    if size % 2 == 0:
        size += 1
    kernel = np.zeros((size, size), dtype=np.float32)
    center = (size - 1) / 2.0
    angle = math.radians(angle_deg)
    dx = math.cos(angle)
    dy = math.sin(angle)

    for i in range(size):
        t = i - center
        x = int(round(center + dx * t))
        y = int(round(center + dy * t))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1.0
    total = kernel.sum()
    if total == 0:
        kernel[size // 2, :] = 1.0
        total = kernel.sum()
    return kernel / total


def apply_kernel_rgb(image: Image.Image, kernel: np.ndarray) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    size = kernel.shape[0]
    pad = size // 2
    padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(arr)
    ys, xs = np.nonzero(kernel)
    weights = kernel[ys, xs]
    for y, x, weight in zip(ys, xs, weights):
        out += padded[y : y + arr.shape[0], x : x + arr.shape[1], :] * weight
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def resize_for_contact(image: Image.Image, width: int = 220) -> Image.Image:
    ratio = width / image.width
    height = int(round(image.height * ratio))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def write_yaml(path: Path, dataset_dir: Path) -> None:
    names = "\n".join(f"  {idx}: {name}" for idx, name in NAMES.items())
    text = (
        f"path: {dataset_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/train\n"
        "test: images/train\n"
        "names:\n"
        f"{names}\n"
    )
    path.write_text(text, encoding="utf-8")


def make_contact_sheet(rows, dst: Path) -> None:
    if not rows:
        return
    cell_w = 220
    gap = 10
    label_h = 24
    resized_rows = []
    for label, images in rows:
        resized = [resize_for_contact(img, cell_w) for img in images]
        resized_rows.append((label, resized))

    row_h = max(max(img.height for img in images) for _, images in resized_rows) + label_h
    cols = max(len(images) for _, images in resized_rows)
    sheet = Image.new("RGB", (cols * cell_w + (cols + 1) * gap, len(rows) * row_h + gap), "white")

    for row_idx, (label, images) in enumerate(resized_rows):
        y = gap + row_idx * row_h + label_h
        for col_idx, img in enumerate(images):
            x = gap + col_idx * (cell_w + gap)
            sheet.paste(img, (x, y))
    sheet.save(dst, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a 100-image YOLO motion-blur preview dataset.")
    parser.add_argument("--source", type=Path, default=Path("datasets/yolo26_seg_100000_ideal_debug"))
    parser.add_argument("--output", type=Path, default=Path("datasets/yolo26_seg_100_motionblur_preview"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2600)
    args = parser.parse_args()

    src_images = args.source / "images" / "train"
    src_labels = args.source / "labels" / "train"
    out_images = args.output / "images" / "train"
    out_labels = args.output / "labels" / "train"
    preview_dir = args.output / "preview"

    if args.output.exists():
        shutil.rmtree(args.output)
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(src_images.glob("*.jpg"))
    rng = random.Random(args.seed)
    chosen = rng.sample(images, min(args.count, len(images)))

    angles = [0, 90, 45, 135]
    sizes = [9, 15, 21]
    contact_rows = []

    for idx, img_path in enumerate(chosen):
        label_path = src_labels / f"{img_path.stem}.txt"
        angle = angles[idx % len(angles)]
        size = sizes[(idx // len(angles)) % len(sizes)]
        image = Image.open(img_path)
        blurred = apply_kernel_rgb(image, motion_kernel(size, angle))

        out_name = f"{img_path.stem}_motionblur_k{size}_a{angle}.jpg"
        out_label = f"{Path(out_name).stem}.txt"
        blurred.save(out_images / out_name, quality=92)
        if label_path.exists():
            shutil.copy2(label_path, out_labels / out_label)

        if len(contact_rows) < 12:
            mild = apply_kernel_rgb(image, motion_kernel(9, angle))
            medium = apply_kernel_rgb(image, motion_kernel(15, angle))
            strong = apply_kernel_rgb(image, motion_kernel(25, angle))
            contact_rows.append((img_path.stem, [image, mild, medium, strong]))

    write_yaml(args.output / "data.yaml", args.output.resolve())
    make_contact_sheet(contact_rows[:6], preview_dir / "motionblur_compare_01.jpg")
    make_contact_sheet(contact_rows[6:12], preview_dir / "motionblur_compare_02.jpg")

    print(f"created: {args.output.resolve()}")
    print(f"images: {len(list(out_images.glob('*.jpg')))}")
    print(f"labels: {len(list(out_labels.glob('*.txt')))}")
    print(f"preview: {preview_dir.resolve()}")


if __name__ == "__main__":
    main()
