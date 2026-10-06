import argparse
import concurrent.futures
import shutil
import time
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


def motion_offsets(size: int, angle: int):
    if size % 2 == 0:
        size += 1
    center = size // 2
    if angle == 0:
        return [(0, i - center) for i in range(size)]
    if angle == 90:
        return [(i - center, 0) for i in range(size)]
    if angle == 45:
        return [(i - center, i - center) for i in range(size)]
    if angle == 135:
        return [(i - center, center - i) for i in range(size)]
    raise ValueError(f"Unsupported angle: {angle}")


def apply_motion_blur(image_path: Path, output_path: Path, size: int, angle: int) -> None:
    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image, dtype=np.float32)
    pad = size // 2
    padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(arr)
    offsets = motion_offsets(size, angle)

    for dy, dx in offsets:
        y0 = pad + dy
        x0 = pad + dx
        out += padded[y0 : y0 + arr.shape[0], x0 : x0 + arr.shape[1], :]

    out /= len(offsets)
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB").save(output_path, quality=92)


def copy_split(source: Path, output: Path, split: str) -> None:
    for kind, pattern in (("images", "*.jpg"), ("labels", "*.txt")):
        src_dir = source / kind / split
        dst_dir = output / kind / split
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in src_dir.glob(pattern):
            dst = dst_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)


def write_yaml(path: Path, dataset_dir: Path) -> None:
    names = "\n".join(f"  {idx}: {name}" for idx, name in NAMES.items())
    path.write_text(
        f"path: {dataset_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        f"{names}\n",
        encoding="utf-8",
    )


def blur_one(task):
    idx, image_path, source_labels, out_images, out_labels, sizes, angles = task
    angle = angles[idx % len(angles)]
    size = sizes[(idx // len(angles)) % len(sizes)]
    stem = image_path.stem
    suffix = f"_motionblur_k{size}_a{angle}"
    output_image = out_images / f"{stem}{suffix}.jpg"
    output_label = out_labels / f"{stem}{suffix}.txt"

    if not output_image.exists():
        apply_motion_blur(image_path, output_image, size, angle)
    if not output_label.exists():
        src_label = source_labels / f"{stem}.txt"
        if src_label.exists():
            shutil.copy2(src_label, output_label)
    return output_image.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Create clean+motion-blur YOLO segmentation augmentation dataset.")
    parser.add_argument("--source", type=Path, default=Path("datasets/yolo26_seg_100000_ideal_debug"))
    parser.add_argument("--output", type=Path, default=Path("datasets/yolo26_seg_100000_ideal_motionblur_aug"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Optional train image limit for testing. 0 means all.")
    args = parser.parse_args()

    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    print(f"source: {args.source.resolve()}", flush=True)
    print(f"output: {output.resolve()}", flush=True)
    print("copying clean train/val/test jpg/txt files...", flush=True)
    for split in ("train", "val", "test"):
        copy_split(args.source, output, split)

    out_images = output / "images" / "train"
    out_labels = output / "labels" / "train"
    source_images = args.source / "images" / "train"
    source_labels = args.source / "labels" / "train"
    images = sorted(source_images.glob("*.jpg"))
    if args.limit > 0:
        images = images[: args.limit]

    sizes = [9, 15, 21]
    angles = [0, 90, 45, 135]
    tasks = [
        (idx, image_path, source_labels, out_images, out_labels, sizes, angles)
        for idx, image_path in enumerate(images)
    ]

    print(f"creating motion-blur train copies: {len(tasks)} images", flush=True)
    started = time.time()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(blur_one, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            future.result()
            done += 1
            if done % 1000 == 0 or done == len(tasks):
                elapsed = max(time.time() - started, 1)
                rate = done / elapsed
                remaining = (len(tasks) - done) / max(rate, 0.1)
                print(f"blurred {done}/{len(tasks)} ({rate:.1f}/s, eta {remaining/60:.1f}m)", flush=True)

    write_yaml(output / "data.yaml", output)
    print("done", flush=True)
    print(f"train images: {len(list((output / 'images' / 'train').glob('*.jpg')))}", flush=True)
    print(f"train labels: {len(list((output / 'labels' / 'train').glob('*.txt')))}", flush=True)
    print(f"data yaml: {(output / 'data.yaml').resolve()}", flush=True)


if __name__ == "__main__":
    main()
