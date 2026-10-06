import argparse
import json
import os
import random
import shutil
from pathlib import Path

from tqdm import tqdm


CLASS_NAMES = ["apple", "orange", "banana", "pineapple", "plain"]


def as_posix(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str) -> str:
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


def label_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx, part in enumerate(parts):
        if part == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    raise ValueError(f"image path does not contain images directory: {image_path}")


def collect_images(root: Path, split: str) -> list[Path]:
    image_dir = root / "images" / split
    return sorted(p for p in image_dir.glob("*.jpg") if label_for_image(p).exists())


def write_data_yaml(root: Path) -> None:
    lines = [
        f"path: {as_posix(root)}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for idx, name in enumerate(CLASS_NAMES):
        lines.append(f"  {idx}: {name}")
    (root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def materialize_split(
    jobs: list[tuple[Path, str]],
    output_root: Path,
    split: str,
    mode: str,
) -> dict:
    stats = {"images": 0, "labels": 0, "hardlink": 0, "copy": 0, "copy_fallback": 0, "exists": 0}
    used_names = set()
    for src_image, prefix in tqdm(jobs, desc=f"materialize {split}", unit="img"):
        src_label = label_for_image(src_image)
        stem = f"{prefix}_{src_image.stem}"
        if stem in used_names:
            suffix = 1
            while f"{stem}_{suffix:02d}" in used_names:
                suffix += 1
            stem = f"{stem}_{suffix:02d}"
        used_names.add(stem)
        dst_image = output_root / "images" / split / f"{stem}{src_image.suffix.lower()}"
        dst_label = output_root / "labels" / split / f"{stem}.txt"
        stats[link_or_copy(src_image, dst_image, mode)] += 1
        stats[link_or_copy(src_label, dst_label, mode)] += 1
        stats["images"] += 1
        stats["labels"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize cube-face unified fine-tune mix as normal YOLO image/label folders.")
    parser.add_argument("--base_dataset", type=Path, default=Path("datasets/meta_v2_50000_cube_face_unified_v1"))
    parser.add_argument("--booster_dataset", type=Path, default=Path("datasets/cube_face_unified_booster_wholefruit_plainhard_v1"))
    parser.add_argument("--output_root", type=Path, default=Path("datasets/cube_face_unified_finetune_wholefruit_plainhard_materialized_v1"))
    parser.add_argument("--base_sample_count", type=int, default=22000)
    parser.add_argument("--base_val_count", type=int, default=2500)
    parser.add_argument("--booster_repeat", type=int, default=2)
    parser.add_argument("--mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if args.reset:
        reset_dir(args.output_root)
    for split in ["train", "val"]:
        (args.output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    base_train = collect_images(args.base_dataset, "train")
    base_val = collect_images(args.base_dataset, "val")
    booster_train = collect_images(args.booster_dataset, "train")
    booster_val = collect_images(args.booster_dataset, "val")
    if not base_train or not booster_train:
        raise RuntimeError("base and booster train images are required")

    base_train_sample = rng.sample(base_train, min(args.base_sample_count, len(base_train)))
    base_val_sample = rng.sample(base_val, min(args.base_val_count, len(base_val)))
    train_jobs: list[tuple[Path, str]] = [(p, "base") for p in base_train_sample]
    for repeat_idx in range(max(1, args.booster_repeat)):
        train_jobs.extend((p, f"boost{repeat_idx}") for p in booster_train)
    val_jobs: list[tuple[Path, str]] = [(p, "base") for p in base_val_sample]
    val_jobs.extend((p, "boost") for p in booster_val)
    rng.shuffle(train_jobs)
    rng.shuffle(val_jobs)

    train_stats = materialize_split(train_jobs, args.output_root, "train", args.mode)
    val_stats = materialize_split(val_jobs, args.output_root, "val", args.mode)
    write_data_yaml(args.output_root)
    manifest = {
        "task": "cube_face_unified_materialized_finetune_mix",
        "base_dataset": str(args.base_dataset),
        "booster_dataset": str(args.booster_dataset),
        "output_root": str(args.output_root),
        "base_train_sample": len(base_train_sample),
        "base_val_sample": len(base_val_sample),
        "booster_train_images": len(booster_train),
        "booster_val_images": len(booster_val),
        "booster_repeat": args.booster_repeat,
        "mode": args.mode,
        "train": train_stats,
        "val": val_stats,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
