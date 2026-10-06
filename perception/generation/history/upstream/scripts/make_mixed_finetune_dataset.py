import argparse
import json
import random
import shutil
import os
from collections import Counter
from pathlib import Path

NAMES = [
    "banana",
    "orange",
    "pineapple",
    "apple",
    "cube",
    "octahedron",
    "dodecahedron",
    "icosahedron",
]


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def copy_or_symlink(src, dst, mode, resume=False):
    ensure_dir(dst.parent)
    if resume and dst.exists():
        return
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        try:
            os.symlink(src.resolve(), dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def collect_stems(dataset, split):
    image_dir = Path(dataset) / "images" / split
    return sorted(path.stem for path in image_dir.glob("*.jpg"))


def read_label_counts(label_path):
    counts = Counter()
    if not label_path.exists():
        return counts
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            cls = int(float(parts[0]))
        except ValueError:
            continue
        counts[cls] += 1
    return counts


def copy_record(src_dataset, src_split, stem, out_root, out_stem, mode, resume):
    src_dataset = Path(src_dataset)
    for group, suffix in (("images", ".jpg"), ("labels", ".txt"), ("_meta", ".json")):
        src = src_dataset / group / src_split / f"{stem}{suffix}"
        if not src.exists():
            continue
        dst = out_root / group / "train" / f"{out_stem}{suffix}"
        copy_or_symlink(src, dst, mode, resume=resume)


def write_data_yaml(out_root):
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(NAMES))
    text = (
        f"path: {out_root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/train\n"
        "test: images/train\n"
        "names:\n"
        f"{names}\n"
    )
    (out_root / "data.yaml").write_text(text, encoding="utf-8")


def audit(out_root):
    label_dir = out_root / "labels" / "train"
    counts = Counter()
    empty = 0
    labels = sorted(label_dir.glob("*.txt"))
    for label in labels:
        label_counts = read_label_counts(label)
        if not label_counts:
            empty += 1
        counts.update(label_counts)
    print("mixed dataset class distribution")
    print(f"labels: {len(labels)} empty: {empty}")
    for idx, name in enumerate(NAMES):
        print(f"{idx} {name}: {counts[idx]}")
    report = {
        "labels": len(labels),
        "empty_labels": empty,
        "class_counts": {NAMES[idx]: counts[idx] for idx in range(len(NAMES))},
    }
    (out_root / "class_distribution.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Build a mixed fine-tune YOLO dataset from base synthetic data plus arena booster data.")
    parser.add_argument("--base_dataset", type=Path, required=True)
    parser.add_argument("--booster_dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base_split", default="train")
    parser.add_argument("--booster_split", default="train")
    parser.add_argument("--base_count", type=int, default=12000)
    parser.add_argument("--booster_count", type=int, default=3000, help="Use 0 for all booster images.")
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--copy_mode", choices=["copy", "symlink"], default="copy")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out_root = args.output.resolve()
    if args.reset and out_root.exists():
        shutil.rmtree(out_root)
    for group in ("images", "labels", "_meta"):
        ensure_dir(out_root / group / "train")

    rng = random.Random(args.seed)
    base_stems = collect_stems(args.base_dataset, args.base_split)
    booster_stems = collect_stems(args.booster_dataset, args.booster_split)
    if len(base_stems) < args.base_count:
        raise RuntimeError(f"Need {args.base_count} base images, found {len(base_stems)}")
    base_sample = rng.sample(base_stems, args.base_count)
    booster_count = len(booster_stems) if args.booster_count <= 0 else min(args.booster_count, len(booster_stems))
    booster_sample = booster_stems if booster_count == len(booster_stems) else rng.sample(booster_stems, booster_count)

    for idx, stem in enumerate(sorted(base_sample)):
        copy_record(args.base_dataset, args.base_split, stem, out_root, f"base_{idx:06d}_{stem}", args.copy_mode, args.resume)
    for idx, stem in enumerate(sorted(booster_sample)):
        copy_record(args.booster_dataset, args.booster_split, stem, out_root, f"arena_{idx:06d}_{stem}", args.copy_mode, args.resume)

    write_data_yaml(out_root)
    audit(out_root)
    print(f"wrote mixed fine-tune dataset: {out_root}")
    print(f"base images: {len(base_sample)}")
    print(f"booster images: {len(booster_sample)}")
    print(f"booster ratio: {len(booster_sample) / max(len(base_sample) + len(booster_sample), 1):.3f}")


if __name__ == "__main__":
    main()
