"""Filter A1 crop-jitter booster samples by a pruned materialized source set.

This script keeps only samples whose primary source image and optional context
image still exist in the pruned materialized dataset. It does not create new
geometry and does not use user-provided hard-case photos.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = ["apple", "orange", "banana", "pineapple", "plain"]
CLASS_COLORS = {
    "apple": (40, 40, 230),
    "orange": (0, 150, 255),
    "banana": (0, 220, 255),
    "pineapple": (60, 180, 60),
    "plain": (230, 230, 230),
}


def safe_reset(path: Path, workspace: Path) -> None:
    resolved = path.resolve()
    workspace = workspace.resolve()
    if workspace not in resolved.parents and resolved != workspace:
        raise RuntimeError(f"refusing to reset path outside workspace: {resolved}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def as_posix(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def label_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx, part in enumerate(parts):
        if part == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    raise ValueError(f"image path has no images segment: {image_path}")


def link_or_copy(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy_fallback"


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    return image


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


def load_pruned_names(pruned_root: Path) -> dict[str, set[str]]:
    names: dict[str, set[str]] = {}
    for split in ["train", "val"]:
        split_dir = pruned_root / "images" / split
        if not split_dir.exists():
            raise FileNotFoundError(split_dir)
        names[split] = {p.name for p in split_dir.glob("*.jpg")}
    return names


def row_is_kept(row: dict[str, str], kept_names: dict[str, set[str]]) -> bool:
    split = row["split"]
    source_name = Path(row["source"]).name
    context_name = Path(row.get("context", "")).name if row.get("context") else ""
    if source_name not in kept_names[split]:
        return False
    if context_name and context_name not in kept_names[split]:
        return False
    return True


def balanced_select(rows: list[dict[str, str]], per_class_cap: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_class[row["primary_class"]].append(row)
    selected: list[dict[str, str]] = []
    for class_name in CLASS_NAMES:
        class_rows = by_class.get(class_name, [])
        rng.shuffle(class_rows)
        if per_class_cap > 0:
            class_rows = class_rows[:per_class_cap]
        selected.extend(class_rows)
    selected.sort(key=lambda r: (r["split"], r["primary_class"], r["case"], Path(r["image"]).name))
    return selected


def parse_label(path: Path, size: int) -> list[tuple[str, np.ndarray]]:
    items: list[tuple[str, np.ndarray]] = []
    if not path.exists():
        return items
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 9:
            continue
        cls_id = int(float(parts[0]))
        coords = np.array([float(v) for v in parts[1:]], dtype=np.float32).reshape(-1, 2)
        coords[:, 0] *= size
        coords[:, 1] *= size
        class_name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else "unknown"
        items.append((class_name, coords))
    return items


def make_preview(root: Path, rows: list[dict[str, str]], count: int, tile: int, seed: int) -> str | None:
    if count <= 0 or not rows:
        return None
    rng = random.Random(seed)
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_class[row["primary_class"]].append(row)
    for class_rows in by_class.values():
        rng.shuffle(class_rows)
    preview_rows: list[dict[str, str]] = []
    while len(preview_rows) < min(count, len(rows)):
        added = False
        for class_name in CLASS_NAMES:
            class_rows = by_class.get(class_name, [])
            if class_rows:
                preview_rows.append(class_rows.pop())
                added = True
                if len(preview_rows) >= min(count, len(rows)):
                    break
        if not added:
            break
    cols = 8
    header_h = 20
    rows_n = int(np.ceil(len(preview_rows) / cols))
    sheet = np.full((rows_n * (tile + header_h), cols * tile, 3), 245, dtype=np.uint8)
    for idx, row in enumerate(preview_rows):
        image_path = Path(row["image"])
        label_path = Path(row["label"])
        image = read_image(image_path)
        original_h, original_w = image.shape[:2]
        image = cv2.resize(image, (tile, tile), interpolation=cv2.INTER_AREA)
        scale_x = tile / float(original_w)
        scale_y = tile / float(original_h)
        for class_name, pts in parse_label(label_path, size=224):
            pts = pts.copy()
            pts[:, 0] *= scale_x
            pts[:, 1] *= scale_y
            pts = pts.astype(np.int32)
            color = CLASS_COLORS.get(class_name, (255, 255, 255))
            cv2.polylines(image, [pts], True, color, 1, cv2.LINE_AA)
        label = f"{row['primary_class']} {row['case']}"
        y = (idx // cols) * (tile + header_h)
        x = (idx % cols) * tile
        cv2.putText(sheet, label[:24], (x + 2, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
        sheet[y + header_h : y + header_h + tile, x : x + tile] = image
    out_dir = root / "previews"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "a1_crop_jitter_context_pruned_contact_sheet.jpg"
    cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return str(out_path)


def build(args: argparse.Namespace) -> dict:
    if args.reset:
        safe_reset(args.output_root, args.workspace)
    for split in ["train", "val"]:
        (args.output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    kept_names = load_pruned_names(args.pruned_root)
    all_rows: dict[str, list[dict[str, str]]] = {"train": [], "val": []}
    available_counts: Counter = Counter()
    rejected_counts: Counter = Counter()
    with args.records.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            split = row["split"]
            class_name = row["primary_class"]
            if row_is_kept(row, kept_names):
                all_rows[split].append(row)
                available_counts[(split, class_name, row["case"])] += 1
            else:
                rejected_counts[(split, class_name)] += 1

    selected_rows = {
        "train": balanced_select(all_rows["train"], args.train_per_class, args.seed),
        "val": balanced_select(all_rows["val"], args.val_per_class, args.seed + 1),
    }

    link_counts: Counter = Counter()
    selected_counts: Counter = Counter()
    output_rows: list[dict[str, str]] = []
    for split, rows in selected_rows.items():
        for out_idx, row in enumerate(rows):
            src_image = Path(row["image"])
            src_label = Path(row["label"])
            suffix = f"{out_idx:06d}_{row['primary_class']}_{row['case']}_{src_image.stem[-12:]}"
            dst_image = args.output_root / "images" / split / f"{suffix}.jpg"
            dst_label = args.output_root / "labels" / split / f"{suffix}.txt"
            link_counts[link_or_copy(src_image, dst_image, args.mode)] += 1
            link_counts[link_or_copy(src_label, dst_label, args.mode)] += 1
            selected_counts[(split, row["primary_class"], row["case"])] += 1
            out_row = dict(row)
            out_row["image"] = str(dst_image)
            out_row["label"] = str(dst_label)
            output_rows.append(out_row)

    write_data_yaml(args.output_root)
    records_path = args.output_root / "records.csv"
    with records_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(output_rows[0].keys()) if output_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    preview = make_preview(args.output_root, output_rows, args.preview_count, args.preview_tile, args.seed)
    manifest = {
        "task": "cube_face_unified_a1_crop_jitter_context_pruned_booster",
        "policy": "Filtered from an existing A1 crop-jitter booster; no user hard-case photos, ideal geometry, or new icon rendering.",
        "source_booster_root": str(args.source_booster_root),
        "source_records": str(args.records),
        "pruned_root": str(args.pruned_root),
        "output_root": str(args.output_root),
        "train_per_class_cap": args.train_per_class,
        "val_per_class_cap": args.val_per_class,
        "available_counts": {"/".join(k): v for k, v in sorted(available_counts.items())},
        "rejected_counts": {"/".join(k): v for k, v in sorted(rejected_counts.items())},
        "selected_counts": {"/".join(k): v for k, v in sorted(selected_counts.items())},
        "link_counts": dict(link_counts),
        "records": str(records_path),
        "preview": preview,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a pruned A1 crop-jitter/context booster.")
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--source_booster_root", type=Path, default=Path("datasets/cube_face_unified_booster_a1_crop_jitter_context_v1"))
    parser.add_argument("--records", type=Path, default=Path("datasets/cube_face_unified_booster_a1_crop_jitter_context_v1/records.csv"))
    parser.add_argument(
        "--pruned_root",
        type=Path,
        default=Path("datasets/cube_face_unified_finetune_meta_v2_50000_plus_hsv_ratio_20000_stronger_pruned_coloroutlier_v1"),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("datasets/cube_face_unified_booster_a1_crop_jitter_context_pruned_v1"),
    )
    parser.add_argument("--train_per_class", type=int, default=900)
    parser.add_argument("--val_per_class", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--preview_count", type=int, default=160)
    parser.add_argument("--preview_tile", type=int, default=96)
    parser.add_argument("--mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
