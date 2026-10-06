from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


CLASS_NAMES = ["apple", "orange", "banana", "pineapple", "plain"]
FRUIT_CLASS_IDS = {0, 1, 2, 3}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
CASE_WEIGHTS = {"normal": 0.55, "border": 0.35, "tight": 0.10}


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def label_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx, part in enumerate(parts):
        if part == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    raise ValueError(f"image path does not contain images directory: {image_path}")


def collect_images(root: Path, split: str) -> list[Path]:
    image_dir = root / "images" / split
    return sorted(
        path
        for path in image_dir.glob("*")
        if path.suffix.lower() in IMAGE_EXTS and label_for_image(path).exists()
    )


def read_label(path: Path) -> list[tuple[int, np.ndarray]]:
    records: list[tuple[int, np.ndarray]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 7 or len(parts) % 2 == 0:
            continue
        class_id = int(float(parts[0]))
        coords = np.array([float(v) for v in parts[1:]], dtype=np.float32).reshape(-1, 2)
        records.append((class_id, coords))
    return records


def primary_class_id(records: list[tuple[int, np.ndarray]]) -> int:
    for class_id, _ in records:
        if class_id in FRUIT_CLASS_IDS:
            return class_id
    return 4


def read_image(path: Path, crop_size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    if image.shape[:2] != (crop_size, crop_size):
        image = cv2.resize(image, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
    return image


def low_detail_context(image: np.ndarray, rng: random.Random) -> np.ndarray:
    out = image.copy()
    if rng.random() < 0.85:
        sigma = rng.uniform(2.0, 6.0)
        out = cv2.GaussianBlur(out, (0, 0), sigma)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= rng.uniform(0.25, 0.70)
    hsv[:, :, 2] *= rng.uniform(0.78, 1.12)
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    alpha = rng.uniform(0.55, 0.82)
    neutral = np.full_like(out, rng.randint(185, 230))
    out = cv2.addWeighted(out, alpha, neutral, 1.0 - alpha, rng.uniform(-6, 6))
    return out


def choose_case(rng: random.Random, weights: dict[str, float]) -> str:
    names = list(weights.keys())
    values = [max(0.0, float(weights[name])) for name in names]
    total = sum(values)
    if total <= 0:
        return "normal"
    return rng.choices(names, weights=values, k=1)[0]


def affine_params(case: str, crop_size: int, rng: random.Random, args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    if case == "normal":
        scale = rng.uniform(0.90, 1.04)
        shift_frac = rng.uniform(0.00, args.normal_shift_frac)
        angle = rng.uniform(-args.normal_rotation_deg, args.normal_rotation_deg)
    elif case == "tight":
        scale = rng.uniform(args.tight_scale_min, args.tight_scale_max)
        shift_frac = rng.uniform(0.02, args.tight_shift_frac)
        angle = rng.uniform(-args.tight_rotation_deg, args.tight_rotation_deg)
    else:
        scale = rng.uniform(args.border_scale_min, args.border_scale_max)
        shift_frac = rng.uniform(args.normal_shift_frac, args.border_shift_frac)
        angle = rng.uniform(-args.border_rotation_deg, args.border_rotation_deg)

    theta = rng.uniform(0.0, math.tau)
    dx = math.cos(theta) * shift_frac * crop_size
    dy = math.sin(theta) * shift_frac * crop_size
    matrix = cv2.getRotationMatrix2D((crop_size / 2.0, crop_size / 2.0), angle, scale)
    matrix[0, 2] += dx
    matrix[1, 2] += dy
    meta = {
        "case": case,
        "scale": round(float(scale), 4),
        "angle_deg": round(float(angle), 4),
        "shift_frac": round(float(shift_frac), 4),
        "dx_px": round(float(dx), 3),
        "dy_px": round(float(dy), 3),
    }
    return matrix.astype(np.float32), meta


def warp_image(
    source: np.ndarray,
    context: np.ndarray,
    matrix: np.ndarray,
    crop_size: int,
    rng: random.Random,
) -> np.ndarray:
    warped = cv2.warpAffine(
        source,
        matrix,
        (crop_size, crop_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    alpha = cv2.warpAffine(
        np.full((crop_size, crop_size), 255, dtype=np.uint8),
        matrix,
        (crop_size, crop_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if rng.random() < 0.50:
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    a = (alpha.astype(np.float32) / 255.0)[:, :, None]
    out = context.astype(np.float32) * (1.0 - a) + warped.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def transformed_label_segments(
    records: list[tuple[int, np.ndarray]],
    matrix: np.ndarray,
    crop_size: int,
    min_area: float,
) -> list[tuple[int, np.ndarray]]:
    transformed: list[tuple[int, np.ndarray]] = []
    for class_id, norm_points in records:
        px = norm_points.copy()
        px[:, 0] *= crop_size
        px[:, 1] *= crop_size
        ones = np.ones((px.shape[0], 1), dtype=np.float32)
        pts = np.hstack([px, ones]) @ matrix.T
        mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            epsilon = max(0.75, 0.003 * cv2.arcLength(contour, True))
            approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(np.float32)
            if approx.shape[0] < 3:
                continue
            approx[:, 0] = np.clip(approx[:, 0] / crop_size, 0.0, 1.0)
            approx[:, 1] = np.clip(approx[:, 1] / crop_size, 0.0, 1.0)
            transformed.append((class_id, approx))
    return transformed


def write_label(path: Path, records: list[tuple[int, np.ndarray]]) -> None:
    lines = []
    for class_id, points in records:
        coords = " ".join(f"{float(v):.6f}" for v in points.reshape(-1))
        lines.append(f"{class_id} {coords}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_data_yaml(root: Path) -> None:
    lines = [
        f"path: {str(root.resolve()).replace(os.sep, '/')}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for idx, name in enumerate(CLASS_NAMES):
        lines.append(f"  {idx}: {name}")
    (root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def class_color(class_id: int) -> tuple[int, int, int]:
    palette = {
        0: (30, 30, 230),
        1: (20, 120, 255),
        2: (35, 220, 235),
        3: (40, 170, 75),
        4: (220, 220, 220),
    }
    return palette.get(class_id, (255, 255, 255))


def draw_label_overlay(image: np.ndarray, records: list[tuple[int, np.ndarray]]) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    for class_id, points in records:
        pts = points.copy()
        pts[:, 0] *= w
        pts[:, 1] *= h
        contour = np.round(pts).astype(np.int32)
        color = class_color(class_id)
        cv2.polylines(out, [contour], True, color, 2, cv2.LINE_AA)
        if contour.size:
            x, y = contour[0]
            cv2.putText(
                out,
                CLASS_NAMES[class_id],
                (int(x), max(14, int(y) - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )
    return out


def make_contact_sheet(records: list[dict], output: Path, crop_size: int, max_items: int) -> None:
    chosen = records[:]
    random.Random(31).shuffle(chosen)
    chosen = chosen[:max_items]
    if not chosen:
        return
    thumb = min(160, crop_size)
    header = 22
    cols = 8
    rows = math.ceil(len(chosen) / cols)
    canvas = np.full((rows * (thumb + header), cols * thumb, 3), 245, dtype=np.uint8)
    for idx, rec in enumerate(chosen):
        image = cv2.imread(rec["image"], cv2.IMREAD_COLOR)
        if image is None:
            continue
        label_records = read_label(Path(rec["label"]))
        image = draw_label_overlay(image, label_records)
        image = cv2.resize(image, (thumb, thumb), interpolation=cv2.INTER_AREA)
        row, col = divmod(idx, cols)
        x = col * thumb
        y = row * (thumb + header)
        canvas[y + header : y + header + thumb, x : x + thumb] = image
        text = f"{rec['primary_class']} {rec['case']}"[:28]
        cv2.putText(canvas, text, (x + 3, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 93])


def group_by_primary(images: list[Path]) -> dict[int, list[tuple[Path, list[tuple[int, np.ndarray]]]]]:
    groups: dict[int, list[tuple[Path, list[tuple[int, np.ndarray]]]]] = defaultdict(list)
    for image_path in images:
        records = read_label(label_for_image(image_path))
        if not records:
            continue
        groups[primary_class_id(records)].append((image_path, records))
    return groups


def balanced_jobs(
    groups: dict[int, list[tuple[Path, list[tuple[int, np.ndarray]]]]],
    count: int,
    rng: random.Random,
) -> list[tuple[Path, list[tuple[int, np.ndarray]]]]:
    non_empty = [class_id for class_id in range(len(CLASS_NAMES)) if groups.get(class_id)]
    if not non_empty:
        raise RuntimeError("no labeled source images found")
    jobs: list[tuple[Path, list[tuple[int, np.ndarray]]]] = []
    per_class = math.ceil(count / len(non_empty))
    for class_id in non_empty:
        pool = groups[class_id][:]
        rng.shuffle(pool)
        if len(pool) >= per_class:
            jobs.extend(pool[:per_class])
        else:
            jobs.extend(pool)
            while len([job for job in jobs if primary_class_id(job[1]) == class_id]) < per_class:
                jobs.append(rng.choice(pool))
    rng.shuffle(jobs)
    return jobs[:count]


def materialize_split(
    source_root: Path,
    output_root: Path,
    split: str,
    count: int,
    rng: random.Random,
    args: argparse.Namespace,
    case_weights: dict[str, float],
) -> tuple[list[dict], Counter]:
    images = collect_images(source_root, split)
    if not images:
        raise RuntimeError(f"no source images for split={split}: {source_root}")
    groups = group_by_primary(images)
    jobs = balanced_jobs(groups, count, rng)
    context_images = images[:]
    stats: Counter = Counter()
    records_out: list[dict] = []

    for idx, (src_image, label_records) in enumerate(tqdm(jobs, desc=f"a1 crop jitter {split}", unit="img")):
        context_path = rng.choice(context_images)
        source = read_image(src_image, args.crop_size)
        context = low_detail_context(read_image(context_path, args.crop_size), rng)
        case = choose_case(rng, case_weights)
        matrix, meta = affine_params(case, args.crop_size, rng, args)
        boosted = warp_image(source, context, matrix, args.crop_size, rng)
        transformed = transformed_label_segments(label_records, matrix, args.crop_size, args.min_segment_area)
        if not transformed:
            continue
        primary_id = primary_class_id(label_records)
        digest = hashlib.sha1(str(src_image).encode("utf-8")).hexdigest()[:12]
        stem = f"{idx:06d}_{CLASS_NAMES[primary_id]}_{case}_{digest}"
        dst_image = output_root / "images" / split / f"{stem}.jpg"
        dst_label = output_root / "labels" / split / f"{stem}.txt"
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(dst_image), boosted, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
        if not ok:
            raise RuntimeError(f"failed to write image: {dst_image}")
        write_label(dst_label, transformed)

        classes_after = [CLASS_NAMES[class_id] for class_id, _ in transformed]
        stats[f"{split}_{CLASS_NAMES[primary_id]}"] += 1
        stats[f"{split}_{case}"] += 1
        for name in classes_after:
            stats[f"{split}_label_{name}"] += 1
        records_out.append(
            {
                "split": split,
                "source": str(src_image),
                "context": str(context_path),
                "image": str(dst_image),
                "label": str(dst_label),
                "primary_class": CLASS_NAMES[primary_id],
                "labels_after": "|".join(classes_after),
                **meta,
            }
        )
    return records_out, stats


def write_records_csv(records: list[dict], output: Path) -> None:
    fields = [
        "split",
        "primary_class",
        "case",
        "scale",
        "angle_deg",
        "shift_frac",
        "dx_px",
        "dy_px",
        "labels_after",
        "source",
        "context",
        "image",
        "label",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow({field: rec.get(field, "") for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an A1-crop-jitter/context booster from existing cube-face unified crops. "
            "No new cube geometry, fruit icons, or synthetic labels are rendered."
        )
    )
    parser.add_argument(
        "--source_root",
        type=Path,
        default=Path("datasets/cube_face_unified_finetune_meta_v2_50000_plus_hsv_ratio_20000_stronger_v1"),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("datasets/cube_face_unified_booster_a1_crop_jitter_context_v1"),
    )
    parser.add_argument("--train_count", type=int, default=10000)
    parser.add_argument("--val_count", type=int, default=1000)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--preview_count", type=int, default=160)
    parser.add_argument("--jpeg_quality", type=int, default=94)
    parser.add_argument("--min_segment_area", type=float, default=80.0)
    parser.add_argument("--normal_ratio", type=float, default=CASE_WEIGHTS["normal"])
    parser.add_argument("--border_ratio", type=float, default=CASE_WEIGHTS["border"])
    parser.add_argument("--tight_ratio", type=float, default=CASE_WEIGHTS["tight"])
    parser.add_argument("--normal_shift_frac", type=float, default=0.08)
    parser.add_argument("--border_shift_frac", type=float, default=0.18)
    parser.add_argument("--tight_shift_frac", type=float, default=0.12)
    parser.add_argument("--normal_rotation_deg", type=float, default=1.5)
    parser.add_argument("--border_rotation_deg", type=float, default=3.0)
    parser.add_argument("--tight_rotation_deg", type=float, default=2.2)
    parser.add_argument("--border_scale_min", type=float, default=0.82)
    parser.add_argument("--border_scale_max", type=float, default=0.96)
    parser.add_argument("--tight_scale_min", type=float, default=1.03)
    parser.add_argument("--tight_scale_max", type=float, default=1.12)
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reset:
        reset_dir(args.output_root)
    for split in ["train", "val"]:
        (args.output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    total_case = args.normal_ratio + args.border_ratio + args.tight_ratio
    case_weights = {
        "normal": args.normal_ratio / total_case,
        "border": args.border_ratio / total_case,
        "tight": args.tight_ratio / total_case,
    }
    all_records: list[dict] = []
    stats: Counter = Counter()
    for split, count in [("train", args.train_count), ("val", args.val_count)]:
        records, split_stats = materialize_split(args.source_root, args.output_root, split, count, rng, args, case_weights)
        all_records.extend(records)
        stats.update(split_stats)

    write_data_yaml(args.output_root)
    write_records_csv(all_records, args.output_root / "records.csv")
    preview = args.output_root / "previews" / "a1_crop_jitter_context_contact_sheet.jpg"
    make_contact_sheet(all_records, preview, args.crop_size, args.preview_count)
    manifest = {
        "task": "cube_face_unified_a1_crop_jitter_context_booster",
        "policy": (
            "Derived only from existing cube-face unified crop images and YOLO segmentation labels. "
            "No ideal geometry, flat icons, or new cube rendering is generated."
        ),
        "intent": [
            "simulate A1 bbox crop jitter before unified face classification",
            "expose mild neighboring/context pixels at crop borders",
            "keep original visual realism and transform labels with the same affine matrix",
        ],
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "train_count_requested": args.train_count,
        "val_count_requested": args.val_count,
        "records_written": len(all_records),
        "case_weights": case_weights,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "class_names": CLASS_NAMES,
        "stats": dict(stats),
        "preview": str(preview),
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
