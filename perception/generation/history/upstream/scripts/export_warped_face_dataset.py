import argparse
import csv
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


FRUIT_CLASSES = ("apple", "orange", "banana", "pineapple")
OUTPUT_CLASSES = ("apple", "orange", "banana", "pineapple", "plain", "unknown")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export fruit-face classifier crops from YOLO26 metadata."
    )
    parser.add_argument("--source_dataset", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--min_face_pixels", type=int, default=900)
    parser.add_argument("--min_face_side", type=int, default=20)
    parser.add_argument(
        "--warp_method",
        choices=["bbox", "approx_quad", "min_area_rect"],
        default="bbox",
        help=(
            "Crop geometry. Use bbox for the current 100k metadata; approx_quad/min_area_rect "
            "are experimental because visible_face_segments are masks, not true cube-face quads."
        ),
    )
    parser.add_argument(
        "--approx_epsilon_ratio",
        type=float,
        default=0.02,
        help="Initial approxPolyDP epsilon ratio for converting face contour to a 4-corner perspective quad.",
    )
    parser.add_argument("--include_unknown_from_failed", action="store_true", default=True)
    parser.add_argument("--no_unknown_from_failed", dest="include_unknown_from_failed", action="store_false")
    parser.add_argument("--unknown_keep_ratio", type=float, default=0.35)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--limit", type=int, default=0, help="Debug limit per split before filtering. 0 means no limit.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def reset_output(path: Path):
    if path.exists():
        shutil.rmtree(path)


def ensure_dirs(output_root: Path, splits):
    for split in splits:
        for class_name in OUTPUT_CLASSES:
            (output_root / split / class_name).mkdir(parents=True, exist_ok=True)


def find_image(image_dir: Path, stem: str):
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def face_passes(obj, min_face_pixels, min_face_side):
    if not obj.get("fruit_visibility_pass", False):
        return False
    if int(obj.get("visible_face_pixels") or 0) < min_face_pixels:
        return False
    if int(obj.get("visible_face_bbox_width") or 0) < min_face_side:
        return False
    if int(obj.get("visible_face_bbox_height") or 0) < min_face_side:
        return False
    if not obj.get("visible_face_segments"):
        return False
    return True


def segment_to_points_px(segment, width, height):
    points = []
    for idx in range(0, len(segment), 2):
        x = float(segment[idx]) * max(width - 1, 1)
        y = float(segment[idx + 1]) * max(height - 1, 1)
        points.append([x, y])
    return np.asarray(points, dtype=np.float32)


def polygon_area(points):
    if points is None or len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def largest_segment(obj, width, height):
    best = None
    best_area = -1.0
    for segment in obj.get("visible_face_segments") or []:
        if len(segment) < 6:
            continue
        points = segment_to_points_px(segment, width, height)
        area = polygon_area(points)
        if area > best_area:
            best_area = area
            best = points
    return best, best_area


def order_quad_points(points):
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("quad must be shape (4, 2)")
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def quad_from_points(points):
    if points is None or len(points) < 3:
        return None
    rect = cv2.minAreaRect(points.astype(np.float32))
    box = cv2.boxPoints(rect)
    if polygon_area(box) <= 1.0:
        return None
    return order_quad_points(box)


def approx_quad_from_points(points, epsilon_ratio):
    if points is None or len(points) < 4:
        return None
    contour = points.astype(np.float32).reshape(-1, 1, 2)
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    if perimeter <= 1.0:
        return None
    ratios = [
        max(0.001, float(epsilon_ratio)),
        0.01,
        0.015,
        0.025,
        0.035,
        0.05,
        0.075,
        0.10,
    ]
    seen = set()
    for ratio in ratios:
        key = round(ratio, 4)
        if key in seen:
            continue
        seen.add(key)
        approx = cv2.approxPolyDP(hull, perimeter * ratio, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            if polygon_area(quad) > 4.0:
                return order_quad_points(quad)
    return None


def bbox_from_norm(bbox, width, height, pad_ratio=0.08):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    left = x1 * width
    top = y1 * height
    right = x2 * width
    bottom = y2 * height
    bw = max(1.0, right - left)
    bh = max(1.0, bottom - top)
    pad = max(bw, bh) * pad_ratio
    left = max(0, int(math.floor(left - pad)))
    top = max(0, int(math.floor(top - pad)))
    right = min(width, int(math.ceil(right + pad)))
    bottom = min(height, int(math.ceil(bottom + pad)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def warp_quad(image, quad, crop_size):
    dst = np.asarray(
        [[0, 0], [crop_size - 1, 0], [crop_size - 1, crop_size - 1], [0, crop_size - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(image, matrix, (crop_size, crop_size), flags=cv2.INTER_LINEAR)


def crop_resize(image, box, crop_size):
    left, top, right, bottom = box
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)


def safe_name(value):
    text = str(value or "object")
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)


def save_crop(path: Path, image, quality):
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]))


def export_positive_crop(args, split, image, image_path, obj, obj_idx, writer, stats):
    height, width = image.shape[:2]
    points, area = largest_segment(obj, width, height)
    if points is None:
        stats["no_segment"] += 1
        return

    method = args.warp_method
    warped = None
    quad = None
    if method == "approx_quad":
        quad = approx_quad_from_points(points, args.approx_epsilon_ratio)
        if quad is not None:
            warped = warp_quad(image, quad, args.crop_size)
        else:
            method = "min_area_rect"

    if warped is None and method == "min_area_rect":
        quad = quad_from_points(points)
        if quad is not None:
            warped = warp_quad(image, quad, args.crop_size)
        else:
            method = "bbox"

    if warped is None:
        box = bbox_from_norm(obj.get("visible_face_bbox") or [], width, height)
        if box is None:
            stats["bad_bbox"] += 1
            return
        warped = crop_resize(image, box, args.crop_size)
        if warped is None:
            stats["bad_bbox"] += 1
            return

    class_name = str(obj.get("class"))
    if class_name not in FRUIT_CLASSES:
        stats["non_fruit_class"] += 1
        return
    object_name = safe_name(obj.get("object_name"))
    out_name = f"{image_path.stem}_{obj_idx:02d}_{object_name}.jpg"
    out_path = args.output_root / split / class_name / out_name
    if args.resume and out_path.exists():
        stats["skipped_resume"] += 1
        return
    if save_crop(out_path, warped, args.jpeg_quality):
        stats[f"class_{class_name}"] += 1
        stats["positive"] += 1
        writer.writerow({
            "split": split,
            "class": class_name,
            "path": str(out_path),
            "source_image": str(image_path),
            "object_name": obj.get("object_name"),
            "warp_method": method,
            "segment_area_px": round(area, 3),
            "visible_face_pixels": obj.get("visible_face_pixels"),
            "visible_face_bbox_width": obj.get("visible_face_bbox_width"),
            "visible_face_bbox_height": obj.get("visible_face_bbox_height"),
            "quality": "positive",
        })
        stats[f"method_{method}"] += 1


def export_unknown_crop(args, split, image, image_path, obj, obj_idx, rng, writer, stats):
    if not args.include_unknown_from_failed or rng.random() > args.unknown_keep_ratio:
        stats["unknown_dropped"] += 1
        return
    bbox = obj.get("visible_face_bbox")
    if not bbox:
        stats["unknown_no_bbox"] += 1
        return
    height, width = image.shape[:2]
    box = bbox_from_norm(bbox, width, height, pad_ratio=0.18)
    if box is None:
        stats["unknown_bad_bbox"] += 1
        return
    crop = crop_resize(image, box, args.crop_size)
    if crop is None:
        stats["unknown_bad_bbox"] += 1
        return
    object_name = safe_name(obj.get("object_name"))
    out_name = f"{image_path.stem}_{obj_idx:02d}_{object_name}_unknown.jpg"
    out_path = args.output_root / split / "unknown" / out_name
    if args.resume and out_path.exists():
        stats["skipped_resume"] += 1
        return
    if save_crop(out_path, crop, args.jpeg_quality):
        stats["unknown"] += 1
        writer.writerow({
            "split": split,
            "class": "unknown",
            "path": str(out_path),
            "source_image": str(image_path),
            "object_name": obj.get("object_name"),
            "warp_method": "bbox_failed_face",
            "segment_area_px": "",
            "visible_face_pixels": obj.get("visible_face_pixels"),
            "visible_face_bbox_width": obj.get("visible_face_bbox_width"),
            "visible_face_bbox_height": obj.get("visible_face_bbox_height"),
            "quality": "failed_threshold",
        })


def process_split(args, split, rng, writer):
    image_dir = args.source_dataset / "images" / split
    meta_dir = args.source_dataset / "_meta" / split
    meta_files = sorted(meta_dir.glob("*.json"))
    if args.limit > 0:
        meta_files = meta_files[:args.limit]

    stats = Counter()
    iterator = tqdm(meta_files, desc=f"{split} warped faces", unit="img")
    for meta_path in iterator:
        meta = read_json(meta_path)
        if meta is None:
            stats["bad_meta"] += 1
            continue
        image_path = find_image(image_dir, meta_path.stem)
        if image_path is None:
            stats["missing_image"] += 1
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            stats["bad_image"] += 1
            continue
        for obj_idx, obj in enumerate(meta.get("fruit_objects", []) or []):
            if face_passes(obj, args.min_face_pixels, args.min_face_side):
                export_positive_crop(args, split, image, image_path, obj, obj_idx, writer, stats)
            else:
                export_unknown_crop(args, split, image, image_path, obj, obj_idx, rng, writer, stats)
        iterator.set_postfix({
            "pos": stats["positive"],
            "unk": stats["unknown"],
        })
    return dict(stats)


def main():
    args = parse_args()
    args.source_dataset = args.source_dataset.resolve()
    args.output_root = args.output_root.resolve()
    if args.reset:
        reset_output(args.output_root)
    ensure_dirs(args.output_root, args.splits)

    rng = random.Random(args.seed)
    audit = {
        "source_dataset": str(args.source_dataset),
        "output_root": str(args.output_root),
        "crop_size": args.crop_size,
        "min_face_pixels": args.min_face_pixels,
        "min_face_side": args.min_face_side,
        "warp_method": args.warp_method,
        "classes": OUTPUT_CLASSES,
        "splits": {},
    }
    audit_csv = args.output_root / "warped_face_audit.csv"
    with audit_csv.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "split",
            "class",
            "path",
            "source_image",
            "object_name",
            "warp_method",
            "segment_area_px",
            "visible_face_pixels",
            "visible_face_bbox_width",
            "visible_face_bbox_height",
            "quality",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for split in args.splits:
            audit["splits"][split] = process_split(args, split, rng, writer)

    audit_json = args.output_root / "warped_face_audit.json"
    audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote: {args.output_root}")
    print(f"audit csv: {audit_csv}")
    print(f"audit json: {audit_json}")
    for split, stats in audit["splits"].items():
        print(f"{split}: {stats}")
    print("note: plain class directories are created but not populated by this script; plain face GT needs cube-face metadata.")
    print("note: default crop geometry is bbox. True perspective warp needs cube face 4-corner quad metadata.")


if __name__ == "__main__":
    main()
