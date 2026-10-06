import argparse
import json
import math
import random
import shutil
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
from tqdm import tqdm


FRUIT_CLASS_IDS = {
    0: "banana",
    1: "orange",
    2: "pineapple",
    3: "apple",
}
CUBE_CLASS_ID = 4
CUBE_LIKE_CLASS_IDS = set(FRUIT_CLASS_IDS) | {CUBE_CLASS_ID}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a YOLO-seg fruit_face_patch dataset from existing YOLO26 metadata."
    )
    parser.add_argument("--source_dataset", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--crop_mode", choices=["object_crop", "full_frame"], default="object_crop")
    parser.add_argument("--copy_mode", choices=["hardlink", "copy", "symlink"], default="hardlink")
    parser.add_argument("--min_face_pixels", type=int, default=900)
    parser.add_argument("--min_face_side", type=int, default=20)
    parser.add_argument("--crop_pad", type=float, default=0.18)
    parser.add_argument("--negative_keep_ratio", type=float, default=1.0)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--limit", type=int, default=0, help="Debug limit per split before filtering. 0 means no limit.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def clamp01(value):
    return min(1.0, max(0.0, float(value)))


def write_data_yaml(output_root: Path):
    text = f"""path: {output_root.resolve().as_posix()}
train: images/train
val: images/val
test: images/test
names:
  0: fruit_face_patch
"""
    (output_root / "data.yaml").write_text(text, encoding="utf-8")


def ensure_dirs(output_root: Path, splits):
    for split in splits:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "_meta" / split).mkdir(parents=True, exist_ok=True)


def reset_output(path: Path):
    if path.exists():
        shutil.rmtree(path)


def link_or_copy(src: Path, dst: Path, mode: str):
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        try:
            dst.hardlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)


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


def read_label_lines(path: Path):
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_yolo_segment(line: str):
    parts = line.split()
    if len(parts) < 7:
        return None
    try:
        class_id = int(float(parts[0]))
        coords = [float(x) for x in parts[1:]]
    except ValueError:
        return None
    if len(coords) % 2 != 0 or len(coords) < 6:
        return None
    points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
    return class_id, points


def polygon_bbox_norm(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def padded_bbox_px(bbox_norm, width, height, pad_ratio):
    x1, y1, x2, y2 = bbox_norm
    px1 = x1 * width
    py1 = y1 * height
    px2 = x2 * width
    py2 = y2 * height
    bw = max(1.0, px2 - px1)
    bh = max(1.0, py2 - py1)
    pad = max(bw, bh) * pad_ratio
    left = max(0, int(math.floor(px1 - pad)))
    top = max(0, int(math.floor(py1 - pad)))
    right = min(width, int(math.ceil(px2 + pad)))
    bottom = min(height, int(math.ceil(py2 + pad)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def transform_segment_to_crop(segment, crop_box, image_width, image_height):
    left, top, right, bottom = crop_box
    crop_w = max(1, right - left)
    crop_h = max(1, bottom - top)
    out = []
    for idx in range(0, len(segment), 2):
        x = float(segment[idx]) * image_width
        y = float(segment[idx + 1]) * image_height
        u = (x - left) / crop_w
        v = (y - top) / crop_h
        out.extend([clamp01(u), clamp01(v)])
    return out


def segment_area_norm(segment):
    if len(segment) < 6:
        return 0.0
    pts = []
    for idx in range(0, len(segment), 2):
        pts.append((float(segment[idx]), float(segment[idx + 1])))
    area = 0.0
    for idx, (x1, y1) in enumerate(pts):
        x2, y2 = pts[(idx + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


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


def fruit_object_queues(meta):
    queues = defaultdict(deque)
    for obj in meta.get("fruit_objects", []) or []:
        cls = obj.get("class")
        if cls:
            queues[str(cls)].append(obj)
    return queues


def yolo_line_from_segment(segment):
    return "0 " + " ".join(f"{clamp01(value):.6f}" for value in segment)


def save_meta(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def process_full_frame(args, split, image_path, meta, stats):
    out_image = args.output_root / "images" / split / image_path.name
    out_label = args.output_root / "labels" / split / f"{image_path.stem}.txt"
    out_meta = args.output_root / "_meta" / split / f"{image_path.stem}.json"
    if args.resume and out_image.exists() and out_label.exists():
        stats["skipped_resume"] += 1
        return

    lines = []
    positive_objects = []
    for obj in meta.get("fruit_objects", []) or []:
        if not face_passes(obj, args.min_face_pixels, args.min_face_side):
            continue
        kept_segments = []
        for segment in obj.get("visible_face_segments") or []:
            if len(segment) >= 6 and segment_area_norm(segment) > 0:
                lines.append(yolo_line_from_segment(segment))
                kept_segments.append(segment)
        if kept_segments:
            positive_objects.append(obj.get("object_name"))

    link_or_copy(image_path, out_image, args.copy_mode)
    out_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    save_meta(out_meta, {
        "source_image": str(image_path),
        "crop_mode": "full_frame",
        "positive_objects": positive_objects,
        "labels": len(lines),
    })
    stats["images"] += 1
    stats["positive_images"] += int(bool(lines))
    stats["labels"] += len(lines)


def process_object_crops(args, split, image_path, label_path, meta, rng, stats):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        stats["missing_or_bad_image"] += 1
        return
    height, width = image.shape[:2]
    fruit_queues = fruit_object_queues(meta)

    label_lines = read_label_lines(label_path)
    for object_idx, line in enumerate(label_lines):
        parsed = parse_yolo_segment(line)
        if parsed is None:
            stats["invalid_object_labels"] += 1
            continue
        old_class_id, object_points = parsed
        if old_class_id not in CUBE_LIKE_CLASS_IDS:
            continue

        is_negative = old_class_id == CUBE_CLASS_ID
        fruit_obj = None
        if old_class_id in FRUIT_CLASS_IDS:
            fruit_obj = fruit_queues[FRUIT_CLASS_IDS[old_class_id]].popleft() if fruit_queues[FRUIT_CLASS_IDS[old_class_id]] else None
            is_negative = not (fruit_obj and face_passes(fruit_obj, args.min_face_pixels, args.min_face_side))

        if is_negative and rng.random() > args.negative_keep_ratio:
            stats["negative_dropped"] += 1
            continue

        bbox = polygon_bbox_norm(object_points)
        crop_box = padded_bbox_px(bbox, width, height, args.crop_pad)
        if crop_box is None:
            stats["invalid_crop"] += 1
            continue
        left, top, right, bottom = crop_box
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            stats["invalid_crop"] += 1
            continue

        out_stem = f"{image_path.stem}_obj{object_idx:02d}"
        out_image = args.output_root / "images" / split / f"{out_stem}.jpg"
        out_label = args.output_root / "labels" / split / f"{out_stem}.txt"
        out_meta = args.output_root / "_meta" / split / f"{out_stem}.json"
        if args.resume and out_image.exists() and out_label.exists():
            stats["skipped_resume"] += 1
            continue

        label_segments = []
        if fruit_obj and not is_negative:
            for segment in fruit_obj.get("visible_face_segments") or []:
                transformed = transform_segment_to_crop(segment, crop_box, width, height)
                if len(transformed) >= 6 and segment_area_norm(transformed) > 0:
                    label_segments.append(transformed)

        label_text = "\n".join(yolo_line_from_segment(segment) for segment in label_segments)
        cv2.imwrite(str(out_image), crop, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)])
        out_label.write_text(label_text + ("\n" if label_text else ""), encoding="utf-8")
        save_meta(out_meta, {
            "source_image": str(image_path),
            "source_label": str(label_path),
            "source_object_index": object_idx,
            "source_class_id": old_class_id,
            "source_class_name": FRUIT_CLASS_IDS.get(old_class_id, "cube"),
            "source_object_name": fruit_obj.get("object_name") if fruit_obj else None,
            "crop_mode": "object_crop",
            "crop_xyxy": [left, top, right, bottom],
            "labels": len(label_segments),
            "negative": len(label_segments) == 0,
        })
        stats["images"] += 1
        stats["positive_images"] += int(bool(label_segments))
        stats["negative_images"] += int(not label_segments)
        stats["labels"] += len(label_segments)


def process_split(args, split, rng):
    image_dir = args.source_dataset / "images" / split
    label_dir = args.source_dataset / "labels" / split
    meta_dir = args.source_dataset / "_meta" / split
    meta_files = sorted(meta_dir.glob("*.json"))
    if args.limit > 0:
        meta_files = meta_files[:args.limit]

    stats = Counter()
    iterator = tqdm(meta_files, desc=f"{split} fruit_face_patch", unit="img")
    for meta_path in iterator:
        meta = read_json(meta_path)
        if meta is None:
            stats["bad_meta"] += 1
            continue
        image_path = find_image(image_dir, meta_path.stem)
        if image_path is None:
            stats["missing_image"] += 1
            continue
        if args.crop_mode == "full_frame":
            process_full_frame(args, split, image_path, meta, stats)
        else:
            process_object_crops(args, split, image_path, label_dir / f"{meta_path.stem}.txt", meta, rng, stats)
        iterator.set_postfix({
            "out": stats["images"],
            "pos": stats["positive_images"],
            "neg": stats["negative_images"],
        })
    return dict(stats)


def main():
    args = parse_args()
    args.source_dataset = args.source_dataset.resolve()
    args.output_root = args.output_root.resolve()
    if args.reset:
        reset_output(args.output_root)
    ensure_dirs(args.output_root, args.splits)
    write_data_yaml(args.output_root)

    rng = random.Random(args.seed)
    audit = {
        "source_dataset": str(args.source_dataset),
        "output_root": str(args.output_root),
        "crop_mode": args.crop_mode,
        "min_face_pixels": args.min_face_pixels,
        "min_face_side": args.min_face_side,
        "splits": {},
    }
    for split in args.splits:
        audit["splits"][split] = process_split(args, split, rng)

    audit_path = args.output_root / "fruit_face_patch_dataset_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote: {args.output_root}")
    print(f"data.yaml: {args.output_root / 'data.yaml'}")
    print(f"audit: {audit_path}")
    for split, stats in audit["splits"].items():
        print(f"{split}: {stats}")


if __name__ == "__main__":
    main()
