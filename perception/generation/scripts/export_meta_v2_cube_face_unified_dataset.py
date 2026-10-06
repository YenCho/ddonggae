import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from export_meta_v2_model_datasets import (
    bbox_from_points,
    crop_resize,
    expand_box,
    load_image,
    mask_crop_to_segments,
    normalize_point_in_box,
    normalized_bbox_from_points,
    read_json,
    read_mask,
    save_image,
)


FACE_CLASSES = ("apple", "orange", "banana", "pineapple", "plain")
FACE_CLASS_TO_ID = {name: idx for idx, name in enumerate(FACE_CLASSES)}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export a cube-crop unified face dataset from meta_v2. "
            "Each cube crop gets multi-class visible-face YOLO segmentation labels "
            "plus sidecar quad/keypoint metadata for future multitask heads."
        )
    )
    parser.add_argument("--source_dataset", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--source_split", default="train")
    parser.add_argument("--split_manifest", type=Path,
                        help="Shared scene-to-split JSON; prevents A1/face/pair leakage.")
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--crop_pad", type=float, default=0.18)
    parser.add_argument("--min_object_pixels", type=int, default=80)
    parser.add_argument("--min_face_pixels", type=int, default=120)
    parser.add_argument("--min_segment_area", type=float, default=8.0)
    parser.add_argument("--include_partial_faces", action="store_true", default=True)
    parser.add_argument("--no_include_partial_faces", dest="include_partial_faces", action="store_false")
    parser.add_argument("--include_bad_faces", action="store_true", default=False)
    parser.add_argument(
        "--keep_empty_crops",
        action="store_true",
        default=True,
        help="Keep cube crops with no accepted face labels as hard negatives.",
    )
    parser.add_argument("--no_keep_empty_crops", dest="keep_empty_crops", action="store_false")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def ensure_parent(path):
    path.parent.mkdir(parents=True, exist_ok=True)


def write_data_yaml(output_root):
    lines = [
        f"path: {output_root.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for idx, name in enumerate(FACE_CLASSES):
        lines.append(f"  {idx}: {name}")
    (output_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def face_class(face):
    if face.get("face_kind") == "plain_face":
        return "plain"
    fruit_class = face.get("fruit_class")
    if fruit_class in FACE_CLASS_TO_ID:
        return fruit_class
    return None


def split_meta_files(meta_files, val_ratio, seed, limit):
    files = list(meta_files)
    rng = random.Random(seed)
    rng.shuffle(files)
    if limit > 0:
        files = files[:limit]
    val_count = int(round(len(files) * max(0.0, min(0.9, val_ratio))))
    val_set = {p.name for p in files[:val_count]}
    return [(p, "val" if p.name in val_set else "train") for p in files]


def source_image_path(source_dataset, split, meta_path):
    for suffix in (".jpg", ".png", ".jpeg"):
        candidate = source_dataset / "images" / split / f"{meta_path.stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def export_object_crop(args, source_split, out_split, image, meta_path, obj, stats):
    h, w = image.shape[:2]
    if obj.get("object_class") != "cube_like_object":
        return
    if int(obj.get("visible_pixels") or 0) < args.min_object_pixels:
        stats["skipped_small_object"] += 1
        return

    bbox = obj.get("visible_bbox_xyxy") or obj.get("full_bbox_xyxy")
    if not bbox:
        stats["skipped_missing_object_bbox"] += 1
        return
    box = expand_box(bbox, w, h, args.crop_pad)
    if box is None:
        stats["skipped_bad_object_box"] += 1
        return

    object_name = str(obj.get("object_name") or f"obj{int(obj.get('object_id', 0)):03d}")
    out_stem = f"{meta_path.stem}_obj{int(obj.get('object_id', 0)):03d}_{object_name}"
    image_out = args.output_root / "images" / out_split / f"{out_stem}.jpg"
    label_out = args.output_root / "labels" / out_split / f"{out_stem}.txt"
    quad_out = args.output_root / "quads" / out_split / f"{out_stem}.json"

    if args.resume and image_out.exists() and label_out.exists() and quad_out.exists():
        stats["resumed_existing_crops"] += 1
        return

    crop = crop_resize(image, box, args.crop_size)
    if crop is None:
        stats["skipped_empty_crop"] += 1
        return

    lines = []
    face_records = []
    for face in obj.get("faces", []):
        quality = face.get("quality")
        if quality == "bad" and not args.include_bad_faces:
            stats["skipped_bad_face"] += 1
            continue
        if quality == "partial" and not args.include_partial_faces:
            stats["skipped_partial_face"] += 1
            continue
        if int(face.get("visible_pixels") or 0) < args.min_face_pixels:
            stats["skipped_small_face"] += 1
            continue

        label = face_class(face)
        if label is None:
            stats["skipped_unknown_face_class"] += 1
            continue

        mask = read_mask(args.source_dataset, face.get("visible_mask"))
        if mask is None:
            stats["skipped_missing_face_mask"] += 1
            continue
        segments = mask_crop_to_segments(
            mask,
            box,
            args.crop_size,
            min_area=args.min_segment_area,
        )
        if not segments:
            stats["skipped_empty_face_segments"] += 1
            continue

        class_id = FACE_CLASS_TO_ID[label]
        for segment in segments:
            lines.append(f"{class_id} " + " ".join(f"{float(v):.6f}" for v in segment))
            stats[f"instances_{label}"] += 1

        quad = face.get("quad_xy") or []
        quad_norm = [normalize_point_in_box(point, box) for point in quad] if len(quad) == 4 else []
        face_record = {
            "face_id": face.get("face_id"),
            "face_name": face.get("face_name"),
            "class_id": class_id,
            "class_name": label,
            "quality": quality,
            "occluded": bool(face.get("occluded") or face.get("occluder_object_ids")),
            "occluder_object_ids": face.get("occluder_object_ids") or [],
            "visible_pixels": int(face.get("visible_pixels") or 0),
            "full_pixels": int(face.get("full_pixels") or 0),
            "visible_ratio": float(face.get("visible_ratio") or 0.0),
            "quad_xy_norm_in_crop": quad_norm,
            "corner_visibility": face.get("corner_visibility") or [],
        }
        if quad_norm:
            face_record["quad_bbox_norm_in_crop"] = normalized_bbox_from_points(quad_norm)
        if face.get("visible_bbox_xyxy"):
            face_record["visible_bbox_norm_in_crop"] = normalized_bbox_from_points(
                [normalize_point_in_box(point, box) for point in [
                    [face["visible_bbox_xyxy"][0], face["visible_bbox_xyxy"][1]],
                    [face["visible_bbox_xyxy"][2], face["visible_bbox_xyxy"][1]],
                    [face["visible_bbox_xyxy"][2], face["visible_bbox_xyxy"][3]],
                    [face["visible_bbox_xyxy"][0], face["visible_bbox_xyxy"][3]],
                ]]
            )
        face_records.append(face_record)

    if not lines and not args.keep_empty_crops:
        stats["skipped_empty_label_crop"] += 1
        return

    save_image(image_out, crop)
    ensure_parent(label_out)
    label_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    ensure_parent(quad_out)
    quad_payload = {
        "source_dataset": str(args.source_dataset),
        "source_split": source_split,
        "source_image": f"images/{source_split}/{meta_path.stem}.jpg",
        "source_meta": f"_meta/{source_split}/{meta_path.name}",
        "object_id": obj.get("object_id"),
        "object_name": object_name,
        "crop_box_xyxy": box,
        "crop_size": args.crop_size,
        "face_count": len(face_records),
        "faces": face_records,
    }
    quad_out.write_text(json.dumps(quad_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    stats["crops"] += 1
    if not lines:
        stats["empty_label_crops"] += 1


def main():
    args = parse_args()
    if args.reset and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_data_yaml(args.output_root)

    meta_dir = args.source_dataset / "_meta" / args.source_split
    meta_files = sorted(meta_dir.glob("*.json"))
    if args.split_manifest:
        mapping = json.loads(args.split_manifest.read_text(encoding="utf-8"))["scenes"]
        if set(mapping) != {p.stem for p in meta_files}:
            raise RuntimeError("Split manifest does not match source scenes")
        if not set(mapping.values()) <= {"train", "val", "test"}:
            raise RuntimeError("Invalid split name in manifest")
        split_items = [(p, mapping[p.stem]) for p in meta_files]
    else:
        split_items = split_meta_files(meta_files, args.val_ratio, args.seed, args.limit)
    stats = Counter()
    stats["source_meta_files"] = len(split_items)

    for meta_path, out_split in tqdm(split_items, desc="cube-face unified export", unit="img"):
        meta = read_json(meta_path)
        meta_v2 = (meta or {}).get("meta_v2")
        if not meta_v2:
            stats["missing_meta_v2"] += 1
            continue
        image_path = source_image_path(args.source_dataset, args.source_split, meta_path)
        if image_path is None:
            stats["missing_source_image"] += 1
            continue
        image = load_image(image_path)
        for obj in meta_v2.get("objects", []):
            export_object_crop(args, args.source_split, out_split, image, meta_path, obj, stats)

    manifest = {
        "task": "cube_face_unified_segmentation_with_quad_sidecars",
        "classes": {idx: name for idx, name in enumerate(FACE_CLASSES)},
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "stats": dict(stats),
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["stats"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
