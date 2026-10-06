import argparse
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


A1_NAMES = {
    0: "cube_like_object",
    1: "octahedron",
    2: "dodecahedron",
    3: "icosahedron",
}
A2_NAMES = {
    0: "plain_face",
    1: "fruit_face",
}
C_CLASSES = ("apple", "orange", "banana", "pineapple", "plain", "unknown")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export A1/A2/B/C training inputs and outputs from visibility_meta_v2."
    )
    parser.add_argument("--source_dataset", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--copy_mode", choices=["copy", "hardlink", "symlink"], default="hardlink")
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--a2_crop_pad", type=float, default=0.18)
    parser.add_argument("--b_crop_pad", type=float, default=0.12)
    parser.add_argument(
        "--b_input_mode",
        choices=["mask", "raw", "raw_blacken_occlusion"],
        default="mask",
        help="B/facequad input. mask matches runtime A2 visible-face mask; raw modes are for visual ablations.",
    )
    parser.add_argument("--min_object_pixels", type=int, default=80)
    parser.add_argument("--min_face_pixels", type=int, default=120)
    parser.add_argument("--occlusion_fill", type=int, default=0)
    parser.add_argument("--mask_a2_target_object", action="store_true", default=False)
    parser.add_argument("--no_mask_a2_target_object", dest="mask_a2_target_object", action="store_false")
    parser.add_argument("--mask_b_target_object", action="store_true", default=False)
    parser.add_argument("--no_mask_b_target_object", dest="mask_b_target_object", action="store_false")
    parser.add_argument("--mask_c_visible_face", action="store_true", default=False)
    parser.add_argument("--no_mask_c_visible_face", dest="mask_c_visible_face", action="store_false")
    parser.add_argument("--blacken_b_occluded_face_area", action="store_true", default=True)
    parser.add_argument("--no_blacken_b_occluded_face_area", dest="blacken_b_occluded_face_area", action="store_false")
    parser.add_argument("--blacken_c_occluded_face_area", action="store_true", default=True)
    parser.add_argument("--no_blacken_c_occluded_face_area", dest="blacken_c_occluded_face_area", action="store_false")
    parser.add_argument("--b_min_quad_area", type=float, default=80.0)
    parser.add_argument("--b_min_quad_side", type=float, default=4.0)
    parser.add_argument("--b_max_quad_aspect", type=float, default=24.0)
    parser.add_argument("--c_min_quad_area", type=float, default=400.0)
    parser.add_argument("--c_min_quad_side", type=float, default=10.0)
    parser.add_argument("--c_max_quad_aspect", type=float, default=14.0)
    parser.add_argument("--c_max_visible_to_full_ratio", type=float, default=1.15)
    parser.add_argument("--c_min_visible_pixels", type=int, default=400)
    parser.add_argument("--c_min_visible_ratio", type=float, default=0.40)
    parser.add_argument("--c_max_black_fraction", type=float, default=0.60)
    parser.add_argument("--c_include_partial_faces", action="store_true", default=True)
    parser.add_argument("--no_c_include_partial_faces", dest="c_include_partial_faces", action="store_false")
    parser.add_argument("--c_include_bad_faces", action="store_true", default=False)
    parser.add_argument("--c_partial_requires_occluder", action="store_true", default=True)
    parser.add_argument("--no_c_partial_requires_occluder", dest="c_partial_requires_occluder", action="store_false")
    parser.add_argument(
        "--include_partial_as_unknown",
        action="store_true",
        default=False,
        help="Export partial/bad visible faces as C/unknown. Off by default because non-front-facing faces often warp into non-face crops.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def reset_output(path):
    if path.exists():
        shutil.rmtree(path)


def ensure_parent(path):
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_file(src, dst, mode, resume=False):
    if resume and dst.exists():
        return False
    ensure_parent(dst)
    if dst.exists():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)
    return True


def write_yaml(path, names, extra=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"path: {path.parent.resolve().as_posix()}",
        "train: images/train",
        "val: images/train",
        "names:",
    ]
    for idx, name in names.items():
        lines.append(f"  {idx}: {name}")
    if extra:
        lines.append(extra.rstrip())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    return image


def read_mask(source_root, mask_rel):
    if not mask_rel:
        return None
    path = source_root / mask_rel
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return mask > 127


def bbox_from_points(points):
    pts = np.asarray(points, dtype=np.float32)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return [float(x1), float(y1), float(x2), float(y2)]


def expand_box(box, width, height, pad_ratio):
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(1.0, x2 - x1 + 1.0)
    bh = max(1.0, y2 - y1 + 1.0)
    pad = max(bw, bh) * float(pad_ratio)
    x1 = max(0, int(math.floor(x1 - pad)))
    y1 = max(0, int(math.floor(y1 - pad)))
    x2 = min(width - 1, int(math.ceil(x2 + pad)))
    y2 = min(height - 1, int(math.ceil(y2 + pad)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def crop_resize(image, box, crop_size):
    x1, y1, x2, y2 = box
    crop = image[y1:y2 + 1, x1:x2 + 1]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)


def apply_binary_mask(image, mask, fill_value=0):
    if mask is None:
        return image
    fill = int(np.clip(fill_value, 0, 255))
    masked = np.full_like(image, fill)
    masked[mask.astype(bool)] = image[mask.astype(bool)]
    return masked


def full_quad_mask(shape, quad):
    h, w = shape[:2]
    pts = order_quad_points(quad)
    if pts is None:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(pts).astype(np.int32).reshape(-1, 1, 2)], 1)
    return mask.astype(bool)


def blacken_hidden_face_area(image, visible_face_mask, quad, fill_value=0, visible_dilate=3):
    if visible_face_mask is None:
        return image
    face_full = full_quad_mask(image.shape, quad)
    if face_full is None or not face_full.any():
        return image
    visible = visible_face_mask.astype(bool)
    if visible_dilate > 0:
        kernel = np.ones((visible_dilate, visible_dilate), dtype=np.uint8)
        visible = cv2.dilate(visible.astype(np.uint8), kernel, iterations=1).astype(bool)
    hidden = face_full & (~visible)
    if not hidden.any():
        return image
    out = image.copy()
    out[hidden] = int(np.clip(fill_value, 0, 255))
    return out


def mask_crop_to_segments(mask, box, crop_size, min_area=8, epsilon_ratio=0.002):
    x1, y1, x2, y2 = box
    crop = mask[y1:y2 + 1, x1:x2 + 1].astype(np.uint8) * 255
    crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < min_area:
            continue
        perimeter = cv2.arcLength(contour.astype(np.float32), True)
        approx = cv2.approxPolyDP(contour.astype(np.float32), max(0.5, perimeter * epsilon_ratio), True)
        if len(approx) < 3:
            continue
        segment = []
        for x, y in approx.reshape(-1, 2):
            segment.extend([
                float(np.clip(x / max(crop_size - 1, 1), 0.0, 1.0)),
                float(np.clip(y / max(crop_size - 1, 1), 0.0, 1.0)),
            ])
        if len(segment) >= 6:
            segments.append(segment)
    return segments


def normalize_point_in_box(point, box):
    x1, y1, x2, y2 = box
    w = max(1.0, x2 - x1 + 1.0)
    h = max(1.0, y2 - y1 + 1.0)
    return [
        float(np.clip((float(point[0]) - x1) / w, 0.0, 1.0)),
        float(np.clip((float(point[1]) - y1) / h, 0.0, 1.0)),
    ]


def box_yolo_from_xyxy(box, width, height):
    x1, y1, x2, y2 = [float(v) for v in box]
    return [
        ((x1 + x2) * 0.5) / max(width, 1),
        ((y1 + y2) * 0.5) / max(height, 1),
        max(1.0, x2 - x1 + 1.0) / max(width, 1),
        max(1.0, y2 - y1 + 1.0) / max(height, 1),
    ]


def normalized_bbox_from_points(points):
    pts = np.asarray(points, dtype=np.float32)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return [
        float(np.clip((x1 + x2) * 0.5, 0.0, 1.0)),
        float(np.clip((y1 + y2) * 0.5, 0.0, 1.0)),
        float(np.clip(x2 - x1, 0.0, 1.0)),
        float(np.clip(y2 - y1, 0.0, 1.0)),
    ]


def quad_area(points):
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        return 0.0
    return float(abs(cv2.contourArea(pts.reshape(-1, 1, 2))))


def order_quad_points(points):
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        return None
    if np.unique(np.round(pts, 3), axis=0).shape[0] != 4:
        return None
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -start, axis=0)
    if quad_area(ordered) <= 0.0:
        return None
    return ordered.astype(np.float32)


def quad_metrics(points):
    pts = order_quad_points(points)
    if pts is None:
        return None
    sides = [
        float(np.linalg.norm(pts[(idx + 1) % 4] - pts[idx]))
        for idx in range(4)
    ]
    min_side = min(sides) if sides else 0.0
    max_side = max(sides) if sides else 0.0
    return {
        "quad": pts,
        "area": quad_area(pts),
        "min_side": min_side,
        "max_side": max_side,
        "aspect": float(max_side / max(min_side, 1e-6)),
    }


def quad_passes(points, min_area, min_side, max_aspect):
    metrics = quad_metrics(points)
    if metrics is None:
        return None
    if metrics["area"] < min_area:
        return None
    if metrics["min_side"] < min_side:
        return None
    if metrics["aspect"] > max_aspect:
        return None
    return metrics["quad"]


def warp_quad(image, quad, crop_size):
    src = order_quad_points(quad)
    if src is None:
        return None
    dst = np.asarray(
        [[0, 0], [crop_size - 1, 0], [crop_size - 1, crop_size - 1], [0, crop_size - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (crop_size, crop_size), flags=cv2.INTER_LINEAR)


def save_image(path, image):
    ensure_parent(path)
    return bool(cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 94]))


def export_a1(args, split, image_path, meta_v2, stats):
    dst_image = args.output_root / "a1_objectseg" / "images" / split / image_path.name
    copy_file(image_path, dst_image, args.copy_mode, resume=args.resume)
    label_path = args.output_root / "a1_objectseg" / "labels" / split / f"{image_path.stem}.txt"
    if args.resume and label_path.exists():
        return
    ensure_parent(label_path)
    lines = []
    for obj in meta_v2.get("objects", []):
        if int(obj.get("visible_pixels") or 0) < args.min_object_pixels:
            continue
        class_id = obj.get("a1_class_id")
        for segment in obj.get("visible_segments") or []:
            lines.append(f"{int(class_id)} " + " ".join(f"{float(v):.6f}" for v in segment))
            stats[f"a1_{A1_NAMES[int(class_id)]}"] += 1
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    stats["a1_images"] += 1


def export_a2(args, split, image, image_path, meta_v2, stats):
    h, w = image.shape[:2]
    source_root = args.source_dataset
    for obj in meta_v2.get("objects", []):
        if obj.get("object_class") != "cube_like_object":
            continue
        bbox = obj.get("visible_bbox_xyxy") or obj.get("full_bbox_xyxy")
        if not bbox:
            continue
        box = expand_box(bbox, w, h, args.a2_crop_pad)
        if box is None:
            continue
        out_stem = f"{image_path.stem}_obj{int(obj['object_id']):03d}_{obj['object_name']}"
        out_image = args.output_root / "a2_faceseg" / "images" / split / f"{out_stem}.jpg"
        object_mask = read_mask(source_root, obj.get("visible_mask"))
        source_image = apply_binary_mask(image, object_mask, args.occlusion_fill) if args.mask_a2_target_object else image
        crop = crop_resize(source_image, box, args.crop_size)
        if crop is None:
            continue
        if not (args.resume and out_image.exists()):
            save_image(out_image, crop)
        lines = []
        for face in obj.get("faces", []):
            if int(face.get("visible_pixels") or 0) < args.min_face_pixels:
                continue
            mask = read_mask(source_root, face.get("visible_mask"))
            if mask is None:
                continue
            segments = mask_crop_to_segments(mask, box, args.crop_size)
            for segment in segments:
                lines.append(f"{int(face['a2_class_id'])} " + " ".join(f"{float(v):.6f}" for v in segment))
                stats[f"a2_{A2_NAMES[int(face['a2_class_id'])]}"] += 1
        label_path = args.output_root / "a2_faceseg" / "labels" / split / f"{out_stem}.txt"
        if not (args.resume and label_path.exists()):
            ensure_parent(label_path)
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        stats["a2_crops"] += 1


def export_b(args, split, image, image_path, meta_v2, stats):
    h, w = image.shape[:2]
    source_root = args.source_dataset
    for obj in meta_v2.get("objects", []):
        if obj.get("object_class") != "cube_like_object":
            continue
        object_mask = read_mask(source_root, obj.get("visible_mask"))
        for face in obj.get("faces", []):
            if int(face.get("visible_pixels") or 0) < args.min_face_pixels:
                continue
            face_mask = read_mask(source_root, face.get("visible_mask"))
            if face_mask is None:
                continue
            quad = face.get("quad_xy")
            if not quad or len(quad) != 4:
                continue
            quad = quad_passes(quad, args.b_min_quad_area, args.b_min_quad_side, args.b_max_quad_aspect)
            if quad is None:
                stats["b_skipped_bad_quad"] += 1
                continue
            box = expand_box(bbox_from_points(quad), w, h, args.b_crop_pad)
            if box is None:
                continue
            if args.b_input_mode == "mask":
                source_image = np.zeros_like(image)
                source_image[face_mask.astype(bool)] = 255
                stats["b_mask_input"] += 1
            else:
                source_image = apply_binary_mask(image, object_mask, args.occlusion_fill) if args.mask_b_target_object else image
                if args.b_input_mode == "raw_blacken_occlusion" and args.blacken_b_occluded_face_area and face.get("occluder_object_ids"):
                    source_image = blacken_hidden_face_area(source_image, face_mask, quad, args.occlusion_fill)
                    stats["b_blackened_occluded_face"] += 1
            crop = crop_resize(source_image, box, args.crop_size)
            if crop is None:
                continue
            out_stem = f"{image_path.stem}_obj{int(obj['object_id']):03d}_face{int(face['face_id']):02d}"
            out_image = args.output_root / "b_facequad" / "images" / split / f"{out_stem}.jpg"
            if not (args.resume and out_image.exists()):
                save_image(out_image, crop)
            quad_norm = [normalize_point_in_box(point, box) for point in quad]
            bbox_norm = normalized_bbox_from_points(quad_norm)
            kpts = []
            for point, visible in zip(quad_norm, face.get("corner_visibility") or [True] * 4):
                kpts.extend([point[0], point[1], 2.0 if visible else 1.0])
            line = "0 " + " ".join(f"{float(v):.6f}" for v in [*bbox_norm, *kpts])
            label_path = args.output_root / "b_facequad" / "labels" / split / f"{out_stem}.txt"
            if not (args.resume and label_path.exists()):
                ensure_parent(label_path)
                label_path.write_text(line + "\n", encoding="utf-8")
            stats[f"b_{face.get('quality', 'unknown')}"] += 1
            stats["b_crops"] += 1


def c_class_for_face(face):
    if face.get("face_kind") == "plain_face":
        return "plain"
    fruit_class = face.get("fruit_class")
    return fruit_class if fruit_class in {"apple", "orange", "banana", "pineapple"} else "unknown"


def export_c(args, split, image, image_path, meta_v2, stats):
    source_root = args.source_dataset
    for obj in meta_v2.get("objects", []):
        if obj.get("object_class") != "cube_like_object":
            continue
        for face in obj.get("faces", []):
            quality = face.get("quality")
            if quality == "bad" and not args.c_include_bad_faces:
                stats["c_skipped_bad_quality"] += 1
                continue
            if quality == "partial" and not args.c_include_partial_faces:
                stats["c_skipped_partial_or_bad"] += 1
                continue
            if quality == "partial" and args.c_partial_requires_occluder and not face.get("occluder_object_ids"):
                stats["c_skipped_partial_without_occluder"] += 1
                continue
            if int(face.get("visible_pixels") or 0) < max(args.min_face_pixels, args.c_min_visible_pixels):
                stats["c_skipped_too_few_visible_pixels"] += 1
                continue
            quad = face.get("quad_xy")
            if not quad or len(quad) != 4:
                continue
            quad = quad_passes(quad, args.c_min_quad_area, args.c_min_quad_side, args.c_max_quad_aspect)
            if quad is None:
                stats["c_skipped_bad_quad"] += 1
                continue
            visible_pixels = float(face.get("visible_pixels") or 0)
            full_pixels = float(face.get("full_pixels") or 0)
            if full_pixels < args.min_face_pixels:
                stats["c_skipped_tiny_full_face"] += 1
                continue
            if full_pixels > 0 and visible_pixels / full_pixels > args.c_max_visible_to_full_ratio:
                stats["c_skipped_inconsistent_visible_full"] += 1
                continue
            visible_ratio = float(face.get("visible_ratio") or 0)
            if visible_ratio < args.c_min_visible_ratio:
                stats["c_skipped_low_visible_ratio"] += 1
                continue
            label = c_class_for_face(face)
            if label == "unknown" and not args.include_partial_as_unknown:
                stats["c_skipped_unknown_label"] += 1
                continue
            face_mask = read_mask(source_root, face.get("visible_mask"))
            if args.mask_c_visible_face:
                source_image = apply_binary_mask(image, face_mask, args.occlusion_fill)
            elif args.blacken_c_occluded_face_area and face.get("occluder_object_ids"):
                source_image = blacken_hidden_face_area(image, face_mask, quad, args.occlusion_fill)
                stats["c_blackened_occluded_face"] += 1
            else:
                source_image = image
            crop = warp_quad(source_image, quad, args.crop_size)
            if crop is None or crop.size == 0:
                continue
            black_fraction = float(((crop[:, :, 0] < 8) & (crop[:, :, 1] < 8) & (crop[:, :, 2] < 8)).mean())
            if black_fraction > args.c_max_black_fraction:
                stats["c_skipped_too_much_black"] += 1
                continue
            out_stem = f"{image_path.stem}_obj{int(obj['object_id']):03d}_face{int(face['face_id']):02d}_{label}.jpg"
            out_path = args.output_root / "c_facecls" / split / label / out_stem
            if not (args.resume and out_path.exists()):
                save_image(out_path, crop)
            stats[f"c_{label}"] += 1


def init_outputs(output_root):
    write_yaml(output_root / "a1_objectseg" / "data.yaml", A1_NAMES)
    write_yaml(output_root / "a2_faceseg" / "data.yaml", A2_NAMES)
    write_yaml(
        output_root / "b_facequad" / "data.yaml",
        {0: "face_quad"},
        extra="kpt_shape: [4, 3]",
    )
    for split in ("train", "val", "test"):
        for class_name in C_CLASSES:
            (output_root / "c_facecls" / split / class_name).mkdir(parents=True, exist_ok=True)


def process_split(args, split):
    meta_dir = args.source_dataset / "_meta" / split
    meta_files = sorted(meta_dir.glob("*.json"))
    if args.limit > 0:
        meta_files = meta_files[:args.limit]
    stats = Counter()
    for meta_path in tqdm(meta_files, desc=f"{split} meta_v2 export", unit="img"):
        meta = read_json(meta_path)
        meta_v2 = (meta or {}).get("meta_v2")
        if not meta_v2:
            stats["missing_meta_v2"] += 1
            continue
        image_rel = meta_v2.get("image_path") or f"images/{split}/{meta_path.stem}.jpg"
        image_path = args.source_dataset / image_rel
        if not image_path.exists():
            stats["missing_image"] += 1
            continue
        image = load_image(image_path)
        export_a1(args, split, image_path, meta_v2, stats)
        export_a2(args, split, image, image_path, meta_v2, stats)
        export_b(args, split, image, image_path, meta_v2, stats)
        export_c(args, split, image, image_path, meta_v2, stats)
        stats["images"] += 1
    return dict(stats)


def main():
    args = parse_args()
    args.source_dataset = args.source_dataset.resolve()
    args.output_root = args.output_root.resolve()
    if args.reset:
        reset_output(args.output_root)
    init_outputs(args.output_root)
    audit = {
        "source_dataset": str(args.source_dataset),
        "output_root": str(args.output_root),
        "splits": {},
        "classes": {
            "a1": A1_NAMES,
            "a2": A2_NAMES,
            "b": {0: "face_quad"},
            "c": C_CLASSES,
        },
    }
    for split in args.splits:
        audit["splits"][split] = process_split(args, split)
    audit_path = args.output_root / "meta_v2_export_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(f"wrote: {args.output_root}")


if __name__ == "__main__":
    main()
