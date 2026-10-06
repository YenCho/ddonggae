import argparse
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# Export 224px cube crops + YOLO-seg face labels from a mimic-arena probe
# source rendered by scripts/render_mimic_arena_scene.py (full frames +
# per-scene metadata/*.json with projected face polygons and ray-cast
# visibility). Convention follows export_meta_v2_cube_face_unified_dataset.py:
# one crop per sufficiently visible cube, labels only for that cube's faces,
# classes 0 apple / 1 orange / 2 banana / 3 pineapple / 4 plain.

FACE_CLASSES = ("apple", "orange", "banana", "pineapple", "plain")
FACE_CLASS_TO_ID = {name: idx for idx, name in enumerate(FACE_CLASSES)}
VERIFY_COLORS = {
    "apple": (60, 60, 230),
    "orange": (40, 150, 255),
    "banana": (60, 220, 240),
    "pineapple": (90, 200, 90),
    "plain": (200, 200, 200),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Export 224 cube-face crops from an arena mimic probe source.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--crop_pad", type=float, default=0.18)
    parser.add_argument("--min_bbox_area", type=float, default=900.0)
    parser.add_argument("--min_face_area", type=float, default=120.0, help="At crop (224) scale, px^2.")
    parser.add_argument("--min_cube_visible", type=float, default=0.25)
    parser.add_argument("--min_face_visible", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--verify_count", type=int, default=8)
    parser.add_argument("--verify_out", type=Path, default=None)
    parser.add_argument("--verify_only", action="store_true")
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def write_data_yaml(output_root: Path) -> None:
    lines = [
        f"path: {output_root.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for idx, name in enumerate(FACE_CLASSES):
        lines.append(f"  {idx}: {name}")
    (output_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def clip_polygon(poly, x_min, y_min, x_max, y_max):
    """Sutherland-Hodgman clip of a polygon against an axis-aligned rect."""

    def clip_edge(points, inside, intersect):
        out = []
        n = len(points)
        for i in range(n):
            cur = points[i]
            prev = points[i - 1]
            cur_in = inside(cur)
            prev_in = inside(prev)
            if cur_in:
                if not prev_in:
                    out.append(intersect(prev, cur))
                out.append(cur)
            elif prev_in:
                out.append(intersect(prev, cur))
        return out

    def x_cross(p, q, x):
        t = (x - p[0]) / (q[0] - p[0]) if q[0] != p[0] else 0.0
        return [x, p[1] + t * (q[1] - p[1])]

    def y_cross(p, q, y):
        t = (y - p[1]) / (q[1] - p[1]) if q[1] != p[1] else 0.0
        return [p[0] + t * (q[0] - p[0]), y]

    pts = [list(p) for p in poly]
    for inside, intersect in (
        (lambda p: p[0] >= x_min, lambda p, q: x_cross(p, q, x_min)),
        (lambda p: p[0] <= x_max, lambda p, q: x_cross(p, q, x_max)),
        (lambda p: p[1] >= y_min, lambda p, q: y_cross(p, q, y_min)),
        (lambda p: p[1] <= y_max, lambda p, q: y_cross(p, q, y_max)),
    ):
        if not pts:
            return []
        pts = clip_edge(pts, inside, intersect)
    return pts


def polygon_area(pts) -> float:
    if len(pts) < 3:
        return 0.0
    area = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def crop_square_replicate(image, cx, cy, side):
    """Square crop centered at (cx, cy); out-of-image regions replicate-padded.
    Returns (crop, x0, y0) with x0/y0 the crop origin in image coordinates."""
    h, w = image.shape[:2]
    side = max(2, int(round(side)))
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x1, y1 = x0 + side, y0 + side
    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - w)
    pad_b = max(0, y1 - h)
    sub = image[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if sub.size == 0:
        return None, x0, y0
    if pad_l or pad_t or pad_r or pad_b:
        sub = cv2.copyMakeBorder(sub, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
    return sub, x0, y0


def export_crops(args) -> Counter:
    meta_dir = args.source / "metadata"
    meta_files = sorted(meta_dir.glob("*.json"))
    if not meta_files:
        raise RuntimeError(f"no metadata under {meta_dir}")
    image_dir = args.output / "images" / args.split
    label_dir = args.output / "labels" / args.split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    stats["scenes"] = len(meta_files)
    for meta_path in meta_files:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        image_path = args.source / meta["image"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            stats["missing_image"] += 1
            continue
        h, w = image.shape[:2]
        for cube in meta.get("cubes", []):
            if cube.get("behind_camera"):
                stats["skipped_behind_camera"] += 1
                continue
            bx0, by0, bx1, by1 = cube["bbox"]
            cx0, cy0 = max(0.0, bx0), max(0.0, by0)
            cx1, cy1 = min(float(w), bx1), min(float(h), by1)
            bw, bh = cx1 - cx0, cy1 - cy0
            if bw <= 0 or bh <= 0 or bw * bh < args.min_bbox_area:
                stats["skipped_small_bbox"] += 1
                continue
            if float(cube.get("visible_frac", 0.0)) < args.min_cube_visible:
                stats["skipped_occluded_cube"] += 1
                continue

            side = max(bw, bh) * (1.0 + 2.0 * args.crop_pad)
            crop, x0, y0 = crop_square_replicate(
                image, (cx0 + cx1) / 2.0, (cy0 + cy1) / 2.0, side)
            if crop is None:
                stats["skipped_empty_crop"] += 1
                continue
            side_px = crop.shape[0]
            scale = args.crop_size / float(side_px)

            lines = []
            for face in cube.get("faces", []):
                if not face.get("front_facing"):
                    continue
                if float(face.get("visible_frac", 0.0)) < args.min_face_visible:
                    stats["skipped_occluded_face"] += 1
                    continue
                local = [[p[0] - x0, p[1] - y0] for p in face["poly"]]
                clipped = clip_polygon(local, 0.0, 0.0, float(side_px), float(side_px))
                if len(clipped) < 3:
                    stats["skipped_out_of_crop_face"] += 1
                    continue
                area_crop_scale = polygon_area(clipped) * scale * scale
                if area_crop_scale < args.min_face_area:
                    stats["skipped_small_face"] += 1
                    continue
                label = face["face_class"]
                class_id = FACE_CLASS_TO_ID[label]
                coords = []
                for px, py in clipped:
                    coords.append(min(1.0, max(0.0, px / side_px)))
                    coords.append(min(1.0, max(0.0, py / side_px)))
                lines.append(f"{class_id} " + " ".join(f"{v:.6f}" for v in coords))
                stats[f"instances_{label}"] += 1

            if not lines:
                stats["skipped_empty_label_crop"] += 1
                continue
            out_stem = f"{meta_path.stem}_{cube['name']}"
            resized = cv2.resize(crop, (args.crop_size, args.crop_size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(image_dir / f"{out_stem}.jpg"), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
            (label_dir / f"{out_stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            stats["crops"] += 1
            crop_kind = cube["class"] if cube["kind"] == "fruit_cube" else "plain_cube"
            stats[f"crops_{crop_kind}"] += 1
    return stats


def make_verify_sheet(args) -> Path:
    image_dir = args.output / "images" / args.split
    label_dir = args.output / "labels" / args.split
    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        raise RuntimeError(f"no crops to verify under {image_dir}")
    rng = random.Random(args.seed)
    picks = rng.sample(images, min(args.verify_count, len(images)))
    tile = 360
    cells = []
    for path in picks:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_NEAREST)
        label_path = label_dir / f"{path.stem}.txt"
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 7:
                continue
            class_id = int(float(parts[0]))
            name = FACE_CLASSES[class_id]
            pts = np.array([
                [float(parts[i]) * tile, float(parts[i + 1]) * tile]
                for i in range(1, len(parts) - 1, 2)
            ], dtype=np.int32)
            color = VERIFY_COLORS[name]
            cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
            cx, cy = pts.mean(axis=0).astype(int)
            cv2.putText(img, name, (max(2, cx - 40), max(14, cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, name, (max(2, cx - 40), max(14, cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        bar = np.full((26, tile, 3), 245, dtype=np.uint8)
        cv2.putText(bar, path.stem[-40:], (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
        cells.append(np.vstack([bar, img]))
    cols = 4
    rows_n = int(math.ceil(len(cells) / cols))
    ch, cw = cells[0].shape[:2]
    sheet = np.full((rows_n * ch, cols * cw, 3), 245, dtype=np.uint8)
    for idx, cell in enumerate(cells):
        r, c = divmod(idx, cols)
        sheet[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = cell
    out = args.verify_out or (args.output / "verify_label_sheet.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    return out


def main() -> None:
    args = parse_args()
    if not args.verify_only:
        if args.reset and args.output.exists():
            shutil.rmtree(args.output)
        args.output.mkdir(parents=True, exist_ok=True)
        write_data_yaml(args.output)
        stats = export_crops(args)
        (args.output / "manifest.json").write_text(
            json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "stats": dict(stats)},
                       ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(dict(stats), ensure_ascii=False, indent=2))
    if args.verify_count > 0:
        out = make_verify_sheet(args)
        print(f"verify sheet: {out}")


if __name__ == "__main__":
    main()
