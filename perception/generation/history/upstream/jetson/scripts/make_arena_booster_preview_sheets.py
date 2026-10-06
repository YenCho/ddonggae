import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


def load_meta(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def title_bar(width, text, height=26):
    bar = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(bar, text[:90], (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
    return bar


def read_thumb(image_path, title, thumb):
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = thumb / max(h, w)
    resized = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((thumb, thumb, 3), 255, dtype=np.uint8)
    y = (thumb - resized.shape[0]) // 2
    x = (thumb - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return np.vstack([title_bar(thumb, title), canvas])


def make_sheet(items, output, thumb=180, cols=5):
    if not items:
        return
    cells = [read_thumb(path, title, thumb) for path, title in items]
    cells = [cell for cell in cells if cell is not None]
    if not cells:
        return
    rows = []
    cell_h = cells[0].shape[0]
    for start in range(0, len(cells), cols):
        row_cells = cells[start:start + cols]
        while len(row_cells) < cols:
            row_cells.append(np.full((cell_h, thumb, 3), 255, dtype=np.uint8))
        rows.append(np.hstack(row_cells))
    sheet = np.vstack(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def collect(dataset, split):
    image_dir = dataset / "images" / split
    meta_dir = dataset / "_meta" / split
    rows = []
    for image in sorted(image_dir.glob("*.jpg")):
        meta = load_meta(meta_dir / f"{image.stem}.json")
        rows.append((image, meta))
    return rows


def has_fruit(meta, names):
    return bool(set(meta.get("scene_fruit_classes", [])) & set(names))


def has_visible_face(meta):
    for obj in meta.get("fruit_objects", []):
        if obj.get("fruit_visibility_pass"):
            return True
    return False


def sample(items, count, seed):
    rng = random.Random(seed)
    if len(items) > count:
        return rng.sample(items, count)
    return items


def parse_args():
    parser = argparse.ArgumentParser(description="Make arena booster QA contact sheets.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--thumb", type=int, default=180)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260530)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = args.dataset.resolve()
    out = args.output_dir or (dataset / "preview_sheets")
    rows = collect(dataset, args.split)

    arena = [(img, meta) for img, meta in rows if str(meta.get("background", "")).startswith("arena_")]
    plain_cube = [(img, meta) for img, meta in rows if meta.get("arena_scene_profile") == "plain_cube_hard_negative"]
    apple_orange = [(img, meta) for img, meta in rows if has_fruit(meta, {"apple", "orange"})]
    fruit_face = [(img, meta) for img, meta in rows if has_visible_face(meta)]

    groups = {
        "sun111_sun168_arena": arena,
        "plain_cube_hard_negative": plain_cube,
        "apple_orange_confusion": apple_orange,
        "fruit_face_visibility": fruit_face,
    }
    for idx, (name, group) in enumerate(groups.items()):
        picked = sample(group, args.count, args.seed + idx)
        items = [
            (img, f"{img.stem} {meta.get('arena_scene_profile', '')} {'/'.join(meta.get('scene_fruit_classes', []))}")
            for img, meta in picked
        ]
        make_sheet(items, out / f"{name}.jpg", thumb=args.thumb, cols=args.cols)
        print(f"{name}: {len(group)} candidates, wrote {min(len(group), args.count)}")
    print(f"preview sheets: {out}")


if __name__ == "__main__":
    main()
