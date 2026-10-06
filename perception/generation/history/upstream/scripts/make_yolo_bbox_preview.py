import argparse
import json
from pathlib import Path

import cv2
import numpy as np


NAMES = ["banana", "orange", "pineapple", "apple", "cube", "octahedron", "dodecahedron", "icosahedron"]
FRUIT_CLASSES = {"banana", "orange", "pineapple", "apple"}
SHORT_FRUIT_NAMES = {
    "apple": "app",
    "banana": "ban",
    "orange": "org",
    "pineapple": "pine",
}
COLORS = {
    0: (40, 230, 240),
    1: (40, 150, 255),
    2: (60, 220, 180),
    3: (60, 60, 255),
    4: (255, 70, 70),
    5: (70, 220, 70),
    6: (70, 140, 255),
    7: (210, 90, 255),
}


def draw_label(image, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y0 = max(0, y - th - baseline - 4)
    cv2.rectangle(image, (x, y0), (x + tw + 6, y0 + th + baseline + 5), color, -1)
    cv2.putText(image, text, (x + 3, y0 + th + 1), font, scale, (20, 20, 20), thickness, cv2.LINE_AA)


def fmt_ratio(value):
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def load_meta(meta_path):
    if not meta_path or not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def object_visible_ratio(item):
    return item.get("object_visible_ratio", item.get("visible_object_ratio"))


def fruit_visible_ratio(item):
    return item.get("fruit_visible_ratio", item.get("actual_fruit_visible_ratio"))


def debug_text(name, item):
    if not item:
        return name
    if item.get("source_class"):
        return f"{name}<-{item['source_class']} obj_vis={fmt_ratio(object_visible_ratio(item))}"
    if name in {"banana", "orange", "pineapple", "apple"}:
        tier = item.get("visibility_tier") or "?"
        fruit = fmt_ratio(fruit_visible_ratio(item))
        obj = fmt_ratio(object_visible_ratio(item))
        passed = "OK" if item.get("fruit_visibility_pass") else "DROP"
        return f"{name} {tier} fruit_vis={fruit} obj_vis={obj} {passed}"
    return f"{name} obj_vis={fmt_ratio(object_visible_ratio(item))}"


def object_caption(index, name, item):
    if not item:
        return f"#{index}\t{name}\t-\t-\t-\tno meta"
    if item.get("source_class"):
        return (
            f"#{index}\t{name}\tfrom={item['source_class']}\t-\t"
            f"obj_vis={fmt_ratio(object_visible_ratio(item))}\tfallback"
        )
    if name in {"banana", "orange", "pineapple", "apple"}:
        tier = item.get("visibility_tier") or "?"
        passed = "OK" if item.get("fruit_visibility_pass") else "DROP"
        return (
            f"#{index}\t{name}\ttier={tier}\tfruit_vis={fmt_ratio(fruit_visible_ratio(item))}\t"
            f"obj_vis={fmt_ratio(object_visible_ratio(item))}\t{passed}"
        )
    return f"#{index}\t{name}\t-\t-\tobj_vis={fmt_ratio(object_visible_ratio(item))}\t-"


def draw_text_lines(canvas, lines, x, y, line_height=20):
    font = cv2.FONT_HERSHEY_SIMPLEX
    columns = [x + 16, x + 58, x + 190, x + 300, x + 410, x + 520]
    for line_idx, (text, color) in enumerate(lines):
        yy = y + line_idx * line_height
        swatch_y = yy - 10
        cv2.rectangle(canvas, (x, swatch_y), (x + 10, swatch_y + 10), color, -1)
        cv2.rectangle(canvas, (x, swatch_y), (x + 10, swatch_y + 10), (70, 70, 70), 1)
        for col_idx, value in enumerate(text.split("\t")):
            if col_idx >= len(columns):
                break
            cv2.putText(canvas, value, (columns[col_idx], yy), font, 0.44, (35, 35, 35), 1, cv2.LINE_AA)


def append_caption_panel(image, lines):
    if not lines:
        return image
    h, w = image.shape[:2]
    line_height = 20
    pad = 12
    output_w = max(w, 640)
    if output_w != w:
        padded = np.zeros((h, output_w, 3), dtype=np.uint8)
        padded[:, :w] = image
        image = padded
    panel_h = pad * 2 + line_height * len(lines)
    panel = np.full((panel_h, output_w, 3), 245, dtype=np.uint8)
    draw_text_lines(panel, lines, pad, pad + 12, line_height=line_height)
    return np.vstack([image, panel])


def select_evenly(items, limit):
    if limit <= 0 or len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[0]]
    step = (len(items) - 1) / float(limit - 1)
    return [items[int(round(idx * step))] for idx in range(limit)]


def contact_sheet_summary(meta):
    fruit_objects = [item for item in meta.get("fruit_objects", []) if item.get("class") in FRUIT_CLASSES]
    classes = sorted({item.get("class") for item in fruit_objects})
    if not classes:
        return ""

    source_counts = {"lab": 0, "raw": 0}
    for item in fruit_objects:
        for texture in item.get("face_textures") or []:
            path = Path(str(texture.get("texture") or ""))
            if "_lab" in path.stem:
                source_counts["lab"] += 1
            else:
                source_counts["raw"] += 1
    source_text = []
    if source_counts["lab"]:
        source_text.append(f"lab{source_counts['lab']}")
    if source_counts["raw"]:
        source_text.append(f"raw{source_counts['raw']}")
    short_classes = [SHORT_FRUIT_NAMES.get(name, name) for name in classes]
    return f"{','.join(short_classes)} {'/'.join(source_text)}".strip()


def draw_sheet_label(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    scale = 0.42
    while scale > 0.24:
        (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
        if tw <= image.shape[1] - 8:
            break
        scale -= 0.03
    cv2.rectangle(image, (0, 0), (image.shape[1] - 1, 22), (255, 255, 255), -1)
    cv2.putText(image, text, (3, 16), font, scale, (25, 25, 25), thickness, cv2.LINE_AA)


def make_contact_sheet(dataset, split, image_paths, output_path, sheet_count=100, columns=5, thumb_size=180):
    selected = select_evenly(image_paths, sheet_count)
    if not selected:
        return None
    rows = int(np.ceil(len(selected) / float(columns)))
    canvas = np.full((rows * thumb_size, columns * thumb_size, 3), 245, dtype=np.uint8)
    meta_dir = dataset / "_meta" / split
    for idx, image_path in enumerate(selected):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.resize(image, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
        meta = load_meta(meta_dir / f"{image_path.stem}.json")
        summary = contact_sheet_summary(meta)
        label = image_path.stem if not summary else f"{image_path.stem} {summary}"
        draw_sheet_label(image, label)
        y = (idx // columns) * thumb_size
        x = (idx % columns) * thumb_size
        canvas[y:y + thumb_size, x:x + thumb_size] = image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return output_path


def shade_visible_fruit_faces(image, meta):
    h, w = image.shape[:2]
    overlay = image.copy()
    for item in meta.get("fruit_objects", []):
        segments = item.get("visible_face_segments") or []
        if not segments:
            continue
        color = (40, 240, 255) if item.get("fruit_visibility_pass") else (40, 80, 255)
        for segment in segments:
            coords = np.array(segment, dtype=np.float32).reshape(-1, 2)
            if len(coords) < 3:
                continue
            pts = np.empty_like(coords, dtype=np.int32)
            pts[:, 0] = np.clip(np.rint(coords[:, 0] * w), 0, w - 1).astype(np.int32)
            pts[:, 1] = np.clip(np.rint(coords[:, 1] * h), 0, h - 1).astype(np.int32)
            cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.22, image, 0.78, 0, image)


def draw_one(image_path, label_path, output_path, meta_path=None, debug_visibility=False):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    h, w = image.shape[:2]
    overlay = image.copy()
    meta = load_meta(meta_path)
    label_meta = meta.get("label_objects", []) if debug_visibility else []
    if label_path.exists():
        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        lines = []

    label_positions = []
    caption_lines = []
    for line_idx, line in enumerate(lines):
        parts = line.split()
        cls = int(float(parts[0]))
        color = COLORS.get(cls, (255, 255, 255))
        name = NAMES[cls] if 0 <= cls < len(NAMES) else str(cls)
        item = label_meta[line_idx] if line_idx < len(label_meta) else {}
        display_idx = line_idx + 1
        text = f"#{display_idx} {name}" if debug_visibility else name
        if debug_visibility:
            caption_lines.append((object_caption(display_idx, name, item), color))
        if len(parts) == 5:
            xc, yc, bw, bh = [float(v) for v in parts[1:]]
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label_positions.append((text, x1, y1, color))
        elif len(parts) >= 7 and len(parts[1:]) % 2 == 0:
            coords = np.array([float(v) for v in parts[1:]], dtype=np.float32).reshape(-1, 2)
            pts = np.empty_like(coords, dtype=np.int32)
            pts[:, 0] = np.clip(np.rint(coords[:, 0] * w), 0, w - 1).astype(np.int32)
            pts[:, 1] = np.clip(np.rint(coords[:, 1] * h), 0, h - 1).astype(np.int32)
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(image, [pts], True, color, 2, cv2.LINE_AA)
            x1, y1 = pts.min(axis=0)
            label_positions.append((text, int(x1), int(y1), color))

    cv2.addWeighted(overlay, 0.28, image, 0.72, 0, image)
    if debug_visibility:
        shade_visible_fruit_faces(image, meta)
    for text, x, y, color in label_positions:
        draw_label(image, text, x, y, color)
    if debug_visibility:
        image = append_caption_panel(image, caption_lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--debug_visibility", action="store_true")
    parser.add_argument("--contact_sheet", action="store_true", default=True)
    parser.add_argument("--no_contact_sheet", dest="contact_sheet", action="store_false")
    parser.add_argument("--sheet_count", type=int, default=100)
    parser.add_argument("--sheet_columns", type=int, default=5)
    parser.add_argument("--sheet_thumb_size", type=int, default=180)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    image_dir = dataset / "images" / args.split
    label_dir = dataset / "labels" / args.split
    meta_dir = dataset / "_meta" / args.split
    default_preview = "visibility_preview" if args.debug_visibility else "bbox_preview"
    output_dir = Path(args.output) if args.output else dataset / default_preview / args.split

    images = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))[:args.count]
    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        meta_path = meta_dir / f"{image_path.stem}.json"
        suffix = "_visibility.jpg" if args.debug_visibility else "_bbox.jpg"
        draw_one(image_path, label_path, output_dir / f"{image_path.stem}{suffix}", meta_path, args.debug_visibility)

    sheet_path = None
    if args.contact_sheet:
        sheet_path = make_contact_sheet(
            dataset,
            args.split,
            images,
            output_dir / "_contact_sheet.jpg",
            sheet_count=args.sheet_count,
            columns=max(1, args.sheet_columns),
            thumb_size=max(64, args.sheet_thumb_size),
        )

    print(f"preview images: {len(images)}")
    print(f"preview dir: {output_dir}")
    if sheet_path:
        print(f"contact sheet: {sheet_path}")


if __name__ == "__main__":
    main()
