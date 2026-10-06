import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


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
    parser = argparse.ArgumentParser(description="Create A1/A2/B/C preview sheets from exported Meta V2 model datasets.")
    parser.add_argument("--model_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--c_per_class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--panel_size", type=int, default=220)
    parser.add_argument("--cols", type=int, default=6)
    return parser.parse_args()


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    return image


def fit_square(image, size):
    h, w = image.shape[:2]
    scale = min(size / max(w, 1), size / max(h, 1))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 246, dtype=np.uint8)
    x = (size - nw) // 2
    y = (size - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def draw_text_box(image, lines, origin=(6, 18), scale=0.45):
    out = image.copy()
    x, y = origin
    line_h = int(22 * scale / 0.45)
    box_h = max(24, line_h * len(lines) + 8)
    box_w = min(out.shape[1] - 4, max(80, max(len(line) for line in lines) * 8 + 12))
    overlay = out.copy()
    cv2.rectangle(overlay, (x - 3, max(0, y - 16)), (x - 3 + box_w, max(0, y - 16) + box_h), (255, 255, 255), -1)
    out = cv2.addWeighted(overlay, 0.72, out, 0.28, 0)
    for idx, line in enumerate(lines):
        cv2.putText(out, line, (x, y + idx * line_h), cv2.FONT_HERSHEY_SIMPLEX, scale, (25, 25, 25), 1, cv2.LINE_AA)
    return out


def make_panel(image, title, subtitle, size):
    header = 54
    panel = np.full((size + header, size, 3), 255, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (size - 1, size + header - 1), (210, 210, 210), 1)
    cv2.putText(panel, title[:28], (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(panel, subtitle[:34], (8, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70, 70, 70), 1, cv2.LINE_AA)
    panel[header:header + size, 0:size] = fit_square(image, size)
    return panel


def make_sheet(panels, cols, fill=(250, 250, 250)):
    if not panels:
        empty = np.full((260, 520, 3), fill, dtype=np.uint8)
        cv2.putText(empty, "No preview samples", (36, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (70, 70, 70), 2, cv2.LINE_AA)
        return empty
    ph, pw = panels[0].shape[:2]
    rows = int(np.ceil(len(panels) / cols))
    sheet = np.full((rows * ph, cols * pw, 3), fill, dtype=np.uint8)
    for idx, panel in enumerate(panels):
        r = idx // cols
        c = idx % cols
        sheet[r * ph:(r + 1) * ph, c * pw:(c + 1) * pw] = panel
    return sheet


def label_path_for_image(model_root, model_name, split, image_path):
    return model_root / model_name / "labels" / split / f"{image_path.stem}.txt"


def denorm_segment(values, width, height):
    pts = np.asarray(values, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] = np.clip(pts[:, 0], 0, 1) * (width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, 1) * (height - 1)
    return np.rint(pts).astype(np.int32)


def draw_seg_overlay(image, label_path, names, colors):
    out = image.copy()
    overlay = image.copy()
    h, w = image.shape[:2]
    counts = {}
    if not label_path.exists():
        return draw_text_box(out, ["missing label"])
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        class_id = int(float(parts[0]))
        values = [float(v) for v in parts[1:]]
        if len(values) < 6 or len(values) % 2 != 0:
            continue
        pts = denorm_segment(values, w, h)
        color = colors[class_id % len(colors)]
        cv2.fillPoly(overlay, [pts.reshape(-1, 1, 2)], color)
        cv2.polylines(out, [pts.reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
        x, y = pts[0]
        cv2.putText(out, names.get(class_id, str(class_id)), (int(x), max(16, int(y) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, color, 1, cv2.LINE_AA)
        counts[class_id] = counts.get(class_id, 0) + 1
    out = cv2.addWeighted(overlay, 0.32, out, 0.68, 0)
    summary = [f"{names.get(k, k)}:{v}" for k, v in sorted(counts.items())]
    if summary:
        out = draw_text_box(out, summary[:3])
    return out


def select_images(image_dir, label_dir, count, rng, prefer_nonempty=True):
    paths = sorted(image_dir.glob("*.jpg"))
    if prefer_nonempty:
        nonempty = []
        empty = []
        for path in paths:
            label_path = label_dir / f"{path.stem}.txt"
            if label_path.exists() and label_path.read_text(encoding="utf-8").strip():
                nonempty.append(path)
            else:
                empty.append(path)
        paths = nonempty + empty
    if len(paths) > count:
        first = paths[: min(len(paths), count // 2)]
        rest = paths[min(len(paths), count // 2):]
        rng.shuffle(rest)
        paths = first + rest[: count - len(first)]
    return paths[:count]


def make_a1_preview(model_root, output_dir, split, count, rng, panel_size, cols):
    image_dir = model_root / "a1_objectseg" / "images" / split
    label_dir = model_root / "a1_objectseg" / "labels" / split
    images = select_images(image_dir, label_dir, count, rng)
    colors = [(40, 220, 40), (255, 190, 0), (255, 0, 180), (0, 190, 255)]
    panels = []
    for path in images:
        image = load_image(path)
        overlay = draw_seg_overlay(image, label_path_for_image(model_root, "a1_objectseg", split, path), A1_NAMES, colors)
        panels.append(make_panel(overlay, path.stem, "full image object seg", panel_size))
    sheet = make_sheet(panels, cols)
    out = output_dir / "a1_objectseg_preview.jpg"
    cv2.imwrite(str(out), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    return {"path": str(out), "samples": len(images)}


def make_a2_preview(model_root, output_dir, split, count, rng, panel_size, cols):
    image_dir = model_root / "a2_faceseg" / "images" / split
    label_dir = model_root / "a2_faceseg" / "labels" / split
    images = select_images(image_dir, label_dir, count, rng)
    colors = [(235, 235, 235), (0, 220, 255)]
    panels = []
    for path in images:
        image = load_image(path)
        overlay = draw_seg_overlay(image, label_path_for_image(model_root, "a2_faceseg", split, path), A2_NAMES, colors)
        panels.append(make_panel(overlay, path.stem, "cube crop face seg", panel_size))
    sheet = make_sheet(panels, cols)
    out = output_dir / "a2_faceseg_preview.jpg"
    cv2.imwrite(str(out), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    return {"path": str(out), "samples": len(images)}


def parse_b_label(label_path):
    if not label_path.exists():
        return None
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    parts = [float(v) for v in text.split()]
    if len(parts) != 17:
        return None
    kpts = parts[5:17]
    points = []
    for idx in range(0, len(kpts), 3):
        points.append((kpts[idx], kpts[idx + 1], kpts[idx + 2]))
    return points


def draw_b_overlay(image, label_path):
    out = image.copy()
    points = parse_b_label(label_path)
    if not points:
        return draw_text_box(out, ["missing quad"])
    h, w = image.shape[:2]
    pts = np.asarray([[p[0] * (w - 1), p[1] * (h - 1)] for p in points], dtype=np.int32)
    cv2.polylines(out, [pts.reshape(-1, 1, 2)], True, (0, 255, 255), 2, cv2.LINE_AA)
    hidden = 0
    for idx, (x_norm, y_norm, vis) in enumerate(points):
        x = int(round(x_norm * (w - 1)))
        y = int(round(y_norm * (h - 1)))
        if vis >= 2:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
            hidden += 1
        cv2.circle(out, (x, y), 5, color, -1, cv2.LINE_AA)
        cv2.putText(out, str(idx + 1), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return draw_text_box(out, [f"hidden corners:{hidden}"])


def make_b_preview(model_root, output_dir, split, count, rng, panel_size, cols):
    image_dir = model_root / "b_facequad" / "images" / split
    label_dir = model_root / "b_facequad" / "labels" / split
    images = sorted(image_dir.glob("*.jpg"))
    hidden = []
    normal = []
    for path in images:
        points = parse_b_label(label_dir / f"{path.stem}.txt")
        if points and any(p[2] < 2 for p in points):
            hidden.append(path)
        else:
            normal.append(path)
    rng.shuffle(hidden)
    rng.shuffle(normal)
    selected = (hidden[: max(1, count // 2)] + normal)[:count]
    panels = []
    for path in selected:
        image = load_image(path)
        overlay = draw_b_overlay(image, label_dir / f"{path.stem}.txt")
        panels.append(make_panel(overlay, path.stem, "mask input + amodal quad", panel_size))
    sheet = make_sheet(panels, cols)
    out = output_dir / "b_facequad_preview.jpg"
    cv2.imwrite(str(out), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    return {"path": str(out), "samples": len(selected), "hidden_corner_candidates": len(hidden)}


def make_c_preview(model_root, output_dir, split, c_per_class, rng, panel_size):
    panels = []
    counts = {}
    for class_name in C_CLASSES:
        paths = sorted((model_root / "c_facecls" / split / class_name).glob("*.jpg"))
        counts[class_name] = len(paths)
        rng.shuffle(paths)
        selected = paths[:c_per_class]
        if not selected:
            blank = np.full((panel_size, panel_size, 3), 238, dtype=np.uint8)
            cv2.putText(blank, "empty", (panel_size // 2 - 42, panel_size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2, cv2.LINE_AA)
            panels.append(make_panel(blank, class_name, "0 crops", panel_size))
            continue
        for path in selected:
            image = load_image(path)
            panels.append(make_panel(image, class_name, path.stem[:34], panel_size))
    sheet = make_sheet(panels, c_per_class if c_per_class > 1 else 1)
    out = output_dir / "c_facecls_preview.jpg"
    cv2.imwrite(str(out), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    return {"path": str(out), "class_counts": counts}


def main():
    args = parse_args()
    args.model_root = args.model_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    report = {
        "model_root": str(args.model_root),
        "output_dir": str(args.output_dir),
        "split": args.split,
        "a1": make_a1_preview(args.model_root, args.output_dir, args.split, args.count, rng, args.panel_size, args.cols),
        "a2": make_a2_preview(args.model_root, args.output_dir, args.split, args.count, rng, args.panel_size, args.cols),
        "b": make_b_preview(args.model_root, args.output_dir, args.split, args.count, rng, args.panel_size, args.cols),
        "c": make_c_preview(args.model_root, args.output_dir, args.split, args.c_per_class, rng, args.panel_size),
    }
    report_path = args.output_dir / "model_previews_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
