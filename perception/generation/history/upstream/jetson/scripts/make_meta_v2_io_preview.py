import argparse
import json
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create one A1/A2/B/C input-output preview from a Meta V2 dataset."
    )
    parser.add_argument("--source_dataset", type=Path, required=True)
    parser.add_argument("--model_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--image_id", default="")
    parser.add_argument("--prefer_occlusion", action="store_true")
    parser.add_argument("--panel_size", type=int, default=360)
    return parser.parse_args()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    return image


def c_class_for_face(face):
    if face.get("face_kind") == "plain_face":
        return "plain"
    fruit_class = face.get("fruit_class")
    if fruit_class in {"apple", "orange", "banana", "pineapple"}:
        return fruit_class
    return "unknown"


def a2_stem(image_stem, obj):
    return f"{image_stem}_obj{int(obj['object_id']):03d}_{obj['object_name']}"


def b_stem(image_stem, obj, face):
    return f"{image_stem}_obj{int(obj['object_id']):03d}_face{int(face['face_id']):02d}"


def c_path(model_root, split, image_stem, obj, face):
    label = c_class_for_face(face)
    name = f"{b_stem(image_stem, obj, face)}_{label}.jpg"
    return model_root / "c_facecls" / split / label / name


def count_fanout(model_root, split, image_stem, obj=None):
    a2_dir = model_root / "a2_faceseg" / "images" / split
    b_dir = model_root / "b_facequad" / "images" / split
    c_dir = model_root / "c_facecls" / split

    if obj is None:
        prefix = f"{image_stem}_"
    else:
        prefix = f"{image_stem}_obj{int(obj['object_id']):03d}_"

    a2_count = len(list(a2_dir.glob(f"{prefix}*.jpg")))
    b_count = len(list(b_dir.glob(f"{prefix}*.jpg")))
    c_count = 0
    for class_dir in c_dir.glob("*"):
        if class_dir.is_dir():
            c_count += len(list(class_dir.glob(f"{prefix}*.jpg")))
    return {
        "a2_crops": a2_count,
        "b_face_samples": b_count,
        "c_face_crops": c_count,
        "c_filtered_from_b": max(0, b_count - c_count),
    }


def summarize_dataset(source_dataset, split):
    meta_files = sorted((source_dataset / "_meta" / split).glob("*.json"))
    summary = {
        "images": len(meta_files),
        "cube_objects": 0,
        "faces": 0,
        "occluded_faces": 0,
        "partial_faces": 0,
        "good_faces": 0,
        "bad_faces": 0,
    }
    for meta_path in meta_files:
        meta = read_json(meta_path)
        meta_v2 = meta.get("meta_v2") or {}
        for obj in meta_v2.get("objects", []):
            if obj.get("object_class") != "cube_like_object":
                continue
            summary["cube_objects"] += 1
            for face in obj.get("faces", []):
                summary["faces"] += 1
                if face.get("occluder_object_ids"):
                    summary["occluded_faces"] += 1
                quality = face.get("quality")
                if quality in {"good", "partial", "bad"}:
                    summary[f"{quality}_faces"] += 1
    return summary


def find_case(args):
    meta_dir = args.source_dataset / "_meta" / args.split
    if args.image_id:
        meta_files = [meta_dir / f"{Path(args.image_id).stem}.json"]
    else:
        meta_files = sorted(meta_dir.glob("*.json"))

    candidates = []
    for meta_path in meta_files:
        if not meta_path.exists():
            continue
        meta = read_json(meta_path)
        meta_v2 = meta.get("meta_v2") or {}
        image_rel = meta_v2.get("image_path") or f"images/{args.split}/{meta_path.stem}.jpg"
        image_path = args.source_dataset / image_rel
        image_stem = image_path.stem
        for obj in meta_v2.get("objects", []):
            if obj.get("object_class") != "cube_like_object":
                continue
            a2_image = args.model_root / "a2_faceseg" / "images" / args.split / f"{a2_stem(image_stem, obj)}.jpg"
            if not a2_image.exists():
                continue
            for face in obj.get("faces", []):
                b_image = args.model_root / "b_facequad" / "images" / args.split / f"{b_stem(image_stem, obj, face)}.jpg"
                if not b_image.exists():
                    continue
                candidate = {
                    "meta_path": meta_path,
                    "meta": meta,
                    "meta_v2": meta_v2,
                    "image_path": image_path,
                    "image_stem": image_stem,
                    "obj": obj,
                    "face": face,
                    "a2_image": a2_image,
                    "b_image": b_image,
                    "c_image": c_path(args.model_root, args.split, image_stem, obj, face),
                }
                candidates.append(candidate)
    if not candidates:
        raise RuntimeError("no usable A1/A2/B/C case found")

    def score(candidate):
        face = candidate["face"]
        has_c = candidate["c_image"].exists()
        has_occ = bool(face.get("occluder_object_ids"))
        visible_ratio = float(face.get("visible_ratio") or 0.0)
        quality = face.get("quality")
        occlusion_severity = max(0.0, 1.0 - visible_ratio)
        if args.prefer_occlusion:
            return (
                1 if has_c else 0,
                1 if has_occ else 0,
                1 if quality == "partial" else 0,
                occlusion_severity,
                float(face.get("visible_pixels") or 0),
            )
        return (
            1 if has_c else 0,
            1 if quality == "good" else 0,
            float(face.get("visible_pixels") or 0),
        )

    return max(candidates, key=score)


def denorm_segment(values, width, height):
    pts = np.asarray(values, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] = np.clip(pts[:, 0], 0, 1) * (width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, 1) * (height - 1)
    return np.rint(pts).astype(np.int32)


def draw_segments(image, lines, names, colors):
    h, w = image.shape[:2]
    out = image.copy()
    for raw in lines:
        parts = raw.strip().split()
        if len(parts) < 7:
            continue
        class_id = int(float(parts[0]))
        values = [float(v) for v in parts[1:]]
        pts = denorm_segment(values, w, h)
        color = colors[class_id % len(colors)]
        cv2.polylines(out, [pts.reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
        cv2.fillPoly(out, [pts.reshape(-1, 1, 2)], color)
        out = cv2.addWeighted(out, 0.35, image, 0.65, 0)
        x, y = pts[0]
        cv2.putText(out, names.get(class_id, str(class_id)), (int(x), max(15, int(y) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def draw_a1(case):
    image = load_image(case["image_path"])
    lines = []
    for obj in case["meta_v2"].get("objects", []):
        class_id = int(obj.get("a1_class_id", 0))
        for segment in obj.get("visible_segments") or []:
            lines.append(" ".join([str(class_id), *[str(v) for v in segment]]))
    colors = [(40, 220, 40), (255, 190, 0), (255, 0, 180), (0, 190, 255)]
    out = draw_segments(image, lines, A1_NAMES, colors)

    selected = case["obj"]
    bbox = selected.get("visible_bbox_xyxy") or selected.get("full_bbox_xyxy")
    if bbox:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 3, cv2.LINE_AA)
    return out


def draw_a2(case, split, model_root):
    image = load_image(case["a2_image"])
    label_path = model_root / "a2_faceseg" / "labels" / split / f"{a2_stem(case['image_stem'], case['obj'])}.txt"
    lines = label_path.read_text(encoding="utf-8").splitlines() if label_path.exists() else []
    colors = [(240, 240, 240), (0, 220, 255)]
    return draw_segments(image, lines, A2_NAMES, colors)


def draw_b(case, split, model_root):
    image = load_image(case["b_image"])
    label_path = model_root / "b_facequad" / "labels" / split / f"{b_stem(case['image_stem'], case['obj'], case['face'])}.txt"
    if not label_path.exists():
        return image
    parts = [float(v) for v in label_path.read_text(encoding="utf-8").strip().split()]
    if len(parts) < 17:
        return image
    h, w = image.shape[:2]
    kpts = parts[5:17]
    pts = []
    vis = []
    for idx in range(0, len(kpts), 3):
        pts.append([kpts[idx] * (w - 1), kpts[idx + 1] * (h - 1)])
        vis.append(kpts[idx + 2])
    pts = np.asarray(pts, dtype=np.int32)
    cv2.polylines(image, [pts.reshape(-1, 1, 2)], True, (0, 255, 255), 2, cv2.LINE_AA)
    for idx, (x, y) in enumerate(pts):
        color = (0, 255, 0) if vis[idx] >= 2 else (0, 0, 255)
        cv2.circle(image, (int(x), int(y)), 5, color, -1, cv2.LINE_AA)
        cv2.putText(image, str(idx + 1), (int(x) + 5, int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return image


def draw_c(case):
    if case["c_image"].exists():
        return load_image(case["c_image"])
    image = np.full((224, 224, 3), 34, dtype=np.uint8)
    cv2.putText(image, "C crop", (50, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(image, "filtered", (50, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 190, 255), 2, cv2.LINE_AA)
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


def make_panel(title, image, lines, size):
    header = 86
    panel = np.full((size + header, size, 3), 255, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (size - 1, size + header - 1), (210, 210, 210), 1)
    cv2.putText(panel, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
    y = 50
    for line in lines[:2]:
        cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (70, 70, 70), 1, cv2.LINE_AA)
        y += 20
    panel[header:header + size, 0:size] = fit_square(image, size)
    return panel


def black_fraction(path):
    if not path.exists():
        return None
    image = load_image(path)
    return float(((image[:, :, 0] < 8) & (image[:, :, 1] < 8) & (image[:, :, 2] < 8)).mean())


def main():
    args = parse_args()
    args.source_dataset = args.source_dataset.resolve()
    args.model_root = args.model_root.resolve()
    args.output = args.output.resolve()

    summary = summarize_dataset(args.source_dataset, args.split)
    case = find_case(args)
    obj = case["obj"]
    face = case["face"]
    label = c_class_for_face(face)
    occ = face.get("occluder_object_ids") or []
    c_black = black_fraction(case["c_image"])
    image_fanout = count_fanout(args.model_root, args.split, case["image_stem"])
    object_fanout = count_fanout(args.model_root, args.split, case["image_stem"], obj)

    panels = [
        make_panel(
            "A1 input/output",
            draw_a1(case),
            [
                f"1 full image -> {image_fanout['a2_crops']} A2 crops",
                f"-> {image_fanout['b_face_samples']} B masks, {image_fanout['c_face_crops']} C crops",
            ],
            args.panel_size,
        ),
        make_panel(
            "A2 input/output",
            draw_a2(case, args.split, args.model_root),
            [
                f"selected obj{int(obj['object_id']):03d}: {obj.get('source_class')}",
                f"this cube -> {object_fanout['b_face_samples']} B masks, {object_fanout['c_face_crops']} C crops",
            ],
            args.panel_size,
        ),
        make_panel(
            "B input/output",
            draw_b(case, args.split, args.model_root),
            [
                "visible mask -> amodal quad",
                f"quality={face.get('quality')} occluders={occ}",
            ],
            args.panel_size,
        ),
        make_panel(
            "C input/output",
            draw_c(case),
            [
                f"warped face -> {label}",
                "black_fraction=" + ("n/a" if c_black is None else f"{c_black:.3f}"),
            ],
            args.panel_size,
        ),
    ]

    sheet = np.concatenate(panels, axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 94])

    report = {
        "source_dataset": str(args.source_dataset),
        "model_root": str(args.model_root),
        "output": str(args.output),
        "summary": summary,
        "selected": {
            "image": str(case["image_path"]),
            "image_fanout": image_fanout,
            "selected_object_fanout": object_fanout,
            "object_id": int(obj["object_id"]),
            "object_name": obj.get("object_name"),
            "source_class": obj.get("source_class"),
            "face_id": int(face["face_id"]),
            "face_kind": face.get("face_kind"),
            "fruit_class": face.get("fruit_class"),
            "quality": face.get("quality"),
            "visible_ratio": face.get("visible_ratio"),
            "visible_pixels": face.get("visible_pixels"),
            "full_pixels": face.get("full_pixels"),
            "occluder_object_ids": occ,
            "c_class": label,
            "c_exported": case["c_image"].exists(),
            "c_black_fraction": c_black,
        },
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
