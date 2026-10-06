import argparse
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


CLASS_NAMES = ["apple", "orange", "banana", "pineapple", "plain"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
FRUIT_CLASSES = ["apple", "orange", "banana", "pineapple"]
SLICE_WORDS = {
    "slice",
    "sliced",
    "cut",
    "half",
    "halves",
    "wedge",
    "segment",
    "section",
    "piece",
    "pieces",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def as_posix_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def clamp_points(points: np.ndarray, size: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    points[:, 0] = np.clip(points[:, 0], 0, size - 1)
    points[:, 1] = np.clip(points[:, 1], 0, size - 1)
    return points


def polygon_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def yolo_seg_line(class_name: str, quad: np.ndarray, size: int) -> str:
    norm = quad.astype(np.float32).copy()
    norm[:, 0] /= size
    norm[:, 1] /= size
    norm = np.clip(norm, 0.0, 1.0)
    coords = " ".join(f"{v:.6f}" for v in norm.reshape(-1))
    return f"{CLASS_TO_ID[class_name]} {coords}"


def safe_reset(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def collect_textures(texture_roots: list[Path]) -> dict[str, list[Path]]:
    textures: dict[str, list[Path]] = {name: [] for name in FRUIT_CLASSES}
    for root in texture_roots:
        if not root.exists():
            continue
        for class_name in FRUIT_CLASSES:
            class_dir = root / class_name
            if not class_dir.exists():
                continue
            for path in class_dir.rglob("*"):
                if path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                lowered = " ".join(path.parts[-3:]).lower()
                if any(word in lowered for word in SLICE_WORDS):
                    continue
                textures[class_name].append(path)
    missing = [name for name, paths in textures.items() if not paths]
    if missing:
        raise RuntimeError(f"no whole-fruit textures found for: {', '.join(missing)}")
    return textures


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    return image


def center_square_crop(image: np.ndarray, rng: random.Random, min_keep: float = 0.78) -> np.ndarray:
    h, w = image.shape[:2]
    side = min(h, w)
    keep = rng.uniform(min_keep, 1.0)
    side = max(8, int(side * keep))
    cx = w // 2 + rng.randint(-max(1, w // 14), max(1, w // 14))
    cy = h // 2 + rng.randint(-max(1, h // 14), max(1, h // 14))
    x0 = max(0, min(w - side, cx - side // 2))
    y0 = max(0, min(h - side, cy - side // 2))
    return image[y0 : y0 + side, x0 : x0 + side]


def make_background(size: int, rng: random.Random) -> np.ndarray:
    base_choices = [
        np.array([205, 202, 194], dtype=np.float32),
        np.array([191, 196, 199], dtype=np.float32),
        np.array([202, 188, 160], dtype=np.float32),
        np.array([184, 171, 136], dtype=np.float32),
        np.array([214, 211, 203], dtype=np.float32),
    ]
    base = rng.choice(base_choices) + np.array(
        [rng.uniform(-12, 12), rng.uniform(-12, 12), rng.uniform(-12, 12)],
        dtype=np.float32,
    )
    x_grad = np.linspace(-1.0, 1.0, size, dtype=np.float32)[None, :, None]
    y_grad = np.linspace(-1.0, 1.0, size, dtype=np.float32)[:, None, None]
    grad = x_grad * rng.uniform(-16, 16) + y_grad * rng.uniform(-14, 14)
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(1.5, 5.0), (size, size, 1))
    image = np.ones((size, size, 3), dtype=np.float32) * base + grad + noise
    if rng.random() < 0.55:
        # Subtle table/wood grain: visible enough to be a realistic nuisance, not a fruit texture.
        for _ in range(rng.randint(6, 18)):
            y = rng.randint(0, size - 1)
            color = base + rng.uniform(-24, 18)
            cv2.line(image, (0, y), (size - 1, y + rng.randint(-6, 6)), color.tolist(), rng.randint(1, 2), cv2.LINE_AA)
    return np.clip(image, 0, 255).astype(np.uint8)


def plain_face_texture(width: int, height: int, rng: random.Random, warm: bool = False) -> np.ndarray:
    palettes = [
        np.array([226, 229, 224], dtype=np.float32),
        np.array([213, 221, 218], dtype=np.float32),
        np.array([234, 232, 218], dtype=np.float32),
        np.array([218, 214, 198], dtype=np.float32),
        np.array([201, 211, 219], dtype=np.float32),
    ]
    if warm:
        palettes.extend(
            [
                np.array([221, 218, 178], dtype=np.float32),
                np.array([220, 207, 165], dtype=np.float32),
                np.array([214, 198, 152], dtype=np.float32),
                np.array([226, 222, 190], dtype=np.float32),
            ]
        )
    base = rng.choice(palettes) + np.array(
        [rng.uniform(-10, 10), rng.uniform(-10, 10), rng.uniform(-10, 10)],
        dtype=np.float32,
    )
    x_grad = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :, None]
    y_grad = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None, None]
    grad = x_grad * rng.uniform(-22, 22) + y_grad * rng.uniform(-22, 22)
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(1.0, 4.0), (height, width, 1))
    patch = np.ones((height, width, 3), dtype=np.float32) * base + grad + noise
    if rng.random() < 0.35:
        edge_color = np.clip(base + rng.uniform(-42, -18), 0, 255).tolist()
        cv2.line(patch, (0, height - 1), (width - 1, height - 1), edge_color, 1, cv2.LINE_AA)
    return np.clip(patch, 0, 255).astype(np.uint8)


def whole_fruit_texture(path: Path, size: int, rng: random.Random) -> np.ndarray:
    image = read_image(path)
    image = center_square_crop(image, rng)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.25:
        alpha = rng.uniform(0.94, 1.08)
        beta = rng.uniform(-8, 10)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return image


def warp_patch(dst: np.ndarray, patch: np.ndarray, quad: np.ndarray, border_shadow: bool, rng: random.Random) -> None:
    h, w = patch.shape[:2]
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    warped = cv2.warpPerspective(patch, matrix, (dst.shape[1], dst.shape[0]), flags=cv2.INTER_LINEAR)
    mask = np.zeros(dst.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255, cv2.LINE_AA)
    if border_shadow:
        shadow = cv2.GaussianBlur(mask, (0, 0), rng.uniform(1.5, 4.0))
        dark = dst.astype(np.float32) * rng.uniform(0.74, 0.88)
        alpha = (shadow.astype(np.float32) / 255.0 * rng.uniform(0.08, 0.18))[:, :, None]
        dst[:] = np.clip(dst.astype(np.float32) * (1.0 - alpha) + dark * alpha, 0, 255).astype(np.uint8)
    dst[mask > 0] = warped[mask > 0]


def draw_face_edges(image: np.ndarray, faces: list[tuple[str, np.ndarray]], rng: random.Random) -> None:
    for _, quad in faces:
        color = rng.choice([(235, 235, 230), (205, 212, 218), (185, 190, 190), (226, 218, 198)])
        cv2.polylines(image, [quad.astype(np.int32)], True, color, rng.choice([1, 1, 2]), cv2.LINE_AA)


def random_cube_geometry(size: int, rng: random.Random, kind: str) -> list[np.ndarray]:
    cx = rng.uniform(size * 0.38, size * 0.62)
    cy = rng.uniform(size * 0.38, size * 0.63)
    face = rng.uniform(size * 0.42, size * 0.72)
    if kind == "close":
        face = rng.uniform(size * 0.58, size * 0.86)
    if kind == "edge":
        face = rng.uniform(size * 0.46, size * 0.76)
    skew_x = rng.uniform(-0.18, 0.18) * face
    skew_y = rng.uniform(-0.08, 0.12) * face
    w = face * rng.uniform(0.78, 1.05)
    h = face * rng.uniform(0.72, 1.04)
    front = np.array(
        [
            [cx - w / 2 + skew_x * 0.25, cy - h / 2 + skew_y * 0.15],
            [cx + w / 2 + skew_x, cy - h / 2 - skew_y * 0.15],
            [cx + w / 2 - skew_x * 0.25, cy + h / 2 + skew_y * 0.25],
            [cx - w / 2 - skew_x, cy + h / 2 - skew_y * 0.10],
        ],
        dtype=np.float32,
    )
    depth = face * rng.uniform(0.12, 0.34)
    if kind == "edge":
        depth = face * rng.uniform(0.05, 0.16)
    dx = rng.choice([-1, 1]) * depth * rng.uniform(0.55, 1.15)
    dy = -depth * rng.uniform(0.45, 0.95)
    side = np.array([front[1], front[1] + [dx, dy], front[2] + [dx * 0.85, dy * 0.78], front[2]], dtype=np.float32)
    top = np.array([front[0], front[1], front[1] + [dx, dy], front[0] + [dx * 0.85, dy * 0.78]], dtype=np.float32)
    faces = [front]
    if rng.random() < (0.82 if kind != "close" else 0.55):
        faces.append(side)
    if rng.random() < (0.64 if kind != "close" else 0.40):
        faces.append(top)
    return [clamp_points(face_pts, size) for face_pts in faces if polygon_area(face_pts) > 120]


def make_empty_background_case(image: np.ndarray, rng: random.Random) -> list[tuple[str, np.ndarray]]:
    size = image.shape[0]
    if rng.random() < 0.7:
        x = rng.randint(size // 6, size * 5 // 6)
        y = rng.randint(size // 6, size * 5 // 6)
        color = rng.choice([(230, 232, 225), (210, 218, 210), (224, 214, 184), (190, 201, 210)])
        cv2.rectangle(
            image,
            (max(0, x - rng.randint(15, 45)), max(0, y - rng.randint(12, 40))),
            (min(size - 1, x + rng.randint(20, 55)), min(size - 1, y + rng.randint(18, 55))),
            color,
            -1,
        )
    for _ in range(rng.randint(1, 4)):
        p1 = (rng.randint(0, size - 1), rng.randint(0, size - 1))
        p2 = (rng.randint(0, size - 1), rng.randint(0, size - 1))
        cv2.line(image, p1, p2, rng.choice([(180, 185, 182), (230, 230, 218), (160, 150, 130)]), 1, cv2.LINE_AA)
    return []


def apply_camera_degradation(image: np.ndarray, rng: random.Random, hard: bool) -> np.ndarray:
    out = image.copy()
    alpha = rng.uniform(0.82, 1.18 if not hard else 1.28)
    beta = rng.uniform(-20, 22 if not hard else 34)
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    if rng.random() < (0.38 if hard else 0.18):
        k = rng.choice([3, 5, 7, 9])
        if rng.random() < 0.65:
            kernel = np.zeros((k, k), dtype=np.float32)
            if rng.random() < 0.5:
                kernel[k // 2, :] = 1.0
            else:
                kernel[:, k // 2] = 1.0
            kernel /= kernel.sum()
            out = cv2.filter2D(out, -1, kernel)
        else:
            out = cv2.GaussianBlur(out, (k, k), 0)
    if rng.random() < 0.55:
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(1.0, 6.0 if hard else 3.0), out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < (0.45 if hard else 0.20):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), rng.randint(45 if hard else 62, 88)]
        ok, encoded = cv2.imencode(".jpg", out, encode_param)
        if ok:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is not None:
                out = decoded
    return out


def choose_case(rng: random.Random, plain_ratio: float, background_empty_ratio: float) -> str:
    r = rng.random()
    if r < background_empty_ratio:
        return "background_empty"
    if r < background_empty_ratio + plain_ratio:
        return rng.choice(["plain_warm", "plain_edge", "plain_shadow"])
    return rng.choice(["fruit_whole", "fruit_close", "fruit_edge"])


def render_sample(
    idx: int,
    split: str,
    out_root: Path,
    textures: dict[str, list[Path]],
    rng: random.Random,
    args: argparse.Namespace,
) -> dict:
    size = args.crop_size
    case = choose_case(rng, args.plain_ratio, args.background_empty_ratio)
    image = make_background(size, rng)
    labels: list[str] = []
    faces_for_edges: list[tuple[str, np.ndarray]] = []
    class_counts = Counter()

    if case == "background_empty":
        faces_for_edges = make_empty_background_case(image, rng)
    else:
        geom_kind = "close" if case == "fruit_close" else "edge" if "edge" in case else "normal"
        quads = random_cube_geometry(size, rng, geom_kind)
        fruit_class = None
        if case.startswith("fruit"):
            fruit_class = rng.choices(
                FRUIT_CLASSES,
                weights=[args.apple_weight, args.orange_weight, args.banana_weight, args.pineapple_weight],
                k=1,
            )[0]

        for face_idx, quad in enumerate(quads):
            is_main = face_idx == 0
            class_name = fruit_class if fruit_class and is_main else "plain"
            if class_name == "plain":
                patch = plain_face_texture(size, size, rng, warm=True)
            else:
                patch = whole_fruit_texture(rng.choice(textures[class_name]), size, rng)
            warp_patch(image, patch, quad, border_shadow=True, rng=rng)
            faces_for_edges.append((class_name, quad))
            if polygon_area(quad) >= args.min_label_area:
                labels.append(yolo_seg_line(class_name, quad, size))
                class_counts[class_name] += 1

    if faces_for_edges:
        draw_face_edges(image, faces_for_edges, rng)
    image = apply_camera_degradation(image, rng, hard=case in {"plain_warm", "plain_edge", "fruit_edge"})

    stem = f"{idx:06d}_{case}"
    img_path = out_root / "images" / split / f"{stem}.jpg"
    label_path = out_root / "labels" / split / f"{stem}.txt"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")

    return {
        "split": split,
        "case": case,
        "image": str(img_path),
        "label": str(label_path),
        "labels": len(labels),
        "class_counts": dict(class_counts),
    }


def write_data_yaml(root: Path) -> None:
    lines = [
        f"path: {as_posix_path(root)}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for idx, name in enumerate(CLASS_NAMES):
        lines.append(f"  {idx}: {name}")
    (root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_preview(root: Path, records: list[dict], count: int, size: int) -> Path | None:
    selected = records[:count]
    if not selected:
        return None
    thumbs = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for rec in selected:
        image = cv2.imread(rec["image"])
        if image is None:
            continue
        label_path = Path(rec["label"])
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                cls = int(float(parts[0]))
                pts = np.array([float(v) for v in parts[1:]], dtype=np.float32).reshape(-1, 2)
                pts[:, 0] *= image.shape[1]
                pts[:, 1] *= image.shape[0]
                color = [(60, 60, 255), (0, 150, 255), (0, 230, 255), (0, 180, 0), (235, 235, 235)][cls]
                cv2.polylines(image, [pts.astype(np.int32)], True, color, 2, cv2.LINE_AA)
                cv2.putText(image, CLASS_NAMES[cls], tuple(pts[0].astype(int)), font, 0.42, color, 1, cv2.LINE_AA)
        header = np.zeros((22, image.shape[1], 3), dtype=np.uint8)
        cv2.putText(header, rec["case"], (4, 16), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(np.vstack([header, image]))
    if not thumbs:
        return None
    cols = min(8, max(1, int(math.sqrt(len(thumbs)) + 1)))
    rows = math.ceil(len(thumbs) / cols)
    tile_h, tile_w = thumbs[0].shape[:2]
    sheet = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        y = (idx // cols) * tile_h
        x = (idx % cols) * tile_w
        sheet[y : y + tile_h, x : x + tile_w] = thumb
    out = root / "previews" / "booster_contact_sheet.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return out


def write_mix_dataset(args: argparse.Namespace, booster_root: Path, records: list[dict]) -> Path:
    base_root = args.base_dataset
    mix_root = args.mix_output
    mix_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed + 101)

    base_train = sorted((base_root / "images" / "train").glob("*.jpg"))
    base_val = sorted((base_root / "images" / "val").glob("*.jpg"))
    if not base_train:
        raise RuntimeError(f"no base train images found: {base_root}")
    booster_train = sorted((booster_root / "images" / "train").glob("*.jpg"))
    booster_val = sorted((booster_root / "images" / "val").glob("*.jpg"))
    base_train_sample = rng.sample(base_train, min(args.base_sample_count, len(base_train)))
    base_val_sample = rng.sample(base_val, min(args.base_val_count, len(base_val))) if base_val else []

    train_paths = base_train_sample + booster_train * max(1, args.booster_repeat)
    val_paths = base_val_sample + booster_val
    rng.shuffle(train_paths)
    rng.shuffle(val_paths)

    train_txt = mix_root / "train.txt"
    val_txt = mix_root / "val.txt"
    train_txt.write_text("\n".join(as_posix_path(p) for p in train_paths) + "\n", encoding="utf-8")
    val_txt.write_text("\n".join(as_posix_path(p) for p in val_paths) + "\n", encoding="utf-8")

    data_lines = [
        f"path: {as_posix_path(Path.cwd())}",
        f"train: {as_posix_path(train_txt)}",
        f"val: {as_posix_path(val_txt)}",
        "names:",
    ]
    for idx, name in enumerate(CLASS_NAMES):
        data_lines.append(f"  {idx}: {name}")
    (mix_root / "data.yaml").write_text("\n".join(data_lines) + "\n", encoding="utf-8")

    manifest = {
        "task": "cube_face_unified_finetune_mix",
        "base_dataset": str(base_root),
        "booster_dataset": str(booster_root),
        "train_images": len(train_paths),
        "val_images": len(val_paths),
        "base_train_sample": len(base_train_sample),
        "base_val_sample": len(base_val_sample),
        "booster_train_images": len(booster_train),
        "booster_val_images": len(booster_val),
        "booster_repeat": args.booster_repeat,
        "note": "Duplicate booster paths intentionally oversample whole-fruit/plain hard cases.",
    }
    (mix_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return mix_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a whole-fruit + plain hard-negative booster for the cube-face unified YOLO-seg model."
    )
    parser.add_argument("--output_root", type=Path, default=Path("datasets/cube_face_unified_booster_wholefruit_plainhard_v1"))
    parser.add_argument("--base_dataset", type=Path, default=Path("datasets/meta_v2_50000_cube_face_unified_v1"))
    parser.add_argument(
        "--texture_roots",
        type=Path,
        nargs="+",
        default=[Path("datasets/fruit_textures/final_fruits36070_original30_color_filtered_v2")],
    )
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--plain_ratio", type=float, default=0.46)
    parser.add_argument("--background_empty_ratio", type=float, default=0.07)
    parser.add_argument("--min_label_area", type=float, default=160.0)
    parser.add_argument("--apple_weight", type=float, default=1.0)
    parser.add_argument("--orange_weight", type=float, default=1.0)
    parser.add_argument("--banana_weight", type=float, default=0.75)
    parser.add_argument("--pineapple_weight", type=float, default=0.75)
    parser.add_argument("--preview_count", type=int, default=96)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--make_mix", action="store_true")
    parser.add_argument("--mix_output", type=Path, default=Path("datasets/cube_face_unified_finetune_wholefruit_plainhard_v1"))
    parser.add_argument("--base_sample_count", type=int, default=22000)
    parser.add_argument("--base_val_count", type=int, default=2500)
    parser.add_argument("--booster_repeat", type=int, default=2)
    return parser.parse_args()


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return value


def main() -> None:
    args = parse_args()
    if args.reset and args.output_root.exists():
        safe_reset(args.output_root)
    for split in ["train", "val"]:
        (args.output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    textures = collect_textures(args.texture_roots)
    write_data_yaml(args.output_root)
    rng = random.Random(args.seed)
    records = []
    stats = {
        "cases": Counter(),
        "instances": Counter(),
        "splits": Counter(),
        "texture_counts": {name: len(paths) for name, paths in textures.items()},
    }

    existing = set()
    if args.resume:
        existing = {p.stem for p in (args.output_root / "images").rglob("*.jpg")}

    for idx in tqdm(range(args.count), desc="cube-face booster", unit="img", file=sys.stdout):
        split = "val" if rng.random() < args.val_ratio else "train"
        # Existing stems include the case suffix, so only use resume for exact generated files if present.
        if args.resume and any(stem.startswith(f"{idx:06d}_") for stem in existing):
            continue
        rec = render_sample(idx, split, args.output_root, textures, rng, args)
        records.append(rec)
        stats["cases"][rec["case"]] += 1
        stats["splits"][rec["split"]] += 1
        for class_name, count in rec["class_counts"].items():
            stats["instances"][class_name] += count

    preview = make_preview(args.output_root, records, args.preview_count, args.crop_size)
    manifest = {
        "task": "cube_face_unified_booster_wholefruit_plainhard",
        "purpose": [
            "whole-fruit face reinforcement without slice/cut/half texture files",
            "warm white/yellow/orange plain-face hard negatives",
            "cube edge/corner and background-empty negatives for runtime crop leakage",
        ],
        "args": json_safe(vars(args)),
        "stats": {
            "cases": dict(stats["cases"]),
            "instances": dict(stats["instances"]),
            "splits": dict(stats["splits"]),
            "texture_counts": stats["texture_counts"],
        },
        "class_names": CLASS_NAMES,
        "preview": str(preview) if preview else None,
        "slice_filter_words": sorted(SLICE_WORDS),
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    mix_root = None
    if args.make_mix:
        mix_root = write_mix_dataset(args, args.output_root, records)

    print(json.dumps(manifest["stats"], indent=2), flush=True)
    print(f"booster: {args.output_root}", flush=True)
    if preview:
        print(f"preview: {preview}", flush=True)
    if mix_root:
        print(f"mix data: {mix_root / 'data.yaml'}", flush=True)


if __name__ == "__main__":
    main()
