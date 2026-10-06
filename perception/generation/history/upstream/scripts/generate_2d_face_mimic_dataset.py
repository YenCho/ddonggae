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
C_CLASSES = ("apple", "orange", "banana", "pineapple", "plain", "unknown")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a 2D mimic dataset for C face classifier without Blender/3D rendering. "
            "The output matches train_face_mobilenetv3.py: split/class/*.jpg."
        )
    )
    parser.add_argument(
        "--texture_dir",
        type=Path,
        default=Path("datasets/fruit_textures/final_fruits36065_original25_fruitseg30_10"),
        help="Fruit texture root with apple/orange/banana/pineapple subdirectories.",
    )
    parser.add_argument(
        "--extra_orange_texture_dir",
        type=Path,
        default=None,
        help="Optional extra orange texture directory or root/orange directory for targeted orange hard-case experiments.",
    )
    parser.add_argument("--extra_orange_weight", type=float, default=4.0)
    parser.add_argument(
        "--background_dir",
        type=Path,
        default=Path("datasets/backgrounds/coco2017/val2017"),
        help="Optional image pool for unknown no-face crops.",
    )
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--classes", nargs="+", choices=C_CLASSES, default=list(C_CLASSES))
    parser.add_argument("--train_per_class", type=int, default=2000)
    parser.add_argument("--val_per_class", type=int, default=250)
    parser.add_argument("--test_per_class", type=int, default=250)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--apple_multiplier", type=float, default=1.0)
    parser.add_argument("--orange_multiplier", type=float, default=1.0)
    parser.add_argument("--banana_multiplier", type=float, default=1.0)
    parser.add_argument("--pineapple_multiplier", type=float, default=1.0)
    parser.add_argument("--plain_multiplier", type=float, default=1.0)
    parser.add_argument("--unknown_multiplier", type=float, default=1.0)
    parser.add_argument(
        "--apple_orange_boost",
        action="store_true",
        help="Increase apple/orange count and hard framing probability for apple/orange confusion work.",
    )
    parser.add_argument("--hard_ratio", type=float, default=0.55)
    parser.add_argument("--orange_hard_ratio", type=float, default=0.70)
    parser.add_argument("--white_margin_hard_ratio", type=float, default=0.55)
    parser.add_argument(
        "--orange_green_distractor_prob",
        type=float,
        default=0.20,
        help="Draw a small green leaf-like distractor on some orange hard cases so C learns not to treat green as apple-only.",
    )
    parser.add_argument("--occlusion_prob", type=float, default=0.16)
    parser.add_argument("--motion_blur_prob", type=float, default=0.32)
    parser.add_argument("--jpeg_quality_min", type=int, default=38)
    parser.add_argument("--jpeg_quality_max", type=int, default=92)
    parser.add_argument("--render_dirty", action="store_true", default=True)
    parser.add_argument("--no_render_dirty", dest="render_dirty", action="store_false")
    parser.add_argument(
        "--projected_resolution_prob",
        type=float,
        default=0.88,
        help="Simulate a face that was rendered small in the 640 frame and then warped/resized for C.",
    )
    parser.add_argument("--lens_like_distortion_prob", type=float, default=0.35)
    parser.add_argument("--preview_count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def reset_output(path):
    if path.exists():
        shutil.rmtree(path)


def ensure_dirs(output_root, splits):
    for split in splits:
        for class_name in C_CLASSES:
            (output_root / split / class_name).mkdir(parents=True, exist_ok=True)
    (output_root / "_preview").mkdir(parents=True, exist_ok=True)


def read_image(path, unchanged=False):
    flag = cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_COLOR
    data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, flag)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    return image


def write_jpg(path, image, quality=94):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"failed to encode jpg: {path}")
    path.write_bytes(buf.tobytes())


def collect_images(root):
    root = Path(root)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def collect_texture_pool(texture_dir):
    pools = {}
    for class_name in FRUIT_CLASSES:
        paths = collect_images(texture_dir / class_name)
        if not paths:
            raise RuntimeError(f"no textures for {class_name}: {texture_dir / class_name}")
        pools[class_name] = paths
    return pools


def extend_orange_pool(pools, extra_dir, weight):
    if not extra_dir:
        return
    extra_dir = Path(extra_dir)
    candidates = []
    if (extra_dir / "orange").exists():
        candidates.extend(collect_images(extra_dir / "orange"))
    candidates.extend(collect_images(extra_dir))
    unique = []
    seen = set()
    for path in candidates:
        key = str(path.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    if not unique:
        raise RuntimeError(f"no extra orange textures found: {extra_dir}")
    repeat = max(1, int(round(float(weight))))
    pools["orange"] = list(pools["orange"]) + unique * repeat


def split_count(args, split, class_name):
    if split == "train":
        base = args.train_per_class
    elif split == "val":
        base = args.val_per_class
    elif split == "test":
        base = args.test_per_class
    else:
        base = args.train_per_class

    multipliers = {
        "apple": args.apple_multiplier,
        "orange": args.orange_multiplier,
        "banana": args.banana_multiplier,
        "pineapple": args.pineapple_multiplier,
        "plain": args.plain_multiplier,
        "unknown": args.unknown_multiplier,
    }
    multiplier = multipliers[class_name]
    if args.apple_orange_boost and class_name in {"apple", "orange"}:
        multiplier *= 1.35
    return int(round(base * multiplier))


def make_face_background(size, rng, mode="paper"):
    base = rng.randint(224, 252)
    color = np.asarray(
        [
            clamp(base + rng.randint(-8, 8), 0, 255),
            clamp(base + rng.randint(-8, 8), 0, 255),
            clamp(base + rng.randint(-8, 8), 0, 255),
        ],
        dtype=np.float32,
    )
    canvas = np.ones((size, size, 3), dtype=np.float32) * color
    noise = rng.uniform(0.8, 4.0) if mode == "paper" else rng.uniform(2.0, 10.0)
    canvas += rng.normalvariate(0.0, noise) * np.ones_like(canvas)
    pixel_noise = np.random.default_rng(rng.randrange(2**32)).normal(0, noise, canvas.shape)
    canvas += pixel_noise

    if rng.random() < 0.75:
        gx = np.linspace(rng.uniform(-10, 8), rng.uniform(-8, 10), size, dtype=np.float32)
        gy = np.linspace(rng.uniform(-8, 10), rng.uniform(-10, 8), size, dtype=np.float32)
        grad = gx[None, :] + gy[:, None]
        canvas += grad[:, :, None]

    if mode == "plain" and rng.random() < 0.45:
        x = rng.randint(0, size - 1)
        cv2.line(canvas, (x, 0), (clamp(x + rng.randint(-30, 30), 0, size - 1), size - 1), rng.randint(185, 235), 1)

    return np.clip(canvas, 0, 255).astype(np.uint8)


def normalize_texture(image):
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        alpha = None
    elif image.shape[2] == 4:
        bgr = image[:, :, :3]
        alpha = image[:, :, 3].astype(np.float32) / 255.0
        white = np.full_like(bgr, 255)
        image = (bgr.astype(np.float32) * alpha[:, :, None] + white.astype(np.float32) * (1.0 - alpha[:, :, None])).astype(np.uint8)
    else:
        image = image[:, :, :3]
    return image


def random_source_crop(image, rng, hard):
    h, w = image.shape[:2]
    if h < 4 or w < 4:
        return image
    keep = rng.uniform(0.70, 1.0) if hard else rng.uniform(0.86, 1.0)
    crop_w = max(4, int(w * keep))
    crop_h = max(4, int(h * keep))
    if crop_w >= w or crop_h >= h:
        return image
    x1 = rng.randint(0, w - crop_w)
    y1 = rng.randint(0, h - crop_h)
    return image[y1 : y1 + crop_h, x1 : x1 + crop_w]


def rotate_patch(image, angle, border_value):
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, 1.0)
    corners = np.asarray([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    ones = np.ones((4, 1), dtype=np.float32)
    new = np.hstack([corners, ones]) @ matrix.T
    min_xy = new.min(axis=0)
    max_xy = new.max(axis=0)
    new_w = int(math.ceil(max_xy[0] - min_xy[0]))
    new_h = int(math.ceil(max_xy[1] - min_xy[1]))
    matrix[0, 2] -= min_xy[0]
    matrix[1, 2] -= min_xy[1]
    return cv2.warpAffine(
        image,
        matrix,
        (max(new_w, 1), max(new_h, 1)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=tuple(int(v) for v in border_value),
    )


def paste_patch(canvas, patch, center_x, center_y):
    out = canvas.copy()
    h, w = out.shape[:2]
    ph, pw = patch.shape[:2]
    x1 = int(round(center_x - pw * 0.5))
    y1 = int(round(center_y - ph * 0.5))
    x2 = x1 + pw
    y2 = y1 + ph
    sx1 = max(0, -x1)
    sy1 = max(0, -y1)
    sx2 = pw - max(0, x2 - w)
    sy2 = ph - max(0, y2 - h)
    dx1 = max(0, x1)
    dy1 = max(0, y1)
    dx2 = dx1 + max(0, sx2 - sx1)
    dy2 = dy1 + max(0, sy2 - sy1)
    if dx2 <= dx1 or dy2 <= dy1:
        return out
    out[dy1:dy2, dx1:dx2] = patch[sy1:sy2, sx1:sx2]
    return out


def color_jitter(image, rng, hard=False, class_name=None):
    out = image.astype(np.float32)
    contrast = rng.uniform(0.82, 1.22) if hard else rng.uniform(0.90, 1.12)
    brightness = rng.uniform(-18, 18) if hard else rng.uniform(-10, 10)
    out = out * contrast + brightness
    out = np.clip(out, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue_shift = rng.uniform(-5, 5) if not hard else rng.uniform(-9, 9)
    sat_scale = rng.uniform(0.75, 1.25) if not hard else rng.uniform(0.58, 1.42)
    val_scale = rng.uniform(0.85, 1.15) if not hard else rng.uniform(0.72, 1.28)
    if class_name == "orange" and hard and rng.random() < 0.35:
        hue_shift += rng.uniform(-4, 4)
        sat_scale *= rng.uniform(0.90, 1.25)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_scale, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    gamma = rng.uniform(0.78, 1.30) if hard else rng.uniform(0.88, 1.15)
    lut = np.asarray([np.clip((i / 255.0) ** gamma * 255.0, 0, 255) for i in range(256)], dtype=np.uint8)
    return cv2.LUT(out, lut)


def perspective_jitter(image, rng, hard=False):
    size = image.shape[0]
    max_jitter = size * (0.055 if not hard else 0.105)
    src = np.asarray([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    dst = src + np.asarray(
        [[rng.uniform(-max_jitter, max_jitter), rng.uniform(-max_jitter, max_jitter)] for _ in range(4)],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    border = int(np.median(image.reshape(-1, 3)))
    return cv2.warpPerspective(
        image,
        matrix,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(border, border, border),
    )


def add_occlusion(image, rng, prob):
    if rng.random() >= prob:
        return image
    out = image.copy()
    size = out.shape[0]
    side = rng.choice(["left", "right", "top", "bottom"])
    color = rng.choice([(0, 0, 0), (20, 20, 20), (245, 245, 245)])
    thickness = rng.randint(max(8, size // 12), max(16, size // 4))
    if side == "left":
        pts = np.asarray([[0, 0], [thickness, rng.randint(0, size)], [rng.randint(0, thickness), size], [0, size]], dtype=np.int32)
    elif side == "right":
        pts = np.asarray([[size, 0], [size - thickness, rng.randint(0, size)], [size - rng.randint(0, thickness), size], [size, size]], dtype=np.int32)
    elif side == "top":
        pts = np.asarray([[0, 0], [size, 0], [rng.randint(0, size), thickness], [rng.randint(0, size), rng.randint(0, thickness)]], dtype=np.int32)
    else:
        pts = np.asarray([[0, size], [size, size], [rng.randint(0, size), size - thickness], [rng.randint(0, size), size - rng.randint(0, thickness)]], dtype=np.int32)
    cv2.fillPoly(out, [pts.reshape(-1, 1, 2)], color)
    return out


def motion_blur(image, rng):
    k = rng.choice([3, 5, 7, 9])
    kernel = np.zeros((k, k), dtype=np.float32)
    if rng.random() < 0.5:
        kernel[k // 2, :] = 1.0
    else:
        kernel[:, k // 2] = 1.0
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel)


def motion_blur_any_angle(image, rng, kernel_size=None):
    k = int(kernel_size or rng.choice([7, 9, 13, 17]))
    if k % 2 == 0:
        k += 1
    angle = math.radians(rng.choice([0, 20, 45, 70, 90, 110, 135, 160]))
    center = k // 2
    kernel = np.zeros((k, k), dtype=np.float32)
    length = k - 1
    dx = math.cos(angle) * length * 0.5
    dy = math.sin(angle) * length * 0.5
    p1 = (int(round(center - dx)), int(round(center - dy)))
    p2 = (int(round(center + dx)), int(round(center + dy)))
    cv2.line(kernel, p1, p2, 1.0, 1)
    total = float(kernel.sum())
    if total <= 0:
        kernel[center, :] = 1.0
        total = float(kernel.sum())
    kernel /= total
    return cv2.filter2D(image, -1, kernel)


def jpeg_roundtrip(image, rng, qmin, qmax):
    quality = rng.randint(clamp(qmin, 1, 100), clamp(qmax, 1, 100))
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return image
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return decoded if decoded is not None else image


def lowres_resample(image, rng, hard=False, strong=False):
    h, w = image.shape[:2]
    if hard or strong:
        scale = rng.uniform(0.22, 0.58)
    else:
        scale = rng.uniform(0.34, 0.82)
    low_w = max(24, int(w * scale))
    low_h = max(24, int(h * scale))
    interp_down = rng.choice([cv2.INTER_AREA, cv2.INTER_LINEAR])
    interp_up = rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST])
    return cv2.resize(cv2.resize(image, (low_w, low_h), interpolation=interp_down), (w, h), interpolation=interp_up)


def apply_lens_like_distortion(image, rng, probability):
    if rng.random() > probability:
        return image
    h, w = image.shape[:2]
    yy, xx = np.indices((h, w), dtype=np.float32)
    cx = w * rng.uniform(0.47, 0.53)
    cy = h * rng.uniform(0.47, 0.53)
    fx = w * rng.uniform(0.72, 1.15)
    fy = h * rng.uniform(0.72, 1.15)
    x = (xx - cx) / fx
    y = (yy - cy) / fy
    r2 = x * x + y * y
    k1 = rng.uniform(-0.24, 0.18)
    k2 = rng.uniform(-0.08, 0.06)
    p1 = rng.uniform(-0.004, 0.004)
    p2 = rng.uniform(-0.004, 0.004)
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    x_dist = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    y_dist = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    map_x = x_dist * fx + cx
    map_y = y_dist * fy + cy
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def add_vignette(image, rng):
    h, w = image.shape[:2]
    yy, xx = np.indices((h, w), dtype=np.float32)
    cx, cy = w / 2.0, h / 2.0
    radius = np.sqrt(((xx - cx) / max(cx, 1)) ** 2 + ((yy - cy) / max(cy, 1)) ** 2)
    vignette = 1.0 - np.clip(radius * rng.uniform(0.05, 0.38), 0, 0.55)
    return np.clip(image.astype(np.float32) * vignette[:, :, None], 0, 255).astype(np.uint8)


def channel_shift(image, rng):
    out = image.copy()
    max_shift = rng.choice([1, 1, 2, 3])
    for channel in range(3):
        dx = rng.randint(-max_shift, max_shift)
        dy = rng.randint(-max_shift, max_shift)
        if dx or dy:
            matrix = np.asarray([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
            out[:, :, channel] = cv2.warpAffine(
                out[:, :, channel],
                matrix,
                (out.shape[1], out.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )
    return out


def print_texture_damage(image, rng, hard=False):
    out = image.copy()
    if rng.random() < 0.85:
        out = lowres_resample(out, rng, hard=hard, strong=hard)
    if rng.random() < (0.75 if hard else 0.45):
        k = rng.choice([3, 5, 7]) if hard else rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)
    if rng.random() < 0.65:
        out = jpeg_roundtrip(out, rng, 42 if hard else 55, 86)
    noise_sigma = rng.uniform(1.0, 7.0 if hard else 4.0)
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, noise_sigma, out.shape)
    return np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_camera_artifacts_like_generator(image, rng, args, hard=False):
    out = image.astype(np.float32)
    if rng.random() < 0.80:
        gains = np.asarray(
            [
                rng.uniform(0.82, 1.22),
                rng.uniform(0.88, 1.12),
                rng.uniform(0.80, 1.28),
            ],
            dtype=np.float32,
        )
        out *= gains
    if rng.random() < 0.85:
        out = out * rng.uniform(0.68, 1.22) + rng.uniform(-30, 18)
    if rng.random() < 0.65:
        gamma = rng.uniform(0.78, 1.35)
        out = 255.0 * np.power(np.clip(out, 0, 255) / 255.0, gamma)
    if rng.random() < 0.55:
        sigma = rng.uniform(2.0, 13.0 if hard else 9.0)
        out += np.random.default_rng(rng.randrange(2**32)).normal(0, sigma, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)

    if rng.random() < (0.55 if not hard else 0.78):
        k = rng.choice([3, 5, 7]) if hard else rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)
    if rng.random() < args.motion_blur_prob:
        out = motion_blur_any_angle(out, rng, kernel_size=rng.choice([7, 9, 13, 17] if hard else [5, 7, 9]))
    if rng.random() < args.projected_resolution_prob:
        out = lowres_resample(out, rng, hard=hard, strong=hard)
    if rng.random() < 0.35:
        out = channel_shift(out, rng)
    if rng.random() < 0.65:
        out = add_vignette(out, rng)
    if rng.random() < 0.88:
        out = jpeg_roundtrip(out, rng, args.jpeg_quality_min, args.jpeg_quality_max)

    white_frac = float(np.mean(np.all(out > 238, axis=2)))
    if white_frac > 0.22:
        out = np.clip(out.astype(np.float32) * rng.uniform(0.68, 0.86), 0, 255).astype(np.uint8)
    mean_luma = float(np.mean(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)))
    if mean_luma < 35.0:
        out = np.clip(out.astype(np.float32) + (rng.uniform(35.0, 55.0) - mean_luma), 0, 255).astype(np.uint8)
    return out


def final_degrade(image, rng, args, hard=False):
    out = color_jitter(image, rng, hard=hard)
    out = perspective_jitter(out, rng, hard=hard)
    out = apply_lens_like_distortion(out, rng, args.lens_like_distortion_prob)
    out = add_occlusion(out, rng, args.occlusion_prob * (1.5 if hard else 1.0))
    if args.render_dirty:
        out = apply_camera_artifacts_like_generator(out, rng, args, hard=hard)
    else:
        if rng.random() < (args.motion_blur_prob * (1.35 if hard else 1.0)):
            out = motion_blur(out, rng)
        elif rng.random() < 0.25:
            k = rng.choice([3, 5])
            out = cv2.GaussianBlur(out, (k, k), 0)
        noise_sigma = rng.uniform(0.0, 4.0 if not hard else 8.0)
        if noise_sigma > 0.5:
            noise = np.random.default_rng(rng.randrange(2**32)).normal(0, noise_sigma, out.shape)
            out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        out = jpeg_roundtrip(out, rng, args.jpeg_quality_min, args.jpeg_quality_max)
    return out


def draw_orange_green_distractor(canvas, rng):
    size = canvas.shape[0]
    if rng.random() < 0.5:
        cx = rng.randint(size // 5, size // 2)
        cy = rng.randint(size // 8, size // 3)
    else:
        cx = rng.randint(size // 2, size * 4 // 5)
        cy = rng.randint(size // 8, size // 3)
    axes = (rng.randint(size // 14, size // 8), rng.randint(size // 28, size // 14))
    angle = rng.uniform(-45, 45)
    color = (rng.randint(20, 80), rng.randint(95, 175), rng.randint(25, 85))
    cv2.ellipse(canvas, (cx, cy), axes, angle, 0, 360, color, -1, cv2.LINE_AA)
    return canvas


def make_fruit_mimic(class_name, texture_path, args, rng, stats):
    size = args.crop_size
    hard_base = args.orange_hard_ratio if class_name == "orange" else args.hard_ratio
    hard = rng.random() < hard_base
    white_margin_hard = hard and rng.random() < args.white_margin_hard_ratio

    canvas = make_face_background(size, rng, mode="paper")
    tex = normalize_texture(read_image(texture_path, unchanged=True))
    tex = random_source_crop(tex, rng, hard=hard)

    if white_margin_hard:
        scale = rng.uniform(0.38, 0.72)
        offset = size * rng.uniform(0.08, 0.23)
    else:
        scale = rng.uniform(0.60, 1.10) if hard else rng.uniform(0.72, 1.02)
        offset = size * rng.uniform(0.00, 0.13)
    patch_side = max(12, int(round(size * scale)))
    tex = cv2.resize(tex, (patch_side, patch_side), interpolation=cv2.INTER_AREA if patch_side < max(tex.shape[:2]) else cv2.INTER_CUBIC)
    tex = print_texture_damage(tex, rng, hard=hard)

    if rng.random() < 0.85:
        angle = rng.uniform(-42, 42) if hard else rng.uniform(-24, 24)
        tex = rotate_patch(tex, angle, border_value=(245, 245, 245))

    center_x = size * 0.5 + rng.uniform(-offset, offset)
    center_y = size * 0.5 + rng.uniform(-offset, offset)
    if white_margin_hard and rng.random() < 0.5:
        center_y += rng.uniform(size * 0.04, size * 0.18)
    canvas = paste_patch(canvas, tex, center_x, center_y)

    if class_name == "orange" and hard and rng.random() < args.orange_green_distractor_prob:
        canvas = draw_orange_green_distractor(canvas, rng)
        stats["orange_green_distractor"] += 1

    out = final_degrade(canvas, rng, args, hard=hard)
    stats[f"{class_name}_hard"] += int(hard)
    stats[f"{class_name}_white_margin_hard"] += int(white_margin_hard)
    return out, {
        "class": class_name,
        "source_texture": str(texture_path),
        "hard": hard,
        "white_margin_hard": white_margin_hard,
    }


def make_plain_mimic(args, rng, stats):
    canvas = make_face_background(args.crop_size, rng, mode="plain")
    if rng.random() < 0.35:
        x1 = rng.randint(0, args.crop_size - 1)
        x2 = clamp(x1 + rng.randint(-60, 60), 0, args.crop_size - 1)
        cv2.line(canvas, (x1, 0), (x2, args.crop_size - 1), (rng.randint(180, 235),) * 3, rng.choice([1, 2]))
    hard = rng.random() < args.hard_ratio
    out = final_degrade(canvas, rng, args, hard=hard)
    stats["plain_hard"] += int(hard)
    return out, {"class": "plain", "mode": "plain_face", "hard": hard}


def random_crop_resize(image, size, rng):
    h, w = image.shape[:2]
    if h <= 2 or w <= 2:
        return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    side = rng.randint(max(2, min(h, w) // 5), min(h, w))
    x1 = rng.randint(0, max(0, w - side))
    y1 = rng.randint(0, max(0, h - side))
    crop = image[y1 : y1 + side, x1 : x1 + side]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def make_unknown_mimic(args, rng, background_paths, texture_pools, stats):
    size = args.crop_size
    mode = rng.choices(
        ["background", "edge_shadow", "blank_noisy", "ambiguous_blur"],
        weights=[0.45 if background_paths else 0.0, 0.25, 0.20, 0.10],
        k=1,
    )[0]
    if mode == "background":
        image = read_image(rng.choice(background_paths), unchanged=False)
        canvas = random_crop_resize(image, size, rng)
    elif mode == "ambiguous_blur":
        cls = rng.choice(FRUIT_CLASSES)
        image = normalize_texture(read_image(rng.choice(texture_pools[cls]), unchanged=True))
        canvas = random_crop_resize(image, size, rng)
        canvas = cv2.GaussianBlur(canvas, (rng.choice([9, 13, 17]),) * 2, 0)
        if rng.random() < 0.7:
            canvas = cv2.cvtColor(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    else:
        canvas = make_face_background(size, rng, mode="unknown")
        if mode == "edge_shadow":
            color = rng.choice([(0, 0, 0), (30, 30, 30), (245, 245, 245)])
            pts = np.asarray(
                [
                    [rng.randint(0, size // 4), rng.randint(0, size)],
                    [rng.randint(size // 2, size), rng.randint(0, size)],
                    [rng.randint(size // 2, size), rng.randint(0, size)],
                    [rng.randint(0, size // 4), rng.randint(0, size)],
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(canvas, [pts.reshape(-1, 1, 2)], color)
    hard = True
    canvas = final_degrade(canvas, rng, args, hard=hard)
    stats[f"unknown_{mode}"] += 1
    return canvas, {"class": "unknown", "mode": mode, "hard": hard}


def make_sample(class_name, split, sample_index, texture_pools, background_paths, args, rng, stats):
    if class_name in FRUIT_CLASSES:
        texture_path = rng.choice(texture_pools[class_name])
        image, meta = make_fruit_mimic(class_name, texture_path, args, rng, stats)
    elif class_name == "plain":
        image, meta = make_plain_mimic(args, rng, stats)
    elif class_name == "unknown":
        image, meta = make_unknown_mimic(args, rng, background_paths, texture_pools, stats)
    else:
        raise ValueError(class_name)

    quality = rng.randint(args.jpeg_quality_min, args.jpeg_quality_max)
    out_name = f"{split}_{class_name}_{sample_index:06d}.jpg"
    out_path = args.output_root / split / class_name / out_name
    meta.update({"split": split, "path": str(out_path), "jpeg_quality": quality})
    return out_path, image, quality, meta


def make_contact_sheet(output_root, splits, count_per_class):
    thumbs = []
    split = "train" if "train" in splits else splits[0]
    for class_name in C_CLASSES:
        paths = sorted((output_root / split / class_name).glob("*.jpg"))[:count_per_class]
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
            cv2.rectangle(image, (0, 0), (128, 22), (0, 0, 0), -1)
            cv2.putText(image, class_name, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            thumbs.append(image)
    if not thumbs:
        return None
    cols = min(6, max(1, len(C_CLASSES)))
    rows = []
    for idx in range(0, len(thumbs), cols):
        row = thumbs[idx : idx + cols]
        while len(row) < cols:
            row.append(np.full((128, 128, 3), 245, dtype=np.uint8))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    out_path = output_root / "_preview" / "contact_sheet.jpg"
    write_jpg(out_path, sheet, quality=95)
    return out_path


def write_classes_file(output_root):
    data = {"classes": list(C_CLASSES), "format": "split/class/*.jpg", "trainer": "scripts/train_face_mobilenetv3.py"}
    (output_root / "classes.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    args.texture_dir = args.texture_dir.resolve()
    args.output_root = args.output_root.resolve()
    args.background_dir = args.background_dir.resolve()
    if args.apple_orange_boost:
        args.orange_hard_ratio = max(args.orange_hard_ratio, 0.55)
        args.white_margin_hard_ratio = max(args.white_margin_hard_ratio, 0.45)

    if args.reset:
        reset_output(args.output_root)
    ensure_dirs(args.output_root, args.splits)
    write_classes_file(args.output_root)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    texture_pools = collect_texture_pool(args.texture_dir)
    extend_orange_pool(texture_pools, args.extra_orange_texture_dir, args.extra_orange_weight)
    background_paths = collect_images(args.background_dir)

    audit = {
        "script": "generate_2d_face_mimic_dataset.py",
        "texture_dir": str(args.texture_dir),
        "background_dir": str(args.background_dir),
        "output_root": str(args.output_root),
        "crop_size": args.crop_size,
        "render_dirty": bool(args.render_dirty),
        "projected_resolution_prob": args.projected_resolution_prob,
        "lens_like_distortion_prob": args.lens_like_distortion_prob,
        "seed": args.seed,
        "splits": list(args.splits),
        "texture_counts": {key: len(value) for key, value in texture_pools.items()},
        "background_count": len(background_paths),
        "counts": {},
        "stats": {},
        "notes": [
            "No Blender/3D rendering is used.",
            "Images are synthetic 2D face patches meant to augment C classifier robustness.",
            "Default mode intentionally mimics dirty final-render C crops: blur, low-res resampling, lens-like warp, noise, vignetting, JPEG, and crop jitter.",
            "Orange hard cases intentionally include white margin, crop jitter, and optional green distractors.",
        ],
    }
    stats = Counter()
    metadata_path = args.output_root / "mimic_metadata.csv"
    fields = ["split", "class", "path", "source_texture", "mode", "hard", "white_margin_hard", "jpeg_quality"]
    active_classes = tuple(args.classes)

    with metadata_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for split in args.splits:
            audit["counts"][split] = {}
            for class_name in active_classes:
                count = split_count(args, split, class_name)
                audit["counts"][split][class_name] = count
                iterator = tqdm(range(count), desc=f"{split}/{class_name}", unit="img")
                for sample_index in iterator:
                    out_path = args.output_root / split / class_name / f"{split}_{class_name}_{sample_index:06d}.jpg"
                    if args.resume and out_path.exists():
                        stats["skipped_resume"] += 1
                        continue
                    out_path, image, quality, meta = make_sample(
                        class_name, split, sample_index, texture_pools, background_paths, args, rng, stats
                    )
                    write_jpg(out_path, image, quality=quality)
                    row = {field: meta.get(field, "") for field in fields}
                    writer.writerow(row)
                    stats[f"class_{class_name}"] += 1

    preview = make_contact_sheet(args.output_root, args.splits, args.preview_count)
    audit["stats"] = dict(stats)
    audit["preview"] = str(preview) if preview else None
    audit["metadata_csv"] = str(metadata_path)
    audit_path = args.output_root / "mimic_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote: {args.output_root}")
    print(f"audit: {audit_path}")
    print(f"metadata: {metadata_path}")
    if preview:
        print(f"preview: {preview}")
    print(json.dumps(audit["counts"], ensure_ascii=False, indent=2))
    print(json.dumps(dict(stats), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
