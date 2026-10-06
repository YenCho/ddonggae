import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Add small warm-tinted plain hard negatives to C face classifier data. "
            "These simulate white cube faces reflecting a warm wood floor."
        )
    )
    parser.add_argument("--c_facecls_root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--count", type=int, default=1500)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--prefix", default="warmplain")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--preview", type=Path, default=None)
    parser.add_argument("--preview_count", type=int, default=40)
    return parser.parse_args()


def collect_plain_sources(plain_dir, prefix):
    paths = []
    for path in plain_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        if path.name.startswith(f"{prefix}_"):
            continue
        paths.append(path)
    return sorted(paths)


def read_image(path):
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    return image


def write_jpg(path, image, quality):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"failed to encode jpg: {path}")
    path.write_bytes(buf.tobytes())


def smooth_noise(shape, rng, scale):
    h, w = shape[:2]
    small_h = max(2, h // rng.randint(18, 36))
    small_w = max(2, w // rng.randint(18, 36))
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0.0, 1.0, (small_h, small_w)).astype(np.float32)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=rng.uniform(2.0, 6.0))
    return noise * float(scale)


def warm_reflection_variant(image, rng, crop_size):
    image = cv2.resize(image, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
    # Existing plain crops can inherit odd scene colors. Pull them back toward a
    # low-saturation white/gray cube face before adding wood-floor warmth.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    out = image.astype(np.float32) * 0.22 + gray_bgr * 0.78
    h, w = out.shape[:2]

    # Pale warm BGR colors only. Keep saturation low so this stays a plain face,
    # not a fruit-like orange patch.
    warm_color = np.asarray(
        [
            rng.uniform(185, 226),  # B
            rng.uniform(215, 246),  # G
            rng.uniform(232, 255),  # R
        ],
        dtype=np.float32,
    )

    x = np.linspace(0.0, 1.0, w, dtype=np.float32)
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    direction = rng.choice(["left", "right", "top", "bottom", "corner"])
    if direction == "left":
        field = 1.0 - xx
    elif direction == "right":
        field = xx
    elif direction == "top":
        field = 1.0 - yy
    elif direction == "bottom":
        field = yy
    else:
        cx = rng.choice([0.0, 1.0])
        cy = rng.choice([0.0, 1.0])
        field = 1.0 - np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / np.sqrt(2.0)
    field = np.clip(field, 0.0, 1.0)

    strength = rng.uniform(0.10, 0.26)
    field = field ** rng.uniform(0.75, 1.65)
    alpha = (strength * field)[:, :, None]
    out = out * (1.0 - alpha) + warm_color[None, None, :] * alpha

    if rng.random() < 0.55:
        grain_axis = xx if rng.random() < 0.5 else yy
        freq = rng.uniform(7.0, 18.0)
        phase = rng.uniform(0.0, 6.28318)
        grain = np.sin((grain_axis * freq + smooth_noise(out.shape, rng, 0.07)) * 6.28318 + phase)
        grain += smooth_noise(out.shape, rng, rng.uniform(0.25, 0.65))
        grain = cv2.GaussianBlur(grain.astype(np.float32), (0, 0), sigmaX=rng.uniform(0.8, 1.8))
        out += grain[:, :, None] * rng.uniform(1.0, 4.5)

    if rng.random() < 0.60:
        shadow = smooth_noise(out.shape, rng, rng.uniform(3.0, 9.0))
        out += shadow[:, :, None]

    if rng.random() < 0.45:
        # Mild edge/corner attenuation, similar to a real warped face crop.
        edge = np.minimum.reduce([xx, yy, 1.0 - xx, 1.0 - yy])
        edge = np.clip(edge / rng.uniform(0.08, 0.22), 0.0, 1.0)
        out *= (0.94 + 0.06 * edge[:, :, None])

    if rng.random() < 0.55:
        alpha_b = rng.uniform(0.92, 1.10)
        beta = rng.uniform(-8.0, 9.0)
        out = out * alpha_b + beta

    if rng.random() < 0.45:
        low = rng.randint(70, 170)
        low_img = cv2.resize(np.clip(out, 0, 255).astype(np.uint8), (low, low), interpolation=cv2.INTER_AREA)
        out = cv2.resize(low_img, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    if rng.random() < 0.38:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(np.clip(out, 0, 255).astype(np.uint8), (k, k), rng.uniform(0.3, 1.1)).astype(np.float32)

    if rng.random() < 0.28:
        angle = rng.uniform(0.0, np.pi)
        length = rng.choice([5, 7, 9])
        kernel = np.zeros((length, length), dtype=np.float32)
        cx = length // 2
        for i in range(length):
            x0 = int(round(cx + (i - cx) * np.cos(angle)))
            y0 = int(round(cx + (i - cx) * np.sin(angle)))
            if 0 <= x0 < length and 0 <= y0 < length:
                kernel[y0, x0] = 1.0
        kernel /= max(float(kernel.sum()), 1.0)
        out = cv2.filter2D(np.clip(out, 0, 255).astype(np.uint8), -1, kernel).astype(np.float32)

    out = np.clip(out, 0, 255).astype(np.uint8)

    if rng.random() < 0.75:
        quality = rng.randint(48, 88)
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out


def make_contact_sheet(paths, output_path, cols=8, tile=112):
    if not paths:
        return
    images = []
    for path in paths:
        image = read_image(path)
        image = cv2.resize(image, (tile, tile), interpolation=cv2.INTER_AREA)
        images.append(image)
    rows = int(np.ceil(len(images) / cols))
    sheet = np.full((rows * tile, cols * tile, 3), 245, dtype=np.uint8)
    for idx, image in enumerate(images):
        y = (idx // cols) * tile
        x = (idx % cols) * tile
        sheet[y : y + tile, x : x + tile] = image
    write_jpg(output_path, sheet, 92)


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    plain_dir = args.c_facecls_root / args.split / "plain"
    if not plain_dir.exists():
        raise RuntimeError(f"plain directory missing: {plain_dir}")

    if args.reset:
        for path in plain_dir.glob(f"{args.prefix}_*.jpg"):
            path.unlink()

    existing = sorted(plain_dir.glob(f"{args.prefix}_*.jpg"))
    if len(existing) >= args.count and args.resume:
        print(f"warm plain already present: {len(existing)} >= {args.count}")
        return

    sources = collect_plain_sources(plain_dir, args.prefix)
    if not sources:
        raise RuntimeError(f"no plain source images found: {plain_dir}")

    made = len(existing) if args.resume else 0
    if not args.resume:
        for path in existing:
            path.unlink()
        made = 0

    preview_paths = []
    with tqdm(total=args.count, initial=made, desc="warm plain", unit="img") as pbar:
        while made < args.count:
            source = rng.choice(sources)
            try:
                image = read_image(source)
            except RuntimeError:
                continue
            variant = warm_reflection_variant(image, rng, args.crop_size)
            out_path = plain_dir / f"{args.prefix}_{made:06d}_{source.stem[:72]}.jpg"
            if out_path.exists() and args.resume:
                made += 1
                pbar.update(1)
                continue
            write_jpg(out_path, variant, rng.randint(78, 94))
            if len(preview_paths) < args.preview_count:
                preview_paths.append(out_path)
            made += 1
            pbar.update(1)

    report = {
        "c_facecls_root": str(args.c_facecls_root),
        "split": args.split,
        "class": "plain",
        "prefix": args.prefix,
        "count": args.count,
        "source_count": len(sources),
        "seed": args.seed,
        "note": "Pale warm plain hard negatives only; no fruit-like texture is added.",
    }
    report_path = args.c_facecls_root / "_preview" / f"{args.prefix}_{args.split}_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.preview:
        make_contact_sheet(preview_paths, args.preview)

    print(f"warm plain hard negatives written: {made}/{args.count}")
    print(f"output: {plain_dir}")
    print(f"report: {report_path}")
    if args.preview:
        print(f"preview: {args.preview}")


if __name__ == "__main__":
    main()
