import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


CLASSES = ("apple", "orange", "banana", "pineapple", "plain", "unknown")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a tiny balanced C classifier anchor dataset from real target crops.")
    parser.add_argument("--apple_image", type=Path, required=True)
    parser.add_argument("--orange_image", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--samples_per_class", type=int, default=2000)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def imread_any(path):
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    return image


def write_jpg(path, image, quality):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"failed to encode {path}")
    path.write_bytes(buf.tobytes())


def base_variants(image):
    variants = []
    variants.append(("full", image.copy()))
    if image.shape[0] > 45:
        variants.append(("remove_top_30", image[30:].copy()))
        variants.append(("remove_top_24", image[24:].copy()))
        variants.append(("remove_top_36", image[36:].copy()))
    white = image.copy()
    white[: min(32, white.shape[0]), :, :] = 255
    variants.append(("top_whitened", white))
    gray = image.copy()
    gray[: min(32, gray.shape[0]), :, :] = 232
    variants.append(("top_gray", gray))
    return variants


def random_crop_resize(image, size, rng):
    h, w = image.shape[:2]
    pad = max(0, int(round(min(h, w) * rng.uniform(0.00, 0.04))))
    if pad > 0:
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT101)
        h, w = image.shape[:2]
    crop_margin_x = int(round(w * rng.uniform(0.00, 0.045)))
    crop_margin_y = int(round(h * rng.uniform(0.00, 0.045)))
    x1 = rng.randint(0, max(0, crop_margin_x))
    y1 = rng.randint(0, max(0, crop_margin_y))
    x2 = w - rng.randint(0, max(0, crop_margin_x))
    y2 = h - rng.randint(0, max(0, crop_margin_y))
    if x2 <= x1 + 8 or y2 <= y1 + 8:
        crop = image
    else:
        crop = image[y1:y2, x1:x2]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def color_jitter(image, rng):
    out = image.astype(np.float32)
    contrast = rng.uniform(0.90, 1.10)
    brightness = rng.uniform(-9, 9)
    out = np.clip(out * contrast + brightness, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-2.5, 2.5)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.88, 1.12), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * rng.uniform(0.92, 1.08), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def mild_geometric(image, rng):
    size = image.shape[0]
    angle = rng.uniform(-3.0, 3.0)
    scale = rng.uniform(0.975, 1.025)
    matrix = cv2.getRotationMatrix2D((size * 0.5, size * 0.5), angle, scale)
    matrix[0, 2] += rng.uniform(-2.5, 2.5)
    matrix[1, 2] += rng.uniform(-2.5, 2.5)
    return cv2.warpAffine(
        image,
        matrix,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def add_noise_and_blur(image, rng):
    out = image.copy()
    if rng.random() < 0.18:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)
    if rng.random() < 0.08:
        k = rng.choice([5, 7])
        kernel = np.zeros((k, k), dtype=np.float32)
        if rng.random() < 0.5:
            kernel[k // 2, :] = 1.0
        else:
            kernel[:, k // 2] = 1.0
        kernel /= kernel.sum()
        out = cv2.filter2D(out, -1, kernel)
    noise_sigma = rng.uniform(0.0, 3.5)
    if noise_sigma > 0.5:
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0, noise_sigma, out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def make_anchor_sample(variants, size, rng):
    name, image = rng.choice(variants)
    out = random_crop_resize(image, size, rng)
    out = mild_geometric(out, rng)
    out = color_jitter(out, rng)
    out = add_noise_and_blur(out, rng)
    if rng.random() < 0.20:
        out = cv2.resize(cv2.resize(out, (rng.randint(112, 180), rng.randint(112, 180)), interpolation=cv2.INTER_AREA), (size, size), interpolation=cv2.INTER_LINEAR)
    return out, name


def make_sheet(output_root, count=24):
    thumbs = []
    for class_name in ("apple", "orange"):
        for path in sorted((output_root / "train" / class_name).glob("*.jpg"))[:count]:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
            cv2.rectangle(image, (0, 0), (128, 20), (0, 0, 0), -1)
            cv2.putText(image, class_name, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            thumbs.append(image)
    if not thumbs:
        return None
    cols = 8
    rows = []
    for idx in range(0, len(thumbs), cols):
        row = thumbs[idx : idx + cols]
        while len(row) < cols:
            row.append(np.full((128, 128, 3), 245, dtype=np.uint8))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    out = output_root / "_preview" / "anchor_pair_contact_sheet.jpg"
    write_jpg(out, sheet, 94)
    return out


def main():
    args = parse_args()
    if args.reset and args.output_root.exists():
        shutil.rmtree(args.output_root)
    for class_name in CLASSES:
        (args.output_root / "train" / class_name).mkdir(parents=True, exist_ok=True)
    (args.output_root / "_preview").mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    sources = {
        "apple": {
            "image": args.apple_image,
            "variants": base_variants(imread_any(args.apple_image)),
        },
        "orange": {
            "image": args.orange_image,
            "variants": base_variants(imread_any(args.orange_image)),
        },
    }
    audit = {
        "script": "generate_c_anchor_pair_dataset.py",
        "samples_per_class": args.samples_per_class,
        "crop_size": args.crop_size,
        "seed": args.seed,
        "sources": {k: str(v["image"]) for k, v in sources.items()},
        "variant_counts": {k: len(v["variants"]) for k, v in sources.items()},
        "counts": {},
    }

    metadata = []
    for class_name, src in sources.items():
        audit["counts"][class_name] = args.samples_per_class
        for idx in tqdm(range(args.samples_per_class), desc=f"anchor/{class_name}", unit="img"):
            image, variant_name = make_anchor_sample(src["variants"], args.crop_size, rng)
            out = args.output_root / "train" / class_name / f"anchor_{class_name}_{idx:06d}.jpg"
            quality = rng.randint(74, 96)
            write_jpg(out, image, quality)
            metadata.append(
                {
                    "path": str(out),
                    "class": class_name,
                    "source": str(src["image"]),
                    "variant": variant_name,
                    "jpeg_quality": quality,
                }
            )

    (args.output_root / "classes.json").write_text(
        json.dumps({"classes": list(CLASSES), "format": "split/class/*.jpg"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_root / "anchor_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    preview = make_sheet(args.output_root)
    audit["preview"] = str(preview) if preview else None
    (args.output_root / "anchor_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
