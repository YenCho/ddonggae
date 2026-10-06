import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add conservative unknown hard negatives to c_facecls from A2 crops."
    )
    parser.add_argument("--model_root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--min_side", type=int, default=48)
    parser.add_argument("--max_side", type=int, default=120)
    parser.add_argument("--max_face_overlap", type=float, default=0.02)
    parser.add_argument("--max_trials_per_image", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--reset_unknown", action="store_true")
    return parser.parse_args()


def denorm_segment(values, width, height):
    pts = np.asarray(values, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] = np.clip(pts[:, 0], 0, 1) * (width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, 1) * (height - 1)
    return np.rint(pts).astype(np.int32)


def load_face_mask(label_path, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)
    if not label_path.exists():
        return mask
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        values = [float(v) for v in parts[1:]]
        if len(values) < 6 or len(values) % 2 != 0:
            continue
        pts = denorm_segment(values, width, height)
        cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 255)
    return mask


def valid_crop(image, face_mask, box, max_face_overlap):
    x1, y1, x2, y2 = box
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    mask_crop = face_mask[y1:y2, x1:x2]
    overlap = float((mask_crop > 0).mean())
    if overlap > max_face_overlap:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if gray.std() < 3.0:
        return False
    return True


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    image_dir = args.model_root / "a2_faceseg" / "images" / args.split
    label_dir = args.model_root / "a2_faceseg" / "labels" / args.split
    out_dir = args.model_root / "c_facecls" / args.split / "unknown"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.reset_unknown:
        for path in out_dir.glob("*.jpg"):
            path.unlink()

    images = sorted(image_dir.glob("*.jpg"))
    rng.shuffle(images)
    made = 0

    with tqdm(total=args.count, desc="unknown hard negatives", unit="crop") as pbar:
        while made < args.count:
            progressed = False
            for image_path in images:
                if made >= args.count:
                    break
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                h, w = image.shape[:2]
                face_mask = load_face_mask(label_dir / f"{image_path.stem}.txt", w, h)
                for _ in range(args.max_trials_per_image):
                    side = rng.randint(args.min_side, min(args.max_side, w, h))
                    if w <= side or h <= side:
                        continue
                    x1 = rng.randint(0, w - side)
                    y1 = rng.randint(0, h - side)
                    box = (x1, y1, x1 + side, y1 + side)
                    if not valid_crop(image, face_mask, box, args.max_face_overlap):
                        continue
                    crop = image[y1:y1 + side, x1:x1 + side]
                    crop = cv2.resize(crop, (args.crop_size, args.crop_size), interpolation=cv2.INTER_AREA)
                    out_path = out_dir / f"{image_path.stem}_unknown_{made:06d}.jpg"
                    cv2.imwrite(str(out_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
                    made += 1
                    progressed = True
                    pbar.update(1)
                    break
            if not progressed:
                break

    print(f"unknown hard negatives written: {made}/{args.count}")
    print(f"output: {out_dir}")


if __name__ == "__main__":
    main()
