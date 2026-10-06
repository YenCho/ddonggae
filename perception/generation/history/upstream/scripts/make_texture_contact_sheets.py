import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def make_sheet(paths, thumb_size, columns):
    thumbs = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            bgr = image[..., :3].astype(np.float32)
            alpha = image[..., 3:4].astype(np.float32) / 255.0
            white = np.full_like(bgr, 245, dtype=np.float32)
            image = np.clip(bgr * alpha + white * (1.0 - alpha), 0, 255).astype(np.uint8)
        else:
            image = image[..., :3]
        h, w = image.shape[:2]
        scale = (thumb_size - 8) / max(h, w)
        resized = cv2.resize(
            image,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        canvas = np.full((thumb_size, thumb_size, 3), 245, np.uint8)
        y0 = (thumb_size - resized.shape[0]) // 2
        x0 = (thumb_size - resized.shape[1]) // 2
        canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
        thumbs.append(canvas)

    if not thumbs:
        return None
    while len(thumbs) % columns:
        thumbs.append(np.full((thumb_size, thumb_size, 3), 245, np.uint8))
    rows = [np.hstack(thumbs[i:i + columns]) for i in range(0, len(thumbs), columns)]
    return np.vstack(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--thumb_size", type=int, default=104)
    parser.add_argument("--columns", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.root)
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        paths = sorted(path for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)[:args.count]
        sheet = make_sheet(paths, args.thumb_size, args.columns)
        if sheet is None:
            continue
        output = root / f"_preview_{class_dir.name}.jpg"
        cv2.imwrite(str(output), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        print(output)


if __name__ == "__main__":
    main()
