from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


CLASSES = ["apple", "orange", "banana", "pineapple"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def list_images(root: Path, class_name: str) -> list[Path]:
    class_dir = root / class_name
    if not class_dir.exists():
        return []
    return sorted(path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def read_image(path: Path, max_side: int) -> tuple[np.ndarray, np.ndarray | None]:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    alpha = None
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3]
        bgr = image[:, :, :3]
    elif image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr = image[:, :, :3]

    h, w = bgr.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        bgr = cv2.resize(bgr, new_size, interpolation=cv2.INTER_AREA)
        if alpha is not None:
            alpha = cv2.resize(alpha, new_size, interpolation=cv2.INTER_AREA)
    return bgr, alpha


def smooth_hist(hist: np.ndarray) -> np.ndarray:
    kernel = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    padded = np.r_[hist[-2:], hist, hist[:2]]
    return np.convolve(padded, kernel / kernel.sum(), mode="same")[2:-2]


def hue_metrics(path: Path, max_side: int) -> dict[str, float | int | str]:
    bgr, alpha = read_image(path, max_side)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = (s >= 24) & (v >= 30) & ~((s < 38) & (v > 220))
    if alpha is not None:
        mask &= alpha > 20
    if int(mask.sum()) < max(80, int(h.size * 0.01)):
        mask = (s >= 14) & (v >= 25)
        if alpha is not None:
            mask &= alpha > 20
    if int(mask.sum()) == 0:
        mask = np.ones_like(h, dtype=bool)

    hv = h[mask].astype(np.int32)
    sv = s[mask].astype(np.float32)
    vv = v[mask].astype(np.float32)
    weights = np.clip(sv, 8, 255)
    hist = np.bincount(hv, weights=weights, minlength=180).astype(np.float64)
    if hist.sum() <= 0:
        hist = np.bincount(hv, minlength=180).astype(np.float64)
    hist /= max(float(hist.sum()), 1e-9)
    dominant_h = int(smooth_hist(hist).argmax())

    def mass(lo: int, hi: int) -> float:
        return float(hist[lo:hi].sum())

    red_wrap = float(hist[:8].sum() + hist[172:].sum())
    red_orange = mass(8, 15)
    orange = mass(15, 27)
    yellow = mass(27, 40)
    yellow_green = mass(40, 65)
    green = mass(65, 90)
    cyan_blue = mass(90, 130)
    purple_magenta = mass(130, 172)
    return {
        "dominant_h": dominant_h,
        "mask_pixels": int(mask.sum()),
        "mask_ratio": float(mask.mean()),
        "mean_s": float(sv.mean()) if len(sv) else 0.0,
        "mean_v": float(vv.mean()) if len(vv) else 0.0,
        "red_wrap": red_wrap,
        "red_orange": red_orange,
        "orange": orange,
        "yellow": yellow,
        "yellow_green": yellow_green,
        "green": green,
        "cyan_blue": cyan_blue,
        "purple_magenta": purple_magenta,
        "redish": red_wrap + red_orange,
        "strict_oy": orange + yellow,
        "warm": red_orange + orange + yellow,
        "greenish": yellow_green + green,
        "cool": cyan_blue + purple_magenta,
    }


def decision(class_name: str, m: dict[str, float | int | str]) -> tuple[bool, str]:
    h = int(m["dominant_h"])
    redish = float(m["redish"])
    strict_oy = float(m["strict_oy"])
    warm = float(m["warm"])
    greenish = float(m["greenish"])
    cool = float(m["cool"])
    red_wrap = float(m["red_wrap"])
    orange = float(m["orange"])
    yellow = float(m["yellow"])
    mean_s = float(m["mean_s"])
    mask_ratio = float(m["mask_ratio"])

    if mask_ratio < 0.01 or mean_s < 18:
        return False, "too_low_saturation_or_mask"

    if class_name == "apple":
        if cool > 0.12:
            return False, "apple_cool_outlier"
        if greenish > 0.35:
            return False, "apple_green_outlier"
        if 40 <= h < 90 and greenish > 0.18:
            return False, "apple_green_dominant"
        if 15 <= h < 40 and strict_oy > 0.60 and redish < 0.35:
            return False, "apple_orange_yellow_dominant"
        if strict_oy > 0.78 and redish < 0.25:
            return False, "apple_strong_yellow_orange"
        return True, "keep"

    if class_name == "orange":
        if cool > 0.08:
            return False, "orange_cool_outlier"
        if greenish > 0.18:
            return False, "orange_green_outlier"
        if warm < 0.55 and redish + orange < 0.45:
            return False, "orange_not_warm_enough"
        if 35 <= h < 165:
            return False, "orange_hue_outlier"
        return True, "keep"

    if class_name == "banana":
        if cool > 0.10:
            return False, "banana_cool_outlier"
        if greenish > 0.30:
            return False, "banana_green_outlier"
        if warm < 0.50 and yellow < 0.35:
            return False, "banana_not_yellow_enough"
        if not (8 <= h < 45):
            return False, "banana_hue_outlier"
        return True, "keep"

    if class_name == "pineapple":
        if cool > 0.16:
            return False, "pineapple_cool_outlier"
        if red_wrap > 0.75 and orange + yellow + greenish < 0.20:
            return False, "pineapple_red_outlier"
        if yellow > 0.78 and orange + float(m["red_orange"]) + greenish < 0.18:
            return False, "pineapple_yellow_flesh_like_outlier"
        if warm + greenish < 0.45:
            return False, "pineapple_not_brown_green_enough"
        return True, "keep"

    return True, "keep"


def strict_identity_decision(class_name: str, m: dict[str, float | int | str]) -> tuple[bool, str]:
    """Tighter identity-preserving filter for noisy web-crawled fruit textures."""
    h = int(m["dominant_h"])
    red_wrap = float(m["red_wrap"])
    red_orange = float(m["red_orange"])
    orange = float(m["orange"])
    yellow = float(m["yellow"])
    redish = float(m["redish"])
    strict_oy = float(m["strict_oy"])
    warm = float(m["warm"])
    greenish = float(m["greenish"])
    cool = float(m["cool"])
    mean_s = float(m["mean_s"])
    mask_ratio = float(m["mask_ratio"])

    if mask_ratio < 0.025 or mean_s < 32:
        return False, f"{class_name}_strict_low_saturation_or_mask"

    if class_name == "apple":
        if cool > 0.08 or greenish > 0.22:
            return False, "apple_strict_cool_green_outlier"
        if strict_oy > 0.62 and red_wrap < 0.22:
            return False, "apple_strict_orange_yellow_outlier"
        if 13 <= h < 42 and red_wrap < 0.20:
            return False, "apple_strict_orange_dominant"
        if redish < 0.30 and red_wrap < 0.18:
            return False, "apple_strict_not_red_enough"
        return True, "keep"

    if class_name == "orange":
        if cool > 0.045 or greenish > 0.105:
            return False, "orange_strict_cool_green_outlier"
        if not (6 <= h < 31):
            return False, "orange_strict_hue_outlier"
        if orange < 0.32 or warm < 0.68:
            return False, "orange_strict_not_orange_enough"
        if red_wrap > 0.42 and orange < 0.48:
            return False, "orange_strict_apple_red_outlier"
        if yellow > 0.58 and orange < 0.36:
            return False, "orange_strict_yellow_outlier"
        return True, "keep"

    if class_name == "banana":
        if cool > 0.075 or greenish > 0.22:
            return False, "banana_strict_cool_green_outlier"
        if not (13 <= h < 42):
            return False, "banana_strict_hue_outlier"
        if yellow < 0.30 and strict_oy < 0.58:
            return False, "banana_strict_not_yellow_enough"
        if red_wrap > 0.32 and yellow < 0.42:
            return False, "banana_strict_red_outlier"
        return True, "keep"

    if class_name == "pineapple":
        brown_green = red_orange + orange + yellow + greenish
        if cool > 0.11:
            return False, "pineapple_strict_cool_outlier"
        if red_wrap > 0.55 and brown_green < 0.45:
            return False, "pineapple_strict_red_outlier"
        if yellow > 0.68 and greenish < 0.08 and red_orange + orange < 0.18:
            return False, "pineapple_strict_yellow_flesh_outlier"
        if brown_green < 0.52:
            return False, "pineapple_strict_not_brown_green_enough"
        return True, "keep"

    return True, "keep"


def hue_distance_to_interval(h: int, lo: int, hi: int) -> int:
    """Circular distance from hue h to a half-open hue interval in OpenCV H space."""
    if lo <= h < hi:
        return 0
    return min((lo - h) % 180, (h - (hi - 1)) % 180)


def soft_outlier_score(class_name: str, m: dict[str, float | int | str]) -> tuple[float, str]:
    h = int(m["dominant_h"])
    red_wrap = float(m["red_wrap"])
    red_orange = float(m["red_orange"])
    orange = float(m["orange"])
    yellow = float(m["yellow"])
    redish = float(m["redish"])
    strict_oy = float(m["strict_oy"])
    warm = float(m["warm"])
    greenish = float(m["greenish"])
    cool = float(m["cool"])
    mean_s = float(m["mean_s"])

    low_s = max(0.0, 45.0 - mean_s) / 45.0
    if class_name == "apple":
        orange_yellow_pressure = strict_oy * 3.8 + max(0.0, strict_oy - 0.34) * 3.2
        red_credit = red_wrap * 1.1 + redish * 0.7
        dominant_oy = 0.9 if 12 <= h < 42 else 0.0
        score = orange_yellow_pressure + greenish * 1.4 + cool * 2.0 + low_s * 0.3 + dominant_oy - red_credit
        if strict_oy >= max(greenish, cool):
            return score, "apple_target10_orange_yellow_soft"
        if greenish >= cool:
            return score, "apple_target10_green_soft"
        return score, "apple_target10_cool_soft"

    if class_name == "orange":
        hue_distance = hue_distance_to_interval(h, 6, 34) / 35.0
        score = hue_distance + cool * 3.0 + greenish * 2.4 + max(0.0, 0.58 - warm) * 2.2 + low_s * 0.4
        if cool >= greenish and cool > 0.02:
            return score, "orange_target10_cool_soft"
        if greenish > 0.04:
            return score, "orange_target10_green_soft"
        return score, "orange_target10_hue_soft"

    if class_name == "banana":
        hue_distance = hue_distance_to_interval(h, 12, 45) / 35.0
        yellow_credit = yellow * 1.1
        score = hue_distance + greenish * 2.7 + cool * 2.8 + max(0.0, 0.55 - warm) * 1.8 + low_s * 0.3 - yellow_credit
        if greenish >= max(cool, strict_oy):
            return score, "banana_target10_green_soft"
        if cool > 0.04:
            return score, "banana_target10_cool_soft"
        return score, "banana_target10_hue_soft"

    if class_name == "pineapple":
        brown_green = red_orange + orange + yellow + greenish
        hue_distance = min(hue_distance_to_interval(h, 8, 64), hue_distance_to_interval(h, 170, 180)) / 45.0
        yellow_flesh_like = max(0.0, yellow - (orange + red_orange + greenish))
        score = hue_distance + cool * 2.5 + yellow_flesh_like * 2.4 + max(0.0, 0.50 - brown_green) * 2.0 + red_wrap * 0.5 + low_s * 0.3
        if yellow_flesh_like > max(cool, 0.08):
            return score, "pineapple_target10_yellow_flesh_soft"
        if cool > 0.05:
            return score, "pineapple_target10_cool_soft"
        return score, "pineapple_target10_hue_soft"

    return 0.0, "target10_soft"


def link_or_copy(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy_fallback"


def make_contact_sheet(rows: list[dict], output: Path, title_key: str, max_items: int = 96) -> None:
    if not rows:
        return
    rows = rows[:max_items]
    thumb = 104
    label_h = 22
    cols = 8
    canvas = np.full((math.ceil(len(rows) / cols) * (thumb + label_h), cols * thumb, 3), 245, dtype=np.uint8)
    for idx, row in enumerate(rows):
        img, _ = read_image(Path(row["source"]), thumb)
        img = cv2.resize(img, (thumb, thumb), interpolation=cv2.INTER_AREA)
        y = (idx // cols) * (thumb + label_h)
        x = (idx % cols) * thumb
        canvas[y + label_h : y + label_h + thumb, x : x + thumb] = img
        text = f"H{int(row['dominant_h'])} {row[title_key]}"[:24]
        cv2.putText(canvas, text, (x + 2, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (20, 20, 20), 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 94])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter fruit texture roots by conservative class-specific HSV hue ranges.")
    parser.add_argument("--source_root", type=Path, default=Path("datasets/fruit_textures/final_fruits36070_original30"))
    parser.add_argument("--output_root", type=Path, default=Path("datasets/fruit_textures/final_fruits36070_original30_color_filtered_v2"))
    parser.add_argument("--report_root", type=Path, default=Path("reports/cube_face_unified_eval/fruit_texture_color_filtered_v2"))
    parser.add_argument("--mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--max_side", type=int, default=384)
    parser.add_argument(
        "--target_reject_ratio",
        type=float,
        default=0.10,
        help="Optional per-class final reject ratio. Hard rejects are kept, then high-scoring soft outliers fill the target.",
    )
    parser.add_argument(
        "--strict_identity",
        action="store_true",
        help="Use tighter class-color gates for noisy web-crawled textures. Recommended for internet photos, not for already curated lab datasets.",
    )
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reset:
        reset_dir(args.output_root)
        reset_dir(args.report_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for class_name in CLASSES:
        for src in list_images(args.source_root, class_name):
            metrics = hue_metrics(src, args.max_side)
            keep, reason = decision(class_name, metrics)
            if keep and args.strict_identity:
                keep, reason = strict_identity_decision(class_name, metrics)
            soft_score, soft_reason = soft_outlier_score(class_name, metrics)
            dst = args.output_root / class_name / src.name
            row = {
                "class": class_name,
                "source": str(src),
                "output": "",
                "keep": int(keep),
                "reason": reason,
                "initial_keep": int(keep),
                "initial_reason": reason,
                "soft_score": float(soft_score),
                "soft_reason": soft_reason,
                "target_reject_ratio": float(args.target_reject_ratio),
                "_dst": str(dst),
            }
            row.update(metrics)
            rows.append(row)

    if args.target_reject_ratio < 0 or args.target_reject_ratio >= 1:
        raise ValueError("--target_reject_ratio must be >= 0 and < 1")

    if args.target_reject_ratio > 0:
        for class_name in CLASSES:
            class_rows = [row for row in rows if row["class"] == class_name]
            target_rejects = int(round(len(class_rows) * args.target_reject_ratio))
            current_rejects = sum(1 for row in class_rows if not row["keep"])
            extra_needed = max(0, target_rejects - current_rejects)
            if extra_needed <= 0:
                continue
            candidates = [row for row in class_rows if row["keep"]]
            candidates.sort(key=lambda row: float(row["soft_score"]), reverse=True)
            for row in candidates[:extra_needed]:
                row["keep"] = 0
                row["reason"] = row["soft_reason"]

    stats: Counter = Counter()
    for row in rows:
        if row["keep"]:
            dst = Path(str(row["_dst"]))
            link_or_copy(Path(str(row["source"])), dst, args.mode)
            row["output"] = str(dst)
        row.pop("_dst", None)
        class_name = str(row["class"])
        keep = bool(row["keep"])
        stats[f"{class_name}_{'keep' if keep else 'reject'}"] += 1
        if not keep:
            stats[f"{class_name}_{row['reason']}"] += 1

    fields = sorted({key for row in rows for key in row.keys()})
    with (args.report_root / "records.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    rejected = [row for row in rows if not row["keep"]]
    for class_name in CLASSES:
        class_rejected = [row for row in rejected if row["class"] == class_name]
        class_rejected.sort(key=lambda row: (row["reason"], -float(row["cool"]), -float(row["strict_oy"]), -float(row["greenish"])))
        make_contact_sheet(class_rejected, args.report_root / f"rejected_{class_name}_contact_sheet.jpg", "reason")
        kept = [row for row in rows if row["class"] == class_name and row["keep"]]
        kept.sort(key=lambda row: abs(int(row["dominant_h"]) - {"apple": 2, "orange": 14, "banana": 23, "pineapple": 14}[class_name]))
        make_contact_sheet(kept[:96], args.report_root / f"kept_{class_name}_sample_contact_sheet.jpg", "class")
        soft_rejected = [
            row for row in class_rejected if int(row["initial_keep"]) == 1 and str(row["reason"]).startswith(f"{class_name}_target")
        ]
        soft_rejected.sort(key=lambda row: float(row["soft_score"]), reverse=True)
        make_contact_sheet(soft_rejected, args.report_root / f"rejected_{class_name}_soft_target_contact_sheet.jpg", "reason")

    summary = {
        "task": "fruit_texture_color_filter",
        "policy": "Create a new filtered texture root only; do not modify the source texture root.",
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "report_root": str(args.report_root),
        "mode": args.mode,
        "target_reject_ratio": args.target_reject_ratio,
        "strict_identity": args.strict_identity,
        "stats": dict(stats),
        "criteria": {
            "hard_filter": "remove clear class-specific HSV outliers first",
            "target_ratio": "when target_reject_ratio is set, fill the per-class reject target with the highest soft outlier scores",
            "apple": "prioritize orange/yellow-like apple outliers, then green/cool outliers, while keeping red/red-orange apples",
            "orange": "keep warm orange/red-orange/yellow oranges; softly remove cool/green/hue-edge outliers",
            "banana": "keep yellow bananas; softly remove green/cool/out-of-range hue outliers",
            "pineapple": "keep brown/green whole-pineapple hues; softly remove cool, red-only, or yellow-flesh-like outliers",
        },
    }
    (args.report_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
