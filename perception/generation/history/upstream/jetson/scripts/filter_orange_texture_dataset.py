import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BAD_URL_TOKENS = (
    "aircraft",
    "airplane",
    "airport",
    "australia",
    "cabin",
    "cleanpark",
    "concord",
    "garbage",
    "gomi",
    "gomiflow",
    "highway",
    "hotate",
    "interior",
    "jet",
    "map",
    "maps",
    "passenger",
    "plane",
    "scallop",
    "shaded_relief",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Filter downloaded orange texture images with ImageNet + color checks.")
    parser.add_argument("--input_root", type=Path, required=True, help="Root containing orange/*.jpg or the orange dir itself.")
    parser.add_argument("--output_root", type=Path, required=True, help="Clean output root. Writes output_root/orange/*.jpg.")
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--mode", choices=["copy", "move", "hardlink"], default="copy")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--min_orange_prob", type=float, default=0.20)
    parser.add_argument("--min_citrus_prob", type=float, default=0.35)
    parser.add_argument("--max_blue_cyan", type=float, default=0.22)
    parser.add_argument("--max_green", type=float, default=0.34)
    parser.add_argument("--max_valid_if_not_orange_dense", type=float, default=0.90)
    parser.add_argument("--allow_url_blacklist", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def image_dir(root):
    candidate = root / "orange"
    return candidate if candidate.exists() else root


def collect_images(root):
    src_dir = image_dir(root)
    return sorted(p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_metadata(path):
    if not path or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for item in data:
        name = Path(item.get("file", "")).name
        if name:
            result[name] = item
    return result


def hsv_ratios(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    valid = (s > 35) & (v > 45)
    orange = ((h >= 3) & (h <= 28)) & (s > 45) & (v > 50)
    red_orange = (((h <= 12) | ((h >= 170) & (h <= 179))) | ((h >= 13) & (h <= 28))) & (s > 40) & (v > 45)
    green = ((h >= 35) & (h <= 95)) & (s > 25) & (v > 35)
    blue_cyan = ((h >= 85) & (h <= 130)) & (s > 35) & (v > 45)
    white = (s < 35) & (v > 180)
    return {
        "orange": float(orange.mean()),
        "red_orange": float(red_orange.mean()),
        "green": float(green.mean()),
        "blue_cyan": float(blue_cyan.mean()),
        "white": float(white.mean()),
        "valid": float(valid.mean()),
    }


def url_blacklisted(meta):
    text = " ".join(str(meta.get(k, "")) for k in ("url", "file", "query")).lower()
    return any(token in text for token in BAD_URL_TOKENS)


def load_imagenet_model(device_arg):
    device = torch.device(device_arg if device_arg != "cuda" or torch.cuda.is_available() else "cpu")
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = mobilenet_v3_large(weights=weights).to(device).eval()
    return model, weights.transforms(), weights.meta["categories"], device


def classify_image(path, model, preprocess, classes, device):
    image = Image.open(path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].detach().cpu().numpy()
    class_probs = {name: float(probs[idx]) for idx, name in enumerate(classes)}
    top_idx = np.argsort(-probs)[:5]
    top5 = [{"class": classes[int(idx)], "prob": float(probs[int(idx)])} for idx in top_idx]
    orange_prob = class_probs.get("orange", 0.0)
    lemon_prob = class_probs.get("lemon", 0.0)
    return orange_prob, lemon_prob, top5


def read_bgr(path):
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to decode {path}")
    return image


def accept_image(path, meta, args, model, preprocess, classes, device):
    image = read_bgr(path)
    ratios = hsv_ratios(image)
    orange_prob, lemon_prob, top5 = classify_image(path, model, preprocess, classes, device)
    citrus_prob = orange_prob + lemon_prob

    reasons = []
    if not args.allow_url_blacklist and url_blacklisted(meta):
        reasons.append("url_blacklist")
    if ratios["blue_cyan"] > args.max_blue_cyan and orange_prob < 0.60:
        reasons.append("too_much_blue_cyan")
    if ratios["green"] > args.max_green and orange_prob < 0.60:
        reasons.append("too_much_green")
    if ratios["valid"] > args.max_valid_if_not_orange_dense and ratios["orange"] < 0.35:
        reasons.append("scene_like_saturated_image")
    if orange_prob < args.min_orange_prob and citrus_prob < args.min_citrus_prob:
        reasons.append("imagenet_not_citrus")
    if ratios["orange"] < 0.035 or ratios["red_orange"] < 0.05:
        reasons.append("not_enough_orange_pixels")

    return not reasons, {
        "ratios": ratios,
        "imagenet_orange_prob": orange_prob,
        "imagenet_lemon_prob": lemon_prob,
        "imagenet_citrus_prob": citrus_prob,
        "imagenet_top5": top5,
        "reject_reasons": reasons,
    }


def transfer(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    else:
        try:
            dst.hardlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)


def main():
    args = parse_args()
    if args.reset and args.output_root.exists():
        shutil.rmtree(args.output_root)
    accepted_dir = args.output_root / "orange"
    rejected_dir = args.output_root / "_rejected"
    accepted_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.metadata)
    model, preprocess, classes, device = load_imagenet_model(args.device)
    audit = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "accepted": [],
        "rejected": [],
        "settings": {
            "min_orange_prob": args.min_orange_prob,
            "min_citrus_prob": args.min_citrus_prob,
            "max_blue_cyan": args.max_blue_cyan,
            "max_green": args.max_green,
        },
    }

    accepted_count = 0
    for src in tqdm(collect_images(args.input_root), desc="filter orange textures", unit="img"):
        meta = metadata.get(src.name, {})
        try:
            ok, details = accept_image(src, meta, args, model, preprocess, classes, device)
        except Exception as exc:
            ok = False
            details = {"reject_reasons": [f"error:{exc}"]}

        record = {
            "source": str(src),
            "source_name": src.name,
            "url": meta.get("url"),
            "query": meta.get("query"),
            **details,
        }
        if ok:
            dst = accepted_dir / f"orange_hard_clean_{accepted_count:04d}.jpg"
            transfer(src, dst, args.mode)
            record["output"] = str(dst)
            audit["accepted"].append(record)
            accepted_count += 1
        else:
            dst = rejected_dir / src.name
            transfer(src, dst, "copy")
            record["output"] = str(dst)
            audit["rejected"].append(record)

    (args.output_root / "filter_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    clean_meta = []
    for item in audit["accepted"]:
        clean_meta.append(
            {
                "file": item["output"],
                "source": item["source"],
                "url": item.get("url"),
                "query": item.get("query"),
                "ratios": item.get("ratios"),
                "imagenet_orange_prob": item.get("imagenet_orange_prob"),
                "imagenet_lemon_prob": item.get("imagenet_lemon_prob"),
                "imagenet_top5": item.get("imagenet_top5"),
            }
        )
    (args.output_root / "metadata.json").write_text(json.dumps(clean_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"accepted": len(audit["accepted"]), "rejected": len(audit["rejected"])}, ensure_ascii=False, indent=2))
    print(f"clean_output={args.output_root}")


if __name__ == "__main__":
    main()
