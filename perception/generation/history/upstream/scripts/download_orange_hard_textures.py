import argparse
import concurrent.futures
import io
import json
import random
import re
import shutil
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import requests


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"
}

QUERIES = [
    "orange slice with green leaf white background photo",
    "orange wedge with leaf fruit photo",
    "orange fruit slice leaf isolated photo",
    "mandarin orange slice with leaf white background",
    "orange half slice leaf close up photo",
    "orange slices green leaf on white background",
    "orange fruit cut slice with leaves photo",
    "fresh orange slice leaf isolated",
]

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
    parser = argparse.ArgumentParser(description="Download targeted hard orange textures for C classifier correction.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=250)
    parser.add_argument("--search_results", type=int, default=120)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--semantic_filter", action="store_true", default=True)
    parser.add_argument("--no_semantic_filter", dest="semantic_filter", action="store_false")
    parser.add_argument("--min_imagenet_orange_prob", type=float, default=0.20)
    parser.add_argument("--min_imagenet_citrus_prob", type=float, default=0.35)
    return parser.parse_args()


def url_blacklisted(url):
    lower = url.lower()
    return any(token in lower for token in BAD_URL_TOKENS)


def collect_urls(search_results):
    seen = set()
    items = []
    for query in QUERIES:
        for first in range(0, search_results, 35):
            params = {"q": query, "first": str(first), "form": "HDRSC2", "safeSearch": "moderate"}
            response = requests.get("https://www.bing.com/images/search", params=params, headers=HEADERS, timeout=20)
            response.raise_for_status()
            html = response.text
            matches = re.findall(r'murl&quot;:&quot;(.*?)&quot;', html)
            matches += re.findall(r'"murl":"(.*?)"', html)
            for url in matches:
                url = url.replace("\\/", "/")
                if not url.startswith("http") or url in seen:
                    continue
                if url_blacklisted(url):
                    continue
                seen.add(url)
                items.append({"url": url, "query": query})
    random.shuffle(items)
    return items


def hsv_ratios(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
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


def usable_orange_hard(image):
    h, w = image.shape[:2]
    if min(h, w) < 120 or max(h, w) / max(1, min(h, w)) > 4.0:
        return False, {}
    ratios = hsv_ratios(image)
    # Color is only a pre-filter. The semantic ImageNet pass below is what rejects
    # orange life jackets, maps, airplane cabins, and other non-fruit scenes.
    ok = (
        ratios["orange"] >= 0.035
        and ratios["red_orange"] >= 0.05
        and ratios["valid"] >= 0.08
        and ratios["blue_cyan"] <= 0.25
        and ratios["green"] <= 0.40
        and not (ratios["valid"] > 0.90 and ratios["orange"] < 0.35)
    )
    return ok, ratios


def load_semantic_filter():
    import torch
    from PIL import Image
    from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = mobilenet_v3_large(weights=weights).eval()
    return {
        "torch": torch,
        "Image": Image,
        "model": model,
        "preprocess": weights.transforms(),
        "classes": weights.meta["categories"],
    }


def semantic_orange_accept(data, semantic, args):
    torch = semantic["torch"]
    image = semantic["Image"].open(io.BytesIO(data)).convert("RGB")
    tensor = semantic["preprocess"](image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(semantic["model"](tensor), dim=1)[0].detach().cpu().numpy()
    classes = semantic["classes"]
    class_probs = {name: float(probs[idx]) for idx, name in enumerate(classes)}
    orange_prob = class_probs.get("orange", 0.0)
    lemon_prob = class_probs.get("lemon", 0.0)
    top_idx = np.argsort(-probs)[:5]
    detail = {
        "imagenet_orange_prob": orange_prob,
        "imagenet_lemon_prob": lemon_prob,
        "imagenet_citrus_prob": orange_prob + lemon_prob,
        "imagenet_top5": [{"class": classes[int(i)], "prob": float(probs[int(i)])} for i in top_idx],
    }
    ok = orange_prob >= args.min_imagenet_orange_prob or (
        orange_prob + lemon_prob >= args.min_imagenet_citrus_prob
    )
    return ok, detail


def ext_from_url(url):
    suffix = Path(urlparse(url).path).suffix.lower()
    return ".png" if suffix == ".png" else ".jpg"


def download_one(item):
    response = requests.get(item["url"], headers=HEADERS, timeout=18)
    response.raise_for_status()
    data = response.content
    if len(data) < 4096:
        raise RuntimeError("too small")
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("decode failed")
    ok, ratios = usable_orange_hard(image)
    if not ok:
        raise RuntimeError(f"filter failed {ratios}")
    max_side = 768
    h, w = image.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        image = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    if not ok:
        raise RuntimeError("encode failed")
    return encoded.tobytes(), ratios


def main():
    args = parse_args()
    random.seed(args.seed)
    out_root = args.output
    out_dir = out_root / "orange"
    if args.clear and out_root.exists():
        shutil.rmtree(out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(out_dir.glob("*.jpg")))
    items = collect_urls(args.search_results)
    print(f"candidates={len(items)} existing={existing} target={args.count}")
    metadata = []
    saved = existing
    semantic = load_semantic_filter() if args.semantic_filter else None
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_one, item): item for item in items}
        for future in concurrent.futures.as_completed(futures):
            if saved >= args.count:
                break
            item = futures[future]
            try:
                data, ratios = future.result()
            except Exception:
                continue
            semantic_detail = {}
            if semantic is not None:
                ok, semantic_detail = semantic_orange_accept(data, semantic, args)
                if not ok:
                    continue
            path = out_dir / f"orange_hard_{saved:04d}.jpg"
            path.write_bytes(data)
            metadata.append({**item, "file": str(path), "ratios": ratios, **semantic_detail})
            saved += 1
            if saved % 25 == 0:
                print(f"saved={saved}/{args.count}")

    (out_root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={saved}")
    print(f"output={out_root}")


if __name__ == "__main__":
    main()
