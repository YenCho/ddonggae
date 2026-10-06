from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import html
import json
import random
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np
import requests


CLASSES = ("apple", "orange", "banana", "pineapple")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
}

QUERIES = {
    "apple": [
        '"whole red apple" "white background" fruit photo',
        '"red apple fruit" "isolated" "white background"',
        '"red delicious apple" "white background"',
        '"single red apple" "isolated on white"',
        '"whole apple" "red" "white background" fruit',
    ],
    "orange": [
        '"whole orange fruit" "white background" photo',
        '"orange fruit" "isolated" "white background"',
        '"single orange fruit" "isolated on white"',
        '"round orange fruit" "white background"',
        '"fresh whole orange" "white background" fruit',
    ],
    "banana": [
        '"yellow banana" "white background" photo',
        '"banana fruit" "isolated" "white background"',
        '"single banana" "isolated on white"',
        '"ripe yellow banana" "white background"',
        '"banana bunch" "yellow" "white background"',
    ],
    "pineapple": [
        '"whole pineapple" "white background" photo',
        '"pineapple fruit" "isolated" "white background"',
        '"brown pineapple" "white background" exterior',
        '"fresh whole pineapple" "white background"',
        '"pineapple with crown" "isolated on white"',
    ],
}

NEGATIVE_QUERY_SUFFIX = (
    " -slice -sliced -cut -half -peeled -juice -smoothie -cartoon -clipart -logo"
    " -whiskey -whisky -bottle -beer -wine -map -chart -person -people -shoe"
)

BAD_TOKENS_COMMON = (
    "cartoon",
    "clipart",
    "clip-art",
    "drawing",
    "illustration",
    "icon",
    "logo",
    "svg",
    "vector",
    "emoji",
    "coloring",
    "pattern",
    "wallpaper",
    "tree",
    "orchard",
    "flower",
    "juice",
    "smoothie",
    "salad",
    "cake",
    "pie",
    "dessert",
    "dish",
    "plate",
    "bowl",
    "people",
    "person",
    "hand",
    "woman",
    "man",
    "child",
    "phone",
    "iphone",
    "whiskey",
    "whisky",
    "bourbon",
    "jim beam",
    "bottle",
    "beer",
    "wine",
    "vodka",
    "liquor",
    "label",
    "brand",
    "advertisement",
    "poster",
    "map",
    "chart",
    "diagram",
    "graph",
    "profile",
    "portrait",
    "animal",
    "cat",
    "dog",
    "rabbit",
    "shoe",
    "bread",
    "bath",
    "olive",
    "pepperoncini",
    "pickled",
    "jar",
    "can",
    "toy",
    "bus",
    "tank",
    "airplane",
    "aircraft",
)

BAD_TOKENS_CUT = (
    "peeled",
    "peel-off",
    "skinless",
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
    "chunk",
    "chunks",
    "cross-section",
    "inside",
    "flesh",
)

CLASS_BAD_TOKENS = {
    "apple": ("green apple", "granny smith", "yellow apple"),
    "orange": ("orange slice", "blood orange slice", "mandarin segment", "peeled orange"),
    "banana": ("banana leaf", "banana tree", "green banana", "plantain"),
    "pineapple": ("pineapple slice", "cut pineapple", "peeled pineapple", "pineapple chunks", "pineapple rings"),
}


@dataclass
class Candidate:
    url: str
    query: str
    engine: str
    title: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an isolated other fruit texture set from search results with white-background cutouts."
    )
    parser.add_argument("--output", type=Path, default=Path("datasets/fruit_textures/other/web_wholefruit_cutouts_v1"))
    parser.add_argument("--target_per_class", type=int, default=160)
    parser.add_argument("--search_results", type=int, default=240)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--engines", nargs="+", default=["bing", "google"], choices=["bing", "google"])
    parser.add_argument("--min_white_border", type=float, default=0.38)
    parser.add_argument("--max_download_mb", type=float, default=8.0)
    parser.add_argument("--contact_sheet_cols", type=int, default=10)
    parser.add_argument("--semantic_filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min_semantic_prob", type=float, default=0.08)
    return parser.parse_args()


def normalize_url(raw: str) -> str:
    value = html.unescape(raw).replace("\\/", "/")
    value = unquote(value)
    if value.startswith("/url?"):
        query = parse_qs(urlparse(value).query)
        value = (query.get("url") or query.get("q") or [""])[0]
    return value


def is_probably_image_url(url: str) -> bool:
    lower = url.lower()
    if not lower.startswith(("http://", "https://")):
        return False
    if any(token in lower for token in BAD_TOKENS_COMMON):
        return False
    return any(ext in lower for ext in (".jpg", ".jpeg", ".png", ".webp")) or "images" in lower


def has_bad_tokens(class_name: str, text: str) -> bool:
    lower = text.lower()
    tokens = BAD_TOKENS_COMMON + BAD_TOKENS_CUT + CLASS_BAD_TOKENS.get(class_name, ())
    return any(token in lower for token in tokens)


def collect_bing(class_name: str, query: str, search_results: int) -> list[Candidate]:
    items: list[Candidate] = []
    seen = set()
    for first in range(0, search_results, 35):
        params = {"q": query + NEGATIVE_QUERY_SUFFIX, "first": str(first), "form": "HDRSC2", "safeSearch": "moderate"}
        response = requests.get("https://www.bing.com/images/search", params=params, headers=HEADERS, timeout=20)
        response.raise_for_status()
        text = response.text
        records = re.findall(r'<a class="iusc"[^>]+m="([^"]+)"', text)
        for record in records:
            try:
                decoded = html.unescape(record)
                data = json.loads(decoded)
            except Exception:
                data = {}
            url = normalize_url(str(data.get("murl") or ""))
            title = str(data.get("t") or "")
            if not is_probably_image_url(url) or url in seen:
                continue
            if has_bad_tokens(class_name, " ".join([url, title, query])):
                continue
            seen.add(url)
            items.append(Candidate(url=url, query=query, engine="bing", title=title))
        matches = re.findall(r'murl&quot;:&quot;(.*?)&quot;', text) + re.findall(r'"murl":"(.*?)"', text)
        for raw in matches:
            url = normalize_url(raw)
            if not is_probably_image_url(url) or url in seen:
                continue
            if has_bad_tokens(class_name, " ".join([url, query])):
                continue
            seen.add(url)
            items.append(Candidate(url=url, query=query, engine="bing"))
    return items


def collect_google(class_name: str, query: str, search_results: int) -> list[Candidate]:
    # Best-effort HTML parsing. This may return fewer URLs than Bing, but it is
    # useful when Google exposes original image URLs in page JSON.
    items: list[Candidate] = []
    seen = set()
    params = {"tbm": "isch", "q": query + NEGATIVE_QUERY_SUFFIX, "safe": "active", "ijn": "0"}
    response = requests.get("https://www.google.com/search", params=params, headers=HEADERS, timeout=25)
    response.raise_for_status()
    text = response.text
    patterns = [
        r'\["(https?://[^"]+?\.(?:jpg|jpeg|png|webp)[^"]*)"',
        r'"(https?://[^"]+?\.(?:jpg|jpeg|png|webp)[^"]*)"',
    ]
    for pattern in patterns:
        for raw in re.findall(pattern, text, flags=re.IGNORECASE):
            if len(items) >= search_results:
                break
            url = normalize_url(raw)
            if not is_probably_image_url(url) or url in seen:
                continue
            if has_bad_tokens(class_name, " ".join([url, query])):
                continue
            seen.add(url)
            items.append(Candidate(url=url, query=query, engine="google"))
    return items


def collect_candidates(class_name: str, engines: list[str], search_results: int) -> list[Candidate]:
    seen = set()
    candidates: list[Candidate] = []
    per_query = max(40, search_results)
    for query in QUERIES[class_name]:
        sources: list[Candidate] = []
        if "bing" in engines:
            try:
                sources.extend(collect_bing(class_name, query, per_query))
            except Exception as exc:
                print(f"{class_name}: bing failed for {query!r}: {exc}")
        if "google" in engines:
            try:
                sources.extend(collect_google(class_name, query, per_query))
            except Exception as exc:
                print(f"{class_name}: google failed for {query!r}: {exc}")
        for item in sources:
            if item.url in seen:
                continue
            seen.add(item.url)
            candidates.append(item)
    random.shuffle(candidates)
    return candidates


def download_bytes(url: str, max_download_mb: float) -> bytes:
    with requests.get(url, headers=HEADERS, stream=True, timeout=20) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if content_type and not any(token in content_type for token in ("image", "octet-stream")):
            raise RuntimeError(f"not image content-type={content_type}")
        limit = int(max_download_mb * 1024 * 1024)
        chunks = []
        size = 0
        for chunk in response.iter_content(8192):
            if not chunk:
                continue
            chunks.append(chunk)
            size += len(chunk)
            if size > limit:
                raise RuntimeError("download too large")
    data = b"".join(chunks)
    if len(data) < 4096:
        raise RuntimeError("too small")
    return data


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError("decode failed")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 3:
        return image
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        bgr = image[:, :, :3].astype(np.float32)
        white = np.full_like(bgr, 255, dtype=np.float32)
        return np.clip(bgr * alpha + white * (1.0 - alpha), 0, 255).astype(np.uint8)
    raise RuntimeError(f"unsupported channels: {image.shape}")


def white_background_mask(image: np.ndarray, min_white_border: float) -> tuple[np.ndarray, dict[str, float]]:
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    b, g, r = cv2.split(image)
    near_white = (v > 205) & (s < 55) & (np.abs(r.astype(np.int16) - g.astype(np.int16)) < 35) & (
        np.abs(g.astype(np.int16) - b.astype(np.int16)) < 35
    )

    border = np.zeros((h, w), np.uint8)
    bw = max(4, int(round(min(h, w) * 0.045)))
    border[:bw, :] = 1
    border[-bw:, :] = 1
    border[:, :bw] = 1
    border[:, -bw:] = 1
    border_white = float((near_white & (border > 0)).sum() / max(1, border.sum()))
    white_total = float(near_white.mean())
    if border_white < min_white_border:
        raise RuntimeError(f"not enough white border {border_white:.3f}")

    bg = np.zeros((h + 2, w + 2), np.uint8)
    seed_mask = (near_white & (border > 0)).astype(np.uint8)
    canvas = near_white.astype(np.uint8) * 255
    # Flood fill all near-white components connected to any border seed.
    for y, x in zip(*np.where(seed_mask > 0)):
        if bg[y + 1, x + 1] == 0:
            cv2.floodFill(canvas, bg, (int(x), int(y)), 128, flags=4)

    connected_bg = bg[1:-1, 1:-1] > 0
    foreground = ~connected_bg
    # Also reject very bright low-chroma speckles as background.
    foreground &= ~((v > 238) & (s < 32))
    foreground = foreground.astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel, iterations=1)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=2)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    if num <= 1:
        raise RuntimeError("empty foreground")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    largest_area = int(stats[largest, cv2.CC_STAT_AREA])
    total_area = h * w
    if largest_area < total_area * 0.035:
        raise RuntimeError(f"foreground too small {largest_area / total_area:.3f}")
    if largest_area > total_area * 0.88:
        raise RuntimeError(f"foreground too large {largest_area / total_area:.3f}")

    mask = (labels == largest).astype(np.uint8) * 255
    mask = cv2.GaussianBlur(mask, (0, 0), 0.75)
    info = {
        "border_white": border_white,
        "white_total": white_total,
        "foreground_area": float(largest_area / total_area),
    }
    return mask, info


def color_ratios(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    fg = mask > 32
    denom = max(1, int(fg.sum()))
    valid = fg & (s > 35) & (v > 35)
    return {
        "red": float((valid & ((h <= 10) | (h >= 170))).sum() / denom),
        "red_orange": float((valid & (((h <= 12) | (h >= 168)) | ((h >= 13) & (h <= 28)))).sum() / denom),
        "orange": float((valid & (h >= 4) & (h <= 25) & (s > 55) & (v > 55)).sum() / denom),
        "yellow": float((valid & (h >= 18) & (h <= 42) & (s > 45) & (v > 60)).sum() / denom),
        "green": float((valid & (h >= 35) & (h <= 95) & (s > 35) & (v > 35)).sum() / denom),
        "brown": float((valid & (h >= 5) & (h <= 30) & (s > 35) & (v > 25) & (v < 175)).sum() / denom),
        "bright_yellow": float((valid & (h >= 18) & (h <= 38) & (s > 65) & (v > 150)).sum() / denom),
        "valid": float(valid.sum() / denom),
    }


def passes_class_filter(class_name: str, ratios: dict[str, float], mask_info: dict[str, float]) -> tuple[bool, str]:
    area = mask_info["foreground_area"]
    if not 0.04 <= area <= 0.82:
        return False, "bad_area"
    if class_name == "apple":
        ok = ratios["red_orange"] >= 0.12 and ratios["red"] >= 0.035 and ratios["yellow"] < 0.42
    elif class_name == "orange":
        ok = ratios["orange"] >= 0.16 and ratios["red"] < 0.18 and ratios["green"] < 0.18
    elif class_name == "banana":
        ok = ratios["yellow"] >= 0.12 and ratios["orange"] < 0.38 and ratios["red"] < 0.12
    elif class_name == "pineapple":
        exterior = ratios["brown"] + ratios["green"]
        ok = exterior >= 0.14 and ratios["bright_yellow"] < 0.42 and ratios["red"] < 0.12
    else:
        ok = False
    return ok, "ok" if ok else "class_color_filter"


def square_rgba(image: np.ndarray, mask: np.ndarray, size: int, pad_ratio: float = 0.08) -> np.ndarray:
    ys, xs = np.where(mask > 32)
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("empty mask")
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    pad = int(round(max(x2 - x1, y2 - y1) * pad_ratio))
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(image.shape[1], x2 + pad), min(image.shape[0], y2 + pad)
    crop = image[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]
    scale = (size * 0.92) / max(crop.shape[:2])
    new_w = max(1, int(round(crop.shape[1] * scale)))
    new_h = max(1, int(round(crop.shape[0] * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(crop, (new_w, new_h), interpolation=interp)
    resized_mask = cv2.resize(crop_mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w, :3] = resized
    canvas[y0 : y0 + new_h, x0 : x0 + new_w, 3] = resized_mask
    return canvas


def rgba_to_white_bgr(rgba: np.ndarray) -> np.ndarray:
    bgr = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    white = np.full_like(bgr, 255.0)
    return np.clip(bgr * alpha + white * (1.0 - alpha), 0, 255).astype(np.uint8)


def sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def process_candidate(class_name: str, item: Candidate, args: argparse.Namespace) -> tuple[bool, dict, bytes | None, np.ndarray | None, np.ndarray | None]:
    data = download_bytes(item.url, args.max_download_mb)
    image = decode_image(data)
    h, w = image.shape[:2]
    if min(h, w) < 150:
        raise RuntimeError("image too small")
    if max(h, w) / max(1, min(h, w)) > 3.6:
        raise RuntimeError("extreme aspect")
    mask, mask_info = white_background_mask(image, args.min_white_border)
    ratios = color_ratios(image, mask)
    ok, reason = passes_class_filter(class_name, ratios, mask_info)
    if not ok:
        raise RuntimeError(reason)
    rgba = square_rgba(image, mask, args.size)
    white = rgba_to_white_bgr(rgba)
    meta = {
        "class": class_name,
        "url": item.url,
        "query": item.query,
        "engine": item.engine,
        "title": item.title,
        "sha1": sha1(data),
        "source_width": int(w),
        "source_height": int(h),
        "mask": mask_info,
        "color": ratios,
    }
    return True, meta, data, rgba, white


def safe_suffix(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        return ".jpg"
    return ".jpg" if suffix == ".jpeg" else suffix


def make_contact_sheet(samples: list[tuple[str, Path]], output: Path, cols: int, tile: int = 140) -> None:
    if not samples:
        return
    rows = int(np.ceil(len(samples) / cols))
    label_h = 24
    sheet = np.full((rows * (tile + label_h), cols * tile, 3), 245, dtype=np.uint8)
    for idx, (label, path) in enumerate(samples):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        if image.shape[2] == 4:
            image = rgba_to_white_bgr(image)
        resized = cv2.resize(image[:, :, :3], (tile, tile), interpolation=cv2.INTER_AREA)
        r, c = divmod(idx, cols)
        x, y = c * tile, r * (tile + label_h)
        sheet[y + label_h : y + label_h + tile, x : x + tile] = resized
        cv2.putText(sheet, label[:22], (x + 3, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


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


SEMANTIC_LABELS = {
    # ImageNet has no red-delicious class, so accept the apple-like labels only
    # as a semantic guard against bottles/maps/people rather than as final truth.
    "apple": {"Granny Smith", "custard apple"},
    "orange": {"orange", "lemon"},
    "banana": {"banana"},
    "pineapple": {"pineapple"},
}


def semantic_accept(class_name: str, bgr_image: np.ndarray, semantic, min_prob: float) -> tuple[bool, dict]:
    image_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    pil = semantic["Image"].fromarray(image_rgb)
    tensor = semantic["preprocess"](pil).unsqueeze(0)
    with semantic["torch"].no_grad():
        probs = semantic["torch"].softmax(semantic["model"](tensor), dim=1)[0].detach().cpu().numpy()
    classes = semantic["classes"]
    wanted = SEMANTIC_LABELS[class_name]
    wanted_prob = 0.0
    for idx, label in enumerate(classes):
        if label in wanted:
            wanted_prob += float(probs[idx])
    top_idx = np.argsort(-probs)[:8]
    top = [{"class": classes[int(i)], "prob": float(probs[int(i)])} for i in top_idx]
    top_names = {item["class"] for item in top[:3]}
    ok = wanted_prob >= min_prob or bool(wanted & top_names)
    # A red apple often gets weak ImageNet confidence; allow it only when the
    # top predictions are fruit-like rather than obvious non-fruit objects.
    if class_name == "apple" and not ok:
        fruitish = {"pomegranate", "fig", "strawberry", "orange", "pineapple", "banana"}
        ok = bool(fruitish & top_names) and wanted_prob >= min_prob * 0.25
    detail = {
        "semantic_prob": wanted_prob,
        "semantic_top8": top,
    }
    return ok, detail


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    output = args.output
    if args.clear and output.exists():
        shutil.rmtree(output)
    for sub in ("raw", "cutouts_512", "white_512", "rejected", "previews"):
        (output / sub).mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    rejections: list[dict] = []
    preview_samples: list[tuple[str, Path]] = []
    semantic = load_semantic_filter() if args.semantic_filter else None

    for class_name in CLASSES:
        for sub in ("raw", "cutouts_512", "white_512", "rejected"):
            (output / sub / class_name).mkdir(parents=True, exist_ok=True)

        existing = len(list((output / "cutouts_512" / class_name).glob("*.png")))
        target = args.target_per_class
        if existing >= target:
            print(f"{class_name}: already has {existing}/{target}")
            continue

        candidates = collect_candidates(class_name, args.engines, args.search_results)
        print(f"{class_name}: candidates={len(candidates)} existing={existing} target={target}")
        saved = existing
        seen_hashes = {p.stem.split("_")[-1] for p in (output / "raw" / class_name).glob("*")}
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_candidate, class_name, item, args): item for item in candidates}
            for future in concurrent.futures.as_completed(futures):
                item = futures[future]
                if saved >= target:
                    break
                try:
                    _ok, meta, raw_data, rgba, white = future.result()
                except Exception as exc:
                    rejections.append({
                        "class": class_name,
                        "url": item.url,
                        "query": item.query,
                        "engine": item.engine,
                        "reason": str(exc)[:160],
                    })
                    continue
                assert raw_data is not None and rgba is not None and white is not None
                if semantic is not None:
                    ok, semantic_detail = semantic_accept(class_name, white, semantic, args.min_semantic_prob)
                    meta.update(semantic_detail)
                    if not ok:
                        rejections.append({
                            "class": class_name,
                            "url": item.url,
                            "query": item.query,
                            "engine": item.engine,
                            "reason": "semantic_filter",
                            "semantic": semantic_detail,
                        })
                        continue
                digest = meta["sha1"][:12]
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)
                stem = f"{class_name}_{saved:04d}_web_{digest}"
                raw_path = output / "raw" / class_name / f"{stem}{safe_suffix(item.url)}"
                cutout_path = output / "cutouts_512" / class_name / f"{stem}.png"
                white_path = output / "white_512" / class_name / f"{stem}.jpg"
                raw_path.write_bytes(raw_data)
                cv2.imwrite(str(cutout_path), rgba, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
                cv2.imwrite(str(white_path), white, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
                meta.update({
                    "raw_path": str(raw_path),
                    "cutout_path": str(cutout_path),
                    "white_path": str(white_path),
                })
                rows.append(meta)
                if len(preview_samples) < 120:
                    preview_samples.append((f"{class_name} {saved:03d}", cutout_path))
                saved += 1
                if saved % 20 == 0:
                    print(f"{class_name}: saved={saved}/{target}")
        print(f"{class_name}: final={saved}/{target}")

    manifest = output / "manifest.csv"
    fields = [
        "class",
        "engine",
        "query",
        "url",
        "sha1",
        "source_width",
        "source_height",
        "raw_path",
        "cutout_path",
        "white_path",
        "mask",
        "color",
    ]
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["mask"] = json.dumps(row.get("mask", {}), ensure_ascii=False)
            out["color"] = json.dumps(row.get("color", {}), ensure_ascii=False)
            writer.writerow(out)
    (output / "rejections.json").write_text(json.dumps(rejections, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = {
        class_name: len(list((output / "cutouts_512" / class_name).glob("*.png")))
        for class_name in CLASSES
    }
    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output": str(output),
        "policy": {
            "apple": "red, unpeeled, whole fruit",
            "orange": "orange-colored, unpeeled, whole fruit",
            "banana": "yellow, unpeeled, whole banana",
            "pineapple": "whole brown/green exterior, no visible yellow cut flesh",
            "background": "near-white border required; alpha cutout by border-connected white flood fill",
        },
        "counts": counts,
        "engines": args.engines,
        "target_per_class": args.target_per_class,
        "search_results": args.search_results,
        "semantic_filter": args.semantic_filter,
        "min_semantic_prob": args.min_semantic_prob,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    make_contact_sheet(preview_samples, output / "previews" / "other_web_cutouts_contact_sheet.jpg", args.contact_sheet_cols)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
