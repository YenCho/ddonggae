import argparse
import concurrent.futures
import json
import re
import random
import shutil
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import requests


QUERIES = {
    "apple": [
        "red apple fruit supermarket photo",
        "red apples grocery store photo",
        "red apple on table real photo",
        "red apple in hand photo",
        "red apples market photo",
    ],
    "banana": [
        "yellow banana supermarket photo",
        "yellow bananas grocery store photo",
        "banana bunch real photo",
        "yellow banana on table photo",
        "banana in hand photo",
    ],
    "orange": [
        "orange fruit supermarket photo",
        "oranges grocery store photo",
        "orange fruit on table real photo",
        "orange fruit in hand photo",
        "mandarin orange fruit photo",
    ],
    "pineapple": [
        "cut pineapple yellow fruit photo",
        "pineapple slices yellow photo",
        "peeled pineapple yellow fruit photo",
        "pineapple chunks yellow photo",
        "cut pineapple supermarket photo",
    ],
}

BAD_TITLE_WORDS = {
    "apple": ["green apple", "tree", "orchard", "logo", "iphone", "macbook"],
    "banana": ["tree", "plant", "leaf", "logo", "cartoon"],
    "orange": ["tree", "flower", "color palette", "logo", "cartoon"],
    "pineapple": ["whole pineapple", "tree", "plant", "logo", "cartoon"],
}

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}


def collect_urls(class_name, max_results):
    urls = []
    seen = set()
    for query in QUERIES[class_name]:
        for first in range(0, max_results, 35):
            params = {"q": query, "first": str(first), "form": "HDRSC2", "safeSearch": "moderate"}
            response = requests.get("https://www.bing.com/images/search", params=params, headers=HEADERS, timeout=20)
            response.raise_for_status()
            html = response.text
            matches = re.findall(r'murl&quot;:&quot;(.*?)&quot;', html)
            matches += re.findall(r'"murl":"(.*?)"', html)
            for url in matches:
                url = url.replace("\\/", "/")
                title = query.lower()
                if any(word in title for word in BAD_TITLE_WORDS.get(class_name, [])):
                    continue
                if url.startswith("http") and url not in seen:
                    seen.add(url)
                    urls.append({"url": url, "title": "", "query": query})
                    if len(urls) >= max_results * len(QUERIES[class_name]):
                        break
    random.shuffle(urls)
    return urls


def color_score(class_name, image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    valid = (s > 45) & (v > 55)
    if class_name == "apple":
        mask = (((h <= 12) | (h >= 168)) | ((h >= 13) & (h <= 28))) & valid
    elif class_name == "banana":
        mask = ((h >= 18) & (h <= 38)) & valid
    elif class_name == "orange":
        mask = ((h >= 4) & (h <= 25)) & (s > 55) & (v > 60)
    elif class_name == "pineapple":
        mask = ((h >= 18) & (h <= 42)) & (s > 30) & (v > 65)
    else:
        mask = valid
    return float(mask.mean())


def ext_from_url(url):
    suffix = Path(urlparse(url).path).suffix.lower()
    return ".png" if suffix == ".png" else ".jpg"


def download_and_filter(class_name, item):
    response = requests.get(item["url"], headers=HEADERS, timeout=15)
    response.raise_for_status()
    data = response.content
    if len(data) < 4096:
        raise RuntimeError("too small")
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("not image")
    h, w = image.shape[:2]
    if min(h, w) < 160 or max(h, w) / max(1, min(h, w)) > 4.0:
        raise RuntimeError("bad shape")
    score = color_score(class_name, image)
    threshold = {"apple": 0.040, "banana": 0.035, "orange": 0.050, "pineapple": 0.030}[class_name]
    if score < threshold:
        raise RuntimeError(f"bad color {score:.3f}")
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("encode failed")
    return encoded.tobytes(), score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="datasets/fruit_textures/web_faces")
    parser.add_argument("--per_class", type=int, default=150)
    parser.add_argument("--search_results", type=int, default=120)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    out = Path(args.output)
    if args.clear and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    metadata = {}

    for class_name in QUERIES:
        class_dir = out / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        urls = collect_urls(class_name, args.search_results)
        print(f"{class_name}: candidates={len(urls)}")
        saved = 0
        metadata[class_name] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(download_and_filter, class_name, item): item for item in urls}
            for future in concurrent.futures.as_completed(futures):
                if saved >= args.per_class:
                    break
                item = futures[future]
                try:
                    data, score = future.result()
                except Exception:
                    continue
                target = class_dir / f"{class_name}_{saved:04d}.jpg"
                target.write_bytes(data)
                metadata[class_name].append({**item, "file": str(target), "color_score": score})
                saved += 1
                if saved % 25 == 0:
                    print(f"{class_name}: {saved}/{args.per_class}")
        print(f"{class_name}: saved={saved}")

    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    total = sum(len(list((out / c).glob("*.jpg"))) for c in QUERIES)
    print(f"total={total}")
    print(f"output={out}")


if __name__ == "__main__":
    main()
