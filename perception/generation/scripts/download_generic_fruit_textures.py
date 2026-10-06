import argparse
import concurrent.futures
import json
import shutil
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import requests


API_URL = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "DataGenerationBlender/1.0 (local synthetic dataset preparation)"}

SEARCHES = {
    "apple": [
        '"red apple" fruit photo',
        '"red apples" grocery',
        '"red apple" supermarket',
        '"red apple" on table',
        '"red apple" hand',
        '"red apple" isolated',
        '"red apple" white background',
        '"apple fruit" red',
    ],
    "banana": [
        '"yellow banana" fruit photo',
        '"yellow bananas" grocery',
        '"banana bunch" yellow',
        '"banana" supermarket',
        '"banana" on table',
        '"yellow banana" isolated',
        '"yellow banana" white background',
    ],
    "orange": [
        '"orange fruit" photo',
        '"oranges" grocery',
        '"orange fruit" supermarket',
        '"orange fruit" on table',
        '"orange fruit" isolated',
        '"orange fruit" white background',
        '"mandarin orange" fruit',
    ],
    "pineapple": [
        '"pineapple fruit" whole',
        '"whole pineapple" photo',
        '"pineapple" supermarket',
        '"pineapple fruit" isolated',
        '"pineapple" white background',
        '"pineapple" on table',
    ],
}


def query(search, limit):
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrnamespace": 6,
        "gsrsearch": search,
        "gsrlimit": min(50, limit),
        "prop": "imageinfo",
        "iiprop": "url|mime|size",
        "iiurlwidth": 640,
    }
    response = requests.get(API_URL, params=params, headers=HEADERS, timeout=25)
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", {})
    items = []
    for page in pages.values():
        infos = page.get("imageinfo") or []
        if not infos:
            continue
        info = infos[0]
        if info.get("mime") not in {"image/jpeg", "image/png"}:
            continue
        if int(info.get("width", 0)) < 180 or int(info.get("height", 0)) < 180:
            continue
        url = info.get("thumburl") or info.get("url")
        if url:
            items.append({"url": url, "title": page.get("title", ""), "query": search})
    return items


def collect_candidates(per_class):
    candidates = {}
    for class_name, searches in SEARCHES.items():
        seen = set()
        items = []
        for search in searches:
            for item in query(search, per_class * 2):
                if item["url"] not in seen:
                    seen.add(item["url"])
                    items.append(item)
        candidates[class_name] = items
        print(f"{class_name}: candidates={len(items)}")
    return candidates


def ext_from_url(url):
    suffix = Path(urlparse(url).path).suffix.lower()
    return ".png" if suffix == ".png" else ".jpg"


def color_score(class_name, image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    valid = (s > 45) & (v > 55)

    if class_name == "apple":
        mask = (((h <= 12) | (h >= 168)) | ((h >= 14) & (h <= 28))) & valid
    elif class_name == "banana":
        mask = ((h >= 18) & (h <= 38)) & valid
    elif class_name == "orange":
        mask = ((h >= 5) & (h <= 25)) & (s > 55) & (v > 60)
    elif class_name == "pineapple":
        green = ((h >= 35) & (h <= 92)) & (s > 25) & (v > 35)
        yellow_brown = ((h >= 8) & (h <= 42)) & (s > 28) & (v > 30)
        mask = green | yellow_brown
    else:
        mask = valid

    return float(mask.mean())


def looks_usable(class_name, bytes_data):
    arr = np.frombuffer(bytes_data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return False, 0.0
    h, w = image.shape[:2]
    if min(h, w) < 180:
        return False, 0.0
    score = color_score(class_name, image)
    threshold = {
        "apple": 0.045,
        "banana": 0.040,
        "orange": 0.055,
        "pineapple": 0.030,
    }[class_name]
    return score >= threshold, score


def download_candidate(class_name, item):
    response = requests.get(item["url"], headers=HEADERS, timeout=35)
    response.raise_for_status()
    data = response.content
    if len(data) < 4096:
        raise RuntimeError("too small")
    ok, score = looks_usable(class_name, data)
    if not ok:
        raise RuntimeError(f"color filter failed score={score:.3f}")
    return data, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="datasets/fruit_textures/generic_faces")
    parser.add_argument("--per_class", type=int, default=150)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    out = Path(args.output)
    if args.clear and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    candidates = collect_candidates(args.per_class)
    metadata = {}
    total = 0

    for class_name, items in candidates.items():
        class_dir = out / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        saved = len([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".png"}])
        metadata[class_name] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(download_candidate, class_name, item): item for item in items}
            for future in concurrent.futures.as_completed(futures):
                if saved >= args.per_class:
                    break
                item = futures[future]
                try:
                    data, score = future.result()
                except Exception:
                    continue
                target = class_dir / f"{class_name}_{saved:04d}{ext_from_url(item['url'])}"
                target.write_bytes(data)
                metadata[class_name].append({**item, "file": str(target), "color_score": score})
                saved += 1
                if saved % 25 == 0:
                    print(f"{class_name}: {saved}/{args.per_class}")

        print(f"{class_name}: {saved}")
        total += saved

    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"total: {total}")
    print(f"output: {out}")


if __name__ == "__main__":
    main()
