import argparse
import csv
import hashlib
import json
import shutil
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np


DATASET_ID = "vkht8pfsp3"
VERSION = 3
API_ROOT = "https://data.mendeley.com/public-api"
HEADERS = {
    "Accept": "application/vnd.mendeley-public-dataset.1+json",
    "User-Agent": "Mozilla/5.0",
}
CLASS_MAP = {
    "apple": ["Apple_Gala"],
    "banana": ["Banana"],
    "orange": ["Orange"],
    "pineapple": ["Pineapple"],
}
IMAGE_EXTS = {".jpg", ".jpeg"}
MASK_EXTS = {".png"}


def request_json(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def download_file(url, target, expected_sha256=None, retries=4):
    target = Path(target)
    if target.exists() and (expected_sha256 is None or sha256_file(target) == expected_sha256):
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=90) as response:
                target.write_bytes(response.read())
            if expected_sha256 and sha256_file(target) != expected_sha256:
                raise RuntimeError(f"sha256 mismatch for {target}")
            return True
        except Exception as exc:
            last_error = exc
            if target.exists():
                target.unlink()
            time.sleep(min(2.0 * attempt, 8.0))
    raise RuntimeError(f"failed to download {url}: {last_error}")


def sha256_file(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def index_folders(folders):
    by_parent = {}
    by_id = {}
    for folder in folders:
        by_parent.setdefault(folder.get("parent_id"), []).append(folder)
        by_id[folder["id"]] = folder
    return by_parent, by_id


def child_by_name(by_parent, parent_id, name):
    for folder in by_parent.get(parent_id, []):
        if folder["name"].lower() == name.lower():
            return folder
    raise RuntimeError(f"missing {name} child folder under {parent_id}")


def selected_source_folders(folders):
    by_parent, _ = index_folders(folders)
    root_ids = {
        folder["id"]
        for folder in folders
        if folder["name"] in {name for names in CLASS_MAP.values() for name in names}
    }
    selected = []
    for class_name, source_names in CLASS_MAP.items():
        for source_name in source_names:
            source_folder = next((folder for folder in folders if folder["name"] == source_name and folder["id"] in root_ids), None)
            if source_folder is None:
                raise RuntimeError(f"missing FruitSeg30 source folder: {source_name}")
            selected.append({
                "class": class_name,
                "source_name": source_name,
                "source_id": source_folder["id"],
                "image_id": child_by_name(by_parent, source_folder["id"], "Images")["id"],
                "mask_id": child_by_name(by_parent, source_folder["id"], "Mask")["id"],
            })
    return selected


def stem_key(path):
    stem = Path(path).stem
    if stem.endswith("_mask"):
        return stem[:-5]
    return stem


def files_by_folder(files):
    grouped = {}
    for item in files:
        grouped.setdefault(item.get("folder_id"), []).append(item)
    return grouped


def download_pairs(raw_root, grouped_files, source):
    image_items = {
        stem_key(item["filename"]): item
        for item in grouped_files.get(source["image_id"], [])
        if Path(item["filename"]).suffix.lower() in IMAGE_EXTS
    }
    mask_items = {
        stem_key(item["filename"]): item
        for item in grouped_files.get(source["mask_id"], [])
        if Path(item["filename"]).suffix.lower() in MASK_EXTS
    }
    keys = sorted(set(image_items) & set(mask_items), key=lambda value: int(value) if value.isdigit() else value)
    pairs = []
    for key in keys:
        image_item = image_items[key]
        mask_item = mask_items[key]
        source_dir = Path(raw_root) / source["source_name"]
        image_path = source_dir / "Images" / image_item["filename"]
        mask_path = source_dir / "Mask" / mask_item["filename"]
        download_file(
            image_item["content_details"]["download_url"],
            image_path,
            image_item["content_details"].get("sha256_hash"),
        )
        download_file(
            mask_item["content_details"]["download_url"],
            mask_path,
            mask_item["content_details"].get("sha256_hash"),
        )
        pairs.append((key, image_path, mask_path))
    return pairs


def rgba_cutout(image, mask, size, padding_ratio):
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = (mask > 0).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    pad = int(round(max(x2 - x1, y2 - y1) * padding_ratio))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(image.shape[1], x2 + pad)
    y2 = min(image.shape[0], y2 + pad)

    crop = image[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]
    scale = (size * 0.92) / max(crop.shape[:2])
    new_w = max(1, int(round(crop.shape[1] * scale)))
    new_h = max(1, int(round(crop.shape[0] * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
    resized_mask = cv2.resize(crop_mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.GaussianBlur(resized_mask, (0, 0), 0.75)

    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w, :3] = resized
    canvas[y0:y0 + new_h, x0:x0 + new_w, 3] = resized_mask
    return canvas


def write_cutouts(raw_root, output_root, grouped_files, selected_sources, size, padding_ratio, max_per_source):
    rows = []
    counts = {class_name: 0 for class_name in CLASS_MAP}
    for source in selected_sources:
        pairs = download_pairs(raw_root, grouped_files, source)
        if max_per_source > 0:
            pairs = pairs[:max_per_source]
        class_dir = Path(output_root) / source["class"]
        class_dir.mkdir(parents=True, exist_ok=True)
        for key, image_path, mask_path in pairs:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                continue
            cutout = rgba_cutout(image, mask, size, padding_ratio)
            if cutout is None:
                continue
            idx = counts[source["class"]]
            target = class_dir / f"{source['class']}_{idx:04d}_fruitseg30_{source['source_name'].replace(' ', '_')}_{key}.png"
            cv2.imwrite(str(target), cutout, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
            rows.append({
                "class": source["class"],
                "output": str(target),
                "source_dataset": "FruitSeg30",
                "source_class": source["source_name"],
                "source_image": str(image_path),
                "source_mask": str(mask_path),
            })
            counts[source["class"]] += 1
        print(f"{source['class']} <- {source['source_name']}: {len(pairs)} pairs")
    return rows, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, default=DATASET_ID)
    parser.add_argument("--version", type=int, default=VERSION)
    parser.add_argument("--raw", type=str, default="datasets/fruit_textures/_raw/fruitseg30")
    parser.add_argument("--output", type=str, default="datasets/fruit_textures/_sources/fruitseg30_cutouts_512")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--padding_ratio", type=float, default=0.06)
    parser.add_argument("--max_per_source", type=int, default=0, help="0 means all matching files.")
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    raw_root = Path(args.raw)
    output_root = Path(args.output)
    if args.clear and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    metadata = request_json(f"{API_ROOT}/datasets/{args.dataset_id}")
    folders = request_json(f"{API_ROOT}/datasets/{args.dataset_id}/folders/{args.version}")
    selected_sources = selected_source_folders(folders)
    grouped_files = files_by_folder(metadata["files"])
    rows, counts = write_cutouts(
        raw_root,
        output_root,
        grouped_files,
        selected_sources,
        args.size,
        args.padding_ratio,
        args.max_per_source,
    )

    manifest = output_root / "_manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["class", "output", "source_dataset", "source_class", "source_image", "source_mask"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metadata_out = output_root / "_source_dataset.json"
    metadata_out.write_text(
        json.dumps({
            "dataset": "FruitSeg30_Segmentation Dataset & Mask Annotations",
            "dataset_id": args.dataset_id,
            "version": args.version,
            "doi": metadata.get("doi", {}).get("id"),
            "licence": metadata.get("data_licence") or metadata.get("licence"),
            "class_map": CLASS_MAP,
            "counts": counts,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"counts: {counts}")
    print(f"output: {output_root}")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
