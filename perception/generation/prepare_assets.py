"""Fetch a fixed Fruits-360 snapshot, checking every image's Git blob hash.

Uses only public GitHub endpoints. No Kaggle login, scraper or original local pack.
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO = "Horea94/Fruit-Images-Dataset"
COMMIT = "ffda2d14eada57a0c5537700190b309cfea2e120"
API = f"https://api.github.com/repos/{REPO}"
CLASSES = ("apple", "orange", "banana", "pineapple")


def matches_class(folder_name, label):
    return re.search(rf"\b{re.escape(label)}\b", folder_name.lower()) is not None and not any(
        word in folder_name.lower() for word in ("pepper", "tomato")
    )


def fetch(url):
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers={"User-Agent": "ddonggae-reproduction/1"}), timeout=60) as response:
                return response.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def tree(sha):
    return json.loads(fetch(f"{API}/git/trees/{sha}"))["tree"]


def training_tree(sha, prefix="", depth=0):
    entries = tree(sha)
    for item in entries:
        if item["type"] == "tree" and item["path"].lower() == "training":
            return item["sha"], prefix + item["path"]
    if depth < 4:
        # Some revisions wrap the 100x100 dataset in a top-level directory.
        for item in entries:
            if item["type"] == "tree" and any(s in item["path"].lower() for s in ("fruit", "100", "dataset")):
                result = training_tree(item["sha"], prefix + item["path"] + "/", depth + 1)
                if result:
                    return result
    return None


def blob_hash(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=256)
    args = parser.parse_args()
    if args.per_class < 1:
        parser.error("--per-class must be positive")
    manifest_path = args.out / "assets-manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved["commit"] != COMMIT or saved["per_class"] != args.per_class:
            raise RuntimeError("Existing asset pack has another recipe; choose a new output directory")
        for item in saved["files"]:
            label = Path(item["local"]).parts[0]
            source_folder = item["source"].split("/")[-2]
            if label not in CLASSES or not matches_class(source_folder, label):
                raise RuntimeError(
                    f"Cached asset has an incorrect fruit class: {item['local']} <- {item['source']}. "
                    "Rebuild the texture pack in a new output directory."
                )
            path = args.out / item["local"]
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(fetch(item["url"]))
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise RuntimeError(f"Asset checksum mismatch: {path}")
        print(f"Verified cached assets: {manifest_path}")
        return
    result = training_tree(COMMIT)
    if result is None:
        raise RuntimeError("Pinned snapshot contains no Training directory")
    sha, prefix = result
    folders = sorted(tree(sha), key=lambda item: item["path"])
    records = []
    for label in CLASSES:
        candidates = []
        for folder in folders:
            if folder["type"] != "tree" or not matches_class(folder["path"], label):
                continue
            if any(word in folder["path"].lower() for word in ("pepper", "tomato")):
                continue
            entries = sorted(tree(folder["sha"]), key=lambda item: item["path"])
            entries = [item for item in entries if item["type"] == "blob" and Path(item["path"]).suffix.lower() in {".jpg", ".png"}]
            candidates.append((folder["path"], entries))
        # Round-robin varieties instead of taking only the first apple variety.
        selected = []
        index = 0
        while len(selected) < args.per_class:
            added = False
            for folder, entries in candidates:
                if index < len(entries) and len(selected) < args.per_class:
                    selected.append((folder, entries[index]))
                    added = True
            if not added:
                raise RuntimeError(f"Not enough textures for {label}: {len(selected)}")
            index += 1
        for index, (folder, entry) in enumerate(selected):
            upstream = f"{prefix}/{folder}/{entry['path']}"
            url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{quote(upstream, safe='/')}"
            local = f"{label}/{label}_{index:04d}{Path(entry['path']).suffix.lower()}"
            path = args.out / local
            path.parent.mkdir(parents=True, exist_ok=True)
            data = path.read_bytes() if path.exists() else fetch(url)
            if blob_hash(data) != entry["sha"]:
                raise RuntimeError(f"Git blob checksum mismatch: {upstream}")
            path.write_bytes(data)
            records.append({"local": local, "source": upstream, "url": url, "git_blob": entry["sha"], "sha256": hashlib.sha256(data).hexdigest()})
        print(f"{label}: {len(selected)} checked textures", flush=True)
    manifest_path.write_text(json.dumps({"repository": REPO, "commit": COMMIT,
        "per_class": args.per_class, "license": "CC BY-SA 4.0; retain upstream attribution",
        "files": records}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
