import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        print(f"already downloaded: {target}")
        return
    print(f"downloading: {url}")
    urllib.request.urlretrieve(url, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="datasets/backgrounds/coco128")
    parser.add_argument("--cache", type=str, default="datasets/_downloads/coco128.zip")
    args = parser.parse_args()

    output = Path(args.output)
    cache = Path(args.cache)
    download(COCO128_URL, cache)

    extract_root = output.parent
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(cache, "r") as zf:
        zf.extractall(extract_root)

    extracted = extract_root / "coco128"
    if extracted.resolve() != output.resolve():
        if output.exists():
            shutil.rmtree(output)
        shutil.move(str(extracted), str(output))

    image_dir = output / "images" / "train2017"
    images = sorted(image_dir.glob("*.jpg"))
    print(f"background images: {len(images)}")
    print(f"background dir: {image_dir}")


if __name__ == "__main__":
    main()
