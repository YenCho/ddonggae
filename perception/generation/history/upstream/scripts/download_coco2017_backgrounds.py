import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


URLS = {
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val2017", "train2017"], default="train2017")
    parser.add_argument("--output", type=str, default="datasets/backgrounds/coco2017")
    parser.add_argument("--cache", type=str, default="datasets/_downloads")
    args = parser.parse_args()

    output = Path(args.output)
    image_dir = output / args.split
    if image_dir.exists() and len(list(image_dir.glob("*.jpg"))) > 1000:
        print(f"already extracted: {image_dir}")
        print(f"images: {len(list(image_dir.glob('*.jpg')))}")
        return

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    zip_path = cache / f"{args.split}.zip"
    cached_zip_ok = zip_path.exists() and zip_path.stat().st_size >= 1024 * 1024
    if cached_zip_ok and not zipfile.is_zipfile(zip_path):
        print(f"cached zip is incomplete or invalid, deleting: {zip_path}")
        zip_path.unlink()
        cached_zip_ok = False

    if not cached_zip_ok:
        print(f"downloading {URLS[args.split]} -> {zip_path}")
        urlretrieve(URLS[args.split], zip_path)
        if not zipfile.is_zipfile(zip_path):
            raise RuntimeError(f"downloaded file is not a valid zip: {zip_path}")
    else:
        print(f"using cached zip: {zip_path}")

    output.mkdir(parents=True, exist_ok=True)
    print(f"extracting to {output}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output)

    print(f"images: {len(list(image_dir.glob('*.jpg')))}")
    print(f"background dir: {image_dir}")


if __name__ == "__main__":
    main()
