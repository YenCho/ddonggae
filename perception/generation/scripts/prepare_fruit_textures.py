import argparse
import shutil
from pathlib import Path


CLASS_PATTERNS = {
    "apple": ["apple"],
    "banana": ["banana"],
    "orange": ["orange"],
    "pineapple": ["pineapple"],
}

EXCLUDE = {
    "orange": ["pepper", "tomato"],
}


def matches_class(folder_name, class_name):
    name = folder_name.lower()
    if any(word not in name for word in CLASS_PATTERNS[class_name]):
        return False
    if any(word in name for word in EXCLUDE.get(class_name, [])):
        return False
    return True


def collect_images(source, class_name):
    images = []
    for split in ["Training", "Test"]:
        split_dir = source / split
        if not split_dir.exists():
            continue
        for folder in split_dir.iterdir():
            if folder.is_dir() and matches_class(folder.name, class_name):
                images.extend(sorted(folder.glob("*.jpg")))
                images.extend(sorted(folder.glob("*.png")))
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="datasets/fruit_textures/fruits360_repo")
    parser.add_argument("--output", type=str, default="datasets/fruit_textures/fruit_faces")
    parser.add_argument("--max_per_class", type=int, default=800)
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    total = 0
    for class_name in CLASS_PATTERNS:
        class_out = output / class_name
        if class_out.exists():
            shutil.rmtree(class_out)
        class_out.mkdir(parents=True, exist_ok=True)

        images = collect_images(source, class_name)[:args.max_per_class]
        for idx, image in enumerate(images):
            target = class_out / f"{class_name}_{idx:04d}{image.suffix.lower()}"
            shutil.copy2(image, target)
        total += len(images)
        print(f"{class_name}: {len(images)}")

    print(f"total: {total}")
    print(f"output: {output}")


if __name__ == "__main__":
    main()
