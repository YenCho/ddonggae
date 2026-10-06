import argparse
import json
from pathlib import Path


FRUIT_CLASSES = {"banana", "orange", "pineapple", "apple"}


def iter_meta_files(dataset):
    root = Path(dataset)
    candidates = [
        root / "_meta" / "train",
        root / "_generated_all" / "_meta" / "train",
        root,
    ]
    for candidate in candidates:
        if candidate.exists():
            yield from sorted(candidate.glob("*.json"))
            return


def face_texture_issues(fruit, expect_single_texture_per_cube):
    issues = []
    fruit_class = fruit.get("class")
    textures = fruit.get("face_textures") or []
    if not textures:
        issues.append("missing face_textures metadata")
        return issues
    texture_paths = {texture.get("texture") for texture in textures if texture.get("texture")}
    if expect_single_texture_per_cube and len(texture_paths) > 1:
        issues.append(f"multiple source textures on one fruit cube: {sorted(texture_paths)}")
    for texture in textures:
        face_class = texture.get("class")
        declared = texture.get("texture_declared_class")
        if face_class and face_class != fruit_class:
            issues.append(f"face class {face_class} != fruit class {fruit_class}")
        if declared and declared != fruit_class:
            issues.append(f"texture declared class {declared} != fruit class {fruit_class}")
    return issues


def audit_file(path, expect_single_fruit_class, expect_single_texture_per_cube, require_face_texture_meta):
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    issues = []
    fruit_objects = [item for item in meta.get("fruit_objects", []) if item.get("class") in FRUIT_CLASSES]
    classes = sorted({item.get("class") for item in fruit_objects})
    if expect_single_fruit_class and len(classes) > 1:
        issues.append(f"multiple fruit classes in image: {classes}")

    if require_face_texture_meta:
        for fruit in fruit_objects:
            issues.extend(
                f"{fruit.get('object_name', '<unknown>')}: {issue}"
                for issue in face_texture_issues(fruit, expect_single_texture_per_cube)
            )

    return {
        "path": path,
        "fruit_objects": len(fruit_objects),
        "fruit_classes": classes,
        "issues": issues,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit generated fruit-cube metadata for mixed fruit classes.")
    parser.add_argument("--dataset", required=True, help="Dataset root or _meta/train directory.")
    parser.add_argument("--expect_single_fruit_class_per_image", action="store_true", default=False)
    parser.add_argument(
        "--allow_mixed_fruit_classes_per_image",
        dest="expect_single_fruit_class_per_image",
        action="store_false",
    )
    parser.add_argument("--expect_single_texture_per_cube", action="store_true", default=False)
    parser.add_argument(
        "--allow_multiple_textures_per_cube",
        dest="expect_single_texture_per_cube",
        action="store_false",
    )
    parser.add_argument("--require_face_texture_meta", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    meta_files = list(iter_meta_files(args.dataset))
    if args.limit > 0:
        meta_files = meta_files[: args.limit]
    if not meta_files:
        raise RuntimeError(f"No metadata JSON files found under {args.dataset}")

    audited = [
        audit_file(
            path,
            args.expect_single_fruit_class_per_image,
            args.expect_single_texture_per_cube,
            args.require_face_texture_meta,
        )
        for path in meta_files
    ]
    failures = [item for item in audited if item["issues"]]
    fruit_images = sum(1 for item in audited if item["fruit_objects"] > 0)
    fruit_class_sets = {}
    for item in audited:
        key = ",".join(item["fruit_classes"]) if item["fruit_classes"] else "<none>"
        fruit_class_sets[key] = fruit_class_sets.get(key, 0) + 1

    print(f"audited: {len(audited)} metadata files")
    print(f"images with fruit objects: {fruit_images}")
    print(f"fruit class sets: {fruit_class_sets}")
    print(f"failures: {len(failures)}")
    for item in failures[:20]:
        print(f"- {item['path']}: {'; '.join(item['issues'])}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
