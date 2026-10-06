from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

try:
    import ujson as fast_json
except Exception:  # pragma: no cover - optional local speedup
    fast_json = json


CLASS_NAMES = ["apple", "orange", "banana", "pineapple", "plain"]


def norm_path(value: str | Path) -> str:
    return str(value).replace("\\", "/").strip().lower()


def as_posix(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def label_from_stem(stem: str) -> str:
    parts = stem.split("_")
    return parts[2] if len(parts) >= 4 else "unknown"


def label_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx, part in enumerate(parts):
        if part == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    raise ValueError(f"image path does not contain an images directory: {image_path}")


def safe_reset(path: Path, workspace: Path) -> None:
    resolved = path.resolve()
    workspace = workspace.resolve()
    if workspace not in resolved.parents and resolved != workspace:
        raise RuntimeError(f"refusing to reset path outside workspace: {resolved}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy_fallback"


def write_data_yaml(root: Path) -> None:
    lines = [
        f"path: {as_posix(root)}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for idx, name in enumerate(CLASS_NAMES):
        lines.append(f"  {idx}: {name}")
    (root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_bad_textures(records_csv: Path) -> tuple[set[str], set[str], Counter]:
    bad_textures: set[str] = set()
    bad_names: set[str] = set()
    by_class: Counter = Counter()
    with records_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("keep", "")).strip() != "0":
                continue
            source = row["source"]
            bad_textures.add(norm_path(source))
            bad_names.add(Path(source).name)
            by_class[row.get("class", "unknown")] += 1
    if not bad_textures:
        raise RuntimeError(f"no rejected textures found in {records_csv}")
    return bad_textures, bad_names, by_class


def load_booster_source_map(booster_records_csv: Path) -> tuple[dict[str, str], dict[str, str]]:
    source_by_booster_stem: dict[str, str] = {}
    class_by_booster_stem: dict[str, str] = {}
    with booster_records_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            booster_stem = Path(row["image"]).stem
            source_stem = Path(row["source"]).stem
            source_by_booster_stem[booster_stem] = source_stem
            classes = [part for part in str(row.get("classes", "")).split("|") if part and part != "plain"]
            class_by_booster_stem[booster_stem] = classes[0] if classes else label_from_stem(source_stem)
    return source_by_booster_stem, class_by_booster_stem


def source_stem_for_materialized_stem(stem: str, booster_source_by_stem: dict[str, str]) -> tuple[str | None, str, str]:
    if stem.startswith("base_"):
        source_stem = stem[len("base_") :]
        return source_stem, "base", label_from_stem(source_stem)
    if stem.startswith("boost") and "_" in stem:
        booster_stem = stem.split("_", 1)[1]
        source_stem = booster_source_by_stem.get(booster_stem)
        return source_stem, "boost", label_from_stem(source_stem or "")
    return None, "unknown", "unknown"


def collect_materialized_entries(
    materialized_root: Path,
    booster_source_by_stem: dict[str, str],
) -> tuple[list[dict], dict[str, None], Counter, Counter]:
    entries: list[dict] = []
    needed_unified: dict[str, None] = {}
    split_counts: Counter = Counter()
    kind_counts: Counter = Counter()
    for split in ["train", "val"]:
        for image_path in sorted((materialized_root / "images" / split).glob("*.jpg")):
            source_stem, kind, class_name = source_stem_for_materialized_stem(image_path.stem, booster_source_by_stem)
            if source_stem:
                needed_unified.setdefault(source_stem, None)
            entries.append(
                {
                    "split": split,
                    "image": image_path,
                    "label": label_for_image(image_path),
                    "stem": image_path.stem,
                    "kind": kind,
                    "source_stem": source_stem or "",
                    "class": class_name,
                }
            )
            split_counts[split] += 1
            kind_counts[kind] += 1
    return entries, needed_unified, split_counts, kind_counts


def resolve_unified_quads(unified_root: Path, needed_stems: dict[str, None]) -> tuple[dict[str, tuple[str, str, str]], defaultdict]:
    unified_info: dict[str, tuple[str, str, str]] = {}
    meta_to_objects: defaultdict[str, set[str]] = defaultdict(set)
    for stem in needed_stems:
        quad_path = unified_root / "quads" / "train" / f"{stem}.json"
        if not quad_path.exists():
            quad_path = unified_root / "quads" / "val" / f"{stem}.json"
        if not quad_path.exists():
            continue
        data = fast_json.loads(quad_path.read_text(encoding="utf-8"))
        meta_rel = data["source_meta"]
        object_name = data["object_name"]
        class_name = label_from_stem(stem)
        unified_info[stem] = (meta_rel, object_name, class_name)
        meta_to_objects[meta_rel].add(object_name)
    return unified_info, meta_to_objects


def texture_candidates(face_texture: dict) -> list[str]:
    candidates: list[str] = []
    texture = face_texture.get("texture")
    if texture:
        candidates.append(str(texture))
    candidates.extend(str(item) for item in (face_texture.get("source_textures") or []))
    return candidates


def find_bad_objects(
    source_meta_root: Path,
    meta_to_objects: dict[str, set[str]],
    bad_textures: set[str],
    bad_names: set[str],
) -> tuple[set[tuple[str, str]], Counter, int]:
    bad_name_re = re.compile("|".join(re.escape(name) for name in sorted(bad_names)))
    bad_objects: set[tuple[str, str]] = set()
    by_class: Counter = Counter()
    parsed_count = 0
    for meta_rel, object_names in meta_to_objects.items():
        meta_path = source_meta_root / meta_rel
        if not meta_path.exists():
            continue
        text = meta_path.read_text(encoding="utf-8", errors="ignore")
        if not bad_name_re.search(text):
            continue
        parsed_count += 1
        data = fast_json.loads(text)
        for obj in data.get("label_objects", []):
            object_name = obj.get("object_name")
            if object_name not in object_names:
                continue
            hit = False
            for face_texture in obj.get("face_textures", []) or []:
                if any(norm_path(candidate) in bad_textures for candidate in texture_candidates(face_texture)):
                    hit = True
                    break
            if hit:
                bad_objects.add((meta_rel, object_name))
                by_class[obj.get("class", "unknown")] += 1
    return bad_objects, by_class, parsed_count


def make_pruned_dataset(args: argparse.Namespace) -> dict:
    bad_textures, bad_names, bad_textures_by_class = load_bad_textures(args.rejected_texture_records)
    booster_source_by_stem, booster_class_by_stem = load_booster_source_map(args.booster_records)
    entries, needed_unified, materialized_counts, materialized_kind_counts = collect_materialized_entries(
        args.materialized_root,
        booster_source_by_stem,
    )
    unified_info, meta_to_objects = resolve_unified_quads(args.unified_root, needed_unified)
    bad_objects, bad_objects_by_class, parsed_meta_count = find_bad_objects(
        args.source_meta_root,
        meta_to_objects,
        bad_textures,
        bad_names,
    )

    bad_unified_stems: set[str] = set()
    bad_unified_by_class: Counter = Counter()
    for stem, (meta_rel, object_name, class_name) in unified_info.items():
        if (meta_rel, object_name) in bad_objects:
            bad_unified_stems.add(stem)
            bad_unified_by_class[class_name] += 1

    if args.reset:
        safe_reset(args.output_root, args.workspace)
    for split in ["train", "val"]:
        (args.output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    kept_counts: Counter = Counter()
    removed_counts: Counter = Counter()
    removed_by_class: Counter = Counter()
    removed_by_kind: Counter = Counter()
    link_counts: Counter = Counter()
    record_rows: list[dict] = []
    for entry in entries:
        source_stem = entry["source_stem"]
        remove = bool(source_stem and source_stem in bad_unified_stems)
        split = entry["split"]
        if remove:
            removed_counts[split] += 1
            removed_by_class[entry["class"]] += 1
            removed_by_kind[entry["kind"]] += 1
        else:
            kept_counts[split] += 1
            dst_image = args.output_root / "images" / split / entry["image"].name
            dst_label = args.output_root / "labels" / split / entry["label"].name
            link_counts[link_or_copy(entry["image"], dst_image, args.mode)] += 1
            link_counts[link_or_copy(entry["label"], dst_label, args.mode)] += 1
        record_rows.append(
            {
                "split": split,
                "stem": entry["stem"],
                "kind": entry["kind"],
                "class": entry["class"],
                "source_stem": source_stem,
                "keep": int(not remove),
                "reason": "keep" if not remove else "uses_rejected_fruit_texture",
            }
        )

    write_data_yaml(args.output_root)
    records_path = args.output_root / "prune_records.csv"
    with records_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "stem", "kind", "class", "source_stem", "keep", "reason"])
        writer.writeheader()
        writer.writerows(record_rows)

    manifest = {
        "task": "cube_face_unified_pruned_by_rejected_fruit_textures",
        "policy": "Create a new hardlinked materialized dataset only; do not modify the source dataset.",
        "materialized_root": str(args.materialized_root),
        "output_root": str(args.output_root),
        "rejected_texture_records": str(args.rejected_texture_records),
        "source_meta_root": str(args.source_meta_root),
        "unified_root": str(args.unified_root),
        "booster_records": str(args.booster_records),
        "mode": args.mode,
        "bad_textures_by_class": dict(bad_textures_by_class),
        "materialized_total": dict(materialized_counts),
        "materialized_kind_counts": dict(materialized_kind_counts),
        "needed_unified_crops": len(needed_unified),
        "resolved_unified_quads": len(unified_info),
        "source_metas_needed": len(meta_to_objects),
        "source_metas_parsed_after_bad_name_prefilter": parsed_meta_count,
        "bad_source_objects": len(bad_objects),
        "bad_source_objects_by_class": dict(bad_objects_by_class),
        "bad_unified_source_crops": len(bad_unified_stems),
        "bad_unified_source_crops_by_class": dict(bad_unified_by_class),
        "kept": dict(kept_counts),
        "removed": dict(removed_counts),
        "removed_by_class": dict(removed_by_class),
        "removed_by_kind": dict(removed_by_kind),
        "link_counts": dict(link_counts),
        "records": str(records_path),
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune a materialized cube-face unified dataset by rejected fruit textures.")
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument(
        "--materialized_root",
        type=Path,
        default=Path("datasets/cube_face_unified_finetune_meta_v2_50000_plus_hsv_ratio_20000_stronger_v1"),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("datasets/cube_face_unified_finetune_meta_v2_50000_plus_hsv_ratio_20000_stronger_pruned_coloroutlier_v1"),
    )
    parser.add_argument(
        "--rejected_texture_records",
        type=Path,
        default=Path("reports/cube_face_unified_eval/fruit_texture_production_color_filtered_v2/records.csv"),
    )
    parser.add_argument("--source_meta_root", type=Path, default=Path("datasets/meta_v2_50000_coco_texture_v1"))
    parser.add_argument("--unified_root", type=Path, default=Path("datasets/meta_v2_50000_cube_face_unified_v1"))
    parser.add_argument(
        "--booster_records",
        type=Path,
        default=Path("datasets/cube_face_unified_booster_meta_v2_hsv_ratio_20000_stronger_v1/records.csv"),
    )
    parser.add_argument("--mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = make_pruned_dataset(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
