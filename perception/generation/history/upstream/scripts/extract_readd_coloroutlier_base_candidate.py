from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path


CLASS_NAMES = ["apple", "orange", "banana", "pineapple", "plain"]


def as_posix(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recover color-outlier pruned cube-face crops whose source textures were REAL "
            "(kind=base, no HSV-shift boost) from the unpruned materialized dataset. "
            "These are hue-ambiguous but correctly labeled real-variety crops "
            "(e.g. reddish oranges, orange/yellow apples) that the hue filter removed. "
            "Creates a new hardlinked dataset only; source datasets are not modified."
        )
    )
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument(
        "--materialized_root",
        type=Path,
        default=Path("datasets/cube_face_unified_finetune_meta_v2_50000_plus_hsv_ratio_20000_stronger_v1"),
        help="Unpruned materialized dataset that still contains the removed crops.",
    )
    parser.add_argument(
        "--pruned_root",
        type=Path,
        default=Path("datasets/cube_face_unified_finetune_meta_v2_50000_plus_hsv_ratio_20000_stronger_pruned_coloroutlier_v1"),
        help="Pruned dataset root containing prune_records.csv.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("datasets/cube_face_unified_readd_coloroutlier_base_v1"),
    )
    parser.add_argument(
        "--include_kinds",
        default="base",
        help="Comma-separated crop kinds to recover. Default base only (boost = synthetic HSV shift, excluded).",
    )
    parser.add_argument("--mode", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    records_csv = args.pruned_root / "prune_records.csv"
    if not records_csv.exists():
        raise RuntimeError(f"prune records not found: {records_csv}")

    include_kinds = {kind.strip() for kind in args.include_kinds.split(",") if kind.strip()}

    if args.reset:
        safe_reset(args.output_root, args.workspace)
    for split in ["train", "val"]:
        (args.output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    selected: Counter = Counter()
    selected_by_class: Counter = Counter()
    skipped_kind: Counter = Counter()
    missing: list[str] = []
    link_counts: Counter = Counter()

    with records_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("keep", "")).strip() != "0":
                continue
            kind = row.get("kind", "unknown")
            if kind not in include_kinds:
                skipped_kind[kind] += 1
                continue
            split = row["split"]
            stem = row["stem"]
            src_image = args.materialized_root / "images" / split / f"{stem}.jpg"
            src_label = args.materialized_root / "labels" / split / f"{stem}.txt"
            if not src_image.exists() or not src_label.exists():
                missing.append(stem)
                continue
            dst_image = args.output_root / "images" / split / src_image.name
            dst_label = args.output_root / "labels" / split / src_label.name
            link_counts[link_or_copy(src_image, dst_image, args.mode)] += 1
            link_counts[link_or_copy(src_label, dst_label, args.mode)] += 1
            selected[split] += 1
            selected_by_class[row.get("class", "unknown")] += 1

    if sum(selected.values()) == 0:
        raise RuntimeError("no crops recovered; check prune_records.csv and include_kinds")

    write_data_yaml(args.output_root)
    manifest = {
        "task": "cube_face_unified_readd_coloroutlier_base_candidate",
        "policy": (
            "Recover keep=0 crops of the selected kinds from the unpruned materialized dataset. "
            "Real-texture (base) crops only by default; HSV-shift boost crops stay excluded. "
            "New hardlinked dataset only; sources unmodified."
        ),
        "materialized_root": str(args.materialized_root),
        "pruned_records": str(records_csv),
        "output_root": str(args.output_root),
        "include_kinds": sorted(include_kinds),
        "mode": args.mode,
        "selected": dict(selected),
        "selected_by_class": dict(selected_by_class),
        "skipped_by_kind": dict(skipped_kind),
        "missing_sources": len(missing),
        "missing_examples": missing[:20],
        "link_counts": dict(link_counts),
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
