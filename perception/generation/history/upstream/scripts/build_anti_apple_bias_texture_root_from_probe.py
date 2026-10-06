#!/usr/bin/env python3
"""Build a non-apple hard-case texture root to reduce apple-biased predictions.

This script mines existing unified-model probe CSVs and creates a texture root
that can be used as a conservative booster source. It does not train a model.

Default policy:
- use reviewed candidate probe rows, not noisy web rows
- select orange/banana/pineapple rows that were predicted as apple
- add low-confidence/no-detection non-apple rows
- keep a small apple anchor set so the model does not forget apple
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import cv2
except ModuleNotFoundError:  # Contact sheets are optional; manifest generation still works.
    cv2 = None


CLASSES = ("apple", "orange", "banana", "pineapple")
NON_APPLE = ("orange", "banana", "pineapple")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine non-apple apple-confusion cases into a training texture root."
    )
    parser.add_argument(
        "--probe_csv",
        type=Path,
        default=Path("reports/cube_face_unified_eval/fruit_texture_candidate_probe_20260702/predictions.csv"),
        help="Probe CSV from the preferred unified model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/fruit_textures/other/targeted_anti_apple_bias_v1"),
    )
    parser.add_argument(
        "--allowed_datasets",
        nargs="+",
        default=["ai_wholefruit_balanced", "ai_printed_fullsquare", "combined_curated"],
        help="Datasets allowed for mining. Web cutouts are intentionally excluded by default.",
    )
    parser.add_argument("--apple_confusion_max_per_class", type=int, default=220)
    parser.add_argument("--low_conf_max_per_class", type=int, default=220)
    parser.add_argument("--strong_anchor_max_per_nonapple", type=int, default=160)
    parser.add_argument("--apple_anchor_max", type=int, default=120)
    parser.add_argument("--apple_anchor_min_conf", type=float, default=0.80)
    parser.add_argument("--copy_mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--clear", action="store_true")
    return parser.parse_args()


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy_fallback"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def row_path(root: Path, row: dict[str, str]) -> Path:
    raw = Path(row["path"])
    return raw if raw.is_absolute() else root / raw


def score_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


def output_name(row: dict[str, str], index: int, src: Path, reason: str) -> str:
    conf = int(round(score_float(row, "conf") * 1000))
    return (
        f"{index:04d}_{safe_name(row.get('dataset', 'dataset'))}_"
        f"{reason}_pred-{safe_name(row.get('pred', 'pred'))}_"
        f"conf-{conf:03d}_{safe_name(src.stem)}{src.suffix.lower()}"
    )


def unique_existing_rows(root: Path, rows: list[dict[str, str]], allowed: set[str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[Path] = set()
    for row in rows:
        expected = row.get("expected", "")
        src = row_path(root, row)
        if row.get("dataset") not in allowed:
            continue
        if expected not in CLASSES:
            continue
        if src.suffix.lower() not in IMAGE_EXTS:
            continue
        if not src.exists():
            continue
        resolved = src.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(row)
    return out


def select_rows(args: argparse.Namespace, root: Path, rows: list[dict[str, str]]) -> dict[str, list[tuple[str, dict[str, str]]]]:
    allowed = set(args.allowed_datasets)
    rows = unique_existing_rows(root, rows, allowed)
    selected: dict[str, list[tuple[str, dict[str, str]]]] = {name: [] for name in CLASSES}

    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_class[row.get("expected", "")].append(row)

    for class_name in NON_APPLE:
        class_rows = by_class[class_name]

        apple_confusions = [
            row
            for row in class_rows
            if row.get("status") == "wrong" and row.get("pred") == "apple"
        ]
        apple_confusions.sort(key=lambda row: score_float(row, "conf"), reverse=True)
        selected[class_name].extend(
            ("apple_confusion", row) for row in apple_confusions[: args.apple_confusion_max_per_class]
        )

        low_conf = [
            row
            for row in class_rows
            if row.get("status") in {"correct_low_conf", "no_detection"}
        ]
        low_conf.sort(
            key=lambda row: (
                0 if row.get("status") == "no_detection" else 1,
                score_float(row, "conf"),
            )
        )
        selected[class_name].extend(
            ("low_conf_or_no_det", row) for row in low_conf[: args.low_conf_max_per_class]
        )

        anchors = [
            row
            for row in class_rows
            if row.get("status") == "correct_strong" and row.get("pred") == class_name
        ]
        anchors.sort(key=lambda row: score_float(row, "conf"), reverse=True)
        selected[class_name].extend(
            ("strong_anchor", row) for row in anchors[: args.strong_anchor_max_per_nonapple]
        )

    apple_anchors = [
        row
        for row in by_class["apple"]
        if row.get("status") == "correct_strong"
        and row.get("pred") == "apple"
        and score_float(row, "conf") >= args.apple_anchor_min_conf
    ]
    apple_anchors.sort(key=lambda row: score_float(row, "conf"), reverse=True)
    selected["apple"].extend(("apple_anchor", row) for row in apple_anchors[: args.apple_anchor_max])

    return selected


def make_contact_sheet(paths: list[Path], output: Path, title: str, tile: int = 128, cols: int = 8) -> None:
    if cv2 is None:
        return
    selected = paths[: min(len(paths), 64)]
    if not selected:
        return
    rows = int(np.ceil(len(selected) / cols))
    header = 30
    canvas = np.full((rows * tile + header, cols * tile, 3), 245, dtype=np.uint8)
    cv2.putText(canvas, title[:96], (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 1, cv2.LINE_AA)
    for idx, path in enumerate(selected):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.resize(image, (tile, tile), interpolation=cv2.INTER_AREA)
        y = header + (idx // cols) * tile
        x = (idx % cols) * tile
        canvas[y : y + tile, x : x + tile] = image
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    probe_csv = args.probe_csv if args.probe_csv.is_absolute() else root / args.probe_csv
    output = args.output if args.output.is_absolute() else root / args.output

    if args.clear:
        reset_dir(output)
    output.mkdir(parents=True, exist_ok=True)
    for class_name in CLASSES:
        (output / class_name).mkdir(parents=True, exist_ok=True)

    rows = read_rows(probe_csv)
    selected = select_rows(args, root, rows)

    manifest: list[dict[str, str | float]] = []
    copied_paths: dict[str, list[Path]] = defaultdict(list)
    copy_stats = Counter()
    reason_counts = Counter()

    for class_name, entries in selected.items():
        seen: set[Path] = set()
        for index, (reason, row) in enumerate(entries):
            src = row_path(root, row).resolve()
            if src in seen:
                continue
            seen.add(src)
            dst = output / class_name / output_name(row, index, src, reason)
            op = link_or_copy(src, dst, args.copy_mode)
            copy_stats[op] += 1
            reason_counts[f"{class_name}:{reason}"] += 1
            copied_paths[class_name].append(dst)
            manifest.append(
                {
                    "class": class_name,
                    "reason": reason,
                    "source": str(src),
                    "output": str(dst.relative_to(output)),
                    "probe_dataset": row.get("dataset", ""),
                    "expected": row.get("expected", ""),
                    "probe_pred": row.get("pred", ""),
                    "probe_conf": score_float(row, "conf"),
                    "probe_status": row.get("status", ""),
                    "copy_op": op,
                }
            )

    for class_name, paths in copied_paths.items():
        make_contact_sheet(paths, output / "previews" / f"{class_name}_contact_sheet.jpg", class_name)

    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "class",
            "reason",
            "source",
            "output",
            "probe_dataset",
            "expected",
            "probe_pred",
            "probe_conf",
            "probe_status",
            "copy_op",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(output),
        "probe_csv": str(probe_csv),
        "allowed_datasets": args.allowed_datasets,
        "policy": {
            "goal": "reduce non-apple fruit being predicted as apple without destroying apple recall",
            "non_apple": "mine rows where orange/banana/pineapple were predicted as apple, plus low-confidence/no-detection rows",
            "apple": "small high-confidence apple anchor set only",
            "web": "excluded by default; use only after manual cleanup",
        },
        "counts_by_class": dict(Counter(row["class"] for row in manifest)),
        "counts_by_reason": dict(reason_counts),
        "copy_stats": dict(copy_stats),
        "manual_check": "Inspect previews/*.jpg before using this root for any booster training.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
