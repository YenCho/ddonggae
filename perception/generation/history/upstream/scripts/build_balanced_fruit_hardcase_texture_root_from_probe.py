#!/usr/bin/env python3
"""Build a balanced fruit hard-case texture root from unified-model probes.

This is the preferred booster-prep script when the symptom is a general fruit
classification bias. It does not target a single label. Instead, it mines each
fruit class with the same policy:

- high-risk rows: wrong, no_detection
- anchor rows: correct_strong high-confidence examples
- default source policy: verified printed/full-square probes only; web and
  noisy mined sets are excluded unless manually inspected

The script only prepares a texture root with manifest/summary/previews. It does
not train or promote any model.
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
except ModuleNotFoundError:  # Preview sheets are optional.
    cv2 = None


CLASSES = ("apple", "orange", "banana", "pineapple")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
HARD_STATUS_ORDER = {"wrong": 0, "no_detection": 1, "correct_low_conf": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine balanced per-class fruit hard cases from probe CSVs."
    )
    parser.add_argument(
        "--probe_csv",
        type=Path,
        nargs="+",
        default=[Path("reports/cube_face_unified_eval/fruit_texture_candidate_probe_20260702/predictions.csv")],
        help="One or more probe CSVs from the preferred unified model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/fruit_textures/other/balanced_fruit_hardcase_v1"),
    )
    parser.add_argument(
        "--allowed_datasets",
        nargs="+",
        default=["ai_printed_fullsquare"],
        help=(
            "Probe dataset names allowed for mining. Default is verified-only; "
            "do not include combined_curated/web sets unless manually inspected."
        ),
    )
    parser.add_argument(
        "--include_correct_low_conf",
        action="store_true",
        help="Also include correct_low_conf rows. Default keeps only wrong/no_detection hard rows plus anchors.",
    )
    parser.add_argument("--hard_max_per_class", type=int, default=420)
    parser.add_argument("--anchor_max_per_class", type=int, default=180)
    parser.add_argument("--anchor_min_conf", type=float, default=0.75)
    parser.add_argument("--max_per_class", type=int, default=600)
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


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["_probe_csv"] = str(path)
                rows.append(row)
    return rows


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


def existing_probe_rows(root: Path, rows: list[dict[str, str]], allowed: set[str]) -> list[dict[str, str]]:
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


def hard_sort_key(row: dict[str, str]) -> tuple[int, float]:
    status = row.get("status", "")
    conf = score_float(row, "conf")
    # For wrong rows, high-confidence mistakes are the most valuable. For
    # low-confidence correct rows, lower confidence is harder.
    if status == "wrong":
        return (HARD_STATUS_ORDER[status], -conf)
    return (HARD_STATUS_ORDER.get(status, 99), conf)


def select_rows(args: argparse.Namespace, root: Path, rows: list[dict[str, str]]) -> dict[str, list[tuple[str, dict[str, str]]]]:
    allowed = set(args.allowed_datasets)
    hard_statuses = {"wrong", "no_detection"}
    if args.include_correct_low_conf:
        hard_statuses.add("correct_low_conf")
    rows = existing_probe_rows(root, rows, allowed)
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_class[row.get("expected", "")].append(row)

    selected: dict[str, list[tuple[str, dict[str, str]]]] = {name: [] for name in CLASSES}
    for class_name in CLASSES:
        class_rows = by_class[class_name]

        hard = [
            row
            for row in class_rows
            if row.get("status") in hard_statuses
        ]
        hard.sort(key=hard_sort_key)
        selected[class_name].extend(("hard_" + row.get("status", "status"), row) for row in hard[: args.hard_max_per_class])

        anchors = [
            row
            for row in class_rows
            if row.get("status") == "correct_strong"
            and row.get("pred") == class_name
            and score_float(row, "conf") >= args.anchor_min_conf
        ]
        anchors.sort(key=lambda row: score_float(row, "conf"), reverse=True)
        remaining = max(0, args.max_per_class - len(selected[class_name]))
        anchor_limit = min(args.anchor_max_per_class, remaining)
        selected[class_name].extend(("strong_anchor", row) for row in anchors[:anchor_limit])

        if len(selected[class_name]) > args.max_per_class:
            selected[class_name] = selected[class_name][: args.max_per_class]

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
    probe_csvs = [path if path.is_absolute() else root / path for path in args.probe_csv]
    output = args.output if args.output.is_absolute() else root / args.output

    if args.clear:
        reset_dir(output)
    output.mkdir(parents=True, exist_ok=True)
    for class_name in CLASSES:
        (output / class_name).mkdir(parents=True, exist_ok=True)

    rows = read_rows(probe_csvs)
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
                    "probe_csv": row.get("_probe_csv", ""),
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
            "probe_csv",
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
        "probe_csv": [str(path) for path in probe_csvs],
        "allowed_datasets": args.allowed_datasets,
        "policy": {
            "goal": "balanced fruit confidence improvement without targeting one label",
            "hard_rows": (
                "wrong/no_detection rows per class"
                + (" plus correct_low_conf rows" if args.include_correct_low_conf else "")
            ),
            "anchors": "correct_strong rows per class to preserve class identity",
            "web": "excluded by default; include only after manual cleanup",
            "combined_curated": "excluded by default after preview inspection showed contaminated apple examples",
        },
        "limits": {
            "hard_max_per_class": args.hard_max_per_class,
            "anchor_max_per_class": args.anchor_max_per_class,
            "anchor_min_conf": args.anchor_min_conf,
            "max_per_class": args.max_per_class,
        },
        "counts_by_class": dict(Counter(row["class"] for row in manifest)),
        "counts_by_reason": dict(reason_counts),
        "copy_stats": dict(copy_stats),
        "manual_check": "Inspect previews/*.jpg before using this root for booster training.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
