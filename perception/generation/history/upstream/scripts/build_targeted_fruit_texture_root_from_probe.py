from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


CLASSES = ("apple", "orange", "banana", "pineapple")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a clean targeted fruit texture root from unified-model probe CSVs. "
            "The default recipe mines banana hard cases while keeping high-confidence "
            "control samples from the other fruit classes."
        )
    )
    parser.add_argument(
        "--probe_csv",
        type=Path,
        default=Path("reports/cube_face_unified_eval/fruit_texture_candidate_probe_20260702/predictions.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/fruit_textures/other/targeted_banana_hard_v1"),
    )
    parser.add_argument("--target_class", choices=CLASSES, default="banana")
    parser.add_argument(
        "--allowed_datasets",
        nargs="+",
        default=["ai_wholefruit_balanced", "ai_printed_fullsquare"],
        help="Probe dataset names allowed for mining. Keep web sets out unless manually cleaned.",
    )
    parser.add_argument("--target_max", type=int, default=420)
    parser.add_argument("--target_easy_max", type=int, default=80)
    parser.add_argument("--control_max_per_class", type=int, default=160)
    parser.add_argument("--control_min_conf", type=float, default=0.75)
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


def clean_name(row: dict[str, str], index: int, src: Path) -> str:
    dataset = row.get("dataset", "dataset")
    status = row.get("status", "status")
    pred = row.get("pred", "pred")
    conf = int(round(score_float(row, "conf") * 1000))
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in src.stem)
    return f"{index:04d}_{dataset}_{status}_pred-{pred}_conf-{conf:03d}_{stem}{src.suffix.lower()}"


def select_rows(args: argparse.Namespace, root: Path, rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    allowed = set(args.allowed_datasets)
    mined: dict[str, list[dict[str, str]]] = {name: [] for name in CLASSES}

    rows = [
        row
        for row in rows
        if row.get("dataset") in allowed
        and row.get("expected") in CLASSES
        and row_path(root, row).suffix.lower() in IMAGE_EXTS
        and row_path(root, row).exists()
    ]

    target_hard = [
        row
        for row in rows
        if row.get("expected") == args.target_class
        and row.get("status") in {"wrong", "correct_low_conf", "no_detection"}
    ]
    target_hard.sort(
        key=lambda row: (
            0 if row.get("status") == "wrong" else 1 if row.get("status") == "no_detection" else 2,
            score_float(row, "conf"),
        )
    )
    mined[args.target_class].extend(target_hard[: args.target_max])

    target_easy = [
        row
        for row in rows
        if row.get("expected") == args.target_class
        and row.get("pred") == args.target_class
        and row.get("status") == "correct_strong"
    ]
    target_easy.sort(key=lambda row: score_float(row, "conf"), reverse=True)
    mined[args.target_class].extend(target_easy[: args.target_easy_max])

    for class_name in CLASSES:
        if class_name == args.target_class:
            continue
        controls = [
            row
            for row in rows
            if row.get("expected") == class_name
            and row.get("pred") == class_name
            and row.get("status") == "correct_strong"
            and score_float(row, "conf") >= args.control_min_conf
        ]
        controls.sort(key=lambda row: score_float(row, "conf"), reverse=True)
        mined[class_name].extend(controls[: args.control_max_per_class])

    return mined


def make_contact_sheet(paths: list[Path], output: Path, title: str, tile: int = 128, cols: int = 8) -> None:
    selected = paths[: min(len(paths), 64)]
    if not selected:
        return
    rows = int(np.ceil(len(selected) / cols))
    header = 28
    canvas = np.full((rows * tile + header, cols * tile, 3), 245, dtype=np.uint8)
    cv2.putText(canvas, title[:96], (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 1, cv2.LINE_AA)
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
    args.probe_csv = args.probe_csv if args.probe_csv.is_absolute() else root / args.probe_csv
    args.output = args.output if args.output.is_absolute() else root / args.output

    if args.clear:
        reset_dir(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    for class_name in CLASSES:
        (args.output / class_name).mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.probe_csv)
    selected = select_rows(args, root, rows)

    manifest_rows: list[dict[str, str | int | float]] = []
    copy_stats = Counter()
    for class_name, class_rows in selected.items():
        written_paths: list[Path] = []
        seen_sources: set[Path] = set()
        for idx, row in enumerate(class_rows):
            src = row_path(root, row).resolve()
            if src in seen_sources:
                continue
            seen_sources.add(src)
            dst = args.output / class_name / clean_name(row, idx, src)
            op = link_or_copy(src, dst, args.copy_mode)
            copy_stats[op] += 1
            written_paths.append(dst)
            manifest_rows.append(
                {
                    "class": class_name,
                    "source": str(src),
                    "output": str(dst.relative_to(args.output)),
                    "probe_dataset": row.get("dataset", ""),
                    "expected": row.get("expected", ""),
                    "probe_pred": row.get("pred", ""),
                    "probe_conf": score_float(row, "conf"),
                    "probe_status": row.get("status", ""),
                    "copy_op": op,
                }
            )
        make_contact_sheet(written_paths, args.output / "previews" / f"{class_name}_contact_sheet.jpg", class_name)

    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in manifest_rows for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(args.output),
        "probe_csv": str(args.probe_csv),
        "target_class": args.target_class,
        "allowed_datasets": args.allowed_datasets,
        "selection_policy": {
            "target_hard": "target class rows with wrong, no_detection, or correct_low_conf status",
            "target_easy": "small high-confidence target-class anchor set",
            "controls": "high-confidence correct non-target fruit rows",
            "web_data": "excluded by default because probe previews showed high non-fruit noise",
        },
        "counts": Counter(row["class"] for row in manifest_rows),
        "copy_stats": dict(copy_stats),
    }
    summary["counts"] = dict(summary["counts"])
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
