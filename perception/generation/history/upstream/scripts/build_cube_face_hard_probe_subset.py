from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


FRUITS = {"apple", "orange", "banana", "pineapple"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a cube-face unified hard probe subset from existing evaluated crops."
    )
    parser.add_argument("--prediction_csv", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target_offdiag_rate", type=float, default=0.12)
    parser.add_argument("--weak_true_conf", type=float, default=0.80)
    parser.add_argument("--max_images", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260705)
    return parser.parse_args()


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                row["_source_csv"] = str(path)
                rows.append(row)
    return rows


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for idx, part in enumerate(parts):
        if part == "images":
            parts[idx] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def copy_sample(row: dict[str, str], output: Path, idx: int) -> dict[str, str]:
    image = Path(row["image"])
    label = label_path_for_image(image)
    if not image.exists():
        raise FileNotFoundError(image)
    if not label.exists():
        raise FileNotFoundError(label)

    true = row.get("true", "none")
    pred = row.get("pred", "none")
    stem = f"{idx:06d}_{true}_pred_{pred}_{image.stem}"
    image_out = output / "images" / "val" / f"{stem}{image.suffix.lower()}"
    label_out = output / "labels" / "val" / f"{stem}.txt"
    image_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, image_out)
    shutil.copy2(label, label_out)
    copied = dict(row)
    copied["image"] = str(image_out)
    copied["label"] = str(label_out)
    return copied


def write_data_yaml(output: Path) -> None:
    yaml = "\n".join(
        [
            f"path: {output.as_posix()}",
            "train: images/val",
            "val: images/val",
            "names:",
            "  0: apple",
            "  1: orange",
            "  2: banana",
            "  3: pineapple",
            "  4: plain",
            "",
        ]
    )
    (output / "data.yaml").write_text(yaml, encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    rows = read_rows(args.prediction_csv)
    fruit_rows = [row for row in rows if row.get("true") in FRUITS]
    wrong = [
        row
        for row in fruit_rows
        if row.get("pred") in FRUITS and row.get("pred") != row.get("true")
    ]
    weak = [
        row
        for row in fruit_rows
        if row not in wrong and float(row.get("true_conf") or 0.0) < args.weak_true_conf
    ]
    correct_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in fruit_rows:
        if row in wrong or row in weak:
            continue
        if row.get("pred") == row.get("true"):
            correct_by_class[row["true"]].append(row)

    rng.shuffle(wrong)
    rng.shuffle(weak)
    for class_rows in correct_by_class.values():
        rng.shuffle(class_rows)

    min_total_for_target = math.ceil(len(wrong) / max(args.target_offdiag_rate, 1e-6))
    target_total = min(args.max_images, max(min_total_for_target, len(wrong) + len(weak), 40))
    selected = list(wrong)

    for row in weak:
        if len(selected) >= target_total:
            break
        selected.append(row)

    class_cycle = list(sorted(FRUITS))
    cycle_idx = 0
    while len(selected) < target_total and any(correct_by_class.values()):
        cls = class_cycle[cycle_idx % len(class_cycle)]
        cycle_idx += 1
        if correct_by_class[cls]:
            selected.append(correct_by_class[cls].pop())

    if args.output.exists():
        shutil.rmtree(args.output)
    copied_rows = [copy_sample(row, args.output, idx) for idx, row in enumerate(selected)]
    write_data_yaml(args.output)

    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for row in copied_rows:
        matrix[row.get("true", "none")][row.get("pred", "none")] += 1
    fruit_total = sum(matrix[t][p] for t in FRUITS for p in FRUITS | {"plain", "none"})
    fruit_to_fruit_wrong = sum(matrix[t][p] for t in FRUITS for p in FRUITS if p != t)
    fruit_to_plain = sum(matrix[t][p] for t in FRUITS for p in {"plain", "none"})
    summary = {
        "source_csv": [str(path) for path in args.prediction_csv],
        "output": str(args.output),
        "target_offdiag_rate": args.target_offdiag_rate,
        "weak_true_conf": args.weak_true_conf,
        "images": len(copied_rows),
        "fruit_total": fruit_total,
        "fruit_to_fruit_wrong": fruit_to_fruit_wrong,
        "fruit_to_fruit_wrong_rate": fruit_to_fruit_wrong / max(1, fruit_total),
        "fruit_to_plain_or_none": fruit_to_plain,
        "fruit_to_plain_or_none_rate": fruit_to_plain / max(1, fruit_total),
        "matrix": {t: dict(matrix[t]) for t in sorted(matrix)},
    }
    (args.output / "hard_subset_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (args.output / "selected_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(copied_rows[0].keys()) if copied_rows else ["image"])
        writer.writeheader()
        writer.writerows(copied_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
