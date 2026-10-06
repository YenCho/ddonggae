from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


FRUITS = ("apple", "orange", "banana", "pineapple")
LABELS = ("apple", "orange", "banana", "pineapple", "plain", "none")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate cube-face probe reports and rank realistic fruit-to-fruit confusion."
    )
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min_fruit_to_fruit_rate", type=float, default=0.05)
    parser.add_argument("--max_fruit_to_plain_rate", type=float, default=0.18)
    parser.add_argument("--max_sheet_items", type=int, default=80)
    return parser.parse_args()


def load_rows(report: Path) -> list[dict[str, str]]:
    csv_path = report / "crop_top_class_predictions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def compute_summary(rows: list[dict[str, str]]) -> dict:
    matrix: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        matrix[row.get("true", "none")][row.get("pred", "none")] += 1

    fruit_total = 0
    fruit_correct = 0
    fruit_to_fruit_wrong = 0
    fruit_to_plain = 0
    offdiag: Counter[tuple[str, str]] = Counter()

    for true in FRUITS:
        for pred in LABELS:
            n = int(matrix[true][pred])
            fruit_total += n
            if pred == true:
                fruit_correct += n
            elif pred in FRUITS:
                fruit_to_fruit_wrong += n
                if n > 0:
                    offdiag[(true, pred)] += n
            elif pred in {"plain", "none"}:
                fruit_to_plain += n

    total = max(1, fruit_total)
    return {
        "fruit_total": fruit_total,
        "fruit_correct": fruit_correct,
        "fruit_to_fruit_wrong": fruit_to_fruit_wrong,
        "fruit_to_plain_or_none": fruit_to_plain,
        "fruit_to_fruit_wrong_rate": fruit_to_fruit_wrong / total,
        "fruit_to_plain_or_none_rate": fruit_to_plain / total,
        "matrix": {t: {p: int(matrix[t][p]) for p in LABELS} for t in LABELS},
        "top_fruit_to_fruit": [
            {"true": t, "pred": p, "count": c}
            for (t, p), c in offdiag.most_common(12)
        ],
    }


def draw_matrix(matrix: dict, output: Path, normalize: bool = False) -> None:
    cell = 86
    left = 122
    top = 74
    labels = list(LABELS)
    image = np.full((top + cell * len(labels) + 38, left + cell * len(labels) + 30, 3), 248, dtype=np.uint8)
    cv2.putText(
        image,
        "normalized" if normalize else "counts",
        (left, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    values = np.array([[float(matrix.get(t, {}).get(p, 0)) for p in labels] for t in labels], dtype=np.float32)
    if normalize:
        denom = values.sum(axis=1, keepdims=True)
        values = np.divide(values, np.maximum(denom, 1.0))
    max_value = float(values.max()) if values.size else 1.0
    max_value = max(max_value, 1e-6)

    for i, true in enumerate(labels):
        y = top + i * cell
        cv2.putText(image, true, (8, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (35, 35, 35), 1, cv2.LINE_AA)
    for j, pred in enumerate(labels):
        x = left + j * cell
        cv2.putText(image, pred[:8], (x + 4, top - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (35, 35, 35), 1, cv2.LINE_AA)

    for i in range(len(labels)):
        for j in range(len(labels)):
            x = left + j * cell
            y = top + i * cell
            v = float(values[i, j])
            intensity = int(255 - 155 * (v / max_value))
            color = (255, intensity, intensity) if i != j else (intensity, 255, intensity)
            cv2.rectangle(image, (x, y), (x + cell - 2, y + cell - 2), color, -1)
            cv2.rectangle(image, (x, y), (x + cell - 2, y + cell - 2), (210, 210, 210), 1)
            text = f"{v:.2f}" if normalize else str(int(values[i, j]))
            cv2.putText(image, text, (x + 16, y + 49), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 1, cv2.LINE_AA)

    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image)


def make_contact_sheet(rows: list[dict[str, str]], output: Path, cols: int = 5, tile: int = 160) -> None:
    cells = []
    for row in rows[:80]:
        image_path = Path(row["image"])
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA)
        canvas = np.full((tile + 30, tile, 3), 245, dtype=np.uint8)
        canvas[30:] = img
        label = f'{row["true"]}->{row["pred"]} {float(row["conf"]):.2f}'
        cv2.putText(canvas, label[:36], (4, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
        cells.append(canvas)
    if not cells:
        cells = [np.full((tile + 30, tile, 3), 245, dtype=np.uint8)]
        cv2.putText(cells[0], "no fruit-to-fruit errors", (5, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
    rows_n = int(np.ceil(len(cells) / cols))
    sheet = np.full((rows_n * cells[0].shape[0], cols * cells[0].shape[1], 3), 245, dtype=np.uint8)
    for idx, cell_img in enumerate(cells):
        r, c = divmod(idx, cols)
        h, w = cell_img.shape[:2]
        sheet[r * h : (r + 1) * h, c * w : (c + 1) * w] = cell_img
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 94])


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    report_summaries = []
    all_fruit_to_fruit_rows: list[dict[str, str]] = []
    for report in args.reports:
        rows = load_rows(report)
        summary = compute_summary(rows)
        status = "candidate" if (
            summary["fruit_to_fruit_wrong_rate"] >= args.min_fruit_to_fruit_rate
            and summary["fruit_to_plain_or_none_rate"] <= args.max_fruit_to_plain_rate
        ) else "reject"
        summary["status"] = status
        summary["report"] = str(report)
        report_summaries.append(summary)

        draw_matrix(summary["matrix"], args.output / f"{report.name}_confusion_counts.png", normalize=False)
        draw_matrix(summary["matrix"], args.output / f"{report.name}_confusion_normalized.png", normalize=True)

        for row in rows:
            if row.get("true") in FRUITS and row.get("pred") in FRUITS and row.get("true") != row.get("pred"):
                copied = dict(row)
                copied["report"] = report.name
                all_fruit_to_fruit_rows.append(copied)

    all_fruit_to_fruit_rows.sort(key=lambda row: float(row.get("conf", 0.0)), reverse=True)
    make_contact_sheet(all_fruit_to_fruit_rows[: args.max_sheet_items], args.output / "fruit_to_fruit_error_sheet.jpg")

    payload = {
        "min_fruit_to_fruit_rate": args.min_fruit_to_fruit_rate,
        "max_fruit_to_plain_rate": args.max_fruit_to_plain_rate,
        "reports": report_summaries,
    }
    (args.output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Fruit-to-Fruit Confusion Search",
        "",
        "Goal: find realistic COCO-based probe settings that create fruit-to-fruit confusion, not fruit-to-plain failures.",
        "",
        "| status | report | fruit total | fruit->fruit wrong | rate | fruit->plain/none | plain rate | top fruit off-diagonal |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for summary in report_summaries:
        top = ", ".join(
            f'{item["true"]}->{item["pred"]}:{item["count"]}'
            for item in summary["top_fruit_to_fruit"][:4]
        ) or "none"
        lines.append(
            "| {status} | `{name}` | {total} | {ff} | {ffr:.3f} | {fp} | {fpr:.3f} | {top} |".format(
                status=summary["status"],
                name=Path(summary["report"]).name,
                total=summary["fruit_total"],
                ff=summary["fruit_to_fruit_wrong"],
                ffr=summary["fruit_to_fruit_wrong_rate"],
                fp=summary["fruit_to_plain_or_none"],
                fpr=summary["fruit_to_plain_or_none_rate"],
                top=top,
            )
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- JSON: `{args.output / 'summary.json'}`",
            f"- fruit-to-fruit error sheet: `{args.output / 'fruit_to_fruit_error_sheet.jpg'}`",
            "- per-report confusion matrix PNG files are saved in this directory.",
        ]
    )
    (args.output / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
