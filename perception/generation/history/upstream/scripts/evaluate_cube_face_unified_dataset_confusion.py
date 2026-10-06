from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


CLASSES = ("apple", "orange", "banana", "pineapple", "plain")
FRUIT_IDS = {0, 1, 2, 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize crop-level top-class confusion for a cube-face unified YOLO dataset."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument(
        "--prediction_mode",
        choices=["all", "fruit_for_fruit_truth"],
        default="all",
        help="When set to fruit_for_fruit_truth, fruit-labeled crops are summarized by the highest-confidence fruit prediction if one exists.",
    )
    parser.add_argument("--max_sheet_items", type=int, default=80)
    return parser.parse_args()


def read_true_class(label_path: Path) -> str:
    if not label_path.exists():
        return "none"
    ids: list[int] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            ids.append(int(float(parts[0])))
        except ValueError:
            continue
    fruit_ids = [idx for idx in ids if idx in FRUIT_IDS]
    if fruit_ids:
        return CLASSES[Counter(fruit_ids).most_common(1)[0][0]]
    if 4 in ids:
        return "plain"
    return "none"


def class_confidences(result) -> dict[str, float]:
    scores = {name: 0.0 for name in CLASSES}
    if result.boxes is None or len(result.boxes) == 0:
        return scores
    confs = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    for class_id, conf in zip(classes, confs):
        if 0 <= int(class_id) < len(CLASSES):
            label = CLASSES[int(class_id)]
            scores[label] = max(scores[label], float(conf))
    return scores


def top_prediction(result, names: dict[int, str], true_class: str = "none", prediction_mode: str = "all") -> tuple[str, float]:
    if result.boxes is None or len(result.boxes) == 0:
        return "none", 0.0
    confs = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    if prediction_mode == "fruit_for_fruit_truth" and true_class in CLASSES[:4]:
        fruit_indices = [idx for idx, class_id in enumerate(classes) if int(class_id) in FRUIT_IDS]
        if fruit_indices:
            idx = max(fruit_indices, key=lambda item: float(confs[item]))
            class_id = int(classes[idx])
            return str(names.get(class_id, class_id)), float(confs[idx])
    idx = int(np.argmax(confs))
    class_id = int(classes[idx])
    return str(names.get(class_id, class_id)), float(confs[idx])


def draw_label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    bar_h = 32
    canvas = np.full((out.shape[0] + bar_h, out.shape[1], 3), 245, dtype=np.uint8)
    canvas[bar_h:] = out
    cv2.putText(canvas, text[:42], (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def make_contact_sheet(rows: list[dict], output: Path, tile: int = 160, cols: int = 5) -> None:
    if not rows:
        blank = np.full((tile, tile, 3), 245, dtype=np.uint8)
        cv2.putText(blank, "no errors", (18, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.imwrite(str(output), blank)
        return
    cells = []
    for row in rows:
        image = cv2.imread(row["image"], cv2.IMREAD_COLOR)
        if image is None:
            continue
        image = cv2.resize(image, (tile, tile), interpolation=cv2.INTER_AREA)
        label = f'{row["true"]}->{row["pred"]} {float(row["conf"]):.2f}'
        cells.append(draw_label(image, label))
    if not cells:
        return
    cell_h, cell_w = cells[0].shape[:2]
    rows_n = int(np.ceil(len(cells) / cols))
    sheet = np.full((rows_n * cell_h, cols * cell_w, 3), 245, dtype=np.uint8)
    for idx, cell in enumerate(cells):
        r, c = divmod(idx, cols)
        sheet[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w] = cell
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 94])


def main() -> None:
    args = parse_args()
    image_dir = args.dataset / "images" / args.split
    label_dir = args.dataset / "labels" / args.split
    images = sorted([p for p in image_dir.glob("*.jpg")])
    if not images:
        raise RuntimeError(f"no images found: {image_dir}")

    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model), task="segment")

    rows: list[dict] = []
    confusion: dict[str, Counter] = defaultdict(Counter)
    for start in range(0, len(images), args.batch):
        batch_paths = images[start : start + args.batch]
        results = model.predict(
            source=[str(p) for p in batch_paths],
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )
        for image_path, result in zip(batch_paths, results):
            true = read_true_class(label_dir / f"{image_path.stem}.txt")
            pred, conf = top_prediction(result, model.names, true, args.prediction_mode)
            class_scores = class_confidences(result)
            true_conf = class_scores.get(true, 0.0)
            confusion[true][pred] += 1
            rows.append(
                {
                    "image": str(image_path),
                    "true": true,
                    "pred": pred,
                    "conf": f"{conf:.6f}",
                    "true_conf": f"{true_conf:.6f}",
                    "apple_conf": f"{class_scores['apple']:.6f}",
                    "orange_conf": f"{class_scores['orange']:.6f}",
                    "banana_conf": f"{class_scores['banana']:.6f}",
                    "pineapple_conf": f"{class_scores['pineapple']:.6f}",
                    "plain_conf": f"{class_scores['plain']:.6f}",
                    "correct": str(true == pred).lower(),
                }
            )

    errors = [row for row in rows if row["true"] != row["pred"]]
    weak = [row for row in rows if row["true"] == row["pred"] and float(row["conf"]) < 0.55]
    weak_true = [row for row in rows if row["true"] in CLASSES and float(row["true_conf"]) < 0.80]
    weak_orange = [row for row in rows if row["true"] == "orange" and float(row["orange_conf"]) < 0.80]

    csv_path = args.output / "crop_top_class_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "true",
                "pred",
                "conf",
                "true_conf",
                "apple_conf",
                "orange_conf",
                "banana_conf",
                "pineapple_conf",
                "plain_conf",
                "correct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    labels = list(CLASSES) + ["none"]
    matrix = {true: {pred: int(confusion[true][pred]) for pred in labels} for true in labels}
    (args.output / "crop_top_class_confusion.json").write_text(
        json.dumps(matrix, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    make_contact_sheet(errors[: args.max_sheet_items], args.output / "crop_top_class_error_sheet.jpg")
    make_contact_sheet(weak[: args.max_sheet_items], args.output / "crop_top_class_weak_sheet.jpg")
    make_contact_sheet(weak_true[: args.max_sheet_items], args.output / "crop_true_conf_lt080_sheet.jpg")
    make_contact_sheet(weak_orange[: args.max_sheet_items], args.output / "crop_orange_conf_lt080_sheet.jpg")

    lines = [
        "# Cube-Face Unified Crop Top-Class Confusion",
        "",
        f"- dataset: `{args.dataset}`",
        f"- split: `{args.split}`",
        f"- model: `{args.model}`",
        f"- prediction mode: `{args.prediction_mode}`",
        f"- images: {len(rows)}",
        f"- top-class errors: {len(errors)}",
        f"- true-class confidence < 0.80: {len(weak_true)}",
        f"- orange confidence < 0.80 on true orange: {len(weak_orange)}",
        "",
        "| true | correct | total | accuracy | main wrong prediction |",
        "|---|---:|---:|---:|---|",
    ]
    for true in labels:
        total = sum(confusion[true].values())
        if total == 0:
            continue
        correct = confusion[true][true]
        wrong = Counter({k: v for k, v in confusion[true].items() if k != true})
        main_wrong = "none"
        if wrong:
            pred, count = wrong.most_common(1)[0]
            main_wrong = f"{pred} ({count})"
        lines.append(f"| {true} | {correct} | {total} | {correct / total:.3f} | {main_wrong} |")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- CSV: `{csv_path}`",
            f"- JSON confusion: `{args.output / 'crop_top_class_confusion.json'}`",
            f"- error sheet: `{args.output / 'crop_top_class_error_sheet.jpg'}`",
            f"- weak sheet: `{args.output / 'crop_top_class_weak_sheet.jpg'}`",
            f"- true_conf < 0.80 sheet: `{args.output / 'crop_true_conf_lt080_sheet.jpg'}`",
            f"- orange_conf < 0.80 sheet: `{args.output / 'crop_orange_conf_lt080_sheet.jpg'}`",
        ]
    )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
