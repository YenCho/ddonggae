from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
CONTACT_SHEET_THUMB_SIZE = (360, 270)
CONTACT_SHEET_LABEL_HEIGHT = 24
CONTACT_SHEET_MAX_DIMENSION = 60000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render side-by-side YOLO model predictions and simulation YOLO labels "
            "for a Replicator run."
        )
    )
    parser.add_argument(
        "dataset",
        help="Replicator run directory containing yolo/, or the YOLO output directory itself.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="YOLO model checkpoint, for example results/training/.../weights/best.pt.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Defaults to <run-dir>/model_compare.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold.")
    parser.add_argument("--device", default="0", help="Ultralytics device, for example 0 or cpu.")
    parser.add_argument("--batch", type=int, default=25, help="Inference batch size.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of images, 0 means all.")
    parser.add_argument(
        "--split",
        choices=("all", "train", "val", "test"),
        default="all",
        help="Use a split file from yolo/splits instead of all images.",
    )
    parser.add_argument("--contact-sheet-columns", type=int, default=4, help="Contact sheet columns.")
    parser.add_argument(
        "--contact-sheet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write a contact sheet for side-by-side images.",
    )
    return parser.parse_args()


def resolve_yolo_dir(dataset: Path) -> Path:
    dataset = dataset.expanduser().resolve()
    if (dataset / "images").is_dir() and (dataset / "labels").is_dir():
        return dataset
    candidate = dataset / "yolo"
    if (candidate / "images").is_dir() and (candidate / "labels").is_dir():
        return candidate
    raise FileNotFoundError(f"Could not find YOLO images/labels under {dataset}")


def default_output_dir(dataset: Path, yolo_dir: Path) -> Path:
    if yolo_dir.name == "yolo":
        return yolo_dir.parent / "model_compare"
    return yolo_dir / "model_compare"


def read_class_names(yolo_dir: Path) -> List[str]:
    classes_path = yolo_dir / "classes.txt"
    if classes_path.is_file():
        return [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    data_yaml = yolo_dir / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Missing classes.txt or data.yaml under {yolo_dir}")

    class_names = []
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if ":" not in stripped or not stripped[0].isdigit():
            continue
        _, value = stripped.split(":", 1)
        class_names.append(value.strip().strip("'\""))
    if not class_names:
        raise ValueError(f"No class names found in {data_yaml}")
    return class_names


def iter_images(images_dir: Path, max_images: int) -> List[Path]:
    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if max_images > 0:
        images = images[:max_images]
    if not images:
        raise FileNotFoundError(f"No images found under {images_dir}")
    return images


def iter_split_images(yolo_dir: Path, split: str, max_images: int) -> List[Path]:
    if split == "all":
        return iter_images(yolo_dir / "images", max_images)

    split_path = yolo_dir / "splits" / f"{split}.txt"
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    images = [Path(line.strip()) for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_images > 0:
        images = images[:max_images]
    missing = [path for path in images if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing image from {split_path}: {missing[0]}")
    if not images:
        raise FileNotFoundError(f"No images listed in split file: {split_path}")
    return images


def class_color(class_id: int) -> Tuple[int, int, int]:
    palette = [
        (238, 75, 43),
        (37, 150, 190),
        (80, 170, 80),
        (245, 166, 35),
        (142, 91, 191),
        (0, 170, 160),
        (230, 95, 140),
        (120, 120, 40),
    ]
    return palette[class_id % len(palette)]


def yolo_to_xyxy(values: List[float], image_width: int, image_height: int) -> List[float]:
    x_center, y_center, width, height = values
    xmin = (x_center - width / 2.0) * image_width
    ymin = (y_center - height / 2.0) * image_height
    xmax = (x_center + width / 2.0) * image_width
    ymax = (y_center + height / 2.0) * image_height
    return [
        max(0.0, min(float(image_width), xmin)),
        max(0.0, min(float(image_height), ymin)),
        max(0.0, min(float(image_width), xmax)),
        max(0.0, min(float(image_height), ymax)),
    ]


def read_ground_truth_boxes(
    label_path: Path,
    class_names: List[str],
    image_width: int,
    image_height: int,
) -> List[Dict]:
    if not label_path.is_file():
        return []

    boxes = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected 5 YOLO fields")
        class_id = int(fields[0])
        values = [float(value) for value in fields[1:]]
        label = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
        boxes.append(
            {
                "class_id": class_id,
                "label": label,
                "confidence": None,
                "xyxy": yolo_to_xyxy(values, image_width, image_height),
            }
        )
    return boxes


def prediction_boxes(result, class_names: List[str]) -> List[Dict]:
    boxes = []
    if result.boxes is None:
        return boxes
    for box in result.boxes:
        class_id = int(box.cls.item())
        label = class_names[class_id] if 0 <= class_id < len(class_names) else str(class_id)
        boxes.append(
            {
                "class_id": class_id,
                "label": label,
                "confidence": float(box.conf.item()),
                "xyxy": [float(value) for value in box.xyxy[0].tolist()],
            }
        )
    return boxes


def draw_boxes(canvas, boxes: Iterable[Dict], empty_text: str) -> None:
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    line_width = max(2, round(min(canvas.size) / 180))
    drew_any = False

    for item in boxes:
        class_id = int(item["class_id"])
        label = str(item["label"])
        confidence = item.get("confidence")
        xmin, ymin, xmax, ymax = [float(value) for value in item["xyxy"]]
        color = class_color(class_id)

        draw.rectangle((xmin, ymin, xmax, ymax), outline=color, width=line_width)
        text = f"{class_id}:{label}" if confidence is None else f"{class_id}:{label} {float(confidence):.2f}"
        text_box = draw.textbbox((0, 0), text, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_x = max(0.0, min(xmin, canvas.size[0] - text_width - 8))
        label_y = ymin - text_height - 6
        if label_y < 0.0:
            label_y = min(ymin + 2, canvas.size[1] - text_height - 6)
        draw.rectangle((label_x, label_y, label_x + text_width + 6, label_y + text_height + 5), fill=color)
        draw.text((label_x + 3, label_y + 2), text, fill=(0, 0, 0), font=font)
        drew_any = True

    if not drew_any:
        text_box = draw.textbbox((0, 0), empty_text, font=font)
        draw.rectangle((8, 8, text_box[2] + 18, text_box[3] + 18), fill=(20, 20, 20))
        draw.text((13, 13), empty_text, fill=(255, 255, 255), font=font)


def draw_header(draw, x: int, y: int, width: int, text: str, fill: Tuple[int, int, int], font) -> None:
    draw.rectangle((x, y, x + width, y + 34), fill=fill)
    text_box = draw.textbbox((0, 0), text, font=font)
    text_height = text_box[3] - text_box[1]
    draw.text((x + 10, y + (34 - text_height) // 2), text, fill=(255, 255, 255), font=font)


def write_side_by_side(
    image_path: Path,
    result,
    labels_dir: Path,
    class_names: List[str],
    pred_dir: Path,
    gt_dir: Path,
    side_dir: Path,
) -> Dict:
    from PIL import Image, ImageDraw, ImageFont

    with Image.open(image_path) as source:
        original = source.convert("RGB")

    pred_boxes = prediction_boxes(result, class_names)
    gt_boxes = read_ground_truth_boxes(labels_dir / f"{image_path.stem}.txt", class_names, original.size[0], original.size[1])

    pred_canvas = original.copy()
    gt_canvas = original.copy()
    draw_boxes(pred_canvas, pred_boxes, "no predictions")
    draw_boxes(gt_canvas, gt_boxes, "no simulation labels")

    pred_path = pred_dir / image_path.name
    gt_path = gt_dir / image_path.name
    side_path = side_dir / image_path.name
    pred_canvas.save(pred_path, quality=92)
    gt_canvas.save(gt_path, quality=92)

    width, height = original.size
    header_height = 34
    combined = Image.new("RGB", (width * 2, height + header_height), (28, 28, 28))
    draw = ImageDraw.Draw(combined)
    font = ImageFont.load_default()
    draw_header(draw, 0, 0, width, "MODEL PREDICTION", (37, 150, 190), font)
    draw_header(draw, width, 0, width, "SIMULATION GT", (80, 170, 80), font)
    combined.paste(pred_canvas, (0, header_height))
    combined.paste(gt_canvas, (width, header_height))
    combined.save(side_path, quality=92)

    return {
        "image": image_path.name,
        "prediction_count": len(pred_boxes),
        "ground_truth_count": len(gt_boxes),
        "predictions": ";".join(f"{item['label']}:{float(item['confidence']):.3f}" for item in pred_boxes),
        "ground_truth": ";".join(str(item["label"]) for item in gt_boxes),
        "comparison_path": str(side_path),
        "prediction_path": str(pred_path),
        "ground_truth_path": str(gt_path),
    }


def contact_sheet_page_name(sheet_path: Path, page_index: int, total_pages: int) -> Path:
    if total_pages == 1:
        return sheet_path
    return sheet_path.with_name(f"{sheet_path.stem}_{page_index + 1:03d}{sheet_path.suffix}")


def write_contact_sheet(image_paths: List[Path], sheet_path: Path, columns: int) -> List[str]:
    if not image_paths:
        return []

    from PIL import Image, ImageDraw, ImageFont, ImageOps

    columns = max(1, min(columns, max(1, CONTACT_SHEET_MAX_DIMENSION // CONTACT_SHEET_THUMB_SIZE[0])))
    row_height = CONTACT_SHEET_THUMB_SIZE[1] + CONTACT_SHEET_LABEL_HEIGHT
    max_rows = max(1, CONTACT_SHEET_MAX_DIMENSION // row_height)
    max_entries = max(1, columns * max_rows)
    total_pages = math.ceil(len(image_paths) / max_entries)
    written = []

    for page_index in range(total_pages):
        page_paths = image_paths[page_index * max_entries : (page_index + 1) * max_entries]
        rows = math.ceil(len(page_paths) / columns)
        sheet = Image.new("RGB", (columns * CONTACT_SHEET_THUMB_SIZE[0], rows * row_height), (242, 242, 242))
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default()

        for index, path in enumerate(page_paths):
            with Image.open(path) as source:
                thumb = ImageOps.contain(source.convert("RGB"), CONTACT_SHEET_THUMB_SIZE)
            col = index % columns
            row = index // columns
            x0 = col * CONTACT_SHEET_THUMB_SIZE[0]
            y0 = row * row_height
            sheet.paste(thumb, (x0 + (CONTACT_SHEET_THUMB_SIZE[0] - thumb.size[0]) // 2, y0))
            draw.text((x0 + 4, y0 + CONTACT_SHEET_THUMB_SIZE[1] + 5), path.name, fill=(20, 20, 20), font=font)

        page_path = contact_sheet_page_name(sheet_path, page_index, total_pages)
        sheet.save(page_path, quality=90)
        written.append(str(page_path))

    return written


def write_summary(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    dataset = Path(args.dataset)
    yolo_dir = resolve_yolo_dir(dataset)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir(dataset, yolo_dir)
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    labels_dir = yolo_dir / "labels"
    pred_dir = output_dir / "predictions"
    gt_dir = output_dir / "ground_truth"
    side_dir = output_dir / "side_by_side"
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    side_dir.mkdir(parents=True, exist_ok=True)

    class_names = read_class_names(yolo_dir)
    image_paths = iter_split_images(yolo_dir, args.split, args.max_images)

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        batch=args.batch,
        verbose=False,
        save=False,
    )
    if len(results) != len(image_paths):
        raise RuntimeError(f"Prediction count mismatch: {len(results)} results for {len(image_paths)} images")

    rows = []
    side_paths = []
    for image_path, result in zip(image_paths, results):
        row = write_side_by_side(image_path, result, labels_dir, class_names, pred_dir, gt_dir, side_dir)
        rows.append(row)
        side_paths.append(Path(row["comparison_path"]))

    write_summary(rows, output_dir / "prediction_summary.csv")
    contact_sheets = []
    if args.contact_sheet:
        contact_sheets = write_contact_sheet(side_paths, output_dir / "comparison_contact_sheet.jpg", args.contact_sheet_columns)

    print(f"YOLO dir: {yolo_dir}")
    print(f"Model: {model_path}")
    print(f"Split: {args.split}")
    print(f"Images: {len(image_paths)}")
    print(f"Output: {output_dir}")
    print(f"Side-by-side: {side_dir}")
    if contact_sheets:
        if len(contact_sheets) == 1:
            print(f"Contact sheet: {contact_sheets[0]}")
        else:
            print(f"Contact sheets: {len(contact_sheets)} files, starting at {contact_sheets[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())