from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


CLASS_COLORS = {
    "apple": (40, 40, 230),
    "orange": (0, 150, 255),
    "banana": (0, 220, 255),
    "pineapple": (40, 180, 40),
    "plain": (220, 220, 220),
    "unknown": (210, 120, 210),
}


@dataclass
class FacePrediction:
    class_name: str
    confidence: float
    box_xyxy_full: list[float]
    crop_index: int


@dataclass
class ObjectPrediction:
    class_name: str
    confidence: float
    box_xyxy: list[float]
    crop_box_xyxy: list[int]
    faces: list[FacePrediction]


@dataclass
class ImageReport:
    image: str
    mode: str
    objects: list[ObjectPrediction]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the cube-face unified YOLO model on A1 runtime crops or direct crop images."
    )
    parser.add_argument("--images", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--a1_model",
        type=Path,
        default=Path("runs/meta_v2_a1_objectseg/a1_yolo26s_seg_meta_v2_50000/weights/best.pt"),
    )
    parser.add_argument(
        "--face_model",
        type=Path,
        default=Path("runs/segment/cube_face_unified_yolo26n_seg_v1/weights/best.pt"),
    )
    parser.add_argument("--output", type=Path, default=Path("reports/cube_face_unified_eval/runtime_crop_test"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--a1_imgsz", type=int, default=640)
    parser.add_argument("--face_imgsz", type=int, default=224)
    parser.add_argument("--a1_conf", type=float, default=0.25)
    parser.add_argument("--face_conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--crop_pad", type=float, default=0.18)
    parser.add_argument("--direct_crop_mode", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def expand_box(box: np.ndarray, width: int, height: int, pad: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(1.0, x2 - x1 + 1.0)
    bh = max(1.0, y2 - y1 + 1.0)
    crop_pad = max(bw, bh) * float(pad)
    x1 -= crop_pad
    x2 += crop_pad
    y1 -= crop_pad
    y2 += crop_pad
    return (
        int(max(0, np.floor(x1))),
        int(max(0, np.floor(y1))),
        int(min(width - 1, np.ceil(x2))),
        int(min(height - 1, np.ceil(y2))),
    )


def color_for(name: str) -> tuple[int, int, int]:
    return CLASS_COLORS.get(name, (255, 255, 255))


def draw_label(image: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y0 = max(0, y - th - baseline - 4)
    x0 = max(0, min(x, image.shape[1] - tw - 4))
    cv2.rectangle(image, (x0, y0), (x0 + tw + 4, y0 + th + baseline + 4), color, -1)
    cv2.putText(image, text, (x0 + 2, y0 + th + 2), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def draw_face_result(
    image: np.ndarray,
    result,
    names: dict[int, str],
    offset: tuple[int, int] = (0, 0),
    coord_scale: tuple[float, float] = (1.0, 1.0),
    crop_index: int = 0,
) -> list[FacePrediction]:
    predictions: list[FacePrediction] = []
    if result.boxes is None or len(result.boxes) == 0:
        return predictions

    ox, oy = offset
    sx, sy = float(coord_scale[0]), float(coord_scale[1])
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    confs = result.boxes.conf.detach().cpu().numpy()

    mask_polys = []
    if result.masks is not None and result.masks.xy is not None:
        mask_polys = result.masks.xy

    overlay = image.copy()
    for idx, (box, class_id, conf) in enumerate(zip(boxes, classes, confs)):
        name = str(names.get(int(class_id), class_id))
        color = color_for(name)
        x1, y1, x2, y2 = box
        full_box = [
            float(x1 * sx + ox),
            float(y1 * sy + oy),
            float(x2 * sx + ox),
            float(y2 * sy + oy),
        ]

        if idx < len(mask_polys):
            poly = np.asarray(mask_polys[idx], dtype=np.float32)
            if len(poly) >= 3:
                poly[:, 0] *= sx
                poly[:, 1] *= sy
                poly[:, 0] += ox
                poly[:, 1] += oy
                pts = np.round(poly).astype(np.int32)
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(image, [pts], True, color, 2, cv2.LINE_AA)

        p1 = (int(round(full_box[0])), int(round(full_box[1])))
        p2 = (int(round(full_box[2])), int(round(full_box[3])))
        cv2.rectangle(image, p1, p2, color, 2)
        draw_label(image, f"{name} {conf:.2f}", p1[0], p1[1], color)
        predictions.append(
            FacePrediction(
                class_name=name,
                confidence=float(conf),
                box_xyxy_full=[round(v, 2) for v in full_box],
                crop_index=crop_index,
            )
        )

    cv2.addWeighted(overlay, 0.28, image, 0.72, 0, dst=image)
    return predictions


def make_contact_sheet(images: list[np.ndarray], labels: list[str], cols: int = 4, tile: int = 224) -> np.ndarray:
    if not images:
        return np.zeros((tile, tile, 3), dtype=np.uint8)
    rows = int(np.ceil(len(images) / cols))
    sheet = np.full((rows * (tile + 24), cols * tile, 3), 245, dtype=np.uint8)
    for idx, image in enumerate(images):
        row, col = divmod(idx, cols)
        x = col * tile
        y = row * (tile + 24)
        resized = cv2.resize(image, (tile, tile), interpolation=cv2.INTER_AREA)
        sheet[y + 24 : y + 24 + tile, x : x + tile] = resized
        cv2.putText(
            sheet,
            labels[idx][:28],
            (x + 3, y + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return sheet


def evaluate_a1_crops(args: argparse.Namespace, a1: YOLO, face_model: YOLO, image_path: Path) -> ImageReport:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    height, width = image.shape[:2]
    annotated = image.copy()

    a1_result = a1.predict(
        source=image,
        imgsz=args.a1_imgsz,
        conf=args.a1_conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
    )[0]

    crops: list[np.ndarray] = []
    crop_original_shapes: list[tuple[int, int]] = []
    crop_boxes: list[tuple[int, int, int, int]] = []
    objects: list[ObjectPrediction] = []
    if a1_result.boxes is not None:
        boxes = a1_result.boxes.xyxy.detach().cpu().numpy()
        classes = a1_result.boxes.cls.detach().cpu().numpy().astype(int)
        confs = a1_result.boxes.conf.detach().cpu().numpy()
        for box, cls_id, conf in zip(boxes, classes, confs):
            class_name = str(a1.names.get(int(cls_id), cls_id))
            if int(cls_id) != 0:
                continue
            crop_box = expand_box(box, width, height, args.crop_pad)
            x1, y1, x2, y2 = crop_box
            crop = image[y1 : y2 + 1, x1 : x2 + 1]
            if crop.size == 0:
                continue
            crop_h, crop_w = crop.shape[:2]
            face_input = cv2.resize(
                crop,
                (int(args.face_imgsz), int(args.face_imgsz)),
                interpolation=cv2.INTER_AREA,
            )
            crops.append(face_input)
            crop_original_shapes.append((crop_h, crop_w))
            crop_boxes.append(crop_box)
            objects.append(
                ObjectPrediction(
                    class_name=class_name,
                    confidence=float(conf),
                    box_xyxy=[round(float(v), 2) for v in box],
                    crop_box_xyxy=[int(v) for v in crop_box],
                    faces=[],
                )
            )
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 180, 40), 2)
            draw_label(annotated, f"A1 {class_name} {conf:.2f}", x1, y1, (255, 180, 40))

    crop_previews: list[np.ndarray] = []
    crop_labels: list[str] = []
    if crops:
        face_results = face_model.predict(
            source=crops,
            imgsz=args.face_imgsz,
            conf=args.face_conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )
        for idx, (crop, original_shape, crop_box, result) in enumerate(zip(crops, crop_original_shapes, crop_boxes, face_results)):
            x1, y1, _x2, _y2 = crop_box
            crop_h, crop_w = original_shape
            result_h, result_w = result.orig_shape if getattr(result, "orig_shape", None) else (args.face_imgsz, args.face_imgsz)
            coord_scale = (
                float(crop_w) / max(float(result_w), 1.0),
                float(crop_h) / max(float(result_h), 1.0),
            )
            objects[idx].faces = draw_face_result(
                annotated,
                result,
                face_model.names,
                (x1, y1),
                coord_scale,
                idx,
            )
            crop_annotated = crop.copy()
            draw_face_result(crop_annotated, result, face_model.names, (0, 0), idx)
            crop_previews.append(crop_annotated)
            counts: dict[str, int] = {}
            for face in objects[idx].faces:
                counts[face.class_name] = counts.get(face.class_name, 0) + 1
            crop_labels.append(f"obj{idx} " + ", ".join(f"{k}:{v}" for k, v in counts.items()))

    out_stem = image_path.stem
    cv2.imwrite(str(args.output / f"{out_stem}_unified_overlay.jpg"), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if crop_previews:
        cv2.imwrite(
            str(args.output / f"{out_stem}_cube_crop_sheet.jpg"),
            make_contact_sheet(crop_previews, crop_labels),
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        )
    return ImageReport(image=str(image_path), mode="a1_crop", objects=objects)


def evaluate_direct_crop(args: argparse.Namespace, face_model: YOLO, image_path: Path) -> ImageReport:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    face_input = cv2.resize(
        image,
        (int(args.face_imgsz), int(args.face_imgsz)),
        interpolation=cv2.INTER_AREA,
    )
    result = face_model.predict(
        source=face_input,
        imgsz=args.face_imgsz,
        conf=args.face_conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
    )[0]
    annotated = image.copy()
    result_h, result_w = result.orig_shape if getattr(result, "orig_shape", None) else (args.face_imgsz, args.face_imgsz)
    coord_scale = (
        float(image.shape[1]) / max(float(result_w), 1.0),
        float(image.shape[0]) / max(float(result_h), 1.0),
    )
    faces = draw_face_result(annotated, result, face_model.names, (0, 0), coord_scale, 0)
    out_stem = image_path.stem
    cv2.imwrite(str(args.output / f"{out_stem}_direct_unified.jpg"), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    obj = ObjectPrediction(
        class_name="direct_crop",
        confidence=1.0,
        box_xyxy=[0, 0, image.shape[1] - 1, image.shape[0] - 1],
        crop_box_xyxy=[0, 0, image.shape[1] - 1, image.shape[0] - 1],
        faces=faces,
    )
    return ImageReport(image=str(image_path), mode="direct_crop", objects=[obj])


def main() -> None:
    args = parse_args()
    args.a1_model = resolve(args.a1_model)
    args.face_model = resolve(args.face_model)
    args.output = resolve(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    face_model = YOLO(str(args.face_model), task="segment")
    a1 = None if args.direct_crop_mode else YOLO(str(args.a1_model), task="segment")

    reports: list[ImageReport] = []
    for raw_path in args.images:
        image_path = resolve(raw_path)
        if args.direct_crop_mode:
            reports.append(evaluate_direct_crop(args, face_model, image_path))
        else:
            assert a1 is not None
            reports.append(evaluate_a1_crops(args, a1, face_model, image_path))

    report_data = [asdict(item) for item in reports]
    (args.output / "report.json").write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report_data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
