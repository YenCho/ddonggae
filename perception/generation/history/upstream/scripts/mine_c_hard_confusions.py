import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_face_mobilenetv3 import CLASSES, make_model  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Mine C classifier hard confusions from split/class image folders.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--true_class", choices=CLASSES, required=True)
    parser.add_argument("--target_class", choices=CLASSES, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--top_k", type=int, default=80)
    parser.add_argument("--copy_top_k", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--device", default="0")
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def choose_device(device_arg):
    if device_arg.lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{int(device_arg.split(',')[0])}")


def load_model(checkpoint, device):
    model, _, _ = make_model(len(CLASSES), pretrained=False)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt))
    return model.to(device).eval()


def list_images(root):
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def imread_any(path):
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    return image


def predict(model, path, imgsz, device):
    image = imread_any(path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (imgsz, imgsz), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    tensor = torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1)[None]).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).detach().cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return probs, CLASSES[pred_idx], float(probs[pred_idx])


def make_contact_sheet(records, output_path, cols=8, thumb=128):
    if not records:
        return None
    rows = int(np.ceil(len(records) / cols))
    sheet = np.full((rows * (thumb + 28), cols * thumb, 3), 245, dtype=np.uint8)
    for idx, rec in enumerate(records):
        image = imread_any(Path(rec["path"]))
        image = cv2.resize(image, (thumb, thumb), interpolation=cv2.INTER_AREA)
        x = (idx % cols) * thumb
        y = (idx // cols) * (thumb + 28)
        sheet[y : y + thumb, x : x + thumb] = image
        label = f"{rec['pred']} {rec['pred_prob']:.2f} tgt {rec['target_prob']:.2f}"
        cv2.rectangle(sheet, (x, y), (x + thumb, y + 18), (0, 0, 0), -1)
        cv2.putText(sheet, label[:24], (x + 3, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)
    return output_path


def main():
    args = parse_args()
    if args.reset and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    mined_dir = args.output_root / f"{args.true_class}_toward_{args.target_class}"
    mined_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    model = load_model(args.checkpoint, device)
    true_idx = CLASSES.index(args.true_class)
    target_idx = CLASSES.index(args.target_class)
    src_dir = args.data_root / args.split / args.true_class
    paths = list_images(src_dir)

    records = []
    top1_target = []
    for path in tqdm(paths, desc=f"mine {args.true_class}->{args.target_class}", unit="img"):
        probs, pred, pred_prob = predict(model, path, args.imgsz, device)
        rec = {
            "path": str(path),
            "true": args.true_class,
            "pred": pred,
            "pred_prob": pred_prob,
            "true_prob": float(probs[true_idx]),
            "target_prob": float(probs[target_idx]),
            "probs": {name: float(probs[idx]) for idx, name in enumerate(CLASSES)},
        }
        records.append(rec)
        if pred == args.target_class:
            top1_target.append(rec)

    records.sort(key=lambda item: item["target_prob"], reverse=True)
    top = records[: args.top_k]
    for idx, rec in enumerate(top[: args.copy_top_k]):
        src = Path(rec["path"])
        dst = mined_dir / f"{idx:04d}_target{rec['target_prob']:.4f}_pred-{rec['pred']}_{src.name}"
        shutil.copy2(src, dst)
        rec["copied_to"] = str(dst)

    sheet_path = make_contact_sheet(top[: args.copy_top_k], args.output_root / "contact_sheet.jpg")
    report = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "split": args.split,
        "true_class": args.true_class,
        "target_class": args.target_class,
        "num_images": len(paths),
        "top1_target_count": len(top1_target),
        "top1_target_rate": len(top1_target) / max(len(paths), 1),
        "top_k": top,
        "contact_sheet": str(sheet_path) if sheet_path else None,
    }
    report_path = args.output_root / "hard_confusions.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["num_images", "top1_target_count", "top1_target_rate", "contact_sheet"]}, ensure_ascii=False, indent=2))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
