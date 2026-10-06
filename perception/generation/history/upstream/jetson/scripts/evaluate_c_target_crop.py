import argparse
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_face_mobilenetv3 import CLASSES, FaceClsDataset, make_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate C model on the target failure crop and original C val split.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target_image", type=Path, default=None)
    parser.add_argument("--original_data", type=Path, default=Path("datasets/meta_v2_10000_coco_v1_models/c_facecls"))
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--val_fraction", type=float, default=0.10)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output_json", type=Path, default=None)
    parser.add_argument("--skip_val", action="store_true")
    return parser.parse_args()


def choose_device(device_arg):
    if device_arg.lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{int(device_arg.split(',')[0])}")


def find_target(path):
    if path:
        return path
    matches = list(Path.cwd().glob("*cand01_face01_C_output.png"))
    if not matches:
        raise FileNotFoundError("target image not found: *cand01_face01_C_output.png")
    return matches[0]


def imread_any(path):
    data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")
    return image


def load_model(checkpoint, device):
    model, model_name, _ = make_model(len(CLASSES), pretrained=False)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt))
    model = model.to(device).eval()
    return model, tuple(ckpt.get("classes") or CLASSES), model_name


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def predict_image(model, classes, image, device):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    tensor = torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1)[None]).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).detach().cpu().numpy()[0]
    order = np.argsort(-probs)
    return [{"class": classes[i], "prob": float(probs[i])} for i in order]


def target_variants(image):
    variants = {"full_with_preview_label": image}
    if image.shape[0] > 40:
        variants["remove_top_30_resize"] = image[30:, :, :]
    white = image.copy()
    white[: min(32, white.shape[0]), :, :] = 255
    variants["top_label_whitened"] = white
    return variants


def eval_original_val(model, classes, original_data, seed, val_fraction, batch, device):
    dataset = FaceClsDataset(original_data, "train", 128, augment=False)
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    val_len = max(1, int(round(len(dataset) * val_fraction)))
    val_set = Subset(dataset, indices[:val_len])
    loader = DataLoader(val_set, batch_size=batch, shuffle=False, num_workers=0, pin_memory=False)
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    total = 0
    correct = 0
    with torch.no_grad():
        for image, label in loader:
            image = image.to(device)
            label = label.to(device)
            pred = model(image).argmax(1)
            for t, p in zip(label.cpu().numpy(), pred.cpu().numpy()):
                cm[int(t), int(p)] += 1
            correct += int((pred == label).sum().cpu())
            total += int(label.numel())
    per_class = {}
    for idx, class_name in enumerate(classes):
        count = int(cm[idx].sum())
        ok = int(cm[idx, idx])
        per_class[class_name] = {
            "total": count,
            "correct": ok,
            "errors": count - ok,
            "acc": ok / max(count, 1),
        }
    return {
        "acc": correct / max(total, 1),
        "total": total,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def best_row_for_checkpoint(checkpoint):
    results = checkpoint.parent.parent / "results.csv"
    if not results.exists():
        return None
    with results.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get("val_acc", 0.0)))


def main():
    args = parse_args()
    device = choose_device(args.device)
    target_path = find_target(args.target_image)
    model, classes, model_name = load_model(args.checkpoint, device)
    image = imread_any(target_path)

    target = {}
    for name, variant in target_variants(image).items():
        target[name] = predict_image(model, classes, variant, device)[:6]

    result = {
        "checkpoint": str(args.checkpoint),
        "model_name": model_name,
        "classes": list(classes),
        "target_image": str(target_path),
        "target": target,
        "best_row": best_row_for_checkpoint(args.checkpoint),
    }
    if not args.skip_val:
        result["original_val"] = eval_original_val(
            model,
            classes,
            args.original_data,
            args.seed,
            args.val_fraction,
            args.batch,
            device,
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
