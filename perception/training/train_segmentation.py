"""Portable A1 / face YOLO training entry point for the integrated recipe."""
import argparse
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

CLASSES = {
    "a1": ["cube_like_object", "octahedron", "dodecahedron", "icosahedron"],
    "face": ["apple", "orange", "banana", "pineapple", "plain"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=CLASSES, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--init", default="yolo26s-seg.yaml")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hsv-h", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()
    import yaml
    from ultralytics import YOLO
    data = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    names = data["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    if names != CLASSES[args.task]:
        raise RuntimeError(f"Class order mismatch: {names}; expected {CLASSES[args.task]}")
    if data.get("val") == data.get("train"):
        raise RuntimeError("Validation must not be the training split")
    run = args.project.resolve() / args.name
    if run.exists():
        raise RuntimeError(f"Run already exists; select a new --name: {run}")
    init = Path(args.init).resolve().as_posix() if Path(args.init).exists() else args.init
    package_versions = {}
    for package in ("ultralytics", "torch", "torchvision", "numpy", "opencv-python"):
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = "not-installed"
    recipe = {
        **vars(args),
        "task_classes": CLASSES[args.task],
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions,
        "resolved_init": init,
        "imgsz": 640 if args.task == "a1" else 224,
        "pretrained": False,
    }
    run.mkdir(parents=True, exist_ok=False)
    (run / "recipe.json").write_text(
        json.dumps(recipe, default=str, indent=2) + "\n", encoding="utf-8"
    )
    model = YOLO(init)
    model.train(
        data=str(args.data.resolve()), epochs=args.epochs,
        imgsz=640 if args.task == "a1" else 224,
        batch=args.batch, device=args.device, workers=args.workers,
        optimizer="AdamW", lr0=args.lr, lrf=0.05, cos_lr=True,
        seed=args.seed, deterministic=True, pretrained=False,
        cache=False, amp=args.device != "cpu", warmup_epochs=1,
        mosaic=0.0, mixup=0.0, copy_paste=0.0, close_mosaic=0,
        hsv_h=args.hsv_h, project=str(args.project.resolve()), name=args.name,
        patience=args.patience, exist_ok=True,
    )


if __name__ == "__main__":
    main()
