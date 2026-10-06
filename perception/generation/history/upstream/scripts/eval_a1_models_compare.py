"""Compare two (or more) A1 whole-object segmentation models on the held-out
arena A1 eval set built by scripts/build_arena_a1_eval.py.

For each model it runs the equivalent of `yolo segment val` (ultralytics
YOLO(...).val() with task='segment') on datasets/arena_v3_a1_eval_v1/data.yaml
and prints a comparison table of box mAP50 / box mAP50-95 / mask mAP50
(+ mask mAP50-95) per model.

Run AFTER A1 training finishes and the eval set exists. Example:
  C:\\Users\\user\\anaconda3\\envs\\ai_robotics\\python.exe ^
      scripts\\eval_a1_models_compare.py ^
      --models current=runs/.../a1/weights/best.pt lowlr=runs/.../a1_lowlr/weights/best.pt ^
      --device 0
"""
import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "datasets" / "arena_v3_a1_eval_v1" / "data.yaml"
DEFAULT_PROJECT = REPO_ROOT / "reports" / "a1_eval_compare_v1"


def parse_models(items: list) -> list:
    """Parse 'name=path' (or bare 'path') CLI entries into (name, path) pairs."""
    models = []
    for item in items:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            name, path = Path(item).stem, item
        name, path = name.strip(), path.strip()
        if not path:
            raise ValueError(f"Empty model path in --models entry {item!r}")
        models.append((name, path))
    return models


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare A1 segmentation models on the arena A1 eval set.")
    p.add_argument("--models", nargs="+", required=True, metavar="name=path",
                   help="One or more models as name=path (repeatable).")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--device", default="cpu", help="cpu (default) or a CUDA id like 0.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--split", default="val")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--project", type=Path, default=DEFAULT_PROJECT,
                   help="Where val() writes its per-model run dir (kept out of runs/).")
    return p.parse_args()


def evaluate_model(name: str, path: str, args: argparse.Namespace) -> dict:
    from ultralytics import YOLO
    model = YOLO(str(path), task="segment")
    metrics = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        split=args.split,
        batch=args.batch,
        project=str(args.project),
        name=name,
        exist_ok=True,
        plots=False,
        save_json=False,
        verbose=False,
    )
    return {
        "name": name,
        "path": path,
        "box_map50": float(metrics.box.map50),
        "box_map50_95": float(metrics.box.map),
        "mask_map50": float(metrics.seg.map50),
        "mask_map50_95": float(metrics.seg.map),
    }


def print_table(rows: list) -> None:
    headers = ["model", "box mAP50", "box mAP50-95", "mask mAP50", "mask mAP50-95"]
    name_w = max(len(headers[0]), *(len(r["name"]) for r in rows))
    widths = [name_w, 11, 13, 11, 13]

    def fmt_row(cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))

    print()
    print(fmt_row(headers))
    print(fmt_row(["-" * w for w in widths]))
    for r in rows:
        print(fmt_row([
            r["name"],
            f"{r['box_map50']:.4f}",
            f"{r['box_map50_95']:.4f}",
            f"{r['mask_map50']:.4f}",
            f"{r['mask_map50_95']:.4f}",
        ]))
    if len(rows) == 2:
        a, b = rows
        print(fmt_row([
            f"delta({b['name']}-{a['name']})",
            f"{b['box_map50'] - a['box_map50']:+.4f}",
            f"{b['box_map50_95'] - a['box_map50_95']:+.4f}",
            f"{b['mask_map50'] - a['mask_map50']:+.4f}",
            f"{b['mask_map50_95'] - a['mask_map50_95']:+.4f}",
        ]))
    print()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(
            f"Eval data.yaml not found: {args.data}. Build it first with "
            "scripts/build_arena_a1_eval.py.")
    models = parse_models(args.models)
    rows = [evaluate_model(name, path, args) for name, path in models]
    print_table(rows)

    args.project.mkdir(parents=True, exist_ok=True)
    out_json = args.project / "a1_eval_comparison.json"
    out_json.write_text(json.dumps({"data": str(args.data), "results": rows},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
