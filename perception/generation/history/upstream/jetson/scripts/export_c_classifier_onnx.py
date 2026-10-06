from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

JETSON_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = JETSON_ROOT.parent
sys.path.insert(0, str(JETSON_ROOT))

from abc_inference import C_CLASSES, make_c_model, torch_load  # noqa: E402


def resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the ABC C classifier checkpoint to ONNX.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/meta_v2_c_facecls/c_mobilenetv3small_meta_v2_50000_runtimewarp_warmplain_ft/weights/last.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/abc_runtime_optimization/artifacts/c_mobilenetv3small_runtimewarp_warmplain_ft.onnx"),
    )
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve(args.checkpoint)
    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = torch_load(checkpoint_path, torch.device("cpu"))
    classes = tuple(checkpoint.get("classes", C_CLASSES)) if isinstance(checkpoint, dict) else C_CLASSES
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    model = make_c_model(len(classes))
    model.load_state_dict(state)
    model.eval()

    dummy = torch.zeros(1, 3, int(args.imgsz), int(args.imgsz), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=int(args.opset),
        dynamo=False,
    )
    sidecar = output_path.with_suffix(".classes.json")
    sidecar.write_text(json.dumps({"classes": list(classes)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"onnx={output_path}")
    print(f"classes={sidecar}")


if __name__ == "__main__":
    main()
