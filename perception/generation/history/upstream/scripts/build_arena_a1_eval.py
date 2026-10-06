"""Build a HELD-OUT generalization eval set for the A1 whole-object detector
("Cube Detector") from the validated mimic-arena renderer.

It renders N arena scenes with scripts/render_mimic_arena_scene.py (each scene
places 28 objects: 16 cubes + 12 non-cube polyhedra), then converts the new
per-object A1 metadata ("objects" key: A1 class + convex-hull silhouette) into a
YOLO-seg dataset under datasets/arena_v3_a1_eval_v1/{images,labels}/val plus a
data.yaml whose 4 classes are IN THE SAME ORDER as the A1 data.yaml.

Example (run AFTER A1 training finishes; CPU render, GPU not required):
  C:\\Users\\user\\anaconda3\\envs\\ai_robotics\\python.exe ^
      scripts\\build_arena_a1_eval.py --num_images 60
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RENDERER = REPO_ROOT / "scripts" / "render_mimic_arena_scene.py"
DEFAULT_A1_DATA_YAML = (
    REPO_ROOT / "datasets" / "meta_v2_50000_coco_texture_v1_models"
    / "a1_objectseg" / "data.yaml"
)
DEFAULT_OUTPUT = REPO_ROOT / "datasets" / "arena_v3_a1_eval_v1"


def resolve_blenderproc() -> str:
    """Locate the blenderproc launcher (same strategy as run_yolo_parallel.py)."""
    found = shutil.which("blenderproc")
    if found:
        return found
    scripts_dir = Path(sys.executable).resolve().parent
    candidates = [
        scripts_dir / "blenderproc.exe",
        scripts_dir / "blenderproc",
        scripts_dir / "Scripts" / "blenderproc.exe",
        scripts_dir / "Scripts" / "blenderproc",
        scripts_dir.parent / "Scripts" / "blenderproc.exe",
        scripts_dir.parent / "Scripts" / "blenderproc",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        "Could not find blenderproc. Activate the ai_robotics environment or add "
        f"its Scripts directory to PATH. sys.executable={sys.executable!r}"
    )


def read_a1_class_names(data_yaml: Path) -> list:
    """Ordered A1 class names from the A1 data.yaml `names:` mapping. Reused so
    this eval set is guaranteed to match A1's class order exactly."""
    text = data_yaml.read_text(encoding="utf-8")
    try:
        import yaml  # ships with ultralytics
        names = yaml.safe_load(text).get("names")
        if isinstance(names, dict):
            return [names[k] for k in sorted(names, key=int)]
        if isinstance(names, list):
            return list(names)
    except Exception:
        pass
    # Manual fallback: collect "  <id>: <name>" lines under the names: block.
    found = {}
    in_names = False
    for line in text.splitlines():
        if re.match(r"^\s*names\s*:", line):
            in_names = True
            continue
        if in_names:
            m = re.match(r"^\s+(\d+)\s*:\s*(.+?)\s*$", line)
            if m:
                found[int(m.group(1))] = m.group(2).strip().strip("'\"")
            elif line.strip() and not line[0].isspace():
                break
    if not found:
        raise ValueError(f"No class names parsed from {data_yaml}")
    return [found[k] for k in sorted(found)]


def run_render(bp_exe: str, staging: Path, args: argparse.Namespace) -> None:
    """Invoke the mimic-arena renderer (CPU-only) into the staging directory."""
    cmd = [
        bp_exe, "run", str(RENDERER),
        "--output", str(staging.resolve()),
        "--num_images", str(args.num_images),
        "--frames_per_layout", str(args.frames_per_layout),
        "--resolution", str(args.resolution),
        "--samples", str(args.samples),
        "--seed", str(args.seed),
    ]
    for flag, value in (
        ("--asset_dir", args.asset_dir),
        ("--texture_root", args.texture_root),
        ("--background_dir", args.background_dir),
        ("--floor_photo", args.floor_photo),
    ):
        if value is not None:
            cmd += [flag, str(Path(value).resolve())]
    print("[build_arena_a1_eval] rendering:\n  " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def silhouette_to_label(class_id: int, silhouette: list, resolution: int) -> str:
    """One YOLO-seg polygon line: '<class> x1 y1 x2 y2 ...' normalized [0,1]."""
    coords = []
    for x, y in silhouette:
        nx = min(max(x / resolution, 0.0), 1.0)
        ny = min(max(y / resolution, 0.0), 1.0)
        coords.append(f"{nx:.6f}")
        coords.append(f"{ny:.6f}")
    return f"{class_id} " + " ".join(coords)


def write_data_yaml(output: Path, class_names: list) -> None:
    lines = [
        f"path: {output.resolve().as_posix()}",
        "train: images/val",
        "val: images/val",
        "names:",
    ]
    lines += [f"  {i}: {name}" for i, name in enumerate(class_names)]
    (output / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_dataset(staging: Path, output: Path, class_names: list,
                  append: bool) -> dict:
    class_to_id = {name: i for i, name in enumerate(class_names)}
    img_out = output / "images" / "val"
    lbl_out = output / "labels" / "val"
    if not append:
        for d in (img_out, lbl_out):
            if d.exists():
                shutil.rmtree(d)
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    meta_paths = sorted((staging / "metadata").glob("*.json"))
    if not meta_paths:
        raise FileNotFoundError(f"No renderer metadata under {staging / 'metadata'}")

    n_images = 0
    n_objects = 0
    per_class = {name: 0 for name in class_names}
    for meta_path in meta_paths:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        image_name = meta.get("image", meta_path.stem + ".jpg")
        src_img = staging / image_name
        if not src_img.exists():
            print(f"[build_arena_a1_eval] WARN missing image {src_img}", flush=True)
            continue
        resolution = meta.get("resolution", 640)
        lines = []
        for obj in meta.get("objects", []):
            if not obj.get("on_screen"):
                continue
            cls = obj.get("a1_class")
            silhouette = obj.get("silhouette", [])
            if cls not in class_to_id or len(silhouette) < 3:
                continue
            lines.append(silhouette_to_label(class_to_id[cls], silhouette, resolution))
            per_class[cls] += 1
            n_objects += 1
        stem = src_img.stem
        shutil.copyfile(src_img, img_out / f"{stem}.jpg")
        (lbl_out / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        n_images += 1

    write_data_yaml(output, class_names)
    return {"images": n_images, "objects": n_objects, "per_class": per_class}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the held-out arena A1 eval set (YOLO-seg).")
    p.add_argument("--num_images", type=int, default=60)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--a1_data_yaml", type=Path, default=DEFAULT_A1_DATA_YAML)
    p.add_argument("--render_dir", type=Path, default=None,
                   help="Renderer staging dir (default: <output>/_render).")
    p.add_argument("--skip_render", action="store_true",
                   help="Reuse an existing --render_dir instead of rendering.")
    p.add_argument("--frames_per_layout", type=int, default=3)
    p.add_argument("--resolution", type=int, default=640)
    p.add_argument("--samples", type=int, default=24)
    p.add_argument("--seed", type=int, default=20260709)
    p.add_argument("--asset_dir", type=Path, default=None)
    p.add_argument("--texture_root", type=Path, default=None)
    p.add_argument("--background_dir", type=Path, default=None)
    p.add_argument("--floor_photo", type=Path, default=None)
    p.add_argument("--append", action="store_true",
                   help="Keep existing val images/labels instead of clearing them.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    class_names = read_a1_class_names(args.a1_data_yaml)
    print(f"[build_arena_a1_eval] A1 classes: {class_names}", flush=True)

    output = args.output
    staging = args.render_dir if args.render_dir is not None else output / "_render"
    output.mkdir(parents=True, exist_ok=True)

    if not args.skip_render:
        run_render(resolve_blenderproc(), staging, args)
    elif not (staging / "metadata").exists():
        raise FileNotFoundError(
            f"--skip_render set but no metadata under {staging / 'metadata'}")

    stats = build_dataset(staging, output, class_names, args.append)
    stats.update({"output": str(output.resolve()),
                  "data_yaml": str((output / "data.yaml").resolve())})
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
