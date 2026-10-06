import argparse
import subprocess
import time
from pathlib import Path


def write_data_yaml(output):
    output = Path(output).resolve()
    dataset_path = output.as_posix()
    yaml_text = f"""path: {dataset_path}
train: images/train
val: images/train
names:
  0: banana
  1: orange
  2: pineapple
  3: apple
  4: cube
  5: octahedron
  6: dodecahedron
  7: icosahedron
"""
    (output / "data.yaml").write_text(yaml_text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--asset_dir", type=str, default="assets/generated")
    parser.add_argument("--background_dir", type=str, default="datasets/backgrounds/coco2017")
    parser.add_argument("--fruit_texture_dir", type=str, default="datasets/fruit_textures/final_fruits36065_original25_fruitseg30_10")
    parser.add_argument("--fruit_texture_aug", choices=["none", "light", "strong"], default="strong")
    parser.add_argument("--output", type=str, default="datasets/yolo_coco_composite")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--script", type=str, default="scripts/generate_yolo_coco_composite.py")
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--min_objects", type=int, default=1)
    parser.add_argument("--max_objects", type=int, default=8)
    parser.add_argument("--scale_min", type=float, default=0.45)
    parser.add_argument("--scale_max", type=float, default=2.35)
    parser.add_argument("--max_covered_ratio", type=float, default=0.80)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    write_data_yaml(args.output)
    start_time = time.time()
    completed_this_run = 0

    for image_id in range(args.num_images):
        image_path = Path(args.output) / "images" / "train" / f"{image_id:06d}.jpg"
        label_path = Path(args.output) / "labels" / "train" / f"{image_id:06d}.txt"
        if args.resume and image_path.exists() and label_path.exists():
            continue

        seed = args.seed + image_id
        cmd = [
            "blenderproc", "run", args.script,
            "--asset_dir", args.asset_dir,
            "--background_dir", args.background_dir,
            "--fruit_texture_dir", args.fruit_texture_dir,
            "--fruit_texture_aug", args.fruit_texture_aug,
            "--output", args.output,
            "--image_id", str(image_id),
            "--width", str(args.width),
            "--height", str(args.height),
            "--samples", str(args.samples),
            "--seed", str(seed),
            "--min_objects", str(args.min_objects),
            "--max_objects", str(args.max_objects),
            "--scale_min", str(args.scale_min),
            "--scale_max", str(args.scale_max),
            "--max_covered_ratio", str(args.max_covered_ratio),
        ]
        done_before = len(list((Path(args.output) / "images" / "train").glob("*.jpg"))) if (Path(args.output) / "images" / "train").exists() else 0
        elapsed = time.time() - start_time
        rate = completed_this_run / elapsed if elapsed > 0 and completed_this_run > 0 else 0
        remaining = max(0, args.num_images - done_before)
        eta_min = remaining / rate / 60 if rate > 0 else None
        if eta_min is None:
            print(f"[{done_before + 1}/{args.num_images}] seed={seed} ETA=calculating", flush=True)
        else:
            print(f"[{done_before + 1}/{args.num_images}] seed={seed} ETA={eta_min:.1f} min", flush=True)
        subprocess.run(cmd, check=True)
        completed_this_run += 1


if __name__ == "__main__":
    main()
