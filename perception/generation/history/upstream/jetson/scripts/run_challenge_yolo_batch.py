import argparse
import subprocess
from pathlib import Path


def write_data_yaml(output):
    yaml_text = """path: .
train: images/train
val: images/train
names:
  0: plain_cube
  1: octahedron
  2: dodecahedron
  3: icosahedron
"""
    Path(output, "data.yaml").write_text(yaml_text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--asset_dir", type=str, default="assets/generated")
    parser.add_argument("--output", type=str, default="datasets/yolo_challenge_arena")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--scale_min", type=float, default=0.70)
    parser.add_argument("--scale_max", type=float, default=1.45)
    parser.add_argument("--partial_set", action="store_true")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    write_data_yaml(args.output)

    for image_id in range(args.num_images):
        cmd = [
            "blenderproc", "run", "scripts/generate_challenge_yolo_scene.py",
            "--asset_dir", args.asset_dir,
            "--output", args.output,
            "--image_id", str(image_id),
            "--width", str(args.width),
            "--height", str(args.height),
            "--samples", str(args.samples),
            "--seed", str(args.seed + image_id),
            "--scale_min", str(args.scale_min),
            "--scale_max", str(args.scale_max),
        ]
        if args.partial_set:
            cmd.append("--partial_set")
        print(f"[{image_id + 1}/{args.num_images}] seed={args.seed + image_id}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
