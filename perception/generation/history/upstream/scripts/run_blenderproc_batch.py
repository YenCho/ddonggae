import argparse
import random
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_scenes", type=int, default=100)
    parser.add_argument("--asset_dir", type=str, default="assets/generated")
    parser.add_argument("--output", type=str, default="datasets/blenderproc_coco")
    parser.add_argument("--views_per_scene", type=int, default=3)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--script", type=str, default="scripts/generate_robot_scene.py")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    # 추천 비율:
    # arena 65%, closeup 25%, negative 10%
    modes = ["tabletop", "closeup", "negative"]
    weights = [0.70, 0.20, 0.10]

    for scene_id in range(args.num_scenes):
        mode = random.choices(modes, weights=weights, k=1)[0]
        seed = random.randint(0, 10**9)

        cmd = [
            "blenderproc", "run", args.script,
            "--asset_dir", args.asset_dir,
            "--output", args.output,
            "--scene_id", str(scene_id),
            "--mode", mode,
            "--views_per_scene", str(args.views_per_scene),
            "--width", str(args.width),
            "--height", str(args.height),
            "--samples", str(args.samples),
            "--seed", str(seed),
        ]

        print(f"[{scene_id + 1}/{args.num_scenes}] mode={mode}, seed={seed}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
