#!/usr/bin/env python3
"""Create a deterministic, scene-level train/val/test manifest for Meta V2 data."""
import argparse
import hashlib
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    args = parser.parse_args()
    if args.val_ratio <= 0 or args.test_ratio <= 0 or args.val_ratio + args.test_ratio >= 1:
        parser.error("val and test ratios must be positive and sum to less than one")

    meta_dir = args.dataset / "_meta" / "train"
    scenes = sorted(p.stem for p in meta_dir.glob("*.json"))
    if not scenes:
        raise SystemExit(f"No scene metadata found in {meta_dir}")
    for scene in scenes:
        if not any((args.dataset / "images" / "train" / f"{scene}{ext}").is_file()
                   for ext in (".jpg", ".jpeg", ".png")):
            raise SystemExit(f"Missing image for scene {scene}")
        if not (args.dataset / "labels" / "train" / f"{scene}.txt").is_file():
            raise SystemExit(f"Missing segmentation label for scene {scene}")
        try:
            record = json.loads((meta_dir / f"{scene}.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Invalid metadata for scene {scene}: {exc}") from exc
        if not record.get("meta_v2"):
            raise SystemExit(f"Scene {scene} does not contain Meta V2 metadata")

    shuffled = list(scenes)
    random.Random(args.seed).shuffle(shuffled)
    n_val = round(len(scenes) * args.val_ratio)
    n_test = round(len(scenes) * args.test_ratio)
    mapping = {name: "train" for name in scenes}
    for name in shuffled[:n_val]:
        mapping[name] = "val"
    for name in shuffled[n_val:n_val + n_test]:
        mapping[name] = "test"
    payload = {
        "schema": "ddonggae.meta-v2-split.v1",
        "seed": args.seed,
        "ratios": {"train": 1 - args.val_ratio - args.test_ratio,
                   "val": args.val_ratio, "test": args.test_ratio},
        "source_scenes": len(scenes),
        "source_scene_ids_sha256": hashlib.sha256("\n".join(scenes).encode()).hexdigest(),
        "counts": {name: sum(v == name for v in mapping.values())
                   for name in ("train", "val", "test")},
        "scenes": mapping,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["counts"], indent=2))
    print(f"manifest: {args.out}")


if __name__ == "__main__":
    main()
