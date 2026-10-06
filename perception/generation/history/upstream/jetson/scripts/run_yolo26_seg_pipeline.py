import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path


NAMES = [
    "banana",
    "orange",
    "pineapple",
    "apple",
    "cube",
    "octahedron",
    "dodecahedron",
    "icosahedron",
]


def run(cmd, dry_run=False):
    print("\n$ " + " ".join(str(part) for part in cmd), flush=True)
    if dry_run:
        return
    subprocess.run([str(part) for part in cmd], check=True)


def ensure_inside_workspace(path):
    cwd = Path.cwd().resolve()
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(cwd)
    except ValueError as exc:
        raise RuntimeError(f"Refusing to manage path outside workspace: {resolved}") from exc
    return resolved


def reset_dir(path, dry_run=False):
    path = ensure_inside_workspace(path)
    if dry_run:
        print(f"[dry-run] reset {path}")
        return
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_data_yaml(dataset):
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(NAMES))
    text = (
        f"path: {dataset.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        f"{names}\n"
    )
    (dataset / "data.yaml").write_text(text, encoding="utf-8")


def expected_counts(args):
    explicit_counts = [args.train_count, args.val_count, args.test_count]
    if any(value is not None for value in explicit_counts):
        if any(value is None for value in explicit_counts):
            raise RuntimeError("Set all of --train_count/--val_count/--test_count, or set none of them.")
        counts = {
            "train": args.train_count,
            "val": args.val_count,
            "test": args.test_count,
        }
    else:
        if args.train_ratio < 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
            raise RuntimeError("--train_ratio and --val_ratio must be non-negative and sum to less than 1.")
        train = int(round(args.total * args.train_ratio))
        val = int(round(args.total * args.val_ratio))
        test = args.total - train - val
        counts = {"train": train, "val": val, "test": test}

    if sum(counts.values()) != args.total:
        raise RuntimeError(
            f"train/val/test counts must sum to total: "
            f"{counts['train']}+{counts['val']}+{counts['test']}!={args.total}"
        )
    return counts


def copy_optional(src, dst):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def split_generated_dataset(dataset, raw_dir, counts, seed, dry_run=False):
    image_dir = raw_dir / "images" / "train"
    images = sorted(image_dir.glob("*.jpg"))
    total = sum(counts.values())
    if dry_run and len(images) < total:
        print(f"[dry-run] would split {total} generated images from {image_dir}")
        for split, count in counts.items():
            print(f"{split}: {count} images")
        return
    if len(images) < total:
        raise RuntimeError(f"Need {total} generated images, found {len(images)} in {image_dir}")

    rng = random.Random(seed)
    stems = [image.stem for image in images]
    rng.shuffle(stems)

    for group in ["images", "labels", "_meta", "ideal_visibility_debug", "visibility_preview"]:
        for split in counts:
            reset_dir(dataset / group / split, dry_run=dry_run)

    offset = 0
    for split, count in counts.items():
        split_stems = sorted(stems[offset : offset + count])
        offset += count
        print(f"{split}: {len(split_stems)} images", flush=True)
        if dry_run:
            continue
        for stem in split_stems:
            copy_optional(raw_dir / "images" / "train" / f"{stem}.jpg", dataset / "images" / split / f"{stem}.jpg")
            copy_optional(raw_dir / "labels" / "train" / f"{stem}.txt", dataset / "labels" / split / f"{stem}.txt")
            copy_optional(raw_dir / "_meta" / "train" / f"{stem}.json", dataset / "_meta" / split / f"{stem}.json")
            copy_optional(
                raw_dir / "ideal_visibility_debug" / "train" / f"{stem}_ideal_visibility.jpg",
                dataset / "ideal_visibility_debug" / split / f"{stem}_ideal_visibility.jpg",
            )
    if not dry_run:
        write_data_yaml(dataset)


def preview_count_for_split(args, split, count):
    if args.preview_count < 0:
        return count
    return min(args.preview_count, count)


def generate_dataset(args, raw_dir):
    cmd = [
        sys.executable,
        "scripts/run_yolo_parallel.py",
        "--output",
        raw_dir,
        "--fruit_texture_dir",
        args.fruit_texture_dir,
        "--background_dir",
        args.background_dir,
        "--arena_background_ratio",
        args.arena_background_ratio,
        "--arena_floor_material",
        args.arena_floor_material,
        "--arena_wall_material",
        args.arena_wall_material,
        "--arena_full_background_ratio",
        args.arena_full_background_ratio,
        "--wall_contact_ratio",
        args.wall_contact_ratio,
        "--corner_scene_ratio",
        args.corner_scene_ratio,
        "--motion_blur_hard_negative_ratio",
        args.motion_blur_hard_negative_ratio,
        "--plain_cube_hard_negative_ratio",
        args.plain_cube_hard_negative_ratio,
        "--num_images",
        args.total,
        "--workers",
        args.gen_workers,
        "--worker_start_delay",
        args.worker_start_delay,
        "--width",
        args.imgsz,
        "--height",
        args.imgsz,
        "--samples",
        args.samples,
        "--cpu_threads",
        args.cpu_threads,
        "--min_objects",
        args.min_objects,
        "--max_objects",
        args.max_objects,
        "--scale_min",
        args.scale_min,
        "--scale_max",
        args.scale_max,
        "--max_covered_ratio",
        args.max_covered_ratio,
        "--min_object_visible_ratio",
        args.min_object_visible_ratio,
        "--min_projected_area",
        args.min_projected_area,
        "--single_object_min_projected_area",
        args.single_object_min_projected_area,
        "--min_fruit_visible_ratio",
        args.min_fruit_visible_ratio,
        "--min_fruit_face_pixels",
        args.min_fruit_face_pixels,
        "--single_object_min_fruit_face_pixels",
        args.single_object_min_fruit_face_pixels,
        "--min_fruit_face_side",
        args.min_fruit_face_side,
        "--label_format",
        "segment",
        "--seg_contour_mode",
        "largest",
        "--seg_contour_epsilon_ratio",
        args.seg_contour_epsilon_ratio,
        "--negative_ratio",
        args.negative_ratio,
        "--fruit_visibility_tier",
        "mixed",
        "--fruit_visibility_easy_weight",
        args.fruit_visibility_easy_weight,
        "--fruit_visibility_mid_weight",
        args.fruit_visibility_mid_weight,
        "--fruit_visibility_hard_weight",
        args.fruit_visibility_hard_weight,
        "--fruit_class_weight_scale",
        args.fruit_class_weight_scale,
        "--hard_min_fruit_visible_ratio",
        args.hard_min_fruit_visible_ratio,
        "--hard_min_fruit_face_pixels",
        args.hard_min_fruit_face_pixels,
        "--lighting_mode",
        args.lighting_mode,
        "--fruit_texture_aug",
        args.fruit_texture_aug,
        "--fruit_texture_layout",
        args.fruit_texture_layout,
        "--fruit_texture_collage_prob",
        args.fruit_texture_collage_prob,
        "--ideal_visibility",
        "--ideal_visibility_max_attempts",
        args.ideal_visibility_max_attempts,
        "--seed",
        args.seed,
    ]
    if args.single_fruit_class_per_image:
        cmd.append("--single_fruit_class_per_image")
    else:
        cmd.append("--allow_mixed_fruit_classes_per_image")
    if args.single_fruit_texture_per_cube:
        cmd.append("--single_fruit_texture_per_cube")
    else:
        cmd.append("--allow_multiple_fruit_textures_per_cube")
    if args.canonical_fruit_textures:
        cmd.append("--canonical_fruit_textures")
    if args.arena_booster_mode:
        cmd.append("--arena_booster_mode")
    if args.robot_camera_view:
        cmd.append("--robot_camera_view")
    if args.apple_orange_boost:
        cmd.append("--apple_orange_boost")
    if args.email_to:
        cmd.extend(
            [
                "--email_to",
                args.email_to,
                "--email_every",
                args.email_every,
                "--email_config",
                args.email_config,
            ]
        )
    if args.resume_generate:
        cmd.append("--resume")
    if args.save_ideal_visibility_debug:
        cmd.append("--ideal_visibility_debug")
    run(cmd, dry_run=args.dry_run)


def make_previews(args, dataset, counts):
    for split, count in counts.items():
        preview_count = preview_count_for_split(args, split, count)
        if preview_count <= 0:
            continue
        run(
            [
                sys.executable,
                "scripts/make_yolo_bbox_preview.py",
                "--dataset",
                dataset,
                "--split",
                split,
                "--count",
                preview_count,
                "--debug_visibility",
                "--sheet_count",
                args.preview_sheet_count,
                "--sheet_columns",
                args.preview_sheet_columns,
                "--sheet_thumb_size",
                args.preview_sheet_thumb_size,
            ],
            dry_run=args.dry_run,
        )


def audit_generated_dataset(args, raw_dir):
    if args.skip_audit:
        return
    cmd = [
        sys.executable,
        "scripts/audit_fruit_generation.py",
        "--dataset",
        raw_dir,
        "--require_face_texture_meta",
    ]
    if args.single_fruit_class_per_image:
        cmd.append("--expect_single_fruit_class_per_image")
    else:
        cmd.append("--allow_mixed_fruit_classes_per_image")
    if args.single_fruit_texture_per_cube:
        cmd.append("--expect_single_texture_per_cube")
    else:
        cmd.append("--allow_multiple_textures_per_cube")
    run(cmd, dry_run=args.dry_run)


def segment_run_dir(project, name):
    project_path = Path(project)
    if project_path.is_absolute():
        return (project_path / name).resolve()
    return (Path("runs") / "segment" / project_path / name).resolve()


def start_training_email_monitor(args, dataset):
    if not args.email_to or args.dry_run or args.disable_training_monitor:
        return

    run_dir = segment_run_dir(args.project, args.name)
    script = Path("scripts/send_yolo_50000_status_email_loop.ps1").resolve()
    cmd = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        "-Workspace",
        str(Path.cwd().resolve()),
        "-Dataset",
        str(dataset.resolve()),
        "-RunDir",
        str(run_dir),
        "-To",
        args.email_to,
        "-Subject",
        f"[30min check] YOLO26n-seg training status {args.name}",
        "-TaskLabel",
        f"YOLO26n-seg training {args.name}",
        "-Epochs",
        str(args.epochs),
        "-Config",
        args.email_config,
        "-IntervalSeconds",
        str(args.training_monitor_interval),
        "-ProcessPattern",
        "yolo segment train",
        args.name,
    ]
    print("\n$ " + " ".join(cmd), flush=True)
    creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    subprocess.Popen(cmd, cwd=Path.cwd(), creationflags=creationflags)


def train_model(args, dataset):
    start_training_email_monitor(args, dataset)
    cmd = [
        "yolo",
        "segment",
        "train",
        f"model={args.model}",
        f"data={(dataset / 'data.yaml').resolve().as_posix()}",
        f"epochs={args.epochs}",
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
        f"device={args.device}",
        f"workers={args.train_workers}",
        f"cache={args.cache}",
        f"amp={args.amp}",
        f"deterministic={args.deterministic}",
        f"patience={args.patience}",
        f"project={args.project}",
        f"name={args.name}",
        "exist_ok=True",
    ]
    if args.train_low_aug:
        cmd.extend([
            "hsv_h=0.003",
            "hsv_s=0.15",
            "hsv_v=0.12",
            "mosaic=0",
            "close_mosaic=0",
            "erasing=0",
            "auto_augment=None",
        ])
    run(cmd, dry_run=args.dry_run)


def predict_test(args, dataset):
    weights = segment_run_dir(args.project, args.name) / "weights" / "best.pt"
    run(
        [
            "yolo",
            "segment",
            "predict",
            f"model={weights.as_posix()}",
            f"source={(dataset / 'images' / 'test').resolve().as_posix()}",
            f"imgsz={args.imgsz}",
            f"conf={args.conf}",
            "save=True",
            "save_txt=True",
            f"project={(dataset / 'predictions').as_posix()}",
            "name=test",
            "exist_ok=True",
        ],
        dry_run=args.dry_run,
    )


def title_bar(width, text):
    import cv2
    import numpy as np

    bar = np.full((34, width, 3), 245, dtype=np.uint8)
    cv2.putText(bar, text, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 1, cv2.LINE_AA)
    return bar


def pad_to_height(image, height):
    import numpy as np

    if image.shape[0] == height:
        return image
    canvas = np.full((height, image.shape[1], 3), 245, dtype=np.uint8)
    canvas[: image.shape[0], : image.shape[1]] = image
    return canvas


def make_side_by_side(args, dataset):
    import cv2
    import numpy as np

    gt_dir = dataset / "visibility_preview" / "test"
    pred_dir = dataset / "predictions" / "test"
    out_dir = dataset / "side_by_side" / "test"
    reset_dir(out_dir, dry_run=args.dry_run)
    if args.dry_run:
        return

    test_images = sorted((dataset / "images" / "test").glob("*.jpg"))[: args.side_by_side_count]
    made = 0
    for image_path in test_images:
        stem = image_path.stem
        gt = cv2.imread(str(gt_dir / f"{stem}_visibility.jpg"), cv2.IMREAD_COLOR)
        pred = cv2.imread(str(pred_dir / f"{stem}.jpg"), cv2.IMREAD_COLOR)
        if gt is None or pred is None:
            continue
        gt = np.vstack([title_bar(gt.shape[1], "ground truth / visibility debug"), gt])
        pred = np.vstack([title_bar(pred.shape[1], "YOLO26n-seg prediction"), pred])
        height = max(gt.shape[0], pred.shape[0])
        combined = np.hstack([pad_to_height(gt, height), pad_to_height(pred, height)])
        cv2.imwrite(str(out_dir / f"{stem}_side_by_side.jpg"), combined, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        made += 1
    print(f"side-by-side images: {made}")
    print(f"side-by-side dir: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate a split YOLO segmentation dataset, train, predict, and compare.")
    parser.add_argument("--dataset", type=str, default="datasets/yolo26_seg_ideal_debug")
    parser.add_argument("--raw_dir", type=str, default=None)
    parser.add_argument("--total", type=int, default=1000)
    parser.add_argument("--train_count", type=int, default=None)
    parser.add_argument("--val_count", type=int, default=None)
    parser.add_argument("--test_count", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=0.80)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260517)

    parser.add_argument("--gen_workers", type=int, default=4)
    parser.add_argument("--worker_start_delay", type=float, default=0.0)
    parser.add_argument("--cpu_threads", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--min_objects", type=int, default=1)
    parser.add_argument("--max_objects", type=int, default=4)
    parser.add_argument("--scale_min", type=float, default=0.40)
    parser.add_argument("--scale_max", type=float, default=2.45)
    parser.add_argument("--max_covered_ratio", type=float, default=0.80)
    parser.add_argument("--min_object_visible_ratio", type=float, default=0.10)
    parser.add_argument("--min_projected_area", type=int, default=900)
    parser.add_argument("--single_object_min_projected_area", type=int, default=2600)
    parser.add_argument("--min_fruit_visible_ratio", type=float, default=0.10)
    parser.add_argument("--min_fruit_face_pixels", type=int, default=300)
    parser.add_argument("--single_object_min_fruit_face_pixels", type=int, default=850)
    parser.add_argument("--min_fruit_face_side", type=int, default=10)
    parser.add_argument("--seg_contour_epsilon_ratio", type=float, default=0.01)
    parser.add_argument("--negative_ratio", type=float, default=0.18)
    parser.add_argument("--ideal_visibility_max_attempts", type=int, default=80)
    parser.add_argument("--background_dir", type=str, default="datasets/backgrounds/coco2017")
    parser.add_argument("--arena_background_ratio", type=float, default=0.0)
    parser.add_argument("--arena_booster_mode", action="store_true")
    parser.add_argument("--arena_floor_material", choices=["sun111_wood"], default="sun111_wood")
    parser.add_argument("--arena_wall_material", choices=["sun168_beige"], default="sun168_beige")
    parser.add_argument("--arena_full_background_ratio", type=float, default=1.0)
    parser.add_argument("--robot_camera_view", action="store_true")
    parser.add_argument("--wall_contact_ratio", type=float, default=0.25)
    parser.add_argument("--corner_scene_ratio", type=float, default=0.10)
    parser.add_argument("--motion_blur_hard_negative_ratio", type=float, default=0.10)
    parser.add_argument("--plain_cube_hard_negative_ratio", type=float, default=0.05)
    parser.add_argument("--apple_orange_boost", action="store_true")
    parser.add_argument("--fruit_texture_dir", type=str, default="datasets/fruit_textures/final_fruits36065_original25_fruitseg30_10")
    parser.add_argument("--fruit_texture_aug", choices=["none", "light", "strong"], default="strong")
    parser.add_argument("--fruit_texture_layout", choices=["single", "collage", "mixed"], default="single")
    parser.add_argument("--fruit_texture_collage_prob", type=float, default=0.35)
    parser.add_argument("--fruit_visibility_easy_weight", type=float, default=0.50)
    parser.add_argument("--fruit_visibility_mid_weight", type=float, default=0.45)
    parser.add_argument("--fruit_visibility_hard_weight", type=float, default=0.05)
    parser.add_argument("--fruit_class_weight_scale", type=float, default=1.20)
    parser.add_argument("--hard_min_fruit_visible_ratio", type=float, default=-1.0)
    parser.add_argument("--hard_min_fruit_face_pixels", type=int, default=-1)
    parser.add_argument("--lighting_mode", choices=["random", "soft_overhead"], default="random")
    parser.add_argument("--canonical_fruit_textures", action="store_true")
    parser.add_argument(
        "--single_fruit_class_per_image",
        action="store_true",
        default=False,
        help="Force every fruit cube in one generated image to use the same fruit class.",
    )
    parser.add_argument(
        "--allow_mixed_fruit_classes_per_image",
        dest="single_fruit_class_per_image",
        action="store_false",
        help="Allow different fruit classes to appear in the same generated image.",
    )
    parser.add_argument(
        "--single_fruit_texture_per_cube",
        action="store_true",
        default=False,
        help="Use one source fruit texture on every fruit face of a cube.",
    )
    parser.add_argument(
        "--allow_multiple_fruit_textures_per_cube",
        dest="single_fruit_texture_per_cube",
        action="store_false",
        help="Allow different same-class source texture files on different faces of one cube.",
    )
    parser.add_argument(
        "--save_ideal_visibility_debug",
        action="store_true",
        help="Save per-image ideal visibility debug panels during generation. This is useful for QA but slows large runs.",
    )
    parser.add_argument("--email_to", type=str, default="")
    parser.add_argument("--email_every", type=int, default=1000)
    parser.add_argument("--email_config", type=str, default="config/email_smtp.json")
    parser.add_argument("--training_monitor_interval", type=int, default=1800)
    parser.add_argument("--disable_training_monitor", action="store_true")

    parser.add_argument("--model", type=str, default="yolo26n-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--train_workers", type=int, default=4)
    parser.add_argument("--cache", type=str, default="disk")
    parser.add_argument("--amp", type=str, default="False")
    parser.add_argument("--deterministic", type=str, default="False")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--project", type=str, default="runs/yolo26_seg_train")
    parser.add_argument("--name", type=str, default="yolo26n_seg_ideal_debug")
    parser.add_argument("--train_low_aug", action="store_true")

    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--preview_count", type=int, default=-1, help="Per split. -1 means all images.")
    parser.add_argument("--preview_sheet_count", type=int, default=100, help="Evenly sampled images per split contact sheet.")
    parser.add_argument("--preview_sheet_columns", type=int, default=5)
    parser.add_argument("--preview_sheet_thumb_size", type=int, default=180)
    parser.add_argument("--side_by_side_count", type=int, default=100)

    parser.add_argument("--skip_generate", action="store_true")
    parser.add_argument("--skip_split", action="store_true")
    parser.add_argument("--skip_preview", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_predict", action="store_true")
    parser.add_argument("--skip_side_by_side", action="store_true")
    parser.add_argument("--skip_audit", action="store_true")
    parser.add_argument("--resume_generate", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.arena_booster_mode:
        args.arena_background_ratio = args.arena_full_background_ratio
        args.robot_camera_view = True if not args.robot_camera_view else args.robot_camera_view
        args.lighting_mode = "soft_overhead" if args.lighting_mode == "random" else args.lighting_mode
        args.apple_orange_boost = True if not args.apple_orange_boost else args.apple_orange_boost

    dataset = Path(args.dataset)
    raw_dir = Path(args.raw_dir) if args.raw_dir else dataset / "_generated_all"
    counts = expected_counts(args)

    if not args.dry_run:
        dataset.mkdir(parents=True, exist_ok=True)
    if not args.skip_generate:
        generate_dataset(args, raw_dir)
    audit_generated_dataset(args, raw_dir)
    if not args.skip_split:
        split_generated_dataset(dataset, raw_dir, counts, args.seed, dry_run=args.dry_run)
    if not args.skip_preview:
        make_previews(args, dataset, counts)
    if not args.skip_train:
        train_model(args, dataset)
    if not args.skip_predict:
        predict_test(args, dataset)
    if not args.skip_side_by_side:
        make_side_by_side(args, dataset)


if __name__ == "__main__":
    main()
