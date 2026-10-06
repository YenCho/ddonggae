import blenderproc as bproc
import argparse
import json
import math
import random
import shutil
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


# Diagnostic only. Do not use this direct-render crop script for training:
# training boosters must follow the existing COCO full-scene -> meta_v2 -> unified export pipeline.
CLASS_NAMES = ["apple", "orange", "banana", "pineapple", "plain"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
FRUIT_CLASSES = CLASS_NAMES[:4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a balanced cube-face unified print-label mimic dataset. "
            "This creates 224x224 unified-crop-style images with YOLO segment labels. "
            "Use first as a probe; promote to booster only after preview and current-model hardness checks."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--texture-root", type=Path, default=Path("datasets/fruit_textures/production_meta_v2_50000_v1_color_filtered_v2"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument(
        "--profile",
        choices=["legacy_hard", "realistic_a4_mild_exposure"],
        default="legacy_hard",
        help="legacy_hard keeps the older intentionally hard print-label probe; realistic_a4_mild_exposure keeps paper thin, colors natural, and exposure mild.",
    )
    parser.add_argument(
        "--label-policy",
        choices=["face", "patch"],
        default="face",
        help="face matches the original unified dataset policy: visible cube face gets the fruit/plain class.",
    )
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def ensure_inside_workspace(path: Path) -> Path:
    cwd = Path.cwd().resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(cwd)
    except ValueError as exc:
        raise RuntimeError(f"Refusing to reset path outside workspace: {resolved}") from exc
    return resolved


def reset_output(output: Path) -> None:
    output = ensure_inside_workspace(output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def make_mat(name: str, color: tuple[float, float, float, float], roughness: float = 0.8) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = roughness
    return mat


def make_image_mat(
    name: str,
    image_path: Path,
    *,
    saturation: float,
    value: float,
    roughness: float = 0.96,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    bsdf = tree.nodes.get("Principled BSDF")
    tex = tree.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(image_path))
    tex.extension = "CLIP"
    hsv = tree.nodes.new("ShaderNodeHueSaturation")
    hsv.inputs["Saturation"].default_value = float(saturation)
    hsv.inputs["Value"].default_value = float(value)
    tree.links.new(tex.outputs["Color"], hsv.inputs["Color"])
    if bsdf is not None:
        tree.links.new(hsv.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = roughness
    return mat


def look_at(obj: bpy.types.Object, target: tuple[float, float, float]) -> None:
    loc = Vector(obj.location)
    direction = Vector(target) - loc
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_cube(cube_color: tuple[float, float, float, float]) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    cube = bpy.context.object
    cube.name = "white_cube"
    cube.data.materials.append(make_mat("printed_white_cube", cube_color, roughness=random.uniform(0.74, 0.90)))
    bevel = cube.modifiers.new("small_real_cube_bevel", "BEVEL")
    bevel.width = random.uniform(0.018, 0.045)
    bevel.segments = random.choice([1, 2])
    cube.modifiers.new("weighted_normals", "WEIGHTED_NORMAL")
    return cube


def add_top_plane(
    name: str,
    size: float,
    z: float,
    mat: bpy.types.Material,
    x: float,
    y: float,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(size=size, location=(x, y, z))
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)
    return obj


def add_side_plane(
    name: str,
    size: float,
    x: float,
    y: float,
    z: float,
    mat: bpy.types.Material,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(size=size, location=(x, y, z), rotation=(0.0, math.radians(90.0), 0.0))
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)
    return obj


def add_camera_and_light(params: dict) -> bpy.types.Object:
    bpy.ops.object.light_add(type="AREA", location=params["light_loc"])
    light = bpy.context.object
    light.name = "large_softbox"
    light.data.energy = params["light_energy"]
    light.data.size = params["light_size"]
    light.data.color = params["light_color"]

    bpy.ops.object.camera_add(location=params["camera_loc"])
    camera = bpy.context.object
    look_at(camera, params["look_at"])
    camera.data.lens = params["lens"]
    bpy.context.scene.camera = camera
    return camera


def plane_quad_world(obj: bpy.types.Object, size: float) -> list[Vector]:
    half = size / 2.0
    local = [
        Vector((-half, -half, 0.0)),
        Vector((half, -half, 0.0)),
        Vector((half, half, 0.0)),
        Vector((-half, half, 0.0)),
    ]
    return [obj.matrix_world @ p for p in local]


def face_quad_world(face: str) -> list[Vector]:
    h = 1.0
    if face == "front":
        return [Vector((-h, -h, -h)), Vector((h, -h, -h)), Vector((h, -h, h)), Vector((-h, -h, h))]
    if face == "right":
        return [Vector((h, -h, -h)), Vector((h, h, -h)), Vector((h, h, h)), Vector((h, -h, h))]
    if face == "top":
        return [Vector((-h, -h, h)), Vector((h, -h, h)), Vector((h, h, h)), Vector((-h, h, h))]
    raise ValueError(face)


def project_polygon(points: list[Vector], width: int, height: int) -> list[tuple[float, float]]:
    scene = bpy.context.scene
    camera = scene.camera
    out: list[tuple[float, float]] = []
    for p in points:
        co = world_to_camera_view(scene, camera, p)
        out.append((float(co.x) * (width - 1), float(1.0 - co.y) * (height - 1)))
    return out


def clip_polygon(poly: list[tuple[float, float]], width: int, height: int) -> list[tuple[float, float]]:
    return [(max(0.0, min(width - 1.0, x)), max(0.0, min(height - 1.0, y))) for x, y in poly]


def polygon_area(poly: list[tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    area = 0.0
    for i, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(i + 1) % len(poly)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def yolo_seg_line(class_name: str, poly: list[tuple[float, float]], width: int, height: int) -> str | None:
    if polygon_area(poly) < 12.0:
        return None
    coords: list[str] = []
    for x, y in poly:
        coords.append(f"{x / max(width - 1, 1):.6f}")
        coords.append(f"{y / max(height - 1, 1):.6f}")
    return f"{CLASS_TO_ID[class_name]} " + " ".join(coords)


def collect_textures(texture_root: Path) -> dict[str, list[Path]]:
    textures: dict[str, list[Path]] = {}
    for cls in FRUIT_CLASSES:
        class_dir = texture_root / cls
        paths = sorted(p for p in class_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not paths:
            raise RuntimeError(f"No textures for {cls}: {class_dir}")
        textures[cls] = paths
    return textures


def variant_params(cls_name: str, index: int, profile: str) -> dict:
    if profile == "realistic_a4_mild_exposure":
        hard = index % 5
        top_size = [1.30, 1.22, 1.14, 1.06, 0.98][hard] * random.uniform(0.96, 1.04)
        side_size = [1.06, 0.98, 0.90, 0.84, 0.78][hard] * random.uniform(0.96, 1.05)
        sat = [1.02, 0.98, 0.94, 0.90, 0.86][hard] * random.uniform(0.98, 1.04)
        value = [1.03, 1.00, 0.98, 0.95, 0.92][hard] * random.uniform(0.98, 1.04)
        if cls_name == "plain":
            sat = 1.0
            value = 1.0
        return {
            "top_size": top_size,
            "side_size": side_size,
            "top_x": random.uniform(-0.035, 0.035),
            "top_y": random.uniform(-0.08, -0.015),
            "side_y": random.uniform(-0.28, -0.16),
            "side_z": random.uniform(0.21, 0.35),
            "saturation": max(0.80, min(1.08, sat)),
            "value": max(0.88, min(1.08, value)),
            "paper_color": tuple(float(v) for v in random.choice([
                (0.94, 0.94, 0.91, 1.0),
                (0.91, 0.91, 0.88, 1.0),
                (0.88, 0.89, 0.86, 1.0),
            ])),
            "paper_edge_color": tuple(float(v) for v in random.choice([
                (0.78, 0.78, 0.75, 1.0),
                (0.72, 0.72, 0.70, 1.0),
                (0.82, 0.82, 0.79, 1.0),
            ])),
            "cube_color": tuple(float(v) for v in random.choice([
                (0.88, 0.89, 0.85, 1.0),
                (0.86, 0.87, 0.83, 1.0),
                (0.84, 0.85, 0.81, 1.0),
            ])),
            "floor_color": tuple(float(v) for v in random.choice([
                (0.58, 0.48, 0.37, 1.0),
                (0.63, 0.53, 0.41, 1.0),
                (0.54, 0.45, 0.36, 1.0),
            ])),
            "light_color": tuple(float(v) for v in random.choice([
                (1.0, 0.94, 0.86),
                (1.0, 0.91, 0.80),
                (0.95, 0.97, 1.0),
            ])),
            "light_energy": random.uniform(360, 620),
            "light_size": random.uniform(4.2, 6.2),
            "exposure": random.uniform(-0.16, 0.08),
            "gamma": random.uniform(0.98, 1.06),
            "camera_loc": (
                random.uniform(3.02, 3.46),
                random.uniform(-5.38, -4.88),
                random.uniform(2.12, 2.58),
            ),
            "look_at": (
                random.uniform(-0.04, 0.04),
                random.uniform(-0.07, 0.02),
                random.uniform(0.02, 0.15),
            ),
            "lens": random.uniform(45.0, 54.0),
            "top_edge_z": 1.0020,
            "top_image_z": 1.0038,
            "side_edge_x": 1.0020,
            "side_image_x": 1.0038,
            "hard_level": hard,
            "profile": profile,
        }

    hard = index % 5
    top_size = [0.92, 0.84, 0.78, 0.72, 0.66][hard] * random.uniform(0.94, 1.08)
    side_size = top_size * random.uniform(0.70, 0.88)
    sat = [0.92, 0.78, 0.64, 0.55, 0.48][hard] * random.uniform(0.92, 1.06)
    value = [0.92, 0.82, 0.74, 0.68, 0.62][hard] * random.uniform(0.94, 1.04)
    warm = hard >= 2
    if cls_name == "plain":
        sat = 1.0
        value = 1.0
    return {
        "top_size": top_size,
        "side_size": side_size,
        "top_x": random.uniform(-0.05, 0.05),
        "top_y": random.uniform(-0.11, -0.02),
        "side_y": random.uniform(-0.32, -0.18),
        "side_z": random.uniform(0.20, 0.34),
        "saturation": max(0.38, min(1.12, sat)),
        "value": max(0.50, min(1.05, value)),
        "paper_color": tuple(float(v) for v in random.choice([
            (0.88, 0.88, 0.85, 1.0),
            (0.84, 0.84, 0.81, 1.0),
            (0.80, 0.80, 0.77, 1.0),
        ])),
        "paper_edge_color": tuple(float(v) for v in random.choice([
            (0.62, 0.62, 0.60, 1.0),
            (0.56, 0.56, 0.54, 1.0),
            (0.70, 0.70, 0.67, 1.0),
        ])),
        "cube_color": tuple(float(v) for v in random.choice([
            (0.84, 0.85, 0.81, 1.0),
            (0.80, 0.81, 0.77, 1.0),
            (0.78, 0.79, 0.75, 1.0),
        ])),
        "floor_color": tuple(float(v) for v in random.choice([
            (0.58, 0.47, 0.36, 1.0),
            (0.63, 0.52, 0.40, 1.0),
            (0.54, 0.45, 0.36, 1.0),
        ])),
        "light_color": tuple(float(v) for v in ((1.0, 0.88, 0.72) if warm else (1.0, 0.94, 0.84))),
        "light_energy": random.uniform(230, 390) if warm else random.uniform(330, 540),
        "light_size": random.uniform(3.8, 5.2),
        "exposure": random.uniform(-0.65, -0.34) if warm else random.uniform(-0.38, -0.16),
        "gamma": random.uniform(1.02, 1.12),
        "camera_loc": (
            random.uniform(3.05, 3.48),
            random.uniform(-5.45, -4.95),
            random.uniform(2.18, 2.62),
        ),
        "look_at": (
            random.uniform(-0.04, 0.04),
            random.uniform(-0.07, 0.02),
            random.uniform(0.02, 0.14),
        ),
        "lens": random.uniform(45.0, 54.0),
        "top_edge_z": 1.014,
        "top_image_z": 1.021,
        "side_edge_x": 1.014,
        "side_image_x": 1.021,
        "hard_level": hard,
        "profile": profile,
    }


def write_data_yaml(output: Path, split: str) -> None:
    train_split = "train" if (output / "images" / "train").exists() else split
    val_split = "val" if (output / "images" / "val").exists() else split
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(CLASS_NAMES))
    (output / "data.yaml").write_text(
        f"path: {output.resolve().as_posix()}\n"
        f"train: images/{train_split}\n"
        f"val: images/{val_split}\n"
        "names:\n"
        f"{names}\n",
        encoding="utf-8",
    )


def render_one(
    *,
    output: Path,
    split: str,
    stem: str,
    cls_name: str,
    texture_path: Path | None,
    params: dict,
    resolution: int,
    samples: int,
    label_policy: str,
) -> dict:
    clear_scene()
    add_cube(params["cube_color"])

    paper = make_mat("paper_label", params["paper_color"], roughness=0.96)
    edge = make_mat("paper_edge", params["paper_edge_color"], roughness=0.96)
    if cls_name == "plain":
        image_mat = make_mat("blank_plain_label", params["paper_color"], roughness=0.96)
    else:
        image_mat = make_image_mat(
            f"{cls_name}_printed_texture",
            texture_path,
            saturation=params["saturation"],
            value=params["value"],
        )

    top_edge = add_top_plane(
        "top_paper_edge", params["top_size"] * 1.035, params["top_edge_z"], edge, params["top_x"], params["top_y"]
    )
    top_image = add_top_plane("top_print_label", params["top_size"], params["top_image_z"], image_mat, params["top_x"], params["top_y"])
    side_edge = add_side_plane(
        "side_paper_edge", params["side_size"] * 1.035, params["side_edge_x"], params["side_y"], params["side_z"], edge
    )
    side_image = add_side_plane("side_print_label", params["side_size"], params["side_image_x"], params["side_y"], params["side_z"], image_mat)

    floor = make_mat("warm_floor", params["floor_color"], roughness=0.78)
    bpy.ops.mesh.primitive_plane_add(size=9.0, location=(0.0, 0.0, -1.01))
    bpy.context.object.data.materials.append(floor)

    params = dict(params)
    params["light_loc"] = (
        random.uniform(-2.4, -1.3),
        random.uniform(-3.8, -2.7),
        random.uniform(2.7, 4.2),
    )
    add_camera_and_light(params)

    scene = bpy.context.scene
    try:
        scene.render.engine = "CYCLES"
        scene.cycles.samples = int(samples)
        scene.cycles.use_denoising = True
    except Exception:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = int(resolution)
    scene.render.resolution_y = int(resolution)
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = float(params["exposure"])
    scene.view_settings.gamma = float(params["gamma"])

    image_dir = output / "images" / split
    label_dir = output / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{stem}.jpg"
    scene.render.filepath = str(image_path)
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = random.randint(76, 92)
    bpy.ops.render.render(write_still=True)

    lines: list[str] = []
    width = height = resolution
    if label_policy == "patch":
        top_poly = clip_polygon(project_polygon(plane_quad_world(top_image, params["top_size"]), width, height), width, height)
        side_poly = clip_polygon(project_polygon(plane_quad_world(side_image, params["side_size"]), width, height), width, height)
    else:
        top_poly = clip_polygon(project_polygon(face_quad_world("top"), width, height), width, height)
        side_poly = clip_polygon(project_polygon(face_quad_world("right"), width, height), width, height)
    for class_for_label, poly in [(cls_name, top_poly), (cls_name, side_poly)]:
        line = yolo_seg_line(class_for_label, poly, width, height)
        if line:
            lines.append(line)

    # Plain visible cube faces remain part of the unified task and help keep fruit-vs-plain balanced.
    front_poly = clip_polygon(project_polygon(face_quad_world("front"), width, height), width, height)
    line = yolo_seg_line("plain", front_poly, width, height)
    if line:
        lines.append(line)

    (label_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "stem": stem,
        "split": split,
        "class": cls_name,
        "image": str(image_path),
        "label": str(label_dir / f"{stem}.txt"),
        "texture": str(texture_path) if texture_path else "",
        "hard_level": params["hard_level"],
        "top_size": params["top_size"],
        "side_size": params["side_size"],
        "saturation": params["saturation"],
        "value": params["value"],
        "exposure": params["exposure"],
        "gamma": params["gamma"],
        "label_policy": label_policy,
        "profile": params["profile"],
        "policy": "balanced all-class print-label mimic; no user hard-case photos; use as probe first",
    }


def merge_manifest(output: Path, split: str, rows: list[dict]) -> list[dict]:
    split_manifest = output / f"manifest_{split}.json"
    split_manifest.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    merged: list[dict] = []
    for split_name in ["train", "val"]:
        path = output / f"manifest_{split_name}.json"
        if path.exists():
            merged.extend(json.loads(path.read_text(encoding="utf-8")))
    if not merged:
        merged = rows
    (output / "manifest.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged


def main() -> None:
    args = parse_args()
    args.output = resolve(args.output)
    args.texture_root = resolve(args.texture_root)
    random.seed(args.seed)
    if args.reset:
        reset_output(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    textures = collect_textures(args.texture_root)

    rows: list[dict] = []
    counts = {name: 0 for name in CLASS_NAMES}
    for idx in range(args.count):
        cls_name = CLASS_NAMES[idx % len(CLASS_NAMES)]
        class_idx = counts[cls_name]
        counts[cls_name] += 1
        texture_path = None
        if cls_name in FRUIT_CLASSES:
            texture_path = random.choice(textures[cls_name])
        params = variant_params(cls_name, class_idx, args.profile)
        stem = f"{idx:06d}_{cls_name}_printlabel_h{params['hard_level']}"
        rows.append(
            render_one(
                output=args.output,
                split=args.split,
                stem=stem,
                cls_name=cls_name,
                texture_path=texture_path,
                params=params,
                resolution=args.resolution,
                samples=args.samples,
                label_policy=args.label_policy,
            )
        )

    write_data_yaml(args.output, args.split)
    merged_rows = merge_manifest(args.output, args.split, rows)
    (args.output / "README_DO_NOT_TRAIN_UNTIL_REVIEWED.md").write_text(
        "# Print-Label Mimic Probe Dataset\n\n"
        "이 데이터셋은 실제 orange/apple 흔들림 원인을 바탕으로 만든 balanced all-class probe입니다.\n"
        f"라벨 정책: `{args.label_policy}`. 기본 `face`는 기존 unified 데이터처럼 fruit가 보이는 cube face 전체를 해당 class로 라벨링합니다.\n"
        "사용자 제공 orange holdout 사진은 포함하지 않았습니다.\n"
        "먼저 preview와 current-model hardness를 확인한 뒤에만 booster 후보로 확장하세요.\n",
        encoding="utf-8",
    )
    (args.output / "README_DO_NOT_TRAIN_UNTIL_REVIEWED.md").write_text(
        "# Print-Label Mimic Probe Dataset\n\n"
        "이 데이터셋은 실제 webcam에서 보인 작은 인쇄 라벨, 저채도, 어두운 조명, 따뜻한 색온도 조건을 닮게 만든 balanced all-class probe입니다.\n"
        f"라벨 정책: `{args.label_policy}`. 기본 `face`는 기존 unified 데이터처럼 fruit가 보이는 cube face 전체를 해당 class로 라벨링합니다.\n"
        "사용자가 제공한 orange holdout 사진은 포함하지 않습니다.\n"
        "먼저 preview와 current-model hardness를 확인한 뒤에만 booster 후보로 확장하세요.\n"
        f"현재 누적 샘플 수: {len(merged_rows)}장.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "count": len(rows), "merged_count": len(merged_rows), "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
