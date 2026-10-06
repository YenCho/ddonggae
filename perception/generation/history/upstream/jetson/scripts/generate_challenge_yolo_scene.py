import blenderproc as bproc

import argparse
import math
import random
from pathlib import Path

import bpy
import cv2
import numpy as np
from mathutils import Vector


CLASS_TO_ID = {
    "plain_cube": 1,
    "octahedron": 2,
    "dodecahedron": 3,
    "icosahedron": 4,
}

YOLO_CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_TO_ID)}
SUPER_CATEGORY = "Data_Generation_Blender"
ARENA_LIMIT = 2.0
START_ZONE_X = (1.6, 2.0)
START_ZONE_Y = (-2.0, -1.6)


def make_material(name, color, roughness=0.8):
    mat = bproc.material.create(name)
    mat.set_principled_shader_value("Base Color", color)
    mat.set_principled_shader_value("Roughness", roughness)
    return mat


def create_box(name, location, scale, material):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    bpy_obj = bpy.context.object
    bpy_obj.name = name
    obj = bproc.object.convert_to_meshes([bpy_obj])[0]
    obj.set_scale(scale)
    obj.replace_materials(material)
    obj.set_cp("category_id", 0)
    obj.set_cp("supercategory", "coco_annotations")
    return obj


def create_arena():
    floor_mat = make_material(
        "plywood_floor",
        random.choice([
            [0.46, 0.32, 0.18, 1.0],
            [0.58, 0.43, 0.25, 1.0],
            [0.66, 0.52, 0.33, 1.0],
        ]),
        roughness=random.uniform(0.45, 0.92),
    )
    wall_mat = make_material(
        "matte_white_acrylic_wall",
        random.choice([
            [0.86, 0.86, 0.83, 1.0],
            [0.94, 0.94, 0.90, 1.0],
            [0.78, 0.80, 0.82, 1.0],
        ]),
        roughness=random.uniform(0.55, 0.98),
    )
    venue_mat = make_material(
        "distant_venue_background",
        random.choice([
            [0.38, 0.38, 0.36, 1.0],
            [0.55, 0.56, 0.58, 1.0],
            [0.66, 0.63, 0.58, 1.0],
        ]),
        roughness=0.95,
    )
    rim_mat = make_material("storage_printed_rim", [0.10, 0.10, 0.10, 1.0], roughness=0.9)

    create_box("floor", [0, 0, -0.01], [4.0, 4.0, 0.02], floor_mat)
    h = 0.30
    t = 0.03
    create_box("wall_pos_x", [2.0 + t / 2, 0, h / 2], [t, 4.0, h], wall_mat)
    create_box("wall_neg_x", [-2.0 - t / 2, 0, h / 2], [t, 4.0, h], wall_mat)
    create_box("wall_pos_y", [0, 2.0 + t / 2, h / 2], [4.0, t, h], wall_mat)
    create_box("wall_neg_y", [0, -2.0 - t / 2, h / 2], [4.0, t, h], wall_mat)
    create_box("venue_backdrop", [0, 2.55, 1.35], [6.0, 0.04, 2.1], venue_mat)

    create_box("storage_rim_right", [-1.6, -1.8, 0.0015], [0.010, 0.40, 0.003], rim_mat)
    create_box("storage_rim_top", [-1.8, -1.6, 0.0015], [0.40, 0.010, 0.003], rim_mat)


def add_print_material(obj):
    base = random.uniform(0.75, 0.99)
    color = np.clip([
        base + random.uniform(-0.05, 0.04),
        base + random.uniform(-0.05, 0.04),
        base + random.uniform(-0.05, 0.04),
        1.0,
    ], 0, 1).tolist()
    obj.replace_materials(make_material("white_printed_object", color, random.uniform(0.42, 0.94)))


def add_bevel(obj):
    try:
        bevel = obj.blender_obj.modifiers.new("small_print_bevel", "BEVEL")
        bevel.width = random.uniform(0.0008, 0.004)
        bevel.segments = random.choice([1, 2])
        normal = obj.blender_obj.modifiers.new("weighted_normal", "WEIGHTED_NORMAL")
        normal.keep_sharp = True
    except Exception:
        pass


def min_z(obj):
    bpy.context.view_layer.update()
    matrix = obj.blender_obj.matrix_world
    world_bbox = np.array([matrix @ Vector(corner) for corner in obj.blender_obj.bound_box])
    return float(np.min(world_bbox[:, 2]))


def place_on_floor(obj, x, y):
    obj.set_location([x, y, 0.15])
    loc = obj.get_location()
    obj.set_location([loc[0], loc[1], loc[2] - min_z(obj) + 0.003])


def in_start_zone(x, y):
    return START_ZONE_X[0] <= x <= START_ZONE_X[1] and START_ZONE_Y[0] <= y <= START_ZONE_Y[1]


def sample_position(existing, min_dist):
    for _ in range(1000):
        if random.random() < 0.20:
            side = random.choice(["left", "right", "top", "bottom"])
            if side == "left":
                x, y = random.uniform(-1.93, -1.74), random.uniform(-1.82, 1.82)
            elif side == "right":
                x, y = random.uniform(1.74, 1.93), random.uniform(-1.82, 1.82)
            elif side == "top":
                x, y = random.uniform(-1.82, 1.82), random.uniform(1.74, 1.93)
            else:
                x, y = random.uniform(-1.82, 1.82), random.uniform(-1.93, -1.74)
        else:
            x, y = random.uniform(-1.72, 1.72), random.uniform(-1.72, 1.72)

        if in_start_zone(x, y):
            continue
        if np.linalg.norm(np.array([x, y]) - np.array([1.8, -1.8])) < 0.60:
            continue
        if all(np.linalg.norm(np.array([x, y]) - p) >= min_dist for p in existing):
            return x, y
    return random.uniform(-1.4, 1.4), random.uniform(-1.2, 1.4)


def load_challenge_objects(asset_dir, scale_min, scale_max, full_rulebook_set):
    asset_dir = Path(asset_dir)
    if full_rulebook_set:
        classes = [name for name in CLASS_TO_ID for _ in range(4)]
    else:
        classes = random.choices(list(CLASS_TO_ID), k=random.randint(5, 16))
    random.shuffle(classes)

    objects = []
    positions = []
    for idx, cls_name in enumerate(classes):
        obj = bproc.loader.load_obj(str(asset_dir / f"{cls_name}.obj"))[0]
        obj.set_name(f"{cls_name}_{idx:03d}")
        obj.set_cp("category_id", CLASS_TO_ID[cls_name])
        obj.set_cp("supercategory", SUPER_CATEGORY)
        add_print_material(obj)
        add_bevel(obj)
        obj.set_rotation_euler([
            random.uniform(0, math.pi * 2),
            random.uniform(0, math.pi * 2),
            random.uniform(0, math.pi * 2),
        ])
        scale = random.uniform(scale_min, scale_max)
        obj.set_scale([scale, scale, scale])
        x, y = sample_position(positions, min_dist=random.uniform(0.16, 0.36) * scale)
        positions.append(np.array([x, y]))
        place_on_floor(obj, x, y)
        objects.append(obj)
    return objects


def add_challenge_lights():
    bpy.context.scene.world.color = random.choice([
        [0.22, 0.22, 0.22],
        [0.38, 0.38, 0.36],
        [0.55, 0.56, 0.58],
        [0.70, 0.68, 0.62],
    ])
    for _ in range(random.randint(1, 5)):
        light = bproc.types.Light()
        light.set_type(random.choice(["AREA", "POINT", "SUN"]))
        if light.get_type() == "SUN":
            light.set_rotation_euler([random.uniform(0.3, 1.4), 0, random.uniform(0, math.pi * 2)])
            light.set_energy(random.uniform(0.3, 3.2))
        else:
            light.set_location([
                random.uniform(-2.2, 2.2),
                random.uniform(-2.2, 2.2),
                random.uniform(0.25, 3.0),
            ])
            light.set_energy(random.uniform(35, 850))
            if light.get_type() == "AREA":
                light.set_radius(random.uniform(0.08, 2.0))
        light.set_color(random.choice([
            [1.0, 0.80, 0.58],
            [1.0, 0.94, 0.82],
            [0.78, 0.88, 1.0],
            [1.0, 1.0, 1.0],
        ]))


def look_at(cam_location, target):
    direction = np.asarray(target, dtype=float) - np.asarray(cam_location, dtype=float)
    rotation_matrix = bproc.camera.rotation_from_forward_vec(direction)
    return bproc.math.build_transformation_mat(cam_location, rotation_matrix)


def add_camera(objects, width, height):
    bproc.camera.set_resolution(width, height)
    bproc.camera.set_intrinsics_from_blender_params(
        lens=random.uniform(16.0, 65.0),
        image_width=width,
        image_height=height,
        lens_unit="MILLIMETERS",
        clip_start=0.01,
        clip_end=20.0,
    )
    if random.random() < 0.55:
        cam = np.array([
            random.uniform(1.45, 1.95),
            random.uniform(-1.95, -1.45),
            random.uniform(0.12, 0.42),
        ])
    else:
        cam = np.array([
            random.uniform(-1.7, 1.7),
            random.uniform(-1.8, 1.8),
            random.uniform(0.10, 0.55),
        ])

    if objects and random.random() < 0.85:
        target = np.mean([np.asarray(random.choice(objects).get_location()) for _ in range(random.randint(1, 3))], axis=0)
    else:
        target = np.array([random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(0.02, 0.12)])
    target += np.array([random.uniform(-0.25, 0.25), random.uniform(-0.25, 0.25), random.uniform(-0.03, 0.10)])
    bproc.camera.add_camera_pose(look_at(cam, target), frame=0)


def color_to_bgr(color):
    arr = np.asarray(color)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return arr[..., :3][..., ::-1].copy()


def labels_from_segmap(segmap, attr_map, min_area):
    labels = []
    h, w = segmap.shape[:2]
    for inst in attr_map:
        category_id = int(inst.get("category_id", 0))
        if category_id == 0:
            continue
        mask = segmap == int(inst["idx"])
        if int(mask.sum()) < min_area:
            continue
        ys, xs = np.where(mask)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        bw = max(1, x2 - x1 + 1)
        bh = max(1, y2 - y1 + 1)
        cls_name = next(name for name, cid in CLASS_TO_ID.items() if cid == category_id)
        labels.append([YOLO_CLASS_TO_ID[cls_name], (x1 + bw / 2) / w, (y1 + bh / 2) / h, bw / w, bh / h])
    return labels


def apply_camera_artifacts(image):
    img = image.astype(np.float32)
    img = img * random.uniform(0.65, 1.40) + random.uniform(-28, 28)
    if random.random() < 0.55:
        gamma = random.uniform(0.72, 1.45)
        img = 255.0 * np.power(np.clip(img, 0, 255) / 255.0, gamma)
    if random.random() < 0.45:
        img += np.random.normal(0, random.uniform(1.0, 10.0), img.shape)
    out = np.clip(img, 0, 255).astype(np.uint8)
    if random.random() < 0.20:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_dir", type=str, default="assets/generated")
    parser.add_argument("--output", type=str, default="datasets/yolo_challenge_arena")
    parser.add_argument("--image_id", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min_area", type=int, default=80)
    parser.add_argument("--scale_min", type=float, default=0.70)
    parser.add_argument("--scale_max", type=float, default=1.45)
    parser.add_argument("--partial_set", action="store_true")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    bproc.init()
    create_arena()
    add_challenge_lights()
    objects = load_challenge_objects(args.asset_dir, args.scale_min, args.scale_max, not args.partial_set)
    add_camera(objects, args.width, args.height)
    bproc.renderer.set_max_amount_of_samples(args.samples)
    data = bproc.renderer.render()
    seg_data = bproc.renderer.render_segmap(
        map_by=["instance", "class", "cp_supercategory"],
        default_values={"class": 0, "cp_supercategory": "coco_annotations"},
    )

    image = apply_camera_artifacts(color_to_bgr(data["colors"][0]))
    labels = labels_from_segmap(seg_data["instance_segmaps"][0], seg_data["instance_attribute_maps"][0], args.min_area)

    out = Path(args.output)
    image_dir = out / "images" / "train"
    label_dir = out / "labels" / "train"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.image_id:06d}"
    cv2.imwrite(str(image_dir / f"{stem}.jpg"), image, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    with open(label_dir / f"{stem}.txt", "w", encoding="utf-8") as f:
        for label in labels:
            f.write("%d %.6f %.6f %.6f %.6f\n" % tuple(label))
    print(f"image: {image_dir / f'{stem}.jpg'}")
    print(f"label: {label_dir / f'{stem}.txt'}")
    print(f"objects: {len(labels)}")


if __name__ == "__main__":
    main()
