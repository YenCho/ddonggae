import blenderproc as bproc

import argparse
import math
import os
import random
from pathlib import Path

import bpy
import numpy as np
from blenderproc.python.utility.LabelIdMapping import LabelIdMapping


CLASS_TO_ID = {
    "plain_cube": 1,
    "octahedron": 2,
    "dodecahedron": 3,
    "icosahedron": 4,
}

SUPER_CATEGORY = "Data_Generation_Blender"
TABLE_HALF_EXTENT = 2.1


def make_material(name, color, roughness=0.8, metallic=0.0):
    mat = bproc.material.create(name)
    mat.set_principled_shader_value("Base Color", color)
    mat.set_principled_shader_value("Roughness", roughness)
    mat.set_principled_shader_value("Metallic", metallic)
    return mat


def create_box(name, location, scale, material):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    bpy_obj = bpy.context.object
    bpy_obj.name = name
    obj = bproc.object.convert_to_meshes([bpy_obj])[0]
    obj.set_scale(scale)
    obj.replace_materials(material)
    return obj


def create_cylinder(name, location, radius, depth, material, vertices=48):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=location)
    bpy_obj = bpy.context.object
    bpy_obj.name = name
    obj = bproc.object.convert_to_meshes([bpy_obj])[0]
    obj.replace_materials(material)
    return obj


def create_tabletop_room():
    floor_mat = make_material(
        "warm_wood_table",
        random.choice([
            [0.47, 0.31, 0.18, 1.0],
            [0.58, 0.41, 0.25, 1.0],
            [0.36, 0.25, 0.18, 1.0],
        ]),
        roughness=random.uniform(0.52, 0.85),
    )
    wall_mat = make_material(
        "soft_room_wall",
        random.choice([
            [0.72, 0.75, 0.71, 1.0],
            [0.76, 0.72, 0.66, 1.0],
            [0.67, 0.71, 0.76, 1.0],
        ]),
        roughness=random.uniform(0.70, 0.98),
    )
    shadow_mat = make_material("dark_table_edge", [0.12, 0.10, 0.08, 1.0], roughness=0.88)

    create_box("tabletop", [0, 0, -0.025], [4.4, 4.4, 0.05], floor_mat)
    create_box("back_wall", [0, 2.18, 0.95], [4.4, 0.05, 1.9], wall_mat)
    create_box("left_wall", [-2.18, 0, 0.95], [0.05, 4.4, 1.9], wall_mat)
    create_box("front_table_lip", [0, -2.18, 0.055], [4.4, 0.06, 0.10], shadow_mat)


def add_realistic_distractors():
    # Everyday tabletop clutter. These objects are intentionally unlabeled.
    paper_mats = [
        make_material("paper_white", [0.94, 0.93, 0.88, 1.0], roughness=0.92),
        make_material("paper_blue", [0.70, 0.79, 0.88, 1.0], roughness=0.90),
        make_material("paper_yellow", [0.94, 0.84, 0.48, 1.0], roughness=0.88),
    ]
    plastic_mats = [
        make_material("muted_red_plastic", [0.60, 0.13, 0.10, 1.0], roughness=0.62),
        make_material("muted_green_plastic", [0.18, 0.41, 0.28, 1.0], roughness=0.66),
        make_material("charcoal_plastic", [0.05, 0.06, 0.07, 1.0], roughness=0.72),
    ]
    metal_mat = make_material("brushed_metal", [0.55, 0.57, 0.56, 1.0], roughness=0.35, metallic=0.4)

    for i in range(random.randint(4, 8)):
        mat = random.choice(paper_mats)
        x, y = random.uniform(-1.7, 1.7), random.uniform(-1.7, 1.5)
        sx, sy = random.uniform(0.20, 0.55), random.uniform(0.12, 0.38)
        paper = create_box(f"paper_note_{i}", [x, y, 0.003], [sx, sy, 0.004], mat)
        paper.set_rotation_euler([0, 0, random.uniform(0, math.pi)])

    for i in range(random.randint(1, 3)):
        cup_mat = random.choice(plastic_mats)
        x, y = random.uniform(-1.6, 1.6), random.uniform(-1.5, 1.4)
        cup = create_cylinder(f"cup_or_cap_{i}", [x, y, 0.07], random.uniform(0.045, 0.085), random.uniform(0.08, 0.16), cup_mat)
        cup.set_rotation_euler([random.uniform(-0.08, 0.08), random.uniform(-0.08, 0.08), random.uniform(0, math.pi)])

    for i in range(random.randint(2, 5)):
        mat = random.choice(plastic_mats + [metal_mat])
        x, y = random.uniform(-1.7, 1.7), random.uniform(-1.7, 1.5)
        sx, sy, sz = random.uniform(0.06, 0.32), random.uniform(0.018, 0.055), random.uniform(0.012, 0.045)
        block = create_box(f"desk_tool_{i}", [x, y, sz / 2 + 0.002], [sx, sy, sz], mat)
        block.set_rotation_euler([0, 0, random.uniform(0, math.pi)])

    cable_mat = make_material("black_cable", [0.01, 0.01, 0.012, 1.0], roughness=0.82)
    for i in range(random.randint(1, 3)):
        cable = create_box(
            f"short_cable_{i}",
            [random.uniform(-1.7, 1.7), random.uniform(-1.7, 1.5), 0.012],
            [random.uniform(0.35, 0.85), 0.012, 0.012],
            cable_mat,
        )
        cable.set_rotation_euler([0, 0, random.uniform(0, math.pi)])


def set_random_world_background():
    bpy.context.scene.world.color = random.choice([
        [0.76, 0.80, 0.86],
        [0.88, 0.84, 0.77],
        [0.80, 0.82, 0.78],
    ])


def add_realistic_lights():
    key = bproc.types.Light()
    key.set_type("AREA")
    key.set_location([random.uniform(-1.2, 1.4), random.uniform(-1.4, 0.9), random.uniform(1.2, 2.0)])
    key.set_energy(random.uniform(280, 650))
    key.set_radius(random.uniform(0.7, 1.5))
    key.set_color(random.choice([[1.0, 0.93, 0.82], [0.93, 0.96, 1.0], [1.0, 1.0, 1.0]]))

    if random.random() < 0.8:
        fill = bproc.types.Light()
        fill.set_type("POINT")
        fill.set_location([random.uniform(-2.0, 2.0), random.uniform(0.3, 2.0), random.uniform(0.45, 1.2)])
        fill.set_energy(random.uniform(20, 90))
        fill.set_color([0.88, 0.92, 1.0])


def apply_target_material(obj):
    base = random.uniform(0.77, 0.98)
    tint = np.array([
        base + random.uniform(-0.04, 0.04),
        base + random.uniform(-0.04, 0.04),
        base + random.uniform(-0.04, 0.04),
        1.0,
    ])
    mat = make_material("slightly_off_white_target", np.clip(tint, 0, 1).tolist(), roughness=random.uniform(0.44, 0.88))
    obj.replace_materials(mat)


def add_small_bevel(obj):
    try:
        bevel = obj.blender_obj.modifiers.new("tiny_bevel", "BEVEL")
        bevel.width = random.uniform(0.0008, 0.0035)
        bevel.segments = random.choice([1, 2])
        normal = obj.blender_obj.modifiers.new("weighted_normal", "WEIGHTED_NORMAL")
        normal.keep_sharp = True
    except Exception:
        pass


def get_min_z(obj):
    bbox = np.asarray(obj.get_bound_box())
    return float(np.min(bbox[:, 2]))


def place_object_on_floor(obj, x, y):
    obj.set_location([x, y, 0.08])
    min_z = get_min_z(obj)
    loc = obj.get_location()
    obj.set_location([loc[0], loc[1], loc[2] - min_z + 0.006])


def sample_xy(existing_positions, min_dist=0.24):
    for _ in range(800):
        if random.random() < 0.35:
            x = random.choice([random.uniform(-1.8, -1.25), random.uniform(1.25, 1.8)])
            y = random.uniform(-1.55, 1.45)
        else:
            x = random.uniform(-1.55, 1.55)
            y = random.uniform(-1.55, 1.35)

        p = np.array([x, y])
        if all(np.linalg.norm(p - q) >= min_dist for q in existing_positions):
            return x, y

    return random.uniform(-1.2, 1.2), random.uniform(-1.2, 1.2)


def load_random_objects(asset_dir, mode):
    asset_dir = Path(asset_dir)
    if mode == "negative":
        return []

    if mode == "closeup":
        n_obj = random.randint(1, 3)
    else:
        n_obj = random.randint(4, 9)

    objects = []
    positions = []
    classes = list(CLASS_TO_ID.keys())

    for idx, cls_name in enumerate(random.choices(classes, k=n_obj)):
        obj_path = asset_dir / f"{cls_name}.obj"
        if not obj_path.exists():
            raise FileNotFoundError(f"Missing asset: {obj_path}")

        obj = bproc.loader.load_obj(str(obj_path))[0]
        obj.set_name(f"{cls_name}_{idx:03d}")
        obj.set_cp("category_id", CLASS_TO_ID[cls_name])
        obj.set_cp("supercategory", SUPER_CATEGORY)

        apply_target_material(obj)
        add_small_bevel(obj)
        obj.set_rotation_euler([
            random.uniform(0, math.pi * 2),
            random.uniform(0, math.pi * 2),
            random.uniform(0, math.pi * 2),
        ])

        s = random.uniform(0.75, 1.45)
        obj.set_scale([s, s, s])

        x, y = sample_xy(positions, min_dist=random.uniform(0.20, 0.38))
        positions.append(np.array([x, y]))
        place_object_on_floor(obj, x, y)
        objects.append(obj)

    return objects


def add_confusers_for_negative():
    add_realistic_distractors()
    for i in range(random.randint(4, 9)):
        mat = make_material(f"white_unlabeled_scrap_{i}", random.choice([
            [0.82, 0.82, 0.80, 1.0],
            [0.92, 0.91, 0.86, 1.0],
            [0.70, 0.72, 0.72, 1.0],
        ]), roughness=random.uniform(0.5, 0.95))
        obj = create_box(
            f"unlabeled_scrap_{i}",
            [random.uniform(-1.7, 1.7), random.uniform(-1.7, 1.45), random.uniform(0.01, 0.05)],
            [random.uniform(0.03, 0.16), random.uniform(0.03, 0.18), random.uniform(0.006, 0.045)],
            mat,
        )
        obj.set_rotation_euler([random.uniform(0, 0.3), random.uniform(0, 0.3), random.uniform(0, math.pi)])


def get_object_center(obj):
    bbox = np.asarray(obj.get_bound_box())
    return np.mean(bbox, axis=0)


def look_at(cam_location, target):
    direction = np.asarray(target, dtype=float) - np.asarray(cam_location, dtype=float)
    rotation_matrix = bproc.camera.rotation_from_forward_vec(direction)
    return bproc.math.build_transformation_mat(cam_location, rotation_matrix)


def add_camera_views(objects, mode, n_views):
    for frame in range(n_views):
        if mode == "closeup" and objects:
            target = get_object_center(random.choice(objects))
            radius = random.uniform(0.30, 0.85)
            theta = random.uniform(0, 2 * math.pi)
            cam_location = np.array([
                target[0] + radius * math.cos(theta),
                target[1] + radius * math.sin(theta),
                random.uniform(0.16, 0.42),
            ])
            target += np.array([random.uniform(-0.03, 0.03), random.uniform(-0.03, 0.03), random.uniform(-0.02, 0.04)])
        else:
            cam_location = np.array([
                random.uniform(-1.75, 1.75),
                random.uniform(-1.95, -0.55),
                random.uniform(0.34, 1.05),
            ])
            if objects and random.random() < 0.82:
                centers = np.array([get_object_center(obj) for obj in random.sample(objects, k=min(len(objects), random.randint(1, 3)))])
                target = np.mean(centers, axis=0)
            else:
                target = np.array([random.uniform(-0.7, 0.7), random.uniform(-0.4, 0.9), random.uniform(0.03, 0.13)])

        bproc.camera.add_camera_pose(look_at(cam_location, target), frame=frame)


def set_camera_and_renderer(args):
    bproc.camera.set_resolution(args.width, args.height)
    lens_mm = random.uniform(args.lens_min, args.lens_max)
    bproc.camera.set_intrinsics_from_blender_params(
        lens=lens_mm,
        image_width=args.width,
        image_height=args.height,
        lens_unit="MILLIMETERS",
        clip_start=0.01,
        clip_end=20.0,
    )
    bproc.renderer.set_max_amount_of_samples(args.samples)
    try:
        bproc.renderer.enable_depth_output(False)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_dir", type=str, default="assets/generated")
    parser.add_argument("--output", type=str, default="datasets/blenderproc_coco")
    parser.add_argument("--scene_id", type=int, default=0)
    parser.add_argument("--mode", type=str, default="tabletop", choices=["tabletop", "closeup", "negative"])
    parser.add_argument("--views_per_scene", type=int, default=3)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--lens_min", type=float, default=22.0)
    parser.add_argument("--lens_max", type=float, default=45.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug_segmap", action="store_true")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    bproc.init()
    set_camera_and_renderer(args)
    set_random_world_background()
    create_tabletop_room()
    add_realistic_lights()

    objects = load_random_objects(args.asset_dir, args.mode)
    if args.mode == "negative":
        add_confusers_for_negative()
    else:
        add_realistic_distractors()

    add_camera_views(objects, args.mode, args.views_per_scene)
    data = bproc.renderer.render()
    seg_data = bproc.renderer.render_segmap(
        map_by=["instance", "class", "name", "cp_supercategory"],
        default_values={"class": 0, "name": "background", "cp_supercategory": "coco_annotations"},
    )
    if args.debug_segmap:
        print("seg_data_keys", sorted(seg_data.keys()))
        print("instance_attribute_maps", seg_data["instance_attribute_maps"][0])

    coco_dir = os.path.join(args.output, "coco_data")
    os.makedirs(coco_dir, exist_ok=True)

    bproc.writer.write_coco_annotations(
        coco_dir,
        instance_segmaps=seg_data["instance_segmaps"],
        instance_attribute_maps=seg_data["instance_attribute_maps"],
        colors=data["colors"],
        color_file_format="JPEG",
        mask_encoding_format="polygon",
        supercategory=SUPER_CATEGORY,
        label_mapping=LabelIdMapping.from_dict(CLASS_TO_ID),
        append_to_existing_output=True,
        jpg_quality=92,
        file_prefix=f"{args.scene_id:06d}_{args.mode}_",
        indent=2,
    )


if __name__ == "__main__":
    main()
