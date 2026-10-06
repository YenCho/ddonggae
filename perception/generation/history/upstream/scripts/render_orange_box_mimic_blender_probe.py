import blenderproc as bproc
import argparse
import json
import math
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a tiny evaluation-only Blender mimic set for the real orange-cube boundary issue. "
            "This does not create training labels and must not be used as a booster dataset."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--samples", type=int, default=48)
    return parser.parse_args()


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def make_mat(name: str, color: tuple[float, float, float, float], roughness: float = 0.65) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
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
    cube.data.materials.append(make_mat("slightly_warm_white_cube", cube_color, roughness=0.82))
    bevel = cube.modifiers.new("small_real_cube_bevel", "BEVEL")
    bevel.width = 0.035
    bevel.segments = 2
    cube.modifiers.new("weighted_normals", "WEIGHTED_NORMAL")
    return cube


def add_plane_square(
    name: str,
    size: float,
    z: float,
    mat: bpy.types.Material,
    x: float = 0.0,
    y: float = 0.0,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(size=size, location=(x, y, z))
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)
    return obj


def add_right_face_square(
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


def add_disc(
    name: str,
    x: float,
    y: float,
    z: float,
    radius: float,
    mat: bpy.types.Material,
    scale_y: float = 1.0,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_circle_add(vertices=96, radius=radius, fill_type="TRIFAN", location=(x, y, z))
    obj = bpy.context.object
    obj.name = name
    obj.scale.y = scale_y
    obj.data.materials.append(mat)
    return obj


def add_right_face_disc(
    name: str,
    y: float,
    z: float,
    x: float,
    radius: float,
    mat: bpy.types.Material,
    scale_y: float = 1.0,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_circle_add(
        vertices=96,
        radius=radius,
        fill_type="TRIFAN",
        location=(x, y, z),
        rotation=(0.0, math.radians(90.0), 0.0),
    )
    obj = bpy.context.object
    obj.name = name
    obj.scale.y = scale_y
    obj.data.materials.append(mat)
    return obj


def add_camera_and_lights(params: dict) -> None:
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


def render_variant(output: Path, variant: dict) -> dict:
    clear_scene()
    add_cube(variant["cube_color"])

    edge = make_mat("paper_label_edge", variant["label_edge_color"], roughness=0.92)
    paper = make_mat("paper_label", variant["label_color"], roughness=0.96)
    orange = make_mat("printed_orange_ink", variant["orange_color"], roughness=0.98)
    shadow = make_mat("slightly_dirty_print_shadow", variant["shadow_color"], roughness=0.98)

    label_size = variant["label_size"]
    add_plane_square("paper_edge_square", label_size * 1.045, 1.013, edge, variant["label_x"], variant["label_y"])
    add_plane_square("paper_square", label_size, 1.018, paper, variant["label_x"], variant["label_y"])

    if variant["shape"] == "single":
        add_disc("single_orange_shadow", variant["label_x"], variant["label_y"], 1.025, variant["radius"] * 1.02, shadow, 0.94)
        add_disc("single_orange_print", variant["label_x"], variant["label_y"], 1.031, variant["radius"], orange, 0.94)
    else:
        dx = variant["pair_dx"]
        for idx, x in enumerate([variant["label_x"] - dx, variant["label_x"] + dx]):
            add_disc(f"orange_shadow_{idx}", x, variant["label_y"], 1.025, variant["radius"] * 1.02, shadow, 0.93)
            add_disc(f"orange_print_{idx}", x, variant["label_y"], 1.031, variant["radius"], orange, 0.93)

    if variant.get("side_label", False):
        side_size = variant["side_label_size"]
        add_right_face_square(
            "right_paper_edge_square",
            side_size * 1.035,
            1.013,
            variant["side_y"],
            variant["side_z"],
            edge,
        )
        add_right_face_square("right_paper_square", side_size, 1.018, variant["side_y"], variant["side_z"], paper)
        add_right_face_disc(
            "right_orange_shadow",
            variant["side_y"],
            variant["side_z"],
            1.025,
            variant["side_radius"] * 1.02,
            shadow,
            0.92,
        )
        add_right_face_disc(
            "right_orange_print",
            variant["side_y"],
            variant["side_z"],
            1.031,
            variant["side_radius"],
            orange,
            0.92,
        )

    floor_mat = make_mat("warm_desk_floor", variant["floor_color"], roughness=0.76)
    bpy.ops.mesh.primitive_plane_add(size=9.0, location=(0.0, 0.0, -1.01))
    floor = bpy.context.object
    floor.name = "desk_floor"
    floor.data.materials.append(floor_mat)

    add_camera_and_lights(variant)

    scene = bpy.context.scene
    try:
        scene.render.engine = "CYCLES"
        scene.cycles.samples = int(variant["samples"])
        scene.cycles.use_denoising = True
    except Exception:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = int(variant["resolution"])
    scene.render.resolution_y = int(variant["resolution"])
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = variant["exposure"]
    scene.view_settings.gamma = variant["gamma"]

    out_file = output / f"{variant['name']}.png"
    scene.render.filepath = str(out_file)
    bpy.ops.render.render(write_still=True)
    return {
        "name": variant["name"],
        "image": str(out_file),
        "shape": variant["shape"],
        "label_size": variant["label_size"],
        "radius": variant["radius"],
        "orange_color_rgba": variant["orange_color"],
        "light_color": variant["light_color"],
        "exposure": variant["exposure"],
        "camera_loc": variant["camera_loc"],
        "note": variant["note"],
    }


def variants(resolution: int, samples: int) -> list[dict]:
    base_camera = (2.65, -3.75, 2.25)
    return [
        {
            "name": "v00_single_large_clean_orange",
            "note": "large saturated one-orange print; should be the easy orange side",
            "shape": "single",
            "label_size": 1.42,
            "label_x": 0.0,
            "label_y": 0.0,
            "radius": 0.34,
            "pair_dx": 0.20,
            "orange_color": (1.00, 0.44, 0.04, 1.0),
            "shadow_color": (0.78, 0.24, 0.08, 1.0),
            "label_color": (0.93, 0.93, 0.91, 1.0),
            "label_edge_color": (0.73, 0.73, 0.71, 1.0),
            "cube_color": (0.86, 0.86, 0.82, 1.0),
            "floor_color": (0.62, 0.50, 0.39, 1.0),
            "light_loc": (-2.4, -3.0, 4.2),
            "light_energy": 560.0,
            "light_size": 3.2,
            "light_color": (1.00, 0.96, 0.88),
            "camera_loc": base_camera,
            "look_at": (0.0, 0.0, 0.15),
            "lens": 62.0,
            "exposure": -0.15,
            "gamma": 1.0,
            "resolution": resolution,
            "samples": samples,
        },
        {
            "name": "v01_pair_medium_clean_orange",
            "note": "two orange blobs on paper label, close to the reported orange card shape",
            "shape": "pair",
            "label_size": 1.25,
            "label_x": 0.02,
            "label_y": -0.03,
            "radius": 0.23,
            "pair_dx": 0.19,
            "orange_color": (1.00, 0.38, 0.05, 1.0),
            "shadow_color": (0.70, 0.22, 0.08, 1.0),
            "label_color": (0.91, 0.91, 0.88, 1.0),
            "label_edge_color": (0.68, 0.68, 0.66, 1.0),
            "cube_color": (0.84, 0.85, 0.81, 1.0),
            "floor_color": (0.60, 0.49, 0.39, 1.0),
            "light_loc": (-2.2, -3.4, 3.6),
            "light_energy": 430.0,
            "light_size": 3.8,
            "light_color": (1.00, 0.93, 0.82),
            "camera_loc": base_camera,
            "look_at": (0.0, 0.0, 0.15),
            "lens": 64.0,
            "exposure": -0.30,
            "gamma": 1.0,
            "resolution": resolution,
            "samples": samples,
        },
        {
            "name": "v02_pair_small_redorange_dim",
            "note": "small red-orange low-margin print under warm dim lighting",
            "shape": "pair",
            "label_size": 1.05,
            "label_x": 0.02,
            "label_y": -0.05,
            "radius": 0.17,
            "pair_dx": 0.15,
            "orange_color": (0.86, 0.27, 0.08, 1.0),
            "shadow_color": (0.55, 0.16, 0.07, 1.0),
            "label_color": (0.86, 0.85, 0.82, 1.0),
            "label_edge_color": (0.62, 0.62, 0.60, 1.0),
            "cube_color": (0.80, 0.81, 0.77, 1.0),
            "floor_color": (0.58, 0.47, 0.37, 1.0),
            "light_loc": (-1.6, -3.2, 3.0),
            "light_energy": 315.0,
            "light_size": 4.2,
            "light_color": (1.00, 0.89, 0.75),
            "camera_loc": (2.45, -3.50, 2.05),
            "look_at": (0.0, 0.0, 0.12),
            "lens": 66.0,
            "exposure": -0.45,
            "gamma": 1.04,
            "resolution": resolution,
            "samples": samples,
        },
        {
            "name": "v03_pair_tiny_low_sat_paper",
            "note": "tiny desaturated orange print on grayish paper label",
            "shape": "pair",
            "label_size": 0.92,
            "label_x": 0.03,
            "label_y": -0.08,
            "radius": 0.135,
            "pair_dx": 0.12,
            "orange_color": (0.72, 0.30, 0.11, 1.0),
            "shadow_color": (0.48, 0.17, 0.08, 1.0),
            "label_color": (0.80, 0.80, 0.77, 1.0),
            "label_edge_color": (0.58, 0.58, 0.56, 1.0),
            "cube_color": (0.78, 0.79, 0.75, 1.0),
            "floor_color": (0.56, 0.46, 0.36, 1.0),
            "light_loc": (-1.4, -3.0, 2.8),
            "light_energy": 260.0,
            "light_size": 4.8,
            "light_color": (1.00, 0.88, 0.72),
            "camera_loc": (2.38, -3.38, 1.98),
            "look_at": (0.0, 0.0, 0.08),
            "lens": 68.0,
            "exposure": -0.55,
            "gamma": 1.08,
            "resolution": resolution,
            "samples": samples,
        },
        {
            "name": "v04_pair_small_cool_flat",
            "note": "small print with cooler lighting to separate color-cast from shape/scale",
            "shape": "pair",
            "label_size": 1.02,
            "label_x": 0.01,
            "label_y": -0.05,
            "radius": 0.165,
            "pair_dx": 0.15,
            "orange_color": (0.88, 0.32, 0.07, 1.0),
            "shadow_color": (0.54, 0.18, 0.08, 1.0),
            "label_color": (0.86, 0.87, 0.86, 1.0),
            "label_edge_color": (0.60, 0.61, 0.61, 1.0),
            "cube_color": (0.80, 0.82, 0.82, 1.0),
            "floor_color": (0.56, 0.48, 0.40, 1.0),
            "light_loc": (-1.9, -3.4, 3.2),
            "light_energy": 315.0,
            "light_size": 4.0,
            "light_color": (0.88, 0.94, 1.00),
            "camera_loc": (2.45, -3.50, 2.05),
            "look_at": (0.0, 0.0, 0.12),
            "lens": 66.0,
            "exposure": -0.42,
            "gamma": 1.04,
            "resolution": resolution,
            "samples": samples,
        },
        {
            "name": "v10_actual_like_pair_side_clean",
            "note": "actual-like framing with top pair label and a visible right-side orange label",
            "shape": "pair",
            "label_size": 0.98,
            "label_x": 0.00,
            "label_y": -0.04,
            "radius": 0.165,
            "pair_dx": 0.145,
            "side_label": True,
            "side_label_size": 0.78,
            "side_y": -0.26,
            "side_z": 0.28,
            "side_radius": 0.20,
            "orange_color": (0.96, 0.35, 0.06, 1.0),
            "shadow_color": (0.62, 0.20, 0.08, 1.0),
            "label_color": (0.88, 0.88, 0.85, 1.0),
            "label_edge_color": (0.64, 0.64, 0.61, 1.0),
            "cube_color": (0.82, 0.83, 0.79, 1.0),
            "floor_color": (0.58, 0.48, 0.38, 1.0),
            "light_loc": (-2.1, -3.5, 3.8),
            "light_energy": 380.0,
            "light_size": 4.2,
            "light_color": (1.00, 0.93, 0.80),
            "camera_loc": (3.25, -5.20, 2.45),
            "look_at": (0.0, -0.02, 0.10),
            "lens": 48.0,
            "exposure": -0.32,
            "gamma": 1.03,
            "resolution": resolution,
            "samples": samples,
        },
        {
            "name": "v11_actual_like_pair_side_red_dim",
            "note": "actual-like framing plus smaller red-orange low-saturation print",
            "shape": "pair",
            "label_size": 0.90,
            "label_x": 0.01,
            "label_y": -0.06,
            "radius": 0.135,
            "pair_dx": 0.125,
            "side_label": True,
            "side_label_size": 0.72,
            "side_y": -0.26,
            "side_z": 0.26,
            "side_radius": 0.17,
            "orange_color": (0.76, 0.25, 0.09, 1.0),
            "shadow_color": (0.48, 0.15, 0.07, 1.0),
            "label_color": (0.82, 0.82, 0.79, 1.0),
            "label_edge_color": (0.57, 0.57, 0.55, 1.0),
            "cube_color": (0.78, 0.79, 0.75, 1.0),
            "floor_color": (0.56, 0.46, 0.36, 1.0),
            "light_loc": (-1.6, -3.2, 3.0),
            "light_energy": 270.0,
            "light_size": 4.8,
            "light_color": (1.00, 0.88, 0.72),
            "camera_loc": (3.25, -5.20, 2.45),
            "look_at": (0.0, -0.02, 0.08),
            "lens": 48.0,
            "exposure": -0.55,
            "gamma": 1.08,
            "resolution": resolution,
            "samples": samples,
        },
        {
            "name": "v12_actual_like_pair_side_tiny_flat",
            "note": "actual-like framing with tiny flat print and weak paper contrast",
            "shape": "pair",
            "label_size": 0.82,
            "label_x": 0.02,
            "label_y": -0.08,
            "radius": 0.115,
            "pair_dx": 0.105,
            "side_label": True,
            "side_label_size": 0.62,
            "side_y": -0.27,
            "side_z": 0.23,
            "side_radius": 0.14,
            "orange_color": (0.67, 0.25, 0.10, 1.0),
            "shadow_color": (0.42, 0.14, 0.07, 1.0),
            "label_color": (0.79, 0.79, 0.76, 1.0),
            "label_edge_color": (0.55, 0.55, 0.53, 1.0),
            "cube_color": (0.77, 0.78, 0.74, 1.0),
            "floor_color": (0.55, 0.45, 0.35, 1.0),
            "light_loc": (-1.4, -3.0, 2.8),
            "light_energy": 245.0,
            "light_size": 5.0,
            "light_color": (1.00, 0.86, 0.70),
            "camera_loc": (3.25, -5.20, 2.45),
            "look_at": (0.0, -0.02, 0.06),
            "lens": 48.0,
            "exposure": -0.60,
            "gamma": 1.10,
            "resolution": resolution,
            "samples": samples,
        },
    ]


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in variants(args.resolution, args.samples):
        rows.append(render_variant(args.output, variant))
    (args.output / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
