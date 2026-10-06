import blenderproc as bproc
import argparse
import json
import math
import random
from pathlib import Path

import bpy
from mathutils import Vector


FRUITS = ("apple", "orange", "banana", "pineapple")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render diagnostic-only Blender scenes for unified cube-face context conflicts. "
            "This keeps fruit texture colors unchanged and changes only context/occlusion."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--texture-root",
        type=Path,
        default=Path("datasets/fruit_textures/production_meta_v2_50000_v1_color_filtered_v2"),
    )
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument(
        "--mode",
        choices=("context", "solo", "solo_a4"),
        default="context",
        help="context includes nearby cubes; solo renders only the target orange cube; solo_a4 uses pasted A4-paper labels.",
    )
    return parser.parse_args()


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def make_mat(name: str, color: tuple[float, float, float, float], roughness: float = 0.82) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = roughness
    return mat


def make_image_mat(name: str, image_path: Path, roughness: float = 0.96) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    bsdf = tree.nodes.get("Principled BSDF")
    coord = tree.nodes.new("ShaderNodeTexCoord")
    tex = tree.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(image_path.resolve()))
    tex.extension = "CLIP"
    tree.links.new(coord.outputs["UV"], tex.inputs["Vector"])
    if bsdf is not None:
        tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = roughness
    return mat


def look_at(obj: bpy.types.Object, target: tuple[float, float, float]) -> None:
    direction = Vector(target) - Vector(obj.location)
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_cube(name: str, loc: tuple[float, float, float], color=(0.86, 0.86, 0.82, 1.0)) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=loc)
    cube = bpy.context.object
    cube.name = name
    cube.data.materials.append(make_mat(f"{name}_white_plastic", color, roughness=0.84))
    bevel = cube.modifiers.new(f"{name}_small_bevel", "BEVEL")
    bevel.width = 0.035
    bevel.segments = 2
    cube.modifiers.new(f"{name}_weighted_normals", "WEIGHTED_NORMAL")
    return cube


def ensure_unit_uv(obj: bpy.types.Object) -> None:
    mesh = obj.data
    uv_layer = mesh.uv_layers.active or mesh.uv_layers.new(name="UVMap")
    coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    for poly in mesh.polygons:
        for idx, loop_index in enumerate(poly.loop_indices):
            uv_layer.data[loop_index].uv = coords[idx % 4]


def add_top_label(
    name: str,
    cube_loc: tuple[float, float, float],
    size: float,
    mat: bpy.types.Material,
    *,
    xoff: float = 0.0,
    yoff: float = -0.05,
    zlift: float = 0.026,
) -> bpy.types.Object:
    x, y, z = cube_loc
    bpy.ops.mesh.primitive_plane_add(size=size, location=(x + xoff, y + yoff, z + 1.0 + zlift))
    obj = bpy.context.object
    obj.name = name
    ensure_unit_uv(obj)
    obj.data.materials.append(mat)
    return obj


def add_top_rect_label(
    name: str,
    cube_loc: tuple[float, float, float],
    width: float,
    height: float,
    mat: bpy.types.Material,
    *,
    xoff: float = 0.0,
    yoff: float = -0.05,
    zlift: float = 0.026,
) -> bpy.types.Object:
    x, y, z = cube_loc
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(x + xoff, y + yoff, z + 1.0 + zlift))
    obj = bpy.context.object
    obj.name = name
    obj.scale = (width, height, 1.0)
    ensure_unit_uv(obj)
    obj.data.materials.append(mat)
    return obj


def add_right_label(
    name: str,
    cube_loc: tuple[float, float, float],
    size: float,
    mat: bpy.types.Material,
    *,
    yoff: float = -0.22,
    zoff: float = 0.23,
    xlift: float = 0.026,
) -> bpy.types.Object:
    x, y, z = cube_loc
    bpy.ops.mesh.primitive_plane_add(
        size=size,
        location=(x + 1.0 + xlift, y + yoff, z + zoff),
        rotation=(0.0, math.radians(90.0), 0.0),
    )
    obj = bpy.context.object
    obj.name = name
    ensure_unit_uv(obj)
    obj.data.materials.append(mat)
    return obj


def add_right_rect_label(
    name: str,
    cube_loc: tuple[float, float, float],
    width: float,
    height: float,
    mat: bpy.types.Material,
    *,
    yoff: float = -0.22,
    zoff: float = 0.23,
    xlift: float = 0.026,
) -> bpy.types.Object:
    x, y, z = cube_loc
    bpy.ops.mesh.primitive_plane_add(
        size=1.0,
        location=(x + 1.0 + xlift, y + yoff, z + zoff),
        rotation=(0.0, math.radians(90.0), 0.0),
    )
    obj = bpy.context.object
    obj.name = name
    obj.scale = (width, height, 1.0)
    ensure_unit_uv(obj)
    obj.data.materials.append(mat)
    return obj


def collect_textures(root: Path) -> dict[str, list[Path]]:
    textures: dict[str, list[Path]] = {}
    for cls in FRUITS:
        folder = root / cls
        paths = []
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            paths.extend(folder.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No textures for {cls} under {folder}")
        textures[cls] = sorted(paths)
    return textures


def add_scene_light_and_camera(params: dict) -> None:
    bpy.ops.object.light_add(type="AREA", location=params["light_loc"])
    light = bpy.context.object
    light.name = "soft_webcam_room_light"
    light.data.energy = params["light_energy"]
    light.data.size = params["light_size"]
    light.data.color = params["light_color"]

    bpy.ops.object.camera_add(location=params["camera_loc"])
    camera = bpy.context.object
    look_at(camera, params["look_at"])
    camera.data.lens = params["lens"]
    bpy.context.scene.camera = camera


def make_variant_list() -> list[dict]:
    base = {
        "camera_loc": (3.35, -4.85, 2.35),
        "look_at": (0.16, -0.02, 0.18),
        "lens": 50.0,
        "light_loc": (-2.2, -3.2, 3.8),
        "light_energy": 420.0,
        "light_size": 4.5,
        "light_color": (1.0, 0.94, 0.84),
        "exposure": -0.22,
        "gamma": 1.04,
        "target_loc": (0.0, 0.0, 0.0),
        "orange_top_size": 0.90,
        "orange_side_size": 0.78,
        "foreground_plain": False,
        "foreground_loc": (-0.72, -0.95, -0.02),
        "distractor_class": None,
        "distractor_loc": (1.72, -0.06, 0.0),
        "distractor_top": False,
        "distractor_side": False,
        "note": "baseline target orange cube only",
    }
    variants: list[dict] = []

    def add(name: str, **kwargs: object) -> None:
        row = dict(base)
        row.update(kwargs)
        row["name"] = name
        variants.append(row)

    add("v00_orange_clean")
    add("v01_orange_with_plain_foreground", foreground_plain=True, note="foreground plain cube intrudes into A1-style crop")
    add(
        "v02_orange_with_apple_side_intrusion",
        distractor_class="apple",
        distractor_side=True,
        note="nearby apple side face appears in the same crop",
    )
    add(
        "v03_orange_with_apple_top_intrusion",
        distractor_class="apple",
        distractor_top=True,
        distractor_loc=(1.52, -0.18, 0.0),
        note="nearby apple top face appears behind/right of orange",
    )
    add(
        "v04_orange_plain_and_apple_side",
        foreground_plain=True,
        distractor_class="apple",
        distractor_side=True,
        note="plain foreground plus nearby apple side label",
    )
    add(
        "v05_orange_with_pineapple_side_intrusion",
        distractor_class="pineapple",
        distractor_side=True,
        note="nearby pineapple side face appears in the same crop",
    )
    add(
        "v06_orange_plain_and_pineapple_side",
        foreground_plain=True,
        distractor_class="pineapple",
        distractor_side=True,
        note="plain foreground plus nearby pineapple side label",
    )
    add(
        "v07_orange_tight_multiface",
        foreground_plain=True,
        distractor_class="apple",
        distractor_top=True,
        distractor_side=True,
        distractor_loc=(1.46, -0.11, 0.0),
        camera_loc=(3.15, -4.45, 2.08),
        look_at=(0.24, -0.02, 0.11),
        lens=58.0,
        note="tight crop with orange top/side and apple top/side context",
    )
    add(
        "v08_orange_low_angle_apple_context",
        distractor_class="apple",
        distractor_side=True,
        camera_loc=(3.7, -5.15, 1.72),
        look_at=(0.22, 0.0, 0.02),
        lens=55.0,
        note="lower camera angle, apple side context",
    )
    add(
        "v09_orange_context_cluster",
        foreground_plain=True,
        distractor_class="pineapple",
        distractor_top=True,
        distractor_side=True,
        distractor_loc=(1.40, -0.10, 0.0),
        camera_loc=(3.0, -4.55, 2.05),
        look_at=(0.18, -0.03, 0.10),
        lens=60.0,
        note="clustered crop, similar to multi-object conflict frames",
    )
    return variants


def make_solo_variant_list() -> list[dict]:
    base = {
        "camera_loc": (3.35, -4.85, 2.35),
        "look_at": (0.16, -0.02, 0.18),
        "lens": 50.0,
        "light_loc": (-2.2, -3.2, 3.8),
        "light_energy": 420.0,
        "light_size": 4.5,
        "light_color": (1.0, 0.94, 0.84),
        "exposure": -0.22,
        "gamma": 1.04,
        "target_loc": (0.0, 0.0, 0.0),
        "orange_top_size": 0.90,
        "orange_side_size": 0.78,
        "top_label": True,
        "side_label": True,
        "top_xoff": 0.0,
        "top_yoff": -0.05,
        "side_yoff": -0.22,
        "side_zoff": 0.23,
        "foreground_plain": False,
        "foreground_loc": (-0.72, -0.95, -0.02),
        "distractor_class": None,
        "distractor_loc": (1.72, -0.06, 0.0),
        "distractor_top": False,
        "distractor_side": False,
        "note": "solo target orange cube only",
    }
    variants: list[dict] = []

    def add(name: str, **kwargs: object) -> None:
        row = dict(base)
        row.update(kwargs)
        row["name"] = name
        variants.append(row)

    add("s00_solo_clean_top_side")
    add("s01_solo_top_only", side_label=False, note="only top orange label visible")
    add("s02_solo_side_only", top_label=False, note="only right-side orange label visible")
    add(
        "s03_solo_low_angle_side_dominant",
        camera_loc=(3.7, -5.1, 1.70),
        look_at=(0.18, -0.02, 0.02),
        lens=55.0,
        note="lower webcam angle, side label dominates",
    )
    add(
        "s04_solo_high_angle_top_band",
        camera_loc=(3.15, -4.60, 2.70),
        look_at=(0.12, -0.02, 0.32),
        lens=58.0,
        note="higher angle, top label becomes a thin band plus side label",
    )
    add(
        "s05_solo_tight_multiface",
        camera_loc=(3.05, -4.40, 2.02),
        look_at=(0.25, -0.02, 0.08),
        lens=63.0,
        note="tight single cube crop with top and side visible",
    )
    add(
        "s06_solo_label_shifted_forward",
        top_yoff=-0.18,
        side_yoff=-0.34,
        note="orange print shifted toward visible front edge, no outside cube",
    )
    add(
        "s07_solo_label_shifted_back",
        top_yoff=0.08,
        side_yoff=-0.08,
        note="orange print shifted away from visible front edge, no outside cube",
    )
    add(
        "s08_solo_warm_dim_room",
        light_energy=300.0,
        light_color=(1.0, 0.90, 0.76),
        exposure=-0.36,
        gamma=1.06,
        note="same orange texture under mild warm dim room light",
    )
    add(
        "s09_solo_flat_frontish_view",
        camera_loc=(4.10, -5.25, 1.92),
        look_at=(0.46, -0.02, 0.04),
        lens=70.0,
        note="front/side-heavy single cube view similar to A1 crop",
    )
    return variants


def make_solo_a4_variant_list() -> list[dict]:
    base = {
        "camera_loc": (3.35, -4.85, 2.35),
        "look_at": (0.16, -0.02, 0.18),
        "lens": 50.0,
        "light_loc": (-2.2, -3.2, 3.8),
        "light_energy": 390.0,
        "light_size": 4.5,
        "light_color": (1.0, 0.94, 0.84),
        "exposure": -0.24,
        "gamma": 1.04,
        "target_loc": (0.0, 0.0, 0.0),
        "cube_color": (0.82, 0.82, 0.78, 1.0),
        "paper_patch_style": "a4",
        "top_label": True,
        "side_label": True,
        "top_xoff": 0.0,
        "top_yoff": -0.05,
        "side_yoff": -0.22,
        "side_zoff": 0.23,
        "top_paper_width": 1.30,
        "top_paper_height": 1.02,
        "side_paper_width": 0.96,
        "side_paper_height": 1.16,
        "top_fruit_width": 1.00,
        "top_fruit_height": 0.78,
        "side_fruit_width": 0.72,
        "side_fruit_height": 0.90,
        "foreground_plain": False,
        "foreground_loc": (-0.72, -0.95, -0.02),
        "distractor_class": None,
        "distractor_loc": (1.72, -0.06, 0.0),
        "distractor_top": False,
        "distractor_side": False,
        "note": "solo orange cube with off-white cube body and pasted A4-paper label",
    }
    variants: list[dict] = []

    def add(name: str, **kwargs: object) -> None:
        row = dict(base)
        row.update(kwargs)
        row["name"] = name
        variants.append(row)

    add("a00_a4_clean_top_side")
    add("a01_a4_side_only", top_label=False, note="A4 paper on side only")
    add("a02_a4_top_only", side_label=False, note="A4 paper on top only")
    add(
        "a03_a4_small_side_print",
        side_fruit_width=0.54,
        side_fruit_height=0.70,
        note="side A4 paper with smaller printed orange photo",
    )
    add(
        "a04_a4_label_shifted_back",
        top_yoff=0.08,
        side_yoff=-0.08,
        note="A4 paper shifted toward the back of each visible face",
    )
    add(
        "a05_a4_label_shifted_front",
        top_yoff=-0.18,
        side_yoff=-0.34,
        note="A4 paper shifted toward visible front edge",
    )
    add(
        "a06_a4_low_angle_side_dominant",
        camera_loc=(3.7, -5.1, 1.70),
        look_at=(0.18, -0.02, 0.02),
        lens=55.0,
        note="lower webcam angle with A4 side label dominant",
    )
    add(
        "a07_a4_tight_multiface",
        camera_loc=(3.05, -4.40, 2.02),
        look_at=(0.25, -0.02, 0.08),
        lens=63.0,
        note="tight A4 single cube crop with top and side visible",
    )
    add(
        "a08_a4_low_contrast_paper_cube",
        cube_color=(0.88, 0.88, 0.84, 1.0),
        light_energy=340.0,
        exposure=-0.32,
        note="A4 paper and cube body have lower contrast",
    )
    add(
        "a09_a4_frontish_side",
        camera_loc=(4.10, -5.25, 1.92),
        look_at=(0.46, -0.02, 0.04),
        lens=70.0,
        note="front/side-heavy A4 single cube view",
    )
    return variants


def render_variant(output: Path, textures: dict[str, list[Path]], params: dict, resolution: int, samples: int) -> dict:
    clear_scene()
    texture_pick = {
        cls: textures[cls][abs(hash(params["name"] + cls)) % len(textures[cls])]
        for cls in FRUITS
    }
    orange_mat = make_image_mat("target_orange_print", texture_pick["orange"])
    paper_mat = make_mat("paper_label", (0.90, 0.90, 0.87, 1.0), roughness=0.96)
    paper_edge_mat = make_mat("paper_edge", (0.70, 0.70, 0.67, 1.0), roughness=0.96)

    target_loc = params["target_loc"]
    add_cube("target_orange_cube", target_loc, color=tuple(params.get("cube_color", (0.86, 0.86, 0.82, 1.0))))
    if params.get("top_label", True):
        if params.get("paper_patch_style") == "a4":
            add_top_rect_label(
                "target_top_paper_shadow",
                target_loc,
                float(params.get("top_paper_width", 1.18)) * 1.025,
                float(params.get("top_paper_height", 0.92)) * 1.025,
                paper_edge_mat,
                xoff=float(params.get("top_xoff", 0.0)),
                yoff=float(params.get("top_yoff", -0.05)),
                zlift=0.022,
            )
            add_top_rect_label(
                "target_top_a4_paper",
                target_loc,
                float(params.get("top_paper_width", 1.18)),
                float(params.get("top_paper_height", 0.92)),
                paper_mat,
                xoff=float(params.get("top_xoff", 0.0)),
                yoff=float(params.get("top_yoff", -0.05)),
                zlift=0.030,
            )
            add_top_rect_label(
                "target_top_orange_print",
                target_loc,
                float(params.get("top_fruit_width", 0.76)),
                float(params.get("top_fruit_height", 0.58)),
                orange_mat,
                xoff=float(params.get("top_xoff", 0.0)),
                yoff=float(params.get("top_yoff", -0.05)),
                zlift=0.040,
            )
        else:
            add_top_label(
                "target_top_paper_edge",
                target_loc,
                params["orange_top_size"] * 1.05,
                paper_edge_mat,
                xoff=float(params.get("top_xoff", 0.0)),
                yoff=float(params.get("top_yoff", -0.05)),
                zlift=0.024,
            )
            add_top_label(
                "target_top_orange",
                target_loc,
                params["orange_top_size"],
                orange_mat,
                xoff=float(params.get("top_xoff", 0.0)),
                yoff=float(params.get("top_yoff", -0.05)),
                zlift=0.034,
            )
    if params.get("side_label", True):
        if params.get("paper_patch_style") == "a4":
            add_right_rect_label(
                "target_side_paper_shadow",
                target_loc,
                float(params.get("side_paper_width", 0.78)) * 1.025,
                float(params.get("side_paper_height", 1.06)) * 1.025,
                paper_edge_mat,
                yoff=float(params.get("side_yoff", -0.22)),
                zoff=float(params.get("side_zoff", 0.23)),
                xlift=0.022,
            )
            add_right_rect_label(
                "target_side_a4_paper",
                target_loc,
                float(params.get("side_paper_width", 0.78)),
                float(params.get("side_paper_height", 1.06)),
                paper_mat,
                yoff=float(params.get("side_yoff", -0.22)),
                zoff=float(params.get("side_zoff", 0.23)),
                xlift=0.030,
            )
            add_right_rect_label(
                "target_side_orange_print",
                target_loc,
                float(params.get("side_fruit_width", 0.54)),
                float(params.get("side_fruit_height", 0.74)),
                orange_mat,
                yoff=float(params.get("side_yoff", -0.22)),
                zoff=float(params.get("side_zoff", 0.23)),
                xlift=0.040,
            )
        else:
            add_right_label(
                "target_side_paper_edge",
                target_loc,
                params["orange_side_size"] * 1.05,
                paper_edge_mat,
                yoff=float(params.get("side_yoff", -0.22)),
                zoff=float(params.get("side_zoff", 0.23)),
                xlift=0.024,
            )
            add_right_label(
                "target_side_orange",
                target_loc,
                params["orange_side_size"],
                orange_mat,
                yoff=float(params.get("side_yoff", -0.22)),
                zoff=float(params.get("side_zoff", 0.23)),
                xlift=0.034,
            )

    if params["foreground_plain"]:
        add_cube("foreground_plain_cube", params["foreground_loc"], color=(0.88, 0.88, 0.84, 1.0))

    distractor = params.get("distractor_class")
    if distractor:
        loc = params["distractor_loc"]
        add_cube(f"distractor_{distractor}_cube", loc)
        dmat = make_image_mat(f"distractor_{distractor}_print", texture_pick[distractor])
        if params["distractor_top"]:
            add_top_label(f"distractor_{distractor}_top_edge", loc, 0.82, paper_edge_mat, xoff=0.02, yoff=-0.04, zlift=0.024)
            add_top_label(f"distractor_{distractor}_top", loc, 0.78, dmat, xoff=0.02, yoff=-0.04, zlift=0.034)
        if params["distractor_side"]:
            add_right_label(f"distractor_{distractor}_side_edge", loc, 0.72, paper_edge_mat, yoff=-0.20, zoff=0.22, xlift=0.024)
            add_right_label(f"distractor_{distractor}_side", loc, 0.68, dmat, yoff=-0.20, zoff=0.22, xlift=0.034)

    floor_mat = make_mat("desk_floor", (0.58, 0.48, 0.38, 1.0), roughness=0.78)
    bpy.ops.mesh.primitive_plane_add(size=8.0, location=(0.0, 0.0, -1.01))
    floor = bpy.context.object
    floor.name = "desk_floor"
    floor.data.materials.append(floor_mat)

    add_scene_light_and_camera(params)

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

    output.mkdir(parents=True, exist_ok=True)
    image_path = output / f"{params['name']}.png"
    scene.render.filepath = str(image_path)
    bpy.ops.render.render(write_still=True)
    return {
        "name": params["name"],
        "image": str(image_path),
        "note": params["note"],
        "foreground_plain": bool(params["foreground_plain"]),
        "distractor_class": params.get("distractor_class"),
        "distractor_top": bool(params.get("distractor_top")),
        "distractor_side": bool(params.get("distractor_side")),
        "textures": {cls: str(path) for cls, path in texture_pick.items()},
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    bproc.init()
    args.output = args.output.resolve()
    args.texture_root = args.texture_root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    textures = collect_textures(args.texture_root)
    if args.mode == "solo":
        variants = make_solo_variant_list()
    elif args.mode == "solo_a4":
        variants = make_solo_a4_variant_list()
    else:
        variants = make_variant_list()
    rows = [
        render_variant(args.output, textures, variant, args.resolution, args.samples)
        for variant in variants
    ]
    (args.output / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "count": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
