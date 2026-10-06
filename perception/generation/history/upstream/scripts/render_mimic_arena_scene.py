import blenderproc as bproc
import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy
import cv2
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Euler, Quaternion, Vector

# Repo-standard printed-label generator (varied realistic_a4_* profiles).
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from generate_yolo_coco_composite import make_fruit_print_label_texture

# Standalone preview renderer for the real 4m x 4m competition arena
# ("mimic arena"), following docs/arena_final_environment.md.
#
# Run (same mechanism as the other blenderproc probe scripts /
# run_yolo_parallel.py resolve_blenderproc()):
#   C:\Users\user\anaconda3\envs\ai_robotics\Scripts\blenderproc.exe run ^
#       scripts\render_mimic_arena_scene.py ^
#       --output reports\arena_mimic_preview_v1 --num_images 10
#
# CPU rendering only (GPU is busy training).

FRUITS = ("apple", "orange", "banana", "pineapple")
POLYHEDRA = ("plain_cube", "octahedron", "dodecahedron", "icosahedron")

# A1 ("Cube Detector") whole-object detector classes, in the EXACT order of
# datasets/meta_v2_50000_coco_texture_v1_models/a1_objectseg/data.yaml. Every
# cube (the 4 set-1 plain cubes + the 12 fruit-print cubes) is one
# "cube_like_object"; each non-cube polyhedron keeps its own class.
A1_CLASSES = ("cube_like_object", "octahedron", "dodecahedron", "icosahedron")
A1_MIN_BBOX_AREA = 200.0  # px^2 on-screen gate, applied to cubes AND polyhedra


def a1_class_for_kind(kind: str):
    """Map a placed-object 'kind' (see place_objects records) to its A1 class
    name, or None if the object is not an A1 detection target."""
    if kind in ("fruit_cube", "plain_cube"):
        return "cube_like_object"
    if kind in ("octahedron", "dodecahedron", "icosahedron"):
        return kind
    return None

# Per-cube print-label style mix (repo-standard profiles, strength "light").
PRINT_PROFILE_MIX = (
    ("realistic_a4_mild", 0.35),
    ("realistic_a4_fruit_visible", 0.25),
    ("realistic_a4_sparse_icon", 0.25),
    ("realistic_a4_label_offset", 0.15),
)
PRINT_STRENGTH = "light"

ARENA_HALF = 2.0          # 4m x 4m floor
FENCE_HEIGHT = 0.29       # SUN-168 fence
FENCE_THICKNESS = 0.018
OBJECT_HEIGHT = 0.08      # lying height 8cm
STORAGE_SIZE = 0.4        # storage zone at corner (-2, -2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render realistic mimic-arena preview images (no labels).")
    parser.add_argument("--output", type=Path, default=Path("reports/arena_mimic_preview_v1"))
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--approach", type=int, default=0,
                        help="Make the first N frames pickup-approach close-ups of a random fruit cube.")
    parser.add_argument("--frames_per_layout", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--asset_dir", type=Path, default=Path("assets/generated"))
    parser.add_argument(
        "--texture_root",
        type=Path,
        default=Path("datasets/fruit_textures/production_meta_v2_50000_v1_color_filtered_v2"),
    )
    parser.add_argument("--background_dir", type=Path, default=Path("datasets/backgrounds/coco2017"))
    parser.add_argument("--floor_photo", type=Path, default=Path("docs/reference_photos/KakaoTalk_20260707_092558292.jpg"),
                        help="Real arena photo used to derive the floor texture (falls back to procedural).")
    parser.add_argument("--light_energy", type=float, default=135.0)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Procedural texture generation (numpy / cv2)
# ---------------------------------------------------------------------------

def make_sun111_floor_texture(path: Path, size: int = 2048) -> None:
    """SUN-111 walnut/teak laminate matched to the real arena photo
    (docs/reference_photos/KakaoTalk_20260707_092558292.jpg). Sampled floor RGB
    percentiles (cv2):
    light stripes p50~(183,158,140) / p75~(190,165,147); dark grain streaks
    p25~(122,89,54) / p5~(102,72,39). Strong, irregular lengthwise grain."""
    light = np.array([183, 158, 140], dtype=np.float32)
    mid = np.array([150, 118, 88], dtype=np.float32)
    dark = np.array([120, 86, 54], dtype=np.float32)
    xdark = np.array([100, 70, 42], dtype=np.float32)

    # Irregular stripe profile (widths 4-25 px): broad light areas broken by
    # thin dark streaks, like the real walnut laminate.
    profile = np.empty(size, dtype=np.float32)
    x0 = 0
    while x0 < size:
        r = random.random()
        if r < 0.55:
            wpx = random.randint(8, 25)
            t = random.uniform(0.78, 1.0)     # broad light stripe
        elif r < 0.82:
            wpx = random.randint(5, 15)
            t = random.uniform(0.45, 0.78)    # mid tone
        elif r < 0.96:
            wpx = random.randint(4, 10)
            t = random.uniform(0.14, 0.45)    # thin dark streak
        else:
            wpx = random.randint(4, 7)
            t = random.uniform(0.0, 0.14)     # extra-dark hairline
        profile[x0:x0 + wpx] = t
        x0 += wpx
    profile = cv2.GaussianBlur(profile[None, :], (0, 0), 1.1)[0]

    t_img = profile[None, :].repeat(size, axis=0)
    # Waviness + slow drift so the grain is not perfectly straight.
    y = np.linspace(0.0, 1.0, size, dtype=np.float32)
    wave = (np.sin(y * math.pi * random.uniform(2.5, 5.0)) * random.uniform(3.0, 8.0)
            + np.sin(y * math.pi * random.uniform(9.0, 16.0)) * random.uniform(1.0, 2.5))[:, None]
    shift = (wave + np.random.normal(0, 0.7, (size, 1))).astype(np.float32)
    map_x = np.clip(np.arange(size, dtype=np.float32)[None, :] + shift, 0, size - 1)
    map_y = np.arange(size, dtype=np.float32)[:, None].repeat(size, axis=1)
    t_img = cv2.remap(t_img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    t_img += np.random.normal(0, 0.035, (size, size)).astype(np.float32)
    t_img = np.clip(cv2.GaussianBlur(t_img, (0, 0), 0.8), 0.0, 1.0)

    # Piecewise color LUT: xdark -> dark -> mid -> light.
    knots_t = np.array([0.0, 0.33, 0.66, 1.0], dtype=np.float32)
    knots_c = np.stack([xdark, dark, mid, light], axis=0)
    img = np.stack(
        [np.interp(t_img, knots_t, knots_c[:, ch]) for ch in range(3)], axis=-1
    ).astype(np.float32)

    # Mild low-frequency tone variation between areas.
    blotch = np.random.normal(0, 1.0, (size // 16, size // 16)).astype(np.float32)
    blotch = cv2.resize(blotch, (size, size), interpolation=cv2.INTER_CUBIC)
    blotch = cv2.GaussianBlur(blotch, (0, 0), 24.0)
    img *= (1.0 + blotch[..., None] * 0.045)

    # Subtle plank borders along the grain direction.
    plank_w = random.randint(size // 14, size // 9)
    xb = random.randint(0, plank_w)
    while xb < size:
        cv2.line(img, (xb, 0), (xb + random.randint(-3, 3), size),
                 (dark * random.uniform(0.75, 0.95)).tolist(), 2, cv2.LINE_AA)
        img[:, xb:min(size, xb + plank_w)] *= random.uniform(0.97, 1.03)
        xb += plank_w + random.randint(-8, 8)

    # 1-3 strong straight plate seam lines across the floor.
    for _ in range(random.randint(1, 3)):
        yline = random.randint(int(size * 0.2), int(size * 0.8))
        cv2.line(img, (0, yline), (size, yline + random.randint(-4, 4)),
                 (xdark * random.uniform(0.65, 0.85)).tolist(), random.choice([3, 4]), cv2.LINE_AA)

    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])


def make_floor_texture_from_photo(photo_path: Path, path: Path, size: int = 3072) -> bool:
    """Build the floor texture from the real arena photo: crop the lower-left
    floor region (bottom ~30%, smallest perspective distortion, no flags/tape),
    mildly rectify perspective, flatten illumination, restore the sheen-washed
    saturation, then mirror-tile to `size` with +-4% per-tile brightness and
    faint plank/plate seam lines. Grain runs along one axis. Returns False if
    the photo is unavailable (caller falls back to the procedural texture)."""
    photo = cv2.imread(str(photo_path), cv2.IMREAD_COLOR)
    if photo is None:
        return False
    h, w = photo.shape[:2]
    x0, x1 = int(0.02 * w), int(0.62 * w)
    y0, y1 = int(0.68 * h), int(0.985 * h)
    crop = photo[y0:y1, x0:x1].astype(np.float32)
    ch, cw = crop.shape[:2]

    # Mild perspective rectification: grain converges slightly toward the top
    # of the crop; stretch the top edge outward so the grain runs parallel.
    inset = 0.10
    src = np.float32([[cw * inset, 0], [cw * (1 - inset), 0], [cw, ch], [0, ch]])
    dst = np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]])
    warp = cv2.getPerspectiveTransform(src, dst)
    crop = cv2.warpPerspective(crop, warp, (cw, ch), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)

    # Flatten low-frequency illumination so tiles do not show blocks.
    lf = cv2.GaussianBlur(crop, (0, 0), min(ch, cw) / 6.0)
    crop = np.clip(crop - lf + lf.mean(axis=(0, 1), keepdims=True), 0, 255)

    # This photo region is washed out by the sheet sheen: restore saturation
    # and grain contrast toward the photo's mid-floor tone.
    hsv = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.28, 0, 255)
    crop = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    mean_c = crop.mean(axis=(0, 1), keepdims=True)
    crop = np.clip((crop - mean_c) * 1.10 + mean_c * 0.97, 0, 255)

    # Mirror-tile with slight per-tile brightness variation (+-4%).
    tiles_x = math.ceil(size / (2 * cw))
    tiles_y = math.ceil(size / (2 * ch))
    rows = []
    for _ in range(tiles_y):
        row_tiles = []
        for _ in range(tiles_x):
            block_a = crop * random.uniform(0.96, 1.04)
            block_b = cv2.flip(crop, 1) * random.uniform(0.96, 1.04)
            block = cv2.hconcat([block_a, block_b])
            block = cv2.vconcat([block, cv2.flip(block, 0) * random.uniform(0.97, 1.03)])
            row_tiles.append(block)
        rows.append(cv2.hconcat(row_tiles))
    big = cv2.vconcat(rows)[:size, :size]

    # Faint plank seam lines along the grain + 1-2 plate seams across it.
    seam_color = (big.reshape(-1, 3).mean(axis=0) * 0.55).tolist()
    plank_w = random.randint(size // 12, size // 8)
    xb = random.randint(0, plank_w)
    while xb < size:
        cv2.line(big, (xb, 0), (xb + random.randint(-4, 4), size), seam_color, 2, cv2.LINE_AA)
        xb += plank_w + random.randint(-10, 10)
    for _ in range(random.randint(1, 2)):
        yline = random.randint(int(size * 0.25), int(size * 0.75))
        cv2.line(big, (0, yline), (size, yline + random.randint(-4, 4)),
                 (np.array(seam_color) * 0.6).tolist(), 3, cv2.LINE_AA)

    cv2.imwrite(str(path), np.clip(big, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return True


def make_sun168_fence_texture(path: Path, width: int = 1024, height: int = 256) -> None:
    """Light beige SUN-168 laminate with subtle horizontal grain. Color sampled
    from the real arena photo fence panels: p50~(143,133,118), p75~(151,143,134)."""
    base_rgb = np.array([158, 148, 132], dtype=np.float32)
    img = np.ones((height, width, 3), dtype=np.float32) * base_rgb[None, None, :]

    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)
    grain = np.sin((yy * random.uniform(24, 40) + random.random()) * math.pi * 2)
    grain += 0.5 * np.sin((yy * random.uniform(60, 95) + random.random()) * math.pi * 2)
    grain = grain[:, None].repeat(width, axis=1)
    grain += np.random.normal(0, 0.22, (height, width)).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (0, 0), 1.2)
    img += grain[..., None] * np.array([6.0, 5.5, 4.5], dtype=np.float32)

    # A few faint horizontal grain lines like the laminate print.
    for _ in range(random.randint(4, 8)):
        yl = random.randint(0, height - 1)
        cv2.line(img, (0, yl), (width, yl + random.randint(-2, 2)),
                 (base_rgb * random.uniform(0.88, 0.95)).tolist(), 1, cv2.LINE_AA)

    noise = np.random.normal(0, 1.6, (height, width)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 2.4)
    img += noise[..., None]
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])


def make_concrete_texture(path: Path, size: int = 1024) -> None:
    """Workshop concrete floor sampled from the photo: p50~(204,211,208),
    with stains and blotches."""
    base = np.array([200, 207, 203], dtype=np.float32)
    img = np.ones((size, size, 3), dtype=np.float32) * base[None, None, :]
    blotch = np.random.normal(0, 1.0, (size // 8, size // 8)).astype(np.float32)
    blotch = cv2.resize(blotch, (size, size), interpolation=cv2.INTER_CUBIC)
    blotch = cv2.GaussianBlur(blotch, (0, 0), 14.0)
    img *= (1.0 + blotch[..., None] * 0.08)
    # Darker stains / patches.
    for _ in range(random.randint(8, 16)):
        cx, cy = random.randint(0, size), random.randint(0, size)
        ax = random.randint(size // 30, size // 8)
        ay = random.randint(size // 30, size // 8)
        stain = np.zeros((size, size), dtype=np.float32)
        cv2.ellipse(stain, (cx, cy), (ax, ay), random.uniform(0, 180), 0, 360, 1.0, -1)
        stain = cv2.GaussianBlur(stain, (0, 0), random.uniform(6, 20))
        img *= (1.0 - stain[..., None] * random.uniform(0.06, 0.22))
    noise = np.random.normal(0, 3.0, (size, size)).astype(np.float32)
    img += cv2.GaussianBlur(noise, (0, 0), 1.5)[..., None]
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])


def make_backdrop_texture(coco_paths: list, path: Path, width: int = 3072, height: int = 768) -> None:
    """Continuous workshop backdrop strip: COCO images tiled side by side,
    desaturated and darkened so they read as background clutter, not posters."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x0 = 0
    while x0 < width:
        img = cv2.imread(str(random.choice(coco_paths)), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = height / h
        nw = max(64, int(w * scale))
        tile = cv2.resize(img, (nw, height), interpolation=cv2.INTER_AREA)
        tile = tile[:, : min(nw, width - x0)]
        canvas[:, x0:x0 + tile.shape[1]] = tile
        x0 += tile.shape[1]
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
    out = canvas.astype(np.float32) * 0.55 + gray3 * 0.45   # desaturate
    out *= 0.62                                             # darker than arena
    out = cv2.GaussianBlur(out, (0, 0), 1.2)
    cv2.imwrite(str(path), np.clip(out, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 90])


def make_korean_flag_texture(path: Path, width: int = 512, height: int = 358) -> None:
    """Small Korean-flag sticker approximation: white field, red-over-blue
    taegeuk disc, 4 black trigram bar groups near the corners."""
    img = np.full((height, width, 3), 252, dtype=np.uint8)
    cx, cy = width // 2, height // 2
    r = int(height * 0.25)
    red = (222, 60, 48)
    blue = (28, 70, 158)
    # Upper half disc red, lower half disc blue (yin-yang approximation),
    # slightly tilted like the real taegeuk.
    tilt = -18
    cv2.ellipse(img, (cx, cy), (r, r), tilt, 180, 360, red, -1, cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy), (r, r), tilt, 0, 180, blue, -1, cv2.LINE_AA)
    # Small yin-yang lobes.
    ang = math.radians(tilt)
    ox, oy = int(r / 2 * math.cos(ang)), int(r / 2 * math.sin(ang))
    cv2.circle(img, (cx - ox, cy - oy), r // 2, red, -1, cv2.LINE_AA)
    cv2.circle(img, (cx + ox, cy + oy), r // 2, blue, -1, cv2.LINE_AA)

    # 4 trigram bar groups (3 short bars each), rotated 45 deg toward center.
    black = (20, 20, 20)
    bar_l, bar_t, gap = int(height * 0.22), max(3, height // 28), max(6, height // 16)
    for corner_x, corner_y, angle in (
        (int(width * 0.16), int(height * 0.22), 55),
        (int(width * 0.84), int(height * 0.22), -55),
        (int(width * 0.16), int(height * 0.78), -55),
        (int(width * 0.84), int(height * 0.78), 55),
    ):
        a = math.radians(angle)
        dx, dy = math.cos(a) * bar_l / 2, math.sin(a) * bar_l / 2
        nx, ny = -math.sin(a), math.cos(a)
        for k in (-1, 0, 1):
            ox2, oy2 = nx * gap * k, ny * gap * k
            p0 = (int(corner_x - dx + ox2), int(corner_y - dy + oy2))
            p1 = (int(corner_x + dx + ox2), int(corner_y + dy + oy2))
            cv2.line(img, p0, p1, black, bar_t, cv2.LINE_AA)

    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])


def choose_print_profile() -> str:
    r = random.random()
    acc = 0.0
    for name, weight in PRINT_PROFILE_MIX:
        acc += weight
        if r < acc:
            return name
    return PRINT_PROFILE_MIX[-1][0]


def make_print_label_face_textures(texture_root: Path, cls: str, out_dir: Path, tag: str):
    """3 printed-label face textures for one cube using the repo-standard
    generator (generate_yolo_coco_composite.make_fruit_print_label_texture).
    One style profile per cube, same class on all 3 faces."""
    profile = choose_print_profile()
    paths = []
    for face_idx in range(3):
        image, _sources = make_fruit_print_label_texture(
            str(texture_root), cls, PRINT_STRENGTH, profile=profile)
        path = out_dir / f"print_{tag}_{cls}_{face_idx}_{random.getrandbits(32):08x}.jpg"
        cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(90, 97)])
        paths.append(path)
    return profile, paths


# ---------------------------------------------------------------------------
# Blender helpers (same conventions as the probe scripts)
# ---------------------------------------------------------------------------

def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    # Drop orphaned meshes/materials/images so 100-scene runs do not bloat.
    try:
        for _ in range(3):
            bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    except Exception:
        pass


def make_mat(name: str, color, roughness: float = 0.7, metallic: float = 0.0) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = tuple(color)
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic
    return mat


def make_image_mat(name: str, image_path: Path, roughness: float = 0.85, coat: float = 0.0) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    bsdf = tree.nodes.get("Principled BSDF")
    coord = tree.nodes.new("ShaderNodeTexCoord")
    tex = tree.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(Path(image_path).resolve()), check_existing=True)
    tree.links.new(coord.outputs["UV"], tex.inputs["Vector"])
    if bsdf is not None:
        tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = roughness
        if coat > 0.0:
            try:
                bsdf.inputs["Coat Weight"].default_value = coat
            except KeyError:
                pass
    return mat


def ensure_unit_uv(obj: bpy.types.Object) -> None:
    mesh = obj.data
    uv_layer = mesh.uv_layers.active or mesh.uv_layers.new(name="UVMap")
    coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    for poly in mesh.polygons:
        for idx, loop_index in enumerate(poly.loop_indices):
            uv_layer.data[loop_index].uv = coords[idx % 4]


def add_box(name: str, dims, loc, mat: bpy.types.Material) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.scale = (dims[0], dims[1], dims[2])
    obj.data.materials.append(mat)
    return obj


def add_decal_plane(name: str, width: float, height: float, loc, rot, mat: bpy.types.Material) -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=loc, rotation=rot)
    obj = bpy.context.object
    obj.name = name
    obj.scale = (width, height, 1.0)
    ensure_unit_uv(obj)
    obj.data.materials.append(mat)
    return obj


def look_at(obj: bpy.types.Object, target) -> None:
    direction = Vector(target) - Vector(obj.location)
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


# ---------------------------------------------------------------------------
# Arena construction
# ---------------------------------------------------------------------------

def build_static_arena(tex: dict, backdrop_paths: list, light_energy: float) -> None:
    # Surrounding room half-size: walls 3-5m beyond the fence.
    room_half = ARENA_HALF + random.uniform(3.0, 4.5)

    # Floor: 4x4m SUN-111 wood; the transparent-sheet sheen is kept subtle.
    floor_mat = make_image_mat("sun111_floor", tex["floor"], roughness=0.55, coat=0.05)
    bpy.ops.mesh.primitive_plane_add(size=ARENA_HALF * 2.0, location=(0.0, 0.0, 0.0))
    floor = bpy.context.object
    floor.name = "arena_floor"
    ensure_unit_uv(floor)
    floor.data.materials.append(floor_mat)

    # Outside ground: workshop concrete extending to the backdrop walls.
    concrete = make_image_mat("workshop_concrete", tex["concrete"], roughness=0.9)
    bpy.ops.mesh.primitive_plane_add(size=room_half * 2.0 + 0.2, location=(0.0, 0.0, -0.003))
    ground = bpy.context.object
    ground.name = "outside_ground"
    ensure_unit_uv(ground)
    ground.data.materials.append(concrete)

    # Continuous backdrop room: 4 large COCO-textured walls + gray ceiling, so
    # there is no void between the fence top and the background.
    wall_h = 4.5
    wall_len = room_half * 2.0 + 0.2
    wall_defs = (
        ((0.0, room_half, wall_h / 2.0), (math.radians(90.0), 0.0, 0.0)),
        ((0.0, -room_half, wall_h / 2.0), (math.radians(90.0), 0.0, math.radians(180.0))),
        ((room_half, 0.0, wall_h / 2.0), (math.radians(90.0), 0.0, math.radians(-90.0))),
        ((-room_half, 0.0, wall_h / 2.0), (math.radians(90.0), 0.0, math.radians(90.0))),
    )
    for i, (loc, rot) in enumerate(wall_defs):
        mat = make_image_mat(f"backdrop_wall_{i}_{random.getrandbits(16):04x}",
                             backdrop_paths[i % len(backdrop_paths)], roughness=0.95)
        add_decal_plane(f"backdrop_wall_{i}", wall_len, wall_h, loc, rot, mat)
    ceiling = make_mat("workshop_ceiling", (0.52, 0.53, 0.55, 1.0), roughness=0.95)
    bpy.ops.mesh.primitive_plane_add(size=wall_len, location=(0.0, 0.0, wall_h + 0.05),
                                     rotation=(math.radians(180.0), 0.0, 0.0))
    ceil_obj = bpy.context.object
    ceil_obj.name = "backdrop_ceiling"
    ceil_obj.data.materials.append(ceiling)

    # Fence: 4 walls, SUN-168 beige with subtle horizontal grain. Small bevel
    # so the 15-18mm top-edge lip catches the overhead light.
    fence_mat = make_image_mat("sun168_fence", tex["fence"], roughness=0.62)
    t = FENCE_THICKNESS
    h = FENCE_HEIGHT
    length = ARENA_HALF * 2.0 + 2.0 * t
    for name, dims, loc in (
        ("fence_north", (length, t, h), (0.0, ARENA_HALF + t / 2.0, h / 2.0)),
        ("fence_south", (length, t, h), (0.0, -ARENA_HALF - t / 2.0, h / 2.0)),
        ("fence_east", (t, length, h), (ARENA_HALF + t / 2.0, 0.0, h / 2.0)),
        ("fence_west", (t, length, h), (-ARENA_HALF - t / 2.0, 0.0, h / 2.0)),
    ):
        wall = add_box(name, dims, loc, fence_mat)
        try:
            bevel = wall.modifiers.new("fence_edge_bevel", "BEVEL")
            bevel.width = 0.0018
            bevel.segments = 2
        except Exception:
            pass

    # Aluminum joint plates at all 4 inside corners (like the photo): light
    # gray metal strips protruding ~3-4mm inward with 2 bolt dots per leg.
    # Low metallic so the plate reads light gray under the dim surroundings
    # (fully metallic renders near-black with a dark world background).
    alu = make_mat("aluminum_plate", (0.70, 0.71, 0.73, 1.0), roughness=0.42, metallic=0.25)
    bolt_mat = make_mat("alu_bolt", (0.45, 0.45, 0.47, 1.0), roughness=0.40, metallic=0.35)
    for cx, cy in ((ARENA_HALF, ARENA_HALF), (-ARENA_HALF, ARENA_HALF),
                   (ARENA_HALF, -ARENA_HALF), (-ARENA_HALF, -ARENA_HALF)):
        sx = 1.0 if cx > 0 else -1.0
        sy = 1.0 if cy > 0 else -1.0
        leg = 0.065
        add_box(f"alu_x_{sx:+.0f}{sy:+.0f}", (0.004, leg, h - 0.012),
                (cx - sx * 0.002, cy - sy * (leg / 2.0 + 0.001), h / 2.0), alu)
        add_box(f"alu_y_{sx:+.0f}{sy:+.0f}", (leg, 0.004, h - 0.012),
                (cx - sx * (leg / 2.0 + 0.001), cy - sy * 0.002, h / 2.0), alu)
        for bz in (0.075, 0.215):
            bpy.ops.mesh.primitive_uv_sphere_add(
                radius=0.0048, segments=16, ring_count=8,
                location=(cx - sx * 0.0055, cy - sy * (leg / 2.0 + 0.001), bz))
            bpy.context.object.data.materials.append(bolt_mat)
            bpy.ops.mesh.primitive_uv_sphere_add(
                radius=0.0048, segments=16, ring_count=8,
                location=(cx - sx * (leg / 2.0 + 0.001), cy - sy * 0.0055, bz))
            bpy.context.object.data.materials.append(bolt_mat)

    # Storage zone at corner (-2, -2): two black strips (10mm wide, 3mm high)
    # on the two open sides of the 0.4x0.4m square.
    black = make_mat("storage_border_black", (0.015, 0.015, 0.015, 1.0), roughness=0.55)
    sx = -ARENA_HALF + STORAGE_SIZE  # open side x = -1.6
    sy = -ARENA_HALF + STORAGE_SIZE  # open side y = -1.6
    add_box("storage_strip_x", (0.01, STORAGE_SIZE + 0.01, 0.003),
            (sx, -ARENA_HALF + STORAGE_SIZE / 2.0, 0.0015), black)
    add_box("storage_strip_y", (STORAGE_SIZE + 0.01, 0.01, 0.003),
            (-ARENA_HALF + STORAGE_SIZE / 2.0, sy, 0.0015), black)

    # Korean-flag stickers: ~6 on the two fence walls near the corner,
    # 4 flat on the floor inside the zone. Sticker ~5 x 3.5 cm.
    flag_mat = make_image_mat("korean_flag_sticker", tex["flag"], roughness=0.6)
    flag_w, flag_h = 0.05, 0.035
    for i in range(3):  # west wall (inner face at x = -2, faces +X)
        y = -ARENA_HALF + 0.07 + i * 0.13 + random.uniform(-0.015, 0.015)
        z = random.uniform(0.15, 0.23)
        add_decal_plane(f"flag_wall_w_{i}", flag_w, flag_h,
                        (-ARENA_HALF + 0.0015, y, z),
                        (math.radians(90.0), 0.0, math.radians(90.0)), flag_mat)
    for i in range(3):  # south wall (inner face at y = -2, faces +Y)
        x = -ARENA_HALF + 0.07 + i * 0.13 + random.uniform(-0.015, 0.015)
        z = random.uniform(0.15, 0.23)
        add_decal_plane(f"flag_wall_s_{i}", flag_w, flag_h,
                        (x, -ARENA_HALF + 0.0015, z),
                        (math.radians(90.0), 0.0, 0.0), flag_mat)
    for i in range(4):  # floor inside the zone
        fx = random.uniform(-ARENA_HALF + 0.06, -ARENA_HALF + STORAGE_SIZE - 0.06)
        fy = random.uniform(-ARENA_HALF + 0.06, -ARENA_HALF + STORAGE_SIZE - 0.06)
        add_decal_plane(f"flag_floor_{i}", flag_w, flag_h,
                        (fx, fy, 0.0012),
                        (0.0, 0.0, random.uniform(0.0, math.pi * 2)), flag_mat)

    # 4 cool-white circular high-bay area lights in a rectangle near the arena
    # center. Dominant light source (world/ambient kept very low) so objects
    # cast visible soft shadows with center-weighted brightness pooling.
    for bx, by in ((0.8, 0.7), (-0.8, 0.7), (0.8, -0.7), (-0.8, -0.7)):
        bpy.ops.object.light_add(type="AREA", location=(
            bx + random.uniform(-0.08, 0.08),
            by + random.uniform(-0.08, 0.08),
            random.uniform(3.1, 3.6),
        ))
        light = bpy.context.object
        light.name = f"highbay_{bx:+.1f}_{by:+.1f}"
        try:
            light.data.shape = "DISK"
        except Exception:
            pass
        light.data.size = 0.55
        light.data.energy = light_energy * random.uniform(0.90, 1.10)
        light.data.color = (0.95, 0.97, 1.0)

    # Very dim ambient so the disk lights dominate the shading.
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("arena_world")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (0.025, 0.025, 0.028, 1.0)
        bg.inputs[1].default_value = 1.0
    world.color = (0.025, 0.025, 0.028)


# ---------------------------------------------------------------------------
# Competition objects
# ---------------------------------------------------------------------------

def load_obj_mesh(path: Path):
    objs = bproc.loader.load_obj(str(path))
    return objs[0].blender_obj


def face_down_rotation(obj: bpy.types.Object) -> Euler:
    """Rotate the object so a random face rests on the floor, plus a random spin."""
    polys = obj.data.polygons
    normal = Vector(polys[random.randrange(len(polys))].normal).normalized()
    quat = normal.rotation_difference(Vector((0.0, 0.0, -1.0)))
    spin = Quaternion(Vector((0.0, 0.0, 1.0)), random.uniform(0.0, math.pi * 2))
    return (spin @ quat).to_euler()


def settle_on_floor(obj: bpy.types.Object, target_height: float = OBJECT_HEIGHT):
    """Scale so lying height == target_height and drop onto z = 0.
    Returns (half_extent_xy, applied_scale)."""
    rot = np.array(obj.rotation_euler.to_matrix())
    verts = np.array([v.co[:] for v in obj.data.vertices], dtype=np.float64)
    verts = verts * np.array(obj.scale)[None, :]
    world = verts @ rot.T
    z_extent = world[:, 2].max() - world[:, 2].min()
    factor = target_height / max(z_extent, 1e-6)
    obj.scale = tuple(s * factor for s in obj.scale)
    world *= factor
    z_off = -world[:, 2].min()
    half_xy = 0.5 * max(world[:, 0].max() - world[:, 0].min(), world[:, 1].max() - world[:, 1].min())
    return half_xy, z_off


def white_plastic() -> bpy.types.Material:
    shade = random.uniform(0.82, 0.92)
    return make_mat(f"white_pla_{random.getrandbits(24):06x}", (shade, shade, shade * 0.985, 1.0),
                    roughness=random.uniform(0.55, 0.8))


def add_object_bevel(obj: bpy.types.Object) -> None:
    try:
        bevel = obj.modifiers.new("tiny_bevel", "BEVEL")
        bevel.width = 0.0012
        bevel.segments = 2
        obj.modifiers.new("weighted_normals", "WEIGHTED_NORMAL")
    except Exception:
        pass


def create_polyhedron(asset_dir: Path, shape: str, idx: int) -> bpy.types.Object:
    obj = load_obj_mesh(asset_dir / f"{shape}.obj")
    obj.name = f"{shape}_{idx:02d}"
    obj.data.materials.clear()
    obj.data.materials.append(white_plastic())
    add_object_bevel(obj)
    return obj

# Fruit prints go on +X / -X / +Y: the real cubes have the 3 printed faces in a
# "ㄷ" (U) arrangement -- two opposite side faces plus the bridging face -- so at
# least one fruit face is visible from any viewing direction. Same convention as
# FRUIT_FACE_NORMALS in scripts/generate_yolo_coco_composite.py.
FRUIT_FACE_NORMALS = ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def create_fruit_cube(cls: str, idx: int, face_textures: list) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=OBJECT_HEIGHT, location=(0.0, 0.0, 0.0))
    cube = bpy.context.object
    cube.name = f"fruit_{cls}_{idx:02d}"
    cube.data.materials.append(white_plastic())
    add_object_bevel(cube)

    half = OBJECT_HEIGHT / 2.0
    lift = 0.0006
    for face_idx, normal in enumerate(FRUIT_FACE_NORMALS):
        nx, ny, _ = normal
        if nx > 0:
            loc, rot = (half + lift, 0.0, 0.0), (0.0, math.radians(90.0), 0.0)
        elif nx < 0:
            loc, rot = (-half - lift, 0.0, 0.0), (0.0, math.radians(-90.0), 0.0)
        else:
            loc, rot = (0.0, half + lift, 0.0), (math.radians(-90.0), 0.0, 0.0)
        mat = make_image_mat(
            f"{cube.name}_face{face_idx}_{random.getrandbits(24):06x}",
            face_textures[face_idx % len(face_textures)],
            roughness=0.9,
        )
        bpy.ops.mesh.primitive_plane_add(size=OBJECT_HEIGHT * 0.985, location=loc, rotation=rot)
        face = bpy.context.object
        face.name = f"{cube.name}_face{face_idx}"
        ensure_unit_uv(face)
        face.data.materials.append(mat)
        face.parent = cube
        face.matrix_parent_inverse.identity()
    return cube


def place_objects(asset_dir: Path, texture_root: Path, tex_dir: Path) -> list:
    """16 polyhedra + 12 fruit cubes, random non-touching placement,
    min gap ~3cm, some wall contact. Returns per-object records for the
    label/metadata export."""
    plan = [("shape", shape) for shape in POLYHEDRA for _ in range(4)]
    plan += [("fruit", cls) for cls in FRUITS for _ in range(3)]
    random.shuffle(plan)

    records = []
    placed = []  # (x, y, radius)
    margin = FENCE_THICKNESS
    for idx, (kind, name) in enumerate(plan):
        profile = None
        if kind == "shape":
            obj = create_polyhedron(asset_dir, name, idx)
            obj.rotation_euler = face_down_rotation(obj)
        else:
            profile, textures = make_print_label_face_textures(
                texture_root, name, tex_dir, f"{idx:02d}")
            obj = create_fruit_cube(name, idx, textures)
            # Keep the ㄷ arrangement guarantee: never let a printed face point
            # down. Upright keeps prints on 3 of the 4 side faces; the optional
            # +90deg x-tip moves the +Y bridge print to the TOP (never bottom).
            euler = Euler((0.0, 0.0, random.uniform(0.0, math.pi * 2)))
            if random.random() < 0.35:
                euler = Euler((math.pi / 2, 0.0, random.uniform(0.0, math.pi * 2)))
            obj.rotation_euler = euler
        half_xy, z_off = settle_on_floor(obj)

        pos = None
        for attempt in range(400):
            wall_contact = random.random() < 0.18
            if wall_contact:
                side = random.choice(("n", "s", "e", "w"))
                if side == "n":
                    x, y = random.uniform(-1.8, 1.8), ARENA_HALF - half_xy - 0.001
                elif side == "s":
                    x, y = random.uniform(-1.8, 1.8), -ARENA_HALF + half_xy + 0.001
                elif side == "e":
                    x, y = ARENA_HALF - half_xy - 0.001, random.uniform(-1.8, 1.8)
                else:
                    x, y = -ARENA_HALF + half_xy + 0.001, random.uniform(-1.8, 1.8)
            else:
                x = random.uniform(-ARENA_HALF + half_xy + margin, ARENA_HALF - half_xy - margin)
                y = random.uniform(-ARENA_HALF + half_xy + margin, ARENA_HALF - half_xy - margin)
            ok = all(math.hypot(x - px, y - py) > half_xy + pr + 0.03 for px, py, pr in placed)
            if ok:
                pos = (x, y)
                break
        if pos is None:
            pos = (random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5))
        obj.location = (pos[0], pos[1], z_off)
        placed.append((pos[0], pos[1], half_xy))
        records.append({
            "name": obj.name,
            "kind": "fruit_cube" if kind == "fruit" else name,
            "class": name if kind == "fruit" else None,
            "profile": profile,
            "xy": [float(pos[0]), float(pos[1])],
        })
    return records


# ---------------------------------------------------------------------------
# Camera + render
# ---------------------------------------------------------------------------

def add_robot_camera() -> bpy.types.Object:
    bpy.ops.object.camera_add(location=(0.0, -1.0, 0.3))
    camera = bpy.context.object
    camera.name = "robot_camera"
    camera.data.clip_start = 0.01
    camera.data.clip_end = 60.0
    bpy.context.scene.camera = camera
    return camera


def sample_robot_view(camera: bpy.types.Object, mode: str = "general", target_xy=None) -> dict:
    if mode == "approach_cube" and target_xy is not None:
        # Robot pickup-approach: camera 0.40-0.70m from a chosen cube at robot height,
        # looking down at it. Makes the target cube large in-frame (unlike wide surveys),
        # exercising the full crop -> Face Classifier -> decide_cube path end-to-end.
        tx, ty = float(target_xy[0]), float(target_xy[1])
        ang = random.uniform(0.0, 6.283185)
        dist = random.uniform(0.40, 0.70)
        cx = tx + dist * math.cos(ang)
        cy = ty + dist * math.sin(ang)
        cx = max(-ARENA_HALF + 0.05, min(ARENA_HALF - 0.05, cx))
        cy = max(-ARENA_HALF + 0.05, min(ARENA_HALF - 0.05, cy))
        cz = random.uniform(0.18, 0.34)
        tz = random.uniform(0.03, 0.06)
        camera.location = (cx, cy, cz)
        look_at(camera, (tx, ty, tz))
        camera.data.lens = random.uniform(28.0, 38.0)
        return {"mode": mode, "camera": [cx, cy, cz], "target": [tx, ty, tz], "lens": camera.data.lens}
    if mode == "storage_close":
        # Close-up of the storage corner: flags + black border + aluminum
        # plate visible at 0.5-1.2m, camera height 0.2-0.35m.
        cx = -ARENA_HALF + random.uniform(0.50, 0.95)
        cy = -ARENA_HALF + random.uniform(0.50, 0.95)
        cz = random.uniform(0.20, 0.35)
        tx, ty = -ARENA_HALF + 0.10, -ARENA_HALF + 0.10
        tz = random.uniform(0.04, 0.12)
    elif mode == "along_fence":
        # Looking down along the south fence wall toward the storage corner.
        cx = random.uniform(0.7, 1.6)
        cy = -ARENA_HALF + random.uniform(0.12, 0.22)
        cz = random.uniform(0.20, 0.32)
        tx = -ARENA_HALF + 0.3
        ty = -ARENA_HALF + random.uniform(0.05, 0.12)
        tz = random.uniform(0.04, 0.12)
    else:
        for _ in range(50):
            cx = random.uniform(-1.45, 1.45)
            cy = random.uniform(-1.45, 1.45)
            cz = random.uniform(0.15, 0.40)
            tx = random.uniform(-0.9, 0.9)
            ty = random.uniform(-0.9, 0.9)
            tz = random.uniform(0.0, 0.10)
            if math.hypot(tx - cx, ty - cy) > 0.9:
                break
    camera.location = (cx, cy, cz)
    look_at(camera, (tx, ty, tz))
    camera.data.lens = random.uniform(24.0, 40.0) if mode == "general" else random.uniform(30.0, 40.0)
    return {"mode": mode, "camera": [cx, cy, cz], "target": [tx, ty, tz], "lens": camera.data.lens}


def configure_render(resolution: int, samples: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(samples)
    scene.cycles.use_denoising = True
    scene.cycles.device = "CPU"  # GPU is busy training -- CPU only.
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "NONE"
    except Exception:
        pass
    scene.render.resolution_x = int(resolution)
    scene.render.resolution_y = int(resolution)
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 92
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium Contrast"
    scene.view_settings.exposure = -0.28
    scene.view_settings.gamma = 1.0


# ---------------------------------------------------------------------------
# Per-scene cube/face metadata (projected polygons + ray-cast visibility)
# ---------------------------------------------------------------------------

def cube_local_faces(obj: bpy.types.Object) -> list:
    """6 axis-aligned faces from the object's local bounding box:
    [(local_normal_tuple, [4 corner Vectors]), ...]. Exact for both the
    primitive fruit cubes and the axis-aligned plain_cube.obj."""
    bb = np.array([list(c) for c in obj.bound_box], dtype=np.float64)
    mins, maxs = bb.min(axis=0), bb.max(axis=0)
    faces = []
    for axis in range(3):
        for sign in (1.0, -1.0):
            fixed = maxs[axis] if sign > 0 else mins[axis]
            others = [a for a in range(3) if a != axis]
            corners = []
            for da, db in ((0, 0), (1, 0), (1, 1), (0, 1)):
                c = [0.0, 0.0, 0.0]
                c[axis] = fixed
                c[others[0]] = maxs[others[0]] if da else mins[others[0]]
                c[others[1]] = maxs[others[1]] if db else mins[others[1]]
                corners.append(Vector(c))
            normal = [0.0, 0.0, 0.0]
            normal[axis] = sign
            faces.append((tuple(normal), corners))
    return faces


def face_sample_points(corners: list, grid: int = 4, inset: float = 0.10) -> list:
    c0, c1, c2, c3 = corners
    pts = []
    for i in range(grid):
        for j in range(grid):
            u = inset + (1.0 - 2.0 * inset) * (i + 0.5) / grid
            v = inset + (1.0 - 2.0 * inset) * (j + 0.5) / grid
            a = c0.lerp(c1, u)
            b = c3.lerp(c2, u)
            pts.append(a.lerp(b, v))
    return pts


def ray_visible_fraction(depsgraph, cam_loc: Vector, points: list, tol: float = 0.006) -> float:
    """Fraction of sample points whose first scene hit from the camera lands
    within `tol` of the point (handles self- and cross-object occlusion; the
    fruit decal planes sit 0.6mm above the cube faces, well inside tol)."""
    scene = bpy.context.scene
    visible = 0
    for p in points:
        d = p - cam_loc
        dist = d.length
        if dist < 1e-6:
            continue
        d.normalize()
        hit, loc, _n, _i, _obj, _m = scene.ray_cast(depsgraph, cam_loc, d, distance=dist + 0.05)
        if hit and (loc - p).length <= tol:
            visible += 1
    return visible / max(1, len(points))


def object_world_vertices(obj: bpy.types.Object) -> list:
    """All mesh vertices of `obj` (and its mesh children -- e.g. the fruit
    decal planes parented to a cube) expressed in world space."""
    verts = []
    if obj.type == "MESH":
        mw = obj.matrix_world
        verts.extend(mw @ v.co for v in obj.data.vertices)
    for child in obj.children:
        if child.type == "MESH":
            cmw = child.matrix_world
            verts.extend(cmw @ v.co for v in child.data.vertices)
    return verts


def project_object_silhouette(scene, camera: bpy.types.Object, obj: bpy.types.Object,
                              resolution: int) -> dict:
    """Project every mesh vertex of `obj` to image pixels, clip to the frame,
    and convex-hull the projected points into a silhouette polygon. Returns the
    projected 2D bbox, its area, the silhouette polygon (pixels), and a
    behind_camera flag (True if any vertex is at/behind the camera plane, where
    the projection is unreliable). Returns None for a vertex-less object."""
    world_verts = object_world_vertices(obj)
    if not world_verts:
        return None
    pix = []
    behind_camera = False
    res = float(resolution)
    for wv in world_verts:
        p = world_to_camera_view(scene, camera, wv)
        if p.z < 0.01:
            behind_camera = True
        x = min(max(p.x * res, 0.0), res)
        y = min(max((1.0 - p.y) * res, 0.0), res)
        pix.append((x, y))
    pts = np.array(pix, dtype=np.float32)
    hull = cv2.convexHull(pts.reshape(-1, 1, 2)).reshape(-1, 2)
    x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
    x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
    return {
        "behind_camera": bool(behind_camera),
        "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
        "bbox_area": round((x1 - x0) * (y1 - y0), 2),
        "silhouette": [[round(float(px), 2), round(float(py), 2)] for px, py in hull],
    }


def collect_a1_objects(records: list, camera: bpy.types.Object, resolution: int,
                       min_area: float = A1_MIN_BBOX_AREA) -> list:
    """A1 record for EVERY placed object (16 cubes + 12 non-cube polyhedra):
    its A1 class, projected 2D bbox and convex-hull silhouette polygon. Each is
    flagged on_screen when it is in front of the camera and its clipped bbox
    area clears `min_area` -- the SAME visibility gate for cubes and polyhedra."""
    scene = bpy.context.scene
    name_to_obj = {o.name: o for o in bpy.data.objects}
    objects = []
    for rec in records:
        a1_cls = a1_class_for_kind(rec["kind"])
        if a1_cls is None:
            continue
        obj = name_to_obj.get(rec["name"])
        if obj is None:
            continue
        sil = project_object_silhouette(scene, camera, obj, resolution)
        if sil is None:
            continue
        on_screen = (not sil["behind_camera"]) and sil["bbox_area"] >= min_area
        objects.append({
            "name": rec["name"],
            "kind": rec["kind"],
            "a1_class": a1_cls,
            "a1_class_id": A1_CLASSES.index(a1_cls),
            "on_screen": bool(on_screen),
            **sil,
        })
    return objects


def collect_scene_metadata(records: list, camera: bpy.types.Object, resolution: int) -> dict:
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cam_loc = camera.matrix_world.translation
    name_to_obj = {o.name: o for o in bpy.data.objects}
    cubes = []
    for rec in records:
        if rec["kind"] not in ("fruit_cube", "plain_cube"):
            continue
        obj = name_to_obj.get(rec["name"])
        if obj is None:
            continue
        mw = obj.matrix_world
        rot3 = mw.to_3x3()
        faces_meta = []
        all_px = []
        behind_camera = False
        front_total = 0
        front_visible = 0.0
        for normal, corners_local in cube_local_faces(obj):
            corners_world = [mw @ c for c in corners_local]
            proj = [world_to_camera_view(scene, camera, c) for c in corners_world]
            if any(p.z < 0.01 for p in proj):
                behind_camera = True
            poly = [[p.x * resolution, (1.0 - p.y) * resolution] for p in proj]
            all_px.extend(poly)
            n_world = (rot3 @ Vector(normal)).normalized()
            center = (corners_world[0] + corners_world[1] + corners_world[2] + corners_world[3]) / 4.0
            view_dir = (center - cam_loc).normalized()
            front = n_world.dot(view_dir) < -0.05
            if front and not behind_camera:
                pts = face_sample_points(corners_world)
                vis = ray_visible_fraction(depsgraph, cam_loc, pts)
                front_total += len(pts)
                front_visible += vis * len(pts)
            else:
                vis = 0.0
            printed = rec["kind"] == "fruit_cube" and tuple(normal) in FRUIT_FACE_NORMALS
            faces_meta.append({
                "normal": list(normal),
                "face_class": rec["class"] if printed else "plain",
                "printed": printed,
                "poly": [[round(v, 2) for v in pt] for pt in poly],
                "front_facing": bool(front),
                "visible_frac": round(vis, 4),
            })
        xs = [p[0] for p in all_px]
        ys = [p[1] for p in all_px]
        cubes.append({
            "name": rec["name"],
            "kind": rec["kind"],
            "class": rec["class"],
            "profile": rec["profile"],
            "behind_camera": behind_camera,
            "bbox": [round(min(xs), 2), round(min(ys), 2), round(max(xs), 2), round(max(ys), 2)],
            "visible_frac": round(front_visible / max(1, front_total), 4),
            "faces": faces_meta,
        })
    return {
        "resolution": resolution,
        "cubes": cubes,
        "objects": collect_a1_objects(records, camera, resolution),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed % (2 ** 31))
    bproc.init()
    try:
        bproc.renderer.set_render_devices(use_only_cpu=True)
    except Exception:
        pass

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tex_dir = output / "_textures"
    tex_dir.mkdir(parents=True, exist_ok=True)

    coco_paths = sorted(args.background_dir.resolve().rglob("*.jpg"))
    if not coco_paths:
        raise FileNotFoundError(f"No COCO backgrounds under {args.background_dir}")
    texture_root = args.texture_root.resolve()

    # Generate procedural texture images once.
    tex = {
        "floor": tex_dir / "sun111_floor.jpg",
        "fence": tex_dir / "sun168_fence.jpg",
        "flag": tex_dir / "korean_flag.jpg",
        "concrete": tex_dir / "concrete.jpg",
    }
    if not make_floor_texture_from_photo(args.floor_photo.resolve(), tex["floor"], size=3072):
        print(f"floor photo not found at {args.floor_photo}; using procedural floor texture", flush=True)
        make_sun111_floor_texture(tex["floor"], size=3072)
    make_sun168_fence_texture(tex["fence"])
    make_korean_flag_texture(tex["flag"])
    make_concrete_texture(tex["concrete"])

    configure_render(args.resolution, args.samples)
    meta_dir = output / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # View plan: optional pickup-approach close-ups first, then storage/fence, rest general.
    if args.approach > 0:
        view_plan = ["approach_cube"] * min(args.approach, args.num_images)
        view_plan += ["general"] * max(0, args.num_images - len(view_plan))
    else:
        view_plan = ["storage_close", "storage_close", "along_fence"]
        view_plan += ["general"] * max(0, args.num_images - len(view_plan))

    manifest = []
    camera = None
    records = []
    for frame_idx in range(args.num_images):
        if frame_idx % max(1, args.frames_per_layout) == 0:
            layout_idx = frame_idx // max(1, args.frames_per_layout)
            clear_scene()
            backdrop_paths = []
            for widx in range(4):
                bp = tex_dir / f"backdrop_{layout_idx % 25}_{widx}.jpg"
                make_backdrop_texture(coco_paths, bp)
                backdrop_paths.append(bp)
            build_static_arena(tex, backdrop_paths, args.light_energy)
            records = place_objects(args.asset_dir.resolve(), texture_root, tex_dir)
            camera = add_robot_camera()
            configure_render(args.resolution, args.samples)
        approach_target = None
        if view_plan[frame_idx] == "approach_cube":
            fruit_recs = [r for r in records if r.get("kind") == "fruit_cube" and r.get("xy")]
            if fruit_recs:
                approach_target = random.choice(fruit_recs)["xy"]
        view = sample_robot_view(camera, mode=view_plan[frame_idx], target_xy=approach_target)
        out_file = output / f"arena_{frame_idx:03d}.jpg"
        bpy.context.scene.render.filepath = str(out_file)
        bpy.ops.render.render(write_still=True)
        meta = collect_scene_metadata(records, camera, args.resolution)
        meta.update({
            "image": out_file.name,
            "layout": frame_idx // max(1, args.frames_per_layout),
            "view": view,
        })
        (meta_dir / f"{out_file.stem}.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        manifest.append({"image": out_file.name, "layout": frame_idx // max(1, args.frames_per_layout), **view})
        print(f"rendered {out_file}", flush=True)

    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "count": len(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
