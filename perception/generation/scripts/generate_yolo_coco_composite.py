import blenderproc as bproc

import argparse
import colorsys
import gc
import json
import math
import random
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector
import cv2
import numpy as np

CLASS_TO_ID = {
    "banana": 1,
    "orange": 2,
    "pineapple": 3,
    "apple": 4,
    "cube": 5,
    "octahedron": 6,
    "dodecahedron": 7,
    "icosahedron": 8,
}

YOLO_CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_TO_ID)}
SUPER_CATEGORY = "Data_Generation_Blender"
SHAPE_CLASSES = ["cube", "octahedron", "dodecahedron", "icosahedron"]
FRUIT_CLASSES = ["banana", "orange", "pineapple", "apple"]
SHAPE_CLASS_WEIGHTS = {name: 1.0 for name in SHAPE_CLASSES}
FRUIT_CLASS_WEIGHTS = {
    "banana": 1.55,
    "orange": 1.35,
    "pineapple": 1.55,
    "apple": 1.50,
}
ARENA_BOOSTER_FRUIT_CLASS_WEIGHTS = {
    "apple": 3.0,
    "orange": 3.0,
    "banana": 2.0,
    "pineapple": 2.0,
}
ARENA_SCENE_PROFILES = [
    ("normal", 0.600),
    ("sun_shift", 0.200),
    ("motion_blur", 0.100),
    ("closeup", 0.067),
    ("plain_cube_hard_negative", 0.033),
]
FRUIT_VISIBILITY_TIERS = [
    ("easy", 0.50),
    ("mid", 0.45),
    ("hard", 0.05),
]
FRUIT_VISIBILITY_RANGES = {
    "easy": (0.50, 1.01),
    "mid": (0.20, 0.50),
    "hard": (0.10, 0.20),
}
FRUIT_FACE_NORMALS = [
    # Integrated recipe, 2026-10-06: competition rule of 2026-07-18.
    # Printed top + opposing Y sides; bottom and opposing X sides are blank.
    # The supplied July-08 snapshot used opposing X sides + one Y side.
    ("pos_z", (0.0, 0.0, 1.0)),
    ("pos_y", (0.0, 1.0, 0.0)),
    ("neg_y", (0.0, -1.0, 0.0)),
]
FRUIT_FACE_HELPER_CATEGORY_ID = 900
FRUIT_TEXTURE_SIZE = 512
FRUIT_TEXTURE_CACHE = {}
BACKGROUND_IMAGE_CACHE = {}
TEXTURE_FILE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
META_V2_OBJECT_CLASSES = {
    "cube_like_object": 0,
    "octahedron": 1,
    "dodecahedron": 2,
    "icosahedron": 3,
}
META_V2_FACE_CLASSES = {
    "plain_face": 0,
    "fruit_face": 1,
}
CUBE_FACE_NORMALS = [
    ("neg_z", (0.0, 0.0, -1.0)),
    ("pos_z", (0.0, 0.0, 1.0)),
    ("neg_y", (0.0, -1.0, 0.0)),
    ("pos_x", (1.0, 0.0, 0.0)),
    ("pos_y", (0.0, 1.0, 0.0)),
    ("neg_x", (-1.0, 0.0, 0.0)),
]
FRUIT_FACE_INDEX_BY_NORMAL = {name: idx for idx, (name, _) in enumerate(FRUIT_FACE_NORMALS)}


def fruit_visibility_tier_weights(args):
    return [
        ("easy", max(0.0, float(args.fruit_visibility_easy_weight))),
        ("mid", max(0.0, float(args.fruit_visibility_mid_weight))),
        ("hard", max(0.0, float(args.fruit_visibility_hard_weight))),
    ]


def declared_fruit_class_from_texture_path(path):
    path = Path(path)
    parent = path.parent.name.lower()
    if parent in FRUIT_CLASSES:
        return parent
    stem = path.stem.lower()
    for class_name in FRUIT_CLASSES:
        if stem == class_name or stem.startswith(f"{class_name}_"):
            return class_name
    return None


def datablock_name(*parts):
    raw = "__".join(str(part) for part in parts if part is not None and str(part))
    clean = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)
    return clean[:180]


def make_material(name, color, roughness=0.68):
    mat = bproc.material.create(datablock_name(name, f"{random.getrandbits(32):08x}"))
    mat.set_principled_shader_value("Base Color", color)
    mat.set_principled_shader_value("Roughness", roughness)
    return mat


def make_materials_single_user(bpy_obj, name_prefix=None):
    if not hasattr(bpy_obj, "material_slots"):
        return
    for slot_idx, slot in enumerate(bpy_obj.material_slots):
        if slot.material is not None:
            mat = slot.material.copy()
            if name_prefix:
                mat.name = datablock_name(name_prefix, "mat", slot_idx, f"{random.getrandbits(32):08x}")
            if mat.use_nodes:
                for node in mat.node_tree.nodes:
                    if node.type == "TEX_IMAGE" and node.image is not None:
                        image = node.image.copy()
                        image.name = datablock_name(mat.name, "image", f"{random.getrandbits(32):08x}")
                        node.image = image
            slot.material = mat


def first_material_image_name(bpy_obj):
    if not hasattr(bpy_obj, "material_slots") or not bpy_obj.material_slots:
        return None, None
    mat = bpy_obj.material_slots[0].material
    if mat is None:
        return None, None
    image_name = None
    if mat.use_nodes:
        for node in mat.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image is not None:
                image_name = node.image.name
                break
    return mat.name, image_name


def set_background_category(obj):
    obj.set_cp("category_id", 0)
    obj.set_cp("supercategory", "coco_annotations")


def make_image_material(name, image_path):
    image_path = Path(image_path).resolve()
    mat = bpy.data.materials.new(name)
    mat.name = name
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    uv = nodes.new(type="ShaderNodeUVMap")
    uv.uv_map = "UVMap"
    tex = nodes.new(type="ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(image_path))
    tex.image.name = datablock_name(name, Path(image_path).stem, "file_image")
    tex.extension = "EXTEND"
    mix = nodes.new(type="ShaderNodeMixRGB")
    mix.blend_type = "MIX"
    mix.inputs["Color1"].default_value = (1.0, 1.0, 1.0, 1.0)
    mat.node_tree.links.new(uv.outputs["UV"], tex.inputs["Vector"])
    mat.node_tree.links.new(tex.outputs["Alpha"], mix.inputs["Fac"])
    mat.node_tree.links.new(tex.outputs["Color"], mix.inputs["Color2"])
    mat.node_tree.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = random.uniform(0.55, 0.96)
    return mat


def blender_image_from_bgr(name, image):
    image = cv2.resize(image, (FRUIT_TEXTURE_SIZE, FRUIT_TEXTURE_SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    alpha = np.ones(rgb.shape[:2] + (1,), dtype=np.float32)
    rgba = np.dstack([rgb, alpha])
    blender_image = bpy.data.images.new(name, width=FRUIT_TEXTURE_SIZE, height=FRUIT_TEXTURE_SIZE, alpha=True)
    blender_image.pixels.foreach_set(rgba.ravel())
    blender_image.update()
    blender_image.pack()
    blender_image.name = name
    return blender_image


def make_array_image_material(name, image):
    mat = bpy.data.materials.new(name)
    mat.name = name
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    uv = nodes.new(type="ShaderNodeUVMap")
    uv.uv_map = "UVMap"
    tex = nodes.new(type="ShaderNodeTexImage")
    tex.image = blender_image_from_bgr(f"{name}_image", image)
    tex.extension = "EXTEND"
    mat.node_tree.links.new(uv.outputs["UV"], tex.inputs["Vector"])
    mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = random.uniform(0.55, 0.96)
    return mat


def add_cube_uvs(bpy_obj):
    mesh = bpy_obj.data
    if not mesh.uv_layers:
        mesh.uv_layers.new(name="UVMap")
    uv_layer = mesh.uv_layers.active.data
    coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for poly in mesh.polygons:
        for i, loop_index in enumerate(poly.loop_indices):
            uv_layer[loop_index].uv = coords[i % 4]


def apply_printed_white_material(obj):
    # Slight off-white variation approximates common white PLA under different light.
    base = random.uniform(0.70, 0.93)
    color = np.clip([
        base + random.uniform(-0.05, 0.04),
        base + random.uniform(-0.05, 0.04),
        base + random.uniform(-0.05, 0.04),
        1.0,
    ], 0, 1).tolist()
    obj.replace_materials(make_material("printed_white_pla", color, roughness=random.uniform(0.48, 0.92)))


def fruit_color_score(class_name, image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    valid = (s > 45) & (v > 55)
    if class_name == "apple":
        mask = (((h <= 12) | (h >= 168)) | ((h >= 13) & (h <= 28))) & valid
    elif class_name == "banana":
        mask = ((h >= 18) & (h <= 38)) & valid
    elif class_name == "orange":
        mask = ((h >= 4) & (h <= 25)) & (s > 55) & (v > 60)
    else:
        green = ((h >= 35) & (h <= 92)) & (s > 25) & (v > 35)
        yellow_brown = ((h >= 8) & (h <= 42)) & (s > 28) & (v > 30)
        mask = green | yellow_brown
    return float(mask.mean())


def fruit_color_ratios(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return {
        "red": float(((((h <= 10) | (h >= 168)) & (s > 45) & (v > 45))).mean()),
        "yellow": float(((h >= 18) & (h <= 42) & (s > 35) & (v > 58)).mean()),
        "orange": float(((h >= 4) & (h <= 24) & (s > 55) & (v > 50)).mean()),
        "green": float(((h >= 35) & (h <= 88) & (s > 30) & (v > 45)).mean()),
        "brown": float(((h >= 8) & (h <= 26) & (s > 30) & (v > 25) & (v < 150)).mean()),
        "dark": float((v < 38).mean()),
        "valid": float(((s > 35) & (v > 45)).mean()),
    }


def fruit_texture_quality_passes(class_name, image):
    ratios = fruit_color_ratios(image)
    if class_name == "banana":
        # Dense green banana clusters look like pineapple skin once projected
        # onto a small cube face, so keep banana sources strongly yellow.
        return (
            ratios["yellow"] >= 0.10
            and ratios["green"] <= 0.16
            and ratios["red"] <= 0.08
            and ratios["dark"] <= 0.16
        )
    if class_name == "orange":
        return ratios["orange"] >= 0.22 and ratios["green"] <= 0.10 and ratios["yellow"] <= 0.28
    if class_name == "apple":
        return ratios["red"] >= 0.08 or ratios["orange"] >= 0.18
    if class_name == "pineapple":
        return (ratios["brown"] + ratios["green"] + 0.5 * ratios["yellow"]) >= 0.10
    return True


def fruit_texture_canonical_passes(class_name, image):
    ratios = fruit_color_ratios(image)
    if ratios["valid"] < 0.08 or ratios["dark"] > 0.24:
        return False
    if class_name == "banana":
        return (
            ratios["yellow"] >= 0.08
            and ratios["yellow"] >= ratios["green"] * 1.15
            and ratios["red"] <= 0.08
        )
    if class_name == "orange":
        return ratios["orange"] >= 0.16 and ratios["green"] <= 0.08 and ratios["yellow"] <= 0.36
    if class_name == "apple":
        return (ratios["red"] + ratios["orange"]) >= 0.16 and ratios["green"] <= 0.28
    if class_name == "pineapple":
        return (
            (ratios["brown"] + ratios["yellow"]) >= 0.20
            and ratios["yellow"] >= 0.045
            and ratios["red"] <= 0.10
        )
    return True


def read_texture_bgr(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        bgr = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        white = np.full_like(bgr, 255, dtype=np.float32)
        return np.clip(bgr * alpha + white * (1.0 - alpha), 0, 255).astype(np.uint8)
    return image[:, :, :3]


def collect_fruit_textures(texture_dir, class_name, canonical=False):
    cache_key = (str(Path(texture_dir).resolve()), class_name, bool(canonical))
    if cache_key in FRUIT_TEXTURE_CACHE:
        return FRUIT_TEXTURE_CACHE[cache_key]
    class_dir = Path(texture_dir) / class_name
    if not class_dir.exists():
        raise FileNotFoundError(f"No fruit textures for {class_name}: {class_dir}")
    paths = sorted(path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in TEXTURE_FILE_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"No fruit textures for {class_name}: {class_dir}")
    thresholds = {"apple": 0.045, "banana": 0.035, "orange": 0.055, "pineapple": 0.030}
    scored = []
    for path in paths:
        image = read_texture_bgr(path)
        if (
            image is not None
            and fruit_color_score(class_name, image) >= thresholds[class_name]
            and fruit_texture_quality_passes(class_name, image)
        ):
            scored.append(path)
    if canonical:
        canonical_scored = []
        for path in scored:
            image = read_texture_bgr(path)
            if image is not None and fruit_texture_canonical_passes(class_name, image):
                canonical_scored.append(path)
        usable = canonical_scored if canonical_scored else scored
    else:
        usable = scored
    usable = usable if usable else [path for path in paths if "_lab" in path.stem]
    if not usable:
        usable = paths
    mode = "canonical" if canonical else "usable"
    print(f"fruit textures {class_name}: {mode}={len(usable)} total={len(paths)}", flush=True)
    FRUIT_TEXTURE_CACHE[cache_key] = usable
    return usable


def choose_fruit_texture(texture_dir, class_name, canonical=False):
    paths = collect_fruit_textures(texture_dir, class_name, canonical=canonical)
    return random.choice(paths)


def choose_fruit_face_textures(texture_dir, class_name, count, single_texture_per_cube=False, canonical=False):
    paths = list(collect_fruit_textures(texture_dir, class_name, canonical=canonical))
    if single_texture_per_cube:
        texture = random.choice(paths)
        return [texture for _ in range(count)]
    if len(paths) >= count:
        return random.sample(paths, count)
    return [random.choice(paths) for _ in range(count)]


def resize_contain_on_white(image, size):
    h, w = image.shape[:2]
    scale = min(size / max(w, 1), size / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)
    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def jitter_fruit_color(image, class_name, strength):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    if strength == "light":
        hue_limits = {"apple": 7, "banana": 4, "orange": 5, "pineapple": 6}
        sat_range = (0.82, 1.22)
        val_range = (0.86, 1.16)
    else:
        hue_limits = {"apple": 10, "banana": 6, "orange": 8, "pineapple": 8}
        sat_range = (0.65, 1.42)
        val_range = (0.72, 1.25)
    hue_shift = random.uniform(-hue_limits[class_name], hue_limits[class_name])
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(*sat_range), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(*val_range) + random.uniform(-10, 10), 0, 255)
    if class_name in {"banana", "orange"}:
        lo, hi = {"banana": (18, 42), "orange": (5, 25)}[class_name]
        colored = (hsv[..., 1] > 35) & (hsv[..., 2] > 50)
        hsv[..., 0][colored] = np.clip(hsv[..., 0][colored], lo, hi)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    if random.random() < 0.65:
        gains = np.array([
            random.uniform(0.88, 1.16),
            random.uniform(0.90, 1.12),
            random.uniform(0.88, 1.16),
        ], dtype=np.float32)
        out *= gains
    if random.random() < 0.45:
        contrast = random.uniform(0.82, 1.20) if strength == "light" else random.uniform(0.70, 1.32)
        out = (out - 127.5) * contrast + 127.5
    out = np.clip(out, 0, 255).astype(np.uint8)
    # 2026-10-06 integration: clamp AFTER BGR gains/contrast. Otherwise gains
    # can push a labelled fruit into another class's hue band (see history).
    if class_name in {"apple", "banana", "orange"}:
        final_hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        colored = (final_hsv[..., 1] > 35) & (final_hsv[..., 2] > 50)
        hue = final_hsv[..., 0]
        if class_name == "apple":
            red_distance = np.minimum(hue.astype(np.int16), 180 - hue.astype(np.int16))
            fix = colored & (red_distance > 12)
            hue[fix & (hue < 90)] = 12
            hue[fix & (hue >= 90)] = 168
        else:
            lo, hi = {"banana": (18, 42), "orange": (5, 25)}[class_name]
            hue[colored] = np.clip(hue[colored], lo, hi)
        out = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return out


def warp_fruit_texture(image, class_name, strength, size=FRUIT_TEXTURE_SIZE):
    img = resize_contain_on_white(image, size)
    if strength == "none":
        return img
    if random.random() < 0.50:
        img = cv2.flip(img, random.choice([0, 1, -1]))

    angle_limit = 12 if strength == "light" else 28
    scale_range = (0.88, 1.08) if strength == "light" else (0.78, 1.16)
    matrix = cv2.getRotationMatrix2D(
        (size / 2.0, size / 2.0),
        random.uniform(-angle_limit, angle_limit),
        random.uniform(*scale_range),
    )
    border = int(random.uniform(232, 255))
    img = cv2.warpAffine(img, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(border, border, border))

    img = jitter_fruit_color(img, class_name, strength)
    if random.random() < (0.18 if strength == "light" else 0.32):
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if random.random() < (0.20 if strength == "light" else 0.42):
        quality = random.randint(54, 92)
        ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return img


def fruit_foreground_alpha(image, blur_sigma):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(image)
    near_white = (b > 220) & (g > 220) & (r > 220)
    low_chroma_light = (hsv[..., 1] < 42) & (hsv[..., 2] > 150)
    blank = near_white | low_chroma_light | (hsv[..., 2] > 252)
    mask = (~blank).astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    if mask.mean() < 5:
        mask = ((hsv[..., 1] > 28) & (hsv[..., 2] > 35)).astype(np.uint8) * 255
    mask = cv2.GaussianBlur(mask, (0, 0), blur_sigma)
    return np.clip(mask.astype(np.float32) / 255.0, 0.0, 1.0)[..., None]


def sample_collage_centers(size, patch_sizes):
    centers = []
    for patch_size in patch_sizes:
        margin = int(patch_size * 0.34)
        best = None
        best_score = -1.0
        for _ in range(80):
            cx = random.randint(max(0, margin), min(size - 1, size - margin))
            cy = random.randint(max(0, margin), min(size - 1, size - margin))
            if not centers:
                score = 1.0
            else:
                score = min(
                    math.hypot(cx - ox, cy - oy) / max(1.0, 0.5 * (patch_size + other_size))
                    for ox, oy, other_size in centers
                )
            if score > best_score:
                best = (cx, cy)
                best_score = score
        centers.append((best[0], best[1], patch_size))
    random.shuffle(centers)
    return centers


def make_fruit_texture_collage(texture_dir, class_name, strength, size=FRUIT_TEXTURE_SIZE, canonical=False):
    canvas_value = int(random.uniform(236, 255))
    canvas = np.full((size, size, 3), canvas_value, dtype=np.uint8)
    if strength == "none":
        patch_count = 2
        patch_ratio = (0.56, 0.74)
    elif strength == "light":
        patch_count = random.randint(2, 3)
        patch_ratio = (0.52, 0.70)
    else:
        patch_count = random.randint(2, 3)
        patch_ratio = (0.46, 0.66)
    patch_sizes = [random.randint(int(size * patch_ratio[0]), int(size * patch_ratio[1])) for _ in range(patch_count)]
    centers = sample_collage_centers(size, patch_sizes)
    source_textures = []
    for idx, (cx, cy, patch_size) in enumerate(centers):
        texture = choose_fruit_texture(texture_dir, class_name, canonical=canonical)
        patch = read_texture_bgr(texture)
        if patch is None:
            continue
        source_textures.append(Path(texture).as_posix())
        patch = warp_fruit_texture(patch, class_name, strength, patch_size)
        alpha = fruit_foreground_alpha(patch, patch_size * 0.018)

        x0 = int(cx - patch_size / 2)
        y0 = int(cy - patch_size / 2)
        x1 = max(0, x0)
        y1 = max(0, y0)
        x2 = min(size, x0 + patch_size)
        y2 = min(size, y0 + patch_size)
        if x2 <= x1 or y2 <= y1:
            continue
        px1 = x1 - x0
        py1 = y1 - y0
        px2 = px1 + (x2 - x1)
        py2 = py1 + (y2 - y1)
        patch_roi = patch[py1:py2, px1:px2].astype(np.float32)
        alpha_roi = alpha[py1:py2, px1:px2]
        canvas_roi = canvas[y1:y2, x1:x2].astype(np.float32)
        canvas[y1:y2, x1:x2] = np.clip(patch_roi * alpha_roi + canvas_roi * (1.0 - alpha_roi), 0, 255).astype(np.uint8)
    return canvas, source_textures


def generated_face_texture_file(output_dir, owner_name, face_idx, class_name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep this short: Windows/OpenCV image IO can silently fail on long nested paths.
    filename = datablock_name(class_name, face_idx, f"{random.getrandbits(48):012x}") + ".jpg"
    return output_dir / filename


def print_label_texture_quality_ok(image):
    if image is None or image.size == 0:
        return False
    h, w = image.shape[:2]
    border = max(4, int(min(h, w) * 0.08))
    border_pixels = np.concatenate(
        [
            image[:border, :, :].reshape(-1, 3),
            image[-border:, :, :].reshape(-1, 3),
            image[:, :border, :].reshape(-1, 3),
            image[:, -border:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    border_hsv = cv2.cvtColor(border_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    border_white = (border_hsv[:, 1] < 42) & (border_hsv[:, 2] > 178)
    if float(np.mean(border_white)) < 0.58:
        return False

    resized = resize_contain_on_white(image, FRUIT_TEXTURE_SIZE)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    colored = (hsv[..., 1] > 52) & (hsv[..., 2] > 45) & (hsv[..., 2] < 248)
    colored_ratio = float(np.mean(colored))
    return 0.055 <= colored_ratio <= 0.70


def choose_print_label_texture(texture_dir, class_name, canonical=False, attempts=80):
    fallback_texture = None
    fallback_image = None
    for _ in range(max(1, attempts)):
        texture = choose_fruit_texture(texture_dir, class_name, canonical=canonical)
        image = read_texture_bgr(texture)
        if image is None:
            continue
        if fallback_texture is None:
            fallback_texture = texture
            fallback_image = image
        if print_label_texture_quality_ok(image):
            return texture, image
    if fallback_texture is None or fallback_image is None:
        raise RuntimeError(f"failed to read any fruit texture for print label: {class_name}")
    return fallback_texture, fallback_image


def make_fruit_texture_collage_file(texture_dir, class_name, strength, output_dir, owner_name, face_idx, canonical=False):
    collage, source_textures = make_fruit_texture_collage(
        texture_dir,
        class_name,
        strength,
        canonical=canonical,
    )
    texture_path = generated_face_texture_file(output_dir, owner_name, face_idx, class_name)
    cv2.imwrite(str(texture_path), collage, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(90, 97)])
    return texture_path, source_textures


def warp_fruit_texture_no_color_jitter(image, strength, size=FRUIT_TEXTURE_SIZE):
    img = resize_contain_on_white(image, size)
    if strength != "none" and random.random() < 0.50:
        img = cv2.flip(img, random.choice([0, 1, -1]))

    if strength != "none":
        angle_limit = 8 if strength == "light" else 18
        scale_range = (0.92, 1.06) if strength == "light" else (0.86, 1.10)
        matrix = cv2.getRotationMatrix2D(
            (size / 2.0, size / 2.0),
            random.uniform(-angle_limit, angle_limit),
            random.uniform(*scale_range),
        )
        border = int(random.uniform(240, 255))
        img = cv2.warpAffine(
            img,
            matrix,
            (size, size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(border, border, border),
        )

    if random.random() < (0.10 if strength == "light" else 0.22):
        img = cv2.GaussianBlur(img, (3, 3), 0)
    if random.random() < (0.18 if strength == "light" else 0.35):
        ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(72, 95)])
        if ok:
            img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return img


PRINT_LABEL_REALISTIC_PROFILES = {
    "realistic_a4_mild",
    "realistic_a4_sparse_icon",
    "realistic_a4_hard",
    "realistic_a4_boundary",
    "realistic_a4_fruit_visible",
    "realistic_a4_label_offset",
    "realistic_a4_label_offset_strong",
    "realistic_a4_label_offset_lqprint",
    "realistic_a4_tape_edge",
    "realistic_a4_tape_edge_visible",
    "realistic_a4_orange_icon_cluster",
    "realistic_a4_orange_icon_cluster_bright",
}


def overlay_print_label_tape(canvas, rect, paper_value, visible=False):
    """Add subtle semi-transparent tape/gloss around a printed A4 label edge."""
    x0, y0, x1, y1 = rect
    h, w = canvas.shape[:2]
    overlay = canvas.astype(np.float32)
    tape_value = random.randint(230, 255)
    tape_color = np.array(
        [
            np.clip(tape_value - random.randint(4, 12), 0, 255),
            np.clip(tape_value - random.randint(1, 8), 0, 255),
            np.clip(tape_value + random.randint(0, 5), 0, 255),
        ],
        dtype=np.float32,
    )
    alpha = random.uniform(0.20, 0.38) if visible else random.uniform(0.10, 0.22)
    tape_w = max(5, int(min(h, w) * random.uniform(0.045, 0.085))) if visible else max(4, int(min(h, w) * random.uniform(0.025, 0.055)))
    extension = max(8, int(min(h, w) * random.uniform(0.08, 0.16))) if visible else max(6, int(min(h, w) * random.uniform(0.04, 0.10)))

    edges = ["top", "bottom", "left", "right"]
    random.shuffle(edges)
    edge_count = random.choice([3, 4]) if visible else (2 if random.random() < 0.70 else 3)
    for edge in edges[:edge_count]:
        if edge == "top":
            xa = max(0, x0 - extension)
            xb = min(w, x1 + extension)
            ya = max(0, y0 - tape_w // 2)
            yb = min(h, y0 + tape_w)
        elif edge == "bottom":
            xa = max(0, x0 - extension)
            xb = min(w, x1 + extension)
            ya = max(0, y1 - tape_w)
            yb = min(h, y1 + tape_w // 2)
        elif edge == "left":
            xa = max(0, x0 - tape_w // 2)
            xb = min(w, x0 + tape_w)
            ya = max(0, y0 - extension)
            yb = min(h, y1 + extension)
        else:
            xa = max(0, x1 - tape_w)
            xb = min(w, x1 + tape_w // 2)
            ya = max(0, y0 - extension)
            yb = min(h, y1 + extension)
        if xb <= xa or yb <= ya:
            continue
        roi = overlay[ya:yb, xa:xb]
        roi[:] = roi * (1.0 - alpha) + tape_color * alpha

        if random.random() < (0.95 if visible else 0.85):
            if edge in {"top", "bottom"}:
                line_y = random.randint(ya, max(ya, yb - 1))
                cv2.line(overlay, (xa, line_y), (xb - 1, line_y), (255, 255, 255), 1, cv2.LINE_AA)
                if visible and yb - ya > 5:
                    cv2.line(overlay, (xa, max(ya, line_y - tape_w // 3)), (xb - 1, max(ya, line_y - tape_w // 3)), (210, 218, 224), 1, cv2.LINE_AA)
            else:
                line_x = random.randint(xa, max(xa, xb - 1))
                cv2.line(overlay, (line_x, ya), (line_x, yb - 1), (255, 255, 255), 1, cv2.LINE_AA)
                if visible and xb - xa > 5:
                    cv2.line(overlay, (max(xa, line_x - tape_w // 3), ya), (max(xa, line_x - tape_w // 3), yb - 1), (210, 218, 224), 1, cv2.LINE_AA)

        if visible and random.random() < 0.70:
            speckles = np.random.random((yb - ya, xb - xa)) < random.uniform(0.002, 0.007)
            roi = overlay[ya:yb, xa:xb]
            roi[speckles] = np.clip(roi[speckles] + random.uniform(8, 22), 0, 255)

    if random.random() < (0.88 if visible else 0.65):
        cx = random.randint(max(0, x0 - extension), min(w - 1, x1 + extension))
        cy = random.randint(max(0, y0 - extension), min(h - 1, y1 + extension))
        ax = random.randint(max(10, tape_w), max(12, tape_w * (4 if visible else 3)))
        ay = random.randint(max(4, tape_w // 3), max(6, int(tape_w * (1.3 if visible else 1.0))))
        angle = random.uniform(-28, 28)
        highlight = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(highlight, (cx, cy), (ax, ay), angle, 0, 360, 255, -1, cv2.LINE_AA)
        blur = cv2.GaussianBlur(highlight, (0, 0), random.uniform(1.6, 4.2) if visible else random.uniform(2.0, 5.0)).astype(np.float32) / 255.0
        strength = random.uniform(18.0, 46.0) if visible else random.uniform(8.0, 24.0)
        overlay = np.clip(overlay + blur[..., None] * strength, 0, 255)

    return overlay.astype(np.uint8)


def flatten_patch_to_print_icon(patch):
    """Flatten a photo patch into a printed flat-graphic look (posterized label print)."""
    smooth = cv2.bilateralFilter(patch, 9, random.randint(45, 75), random.randint(45, 75))
    pixels = smooth.reshape(-1, 3).astype(np.float32)
    k = random.choice([4, 5, 6])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 8, 1.0)
    _, labels_, centers = cv2.kmeans(pixels, k, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    flat = centers[labels_.flatten()].reshape(smooth.shape).astype(np.uint8)
    if random.random() < 0.5:
        flat = cv2.GaussianBlur(flat, (3, 3), 0)
    return flat


def make_orange_icon_cluster_patch(source, size, bright=False):
    """Create a printed orange-cluster icon from real orange texture colors."""
    source = resize_contain_on_white(source, max(64, size))
    hsv_source = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    orange_mask = (hsv_source[..., 0] >= 5) & (hsv_source[..., 0] <= 35) & (hsv_source[..., 1] > 55) & (hsv_source[..., 2] > 45)
    if int(orange_mask.sum()) > 64:
        orange_hsv = np.median(hsv_source[orange_mask], axis=0).astype(np.float32)
        if bright:
            orange_hsv[0] = np.clip(orange_hsv[0], 12, 24)
            orange_hsv[1] = np.clip(orange_hsv[1], 110, 165)
            orange_hsv[2] = np.clip(orange_hsv[2], 195, 232)
        base_color = cv2.cvtColor(np.uint8([[orange_hsv]]), cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)
    else:
        base_color = np.array([72, 170, 226] if bright else [42, 145, 222], dtype=np.float32)

    patch = np.full((size, size, 3), 255, dtype=np.uint8)
    if bright:
        centers = [
            (0.36, 0.43, 0.20),
            (0.56, 0.39, 0.20),
            (0.68, 0.55, 0.19),
            (0.47, 0.61, 0.21),
        ]
        centers = centers[: random.choice([3, 4])]
    else:
        centers = [
            (0.38, 0.42, 0.23),
            (0.60, 0.42, 0.24),
            (0.49, 0.58, 0.25),
            (0.70, 0.58, 0.20),
        ]
        random.shuffle(centers)
        centers = centers[: random.choice([3, 4])]
    yy, xx = np.indices((size, size), dtype=np.float32)
    for cx_r, cy_r, radius_r in centers:
        cx = int(size * (cx_r + random.uniform(-0.025, 0.025)))
        cy = int(size * (cy_r + random.uniform(-0.025, 0.025)))
        rx = int(size * radius_r * random.uniform(0.86, 1.00 if bright else 1.04))
        ry = int(size * radius_r * random.uniform(0.84, 1.00 if bright else 1.02))
        fruit = (((xx - cx) / max(1, rx)) ** 2 + ((yy - cy) / max(1, ry)) ** 2) <= 1.0
        shade = 1.0 - 0.20 * ((xx - cx) / max(1, rx)) + 0.12 * ((yy - cy) / max(1, ry))
        color = base_color * (random.uniform(0.98, 1.07) if bright else random.uniform(0.94, 1.08))
        fruit_img = np.clip(color[None, None, :] * shade[..., None], 0, 255)
        edge = cv2.GaussianBlur(fruit.astype(np.uint8) * 255, (0, 0), max(1.0, size * 0.010)).astype(np.float32) / 255.0
        patch = np.clip(fruit_img * edge[..., None] + patch.astype(np.float32) * (1.0 - edge[..., None]), 0, 255).astype(np.uint8)
        if random.random() < 0.70:
            hx = int(cx - rx * random.uniform(0.18, 0.34))
            hy = int(cy - ry * random.uniform(0.22, 0.38))
            cv2.ellipse(
                patch,
                (hx, hy),
                (max(2, rx // 5), max(2, ry // 8)),
                random.uniform(-28, 18),
                0,
                360,
                tuple(int(v) for v in np.clip(color * 1.22 + 16, 0, 255)),
                -1,
                cv2.LINE_AA,
            )

    leaf_color = np.array([62, 136, 64] if bright else [58, 128, 62], dtype=np.uint8)
    leaf_count = random.choice([1, 2]) if bright else random.choice([2, 3, 4])
    for _ in range(leaf_count):
        lx = int(size * random.uniform(0.35, 0.72))
        ly = int(size * random.uniform(0.60, 0.76))
        axes = (
            int(size * random.uniform(0.07, 0.115) if bright else size * random.uniform(0.09, 0.15)),
            int(size * random.uniform(0.020, 0.040) if bright else size * random.uniform(0.025, 0.050)),
        )
        angle = random.choice([-34, -22, 24, 36]) + random.uniform(-8, 8)
        color = tuple(int(v) for v in np.clip(leaf_color.astype(np.int16) + random.randint(-14, 18), 0, 255))
        cv2.ellipse(patch, (lx, ly), axes, angle, 0, 360, color, -1, cv2.LINE_AA)
        if random.random() < 0.65:
            cv2.ellipse(patch, (lx, ly), axes, angle, 0, 360, (35, 92, 42), 1, cv2.LINE_AA)

    if random.random() < (0.35 if bright else 0.55):
        low = max(40, int(size * random.uniform(0.70, 0.88) if bright else size * random.uniform(0.58, 0.82)))
        patch = cv2.resize(cv2.resize(patch, (low, low), interpolation=cv2.INTER_AREA), (size, size), interpolation=cv2.INTER_LINEAR)
    if random.random() < (0.18 if bright else 0.35):
        patch = cv2.GaussianBlur(patch, (3, 3), 0)
    if random.random() < (0.30 if bright else 0.45):
        ok, encoded = cv2.imencode(".jpg", patch, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(78, 92) if bright else random.randint(68, 88)])
        if ok:
            patch = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return patch


def make_fruit_print_label_texture(
    texture_dir,
    class_name,
    strength,
    size=FRUIT_TEXTURE_SIZE,
    canonical=False,
    profile="legacy",
):
    if profile in PRINT_LABEL_REALISTIC_PROFILES:
        source_texture, source = choose_print_label_texture(texture_dir, class_name, canonical=canonical)
    else:
        source_texture = choose_fruit_texture(texture_dir, class_name, canonical=canonical)
        source = read_texture_bgr(source_texture)
        if source is None:
            raise RuntimeError(f"failed to read fruit texture: {source_texture}")

    if profile in PRINT_LABEL_REALISTIC_PROFILES:
        if profile == "realistic_a4_boundary":
            face_base = int(random.uniform(226, 242))
            warm_blue_drop = random.randint(12, 24)
            warm_green_drop = random.randint(2, 8)
        elif profile in {
            "realistic_a4_label_offset_strong",
            "realistic_a4_label_offset_lqprint",
            "realistic_a4_tape_edge",
            "realistic_a4_tape_edge_visible",
            "realistic_a4_orange_icon_cluster",
            "realistic_a4_orange_icon_cluster_bright",
        }:
            face_base = int(random.uniform(208, 230))
            warm_blue_drop = random.randint(22, 40)
            warm_green_drop = random.randint(6, 15)
        elif profile in {"realistic_a4_fruit_visible", "realistic_a4_label_offset"}:
            face_base = int(random.uniform(220, 238))
            warm_blue_drop = random.randint(16, 30)
            warm_green_drop = random.randint(3, 10)
        elif profile == "realistic_a4_hard":
            face_base = int(random.uniform(232, 248))
            warm_blue_drop = random.randint(5, 14)
            warm_green_drop = random.randint(0, 5)
        else:
            face_base = int(random.uniform(236, 249))
            warm_blue_drop = random.randint(3, 10)
            warm_green_drop = random.randint(0, 4)
        face_color = np.array(
            [
                np.clip(face_base - warm_blue_drop, 0, 255),
                np.clip(face_base - warm_green_drop, 0, 255),
                np.clip(face_base + random.randint(0, 4), 0, 255),
            ],
            dtype=np.uint8,
        )
        canvas = np.empty((size, size, 3), dtype=np.uint8)
        canvas[:] = face_color
    else:
        face_base = int(random.uniform(238, 254))
        canvas = np.full((size, size, 3), face_base, dtype=np.uint8)

    if profile in PRINT_LABEL_REALISTIC_PROFILES:
        if profile == "realistic_a4_boundary":
            paper_delta = random.randint(14, 28)
        elif profile in {"realistic_a4_label_offset_strong", "realistic_a4_label_offset_lqprint"}:
            paper_delta = random.randint(32, 54)
        elif profile in {
            "realistic_a4_tape_edge",
            "realistic_a4_tape_edge_visible",
            "realistic_a4_orange_icon_cluster",
            "realistic_a4_orange_icon_cluster_bright",
        }:
            paper_delta = random.randint(24, 42)
        elif profile in {"realistic_a4_fruit_visible", "realistic_a4_label_offset"}:
            paper_delta = random.randint(22, 36)
        elif profile == "realistic_a4_hard":
            paper_delta = random.randint(-1, 10)
        else:
            paper_delta = random.randint(-3, 7)
        paper_value = int(np.clip(face_base + paper_delta, 236, 255))
        paper_color = np.array(
            [
                np.clip(paper_value - random.randint(0, 2), 0, 255),
                np.clip(paper_value - random.randint(0, 2), 0, 255),
                np.clip(paper_value, 0, 255),
            ],
            dtype=np.uint8,
        )
        if profile == "realistic_a4_boundary":
            margin_x = (0.040, 0.085)
            margin_y = (0.040, 0.090)
        elif profile in {"realistic_a4_label_offset_strong", "realistic_a4_label_offset_lqprint"}:
            margin_x = (0.025, 0.060)
            margin_y = (0.025, 0.065)
        elif profile in {
            "realistic_a4_tape_edge",
            "realistic_a4_tape_edge_visible",
            "realistic_a4_orange_icon_cluster",
            "realistic_a4_orange_icon_cluster_bright",
        }:
            margin_x = (0.035, 0.075)
            margin_y = (0.035, 0.085)
        elif profile in {"realistic_a4_fruit_visible", "realistic_a4_label_offset"}:
            margin_x = (0.035, 0.075)
            margin_y = (0.035, 0.080)
        elif profile == "realistic_a4_hard":
            margin_x = (0.025, 0.075)
            margin_y = (0.030, 0.085)
        else:
            margin_x = (0.040, 0.095)
            margin_y = (0.045, 0.105)
    else:
        paper_delta = random.randint(-6, 4)
        paper_value = int(np.clip(face_base + paper_delta, 224, 255))
        paper_color = np.array([paper_value, paper_value, paper_value], dtype=np.uint8)
        margin_x = (0.07, 0.15)
        margin_y = (0.07, 0.15)
    x0 = int(size * random.uniform(*margin_x))
    y0 = int(size * random.uniform(*margin_y))
    x1 = size - int(size * random.uniform(*margin_x))
    y1 = size - int(size * random.uniform(*margin_y))
    canvas[y0:y1, x0:x1] = paper_color
    if profile in PRINT_LABEL_REALISTIC_PROFILES:
        if profile == "realistic_a4_boundary":
            shadow_drop = random.randint(18, 34)
        elif profile in {"realistic_a4_label_offset_strong", "realistic_a4_label_offset_lqprint"}:
            shadow_drop = random.randint(26, 48)
        elif profile in {
            "realistic_a4_tape_edge",
            "realistic_a4_tape_edge_visible",
            "realistic_a4_orange_icon_cluster",
            "realistic_a4_orange_icon_cluster_bright",
        }:
            shadow_drop = random.randint(18, 32)
        elif profile in {"realistic_a4_fruit_visible", "realistic_a4_label_offset"}:
            shadow_drop = random.randint(20, 38)
        elif profile == "realistic_a4_hard":
            shadow_drop = random.randint(12, 26)
        else:
            shadow_drop = random.randint(8, 18)
        shadow = int(np.clip(paper_value - shadow_drop, 168, 238))
        highlight = int(np.clip(paper_value + random.randint(1, 5), 0, 255))
        cv2.line(canvas, (x0, y1 - 1), (x1 - 1, y1 - 1), (shadow, shadow, shadow), 1, cv2.LINE_AA)
        cv2.line(canvas, (x1 - 1, y0), (x1 - 1, y1 - 1), (shadow, shadow, shadow), 1, cv2.LINE_AA)
        cv2.line(canvas, (x0, y0), (x1 - 1, y0), (highlight, highlight, highlight), 1, cv2.LINE_AA)
        cv2.line(canvas, (x0, y0), (x0, y1 - 1), (highlight, highlight, highlight), 1, cv2.LINE_AA)
    if profile == "realistic_a4_boundary":
        edge_probability = 0.94
    elif profile in {"realistic_a4_label_offset_strong", "realistic_a4_label_offset_lqprint"}:
        edge_probability = 1.0
    elif profile in {
        "realistic_a4_tape_edge",
        "realistic_a4_tape_edge_visible",
        "realistic_a4_orange_icon_cluster",
        "realistic_a4_orange_icon_cluster_bright",
    }:
        edge_probability = 1.0
    elif profile in {"realistic_a4_fruit_visible", "realistic_a4_label_offset"}:
        edge_probability = 0.97
    elif profile == "realistic_a4_hard":
        edge_probability = 0.86
    elif profile == "realistic_a4_mild":
        edge_probability = 0.55
    else:
        edge_probability = 0.75
    if random.random() < edge_probability:
        if profile == "realistic_a4_boundary":
            edge_drop = random.randint(14, 28)
        elif profile in {"realistic_a4_label_offset_strong", "realistic_a4_label_offset_lqprint"}:
            edge_drop = random.randint(26, 48)
        elif profile in {
            "realistic_a4_tape_edge",
            "realistic_a4_tape_edge_visible",
            "realistic_a4_orange_icon_cluster",
            "realistic_a4_orange_icon_cluster_bright",
        }:
            edge_drop = random.randint(14, 28)
        elif profile in {"realistic_a4_fruit_visible", "realistic_a4_label_offset"}:
            edge_drop = random.randint(18, 34)
        elif profile == "realistic_a4_hard":
            edge_drop = random.randint(8, 18)
        elif profile == "realistic_a4_mild":
            edge_drop = random.randint(3, 11)
        else:
            edge_drop = random.randint(8, 20)
        edge = int(np.clip(paper_value - edge_drop, 190, 246))
        cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), (edge, edge, edge), 1, cv2.LINE_AA)

    if profile in PRINT_LABEL_REALISTIC_PROFILES:
        if profile == "realistic_a4_boundary":
            if strength == "none":
                ratio_range = (0.76, 0.92)
            elif strength == "light":
                ratio_range = (0.70, 0.88)
            else:
                ratio_range = random.choice([(0.68, 0.86), (0.72, 0.90), (0.76, 0.92)])
        elif profile in {"realistic_a4_label_offset_strong", "realistic_a4_label_offset_lqprint"}:
            if strength == "none":
                ratio_range = (0.80, 0.95)
            elif strength == "light":
                ratio_range = (0.76, 0.92)
            else:
                ratio_range = random.choice([(0.74, 0.90), (0.78, 0.94), (0.82, 0.96)])
        elif profile in {"realistic_a4_tape_edge", "realistic_a4_tape_edge_visible"}:
            if strength == "none":
                ratio_range = (0.78, 0.94)
            elif strength == "light":
                ratio_range = (0.74, 0.91)
            else:
                ratio_range = random.choice([(0.72, 0.89), (0.76, 0.92), (0.80, 0.95)])
        elif profile in {"realistic_a4_orange_icon_cluster", "realistic_a4_orange_icon_cluster_bright"}:
            if strength == "none":
                ratio_range = (0.82, 0.96)
            elif strength == "light":
                ratio_range = (0.78, 0.94)
            else:
                ratio_range = random.choice([(0.76, 0.92), (0.80, 0.95), (0.82, 0.96)])
        elif profile in {"realistic_a4_fruit_visible", "realistic_a4_label_offset"}:
            if strength == "none":
                ratio_range = (0.84, 0.96)
            elif strength == "light":
                ratio_range = (0.78, 0.94)
            else:
                ratio_range = random.choice([(0.76, 0.92), (0.80, 0.96), (0.84, 0.98)])
        elif profile == "realistic_a4_hard":
            if strength == "none":
                ratio_range = (0.66, 0.84)
            elif strength == "light":
                ratio_range = (0.58, 0.80)
            else:
                ratio_range = random.choice([(0.56, 0.76), (0.58, 0.80), (0.62, 0.82)])
        elif profile == "realistic_a4_sparse_icon":
            if strength == "none":
                ratio_range = (0.26, 0.42)
            elif strength == "light":
                ratio_range = (0.24, 0.40)
            else:
                ratio_range = random.choice([(0.22, 0.36), (0.24, 0.40), (0.28, 0.44)])
        else:
            if strength == "none":
                ratio_range = (0.70, 0.88)
            elif strength == "light":
                ratio_range = (0.64, 0.86)
            else:
                ratio_range = random.choice([(0.60, 0.82), (0.64, 0.86), (0.68, 0.88)])
    else:
        if strength == "none":
            ratio_range = (0.72, 0.90)
        elif strength == "light":
            ratio_range = (0.66, 0.88)
        else:
            ratio_range = random.choice([(0.60, 0.82), (0.66, 0.90), (0.72, 0.94)])
    patch_size = random.randint(int(size * ratio_range[0]), int(size * ratio_range[1]))
    if profile in {"realistic_a4_orange_icon_cluster", "realistic_a4_orange_icon_cluster_bright"} and class_name == "orange":
        patch = make_orange_icon_cluster_patch(source, patch_size, bright=profile == "realistic_a4_orange_icon_cluster_bright")
    else:
        patch = warp_fruit_texture_no_color_jitter(source, strength, patch_size)
    if profile == "realistic_a4_sparse_icon" and random.random() < 0.75:
        patch = flatten_patch_to_print_icon(patch)
    if profile in PRINT_LABEL_REALISTIC_PROFILES:
        if profile == "realistic_a4_boundary":
            alpha_range = (0.94, 1.0)
        elif profile in {"realistic_a4_label_offset_strong", "realistic_a4_label_offset_lqprint"}:
            alpha_range = (0.975, 1.0)
        elif profile in {
            "realistic_a4_tape_edge",
            "realistic_a4_tape_edge_visible",
            "realistic_a4_orange_icon_cluster",
            "realistic_a4_orange_icon_cluster_bright",
        }:
            alpha_range = (0.985, 1.0)
        elif profile in {"realistic_a4_fruit_visible", "realistic_a4_label_offset", "realistic_a4_sparse_icon"}:
            alpha_range = (0.985, 1.0)
        elif profile == "realistic_a4_hard":
            alpha_range = (0.86, 0.96)
        else:
            alpha_range = (0.90, 0.98)
        alpha = np.full((patch_size, patch_size, 1), random.uniform(*alpha_range), dtype=np.float32)
        if profile == "realistic_a4_hard" and random.random() < 0.32:
            patch = cv2.GaussianBlur(patch, (3, 3), 0)
        if profile == "realistic_a4_hard" and random.random() < 0.45:
            ok, encoded = cv2.imencode(".jpg", patch, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(62, 88)])
            if ok:
                patch = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if profile == "realistic_a4_label_offset_strong" and random.random() < 0.35:
            patch = cv2.GaussianBlur(patch, (3, 3), 0)
        if profile == "realistic_a4_label_offset_strong" and random.random() < 0.55:
            ok, encoded = cv2.imencode(".jpg", patch, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(55, 84)])
            if ok:
                patch = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if profile in {"realistic_a4_tape_edge", "realistic_a4_tape_edge_visible"} and random.random() < 0.18:
            patch = cv2.GaussianBlur(patch, (3, 3), 0)
        if profile in {"realistic_a4_tape_edge", "realistic_a4_tape_edge_visible"} and random.random() < 0.28:
            ok, encoded = cv2.imencode(".jpg", patch, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(70, 90)])
            if ok:
                patch = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if profile == "realistic_a4_label_offset_lqprint":
            low = max(24, int(patch_size * random.uniform(0.34, 0.62)))
            interp_up = random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC])
            patch = cv2.resize(
                cv2.resize(patch, (low, low), interpolation=cv2.INTER_AREA),
                (patch_size, patch_size),
                interpolation=interp_up,
            )
            if random.random() < 0.55:
                patch = cv2.GaussianBlur(patch, (3, 3), 0)
            ok, encoded = cv2.imencode(".jpg", patch, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(48, 78)])
            if ok:
                patch = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    else:
        alpha = fruit_foreground_alpha(patch, patch_size * random.uniform(0.010, 0.018))

    if profile in {"realistic_a4_mild", "realistic_a4_sparse_icon"}:
        cx = int(size * random.uniform(0.42, 0.58))
        cy = int(size * random.uniform(0.39, 0.58))
    elif profile == "realistic_a4_fruit_visible":
        cx = int(size * random.uniform(0.45, 0.55))
        cy = int(size * random.uniform(0.43, 0.56))
    elif profile == "realistic_a4_label_offset":
        if random.random() < 0.50:
            cx = int(size * random.choice([random.uniform(0.36, 0.46), random.uniform(0.54, 0.64)]))
            cy = int(size * random.uniform(0.40, 0.60))
        else:
            cx = int(size * random.uniform(0.40, 0.60))
            cy = int(size * random.choice([random.uniform(0.36, 0.46), random.uniform(0.54, 0.64)]))
    elif profile in {"realistic_a4_label_offset_strong", "realistic_a4_label_offset_lqprint"}:
        if random.random() < 0.60:
            cx = int(size * random.choice([random.uniform(0.30, 0.43), random.uniform(0.57, 0.70)]))
            cy = int(size * random.uniform(0.38, 0.62))
        else:
            cx = int(size * random.uniform(0.38, 0.62))
            cy = int(size * random.choice([random.uniform(0.31, 0.44), random.uniform(0.56, 0.69)]))
    elif profile in {"realistic_a4_tape_edge", "realistic_a4_tape_edge_visible"}:
        if random.random() < 0.45:
            cx = int(size * random.choice([random.uniform(0.34, 0.46), random.uniform(0.54, 0.66)]))
            cy = int(size * random.uniform(0.39, 0.61))
        else:
            cx = int(size * random.uniform(0.40, 0.60))
            cy = int(size * random.choice([random.uniform(0.35, 0.46), random.uniform(0.54, 0.65)]))
    elif profile in {"realistic_a4_orange_icon_cluster", "realistic_a4_orange_icon_cluster_bright"}:
        cx = int(size * random.uniform(0.47, 0.55))
        cy = int(size * random.uniform(0.45, 0.56))
    else:
        cx = int(size * random.uniform(0.43, 0.57))
        cy = int(size * random.uniform(0.40, 0.58))
    x0 = int(cx - patch_size / 2)
    y0 = int(cy - patch_size / 2)
    x1 = max(0, x0)
    y1 = max(0, y0)
    x2 = min(size, x0 + patch_size)
    y2 = min(size, y0 + patch_size)
    px1 = x1 - x0
    py1 = y1 - y0
    px2 = px1 + (x2 - x1)
    py2 = py1 + (y2 - y1)
    if x2 > x1 and y2 > y1:
        canvas_roi = canvas[y1:y2, x1:x2].astype(np.float32)
        patch_roi = patch[py1:py2, px1:px2].astype(np.float32)
        alpha_roi = alpha[py1:py2, px1:px2]
        canvas[y1:y2, x1:x2] = np.clip(patch_roi * alpha_roi + canvas_roi * (1.0 - alpha_roi), 0, 255).astype(np.uint8)

    if profile in {"realistic_a4_tape_edge", "realistic_a4_tape_edge_visible"}:
        canvas = overlay_print_label_tape(canvas, (x0, y0, x1, y1), paper_value, visible=profile == "realistic_a4_tape_edge_visible")

    if profile == "realistic_a4_boundary":
        noise_probability = 0.34
    elif profile in {"realistic_a4_label_offset_strong", "realistic_a4_label_offset_lqprint"}:
        noise_probability = 0.50
    elif profile in {"realistic_a4_tape_edge", "realistic_a4_tape_edge_visible"}:
        noise_probability = 0.30
    elif profile in {"realistic_a4_orange_icon_cluster", "realistic_a4_orange_icon_cluster_bright"}:
        noise_probability = 0.22
    elif profile in {"realistic_a4_fruit_visible", "realistic_a4_label_offset"}:
        noise_probability = 0.22
    elif profile == "realistic_a4_hard":
        noise_probability = 0.58
    elif profile in {"realistic_a4_mild", "realistic_a4_sparse_icon"}:
        noise_probability = 0.42
    else:
        noise_probability = 0.30
    if random.random() < noise_probability:
        if profile == "realistic_a4_boundary":
            sigma = random.uniform(0.2, 0.9)
        elif profile in {"realistic_a4_label_offset_strong", "realistic_a4_label_offset_lqprint"}:
            sigma = random.uniform(0.25, 1.15)
        elif profile in {
            "realistic_a4_tape_edge",
            "realistic_a4_tape_edge_visible",
            "realistic_a4_orange_icon_cluster",
            "realistic_a4_orange_icon_cluster_bright",
        }:
            sigma = random.uniform(0.10, 0.70)
        elif profile in {"realistic_a4_fruit_visible", "realistic_a4_label_offset"}:
            sigma = random.uniform(0.1, 0.55)
        elif profile == "realistic_a4_hard":
            sigma = random.uniform(0.4, 1.8)
        elif profile in {"realistic_a4_mild", "realistic_a4_sparse_icon"}:
            sigma = random.uniform(0.25, 1.0)
        else:
            sigma = random.uniform(0.4, 1.4)
        noise = np.random.normal(0, sigma, canvas.shape[:2]).astype(np.float32)
        canvas = np.clip(canvas.astype(np.float32) + noise[..., None], 0, 255).astype(np.uint8)
    return canvas, [Path(source_texture).as_posix()]


def make_fruit_print_label_texture_file(
    texture_dir,
    class_name,
    strength,
    output_dir,
    owner_name,
    face_idx,
    canonical=False,
    profile="legacy",
):
    image, source_textures = make_fruit_print_label_texture(
        texture_dir,
        class_name,
        strength,
        canonical=canonical,
        profile=profile,
    )
    texture_path = generated_face_texture_file(output_dir, owner_name, face_idx, class_name)
    cv2.imwrite(str(texture_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(90, 97)])
    return texture_path, source_textures


def make_augmented_fruit_texture(texture_dir, class_name, strength, texture_path=None, canonical=False):
    collage_probability = {
        "none": {"apple": 0.00, "banana": 0.00, "orange": 0.00, "pineapple": 0.00},
        "light": {"apple": 0.00, "banana": 0.00, "orange": 0.00, "pineapple": 0.00},
        "strong": {"apple": 0.00, "banana": 0.00, "orange": 0.00, "pineapple": 0.00},
    }[strength][class_name]
    if random.random() < collage_probability:
        return make_fruit_texture_collage(texture_dir, class_name, strength, canonical=canonical)[0]

    texture = texture_path or choose_fruit_texture(texture_dir, class_name, canonical=canonical)
    image = read_texture_bgr(texture)
    if image is None:
        return None
    return warp_fruit_texture(image, class_name, strength)


def sample_face_uv_transform(strength):
    if strength == "none":
        return {"rotation_deg": 0.0, "zoom": 1.0}
    angle_limit = 22 if strength == "light" else 45
    max_zoom = 1.32 if strength == "light" else 1.70
    angle_deg = random.uniform(-angle_limit, angle_limit)
    zoom = math.exp(random.uniform(-math.log(max_zoom), math.log(max_zoom)))
    return {"rotation_deg": angle_deg, "zoom": zoom}


def transformed_face_uvs(transform):
    angle = math.radians(float(transform.get("rotation_deg", 0.0)))
    zoom = max(0.05, float(transform.get("zoom", 1.0)))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    coords = []
    for u, v in [(0, 0), (1, 0), (1, 1), (0, 1)]:
        x = (u - 0.5) / zoom
        y = (v - 0.5) / zoom
        ru = x * cos_a - y * sin_a + 0.5
        rv = x * sin_a + y * cos_a + 0.5
        coords.append((ru, rv))
    return coords


def should_use_collage_texture(texture_layout, collage_probability):
    if texture_layout == "collage":
        return True
    if texture_layout == "mixed":
        return random.random() < max(0.0, min(1.0, float(collage_probability)))
    return False


def storage_texture_path(args, texture_record):
    if not texture_record:
        return None
    if texture_record.get("texture_layout") in {"collage", "print_label"} and not getattr(args, "keep_generated_face_textures", False):
        return None
    return texture_record.get("texture")


def storage_face_texture_records(args, records):
    if not isinstance(records, list):
        return records
    sanitized = []
    for record in records:
        if not isinstance(record, dict):
            sanitized.append(record)
            continue
        item = dict(record)
        if item.get("texture_layout") == "collage" and not getattr(args, "keep_generated_face_textures", False):
            item["texture"] = None
        sanitized.append(item)
    return sanitized


def apply_fruit_cube_materials(
    obj,
    texture_dir,
    class_name,
    texture_aug="strong",
    single_texture_per_cube=False,
    canonical_textures=False,
    texture_layout="single",
    collage_probability=0.35,
    generated_texture_dir=None,
    fruit_print_label_profile="legacy",
):
    if class_name not in FRUIT_CLASSES:
        raise ValueError(f"Fruit cube class must be one of {FRUIT_CLASSES}: {class_name}")
    if fruit_print_label_profile in PRINT_LABEL_REALISTIC_PROFILES:
        if fruit_print_label_profile == "realistic_a4_boundary":
            base = random.uniform(0.76, 0.90)
            blue_drop = random.uniform(0.08, 0.16)
            red_lift = random.uniform(0.035, 0.085)
            green_lift = random.uniform(0.015, 0.055)
        elif fruit_print_label_profile == "realistic_a4_fruit_visible":
            base = random.uniform(0.74, 0.88)
            blue_drop = random.uniform(0.10, 0.18)
            red_lift = random.uniform(0.035, 0.085)
            green_lift = random.uniform(0.015, 0.055)
        elif fruit_print_label_profile == "realistic_a4_hard":
            base = random.uniform(0.80, 0.94)
            blue_drop = random.uniform(0.025, 0.075)
            red_lift = random.uniform(0.006, 0.025)
            green_lift = random.uniform(0.000, 0.018)
        else:
            base = random.uniform(0.82, 0.95)
            blue_drop = random.uniform(0.015, 0.055)
            red_lift = random.uniform(0.006, 0.025)
            green_lift = random.uniform(0.000, 0.018)
        white_rgba = [
            min(1.0, base + red_lift),
            min(1.0, base + green_lift),
            max(0.0, base - blue_drop),
            1.0,
        ]
    else:
        white_rgba = [random.uniform(0.78, 0.96)] * 3 + [1.0]
    white = make_material("fruit_cube_white_pla", white_rgba, random.uniform(0.50, 0.90))
    obj.blender_obj.data.materials.clear()
    add_cube_uvs(obj.blender_obj)
    obj.blender_obj.data.materials.append(white.blender_obj if hasattr(white, "blender_obj") else white)
    make_materials_single_user(obj.blender_obj)
    for poly in obj.blender_obj.data.polygons:
        poly.material_index = 0

    owner_name = obj.blender_obj.name
    face_textures = choose_fruit_face_textures(
        texture_dir,
        class_name,
        len(FRUIT_FACE_NORMALS),
        single_texture_per_cube=single_texture_per_cube,
        canonical=canonical_textures,
    )
    face_texture_records = []
    for face_idx, (_, normal) in enumerate(FRUIT_FACE_NORMALS):
        source_textures = []
        face_texture_layout = "single"
        if texture_layout == "print_label":
            if generated_texture_dir is None:
                raise RuntimeError("--fruit_texture_layout print_label requires a generated texture output directory")
            face_texture, source_textures = make_fruit_print_label_texture_file(
                texture_dir,
                class_name,
                texture_aug,
                generated_texture_dir,
                owner_name,
                face_idx,
                canonical=canonical_textures,
                profile=fruit_print_label_profile,
            )
            face_texture_layout = "print_label"
        elif should_use_collage_texture(texture_layout, collage_probability):
            if generated_texture_dir is None:
                raise RuntimeError("--fruit_texture_layout collage/mixed requires a generated texture output directory")
            face_texture, source_textures = make_fruit_texture_collage_file(
                texture_dir,
                class_name,
                texture_aug,
                generated_texture_dir,
                owner_name,
                face_idx,
                canonical=canonical_textures,
            )
            face_texture_layout = "collage"
        else:
            face_texture = face_textures[face_idx]
        declared_class = declared_fruit_class_from_texture_path(face_texture) or class_name
        if declared_class != class_name:
            raise RuntimeError(
                f"Refusing mixed fruit texture on {owner_name}: "
                f"class={class_name}, texture={face_texture}, declared={declared_class}"
            )
        material_name = datablock_name(
            owner_name,
            "fruitface",
            face_idx,
            class_name,
            Path(face_texture).stem,
            f"{random.getrandbits(32):08x}",
        )
        material = make_image_material(material_name, face_texture)
        uv_transform = sample_face_uv_transform(texture_aug)
        overlay_info = create_fruit_face_overlay(
            obj,
            owner_name,
            face_idx,
            normal,
            material,
            fruit_class=class_name,
            texture_path=face_texture,
            texture_declared_class=declared_class,
            texture_layout=face_texture_layout,
            source_textures=source_textures,
            uv_transform=uv_transform,
        )
        face_texture_records.append({
            "face_index": face_idx,
            "class": class_name,
            "texture_declared_class": declared_class,
            "texture": Path(face_texture).as_posix(),
            "texture_layout": face_texture_layout,
            "print_label_profile": fruit_print_label_profile if face_texture_layout == "print_label" else "",
            "source_textures": source_textures,
            "texture_aug": texture_aug,
            "uv_rotation_deg": uv_transform["rotation_deg"],
            "uv_zoom": uv_transform["zoom"],
            "material_name": overlay_info.get("material_name"),
            "image_name": overlay_info.get("image_name"),
        })
    obj.set_cp("fruit_face_textures", json.dumps(face_texture_records, ensure_ascii=False))


def face_vertices_for_normal(normal, half=0.040, offset=0.00045):
    nx, ny, nz = normal
    if nx > 0:
        x = half + offset
        return [(x, -half, -half), (x, half, -half), (x, half, half), (x, -half, half)]
    if nx < 0:
        x = -half - offset
        return [(x, half, -half), (x, -half, -half), (x, -half, half), (x, half, half)]
    if ny > 0:
        y = half + offset
        return [(-half, y, -half), (half, y, -half), (half, y, half), (-half, y, half)]
    if ny < 0:
        y = -half - offset
        return [(half, y, -half), (-half, y, -half), (-half, y, half), (half, y, half)]
    if nz > 0:
        z = half + offset
        return [(-half, -half, z), (half, -half, z), (half, half, z), (-half, half, z)]
    raise ValueError(f"Unsupported fruit face normal: {normal}")


def create_fruit_face_overlay(
    cube_obj,
    owner_name,
    face_idx,
    normal,
    material,
    fruit_class,
    texture_path,
    texture_declared_class,
    texture_layout,
    source_textures,
    uv_transform,
):
    overlay_name = datablock_name(
        "fruitface",
        owner_name,
        face_idx,
        fruit_class,
        Path(texture_path).stem,
        f"{random.getrandbits(32):08x}",
    )
    mesh = bpy.data.meshes.new(datablock_name(overlay_name, "mesh"))
    mesh.from_pydata(face_vertices_for_normal(normal), [], [(0, 1, 2, 3)])
    mesh.update()
    uv_layer = mesh.uv_layers.new(name="UVMap")
    coords = transformed_face_uvs(uv_transform)
    for i, loop_index in enumerate(mesh.polygons[0].loop_indices):
        uv_layer.data[loop_index].uv = coords[i]

    bpy_obj = bpy.data.objects.new(overlay_name, mesh)
    bpy.context.collection.objects.link(bpy_obj)
    bpy_obj.parent = cube_obj.blender_obj
    bpy_obj.location = (0.0, 0.0, 0.0)
    bpy_obj.rotation_euler = (0.0, 0.0, 0.0)
    bpy_obj.scale = (1.0, 1.0, 1.0)
    bpy_obj.data.materials.append(material)

    face_obj = bproc.object.convert_to_meshes([bpy_obj])[0]
    make_materials_single_user(face_obj.blender_obj, name_prefix=overlay_name)
    actual_material_name, actual_image_name = first_material_image_name(face_obj.blender_obj)
    face_obj.set_cp("category_id", FRUIT_FACE_HELPER_CATEGORY_ID)
    face_obj.set_cp("supercategory", SUPER_CATEGORY)
    face_obj.set_cp("fruit_face_owner", owner_name)
    face_obj.set_cp("fruit_face_index", face_idx)
    face_obj.set_cp("fruit_face_class", fruit_class)
    face_obj.set_cp("fruit_face_texture", Path(texture_path).as_posix())
    face_obj.set_cp("fruit_face_declared_class", texture_declared_class)
    face_obj.set_cp("fruit_face_texture_layout", texture_layout)
    face_obj.set_cp("fruit_face_source_textures", json.dumps(source_textures or [], ensure_ascii=False))
    face_obj.set_cp("fruit_face_material", actual_material_name or "")
    face_obj.set_cp("fruit_face_image", actual_image_name or "")
    face_obj.set_cp("fruit_face_uv_rotation_deg", float(uv_transform["rotation_deg"]))
    face_obj.set_cp("fruit_face_uv_zoom", float(uv_transform["zoom"]))
    face_obj.blender_obj["cp_fruit_face_owner"] = owner_name
    face_obj.blender_obj["cp_fruit_face_index"] = int(face_idx)
    face_obj.blender_obj["cp_fruit_face_class"] = fruit_class
    face_obj.blender_obj["cp_fruit_face_texture"] = Path(texture_path).as_posix()
    face_obj.blender_obj["cp_fruit_face_declared_class"] = texture_declared_class
    face_obj.blender_obj["cp_fruit_face_texture_layout"] = texture_layout
    face_obj.blender_obj["cp_fruit_face_source_textures"] = json.dumps(source_textures or [], ensure_ascii=False)
    face_obj.blender_obj["cp_fruit_face_material"] = actual_material_name or ""
    face_obj.blender_obj["cp_fruit_face_image"] = actual_image_name or ""
    face_obj.blender_obj["cp_fruit_face_uv_rotation_deg"] = float(uv_transform["rotation_deg"])
    face_obj.blender_obj["cp_fruit_face_uv_zoom"] = float(uv_transform["zoom"])
    return {"object": face_obj, "material_name": actual_material_name, "image_name": actual_image_name}


def create_fruit_cube(
    class_name,
    idx,
    texture_dir,
    texture_aug="strong",
    single_texture_per_cube=False,
    canonical_textures=False,
    texture_layout="single",
    collage_probability=0.35,
    generated_texture_dir=None,
    fruit_print_label_profile="legacy",
):
    bpy.ops.mesh.primitive_cube_add(size=0.08, location=[0, 0, 0])
    bpy_obj = bpy.context.object
    bpy_obj.name = f"{class_name}_{idx:03d}"
    obj = bproc.object.convert_to_meshes([bpy_obj])[0]
    obj.set_cp("category_id", CLASS_TO_ID[class_name])
    obj.set_cp("supercategory", SUPER_CATEGORY)
    apply_fruit_cube_materials(
        obj,
        texture_dir,
        class_name,
        texture_aug=texture_aug,
        single_texture_per_cube=single_texture_per_cube,
        canonical_textures=canonical_textures,
        texture_layout=texture_layout,
        collage_probability=collage_probability,
        generated_texture_dir=generated_texture_dir,
        fruit_print_label_profile=fruit_print_label_profile,
    )
    add_small_bevel(obj)
    return obj


def add_small_bevel(obj):
    try:
        bevel = obj.blender_obj.modifiers.new("tiny_print_bevel", "BEVEL")
        bevel.width = random.uniform(0.0002, 0.0012)
        bevel.segments = 1
        normal = obj.blender_obj.modifiers.new("weighted_normal", "WEIGHTED_NORMAL")
        normal.keep_sharp = True
    except Exception:
        pass


def sample_arena_location(existing, scene_profile="normal", wall_contact_ratio=0.25, corner_scene_ratio=0.10):
    wall_scene = random.random() < wall_contact_ratio
    corner_scene = random.random() < corner_scene_ratio
    if scene_profile == "closeup":
        x_range = (-0.28, 0.28)
        y_range = (-0.32, 0.05)
    else:
        x_range = (-0.68, 0.68)
        y_range = (-0.34, 0.48)

    for _ in range(220):
        if corner_scene:
            x = random.choice([random.uniform(x_range[0], -0.46), random.uniform(0.46, x_range[1])])
            y = random.uniform(0.32, y_range[1])
        elif wall_scene:
            x = random.uniform(*x_range)
            y = random.uniform(0.30, y_range[1])
        elif random.random() < 0.12 and existing:
            base = random.choice(existing)
            x = base[0] + random.uniform(-0.22, 0.22)
            y = base[1] + random.uniform(-0.18, 0.18)
        else:
            x = random.uniform(*x_range)
            y = random.uniform(*y_range)
        z = random.uniform(-0.08, 0.12)
        if all(np.linalg.norm(np.array([x, y]) - np.array(p[:2])) > 0.15 for p in existing):
            return x, y, z
    return random.uniform(*x_range), random.uniform(*y_range), random.uniform(-0.08, 0.12)


def sample_location(existing):
    for _ in range(200):
        if random.random() < 0.05 and existing:
            base = random.choice(existing)
            x = base[0] + random.uniform(-0.26, 0.26)
            y = base[1] + random.uniform(-0.20, 0.20)
        else:
            x = random.uniform(-0.48, 0.48)
            y = random.uniform(-0.22, 0.30)
        z = random.uniform(-0.08, 0.14)
        if all(np.linalg.norm(np.array([x, y]) - np.array(p[:2])) > 0.16 for p in existing):
            return x, y, z
    return random.uniform(-0.48, 0.48), random.uniform(-0.22, 0.30), random.uniform(-0.08, 0.14)


def sample_location_for_tier(existing, tier, arena_booster_mode=False, scene_profile="normal", wall_contact_ratio=0.25, corner_scene_ratio=0.10):
    if arena_booster_mode:
        return sample_arena_location(existing, scene_profile, wall_contact_ratio, corner_scene_ratio)
    if not tier or tier == "easy":
        return sample_location(existing)
    if tier == "mid":
        x_range = (-0.60, 0.60)
        y_range = (-0.30, 0.36)
    else:
        x_range = (-0.72, 0.72)
        y_range = (-0.38, 0.44)
    for _ in range(200):
        if random.random() < 0.70:
            x = random.choice([random.uniform(x_range[0], -0.34), random.uniform(0.34, x_range[1])])
        else:
            x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(*y_range)
        z = random.uniform(-0.08, 0.14)
        if all(np.linalg.norm(np.array([x, y]) - np.array(p[:2])) > 0.16 for p in existing):
            return x, y, z
    return random.uniform(*x_range), random.uniform(*y_range), random.uniform(-0.08, 0.14)


def random_rotation_euler():
    return [
        random.uniform(0, math.pi * 2),
        random.uniform(0, math.pi * 2),
        random.uniform(0, math.pi * 2),
    ]


def fruit_rotation_euler(tier):
    if tier == "easy":
        tilt = math.radians(15)
    elif tier == "mid":
        tilt = math.radians(65)
    else:
        return random_rotation_euler()
    return [
        random.uniform(-tilt, tilt),
        random.uniform(-tilt, tilt),
        random.uniform(0, math.pi * 2),
    ]


def sample_fruit_visibility_tier(forced=None, tier_weights=None):
    if forced and forced != "mixed":
        return forced
    weights = tier_weights or FRUIT_VISIBILITY_TIERS
    total = sum(max(0.0, float(weight)) for _, weight in weights)
    if total <= 0.0:
        weights = FRUIT_VISIBILITY_TIERS
        total = sum(weight for _, weight in weights)
    r = random.random()
    cumulative = 0.0
    for tier, weight in weights:
        cumulative += max(0.0, float(weight)) / total
        if r <= cumulative:
            return tier
    return weights[-1][0]


def fruit_scale_multiplier(tier):
    if tier == "easy":
        return random.uniform(1.08, 1.18)
    if tier == "mid":
        return random.uniform(0.98, 1.08)
    return random.uniform(0.82, 1.00)


def sample_object_scale(scale_min, scale_max, object_count, single_object_scale_min):
    if object_count <= 1:
        lower = max(scale_min, single_object_scale_min)
        mode = min(scale_max, max(lower, lower + (scale_max - lower) * 0.38))
    else:
        lower = scale_min
        mode = lower + (scale_max - lower) * 0.46

    # Triangular sampling keeps rare small/large examples, while making the
    # middle sizes much more common than the previous uniform distribution.
    if random.random() < 0.86:
        return random.triangular(lower, scale_max, mode)
    return random.uniform(lower, scale_max)


def sample_target_classes(
    n_obj,
    fruit_only=False,
    single_fruit_class_per_image=False,
    fruit_class_weight_scale=1.0,
    apple_orange_boost=False,
    force_plain_cubes=False,
    force_fruit_class="",
):
    if force_plain_cubes:
        return ["cube"] * n_obj
    if force_fruit_class:
        if force_fruit_class not in FRUIT_CLASSES:
            raise ValueError(f"force_fruit_class must be one of {FRUIT_CLASSES}: {force_fruit_class}")
        if fruit_only:
            return [force_fruit_class] * n_obj
    classes = FRUIT_CLASSES if fruit_only else SHAPE_CLASSES + FRUIT_CLASSES
    fruit_weights_by_name = ARENA_BOOSTER_FRUIT_CLASS_WEIGHTS if apple_orange_boost else FRUIT_CLASS_WEIGHTS
    if fruit_only:
        weights = [fruit_weights_by_name[name] for name in FRUIT_CLASSES]
        if single_fruit_class_per_image:
            class_name = random.choices(classes, weights=weights, k=1)[0]
            return [class_name] * n_obj
        return random.choices(classes, weights=weights, k=n_obj)
    weights = [SHAPE_CLASS_WEIGHTS[name] for name in SHAPE_CLASSES]
    fruit_scale = max(0.0, float(fruit_class_weight_scale))
    weights += [fruit_weights_by_name[name] * fruit_scale for name in FRUIT_CLASSES]
    chosen = random.choices(classes, weights=weights, k=n_obj)
    if force_fruit_class:
        chosen = [force_fruit_class if name in FRUIT_CLASSES else name for name in chosen]
    if single_fruit_class_per_image and any(name in FRUIT_CLASSES for name in chosen):
        if force_fruit_class:
            scene_fruit_class = force_fruit_class
        else:
            fruit_weights = [fruit_weights_by_name[name] for name in FRUIT_CLASSES]
            scene_fruit_class = random.choices(FRUIT_CLASSES, weights=fruit_weights, k=1)[0]
        chosen = [scene_fruit_class if name in FRUIT_CLASSES else name for name in chosen]
    return chosen


def polygon_area(points):
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def clip_polygon_to_unit_square(points):
    def clip_edge(poly, inside, intersect):
        if not poly:
            return []
        out = []
        prev = poly[-1]
        prev_inside = inside(prev)
        for curr in poly:
            curr_inside = inside(curr)
            if curr_inside:
                if not prev_inside:
                    out.append(intersect(prev, curr))
                out.append(curr)
            elif prev_inside:
                out.append(intersect(prev, curr))
            prev = curr
            prev_inside = curr_inside
        return out

    def intersect_x(value):
        def fn(p1, p2):
            x1, y1 = p1
            x2, y2 = p2
            denom = x2 - x1
            t = 0.0 if abs(denom) < 1e-9 else (value - x1) / denom
            return value, y1 + t * (y2 - y1)
        return fn

    def intersect_y(value):
        def fn(p1, p2):
            x1, y1 = p1
            x2, y2 = p2
            denom = y2 - y1
            t = 0.0 if abs(denom) < 1e-9 else (value - y1) / denom
            return x1 + t * (x2 - x1), value
        return fn

    poly = clip_edge(points, lambda p: p[0] >= 0.0, intersect_x(0.0))
    poly = clip_edge(poly, lambda p: p[0] <= 1.0, intersect_x(1.0))
    poly = clip_edge(poly, lambda p: p[1] >= 0.0, intersect_y(0.0))
    poly = clip_edge(poly, lambda p: p[1] <= 1.0, intersect_y(1.0))
    return poly


def projected_points_for_vertices(scene, camera, matrix, vertices):
    points = []
    for vertex in vertices:
        camera_co = world_to_camera_view(scene, camera, matrix @ vertex.co)
        if camera_co.z <= 0:
            return []
        points.append((float(camera_co.x), float(1.0 - camera_co.y)))
    return points


def bbox_area(points):
    if not points:
        return 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return max(0.0, max(xs) - min(xs)) * max(0.0, max(ys) - min(ys))


def convex_hull_points(points):
    if len(points) < 3:
        return points
    pts = np.array(points, dtype=np.float32)
    hull = cv2.convexHull(pts).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in hull]


def projected_object_silhouette_area_pixels(obj, width, height):
    scene = bpy.context.scene
    camera = scene.camera
    if camera is None:
        return 0.0
    bpy.context.view_layer.update()
    points = projected_points_for_vertices(scene, camera, obj.blender_obj.matrix_world, obj.blender_obj.data.vertices)
    if len(points) < 3:
        return 0.0
    return polygon_area(convex_hull_points(points)) * float(width * height)


def projected_mesh_silhouette_mask(obj, width, height):
    scene = bpy.context.scene
    camera = scene.camera
    if camera is None:
        return np.zeros((height, width), dtype=np.uint8)

    bpy.context.view_layer.update()
    mesh = obj.blender_obj.data
    matrix = obj.blender_obj.matrix_world
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in mesh.polygons:
        points = []
        behind_camera = False
        for vertex_idx in poly.vertices:
            camera_co = world_to_camera_view(scene, camera, matrix @ mesh.vertices[vertex_idx].co)
            if camera_co.z <= 0:
                behind_camera = True
                break
            points.append((float(camera_co.x), float(1.0 - camera_co.y)))
        if behind_camera or len(points) < 3:
            continue

        clipped = clip_polygon_to_unit_square(points)
        if len(clipped) < 3:
            continue

        pts = np.empty((len(clipped), 2), dtype=np.int32)
        pts[:, 0] = np.clip(np.rint([p[0] * (width - 1) for p in clipped]), 0, width - 1).astype(np.int32)
        pts[:, 1] = np.clip(np.rint([p[1] * (height - 1) for p in clipped]), 0, height - 1).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def mask_pixel_area(mask):
    if mask is None:
        return 0
    return int(np.count_nonzero(mask))


def projected_polygon_pixels(scene, camera, matrix, mesh, poly, width, height):
    points = []
    depths = []
    for vertex_idx in poly.vertices:
        camera_co = world_to_camera_view(scene, camera, matrix @ mesh.vertices[vertex_idx].co)
        if camera_co.z <= 0:
            return None, None
        points.append((float(camera_co.x), float(1.0 - camera_co.y)))
        depths.append(float(camera_co.z))

    clipped = clip_polygon_to_unit_square(points)
    if len(clipped) < 3:
        return None, None

    pts = np.empty((len(clipped), 2), dtype=np.int32)
    pts[:, 0] = np.clip(np.rint([p[0] * (width - 1) for p in clipped]), 0, width - 1).astype(np.int32)
    pts[:, 1] = np.clip(np.rint([p[1] * (height - 1) for p in clipped]), 0, height - 1).astype(np.int32)
    return pts, float(np.mean(depths))


def fruit_face_children(owner_obj):
    owner_name = owner_obj.blender_obj.name
    children = []
    for bpy_obj in bpy.context.scene.objects:
        if bpy_obj.parent != owner_obj.blender_obj:
            continue
        prop_owner = bpy_obj.get("cp_fruit_face_owner", bpy_obj.get("fruit_face_owner", ""))
        if prop_owner == owner_name or (not prop_owner and fruit_face_owner_from_name(bpy_obj.name) == owner_name):
            children.append(bpy_obj)
    return children


def blender_custom_prop(bpy_obj, name, default=None):
    return bpy_obj.get(f"cp_{name}", bpy_obj.get(name, default))


def ideal_scene_segmentation(objects, width, height):
    scene = bpy.context.scene
    camera = scene.camera
    instance_segmap = np.zeros((height, width), dtype=np.int32)
    depth_map = np.full((height, width), np.inf, dtype=np.float32)
    attr_map = []
    object_full_masks = {}
    next_idx = 1

    def add_attr(
        category_id,
        basename,
        fruit_face_owner="",
        fruit_face_index=None,
        fruit_face_class="",
        fruit_face_texture="",
        fruit_face_declared_class="",
        fruit_face_texture_layout="",
        fruit_face_source_textures="",
        fruit_face_material="",
        fruit_face_image="",
        fruit_face_uv_rotation_deg=None,
        fruit_face_uv_zoom=None,
    ):
        nonlocal next_idx
        idx = next_idx
        next_idx += 1
        attr = {
            "idx": idx,
            "category_id": int(category_id),
            "cf_basename": basename,
            "basename": basename,
            "cp_supercategory": SUPER_CATEGORY,
            "cp_fruit_face_owner": fruit_face_owner,
        }
        if fruit_face_owner:
            attr["fruit_face_owner"] = fruit_face_owner
        if fruit_face_index is not None:
            attr["fruit_face_index"] = int(fruit_face_index)
            attr["cp_fruit_face_index"] = int(fruit_face_index)
        if fruit_face_class:
            attr["fruit_face_class"] = fruit_face_class
            attr["cp_fruit_face_class"] = fruit_face_class
        if fruit_face_texture:
            attr["fruit_face_texture"] = fruit_face_texture
            attr["cp_fruit_face_texture"] = fruit_face_texture
        if fruit_face_declared_class:
            attr["fruit_face_declared_class"] = fruit_face_declared_class
            attr["cp_fruit_face_declared_class"] = fruit_face_declared_class
        if fruit_face_texture_layout:
            attr["fruit_face_texture_layout"] = fruit_face_texture_layout
            attr["cp_fruit_face_texture_layout"] = fruit_face_texture_layout
        if fruit_face_source_textures:
            attr["fruit_face_source_textures"] = fruit_face_source_textures
            attr["cp_fruit_face_source_textures"] = fruit_face_source_textures
        if fruit_face_material:
            attr["fruit_face_material"] = fruit_face_material
            attr["cp_fruit_face_material"] = fruit_face_material
        if fruit_face_image:
            attr["fruit_face_image"] = fruit_face_image
            attr["cp_fruit_face_image"] = fruit_face_image
        if fruit_face_uv_rotation_deg is not None:
            attr["fruit_face_uv_rotation_deg"] = float(fruit_face_uv_rotation_deg)
            attr["cp_fruit_face_uv_rotation_deg"] = float(fruit_face_uv_rotation_deg)
        if fruit_face_uv_zoom is not None:
            attr["fruit_face_uv_zoom"] = float(fruit_face_uv_zoom)
            attr["cp_fruit_face_uv_zoom"] = float(fruit_face_uv_zoom)
        attr_map.append(attr)
        return idx

    face_records = []
    for obj in objects:
        bpy_obj = obj.blender_obj
        obj_name = bpy_obj.name
        try:
            category_id = int(obj.get_cp("category_id"))
        except Exception:
            category_id = 0

        base_idx = add_attr(category_id, obj_name)
        object_full_masks[obj_name] = projected_mesh_silhouette_mask(obj, width, height)
        for poly in bpy_obj.data.polygons:
            pts, depth = projected_polygon_pixels(scene, camera, bpy_obj.matrix_world, bpy_obj.data, poly, width, height)
            if pts is not None:
                face_records.append((depth, base_idx, pts))

        for face_bpy_obj in fruit_face_children(obj):
            face_idx = add_attr(
                FRUIT_FACE_HELPER_CATEGORY_ID,
                face_bpy_obj.name,
                obj_name,
                fruit_face_index=blender_custom_prop(face_bpy_obj, "fruit_face_index"),
                fruit_face_class=blender_custom_prop(face_bpy_obj, "fruit_face_class", ""),
                fruit_face_texture=blender_custom_prop(face_bpy_obj, "fruit_face_texture", ""),
                fruit_face_declared_class=blender_custom_prop(face_bpy_obj, "fruit_face_declared_class", ""),
                fruit_face_texture_layout=blender_custom_prop(face_bpy_obj, "fruit_face_texture_layout", ""),
                fruit_face_source_textures=blender_custom_prop(face_bpy_obj, "fruit_face_source_textures", ""),
                fruit_face_material=blender_custom_prop(face_bpy_obj, "fruit_face_material", ""),
                fruit_face_image=blender_custom_prop(face_bpy_obj, "fruit_face_image", ""),
                fruit_face_uv_rotation_deg=blender_custom_prop(face_bpy_obj, "fruit_face_uv_rotation_deg", None),
                fruit_face_uv_zoom=blender_custom_prop(face_bpy_obj, "fruit_face_uv_zoom", None),
            )
            for poly in face_bpy_obj.data.polygons:
                pts, depth = projected_polygon_pixels(
                    scene,
                    camera,
                    face_bpy_obj.matrix_world,
                    face_bpy_obj.data,
                    poly,
                    width,
                    height,
                )
                if pts is not None:
                    face_records.append((depth, face_idx, pts))

    for depth, instance_idx, pts in sorted(face_records, key=lambda item: item[0], reverse=True):
        poly_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [pts], 1)
        update = (poly_mask > 0) & (depth < depth_map)
        instance_segmap[update] = instance_idx
        depth_map[update] = depth

    return instance_segmap, attr_map, object_full_masks


def class_color_bgr(class_id):
    palette = {
        1: (40, 230, 240),
        2: (40, 150, 255),
        3: (60, 220, 180),
        4: (60, 60, 255),
        5: (255, 70, 70),
        6: (70, 220, 70),
        7: (70, 140, 255),
        8: (210, 90, 255),
        FRUIT_FACE_HELPER_CATEGORY_ID: (40, 255, 255),
    }
    return palette.get(int(class_id), (180, 180, 180))


def draw_panel_title(image, text):
    cv2.rectangle(image, (0, 0), (image.shape[1], 32), (245, 245, 245), -1)
    cv2.putText(image, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (35, 35, 35), 1, cv2.LINE_AA)


def draw_mask_overlay(base, mask, color, alpha=0.36):
    if not np.any(mask):
        return
    overlay = base.copy()
    overlay[mask] = color
    cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0, base)
    contours, _ = cv2.findContours((mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(base, contours, -1, color, 2, cv2.LINE_AA)


def draw_instance_label(image, mask, text, color):
    if not np.any(mask):
        return
    ys, xs = np.where(mask)
    x = int(xs.min())
    y = int(ys.min())
    y = max(16, y - 5)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.43
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(image, (x, y - th - baseline - 5), (x + tw + 8, y + 3), color, -1)
    cv2.putText(image, text, (x + 4, y - 3), font, scale, (20, 20, 20), thickness, cv2.LINE_AA)


def caption_panel(width, rows):
    row_h = 22
    h = max(46, 30 + row_h * max(1, len(rows)))
    panel = np.full((h, width, 3), 246, dtype=np.uint8)
    columns = [
        ("#", 10),
        ("object", 38),
        ("target", 170),
        ("actual", 245),
        ("fruit", 320),
        ("obj", 380),
        ("fruit_px", 435),
    ]
    for title, x in columns:
        cv2.putText(panel, title, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (55, 55, 55), 1, cv2.LINE_AA)
    for idx, row in enumerate(rows, start=1):
        y = 20 + idx * row_h
        color = row["color"]
        cv2.rectangle(panel, (10, y - 12), (21, y - 1), color, -1)
        values = [
            (str(idx), 24),
            (row["name"], 38),
            (row["target"], 170),
            (row["actual"], 245),
            (row["fruit"], 320),
            (row["obj"], 380),
            (str(row["face_px"]), 435),
        ]
        for text, x in values:
            cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (35, 35, 35), 1, cv2.LINE_AA)
    return panel


def save_ideal_visibility_debug(output, image_id, instance_segmap, attr_map, fruit_visibility, object_full_masks):
    out_dir = Path(output) / "ideal_visibility_debug" / "train"
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = instance_segmap.shape[:2]
    attr_by_idx = {int(inst["idx"]): inst for inst in attr_map}

    full_panel = np.zeros((h, w, 3), dtype=np.uint8)
    visible_panel = np.zeros((h, w, 3), dtype=np.uint8)
    face_panel = np.zeros((h, w, 3), dtype=np.uint8)
    rows = []
    object_idx_by_name = {name: idx for idx, name in enumerate(object_full_masks, start=1)}

    for name, mask in object_full_masks.items():
        color = class_color_bgr(object_idx_by_name[name])
        draw_mask_overlay(full_panel, mask.astype(bool), color, alpha=0.30)
        draw_instance_label(full_panel, mask.astype(bool), f"#{object_idx_by_name[name]} {name.split('_', 1)[0]}", color)

    face_masks_by_owner = {}
    for idx in np.unique(instance_segmap):
        if idx == 0:
            continue
        inst = attr_by_idx.get(int(idx), {})
        category_id = int(inst.get("category_id", 0))
        color = class_color_bgr(category_id)
        mask = instance_segmap == idx
        owner = instance_fruit_face_owner(inst)
        if category_id == FRUIT_FACE_HELPER_CATEGORY_ID:
            if owner:
                owner_idx = object_idx_by_name.get(owner, 0)
                color = class_color_bgr(owner_idx)
                draw_mask_overlay(visible_panel, mask, color, alpha=0.36)
            draw_mask_overlay(face_panel, mask, color, alpha=0.54)
            if owner:
                if owner not in face_masks_by_owner:
                    face_masks_by_owner[owner] = np.zeros((h, w), dtype=bool)
                face_masks_by_owner[owner] |= mask
            continue
        name = instance_basename(inst)
        draw_mask_overlay(visible_panel, mask, color, alpha=0.36)
        draw_instance_label(visible_panel, mask, f"#{object_idx_by_name.get(name, 0)} {name.split('_', 1)[0]}", color)

    for owner, mask in face_masks_by_owner.items():
        draw_instance_label(face_panel, mask, f"#{object_idx_by_name.get(owner, 0)} fruit face", class_color_bgr(object_idx_by_name.get(owner, 0)))

    for name in object_full_masks:
        info = fruit_visibility.get(name, {})
        full_mask = object_full_masks.get(name)
        visible_mask = np.zeros((h, w), dtype=bool)
        for idx in np.unique(instance_segmap):
            if idx == 0:
                continue
            inst = attr_by_idx.get(int(idx), {})
            if instance_basename(inst) == name or instance_fruit_face_owner(inst) == name:
                visible_mask |= instance_segmap == idx
        obj_ratio = mask_visible_ratio(visible_mask, mask_pixel_area(full_mask))
        is_fruit = name.split("_", 1)[0] in FRUIT_CLASSES
        rows.append({
            "name": name,
            "target": str(info.get("target_visibility_tier") or "-") if is_fruit else "-",
            "actual": str(info.get("actual_visibility_tier") or "-") if is_fruit else "-",
            "fruit": f"{float(info.get('fruit_visible_ratio') or 0):.2f}" if is_fruit else "-",
            "obj": f"{obj_ratio:.2f}",
            "face_px": int(info.get("visible_face_pixels") or 0) if is_fruit else "-",
            "color": class_color_bgr(object_idx_by_name.get(name, 0)),
        })

    draw_panel_title(full_panel, "1. full ideal silhouettes")
    draw_panel_title(visible_panel, "2. ideal visible objects")
    draw_panel_title(face_panel, "3. ideal visible fruit faces")
    canvas = np.vstack([
        np.hstack([full_panel, visible_panel, face_panel]),
        caption_panel(w * 3, rows),
    ])
    cv2.imwrite(str(out_dir / f"{image_id:06d}_ideal_visibility.jpg"), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def projected_frame_overflow(obj, margin=0.08):
    scene = bpy.context.scene
    camera = scene.camera
    if camera is None:
        return float("inf")
    bpy.context.view_layer.update()
    points = projected_points_for_vertices(scene, camera, obj.blender_obj.matrix_world, obj.blender_obj.data.vertices)
    if len(points) < 3:
        return float("inf")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return max(
        0.0,
        margin - min(xs),
        max(xs) - (1.0 - margin),
        margin - min(ys),
        max(ys) - (1.0 - margin),
    )


def fruit_photo_to_cube_screen_ratio(obj):
    scene = bpy.context.scene
    camera = scene.camera
    if camera is None:
        return 1.0

    bpy.context.view_layer.update()
    mesh = obj.blender_obj.data
    matrix = obj.blender_obj.matrix_world
    camera_loc = camera.matrix_world.translation
    cube_points = projected_points_for_vertices(scene, camera, matrix, mesh.vertices)
    cube_area = bbox_area(cube_points)
    if cube_area <= 1e-9:
        return 0.0

    visible_fruit_area = 0.0
    for poly in mesh.polygons:
        normal = poly.normal
        if not (abs(normal.x) > 0.5 or normal.y > 0.5):
            continue

        world_center = matrix @ poly.center
        world_normal = matrix.to_3x3() @ normal
        if world_normal.dot(camera_loc - world_center) <= 0:
            continue

        projected = []
        in_front = True
        for vertex_index in poly.vertices:
            world_co = matrix @ mesh.vertices[vertex_index].co
            camera_co = world_to_camera_view(scene, camera, world_co)
            if camera_co.z <= 0:
                in_front = False
                break
            projected.append((float(camera_co.x), float(1.0 - camera_co.y)))
        if not in_front:
            continue

        full_area = polygon_area(projected)
        if full_area <= 1e-9:
            continue
        clipped_area = polygon_area(clip_polygon_to_unit_square(projected))
        visible_fruit_area += clipped_area
    return min(1.0, visible_fruit_area / cube_area)


def fruit_visibility_in_tier(ratio, tier):
    min_ratio, max_ratio = FRUIT_VISIBILITY_RANGES[tier]
    return min_ratio <= ratio < max_ratio


def actual_fruit_visibility_tier(ratio):
    if ratio is None:
        return None
    ratio = float(ratio)
    for tier in ("hard", "mid", "easy"):
        min_ratio, max_ratio = FRUIT_VISIBILITY_RANGES[tier]
        if min_ratio <= ratio < max_ratio:
            return tier
    if ratio >= FRUIT_VISIBILITY_RANGES["easy"][1]:
        return "easy"
    return "below_hard"


def fruit_visibility_tier_distance(ratio, tier):
    min_ratio, max_ratio = FRUIT_VISIBILITY_RANGES[tier]
    if ratio < min_ratio:
        return min_ratio - ratio
    if ratio >= max_ratio:
        return ratio - max_ratio
    return 0.0


def create_white_distractor(idx):
    shape = random.choice(["cylinder", "sphere"])
    if shape == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=random.choice([48, 64, 96]),
            radius=random.uniform(0.025, 0.075),
            depth=random.uniform(0.02, 0.11),
        )
    else:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=random.uniform(0.025, 0.085))
    bpy_obj = bpy.context.object
    bpy_obj.name = f"white_distractor_{idx:03d}"
    obj = bproc.object.convert_to_meshes([bpy_obj])[0]
    obj.replace_materials(make_material(
        "unlabeled_white_distractor",
        [random.uniform(0.70, 0.96), random.uniform(0.70, 0.96), random.uniform(0.70, 0.96), 1.0],
        random.uniform(0.55, 0.98),
    ))
    set_background_category(obj)
    add_small_bevel(obj)
    return obj


def load_targets(
    asset_dir,
    fruit_texture_dir,
    min_objects,
    max_objects,
    scale_min,
    scale_max,
    fruit_visibility_tier="mixed",
    fruit_only=False,
    fruit_texture_aug="strong",
    negative=False,
    min_projected_area=900,
    min_fruit_face_pixels=300,
    hard_min_fruit_face_pixels=-1,
    single_object_scale_min=0.82,
    single_object_min_projected_area=2600,
    single_object_min_fruit_face_pixels=850,
    width=640,
    height=640,
    single_fruit_class_per_image=False,
    single_fruit_texture_per_cube=False,
    fruit_class_weight_scale=1.20,
    fruit_visibility_weights=None,
    canonical_fruit_textures=False,
    fruit_texture_layout="single",
    fruit_texture_collage_prob=0.35,
    generated_texture_dir=None,
    fruit_print_label_profile="legacy",
    arena_booster_mode=False,
    arena_scene_profile="normal",
    wall_contact_ratio=0.25,
    corner_scene_ratio=0.10,
    apple_orange_boost=False,
    force_fruit_class="",
):
    asset_dir = Path(asset_dir)
    if negative:
        n_obj = random.randint(1, max(3, max_objects // 2))
        chosen = ["__distractor__"] * n_obj
    else:
        if arena_scene_profile == "plain_cube_hard_negative":
            n_obj = random.randint(1, max(1, min(3, max_objects)))
        elif arena_scene_profile == "closeup":
            n_obj = random.randint(1, max(1, min(3, max_objects)))
        else:
            n_obj = random.randint(min_objects, max_objects)
        chosen = sample_target_classes(
            n_obj,
            fruit_only=fruit_only,
            single_fruit_class_per_image=single_fruit_class_per_image,
            fruit_class_weight_scale=fruit_class_weight_scale,
            apple_orange_boost=apple_orange_boost,
            force_plain_cubes=arena_scene_profile == "plain_cube_hard_negative",
            force_fruit_class=force_fruit_class,
        )
    objects = []
    locations = []
    effective_min_projected_area = single_object_min_projected_area if len(chosen) == 1 else min_projected_area
    effective_min_fruit_face_pixels = single_object_min_fruit_face_pixels if len(chosen) == 1 else min_fruit_face_pixels

    for idx, cls_name in enumerate(chosen):
        fruit_tier = None
        if cls_name == "__distractor__":
            obj = create_white_distractor(idx)
        elif cls_name in FRUIT_CLASSES:
            fruit_tier = sample_fruit_visibility_tier(fruit_visibility_tier, fruit_visibility_weights)
            obj = create_fruit_cube(
                cls_name,
                idx,
                fruit_texture_dir,
                texture_aug=fruit_texture_aug,
                single_texture_per_cube=single_fruit_texture_per_cube,
                canonical_textures=canonical_fruit_textures,
                texture_layout=fruit_texture_layout,
                collage_probability=fruit_texture_collage_prob,
                generated_texture_dir=generated_texture_dir,
                fruit_print_label_profile=fruit_print_label_profile,
            )
        else:
            asset_name = "plain_cube" if cls_name == "cube" else cls_name
            obj_path = asset_dir / f"{asset_name}.obj"
            if not obj_path.exists():
                raise FileNotFoundError(f"Missing asset: {obj_path}")

            obj = bproc.loader.load_obj(str(obj_path))[0]
            obj.set_name(f"{cls_name}_{idx:03d}")
            obj.set_cp("category_id", CLASS_TO_ID[cls_name])
            obj.set_cp("supercategory", SUPER_CATEGORY)
            apply_printed_white_material(obj)
            add_small_bevel(obj)

        best_transform = None
        best_score = float("inf")
        best_ratio = 0.0
        attempts = 300 if fruit_tier else 120
        for _ in range(attempts):
            x, y, z = sample_location_for_tier(
                locations,
                fruit_tier,
                arena_booster_mode=arena_booster_mode,
                scene_profile=arena_scene_profile,
                wall_contact_ratio=wall_contact_ratio,
                corner_scene_ratio=corner_scene_ratio,
            )
            rotation = fruit_rotation_euler(fruit_tier) if fruit_tier else random_rotation_euler()
            scale = sample_object_scale(scale_min, scale_max, len(chosen), single_object_scale_min)
            if arena_scene_profile == "closeup":
                scale *= random.uniform(1.18, 1.42)
            elif arena_scene_profile == "plain_cube_hard_negative":
                scale *= random.uniform(0.92, 1.22)
            if fruit_tier:
                scale *= fruit_scale_multiplier(fruit_tier)
            obj.set_location([x, y, z])
            obj.set_rotation_euler(rotation)
            obj.set_scale([scale, scale, scale])
            overflow = projected_frame_overflow(obj)
            projected_area = projected_object_silhouette_area_pixels(obj, width, height)
            ratio = fruit_photo_to_cube_screen_ratio(obj) if fruit_tier else 1.0
            score = overflow * 100.0
            if projected_area < effective_min_projected_area:
                score += (effective_min_projected_area - projected_area) / max(effective_min_projected_area, 1.0)
            tier_min_fruit_face_pixels = effective_min_fruit_face_pixels
            if fruit_tier == "hard" and hard_min_fruit_face_pixels >= 0:
                tier_min_fruit_face_pixels = hard_min_fruit_face_pixels
            if fruit_tier:
                score += fruit_visibility_tier_distance(ratio, fruit_tier)
                estimated_fruit_pixels = projected_area * ratio
                if estimated_fruit_pixels < tier_min_fruit_face_pixels:
                    score += (tier_min_fruit_face_pixels - estimated_fruit_pixels) / max(tier_min_fruit_face_pixels, 1.0)
            if score < best_score or (score == best_score and ratio > best_ratio):
                best_score = score
                best_ratio = ratio
                best_transform = (x, y, z, rotation, scale, ratio)
            size_ok = projected_area >= effective_min_projected_area
            fruit_size_ok = (not fruit_tier) or (projected_area * ratio >= tier_min_fruit_face_pixels)
            if overflow <= 0.0 and size_ok and fruit_size_ok and (not fruit_tier or fruit_visibility_in_tier(ratio, fruit_tier)):
                break

        x, y, z, rotation, scale, visible_ratio = best_transform
        obj.set_location([x, y, z])
        obj.set_rotation_euler(rotation)
        obj.set_scale([scale, scale, scale])
        if fruit_tier:
            obj.set_cp("visibility_tier", fruit_tier)
            obj.set_cp("fruit_photo_cube_ratio", float(visible_ratio))
        objects.append(obj)
        locations.append((x, y, z))

    return objects


def bright_arena_light_color():
    """High-value near-white light with slight warm/yellow/orange variation."""
    hue = random.uniform(0.055, 0.115)
    saturation = random.uniform(0.015, 0.085)
    value = random.uniform(0.94, 1.00)
    return list(colorsys.hsv_to_rgb(hue, saturation, value))


def add_lights(lighting_mode="random"):
    if lighting_mode == "soft_overhead":
        bpy.context.scene.world.color = random.choice([
            [0.060, 0.060, 0.060],
            [0.075, 0.073, 0.068],
            [0.085, 0.088, 0.090],
        ])

        overhead = bproc.types.Light()
        overhead.set_type("AREA")
        overhead.set_location([
            random.uniform(-0.25, 0.25),
            random.uniform(-0.25, 0.25),
            random.uniform(1.65, 2.65),
        ])
        overhead.set_energy(random.uniform(360, 720))
        overhead.set_radius(random.uniform(1.6, 3.0))
        overhead.set_color(random.choice([
            [1.00, 0.96, 0.88],
            [0.96, 0.98, 1.00],
            [1.00, 1.00, 1.00],
        ]))

        for _ in range(random.randint(1, 2)):
            fill = bproc.types.Light()
            fill.set_type("AREA")
            fill.set_location([
                random.uniform(-1.2, 1.2),
                random.uniform(-1.0, 0.9),
                random.uniform(0.45, 1.35),
            ])
            fill.set_energy(random.uniform(30, 110))
            fill.set_radius(random.uniform(0.8, 2.0))
            fill.set_color(random.choice([
                [1.00, 0.95, 0.88],
                [0.92, 0.96, 1.00],
                [1.00, 1.00, 1.00],
            ]))
        return

    if lighting_mode == "bright_arena":
        bpy.context.scene.world.color = random.choice([
            [0.160, 0.158, 0.148],
            [0.185, 0.180, 0.162],
            [0.145, 0.148, 0.142],
            [0.170, 0.166, 0.154],
        ])

        corner_locations = [
            (-1.95, -1.70),
            (1.95, -1.70),
            (-1.95, 1.70),
            (1.95, 1.70),
        ]
        random.shuffle(corner_locations)
        for idx, (base_x, base_y) in enumerate(corner_locations):
            light = bproc.types.Light()
            light.set_type("AREA")
            light.set_location([
                base_x + random.uniform(-0.28, 0.28),
                base_y + random.uniform(-0.24, 0.24),
                random.uniform(2.15, 3.35),
            ])
            light.set_energy(random.uniform(520, 980))
            light.set_radius(random.uniform(2.4, 5.2))
            light.set_color(bright_arena_light_color())

        if random.random() < 0.82:
            center_reflection = bproc.types.Light()
            center_reflection.set_type("AREA")
            center_reflection.set_location([
                random.uniform(-0.25, 0.25),
                random.uniform(-0.25, 0.25),
                random.uniform(2.2, 3.1),
            ])
            center_reflection.set_energy(random.uniform(90, 260))
            center_reflection.set_radius(random.uniform(3.5, 6.5))
            center_reflection.set_color(bright_arena_light_color())

        front_fill = bproc.types.Light()
        front_fill.set_type("AREA")
        front_fill.set_location([
            random.uniform(-0.55, 0.55),
            random.uniform(-1.45, -0.75),
            random.uniform(0.65, 1.35),
        ])
        front_fill.set_energy(random.uniform(45, 150))
        front_fill.set_radius(random.uniform(1.6, 3.8))
        front_fill.set_color(bright_arena_light_color())

        if random.random() < 0.65:
            side_fill = bproc.types.Light()
            side_fill.set_type("AREA")
            side_fill.set_location([
                random.choice([-1.0, 1.0]) * random.uniform(0.9, 1.7),
                random.uniform(-0.8, 0.9),
                random.uniform(0.45, 1.15),
            ])
            side_fill.set_energy(random.uniform(35, 130))
            side_fill.set_radius(random.uniform(1.2, 2.8))
            side_fill.set_color(bright_arena_light_color())
        return

    bpy.context.scene.world.color = random.choice([
        [0.018, 0.018, 0.020],
        [0.045, 0.047, 0.050],
        [0.080, 0.078, 0.070],
        [0.025, 0.030, 0.038],
    ])

    key = bproc.types.Light()
    key.set_type(random.choice(["AREA", "AREA", "POINT", "SUN"]))
    if key.get_type() == "SUN":
        key.set_rotation_euler([
            random.uniform(math.radians(15), math.radians(75)),
            0,
            random.uniform(0, math.pi * 2),
        ])
        key.set_energy(random.uniform(0.25, 1.8))
    else:
        key.set_location([
            random.uniform(-1.2, 1.2),
            random.uniform(-1.4, 0.6),
            random.uniform(0.25, 2.2),
        ])
        key.set_energy(random.uniform(35, 520))
    if key.get_type() == "AREA":
        key.set_radius(random.uniform(0.08, 1.8))
    key.set_color(random.choice([
        [1.00, 0.82, 0.60],
        [1.00, 0.92, 0.78],
        [0.78, 0.88, 1.00],
        [1.00, 1.00, 1.00],
    ]))

    for _ in range(random.randint(0, 3)):
        fill = bproc.types.Light()
        fill.set_type(random.choice(["POINT", "AREA"]))
        fill.set_location([
            random.uniform(-1.5, 1.5),
            random.uniform(-1.0, 1.2),
            random.uniform(0.08, 1.4),
        ])
        fill.set_energy(random.uniform(4, 130))
        if fill.get_type() == "AREA":
            fill.set_radius(random.uniform(0.12, 1.0))
        fill.set_color(random.choice([
            [1.0, 0.76, 0.55],
            [0.72, 0.84, 1.0],
            [1.0, 1.0, 1.0],
        ]))

    if random.random() < 0.12:
        under = bproc.types.Light()
        under.set_type(random.choice(["POINT", "AREA"]))
        under.set_location([
            random.uniform(-0.45, 0.45),
            random.uniform(-0.35, 0.35),
            random.uniform(-0.18, 0.02),
        ])
        under.set_energy(random.uniform(35, 180))
        if under.get_type() == "AREA":
            under.set_radius(random.uniform(0.08, 0.35))
        under.set_color(random.choice([
            [1.0, 0.70, 0.45],
            [0.60, 0.78, 1.0],
            [1.0, 1.0, 1.0],
        ]))

    if random.random() < 0.18:
        for angle in np.linspace(0, math.pi * 2, 4, endpoint=False):
            ring = bproc.types.Light()
            ring.set_type("POINT")
            ring.set_location([0.75 * math.cos(angle), 0.75 * math.sin(angle), random.uniform(0.10, 0.65)])
            ring.set_energy(random.uniform(12, 75))
            ring.set_color(random.choice([[1.0, 0.92, 0.80], [0.82, 0.90, 1.0], [1.0, 1.0, 1.0]]))


def look_at(cam_location, target):
    direction = np.asarray(target, dtype=float) - np.asarray(cam_location, dtype=float)
    rotation_matrix = bproc.camera.rotation_from_forward_vec(direction)
    return bproc.math.build_transformation_mat(cam_location, rotation_matrix)


def setup_camera(width, height, robot_camera_view=False, arena_scene_profile="normal"):
    bproc.camera.set_resolution(width, height)
    if robot_camera_view:
        lens = random.uniform(24.0, 48.0)
        if arena_scene_profile == "closeup":
            lens = random.uniform(32.0, 58.0)
    else:
        lens = random.uniform(18.0, 85.0)
    bproc.camera.set_intrinsics_from_blender_params(
        lens=lens,
        image_width=width,
        image_height=height,
        lens_unit="MILLIMETERS",
        clip_start=0.01,
        clip_end=20.0,
    )

    if robot_camera_view:
        cam_location = np.array([
            random.uniform(-0.26, 0.26),
            random.uniform(-1.20, -0.42) if arena_scene_profile != "closeup" else random.uniform(-0.82, -0.32),
            random.uniform(0.10, 0.34),
        ])
        target = np.array([
            random.uniform(-0.15, 0.15),
            random.uniform(0.04, 0.36),
            random.uniform(-0.07, 0.04),
        ])
    else:
        cam_location = np.array([
            random.uniform(-0.55, 0.55),
            random.uniform(-1.65, -0.35),
            random.uniform(0.04, 0.48),
        ])
        target = np.array([
            random.uniform(-0.18, 0.18),
            random.uniform(-0.06, 0.20),
            random.uniform(-0.06, 0.10),
        ])
    bproc.camera.add_camera_pose(look_at(cam_location, target), frame=0)
    return cam_location


def make_wood_grain(height, width, strength=1.0):
    x = np.linspace(0, 1, width, dtype=np.float32)
    base = np.sin((x * random.uniform(10, 18) + random.random()) * math.pi * 2)
    fine = np.sin((x * random.uniform(34, 58) + random.random()) * math.pi * 2) * 0.35
    grain = (base + fine)[None, :]
    grain = np.repeat(grain, height, axis=0)
    grain += np.random.normal(0, 0.18, (height, width)).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (0, 0), random.uniform(1.2, 3.5))
    return grain * strength


def make_arena_background(
    width,
    height,
    floor_material="sun111_wood",
    wall_material="sun168_beige",
    scene_profile="normal",
):
    img = np.zeros((height, width, 3), dtype=np.float32)
    horizon = int(height * random.uniform(0.28, 0.42))

    # BGR approximations from the SUN material notice image:
    # SUN-111: beige/yellow plywood, SUN-168: matte paint beige.
    wall = np.array([205, 210, 218], dtype=np.float32)
    floor_base = np.array([120, 155, 195], dtype=np.float32)
    if floor_material != "sun111_wood":
        floor_base = np.array([130, 150, 180], dtype=np.float32)
    if wall_material != "sun168_beige":
        wall = np.array([198, 204, 214], dtype=np.float32)
    if scene_profile == "sun_shift":
        floor_base *= random.uniform(0.88, 1.12)
        wall *= random.uniform(0.90, 1.10)
        floor_base += np.array([
            random.uniform(-8, 8),
            random.uniform(-8, 8),
            random.uniform(-10, 10),
        ], dtype=np.float32)
        wall += random.uniform(-12, 10)

    img[:horizon] = wall
    for y in range(horizon, height):
        t = (y - horizon) / max(1, height - horizon)
        img[y, :] = floor_base * (0.82 + 0.22 * t)

    floor_slice = img[horizon:, :]
    grain = make_wood_grain(height - horizon, width, strength=random.uniform(4.0, 9.0))
    floor_slice += grain[..., None] * np.array([0.55, 0.80, 1.0], dtype=np.float32)
    img[horizon:, :] = floor_slice

    noise = np.random.normal(0, random.uniform(1.0, 3.6), (height, width)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), random.uniform(2.0, 6.0))
    img += noise[..., None]
    for x in range(random.randint(95, 170), width, random.randint(115, 180)):
        cv2.line(
            img,
            (x + random.randint(-8, 8), horizon),
            (x + random.randint(-40, 40), height),
            (floor_base * random.uniform(0.72, 0.92)).tolist(),
            1,
            cv2.LINE_AA,
        )
    cv2.line(
        img,
        (0, horizon + random.randint(-4, 4)),
        (width, horizon + random.randint(-4, 4)),
        (wall * random.uniform(0.72, 0.88)).tolist(),
        random.choice([1, 2]),
        cv2.LINE_AA,
    )
    if random.random() < 0.35:
        # Soft fence/wall panel hints above the horizon. Kept subtle so labels
        # do not overfit to a drawn room.
        for x in range(random.randint(70, 130), width, random.randint(120, 210)):
            cv2.line(img, (x, 0), (x + random.randint(-6, 6), horizon), (wall * 0.82).tolist(), 1, cv2.LINE_AA)
    return np.clip(img, 0, 255).astype(np.uint8), f"arena_{floor_material}_{wall_material}_{scene_profile}"


def arena_background_scheduled(image_id, arena_background_ratio):
    if arena_background_ratio <= 0:
        return False
    if arena_background_ratio >= 1:
        return True
    if image_id is None:
        return random.random() < arena_background_ratio
    return int((int(image_id) + 1) * arena_background_ratio) > int(int(image_id) * arena_background_ratio)


def choose_background(
    background_dir,
    width,
    height,
    arena_background_ratio=0.0,
    image_id=None,
    arena_booster_mode=False,
    arena_floor_material="sun111_wood",
    arena_wall_material="sun168_beige",
    arena_scene_profile="normal",
):
    if arena_booster_mode:
        return make_arena_background(width, height, arena_floor_material, arena_wall_material, arena_scene_profile)
    if arena_background_scheduled(image_id, arena_background_ratio):
        return make_arena_background(width, height, arena_floor_material, arena_wall_material, arena_scene_profile)

    background_dir = Path(background_dir)
    cache_key = str(background_dir.resolve())
    paths = BACKGROUND_IMAGE_CACHE.get(cache_key)
    if paths is None:
        paths = sorted(background_dir.rglob("*.jpg"))
        BACKGROUND_IMAGE_CACHE[cache_key] = paths
        print(f"background images: {len(paths)} from {background_dir}", flush=True)
    if not paths:
        raise FileNotFoundError(f"No .jpg backgrounds found in {background_dir}")
    path = random.choice(paths)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read background: {path}")

    src_h, src_w = img.shape[:2]
    scale = max(width / src_w, height / src_h)
    resized = cv2.resize(img, (int(src_w * scale) + 1, int(src_h * scale) + 1), interpolation=cv2.INTER_AREA)
    y0 = random.randint(0, max(0, resized.shape[0] - height))
    x0 = random.randint(0, max(0, resized.shape[1] - width))
    crop = resized[y0:y0 + height, x0:x0 + width].copy()

    if random.random() < 0.35:
        crop = cv2.GaussianBlur(crop, (3, 3), 0)
    if random.random() < 0.45:
        gain = random.uniform(0.82, 1.10)
        bias = random.uniform(-16, 10)
        crop = np.clip(crop.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    if random.random() < 0.35:
        texture = np.random.normal(0, random.uniform(2.0, 10.0), crop.shape[:2]).astype(np.float32)
        texture = cv2.GaussianBlur(texture, (0, 0), random.uniform(1.0, 4.0))
        crop = np.clip(crop.astype(np.float32) + texture[..., None], 0, 255).astype(np.uint8)
    return crop, path


def apply_fake_shadow(background, object_mask):
    if not np.any(object_mask) or random.random() > 0.92:
        return background

    mask = object_mask.astype(np.uint8) * 255
    shadow = np.zeros_like(mask)
    dx = random.randint(-35, 55)
    dy = random.randint(12, 85)
    transform = np.float32([[1, 0, dx], [0, 1, dy]])
    shadow = cv2.warpAffine(mask, transform, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)
    k = random.choice([31, 41, 55, 71])
    shadow = cv2.GaussianBlur(shadow, (k, k), 0).astype(np.float32) / 255.0
    strength = random.uniform(0.18, 0.58)
    out = background.astype(np.float32) * (1.0 - shadow[..., None] * strength)
    return np.clip(out, 0, 255).astype(np.uint8)


def scheduled_ratio_hit(image_id, ratio, salt=0):
    if ratio <= 0:
        return False
    if ratio >= 1:
        return True
    if image_id is None:
        return random.random() < ratio
    bucket = (int(image_id) * 1103515245 + 12345 + int(salt)) & 0x7FFFFFFF
    return (bucket % 10000) < int(ratio * 10000)


def choose_arena_scene_profile(args, image_id):
    if not args.arena_booster_mode:
        return "normal"
    if scheduled_ratio_hit(image_id, args.plain_cube_hard_negative_ratio, salt=17):
        return "plain_cube_hard_negative"
    if scheduled_ratio_hit(image_id, args.motion_blur_hard_negative_ratio, salt=31):
        return "motion_blur"
    non_hard_profiles = [(name, weight) for name, weight in ARENA_SCENE_PROFILES if name not in {"motion_blur", "plain_cube_hard_negative"}]
    total = sum(weight for _, weight in non_hard_profiles)
    r = random.random() * max(total, 1e-9)
    cumulative = 0.0
    for name, weight in non_hard_profiles:
        cumulative += weight
        if r <= cumulative:
            return name
    return "normal"


def motion_blur_image(image, kernel_size=None, angle=None):
    kernel_size = int(kernel_size or random.choice([9, 13, 17, 21]))
    if kernel_size % 2 == 0:
        kernel_size += 1
    angle = int(angle if angle is not None else random.choice([0, 45, 90, 135]))
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    if angle == 0:
        kernel[center, :] = 1.0
    elif angle == 90:
        kernel[:, center] = 1.0
    elif angle == 45:
        np.fill_diagonal(kernel, 1.0)
    else:
        np.fill_diagonal(np.fliplr(kernel), 1.0)
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel)


def apply_lens_distortion(image, instance_segmap, *extra_images, return_distortion=False, probability=0.35):
    probability = float(np.clip(probability, 0.0, 1.0))
    if random.random() > probability:
        if return_distortion:
            if extra_images:
                return (image, instance_segmap, *extra_images, None)
            return image, instance_segmap, None
        if extra_images:
            return (image, instance_segmap, *extra_images)
        return image, instance_segmap

    h, w = image.shape[:2]
    yy, xx = np.indices((h, w), dtype=np.float32)

    map_x = None
    map_y = None
    distortion = None
    for _ in range(80):
        cx = w * random.uniform(0.47, 0.53)
        cy = h * random.uniform(0.47, 0.53)
        fx = w * random.uniform(0.72, 1.15)
        fy = h * random.uniform(0.72, 1.15)

        x = (xx - cx) / fx
        y = (yy - cy) / fy
        r2 = x * x + y * y
        k1 = random.uniform(-0.30, 0.22)
        k2 = random.uniform(-0.10, 0.08)
        p1 = random.uniform(-0.006, 0.006)
        p2 = random.uniform(-0.006, 0.006)
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        x_dist = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        y_dist = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        candidate_x = x_dist * fx + cx
        candidate_y = y_dist * fy + cy

        if (
            candidate_x.min() >= 0.0
            and candidate_x.max() <= w - 1
            and candidate_y.min() >= 0.0
            and candidate_y.max() <= h - 1
        ):
            map_x = candidate_x
            map_y = candidate_y
            distortion = {
                "cx": float(cx),
                "cy": float(cy),
                "fx": float(fx),
                "fy": float(fy),
                "k1": float(k1),
                "k2": float(k2),
                "p1": float(p1),
                "p2": float(p2),
            }
            break

    if map_x is None or map_y is None:
        if return_distortion:
            if extra_images:
                return (image, instance_segmap, *extra_images, None)
            return image, instance_segmap, None
        if extra_images:
            return (image, instance_segmap, *extra_images)
        return image, instance_segmap

    warped_img = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    warped_seg = cv2.remap(instance_segmap.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_extras = [
        cv2.remap(extra, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        for extra in extra_images
    ]
    if return_distortion:
        if extra_images:
            return (warped_img, warped_seg.astype(instance_segmap.dtype), *warped_extras, distortion)
        return warped_img, warped_seg.astype(instance_segmap.dtype), distortion
    if extra_images:
        return (warped_img, warped_seg.astype(instance_segmap.dtype), *warped_extras)
    return warped_img, warped_seg.astype(instance_segmap.dtype)


def apply_lens_distortion_to_object_masks(image, instance_segmap, background, object_masks, return_distortion=False, probability=0.35):
    names = list(object_masks)
    extras = [background] + [object_masks[name].astype(np.float32) for name in names]
    warped = apply_lens_distortion(
        image,
        instance_segmap,
        *extras,
        return_distortion=return_distortion,
        probability=probability,
    )
    distortion = warped[-1] if return_distortion else None
    payload = warped[:-1] if return_distortion else warped
    warped_image, warped_segmap, warped_background = payload[:3]
    warped_masks = {}
    for name, warped_mask in zip(names, payload[3:]):
        warped_masks[name] = warped_mask >= 0.5
    if return_distortion:
        return warped_image, warped_segmap, warped_background, warped_masks, distortion
    return warped_image, warped_segmap, warped_background, warped_masks


def apply_camera_artifacts(image, profile="default"):
    if profile == "none":
        return image

    if profile in {"webcam_nocolor_mild", "webcam_nocolor_strong", "webcam_nocolor_aggressive"}:
        out = image.copy()

        strong = profile == "webcam_nocolor_strong"
        aggressive = profile == "webcam_nocolor_aggressive"

        if random.random() < (0.12 if not strong and not aggressive else 0.22 if strong else 0.35):
            out = cv2.GaussianBlur(out, (3, 3), 0)

        if random.random() < (0.18 if not strong and not aggressive else 0.32 if strong else 0.48):
            blurred = cv2.GaussianBlur(out, (0, 0), random.uniform(0.45, 0.95 if not strong and not aggressive else 1.15 if strong else 1.35))
            amount = random.uniform(0.20, 0.48 if not strong and not aggressive else 0.72 if strong else 0.95)
            out = np.clip(
                out.astype(np.float32) * (1.0 + amount) - blurred.astype(np.float32) * amount,
                0,
                255,
            ).astype(np.uint8)

        if random.random() < (0.24 if not strong and not aggressive else 0.42 if strong else 0.62):
            h, w = out.shape[:2]
            if aggressive:
                scale = random.uniform(0.42, 0.74)
            else:
                scale = random.uniform(0.72, 0.94) if not strong else random.uniform(0.58, 0.86)
            low_w = max(96, int(w * scale))
            low_h = max(96, int(h * scale))
            out = cv2.resize(
                cv2.resize(out, (low_w, low_h), interpolation=cv2.INTER_AREA),
                (w, h),
                interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC]),
            )

        if random.random() < (0.12 if not strong and not aggressive else 0.25 if strong else 0.40):
            gray_noise = np.random.normal(0, random.uniform(0.5, 2.0 if not strong and not aggressive else 3.8 if strong else 6.0), out.shape[:2])
            out = np.clip(out.astype(np.float32) + gray_noise[..., None], 0, 255).astype(np.uint8)

        if aggressive and random.random() < 0.20:
            out = cv2.bilateralFilter(out, 5, 18, 18)

        if random.random() < (0.74 if not strong and not aggressive else 0.90 if strong else 0.96):
            quality = random.randint(72, 95) if not strong and not aggressive else random.randint(58, 88) if strong else random.randint(42, 82)
            ok, encoded = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if ok:
                out = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        return out

    if profile == "realistic_webcam_boundary":
        img = image.astype(np.float32)

        if random.random() < 0.74:
            channel_gains = np.array([
                random.uniform(0.90, 1.10),
                random.uniform(0.94, 1.06),
                random.uniform(0.88, 1.12),
            ], dtype=np.float32)
            img *= channel_gains

        if random.random() < 0.82:
            gain = random.uniform(0.84, 1.18)
            bias = random.uniform(-10, 18)
            img = img * gain + bias

        if random.random() < 0.50:
            gamma = random.uniform(0.86, 1.16)
            img = 255.0 * np.power(np.clip(img, 0, 255) / 255.0, gamma)

        if random.random() < 0.30:
            noise = np.random.normal(0, random.uniform(1.0, 4.8), img.shape)
            img = img + noise

        out = np.clip(img, 0, 255).astype(np.uint8)

        if random.random() < 0.10:
            out = cv2.GaussianBlur(out, (3, 3), 0)

        if random.random() < 0.16:
            blurred = cv2.GaussianBlur(out, (0, 0), random.uniform(0.5, 0.9))
            amount = random.uniform(0.25, 0.55)
            out = np.clip(out.astype(np.float32) * (1.0 + amount) - blurred.astype(np.float32) * amount, 0, 255).astype(np.uint8)

        if random.random() < 0.28:
            h, w = out.shape[:2]
            scale = random.uniform(0.68, 0.92)
            low_w = max(96, int(w * scale))
            low_h = max(96, int(h * scale))
            out = cv2.resize(
                cv2.resize(out, (low_w, low_h), interpolation=cv2.INTER_AREA),
                (w, h),
                interpolation=cv2.INTER_LINEAR,
            )

        if random.random() < 0.82:
            quality = random.randint(62, 92)
            ok, encoded = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if ok:
                out = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        return out

    if profile == "realistic_webcam_hard":
        img = image.astype(np.float32)

        if random.random() < 0.82:
            channel_gains = np.array([
                random.uniform(0.88, 1.12),
                random.uniform(0.93, 1.07),
                random.uniform(0.86, 1.14),
            ], dtype=np.float32)
            img *= channel_gains

        if random.random() < 0.88:
            gain = random.uniform(0.78, 1.22)
            bias = random.uniform(-14, 22)
            img = img * gain + bias

        if random.random() < 0.58:
            gamma = random.uniform(0.82, 1.20)
            img = 255.0 * np.power(np.clip(img, 0, 255) / 255.0, gamma)

        if random.random() < 0.42:
            noise = np.random.normal(0, random.uniform(1.5, 7.0), img.shape)
            img = img + noise

        out = np.clip(img, 0, 255).astype(np.uint8)

        if random.random() < 0.22:
            out = cv2.GaussianBlur(out, (3, 3), 0)

        if random.random() < 0.20:
            blurred = cv2.GaussianBlur(out, (0, 0), random.uniform(0.6, 1.1))
            amount = random.uniform(0.35, 0.75)
            out = np.clip(out.astype(np.float32) * (1.0 + amount) - blurred.astype(np.float32) * amount, 0, 255).astype(np.uint8)

        if random.random() < 0.42:
            h, w = out.shape[:2]
            scale = random.uniform(0.56, 0.86)
            low_w = max(96, int(w * scale))
            low_h = max(96, int(h * scale))
            out = cv2.resize(
                cv2.resize(out, (low_w, low_h), interpolation=cv2.INTER_AREA),
                (w, h),
                interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC]),
            )

        if random.random() < 0.42:
            h, w = out.shape[:2]
            yy, xx = np.indices((h, w), dtype=np.float32)
            cx, cy = w / 2.0, h / 2.0
            radius = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
            vignette = 1.0 - np.clip(radius * random.uniform(0.05, 0.24), 0, 0.38)
            out = np.clip(out.astype(np.float32) * vignette[..., None], 0, 255).astype(np.uint8)

        if random.random() < 0.88:
            quality = random.randint(54, 88)
            ok, encoded = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if ok:
                out = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        return out

    if profile == "mild_exposure":
        img = image.astype(np.float32)

        if random.random() < 0.70:
            channel_gains = np.array([
                random.uniform(0.92, 1.08),
                random.uniform(0.95, 1.05),
                random.uniform(0.90, 1.10),
            ], dtype=np.float32)
            img *= channel_gains

        if random.random() < 0.78:
            gain = random.uniform(0.88, 1.16)
            bias = random.uniform(-8, 16)
            img = img * gain + bias

        if random.random() < 0.45:
            gamma = random.uniform(0.88, 1.12)
            img = 255.0 * np.power(np.clip(img, 0, 255) / 255.0, gamma)

        if random.random() < 0.28:
            noise = np.random.normal(0, random.uniform(1.0, 4.5), img.shape)
            img = img + noise

        out = np.clip(img, 0, 255).astype(np.uint8)

        if random.random() < 0.14:
            out = cv2.GaussianBlur(out, (3, 3), 0)

        if random.random() < 0.22:
            h, w = out.shape[:2]
            scale = random.uniform(0.70, 0.92)
            low_w = max(96, int(w * scale))
            low_h = max(96, int(h * scale))
            out = cv2.resize(
                cv2.resize(out, (low_w, low_h), interpolation=cv2.INTER_AREA),
                (w, h),
                interpolation=cv2.INTER_LINEAR,
            )

        if random.random() < 0.38:
            h, w = out.shape[:2]
            yy, xx = np.indices((h, w), dtype=np.float32)
            cx, cy = w / 2.0, h / 2.0
            radius = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
            vignette = 1.0 - np.clip(radius * random.uniform(0.03, 0.18), 0, 0.30)
            out = np.clip(out.astype(np.float32) * vignette[..., None], 0, 255).astype(np.uint8)

        if random.random() < 0.75:
            quality = random.randint(68, 95)
            ok, encoded = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if ok:
                out = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        return out

    img = image.astype(np.float32)

    if random.random() < 0.80:
        channel_gains = np.array([
            random.uniform(0.82, 1.22),
            random.uniform(0.88, 1.12),
            random.uniform(0.80, 1.28),
        ], dtype=np.float32)
        img *= channel_gains

    if random.random() < 0.85:
        gain = random.uniform(0.68, 1.22)
        bias = random.uniform(-30, 18)
        img = img * gain + bias

    if random.random() < 0.65:
        gamma = random.uniform(0.78, 1.35)
        img = 255.0 * np.power(np.clip(img, 0, 255) / 255.0, gamma)

    if random.random() < 0.50:
        noise = np.random.normal(0, random.uniform(2.0, 13.0), img.shape)
        img = img + noise

    out = np.clip(img, 0, 255).astype(np.uint8)

    if random.random() < 0.25:
        k = random.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)

    if random.random() < 0.18:
        kernel_size = random.choice([3, 5, 7])
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        if random.random() < 0.5:
            kernel[kernel_size // 2, :] = 1.0
        else:
            kernel[:, kernel_size // 2] = 1.0
        kernel /= kernel_size
        out = cv2.filter2D(out, -1, kernel)

    if random.random() < 0.55:
        h, w = out.shape[:2]
        scale = random.uniform(0.34, 0.82)
        low_w = max(64, int(w * scale))
        low_h = max(64, int(h * scale))
        interp_down = random.choice([cv2.INTER_AREA, cv2.INTER_LINEAR])
        interp_up = random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST])
        out = cv2.resize(cv2.resize(out, (low_w, low_h), interpolation=interp_down), (w, h), interpolation=interp_up)

    if random.random() < 0.65:
        h, w = out.shape[:2]
        yy, xx = np.indices((h, w), dtype=np.float32)
        cx, cy = w / 2.0, h / 2.0
        radius = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
        vignette = 1.0 - np.clip(radius * random.uniform(0.05, 0.38), 0, 0.55)
        out = np.clip(out.astype(np.float32) * vignette[..., None], 0, 255).astype(np.uint8)

    if random.random() < 0.88:
        quality = random.randint(38, 92)
        ok, encoded = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            out = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    white_frac = np.mean(np.all(out > 238, axis=2))
    if white_frac > 0.22:
        out = np.clip(out.astype(np.float32) * random.uniform(0.68, 0.86), 0, 255).astype(np.uint8)

    mean_luma = float(np.mean(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)))
    if mean_luma < 35.0:
        lift = random.uniform(35.0, 55.0) - mean_luma
        out = np.clip(out.astype(np.float32) + lift, 0, 255).astype(np.uint8)

    return out


def color_to_bgr(color):
    arr = np.asarray(color)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.shape[-1] >= 3:
        arr = arr[..., :3]
    return arr[..., ::-1].copy()


def filter_heavily_covered_labels(labels, max_covered_ratio, metas=None):
    kept = []
    kept_metas = []
    for i, label in enumerate(labels):
        _, xc, yc, bw, bh = label
        x1 = xc - bw / 2
        y1 = yc - bh / 2
        x2 = xc + bw / 2
        y2 = yc + bh / 2
        area = max(1e-9, bw * bh)
        heavily_covered = False

        for j, other in enumerate(labels):
            if i == j:
                continue
            _, oxc, oyc, obw, obh = other
            ox1 = oxc - obw / 2
            oy1 = oyc - obh / 2
            ox2 = oxc + obw / 2
            oy2 = oyc + obh / 2
            ix1 = max(x1, ox1)
            iy1 = max(y1, oy1)
            ix2 = min(x2, ox2)
            iy2 = min(y2, oy2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            covered = ((ix2 - ix1) * (iy2 - iy1)) / area
            other_area = max(1e-9, obw * obh)
            if covered >= max_covered_ratio and other_area >= area * 0.60:
                heavily_covered = True
                break

        if not heavily_covered:
            kept.append(label)
            if metas is not None:
                kept_metas.append(metas[i])
    if metas is not None:
        return kept, kept_metas
    return kept


def fruit_face_owner_from_name(name):
    prefix = "fruitface__"
    if not name.startswith(prefix):
        return None
    body = name[len(prefix):]
    if "__" not in body:
        return None
    return body.rsplit("__", 1)[0]


def instance_basename(inst):
    return str(inst.get("cf_basename") or inst.get("basename") or inst.get("name") or "")


def instance_fruit_face_owner(inst):
    owner = inst.get("cp_fruit_face_owner") or inst.get("fruit_face_owner")
    if owner:
        return str(owner)
    return fruit_face_owner_from_name(instance_basename(inst))


def fruit_face_texture_record_from_inst(inst):
    texture = inst.get("cp_fruit_face_texture") or inst.get("fruit_face_texture")
    fruit_class = inst.get("cp_fruit_face_class") or inst.get("fruit_face_class")
    declared_class = inst.get("cp_fruit_face_declared_class") or inst.get("fruit_face_declared_class")
    texture_layout = inst.get("cp_fruit_face_texture_layout") or inst.get("fruit_face_texture_layout")
    source_textures_json = inst.get("cp_fruit_face_source_textures") or inst.get("fruit_face_source_textures")
    material_name = inst.get("cp_fruit_face_material") or inst.get("fruit_face_material")
    image_name = inst.get("cp_fruit_face_image") or inst.get("fruit_face_image")
    uv_rotation_deg = inst.get("cp_fruit_face_uv_rotation_deg", inst.get("fruit_face_uv_rotation_deg"))
    uv_zoom = inst.get("cp_fruit_face_uv_zoom", inst.get("fruit_face_uv_zoom"))
    if not texture and not fruit_class and not declared_class:
        return None
    face_index = inst.get("cp_fruit_face_index", inst.get("fruit_face_index"))
    record = {
        "face_index": int(face_index) if face_index is not None else None,
        "class": str(fruit_class) if fruit_class else None,
        "texture_declared_class": str(declared_class) if declared_class else None,
        "texture": str(texture) if texture else None,
        "texture_layout": str(texture_layout) if texture_layout else None,
        "source_textures": [],
        "material_name": str(material_name) if material_name else None,
        "image_name": str(image_name) if image_name else None,
        "uv_rotation_deg": float(uv_rotation_deg) if uv_rotation_deg is not None else None,
        "uv_zoom": float(uv_zoom) if uv_zoom is not None else None,
    }
    if source_textures_json:
        try:
            source_textures = json.loads(str(source_textures_json))
            if isinstance(source_textures, list):
                record["source_textures"] = [str(path) for path in source_textures]
        except Exception:
            record["source_textures"] = []
    return record


def compute_fruit_face_visibility(instance_segmap, instance_attribute_map, fruit_projected_area_by_name):
    height, width = instance_segmap.shape[:2]
    visibility = {
        name: {
            "face_indices": [],
            "face_textures": [],
            "visible_face_pixels": 0,
            "visible_face_bbox_width": 0,
            "visible_face_bbox_height": 0,
            "visible_face_bbox": None,
            "visible_face_segments": [],
            "projected_cube_area_pixels": float(area),
            "actual_fruit_visible_ratio": 0.0,
            "fruit_visible_ratio": 0.0,
            "actual_visibility_tier": None,
        }
        for name, area in fruit_projected_area_by_name.items()
    }

    for inst in instance_attribute_map:
        owner = instance_fruit_face_owner(inst)
        if owner is None:
            continue
        if owner not in visibility:
            visibility[owner] = {
                "face_indices": [],
                "face_textures": [],
                "visible_face_pixels": 0,
                "visible_face_bbox_width": 0,
                "visible_face_bbox_height": 0,
                "visible_face_bbox": None,
                "visible_face_segments": [],
                "projected_cube_area_pixels": 0.0,
                "actual_fruit_visible_ratio": 0.0,
                "fruit_visible_ratio": 0.0,
                "actual_visibility_tier": None,
            }
        visibility[owner]["face_indices"].append(int(inst["idx"]))
        texture_record = fruit_face_texture_record_from_inst(inst)
        if texture_record is not None and texture_record not in visibility[owner]["face_textures"]:
            visibility[owner]["face_textures"].append(texture_record)

    for owner, info in visibility.items():
        face_mask = np.zeros(instance_segmap.shape[:2], dtype=bool)
        for face_idx in info["face_indices"]:
            face_mask |= instance_segmap == face_idx

        pixels = int(face_mask.sum())
        info["visible_face_pixels"] = pixels
        if pixels > 0:
            ys, xs = np.where(face_mask)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            info["visible_face_bbox_width"] = int(x2 - x1 + 1)
            info["visible_face_bbox_height"] = int(y2 - y1 + 1)
            info["visible_face_bbox"] = [
                float(x1 / max(width - 1, 1)),
                float(y1 / max(height - 1, 1)),
                float(x2 / max(width - 1, 1)),
                float(y2 / max(height - 1, 1)),
            ]
            mask_u8 = (face_mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= 16]
            contours.sort(key=cv2.contourArea, reverse=True)
            info["visible_face_segments"] = [
                segment
                for segment in (
                    contour_to_normalized_segment(cnt, width, height, 0.001)
                    for cnt in contours[:6]
                )
                if segment is not None
            ]

        denom = max(1e-9, float(info.get("projected_cube_area_pixels", 0.0)))
        fruit_visible_ratio = float(np.clip(pixels / denom, 0.0, 1.0))
        info["fruit_visible_ratio"] = fruit_visible_ratio
        info["actual_fruit_visible_ratio"] = fruit_visible_ratio
        info["actual_visibility_tier"] = actual_fruit_visibility_tier(fruit_visible_ratio)

    return visibility


def visible_object_entries(instance_segmap, instance_attribute_map, fruit_visibility=None):
    entries = []
    seen_names = set()
    for inst in instance_attribute_map:
        category_id = int(inst.get("category_id", 0))
        if category_id == 0 or category_id not in CLASS_TO_ID.values():
            continue

        cls_name = next(name for name, cid in CLASS_TO_ID.items() if cid == category_id)
        obj_name = instance_basename(inst)
        seen_names.add(obj_name)
        mask = instance_segmap == int(inst["idx"])
        fruit_info = {}

        if cls_name in FRUIT_CLASSES:
            fruit_info = (fruit_visibility or {}).get(obj_name, {})
            for face_idx in fruit_info.get("face_indices", []):
                mask |= instance_segmap == face_idx

        entries.append({
            "class_name": cls_name,
            "class_id": YOLO_CLASS_TO_ID[cls_name],
            "object_name": obj_name,
            "mask": mask,
            "fruit_info": fruit_info,
        })
    for obj_name, fruit_info in (fruit_visibility or {}).items():
        if obj_name in seen_names:
            continue
        cls_name = obj_name.split("_", 1)[0]
        if cls_name not in FRUIT_CLASSES:
            continue
        mask = np.zeros(instance_segmap.shape[:2], dtype=bool)
        for face_idx in fruit_info.get("face_indices", []):
            mask |= instance_segmap == face_idx
        if not mask.any():
            continue
        entries.append({
            "class_name": cls_name,
            "class_id": YOLO_CLASS_TO_ID[cls_name],
            "object_name": obj_name,
            "mask": mask,
            "fruit_info": fruit_info,
        })
    return entries


def fruit_visibility_passes(
    fruit_info,
    min_fruit_visible_ratio,
    min_fruit_face_pixels,
    min_fruit_face_side,
    hard_min_fruit_visible_ratio=-1.0,
    hard_min_fruit_face_pixels=-1,
):
    face_pixels = int(fruit_info.get("visible_face_pixels", 0))
    face_ratio = float(fruit_info.get("fruit_visible_ratio", fruit_info.get("actual_fruit_visible_ratio", 0.0)))
    face_w = int(fruit_info.get("visible_face_bbox_width", 0))
    face_h = int(fruit_info.get("visible_face_bbox_height", 0))
    target_tier = fruit_info.get("target_visibility_tier", fruit_info.get("visibility_tier"))
    required_ratio = min_fruit_visible_ratio
    required_pixels = min_fruit_face_pixels
    if target_tier == "hard":
        if hard_min_fruit_visible_ratio >= 0:
            required_ratio = hard_min_fruit_visible_ratio
        if hard_min_fruit_face_pixels >= 0:
            required_pixels = hard_min_fruit_face_pixels
    passes = (
        face_pixels >= required_pixels
        and face_ratio >= required_ratio
        and face_w >= min_fruit_face_side
        and face_h >= min_fruit_face_side
    )
    if not passes:
        return False
    return True


def fruit_fallback_cube_entry(entry):
    fallback = dict(entry)
    fallback["source_class_name"] = entry["class_name"]
    fallback["class_name"] = "cube"
    fallback["class_id"] = YOLO_CLASS_TO_ID["cube"]
    return fallback


def mask_visible_ratio(mask, projected_area):
    denom = max(1e-9, float(projected_area or 0.0))
    return float(np.clip(int(mask.sum()) / denom, 0.0, 1.0))


def make_label_meta(entry, mask, object_full_area, label_area=None, fruit_pass=None):
    visible_pixels = int(mask.sum())
    object_visible_ratio = mask_visible_ratio(mask, object_full_area)
    meta = {
        "class": entry["class_name"],
        "class_id": entry["class_id"],
        "object_name": entry["object_name"],
        "visible_pixels": visible_pixels,
        "label_area_pixels": int(label_area if label_area is not None else visible_pixels),
        "projected_object_area_pixels": float(object_full_area or 0.0),
        "actual_object_area_pixels": int(object_full_area or 0),
        "object_visible_ratio": object_visible_ratio,
        "visible_object_ratio": object_visible_ratio,
    }
    if entry.get("source_class_name"):
        meta["source_class"] = entry["source_class_name"]
        meta["fallback_class"] = entry["class_name"]
    if entry["class_name"] in FRUIT_CLASSES:
        fruit_info = entry["fruit_info"] or {}
        meta.update({
            "visibility_tier": fruit_info.get("actual_visibility_tier"),
            "actual_visibility_tier": fruit_info.get("actual_visibility_tier"),
            "target_visibility_tier": fruit_info.get("target_visibility_tier", fruit_info.get("visibility_tier")),
            "fruit_photo_cube_ratio": fruit_info.get("fruit_photo_cube_ratio"),
            "fruit_visibility_pass": bool(fruit_pass),
            "fruit_visible_ratio": fruit_info.get("fruit_visible_ratio", fruit_info.get("actual_fruit_visible_ratio")),
            "actual_fruit_visible_ratio": fruit_info.get("actual_fruit_visible_ratio"),
            "visible_face_pixels": fruit_info.get("visible_face_pixels"),
            "visible_face_bbox_width": fruit_info.get("visible_face_bbox_width"),
            "visible_face_bbox_height": fruit_info.get("visible_face_bbox_height"),
            "visible_face_bbox": fruit_info.get("visible_face_bbox"),
            "visible_face_segments": fruit_info.get("visible_face_segments"),
            "face_textures": fruit_info.get("face_textures"),
            "projected_cube_area_pixels": fruit_info.get("projected_cube_area_pixels"),
        })
    return meta


def build_yolo_labels(
    instance_segmap,
    instance_attribute_map,
    min_area,
    max_covered_ratio,
    projected_area_by_name=None,
    fruit_visibility=None,
    min_fruit_visible_ratio=0.10,
    min_fruit_face_pixels=300,
    min_fruit_face_side=10,
    min_object_visible_ratio=0.10,
    hard_min_fruit_visible_ratio=-1.0,
    hard_min_fruit_face_pixels=-1,
):
    labels = []
    label_meta = []
    for entry in visible_object_entries(instance_segmap, instance_attribute_map, fruit_visibility):
        cls_name = entry["class_name"]
        mask = entry["mask"]
        fruit_pass = None

        if cls_name in FRUIT_CLASSES:
            fruit_pass = fruit_visibility_passes(
                entry["fruit_info"],
                min_fruit_visible_ratio,
                min_fruit_face_pixels,
                min_fruit_face_side,
                hard_min_fruit_visible_ratio=hard_min_fruit_visible_ratio,
                hard_min_fruit_face_pixels=hard_min_fruit_face_pixels,
            )
            if not fruit_pass:
                entry = fruit_fallback_cube_entry(entry)
                cls_name = entry["class_name"]

        area = int(mask.sum())
        if area < min_area:
            continue

        projected_area = (projected_area_by_name or {}).get(entry["object_name"], area)
        if mask_visible_ratio(mask, projected_area) <= min_object_visible_ratio:
            continue

        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            continue

        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        h, w = instance_segmap.shape[:2]
        box_w = max(1, x2 - x1 + 1)
        box_h = max(1, y2 - y1 + 1)
        labels.append([
            entry["class_id"],
            (x1 + box_w / 2) / w,
            (y1 + box_h / 2) / h,
            box_w / w,
            box_h / h,
        ])
        label_meta.append(make_label_meta(entry, mask, projected_area, fruit_pass=fruit_pass))
    return filter_heavily_covered_labels(labels, max_covered_ratio, label_meta)


def contour_to_normalized_segment(contour, width, height, epsilon_ratio):
    contour = contour.reshape(-1, 2)
    if len(contour) < 3:
        return None
    perimeter = cv2.arcLength(contour.astype(np.float32), True)
    epsilon = max(0.5, perimeter * epsilon_ratio)
    approx = cv2.approxPolyDP(contour.astype(np.float32), epsilon, True).reshape(-1, 2)
    if len(approx) < 3:
        return None
    segment = []
    for x, y in approx:
        segment.extend([
            float(np.clip(x / max(width - 1, 1), 0.0, 1.0)),
            float(np.clip(y / max(height - 1, 1), 0.0, 1.0)),
        ])
    return segment if len(segment) >= 6 else None


def build_yolo_segmentation_labels(
    instance_segmap,
    instance_attribute_map,
    min_area,
    max_covered_ratio,
    projected_area_by_name=None,
    fruit_visibility=None,
    min_fruit_visible_ratio=0.10,
    min_fruit_face_pixels=300,
    min_fruit_face_side=10,
    min_segment_area=30,
    contour_mode="all",
    contour_epsilon_ratio=0.002,
    min_object_visible_ratio=0.10,
    hard_min_fruit_visible_ratio=-1.0,
    hard_min_fruit_face_pixels=-1,
):
    labels = []
    label_meta = []
    height, width = instance_segmap.shape[:2]
    for entry in visible_object_entries(instance_segmap, instance_attribute_map, fruit_visibility):
        cls_name = entry["class_name"]
        mask = entry["mask"]
        fruit_pass = None

        if cls_name in FRUIT_CLASSES:
            fruit_pass = fruit_visibility_passes(
                entry["fruit_info"],
                min_fruit_visible_ratio,
                min_fruit_face_pixels,
                min_fruit_face_side,
                hard_min_fruit_visible_ratio=hard_min_fruit_visible_ratio,
                hard_min_fruit_face_pixels=hard_min_fruit_face_pixels,
            )
            if not fruit_pass:
                entry = fruit_fallback_cube_entry(entry)
                cls_name = entry["class_name"]

        if int(mask.sum()) < min_area:
            continue

        projected_area = (projected_area_by_name or {}).get(entry["object_name"], int(mask.sum()))
        if mask_visible_ratio(mask, projected_area) <= min_object_visible_ratio:
            continue
        if mask_visible_ratio(mask, projected_area) < 1.0 - max_covered_ratio:
            continue

        mask_u8 = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_segment_area]
        if not contours:
            continue
        contours.sort(key=cv2.contourArea, reverse=True)
        if contour_mode == "largest":
            contours = contours[:1]

        for contour in contours:
            segment = contour_to_normalized_segment(contour, width, height, contour_epsilon_ratio)
            if segment is None:
                continue
            labels.append([entry["class_id"], *segment])
            label_meta.append(make_label_meta(
                entry,
                mask,
                projected_area,
                label_area=cv2.contourArea(contour),
                fruit_pass=fruit_pass,
            ))

    return labels, label_meta


def soften_mask(mask):
    alpha = (mask.astype(np.uint8) * 255)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0).astype(np.float32) / 255.0
    alpha[~mask] = 0.0
    return alpha[..., None]


def clear_scene():
    for obj in list(bpy.context.scene.objects):
        if obj.type != "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)

    for collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.lights,
        bpy.data.curves,
        bpy.data.textures,
    ):
        for item in list(collection):
            if item.users == 0:
                collection.remove(item)
    gc.collect()


def add_fruit_visibility_targets(objects, fruit_visibility):
    for obj in objects:
        obj_name = obj.blender_obj.name
        if obj_name not in fruit_visibility:
            continue
        try:
            target_tier = obj.get_cp("visibility_tier")
        except Exception:
            target_tier = None
        fruit_visibility[obj_name]["target_visibility_tier"] = target_tier
        fruit_visibility[obj_name]["visibility_tier"] = fruit_visibility[obj_name].get("actual_visibility_tier")
        try:
            fruit_visibility[obj_name]["fruit_photo_cube_ratio"] = float(obj.get_cp("fruit_photo_cube_ratio"))
        except Exception:
            fruit_visibility[obj_name]["fruit_photo_cube_ratio"] = None


def build_projected_area_by_name(object_full_masks_by_name):
    return {
        name: mask_pixel_area(mask)
        for name, mask in object_full_masks_by_name.items()
    }


def class_name_from_category_id(category_id):
    for name, cid in CLASS_TO_ID.items():
        if int(cid) == int(category_id):
            return name
    return None


def meta_v2_object_class(class_name):
    if class_name in FRUIT_CLASSES or class_name == "cube":
        return "cube_like_object"
    if class_name in {"octahedron", "dodecahedron", "icosahedron"}:
        return class_name
    return None


def normalize_xyz(values):
    arr = np.asarray(values, dtype=np.float32)
    denom = float(np.linalg.norm(arr))
    if denom <= 1e-9:
        return arr
    return arr / denom


def cube_face_name_from_normal(normal):
    n = normalize_xyz([float(normal.x), float(normal.y), float(normal.z)])
    best_name = None
    best_dot = -1.0
    for name, ref in CUBE_FACE_NORMALS:
        dot = float(np.dot(n, normalize_xyz(ref)))
        if dot > best_dot:
            best_dot = dot
            best_name = name
    return best_name if best_dot >= 0.75 else None


def parse_fruit_face_texture_records(obj):
    try:
        raw = obj.get_cp("fruit_face_textures")
    except Exception:
        raw = None
    if not raw:
        return {}
    try:
        records = json.loads(raw)
    except Exception:
        return {}
    out = {}
    for record in records if isinstance(records, list) else []:
        try:
            out[int(record.get("face_index"))] = record
        except Exception:
            continue
    return out


def apply_distortion_to_point_xy(point, distortion):
    if not distortion:
        return [float(point[0]), float(point[1])]
    cx = float(distortion["cx"])
    cy = float(distortion["cy"])
    fx = float(distortion["fx"])
    fy = float(distortion["fy"])
    k1 = float(distortion["k1"])
    k2 = float(distortion["k2"])
    p1 = float(distortion["p1"])
    p2 = float(distortion["p2"])
    src_x = (float(point[0]) - cx) / max(fx, 1e-9)
    src_y = (float(point[1]) - cy) / max(fy, 1e-9)
    x = src_x
    y = src_y
    for _ in range(10):
        r2 = x * x + y * y
        radial = max(1e-6, 1.0 + k1 * r2 + k2 * r2 * r2)
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x = (src_x - dx) / radial
        y = (src_y - dy) / radial
    return [float(x * fx + cx), float(y * fy + cy)]


def apply_distortion_to_quad_xy(quad_xy, distortion):
    return [apply_distortion_to_point_xy(point, distortion) for point in quad_xy]


def order_quad_xy_points(points):
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        return None
    rounded = np.round(pts, 3)
    if np.unique(rounded, axis=0).shape[0] != 4:
        return None
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -start, axis=0)
    if quad_area_xy(ordered) < 1.0:
        return None
    return [[float(x), float(y)] for x, y in ordered]


def quad_area_xy(points):
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        return 0.0
    return float(abs(cv2.contourArea(pts.reshape(-1, 1, 2))))


def quad_min_side_xy(points):
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        return 0.0
    sides = [
        float(np.linalg.norm(pts[(idx + 1) % 4] - pts[idx]))
        for idx in range(4)
    ]
    return min(sides) if sides else 0.0


def poly_world_center_and_normal(matrix, mesh, poly):
    local_vertices = [mesh.vertices[int(vertex_idx)].co for vertex_idx in poly.vertices]
    local_center = Vector((
        sum(float(vertex.x) for vertex in local_vertices) / max(1, len(local_vertices)),
        sum(float(vertex.y) for vertex in local_vertices) / max(1, len(local_vertices)),
        sum(float(vertex.z) for vertex in local_vertices) / max(1, len(local_vertices)),
    ))
    world_center = matrix @ local_center
    world_normal = matrix.to_3x3() @ poly.normal
    try:
        world_normal.normalize()
    except Exception:
        pass
    return world_center, world_normal


def projected_poly_quad_xy(scene, camera, matrix, mesh, poly, width, height):
    points = []
    for vertex_idx in poly.vertices:
        camera_co = world_to_camera_view(scene, camera, matrix @ mesh.vertices[vertex_idx].co)
        if camera_co.z <= 0:
            return None
        points.append([
            float(camera_co.x) * float(width - 1),
            float(1.0 - camera_co.y) * float(height - 1),
        ])
    return points if len(points) >= 3 else None


def polygon_mask_from_xy(points_xy, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)
    if not points_xy or len(points_xy) < 3:
        return mask.astype(bool)
    pts = np.asarray(points_xy, dtype=np.float32)
    pts[:, 0] = np.clip(np.rint(pts[:, 0]), -width * 2, width * 3)
    pts[:, 1] = np.clip(np.rint(pts[:, 1]), -height * 2, height * 3)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask.astype(bool)


def mask_bbox_xyxy(mask):
    if mask is None or not np.any(mask):
        return None
    ys, xs = np.where(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def mask_to_normalized_segments(mask, width, height, min_area=12, epsilon_ratio=0.002, limit=8):
    if mask is None or not np.any(mask):
        return []
    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    contours.sort(key=cv2.contourArea, reverse=True)
    segments = []
    for contour in contours[:limit]:
        segment = contour_to_normalized_segment(contour, width, height, epsilon_ratio)
        if segment is not None:
            segments.append(segment)
    return segments


def write_binary_mask(mask, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255), [int(cv2.IMWRITE_PNG_COMPRESSION), 6])


def rel_path(path, root):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def json_safe_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe_value(value.tolist())
    if isinstance(value, dict):
        return {str(k): json_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_value(v) for v in value]
    try:
        return [json_safe_value(v) for v in value]
    except Exception:
        return str(value)


def vector_to_list(values):
    return [float(v) for v in values]


def matrix_to_list(matrix):
    return [[float(value) for value in row] for row in matrix]


def custom_properties_snapshot(bpy_obj):
    props = {}
    for key in bpy_obj.keys():
        if str(key).startswith("_"):
            continue
        props[str(key)] = json_safe_value(bpy_obj[key])
    return props


def material_slots_snapshot(bpy_obj):
    slots = []
    if not hasattr(bpy_obj, "material_slots"):
        return slots
    for slot_idx, slot in enumerate(bpy_obj.material_slots):
        mat = slot.material
        if mat is None:
            slots.append({"slot_index": slot_idx, "material": None})
            continue
        images = []
        if mat.use_nodes and mat.node_tree is not None:
            for node in mat.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image is not None:
                    images.append({
                        "image_name": str(node.image.name),
                        "filepath": str(bpy.path.abspath(node.image.filepath)) if node.image.filepath else "",
                    })
        slots.append({
            "slot_index": slot_idx,
            "material": {
                "name": str(mat.name),
                "diffuse_color": vector_to_list(mat.diffuse_color),
                "use_nodes": bool(mat.use_nodes),
                "texture_images": images,
            },
        })
    return slots


def mesh_geometry_snapshot(bpy_obj, max_vertices=512, max_polygons=512):
    mesh = getattr(bpy_obj, "data", None)
    if mesh is None or not hasattr(mesh, "vertices") or not hasattr(mesh, "polygons"):
        return None

    vertex_count = len(mesh.vertices)
    polygon_count = len(mesh.polygons)
    matrix = bpy_obj.matrix_world
    geometry = {
        "mesh_name": str(mesh.name),
        "vertex_count": int(vertex_count),
        "polygon_count": int(polygon_count),
        "full_geometry_stored": bool(vertex_count <= max_vertices and polygon_count <= max_polygons),
    }
    if not geometry["full_geometry_stored"]:
        return geometry

    geometry["vertices_local"] = [vector_to_list(vertex.co) for vertex in mesh.vertices]
    geometry["vertices_world"] = [vector_to_list(matrix @ vertex.co) for vertex in mesh.vertices]
    polygons = []
    for poly in mesh.polygons:
        world_normal = matrix.to_3x3() @ poly.normal
        try:
            world_normal.normalize()
        except Exception:
            pass
        local_vertices = [mesh.vertices[int(vertex_idx)].co for vertex_idx in poly.vertices]
        local_center = Vector((
            sum(float(vertex.x) for vertex in local_vertices) / max(1, len(local_vertices)),
            sum(float(vertex.y) for vertex in local_vertices) / max(1, len(local_vertices)),
            sum(float(vertex.z) for vertex in local_vertices) / max(1, len(local_vertices)),
        ))
        world_center = matrix @ local_center
        polygons.append({
            "index": int(poly.index),
            "vertices": [int(vertex_idx) for vertex_idx in poly.vertices],
            "normal_local": vector_to_list(poly.normal),
            "normal_world": vector_to_list(world_normal),
            "center_local": vector_to_list(local_center),
            "center_world": vector_to_list(world_center),
            "area_local": float(poly.area),
            "cube_face_name": cube_face_name_from_normal(poly.normal) if len(poly.vertices) == 4 else None,
        })
    geometry["polygons"] = polygons
    return geometry


def object_scene_snapshot(obj, object_id):
    bpy_obj = obj.blender_obj
    category_id = None
    try:
        category_id = int(obj.get_cp("category_id"))
    except Exception:
        pass
    class_name = class_name_from_category_id(category_id) if category_id is not None else None
    object_class = meta_v2_object_class(class_name)
    bbox_world = []
    try:
        bbox_world = [vector_to_list(bpy_obj.matrix_world @ corner) for corner in bpy_obj.bound_box]
    except Exception:
        bbox_world = []

    return {
        "object_id": int(object_id),
        "object_name": str(bpy_obj.name),
        "category_id": category_id,
        "source_class": class_name,
        "meta_v2_object_class": object_class,
        "location_xyz": vector_to_list(bpy_obj.location),
        "rotation_euler_xyz": vector_to_list(bpy_obj.rotation_euler),
        "rotation_mode": str(bpy_obj.rotation_mode),
        "scale_xyz": vector_to_list(bpy_obj.scale),
        "dimensions_xyz": vector_to_list(bpy_obj.dimensions),
        "matrix_world": matrix_to_list(bpy_obj.matrix_world),
        "matrix_world_inverse": matrix_to_list(bpy_obj.matrix_world.inverted()),
        "bound_box_local": [vector_to_list(corner) for corner in bpy_obj.bound_box],
        "bound_box_world": bbox_world,
        "custom_properties": custom_properties_snapshot(bpy_obj),
        "materials": material_slots_snapshot(bpy_obj),
        "fruit_face_textures": parse_fruit_face_texture_records(obj) if class_name in FRUIT_CLASSES else {},
        "mesh_geometry": mesh_geometry_snapshot(bpy_obj),
    }


def camera_scene_snapshot(args):
    scene = bpy.context.scene
    camera = scene.camera
    if camera is None:
        return None
    data = camera.data
    projection_matrix = None
    try:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        projection_matrix = matrix_to_list(data.calc_matrix_camera(
            depsgraph,
            x=int(args.width),
            y=int(args.height),
            scale_x=1.0,
            scale_y=1.0,
        ))
    except Exception:
        projection_matrix = None
    return {
        "name": str(camera.name),
        "type": str(data.type),
        "location_xyz": vector_to_list(camera.location),
        "rotation_euler_xyz": vector_to_list(camera.rotation_euler),
        "matrix_world": matrix_to_list(camera.matrix_world),
        "world_to_camera_matrix": matrix_to_list(camera.matrix_world.inverted()),
        "projection_matrix": projection_matrix,
        "lens_mm": float(data.lens),
        "sensor_width_mm": float(data.sensor_width),
        "sensor_height_mm": float(data.sensor_height),
        "sensor_fit": str(data.sensor_fit),
        "shift_x": float(data.shift_x),
        "shift_y": float(data.shift_y),
        "clip_start": float(data.clip_start),
        "clip_end": float(data.clip_end),
    }


def lights_scene_snapshot():
    lights = []
    for light_obj in bpy.context.scene.objects:
        if light_obj.type != "LIGHT":
            continue
        data = light_obj.data
        lights.append({
            "name": str(light_obj.name),
            "light_type": str(data.type),
            "location_xyz": vector_to_list(light_obj.location),
            "rotation_euler_xyz": vector_to_list(light_obj.rotation_euler),
            "matrix_world": matrix_to_list(light_obj.matrix_world),
            "energy": float(getattr(data, "energy", 0.0)),
            "color": vector_to_list(getattr(data, "color", [1.0, 1.0, 1.0])),
            "shadow_soft_size": float(getattr(data, "shadow_soft_size", 0.0)),
            "size": float(getattr(data, "size", 0.0)) if hasattr(data, "size") else None,
        })
    return lights


def scene_snapshot_meta(args, objects, seed, negative, background_path, final_quality, arena_scene_profile, distortion):
    out = Path(args.output)
    object_snapshots = [
        object_scene_snapshot(obj, object_id)
        for object_id, obj in enumerate(objects)
    ]
    return {
        "schema": "scene_snapshot_v1",
        "purpose": "Regenerate downstream labels, crops, face quads, and model-specific datasets without rerendering Blender.",
        "coordinate_contract": {
            "image_geometry": "Meta V2 visible masks, visible segments, bboxes, and quad_xy are in final saved image coordinates after lens distortion.",
            "quad_xy_raw": "Pre-lens-distortion camera projection is retained for each cube face.",
            "photometric_artifacts": "Color/noise/JPEG changes do not alter geometry and are recorded as render provenance only.",
        },
        "seed": seed,
        "negative": bool(negative),
        "background": str(background_path),
        "background_relative_to_output": rel_path(background_path, out) if str(background_path).startswith(str(out)) else None,
        "arena_scene_profile": arena_scene_profile,
        "jpg_quality": int(final_quality),
        "distortion": distortion,
        "render": {
            "width": int(args.width),
            "height": int(args.height),
            "samples": int(args.samples),
            "cpu_threads": int(args.cpu_threads),
            "label_format": str(args.label_format),
            "lens_distortion_prob": float(args.lens_distortion_prob),
            "camera_artifacts_applied": True,
            "arena_booster_mode": bool(args.arena_booster_mode),
            "robot_camera_view": bool(args.robot_camera_view),
            "lighting_mode": str(args.lighting_mode),
        },
        "sampling_args": {
            "min_objects": int(args.min_objects),
            "max_objects": int(args.max_objects),
            "scale_min": float(args.scale_min),
            "scale_max": float(args.scale_max),
            "negative_ratio": float(args.negative_ratio),
            "fruit_visibility_tier": str(args.fruit_visibility_tier),
            "fruit_visibility_easy_weight": float(args.fruit_visibility_easy_weight),
            "fruit_visibility_mid_weight": float(args.fruit_visibility_mid_weight),
            "fruit_visibility_hard_weight": float(args.fruit_visibility_hard_weight),
            "fruit_class_weight_scale": float(args.fruit_class_weight_scale),
            "force_fruit_class": str(args.force_fruit_class),
            "fruit_texture_aug": str(args.fruit_texture_aug),
            "fruit_texture_layout": str(args.fruit_texture_layout),
            "fruit_texture_collage_prob": float(args.fruit_texture_collage_prob),
            "single_fruit_class_per_image": bool(args.single_fruit_class_per_image),
            "single_fruit_texture_per_cube": bool(args.single_fruit_texture_per_cube),
        },
        "label_thresholds": {
            "min_area": int(args.min_area),
            "max_covered_ratio": float(args.max_covered_ratio),
            "min_object_visible_ratio": float(args.min_object_visible_ratio),
            "min_projected_area": int(args.min_projected_area),
            "min_fruit_visible_ratio": float(args.min_fruit_visible_ratio),
            "min_fruit_face_pixels": int(args.min_fruit_face_pixels),
            "min_fruit_face_side": int(args.min_fruit_face_side),
            "hard_min_fruit_visible_ratio": float(args.hard_min_fruit_visible_ratio),
            "hard_min_fruit_face_pixels": int(args.hard_min_fruit_face_pixels),
            "min_segment_area": int(args.min_segment_area),
            "seg_contour_mode": str(args.seg_contour_mode),
            "seg_contour_epsilon_ratio": float(args.seg_contour_epsilon_ratio),
        },
        "world": {
            "color": vector_to_list(bpy.context.scene.world.color) if bpy.context.scene.world else None,
        },
        "camera": camera_scene_snapshot(args),
        "lights": lights_scene_snapshot(),
        "objects": object_snapshots,
    }


def point_visibility(mask, point, radius=3):
    if mask is None or not np.any(mask):
        return False
    h, w = mask.shape[:2]
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    if x < 0 or x >= w or y < 0 or y >= h:
        return False
    x1 = max(0, x - radius)
    x2 = min(w, x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(h, y + radius + 1)
    return bool(mask[y1:y2, x1:x2].any())


def face_quality(visible_pixels, visible_ratio, corner_visibility):
    if visible_pixels >= 1200 and visible_ratio >= 0.62 and all(corner_visibility):
        return "good"
    if visible_pixels >= 260 and visible_ratio >= 0.18:
        return "partial"
    return "bad"


def occluder_ids_from_masks(full_mask, visible_mask, instance_segmap, attr_by_idx, object_id_by_name, self_name):
    if full_mask is None or visible_mask is None:
        return []
    hidden = full_mask.astype(bool) & (~visible_mask.astype(bool))
    ids = []
    for idx in np.unique(instance_segmap[hidden]):
        idx = int(idx)
        if idx <= 0:
            continue
        inst = attr_by_idx.get(idx, {})
        owner = instance_fruit_face_owner(inst)
        name = owner or instance_basename(inst)
        if not name or name == self_name:
            continue
        object_id = object_id_by_name.get(name)
        if object_id is not None and object_id not in ids:
            ids.append(object_id)
    return ids


def build_visibility_meta_v2(
    args,
    out,
    stem,
    image_path,
    objects,
    instance_segmap,
    attr_map,
    object_full_masks_by_name,
    projected_area_by_name,
    fruit_visibility,
    distortion,
):
    height, width = instance_segmap.shape[:2]
    split = "train"
    attr_by_idx = {int(inst["idx"]): inst for inst in attr_map}
    object_instance_idx_by_name = {}
    face_instance_idx_by_owner_face = {}
    for inst in attr_map:
        idx = int(inst["idx"])
        category_id = int(inst.get("category_id", 0))
        if category_id == FRUIT_FACE_HELPER_CATEGORY_ID:
            owner = instance_fruit_face_owner(inst)
            face_index = inst.get("cp_fruit_face_index", inst.get("fruit_face_index"))
            if owner and face_index is not None:
                face_instance_idx_by_owner_face[(owner, int(face_index))] = idx
        elif category_id in CLASS_TO_ID.values():
            object_instance_idx_by_name[instance_basename(inst)] = idx

    visible_entries_by_name = {
        entry["object_name"]: entry
        for entry in visible_object_entries(instance_segmap, attr_map, fruit_visibility)
    }
    object_id_by_name = {
        obj.blender_obj.name: object_id
        for object_id, obj in enumerate(objects)
    }

    mask_root = Path(out) / "masks"
    object_visible_dir = mask_root / "objects_visible" / split
    object_full_dir = mask_root / "objects_full" / split
    face_visible_dir = mask_root / "faces_visible" / split

    meta_objects = []
    scene = bpy.context.scene
    camera = scene.camera
    for obj in objects:
        obj_name = obj.blender_obj.name
        object_id = object_id_by_name[obj_name]
        try:
            category_id = int(obj.get_cp("category_id"))
        except Exception:
            continue
        class_name = class_name_from_category_id(category_id)
        object_class = meta_v2_object_class(class_name)
        if object_class is None:
            continue

        visible_entry = visible_entries_by_name.get(obj_name)
        visible_mask = visible_entry["mask"] if visible_entry else np.zeros((height, width), dtype=bool)
        full_mask = object_full_masks_by_name.get(obj_name, np.zeros((height, width), dtype=bool)).astype(bool)
        visible_pixels = int(visible_mask.sum())
        full_pixels = int(projected_area_by_name.get(obj_name, mask_pixel_area(full_mask)) or 0)
        visible_ratio = mask_visible_ratio(visible_mask, full_pixels)
        object_visible_mask_path = object_visible_dir / f"{stem}_{object_id:03d}_{obj_name}.png"
        object_full_mask_path = object_full_dir / f"{stem}_{object_id:03d}_{obj_name}.png"
        if visible_pixels > 0:
            write_binary_mask(visible_mask, object_visible_mask_path)
        if full_pixels > 0:
            write_binary_mask(full_mask, object_full_mask_path)

        record = {
            "object_id": object_id,
            "object_name": obj_name,
            "source_class": class_name,
            "object_class": object_class,
            "a1_class_id": META_V2_OBJECT_CLASSES[object_class],
            "visible_pixels": visible_pixels,
            "full_pixels": full_pixels,
            "visible_ratio": visible_ratio,
            "visible_bbox_xyxy": mask_bbox_xyxy(visible_mask),
            "full_bbox_xyxy": mask_bbox_xyxy(full_mask),
            "visible_segments": mask_to_normalized_segments(visible_mask, width, height),
            "visible_mask": rel_path(object_visible_mask_path, out) if visible_pixels > 0 else None,
            "full_mask": rel_path(object_full_mask_path, out) if full_pixels > 0 else None,
            "occluder_object_ids": occluder_ids_from_masks(
                full_mask,
                visible_mask,
                instance_segmap,
                attr_by_idx,
                object_id_by_name,
                obj_name,
            ),
            "faces": [],
        }

        if object_class == "cube_like_object":
            texture_records = parse_fruit_face_texture_records(obj)
            base_instance_idx = object_instance_idx_by_name.get(obj_name)
            mesh = obj.blender_obj.data
            matrix = obj.blender_obj.matrix_world
            for poly in mesh.polygons:
                if len(poly.vertices) != 4:
                    continue
                world_center, world_normal = poly_world_center_and_normal(matrix, mesh, poly)
                if camera is not None and world_normal.dot(camera.matrix_world.translation - world_center) <= 1e-6:
                    continue
                face_name = cube_face_name_from_normal(poly.normal)
                if face_name is None:
                    continue
                raw_quad = projected_poly_quad_xy(scene, camera, matrix, mesh, poly, width, height)
                if raw_quad is None:
                    continue
                raw_quad_ordered = order_quad_xy_points(raw_quad)
                if raw_quad_ordered is None:
                    continue
                quad_xy = order_quad_xy_points(apply_distortion_to_quad_xy(raw_quad_ordered, distortion))
                if quad_xy is None or quad_area_xy(quad_xy) < 8.0 or quad_min_side_xy(quad_xy) < 2.0:
                    continue
                full_face_mask = polygon_mask_from_xy(quad_xy, width, height)
                fruit_face_index = FRUIT_FACE_INDEX_BY_NORMAL.get(face_name) if class_name in FRUIT_CLASSES else None
                helper_idx = face_instance_idx_by_owner_face.get((obj_name, fruit_face_index)) if fruit_face_index is not None else None
                if helper_idx is not None:
                    face_kind = "fruit_face"
                    helper_mask = instance_segmap == helper_idx
                    support_mask = cv2.dilate(
                        full_face_mask.astype(np.uint8),
                        np.ones((5, 5), dtype=np.uint8),
                        iterations=1,
                    ).astype(bool)
                    visible_face_mask = helper_mask & support_mask
                else:
                    face_kind = "plain_face"
                    visible_face_mask = full_face_mask & (instance_segmap == int(base_instance_idx)) if base_instance_idx is not None else np.zeros((height, width), dtype=bool)

                face_visible_pixels = int(visible_face_mask.sum())
                face_full_pixels = int(full_face_mask.sum())
                face_visible_ratio = float(min(1.0, face_visible_pixels / max(face_full_pixels, 1)))
                corner_visibility = [point_visibility(visible_face_mask, point) for point in quad_xy]
                quality = face_quality(face_visible_pixels, face_visible_ratio, corner_visibility)
                face_idx = len(record["faces"])
                face_mask_path = face_visible_dir / f"{stem}_{object_id:03d}_face_{face_idx:02d}_{face_kind}.png"
                if face_visible_pixels > 0:
                    write_binary_mask(visible_face_mask, face_mask_path)
                texture_record = texture_records.get(fruit_face_index, {}) if fruit_face_index is not None else {}
                fruit_class = class_name if face_kind == "fruit_face" and class_name in FRUIT_CLASSES else None
                record["faces"].append({
                    "face_id": face_idx,
                    "face_name": face_name,
                    "source_poly_index": int(poly.index),
                    "face_kind": face_kind,
                    "a2_class_id": META_V2_FACE_CLASSES[face_kind],
                    "fruit_class": fruit_class,
                    "fruit_face_index": fruit_face_index,
                    "texture": storage_texture_path(args, texture_record),
                    "texture_layout": texture_record.get("texture_layout"),
                    "source_textures": texture_record.get("source_textures", []),
                    "quad_xy_raw": [[float(x), float(y)] for x, y in raw_quad_ordered],
                    "quad_xy": [[float(x), float(y)] for x, y in quad_xy],
                    "corner_visibility": corner_visibility,
                    "visible_pixels": face_visible_pixels,
                    "full_pixels": face_full_pixels,
                    "visible_ratio": face_visible_ratio,
                    "visible_bbox_xyxy": mask_bbox_xyxy(visible_face_mask),
                    "visible_segments": mask_to_normalized_segments(visible_face_mask, width, height),
                    "visible_mask": rel_path(face_mask_path, out) if face_visible_pixels > 0 else None,
                    "occluder_object_ids": occluder_ids_from_masks(
                        full_face_mask,
                        visible_face_mask,
                        instance_segmap,
                        attr_by_idx,
                        object_id_by_name,
                        obj_name,
                    ),
                    "quality": quality,
                    "occluded": bool(face_visible_ratio < 0.92),
                })

        meta_objects.append(record)

    return {
        "schema": "visibility_meta_v2",
        "image_id": stem,
        "image_path": rel_path(image_path, out),
        "image_size": {"width": int(width), "height": int(height)},
        "distortion": distortion,
        "scene_snapshot_ref": "#/scene_snapshot",
        "object_classes": META_V2_OBJECT_CLASSES,
        "face_classes": META_V2_FACE_CLASSES,
        "objects": meta_objects,
    }


def build_fruit_projected_area_by_name(projected_area_by_name):
    return {
        name: area
        for name, area in projected_area_by_name.items()
        if name.split("_", 1)[0] in FRUIT_CLASSES
    }


def build_labels_from_visibility(args, instance_segmap, attr_map, projected_area_by_name, fruit_visibility):
    if args.label_format == "segment":
        return build_yolo_segmentation_labels(
            instance_segmap,
            attr_map,
            args.min_area,
            args.max_covered_ratio,
            projected_area_by_name=projected_area_by_name,
            fruit_visibility=fruit_visibility,
            min_fruit_visible_ratio=args.min_fruit_visible_ratio,
            min_fruit_face_pixels=args.min_fruit_face_pixels,
            min_fruit_face_side=args.min_fruit_face_side,
            hard_min_fruit_visible_ratio=args.hard_min_fruit_visible_ratio,
            hard_min_fruit_face_pixels=args.hard_min_fruit_face_pixels,
            min_segment_area=args.min_segment_area,
            contour_mode=args.seg_contour_mode,
            contour_epsilon_ratio=args.seg_contour_epsilon_ratio,
            min_object_visible_ratio=args.min_object_visible_ratio,
        )
    return build_yolo_labels(
        instance_segmap,
        attr_map,
        args.min_area,
        args.max_covered_ratio,
        projected_area_by_name=projected_area_by_name,
        fruit_visibility=fruit_visibility,
        min_fruit_visible_ratio=args.min_fruit_visible_ratio,
        min_fruit_face_pixels=args.min_fruit_face_pixels,
        min_fruit_face_side=args.min_fruit_face_side,
        hard_min_fruit_visible_ratio=args.hard_min_fruit_visible_ratio,
        hard_min_fruit_face_pixels=args.hard_min_fruit_face_pixels,
        min_object_visible_ratio=args.min_object_visible_ratio,
    )


def ideal_visibility_passes(args, objects, instance_segmap, attr_map, projected_area_by_name, fruit_visibility, labels):
    if not labels and not any(instance_basename(inst).startswith("white_distractor") for inst in attr_map):
        return False

    for entry in visible_object_entries(instance_segmap, attr_map, fruit_visibility):
        category = entry["class_name"]
        if category not in CLASS_TO_ID:
            continue
        area = projected_area_by_name.get(entry["object_name"], int(entry["mask"].sum()))
        visible_ratio = mask_visible_ratio(entry["mask"], area)
        if visible_ratio <= args.min_object_visible_ratio:
            continue
        if visible_ratio < 1.0 - args.max_covered_ratio:
            return False

    for obj in objects:
        try:
            category_id = int(obj.get_cp("category_id"))
            cls_name = next(name for name, cid in CLASS_TO_ID.items() if cid == category_id)
        except Exception:
            continue
        if cls_name not in FRUIT_CLASSES:
            continue
        obj_name = obj.blender_obj.name
        info = fruit_visibility.get(obj_name, {})
        if not fruit_visibility_passes(
            info,
            args.min_fruit_visible_ratio,
            args.min_fruit_face_pixels,
            args.min_fruit_face_side,
            hard_min_fruit_visible_ratio=args.hard_min_fruit_visible_ratio,
            hard_min_fruit_face_pixels=args.hard_min_fruit_face_pixels,
        ):
            return False
        if args.ideal_visibility_match_tier:
            target_tier = info.get("target_visibility_tier")
            actual_tier = info.get("actual_visibility_tier")
            if target_tier in FRUIT_VISIBILITY_RANGES and actual_tier != target_tier:
                return False
    return True


def generated_face_texture_attempt_dir(output, image_id, attempt):
    return Path(output) / "_generated_face_textures" / "train" / f"{image_id:06d}" / f"attempt_{attempt:03d}"


def cleanup_generated_face_textures(output, image_id):
    texture_dir = Path(output) / "_generated_face_textures" / "train" / f"{image_id:06d}"
    if not texture_dir.exists():
        return
    for path in sorted(texture_dir.rglob("*"), reverse=True):
        if path.is_file():
            try:
                path.unlink()
            except OSError:
                pass
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


def configure_render_device(args):
    if args.render_device == "auto":
        return
    if args.render_device == "cpu":
        bproc.renderer.set_render_devices(use_only_cpu=True)
        return
    desired = [item.strip().upper() for item in args.gpu_device_type.split(",") if item.strip()]
    bproc.renderer.set_render_devices(
        use_only_cpu=False,
        desired_gpu_device_type=desired or None,
    )


def sample_scene(args, negative, generated_texture_dir=None, arena_scene_profile="normal"):
    clear_scene()
    setup_camera(args.width, args.height, robot_camera_view=args.robot_camera_view, arena_scene_profile=arena_scene_profile)
    add_lights(args.lighting_mode)
    return load_targets(
        args.asset_dir,
        args.fruit_texture_dir,
        args.min_objects,
        args.max_objects,
        args.scale_min,
        args.scale_max,
        fruit_visibility_tier=args.fruit_visibility_tier,
        fruit_only=args.fruit_only,
        fruit_texture_aug=args.fruit_texture_aug,
        negative=negative,
        min_projected_area=args.min_projected_area,
        min_fruit_face_pixels=args.min_fruit_face_pixels,
        hard_min_fruit_face_pixels=args.hard_min_fruit_face_pixels,
        single_object_scale_min=args.single_object_scale_min,
        single_object_min_projected_area=args.single_object_min_projected_area,
        single_object_min_fruit_face_pixels=args.single_object_min_fruit_face_pixels,
        width=args.width,
        height=args.height,
        single_fruit_class_per_image=args.single_fruit_class_per_image,
        single_fruit_texture_per_cube=args.single_fruit_texture_per_cube,
        fruit_class_weight_scale=args.fruit_class_weight_scale,
        fruit_visibility_weights=fruit_visibility_tier_weights(args),
        canonical_fruit_textures=args.canonical_fruit_textures,
        fruit_texture_layout=args.fruit_texture_layout,
        fruit_texture_collage_prob=args.fruit_texture_collage_prob,
        generated_texture_dir=generated_texture_dir,
        fruit_print_label_profile=args.fruit_print_label_profile,
        arena_booster_mode=args.arena_booster_mode,
        arena_scene_profile=arena_scene_profile,
        wall_contact_ratio=args.wall_contact_ratio,
        corner_scene_ratio=args.corner_scene_ratio,
        apple_orange_boost=args.apple_orange_boost,
        force_fruit_class=args.force_fruit_class,
    )


def evaluate_ideal_scene(args, objects):
    instance_segmap, attr_map, object_full_masks_by_name = ideal_scene_segmentation(objects, args.width, args.height)
    projected_area_by_name = build_projected_area_by_name(object_full_masks_by_name)
    fruit_projected_area_by_name = build_fruit_projected_area_by_name(projected_area_by_name)
    fruit_visibility = compute_fruit_face_visibility(instance_segmap, attr_map, fruit_projected_area_by_name)
    add_fruit_visibility_targets(objects, fruit_visibility)
    labels, label_meta = build_labels_from_visibility(
        args,
        instance_segmap,
        attr_map,
        projected_area_by_name,
        fruit_visibility,
    )
    return {
        "instance_segmap": instance_segmap,
        "attr_map": attr_map,
        "object_full_masks_by_name": object_full_masks_by_name,
        "projected_area_by_name": projected_area_by_name,
        "fruit_visibility": fruit_visibility,
        "labels": labels,
        "label_meta": label_meta,
    }


def render_image(args, image_id, negative=False):
    if args.seed is not None:
        seed = args.seed + image_id
        random.seed(seed)
        np.random.seed(seed)
    else:
        seed = None

    arena_scene_profile = choose_arena_scene_profile(args, image_id)
    if args.arena_booster_mode and arena_scene_profile == "plain_cube_hard_negative":
        negative = False

    use_ideal_visibility = args.ideal_visibility or args.ideal_visibility_debug
    ideal_result = None
    cleanup_generated_face_textures(args.output, image_id)
    if use_ideal_visibility:
        last_result = None
        for attempt in range(1, max(1, args.ideal_visibility_max_attempts) + 1):
            cleanup_generated_face_textures(args.output, image_id)
            objects = sample_scene(
                args,
                negative,
                generated_texture_dir=generated_face_texture_attempt_dir(args.output, image_id, attempt),
                arena_scene_profile=arena_scene_profile,
            )
            candidate = evaluate_ideal_scene(args, objects)
            last_result = candidate
            if ideal_visibility_passes(
                args,
                objects,
                candidate["instance_segmap"],
                candidate["attr_map"],
                candidate["projected_area_by_name"],
                candidate["fruit_visibility"],
                candidate["labels"],
            ):
                ideal_result = candidate
                break
        if ideal_result is None:
            ideal_result = last_result
            print(
                f"ideal visibility fallback: no perfect placement after {args.ideal_visibility_max_attempts} attempts",
                flush=True,
            )

        instance_segmap = ideal_result["instance_segmap"]
        attr_map = ideal_result["attr_map"]
        object_full_masks_by_name = ideal_result["object_full_masks_by_name"]
    else:
        objects = sample_scene(
            args,
            negative,
            generated_texture_dir=generated_face_texture_attempt_dir(args.output, image_id, 0),
            arena_scene_profile=arena_scene_profile,
        )
        instance_segmap = None
        attr_map = None
        object_full_masks_by_name = {
            obj.blender_obj.name: projected_mesh_silhouette_mask(obj, args.width, args.height)
            for obj in objects
        }

    data = bproc.renderer.render()
    if not use_ideal_visibility:
        seg_data = bproc.renderer.render_segmap(
            map_by=[
                "instance",
                "class",
                "cf_basename",
                "cp_fruit_face_owner",
                "cp_fruit_face_index",
                "cp_fruit_face_class",
                "cp_fruit_face_texture",
                "cp_fruit_face_declared_class",
                "cp_fruit_face_texture_layout",
                "cp_fruit_face_source_textures",
                "cp_fruit_face_material",
                "cp_fruit_face_image",
                "cp_fruit_face_uv_rotation_deg",
                "cp_fruit_face_uv_zoom",
                "cp_supercategory",
            ],
            default_values={
                "class": 0,
                "cf_basename": "",
                "cp_fruit_face_owner": "",
                "cp_fruit_face_index": -1,
                "cp_fruit_face_class": "",
                "cp_fruit_face_texture": "",
                "cp_fruit_face_declared_class": "",
                "cp_fruit_face_texture_layout": "",
                "cp_fruit_face_source_textures": "",
                "cp_fruit_face_material": "",
                "cp_fruit_face_image": "",
                "cp_fruit_face_uv_rotation_deg": 0.0,
                "cp_fruit_face_uv_zoom": 1.0,
                "cp_supercategory": "coco_annotations",
            },
        )
        instance_segmap = seg_data["instance_segmaps"][0]
        attr_map = seg_data["instance_attribute_maps"][0]

    fg_bgr = color_to_bgr(data["colors"][0])
    if args.debug_seg_attrs:
        debug_attrs = []
        for inst in attr_map:
            debug_attrs.append(dict(inst))
        print("SEG_ATTR_DEBUG:", json.dumps(debug_attrs, ensure_ascii=False), flush=True)
    object_mask = instance_segmap > 0
    valid_instances = {int(inst["idx"]) for inst in attr_map if int(inst.get("category_id", 0)) != 0}
    for idx in valid_instances:
        object_mask |= instance_segmap == idx

    pre_distortion_fruit_projected_area = {
        name: mask_pixel_area(mask)
        for name, mask in object_full_masks_by_name.items()
        if name.split("_", 1)[0] in FRUIT_CLASSES
    }
    pre_distortion_fruit_visibility = compute_fruit_face_visibility(
        instance_segmap,
        attr_map,
        pre_distortion_fruit_projected_area,
    )
    for obj in objects:
        obj_name = obj.blender_obj.name
        if obj_name not in pre_distortion_fruit_visibility:
            continue
        try:
            target_tier = obj.get_cp("visibility_tier")
        except Exception:
            target_tier = None
        pre_distortion_fruit_visibility[obj_name]["target_visibility_tier"] = target_tier
        pre_distortion_fruit_visibility[obj_name]["visibility_tier"] = pre_distortion_fruit_visibility[obj_name].get("actual_visibility_tier")
    if args.ideal_visibility_debug:
        save_ideal_visibility_debug(
            args.output,
            image_id,
            instance_segmap,
            attr_map,
            pre_distortion_fruit_visibility,
            object_full_masks_by_name,
        )
    background, bg_path = choose_background(
        args.background_dir,
        args.width,
        args.height,
        arena_background_ratio=args.arena_background_ratio,
        image_id=image_id,
        arena_booster_mode=args.arena_booster_mode,
        arena_floor_material=args.arena_floor_material,
        arena_wall_material=args.arena_wall_material,
        arena_scene_profile=arena_scene_profile,
    )
    background = apply_fake_shadow(background, object_mask)
    alpha = soften_mask(object_mask)
    composite = (fg_bgr.astype(np.float32) * alpha + background.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    composite, instance_segmap, background, object_full_masks_by_name, distortion = apply_lens_distortion_to_object_masks(
        composite,
        instance_segmap,
        background,
        object_full_masks_by_name,
        return_distortion=True,
        probability=args.lens_distortion_prob,
    )
    projected_area_by_name = {
        name: mask_pixel_area(mask)
        for name, mask in object_full_masks_by_name.items()
    }
    fruit_projected_area_by_name = {
        name: area
        for name, area in projected_area_by_name.items()
        if name.split("_", 1)[0] in FRUIT_CLASSES
    }
    fruit_visibility = compute_fruit_face_visibility(instance_segmap, attr_map, fruit_projected_area_by_name)
    for obj in objects:
        obj_name = obj.blender_obj.name
        if obj_name not in fruit_visibility:
            continue
        try:
            target_tier = obj.get_cp("visibility_tier")
        except Exception:
            target_tier = None
        fruit_visibility[obj_name]["target_visibility_tier"] = target_tier
        fruit_visibility[obj_name]["visibility_tier"] = fruit_visibility[obj_name].get("actual_visibility_tier")
        try:
            fruit_visibility[obj_name]["fruit_photo_cube_ratio"] = float(obj.get_cp("fruit_photo_cube_ratio"))
        except Exception:
            fruit_visibility[obj_name]["fruit_photo_cube_ratio"] = None
    composite = apply_camera_artifacts(composite, profile=args.camera_artifact_profile)
    if args.arena_booster_mode and arena_scene_profile == "motion_blur":
        composite = motion_blur_image(composite)
    if args.label_format == "segment":
        labels, label_meta = build_yolo_segmentation_labels(
            instance_segmap,
            attr_map,
            args.min_area,
            args.max_covered_ratio,
            projected_area_by_name=projected_area_by_name,
            fruit_visibility=fruit_visibility,
            min_fruit_visible_ratio=args.min_fruit_visible_ratio,
            min_fruit_face_pixels=args.min_fruit_face_pixels,
            min_fruit_face_side=args.min_fruit_face_side,
            hard_min_fruit_visible_ratio=args.hard_min_fruit_visible_ratio,
            hard_min_fruit_face_pixels=args.hard_min_fruit_face_pixels,
            min_segment_area=args.min_segment_area,
            contour_mode=args.seg_contour_mode,
            contour_epsilon_ratio=args.seg_contour_epsilon_ratio,
        )
    else:
        labels, label_meta = build_yolo_labels(
            instance_segmap,
            attr_map,
            args.min_area,
            args.max_covered_ratio,
            projected_area_by_name=projected_area_by_name,
            fruit_visibility=fruit_visibility,
            min_fruit_visible_ratio=args.min_fruit_visible_ratio,
            min_fruit_face_pixels=args.min_fruit_face_pixels,
            min_fruit_face_side=args.min_fruit_face_side,
            hard_min_fruit_visible_ratio=args.hard_min_fruit_visible_ratio,
            hard_min_fruit_face_pixels=args.hard_min_fruit_face_pixels,
        )

    out = Path(args.output)
    image_dir = out / "images" / "train"
    label_dir = out / "labels" / "train"
    meta_dir = out / "_meta" / "train"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{image_id:06d}"
    final_quality = random.randint(58, 96)
    image_path = image_dir / f"{stem}.jpg"
    cv2.imwrite(str(image_path), composite, [int(cv2.IMWRITE_JPEG_QUALITY), final_quality])
    with open(label_dir / f"{stem}.txt", "w", encoding="utf-8") as f:
        for label in labels:
            f.write("%d %s\n" % (int(label[0]), " ".join(f"{float(value):.6f}" for value in label[1:])))
    visible_entries_by_name = {
        entry["object_name"]: entry
        for entry in visible_object_entries(instance_segmap, attr_map, fruit_visibility)
    }
    scene_fruit_classes = []
    for obj in objects:
        try:
            category_id = int(obj.get_cp("category_id"))
            cls_name = next(name for name, cid in CLASS_TO_ID.items() if cid == category_id)
        except Exception:
            continue
        if cls_name in FRUIT_CLASSES and cls_name not in scene_fruit_classes:
            scene_fruit_classes.append(cls_name)
    snapshot = scene_snapshot_meta(
        args,
        objects,
        seed,
        negative,
        bg_path,
        final_quality,
        arena_scene_profile,
        distortion,
    )
    meta = {
        "image_id": image_id,
        "seed": seed,
        "negative": negative,
        "label_format": args.label_format,
        "fruit_visibility_tier": args.fruit_visibility_tier,
        "background": str(bg_path),
        "jpg_quality": int(final_quality),
        "lens_distortion": distortion,
        "arena_background": str(bg_path).startswith("arena_"),
        "arena_booster_mode": bool(args.arena_booster_mode),
        "arena_scene_profile": arena_scene_profile,
        "arena_floor_material": args.arena_floor_material,
        "arena_wall_material": args.arena_wall_material,
        "robot_camera_view": bool(args.robot_camera_view),
        "single_fruit_class_per_image": args.single_fruit_class_per_image,
        "single_fruit_texture_per_cube": args.single_fruit_texture_per_cube,
        "fruit_texture_layout": args.fruit_texture_layout,
        "fruit_texture_collage_prob": args.fruit_texture_collage_prob,
        "force_fruit_class": args.force_fruit_class,
        "scene_fruit_classes": scene_fruit_classes,
        "label_objects": label_meta,
        "fruit_objects": [],
        "scene_snapshot": snapshot,
    }
    if args.meta_v2:
        meta["meta_v2"] = build_visibility_meta_v2(
            args,
            out,
            stem,
            image_path,
            objects,
            instance_segmap,
            attr_map,
            object_full_masks_by_name,
            projected_area_by_name,
            fruit_visibility,
            distortion,
        )
    for obj in objects:
        try:
            category_id = int(obj.get_cp("category_id"))
            cls_name = next(name for name, cid in CLASS_TO_ID.items() if cid == category_id)
        except Exception:
            continue
        if cls_name not in FRUIT_CLASSES:
            continue
        try:
            tier = obj.get_cp("visibility_tier")
        except Exception:
            tier = None
        try:
            ratio = float(obj.get_cp("fruit_photo_cube_ratio"))
        except Exception:
            ratio = None
        actual_visibility = fruit_visibility.get(obj.blender_obj.name, {})
        visible_entry = visible_entries_by_name.get(obj.blender_obj.name)
        visible_mask = visible_entry["mask"] if visible_entry else np.zeros(instance_segmap.shape[:2], dtype=bool)
        actual_object_area = projected_area_by_name.get(obj.blender_obj.name, 0)
        object_visible_ratio = mask_visible_ratio(
            visible_mask,
            actual_object_area,
        )
        fruit_pass = fruit_visibility_passes(
            actual_visibility,
            args.min_fruit_visible_ratio,
            args.min_fruit_face_pixels,
            args.min_fruit_face_side,
            hard_min_fruit_visible_ratio=args.hard_min_fruit_visible_ratio,
            hard_min_fruit_face_pixels=args.hard_min_fruit_face_pixels,
        )
        meta["fruit_objects"].append({
            "class": cls_name,
            "object_name": obj.blender_obj.name,
            "visibility_tier": actual_visibility.get("actual_visibility_tier"),
            "actual_visibility_tier": actual_visibility.get("actual_visibility_tier"),
            "target_visibility_tier": tier,
            "fruit_photo_cube_ratio": ratio,
            "fruit_visibility_pass": fruit_pass,
            "fruit_visible_ratio": actual_visibility.get("fruit_visible_ratio", actual_visibility.get("actual_fruit_visible_ratio")),
            "actual_fruit_visible_ratio": actual_visibility.get("actual_fruit_visible_ratio"),
            "visible_face_pixels": actual_visibility.get("visible_face_pixels"),
            "visible_face_bbox_width": actual_visibility.get("visible_face_bbox_width"),
            "visible_face_bbox_height": actual_visibility.get("visible_face_bbox_height"),
            "visible_face_bbox": actual_visibility.get("visible_face_bbox"),
            "visible_face_segments": actual_visibility.get("visible_face_segments"),
            "face_textures": storage_face_texture_records(args, actual_visibility.get("face_textures")),
            "projected_cube_area_pixels": actual_visibility.get("projected_cube_area_pixels"),
            "actual_object_area_pixels": int(actual_object_area or 0),
            "object_visible_ratio": object_visible_ratio,
            "visible_object_ratio": object_visible_ratio,
        })
    with open(meta_dir / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if not getattr(args, "keep_generated_face_textures", False):
        cleanup_generated_face_textures(args.output, image_id)

    print(f"image: {image_dir / f'{stem}.jpg'}")
    print(f"label: {label_dir / f'{stem}.txt'}")
    print(f"background: {bg_path}")
    print(f"objects: {len(labels)}")
    if seed is not None:
        print(f"seed: {seed}")


def image_ids_from_args(args):
    if args.id_file:
        with open(args.id_file, "r", encoding="utf-8") as f:
            return [int(line.strip()) for line in f if line.strip()]
    if args.start_id is None and args.end_id is None:
        return [args.image_id]
    if args.start_id is None or args.end_id is None:
        raise ValueError("--start_id and --end_id must be used together")
    if args.end_id < args.start_id:
        raise ValueError("--end_id must be >= --start_id")
    return list(range(args.start_id, args.end_id + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_dir", type=str, default="assets/generated")
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
    parser.add_argument(
        "--force_fruit_class",
        choices=["", "apple", "orange", "banana", "pineapple"],
        default="",
        help="Force all sampled fruit cubes to this class while leaving non-fruit/plain objects unchanged.",
    )
    parser.add_argument("--fruit_texture_dir", type=str, default="datasets/fruit_textures/final_fruits36065_original25_fruitseg30_10")
    parser.add_argument("--output", type=str, default="datasets/yolo_coco_composite")
    parser.add_argument("--image_id", type=int, default=0)
    parser.add_argument("--start_id", type=int, default=None)
    parser.add_argument("--end_id", type=int, default=None)
    parser.add_argument("--id_file", type=str, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--cpu_threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min_area", type=int, default=80)
    parser.add_argument("--min_objects", type=int, default=1)
    parser.add_argument("--max_objects", type=int, default=8)
    parser.add_argument("--scale_min", type=float, default=0.45)
    parser.add_argument("--scale_max", type=float, default=2.35)
    parser.add_argument("--max_covered_ratio", type=float, default=0.80)
    parser.add_argument("--min_object_visible_ratio", type=float, default=0.10)
    parser.add_argument("--min_projected_area", type=int, default=900)
    parser.add_argument("--min_fruit_visible_ratio", type=float, default=0.10)
    parser.add_argument("--min_fruit_face_pixels", type=int, default=300)
    parser.add_argument("--single_object_scale_min", type=float, default=0.82)
    parser.add_argument("--single_object_min_projected_area", type=int, default=2600)
    parser.add_argument("--single_object_min_fruit_face_pixels", type=int, default=850)
    parser.add_argument("--min_fruit_face_side", type=int, default=10)
    parser.add_argument("--label_format", choices=["bbox", "segment"], default="bbox")
    parser.add_argument("--min_segment_area", type=int, default=30)
    parser.add_argument("--seg_contour_mode", choices=["all", "largest"], default="largest")
    parser.add_argument("--seg_contour_epsilon_ratio", type=float, default=0.01)
    parser.add_argument("--negative_ratio", type=float, default=0.0)
    parser.add_argument("--fruit_visibility_tier", type=str, default="mixed", choices=["mixed", "easy", "mid", "hard"])
    parser.add_argument("--fruit_visibility_easy_weight", type=float, default=0.50)
    parser.add_argument("--fruit_visibility_mid_weight", type=float, default=0.45)
    parser.add_argument("--fruit_visibility_hard_weight", type=float, default=0.05)
    parser.add_argument("--fruit_class_weight_scale", type=float, default=1.20)
    parser.add_argument("--hard_min_fruit_visible_ratio", type=float, default=-1.0)
    parser.add_argument("--hard_min_fruit_face_pixels", type=int, default=-1)
    parser.add_argument("--lighting_mode", choices=["random", "soft_overhead", "bright_arena"], default="random")
    parser.add_argument("--fruit_only", action="store_true")
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
        "--fruit_texture_aug",
        choices=["none", "light", "strong"],
        default="strong",
        help="Augment fruit face textures before attaching them to cubes. Textures stay same-class and use per-face UV scale/rotation.",
    )
    parser.add_argument(
        "--fruit_texture_layout",
        choices=["single", "collage", "mixed", "print_label"],
        default="single",
        help="single: one fruit photo per face. collage: paste several smaller same-class fruits onto each face. mixed: sample both. print_label: printed-label texture without HSV/color jitter.",
    )
    parser.add_argument(
        "--fruit_texture_collage_prob",
        type=float,
        default=0.35,
        help="Probability of using collage texture per fruit face when --fruit_texture_layout mixed.",
    )
    parser.add_argument(
        "--fruit_print_label_profile",
        choices=[
            "legacy",
            "realistic_a4_mild",
            "realistic_a4_sparse_icon",
            "realistic_a4_hard",
            "realistic_a4_boundary",
            "realistic_a4_fruit_visible",
            "realistic_a4_label_offset",
            "realistic_a4_label_offset_strong",
            "realistic_a4_label_offset_lqprint",
            "realistic_a4_tape_edge",
            "realistic_a4_tape_edge_visible",
            "realistic_a4_orange_icon_cluster",
            "realistic_a4_orange_icon_cluster_bright",
        ],
        default="legacy",
        help="Texture profile for --fruit_texture_layout print_label. Default preserves the existing generator.",
    )
    parser.add_argument(
        "--camera_artifact_profile",
        choices=[
            "default",
            "none",
            "mild_exposure",
            "realistic_webcam_hard",
            "realistic_webcam_boundary",
            "webcam_nocolor_mild",
            "webcam_nocolor_strong",
            "webcam_nocolor_aggressive",
        ],
        default="default",
        help="Post-render camera artifact profile. Default preserves the existing generator.",
    )
    parser.add_argument(
        "--lens_distortion_prob",
        type=float,
        default=0.35,
        help="Probability of geometric lens distortion. Meta V2 quads are stored in the final distorted image coordinate system.",
    )
    parser.add_argument(
        "--render_device",
        choices=["auto", "gpu", "cpu"],
        default="auto",
        help="auto: keep BlenderProc default device selection. gpu/cpu: force render device for stability tuning.",
    )
    parser.add_argument(
        "--gpu_device_type",
        type=str,
        default="",
        help="Optional comma-separated BlenderProc GPU device preference, e.g. OPTIX or CUDA.",
    )
    parser.add_argument("--canonical_fruit_textures", action="store_true")
    parser.add_argument("--ideal_visibility", action="store_true")
    parser.add_argument("--ideal_visibility_debug", action="store_true")
    parser.add_argument("--ideal_visibility_max_attempts", type=int, default=80)
    parser.add_argument("--ideal_visibility_match_tier", action="store_true", default=True)
    parser.add_argument("--no_ideal_visibility_match_tier", dest="ideal_visibility_match_tier", action="store_false")
    parser.add_argument("--meta_v2", action="store_true", help="Write visibility_meta_v2 with object masks, cube face masks, distorted quads, occlusion, and quality.")
    parser.add_argument(
        "--keep_generated_face_textures",
        action="store_true",
        help="Keep generated collage face-texture JPEGs under _generated_face_textures for debugging. Off by default to avoid unnecessary dataset files.",
    )
    parser.add_argument("--debug_seg_attrs", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--negative", action="store_true")
    args = parser.parse_args()
    if args.arena_booster_mode:
        args.arena_background_ratio = args.arena_full_background_ratio
        args.robot_camera_view = True if not args.robot_camera_view else args.robot_camera_view
        args.lighting_mode = "bright_arena" if args.lighting_mode == "random" else args.lighting_mode
        args.fruit_class_weight_scale = max(float(args.fruit_class_weight_scale), 1.35)
        if not args.apple_orange_boost:
            args.apple_orange_boost = True

    bproc.init()
    configure_render_device(args)
    bproc.renderer.set_cpu_threads(args.cpu_threads)
    bproc.renderer.set_max_amount_of_samples(args.samples)
    bproc.renderer.set_output_format("PNG")

    out = Path(args.output)
    image_dir = out / "images" / "train"
    label_dir = out / "labels" / "train"
    meta_dir = out / "_meta" / "train"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    ids = image_ids_from_args(args)
    completed = 0
    for image_id in ids:
        stem = f"{image_id:06d}"
        resume_done = (image_dir / f"{stem}.jpg").exists() and (label_dir / f"{stem}.txt").exists()
        if args.meta_v2:
            resume_done = resume_done and (meta_dir / f"{stem}.json").exists()
        if args.resume and resume_done:
            print(f"skip: {stem}", flush=True)
            continue
        if args.negative:
            negative = True
        elif args.negative_ratio > 0 and args.seed is not None:
            negative = (args.seed + image_id) % 100 < int(args.negative_ratio * 100)
        else:
            negative = False
        render_image(args, image_id, negative=negative)
        completed += 1
        print(f"completed: {completed}/{len(ids)} image_id={image_id}", flush=True)


if __name__ == "__main__":
    main()
