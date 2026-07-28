from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path
from typing import Iterable

from isaacsim import SimulationApp


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_STAGE = PROJECT_DIR / "stadium.usd"
DEFAULT_OUTPUT_STAGE = PROJECT_DIR / "stadium_competition_28.usd"
DEFAULT_PROOF_PATH = PROJECT_DIR / "logs" / "stage_inventory" / "stadium_competition_28_inventory.json"

SET1_SOURCES = {
    "cube": "/World/cube_body",
    "octahedron": "/World/octahedron_body",
    "dodecahedron": "/World/dodecahedron_body",
    "icosahedron": "/World/icosahedron_body",
}
SET2_SOURCES = {
    "apple": "/World/apple",
    "banana": "/World/banana",
    "orange": "/World/orange",
    "pineapple": "/World/pineapple",
}
ORIGINAL_OBJECT_ROOTS = tuple(SET1_SOURCES.values()) + tuple(SET2_SOURCES.values())
COMPETITION_ROOT = "/World/CompetitionObjects"
START_XY = (1.8, -1.8)
STORAGE_XY = (-1.8, -1.8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a derived Isaac Sim stadium stage with the AGENTS competition "
            "inventory: set1 4x4 shape objects plus set2 4x3 fruit cubes."
        )
    )
    parser.add_argument("--source-stage", default=str(DEFAULT_SOURCE_STAGE))
    parser.add_argument("--output-stage", default=str(DEFAULT_OUTPUT_STAGE))
    parser.add_argument("--proof-path", default=str(DEFAULT_PROOF_PATH))
    parser.add_argument(
        "--layout",
        choices=("legacy", "grid42"),
        default="legacy",
        help=(
            "legacy: original fixed 7x4 grid. grid42: official 2026-07-06 rule - "
            "28 objects on a random subset of the 42 candidate grid points "
            "(official cm coords x in {50..350}, y in {100..350}, 50 cm pitch)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=14,
        help="Random seed for grid42 point selection and object assignment.",
    )
    parser.add_argument(
        "--target-size-m",
        type=float,
        default=0.0,
        help=(
            "If > 0, uniformly scale each placed object so its largest bbox "
            "dimension equals this size (rule: objects fit in 8x8x8 cm -> 0.08). "
            "0 keeps source sizes."
        ),
    )
    parser.add_argument(
        "--random-yaw",
        action="store_true",
        help=(
            "Give every placed object a seeded random yaw about its center "
            "(hand placement has arbitrary orientation; matters for the "
            "fruit-face rule where only 3 of 6 cube faces carry images)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output USD if it already exists.",
    )
    return parser.parse_args()


def official_cm_to_map_m(x_cm: float, y_cm: float) -> tuple[float, float]:
    """Official arena frame (origin bottom-left/storage corner, cm) -> sim map frame (m).

    The sim map frame is centered on the 4 m arena: map = official/100 - 2.0.
    Storage (0..40, 0..40) -> around (-1.8, -1.8); start (360..400, 0..40) ->
    around (1.8, -1.8), matching STORAGE_XY/START_XY.
    """
    return (x_cm / 100.0 - 2.0, y_cm / 100.0 - 2.0)


def grid42_candidate_points() -> list[tuple[float, float]]:
    """The 42 official candidate grid points in sim map frame (meters)."""
    xs_cm = [50, 100, 150, 200, 250, 300, 350]
    ys_cm = [100, 150, 200, 250, 300, 350]
    return [official_cm_to_map_m(x, y) for y in ys_cm for x in xs_cm]


def distance_xy(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def world_translation(stage, prim_path: str) -> tuple[float, float, float]:
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"missing source prim: {prim_path}")
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return (float(matrix[3][0]), float(matrix[3][1]), float(matrix[3][2]))


def set_translate(stage, prim_path: str, xyz: tuple[float, float, float]) -> None:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"cannot set transform for missing prim: {prim_path}")
    xform = UsdGeom.Xformable(prim)
    translate_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if translate_op is None:
        translate_op = xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
    translate_op.Set(Gf.Vec3d(*xyz))


def set_visibility(stage, prim_path: str, visibility: str) -> None:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"cannot set visibility for missing prim: {prim_path}")
    UsdGeom.Imageable(prim).GetVisibilityAttr().Set(visibility)


def create_object_plan(layout: str = "legacy", seed: int = 14) -> list[dict]:
    if layout == "grid42":
        rng = random.Random(seed)
        candidates = grid42_candidate_points()
        positions = rng.sample(candidates, 28)
    else:
        xs = [-1.55, -1.05, -0.55, -0.05, 0.45, 0.95, 1.45]
        ys = [1.35, 0.85, 0.35, -0.15]
        positions = [(x, y) for y in ys for x in xs]

    entries: list[dict] = []
    for repeat in range(4):
        for class_name, source_path in SET1_SOURCES.items():
            entries.append(
                {
                    "set": "set1_shape",
                    "class_name": class_name,
                    "instance": repeat + 1,
                    "source_path": source_path,
                }
            )
    for repeat in range(3):
        for class_name, source_path in SET2_SOURCES.items():
            entries.append(
                {
                    "set": "set2_fruit",
                    "class_name": class_name,
                    "instance": repeat + 1,
                    "source_path": source_path,
                }
            )

    if len(entries) != len(positions):
        raise RuntimeError(f"layout bug: entries={len(entries)} positions={len(positions)}")
    for index, (entry, xy) in enumerate(zip(entries, positions), start=1):
        entry["index"] = index
        entry["xy"] = xy
        entry["prim_path"] = (
            f"{COMPETITION_ROOT}/{entry['set']}_{entry['class_name']}_{entry['instance']:02d}"
        )
    return entries


def count_by(items: Iterable[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(item[key])
        counts[value] = counts.get(value, 0) + 1
    return counts


def min_pair_distance(items: list[dict]) -> float:
    distances = []
    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            distances.append(distance_xy(tuple(left["xy"]), tuple(right["xy"])))
    return min(distances) if distances else 0.0


def set_uniform_scale(stage, prim_path: str, factor: float) -> None:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"cannot scale missing prim: {prim_path}")
    xform = UsdGeom.Xformable(prim)
    scale_op = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_op = op
            break
    if scale_op is None:
        scale_op = xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble)
    current = scale_op.Get() or Gf.Vec3d(1.0, 1.0, 1.0)
    scale_op.Set(Gf.Vec3d(*(float(c) * factor for c in current)))


def measure_max_dimension(stage, prim_path: str) -> float:
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    prim = stage.GetPrimAtPath(prim_path)
    box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    size = box.GetMax() - box.GetMin()
    return max(float(size[0]), float(size[1]), float(size[2]))


def measure_local_center(stage, prim_path: str) -> tuple[float, float]:
    """Geometry bbox center in the prim's LOCAL frame (XY).

    Several source assets (notably the fruit-image cubes) have their pivot
    far from the geometric center (banana ~0.9 m unscaled). The offset is
    baked into each placed copy as a trailing xformOp so the saved USD has
    prim origin == geometry center; nothing needs correcting at runtime.
    """
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    prim = stage.GetPrimAtPath(prim_path)
    box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    center = (box.GetMax() + box.GetMin()) * 0.5
    world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    local = world.GetInverse().Transform(center)
    return float(local[0]), float(local[1])


def bake_pivot_correction(stage, prim_path: str, local_offset: tuple[float, float]) -> None:
    """Append a local translate that shifts geometry so origin == center."""
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    op = xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble, opSuffix="pivotFix")
    op.Set(Gf.Vec3d(-local_offset[0], -local_offset[1], 0.0))


def set_yaw(stage, prim_path: str, yaw_deg: float) -> None:
    """Rotate the object about its (pivot-corrected) center.

    The op is ordered between the placement translate and the scale/pivotFix
    ops, so it composes as pivotFix -> scale -> rotate -> translate: the
    rotation happens about the geometric center, then the object is placed.
    """
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    rotate_op = xform.AddRotateZOp(precision=UsdGeom.XformOp.PrecisionDouble, opSuffix="yaw")
    rotate_op.Set(float(yaw_deg))
    ops = xform.GetOrderedXformOps()
    translates = [op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
                  and "pivotFix" not in op.GetOpName()]
    rest = [op for op in ops if op not in translates and op != rotate_op]
    xform.SetXformOpOrder(translates + [rotate_op] + rest)


FRUIT_BLANK_RULE = "2026-07-18: fruit images on front/top/back, blank on left/right/bottom"


def apply_fruit_face_rule(stage, prim_path: str) -> list[str]:
    """Rebind a fruit cube's label faces per the 2026-07-18 rule.

    Of the vertical ring {front, top, back, bottom}, 3 faces carry the fruit
    image and the blank ring face is placed at the bottom -> fruit on
    front/back/top (+-Y local, +Z), blank on left/right/bottom (+-X, -Z).
    Source assets have the label on all 6 faces, so the 3 blank faces are
    rebound to the asset's own mat_white.
    """
    from pxr import Usd, UsdGeom, UsdShade

    prim = stage.GetPrimAtPath(prim_path)
    white = None
    faces = []
    for child in Usd.PrimRange(prim):
        if child.GetName() == "mat_white":
            white = UsdShade.Material(child)
        elif child.GetName().startswith("label_face"):
            faces.append(child)
    if white is None or len(faces) != 6:
        raise RuntimeError(
            f"fruit prim {prim_path}: expected 6 label faces + mat_white "
            f"(got {len(faces)} faces, white={white is not None})")
    blanked = []
    for face in faces:
        t = UsdGeom.Xformable(face).GetLocalTransformation().ExtractTranslation()
        side = abs(t[0]) > max(abs(t[1]), abs(t[2]))
        bottom = abs(t[2]) > max(abs(t[0]), abs(t[1])) and t[2] < 0
        if side or bottom:
            binding = UsdShade.MaterialBindingAPI.Apply(face)
            binding.UnbindDirectBinding()
            binding.Bind(white)
            blanked.append(face.GetName())
    if len(blanked) != 3:
        raise RuntimeError(f"fruit prim {prim_path}: blanked {blanked}, expected 3 faces")
    return sorted(blanked)


def create_layout(source_stage: Path, output_stage: Path, proof_path: Path, force: bool, layout: str = "legacy", seed: int = 14, target_size_m: float = 0.0, random_yaw: bool = False) -> dict:
    from pxr import Sdf, UsdGeom

    if not source_stage.is_file():
        raise FileNotFoundError(source_stage)
    if output_stage.exists() and not force:
        raise FileExistsError(f"{output_stage} exists; pass --force to overwrite")
    output_stage.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_stage, output_stage)

    import omni.usd

    context = omni.usd.get_context()
    if not context.open_stage(str(output_stage)):
        raise RuntimeError(f"failed to open output stage: {output_stage}")
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("no stage is open")
    root_layer = stage.GetRootLayer()
    stage.SetEditTarget(root_layer)

    UsdGeom.Xform.Define(stage, COMPETITION_ROOT)

    plan = create_object_plan(layout=layout, seed=seed)
    yaw_rng = random.Random(seed + 1000)  # independent stream: yaw does not disturb point selection
    source_heights = {path: world_translation(stage, path)[2] for path in ORIGINAL_OBJECT_ROOTS}
    # Per-class scale factors so every object fits the rule box (e.g. 8 cm),
    # and pivot->geometric-center offsets so the VISIBLE object sits on the
    # grid point (fruit cubes have off-center pivots).
    scale_factors: dict[str, float] = {}
    local_centers: dict[str, tuple[float, float]] = {}
    for source_path in ORIGINAL_OBJECT_ROOTS:
        local_centers[source_path] = measure_local_center(stage, source_path)
        if target_size_m > 0.0:
            max_dim = measure_max_dimension(stage, source_path)
            scale_factors[source_path] = (target_size_m / max_dim) if max_dim > 0 else 1.0

    for entry in plan:
        src = Sdf.Path(entry["source_path"])
        dst = Sdf.Path(entry["prim_path"])
        if not Sdf.CopySpec(root_layer, src, root_layer, dst):
            raise RuntimeError(f"failed to copy {src} -> {dst}")
        factor = scale_factors.get(entry["source_path"], 1.0)
        entry["scale_factor"] = round(factor, 4)
        if abs(factor - 1.0) > 1e-3:
            set_uniform_scale(stage, entry["prim_path"], factor)
        z = source_heights[entry["source_path"]]
        # Keep the object resting on the floor after scaling (source z is the
        # rest height of the unscaled model above the floor top at 0.05).
        floor_top = 0.05
        z = floor_top + (z - floor_top) * factor
        x, y = entry["xy"]
        set_translate(stage, entry["prim_path"], (float(x), float(y), float(z)))
        # Bake the pivot so the saved USD has prim origin == geometry center
        # (one-time correction; no runtime bbox math needed anywhere).
        local_cx, local_cy = local_centers.get(entry["source_path"], (0.0, 0.0))
        entry["pivot_fix_local"] = [round(local_cx, 4), round(local_cy, 4)]
        if abs(local_cx) > 1e-4 or abs(local_cy) > 1e-4:
            bake_pivot_correction(stage, entry["prim_path"], (local_cx, local_cy))
        if random_yaw:
            yaw_deg = yaw_rng.uniform(0.0, 360.0)
            entry["yaw_deg"] = round(yaw_deg, 1)
            set_yaw(stage, entry["prim_path"], yaw_deg)
        if entry["set"] == "set2_fruit":
            entry["blank_faces"] = apply_fruit_face_rule(stage, entry["prim_path"])
        set_visibility(stage, entry["prim_path"], "inherited")

    parked_originals = []
    for index, root_path in enumerate(ORIGINAL_OBJECT_ROOTS):
        parking_xyz = (-3.0 + 0.25 * index, 3.0, -2.0)
        set_translate(stage, root_path, parking_xyz)
        set_visibility(stage, root_path, "invisible")
        parked_originals.append({"prim_path": root_path, "parking_xyz": parking_xyz})

    if not root_layer.Save():
        raise RuntimeError(f"failed to save {output_stage}")

    proof_items = []
    for entry in plan:
        prim = stage.GetPrimAtPath(entry["prim_path"])
        if not prim.IsValid():
            raise RuntimeError(f"missing generated prim after save: {entry['prim_path']}")
        x, y = entry["xy"]
        z = source_heights[entry["source_path"]]
        proof_items.append(
            {
                **entry,
                "xyz": [float(x), float(y), float(z)],
                "active": bool(prim.IsActive()),
                "visibility": str(UsdGeom.Imageable(prim).GetVisibilityAttr().Get() or "inherited"),
                "distance_from_start_m": distance_xy((x, y), START_XY),
                "distance_from_storage_m": distance_xy((x, y), STORAGE_XY),
            }
        )

    proof = {
        "schema": "mk1_competition_stadium_layout/v1",
        "layout": layout,
        "seed": seed if layout == "grid42" else None,
        "target_size_m": target_size_m if target_size_m > 0 else None,
        "random_yaw": bool(random_yaw),
        "fruit_face_rule": FRUIT_BLANK_RULE,
        "source_stage": str(source_stage),
        "output_stage": str(output_stage),
        "competition_root": COMPETITION_ROOT,
        "total_object_count": len(proof_items),
        "set_counts": count_by(proof_items, "set"),
        "class_counts": count_by(proof_items, "class_name"),
        "min_pair_distance_m": min_pair_distance(proof_items),
        "min_distance_from_start_m": min(item["distance_from_start_m"] for item in proof_items),
        "min_distance_from_storage_m": min(item["distance_from_storage_m"] for item in proof_items),
        "original_roots_parked": parked_originals,
        "objects": proof_items,
    }
    proof_path.parent.mkdir(parents=True, exist_ok=True)
    proof_path.write_text(json.dumps(proof, indent=2, sort_keys=True), encoding="utf-8")
    return proof


def main() -> int:
    args = parse_args()
    source_stage = Path(args.source_stage).expanduser().resolve()
    output_stage = Path(args.output_stage).expanduser().resolve()
    proof_path = Path(args.proof_path).expanduser().resolve()

    app = SimulationApp({"headless": True})
    try:
        proof = create_layout(
            source_stage,
            output_stage,
            proof_path,
            bool(args.force),
            layout=args.layout,
            seed=args.seed,
            target_size_m=float(args.target_size_m),
            random_yaw=bool(args.random_yaw),
        )
        print(json.dumps(proof, indent=2, sort_keys=True), flush=True)
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
