"""Generate the sim-only 4-mecanum-wheel robot URDF (robot_mk3_mecanum_sim).

Builds on the mobile_manipulator_collision robot (same chassis, gripper,
sensors, +Y-forward authoring convention) but replaces the diff-drive
wheels + ball caster with four mecanum wheels. Each wheel follows the
NVIDIA Kaya pattern: passive sphere rollers on free-spinning joints around
the hub, except the roller axes are tilted 45 deg for mecanum behaviour
(Kaya uses 90 deg omni rollers).

Wheel order and body conventions match src/robot_hardware/robot_hardware/
mecanum_control.py: (front_left, front_right, rear_left, rear_right),
+x forward, +y left, +yaw CCW in the ROS body frame. In this URDF's
authoring frame the robot faces +Y (like mobile_manipulator_collision), so
ROS +x forward -> URDF +Y and ROS +y left -> URDF -X.

Roller angle convention (X-configuration, viewed from above): the roller
axes of FL/RR tilt one way and FR/RL the other, so that the wheel force
directions form an X. The gamma sign per wheel is chosen so the repo's
MecanumKinematics IK drives the robot in the commanded direction; it is
validated in sim by tests/sim smoke runs (strafe left => FL,RR backward,
FR,RL forward).

Usage:
    python3 sim/isaacsim/scripts/generate_mecanum_robot_urdf.py \
        [--output src/robot_description/urdf/robot_mk3_mecanum_sim.urdf]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "src" / "robot_description" / "urdf" / "robot_mk3_mecanum_sim.urdf"

# Drive geometry: mirrors mecanum_bridge_node defaults / real.yaml.
# Two wheel sets were purchased for the real MK3 swap: 68 mm and 80 mm
# diameter. The robot competed on the 80 mm set (see hardware/docs/bom.md);
# the 68 mm variant stays generatable for the alternative build.
SUPPORTED_WHEEL_DIAMETERS_MM = (68, 80)
DEFAULT_WHEEL_DIAMETER_MM = 80
HALF_LENGTH_M = 0.150           # wheel center forward offset from body center
HALF_WIDTH_M = 0.125            # wheel center lateral offset from body center

# Mecanum roller layout (Kaya-style passive sphere rollers); roller size
# scales with the wheel radius.
ROLLER_SPHERE_RADIUS_RATIO = 0.012 / 0.034
ROLLER_COUNT = 16               # single ring; self-collision stays disabled
ROLLER_GAMMA_DEG = 45.0
# Rollers are deliberately a bit heavy for solver stability (tiny bodies with
# large contact forces jitter otherwise).
ROLLER_MASS_KG = 0.02
ROLLER_INERTIA = 1.2e-6
# Without an explicit URDF velocity limit the importer clamps
# physxJoint:maxJointVelocity to ~0.25 rad/s, which locks the rollers.
ROLLER_VELOCITY_LIMIT_RAD_S = 200.0

HUB_RADIUS_M = 0.018
HUB_LENGTH_M = 0.030
HUB_MASS_KG = 0.12
HUB_INERTIA = 2.0e-5

# Drive motor: JGB37-520 encoder motor, 12 V, 333 rpm (purchased for the
# real swap). 333 rpm ~= 34.9 rad/s no-load; stall torque ~1 Nm class.
# Per 2026-07-07: the swap should allow ~1.5x faster wheel speeds than the
# current motors (bridge caps stay at real.yaml values; raise via --speed-scale).
WHEEL_EFFORT_LIMIT_NM = 1.0
WHEEL_VELOCITY_LIMIT_RAD_S = 35.0

# Chassis (2026-07-07 user measurements): body front-back length is 5 cm
# shorter than MK3 (0.30 -> 0.25 m) and the body plate WIDTH is 19 cm (MK3
# revision). Gripper arms are thin plates: ~1 cm thick, ~10 cm front-back.
# A backstop wall at the body's front edge keeps gripped objects inside the
# gripper region (they cannot slide into the body).
BODY_FRONT_BACK_M = 0.25
BODY_WIDTH_M = 0.19
FINGER_THICKNESS_M = 0.01
FINGER_LENGTH_M = 0.10
FINGER_HEIGHT_M = 0.09

# (name, x_sign, y_sign, gamma_sign). ROS body frame -> URDF frame:
# ROS +x (forward) -> URDF +Y ; ROS +y (left) -> URDF -X.
WHEELS = (
    ("front_left", -1.0, +1.0, +1.0),
    ("front_right", +1.0, +1.0, -1.0),
    ("rear_left", -1.0, -1.0, -1.0),
    ("rear_right", +1.0, -1.0, +1.0),
)


def _fmt(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def roller_origin_and_axis(theta: float, gamma_sign: float, axis_offset_m: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Roller joint origin and spin axis in the wheel link frame.

    The wheel spins about local +X; the rim lies in the local YZ plane.
    Roller center: d*(0, sin(theta), cos(theta)). Roller spin axis is the rim
    tangent rotated by gamma about the outward radial direction:
        a = t*cos(gamma) + (u x t)*sin(gamma),  u x t = (-1, 0, 0).
    """
    d = axis_offset_m
    gamma = math.radians(ROLLER_GAMMA_DEG) * gamma_sign
    origin = (0.0, d * math.sin(theta), d * math.cos(theta))
    axis = (
        -math.sin(gamma),
        math.cos(gamma) * math.cos(theta),
        -math.cos(gamma) * math.sin(theta),
    )
    return origin, axis


def wheel_assembly(name: str, urdf_x: float, urdf_y: float, gamma_sign: float, wheel_radius_m: float) -> list[str]:
    roller_radius_m = ROLLER_SPHERE_RADIUS_RATIO * wheel_radius_m
    axis_offset_m = wheel_radius_m - roller_radius_m
    lines: list[str] = []
    wheel_link = f"{name}_wheel"
    lines.append(f'  <link name="{wheel_link}">')
    lines.append("    <inertial>")
    lines.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
    lines.append(f'      <mass value="{_fmt(HUB_MASS_KG)}"/>')
    lines.append(
        f'      <inertia ixx="{_fmt(HUB_INERTIA)}" ixy="0" ixz="0" '
        f'iyy="{_fmt(HUB_INERTIA)}" iyz="0" izz="{_fmt(HUB_INERTIA)}"/>'
    )
    lines.append("    </inertial>")
    lines.append("    <visual>")
    lines.append('      <origin xyz="0 0 0" rpy="0 1.5708 0"/>')
    lines.append(
        f"      <geometry><cylinder radius=\"{_fmt(HUB_RADIUS_M)}\" length=\"{_fmt(HUB_LENGTH_M)}\"/></geometry>"
    )
    lines.append('      <material name="rubber"/>')
    lines.append("    </visual>")
    # Hub has no collision: only the rollers touch the ground.
    lines.append("  </link>")
    lines.append(f'  <joint name="{name}_wheel_spin" type="continuous">')
    lines.append(f'    <origin xyz="{_fmt(urdf_x)} {_fmt(urdf_y)} {_fmt(wheel_radius_m)}" rpy="0 0 0"/>')
    lines.append('    <parent link="base_link"/>')
    lines.append(f'    <child link="{wheel_link}"/>')
    lines.append('    <axis xyz="1 0 0"/>')
    # JGB37-520 (12 V, 333 rpm) motor envelope.
    lines.append(
        f'    <limit effort="{_fmt(WHEEL_EFFORT_LIMIT_NM)}" velocity="{_fmt(WHEEL_VELOCITY_LIMIT_RAD_S)}"/>'
    )
    # The Isaac URDF importer maps this damping onto the joint drive, where it
    # acts as the velocity-drive gain (torque = damping * velocity error).
    lines.append('    <dynamics damping="20.0" friction="0.0"/>')
    lines.append("  </joint>")

    for index in range(ROLLER_COUNT):
        theta = 2.0 * math.pi * index / ROLLER_COUNT
        origin, axis = roller_origin_and_axis(theta, gamma_sign, axis_offset_m)
        roller_link = f"{name}_roller_{index:02d}"
        lines.append(f'  <link name="{roller_link}">')
        lines.append("    <inertial>")
        lines.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
        lines.append(f'      <mass value="{_fmt(ROLLER_MASS_KG)}"/>')
        lines.append(
            f'      <inertia ixx="{_fmt(ROLLER_INERTIA)}" ixy="0" ixz="0" '
            f'iyy="{_fmt(ROLLER_INERTIA)}" iyz="0" izz="{_fmt(ROLLER_INERTIA)}"/>'
        )
        lines.append("    </inertial>")
        lines.append("    <visual>")
        lines.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
        lines.append(f"      <geometry><sphere radius=\"{_fmt(roller_radius_m)}\"/></geometry>")
        lines.append('      <material name="rubber"/>')
        lines.append("    </visual>")
        lines.append("    <collision>")
        lines.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
        lines.append(f"      <geometry><sphere radius=\"{_fmt(roller_radius_m)}\"/></geometry>")
        lines.append("    </collision>")
        lines.append("  </link>")
        lines.append(f'  <joint name="{roller_link}_joint" type="continuous">')
        lines.append(
            f'    <origin xyz="{_fmt(origin[0])} {_fmt(origin[1])} {_fmt(origin[2])}" rpy="0 0 0"/>'
        )
        lines.append(f'    <parent link="{wheel_link}"/>')
        lines.append(f'    <child link="{roller_link}"/>')
        lines.append(f'    <axis xyz="{_fmt(axis[0])} {_fmt(axis[1])} {_fmt(axis[2])}"/>')
        lines.append(f'    <limit effort="1000" velocity="{_fmt(ROLLER_VELOCITY_LIMIT_RAD_S)}"/>')
        lines.append('    <dynamics damping="0.0001" friction="0.0"/>')
        lines.append("  </joint>")
    return lines


def build_urdf(wheel_diameter_mm: int = DEFAULT_WHEEL_DIAMETER_MM) -> str:
    if wheel_diameter_mm not in SUPPORTED_WHEEL_DIAMETERS_MM:
        raise ValueError(f"unsupported wheel diameter {wheel_diameter_mm}mm; expected one of {SUPPORTED_WHEEL_DIAMETERS_MM}")
    wheel_radius_m = wheel_diameter_mm / 2000.0
    lines: list[str] = []
    lines.append("<!-- Generated by sim/isaacsim/scripts/generate_mecanum_robot_urdf.py; do not edit by hand. -->")
    # The mecanum_* attributes are read by create_mecanum_competition_scene.py
    # so sim kinematics always match the authored wheel geometry.
    lines.append(
        '<robot name="robot_mk3_mecanum_sim" '
        f'mecanum_wheel_radius_m="{_fmt(wheel_radius_m)}" '
        f'mecanum_half_length_m="{_fmt(HALF_LENGTH_M)}" '
        f'mecanum_half_width_m="{_fmt(HALF_WIDTH_M)}" '
        f'mecanum_wheel_diameter_mm="{wheel_diameter_mm}">'
    )
    lines.append('  <material name="plate"><color rgba="0.54 0.55 0.52 1"/></material>')
    lines.append('  <material name="rubber"><color rgba="0.02 0.022 0.025 1"/></material>')
    lines.append('  <material name="gripper"><color rgba="0.88 0.64 0.16 1"/></material>')
    lines.append('  <material name="camera"><color rgba="0.06 0.11 0.17 1"/></material>')
    lines.append('  <material name="lidar"><color rgba="0.05 0.05 0.055 1"/></material>')
    lines.append('  <material name="jetson"><color rgba="0.05 0.35 0.18 1"/></material>')
    lines.append('  <material name="battery"><color rgba="0.12 0.12 0.14 1"/></material>')

    # Chassis: same visual mesh as mobile_manipulator_collision; collision
    # boxes trimmed so nothing dips below z=0.075 near the wheels (the
    # mecanum wheels + rollers own the space below).
    lines.append('  <link name="base_link">')
    lines.append("    <inertial>")
    lines.append('      <origin xyz="0 0 0.12" rpy="0 0 0"/>')
    lines.append('      <mass value="2.4"/>')
    lines.append('      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/>')
    lines.append("    </inertial>")
    # No STL visual: the old MK2 mesh no longer matches the narrower MK4 body;
    # the collision primitives below are made renderable at import time.
    # Plates: 0.19 wide (MK3 revision) x 0.25 front-back (5 cm shorter).
    for origin, size in (
        ("0 0 0.173", "0.19 0.25 0.006"),
        ("0 0 0.258", "0.19 0.21 0.006"),
        ("0 0 0.333", "0.19 0.17 0.006"),
        ("0 -0.105 0.115", "0.19 0.006 0.08"),
        ("-0.095 0.06 0.12", "0.01 0.14 0.08"),
        ("0.095 0.06 0.12", "0.01 0.14 0.08"),
        ("0 0.112 0.125", "0.17 0.04 0.032"),
        # Gripper backstop: objects stay in the finger region, never under the body.
        ("0 0.125 0.06", "0.15 0.006 0.1"),
    ):
        lines.append("    <collision>")
        lines.append(f'      <origin xyz="{origin}" rpy="0 0 0"/>')
        lines.append(f"      <geometry><box size=\"{size}\"/></geometry>")
        lines.append("    </collision>")
    lines.append("  </link>")

    for name, x_sign, y_sign, gamma_sign in WHEELS:
        lines.extend(
            wheel_assembly(name, x_sign * HALF_WIDTH_M, y_sign * HALF_LENGTH_M, gamma_sign, wheel_radius_m)
        )

    # Gripper fingers: thin linear-bearing arms (2026-07-07 measurements:
    # ~1 cm thick, ~10 cm front-back). Primitive box visuals match reality
    # better than the old thick STL.
    finger_box = f"{_fmt(FINGER_THICKNESS_M)} {_fmt(FINGER_LENGTH_M)} {_fmt(FINGER_HEIGHT_M)}"
    for side, sign in (("left", -1.0), ("right", 1.0)):
        lines.append(f'  <link name="{side}_finger">')
        lines.append("    <inertial>")
        lines.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
        lines.append('      <mass value="0.06"/>')
        lines.append('      <inertia ixx="0.00006" ixy="0" ixz="0" iyy="0.00006" iyz="0" izz="0.00006"/>')
        lines.append("    </inertial>")
        lines.append("    <visual>")
        lines.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
        lines.append(f"      <geometry><box size=\"{finger_box}\"/></geometry>")
        lines.append('      <material name="gripper"/>')
        lines.append("    </visual>")
        lines.append("    <collision>")
        lines.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
        lines.append(f"      <geometry><box size=\"{finger_box}\"/></geometry>")
        lines.append("    </collision>")
        lines.append("  </link>")
        lines.append(f'  <joint name="{side}_finger_slide" type="prismatic">')
        lines.append(f'    <origin xyz="{_fmt(sign * 0.0575)} 0.145 0.055" rpy="0 0 0"/>')
        lines.append('    <parent link="base_link"/>')
        lines.append(f'    <child link="{side}_finger"/>')
        lines.append(f'    <axis xyz="{_fmt(sign)} 0 0"/>')
        lines.append('    <limit lower="0" upper="0.045" effort="40" velocity="0.25"/>')
        lines.append("  </joint>")

    # Fixed sensors/components: identical placement to mobile_manipulator_collision.
    fixed_parts = (
        ("near_d435i", "0 0.118 0.218", "-0.523599 0 0", "camera", '<box size="0.09 0.025 0.025"/>', 0.08),
        ("top_d435i", "0 0.128 0.368", "-0.174533 0 0", "camera", '<box size="0.09 0.025 0.025"/>', 0.08),
        ("lidar", "0 -0.02 0.28", "0 0 0", "lidar", '<cylinder radius="0.048" length="0.035"/>', 0.15),
        ("jetson_orin_nano", "0 -0.045 0.205", "0 0 0", "jetson", '<box size="0.1 0.08 0.035"/>', 0.25),
        ("battery", "0 -0.105 0.09", "0 0 0", "battery", '<box size="0.145 0.06 0.045"/>', 0.7),
    )
    for name, xyz, rpy, material, collision_geom, mass in fixed_parts:
        mesh = name if name != "jetson_orin_nano" else "jetson_orin_nano"
        lines.append(f'  <link name="{name}">')
        lines.append("    <inertial>")
        lines.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
        lines.append(f'      <mass value="{_fmt(mass)}"/>')
        lines.append('      <inertia ixx="0.0002" ixy="0" ixz="0" iyy="0.0002" iyz="0" izz="0.0002"/>')
        lines.append("    </inertial>")
        lines.append("    <visual>")
        lines.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
        lines.append(
            f'      <geometry><mesh filename="meshes/mobile_manipulator_collision/{mesh}.stl" scale="1 1 1"/></geometry>'
        )
        lines.append(f'      <material name="{material}"/>')
        lines.append("    </visual>")
        lines.append("    <collision>")
        lines.append('      <origin xyz="0 0 0" rpy="0 0 0"/>')
        lines.append(f"      <geometry>{collision_geom}</geometry>")
        lines.append("    </collision>")
        lines.append("  </link>")
        lines.append(f'  <joint name="{name}_mount" type="fixed">')
        lines.append(f'    <origin xyz="{xyz}" rpy="{rpy}"/>')
        lines.append('    <parent link="base_link"/>')
        lines.append(f'    <child link="{name}"/>')
        lines.append("  </joint>")

    lines.append("</robot>")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--wheel-diameter-mm",
        type=int,
        choices=SUPPORTED_WHEEL_DIAMETERS_MM,
        default=DEFAULT_WHEEL_DIAMETER_MM,
        help="Purchased wheel options: 68 mm (matches real.yaml 0.034 m radius) or 80 mm.",
    )
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_urdf(args.wheel_diameter_mm), encoding="utf-8")
    wheel_names = ", ".join(name for name, *_ in WHEELS)
    print(f"wrote {output}")
    print(
        f"wheels: {wheel_names} | diameter={args.wheel_diameter_mm}mm "
        f"(radius={args.wheel_diameter_mm / 2000.0}) half_length={HALF_LENGTH_M} "
        f"half_width={HALF_WIDTH_M} rollers/wheel={ROLLER_COUNT} gamma={ROLLER_GAMMA_DEG}deg | "
        f"motor=JGB37-520 12V 333rpm (effort<={WHEEL_EFFORT_LIMIT_NM}Nm, vel<={WHEEL_VELOCITY_LIMIT_RAD_S}rad/s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
