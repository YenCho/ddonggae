"""Mecanum kinematics, odometry, and firmware protocol helpers.

Pairs with hardware/firmware/arduino_mecanum. Wheel order is
always (front_left, front_right, rear_left, rear_right). Body axes: +x
forward, +y left, +yaw CCW.
"""

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from robot_hardware.base_control import clamp, normalize_angle

WHEEL_NAMES = ('front_left', 'front_right', 'rear_left', 'rear_right')

MODE_IDLE = 0
MODE_VELOCITY = 1
MODE_POSITION = 2


@dataclass(frozen=True)
class MecanumStateSample:
    device_ms: int
    ticks: Tuple[int, int, int, int]
    pwm: Tuple[int, int, int, int]
    mode: int


def parse_mecanum_state_line(line):
    """Parse a firmware STATE line.

    STATE,<ms>,<fl_t>,<fr_t>,<rl_t>,<rr_t>,<fl_p>,<fr_p>,<rl_p>,<rr_p>,<mode>
    """
    fields = line.strip().split(',')
    if len(fields) < 11 or fields[0] != 'STATE':
        return None
    try:
        return MecanumStateSample(
            device_ms=int(fields[1]),
            ticks=tuple(int(value) for value in fields[2:6]),
            pwm=tuple(int(value) for value in fields[6:10]),
            mode=int(fields[10]),
        )
    except ValueError:
        return None


def parse_move_done_line(line):
    """Parse a DONE,move,<fl_err>,<fr_err>,<rl_err>,<rr_err> line."""
    fields = line.strip().split(',')
    if len(fields) < 6 or fields[0] != 'DONE' or fields[1] != 'move':
        return None
    try:
        return tuple(float(value) for value in fields[2:6])
    except ValueError:
        return None


class MecanumKinematics:
    """X-configuration mecanum inverse/forward kinematics."""

    def __init__(self, wheel_radius_m, half_length_m, half_width_m):
        self.wheel_radius_m = float(wheel_radius_m)
        self.half_length_m = float(half_length_m)
        self.half_width_m = float(half_width_m)

    @property
    def rotation_factor(self):
        return self.half_length_m + self.half_width_m

    def body_to_wheels(self, vx, vy, wz):
        """Body twist -> wheel angular velocities (rad/s)."""
        k = self.rotation_factor
        r = self.wheel_radius_m
        return (
            (vx - vy - k * wz) / r,
            (vx + vy + k * wz) / r,
            (vx + vy - k * wz) / r,
            (vx - vy + k * wz) / r,
        )

    def wheels_to_body(self, fl, fr, rl, rr):
        """Wheel angular velocities (rad/s) -> body twist (vx, vy, wz)."""
        r = self.wheel_radius_m
        vx = r * (fl + fr + rl + rr) / 4.0
        vy = r * (-fl + fr + rl - rr) / 4.0
        wz = r * (-fl + fr - rl + rr) / (4.0 * self.rotation_factor)
        return vx, vy, wz

    def clamp_body_twist(self, vx, vy, wz, max_wheel_rad_s):
        """Uniformly scale a body twist so no wheel exceeds the limit."""
        wheels = self.body_to_wheels(vx, vy, wz)
        peak = max(abs(value) for value in wheels)
        if peak <= max_wheel_rad_s or peak <= 0.0:
            return vx, vy, wz
        scale = max_wheel_rad_s / peak
        return vx * scale, vy * scale, wz * scale


class MecanumOdometry:
    """Integrate pose from four wheel encoder tick counters.

    Firmware already applies encoder signs, so ticks arriving here are
    positive for physical forward rotation on every wheel.
    """

    def __init__(self, kinematics: MecanumKinematics, encoder_cpr):
        self.kinematics = kinematics
        self.encoder_cpr = float(encoder_cpr)
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_ticks: Optional[Tuple[int, int, int, int]] = None

    @property
    def rad_per_tick(self):
        return 2.0 * math.pi / self.encoder_cpr

    def reset(self, x=0.0, y=0.0, yaw=0.0):
        self.x = float(x)
        self.y = float(y)
        self.yaw = float(yaw)
        self.last_ticks = None

    def update(self, ticks: Sequence[int]):
        """Advance the pose. Returns body-frame deltas (dx, dy, dyaw)."""
        current = tuple(int(value) for value in ticks)
        if self.last_ticks is None:
            self.last_ticks = current
            return 0.0, 0.0, 0.0

        deltas_rad = [
            (now - before) * self.rad_per_tick
            for now, before in zip(current, self.last_ticks)
        ]
        self.last_ticks = current

        dx, dy, dyaw = self.kinematics.wheels_to_body(*deltas_rad)
        midpoint_yaw = self.yaw + 0.5 * dyaw
        cos_yaw = math.cos(midpoint_yaw)
        sin_yaw = math.sin(midpoint_yaw)
        self.x += dx * cos_yaw - dy * sin_yaw
        self.y += dx * sin_yaw + dy * cos_yaw
        self.yaw = normalize_angle(self.yaw + dyaw)
        return dx, dy, dyaw


def scale_wheel_speeds(wheels, max_wheel_rad_s):
    peak = max(abs(value) for value in wheels)
    if peak <= max_wheel_rad_s or peak <= 0.0:
        return tuple(wheels)
    scale = max_wheel_rad_s / peak
    return tuple(value * scale for value in wheels)


def clamp_twist_components(vx, vy, wz, max_vx, max_vy, max_wz):
    return (
        clamp(vx, -max_vx, max_vx),
        clamp(vy, -max_vy, max_vy),
        clamp(wz, -max_wz, max_wz),
    )
