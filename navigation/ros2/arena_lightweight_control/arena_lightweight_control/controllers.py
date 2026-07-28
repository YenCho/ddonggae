#!/usr/bin/env python3
import math
from dataclasses import dataclass
from typing import Optional

from arena_lightweight_control.map_localization import Pose2D, normalize_angle


@dataclass(frozen=True)
class ChassisCommand:
    vx: float
    vy: float
    omega: float
    reached: bool = False
    blocked: bool = False
    phase: str = "idle"


@dataclass(frozen=True)
class GoalControllerConfig:
    xy_tolerance_m: float = 0.15
    yaw_tolerance_rad: float = 0.40
    near_goal_slow_radius_m: float = 0.10
    near_goal_speed_scale: float = 0.5
    rotate_to_goal_threshold_rad: float = 0.35
    obstacle_stop_distance_m: float = 0.22
    max_linear_mps: float = 0.36
    min_linear_mps: float = 0.075
    max_angular_rps: float = 0.85
    final_yaw_min_angular_rps: float = 0.0
    linear_gain: float = 1.425
    angular_gain: float = 1.7
    drive_angular_deadband_rad: float = 0.08
    drive_angular_gain_scale: float = 0.45


class ChassisController:
    def compute_command(
        self,
        pose: Pose2D,
        goal: Pose2D,
        obstacle_front_m: Optional[float],
    ) -> ChassisCommand:
        raise NotImplementedError


class DiffDriveController(ChassisController):
    """Current two-wheel + ball-caster controller."""

    def __init__(self, config: GoalControllerConfig):
        self.config = config

    def compute_command(
        self,
        pose: Pose2D,
        goal: Pose2D,
        obstacle_front_m: Optional[float],
    ) -> ChassisCommand:
        dx = goal.x - pose.x
        dy = goal.y - pose.y
        distance = math.hypot(dx, dy)
        speed_scale = near_goal_speed_scale(self.config, distance)
        if distance <= self.config.xy_tolerance_m:
            if goal.yaw is None or not math.isfinite(float(goal.yaw)):
                return ChassisCommand(0.0, 0.0, 0.0, reached=True, phase="reached")
            yaw_error = normalize_angle(float(goal.yaw) - pose.yaw)
            if abs(yaw_error) <= self.config.yaw_tolerance_rad:
                return ChassisCommand(0.0, 0.0, 0.0, reached=True, phase="reached")
            return ChassisCommand(
                0.0,
                0.0,
                self._final_yaw_angular(yaw_error, speed_scale),
                phase="final_yaw",
            )

        target_heading = math.atan2(dy, dx)
        heading_error = normalize_angle(target_heading - pose.yaw)
        if abs(heading_error) > self.config.rotate_to_goal_threshold_rad:
            return ChassisCommand(
                0.0,
                0.0,
                self._angular(heading_error) * speed_scale,
                phase="rotate_to_goal",
            )

        if (
            obstacle_front_m is not None
            and obstacle_front_m <= self.config.obstacle_stop_distance_m
        ):
            return ChassisCommand(0.0, 0.0, 0.0, blocked=True, phase="obstacle_stop")

        linear = min(self.config.max_linear_mps, self.config.linear_gain * distance)
        linear = max(self.config.min_linear_mps, linear)
        linear *= speed_scale
        if abs(heading_error) <= self.config.drive_angular_deadband_rad:
            angular = 0.0
        else:
            angular = (
                self._angular(heading_error)
                * self.config.drive_angular_gain_scale
                * speed_scale
            )
        return ChassisCommand(linear, 0.0, angular, phase="drive")

    def _angular(self, error: float) -> float:
        return clamp(
            self.config.angular_gain * error,
            -self.config.max_angular_rps,
            self.config.max_angular_rps,
        )

    def _final_yaw_angular(self, error: float, speed_scale: float) -> float:
        return apply_min_abs(
            self._angular(error) * speed_scale,
            self.config.final_yaw_min_angular_rps,
            self.config.max_angular_rps * max(0.0, speed_scale),
        )


class MecanumController(ChassisController):
    """Future holonomic controller; kept separate from planner/localization."""

    def __init__(self, config: GoalControllerConfig):
        self.config = config

    def compute_command(
        self,
        pose: Pose2D,
        goal: Pose2D,
        obstacle_front_m: Optional[float],
    ) -> ChassisCommand:
        dx = goal.x - pose.x
        dy = goal.y - pose.y
        distance = math.hypot(dx, dy)
        speed_scale = near_goal_speed_scale(self.config, distance)
        if distance <= self.config.xy_tolerance_m:
            if goal.yaw is None or not math.isfinite(float(goal.yaw)):
                return ChassisCommand(0.0, 0.0, 0.0, reached=True, phase="reached")
            yaw_error = normalize_angle(float(goal.yaw) - pose.yaw)
            if abs(yaw_error) <= self.config.yaw_tolerance_rad:
                return ChassisCommand(0.0, 0.0, 0.0, reached=True, phase="reached")
            return ChassisCommand(
                0.0,
                0.0,
                self._final_yaw_angular(yaw_error, speed_scale),
                phase="final_yaw",
            )
        cos_yaw = math.cos(-pose.yaw)
        sin_yaw = math.sin(-pose.yaw)
        robot_x = dx * cos_yaw - dy * sin_yaw
        robot_y = dx * sin_yaw + dy * cos_yaw
        scale = min(self.config.max_linear_mps, self.config.linear_gain * distance)
        scale *= speed_scale
        if distance > 1e-6:
            robot_x = robot_x / distance * scale
            robot_y = robot_y / distance * scale
        # Only treat a front obstacle as blocking when the commanded motion
        # actually has a forward component; strafing or backing away past an
        # obstacle is one of the main reasons to run mecanum wheels.
        if (
            obstacle_front_m is not None
            and obstacle_front_m <= self.config.obstacle_stop_distance_m
            and robot_x > 1e-3
        ):
            return ChassisCommand(0.0, 0.0, 0.0, blocked=True, phase="obstacle_stop")
        return ChassisCommand(
            robot_x,
            robot_y,
            self.goal_yaw_command(pose, goal) * speed_scale,
            phase="holonomic_drive",
        )

    def _angular(self, error: float) -> float:
        return clamp(
            self.config.angular_gain * error,
            -self.config.max_angular_rps,
            self.config.max_angular_rps,
        )

    def _final_yaw_angular(self, error: float, speed_scale: float) -> float:
        return apply_min_abs(
            self._angular(error) * speed_scale,
            self.config.final_yaw_min_angular_rps,
            self.config.max_angular_rps * max(0.0, speed_scale),
        )

    def goal_yaw_command(self, pose: Pose2D, goal: Pose2D) -> float:
        if goal.yaw is None or not math.isfinite(float(goal.yaw)):
            return 0.0
        yaw_error = normalize_angle(float(goal.yaw) - pose.yaw)
        # Mirror the diff-drive deadband: while driving, ignore small yaw
        # errors so localization yaw jitter does not turn into a stream of
        # micro wz commands (the main straight-drive curving cause).
        if abs(yaw_error) <= self.config.drive_angular_deadband_rad:
            return 0.0
        return self._angular(yaw_error) * self.config.drive_angular_gain_scale


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def apply_min_abs(value: float, min_abs: float, max_abs: float) -> float:
    if value == 0.0:
        return 0.0
    limit = max(0.0, abs(max_abs))
    if limit == 0.0:
        return 0.0
    floor = min(max(0.0, abs(min_abs)), limit)
    magnitude = clamp(abs(value), floor, limit)
    return math.copysign(magnitude, value)


def near_goal_speed_scale(config: GoalControllerConfig, distance_m: float) -> float:
    radius = float(config.near_goal_slow_radius_m)
    if radius <= 0.0 or distance_m > radius:
        return 1.0
    return clamp(float(config.near_goal_speed_scale), 0.0, 1.0)
