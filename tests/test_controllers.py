import math

import pytest

# arena_control_node imports rclpy. Everything under test here is plain Python,
# but the module-level import pulls ROS 2 in, so a machine without it would fail
# at collection and take the whole suite down with it.
pytest.importorskip("rclpy", reason="ROS 2 not sourced; the rest of the suite runs without it")

from arena_lightweight_control.controllers import (  # noqa: E402
    DiffDriveController,
    GoalControllerConfig,
    MecanumController,
)
from arena_lightweight_control.arena_control_node import (  # noqa: E402
    goal_for_position_latch,
    resolve_goal_yaw,
    update_goal_position_latch,
)
from arena_lightweight_control.map_localization import Pose2D  # noqa: E402


def test_diff_drive_rotates_before_driving():
    controller = DiffDriveController(GoalControllerConfig(max_angular_rps=0.5))

    command = controller.compute_command(
        Pose2D(0.0, 0.0, 0.0),
        Pose2D(0.0, 1.0, 0.0),
        obstacle_front_m=None,
    )

    assert command.phase == "rotate_to_goal"
    assert command.vx == 0.0
    assert command.omega == 0.5


def test_diff_drive_stops_for_front_obstacle():
    controller = DiffDriveController(GoalControllerConfig(obstacle_stop_distance_m=0.3))

    command = controller.compute_command(
        Pose2D(0.0, 0.0, 0.0),
        Pose2D(1.0, 0.0, 0.0),
        obstacle_front_m=0.2,
    )

    assert command.blocked is True
    assert command.vx == 0.0
    assert command.omega == 0.0


def test_diff_drive_reaches_xy_goal_without_final_yaw():
    controller = DiffDriveController(GoalControllerConfig())

    command = controller.compute_command(
        Pose2D(0.0, 0.0, 1.2),
        Pose2D(0.02, 0.02, None),
        obstacle_front_m=None,
    )

    assert command.reached is True
    assert command.phase == "reached"
    assert command.omega == 0.0


def test_diff_drive_uses_fifteen_centimeter_default_xy_tolerance():
    controller = DiffDriveController(GoalControllerConfig())

    inside = controller.compute_command(
        Pose2D(0.0, 0.0, 0.0),
        Pose2D(0.149, 0.0, None),
        obstacle_front_m=None,
    )
    outside = controller.compute_command(
        Pose2D(0.0, 0.0, 0.0),
        Pose2D(0.151, 0.0, None),
        obstacle_front_m=None,
    )

    assert inside.reached is True
    assert outside.reached is False
    assert outside.phase == "drive"


def test_diff_drive_aligns_final_yaw_inside_xy_tolerance():
    controller = DiffDriveController(GoalControllerConfig())

    command = controller.compute_command(
        Pose2D(0.0, 0.0, 0.0),
        Pose2D(0.02, 0.0, math.pi / 2.0),
        obstacle_front_m=None,
    )

    assert command.phase == "final_yaw"
    assert command.vx == 0.0
    assert command.omega > 0.0


def test_diff_drive_slows_final_yaw_inside_near_goal_radius():
    controller = DiffDriveController(GoalControllerConfig())

    command = controller.compute_command(
        Pose2D(0.0, 0.0, 0.0),
        Pose2D(0.02, 0.0, math.pi / 2.0),
        obstacle_front_m=None,
    )

    assert command.phase == "final_yaw"
    assert command.vx == 0.0
    assert math.isclose(command.omega, 0.425)


def test_diff_drive_skips_small_final_yaw_inside_relaxed_tolerance():
    controller = DiffDriveController(GoalControllerConfig())

    command = controller.compute_command(
        Pose2D(0.0, 0.0, 1.36),
        Pose2D(0.02, 0.0, 1.57),
        obstacle_front_m=None,
    )

    assert command.reached is True
    assert command.phase == "reached"
    assert command.omega == 0.0


def test_diff_drive_final_yaw_minimum_is_opt_in():
    controller = DiffDriveController(
        GoalControllerConfig(
            yaw_tolerance_rad=0.20,
            final_yaw_min_angular_rps=0.32,
        )
    )

    command = controller.compute_command(
        Pose2D(0.0, 0.0, 1.36),
        Pose2D(0.02, 0.0, 1.57),
        obstacle_front_m=None,
    )

    assert command.phase == "final_yaw"
    assert command.vx == 0.0
    assert math.isclose(command.omega, 0.32)


def test_diff_drive_stops_final_yaw_inside_yaw_tolerance():
    controller = DiffDriveController(GoalControllerConfig())

    command = controller.compute_command(
        Pose2D(0.0, 0.0, 1.38),
        Pose2D(0.02, 0.0, 1.57),
        obstacle_front_m=None,
    )

    assert command.reached is True
    assert command.phase == "reached"
    assert command.omega == 0.0


def test_goal_without_yaw_defaults_to_map_positive_y():
    assert math.isclose(resolve_goal_yaw(None, math.pi / 2.0), math.pi / 2.0)
    assert math.isclose(resolve_goal_yaw(0.0, math.pi / 2.0), 0.0)
    assert resolve_goal_yaw(None, math.nan) is None


def test_diff_drive_allows_turning_away_from_front_obstacle():
    controller = DiffDriveController(
        GoalControllerConfig(obstacle_stop_distance_m=0.3, max_angular_rps=0.5)
    )

    command = controller.compute_command(
        Pose2D(0.0, 0.0, 0.0),
        Pose2D(0.0, 1.0, 0.0),
        obstacle_front_m=0.2,
    )

    assert command.blocked is False
    assert command.phase == "rotate_to_goal"
    assert command.vx == 0.0
    assert command.omega == 0.5


def test_diff_drive_deadbands_small_drive_heading_error():
    controller = DiffDriveController(
        GoalControllerConfig(
            drive_angular_deadband_rad=0.08,
            drive_angular_gain_scale=0.45,
        )
    )

    command = controller.compute_command(
        Pose2D(0.0, 0.0, 0.03),
        Pose2D(1.0, 0.0, None),
        obstacle_front_m=None,
    )

    assert command.phase == "drive"
    assert command.vx == 0.36
    assert command.omega == 0.0


def test_diff_drive_scales_drive_heading_correction():
    controller = DiffDriveController(
        GoalControllerConfig(
            angular_gain=1.7,
            drive_angular_deadband_rad=0.08,
            drive_angular_gain_scale=0.45,
        )
    )

    command = controller.compute_command(
        Pose2D(0.0, 0.0, -0.12),
        Pose2D(1.0, 0.0, None),
        obstacle_front_m=None,
    )

    assert command.phase == "drive"
    assert 0.08 < command.omega < 0.10


def test_diff_drive_slows_linear_and_angular_inside_near_goal_radius():
    controller = DiffDriveController(
        GoalControllerConfig(
            xy_tolerance_m=0.05,
            near_goal_slow_radius_m=0.10,
            angular_gain=1.7,
            drive_angular_deadband_rad=0.0,
            drive_angular_gain_scale=0.45,
        )
    )

    command = controller.compute_command(
        Pose2D(0.0, 0.0, -0.2),
        Pose2D(0.09, 0.0, None),
        obstacle_front_m=None,
    )

    assert command.phase == "drive"
    assert math.isclose(command.vx, 0.064125)
    assert math.isclose(command.omega, 0.0765)


def test_goal_position_latch_ignores_small_localization_jump():
    latched, release_count = update_goal_position_latch(
        False,
        0,
        distance_m=0.149,
        enter_tolerance_m=0.15,
        release_tolerance_m=0.25,
        release_count_threshold=3,
    )
    assert latched is True
    assert release_count == 0

    latched, release_count = update_goal_position_latch(
        latched,
        release_count,
        distance_m=0.20,
        enter_tolerance_m=0.15,
        release_tolerance_m=0.25,
        release_count_threshold=3,
    )
    assert latched is True
    assert release_count == 0


def test_goal_position_latch_releases_after_consecutive_large_jumps():
    latched = True
    release_count = 0
    for expected_count in (1, 2):
        latched, release_count = update_goal_position_latch(
            latched,
            release_count,
            distance_m=0.26,
            enter_tolerance_m=0.15,
            release_tolerance_m=0.25,
            release_count_threshold=3,
        )
        assert latched is True
        assert release_count == expected_count

    latched, release_count = update_goal_position_latch(
        latched,
        release_count,
        distance_m=0.26,
        enter_tolerance_m=0.15,
        release_tolerance_m=0.25,
        release_count_threshold=3,
    )
    assert latched is False
    assert release_count == 0


def test_latched_goal_only_aligns_yaw_without_xy_drive():
    controller = DiffDriveController(GoalControllerConfig())
    pose = Pose2D(0.13, 0.0, 0.0)
    goal = Pose2D(0.0, 0.0, math.pi / 2.0)

    command = controller.compute_command(
        pose,
        goal_for_position_latch(pose, goal),
        obstacle_front_m=None,
    )

    assert command.phase == "final_yaw"
    assert command.vx == 0.0
    assert command.omega > 0.0


def test_mecanum_controller_keeps_holonomic_command_separate():
    controller = MecanumController(GoalControllerConfig(max_linear_mps=0.2))

    command = controller.compute_command(
        Pose2D(0.0, 0.0, 0.0),
        Pose2D(0.0, 1.0, math.pi / 2.0),
        obstacle_front_m=None,
    )

    assert command.phase == "holonomic_drive"
    assert command.vx == 0.0
    assert command.vy > 0.0
    assert command.omega > 0.0


def test_mecanum_stops_for_front_obstacle_when_moving_forward():
    controller = MecanumController(
        GoalControllerConfig(obstacle_stop_distance_m=0.3)
    )

    command = controller.compute_command(
        Pose2D(0.0, 0.0, 0.0),
        Pose2D(1.0, 0.0, None),
        obstacle_front_m=0.2,
    )

    assert command.blocked
    assert command.phase == "obstacle_stop"


def test_mecanum_allows_strafe_past_front_obstacle():
    controller = MecanumController(
        GoalControllerConfig(obstacle_stop_distance_m=0.3)
    )

    command = controller.compute_command(
        Pose2D(0.0, 0.0, 0.0),
        Pose2D(0.0, 1.0, None),
        obstacle_front_m=0.2,
    )

    assert not command.blocked
    assert command.phase == "holonomic_drive"
    assert command.vy > 0.0
    assert abs(command.vx) < 1e-9


def test_mecanum_drive_yaw_deadband_suppresses_micro_omega():
    controller = MecanumController(
        GoalControllerConfig(drive_angular_deadband_rad=0.08)
    )

    command = controller.compute_command(
        Pose2D(0.0, 0.0, 0.0),
        Pose2D(1.0, 0.0, 0.05),
        obstacle_front_m=None,
    )

    assert command.phase == "holonomic_drive"
    assert command.omega == 0.0
