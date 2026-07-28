import math
import sys
from pathlib import Path


from arena_lightweight_control.map_localization import (  # noqa: E402
    KnownMapLocalizer,
    OccupancyMap,
    Pose2D,
    WallRangeLocalizer,
    min_front_range,
    min_range_in_sector,
    normalize_angle,
    scan_to_world_points,
)


def test_known_map_localizer_prefers_pose_aligned_to_wall():
    occupied = [False] * 100
    for row in range(10):
        occupied[row * 10 + 7] = True
    arena_map = OccupancyMap(10, 10, 0.1, [0.0, 0.0, 0.0], occupied)
    localizer = KnownMapLocalizer(
        arena_map,
        max_beams=8,
        search_xy_m=0.10,
        xy_step_m=0.10,
        search_yaw_rad=0.0,
    )

    result = localizer.match([(0.5, 0.0)], Pose2D(0.1, 0.5, 0.0))

    assert result.success is True
    assert result.pose.x > 0.1
    assert result.latency_ms < 100.0


def test_scan_to_points_downsamples_valid_ranges():
    arena_map = OccupancyMap(4, 4, 0.1, [0.0, 0.0, 0.0], [False] * 16)
    localizer = KnownMapLocalizer(arena_map, max_beams=3)

    points = localizer.scan_to_points(
        [math.inf, 0.2, 0.3, 0.4, 9.0, math.nan],
        angle_min=0.0,
        angle_increment=0.1,
        range_min=0.05,
        range_max=3.0,
    )

    assert len(points) == 3


def test_scan_to_points_applies_yaw_offset():
    arena_map = OccupancyMap(4, 4, 0.1, [0.0, 0.0, 0.0], [False] * 16)
    localizer = KnownMapLocalizer(arena_map, max_beams=8)

    points = localizer.scan_to_points(
        [1.0],
        angle_min=0.0,
        angle_increment=0.1,
        range_min=0.05,
        range_max=3.0,
        yaw_offset=math.pi,
    )

    assert len(points) == 1
    assert points[0][0] < -0.99
    assert abs(points[0][1]) < 1e-6


def test_min_front_range_uses_center_sector_only():
    distance = min_front_range(
        [0.4, 0.2, 0.6],
        angle_min=-0.5,
        angle_increment=0.5,
        range_min=0.05,
        range_max=3.0,
        sector_rad=0.2,
    )

    assert distance == 0.2


def test_min_range_in_sector_applies_yaw_offset():
    distance = min_range_in_sector(
        [0.2, 0.8],
        angle_min=-math.pi,
        angle_increment=math.pi,
        range_min=0.05,
        range_max=3.0,
        sector_rad=0.2,
        sector_center_rad=0.0,
        yaw_offset=math.pi,
    )

    assert distance == 0.2


def test_scan_to_world_points_downsamples_and_uses_pose():
    points, total = scan_to_world_points(
        [1.0, 2.0, math.inf, 3.0],
        angle_min=0.0,
        angle_increment=math.pi / 2.0,
        range_min=0.05,
        range_max=4.0,
        yaw_offset=0.0,
        pose=Pose2D(1.0, 2.0, math.pi / 2.0),
        max_points=2,
    )

    assert total == 3
    assert len(points) == 2
    assert abs(points[0][0] - 1.0) < 1e-6
    assert abs(points[0][1] - 3.0) < 1e-6
    assert abs(points[1][0] - 4.0) < 1e-6
    assert abs(points[1][1] - 2.0) < 1e-6


def test_wall_range_localizer_tracks_square_wall_pose():
    arena_map = square_wall_map()
    localizer = WallRangeLocalizer(
        arena_map,
        max_beams=32,
        search_xy_m=0.12,
        search_yaw_rad=0.14,
        xy_step_m=0.04,
        yaw_step_rad=0.07,
    )
    true_pose = Pose2D(0.42, -0.28, 0.35)
    ranges, angle_min, angle_increment = square_wall_scan(localizer, true_pose)

    result = localizer.match_scan(
        ranges,
        angle_min,
        angle_increment,
        range_min=0.05,
        range_max=6.0,
        yaw_offset=0.0,
        prior_pose=Pose2D(0.32, -0.18, 0.25),
    )

    assert result.success is True
    assert abs(result.pose.x - true_pose.x) <= 0.04
    assert abs(result.pose.y - true_pose.y) <= 0.04
    assert abs(normalize_angle(result.pose.yaw - true_pose.yaw)) <= 0.06
    assert result.beam_count == 32
    assert result.candidate_count <= 320
    assert result.score < 0.07


def test_wall_range_localizer_global_reseed_escapes_bad_prior():
    arena_map = square_wall_map()
    localizer = WallRangeLocalizer(
        arena_map,
        max_beams=32,
        search_xy_m=0.12,
        search_yaw_rad=0.14,
        xy_step_m=0.04,
        yaw_step_rad=0.07,
    )
    true_pose = Pose2D(0.80, -0.45, 0.50)
    ranges, angle_min, angle_increment = square_wall_scan(localizer, true_pose)
    bad_prior = Pose2D(-1.25, 1.10, -2.20)

    local_result = localizer.match_scan(
        ranges,
        angle_min,
        angle_increment,
        range_min=0.05,
        range_max=6.0,
        yaw_offset=0.0,
        prior_pose=bad_prior,
    )
    global_result = localizer.match_scan_global(
        ranges,
        angle_min,
        angle_increment,
        range_min=0.05,
        range_max=6.0,
        yaw_offset=0.0,
        prior_pose=bad_prior,
        xy_step_m=0.32,
        yaw_step_rad=0.35,
        max_beams=16,
    )

    assert global_result.success is True
    assert global_result.score < local_result.score
    assert abs(global_result.pose.x - true_pose.x) <= 0.18
    assert abs(global_result.pose.y - true_pose.y) <= 0.18
    assert abs(normalize_angle(global_result.pose.yaw - true_pose.yaw)) <= 0.18
    assert global_result.beam_count == 16
    assert global_result.candidate_count <= 4000


def test_wall_range_localizer_global_coarse_ignores_prior():
    arena_map = square_wall_map()
    localizer = WallRangeLocalizer(
        arena_map,
        max_beams=32,
        search_xy_m=0.12,
        search_yaw_rad=0.14,
        xy_step_m=0.04,
        yaw_step_rad=0.07,
    )
    true_pose = Pose2D(-0.60, -0.90, 0.80)
    ranges, angle_min, angle_increment = square_wall_scan(localizer, true_pose)

    result = localizer.match_scan_global_coarse(
        ranges,
        angle_min,
        angle_increment,
        range_min=0.05,
        range_max=6.0,
        yaw_offset=0.0,
        prior_pose=Pose2D(1.35, -1.20, 2.20),
        xy_step_m=0.32,
        yaw_step_rad=0.35,
        max_beams=16,
    )

    assert result.success is True
    assert result.reason.startswith("wall_range_global_coarse")
    assert result.beam_count == 16
    assert result.candidate_count <= 3200
    assert abs(result.pose.x - true_pose.x) <= 0.18
    assert abs(result.pose.y - true_pose.y) <= 0.18
    assert abs(normalize_angle(result.pose.yaw - true_pose.yaw)) <= 0.18


def test_wall_range_localizer_ignores_short_object_returns():
    arena_map = square_wall_map()
    localizer = WallRangeLocalizer(
        arena_map,
        max_beams=32,
        search_xy_m=0.12,
        search_yaw_rad=0.14,
        xy_step_m=0.04,
        yaw_step_rad=0.07,
    )
    true_pose = Pose2D(-0.35, 0.25, -0.18)
    ranges, angle_min, angle_increment = square_wall_scan(localizer, true_pose)
    for index in range(0, len(ranges), 9):
        ranges[index] = max(0.08, ranges[index] - 0.65)

    result = localizer.match_scan(
        ranges,
        angle_min,
        angle_increment,
        range_min=0.05,
        range_max=6.0,
        yaw_offset=0.0,
        prior_pose=Pose2D(-0.45, 0.15, -0.10),
    )

    assert result.success is True
    assert abs(result.pose.x - true_pose.x) <= 0.06
    assert abs(result.pose.y - true_pose.y) <= 0.06
    assert abs(normalize_angle(result.pose.yaw - true_pose.yaw)) <= 0.08
    assert result.beam_count == 32
    assert result.candidate_count <= 320


def test_stadium_known_map_match_stays_under_100ms():
    repo_root = Path(__file__).resolve().parents[1]
    arena_map = OccupancyMap.from_yaml(
        repo_root / "navigation" / "ros2" / "arena_lightweight_control"
        / "maps" / "stadium.yaml"
    )
    localizer = KnownMapLocalizer(
        arena_map,
        max_beams=72,
        search_xy_m=0.10,
        search_yaw_rad=0.14,
        xy_step_m=0.02,
        yaw_step_rad=0.035,
    )
    prior = Pose2D(0.06, -0.04, 0.035)

    result = localizer.match(_square_wall_scan_points(72), prior)

    assert result.success is True
    assert result.latency_ms < 100.0
    assert result.beam_count == 72
    assert result.candidate_count <= 1200
    assert result.score < 0.08


def _square_wall_scan_points(count, half_extent=2.0):
    points = []
    for index in range(count):
        angle = -math.pi + (2.0 * math.pi * index / count)
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        distances = []
        if abs(cos_angle) > 1e-6:
            distances.extend((half_extent / cos_angle, -half_extent / cos_angle))
        if abs(sin_angle) > 1e-6:
            distances.extend((half_extent / sin_angle, -half_extent / sin_angle))
        distance = min(value for value in distances if value > 0.0)
        points.append((distance * cos_angle, distance * sin_angle))
    return points


def square_wall_map():
    width = 50
    height = 50
    resolution = 0.1
    origin = [-2.5, -2.5, 0.0]
    occupied = []
    for row in range(height):
        y = origin[1] + (height - 1 - row + 0.5) * resolution
        for col in range(width):
            x = origin[0] + (col + 0.5) * resolution
            occupied.append(abs(abs(x) - 2.0) <= 0.051 or abs(abs(y) - 2.0) <= 0.051)
    return OccupancyMap(width, height, resolution, origin, occupied)


def square_wall_scan(localizer, pose, count=180):
    angle_min = -math.pi
    angle_increment = 2.0 * math.pi / float(count)
    ranges = []
    for index in range(count):
        angle = angle_min + index * angle_increment
        distance = localizer.expected_wall_range(
            pose.x,
            pose.y,
            float(pose.yaw) + angle,
        )
        ranges.append(distance if distance is not None else math.inf)
    return ranges, angle_min, angle_increment
