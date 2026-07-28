#!/usr/bin/env python3
import ast
import heapq
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: Optional[float]


@dataclass(frozen=True)
class MatchResult:
    pose: Pose2D
    score: float
    latency_ms: float
    beam_count: int
    candidate_count: int
    success: bool
    reason: str = ""


@dataclass(frozen=True)
class ArenaBounds:
    xmin: float
    xmax: float
    ymin: float
    ymax: float


@dataclass(frozen=True)
class WallBeam:
    angle: float
    distance: float
    cos_angle: float
    sin_angle: float


class OccupancyMap:
    def __init__(
        self,
        width: int,
        height: int,
        resolution: float,
        origin: Sequence[float],
        occupied: Sequence[bool],
    ):
        if len(occupied) != width * height:
            raise ValueError("occupied grid length does not match map dimensions")
        self.width = int(width)
        self.height = int(height)
        self.resolution = float(resolution)
        self.origin = (
            float(origin[0]),
            float(origin[1]),
            float(origin[2]) if len(origin) >= 3 else 0.0,
        )
        self.occupied = list(bool(value) for value in occupied)
        self.distance_m = self._build_distance_field()

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "OccupancyMap":
        yaml_path = Path(yaml_path).expanduser().resolve()
        config = _read_simple_map_yaml(yaml_path)
        image_path = Path(str(config["image"]))
        if not image_path.is_absolute():
            image_path = yaml_path.parent / image_path

        width, height, pixels = _read_pgm(image_path)
        resolution = float(config.get("resolution", 0.02))
        origin = config.get("origin", [-2.5, -2.5, 0.0])
        occupied_thresh = float(config.get("occupied_thresh", 0.65))
        negate = int(config.get("negate", 0))

        occupied = []
        for pixel in pixels:
            normalized = float(pixel) / 255.0
            occupancy = normalized if negate else 1.0 - normalized
            occupied.append(occupancy >= occupied_thresh)
        return cls(width, height, resolution, origin, occupied)

    def world_to_cell(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        mx = int(math.floor((float(x) - self.origin[0]) / self.resolution))
        my = int(math.floor((float(y) - self.origin[1]) / self.resolution))
        if 0 <= mx < self.width and 0 <= my < self.height:
            return mx, my
        return None

    def distance_at(self, x: float, y: float, outside_penalty_m: float) -> float:
        cell = self.world_to_cell(x, y)
        if cell is None:
            return outside_penalty_m
        mx, my = cell
        image_row = self.height - 1 - my
        return self.distance_m[image_row * self.width + mx]

    def occupancy_for_display(self) -> List[int]:
        return [1 if value else 0 for value in self.occupied]

    def occupied_world_points(self) -> List[Tuple[float, float]]:
        points = []
        for index, is_occupied in enumerate(self.occupied):
            if not is_occupied:
                continue
            row = index // self.width
            col = index % self.width
            image_row = row
            map_y = self.height - 1 - image_row
            points.append(
                (
                    self.origin[0] + (col + 0.5) * self.resolution,
                    self.origin[1] + (map_y + 0.5) * self.resolution,
                )
            )
        return points

    def inner_wall_bounds(self) -> ArenaBounds:
        points = self.occupied_world_points()
        if not points:
            half_width = self.width * self.resolution * 0.5
            half_height = self.height * self.resolution * 0.5
            center_x = self.origin[0] + half_width
            center_y = self.origin[1] + half_height
            return ArenaBounds(
                center_x - half_width,
                center_x + half_width,
                center_y - half_height,
                center_y + half_height,
            )

        center_x = self.origin[0] + self.width * self.resolution * 0.5
        center_y = self.origin[1] + self.height * self.resolution * 0.5
        x_band = self.width * self.resolution * 0.25
        y_band = self.height * self.resolution * 0.25
        vertical_wall_points = [
            (x, y)
            for x, y in points
            if abs(y - center_y) <= y_band
        ]
        horizontal_wall_points = [
            (x, y)
            for x, y in points
            if abs(x - center_x) <= x_band
        ]
        left = [x for x, _y in vertical_wall_points if x < center_x]
        right = [x for x, _y in vertical_wall_points if x > center_x]
        bottom = [y for _x, y in horizontal_wall_points if y < center_y]
        top = [y for _x, y in horizontal_wall_points if y > center_y]
        if not left or not right or not bottom or not top:
            xs = [x for x, _y in points]
            ys = [y for _x, y in points]
            return ArenaBounds(min(xs), max(xs), min(ys), max(ys))
        return ArenaBounds(max(left), min(right), max(bottom), min(top))

    def _build_distance_field(self) -> List[float]:
        total = self.width * self.height
        distances = [math.inf] * total
        queue: list[tuple[float, int, int]] = []

        for index, is_occupied in enumerate(self.occupied):
            if is_occupied:
                row = index // self.width
                col = index % self.width
                distances[index] = 0.0
                heapq.heappush(queue, (0.0, col, row))

        neighbors = (
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        )
        while queue:
            distance_cells, col, row = heapq.heappop(queue)
            index = row * self.width + col
            if distance_cells > distances[index]:
                continue
            for dx, dy, step_cost in neighbors:
                next_col = col + dx
                next_row = row + dy
                if not (0 <= next_col < self.width and 0 <= next_row < self.height):
                    continue
                next_index = next_row * self.width + next_col
                next_distance = distance_cells + step_cost
                if next_distance < distances[next_index]:
                    distances[next_index] = next_distance
                    heapq.heappush(queue, (next_distance, next_col, next_row))

        max_distance = math.hypot(self.width, self.height) * self.resolution
        return [
            (value * self.resolution) if math.isfinite(value) else max_distance
            for value in distances
        ]


class KnownMapLocalizer:
    """Small-window correlative LiDAR matcher against a known occupancy map."""

    def __init__(
        self,
        occupancy_map: OccupancyMap,
        max_beams: int = 72,
        search_xy_m: float = 0.10,
        search_yaw_rad: float = 0.14,
        xy_step_m: float = 0.02,
        yaw_step_rad: float = 0.035,
        max_score_distance_m: float = 0.35,
        prior_penalty: float = 0.05,
    ):
        self.map = occupancy_map
        self.max_beams = max(8, int(max_beams))
        self.search_xy_m = max(0.0, float(search_xy_m))
        self.search_yaw_rad = max(0.0, float(search_yaw_rad))
        self.xy_step_m = max(0.005, float(xy_step_m))
        self.yaw_step_rad = max(0.005, float(yaw_step_rad))
        self.max_score_distance_m = max(0.02, float(max_score_distance_m))
        self.prior_penalty = max(0.0, float(prior_penalty))

    def scan_to_points(
        self,
        ranges: Sequence[float],
        angle_min: float,
        angle_increment: float,
        range_min: float,
        range_max: float,
        yaw_offset: float = 0.0,
    ) -> List[Tuple[float, float]]:
        valid: list[tuple[float, float]] = []
        upper_range = min(float(range_max), 4.0)
        for index, value in enumerate(ranges):
            distance = float(value)
            if not math.isfinite(distance):
                continue
            if distance < range_min or distance > upper_range:
                continue
            angle = (
                float(angle_min)
                + float(index) * float(angle_increment)
                + float(yaw_offset)
            )
            valid.append((distance * math.cos(angle), distance * math.sin(angle)))

        if len(valid) <= self.max_beams:
            return valid

        step = float(len(valid) - 1) / float(self.max_beams - 1)
        return [valid[int(round(i * step))] for i in range(self.max_beams)]

    def match(
        self,
        points: Sequence[Tuple[float, float]],
        prior_pose: Pose2D,
    ) -> MatchResult:
        started = time.perf_counter()
        if not points:
            return MatchResult(
                pose=prior_pose,
                score=math.inf,
                latency_ms=0.0,
                beam_count=0,
                candidate_count=0,
                success=False,
                reason="no_valid_lidar_points",
            )

        best_pose = prior_pose
        best_score = math.inf
        candidate_count = 0
        xy_offsets = _centered_offsets(self.search_xy_m, self.xy_step_m)
        yaw_offsets = _centered_offsets(self.search_yaw_rad, self.yaw_step_rad)
        outside_penalty = self.max_score_distance_m * 2.0

        prior_yaw = float(prior_pose.yaw or 0.0)
        for dyaw in yaw_offsets:
            yaw = normalize_angle(prior_yaw + dyaw)
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            rotated = [
                (
                    px * cos_yaw - py * sin_yaw,
                    px * sin_yaw + py * cos_yaw,
                )
                for px, py in points
            ]
            for dx in xy_offsets:
                x = prior_pose.x + dx
                for dy in xy_offsets:
                    y = prior_pose.y + dy
                    candidate_count += 1
                    score = 0.0
                    for rx, ry in rotated:
                        distance = self.map.distance_at(
                            x + rx,
                            y + ry,
                            outside_penalty,
                        )
                        score += min(distance, self.max_score_distance_m)
                    score /= float(len(rotated))
                    score += self.prior_penalty * (
                        abs(dx) / max(self.xy_step_m, self.search_xy_m, 1e-6)
                        + abs(dy) / max(self.xy_step_m, self.search_xy_m, 1e-6)
                        + abs(dyaw) / max(self.yaw_step_rad, self.search_yaw_rad, 1e-6)
                    )
                    if score < best_score:
                        best_score = score
                        best_pose = Pose2D(x=x, y=y, yaw=yaw)

        latency_ms = (time.perf_counter() - started) * 1000.0
        return MatchResult(
            pose=best_pose,
            score=best_score,
            latency_ms=latency_ms,
            beam_count=len(points),
            candidate_count=candidate_count,
            success=math.isfinite(best_score),
            reason="known_map",
        )


class WallRangeLocalizer:
    """Fast localizer for a rectangular arena using expected wall ranges."""

    def __init__(
        self,
        occupancy_map: OccupancyMap,
        max_beams: int = 32,
        search_xy_m: float = 0.12,
        search_yaw_rad: float = 0.14,
        xy_step_m: float = 0.04,
        yaw_step_rad: float = 0.07,
        max_residual_m: float = 0.35,
        short_return_margin_m: float = 0.22,
        trim_fraction: float = 0.30,
        min_beams: int = 8,
        prior_penalty: float = 0.004,
    ):
        self.bounds = occupancy_map.inner_wall_bounds()
        self.max_beams = max(8, int(max_beams))
        self.search_xy_m = max(0.0, float(search_xy_m))
        self.search_yaw_rad = max(0.0, float(search_yaw_rad))
        self.xy_step_m = max(0.005, float(xy_step_m))
        self.yaw_step_rad = max(0.005, float(yaw_step_rad))
        self.max_residual_m = max(0.02, float(max_residual_m))
        self.short_return_margin_m = max(0.0, float(short_return_margin_m))
        self.trim_fraction = max(0.0, min(0.8, float(trim_fraction)))
        self.min_beams = max(4, int(min_beams))
        self.prior_penalty = max(0.0, float(prior_penalty))

    def match_scan(
        self,
        ranges: Sequence[float],
        angle_min: float,
        angle_increment: float,
        range_min: float,
        range_max: float,
        yaw_offset: float,
        prior_pose: Pose2D,
        yaw_locked: bool = False,
    ) -> MatchResult:
        """yaw_locked: trust the prior yaw completely (IMU feed-forward
        supplies it) and search ONLY x/y on the coarse grid - candidates
        drop ~10x and the match runs well under the real robot's 50 ms."""
        started = time.perf_counter()
        beams = self.wall_beams(
            ranges,
            angle_min,
            angle_increment,
            range_min,
            range_max,
            yaw_offset,
        )
        if len(beams) < self.min_beams:
            return MatchResult(
                pose=prior_pose,
                score=math.inf,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                beam_count=len(beams),
                candidate_count=0,
                success=False,
                reason="wall_range:not_enough_beams",
            )

        prior_yaw = float(prior_pose.yaw or 0.0)
        xy_offsets = _centered_offsets(self.search_xy_m, self.xy_step_m)
        yaw_offsets = [0.0] if yaw_locked else _centered_offsets(self.search_yaw_rad, self.yaw_step_rad)
        best_pose, best_score, best_used, candidate_count = self.search_window(
            float(prior_pose.x),
            float(prior_pose.y),
            prior_yaw,
            xy_offsets,
            yaw_offsets,
            beams,
            prior_pose,
            prior_yaw,
        )

        if math.isfinite(best_score):
            refine_xy_step = self.xy_step_m * 0.5
            refine_yaw_step = self.yaw_step_rad * 0.5
            refine_xy_offsets = (
                [-refine_xy_step, 0.0, refine_xy_step]
                if refine_xy_step >= 0.005
                else [0.0]
            )
            refine_yaw_offsets = (
                [-refine_yaw_step, 0.0, refine_yaw_step]
                if refine_yaw_step >= 0.005 and not yaw_locked
                else [0.0]
            )
            refined_pose, refined_score, refined_used, refined_count = self.search_window(
                best_pose.x,
                best_pose.y,
                float(best_pose.yaw or 0.0),
                refine_xy_offsets,
                refine_yaw_offsets,
                beams,
                prior_pose,
                prior_yaw,
            )
            candidate_count += refined_count
            if refined_score < best_score:
                best_pose = refined_pose
                best_score = refined_score
                best_used = refined_used

        latency_ms = (time.perf_counter() - started) * 1000.0
        return MatchResult(
            pose=best_pose,
            score=best_score,
            latency_ms=latency_ms,
            beam_count=len(beams),
            candidate_count=candidate_count,
            success=math.isfinite(best_score),
            reason=f"wall_range:used={best_used},bounds={self.bounds}",
        )

    def match_scan_global(
        self,
        ranges: Sequence[float],
        angle_min: float,
        angle_increment: float,
        range_min: float,
        range_max: float,
        yaw_offset: float,
        prior_pose: Pose2D,
        xy_step_m: float = 0.32,
        yaw_step_rad: float = 0.35,
        max_beams: int = 16,
        refine_max_beams: Optional[int] = None,
        yaw_fixed: Optional[float] = None,
    ) -> MatchResult:
        started = time.perf_counter()
        beams = self.wall_beams(
            ranges,
            angle_min,
            angle_increment,
            range_min,
            range_max,
            yaw_offset,
        )
        if len(beams) < self.min_beams:
            return MatchResult(
                pose=prior_pose,
                score=math.inf,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                beam_count=len(beams),
                candidate_count=0,
                success=False,
                reason="wall_range_global:not_enough_beams",
            )

        coarse_beams = _downsample_sequence(beams, max(self.min_beams, int(max_beams)))
        coarse_pose, coarse_score, _coarse_used, candidate_count = self.search_global_grid(
            coarse_beams,
            max(0.05, float(xy_step_m)),
            max(0.05, float(yaw_step_rad)),
            yaw_fixed=yaw_fixed,
        )

        best_pose = coarse_pose
        best_score = coarse_score
        best_used = 0
        refine_beams = coarse_beams
        if math.isfinite(coarse_score):
            if refine_max_beams is None:
                refine_limit = min(self.max_beams, max(len(coarse_beams), int(max_beams) * 2))
            else:
                refine_limit = max(self.min_beams, int(refine_max_beams))
            refine_beams = _downsample_sequence(beams, refine_limit)
            refine_xy_step = max(self.xy_step_m, float(xy_step_m) * 0.25)
            refine_yaw_step = max(self.yaw_step_rad, float(yaw_step_rad) * 0.25)
            refine_xy_offsets = _centered_offsets(float(xy_step_m), refine_xy_step)
            refine_yaw_offsets = (
                [0.0]
                if yaw_fixed is not None
                else _centered_offsets(float(yaw_step_rad), refine_yaw_step)
            )
            best_pose, best_score, best_used, refined_count = self.search_window(
                coarse_pose.x,
                coarse_pose.y,
                float(coarse_pose.yaw or 0.0),
                refine_xy_offsets,
                refine_yaw_offsets,
                refine_beams,
                prior_pose,
                float(prior_pose.yaw or 0.0),
                prior_penalty=0.0,
            )
            candidate_count += refined_count

            # [2026-07-22 미세 refine] 종전에는 여기서 끝나 출력 x/y 가
            # refine 격자(0.3*0.25 = 7.5cm) 위로 양자화됐다. 실기 02시 런에서
            # 이 계단이 폐루프 주행에 그대로 보였다: 오차가 7.5cm 쌓여야
            # 첫 보정이 나가고, 동결->버스트 갱신이 점프 감지 오판을 만들었다.
            # 최적점 주변 3x3 halving 2단(스텝 0.0375 -> 0.01875, yaw 고정)으로
            # 최종 해상도를 ~1.9cm 로 낮춘다. 비용: +18 후보 x refine_beams
            # (전체 후보 250 -> 268, 약 +11% -> 매칭 지연 +2~3ms 수준).
            fine_step = refine_xy_step * 0.5
            while fine_step >= 0.015 and math.isfinite(best_score):
                fine_pose, fine_score, fine_used, fine_count = self.search_window(
                    best_pose.x,
                    best_pose.y,
                    float(best_pose.yaw or 0.0),
                    [-fine_step, 0.0, fine_step],
                    [0.0],
                    refine_beams,
                    prior_pose,
                    float(prior_pose.yaw or 0.0),
                    prior_penalty=0.0,
                )
                candidate_count += fine_count
                if fine_score < best_score:
                    best_pose, best_score, best_used = fine_pose, fine_score, fine_used
                fine_step *= 0.5

        latency_ms = (time.perf_counter() - started) * 1000.0
        return MatchResult(
            pose=best_pose,
            score=best_score,
            latency_ms=latency_ms,
            beam_count=len(coarse_beams),
            candidate_count=candidate_count,
            success=math.isfinite(best_score),
            reason=(
                f"wall_range_global:used={best_used},"
                f"coarse_beams={len(coarse_beams)},"
                f"refine_beams={len(refine_beams)},bounds={self.bounds}"
            ),
        )

    def match_scan_global_coarse(
        self,
        ranges: Sequence[float],
        angle_min: float,
        angle_increment: float,
        range_min: float,
        range_max: float,
        yaw_offset: float,
        prior_pose: Pose2D,
        xy_step_m: float = 0.42,
        yaw_step_rad: float = 0.50,
        max_beams: int = 12,
    ) -> MatchResult:
        started = time.perf_counter()
        beams = self.wall_beams(
            ranges,
            angle_min,
            angle_increment,
            range_min,
            range_max,
            yaw_offset,
        )
        if len(beams) < self.min_beams:
            return MatchResult(
                pose=prior_pose,
                score=math.inf,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                beam_count=len(beams),
                candidate_count=0,
                success=False,
                reason="wall_range_global_coarse:not_enough_beams",
            )

        coarse_beams = _downsample_sequence(beams, max(self.min_beams, int(max_beams)))
        best_pose, best_score, best_used, candidate_count = self.search_global_grid(
            coarse_beams,
            max(0.05, float(xy_step_m)),
            max(0.05, float(yaw_step_rad)),
        )

        latency_ms = (time.perf_counter() - started) * 1000.0
        return MatchResult(
            pose=best_pose,
            score=best_score,
            latency_ms=latency_ms,
            beam_count=len(coarse_beams),
            candidate_count=candidate_count,
            success=math.isfinite(best_score),
            reason=(
                f"wall_range_global_coarse:used={best_used},"
                f"coarse_beams={len(coarse_beams)},bounds={self.bounds}"
            ),
        )

    def search_global_grid(
        self,
        beams: Sequence[WallBeam],
        xy_step_m: float,
        yaw_step_rad: float,
        yaw_fixed: Optional[float] = None,
    ) -> Tuple[Pose2D, float, int, int]:
        best_pose = Pose2D(
            (self.bounds.xmin + self.bounds.xmax) * 0.5,
            (self.bounds.ymin + self.bounds.ymax) * 0.5,
            0.0,
        )
        best_score = math.inf
        best_used = 0
        candidate_count = 0
        margin = max(0.04, self.xy_step_m)
        x_values = _range_values(self.bounds.xmin + margin, self.bounds.xmax - margin, xy_step_m)
        y_values = _range_values(self.bounds.ymin + margin, self.bounds.ymax - margin, xy_step_m)
        # With the yaw known (IMU feed-forward) the square-arena solution is
        # UNIQUE - a single-yaw global grid replaces the whole yaw sweep.
        if yaw_fixed is not None:
            yaw_values = [yaw_fixed]
        else:
            yaw_values = _range_values(-math.pi, math.pi, yaw_step_rad, include_stop=False)

        for yaw in yaw_values:
            yaw = normalize_angle(yaw)
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            for x in x_values:
                for y in y_values:
                    candidate_count += 1
                    score, used = self.score_candidate(x, y, cos_yaw, sin_yaw, beams)
                    if used < self.min_beams:
                        continue
                    if score < best_score:
                        best_score = score
                        best_used = used
                        best_pose = Pose2D(x=x, y=y, yaw=yaw)
        return best_pose, best_score, best_used, candidate_count

    def search_window(
        self,
        center_x: float,
        center_y: float,
        center_yaw: float,
        xy_offsets: Sequence[float],
        yaw_offsets: Sequence[float],
        beams: Sequence[WallBeam],
        prior_pose: Pose2D,
        prior_yaw: float,
        prior_penalty: Optional[float] = None,
    ) -> Tuple[Pose2D, float, int, int]:
        best_pose = Pose2D(center_x, center_y, center_yaw)
        best_score = math.inf
        best_used = 0
        candidate_count = 0
        prior_xy_norm = max(self.xy_step_m, self.search_xy_m, 1e-6)
        prior_yaw_norm = max(self.yaw_step_rad, self.search_yaw_rad, 1e-6)

        for dyaw in yaw_offsets:
            yaw = normalize_angle(center_yaw + dyaw)
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            for dx in xy_offsets:
                x = center_x + dx
                if x <= self.bounds.xmin or x >= self.bounds.xmax:
                    continue
                for dy in xy_offsets:
                    y = center_y + dy
                    if y <= self.bounds.ymin or y >= self.bounds.ymax:
                        continue
                    candidate_count += 1
                    score, used = self.score_candidate(x, y, cos_yaw, sin_yaw, beams)
                    if used < self.min_beams:
                        continue
                    score += (self.prior_penalty if prior_penalty is None else prior_penalty) * (
                        abs(x - prior_pose.x) / prior_xy_norm
                        + abs(y - prior_pose.y) / prior_xy_norm
                        + abs(normalize_angle(yaw - prior_yaw)) / prior_yaw_norm
                    )
                    if score < best_score:
                        best_score = score
                        best_used = used
                        best_pose = Pose2D(x=x, y=y, yaw=yaw)
        return best_pose, best_score, best_used, candidate_count

    def wall_beams(
        self,
        ranges: Sequence[float],
        angle_min: float,
        angle_increment: float,
        range_min: float,
        range_max: float,
        yaw_offset: float,
    ) -> List[WallBeam]:
        beams = []
        for index, value in enumerate(ranges):
            distance = float(value)
            if not math.isfinite(distance):
                continue
            if distance < range_min or distance > range_max:
                continue
            angle = (
                float(angle_min)
                + float(index) * float(angle_increment)
                + float(yaw_offset)
            )
            angle = normalize_angle(angle)
            beams.append(WallBeam(angle, distance, math.cos(angle), math.sin(angle)))

        if len(beams) <= self.max_beams:
            return beams
        step = float(len(beams) - 1) / float(self.max_beams - 1)
        return [beams[int(round(i * step))] for i in range(self.max_beams)]

    def score_candidate(
        self,
        x: float,
        y: float,
        cos_yaw: float,
        sin_yaw: float,
        beams: Sequence[WallBeam],
    ) -> Tuple[float, int]:
        residuals = []
        for beam in beams:
            cos_theta = cos_yaw * beam.cos_angle - sin_yaw * beam.sin_angle
            sin_theta = sin_yaw * beam.cos_angle + cos_yaw * beam.sin_angle
            expected = self.expected_wall_range_from_unit(x, y, cos_theta, sin_theta)
            if expected is None:
                continue
            residual = beam.distance - expected
            if residual < -self.short_return_margin_m and not getattr(
                self, "penalize_short_returns", False
            ):
                # Legacy behaviour: treat short returns as occlusions and
                # skip them. With the scan plane ABOVE the tallest object
                # (this arena: 0.32 m vs 8 cm) there are no occlusions, and
                # skipping makes far-off candidates unbeatable in a GLOBAL
                # search (they drop every disagreeing beam and score ~0 on
                # the agreeing axis alone) - set penalize_short_returns.
                continue
            residuals.append(min(abs(residual), self.max_residual_m))

        if len(residuals) < self.min_beams:
            return math.inf, len(residuals)
        residuals.sort()
        keep_count = max(self.min_beams, int(round(len(residuals) * (1.0 - self.trim_fraction))))
        kept = residuals[:keep_count]
        return sum(kept) / float(len(kept)), len(kept)

    def expected_wall_range(self, x: float, y: float, theta: float) -> Optional[float]:
        return self.expected_wall_range_from_unit(x, y, math.cos(theta), math.sin(theta))

    def expected_wall_range_from_unit(
        self,
        x: float,
        y: float,
        cos_theta: float,
        sin_theta: float,
    ) -> Optional[float]:
        best = math.inf
        eps = 1e-9
        if cos_theta > eps:
            distance = (self.bounds.xmax - x) / cos_theta
            if distance > 0.0:
                best = min(best, distance)
        elif cos_theta < -eps:
            distance = (self.bounds.xmin - x) / cos_theta
            if distance > 0.0:
                best = min(best, distance)
        if sin_theta > eps:
            distance = (self.bounds.ymax - y) / sin_theta
            if distance > 0.0:
                best = min(best, distance)
        elif sin_theta < -eps:
            distance = (self.bounds.ymin - y) / sin_theta
            if distance > 0.0:
                best = min(best, distance)
        if not math.isfinite(best):
            return None
        return best


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    return angle


def scan_to_world_points(
    ranges: Sequence[float],
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    yaw_offset: float,
    pose: Pose2D,
    max_points: int,
) -> Tuple[List[Tuple[float, float]], int]:
    upper_range = float(range_max)
    pose_yaw = float(pose.yaw or 0.0)
    cos_pose = math.cos(pose_yaw)
    sin_pose = math.sin(pose_yaw)

    valid_count = 0
    for value in ranges:
        distance = float(value)
        if not math.isfinite(distance):
            continue
        if distance < range_min or distance > upper_range:
            continue
        valid_count += 1

    if valid_count == 0:
        return [], 0

    max_points = max(0, int(max_points))
    if max_points == 1:
        target_indexes = {valid_count // 2}
    elif max_points == 0 or valid_count <= max_points:
        target_indexes = set(range(valid_count))
    else:
        step = float(valid_count - 1) / float(max_points - 1)
        target_indexes = {int(round(i * step)) for i in range(max_points)}

    points: list[tuple[float, float]] = []
    valid_index = 0
    pose_x = float(pose.x)
    pose_y = float(pose.y)
    for index, value in enumerate(ranges):
        distance = float(value)
        if not math.isfinite(distance):
            continue
        if distance < range_min or distance > upper_range:
            continue
        if valid_index in target_indexes:
            angle = (
                float(angle_min)
                + float(index) * float(angle_increment)
                + float(yaw_offset)
            )
            rx = distance * math.cos(angle)
            ry = distance * math.sin(angle)
            points.append(
                (
                    pose_x + rx * cos_pose - ry * sin_pose,
                    pose_y + rx * sin_pose + ry * cos_pose,
                )
            )
        valid_index += 1

    return points, valid_count


def _centered_offsets(radius: float, step: float) -> List[float]:
    if radius <= 1e-9:
        return [0.0]
    count = int(math.floor(radius / step))
    offsets = [0.0]
    for index in range(1, count + 1):
        value = index * step
        offsets.extend((-value, value))
    return sorted(offsets)


def _downsample_sequence(values: Sequence, max_count: int) -> List:
    max_count = max(0, int(max_count))
    if max_count == 0:
        return []
    if len(values) <= max_count:
        return list(values)
    if max_count == 1:
        return [values[len(values) // 2]]

    step = float(len(values) - 1) / float(max_count - 1)
    return [values[int(round(index * step))] for index in range(max_count)]


def _range_values(
    start: float,
    stop: float,
    step: float,
    include_stop: bool = True,
) -> List[float]:
    start = float(start)
    stop = float(stop)
    step = max(1e-6, float(step))
    if stop < start:
        return []

    values: list[float] = []
    value = start
    epsilon = step * 0.1
    while True:
        if include_stop:
            if value > stop + epsilon:
                break
        elif value >= stop:
            break
        if value <= stop or include_stop:
            values.append(min(value, stop))
        value += step

    if include_stop and (not values or abs(values[-1] - stop) > epsilon):
        values.append(stop)
    return values


def _read_simple_map_yaml(path: Path) -> dict:
    data = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith("["):
            data[key] = ast.literal_eval(value)
        elif key in {"resolution", "occupied_thresh", "free_thresh"}:
            data[key] = float(value)
        elif key == "negate":
            data[key] = int(value)
        else:
            data[key] = value.strip("'\"")
    return data


def _read_pgm(path: Path) -> Tuple[int, int, bytes]:
    with path.open("rb") as stream:
        magic = _read_pgm_token(stream)
        if magic not in {b"P5", b"P2"}:
            raise ValueError(f"unsupported PGM magic {magic!r}")
        width = int(_read_pgm_token(stream))
        height = int(_read_pgm_token(stream))
        max_value = int(_read_pgm_token(stream))
        if max_value <= 0 or max_value > 255:
            raise ValueError("only 8-bit PGM maps are supported")
        if magic == b"P5":
            pixels = stream.read(width * height)
            if len(pixels) != width * height:
                raise ValueError("PGM pixel data is truncated")
            return width, height, pixels
        values = [
            int(_read_pgm_token(stream))
            for _ in range(width * height)
        ]
        return width, height, bytes(values)


def _read_pgm_token(stream) -> bytes:
    token = bytearray()
    while True:
        char = stream.read(1)
        if not char:
            return bytes(token)
        if char == b"#":
            stream.readline()
            continue
        if char.isspace():
            if token:
                return bytes(token)
            continue
        token.extend(char)


def min_front_range(
    ranges: Sequence[float],
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    sector_rad: float,
    yaw_offset: float = 0.0,
) -> Optional[float]:
    return min_range_in_sector(
        ranges,
        angle_min,
        angle_increment,
        range_min,
        range_max,
        sector_rad,
        sector_center_rad=0.0,
        yaw_offset=yaw_offset,
    )


def min_range_in_sector(
    ranges: Sequence[float],
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
    sector_rad: float,
    sector_center_rad: float = 0.0,
    yaw_offset: float = 0.0,
) -> Optional[float]:
    best = math.inf
    half_sector = max(0.0, float(sector_rad)) * 0.5
    for index, value in enumerate(ranges):
        distance = float(value)
        if not math.isfinite(distance) or distance < range_min or distance > range_max:
            continue
        angle = normalize_angle(
            float(angle_min)
            + float(index) * float(angle_increment)
            + float(yaw_offset)
            - float(sector_center_rad)
        )
        if abs(angle) <= half_sector:
            best = min(best, distance)
    return best if math.isfinite(best) else None
