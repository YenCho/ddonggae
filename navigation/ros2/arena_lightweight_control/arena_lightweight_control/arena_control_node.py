#!/usr/bin/env python3
import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Optional

import rclpy
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Image, Imu, LaserScan
from std_msgs.msg import String

from arena_lightweight_control.controllers import (
    DiffDriveController,
    GoalControllerConfig,
    MecanumController,
)
from arena_lightweight_control.internal_perception import InternalYoloDetector
from arena_lightweight_control.map_localization import (
    KnownMapLocalizer,
    MatchResult,
    OccupancyMap,
    Pose2D,
    WallRangeLocalizer,
    min_front_range,
    min_range_in_sector,
)
from arena_lightweight_control.web_ui import ArenaWebServer


class ArenaControlNode(Node):
    def __init__(self):
        super().__init__("arena_control_node")

        self.declare_parameter("map_yaml", default_map_yaml())
        self.declare_parameter("scan_topic", "/laser_scan")
        self.declare_parameter("scan_yaw_offset_rad", 0.0)
        self.declare_parameter("cmd_vel_topic", "/cmd_vel_direct")
        self.declare_parameter("gripper_command_topic", "/gripper/command")
        self.declare_parameter("gripper_status_topic", "/gripper/state")
        self.declare_parameter("motor_state_topic", "/motor/state")
        self.declare_parameter("debug_odom_topic", "/odom/wheel")
        self.declare_parameter("debug_imu_topic", "/imu/data")
        self.declare_parameter("status_topic", "/arena_lightweight/status")
        self.declare_parameter("image_topic", "/camera_54/rgb")
        self.declare_parameter("objects_topic", "/detected_objects/map_objects")
        self.declare_parameter("enable_object_avoidance", False)
        self.declare_parameter("object_avoidance_max_age_sec", 0.75)
        self.declare_parameter("object_keepout_half_width_m", 0.15)
        self.declare_parameter("object_keepout_half_depth_m", 0.20)
        self.declare_parameter("object_avoidance_lookahead_m", 0.35)
        self.declare_parameter("object_avoidance_turn_rps", 0.35)
        self.declare_parameter("enable_internal_image_processing", False)
        self.declare_parameter("detector_model_path", "")
        self.declare_parameter("detector_confidence", 0.25)
        self.declare_parameter("detector_imgsz", 416)
        self.declare_parameter("detector_device", "0")
        self.declare_parameter("detector_min_period_sec", 0.20)
        self.declare_parameter("ui_host", "0.0.0.0")
        self.declare_parameter("ui_port", 18765)
        self.declare_parameter("enable_web_ui", False)
        self.declare_parameter("goal_command_topic", "/arena_lightweight/goal")
        self.declare_parameter("pose_command_topic", "/arena_lightweight/pose")
        self.declare_parameter("control_command_topic", "/arena_lightweight/control")
        # 2026-07-19: 구 "항상 북쪽(yaw+90) 정착" 전략 기본값 제거 — NaN이면
        # yaw 없는 goal은 final-yaw 회전 없이 현재 heading으로 정지한다
        # (resolve_goal_yaw가 비유한값을 None으로 해석). 북향이 필요한 호출자는
        # goal에 yaw를 명시하거나 이 파라미터를 직접 설정할 것.
        self.declare_parameter("default_goal_yaw_rad", float("nan"))
        self.declare_parameter("control_rate_hz", 20.0)
        # goal 종료 후 zero-twist를 계속 발행하는 유예 시간(s). 이후 idle 침묵.
        self.declare_parameter("idle_cmd_publish_sec", 0.6)
        self.declare_parameter("odom_yaw_gate_rad", 0.6)
        # IMU-yaw prior (preferred over wheel odom when available): gyro-z
        # integration is slip-free on mecanum wheels and drifts only a few
        # deg/min, so the true yaw is ALWAYS within +-45 deg of the prior -
        # the gate then deterministically resolves the square-arena 90-deg
        # symmetry and flips become impossible.
        self.declare_parameter("imu_topic", "")
        # body-up(yaw) 축의 IMU frame 단위벡터: wz_body = axis · gyro.
        # sim은 body-frame IMU라 [0,0,1]. 실기 D435I(바텀, 54도 틸트)는
        # 회전이 y/z에 나뉘어 음수로 실림 → [0, -0.586, -0.810] (2026-07-15 실측,
        # cos54/sin54와 일치). 축이 틀리면 yaw prior가 반대로 적분돼
        # 대칭 플립을 오히려 조장한다.
        self.declare_parameter("imu_yaw_axis", [0.0, 0.0, 1.0])
        self.declare_parameter("imu_yaw_gate_rad", math.pi / 4.0)
        self.declare_parameter("status_period_sec", 0.25)
        # [2026-07-21] 스캔매칭 최소 주기 [s]. scan_callback 주석 참조.
        # 0.05 = 20Hz 상한 (실기 라이다 13.1Hz 라 실질 무제한, 시뮬 폭주만 차단).
        self.declare_parameter("scan_min_period_sec", 0.05)
        # START 코너(우하단)·북향에서 기동 — wall_range는 정사각 대칭이라 시작점
        # prior가 틀리면 다른 코너 해로 수렴한다.
        # [demo/arena-2m] 경기 1.8 / -1.8. 런치가 덮어쓰지만, 노드를
        # 단독 기동할 때의 기본값도 같은 아레나를 가리켜야 한다.
        self.declare_parameter("initial_pose_x", 0.8)
        self.declare_parameter("initial_pose_y", -0.8)
        self.declare_parameter("initial_pose_yaw", 1.5708)
        self.declare_parameter("localization_max_beams", 24)
        self.declare_parameter("localization_search_xy_m", 0.12)
        self.declare_parameter("localization_search_yaw_rad", 0.14)
        self.declare_parameter("localization_xy_step_m", 0.04)
        self.declare_parameter("localization_yaw_step_rad", 0.07)
        self.declare_parameter("localization_mode", "wall_range")
        self.declare_parameter("wall_localization_max_score", 0.22)
        self.declare_parameter("localization_reseed_period_sec", 2.5)
        self.declare_parameter("localization_reseed_xy_step_m", 0.32)
        self.declare_parameter("localization_reseed_yaw_step_rad", 0.35)
        self.declare_parameter("localization_reseed_max_beams", 16)
        self.declare_parameter("localization_reseed_refine_max_beams", 16)
        self.declare_parameter("localization_reseed_score_margin", 0.035)
        self.declare_parameter("localization_warn_latency_ms", 50.0)
        self.declare_parameter("scan_timeout_sec", 0.5)
        self.declare_parameter("obstacle_sector_deg", 55.0)
        self.declare_parameter("obstacle_stop_distance_m", 0.22)
        self.declare_parameter("controller_type", "diff_drive")
        self.declare_parameter("xy_tolerance_m", 0.15)
        self.declare_parameter("yaw_tolerance_rad", 0.40)
        self.declare_parameter("goal_position_latch", True)
        self.declare_parameter("xy_release_tolerance_m", 0.25)
        self.declare_parameter("xy_release_consecutive_count", 3)
        self.declare_parameter("near_goal_slow_radius_m", 0.10)
        self.declare_parameter("near_goal_speed_scale", 0.5)
        self.declare_parameter("rotate_to_goal_threshold_rad", 0.35)
        self.declare_parameter("max_linear_mps", 0.36)
        self.declare_parameter("min_linear_mps", 0.075)
        self.declare_parameter("max_angular_rps", 0.85)
        self.declare_parameter("final_yaw_min_angular_rps", 0.0)
        self.declare_parameter("linear_gain", 1.425)
        self.declare_parameter("angular_gain", 1.7)
        self.declare_parameter("drive_angular_deadband_rad", 0.08)
        self.declare_parameter("drive_angular_gain_scale", 0.45)

        self.lock = threading.Lock()
        self.map = OccupancyMap.from_yaml(str(self.get_parameter("map_yaml").value))
        # 이 노드가 믿는 아레나 크기를 status 에 실어 보낸다. 러너가 시딩 직후
        # 자기 상수(fl.ARENA_HALF_M)와 대조해서 어긋나면 주행 전에 죽는다 —
        # 맵/상수 불일치는 포즈가 조용히 최대 1m 틀리고 레그1에서 벽으로 가는
        # 실패라, 주행 후에 알아채면 늦다.
        # inner_wall_bounds() 는 맵 전체를 훑으므로 20Hz status 경로에서 매번
        # 부르면 안 된다. 맵은 런타임에 안 바뀌니 여기서 한 번만 만든다.
        self.arena_info = self._arena_info()
        self.pose = Pose2D(
            float(self.get_parameter("initial_pose_x").value),
            float(self.get_parameter("initial_pose_y").value),
            float(self.get_parameter("initial_pose_yaw").value),
        )
        self.goal: Optional[Pose2D] = None
        self.goal_position_latched = False
        self.goal_position_release_count = 0
        self.last_match: Optional[MatchResult] = None
        self.pose_revision = 0
        self.last_localization_apply_started = 0.0
        self.last_localization_reseed_time = 0.0
        self.last_scan_time = 0.0
        self.last_scan_stamp_age_ms = None
        self.obstacle_front_m = None
        self.scan_sector_ranges = {}
        self.command_phase = "idle"
        self.last_command = Twist()
        self.last_gripper_status = ""
        self.last_motor_state = ""
        self.last_debug_odom = {}
        self.last_debug_imu = {}
        # Odom-yaw prior gate: wheel odometry yaw drifts very slowly, so a
        # scan-match whose yaw jumps away from the odom-propagated yaw is a
        # symmetric-arena flip and must be rejected.
        self.odom_yaw: Optional[float] = None
        self.odom_yaw_at_last_apply: Optional[float] = None
        self.yaw_gate_reject_count = 0
        # IMU gyro-z integration (unwrapped); absolute offset is irrelevant,
        # the gate only uses deltas between scan applications.
        self.imu_yaw: Optional[float] = None
        self.imu_yaw_at_last_apply: Optional[float] = None
        self.imu_last_stamp: Optional[float] = None
        self.imu_last_rx = 0.0
        self.latest_objects: list[dict] = []
        self.latest_objects_time = 0.0
        self.object_avoidance_block = None
        self.pending_gripper_commands: list[str] = []
        self.last_command_log_time = 0.0
        self.last_command_log_key = ""
        self.localization_warn_latency_ms = float(
            self.get_parameter("localization_warn_latency_ms").value
        )
        self.odom_yaw_gate_rad = float(self.get_parameter("odom_yaw_gate_rad").value)
        self.imu_yaw_gate_rad = float(self.get_parameter("imu_yaw_gate_rad").value)
        self.scan_timeout_sec = float(self.get_parameter("scan_timeout_sec").value)
        self.scan_yaw_offset_rad = float(
            self.get_parameter("scan_yaw_offset_rad").value
        )
        self.default_goal_yaw_rad = float(
            self.get_parameter("default_goal_yaw_rad").value
        )
        self.obstacle_sector_rad = math.radians(
            float(self.get_parameter("obstacle_sector_deg").value)
        )
        self.object_avoidance_enabled = bool(
            self.get_parameter("enable_object_avoidance").value
        )
        self.object_avoidance_max_age_sec = float(
            self.get_parameter("object_avoidance_max_age_sec").value
        )
        self.object_keepout_half_width_m = float(
            self.get_parameter("object_keepout_half_width_m").value
        )
        self.object_keepout_half_depth_m = float(
            self.get_parameter("object_keepout_half_depth_m").value
        )
        self.object_avoidance_lookahead_m = float(
            self.get_parameter("object_avoidance_lookahead_m").value
        )
        self.object_avoidance_turn_rps = float(
            self.get_parameter("object_avoidance_turn_rps").value
        )
        self.xy_tolerance_m = float(self.get_parameter("xy_tolerance_m").value)
        self.yaw_tolerance_rad = float(self.get_parameter("yaw_tolerance_rad").value)
        self.final_yaw_min_angular_rps = float(
            self.get_parameter("final_yaw_min_angular_rps").value
        )
        self.goal_position_latch_enabled = bool(
            self.get_parameter("goal_position_latch").value
        )
        self.xy_release_tolerance_m = max(
            self.xy_tolerance_m,
            float(self.get_parameter("xy_release_tolerance_m").value),
        )
        self.xy_release_consecutive_count = max(
            1,
            int(self.get_parameter("xy_release_consecutive_count").value),
        )

        self.localizer = KnownMapLocalizer(
            self.map,
            max_beams=int(self.get_parameter("localization_max_beams").value),
            search_xy_m=float(self.get_parameter("localization_search_xy_m").value),
            search_yaw_rad=float(self.get_parameter("localization_search_yaw_rad").value),
            xy_step_m=float(self.get_parameter("localization_xy_step_m").value),
            yaw_step_rad=float(self.get_parameter("localization_yaw_step_rad").value),
        )
        self.wall_localizer = WallRangeLocalizer(
            self.map,
            max_beams=int(self.get_parameter("localization_max_beams").value),
            search_xy_m=float(self.get_parameter("localization_search_xy_m").value),
            search_yaw_rad=float(self.get_parameter("localization_search_yaw_rad").value),
            xy_step_m=float(self.get_parameter("localization_xy_step_m").value),
            yaw_step_rad=float(self.get_parameter("localization_yaw_step_rad").value),
        )
        # The scan plane clears the tallest object (0.32 m vs 8 cm rulebox),
        # so every return IS a wall: penalize short returns instead of
        # skipping them (skipping degenerates the global x/y search).
        self.wall_localizer.penalize_short_returns = True
        self.localization_mode = str(
            self.get_parameter("localization_mode").value
        ).strip().lower()
        self.wall_localization_max_score = float(
            self.get_parameter("wall_localization_max_score").value
        )
        self.localization_reseed_period_sec = float(
            self.get_parameter("localization_reseed_period_sec").value
        )
        self.localization_reseed_xy_step_m = float(
            self.get_parameter("localization_reseed_xy_step_m").value
        )
        self.localization_reseed_yaw_step_rad = float(
            self.get_parameter("localization_reseed_yaw_step_rad").value
        )
        self.localization_reseed_max_beams = int(
            self.get_parameter("localization_reseed_max_beams").value
        )
        self.localization_reseed_refine_max_beams = int(
            self.get_parameter("localization_reseed_refine_max_beams").value
        )
        self.localization_reseed_score_margin = float(
            self.get_parameter("localization_reseed_score_margin").value
        )
        controller_config = GoalControllerConfig(
            xy_tolerance_m=self.xy_tolerance_m,
            yaw_tolerance_rad=self.yaw_tolerance_rad,
            near_goal_slow_radius_m=float(
                self.get_parameter("near_goal_slow_radius_m").value
            ),
            near_goal_speed_scale=float(
                self.get_parameter("near_goal_speed_scale").value
            ),
            rotate_to_goal_threshold_rad=float(
                self.get_parameter("rotate_to_goal_threshold_rad").value
            ),
            obstacle_stop_distance_m=float(
                self.get_parameter("obstacle_stop_distance_m").value
            ),
            max_linear_mps=float(self.get_parameter("max_linear_mps").value),
            min_linear_mps=float(self.get_parameter("min_linear_mps").value),
            max_angular_rps=float(self.get_parameter("max_angular_rps").value),
            final_yaw_min_angular_rps=self.final_yaw_min_angular_rps,
            linear_gain=float(self.get_parameter("linear_gain").value),
            angular_gain=float(self.get_parameter("angular_gain").value),
            drive_angular_deadband_rad=float(
                self.get_parameter("drive_angular_deadband_rad").value
            ),
            drive_angular_gain_scale=float(
                self.get_parameter("drive_angular_gain_scale").value
            ),
        )
        controller_type = str(self.get_parameter("controller_type").value).strip().lower()
        if controller_type == "mecanum":
            self.controller = MecanumController(controller_config)
        else:
            self.controller = DiffDriveController(controller_config)
            controller_type = "diff_drive"
        self.controller_type = controller_type
        # 런타임 파라미터 반영: 컨트롤러 config는 __init__ 에서 한 번만
        # 만들어지므로 `ros2 param set` 이 주행에 아무 영향이 없었다
        # (7/20: e2e 속도 프로파일 safe/normal/fast 가 전부 무시되던 원인).
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.detector = InternalYoloDetector(
            str(self.get_parameter("detector_model_path").value),
            confidence=float(self.get_parameter("detector_confidence").value),
            imgsz=int(self.get_parameter("detector_imgsz").value),
            device=str(self.get_parameter("detector_device").value),
            min_period_sec=float(self.get_parameter("detector_min_period_sec").value),
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            str(self.get_parameter("cmd_vel_topic").value),
            10,
        )
        self.gripper_pub = self.create_publisher(
            String,
            str(self.get_parameter("gripper_command_topic").value),
            10,
        )
        self.status_pub = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("goal_command_topic").value),
            self.goal_command_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("pose_command_topic").value),
            self.pose_command_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("control_command_topic").value),
            self.control_command_callback,
            10,
        )
        # depth=1: when matching falls behind (CPU contention with the sim
        # renderer), stale scans must be DROPPED, not queued - a backlog of
        # 5 scans x 300 ms turned the pose estimate seconds-stale (fuse7).
        scan_qos = QoSProfile(
            depth=1,
            reliability=qos_profile_sensor_data.reliability,
            durability=qos_profile_sensor_data.durability,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(
            LaserScan,
            str(self.get_parameter("scan_topic").value),
            self.scan_callback,
            scan_qos,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("gripper_status_topic").value),
            self.gripper_status_callback,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("motor_state_topic").value),
            self.motor_state_callback,
            10,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("debug_odom_topic").value),
            self.debug_odom_callback,
            10,
        )
        self.create_subscription(
            Imu,
            str(self.get_parameter("debug_imu_topic").value),
            self.debug_imu_callback,
            10,
        )
        imu_topic = str(self.get_parameter("imu_topic").value).strip()
        axis = [float(v) for v in self.get_parameter("imu_yaw_axis").value]
        norm = math.sqrt(sum(v * v for v in axis)) or 1.0
        self.imu_yaw_axis = tuple(v / norm for v in axis)
        if imu_topic:
            # RealSense IMU는 BEST_EFFORT 발행 — RELIABLE 구독이면 QoS 비호환으로
            # 아무것도 못 받는다. BEST_EFFORT 구독은 RELIABLE(sim) 발행과도 호환.
            imu_qos = QoSProfile(
                depth=50,
                history=HistoryPolicy.KEEP_LAST,
                reliability=qos_profile_sensor_data.reliability,
                durability=qos_profile_sensor_data.durability,
            )
            self.create_subscription(
                Imu, imu_topic, self.imu_yaw_callback, imu_qos
            )
        if self.object_avoidance_enabled:
            self.create_subscription(
                String,
                str(self.get_parameter("objects_topic").value),
                self.objects_callback,
                10,
            )
        if bool(self.get_parameter("enable_internal_image_processing").value):
            self.create_subscription(
                Image,
                str(self.get_parameter("image_topic").value),
                self.image_callback,
                2,
            )

        rate_hz = max(2.0, float(self.get_parameter("control_rate_hz").value))
        self.idle_cmd_publish_sec = float(
            self.get_parameter("idle_cmd_publish_sec").value
        )
        self.control_timer = self.create_timer(1.0 / rate_hz, self.control_tick)
        self.scan_min_period_sec = max(
            0.0, float(self.get_parameter("scan_min_period_sec").value))
        status_period = max(0.05, float(self.get_parameter("status_period_sec").value))
        self.status_timer = self.create_timer(status_period, self.publish_status)

        self.web_server = None
        self.ui_url = ""
        if bool(self.get_parameter("enable_web_ui").value):
            self.web_server = ArenaWebServer(
                str(self.get_parameter("ui_host").value),
                int(self.get_parameter("ui_port").value),
                self.map_snapshot,
                self.state_snapshot,
                self.set_goal,
                self.set_pose,
                self.clear_goal,
                self.enqueue_gripper_command,
            )
            self.web_server.start()
            self.ui_url = self.web_server.local_url()
        self.get_logger().info(
            "arena lightweight control ready: "
            f"map={self.get_parameter('map_yaml').value}, "
            f"controller={self.controller_type}, "
            f"localization_mode={self.localization_mode}, "
            f"xy_tolerance={self.xy_tolerance_m:.3f}, "
            f"xy_release={self.xy_release_tolerance_m:.3f}, "
            f"yaw_tolerance={self.yaw_tolerance_rad:.3f}, "
            f"final_yaw_min_wz={self.final_yaw_min_angular_rps:.3f}, "
            f"scan_yaw_offset_rad={self.scan_yaw_offset_rad:.3f}, "
            f"object_avoidance={self.object_avoidance_enabled}, "
            f"object_keepout={self.object_keepout_half_width_m:.2f}x"
            f"{self.object_keepout_half_depth_m:.2f}, "
            f"web_ui={self.ui_url or 'disabled'}"
        )

    # 런타임에 다시 반영할 수 있는 컨트롤러 config 필드
    _LIVE_CONTROLLER_PARAMS = (
        "max_linear_mps",
        "min_linear_mps",
        "max_angular_rps",
        "linear_gain",
        "angular_gain",
        "near_goal_slow_radius_m",
        "near_goal_speed_scale",
        "rotate_to_goal_threshold_rad",
        "obstacle_stop_distance_m",
        "drive_angular_deadband_rad",
        "drive_angular_gain_scale",
    )

    def _on_set_parameters(self, params):
        """`ros2 param set` 을 살아 있는 컨트롤러 config 에 즉시 반영."""
        import dataclasses

        from rcl_interfaces.msg import SetParametersResult

        updates = {}
        for p in params:
            if p.name not in self._LIVE_CONTROLLER_PARAMS:
                continue
            try:
                updates[p.name] = float(p.value)
            except (TypeError, ValueError):
                return SetParametersResult(
                    successful=False, reason=f"{p.name} must be a float"
                )
        if updates:
            # GoalControllerConfig 는 frozen dataclass → 통째로 교체한다.
            self.controller.config = dataclasses.replace(
                self.controller.config, **updates
            )
            self.get_logger().info(
                "controller config updated (live): "
                + ", ".join(f"{k}={v:.3f}" for k, v in sorted(updates.items()))
            )
        return SetParametersResult(successful=True)

    def scan_callback(self, msg: LaserScan):
        started = time.monotonic()
        # Throttle: the sim's PhysX lidar publishes every render frame
        # (30-60 Hz) and back-to-back Python matching saturated this process
        # (GIL) - 6 ms of compute stretched to 300 ms and starved the
        # status/IMU callbacks. The cap protects against that.
        #
        # [2026-07-21] 하드코딩 0.08s -> 파라미터 scan_min_period_sec (기본 0.05).
        #   폐기 사유: 실기 RPLIDAR 실측이 **13.1Hz(주기 76ms)** 라, 0.08s 상한이
        #   76ms 만에 도착한 스캔을 그대로 버렸다 -> 실제 매칭이 5~6Hz 로 반토막
        #   (scan_age 중앙값 0.090s / 최대 0.194s 로 확인). 구 주석의 "~10Hz" 는
        #   실측과 달랐다. 0.05s 로 낮추면 13.1Hz 전량이 통과한다.
        #   ⚠ 비용: 매칭 지연 실측 27ms (전역해 후보 277개) x 13.1Hz = 약 35%/core.
        #     GIL 기아가 재발하면 이 파라미터를 0.08 로 되돌릴 것.
        if (started - getattr(self, "_last_scan_processed", 0.0)
                < getattr(self, "scan_min_period_sec", 0.05)):
            return
        self._last_scan_processed = started
        with self.lock:
            prior_pose = self.pose
            prior_revision = self.pose_revision
            imu_at_capture = self.imu_yaw
        result = self.match_scan(msg, prior_pose)
        front = min_front_range(
            msg.ranges,
            msg.angle_min,
            msg.angle_increment,
            msg.range_min,
            msg.range_max,
            self.obstacle_sector_rad,
            yaw_offset=self.scan_yaw_offset_rad,
        )
        sector_ranges = self.scan_sector_snapshot(
            msg.ranges,
            msg.angle_min,
            msg.angle_increment,
            msg.range_min,
            msg.range_max,
        )
        stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        age_ms = None
        if stamp_sec > 0.0:
            age_ms = max(0.0, (self.get_clock().now().nanoseconds * 1e-9 - stamp_sec) * 1000.0)
        with self.lock:
            can_apply_pose = (
                result.success
                and prior_revision == self.pose_revision
                and started >= self.last_localization_apply_started
            )
            # Yaw prior gate (see init): expected yaw = last applied yaw +
            # prior-source yaw delta since then. Reject flips beyond the
            # gate. IMU gyro integration is the PREFERRED prior when fresh:
            # it is slip-free (mecanum odom is not) and drifts only a few
            # deg/min, so with a +-45 deg gate the square-arena 90-deg flip
            # is impossible; no small escape hatch is needed (a large one
            # remains as a last-resort unlock).
            yaw_gated = False
            imu_fresh = (
                self.imu_yaw is not None
                and (time.monotonic() - self.imu_last_rx) < 0.5
            )
            if imu_fresh:
                # IMU FEED-FORWARD mode: imu_yaw_callback dead-reckons
                # pose.yaw continuously, so the prior captured at scan
                # receipt is already yaw-current and the expectation is
                # simply the prior itself. (Without feed-forward, fast
                # in-place rotations outran the matcher's ±0.14 rad local
                # window and the estimate walked onto a neighbouring
                # symmetric optimum - fuse2 residuals ~1 rad.)
                expected_yaw = prior_pose.yaw
                gate_rad, hatch_limit = self.imu_yaw_gate_rad, 60
                have_prior = True
            else:
                have_prior = (
                    self.odom_yaw is not None and self.odom_yaw_at_last_apply is not None
                )
                odom_dyaw = 0.0
                if have_prior:
                    odom_dyaw = math.atan2(
                        math.sin(self.odom_yaw - self.odom_yaw_at_last_apply),
                        math.cos(self.odom_yaw - self.odom_yaw_at_last_apply),
                    )
                expected_yaw = prior_pose.yaw + odom_dyaw
                gate_rad, hatch_limit = self.odom_yaw_gate_rad, 12
            # SIMPLIFICATION (2026-07-07): with a fresh IMU the yaw is fed
            # forward and the yaw-locked global solve returns exactly the
            # prior yaw - the gate is dead weight there. It only guards the
            # legacy (IMU-less) local-window path now.
            if can_apply_pose and gate_rad > 0.0 and have_prior and not imu_fresh:
                yaw_error = math.atan2(
                    math.sin(result.pose.yaw - expected_yaw),
                    math.cos(result.pose.yaw - expected_yaw),
                )
                if abs(yaw_error) > gate_rad:
                    self.yaw_gate_consecutive = getattr(self, "yaw_gate_consecutive", 0) + 1
                    if self.yaw_gate_consecutive <= hatch_limit:
                        yaw_gated = True
                        self.yaw_gate_reject_count += 1
                        if not imu_fresh:
                            # Odom path: propagate the prior by the odom
                            # delta (the IMU path advances pose.yaw itself).
                            self.pose = Pose2D(prior_pose.x, prior_pose.y, expected_yaw)
                            self.odom_yaw_at_last_apply = self.odom_yaw
                    else:
                        self.yaw_gate_consecutive = 0
                else:
                    self.yaw_gate_consecutive = 0
            if can_apply_pose and not yaw_gated:
                yaw_new = result.pose.yaw
                if imu_fresh and imu_at_capture is not None and self.imu_yaw is not None:
                    # The scan was taken at capture time; bring its yaw
                    # forward by the IMU rotation accumulated during the
                    # ~80 ms matching so an in-progress turn is not rewound.
                    yaw_new = result.pose.yaw + (self.imu_yaw - imu_at_capture)
                self.pose = Pose2D(
                    result.pose.x,
                    result.pose.y,
                    math.atan2(math.sin(yaw_new), math.cos(yaw_new)),
                )
                self.last_localization_apply_started = started
                self.odom_yaw_at_last_apply = self.odom_yaw
                self.imu_yaw_at_last_apply = self.imu_yaw
            self.last_match = result
            self.last_scan_time = started
            self.last_scan_stamp_age_ms = age_ms
            self.obstacle_front_m = front
            self.scan_sector_ranges = sector_ranges
        if yaw_gated:
            self.get_logger().warn(
                f"yaw gate rejected scan-match yaw {result.pose.yaw:.2f} "
                f"(odom-propagated prior kept, rejects={self.yaw_gate_reject_count})",
                throttle_duration_sec=2.0,
            )
        if result.latency_ms > self.localization_warn_latency_ms:
            self.get_logger().warn(
                f"LiDAR localization latency {result.latency_ms:.1f} ms exceeds "
                f"{self.localization_warn_latency_ms:.1f} ms",
                throttle_duration_sec=2.0,
            )

    def objects_callback(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        objects = payload.get("objects") if isinstance(payload, dict) else None
        if not isinstance(objects, list):
            return
        with self.lock:
            self.latest_objects = [obj for obj in objects if isinstance(obj, dict)]
            self.latest_objects_time = time.monotonic()

    def blocking_object_ahead(
        self,
        pose: Pose2D,
        objects: list[dict],
        objects_age: float,
    ) -> Optional[dict]:
        if not self.object_avoidance_enabled:
            return None
        if objects_age > self.object_avoidance_max_age_sec:
            return None

        best = None
        best_forward = math.inf
        cos_yaw = math.cos(pose.yaw)
        sin_yaw = math.sin(pose.yaw)
        lookahead = max(0.0, float(self.object_avoidance_lookahead_m))
        half_width = max(0.0, float(self.object_keepout_half_width_m))
        half_depth = max(0.0, float(self.object_keepout_half_depth_m))

        for obj in objects:
            if obj.get("confirmed") is False:
                continue
            position = obj.get("position")
            if not isinstance(position, dict):
                continue
            try:
                dx = float(position["x"]) - pose.x
                dy = float(position["y"]) - pose.y
            except (KeyError, TypeError, ValueError):
                continue

            forward = dx * cos_yaw + dy * sin_yaw
            left = -dx * sin_yaw + dy * cos_yaw
            radius = max(0.0, float(obj.get("radius_m", 0.0) or 0.0))
            object_front_edge = forward - half_depth - radius
            object_back_edge = forward + half_depth + radius
            if object_back_edge < 0.0 or object_front_edge > lookahead:
                continue
            if abs(left) > half_width + radius:
                continue
            if forward < best_forward:
                best_forward = forward
                best = {
                    "id": obj.get("id"),
                    "class_name": obj.get("class_name"),
                    "forward_m": float(forward),
                    "left_m": float(left),
                    "keepout_half_width_m": float(half_width),
                    "keepout_half_depth_m": float(half_depth),
                    "recent_votes": int(obj.get("recent_votes", 0) or 0),
                }
        return best

    def match_scan(self, msg: LaserScan, prior_pose: Pose2D) -> MatchResult:
        if self.localization_mode in {"wall_range_global", "wall_global"}:
            return self.wall_localizer.match_scan_global(
                msg.ranges,
                msg.angle_min,
                msg.angle_increment,
                msg.range_min,
                msg.range_max,
                self.scan_yaw_offset_rad,
                prior_pose,
                xy_step_m=self.localization_reseed_xy_step_m,
                yaw_step_rad=self.localization_reseed_yaw_step_rad,
                max_beams=self.localization_reseed_max_beams,
                refine_max_beams=self.localization_reseed_refine_max_beams,
            )
        if self.localization_mode in {
            "wall_range_global_coarse",
            "wall_global_coarse",
            "global_coarse",
        }:
            return self.wall_localizer.match_scan_global_coarse(
                msg.ranges,
                msg.angle_min,
                msg.angle_increment,
                msg.range_min,
                msg.range_max,
                self.scan_yaw_offset_rad,
                prior_pose,
                xy_step_m=self.localization_reseed_xy_step_m,
                yaw_step_rad=self.localization_reseed_yaw_step_rad,
                max_beams=self.localization_reseed_max_beams,
            )
        if self.localization_mode in {"wall", "wall_range", "wall_range_fallback"}:
            # With a fresh IMU the prior yaw is trusted outright (feed-
            # forward keeps it current), so the matcher solves x/y only on
            # the coarse grid - ~10x fewer candidates, well under the real
            # robot's 50 ms budget. Gyro bias (~0.05 dps calibrated) still
            # drifts a few deg over a match, so WHEN STATIONARY a normal
            # yaw-searched match runs every ~5 s to trim the drift.
            now_mono = time.monotonic()
            imu_locked = (
                self.imu_yaw is not None
                and (now_mono - self.imu_last_rx) < 0.5
            )
            if imu_locked and abs(getattr(self, "imu_last_wz", 1.0)) < 0.03:
                if now_mono - getattr(self, "_last_yaw_trim", 0.0) > 5.0:
                    self._last_yaw_trim = now_mono
                    imu_locked = False
            if imu_locked:
                # Known yaw makes the square-arena x/y solution UNIQUE, so
                # solve GLOBALLY every scan: no local window to fall behind,
                # no reseed guard to trap a diverged estimate (fuse6 lost
                # 1.3 m for 20 s because both windows were smaller than the
                # error). Single-yaw grid keeps the cost at ~200 candidates.
                result = self.wall_localizer.match_scan_global(
                    msg.ranges,
                    msg.angle_min,
                    msg.angle_increment,
                    msg.range_min,
                    msg.range_max,
                    self.scan_yaw_offset_rad,
                    prior_pose,
                    xy_step_m=0.3,
                    max_beams=self.localization_reseed_max_beams,
                    yaw_fixed=float(prior_pose.yaw or 0.0),
                )
            else:
                result = self.wall_localizer.match_scan(
                    msg.ranges,
                    msg.angle_min,
                    msg.angle_increment,
                    msg.range_min,
                    msg.range_max,
                    self.scan_yaw_offset_rad,
                    prior_pose,
                )
                result = self.maybe_reseed_wall_match(msg, prior_pose, result)
            if (
                result.success
                and result.score <= self.wall_localization_max_score
            ):
                return result
            if self.localization_mode != "wall_range":
                fallback = self.match_known_map(msg, prior_pose)
                if fallback.success:
                    return fallback
            return result
        return self.match_known_map(msg, prior_pose)

    def maybe_reseed_wall_match(
        self,
        msg: LaserScan,
        prior_pose: Pose2D,
        local_result: MatchResult,
    ) -> MatchResult:
        period = max(0.0, self.localization_reseed_period_sec)
        now = time.monotonic()
        local_score_bad = (
            not local_result.success
            or local_result.score > self.wall_localization_max_score
        )
        if not local_score_bad and (
            period <= 0.0
            or now - self.last_localization_reseed_time < period
        ):
            return local_result

        self.last_localization_reseed_time = now
        global_result = self.wall_localizer.match_scan_global(
            msg.ranges,
            msg.angle_min,
            msg.angle_increment,
            msg.range_min,
            msg.range_max,
            self.scan_yaw_offset_rad,
            prior_pose,
            xy_step_m=self.localization_reseed_xy_step_m,
            yaw_step_rad=self.localization_reseed_yaw_step_rad,
            max_beams=self.localization_reseed_max_beams,
            refine_max_beams=self.localization_reseed_refine_max_beams,
        )

        combined_latency_ms = local_result.latency_ms + global_result.latency_ms
        combined_candidates = (
            local_result.candidate_count + global_result.candidate_count
        )
        # 90도 대칭 방어 (2026-07-21, 16:1x 실기 +90도 쿼드런트 플립 재발 방지):
        # 정사각 아레나의 wall_range 관측모델은 90도 축퇴라 전역 재탐색이
        # '이웃 90도 최적해'를 잡을 수 있다. 전파 prior 에서 45도(축퇴 basin
        # 반경) 넘게 벗어난 yaw 의 전역해는 채택하지 않는다 — SEED 가 참 yaw
        # 를 보장하고, 스캔 간 정당한 yaw 보정은 45도를 넘을 수 없다.
        yaw_jump_ok = True
        if global_result.success and global_result.pose is not None:
            dyaw = global_result.pose.yaw - prior_pose.yaw
            yaw_jump_ok = abs(
                math.atan2(math.sin(dyaw), math.cos(dyaw))
            ) <= math.radians(45.0)
        accept_global = (
            global_result.success
            and yaw_jump_ok
            and (
                not local_result.success
                or (
                    local_result.score > self.wall_localization_max_score
                    and global_result.score <= self.wall_localization_max_score
                )
                or global_result.score + self.localization_reseed_score_margin
                < local_result.score
            )
        )
        if accept_global:
            reason = (
                f"{global_result.reason},reseed=accepted,"
                f"local_score={_score_text(local_result.score)}"
            )
            return MatchResult(
                pose=global_result.pose,
                score=global_result.score,
                latency_ms=combined_latency_ms,
                beam_count=global_result.beam_count,
                candidate_count=combined_candidates,
                success=global_result.success,
                reason=reason,
            )

        reason = (
            f"{local_result.reason},reseed=rejected,"
            + ("yaw_jump>45deg," if not yaw_jump_ok else "")
            + f"global_score={_score_text(global_result.score)}"
        )
        return MatchResult(
            pose=local_result.pose,
            score=local_result.score,
            latency_ms=combined_latency_ms,
            beam_count=local_result.beam_count,
            candidate_count=combined_candidates,
            success=local_result.success,
            reason=reason,
        )

    def match_known_map(self, msg: LaserScan, prior_pose: Pose2D) -> MatchResult:
        points = self.localizer.scan_to_points(
            msg.ranges,
            msg.angle_min,
            msg.angle_increment,
            msg.range_min,
            msg.range_max,
            yaw_offset=self.scan_yaw_offset_rad,
        )
        return self.localizer.match(points, prior_pose)

    def control_tick(self):
        gripper_commands = []
        with self.lock:
            if self.pending_gripper_commands:
                gripper_commands = list(self.pending_gripper_commands)
                self.pending_gripper_commands.clear()
            pose = self.pose
            goal = self.goal
            scan_age = time.monotonic() - self.last_scan_time if self.last_scan_time else math.inf
            obstacle_front_m = self.obstacle_front_m
            objects = list(self.latest_objects)
            objects_age = (
                time.monotonic() - self.latest_objects_time
                if self.latest_objects_time
                else math.inf
            )
            goal_position_latched = self.goal_position_latched
            if (
                goal is not None
                and scan_age <= self.scan_timeout_sec
                and self.goal_position_latch_enabled
            ):
                distance = xy_distance(pose, goal)
                (
                    self.goal_position_latched,
                    self.goal_position_release_count,
                ) = update_goal_position_latch(
                    self.goal_position_latched,
                    self.goal_position_release_count,
                    distance,
                    self.xy_tolerance_m,
                    self.xy_release_tolerance_m,
                    self.xy_release_consecutive_count,
                )
                goal_position_latched = self.goal_position_latched

        for command in gripper_commands:
            msg = String()
            msg.data = command
            self.gripper_pub.publish(msg)

        twist = Twist()
        phase = "idle"
        object_block = None
        if goal is not None:
            if scan_age > self.scan_timeout_sec:
                phase = "scan_timeout"
            else:
                control_goal = (
                    goal_for_position_latch(pose, goal)
                    if goal_position_latched
                    else goal
                )
                command = self.controller.compute_command(
                    pose,
                    control_goal,
                    obstacle_front_m,
                )
                phase = command.phase
                twist.linear.x = float(command.vx)
                twist.linear.y = float(command.vy)
                twist.angular.z = float(command.omega)
                object_block = self.blocking_object_ahead(pose, objects, objects_age)
                if object_block is not None and twist.linear.x > 0.0:
                    twist = Twist()
                    turn_direction = -1.0 if float(object_block.get("left_m", 0.0)) >= 0.0 else 1.0
                    twist.angular.z = turn_direction * abs(self.object_avoidance_turn_rps)
                    phase = "object_avoid"
                if command.reached:
                    with self.lock:
                        if self.goal == goal:
                            self.goal = None
                            self.goal_position_latched = False
                            self.goal_position_release_count = 0
                if command.blocked:
                    twist = Twist()
        # 2026-07-19: idle(goal 없음)일 때 무한 zero-twist 발행 중단.
        # /cmd_vel_direct는 mux 최우선이라 idle 발행이 텔레옵(/cmd_vel)을 항상
        # 눌러버렸다. 정지 후 idle_cmd_publish_sec 동안만 zero를 내보내 브레이크를
        # 보장하고, 이후엔 침묵해 mux가 하위 소스로 폴스루하게 한다.
        if goal is not None:
            self._last_goal_active_time = time.monotonic()
            self.cmd_pub.publish(twist)
        elif time.monotonic() - getattr(self, "_last_goal_active_time", 0.0) \
                <= self.idle_cmd_publish_sec:
            self.cmd_pub.publish(twist)
        self.log_command_state(
            phase,
            pose,
            goal,
            twist,
            scan_age,
            obstacle_front_m,
            object_block,
        )
        with self.lock:
            self.last_command = twist
            self.command_phase = phase
            self.object_avoidance_block = object_block

    def publish_status(self):
        msg = String()
        msg.data = json.dumps(self.state_snapshot(), sort_keys=True)
        self.status_pub.publish(msg)

    def _arena_info(self) -> dict:
        """맵에서 유도한 아레나 기하. __init__ 에서 한 번만 만든다.

        half_m 은 네 변까지의 거리 중 **최대**다. inner_wall_bounds() 가 주는 건
        3px 벽의 안쪽 면이라 벽 중심보다 1~2셀 안쪽이고(경기 맵: ±2.0m 벽 ->
        -1.97/+1.99), 변마다 반 셀씩 다르다. 최대를 쓰면 벽 중심에 가장 가까워
        러너 상수와의 대조 오차가 셀 크기(0.02m) 안에 들어온다.
        """
        b = self.map.inner_wall_bounds()
        return {
            "xmin": b.xmin, "xmax": b.xmax, "ymin": b.ymin, "ymax": b.ymax,
            "half_m": max(abs(b.xmin), abs(b.xmax), abs(b.ymin), abs(b.ymax)),
            "map_yaml": str(self.get_parameter("map_yaml").value),
        }

    def map_snapshot(self) -> dict:
        return {
            "width": self.map.width,
            "height": self.map.height,
            "resolution": self.map.resolution,
            "origin": list(self.map.origin),
            "occupied": self.map.occupancy_for_display(),
        }

    def state_snapshot(self) -> dict:
        with self.lock:
            pose = self.pose
            goal = self.goal
            match = self.last_match
            command = self.last_command
            scan_age = time.monotonic() - self.last_scan_time if self.last_scan_time else None
            snapshot = {
                # [2026-07-23 신규 — 조작자 지시] pose 유효시각. 종전엔 status 에
                # 시각이 **하나도 없어** 소비자가 "수신시각"으로만 정렬할 수 있었고,
                # 그건 executor 가 멎었다 깨어날 때 백로그가 한꺼번에 같은 시각으로
                # 찍혀 무의미해진다 (실측: status 20Hz 설정인데 카메라 4스트림
                # 구독 시 6Hz·공백 최대 3.15s). 카메라 Image header.stamp 와 같은
                # ROS 클럭으로 찍어 프레임↔pose 시각 정렬을 가능하게 한다.
                # ⚠ 이 시각이 pose 의 유효시각인 근거: yaw 는 imu_yaw_callback 이
                #   연속 dead-reckoning 으로 갱신하므로(IMU feed-forward, ~200Hz)
                #   스냅샷을 뜨는 이 순간이 곧 yaw 의 시각이다. x/y 는 스캔 주기로
                #   갱신되지만 스텝 스캔은 제자리라 병진이 0 이다.
                "stamp": self.get_clock().now().nanoseconds * 1e-9,
                "arena": self.arena_info,
                "pose": _pose_dict(pose),
                "goal": _pose_dict(goal) if goal is not None else None,
                "command": {
                    "linear_x": command.linear.x,
                    "linear_y": command.linear.y,
                    "angular_z": command.angular.z,
                },
                "command_phase": self.command_phase,
                "controller_type": self.controller_type,
                "goal_position_latched": self.goal_position_latched,
                "goal_position_release_count": self.goal_position_release_count,
                "ui_url": getattr(self, "ui_url", ""),
                "obstacle_front_m": self.obstacle_front_m,
                "object_avoidance_enabled": self.object_avoidance_enabled,
                "object_avoidance_block": self.object_avoidance_block,
                "scan_sector_ranges": dict(self.scan_sector_ranges),
                "scan_age_sec": scan_age,
                "scan_stamp_age_ms": self.last_scan_stamp_age_ms,
                "localization": _match_dict(match),
                "gripper_status": self.last_gripper_status,
                "motor_state": self.last_motor_state,
                "debug_odom": dict(self.last_debug_odom),
                "debug_imu": dict(self.last_debug_imu),
                "imu_prior": {
                    "yaw_integrated": self.imu_yaw,
                    "fresh": (time.monotonic() - self.imu_last_rx) < 0.5 if self.imu_yaw is not None else False,
                    "gate_rejects": self.yaw_gate_reject_count,
                },
                "perception": self.detector.snapshot(),
            }
        return snapshot

    def set_goal(self, x: float, y: float, yaw: Optional[float]):
        goal_yaw = resolve_goal_yaw(yaw, self.default_goal_yaw_rad)
        with self.lock:
            self.goal = Pose2D(float(x), float(y), goal_yaw)
            self.goal_position_latched = False
            self.goal_position_release_count = 0
        yaw_text = "none" if goal_yaw is None else f"{goal_yaw:.3f}"
        self.get_logger().info(f"goal set from UI: x={x:.3f}, y={y:.3f}, yaw={yaw_text}")

    def goal_command_callback(self, msg: String):
        try:
            payload = json.loads(msg.data)
            yaw = payload.get("yaw")
            self.set_goal(
                float(payload["x"]),
                float(payload["y"]),
                None if yaw is None else float(yaw),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().warn(f"ignored invalid goal command: {exc}")

    def scan_sector_snapshot(
        self,
        ranges,
        angle_min: float,
        angle_increment: float,
        range_min: float,
        range_max: float,
    ) -> dict:
        sectors = {
            "front": 0.0,
            "left": math.pi / 2.0,
            "back": math.pi,
            "right": -math.pi / 2.0,
        }
        return {
            name: min_range_in_sector(
                ranges,
                angle_min,
                angle_increment,
                range_min,
                range_max,
                self.obstacle_sector_rad,
                sector_center_rad=center,
                yaw_offset=self.scan_yaw_offset_rad,
            )
            for name, center in sectors.items()
        }

    def set_pose(self, x: float, y: float, yaw: float):
        with self.lock:
            self.pose = Pose2D(float(x), float(y), float(yaw))
            self.last_match = None
            self.pose_revision += 1
            self.goal_position_latched = False
            self.goal_position_release_count = 0
        self.get_logger().info(f"pose reset from UI: x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}")

    def pose_command_callback(self, msg: String):
        try:
            payload = json.loads(msg.data)
            self.set_pose(
                float(payload["x"]),
                float(payload["y"]),
                float(payload.get("yaw", 0.0)),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().warn(f"ignored invalid pose command: {exc}")

    def clear_goal(self):
        with self.lock:
            self.goal = None
            self.goal_position_latched = False
            self.goal_position_release_count = 0
            self.command_phase = "stopped"
        self.cmd_pub.publish(Twist())
        self.get_logger().info("goal cleared from UI")

    def control_command_callback(self, msg: String):
        command = msg.data.strip().upper()
        if command in {"STOP", "CLEAR", "CLEAR_GOAL"}:
            self.clear_goal()
        else:
            self.get_logger().warn(f"ignored unknown control command: {command}")

    def log_command_state(
        self,
        phase: str,
        pose: Pose2D,
        goal: Optional[Pose2D],
        twist: Twist,
        scan_age: float,
        obstacle_front_m: Optional[float],
        object_block: Optional[dict] = None,
    ):
        if goal is None:
            return

        now = time.monotonic()
        key = phase
        if key == self.last_command_log_key and now - self.last_command_log_time < 1.0:
            return

        self.last_command_log_key = key
        self.last_command_log_time = now
        scan_age_text = "inf" if math.isinf(scan_age) else f"{scan_age:.3f}"
        front_text = "none" if obstacle_front_m is None else f"{obstacle_front_m:.3f}"
        object_text = "none"
        if object_block is not None:
            object_text = (
                f"{object_block.get('class_name') or object_block.get('id') or 'object'} "
                f"x={float(object_block.get('forward_m', 0.0)):.3f} "
                f"y={float(object_block.get('left_m', 0.0)):.3f}"
            )
        self.get_logger().info(
            "arena command: "
            f"phase={phase} "
            f"vx={twist.linear.x:.3f} "
            f"vy={twist.linear.y:.3f} "
            f"wz={twist.angular.z:.3f} "
            f"pose=({pose.x:.2f},{pose.y:.2f},{pose.yaw:.2f}) "
            f"goal={self.goal_log_text(goal)} "
            f"scan_age={scan_age_text}s "
            f"front={front_text}m "
            f"object_block={object_text}"
        )

    @staticmethod
    def goal_log_text(goal: Pose2D) -> str:
        yaw_text = "none" if goal.yaw is None else f"{float(goal.yaw):.2f}"
        return f"({goal.x:.2f},{goal.y:.2f},{yaw_text})"

    def enqueue_gripper_command(self, command: str):
        command = command.strip().upper()
        allowed = {
            "OPEN",
            "CLOSE",
            "STOP",
            "DXL_POWER_ON",
            "DXL_POWER_CYCLE",
            "POWER_ON",
            "RECOVER",
        }
        if command not in allowed and not command.startswith("SET_DEG "):
            self.get_logger().warn("rejected unknown gripper command from UI: " + command)
            return
        with self.lock:
            self.pending_gripper_commands.append(command)

    def gripper_status_callback(self, msg: String):
        with self.lock:
            self.last_gripper_status = msg.data[:360]

    def motor_state_callback(self, msg: String):
        with self.lock:
            self.last_motor_state = msg.data[:360]

    def imu_yaw_callback(self, msg: Imu):
        """Integrate gyro-z into an unwrapped yaw prior (D435i-class IMU).

        Only DELTAS between scan applications are consumed, so neither the
        absolute offset nor slow bias drift (few deg/min) matters within a
        match; slip does not exist for a gyro, unlike mecanum wheel odom."""
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        with self.lock:
            # Online zero-rate (bias) estimation, as real IMU drivers do:
            # when the reading is indistinguishable from rest, low-pass it
            # into the bias and integrate (wz - bias). Without this the
            # BMI055-class drift (deg/min) accumulates into the yaw feed-
            # forward and skews wall_range positions by range*yaw_err.
            ax, ay, az = self.imu_yaw_axis
            wz_raw = (
                ax * float(msg.angular_velocity.x)
                + ay * float(msg.angular_velocity.y)
                + az * float(msg.angular_velocity.z)
            )
            bias = getattr(self, "imu_gyro_bias", 0.0)
            if abs(wz_raw - bias) < 0.02:
                self.imu_gyro_bias = 0.98 * bias + 0.02 * wz_raw
            if self.imu_last_stamp is not None:
                dt = stamp - self.imu_last_stamp
                if 0.0 < dt < 0.5:
                    dyaw = (wz_raw - getattr(self, "imu_gyro_bias", 0.0)) * dt
                    base = self.imu_yaw if self.imu_yaw is not None else 0.0
                    self.imu_yaw = base + dyaw
                    # FEED-FORWARD: dead-reckon the pose yaw between scan
                    # applications. The scan matcher then only corrects the
                    # small residual (gyro noise/bias over one scan period)
                    # instead of chasing a fast turn with its ±0.14 rad
                    # window - which is how the estimate used to fall onto a
                    # neighbouring 90-deg symmetric optimum.
                    new_yaw = self.pose.yaw + dyaw
                    self.pose = Pose2D(
                        self.pose.x,
                        self.pose.y,
                        math.atan2(math.sin(new_yaw), math.cos(new_yaw)),
                    )
            elif self.imu_yaw is None:
                self.imu_yaw = 0.0
            self.imu_last_stamp = stamp
            self.imu_last_rx = time.monotonic()
            self.imu_last_wz = wz_raw

    def debug_odom_callback(self, msg: Odometry):
        pose = msg.pose.pose.position
        quat = msg.pose.pose.orientation
        twist = msg.twist.twist
        yaw = math.atan2(
            2.0 * (quat.w * quat.z + quat.x * quat.y),
            1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z),
        )
        with self.lock:
            self.odom_yaw = yaw
            self.last_debug_odom = {
                "x": float(pose.x),
                "y": float(pose.y),
                "yaw": float(yaw),
                "linear_x": float(twist.linear.x),
                "angular_z": float(twist.angular.z),
            }

    def debug_imu_callback(self, msg: Imu):
        with self.lock:
            self.last_debug_imu = {
                "angular_z": float(msg.angular_velocity.z),
                "linear_accel_x": float(msg.linear_acceleration.x),
                "linear_accel_y": float(msg.linear_acceleration.y),
            }

    def image_callback(self, msg: Image):
        self.detector.process_image(msg)

    def destroy_node(self):
        try:
            if self.web_server is not None:
                self.web_server.stop()
        finally:
            super().destroy_node()


# The map used to be installed by example_nav2, which is not part of this
# release. It now ships with this package (setup.py data_files), and the
# source-tree fallback follows the part-directory layout.
# 아레나 맵 파일명. 런치의 map_yaml 기본값과 **같아야 한다** — 노드를
# 단독 기동(ros2 run)하면 런치를 안 거치므로 이 값이 쓰인다. 둘이 갈리면
# 런처로 띄울 때와 단독으로 띄울 때 서로 다른 아레나를 믿게 된다.
DEFAULT_MAP_NAME = "demo2m.yaml"   # [demo/arena-2m] 경기 stadium.yaml
MAP_REL_PATH = (Path("navigation") / "ros2" / "arena_lightweight_control"
                / "maps" / DEFAULT_MAP_NAME)


def default_map_yaml() -> str:
    try:
        return str(
            Path(get_package_share_directory("arena_lightweight_control"))
            / "maps"
            / DEFAULT_MAP_NAME
        )
    except PackageNotFoundError:
        return str(find_repo_root() / MAP_REL_PATH)


def find_repo_root() -> Path:
    env_root = os.environ.get("ROBOT_REPO_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend([Path.cwd(), Path(__file__).resolve()])
    for candidate in candidates:
        for root in [candidate, *candidate.parents]:
            if (root / MAP_REL_PATH).exists():
                return root
    return Path.cwd()


def _pose_dict(pose: Pose2D) -> dict:
    return {"x": pose.x, "y": pose.y, "yaw": pose.yaw}


def _match_dict(match: Optional[MatchResult]) -> Optional[dict]:
    if match is None:
        return None
    return {
        "success": match.success,
        "score": match.score,
        "latency_ms": match.latency_ms,
        "beam_count": match.beam_count,
        "candidate_count": match.candidate_count,
        "reason": match.reason,
    }


def _score_text(score: float) -> str:
    return "inf" if not math.isfinite(score) else f"{score:.3f}"


def xy_distance(pose: Pose2D, goal: Pose2D) -> float:
    return math.hypot(goal.x - pose.x, goal.y - pose.y)


def goal_for_position_latch(pose: Pose2D, goal: Pose2D) -> Pose2D:
    return Pose2D(pose.x, pose.y, goal.yaw)


def update_goal_position_latch(
    latched: bool,
    release_count: int,
    distance_m: float,
    enter_tolerance_m: float,
    release_tolerance_m: float,
    release_count_threshold: int,
) -> tuple[bool, int]:
    if distance_m <= enter_tolerance_m:
        return True, 0
    if not latched:
        return False, 0
    if distance_m <= max(enter_tolerance_m, release_tolerance_m):
        return True, 0

    next_release_count = release_count + 1
    if next_release_count >= max(1, release_count_threshold):
        return False, 0
    return True, next_release_count


def resolve_goal_yaw(
    requested_yaw: Optional[float],
    default_goal_yaw_rad: Optional[float],
) -> Optional[float]:
    if requested_yaw is not None:
        return float(requested_yaw)
    if default_goal_yaw_rad is None:
        return None
    default_yaw = float(default_goal_yaw_rad)
    return default_yaw if math.isfinite(default_yaw) else None


def _apply_cpu_affinity() -> None:
    """`ARENA_CPU_AFFINITY` 가 있으면 이 프로세스를 해당 코어에 고정한다.

    [2026-07-23 신규 — 조작자 승인] 02:14 실기에서 YOLO 배치추론(a1 3.66s)과
    스캔매칭이 코어를 다투며 매칭 지연이 p90 222ms / max 300ms 로 팽창했고,
    pose 유효 갱신이 7.4Hz(라이다 13.1Hz의 56%)로 반토막 나 street_nav 가
    남하 레그의 25%를 정지시켰다 (84틱 trace).

    1차 대책은 **추론 쪽**(e2e_match_test)을 상위 코어에서 몰아내는 것이고
    그것만으로 이 노드가 빈 코어로 옮겨간다 (CFS 로드밸런싱). 이 하드 핀은
    그것으로 부족할 때 쓰는 **2차 수단이라 기본 비활성**이다.
      ⚠ 켤 때 주의: RealSense/라이다 드라이버는 여전히 전 코어를 떠다니므로,
        이 노드를 2코어에 묶으면 그 2코어를 드라이버와 나눠 쓰게 되어 오히려
        나빠질 수 있다. 켠 뒤 반드시 매칭 지연 p90 을 재측정해 비교할 것.
    사용: ARENA_CPU_AFFINITY=4,5 (또는 auto) 로 스택을 기동.
    ⚠ rclpy.init() 보다 먼저 호출해야 DDS/executor 스레드가 마스크를 상속한다.
    """
    raw = os.environ.get("ARENA_CPU_AFFINITY", "").strip()
    if not raw or raw.lower() in ("off", "none", "0"):
        return
    try:
        avail = sorted(os.sched_getaffinity(0))
        if raw.lower() == "auto":
            if len(avail) < 4:
                return
            cores = set(avail[-2:])       # 상위 2코어 = fieldlib 예약분과 동일
        else:
            cores = {int(t) for t in raw.replace(" ", ",").split(",") if t != ""}
        os.sched_setaffinity(0, cores)
        print(f"[arena] CPU 친화도 → {sorted(os.sched_getaffinity(0))} "
              f"(ARENA_CPU_AFFINITY={raw})", flush=True)
    except Exception as e:   # 권한/커널/오타 — 무시하고 종전대로 기동
        print(f"[arena] ⚠ CPU 친화도 설정 실패({e}) — 분리 없이 계속", flush=True)


def main(args=None):
    _apply_cpu_affinity()   # rclpy.init 전에 (스레드 상속)
    rclpy.init(args=args)
    node = ArenaControlNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
