from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    map_yaml = LaunchConfiguration("map_yaml")
    launch_bridge = LaunchConfiguration("launch_bridge")
    launch_cameras = LaunchConfiguration("launch_cameras")
    launch_python_ui = LaunchConfiguration("launch_python_ui")

    return LaunchDescription(
        [
            DeclareLaunchArgument("launch_bridge", default_value="true"),
            DeclareLaunchArgument("launch_cameras", default_value="false"),
            # RGB/depth 해상도 분리 (2026-07-21) — bridge 로 그대로 전달.
            # 기본 FHD RGB + VGA depth (bridge 런치 기본값과 동일하게 유지할 것)
            DeclareLaunchArgument("camera_rgb_profile", default_value="1920x1080x15"),
            # 16:9 계열 필수 — 4:3(640x480)은 depth H-FOV 크롭으로 FHD 가장자리
            # aligned depth 가 빈다 (bridge launch 주석 참조, 2026-07-21)
            DeclareLaunchArgument("camera_depth_profile", default_value="848x480x15"),
            # 비어 있지 않으면 arena 노드가 gyro-z 적분 yaw prior를 사용해
            # 정사각 경기장 wall_range의 90도 대칭 플립을 차단한다.
            # 2026-07-21 기본 활성화: 16:1x 실기 2회의 +90° 쿼드런트 플립
            # (START 코너 오적재)이 imu_topic 공란 → prior 구독 미생성 →
            # 레거시 무제한 yaw 재탐색 경로에서 발생함을 로그로 확정
            # (D435I 하단캠은 /imu/data 를 정상 발행 중이었다 — 브리지
            # enable_bottom_imu 기본 true). 끄려면 imu_topic:="" 명시.
            DeclareLaunchArgument("imu_topic", default_value="/imu/data"),
            DeclareLaunchArgument("launch_python_ui", default_value="true"),
            DeclareLaunchArgument("ui_host", default_value="0.0.0.0"),
            DeclareLaunchArgument("ui_port", default_value="18765"),
            DeclareLaunchArgument("enable_web_ui", default_value="false"),
            DeclareLaunchArgument(
                "map_yaml",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("example_nav2"), "maps", "stadium.yaml"]
                ),
            ),
            DeclareLaunchArgument("scan_topic", default_value="/laser_scan"),
            # [2026-07-21] arena_control_node status 발행 주기 [s].
            # 0.05 = 20Hz (노드가 max(0.05, ...) 로 클램프하므로 이게 상한).
            # 근거·주의사항은 arena_control_node parameters 블록 주석 참조.
            # 부하 걱정 시 되돌리려면 0.25 (종전 노드 기본값).
            DeclareLaunchArgument("status_period_sec", default_value="0.05"),
            # [2026-07-21] 스캔매칭 최소 주기 [s]. 0.05=20Hz 상한(실기 라이다가
            # 13.1Hz 라 실질 무제한). GIL 기아 재발 시 0.08 로 되돌릴 것.
            DeclareLaunchArgument("scan_min_period_sec", default_value="0.05"),
            DeclareLaunchArgument("base_scan_yaw", default_value="3.14159265359"),
            DeclareLaunchArgument("arena_cmd_vel_topic", default_value="/cmd_vel_direct"),
            DeclareLaunchArgument("goal_command_topic", default_value="/arena_lightweight/goal"),
            DeclareLaunchArgument("pose_command_topic", default_value="/arena_lightweight/pose"),
            DeclareLaunchArgument("control_command_topic", default_value="/arena_lightweight/control"),
            # 2026-07-19: 구 북향(yaw+90) 기본 정착 제거 — "nan"이면 yaw 없는
            # goal은 final-yaw 회전 없이 정지 (북향 필요 시 goal에 yaw 명시).
            DeclareLaunchArgument("default_goal_yaw_rad", default_value="nan"),
            DeclareLaunchArgument("gripper_command_topic", default_value="/gripper/command"),
            # [2026-07-23] 기본값을 by-path → by-id glob 으로 교체.
            # 폐기: default_value="/dev/serial/by-path/platform-3610000.usb-usb-0:2.3.3:1.0"
            # by-path 는 **USB 허브의 물리 포트**를 가리킨다. OpenRB 를 다른
            # 포트에 다시 꽂으면 경로가 바뀌고(18:30 실기: 2.3.3 → 2.4.3),
            # 이 인자가 real.yaml 의 serial_port_candidates glob 을 덮어써서
            # 폴백이 사라진다 → gripper_bridge_node 가 포트를 못 열고 죽어
            # 헬스체크 "그리퍼/마스트 보드(OpenRB) 실동작" 결손으로 나타났다.
            # by-id 는 보드 자체를 가리켜 포트를 바꿔 꽂아도 따라온다.
            DeclareLaunchArgument(
                "gripper_serial_port",
                default_value="/dev/serial/by-id/usb-ROBOTIS_OpenRB-150*-if00",
            ),
            DeclareLaunchArgument(
                "gripper_joint_command_topic",
                default_value="/mk1/gripper_joint_command",
            ),
            DeclareLaunchArgument("controller_type", default_value="diff_drive"),
            DeclareLaunchArgument(
                "drive_type",
                default_value="diff",
                description="Motor bridge type passed to the bridge: diff or mecanum.",
            ),
            # START 구역(우하, web_ui.py COMP.start = x[1.6,2.0] y[-2.0,-1.6]) 중심.
            # 과거 기본값 -1.8,-1.8은 STORAGE 구역(x[-2.0,-1.6] y[-2.0,-1.6]) 좌표였음 —
            # 로봇이 물리적으로 START에 있어도 wall_range가 STORAGE에서 시작해
            # 국소탐색으로 서서히 이동하며 한동안 storage로 오분류되는 버그였다.
            DeclareLaunchArgument("initial_pose_x", default_value="1.8"),
            DeclareLaunchArgument("initial_pose_y", default_value="-1.8"),
            DeclareLaunchArgument("initial_pose_yaw", default_value="1.5708"),
            DeclareLaunchArgument("localization_max_beams", default_value="24"),
            DeclareLaunchArgument("localization_search_xy_m", default_value="0.12"),
            DeclareLaunchArgument("localization_search_yaw_rad", default_value="0.14"),
            DeclareLaunchArgument("localization_xy_step_m", default_value="0.04"),
            DeclareLaunchArgument("localization_yaw_step_rad", default_value="0.07"),
            DeclareLaunchArgument("localization_mode", default_value="wall_range"),
            DeclareLaunchArgument("wall_localization_max_score", default_value="0.22"),
            DeclareLaunchArgument("localization_reseed_period_sec", default_value="2.5"),
            DeclareLaunchArgument("localization_reseed_xy_step_m", default_value="0.32"),
            DeclareLaunchArgument("localization_reseed_yaw_step_rad", default_value="0.35"),
            DeclareLaunchArgument("localization_reseed_max_beams", default_value="16"),
            DeclareLaunchArgument("localization_reseed_refine_max_beams", default_value="16"),
            DeclareLaunchArgument("localization_reseed_score_margin", default_value="0.035"),
            DeclareLaunchArgument("obstacle_stop_distance_m", default_value="0.22"),
            DeclareLaunchArgument("xy_tolerance_m", default_value="0.15"),
            DeclareLaunchArgument("yaw_tolerance_rad", default_value="0.40"),
            DeclareLaunchArgument("goal_position_latch", default_value="true"),
            DeclareLaunchArgument("xy_release_tolerance_m", default_value="0.25"),
            DeclareLaunchArgument("xy_release_consecutive_count", default_value="3"),
            DeclareLaunchArgument("near_goal_slow_radius_m", default_value="0.10"),
            DeclareLaunchArgument("near_goal_speed_scale", default_value="0.5"),
            DeclareLaunchArgument("max_linear_mps", default_value="0.36"),
            DeclareLaunchArgument("min_linear_mps", default_value="0.075"),
            DeclareLaunchArgument("final_yaw_min_angular_rps", default_value="0.0"),
            DeclareLaunchArgument("linear_gain", default_value="1.425"),
            DeclareLaunchArgument("drive_angular_deadband_rad", default_value="0.08"),
            DeclareLaunchArgument("drive_angular_gain_scale", default_value="0.45"),
            DeclareLaunchArgument("scan_display_max_points", default_value="120"),
            DeclareLaunchArgument("enable_internal_image_processing", default_value="false"),
            DeclareLaunchArgument("detector_model_path", default_value=""),
            DeclareLaunchArgument("enable_object_avoidance", default_value="false"),
            DeclareLaunchArgument("objects_topic", default_value="/detected_objects/map_objects"),
            DeclareLaunchArgument("object_avoidance_max_age_sec", default_value="0.75"),
            DeclareLaunchArgument("object_keepout_half_width_m", default_value="0.15"),
            DeclareLaunchArgument("object_keepout_half_depth_m", default_value="0.20"),
            DeclareLaunchArgument("object_avoidance_lookahead_m", default_value="0.35"),
            DeclareLaunchArgument("object_avoidance_turn_rps", default_value="0.35"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("robot_bringup"),
                            "launch",
                            "real_competition_bridge.launch.py",
                        ]
                    )
                ),
                launch_arguments={
                    "launch_cameras": launch_cameras,
                    "camera_rgb_profile": LaunchConfiguration("camera_rgb_profile"),
                    "camera_depth_profile": LaunchConfiguration("camera_depth_profile"),
                    "launch_lidar": "true",
                    "launch_motor": "true",
                    "drive_type": LaunchConfiguration("drive_type"),
                    "launch_gripper": "true",
                    "launch_cmd_vel_mux": "true",
                    "launch_scan_watchdog": "true",
                    "cmd_vel_topic": "/cmd_vel_motor",
                    "direct_cmd_vel_topic": LaunchConfiguration("arena_cmd_vel_topic"),
                    "nav_cmd_vel_topic": "/cmd_vel_nav",
                    "default_cmd_vel_topic": "/cmd_vel",
                    "base_scan_yaw": LaunchConfiguration("base_scan_yaw"),
                    "gripper_command_topic": LaunchConfiguration("gripper_command_topic"),
                    "gripper_serial_port": LaunchConfiguration("gripper_serial_port"),
                    "gripper_joint_command_topic": LaunchConfiguration(
                        "gripper_joint_command_topic"
                    ),
                    "scan_watchdog_publish_empty_scan": "false",
                    "scan_watchdog_publish_on_receive": "true",
                    "scan_watchdog_republish_fresh_scan": "false",
                    "scan_watchdog_restamp_scans": "false",
                }.items(),
                condition=IfCondition(launch_bridge),
            ),
            Node(
                package="arena_lightweight_control",
                executable="arena_control_node",
                name="arena_control_node",
                output="screen",
                parameters=[
                    {
                        "map_yaml": map_yaml,
                        "scan_topic": LaunchConfiguration("scan_topic"),
                        "scan_yaw_offset_rad": ParameterValue(
                            LaunchConfiguration("base_scan_yaw"),
                            value_type=float,
                        ),
                        "cmd_vel_topic": LaunchConfiguration("arena_cmd_vel_topic"),
                        # 브리지(real_competition_bridge)는 odom을 /chassis/odom으로
                        # 발행한다. 노드 기본값(/odom/wheel)은 이 스택에서 퍼블리셔가
                        # 없어 odom yaw 게이트가 조용히 비활성화된다 (2026-07-15 실측).
                        "debug_odom_topic": "/chassis/odom",
                        "imu_topic": LaunchConfiguration("imu_topic"),
                        # [2026-07-21] status 발행 주기. 노드 기본 0.25s(4Hz)는
                        # street 폐루프 주행(횡오차 P 보정)의 피드백으로 느리다.
                        # ⚠ 이 값을 낮춰도 **새 정보는 늘지 않는다** — x/y 해는
                        #   scan_callback 스로틀(0.08s)로 12.5Hz가 상한이다.
                        #   얻는 것은 신선도: 새 해가 나온 뒤 소비자가 보기까지의
                        #   최대 지연이 250ms → 50ms 로 줄어든다 (위상여유 직결).
                        #   yaw 는 imu feed-forward 라 이미 200Hz 로 신선하다.
                        # ⚠ 소비자 주의: 20Hz 로 뽑으면 약 37% 샘플이 직전 x/y 와
                        #   동일하다. 고정 dt 로 미분하면 D항이 0/2배로 튀므로,
                        #   x/y 가 실제로 바뀐 샘플에서만 이벤트 구동할 것.
                        # ⚠ 런타임 `ros2 param set` 으로는 안 바뀐다 — 타이머가
                        #   init 때 한 번만 생성되고 _on_set_parameters 는 이
                        #   파라미터를 조용히 무시하며 successful=True 를 준다.
                        "status_period_sec": ParameterValue(
                            LaunchConfiguration("status_period_sec"),
                            value_type=float,
                        ),
                        # [2026-07-21] 스캔매칭 최소 주기. 종전 하드코딩 0.08s 는
                        # 실기 라이다 13.1Hz(76ms)를 걸러 매칭을 5~6Hz 로 반토막
                        # 냈다. 0.05 로 전량 통과 (근거는 노드 scan_callback 주석).
                        "scan_min_period_sec": ParameterValue(
                            LaunchConfiguration("scan_min_period_sec"),
                            value_type=float,
                        ),
                        # 실기 D435I 바텀캠(54도 틸트) gyro의 body-up 투영축
                        # (2026-07-15 회전 실측; sim은 노드 기본값 [0,0,1] 사용).
                        "imu_yaw_axis": [0.0, -0.586, -0.810],
                        "gripper_command_topic": LaunchConfiguration("gripper_command_topic"),
                        "controller_type": LaunchConfiguration("controller_type"),
                        "ui_host": LaunchConfiguration("ui_host"),
                        "ui_port": ParameterValue(
                            LaunchConfiguration("ui_port"),
                            value_type=int,
                        ),
                        "enable_web_ui": ParameterValue(
                            LaunchConfiguration("enable_web_ui"),
                            value_type=bool,
                        ),
                        "goal_command_topic": LaunchConfiguration("goal_command_topic"),
                        "pose_command_topic": LaunchConfiguration("pose_command_topic"),
                        "control_command_topic": LaunchConfiguration("control_command_topic"),
                        "default_goal_yaw_rad": ParameterValue(
                            LaunchConfiguration("default_goal_yaw_rad"),
                            value_type=float,
                        ),
                        "initial_pose_x": ParameterValue(
                            LaunchConfiguration("initial_pose_x"),
                            value_type=float,
                        ),
                        "initial_pose_y": ParameterValue(
                            LaunchConfiguration("initial_pose_y"),
                            value_type=float,
                        ),
                        "initial_pose_yaw": ParameterValue(
                            LaunchConfiguration("initial_pose_yaw"),
                            value_type=float,
                        ),
                        "localization_max_beams": ParameterValue(
                            LaunchConfiguration("localization_max_beams"),
                            value_type=int,
                        ),
                        "localization_search_xy_m": ParameterValue(
                            LaunchConfiguration("localization_search_xy_m"),
                            value_type=float,
                        ),
                        "localization_search_yaw_rad": ParameterValue(
                            LaunchConfiguration("localization_search_yaw_rad"),
                            value_type=float,
                        ),
                        "localization_xy_step_m": ParameterValue(
                            LaunchConfiguration("localization_xy_step_m"),
                            value_type=float,
                        ),
                        "localization_yaw_step_rad": ParameterValue(
                            LaunchConfiguration("localization_yaw_step_rad"),
                            value_type=float,
                        ),
                        "localization_mode": LaunchConfiguration("localization_mode"),
                        "wall_localization_max_score": ParameterValue(
                            LaunchConfiguration("wall_localization_max_score"),
                            value_type=float,
                        ),
                        "localization_reseed_period_sec": ParameterValue(
                            LaunchConfiguration("localization_reseed_period_sec"),
                            value_type=float,
                        ),
                        "localization_reseed_xy_step_m": ParameterValue(
                            LaunchConfiguration("localization_reseed_xy_step_m"),
                            value_type=float,
                        ),
                        "localization_reseed_yaw_step_rad": ParameterValue(
                            LaunchConfiguration("localization_reseed_yaw_step_rad"),
                            value_type=float,
                        ),
                        "localization_reseed_max_beams": ParameterValue(
                            LaunchConfiguration("localization_reseed_max_beams"),
                            value_type=int,
                        ),
                        "localization_reseed_refine_max_beams": ParameterValue(
                            LaunchConfiguration("localization_reseed_refine_max_beams"),
                            value_type=int,
                        ),
                        "localization_reseed_score_margin": ParameterValue(
                            LaunchConfiguration("localization_reseed_score_margin"),
                            value_type=float,
                        ),
                        "obstacle_stop_distance_m": ParameterValue(
                            LaunchConfiguration("obstacle_stop_distance_m"),
                            value_type=float,
                        ),
                        "xy_tolerance_m": ParameterValue(
                            LaunchConfiguration("xy_tolerance_m"),
                            value_type=float,
                        ),
                        "yaw_tolerance_rad": ParameterValue(
                            LaunchConfiguration("yaw_tolerance_rad"),
                            value_type=float,
                        ),
                        "goal_position_latch": ParameterValue(
                            LaunchConfiguration("goal_position_latch"),
                            value_type=bool,
                        ),
                        "xy_release_tolerance_m": ParameterValue(
                            LaunchConfiguration("xy_release_tolerance_m"),
                            value_type=float,
                        ),
                        "xy_release_consecutive_count": ParameterValue(
                            LaunchConfiguration("xy_release_consecutive_count"),
                            value_type=int,
                        ),
                        "near_goal_slow_radius_m": ParameterValue(
                            LaunchConfiguration("near_goal_slow_radius_m"),
                            value_type=float,
                        ),
                        "near_goal_speed_scale": ParameterValue(
                            LaunchConfiguration("near_goal_speed_scale"),
                            value_type=float,
                        ),
                        "max_linear_mps": ParameterValue(
                            LaunchConfiguration("max_linear_mps"),
                            value_type=float,
                        ),
                        "min_linear_mps": ParameterValue(
                            LaunchConfiguration("min_linear_mps"),
                            value_type=float,
                        ),
                        "final_yaw_min_angular_rps": ParameterValue(
                            LaunchConfiguration("final_yaw_min_angular_rps"),
                            value_type=float,
                        ),
                        "linear_gain": ParameterValue(
                            LaunchConfiguration("linear_gain"),
                            value_type=float,
                        ),
                        "drive_angular_deadband_rad": ParameterValue(
                            LaunchConfiguration("drive_angular_deadband_rad"),
                            value_type=float,
                        ),
                        "drive_angular_gain_scale": ParameterValue(
                            LaunchConfiguration("drive_angular_gain_scale"),
                            value_type=float,
                        ),
                        "enable_internal_image_processing": ParameterValue(
                            LaunchConfiguration("enable_internal_image_processing"),
                            value_type=bool,
                        ),
                        "detector_model_path": LaunchConfiguration("detector_model_path"),
                        "enable_object_avoidance": ParameterValue(
                            LaunchConfiguration("enable_object_avoidance"),
                            value_type=bool,
                        ),
                        "objects_topic": LaunchConfiguration("objects_topic"),
                        "object_avoidance_max_age_sec": ParameterValue(
                            LaunchConfiguration("object_avoidance_max_age_sec"),
                            value_type=float,
                        ),
                        "object_keepout_half_width_m": ParameterValue(
                            LaunchConfiguration("object_keepout_half_width_m"),
                            value_type=float,
                        ),
                        "object_keepout_half_depth_m": ParameterValue(
                            LaunchConfiguration("object_keepout_half_depth_m"),
                            value_type=float,
                        ),
                        "object_avoidance_lookahead_m": ParameterValue(
                            LaunchConfiguration("object_avoidance_lookahead_m"),
                            value_type=float,
                        ),
                        "object_avoidance_turn_rps": ParameterValue(
                            LaunchConfiguration("object_avoidance_turn_rps"),
                            value_type=float,
                        ),
                    }
                ],
            ),
            Node(
                package="arena_lightweight_control",
                executable="arena_tk_ui",
                name="arena_tk_ui",
                output="screen",
                parameters=[
                    {
                        "map_yaml": map_yaml,
                        "scan_topic": LaunchConfiguration("scan_topic"),
                        "scan_yaw_offset_rad": ParameterValue(
                            LaunchConfiguration("base_scan_yaw"),
                            value_type=float,
                        ),
                        "status_topic": "/arena_lightweight/status",
                        "goal_command_topic": LaunchConfiguration("goal_command_topic"),
                        "pose_command_topic": LaunchConfiguration("pose_command_topic"),
                        "control_command_topic": LaunchConfiguration("control_command_topic"),
                        "gripper_command_topic": LaunchConfiguration("gripper_command_topic"),
                        "gripper_joint_command_topic": LaunchConfiguration(
                            "gripper_joint_command_topic"
                        ),
                        "scan_display_max_points": ParameterValue(
                            LaunchConfiguration("scan_display_max_points"),
                            value_type=int,
                        ),
                    }
                ],
                condition=IfCondition(launch_python_ui),
            ),
        ]
    )
