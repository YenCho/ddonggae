from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


DEFAULT_CONFIG = Path(get_package_share_directory("robot_bringup")) / "config" / "real.yaml"


def realsense_node(
    name,
    serial_arg,
    rgb_topic,
    depth_topic,
    camera_info_topic,
    enable_depth_arg,
    enable_imu_arg=None,
    use_rsusb_backend=False,
):
    namespace = f"camera/{name}"
    raw_prefix = f"/{namespace}/{name}"
    enable_imu = (
        ParameterValue(LaunchConfiguration(enable_imu_arg), value_type=bool)
        if enable_imu_arg is not None
        else False
    )
    unite_imu_method = (
        ParameterValue(
            PythonExpression(
                [
                    "2 if '",
                    LaunchConfiguration(enable_imu_arg),
                    "'.lower() in ('1', 'true', 'yes', 'on') else 0",
                ]
            ),
            value_type=int,
        )
        if enable_imu_arg is not None
        else 0
    )
    enable_depth = ParameterValue(
        LaunchConfiguration(enable_depth_arg),
        value_type=bool,
    )
    remappings = [
        (f"{raw_prefix}/color/image_raw", rgb_topic),
        (f"{raw_prefix}/color/camera_info", camera_info_topic),
        (f"{raw_prefix}/aligned_depth_to_color/image_raw", depth_topic),
        (f"{raw_prefix}/gyro/sample", f"/{namespace}/gyro/sample"),
        (f"{raw_prefix}/accel/sample", f"/{namespace}/accel/sample"),
    ]
    if enable_imu_arg is not None:
        remappings.append((f"{raw_prefix}/imu", "/imu/data"))

    additional_env = None
    if use_rsusb_backend:
        additional_env = {
            "LD_LIBRARY_PATH": EnvironmentVariable(
                "REALSENSE_RSUSB_LD_LIBRARY_PATH",
                default_value=EnvironmentVariable("LD_LIBRARY_PATH", default_value=""),
            )
        }

    return Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        namespace=namespace,
        name=name,
        output="screen",
        parameters=[
            {
                "camera_name": name,
                "camera_namespace": namespace,
                "serial_no": LaunchConfiguration(serial_arg),
                "enable_color": True,
                "enable_depth": enable_depth,
                "rgb_camera.color_profile": LaunchConfiguration("camera_rgb_profile"),
                # 상/하 캠이 auto 로 따로 수렴하면 스티치 이음매에서 색·밝기가
                # 갈라진다. 둘 다 **동일 고정값**으로 잠근다.
                # 2026-07-20 실측(stitch_sweep 81페어): WB만 잠그고 노출은 auto로
                # 둔 탓에 마스트업에서 이음매 밝기차 Y 중앙 +12.4 (top이 더 밝음).
                # → 노출/게인도 함께 잠금. 값 찾는 법은
                #   perception/docs/stitching-and-calibration.md §1
                "rgb_camera.enable_auto_white_balance": False,
                "rgb_camera.white_balance": LaunchConfiguration("camera_white_balance"),
                "rgb_camera.enable_auto_exposure": LaunchConfiguration(
                    "camera_auto_exposure"
                ),
                "rgb_camera.exposure": LaunchConfiguration("camera_exposure"),
                "rgb_camera.gain": LaunchConfiguration("camera_gain"),
                # 형광등 깜빡임(밴딩) 방지 — 국내 60Hz
                "rgb_camera.power_line_frequency": LaunchConfiguration(
                    "camera_power_line_frequency"
                ),
                "depth_module.depth_profile": LaunchConfiguration("camera_depth_profile"),
                "align_depth.enable": enable_depth,
                "enable_infra1": False,
                "enable_infra2": False,
                "enable_motion": enable_imu,
                "enable_gyro": enable_imu,
                "enable_accel": enable_imu,
                "gyro_fps": 200,
                "accel_fps": 63,
                "unite_imu_method": unite_imu_method,
                "hold_back_imu_for_frames": False,
                "angular_velocity_cov": 0.01,
                "linear_accel_cov": 0.01,
            }
        ],
        remappings=remappings,
        additional_env=additional_env,
        condition=IfCondition(LaunchConfiguration("launch_cameras")),
    )


def generate_launch_description():
    config = LaunchConfiguration("config")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=str(DEFAULT_CONFIG),
                description="Robot real-hardware YAML configuration.",
            ),
            DeclareLaunchArgument("launch_cameras", default_value="true"),
            DeclareLaunchArgument("launch_lidar", default_value="true"),
            DeclareLaunchArgument("launch_scan_watchdog", default_value="true"),
            DeclareLaunchArgument("launch_motor", default_value="true"),
            DeclareLaunchArgument(
                "drive_type",
                default_value="diff",
                description="Base drive type: diff or mecanum.",
            ),
            DeclareLaunchArgument("launch_cmd_vel_mux", default_value="true"),
            DeclareLaunchArgument("launch_gripper", default_value="true"),
            DeclareLaunchArgument("launch_gripper_command_bridge", default_value="true"),
            DeclareLaunchArgument("base_dry_run", default_value="false"),
            DeclareLaunchArgument("gripper_dry_run", default_value="false"),
            # [2026-07-23] by-path → by-id glob (lightweight_real.launch.py 와 동일 사유).
            # 폐기: default_value="/dev/serial/by-path/platform-3610000.usb-usb-0:2.3.3:1.0"
            DeclareLaunchArgument(
                "gripper_serial_port",
                default_value="/dev/serial/by-id/usb-ROBOTIS_OpenRB-150*-if00",
            ),
            DeclareLaunchArgument("enable_top_imu", default_value="false"),
            DeclareLaunchArgument("enable_bottom_imu", default_value="true"),
            DeclareLaunchArgument("enable_camera_depth", default_value="true"),
            DeclareLaunchArgument("cam_top_serial", default_value="_030422070364"),
            DeclareLaunchArgument("cam_bottom_serial", default_value="_112322074553"),
            DeclareLaunchArgument("camera_profile", default_value="640x480x30"),
            # RGB/depth 프로파일 분리 (2026-07-21). 기본 = FHD RGB + 848x480 depth.
            # 구형 640 전체로 돌리려면:
            #   camera_rgb_profile:=640x480x30 camera_depth_profile:=640x480x30
            # ⚠ depth 는 D435 최대 1280x720 — depth 프로파일에 1080p 를 주면
            #   협상이 실패한다. aligned depth 는 자동으로 컬러 해상도로 리샘플.
            # ⚠ depth 는 반드시 16:9 계열(848x480/1280x720/640x360)로 둘 것.
            #   4:3(640x480)은 스테레오 이미저를 가로 크롭해 H-FOV가 ~87°→~65°로
            #   줄고, RGB(69°)보다 좁아져 FHD 프레임 좌우 가장자리의 aligned
            #   depth 가 통째로 비었다 — 가장자리 물체 위치추정 실패의 원인
            #   (2026-07-21). 848x480 은 full-FOV + Intel 권장 최적 해상도.
            DeclareLaunchArgument("camera_rgb_profile", default_value="1920x1080x15"),
            DeclareLaunchArgument("camera_depth_profile", default_value="848x480x15"),
            # D435 WB 범위 2800~6500K. 색이 노랗면 ↓ 파랗면 ↑
            # (realsense 쪽 파라미터 선언이 double 이라 정수 문자열이면 적용 거부됨)
            # 기본값 근거: 2026-07-21 저녁 현장 확정 — stitch_live_view 슬라이더 3514
            # (표시값+2800=실제) ≈ 6300K 가 최적. 7/21 저녁의 3500 은 슬라이더
            # 표시값을 실제값으로 오기입한 것 (구 wb4000/exp312 은 7/21 새벽
            # 108장 스윕값 — stitch_sweep_20260721_040659_eval 참조)
            DeclareLaunchArgument("camera_white_balance", default_value="6300.0"),
            # 노출 잠금. auto 로 두면 두 캠이 따로 수렴해 이음매가 갈라진다.
            # 값은 조명마다 다르므로 stitch_calibrator 로 현장에서 찾아 넣을 것
            # (Y가 100~140 근처, 흰 물체가 안 날아가는 최대값).
            DeclareLaunchArgument("camera_auto_exposure", default_value="false"),
            DeclareLaunchArgument("camera_exposure", default_value="320"),
            DeclareLaunchArgument("camera_gain", default_value="64"),
            DeclareLaunchArgument("camera_power_line_frequency", default_value="2"),
            DeclareLaunchArgument("cmd_vel_topic", default_value="/cmd_vel_motor"),
            DeclareLaunchArgument("direct_cmd_vel_topic", default_value="/cmd_vel_direct"),
            DeclareLaunchArgument("nav_cmd_vel_topic", default_value="/cmd_vel_nav"),
            DeclareLaunchArgument("default_cmd_vel_topic", default_value="/cmd_vel"),
            DeclareLaunchArgument("odom_topic", default_value="/chassis/odom"),
            DeclareLaunchArgument("odom_frame", default_value="odom"),
            DeclareLaunchArgument("base_frame", default_value="base_link"),
            DeclareLaunchArgument("publish_odom_tf", default_value="true"),
            DeclareLaunchArgument("scan_topic", default_value="/laser_scan"),
            DeclareLaunchArgument("scan_alias_topics", default_value="/scan"),
            DeclareLaunchArgument("raw_scan_topic", default_value="/scan_raw"),
            DeclareLaunchArgument("extra_raw_scan_topics", default_value=""),
            DeclareLaunchArgument("lidar_frame", default_value="base_scan"),
            DeclareLaunchArgument("lidar_serial_port", default_value="/dev/rplidar"),
            DeclareLaunchArgument("lidar_serial_baudrate", default_value="256000"),
            DeclareLaunchArgument("lidar_channel_type", default_value="serial"),
            DeclareLaunchArgument("lidar_scan_mode", default_value=""),
            DeclareLaunchArgument("lidar_inverted", default_value="false"),
            DeclareLaunchArgument("lidar_angle_compensate", default_value="true"),
            DeclareLaunchArgument("lidar_scan_frequency", default_value="10.0"),
            DeclareLaunchArgument("scan_watchdog_publish_rate", default_value="15.0"),
            DeclareLaunchArgument("scan_watchdog_raw_timeout", default_value="0.25"),
            DeclareLaunchArgument("scan_watchdog_hold_stale_sec", default_value="0.0"),
            DeclareLaunchArgument("scan_watchdog_qos_depth", default_value="50"),
            DeclareLaunchArgument("scan_watchdog_publish_empty_scan", default_value="false"),
            DeclareLaunchArgument("scan_watchdog_publish_on_receive", default_value="true"),
            DeclareLaunchArgument("scan_watchdog_republish_fresh_scan", default_value="false"),
            DeclareLaunchArgument("scan_watchdog_restamp_scans", default_value="false"),
            DeclareLaunchArgument("base_scan_x", default_value="0.0"),
            DeclareLaunchArgument("base_scan_y", default_value="0.0"),
            DeclareLaunchArgument("base_scan_z", default_value="0.262"),
            DeclareLaunchArgument("base_scan_yaw", default_value="3.14159265359"),
            DeclareLaunchArgument("base_scan_pitch", default_value="0.0"),
            DeclareLaunchArgument("base_scan_roll", default_value="0.0"),
            DeclareLaunchArgument(
                "gripper_joint_command_topic",
                default_value="/mk1/gripper_joint_command",
            ),
            DeclareLaunchArgument(
                "gripper_command_topic",
                default_value="/gripper/command",
            ),
            DeclareLaunchArgument(
                "gripper_status_topic",
                default_value="/gripper/state",
            ),
            realsense_node(
                "top",
                "cam_top_serial",
                "/camera_19/rgb",
                "/camera_19/depth",
                "/camera_19/camera_info",
                "enable_camera_depth",
                "enable_top_imu",
                use_rsusb_backend=False,
            ),
            realsense_node(
                "bottom",
                "cam_bottom_serial",
                "/camera_54/rgb",
                "/camera_54/depth",
                "/camera_54/camera_info",
                "enable_camera_depth",
                "enable_bottom_imu",
                use_rsusb_backend=True,
            ),
            Node(
                package="sllidar_ros2",
                executable="sllidar_node",
                name="sllidar_node",
                output="screen",
                respawn=True,
                respawn_delay=2.0,
                parameters=[
                    {
                        "channel_type": LaunchConfiguration("lidar_channel_type"),
                        "serial_port": LaunchConfiguration("lidar_serial_port"),
                        "serial_baudrate": ParameterValue(
                            LaunchConfiguration("lidar_serial_baudrate"),
                            value_type=int,
                        ),
                        "frame_id": LaunchConfiguration("lidar_frame"),
                        "inverted": ParameterValue(
                            LaunchConfiguration("lidar_inverted"),
                            value_type=bool,
                        ),
                        "angle_compensate": ParameterValue(
                            LaunchConfiguration("lidar_angle_compensate"),
                            value_type=bool,
                        ),
                        "scan_frequency": ParameterValue(
                            LaunchConfiguration("lidar_scan_frequency"),
                            value_type=float,
                        ),
                        "scan_mode": LaunchConfiguration("lidar_scan_mode"),
                    }
                ],
                remappings=[("scan", LaunchConfiguration("raw_scan_topic"))],
                condition=IfCondition(LaunchConfiguration("launch_lidar")),
            ),
            Node(
                package="robot_bringup",
                executable="laser_scan_watchdog_node",
                name="laser_scan_watchdog_node",
                output="screen",
                parameters=[
                    {
                        "scan_in_topic": LaunchConfiguration("raw_scan_topic"),
                        "extra_scan_in_topics": LaunchConfiguration("extra_raw_scan_topics"),
                        "scan_out_topic": LaunchConfiguration("scan_topic"),
                        "extra_scan_out_topics": LaunchConfiguration("scan_alias_topics"),
                        "frame_id": LaunchConfiguration("lidar_frame"),
                        "qos_depth": ParameterValue(
                            LaunchConfiguration("scan_watchdog_qos_depth"),
                            value_type=int,
                        ),
                        "publish_rate_hz": ParameterValue(
                            LaunchConfiguration("scan_watchdog_publish_rate"),
                            value_type=float,
                        ),
                        "raw_fresh_timeout_sec": ParameterValue(
                            LaunchConfiguration("scan_watchdog_raw_timeout"),
                            value_type=float,
                        ),
                        "hold_stale_scan_sec": ParameterValue(
                            LaunchConfiguration("scan_watchdog_hold_stale_sec"),
                            value_type=float,
                        ),
                        "publish_empty_scan": ParameterValue(
                            LaunchConfiguration("scan_watchdog_publish_empty_scan"),
                            value_type=bool,
                        ),
                        "publish_on_receive": ParameterValue(
                            LaunchConfiguration("scan_watchdog_publish_on_receive"),
                            value_type=bool,
                        ),
                        "republish_fresh_scan": ParameterValue(
                            LaunchConfiguration("scan_watchdog_republish_fresh_scan"),
                            value_type=bool,
                        ),
                        "restamp_scans": ParameterValue(
                            LaunchConfiguration("scan_watchdog_restamp_scans"),
                            value_type=bool,
                        ),
                    }
                ],
                condition=IfCondition(LaunchConfiguration("launch_scan_watchdog")),
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="static_base_scan_tf",
                arguments=[
                    LaunchConfiguration("base_scan_x"),
                    LaunchConfiguration("base_scan_y"),
                    LaunchConfiguration("base_scan_z"),
                    LaunchConfiguration("base_scan_yaw"),
                    LaunchConfiguration("base_scan_pitch"),
                    LaunchConfiguration("base_scan_roll"),
                    LaunchConfiguration("base_frame"),
                    LaunchConfiguration("lidar_frame"),
                ],
                output="screen",
                condition=IfCondition(LaunchConfiguration("launch_scan_watchdog")),
            ),
            Node(
                package="robot_bringup",
                executable="cmd_vel_mux_node",
                name="cmd_vel_mux_node",
                output="screen",
                parameters=[
                    {
                        "direct_topic": LaunchConfiguration("direct_cmd_vel_topic"),
                        "nav_topic": LaunchConfiguration("nav_cmd_vel_topic"),
                        "cmd_vel_topic": LaunchConfiguration("default_cmd_vel_topic"),
                        "output_topic": LaunchConfiguration("cmd_vel_topic"),
                    }
                ],
                condition=IfCondition(LaunchConfiguration("launch_cmd_vel_mux")),
            ),
            Node(
                package="robot_hardware",
                executable="motor_bridge_node",
                name="motor_bridge_node",
                output="screen",
                parameters=[
                    config,
                    {
                        "dry_run": ParameterValue(
                            LaunchConfiguration("base_dry_run"),
                            value_type=bool,
                        ),
                        "cmd_vel_topic": LaunchConfiguration("cmd_vel_topic"),
                        "odom_topic": LaunchConfiguration("odom_topic"),
                        "odom_frame": LaunchConfiguration("odom_frame"),
                        "base_frame": LaunchConfiguration("base_frame"),
                        "publish_tf": ParameterValue(
                            LaunchConfiguration("publish_odom_tf"),
                            value_type=bool,
                        ),
                    },
                ],
                condition=IfCondition(
                    PythonExpression(
                        [
                            "'",
                            LaunchConfiguration("launch_motor"),
                            "' == 'true' and '",
                            LaunchConfiguration("drive_type"),
                            "' != 'mecanum'",
                        ]
                    )
                ),
            ),
            Node(
                package="robot_hardware",
                executable="mecanum_bridge_node",
                name="mecanum_bridge_node",
                output="screen",
                parameters=[
                    config,
                    {
                        "dry_run": ParameterValue(
                            LaunchConfiguration("base_dry_run"),
                            value_type=bool,
                        ),
                        "cmd_vel_topic": LaunchConfiguration("cmd_vel_topic"),
                        "odom_topic": LaunchConfiguration("odom_topic"),
                        "odom_frame": LaunchConfiguration("odom_frame"),
                        "base_frame": LaunchConfiguration("base_frame"),
                        "publish_tf": ParameterValue(
                            LaunchConfiguration("publish_odom_tf"),
                            value_type=bool,
                        ),
                    },
                ],
                condition=IfCondition(
                    PythonExpression(
                        [
                            "'",
                            LaunchConfiguration("launch_motor"),
                            "' == 'true' and '",
                            LaunchConfiguration("drive_type"),
                            "' == 'mecanum'",
                        ]
                    )
                ),
            ),
            Node(
                package="robot_hardware",
                executable="gripper_bridge_node",
                name="gripper_bridge_node",
                output="screen",
                # [2026-07-24] respawn: 시리얼 글리치/USB 재열거로 노드가 죽어도
                # ROS 가 자동 재기동 (같은 파일의 sllidar_node 와 동일 패턴). 7/24
                # 실기에서 gripper_bridge_node 가 termios.error 로 죽은 뒤 되살아나지
                # 못해 경기 직전 그리퍼가 전멸한 사고의 2차 방어선(L2).
                # 근거/대책: docs/06-troubleshooting.md
                respawn=True,
                respawn_delay=2.0,
                parameters=[
                    config,
                    {
                        "dry_run": ParameterValue(
                            LaunchConfiguration("gripper_dry_run"),
                            value_type=bool,
                        ),
                        "command_topic": LaunchConfiguration("gripper_command_topic"),
                        "status_topic": LaunchConfiguration("gripper_status_topic"),
                        "serial_port": LaunchConfiguration("gripper_serial_port"),
                    },
                ],
                condition=IfCondition(LaunchConfiguration("launch_gripper")),
            ),
            Node(
                package="robot_bringup",
                executable="gripper_joint_command_bridge",
                name="gripper_joint_command_bridge",
                output="screen",
                parameters=[
                    {
                        "joint_command_topic": LaunchConfiguration(
                            "gripper_joint_command_topic"
                        ),
                        "string_command_topic": LaunchConfiguration(
                            "gripper_command_topic"
                        ),
                    }
                ],
                condition=IfCondition(
                    LaunchConfiguration("launch_gripper_command_bridge")
                ),
            ),
        ]
    )
