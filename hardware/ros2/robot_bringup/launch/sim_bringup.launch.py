from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BEST_MODEL = REPO_ROOT / "perception" / "models" / "best.pt"
DEFAULT_FALLBACK_YOLO_MODEL = REPO_ROOT / "perception" / "models" / "260516.pt"
DEFAULT_YOLO_MODEL = DEFAULT_BEST_MODEL if DEFAULT_BEST_MODEL.exists() else DEFAULT_FALLBACK_YOLO_MODEL
DEFAULT_POLICY = (
    REPO_ROOT
    / "logs"
    / "2026-05"
    / "isaacsim"
    / "rl_games"
    / "mk1_arena_direct"
    / "2026-05-24_12-09-15"
    / "nn"
    / "mk1_arena_direct_torchscript.pt"
)
DEFAULT_ROS_DISTRO = "humble"
DEFAULT_ROS_PYTHON_VERSION = "3.10"
DEFAULT_ISAAC_PYTHON = Path.home() / "isaacsim" / "python.sh"
DEFAULT_SCENE_SCRIPT = REPO_ROOT / "sim" / "isaacsim" / "scripts" / "create_mobile_manipulator_scene.py"
WORKSPACE_PREFIXES = [
    REPO_ROOT / "install" / name
    for name in (
        "robot_bringup",
        "example_nav2",
        "robot_description",
        "robot_hardware",
        "robot_perception",
        "robot_semantic_mapping",
        "robot_task_planner",
    )
]
AMENT_PREFIX = ":".join(str(path) for path in WORKSPACE_PREFIXES) + f":/opt/ros/{DEFAULT_ROS_DISTRO}"
PYTHON_SITE_PREFIX = (
    REPO_ROOT
    / "install"
    / "example_nav2"
    / "lib"
    / f"python{DEFAULT_ROS_PYTHON_VERSION}"
    / "site-packages"
)
VENDOR_LIB_PATHS = sorted(str(path) for path in Path(f"/opt/ros/{DEFAULT_ROS_DISTRO}/opt").glob("*_vendor/lib"))
LD_LIBRARY_PATH = ":".join(
    [
        str(REPO_ROOT / "install" / "example_nav2" / "lib"),
        f"/opt/ros/{DEFAULT_ROS_DISTRO}/lib",
        *VENDOR_LIB_PATHS,
    ]
)


def generate_launch_description():
    robot_description_dir = get_package_share_directory("robot_description")
    urdf_path = Path(robot_description_dir) / "urdf" / "mobile_manipulator_collision.urdf"
    robot_description = urdf_path.read_text(encoding="utf-8")

    use_sim_time = LaunchConfiguration("use_sim_time")
    launch_isaac = LaunchConfiguration("launch_isaac")
    use_robot_state_publisher = LaunchConfiguration("use_robot_state_publisher")
    use_joint_state_publisher = LaunchConfiguration("use_joint_state_publisher")
    use_nav2 = LaunchConfiguration("use_nav2")
    use_ai = LaunchConfiguration("use_ai")
    use_yolo = LaunchConfiguration("use_yolo")
    use_rviz = LaunchConfiguration("use_rviz")
    yolo_model = LaunchConfiguration("yolo_model")
    policy_path = LaunchConfiguration("policy_path")
    target_class = LaunchConfiguration("target_class")

    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("example_nav2"), "launch", "nav2.launch.py"])
        ),
        condition=IfCondition(use_nav2),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "ros_distro": DEFAULT_ROS_DISTRO,
            "use_rviz": use_rviz,
            "rviz_config": PathJoinSubstitution(
                [FindPackageShare("example_nav2"), "rviz", "mobile_manipulator_sim.rviz"]
            ),
            "use_yolo_rgb_detector": use_yolo,
            "use_dual_camera_yolo_detector": "false",
            "use_pinhole_object_projector": "true",
            "use_rgbd_yolo_detector": "false",
            "use_object_map_overlay": "false",
            "use_pick_and_place_task_manager": "false",
            "use_mk1_wheel_joint_bridge": "true",
            "mk1_wheel_left_joint_name": "left_wheel_spin",
            "mk1_wheel_right_joint_name": "right_wheel_spin",
            "scan_topic": "/laser_scan",
            "rgb_topic": "/camera_19/rgb",
            "depth_topic": "/camera_19/depth",
            "camera_info_topic": "/camera_19/camera_info",
            "yolo_model": yolo_model,
            "yolo_rgb_detections_topic": "/mk1/front_view/yolo/detections",
            "yolo_rgb_annotated_topic": "/mk1/front_view/yolo/annotated",
            "detector_debug_image_topic": "/detected_objects/debug_image",
            "pinhole_objects_topic": "/detected_objects/map_objects",
            "object_markers_topic": "/detected_objects/map_markers",
            "pick_target_class": target_class,
            "pick_gripper_joint_names": "left_finger_slide",
            "pick_gripper_open_positions": "0.045",
            "pick_gripper_close_positions": "0.0",
        }.items(),
    )

    ai_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("example_nav2"), "launch", "mk1_rl_interface.launch.py"])
        ),
        condition=IfCondition(use_ai),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "use_rl_io_bridge": "true",
            "use_rl_policy_runner": "true",
            "use_mk1_wheel_joint_bridge": "false",
            "scan_topic": "/laser_scan",
            "odom_topic": "/chassis/odom",
            "detections_topic": "/mk1/front_view/yolo/detections",
            "joint_states_topic": "/joint_states",
            "cmd_vel_topic": "/cmd_vel",
            "gripper_topic": "/mk1/gripper_joint_command",
            "target_class": target_class,
            "detection_model_path": yolo_model,
            "policy_backend": "torchscript",
            "policy_path": policy_path,
            "policy_outputs_normalized": "true",
            "policy_device": "cpu",
            "gripper_joint_names": "left_finger_slide",
            "gripper_open_positions": "0.045",
            "gripper_close_positions": "0.0",
        }.items(),
    )

    return LaunchDescription(
        [
            SetEnvironmentVariable(
                "PATH",
                "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            ),
            SetEnvironmentVariable("ROS_DISTRO", DEFAULT_ROS_DISTRO),
            SetEnvironmentVariable("AMENT_PREFIX_PATH", AMENT_PREFIX),
            SetEnvironmentVariable("LD_LIBRARY_PATH", LD_LIBRARY_PATH),
            SetEnvironmentVariable(
                "PYTHONPATH",
                f"{PYTHON_SITE_PREFIX}:/opt/ros/{DEFAULT_ROS_DISTRO}/lib/python{DEFAULT_ROS_PYTHON_VERSION}/site-packages",
            ),
            SetEnvironmentVariable("VIRTUAL_ENV", ""),
            SetEnvironmentVariable("PYTHONHOME", ""),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument(
                "launch_isaac",
                default_value="false",
                description=(
                    "Start Isaac Sim with the generated mobile_manipulator scene. "
                    "Default is false so ROS bringup can attach to an already-open stage."
                ),
            ),
            DeclareLaunchArgument("isaac_python", default_value=str(DEFAULT_ISAAC_PYTHON)),
            DeclareLaunchArgument("isaac_scene_script", default_value=str(DEFAULT_SCENE_SCRIPT)),
            DeclareLaunchArgument("use_robot_state_publisher", default_value="true"),
            DeclareLaunchArgument("use_joint_state_publisher", default_value="true"),
            DeclareLaunchArgument("use_nav2", default_value="true"),
            DeclareLaunchArgument("use_ai", default_value="false"),
            DeclareLaunchArgument("use_yolo", default_value="true"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("target_class", default_value=""),
            DeclareLaunchArgument("yolo_model", default_value=str(DEFAULT_YOLO_MODEL)),
            DeclareLaunchArgument("policy_path", default_value=str(DEFAULT_POLICY)),
            ExecuteProcess(
                cmd=[
                    "env",
                    "-u",
                    "PYTHONPATH",
                    "-u",
                    "PYTHONHOME",
                    "-u",
                    "VIRTUAL_ENV",
                    f"ROS_DISTRO={DEFAULT_ROS_DISTRO}",
                    f"AMENT_PREFIX_PATH=/opt/ros/{DEFAULT_ROS_DISTRO}",
                    f"LD_LIBRARY_PATH=/opt/ros/{DEFAULT_ROS_DISTRO}/lib",
                    LaunchConfiguration("isaac_python"),
                    LaunchConfiguration("isaac_scene_script"),
                    "--ros-distro",
                    DEFAULT_ROS_DISTRO,
                    "--overwrite",
                    "--open-gui",
                ],
                condition=IfCondition(launch_isaac),
                output="screen",
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                condition=IfCondition(use_robot_state_publisher),
                parameters=[
                    {
                        "use_sim_time": ParameterValue(use_sim_time, value_type=bool),
                        "robot_description": robot_description,
                    }
                ],
                output="screen",
            ),
            Node(
                package="joint_state_publisher",
                executable="joint_state_publisher",
                name="joint_state_publisher",
                condition=IfCondition(use_joint_state_publisher),
                parameters=[
                    {
                        "use_sim_time": ParameterValue(use_sim_time, value_type=bool),
                        "source_list": ["/mk1/gripper_joint_command", "/mk1/wheel_joint_command"],
                        "rate": 30,
                    }
                ],
                output="screen",
            ),
            nav2_launch,
            ai_launch,
        ]
    )
