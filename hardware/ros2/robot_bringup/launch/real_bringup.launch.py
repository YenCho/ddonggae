from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPO_ROOT / 'src' / 'robot_bringup' / 'config' / 'real.yaml'
DEFAULT_MODEL = REPO_ROOT / 'data' / 'yolo' / 'weights' / 'seg1000.pt'


def realsense_node(name, serial_arg, enable_imu_arg=None, use_rsusb_backend=False):
    namespace = f'camera/{name}'
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
    remappings = [
        (f'/{namespace}/{name}/color/image_raw', f'/{namespace}/color/image_raw'),
        (f'/{namespace}/{name}/color/camera_info', f'/{namespace}/color/camera_info'),
        (
            f'/{namespace}/{name}/aligned_depth_to_color/image_raw',
            f'/{namespace}/aligned_depth_to_color/image_raw',
        ),
        (f'/{namespace}/{name}/gyro/sample', f'/{namespace}/gyro/sample'),
        (f'/{namespace}/{name}/accel/sample', f'/{namespace}/accel/sample'),
    ]
    if enable_imu_arg is not None:
        remappings.append((f'/{namespace}/{name}/imu', '/imu/data'))

    additional_env = None
    if use_rsusb_backend:
        additional_env = {
            'LD_LIBRARY_PATH': EnvironmentVariable(
                'REALSENSE_RSUSB_LD_LIBRARY_PATH',
                default_value=EnvironmentVariable('LD_LIBRARY_PATH', default_value=''),
            )
        }

    return Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        namespace=namespace,
        name=name,
        output='screen',
        parameters=[{
            'camera_name': name,
            'camera_namespace': namespace,
            'serial_no': LaunchConfiguration(serial_arg),
            'enable_color': True,
            'enable_depth': True,
            'rgb_camera.color_profile': LaunchConfiguration('camera_profile'),
            'depth_module.depth_profile': LaunchConfiguration('camera_profile'),
            'align_depth.enable': True,
            'enable_infra1': False,
            'enable_infra2': False,
            'enable_motion': enable_imu,
            'enable_gyro': enable_imu,
            'enable_accel': enable_imu,
            'gyro_fps': 200,
            'accel_fps': 63,
            'unite_imu_method': unite_imu_method,
            'hold_back_imu_for_frames': False,
            'angular_velocity_cov': 0.01,
            'linear_accel_cov': 0.01,
        }],
        remappings=remappings,
        additional_env=additional_env,
        condition=IfCondition(LaunchConfiguration('launch_cameras')),
    )


def generate_launch_description():
    config = LaunchConfiguration('config')
    return LaunchDescription([
        DeclareLaunchArgument(
            'config',
            default_value=str(DEFAULT_CONFIG),
            description='Robot real-hardware YAML configuration.',
        ),
        DeclareLaunchArgument('base_dry_run', default_value='false'),
        DeclareLaunchArgument('gripper_dry_run', default_value='false'),
        DeclareLaunchArgument(
            'gripper_serial_port',
            default_value='/dev/serial/by-path/platform-3610000.usb-usb-0:2.3.3:1.0',
        ),
        DeclareLaunchArgument(
            'require_loaded_pwm_calibration',
            default_value='true',
        ),
        DeclareLaunchArgument('launch_cameras', default_value='true'),
        DeclareLaunchArgument('launch_yolo', default_value='true'),
        DeclareLaunchArgument('launch_semantic_mapper', default_value='true'),
        DeclareLaunchArgument('enable_top_imu', default_value='false'),
        DeclareLaunchArgument('enable_bottom_imu', default_value='true'),
        DeclareLaunchArgument('cam_top_serial', default_value='_030422070364'),
        DeclareLaunchArgument('cam_bottom_serial', default_value='_112322074553'),
        DeclareLaunchArgument('camera_profile', default_value='640x480x30'),
        DeclareLaunchArgument(
            'model_path',
            default_value=str(DEFAULT_MODEL),
        ),
        realsense_node('top', 'cam_top_serial', 'enable_top_imu'),
        realsense_node(
            'bottom',
            'cam_bottom_serial',
            'enable_bottom_imu',
            use_rsusb_backend=True,
        ),
        Node(
            package='robot_hardware',
            executable='motor_bridge_node',
            name='motor_bridge_node',
            output='screen',
            parameters=[config, {'dry_run': LaunchConfiguration('base_dry_run')}],
        ),
        Node(
            package='robot_hardware',
            executable='gripper_bridge_node',
            name='gripper_bridge_node',
            output='screen',
            parameters=[
                config,
                {
                    'dry_run': LaunchConfiguration('gripper_dry_run'),
                    'serial_port': LaunchConfiguration('gripper_serial_port'),
                },
            ],
        ),
        Node(
            package='robot_perception',
            executable='camera_stitch_node',
            name='camera_stitch_node',
            output='screen',
            parameters=[config],
            condition=IfCondition(LaunchConfiguration('launch_cameras')),
        ),
        Node(
            package='robot_perception',
            executable='yolo_seg_node',
            name='yolo_seg_node',
            output='screen',
            parameters=[
                config,
                {'model_path': LaunchConfiguration('model_path')},
            ],
            condition=IfCondition(LaunchConfiguration('launch_yolo')),
        ),
        Node(
            package='robot_semantic_mapping',
            executable='semantic_mapper_node',
            name='semantic_mapper_node',
            output='screen',
            parameters=[
                config,
                {'model_path': LaunchConfiguration('model_path')},
            ],
            condition=IfCondition(LaunchConfiguration('launch_semantic_mapper')),
        ),
        Node(
            package='robot_task_planner',
            executable='match_state_machine_node',
            name='match_state_machine_node',
            output='screen',
            parameters=[
                config,
                {
                    'require_loaded_pwm_calibration': LaunchConfiguration(
                        'require_loaded_pwm_calibration'
                    ),
                },
            ],
        ),
        Node(
            package='robot_bringup',
            executable='dashboard_node',
            name='dashboard_node',
            output='screen',
            parameters=[config],
        ),
    ])
