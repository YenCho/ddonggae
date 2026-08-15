from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPO_ROOT / 'src' / 'robot_bringup' / 'config' / 'real.yaml'
DEFAULT_MODEL = REPO_ROOT / 'data' / 'yolo' / 'weights' / 'seg1000.pt'


def generate_launch_description():
    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('robot_bringup'),
            '/launch/real_bringup.launch.py',
        ]),
        launch_arguments={
            'config': LaunchConfiguration('config'),
            'model_path': LaunchConfiguration('model_path'),
        }.items(),
    )
    return LaunchDescription([
        DeclareLaunchArgument(
            'config',
            default_value=str(DEFAULT_CONFIG),
        ),
        DeclareLaunchArgument(
            'model_path',
            default_value=str(DEFAULT_MODEL),
        ),
        DeclareLaunchArgument('target_shape_class', default_value='dodecahedron'),
        DeclareLaunchArgument('target_fruit_class', default_value='banana'),
        bringup,
    ])
