from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPO_ROOT / 'src' / 'robot_bringup' / 'config' / 'real.yaml'


def generate_launch_description():
    config = LaunchConfiguration('config')
    return LaunchDescription([
        DeclareLaunchArgument(
            'config',
            default_value=str(DEFAULT_CONFIG),
        ),
        Node(
            package='robot_perception',
            executable='camera_stitch_node',
            name='camera_stitch_node',
            output='screen',
            parameters=[config],
        ),
        Node(
            package='robot_perception',
            executable='yolo_seg_node',
            name='yolo_seg_node',
            output='screen',
            parameters=[config],
        ),
        Node(
            package='robot_semantic_mapping',
            executable='semantic_mapper_node',
            name='semantic_mapper_node',
            output='screen',
            parameters=[config],
        ),
    ])
