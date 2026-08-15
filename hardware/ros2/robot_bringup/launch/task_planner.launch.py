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
            package='robot_task_planner',
            executable='match_state_machine_node',
            name='match_state_machine_node',
            output='screen',
            parameters=[config],
        ),
    ])
