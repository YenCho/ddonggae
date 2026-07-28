from glob import glob

from setuptools import find_packages, setup

package_name = 'robot_bringup'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/web', glob('web/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Team 14',
    maintainer_email='yencho929@gmail.com',
    description='System launch files, common configuration, and dashboard node.',
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'cmd_vel_mux_node = robot_bringup.cmd_vel_mux_node:main',
            'dashboard_node = robot_bringup.dashboard_node:main',
            'gripper_joint_command_bridge = robot_bringup.gripper_joint_command_bridge:main',
            'laser_scan_watchdog_node = robot_bringup.laser_scan_watchdog_node:main',
        ],
    },
)
