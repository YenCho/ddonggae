from setuptools import find_packages, setup

package_name = 'robot_hardware'

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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Team 14',
    maintainer_email='yencho929@gmail.com',
    description='Arduino motor and OpenRB gripper hardware bridge nodes.',
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'motor_bridge_node = robot_hardware.motor_bridge_node:main',
            'mecanum_bridge_node = robot_hardware.mecanum_bridge_node:main',
            'gripper_bridge_node = robot_hardware.gripper_bridge_node:main',
            'gripper_position_probe = robot_hardware.gripper_position_probe:main',
            'gripper_record_limit = robot_hardware.gripper_record_limit:main',
        ],
    },
)
