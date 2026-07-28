from glob import glob

from setuptools import find_packages, setup

package_name = 'arena_lightweight_control'

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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Team 14',
    maintainer_email='yencho929@gmail.com',
    description='Nav2-free known-map LiDAR localization and lightweight arena control.',
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'arena_control_node = arena_lightweight_control.arena_control_node:main',
            'arena_tk_ui = arena_lightweight_control.tk_ui:main',
        ],
    },
)
