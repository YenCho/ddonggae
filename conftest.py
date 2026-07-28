"""Pytest bootstrap.

The ROS 2 packages in this repository live under three part directories
(``hardware/ros2``, ``navigation/ros2``, ``third_party``) rather than a single
``src/``, and the shared perception library is a plain module in
``perception/``. When the packages are built with colcon they end up on
``PYTHONPATH`` normally; these tests are deliberately runnable **without** a ROS
installation, so the same locations are added here instead.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

_PATHS = (
    REPO_ROOT / "perception",                          # fieldlib, geometry
    REPO_ROOT / "perception" / "tools",                # stitch_calibrator
    REPO_ROOT / "navigation" / "ros2" / "arena_lightweight_control",
    REPO_ROOT / "hardware" / "ros2" / "robot_hardware",
    REPO_ROOT / "hardware" / "ros2" / "robot_bringup",
)

for _p in _PATHS:
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
