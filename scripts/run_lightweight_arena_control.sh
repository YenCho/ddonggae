#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_DISTRO_ARG="${ROS_DISTRO_ARG:-humble}"
FASTDDS_PROFILE="${FASTDDS_PROFILE:-${REPO_ROOT}/hardware/ros2/robot_bringup/config/fastdds_udp_only.xml}"
LAUNCH_CAMERAS="${LAUNCH_CAMERAS:-false}"
# 아레나 맵 교체용. 비워두면 런치 기본값(패키지 share 의 stadium.yaml)을 쓴다.
# 예) MAP_YAML=$PWD/navigation/ros2/arena_lightweight_control/maps/demo2m.yaml
MAP_YAML="${MAP_YAML:-}"
UI_PORT="${UI_PORT:-18765}"
LAUNCH_PYTHON_UI="${LAUNCH_PYTHON_UI:-true}"
ENABLE_WEB_UI="${ENABLE_WEB_UI:-false}"
LOCALIZATION_MODE="${LOCALIZATION_MODE:-wall_range}"
LOCALIZATION_MAX_BEAMS="${LOCALIZATION_MAX_BEAMS:-24}"
LOCALIZATION_RESEED_PERIOD_SEC="${LOCALIZATION_RESEED_PERIOD_SEC:-2.5}"
LOCALIZATION_RESEED_XY_STEP_M="${LOCALIZATION_RESEED_XY_STEP_M:-0.32}"
LOCALIZATION_RESEED_YAW_STEP_RAD="${LOCALIZATION_RESEED_YAW_STEP_RAD:-0.35}"
LOCALIZATION_RESEED_MAX_BEAMS="${LOCALIZATION_RESEED_MAX_BEAMS:-16}"
LOCALIZATION_RESEED_REFINE_MAX_BEAMS="${LOCALIZATION_RESEED_REFINE_MAX_BEAMS:-16}"
SCAN_DISPLAY_MAX_POINTS="${SCAN_DISPLAY_MAX_POINTS:-120}"
XY_TOLERANCE_M="${XY_TOLERANCE_M:-0.15}"
YAW_TOLERANCE_RAD="${YAW_TOLERANCE_RAD:-0.40}"
FINAL_YAW_MIN_ANGULAR_RPS="${FINAL_YAW_MIN_ANGULAR_RPS:-0.0}"
GOAL_POSITION_LATCH="${GOAL_POSITION_LATCH:-true}"
XY_RELEASE_TOLERANCE_M="${XY_RELEASE_TOLERANCE_M:-0.25}"
XY_RELEASE_CONSECUTIVE_COUNT="${XY_RELEASE_CONSECUTIVE_COUNT:-3}"
NEAR_GOAL_SLOW_RADIUS_M="${NEAR_GOAL_SLOW_RADIUS_M:-0.10}"
NEAR_GOAL_SPEED_SCALE="${NEAR_GOAL_SPEED_SCALE:-0.5}"
ENABLE_OBJECT_AVOIDANCE="${ENABLE_OBJECT_AVOIDANCE:-false}"
OBJECT_AVOIDANCE_MAX_AGE_SEC="${OBJECT_AVOIDANCE_MAX_AGE_SEC:-0.75}"
OBJECT_KEEPOUT_HALF_WIDTH_M="${OBJECT_KEEPOUT_HALF_WIDTH_M:-0.15}"
OBJECT_KEEPOUT_HALF_DEPTH_M="${OBJECT_KEEPOUT_HALF_DEPTH_M:-0.20}"
OBJECT_AVOIDANCE_LOOKAHEAD_M="${OBJECT_AVOIDANCE_LOOKAHEAD_M:-0.35}"
OBJECT_AVOIDANCE_TURN_RPS="${OBJECT_AVOIDANCE_TURN_RPS:-0.35}"
# [2026-07-23] by-path -> by-id glob. Retired:
#   GRIPPER_SERIAL_PORT="${GRIPPER_SERIAL_PORT:-/dev/serial/by-path/platform-3610000.usb-usb-0:2.3.3:1.0}"
# A by-path name is the physical USB hub port, so it disappears the moment the
# board is plugged into a different one (field run 18:30: moving 2.3.3 -> 2.4.3
# made gripper_bridge_node fail to start).
GRIPPER_SERIAL_PORT="${GRIPPER_SERIAL_PORT:-/dev/serial/by-id/usb-ROBOTIS_OpenRB-150*-if00}"
# LiDAR mount yaw. Default pi = lidar 0deg facing robot rear (MK3 mount).
# If reassembly rotated the lidar 180deg, run
# scripts/dev/mecanum/40_lidar_orientation_check.py and set BASE_SCAN_YAW=0.0.
BASE_SCAN_YAW="${BASE_SCAN_YAW:-3.14159265359}"
# 비어 있지 않으면 arena 노드 IMU yaw prior 활성화 (예: /imu/data).
# 7/21 기본 /imu/data — 공란이면 arena IMU prior 미생성 → 90도 플립 취약
# (16:1x 실기 START 오적재의 원인). 의도적으로 끌 때만 IMU_TOPIC="" 지정.
IMU_TOPIC="${IMU_TOPIC:-/imu/data}"
# Mecanum conversion switch: DRIVE_TYPE=mecanum swaps the motor bridge and
# the goal controller together. Requires the mecanum_encoder_control
# firmware on the Arduino.
DRIVE_TYPE="${DRIVE_TYPE:-diff}"
if [[ "${DRIVE_TYPE}" == "mecanum" ]]; then
  CONTROLLER_TYPE="${CONTROLLER_TYPE:-mecanum}"
else
  CONTROLLER_TYPE="${CONTROLLER_TYPE:-diff_drive}"
fi

if [[ $# -gt 0 && ( "$1" == "humble" || "$1" == "jazzy" ) ]]; then
  ROS_DISTRO_ARG="$1"
  shift
fi

ROS_SETUP="/opt/ros/${ROS_DISTRO_ARG}/setup.bash"
LOCAL_SETUP="${REPO_ROOT}/install/local_setup.bash"
if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "missing ROS setup file: ${ROS_SETUP}" >&2
  exit 2
fi
if [[ ! -f "${LOCAL_SETUP}" ]]; then
  echo "missing workspace setup file: ${LOCAL_SETUP}" >&2
  echo "build it with: source ${ROS_SETUP} && colcon build --symlink-install --base-paths src" >&2
  exit 2
fi
if [[ ! -f "${FASTDDS_PROFILE}" ]]; then
  echo "missing Fast DDS profile: ${FASTDDS_PROFILE}" >&2
  exit 2
fi

cd "${REPO_ROOT}"
unset ROS_DISTRO AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH PYTHONHOME VIRTUAL_ENV
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

set +u
source "${ROS_SETUP}"
source "${LOCAL_SETUP}"
set -u

export ROS_DISTRO="${ROS_DISTRO_ARG}"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTDDS_PROFILE}"
unset RMW_FASTRTPS_USE_QOS_FROM_XML

for package in arena_lightweight_control robot_bringup robot_hardware sllidar_ros2; do
  if ! ros2 pkg prefix "${package}" >/dev/null 2>&1; then
    echo "missing ROS package: ${package}" >&2
    echo "build it with: source ${ROS_SETUP} && colcon build --symlink-install --base-paths src" >&2
    exit 4
  fi
done

echo "Starting Nav2-free lightweight arena control"
echo "Python Tk UI: ${LAUNCH_PYTHON_UI}"
echo "Web UI: ${ENABLE_WEB_UI} http://localhost:${UI_PORT}"
echo "launch_cameras=${LAUNCH_CAMERAS}"
echo "localization_mode=${LOCALIZATION_MODE}"
echo "localization_max_beams=${LOCALIZATION_MAX_BEAMS} scan_display_max_points=${SCAN_DISPLAY_MAX_POINTS}"
echo "reseed=${LOCALIZATION_RESEED_PERIOD_SEC}s xy_step=${LOCALIZATION_RESEED_XY_STEP_M} yaw_step=${LOCALIZATION_RESEED_YAW_STEP_RAD} beams=${LOCALIZATION_RESEED_MAX_BEAMS}/${LOCALIZATION_RESEED_REFINE_MAX_BEAMS}"
echo "goal_tolerance xy=${XY_TOLERANCE_M} yaw=${YAW_TOLERANCE_RAD} final_yaw_min=${FINAL_YAW_MIN_ANGULAR_RPS} latch=${GOAL_POSITION_LATCH} release=${XY_RELEASE_TOLERANCE_M} count=${XY_RELEASE_CONSECUTIVE_COUNT}"
echo "near_goal_slow=${NEAR_GOAL_SLOW_RADIUS_M}m scale=${NEAR_GOAL_SPEED_SCALE}"
echo "object_avoidance=${ENABLE_OBJECT_AVOIDANCE} keepout_half=${OBJECT_KEEPOUT_HALF_WIDTH_M}x${OBJECT_KEEPOUT_HALF_DEPTH_M} lookahead=${OBJECT_AVOIDANCE_LOOKAHEAD_M}"
echo "gripper_serial_port=${GRIPPER_SERIAL_PORT}"
echo "drive_type=${DRIVE_TYPE} controller_type=${CONTROLLER_TYPE} base_scan_yaw=${BASE_SCAN_YAW}"
echo "map_yaml=${MAP_YAML:-(런치 기본값)}"

# MAP_YAML 이 비었으면 인자 자체를 넘기지 않는다 — 빈 문자열을 넘기면 노드가
# 존재하지 않는 경로를 열려다 죽는다.
MAP_ARGS=()
[[ -n "${MAP_YAML}" ]] && MAP_ARGS+=("map_yaml:=${MAP_YAML}")

exec ros2 launch arena_lightweight_control lightweight_real.launch.py \
  "${MAP_ARGS[@]+"${MAP_ARGS[@]}"}" \
  launch_bridge:=true \
  drive_type:="${DRIVE_TYPE}" \
  controller_type:="${CONTROLLER_TYPE}" \
  base_scan_yaw:="${BASE_SCAN_YAW}" \
  imu_topic:="${IMU_TOPIC}" \
  launch_cameras:="${LAUNCH_CAMERAS}" \
  launch_python_ui:="${LAUNCH_PYTHON_UI}" \
  enable_web_ui:="${ENABLE_WEB_UI}" \
  ui_port:="${UI_PORT}" \
  gripper_serial_port:="${GRIPPER_SERIAL_PORT}" \
  localization_mode:="${LOCALIZATION_MODE}" \
  localization_max_beams:="${LOCALIZATION_MAX_BEAMS}" \
  localization_reseed_period_sec:="${LOCALIZATION_RESEED_PERIOD_SEC}" \
  localization_reseed_xy_step_m:="${LOCALIZATION_RESEED_XY_STEP_M}" \
  localization_reseed_yaw_step_rad:="${LOCALIZATION_RESEED_YAW_STEP_RAD}" \
  localization_reseed_max_beams:="${LOCALIZATION_RESEED_MAX_BEAMS}" \
  localization_reseed_refine_max_beams:="${LOCALIZATION_RESEED_REFINE_MAX_BEAMS}" \
  scan_display_max_points:="${SCAN_DISPLAY_MAX_POINTS}" \
  xy_tolerance_m:="${XY_TOLERANCE_M}" \
  yaw_tolerance_rad:="${YAW_TOLERANCE_RAD}" \
  final_yaw_min_angular_rps:="${FINAL_YAW_MIN_ANGULAR_RPS}" \
  goal_position_latch:="${GOAL_POSITION_LATCH}" \
  xy_release_tolerance_m:="${XY_RELEASE_TOLERANCE_M}" \
  xy_release_consecutive_count:="${XY_RELEASE_CONSECUTIVE_COUNT}" \
  near_goal_slow_radius_m:="${NEAR_GOAL_SLOW_RADIUS_M}" \
  near_goal_speed_scale:="${NEAR_GOAL_SPEED_SCALE}" \
  enable_object_avoidance:="${ENABLE_OBJECT_AVOIDANCE}" \
  object_avoidance_max_age_sec:="${OBJECT_AVOIDANCE_MAX_AGE_SEC}" \
  object_keepout_half_width_m:="${OBJECT_KEEPOUT_HALF_WIDTH_M}" \
  object_keepout_half_depth_m:="${OBJECT_KEEPOUT_HALF_DEPTH_M}" \
  object_avoidance_lookahead_m:="${OBJECT_AVOIDANCE_LOOKAHEAD_M}" \
  object_avoidance_turn_rps:="${OBJECT_AVOIDANCE_TURN_RPS}" \
  "$@"
