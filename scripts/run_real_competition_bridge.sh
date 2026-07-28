#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FASTDDS_PROFILE="${FASTDDS_PROFILE:-${REPO_ROOT}/hardware/ros2/robot_bringup/config/fastdds_udp_only.xml}"
RAW_SCAN_TOPIC="${RAW_SCAN_TOPIC:-/scan_raw}"
EXTRA_RAW_SCAN_TOPICS="${EXTRA_RAW_SCAN_TOPICS:-}"
SCAN_TOPIC="${SCAN_TOPIC:-/laser_scan}"
SCAN_ALIAS_TOPICS="${SCAN_ALIAS_TOPICS:-/scan}"
SCAN_WATCHDOG_PUBLISH_RATE="${SCAN_WATCHDOG_PUBLISH_RATE:-15.0}"
SCAN_WATCHDOG_RAW_TIMEOUT="${SCAN_WATCHDOG_RAW_TIMEOUT:-0.25}"
SCAN_WATCHDOG_HOLD_STALE_SEC="${SCAN_WATCHDOG_HOLD_STALE_SEC:-0.15}"
SCAN_WATCHDOG_QOS_DEPTH="${SCAN_WATCHDOG_QOS_DEPTH:-50}"
SCAN_WATCHDOG_PUBLISH_EMPTY_SCAN="${SCAN_WATCHDOG_PUBLISH_EMPTY_SCAN:-false}"
SCAN_WATCHDOG_PUBLISH_ON_RECEIVE="${SCAN_WATCHDOG_PUBLISH_ON_RECEIVE:-false}"
SCAN_WATCHDOG_REPUBLISH_FRESH_SCAN="${SCAN_WATCHDOG_REPUBLISH_FRESH_SCAN:-true}"
SCAN_WATCHDOG_RESTAMP_SCANS="${SCAN_WATCHDOG_RESTAMP_SCANS:-true}"
REALSENSE_RSUSB_PREFIX="${REALSENSE_RSUSB_PREFIX:-${HOME}/.local/opt/librealsense-rsusb-2.58.2}"
USE_REALSENSE_RSUSB="${USE_REALSENSE_RSUSB:-auto}"
# DRIVE_TYPE env var mirrors run_lightweight_arena_control.sh's convention.
# Without this, DRIVE_TYPE=mecanum was silently ignored and the launch fell
# back to its diff-drive default (2026-07-18 field incident).
DRIVE_TYPE="${DRIVE_TYPE:-diff}"
ROS_DISTRO_ARG="humble"
LAUNCH_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    humble|jazzy)
      ROS_DISTRO_ARG="$1"
      shift
      ;;
    --ros-distro)
      if [[ $# -lt 2 ]]; then
        echo "--ros-distro requires a value: humble or jazzy" >&2
        exit 2
      fi
      ROS_DISTRO_ARG="$2"
      shift 2
      ;;
    --ros-distro=*)
      ROS_DISTRO_ARG="${1#--ros-distro=}"
      shift
      ;;
    ros_distro:=*)
      ROS_DISTRO_ARG="${1#ros_distro:=}"
      shift
      ;;
    --)
      shift
      LAUNCH_ARGS+=("$@")
      break
      ;;
    *)
      LAUNCH_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${ROS_DISTRO_ARG}" != "humble" && "${ROS_DISTRO_ARG}" != "jazzy" ]]; then
  echo "unsupported ROS distro: ${ROS_DISTRO_ARG}; expected humble or jazzy" >&2
  exit 2
fi

if [[ "${DRIVE_TYPE}" != "diff" && "${DRIVE_TYPE}" != "mecanum" ]]; then
  echo "unsupported DRIVE_TYPE=${DRIVE_TYPE}; expected diff or mecanum" >&2
  exit 2
fi

# Kill any bridge processes left running from a prior launch before starting
# a new one. An orphaned diff-drive motor_bridge_node (or a second copy of
# this launch) sharing the same UNO serial port with a fresh mecanum_bridge_node
# corrupts the firmware's command parser (2026-07-18 field incident — required
# a physical USB replug to recover). serial.Serial(exclusive=True) now also
# guards this at the port level, but cleaning up stale processes first avoids
# the failed-connection error entirely.
cleanup_stale_bridge_processes() {
  local patterns=(
    "install/robot_hardware/lib/robot_hardware/motor_bridge_node"
    "install/robot_hardware/lib/robot_hardware/mecanum_bridge_node"
    "install/robot_hardware/lib/robot_hardware/gripper_bridge_node"
    "install/robot_bringup/lib/robot_bringup/gripper_joint_command_bridge"
    "install/robot_bringup/lib/robot_bringup/cmd_vel_mux_node"
    "install/robot_bringup/lib/robot_bringup/laser_scan_watchdog_node"
    "install/sllidar_ros2/lib/sllidar_ros2/sllidar_node"
    "real_competition_bridge.launch.py"
  )
  local found=0
  for pattern in "${patterns[@]}"; do
    if pgrep -f "${pattern}" >/dev/null 2>&1; then
      found=1
    fi
  done
  if [[ "${found}" -eq 0 ]]; then
    return 0
  fi
  echo "found bridge processes still running from a previous session; stopping them first" >&2
  for pattern in "${patterns[@]}"; do
    pkill -f "${pattern}" 2>/dev/null || true
  done
  sleep 1.5
  for pattern in "${patterns[@]}"; do
    pkill -9 -f "${pattern}" 2>/dev/null || true
  done
  sleep 1
  for pattern in "${patterns[@]}"; do
    if pgrep -f "${pattern}" >/dev/null 2>&1; then
      echo "failed to stop stale process matching '${pattern}'; check manually with: pgrep -af '${pattern}'" >&2
      exit 3
    fi
  done
  echo "stale bridge processes cleared" >&2
}
cleanup_stale_bridge_processes

ROS_SETUP="/opt/ros/${ROS_DISTRO_ARG}/setup.bash"
if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "missing ROS setup file: ${ROS_SETUP}" >&2
  exit 2
fi

LOCAL_SETUP="${REPO_ROOT}/install/local_setup.bash"
if [[ ! -f "${LOCAL_SETUP}" ]]; then
  echo "missing workspace setup file: ${LOCAL_SETUP}" >&2
  echo "build it with: source ${ROS_SETUP} && colcon build --symlink-install --base-paths src" >&2
  exit 2
fi

if [[ ! -f "${FASTDDS_PROFILE}" ]]; then
  echo "missing Fast DDS profile: ${FASTDDS_PROFILE}" >&2
  exit 2
fi

clean_colon_var() {
  local name="$1"
  local value="${!name:-}"
  local cleaned=""
  local part

  if [[ -z "${value}" ]]; then
    return 0
  fi

  IFS=':' read -r -a parts <<< "${value}"
  for part in "${parts[@]}"; do
    [[ -z "${part}" ]] && continue
    if [[ "${part}" == /opt/ros/* && "${part}" != /opt/ros/"${ROS_DISTRO_ARG}"* ]]; then
      continue
    fi
    if [[ -z "${cleaned}" ]]; then
      cleaned="${part}"
    else
      cleaned="${cleaned}:${part}"
    fi
  done
  export "${name}=${cleaned}"
}

enable_realsense_rsusb_backend() {
  local mode="${USE_REALSENSE_RSUSB,,}"
  local lib_dir="${REALSENSE_RSUSB_PREFIX}/lib"

  case "${mode}" in
    0|false|no|off|disabled)
      return 0
      ;;
    1|true|yes|on|auto)
      ;;
    *)
      echo "unsupported USE_REALSENSE_RSUSB=${USE_REALSENSE_RSUSB}; expected auto, true, or false" >&2
      exit 2
      ;;
  esac

  if [[ -f "${lib_dir}/librealsense2.so.2.57" || -f "${lib_dir}/librealsense2.so" ]]; then
    export REALSENSE_RSUSB_LIB_DIR="${lib_dir}"
    export REALSENSE_RSUSB_LD_LIBRARY_PATH="${lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export REALSENSE_RSUSB_ACTIVE="true"
    return 0
  fi

  if [[ "${mode}" == "auto" ]]; then
    export REALSENSE_RSUSB_ACTIVE="false"
    echo "RealSense RSUSB backend not found at ${REALSENSE_RSUSB_PREFIX}; using ROS/system librealsense" >&2
    return 0
  fi

  echo "RealSense RSUSB backend requested but missing: ${lib_dir}/librealsense2.so" >&2
  echo "Build/install it with FORCE_RSUSB_BACKEND=ON, or set USE_REALSENSE_RSUSB=auto/false." >&2
  exit 4
}

cd "${REPO_ROOT}"

unset ROS_DISTRO AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH PYTHONHOME VIRTUAL_ENV
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

set +u
source "${ROS_SETUP}"
source "${LOCAL_SETUP}"
set -u

for name in AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH PATH; do
  clean_colon_var "${name}"
done
enable_realsense_rsusb_backend
export ROS_DISTRO="${ROS_DISTRO_ARG}"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTDDS_PROFILE}"
unset RMW_FASTRTPS_USE_QOS_FROM_XML

required_ros_packages=(robot_bringup robot_hardware realsense2_camera sllidar_ros2 tf2_ros)
missing_ros_packages=()
for package in "${required_ros_packages[@]}"; do
  if ! ros2 pkg prefix "${package}" >/dev/null 2>&1; then
    missing_ros_packages+=("${package}")
  fi
done
if [[ ${#missing_ros_packages[@]} -gt 0 ]]; then
  echo "missing ROS ${ROS_DISTRO_ARG} packages: ${missing_ros_packages[*]}" >&2
  echo "install RealSense with: sudo apt install ros-${ROS_DISTRO_ARG}-realsense2-camera" >&2
  echo "build local packages with: source ${ROS_SETUP} && colcon build --symlink-install --base-paths src" >&2
  exit 4
fi

echo "ROS_DISTRO=${ROS_DISTRO}"
echo "DRIVE_TYPE=${DRIVE_TYPE} (diff is the launch default; set DRIVE_TYPE=mecanum before running for the mecanum chassis)"
echo "ros2=$(command -v ros2)"
echo "rmw_implementation=${RMW_IMPLEMENTATION}"
echo "fastdds_profile=${FASTRTPS_DEFAULT_PROFILES_FILE}"
echo "raw_scan_topic=${RAW_SCAN_TOPIC}"
echo "extra_raw_scan_topics=${EXTRA_RAW_SCAN_TOPICS}"
echo "scan_topic=${SCAN_TOPIC}"
echo "scan_alias_topics=${SCAN_ALIAS_TOPICS}"
echo "scan_watchdog_publish_rate=${SCAN_WATCHDOG_PUBLISH_RATE}"
echo "scan_watchdog_hold_stale_sec=${SCAN_WATCHDOG_HOLD_STALE_SEC}"
echo "scan_watchdog_publish_empty_scan=${SCAN_WATCHDOG_PUBLISH_EMPTY_SCAN}"
echo "scan_watchdog_publish_on_receive=${SCAN_WATCHDOG_PUBLISH_ON_RECEIVE}"
echo "scan_watchdog_republish_fresh_scan=${SCAN_WATCHDOG_REPUBLISH_FRESH_SCAN}"
echo "scan_watchdog_restamp_scans=${SCAN_WATCHDOG_RESTAMP_SCANS}"
echo "realsense_rsusb_active=${REALSENSE_RSUSB_ACTIVE:-false}"
if [[ "${REALSENSE_RSUSB_ACTIVE:-false}" == "true" ]]; then
  echo "realsense_rsusb_prefix=${REALSENSE_RSUSB_PREFIX}"
fi
echo "publishing real hardware into competition topics:"
echo "  /camera_19/rgb /camera_19/depth /camera_19/camera_info"
echo "  /camera_54/rgb /camera_54/depth /camera_54/camera_info"
echo "  ${SCAN_TOPIC} ${SCAN_ALIAS_TOPICS} /chassis/odom /tf /joint_states /motor/state"

bridge_args=(
  raw_scan_topic:="${RAW_SCAN_TOPIC}"
  scan_topic:="${SCAN_TOPIC}"
  scan_alias_topics:="${SCAN_ALIAS_TOPICS}"
  scan_watchdog_publish_rate:="${SCAN_WATCHDOG_PUBLISH_RATE}"
  scan_watchdog_raw_timeout:="${SCAN_WATCHDOG_RAW_TIMEOUT}"
  scan_watchdog_hold_stale_sec:="${SCAN_WATCHDOG_HOLD_STALE_SEC}"
  scan_watchdog_qos_depth:="${SCAN_WATCHDOG_QOS_DEPTH}"
  scan_watchdog_publish_empty_scan:="${SCAN_WATCHDOG_PUBLISH_EMPTY_SCAN}"
  scan_watchdog_publish_on_receive:="${SCAN_WATCHDOG_PUBLISH_ON_RECEIVE}"
  scan_watchdog_republish_fresh_scan:="${SCAN_WATCHDOG_REPUBLISH_FRESH_SCAN}"
  scan_watchdog_restamp_scans:="${SCAN_WATCHDOG_RESTAMP_SCANS}"
)
if [[ -n "${EXTRA_RAW_SCAN_TOPICS}" ]]; then
  bridge_args+=(extra_raw_scan_topics:="${EXTRA_RAW_SCAN_TOPICS}")
fi
drive_type_overridden=0
for arg in "${LAUNCH_ARGS[@]}"; do
  if [[ "${arg}" == drive_type:=* ]]; then
    drive_type_overridden=1
  fi
done
if [[ "${drive_type_overridden}" -eq 0 ]]; then
  bridge_args+=(drive_type:="${DRIVE_TYPE}")
fi

exec ros2 launch robot_bringup real_competition_bridge.launch.py \
  "${bridge_args[@]}" \
  "${LAUNCH_ARGS[@]}"
