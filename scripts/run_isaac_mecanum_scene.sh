#!/usr/bin/env bash
set -euo pipefail

# Launch the Isaac Sim 4-mecanum-wheel competition scene with the ROS 2 bridge.
# Usage: scripts/dev/run_isaac_mecanum_scene.sh [--headless] [--drive-mode physics|kinematic] [-- extra scene args]
# ROS distro defaults to humble (AGENTS.md); pass "jazzy" or ros_distro:=jazzy to opt in.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

resolve_default_isaac_python() {
  local candidate
  for candidate in "${HOME}/isaacsim/python.sh" "${HOME}/isaac-sim/python.sh"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  printf '%s\n' "${HOME}/isaacsim/python.sh"
}

ISAAC_PYTHON="${ISAAC_PYTHON:-$(resolve_default_isaac_python)}"
SCENE_SCRIPT="${REPO_ROOT}/simulation/scripts/create_mecanum_competition_scene.py"
FASTDDS_PROFILE="${FASTDDS_PROFILE:-${REPO_ROOT}/hardware/ros2/robot_bringup/config/fastdds_udp_only.xml}"
ROS_DISTRO_ARG="humble"
SCENE_ARGS=()

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
      SCENE_ARGS+=("$@")
      break
      ;;
    *)
      SCENE_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${ROS_DISTRO_ARG}" != "humble" && "${ROS_DISTRO_ARG}" != "jazzy" ]]; then
  echo "unsupported ROS distro: ${ROS_DISTRO_ARG}; expected humble or jazzy" >&2
  exit 2
fi

if [[ ! -x "${ISAAC_PYTHON}" ]]; then
  echo "missing Isaac Sim Python launcher: ${ISAAC_PYTHON}" >&2
  exit 2
fi

resolve_isaac_ros_lib() {
  local isaac_root candidate
  isaac_root="$(cd "$(dirname "$(readlink -f "${ISAAC_PYTHON}")")" && pwd)"
  for candidate in \
    "${isaac_root}/exts/isaacsim.ros2.core/${ROS_DISTRO_ARG}/lib" \
    "${isaac_root}/exts/isaacsim.ros2.bridge/${ROS_DISTRO_ARG}/lib"; do
    if [[ -d "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

if ! ISAAC_ROS_LIB="$(resolve_isaac_ros_lib)"; then
  echo "missing Isaac Sim internal ROS ${ROS_DISTRO_ARG} libraries next to ${ISAAC_PYTHON}" >&2
  exit 2
fi

clean_colon_var() {
  local name="$1"
  local value="${!name:-}"
  local cleaned=""
  local part

  [[ -z "${value}" ]] && return 0

  IFS=':' read -r -a parts <<< "${value}"
  for part in "${parts[@]}"; do
    [[ -z "${part}" ]] && continue
    if [[ "${part}" == /opt/ros/* ]]; then
      continue
    fi
    if [[ "${part}" == */exts/isaacsim.ros2.core/*/lib && "${part}" != "${ISAAC_ROS_LIB}" ]]; then
      continue
    fi
    if [[ "${part}" == */exts/isaacsim.ros2.bridge/*/lib && "${part}" != "${ISAAC_ROS_LIB}" ]]; then
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

unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH
unset PYTHONPATH OLD_PYTHONPATH PYTHONHOME VIRTUAL_ENV
unset FASTRTPS_DEFAULT_PROFILES_FILE RMW_FASTRTPS_USE_QOS_FROM_XML
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

for name in LD_LIBRARY_PATH PATH; do
  clean_colon_var "${name}"
done
export LD_LIBRARY_PATH="${ISAAC_ROS_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export ROS_DISTRO="${ROS_DISTRO_ARG}"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

if [[ -n "${FASTDDS_PROFILE}" ]]; then
  if [[ ! -f "${FASTDDS_PROFILE}" ]]; then
    echo "missing Fast DDS profile: ${FASTDDS_PROFILE}" >&2
    exit 2
  fi
  export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTDDS_PROFILE}"
fi

echo "ROS_DISTRO=${ROS_DISTRO}"
echo "isaac_python=${ISAAC_PYTHON}"
echo "isaac_ros_lib=${ISAAC_ROS_LIB}"
echo "scene_script=${SCENE_SCRIPT}"
echo "fastdds_profile=${FASTDDS_PROFILE:-<disabled>}"

exec env -u PYTHONPATH -u OLD_PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
  -u AMENT_PREFIX_PATH -u CMAKE_PREFIX_PATH -u COLCON_PREFIX_PATH \
  ROS_DISTRO="${ROS_DISTRO}" \
  RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION}" \
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
  PATH="${PATH}" \
  "${ISAAC_PYTHON}" \
  "${SCENE_SCRIPT}" \
  --ros-distro "${ROS_DISTRO}" \
  --overwrite \
  --run \
  "${SCENE_ARGS[@]}"
