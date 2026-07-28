#!/usr/bin/env bash
set -euo pipefail

LIBREALSENSE_TAG="${LIBREALSENSE_TAG:-v2.58.2}"
LIBREALSENSE_VERSION="${LIBREALSENSE_VERSION:-${LIBREALSENSE_TAG#v}}"
REALSENSE_CACHE_DIR="${REALSENSE_CACHE_DIR:-${HOME}/.cache/realsense}"
REALSENSE_RSUSB_PREFIX="${REALSENSE_RSUSB_PREFIX:-${HOME}/.local/opt/librealsense-rsusb-${LIBREALSENSE_VERSION}}"
JOBS="${JOBS:-2}"

SOURCE_DIR="${REALSENSE_CACHE_DIR}/librealsense-${LIBREALSENSE_TAG}"
BUILD_DIR="${SOURCE_DIR}/build-rsusb"

mkdir -p "${REALSENSE_CACHE_DIR}"

if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
  git clone --depth 1 --branch "${LIBREALSENSE_TAG}" \
    https://github.com/IntelRealSense/librealsense.git \
    "${SOURCE_DIR}"
else
  git -C "${SOURCE_DIR}" fetch --depth 1 origin tag "${LIBREALSENSE_TAG}" || true
  git -C "${SOURCE_DIR}" checkout "${LIBREALSENSE_TAG}"
fi

cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DFORCE_RSUSB_BACKEND=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_GRAPHICAL_EXAMPLES=OFF \
  -DBUILD_TOOLS=ON \
  -DBUILD_UNIT_TESTS=OFF \
  -DBUILD_WITH_CUDA=OFF \
  -DBUILD_PYTHON_BINDINGS=OFF \
  -DHWM_OVER_XU=ON

cmake --build "${BUILD_DIR}" --target rs-enumerate-devices -j "${JOBS}"

mkdir -p "${REALSENSE_RSUSB_PREFIX}/bin" \
  "${REALSENSE_RSUSB_PREFIX}/include" \
  "${REALSENSE_RSUSB_PREFIX}/lib"

cp -a "${SOURCE_DIR}/include/librealsense2" "${REALSENSE_RSUSB_PREFIX}/include/"
cp -a "${BUILD_DIR}/Release"/librealsense2.so* "${REALSENSE_RSUSB_PREFIX}/lib/"
cp "${BUILD_DIR}/Release/rs-enumerate-devices" "${REALSENSE_RSUSB_PREFIX}/bin/"

if [[ -e "${REALSENSE_RSUSB_PREFIX}/lib/librealsense2.so.2.58" ]]; then
  ln -sf librealsense2.so.2.58 "${REALSENSE_RSUSB_PREFIX}/lib/librealsense2.so.2.57"
fi

cat <<EOF
Installed RSUSB librealsense at:
  ${REALSENSE_RSUSB_PREFIX}

To test D435I motion profiles:
  LD_LIBRARY_PATH=${REALSENSE_RSUSB_PREFIX}/lib \\
    ${REALSENSE_RSUSB_PREFIX}/bin/rs-enumerate-devices -c

run_real_competition_bridge.sh uses this prefix automatically when present.
Set USE_REALSENSE_RSUSB=false to disable it.
EOF
