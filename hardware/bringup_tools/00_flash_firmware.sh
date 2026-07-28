#!/usr/bin/env bash
# 메카넘 펌웨어(mecanum_encoder_control) 컴파일 + 업로드 원커맨드.
# 사용: bash scripts/dev/mecanum/00_flash_firmware.sh [--compile-only] [포트]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SKETCH="${REPO_ROOT}/firmware/arduino_motor/mecanum_encoder_control"
FQBN="arduino:avr:uno"

COMPILE_ONLY=false
PORT=""
for arg in "$@"; do
  case "$arg" in
    --compile-only) COMPILE_ONLY=true ;;
    *) PORT="$arg" ;;
  esac
done

if ! command -v arduino-cli >/dev/null 2>&1; then
  echo "arduino-cli가 없습니다. 설치:" >&2
  echo "  curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh" >&2
  echo "  export PATH=\$PATH:\$PWD/bin && arduino-cli core install arduino:avr" >&2
  exit 2
fi

if ! arduino-cli core list 2>/dev/null | grep -q "arduino:avr"; then
  echo "arduino:avr core 설치 중..."
  arduino-cli core update-index
  arduino-cli core install arduino:avr
fi

echo "== compile: ${SKETCH}"
arduino-cli compile --fqbn "${FQBN}" "${SKETCH}"

if [[ "${COMPILE_ONLY}" == true ]]; then
  echo "== compile-only 완료 (업로드 생략)"
  exit 0
fi

if [[ -z "${PORT}" ]]; then
  for candidate in \
    /dev/serial/by-id/usb-Arduino__www.arduino.cc__Arduino_14101-if00 \
    /dev/ttyACM0 /dev/ttyACM1; do
    if [[ -e "${candidate}" ]]; then
      PORT="${candidate}"
      break
    fi
  done
fi
if [[ -z "${PORT}" ]]; then
  echo "포트를 찾지 못했습니다. 인자로 지정하세요: $0 /dev/ttyACM0" >&2
  exit 3
fi

echo "== upload: ${PORT}"
echo "   (mecanum_bridge_node 등 시리얼 점유 프로세스가 없는지 확인)"
arduino-cli upload --fqbn "${FQBN}" --port "${PORT}" "${SKETCH}"
echo "== 업로드 완료. 다음: python3 scripts/dev/mecanum/10_wheel_selftest.py"
