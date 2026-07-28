#!/usr/bin/env python3
"""카메라 마스트 리프트(OpenRB ID12) 직접 제어 CLI.

gripper_bridge_node 가 떠 있으면 시리얼 포트를 공유하므로 먼저 STOP 해야 한다
(포트가 잡혀 있으면 열기 실패로 안내함).

사용:
  python3 scripts/lift_cmd.py up          # 최고점(마진 포함)으로 상승
  python3 scripts/lift_cmd.py down        # 바닥(홈)으로 하강
  python3 scripts/lift_cmd.py status      # 현재 위치/홈/스트로크 출력
  python3 scripts/lift_cmd.py home        # 현재 위치를 바닥 홈으로 재지정
  python3 scripts/lift_cmd.py move -5000  # 상대 이동(음수=올림, 양수=내림)
  python3 scripts/lift_cmd.py raw LIFT_TORQUE_OFF   # 임의 LIFT_* 명령 전송

홈 로직: 부팅 후 첫 리프트 명령 시점 위치를 바닥 홈으로 1회 캡처하므로,
반드시 카메라가 바닥에 있을 때 처음 명령을 보내라(아니면 up 전에 down/home).
"""

import argparse
import glob
import sys
import time

import serial

PORT_GLOB = "/dev/serial/by-id/usb-ROBOTIS_OpenRB-150*-if00"

ALIASES = {
    "up": "LIFT_TO_TOP",
    "top": "LIFT_TO_TOP",
    "down": "LIFT_TO_BOTTOM",
    "bottom": "LIFT_TO_BOTTOM",
    "status": "LIFT_STATUS?",
    "home": "LIFT_SET_HOME",
    "stop": "LIFT_STOP",
    "free": "LIFT_TORQUE_OFF",
    "hold": "LIFT_TORQUE_ON",
}


def find_port():
    hits = sorted(glob.glob(PORT_GLOB))
    if not hits:
        sys.exit("OpenRB 포트를 못 찾음 (부트로더 상태거나 미연결). lsusb / ls "
                 "/dev/serial/by-id/ 확인")
    return hits[0]


def send(ser, cmd, wait=0.7):
    ser.write((cmd + "\n").encode())
    ser.flush()
    time.sleep(wait)
    out = []
    end = time.time() + wait
    while time.time() < end:
        ln = ser.readline().decode(errors="replace").strip()
        if ln:
            out.append(ln)
    return out


def monitor_until_stopped(ser, timeout_s=25.0):
    t0 = time.time()
    i = 0
    while time.time() - t0 < timeout_s:
        lines = [l for l in send(ser, "LIFT_STATUS?", 0.5) if l.startswith("LIFT_STATUS")]
        if lines:
            l = lines[-1]
            pos = int(l.split("POS_RAW")[1].split()[0])
            mov = int(l.split("MOVING")[1].split()[0])
            cur = int(l.split("CURRENT_RAW")[1].split()[0])
            print(f"  t~{time.time()-t0:4.1f}s POS_RAW {pos}  MOVING {mov}  CUR {cur}")
            if mov == 0 and i > 2:
                print("  -> 정지 완료")
                return
        i += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", help="up|down|status|home|stop|free|hold|move|raw")
    ap.add_argument("arg", nargs="?", help="move: 상대 ticks / raw: 전송할 명령")
    ap.add_argument("--no-monitor", action="store_true", help="이동 후 정지 대기 생략")
    args = ap.parse_args()

    act = args.action.lower()
    if act == "move":
        if args.arg is None:
            sys.exit("move 는 상대 ticks 인자 필요: 예) move -5000 (음수=올림)")
        cmd = f"LIFT_MOVE {int(args.arg)}"
    elif act == "raw":
        if not args.arg:
            sys.exit("raw 는 전송할 명령 필요: 예) raw LIFT_TORQUE_OFF")
        cmd = args.arg
    elif act in ALIASES:
        cmd = ALIASES[act]
    else:
        sys.exit(f"알 수 없는 action: {act}")

    port = find_port()
    try:
        ser = serial.Serial(port, 115200, timeout=0.4)
    except serial.SerialException as exc:
        sys.exit(f"포트 열기 실패({port}): {exc}\n"
                 "gripper_bridge_node 가 떠 있으면 먼저 종료하세요.")
    time.sleep(2.0)
    ser.reset_input_buffer()

    # 전원 세션 첫 명령이면 DXL 버스 전원이 꺼져 있을 수 있음 → 켜둔다(그리퍼 무해).
    send(ser, "DXL_POWER_ON", 1.5)

    print(f">> {cmd}")
    for l in send(ser, cmd, 0.8):
        print("  ", l)

    is_move = act in ("up", "top", "down", "bottom", "move")
    if is_move and not args.no_monitor:
        monitor_until_stopped(ser)

    ser.close()


if __name__ == "__main__":
    main()
