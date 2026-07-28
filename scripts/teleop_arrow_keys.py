#!/usr/bin/env python3
"""방향키 텔레옵 (메카넘, 경기장 밖 간단 테스트용).

  ↑ / ↓   : 전진 / 후진
  ← / →   : 좌 / 우 스트레이프
  a / d   : 좌 / 우 회전 (CCW / CW)
  space   : 즉시 정지    q 또는 Ctrl-C : 종료

키를 누르고 있는 동안만 움직인다 (0.35s 키 입력 없으면 자동 정지 +
펌웨어 command_timeout 0.5s 이중 안전). /cmd_vel 로 발행 → mux 경유.
arena 노드가 goal을 잡고 있으면 STOP을 먼저 보낸다.

사용: teleop_arrow_keys.py [--vx 0.25] [--vy 0.20] [--wz 0.8]
"""
import argparse
import select
import sys
import termios
import time
import tty

import rclpy
from geometry_msgs.msg import Twist
from std_msgs.msg import String

KEY_HOLD_SEC = 0.35


def read_key(timeout):
    """논블로킹 키 읽기. 화살표는 ESC 시퀀스 → 'UP' 등으로 변환."""
    if not select.select([sys.stdin], [], [], timeout)[0]:
        return None
    ch = sys.stdin.read(1)
    if ch != "\x1b":
        return ch
    if select.select([sys.stdin], [], [], 0.01)[0] and sys.stdin.read(1) == "[":
        code = sys.stdin.read(1)
        return {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT"}.get(code)
    return "ESC"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vx", type=float, default=0.25, help="전후진 속도 m/s")
    ap.add_argument("--vy", type=float, default=0.20, help="스트레이프 속도 m/s")
    ap.add_argument("--wz", type=float, default=0.8, help="회전 속도 rad/s")
    ap.add_argument("--cmd-topic", default="/cmd_vel")
    args = ap.parse_args()

    rclpy.init()
    node = rclpy.create_node("teleop_arrow_keys")
    pub = node.create_publisher(Twist, args.cmd_topic, 10)
    ctrl = node.create_publisher(String, "/arena_lightweight/control", 10)
    time.sleep(0.5)
    ctrl.publish(String(data="STOP"))  # arena goal 해제 (충돌 방지)

    keymap = {
        "UP": (args.vx, 0.0, 0.0), "DOWN": (-args.vx, 0.0, 0.0),
        "LEFT": (0.0, args.vy, 0.0), "RIGHT": (0.0, -args.vy, 0.0),
        "a": (0.0, 0.0, args.wz), "d": (0.0, 0.0, -args.wz),
    }
    print(__doc__.split("사용:")[0])
    print(f"vx={args.vx} vy={args.vy} wz={args.wz} → {args.cmd_topic}")

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    cur = (0.0, 0.0, 0.0)
    last_key_t = 0.0
    try:
        while True:
            key = read_key(0.05)
            now = time.monotonic()
            if key in ("q", "\x03"):
                break
            if key == " ":
                cur = (0.0, 0.0, 0.0)
            elif key in keymap:
                cur = keymap[key]
                last_key_t = now
            elif cur != (0.0, 0.0, 0.0) and now - last_key_t > KEY_HOLD_SEC:
                cur = (0.0, 0.0, 0.0)
            t = Twist()
            t.linear.x, t.linear.y, t.angular.z = cur
            pub.publish(t)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        pub.publish(Twist())
        time.sleep(0.1)
        print("\n정지 발행 — 종료")
        rclpy.shutdown()


if __name__ == "__main__":
    main()
