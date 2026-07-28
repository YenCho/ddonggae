#!/usr/bin/env python3
"""바닥 주행 정밀도 테스트 — /base/move_relative 폐루프 이동 + 줄자 실측.

mecanum_bridge_node가 떠 있는 상태(ROS 경유)에서 실행한다:
  DRIVE_TYPE=mecanum bash scripts/dev/run_lightweight_arena_control.sh
  (또는 브리지 노드만: ros2 run robot_hardware mecanum_bridge_node ...)

전진/후진/좌우 스트레이프/±90° 회전을 순서대로 수행하고, 각 이동 후 실측값을
입력하면 wheel_radius / half_length+half_width(회전) / 스트레이프 슬립 보정
계수를 제안한다. 실측 입력을 건너뛰면(엔터) 명령-결과 확인만 한다.

사용:
  python3 scripts/dev/mecanum/50_move_accuracy.py [--skip-yaw] [--repeat 1]
"""

import argparse
import json
import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

TESTS = [
    ("전진 0.5m", {"dx": 0.5, "dy": 0.0, "dyaw": 0.0}, "linear_x"),
    ("후진 0.5m", {"dx": -0.5, "dy": 0.0, "dyaw": 0.0}, "linear_x"),
    ("좌 스트레이프 0.4m", {"dx": 0.0, "dy": 0.4, "dyaw": 0.0}, "linear_y"),
    ("우 스트레이프 0.4m", {"dx": 0.0, "dy": -0.4, "dyaw": 0.0}, "linear_y"),
    ("좌회전 +90°", {"dx": 0.0, "dy": 0.0, "dyaw": math.pi / 2}, "yaw"),
    ("우회전 -90°", {"dx": 0.0, "dy": 0.0, "dyaw": -math.pi / 2}, "yaw"),
]


class MoveTester(Node):
    def __init__(self):
        super().__init__("mecanum_move_accuracy")
        self.result = None
        self.pub = self.create_publisher(String, "/base/move_relative", 10)
        self.create_subscription(String, "/base/move_result", self.on_result, 10)

    def on_result(self, msg):
        try:
            self.result = json.loads(msg.data)
        except ValueError:
            self.result = {"ok": False, "reason": "bad json: " + msg.data}

    def move(self, payload, timeout=25.0):
        self.result = None
        msg = String()
        msg.data = json.dumps(payload)
        self.pub.publish(msg)
        end = time.time() + timeout
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.2)
            if self.result is not None:
                return self.result
        return {"ok": False, "reason": "timeout (move_result 미수신)"}


def ask_measured(prompt):
    raw = input(prompt).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-yaw", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    print(__doc__)
    input("바닥에 2m 이상 공간 확보, 시작 위치를 테이프로 표시 후 Enter")

    rclpy.init()
    node = MoveTester()
    ratios = {"linear_x": [], "linear_y": [], "yaw": []}
    try:
        for _ in range(args.repeat):
            for name, payload, kind in TESTS:
                if args.skip_yaw and kind == "yaw":
                    continue
                input(f"\n[{name}] Enter로 실행 — 로봇 주변 확인!")
                result = node.move(payload)
                ok = result.get("ok")
                print(f"  move_result: ok={ok} reason={result.get('reason')} "
                      f"elapsed={result.get('elapsed_sec', result.get('elapsed'))}")
                if not ok:
                    print("  !! 실패 — 원인 해결 후 재시도 권장")
                    continue
                commanded = abs(payload["dx"] or payload["dy"] or payload["dyaw"])
                unit = "deg" if kind == "yaw" else "m"
                shown = math.degrees(commanded) if kind == "yaw" else commanded
                measured = ask_measured(
                    f"  실측값 입력 ({shown:.1f}{unit} 명령, 줄자/각도 실측, 건너뛰기=Enter): ")
                if measured is not None:
                    measured_rad = math.radians(measured) if kind == "yaw" else measured
                    ratios[kind].append(measured_rad / commanded)
                    print(f"  실측/명령 비 = {measured_rad / commanded:.4f}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

    print("\n== 보정 제안 ==")
    if ratios["linear_x"]:
        r = sum(ratios["linear_x"]) / len(ratios["linear_x"])
        print(f"  전후진 비 평균 {r:.4f} → wheel_radius_m을 현재값 × {r:.4f} 로 보정")
    if ratios["linear_y"]:
        r = sum(ratios["linear_y"]) / len(ratios["linear_y"])
        print(f"  스트레이프 비 평균 {r:.4f} (롤러 슬립 — sim에서도 ~0.7 수준이었음)")
        print("    → 전략 계층에서 y이동은 폐루프(move_relative/localization) 전제 유지")
    if ratios["yaw"]:
        r = sum(ratios["yaw"]) / len(ratios["yaw"])
        print(f"  회전 비 평균 {r:.4f} → (half_length_m + half_width_m)을 ÷ {r:.4f} 로 보정")
    print("\n갱신 위치: real.yaml mecanum_bridge_node (wheel_radius_m, half_length_m, "
          "half_width_m) — 갱신 후 브리지 재시작(geom 자동 push)")


if __name__ == "__main__":
    main()
