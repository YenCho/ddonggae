#!/usr/bin/env python3
"""라이다 장착 방향(180° 회전 의심) 판별 → base_scan_yaw 권장값 출력.

배치: 로봇 '정면'을 벽/보드에서 25~50cm 앞에 두고, 나머지 방향(특히 후방)은
1m 이상 비운다. 스캔 원시 각도에서 최근접 섹터를 찾아 라이다 0°가 로봇의
어느 방향을 보는지 판정한다.

현재 기본값 base_scan_yaw=π 는 "라이다 0° = 로봇 후방" 장착 가정이다.
조립 중 180° 돌아갔다면 이 스크립트가 base_scan_yaw=0 을 권장하게 된다.

사전 조건: 라이다 드라이버가 떠 있어야 함. 전체 브리지 없이 라이다만 띄우려면
  ros2 launch sllidar_ros2 sllidar_a2m12_launch.py   # 토픽 /scan
  (run_lightweight_arena_control.sh 사용 시 토픽 /laser_scan)

사용:
  python3 scripts/dev/mecanum/40_lidar_orientation_check.py [--topic /laser_scan]
"""

import argparse
import math
import statistics

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan

SECTORS = {  # 스캔 원시 각도 기준
    "0deg(scan +x)": 0.0,
    "+90deg(scan +y)": math.pi / 2,
    "180deg(scan -x)": math.pi,
    "-90deg(scan -y)": -math.pi / 2,
}
HALF_WIDTH = math.radians(20)


def sector_medians(msg):
    out = {}
    for name, center in SECTORS.items():
        vals = []
        angle = msg.angle_min
        for r in msg.ranges:
            diff = math.atan2(math.sin(angle - center), math.cos(angle - center))
            if abs(diff) <= HALF_WIDTH and msg.range_min < r < msg.range_max:
                vals.append(r)
            angle += msg.angle_increment
        out[name] = statistics.median(vals) if vals else float("inf")
    return out


class Checker(Node):
    def __init__(self, topic, samples):
        super().__init__("lidar_orientation_check")
        self.samples_needed = samples
        self.collected = []
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(LaserScan, topic, self.on_scan, qos)
        self.get_logger().info(f"{topic} 대기 중...")

    def on_scan(self, msg):
        med = sector_medians(msg)
        self.collected.append(med)
        pretty = "  ".join(f"{k}={v:.2f}m" for k, v in med.items())
        self.get_logger().info(f"[{len(self.collected)}/{self.samples_needed}] {pretty}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/laser_scan")
    parser.add_argument("--samples", type=int, default=15)
    args = parser.parse_args()

    print(__doc__)
    input("로봇 정면 25~50cm에 벽/보드, 후방은 1m 이상 개방 — 준비되면 Enter")

    rclpy.init()
    node = Checker(args.topic, args.samples)
    try:
        while rclpy.ok() and len(node.collected) < args.samples:
            rclpy.spin_once(node, timeout_sec=1.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if not node.collected:
        raise SystemExit("스캔을 받지 못했습니다 — 라이다 드라이버/토픽명 확인")

    # 섹터별 중앙값의 중앙값
    agg = {
        name: statistics.median(sample[name] for sample in node.collected)
        for name in SECTORS
    }
    nearest = min(agg, key=agg.get)
    print("\n== 섹터별 거리 (중앙값) ==")
    for name, dist in agg.items():
        mark = "  <-- 최근접(로봇 정면 방향)" if name == nearest else ""
        print(f"  {name:16s} {dist:6.2f} m{mark}")

    recommendations = {
        "0deg(scan +x)": ("0.0", "라이다 0° = 로봇 정면 (조립 시 180° 회전된 경우)"),
        "180deg(scan -x)": ("3.14159265359", "라이다 0° = 로봇 후방 (기존 MK3 장착과 동일, 기본값 유지)"),
        "+90deg(scan +y)": ("±1.5707963268", "90° 장착 — 부호는 UI 스캔 오버레이로 확인 필요"),
        "-90deg(scan -y)": ("±1.5707963268", "90° 장착 — 부호는 UI 스캔 오버레이로 확인 필요"),
    }
    value, note = recommendations[nearest]
    print(f"\n== 판정: {note}")
    print(f"   권장 base_scan_yaw = {value}")
    print("\n적용 (run 스크립트 env로 전달):")
    print(f"   BASE_SCAN_YAW={value.lstrip('±')} DRIVE_TYPE=mecanum "
          "bash scripts/dev/run_lightweight_arena_control.sh")
    print("검증: UI 스캔 오버레이에서 벽 4면이 arena 사각형과 겹치고, 전진 시 pose가 "
          "전방으로 이동하는지 확인. 확정되면 이 값을 런북/CLAUDE.md에 기록.")


if __name__ == "__main__":
    main()
