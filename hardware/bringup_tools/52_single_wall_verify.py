#!/usr/bin/env python3
"""단일 벽 + IMU 자이로 기반 주행 정밀도 검증 (경기장 전체 불필요).

경기장 예약 없이 벽 하나만으로 51번(move_accuracy_lidar)을 대체한다:
  - 전/후진: 정면 벽까지 수직거리 변화 (raw /scan, wall_range localizer 불필요)
  - 스트레이프: 옆 벽까지 수직거리 변화 + 자이로 yaw 유지 확인
  - 제자리 회전: 자이로 적분 dyaw (벽 자체가 불필요)

벽 수직거리 = 해당 방향 ±30° 윈도우에서 스캔별 최소 range의 중앙값.
평평한 벽의 최소거리는 수직거리이므로 로봇 yaw가 조금 틀어져도 유효하다.

사용:
  python3 scripts/dev/mecanum/52_single_wall_verify.py --wall front 0,1   # 전/후진
  python3 scripts/dev/mecanum/52_single_wall_verify.py --wall left  2,3   # 스트레이프
  python3 scripts/dev/mecanum/52_single_wall_verify.py --wall none  4,5   # 회전(IMU만)

전제: mecanum bridge + sllidar + IMU 노드 기동 상태 (run_lightweight_arena_control.sh
+ 수동 IMU 노드면 충분. arena localization은 사용하지 않으므로 벽 1개여도 무방).
"""
import argparse
import json
import math
import statistics
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import String

REPO_LOG_DIR = str(Path(__file__).resolve().parents[2] / "logs" / "mecanum_bringup")

# 실기 D435I 바텀캠(54도 틸트) gyro 실측 yaw 축 (CLAUDE.md 2026-07-15)
DEFAULT_YAW_AXIS = (0.0, -0.586, -0.810)

TESTS = [
    ("전진 0.5m",  {"dx": 0.5,  "dy": 0.0,  "dyaw": 0.0}, "front"),
    ("후진 0.5m",  {"dx": -0.5, "dy": 0.0,  "dyaw": 0.0}, "front"),
    ("좌 스트레이프 0.4m", {"dx": 0.0, "dy": 0.4,  "dyaw": 0.0}, "side"),
    ("우 스트레이프 0.4m", {"dx": 0.0, "dy": -0.4, "dyaw": 0.0}, "side"),
    ("좌회전 +90도", {"dx": 0.0, "dy": 0.0, "dyaw": math.pi / 2}, "none"),
    ("우회전 -90도", {"dx": 0.0, "dy": 0.0, "dyaw": -math.pi / 2}, "none"),
]

WALL_BEARING = {"front": 0.0, "left": math.pi / 2, "right": -math.pi / 2}
# 이동 후에도 벽과 충돌하지 않기 위한 최소 시작 여유 (이동량 + margin)
SAFETY_MARGIN_M = 0.35


def yaw_of(odom):
    q = odom.pose.pose.orientation
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))


def ang_diff(a, b):
    return math.remainder(a - b, 2 * math.pi)


class SingleWallTester(Node):
    def __init__(self, args):
        super().__init__("single_wall_verify")
        self.args = args
        self.scan = None
        self.odom_pose = None
        self.move_result = None
        # gyro 적분 상태
        self.gyro_bias = 0.0
        self.gyro_yaw = 0.0
        self.last_imu_t = None
        self.imu_count = 0
        self.bias_samples = None  # list일 때만 수집
        self.create_subscription(LaserScan, args.scan_topic, self.on_scan,
                                 qos_profile_sensor_data)
        self.create_subscription(Imu, args.imu_topic, self.on_imu,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, "/chassis/odom", self.on_odom, 10)
        self.create_subscription(String, "/base/move_result", self.on_result, 10)
        self.move_pub = self.create_publisher(String, "/base/move_relative", 10)

    def on_scan(self, msg):
        self.scan = msg

    def on_imu(self, msg):
        ax, ay, az = self.args.yaw_axis
        rate = msg.angular_velocity.x * ax + msg.angular_velocity.y * ay \
            + msg.angular_velocity.z * az
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.imu_count += 1
        if self.bias_samples is not None:
            self.bias_samples.append(rate)
        elif self.last_imu_t is not None:
            dt = t - self.last_imu_t
            if 0.0 < dt < 0.2:
                self.gyro_yaw += (rate - self.gyro_bias) * dt
        self.last_imu_t = t

    def on_odom(self, msg):
        p = msg.pose.pose.position
        self.odom_pose = (p.x, p.y, yaw_of(msg))

    def on_result(self, msg):
        try:
            self.move_result = json.loads(msg.data)
        except ValueError:
            self.move_result = {"ok": False, "reason": msg.data}

    def spin_for(self, seconds):
        t0 = time.time()
        while time.time() - t0 < seconds:
            rclpy.spin_once(self, timeout_sec=0.05)

    def calibrate_gyro_bias(self, seconds=1.5):
        """정지 상태에서 gyro bias 추정 후 적분 리셋."""
        self.bias_samples = []
        self.spin_for(seconds)
        samples, self.bias_samples = self.bias_samples, None
        if len(samples) < 10:
            return None
        self.gyro_bias = statistics.median(samples)
        self.gyro_yaw = 0.0
        return self.gyro_bias

    def wall_distance(self, bearing, seconds=1.5, half_window=math.radians(30)):
        """bearing 방향 ±window 최소 range(스캔별)의 중앙값 = 벽 수직거리."""
        mins = []
        t0 = time.time()
        seen = None
        while time.time() - t0 < seconds:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.scan is None or self.scan is seen:
                continue
            seen = self.scan
            s = self.scan
            vals = []
            for i, r in enumerate(s.ranges):
                if not (s.range_min < r < s.range_max) or not math.isfinite(r):
                    continue
                b = ang_diff(s.angle_min + i * s.angle_increment
                             + self.args.scan_yaw, 0.0)
                if abs(ang_diff(b, bearing)) <= half_window:
                    vals.append(r)
            if len(vals) >= 5:
                vals.sort()
                mins.append(statistics.median(vals[:3]))
        return statistics.median(mins) if mins else None

    def move(self, payload, timeout=25.0):
        self.move_result = None
        self.move_pub.publish(String(data=json.dumps(payload)))
        t0 = time.time()
        while self.move_result is None and time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.move_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tests", nargs="?", default=None,
                        help="테스트 번호 (예: 0,1). 생략 시 --wall과 맞는 것 전부")
    parser.add_argument("--wall", choices=["front", "left", "right", "none"],
                        required=True,
                        help="벽 위치: front=정면(전/후진), left/right=옆(스트레이프), none=회전만")
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--imu-topic", default="/imu/data")
    parser.add_argument("--scan-yaw", type=float,
                        default=0.0, help="라이다 장착 yaw 오프셋(rad). 실기=0.0")
    parser.add_argument("--yaw-axis", default=None,
                        help='"x,y,z" gyro yaw 축 (기본: D435I 바텀 실측)')
    args = parser.parse_args()
    args.yaw_axis = tuple(float(v) for v in args.yaw_axis.split(",")) \
        if args.yaw_axis else DEFAULT_YAW_AXIS

    if args.tests:
        sel = [int(i) for i in args.tests.split(",")]
    else:
        want = "side" if args.wall in ("left", "right") else args.wall
        sel = [i for i, (_, _, need) in enumerate(TESTS)
               if need == want or need == "none"]

    rclpy.init()
    node = SingleWallTester(args)
    node.spin_for(2.0)
    if node.imu_count == 0:
        print(f"경고: IMU({args.imu_topic}) 미수신 — 회전/yaw 유지 측정 불가. "
              "IMU 노드 기동 확인 (CLAUDE.md 터미널1 명령).")
    if node.odom_pose is None:
        print("오류: /chassis/odom 미수신 — mecanum bridge 기동 확인")
        return

    report = []
    for i in sel:
        name, cmd, need = TESTS[i]
        wall_bearing = None
        if need == "front" and args.wall == "front":
            wall_bearing = WALL_BEARING["front"]
        elif need == "side" and args.wall in ("left", "right"):
            wall_bearing = WALL_BEARING[args.wall]
        elif need != "none":
            print(f"\n== [{i}] {name} == 건너뜀 (--wall {args.wall}과 불일치)")
            continue

        print(f"\n== [{i}] {name} ==")
        bias = node.calibrate_gyro_bias()
        if bias is not None:
            print(f"  gyro bias {bias:+.5f} rad/s ({node.imu_count} msgs 누적)")
        wall0 = node.wall_distance(wall_bearing) if wall_bearing is not None else None
        odom0 = node.odom_pose

        # 벽 접근 방향이면 충돌 여유 확인
        if wall_bearing is not None:
            approach = cmd["dx"] if need == "front" else \
                (cmd["dy"] if args.wall == "left" else -cmd["dy"])
            if wall0 is None:
                print("  벽 거리 측정 실패 — /scan 확인"); continue
            print(f"  시작 벽거리 {wall0:.3f} m")
            if approach > 0 and wall0 < approach + SAFETY_MARGIN_M:
                print(f"  중단: 벽까지 {wall0:.2f} m < 이동 {approach:.2f}+여유 "
                      f"{SAFETY_MARGIN_M} m — 로봇을 벽에서 더 떨어뜨리고 재실행")
                continue

        res = node.move(dict(cmd))
        ok = bool(res and res.get("ok"))
        print(f"  move_result: ok={ok} reason={res.get('reason') if res else 'timeout'} "
              f"elapsed={res.get('elapsed_sec') if res else None}")
        node.spin_for(1.0)

        wall1 = node.wall_distance(wall_bearing) if wall_bearing is not None else None
        odom1 = node.odom_pose
        gyro_dyaw = node.gyro_yaw if node.imu_count else float("nan")
        odom_d = (odom1[0] - odom0[0], odom1[1] - odom0[1],
                  ang_diff(odom1[2], odom0[2]))

        entry = {"test": name, "cmd": cmd, "ok": ok,
                 "reason": res.get("reason") if res else "timeout",
                 "gyro_dyaw_deg": math.degrees(gyro_dyaw),
                 "odom_dyaw_deg": math.degrees(odom_d[2]),
                 "wall_before_m": wall0, "wall_after_m": wall1}

        if wall_bearing is not None and wall0 and wall1:
            moved = wall0 - wall1  # 벽으로 접근한 거리
            cmd_toward = cmd["dx"] if need == "front" else \
                (cmd["dy"] if args.wall == "left" else -cmd["dy"])
            transfer = moved / cmd_toward if abs(cmd_toward) > 1e-6 else float("nan")
            entry["wall_moved_m"] = moved
            entry["transfer_pct"] = transfer * 100.0
            print(f"  벽거리 변화 {moved:+.3f} m (명령 {cmd_toward:+.3f}) "
                  f"→ 전달률 {transfer*100:+.0f}%")
        if abs(cmd["dyaw"]) > 1e-6:
            entry["dyaw_transfer_pct"] = gyro_dyaw / cmd["dyaw"] * 100.0
            print(f"  gyro dyaw {math.degrees(gyro_dyaw):+.1f}deg "
                  f"(명령 {math.degrees(cmd['dyaw']):+.1f}) "
                  f"→ 전달률 {entry['dyaw_transfer_pct']:+.0f}%  "
                  f"odom {math.degrees(odom_d[2]):+.1f}deg")
        else:
            print(f"  yaw 유지: gyro {math.degrees(gyro_dyaw):+.2f}deg "
                  f"odom {math.degrees(odom_d[2]):+.2f}deg")
        report.append(entry)
        time.sleep(0.5)

    out = f"{REPO_LOG_DIR}/single_wall_{time.strftime('%H%M%S')}.json"
    with open(out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out}")
    print("판정 가이드: 전달률 93~107% 합격, yaw 유지 |3도| 이내, "
          "회전은 gyro 전달률 90~110% + ok=True")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
