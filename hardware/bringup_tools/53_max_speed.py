#!/usr/bin/env python3
"""최대속도 측정 — 전/후진, 스트레이프(벽 거리 기준), 제자리 회전(IMU 자이로 기준).

명령 속도를 계단식으로 올리면서 실제 달성 속도를 측정해 구동계 포화점을 찾는다.
  - linear x/y: 벽까지 수직거리 시계열의 기울기 = 지면 실속도 (odom과 병기).
    달성/명령 비율이 --sat-ratio 미만이면 포화로 판정하고 1스텝 더 확인 후 종료.
  - yaw: 각 스텝에서 gyro 적분 dyaw vs odom dyaw, IMU 메시지 수신율,
    gyro 순간최대치를 기록 — 구동계 포화와 IMU 추적 한계를 동시에 판정.

주의: 브리지의 속도 상한을 푼 상태로 기동해야 한다:
  ros2 run robot_hardware mecanum_bridge_node --ros-args \
    --params-file src/robot_bringup/config/real.yaml \
    -p odom_topic:=/chassis/odom -p max_linear_x_mps:=1.5 \
    -p max_linear_y_mps:=1.2 -p max_angular_rps:=8.0 -p max_wheel_rad_s:=40.0

사용:
  53_max_speed.py --axis x                      # 전/후진 (벽 정면)
  53_max_speed.py --axis y --wall left          # 스트레이프 (벽 왼쪽)
  53_max_speed.py --axis yaw                    # 제자리 회전 (벽 불필요)
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
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import String

REPO_LOG_DIR = str(Path(__file__).resolve().parents[2] / "logs" / "mecanum_bringup")
DEFAULT_YAW_AXIS = (0.0, -0.586, -0.810)  # D435I 바텀캠 gyro yaw 투영축 (실측)
WALL_BEARING = {"front": 0.0, "left": math.pi / 2, "right": -math.pi / 2}


def yaw_of(odom):
    q = odom.pose.pose.orientation
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))


def ang_diff(a, b):
    return math.remainder(a - b, 2 * math.pi)


def slope_fit(samples):
    """(t, v) 최소제곱 기울기. 표본 3개 미만이면 None."""
    if len(samples) < 3:
        return None
    n = len(samples)
    mt = sum(t for t, _ in samples) / n
    mv = sum(v for _, v in samples) / n
    denom = sum((t - mt) ** 2 for t, _ in samples)
    if denom < 1e-9:
        return None
    return sum((t - mt) * (v - mv) for t, v in samples) / denom


class MaxSpeedTester(Node):
    def __init__(self, args):
        super().__init__("max_speed_test")
        self.args = args
        self.scan = None
        self.odom = None
        self.imu_count = 0
        self.gyro_bias = 0.0
        self.gyro_yaw = 0.0
        self.gyro_peak = 0.0
        self.last_imu_t = None
        self.bias_samples = None
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(LaserScan, args.scan_topic, self.on_scan,
                                 qos_profile_sensor_data)
        self.create_subscription(Imu, args.imu_topic, self.on_imu,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, args.odom_topic, self.on_odom, 10)

    def on_scan(self, msg):
        self.scan = msg

    def on_imu(self, msg):
        ax, ay, az = self.args.yaw_axis
        rate = msg.angular_velocity.x * ax + msg.angular_velocity.y * ay \
            + msg.angular_velocity.z * az
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.imu_count += 1
        self.gyro_peak = max(self.gyro_peak, abs(rate - self.gyro_bias))
        if self.bias_samples is not None:
            self.bias_samples.append(rate)
        elif self.last_imu_t is not None:
            dt = t - self.last_imu_t
            if 0.0 < dt < 0.2:
                self.gyro_yaw += (rate - self.gyro_bias) * dt
        self.last_imu_t = t

    def on_odom(self, msg):
        self.odom = msg

    def spin_for(self, seconds):
        t0 = time.time()
        while time.time() - t0 < seconds:
            rclpy.spin_once(self, timeout_sec=0.02)

    def calibrate_gyro_bias(self, seconds=1.2):
        self.bias_samples = []
        c0 = self.imu_count
        self.spin_for(seconds)
        samples, self.bias_samples = self.bias_samples, None
        self.imu_rate_baseline = (self.imu_count - c0) / seconds
        if len(samples) < 10:
            return None
        self.gyro_bias = statistics.median(samples)
        self.gyro_yaw = 0.0
        self.gyro_peak = 0.0
        return self.gyro_bias

    def wall_distance_once(self, bearing, half_window=math.radians(30)):
        """최신 스캔 1개에서 벽 수직거리 (스캔별 최소 range의 하위3 중앙값)."""
        s = self.scan
        if s is None:
            return None
        vals = []
        for i, r in enumerate(s.ranges):
            if not (s.range_min < r < s.range_max) or not math.isfinite(r):
                continue
            b = ang_diff(s.angle_min + i * s.angle_increment + self.args.scan_yaw, 0.0)
            if abs(ang_diff(b, bearing)) <= half_window:
                vals.append(r)
        if len(vals) < 5:
            return None
        vals.sort()
        return statistics.median(vals[:3])

    def wall_distance_stable(self, bearing, seconds=1.0):
        vals, seen = [], None
        t0 = time.time()
        while time.time() - t0 < seconds:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.scan is not seen:
                seen = self.scan
                d = self.wall_distance_once(bearing)
                if d is not None:
                    vals.append(d)
        return statistics.median(vals) if vals else None

    def publish_cmd(self, vx=0.0, vy=0.0, wz=0.0):
        t = Twist()
        t.linear.x, t.linear.y, t.angular.z = float(vx), float(vy), float(wz)
        self.cmd_pub.publish(t)

    def burst(self, vx, vy, wz, duration, wall_bearing=None):
        """duration 동안 cmd_vel 유지하며 벽거리/odom/gyro 시계열 수집."""
        wall_ts, odom_ts = [], []
        seen_scan = None
        gyro0 = self.gyro_yaw
        imu0 = self.imu_count
        t0 = time.time()
        while time.time() - t0 < duration:
            self.publish_cmd(vx, vy, wz)
            rclpy.spin_once(self, timeout_sec=0.02)
            now = time.time() - t0
            if wall_bearing is not None and self.scan is not seen_scan:
                seen_scan = self.scan
                d = self.wall_distance_once(wall_bearing)
                if d is not None:
                    wall_ts.append((now, d))
            if self.odom is not None:
                p = self.odom.pose.pose.position
                odom_ts.append((now, p.x, p.y, yaw_of(self.odom)))
        self.publish_cmd(0.0, 0.0, 0.0)
        elapsed = time.time() - t0
        # 정지 대기 (브리지 timeout과 별개로 명시적 0 명령)
        for _ in range(10):
            self.publish_cmd(0.0, 0.0, 0.0)
            rclpy.spin_once(self, timeout_sec=0.05)
        return {
            "wall_ts": wall_ts, "odom_ts": odom_ts,
            "gyro_dyaw": self.gyro_yaw - gyro0,
            "imu_msgs": self.imu_count - imu0,
            "elapsed": elapsed,
        }


def middle_window(ts, lo=0.45, hi=0.95):
    if not ts:
        return []
    t_end = ts[-1][0]
    return [s for s in ts if lo * t_end <= s[0] <= hi * t_end]


def measure_linear(node, args):
    bearing = WALL_BEARING["front"] if args.axis == "x" else WALL_BEARING[args.wall]
    ramp = [float(v) for v in args.ramp.split(",")]
    results = []
    saturated = 0
    for v_cmd in ramp:
        t_burst = min(max(args.budget_m / v_cmd, 0.6), 3.0)
        travel = v_cmd * t_burst
        # 접근 방향을 먼저: x축은 +가 벽 접근(정면), y축은 left벽이면 +dy 접근
        toward_sign = 1.0 if (args.axis == "x" or args.wall == "left") else -1.0
        step = {"cmd_mps": v_cmd, "burst_s": round(t_burst, 2), "runs": []}
        print(f"\n== {v_cmd:.2f} m/s (버스트 {t_burst:.1f}s, 이동 ~{travel:.2f} m) ==")
        for direction in (+1, -1):
            node.calibrate_gyro_bias()
            approaching = direction * toward_sign > 0
            # 항상 "지금 접근하는 쪽" 벽을 감시 — 경기장처럼 사방이 벽이면
            # 반대 방향도 반대쪽 벽 접근이므로 그 벽으로 측정+충돌 확인.
            run_bearing = bearing if approaching else ang_diff(bearing + math.pi, 0.0)
            wall0 = node.wall_distance_stable(run_bearing)
            if wall0 is None:
                if not approaching:  # 열린 공간(벽 없음)이면 확인 없이 진행
                    run_bearing = None
                else:
                    print("  벽 미검출 — 중단"); return results
            elif wall0 < travel + args.margin_m:
                print(f"  건너뜀: 진행방향 벽 {wall0:.2f} m < 필요 "
                      f"{travel + args.margin_m:.2f} m")
                continue
            vx = direction * v_cmd if args.axis == "x" else 0.0
            vy = direction * v_cmd if args.axis == "y" else 0.0
            data = node.burst(vx, vy, 0.0, t_burst, wall_bearing=run_bearing)
            mid_wall = middle_window(data["wall_ts"])
            wall_rate = slope_fit(mid_wall)  # 접근 시 음수
            ground_v = abs(wall_rate) if wall_rate is not None else float("nan")
            mid_odom = middle_window([(t, x, y) for t, x, y, _ in data["odom_ts"]])
            odom_v = float("nan")
            if len(mid_odom) >= 3:
                dt = mid_odom[-1][0] - mid_odom[0][0]
                dd = math.hypot(mid_odom[-1][1] - mid_odom[0][1],
                                mid_odom[-1][2] - mid_odom[0][2])
                odom_v = dd / dt if dt > 0 else float("nan")
            yaw_drift = math.degrees(data["gyro_dyaw"])
            ratio = ground_v / v_cmd if v_cmd else float("nan")
            label = ("전진" if direction > 0 else "후진") if args.axis == "x" \
                else ("좌" if direction > 0 else "우")
            wall_str = f"{wall0:.2f}m" if wall0 is not None else "없음"
            print(f"  [{label}] 지면 {ground_v:.3f} m/s ({ratio*100:.0f}%)  "
                  f"odom {odom_v:.3f} m/s  yaw드리프트 {yaw_drift:+.1f}deg  "
                  f"진행방향 벽 {wall_str}")
            step["runs"].append({
                "dir": direction, "ground_mps": ground_v, "odom_mps": odom_v,
                "ratio": ratio, "yaw_drift_deg": yaw_drift,
                "wall_before_m": wall0, "wall_samples": len(data["wall_ts"]),
            })
            time.sleep(0.5)
        results.append(step)
        ratios = [r["ratio"] for r in step["runs"] if math.isfinite(r["ratio"])]
        if ratios and max(ratios) < args.sat_ratio:
            saturated += 1
            if saturated >= 2:
                print("\n포화 2스텝 연속 — 램프 종료")
                break
        else:
            saturated = 0
    return results


def measure_yaw(node, args):
    ramp = [float(v) for v in args.yaw_ramp.split(",")]
    results = []
    plateau = 0
    prev_rate = 0.0
    for wz_cmd in ramp:
        t_burst = min(max(math.radians(200) / wz_cmd, 0.8), 3.0)
        step = {"cmd_rad_s": wz_cmd, "burst_s": round(t_burst, 2), "runs": []}
        print(f"\n== wz {wz_cmd:.1f} rad/s ({math.degrees(wz_cmd):.0f} deg/s, "
              f"버스트 {t_burst:.1f}s) ==")
        for direction in (+1, -1):
            node.calibrate_gyro_bias()
            data = node.burst(0.0, 0.0, direction * wz_cmd, t_burst)
            gyro_dyaw = data["gyro_dyaw"]
            odom_ts = data["odom_ts"]
            odom_dyaw = ang_diff(odom_ts[-1][3], odom_ts[0][3]) if len(odom_ts) >= 2 \
                else float("nan")
            # unwrap 대신 gyro 적분으로 실회전량 판단 (odom은 ±pi wrap 가능)
            imu_rate = data["imu_msgs"] / data["elapsed"]
            gyro_mean_rate = gyro_dyaw / data["elapsed"]
            achieve = abs(gyro_mean_rate) / wz_cmd
            rate_drop = imu_rate / max(node.imu_rate_baseline, 1.0)
            print(f"  [{'CCW' if direction > 0 else 'CW'}] "
                  f"gyro 평균 {gyro_mean_rate:+.2f} rad/s (달성 {achieve*100:.0f}%) "
                  f"peak {node.gyro_peak:.2f}  odom dyaw {math.degrees(odom_dyaw):+.0f}deg  "
                  f"IMU {imu_rate:.0f} Hz (기준 대비 {rate_drop*100:.0f}%)")
            step["runs"].append({
                "dir": direction, "gyro_mean_rad_s": gyro_mean_rate,
                "gyro_peak_rad_s": node.gyro_peak, "achieve_ratio": achieve,
                "imu_hz": imu_rate, "imu_hz_ratio": rate_drop,
                "odom_dyaw_deg": math.degrees(odom_dyaw),
            })
            time.sleep(0.8)
        results.append(step)
        best = max(abs(r["gyro_mean_rad_s"]) for r in step["runs"])
        if best < prev_rate * 1.05:  # 더 못 빨라짐 = 구동계 포화
            plateau += 1
            if plateau >= 2:
                print("\n회전속도 평탄화 2스텝 연속 — 램프 종료")
                break
        else:
            plateau = 0
        prev_rate = max(prev_rate, best)
        imu_bad = any(r["imu_hz_ratio"] < 0.8 for r in step["runs"])
        if imu_bad:
            print("\nIMU 수신율 20%+ 하락 — 이 속도부터 IMU 추적 불가 판정, 종료")
            break
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", choices=["x", "y", "yaw"], required=True)
    parser.add_argument("--wall", choices=["front", "left", "right"], default="front",
                        help="y축 측정 시 벽 위치 (left/right)")
    parser.add_argument("--ramp", default="0.3,0.5,0.7,0.9,1.1,1.3",
                        help="linear 명령 속도 목록 (m/s)")
    parser.add_argument("--yaw-ramp", default="1.0,2.0,3.0,4.0,5.0,6.0",
                        help="yaw 명령 속도 목록 (rad/s)")
    parser.add_argument("--budget-m", type=float, default=1.0,
                        help="버스트당 이동 거리 예산")
    parser.add_argument("--margin-m", type=float, default=0.6,
                        help="벽 접근 시 최소 잔여 여유")
    parser.add_argument("--sat-ratio", type=float, default=0.88)
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--imu-topic", default="/imu/data")
    parser.add_argument("--odom-topic", default="/chassis/odom")
    parser.add_argument("--scan-yaw", type=float, default=0.0)
    parser.add_argument("--yaw-axis", default=None, help='"x,y,z" gyro 투영축')
    args = parser.parse_args()
    args.yaw_axis = tuple(float(v) for v in args.yaw_axis.split(",")) \
        if args.yaw_axis else DEFAULT_YAW_AXIS
    if args.axis == "y" and args.wall == "front":
        parser.error("--axis y에는 --wall left 또는 right가 필요")

    rclpy.init()
    node = MaxSpeedTester(args)
    t0 = time.time()  # DDS discovery가 수 초 걸릴 수 있어 필수 토픽은 재시도 대기
    while node.odom is None and time.time() - t0 < 15.0:
        node.spin_for(0.5)
    node.spin_for(1.0)
    if node.odom is None:
        print(f"오류: {args.odom_topic} 미수신 — 브리지 기동/odom_topic 확인"); return
    if node.imu_count == 0:
        print(f"경고: IMU({args.imu_topic}) 미수신 — yaw 측정 불가")
        if args.axis == "yaw":
            return
    if args.axis != "yaw" and node.scan is None:
        print(f"오류: {args.scan_topic} 미수신 — 라이다 기동 확인"); return

    results = measure_yaw(node, args) if args.axis == "yaw" \
        else measure_linear(node, args)

    out = f"{REPO_LOG_DIR}/max_speed_{args.axis}_{time.strftime('%H%M%S')}.json"
    with open(out, "w") as f:
        json.dump({"axis": args.axis, "wall": args.wall, "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out}")
    print("권장 운용 최대치 = 측정 포화속도의 80% (real.yaml max_linear_*/max_angular 반영)")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
