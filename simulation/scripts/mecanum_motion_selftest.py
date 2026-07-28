"""Host-side motion self-test for the Isaac mecanum scene.

Runs on the workstation ROS 2 Humble install (NOT inside Isaac python) while
create_mecanum_competition_scene.py --run is up. Publishes body twists on
/cmd_vel and verifies against /sim/ground_truth_pose that the robot actually
moves the commanded way:

  phase forward: +vx moves along the robot heading
  phase strafe:  +vy moves 90 deg left of the heading (the mecanum check)
  phase rotate:  +wz increases yaw with little translation
  final:         wheel odometry stays close to ground truth

Exits 0 and prints a JSON verdict when all checks pass; exits 1 otherwise.

Usage:
  source /opt/ros/humble/setup.bash
  python3 sim/isaacsim/scripts/mecanum_motion_selftest.py [--wait-sec 120]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from std_msgs.msg import String


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class SelfTest:
    def __init__(self, node):
        self.node = node
        self.cmd_pub = node.create_publisher(Twist, "/cmd_vel", 10)
        self.ground_truth = None
        node.create_subscription(String, "/sim/ground_truth_pose", self._gt_callback, 10)

    def _gt_callback(self, msg):
        try:
            self.ground_truth = json.loads(msg.data)
        except ValueError:
            pass

    def spin_for(self, seconds: float, twist: Twist | None = None, rate_hz: float = 20.0):
        deadline = time.monotonic() + seconds
        period = 1.0 / rate_hz
        while time.monotonic() < deadline:
            if twist is not None:
                self.cmd_pub.publish(twist)
            rclpy.spin_once(self.node, timeout_sec=period)

    def wait_for_ground_truth(self, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.2)
            if self.ground_truth is not None:
                return True
        return False

    def snapshot(self) -> dict:
        return dict(self.ground_truth)

    def drive(self, vx: float, vy: float, wz: float, seconds: float) -> tuple[dict, dict, float]:
        """Drive for `seconds` of wall time; return (start, end, sim_elapsed).

        The sim may run slower than real time (physics at 240 Hz), so expected
        displacements are computed from the sim-clock delta, not wall time.
        """
        start = self.snapshot()
        twist = Twist()
        twist.linear.x = vx
        twist.linear.y = vy
        twist.angular.z = wz
        self.spin_for(seconds, twist)
        mid = self.snapshot()  # end of the commanded window
        self.spin_for(1.5, Twist())  # stop and settle
        end = self.snapshot()
        sim_elapsed = float(mid.get("sim_time", 0.0)) - float(start.get("sim_time", 0.0))
        if sim_elapsed <= 0.0:
            sim_elapsed = seconds
        return start, end, sim_elapsed


def direction_error_deg(dx: float, dy: float, expected_rad: float) -> float:
    actual = math.atan2(dy, dx)
    return abs(math.degrees(normalize_angle(actual - expected_rad)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-sec", type=float, default=120.0)
    parser.add_argument("--vx", type=float, default=0.2)
    parser.add_argument("--vy", type=float, default=0.15)
    parser.add_argument("--wz", type=float, default=0.5)
    parser.add_argument("--drive-sec", type=float, default=3.0)
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node("mecanum_motion_selftest")
    test = SelfTest(node)

    results: dict[str, object] = {"ok": False, "checks": []}

    def check(name: str, ok: bool, detail: dict):
        results["checks"].append({"name": name, "ok": bool(ok), **detail})
        return ok

    try:
        if not test.wait_for_ground_truth(args.wait_sec):
            results["error"] = "no /sim/ground_truth_pose within wait window"
            print(json.dumps(results, indent=2))
            return 1
        test.spin_for(2.0)

        all_ok = True

        # Phase 1: forward (+vx along heading)
        start, end, sim_elapsed = test.drive(args.vx, 0.0, 0.0, args.drive_sec)
        dx, dy = end["x"] - start["x"], end["y"] - start["y"]
        dist = math.hypot(dx, dy)
        expected = args.vx * sim_elapsed
        ok = 0.6 * expected <= dist <= 1.3 * expected and direction_error_deg(dx, dy, start["yaw"]) <= 25.0
        all_ok &= check(
            "forward_vx",
            ok,
            {
                "dist_m": round(dist, 3),
                "expected_m": round(expected, 3),
                "sim_elapsed_s": round(sim_elapsed, 2),
                "direction_err_deg": round(direction_error_deg(dx, dy, start["yaw"]), 1),
            },
        )

        # Phase 2: strafe left (+vy at heading + 90 deg) - the mecanum check.
        start, end, sim_elapsed = test.drive(0.0, args.vy, 0.0, args.drive_sec)
        dx, dy = end["x"] - start["x"], end["y"] - start["y"]
        dist = math.hypot(dx, dy)
        expected = args.vy * sim_elapsed
        expected_dir = start["yaw"] + math.pi / 2.0
        ok = 0.5 * expected <= dist <= 1.3 * expected and direction_error_deg(dx, dy, expected_dir) <= 30.0
        all_ok &= check(
            "strafe_vy",
            ok,
            {
                "dist_m": round(dist, 3),
                "expected_m": round(expected, 3),
                "sim_elapsed_s": round(sim_elapsed, 2),
                "direction_err_deg": round(direction_error_deg(dx, dy, expected_dir), 1),
            },
        )

        # Phase 3: rotate in place (+wz CCW)
        start, end, sim_elapsed = test.drive(0.0, 0.0, args.wz, args.drive_sec)
        dyaw = normalize_angle(end["yaw"] - start["yaw"])
        drift = math.hypot(end["x"] - start["x"], end["y"] - start["y"])
        expected = args.wz * sim_elapsed
        ok = 0.5 * expected <= dyaw <= 1.3 * expected and drift <= 0.30
        all_ok &= check(
            "rotate_wz",
            ok,
            {
                "dyaw_rad": round(dyaw, 3),
                "expected_rad": round(expected, 3),
                "sim_elapsed_s": round(sim_elapsed, 2),
                "xy_drift_m": round(drift, 3),
            },
        )

        # Final: wheel odometry vs ground truth agreement.
        gt = test.snapshot()
        odom = gt.get("odom", {})
        err = math.hypot(gt["x"] - odom.get("x", 1e9), gt["y"] - odom.get("y", 1e9))
        yaw_err = abs(normalize_angle(gt["yaw"] - odom.get("yaw", 1e9)))
        ok = err <= 0.30 and yaw_err <= 0.35
        all_ok &= check("odom_vs_ground_truth", ok, {"xy_err_m": round(err, 3), "yaw_err_rad": round(yaw_err, 3)})

        results["ok"] = bool(all_ok)
        print(json.dumps(results, indent=2))
        return 0 if all_ok else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
