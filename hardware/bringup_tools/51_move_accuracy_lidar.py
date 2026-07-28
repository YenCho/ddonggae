#!/usr/bin/env python3
"""50번 대체: /base/move_relative 주행 정밀도 — LiDAR localization을 그라운드트루스로.

각 이동 전후 arena status pose(2초 중앙값)와 /chassis/odom pose를 기록,
시작 yaw 기준 body frame으로 변환해 명령 대비 실이동/odom 이동을 비교한다.
사용: move_accuracy_lidar.py [테스트번호들 예: 0,1  생략=전부]
"""
import json
import math
import statistics
import sys
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String

TESTS = [
    ("전진 0.5m",  {"dx": 0.5,  "dy": 0.0,  "dyaw": 0.0}),
    ("후진 0.5m",  {"dx": -0.5, "dy": 0.0,  "dyaw": 0.0}),
    ("좌 스트레이프 0.4m", {"dx": 0.0, "dy": 0.4,  "dyaw": 0.0}),
    ("우 스트레이프 0.4m", {"dx": 0.0, "dy": -0.4, "dyaw": 0.0}),
    ("좌회전 +90도", {"dx": 0.0, "dy": 0.0, "dyaw": math.pi / 2}),
    ("우회전 -90도", {"dx": 0.0, "dy": 0.0, "dyaw": -math.pi / 2}),
]


def yaw_of(odom):
    q = odom.pose.pose.orientation
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))


def ang_diff(a, b):
    return math.remainder(a - b, 2 * math.pi)


class Tester(Node):
    def __init__(self):
        super().__init__("move_accuracy_lidar")
        self.status_pose = None
        self.odom_pose = None
        self.move_result = None
        self.create_subscription(String, "/arena_lightweight/status", self.on_status, 10)
        self.create_subscription(Odometry, "/chassis/odom", self.on_odom, 10)
        self.create_subscription(String, "/base/move_result", self.on_result, 10)
        self.move_pub = self.create_publisher(String, "/base/move_relative", 10)

    def on_status(self, msg):
        try:
            p = json.loads(msg.data)["pose"]
            self.status_pose = (p["x"], p["y"], p["yaw"])
        except (ValueError, KeyError):
            pass

    def on_odom(self, msg):
        p = msg.pose.pose.position
        self.odom_pose = (p.x, p.y, yaw_of(msg))

    def on_result(self, msg):
        try:
            self.move_result = json.loads(msg.data)
        except ValueError:
            self.move_result = {"ok": False, "reason": msg.data}

    def sample_pose(self, seconds=2.0):
        xs, ys, yaws, oxs, oys, oyaws = [], [], [], [], [], []
        t0 = time.time()
        while time.time() - t0 < seconds:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.status_pose:
                xs.append(self.status_pose[0]); ys.append(self.status_pose[1])
                yaws.append(self.status_pose[2])
            if self.odom_pose:
                oxs.append(self.odom_pose[0]); oys.append(self.odom_pose[1])
                oyaws.append(self.odom_pose[2])
        med = lambda v: statistics.median(v) if v else float("nan")
        return ((med(xs), med(ys), med(yaws)), (med(oxs), med(oys), med(oyaws)))

    def move(self, payload, timeout=25.0):
        self.move_result = None
        self.move_pub.publish(String(data=json.dumps(payload)))
        t0 = time.time()
        while self.move_result is None and time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.move_result


def body_delta(before, after):
    """world pose 변화 → 시작 yaw 기준 body frame (dx 전방, dy 좌측, dyaw)."""
    dx_w = after[0] - before[0]
    dy_w = after[1] - before[1]
    c, s = math.cos(before[2]), math.sin(before[2])
    return (c * dx_w + s * dy_w, -s * dx_w + c * dy_w, ang_diff(after[2], before[2]))


def main():
    sel = [int(i) for i in sys.argv[1].split(",")] if len(sys.argv) > 1 else range(len(TESTS))
    rclpy.init()
    node = Tester()
    t0 = time.time()
    while (node.status_pose is None or node.odom_pose is None) and time.time() - t0 < 10:
        rclpy.spin_once(node, timeout_sec=0.2)
    if node.status_pose is None or node.odom_pose is None:
        print(f"토픽 대기 실패: status={node.status_pose is not None} odom={node.odom_pose is not None}")
        return
    report = []
    for i in sel:
        name, cmd = TESTS[i]
        lidar0, odom0 = node.sample_pose()
        if math.isnan(lidar0[0]):
            print("localization pose 미수신"); break
        print(f"\n== [{i}] {name} ==")
        res = node.move(dict(cmd))
        ok = res and res.get("ok")
        print(f"  move_result: ok={ok} reason={res.get('reason') if res else 'timeout'} "
              f"elapsed={res.get('elapsed_sec') if res else None}")
        time.sleep(1.0)
        lidar1, odom1 = node.sample_pose()
        bl = body_delta(lidar0, lidar1)
        bo = body_delta(odom0, odom1)
        print(f"  명령      dx={cmd['dx']:+.3f} dy={cmd['dy']:+.3f} dyaw={math.degrees(cmd['dyaw']):+.1f}deg")
        print(f"  LiDAR실측 dx={bl[0]:+.3f} dy={bl[1]:+.3f} dyaw={math.degrees(bl[2]):+.1f}deg")
        print(f"  odom      dx={bo[0]:+.3f} dy={bo[1]:+.3f} dyaw={math.degrees(bo[2]):+.1f}deg")
        for axis, idx in (("dx", 0), ("dy", 1)):
            if abs(cmd[axis]) > 1e-6:
                print(f"  {axis} 전달률: LiDAR {bl[idx]/cmd[axis]*100:+.0f}%  odom {bo[idx]/cmd[axis]*100:+.0f}%")
        if abs(cmd["dyaw"]) > 1e-6:
            print(f"  dyaw 전달률: LiDAR {bl[2]/cmd['dyaw']*100:+.0f}%  odom {bo[2]/cmd['dyaw']*100:+.0f}%")
        report.append({"test": name, "cmd": cmd, "ok": bool(ok),
                       "lidar": bl, "odom": bo,
                       "reason": res.get("reason") if res else "timeout"})
        time.sleep(0.5)

    log_dir = Path(__file__).resolve().parents[2] / "logs" / "mecanum_bringup"
    log_dir.mkdir(parents=True, exist_ok=True)
    out = str(log_dir / ("move_accuracy_" + time.strftime("%H%M%S") + ".json"))
    with open(out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
