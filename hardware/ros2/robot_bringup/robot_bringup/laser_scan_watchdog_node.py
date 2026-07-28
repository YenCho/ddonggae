#!/usr/bin/env python3
import math
import threading
import time
from array import array

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan


class LaserScanWatchdogNode(Node):
    """Republish lidar scans at a steady rate and bridge short scan dropouts."""

    def __init__(self):
        super().__init__("laser_scan_watchdog_node")

        self.scan_in_topic = self.declare_parameter("scan_in_topic", "/scan_raw").value
        self.extra_scan_in_topics = self.declare_parameter("extra_scan_in_topics", "").value
        self.scan_out_topic = self.declare_parameter("scan_out_topic", "/laser_scan").value
        self.extra_scan_out_topics = self.declare_parameter("extra_scan_out_topics", "/scan").value
        self.frame_id = self.declare_parameter("frame_id", "base_scan").value
        self.qos_depth = max(1, int(self.declare_parameter("qos_depth", 50).value))
        self.publish_rate_hz = max(
            1.0,
            float(self.declare_parameter("publish_rate_hz", 15.0).value),
        )
        self.raw_fresh_timeout_sec = float(
            self.declare_parameter("raw_fresh_timeout_sec", 0.35).value
        )
        self.hold_stale_scan_sec = float(
            self.declare_parameter("hold_stale_scan_sec", 1.25).value
        )
        self.publish_on_receive = bool(
            self.declare_parameter("publish_on_receive", True).value
        )
        self.republish_fresh_scan = bool(
            self.declare_parameter("republish_fresh_scan", False).value
        )
        self.restamp_scans = bool(
            self.declare_parameter("restamp_scans", False).value
        )
        self.publish_empty_scan = bool(
            self.declare_parameter("publish_empty_scan", True).value
        )
        self.angle_min = float(self.declare_parameter("angle_min", -math.pi).value)
        self.angle_max = float(self.declare_parameter("angle_max", math.pi).value)
        self.angle_increment = float(
            self.declare_parameter("angle_increment", math.radians(1.0)).value
        )
        self.range_min = float(self.declare_parameter("range_min", 0.05).value)
        self.range_max = float(self.declare_parameter("range_max", 8.0).value)

        self.latest_scan = None
        self.latest_scan_time_sec = 0.0
        self.latest_source_topic = ""
        self.last_state = ""
        self.lock = threading.Lock()
        self.callback_group = ReentrantCallbackGroup()
        self.scan_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=self.qos_depth,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.scan_out_topics = self.topic_list(
            self.scan_out_topic,
            self.extra_scan_out_topics,
        )
        self.scan_in_topics = [
            topic
            for topic in self.topic_list(self.scan_in_topic, self.extra_scan_in_topics)
            if topic not in self.scan_out_topics
        ]
        if not self.scan_in_topics:
            raise RuntimeError("laser scan watchdog has no input topics after filtering output aliases")
        self.scan_subscriptions = [
            self.create_subscription(
                LaserScan,
                topic,
                lambda msg, topic=topic: self.scan_callback(msg, topic),
                self.scan_qos,
                callback_group=self.callback_group,
            )
            for topic in self.scan_in_topics
        ]
        self.scan_pubs = [
            self.create_publisher(LaserScan, topic, self.scan_qos)
            for topic in self.scan_out_topics
        ]
        self.timer = self.create_timer(
            1.0 / self.publish_rate_hz,
            self.timer_callback,
            callback_group=self.callback_group,
        )
        self.get_logger().info(
            "laser scan watchdog ready: "
            f"{','.join(self.scan_in_topics)} -> {','.join(self.scan_out_topics)} "
            f"at {self.publish_rate_hz:.1f} Hz, qos_depth={self.qos_depth}"
        )

    @staticmethod
    def topic_list(primary, extra):
        topics = []
        for value in (primary, extra):
            if isinstance(value, str):
                candidates = value.replace(";", ",").split(",")
            else:
                candidates = list(value or [])
            for candidate in candidates:
                topic = str(candidate).strip()
                if topic and topic not in topics:
                    topics.append(topic)
        return topics

    def scan_callback(self, msg: LaserScan, source_topic: str):
        now = self.now_sec()
        with self.lock:
            self.latest_scan = msg
            self.latest_scan_time_sec = now
            self.latest_source_topic = source_topic
        if self.publish_on_receive:
            self.publish_scan(msg, "raw", source_topic, 0.0)

    def timer_callback(self):
        now = self.now_sec()
        with self.lock:
            latest_scan = self.latest_scan
            latest_scan_time_sec = self.latest_scan_time_sec
            latest_source_topic = self.latest_source_topic

        age = now - latest_scan_time_sec if latest_scan is not None else math.inf
        if latest_scan is not None and age <= self.raw_fresh_timeout_sec:
            if self.republish_fresh_scan:
                self.publish_scan(latest_scan, "raw", latest_source_topic, age)
            return

        if latest_scan is not None and age <= self.hold_stale_scan_sec:
            self.publish_scan(latest_scan, "held", latest_source_topic, age)
            return

        if self.publish_empty_scan:
            self.publish_to_all(self.empty_scan())
            self.log_state("empty", age, latest_source_topic)

    def publish_scan(self, scan: LaserScan, state: str, source_topic: str, age: float):
        msg = LaserScan()
        msg.header = scan.header
        if self.restamp_scans:
            msg.header.stamp = self.get_clock().now().to_msg()
        if not msg.header.frame_id:
            msg.header.frame_id = self.frame_id
        msg.angle_min = scan.angle_min
        msg.angle_max = scan.angle_max
        msg.angle_increment = scan.angle_increment
        msg.time_increment = scan.time_increment
        msg.scan_time = scan.scan_time
        msg.range_min = scan.range_min
        msg.range_max = scan.range_max
        msg.ranges = scan.ranges
        msg.intensities = scan.intensities
        self.publish_to_all(msg)
        self.log_state(state, age, source_topic)

    def publish_to_all(self, msg: LaserScan):
        for publisher in self.scan_pubs:
            publisher.publish(msg)

    def empty_scan(self) -> LaserScan:
        scan = LaserScan()
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.header.frame_id = str(self.frame_id)
        scan.angle_min = float(self.angle_min)
        scan.angle_max = float(self.angle_max)
        scan.angle_increment = float(self.angle_increment)
        count = max(
            1,
            int(round((scan.angle_max - scan.angle_min) / scan.angle_increment)) + 1,
        )
        scan.scan_time = 1.0 / self.publish_rate_hz
        scan.time_increment = scan.scan_time / float(max(1, count - 1))
        scan.range_min = float(self.range_min)
        scan.range_max = float(self.range_max)
        scan.ranges = array("f", [math.inf]) * count
        scan.intensities = array("f", [0.0]) * count
        return scan

    def log_state(self, state: str, age: float, source_topic: str):
        state_key = f"{state}:{source_topic}"
        if state_key == self.last_state:
            return
        self.last_state = state_key
        if state == "raw":
            self.get_logger().info(f"laser scan source recovered: {source_topic or 'raw lidar'}")
        elif state == "held":
            self.get_logger().warn(
                f"laser scan input {source_topic or '<none>'} stale for {age:.2f}s; holding last scan"
            )
        else:
            self.get_logger().warn(
                f"laser scan input unavailable for {age:.2f}s; publishing empty scan"
            )

    def now_sec(self) -> float:
        return time.monotonic()


def main(args=None):
    rclpy.init(args=args)
    node = LaserScanWatchdogNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
