#!/usr/bin/env python3
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


@dataclass
class CmdVelSource:
    name: str
    topic: str
    timeout_sec: float
    last_msg: Twist | None = None
    last_time_sec: float = 0.0


class CmdVelMuxNode(Node):
    """Priority mux for real robot velocity commands."""

    def __init__(self):
        super().__init__("cmd_vel_mux_node")

        self.declare_parameter("direct_topic", "/cmd_vel_direct")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("nav_topic", "/cmd_vel_nav")
        self.declare_parameter("output_topic", "/cmd_vel_motor")
        self.declare_parameter("direct_timeout_sec", 0.4)
        self.declare_parameter("cmd_vel_timeout_sec", 0.4)
        self.declare_parameter("nav_timeout_sec", 0.4)
        self.declare_parameter("publish_rate_hz", 30.0)

        self.sources = [
            CmdVelSource(
                "direct",
                str(self.get_parameter("direct_topic").value),
                float(self.get_parameter("direct_timeout_sec").value),
            ),
            CmdVelSource(
                "cmd_vel",
                str(self.get_parameter("cmd_vel_topic").value),
                float(self.get_parameter("cmd_vel_timeout_sec").value),
            ),
            CmdVelSource(
                "nav",
                str(self.get_parameter("nav_topic").value),
                float(self.get_parameter("nav_timeout_sec").value),
            ),
        ]

        self.pub = self.create_publisher(
            Twist,
            str(self.get_parameter("output_topic").value),
            10,
        )
        self.last_output = Twist()
        self.last_source = ""
        for source in self.sources:
            self.create_subscription(
                Twist,
                source.topic,
                lambda msg, item=source: self.command_callback(item, msg),
                10,
            )

        rate_hz = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.timer = self.create_timer(1.0 / rate_hz, self.timer_callback)
        self.get_logger().info(
            "cmd_vel mux ready: "
            + ", ".join(f"{source.name}={source.topic}" for source in self.sources)
            + " -> "
            + str(self.get_parameter("output_topic").value)
        )

    def command_callback(self, source: CmdVelSource, msg: Twist):
        source.last_msg = msg
        source.last_time_sec = self.now_sec()

    def timer_callback(self):
        now = self.now_sec()
        # 0속도 명령은 하위 소스를 가리지 않는다: arena 노드가 idle에도
        # /cmd_vel_direct 로 0을 상시 발행해 teleop(/cmd_vel)이 영구 차단되던
        # 문제(2026-07-17) — 0이 아닌 최상위 소스를 고르고, 전부 0이면
        # 기존처럼 최상위 활성 소스의 0을 내보낸다.
        selected = None
        fallback = None
        for source in self.sources:
            if source.last_msg is None:
                continue
            if now - source.last_time_sec > source.timeout_sec:
                continue
            if fallback is None:
                fallback = source
            if self.is_zero(source.last_msg):
                continue
            selected = source
            break
        if selected is None:
            selected = fallback

        if selected is None:
            self.pub.publish(Twist())
            self.last_output = Twist()
            self.last_source = ""
            return

        self.pub.publish(selected.last_msg)
        self.last_output = selected.last_msg
        if selected.name != self.last_source:
            self.last_source = selected.name
            self.get_logger().info("selected cmd_vel source: " + selected.name)

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def is_zero(msg: Twist) -> bool:
        values = (
            msg.linear.x,
            msg.linear.y,
            msg.linear.z,
            msg.angular.x,
            msg.angular.y,
            msg.angular.z,
        )
        return all(abs(float(value)) <= 1e-9 for value in values)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelMuxNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
