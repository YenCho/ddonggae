#!/usr/bin/env python3
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String


class GripperJointCommandBridge(Node):
    """Convert Isaac-style gripper JointState commands to real gripper strings."""

    def __init__(self):
        super().__init__("gripper_joint_command_bridge")

        self.declare_parameter("joint_command_topic", "/mk1/gripper_joint_command")
        self.declare_parameter("string_command_topic", "/gripper/command")
        self.declare_parameter("open_command", "OPEN")
        self.declare_parameter("close_command", "CLOSE")
        self.declare_parameter("two_finger_aperture_threshold", 0.0)
        self.declare_parameter("single_joint_open_threshold", 0.02)
        self.declare_parameter("repeat_unchanged_after_sec", 2.0)

        self.last_command = ""
        self.last_publish_time = 0.0

        self.command_pub = self.create_publisher(
            String,
            self.get_parameter("string_command_topic").value,
            10,
        )
        self.command_sub = self.create_subscription(
            JointState,
            self.get_parameter("joint_command_topic").value,
            self.command_callback,
            10,
        )

        self.get_logger().info(
            "bridging "
            + str(self.get_parameter("joint_command_topic").value)
            + " -> "
            + str(self.get_parameter("string_command_topic").value)
        )

    def infer_command(self, positions):
        if len(positions) >= 2:
            aperture = float(positions[-1]) - float(positions[0])
            threshold = float(
                self.get_parameter("two_finger_aperture_threshold").value
            )
            if aperture >= threshold:
                return str(self.get_parameter("open_command").value)
            return str(self.get_parameter("close_command").value)

        threshold = float(self.get_parameter("single_joint_open_threshold").value)
        if float(positions[0]) >= threshold:
            return str(self.get_parameter("open_command").value)
        return str(self.get_parameter("close_command").value)

    def command_callback(self, msg):
        if not msg.position:
            self.get_logger().warn(
                "received gripper JointState without positions",
                throttle_duration_sec=2.0,
            )
            return

        command = self.infer_command(msg.position)
        now = time.monotonic()
        repeat_after = float(self.get_parameter("repeat_unchanged_after_sec").value)
        if command == self.last_command and now - self.last_publish_time < repeat_after:
            return

        out = String()
        out.data = command
        self.command_pub.publish(out)
        self.last_command = command
        self.last_publish_time = now
        self.get_logger().info("published gripper command: " + command)


def main(args=None):
    rclpy.init(args=args)
    node = GripperJointCommandBridge()
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
