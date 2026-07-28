#!/usr/bin/env python3

import json
import math
import glob
import os
import time
from pathlib import Path

import rclpy
import serial

from geometry_msgs.msg import Quaternion, TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from tf2_ros import TransformBroadcaster

from robot_hardware.base_control import (
    DifferentialOdometry,
    EncoderSample,
    WheelPiController,
    load_pwm_calibration,
    parse_encoder_line,
    summarize_pwm_calibration,
    twist_to_wheel_speeds,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CALIBRATION_CSV = (
    'perception/calibration/arduino_motor/'
    'pwm_encoder_sweep_20260527_231622_speed_analysis_cpr2464.csv'
)


def resolve_repo_path(path_value):
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def yaw_to_quaternion(yaw):
    half = yaw * 0.5
    return Quaternion(z=math.sin(half), w=math.cos(half))


class MotorBridgeNode(Node):
    def __init__(self):
        super().__init__('motor_bridge_node')

        self.declare_parameter(
            'serial_port',
            '/dev/serial/by-id/'
            'usb-Arduino__www.arduino.cc__Arduino_14101-if00',
        )
        self.declare_parameter(
            'serial_port_candidates',
            '/dev/serial/by-id/usb-Arduino__www.arduino.cc__Arduino_14101-if00,'
            '/dev/serial/by-path/platform-3610000.usb-usb-0:2.3.1:1.0,'
            '/dev/ttyACM0',
        )
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('serial_reconnect_interval_sec', 1.0)
        self.declare_parameter('serial_boot_delay_sec', 0.5)
        self.declare_parameter('serial_write_timeout_sec', 0.1)
        self.declare_parameter('dry_run', False)
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('e_stop_topic', '/safety/e_stop')
        self.declare_parameter('odom_topic', '/odom/wheel')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('publish_tf', False)
        self.declare_parameter('wheel_radius_m', 0.044)
        self.declare_parameter('wheel_separation_m', 0.25)
        self.declare_parameter('encoder_cpr', 2464)
        self.declare_parameter('left_encoder_sign', 1)
        self.declare_parameter('right_encoder_sign', -1)
        self.declare_parameter('left_motor_sign', 1)
        self.declare_parameter('right_motor_sign', 1)
        self.declare_parameter('swap_wheel_commands', False)
        self.declare_parameter('swap_wheel_odometry', False)
        self.declare_parameter('max_linear_mps', 0.30)
        self.declare_parameter('max_angular_rps', 1.20)
        self.declare_parameter('max_wheel_rad_s', 8.0)
        self.declare_parameter('command_timeout_sec', 0.5)
        self.declare_parameter(
            'calibration_csv',
            DEFAULT_CALIBRATION_CSV,
        )
        self.declare_parameter('min_calibration_points', 2)
        self.declare_parameter('min_calibration_duration_sec', 2.0)

        # Feedforward values come from the 2026-05-27 unloaded sweep.
        # PI feedback supplies the extra PWM needed after payload increases.
        self.declare_parameter('min_pwm', 72.0)
        self.declare_parameter('max_pwm', 220.0)
        self.declare_parameter('feedforward_slope', 18.5)
        self.declare_parameter('left_pwm_scale', 0.92)
        self.declare_parameter('right_pwm_scale', 1.08)
        self.declare_parameter('speed_kp', 12.0)
        self.declare_parameter('speed_ki', 7.0)
        self.declare_parameter('integral_limit', 8.0)

        self.wheel_radius = float(
            self.get_parameter('wheel_radius_m').value
        )
        self.wheel_separation = float(
            self.get_parameter('wheel_separation_m').value
        )
        self.encoder_cpr = int(self.get_parameter('encoder_cpr').value)
        self.left_encoder_sign = int(
            self.get_parameter('left_encoder_sign').value
        )
        self.right_encoder_sign = int(
            self.get_parameter('right_encoder_sign').value
        )
        self.left_motor_sign = int(
            self.get_parameter('left_motor_sign').value
        )
        self.right_motor_sign = int(
            self.get_parameter('right_motor_sign').value
        )
        self.swap_wheel_commands = bool(
            self.get_parameter('swap_wheel_commands').value
        )
        self.swap_wheel_odometry = bool(
            self.get_parameter('swap_wheel_odometry').value
        )
        self.max_linear = float(
            self.get_parameter('max_linear_mps').value
        )
        self.max_angular = float(
            self.get_parameter('max_angular_rps').value
        )
        self.max_wheel_speed = float(
            self.get_parameter('max_wheel_rad_s').value
        )
        self.command_timeout = float(
            self.get_parameter('command_timeout_sec').value
        )
        self.dry_run = bool(self.get_parameter('dry_run').value)

        calibration = None
        calibration_csv_value = str(
            self.get_parameter('calibration_csv').value
        ).strip()
        calibration_csv = (
            str(resolve_repo_path(calibration_csv_value))
            if calibration_csv_value
            else ''
        )
        self.calibration_status = {
            'path': calibration_csv,
            'loaded': False,
            'ok': False,
            'error': 'calibration_csv is not configured',
        }
        if calibration_csv:
            self.calibration_status = summarize_pwm_calibration(
                calibration_csv,
                int(self.get_parameter('min_calibration_points').value),
                float(
                    self.get_parameter(
                        'min_calibration_duration_sec'
                    ).value
                ),
            )
            try:
                calibration = load_pwm_calibration(calibration_csv)
                self.get_logger().info(
                    f'loaded PWM calibration: {calibration_csv}'
                )
            except (OSError, ValueError) as exc:
                self.get_logger().warn(
                    f'PWM calibration unavailable, using fallback: {exc}'
                )
            if not self.calibration_status.get('ok', False):
                self.get_logger().warn(
                    'PWM calibration does not meet loaded-robot criteria: '
                    + str(self.calibration_status.get('error', 'unknown'))
                )

        common = {
            'min_pwm': self.get_parameter('min_pwm').value,
            'max_pwm': self.get_parameter('max_pwm').value,
            'feedforward_slope': self.get_parameter(
                'feedforward_slope'
            ).value,
            'kp': self.get_parameter('speed_kp').value,
            'ki': self.get_parameter('speed_ki').value,
            'integral_limit': self.get_parameter(
                'integral_limit'
            ).value,
        }
        self.left_controller = WheelPiController(
            **common,
            scale=self.get_parameter('left_pwm_scale').value,
            positive_curve=(
                calibration['left_forward'] if calibration else None
            ),
            negative_curve=(
                calibration['left_reverse'] if calibration else None
            ),
        )
        self.right_controller = WheelPiController(
            **common,
            scale=self.get_parameter('right_pwm_scale').value,
            positive_curve=(
                calibration['right_forward'] if calibration else None
            ),
            negative_curve=(
                calibration['right_reverse'] if calibration else None
            ),
        )
        odom_left_sign = self.left_encoder_sign
        odom_right_sign = self.right_encoder_sign
        if self.swap_wheel_odometry:
            odom_left_sign = self.right_encoder_sign
            odom_right_sign = self.left_encoder_sign
        self.odometry = DifferentialOdometry(
            self.wheel_radius,
            self.wheel_separation,
            self.encoder_cpr,
            odom_left_sign,
            odom_right_sign,
        )

        self.target_left = 0.0
        self.target_right = 0.0
        self.measured_left = 0.0
        self.measured_right = 0.0
        self.left_pwm = 0
        self.right_pwm = 0
        self.last_command_time = time.monotonic()
        self.last_feedback_time = None
        self.last_sample = None
        self.last_serial_command = None
        self.last_serial_write = 0.0
        self.last_motion_log_time = 0.0
        self.feedback_ok = False
        self.e_stop_active = False
        self.serial_port_name = str(self.get_parameter('serial_port').value)
        self.serial_error = ''
        self.baudrate = int(self.get_parameter('baudrate').value)
        self.last_serial_open_attempt_time = 0.0

        self.serial_port = None
        if not self.dry_run:
            self.open_serial(force=True)
        else:
            self.get_logger().warn('base driver is in dry_run mode')

        cmd_vel_topic = str(self.get_parameter('cmd_vel_topic').value)
        odom_topic = str(self.get_parameter('odom_topic').value)
        self.cmd_sub = self.create_subscription(
            Twist,
            cmd_vel_topic,
            self.cmd_vel_callback,
            10,
        )
        self.e_stop_sub = self.create_subscription(
            Bool,
            str(self.get_parameter('e_stop_topic').value),
            self.e_stop_callback,
            10,
        )
        self.odom_pub = self.create_publisher(Odometry, odom_topic, 20)
        self.joint_pub = self.create_publisher(
            JointState,
            '/joint_states',
            20,
        )
        self.status_pub = self.create_publisher(
            String,
            '/motor/state',
            10,
        )
        self.tf_broadcaster = TransformBroadcaster(self)
        self.read_timer = self.create_timer(0.02, self.read_serial)
        self.command_timer = self.create_timer(0.05, self.command_tick)
        self.status_timer = self.create_timer(0.5, self.publish_status)
        self.dry_run_odom_timer = None
        if self.dry_run:
            self.dry_run_odom_timer = self.create_timer(
                0.05,
                self.publish_dry_run_odometry,
            )

    def serial_port_candidates(self):
        patterns = [self.serial_port_name]
        extra = str(self.get_parameter('serial_port_candidates').value)
        patterns.extend(
            part.strip() for part in extra.split(',') if part.strip()
        )

        candidates = []
        seen = set()
        for pattern in patterns:
            if any(ch in pattern for ch in '*?[]'):
                paths = sorted(glob.glob(pattern))
            else:
                paths = [pattern]
            for path in paths:
                if path and path not in seen:
                    candidates.append(path)
                    seen.add(path)
        return candidates

    def open_serial(self, force=False):
        if self.dry_run or self.serial_port is not None:
            return

        now = time.monotonic()
        interval = float(
            self.get_parameter('serial_reconnect_interval_sec').value
        )
        if (
            not force
            and now - self.last_serial_open_attempt_time < interval
        ):
            return
        self.last_serial_open_attempt_time = now

        candidates = self.serial_port_candidates()
        present = [path for path in candidates if os.path.exists(path)]
        if not present:
            self.serial_error = (
                'serial port is not present; candidates='
                + ','.join(candidates)
            )
            self.get_logger().warn(self.serial_error, throttle_duration_sec=2.0)
            return

        port_name = present[0]
        try:
            # exclusive=True: flock the port so a stray mecanum_bridge_node
            # (or a second copy of this node from an orphaned launch) can't
            # open the same UNO concurrently and interleave commands into
            # the firmware parser (2026-07-18 field incident — corrupted
            # the firmware until USB replug).
            self.serial_port = serial.Serial(
                port_name,
                self.baudrate,
                timeout=0.0,
                write_timeout=float(
                    self.get_parameter('serial_write_timeout_sec').value
                ),
                exclusive=True,
            )
            self.serial_port_name = port_name
            time.sleep(
                float(self.get_parameter('serial_boot_delay_sec').value)
            )
            self.serial_port.reset_input_buffer()
            self.serial_port.write(b'stream 1\n')
            self.serial_error = ''
            self.get_logger().info(
                f'base serial connected: {self.serial_port_name}'
            )
        except (OSError, serial.SerialException) as exc:
            self.serial_error = str(exc)
            self.close_serial()
            self.get_logger().error(
                'failed to open base serial port '
                + port_name
                + ': '
                + self.serial_error
            )

    def close_serial(self):
        if self.serial_port is None:
            return
        try:
            self.serial_port.write(b'p 0 0\n')
        except (OSError, serial.SerialException):
            pass
        try:
            self.serial_port.close()
        except (OSError, serial.SerialException):
            pass
        self.serial_port = None

    def handle_serial_error(self, context, exc):
        self.serial_error = str(exc)
        self.feedback_ok = False
        self.left_pwm = 0
        self.right_pwm = 0
        self.target_left = 0.0
        self.target_right = 0.0
        self.last_sample = None
        self.get_logger().error(context + ': ' + self.serial_error)
        self.close_serial()

    def cmd_vel_callback(self, msg):
        if self.e_stop_active:
            self.target_left = 0.0
            self.target_right = 0.0
            self.left_pwm = 0
            self.right_pwm = 0
            self.last_command_time = time.monotonic()
            return

        linear = max(-self.max_linear, min(self.max_linear, msg.linear.x))
        angular = max(
            -self.max_angular,
            min(self.max_angular, msg.angular.z),
        )
        left, right = twist_to_wheel_speeds(
            linear,
            angular,
            self.wheel_radius,
            self.wheel_separation,
        )
        if self.swap_wheel_commands:
            left, right = right, left
        peak = max(abs(left), abs(right), self.max_wheel_speed)
        if peak > self.max_wheel_speed:
            scale = self.max_wheel_speed / peak
            left *= scale
            right *= scale
        self.target_left = left
        self.target_right = right
        self.last_command_time = time.monotonic()

        if self.last_sample is None:
            self.left_pwm = int(round(
                self.left_controller.feedforward(left)
            ))
            self.right_pwm = int(round(
                self.right_controller.feedforward(right)
            ))

    def e_stop_callback(self, msg):
        self.e_stop_active = bool(msg.data)
        if self.e_stop_active:
            self.target_left = 0.0
            self.target_right = 0.0
            self.left_pwm = 0
            self.right_pwm = 0
            self.write_serial('stop')

    def read_serial(self):
        if self.serial_port is None:
            self.open_serial()
            return

        try:
            while self.serial_port.in_waiting:
                raw = self.serial_port.readline()
                if not raw:
                    break
                line = raw.decode('utf-8', errors='replace').strip()
                sample = parse_encoder_line(line)
                if sample is not None:
                    self.handle_encoder_sample(sample)
        except (OSError, serial.SerialException) as exc:
            self.handle_serial_error('base serial read failed', exc)

    def handle_encoder_sample(self, sample):
        now = time.monotonic()
        odom_left_ticks, odom_right_ticks = self.odometry_ticks(sample)
        self.odometry.update(odom_left_ticks, odom_right_ticks)
        if self.last_sample is not None:
            delta_ms = (
                sample.device_ms - self.last_sample.device_ms
            ) & 0xFFFFFFFF
            dt = delta_ms / 1000.0
            if 0.01 <= dt <= 1.0:
                left_delta = (
                    sample.left_ticks - self.last_sample.left_ticks
                ) * self.left_encoder_sign
                right_delta = (
                    sample.right_ticks - self.last_sample.right_ticks
                ) * self.right_encoder_sign
                scale = 2.0 * math.pi / self.encoder_cpr / dt
                self.measured_left = left_delta * scale
                self.measured_right = right_delta * scale
                self.left_pwm = self.left_controller.update(
                    self.target_left,
                    self.measured_left,
                    dt,
                )
                self.right_pwm = self.right_controller.update(
                    self.target_right,
                    self.measured_right,
                    dt,
                )
                self.feedback_ok = True
                self.log_motion_sample(now, sample)

        self.last_sample = sample
        self.last_feedback_time = now
        self.publish_odometry(sample)

    def odometry_ticks(self, sample):
        if self.swap_wheel_odometry:
            return sample.right_ticks, sample.left_ticks
        return sample.left_ticks, sample.right_ticks

    def physical_wheel_speeds(self):
        if self.swap_wheel_odometry:
            return self.measured_right, self.measured_left
        return self.measured_left, self.measured_right

    def log_motion_sample(self, now, sample):
        if now - self.last_motion_log_time < 0.25:
            return
        if (
            abs(self.left_pwm) < 1
            and abs(self.right_pwm) < 1
            and abs(sample.left_pwm) < 1
            and abs(sample.right_pwm) < 1
            and abs(self.target_left) < 1e-3
            and abs(self.target_right) < 1e-3
        ):
            return
        odom_left, odom_right = self.physical_wheel_speeds()
        linear = self.wheel_radius * 0.5 * (odom_left + odom_right)
        angular = (
            self.wheel_radius
            * (odom_right - odom_left)
            / self.wheel_separation
        )
        self.get_logger().info(
            'base motion: '
            f'serial_pwm=({sample.left_pwm},{sample.right_pwm}) '
            f'cmd_pwm=({self.left_pwm},{self.right_pwm}) '
            f'hw_rad_s=({self.measured_left:.3f},{self.measured_right:.3f}) '
            f'odom_rad_s=({odom_left:.3f},{odom_right:.3f}) '
            f'v={linear:.3f}mps w={angular:.3f}rps '
            f'yaw={math.degrees(self.odometry.yaw):.1f}deg'
        )
        self.last_motion_log_time = now

    def publish_dry_run_odometry(self):
        sample = EncoderSample(
            device_ms=0,
            left_pwm=self.left_pwm,
            right_pwm=self.right_pwm,
            left_ticks=0,
            right_ticks=0,
        )
        self.publish_odometry(sample)

    def command_tick(self):
        now = time.monotonic()
        if now - self.last_command_time > self.command_timeout:
            self.target_left = 0.0
            self.target_right = 0.0
            self.left_pwm = 0
            self.right_pwm = 0
            self.left_controller.reset()
            self.right_controller.reset()

        if (
            self.e_stop_active
            or
            self.last_feedback_time is not None
            and now - self.last_feedback_time > 0.7
        ):
            self.left_pwm = 0
            self.right_pwm = 0
            self.feedback_ok = False

        left = self.left_pwm * self.left_motor_sign
        right = self.right_pwm * self.right_motor_sign
        command = f'p {left} {right}'
        if (
            command != self.last_serial_command
            or now - self.last_serial_write >= 0.2
        ):
            self.write_serial(command)
            self.last_serial_command = command
            self.last_serial_write = now

    def write_serial(self, command):
        if self.serial_port is None:
            self.open_serial()
        if self.serial_port is not None:
            try:
                self.serial_port.write((command + '\n').encode('ascii'))
                self.serial_error = ''
            except (OSError, serial.SerialException) as exc:
                self.handle_serial_error('base serial write failed', exc)

    def publish_odometry(self, sample):
        stamp = self.get_clock().now().to_msg()
        odom_frame = str(self.get_parameter('odom_frame').value)
        base_frame = str(self.get_parameter('base_frame').value)
        orientation = yaw_to_quaternion(self.odometry.yaw)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = odom_frame
        odom.child_frame_id = base_frame
        odom.pose.pose.position.x = self.odometry.x
        odom.pose.pose.position.y = self.odometry.y
        odom.pose.pose.orientation = orientation
        odom_left, odom_right = self.physical_wheel_speeds()
        odom.twist.twist.linear.x = (
            self.wheel_radius
            * 0.5
            * (odom_left + odom_right)
        )
        odom.twist.twist.angular.z = (
            self.wheel_radius
            * (odom_right - odom_left)
            / self.wheel_separation
        )
        self.odom_pub.publish(odom)

        joints = JointState()
        joints.header.stamp = stamp
        joints.name = ['left_wheel_spin', 'right_wheel_spin']
        joint_left_ticks, joint_right_ticks = self.odometry_ticks(sample)
        joint_left_sign = (
            self.right_encoder_sign
            if self.swap_wheel_odometry
            else self.left_encoder_sign
        )
        joint_right_sign = (
            self.left_encoder_sign
            if self.swap_wheel_odometry
            else self.right_encoder_sign
        )
        odom_left, odom_right = self.physical_wheel_speeds()
        joints.position = [
            joint_left_ticks
            * joint_left_sign
            * 2.0
            * math.pi
            / self.encoder_cpr,
            joint_right_ticks
            * joint_right_sign
            * 2.0
            * math.pi
            / self.encoder_cpr,
        ]
        joints.velocity = [odom_left, odom_right]
        self.joint_pub.publish(joints)

        if bool(self.get_parameter('publish_tf').value):
            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = odom_frame
            transform.child_frame_id = base_frame
            transform.transform.translation.x = self.odometry.x
            transform.transform.translation.y = self.odometry.y
            transform.transform.rotation = orientation
            self.tf_broadcaster.sendTransform(transform)

    def publish_status(self):
        status = {
            'connected': self.serial_port is not None,
            'dry_run': self.dry_run,
            'serial_port': self.serial_port_name,
            'serial_error': self.serial_error,
            'e_stop_active': self.e_stop_active,
            'feedback_ok': self.feedback_ok,
            'target_rad_s': [self.target_left, self.target_right],
            'measured_rad_s': [self.measured_left, self.measured_right],
            'odom_rad_s': list(self.physical_wheel_speeds()),
            'pwm': [self.left_pwm, self.right_pwm],
            'serial_pwm': (
                [self.last_sample.left_pwm, self.last_sample.right_pwm]
                if self.last_sample is not None
                else None
            ),
            'swap_wheel_commands': self.swap_wheel_commands,
            'swap_wheel_odometry': self.swap_wheel_odometry,
            'wheel_separation_m': self.wheel_separation,
            'calibration': self.calibration_status,
            'pose': {
                'x': self.odometry.x,
                'y': self.odometry.y,
                'yaw': self.odometry.yaw,
            },
        }
        msg = String()
        msg.data = json.dumps(status)
        self.status_pub.publish(msg)

    def stop(self):
        self.left_pwm = 0
        self.right_pwm = 0
        self.write_serial('stop')
        if self.serial_port is not None:
            try:
                self.serial_port.close()
            except (OSError, serial.SerialException) as exc:
                self.serial_error = str(exc)
            self.serial_port = None


def main(args=None):
    rclpy.init(args=args)
    node = MotorBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
