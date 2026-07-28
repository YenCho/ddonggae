#!/usr/bin/env python3
"""ROS 2 bridge for the mecanum_encoder_control Arduino firmware.

Velocity path: /cmd_vel Twist (vx, vy, wz) -> serial `m` command; the
firmware runs per-wheel velocity PID. Position path: /base/move_relative
JSON {"dx":, "dy":, "dyaw":} -> serial `d` command; the firmware runs a
synchronized trapezoidal position move and this node reports the result on
/base/move_result. Odometry comes from the four encoder counters in the
firmware STATE stream.
"""

import json
import glob
import math
import os
import termios
import time

import rclpy
import serial

from geometry_msgs.msg import Quaternion, TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from tf2_ros import TransformBroadcaster

from robot_hardware.mecanum_control import (
    WHEEL_NAMES,
    MecanumKinematics,
    MecanumOdometry,
    clamp_twist_components,
    parse_mecanum_state_line,
    parse_move_done_line,
)


def yaw_to_quaternion(yaw):
    half = yaw * 0.5
    return Quaternion(z=math.sin(half), w=math.cos(half))


class MecanumBridgeNode(Node):
    def __init__(self):
        super().__init__('mecanum_bridge_node')

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
        self.declare_parameter('serial_boot_delay_sec', 2.0)
        self.declare_parameter('serial_write_timeout_sec', 0.1)
        self.declare_parameter('dry_run', False)
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('move_relative_topic', '/base/move_relative')
        self.declare_parameter('move_result_topic', '/base/move_result')
        self.declare_parameter('e_stop_topic', '/safety/e_stop')
        self.declare_parameter('odom_topic', '/odom/wheel')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('publish_tf', False)

        # Geometry. Re-measure at wheel contact centers after assembly and
        # keep these equal to the values pushed into the firmware.
        self.declare_parameter('wheel_radius_m', 0.034)
        self.declare_parameter('half_length_m', 0.150)
        self.declare_parameter('half_width_m', 0.125)
        self.declare_parameter('encoder_cpr', 2464)

        # Bring-up polarity fixes, pushed to the firmware `sign` command.
        self.declare_parameter('motor_signs', [1, 1, 1, 1])
        self.declare_parameter('encoder_signs', [1, 1, 1, 1])

        self.declare_parameter('max_linear_x_mps', 0.30)
        self.declare_parameter('max_linear_y_mps', 0.30)
        self.declare_parameter('max_angular_rps', 1.20)
        self.declare_parameter('max_wheel_rad_s', 12.0)
        self.declare_parameter('command_timeout_sec', 0.5)

        # Velocity PID (firmware-side), pushed via the `pid` command.
        self.declare_parameter('speed_kp', 12.0)
        self.declare_parameter('speed_ki', 7.0)
        self.declare_parameter('speed_kd', 0.0)
        self.declare_parameter('min_pwm', 45.0)
        self.declare_parameter('max_pwm', 220.0)
        self.declare_parameter('feedforward_slope', 14.0)
        self.declare_parameter('integral_limit', 8.0)

        # Position loop (firmware-side), pushed via the `ppid` command.
        self.declare_parameter('position_kp', 6.0)
        self.declare_parameter('position_kd', 0.0)
        self.declare_parameter('position_max_rad_s', 6.0)
        self.declare_parameter('position_accel_rad_s2', 12.0)
        self.declare_parameter('position_tolerance_rad', 0.035)
        self.declare_parameter('position_hold_ms', 150)
        self.declare_parameter('position_min_rad_s', 0.8)

        self.kinematics = MecanumKinematics(
            float(self.get_parameter('wheel_radius_m').value),
            float(self.get_parameter('half_length_m').value),
            float(self.get_parameter('half_width_m').value),
        )
        self.encoder_cpr = int(self.get_parameter('encoder_cpr').value)
        self.odometry = MecanumOdometry(self.kinematics, self.encoder_cpr)

        self.max_vx = float(self.get_parameter('max_linear_x_mps').value)
        self.max_vy = float(self.get_parameter('max_linear_y_mps').value)
        self.max_wz = float(self.get_parameter('max_angular_rps').value)
        self.max_wheel_rad_s = float(
            self.get_parameter('max_wheel_rad_s').value
        )
        self.command_timeout = float(
            self.get_parameter('command_timeout_sec').value
        )
        self.dry_run = bool(self.get_parameter('dry_run').value)

        self.target_twist = (0.0, 0.0, 0.0)
        self.measured_twist = (0.0, 0.0, 0.0)
        self.measured_wheel_rad_s = [0.0, 0.0, 0.0, 0.0]
        self.last_state_sample = None
        self.firmware_mode = 0
        self.move_active = False
        self.move_started_time = None
        self.last_command_time = time.monotonic()
        self.last_feedback_time = None
        self.last_serial_command = None
        self.last_serial_write = 0.0
        self.feedback_ok = False
        self.e_stop_active = False
        self.serial_port_name = str(self.get_parameter('serial_port').value)
        self.serial_error = ''
        self.baudrate = int(self.get_parameter('baudrate').value)
        self.last_serial_open_attempt_time = 0.0

        self.serial_port = None
        self.serial_rx_buffer = b''
        if not self.dry_run:
            self.open_serial(force=True)
        else:
            self.get_logger().warn('mecanum bridge is in dry_run mode')

        self.cmd_sub = self.create_subscription(
            Twist,
            str(self.get_parameter('cmd_vel_topic').value),
            self.cmd_vel_callback,
            10,
        )
        self.move_sub = self.create_subscription(
            String,
            str(self.get_parameter('move_relative_topic').value),
            self.move_relative_callback,
            10,
        )
        self.e_stop_sub = self.create_subscription(
            Bool,
            str(self.get_parameter('e_stop_topic').value),
            self.e_stop_callback,
            10,
        )
        self.odom_pub = self.create_publisher(
            Odometry,
            str(self.get_parameter('odom_topic').value),
            20,
        )
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 20)
        self.status_pub = self.create_publisher(String, '/motor/state', 10)
        self.move_result_pub = self.create_publisher(
            String,
            str(self.get_parameter('move_result_topic').value),
            10,
        )
        self.tf_broadcaster = TransformBroadcaster(self)
        self.read_timer = self.create_timer(0.02, self.read_serial)
        self.command_timer = self.create_timer(0.05, self.command_tick)
        self.status_timer = self.create_timer(0.5, self.publish_status)

    # ------------------------------------------------------------------
    # Serial management
    # ------------------------------------------------------------------

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
            self.get_logger().warn(
                self.serial_error,
                throttle_duration_sec=2.0,
            )
            return

        port_name = present[0]
        try:
            # exclusive=True: flock the port so a stray diff-drive
            # motor_bridge_node (or a second copy of this node from an
            # orphaned launch) can't open the same UNO concurrently and
            # interleave commands into the firmware parser (2026-07-18
            # field incident — corrupted the firmware until USB replug).
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
            self.serial_rx_buffer = b''
            self.push_firmware_config()
            self.serial_error = ''
            self.get_logger().info(
                f'mecanum serial connected: {self.serial_port_name}'
            )
        except (OSError, serial.SerialException, termios.error) as exc:
            # termios.error는 OSError 하위가 아님 — 재연결 레이스에서
            # reset_input_buffer()가 이걸 던지면 노드째 죽는다 (2026-07-17 실기).
            self.serial_error = str(exc)
            self.close_serial()
            self.get_logger().error(
                'failed to open mecanum serial port '
                + port_name
                + ': '
                + self.serial_error
            )

    def push_firmware_config(self):
        """Send geometry, PID tuning, and signs so the firmware matches ROS
        parameters after every (re)connect or Arduino reset."""
        motor_signs = [
            int(value) for value in self.get_parameter('motor_signs').value
        ]
        encoder_signs = [
            int(value) for value in self.get_parameter('encoder_signs').value
        ]
        commands = [
            'stop',
            'geom {:.5f} {:.5f} {:.5f}'.format(
                self.kinematics.wheel_radius_m,
                self.kinematics.half_length_m,
                self.kinematics.half_width_m,
            ),
            'pid {} {} {} {} {} {} {}'.format(
                float(self.get_parameter('speed_kp').value),
                float(self.get_parameter('speed_ki').value),
                float(self.get_parameter('speed_kd').value),
                float(self.get_parameter('min_pwm').value),
                float(self.get_parameter('max_pwm').value),
                float(self.get_parameter('feedforward_slope').value),
                float(self.get_parameter('integral_limit').value),
            ),
            self.ppid_command(
                float(self.get_parameter('position_max_rad_s').value)
            ),
            'sign {} {} {} {} {} {} {} {}'.format(
                *(motor_signs + encoder_signs)
            ),
            'stream 1',
        ]
        # UNO의 RX 버퍼는 64바이트뿐이고, READY 직후엔 ~500자 help 출력이
        # TX를 점유해 loop()가 수십 ms 동안 RX를 못 비운다. 그 사이에
        # 커맨드를 몰아 보내면 오버플로로 라인이 잘려 ERR usage가 난다
        # (2026-07-15 실측: pid 라인 유실 + stream 1 미적용 → odom 미발행).
        time.sleep(0.3)
        for command in commands:
            self.serial_port.write((command + '\n').encode('ascii'))
            time.sleep(0.06)
        self.active_position_max_rad_s = float(
            self.get_parameter('position_max_rad_s').value
        )

    def ppid_command(self, max_rad_s):
        return 'ppid {} {} {} {} {} {} {}'.format(
            float(self.get_parameter('position_kp').value),
            float(self.get_parameter('position_kd').value),
            float(max_rad_s),
            float(self.get_parameter('position_accel_rad_s2').value),
            float(self.get_parameter('position_tolerance_rad').value),
            int(self.get_parameter('position_hold_ms').value),
            float(self.get_parameter('position_min_rad_s').value),
        )

    def close_serial(self):
        if self.serial_port is None:
            return
        try:
            self.serial_port.write(b'stop\n')
        except (OSError, serial.SerialException, termios.error):
            pass
        try:
            self.serial_port.close()
        except (OSError, serial.SerialException, termios.error):
            pass
        self.serial_port = None
        self.serial_rx_buffer = b''

    def handle_serial_error(self, context, exc):
        self.serial_error = str(exc)
        self.feedback_ok = False
        self.target_twist = (0.0, 0.0, 0.0)
        self.last_state_sample = None
        if self.move_active:
            self.finish_move(False, 'serial error: ' + self.serial_error)
        self.get_logger().error(context + ': ' + self.serial_error)
        self.close_serial()

    def write_serial(self, command):
        if self.serial_port is None:
            self.open_serial()
        if self.serial_port is not None:
            try:
                self.serial_port.write((command + '\n').encode('ascii'))
                self.serial_error = ''
            except (OSError, serial.SerialException, termios.error) as exc:
                self.handle_serial_error('mecanum serial write failed', exc)

    # ------------------------------------------------------------------
    # Command handling
    # ------------------------------------------------------------------

    def cmd_vel_callback(self, msg):
        if self.e_stop_active:
            self.target_twist = (0.0, 0.0, 0.0)
            self.last_command_time = time.monotonic()
            return

        vx, vy, wz = clamp_twist_components(
            msg.linear.x,
            msg.linear.y,
            msg.angular.z,
            self.max_vx,
            self.max_vy,
            self.max_wz,
        )
        vx, vy, wz = self.kinematics.clamp_body_twist(
            vx,
            vy,
            wz,
            self.max_wheel_rad_s,
        )
        nonzero = any(abs(value) > 1e-4 for value in (vx, vy, wz))
        if self.move_active and nonzero:
            # A live velocity command overrides an in-flight position move.
            self.finish_move(False, 'preempted by cmd_vel')
        self.target_twist = (vx, vy, wz)
        self.last_command_time = time.monotonic()

    def move_relative_callback(self, msg):
        if self.e_stop_active:
            self.publish_move_result(False, 'e_stop active')
            return
        try:
            payload = json.loads(msg.data)
            dx = float(payload.get('dx', 0.0))
            dy = float(payload.get('dy', 0.0))
            dyaw = float(payload.get('dyaw', 0.0))
            max_v = payload.get('max_v')
        except (ValueError, TypeError, AttributeError) as exc:
            self.publish_move_result(False, f'bad request: {exc}')
            return
        if self.move_active:
            self.publish_move_result(False, 'move already active')
            return

        # Optional per-move speed cap (same contract as the sim bridge):
        # max_v [m/s] maps to a wheel-speed cap for THIS move's trapezoid
        # profile; without it the default position_max_rad_s applies.
        desired_max_rad_s = float(self.get_parameter('position_max_rad_s').value)
        if max_v is not None:
            try:
                desired_max_rad_s = min(
                    float(self.get_parameter('max_wheel_rad_s').value),
                    max(0.5, float(max_v) / self.kinematics.wheel_radius_m),
                )
            except (ValueError, TypeError):
                pass
        if abs(desired_max_rad_s
               - getattr(self, 'active_position_max_rad_s',
                         desired_max_rad_s + 1.0)) > 1e-6:
            self.write_serial(self.ppid_command(desired_max_rad_s))
            self.active_position_max_rad_s = desired_max_rad_s

        self.target_twist = (0.0, 0.0, 0.0)
        self.move_active = True
        self.move_started_time = time.monotonic()
        self.write_serial(f'd {dx:.5f} {dy:.5f} {dyaw:.5f}')
        self.get_logger().info(
            f'position move: dx={dx:.3f} dy={dy:.3f} dyaw={dyaw:.3f}'
        )

    def e_stop_callback(self, msg):
        self.e_stop_active = bool(msg.data)
        if self.e_stop_active:
            self.target_twist = (0.0, 0.0, 0.0)
            if self.move_active:
                self.finish_move(False, 'e_stop')
            self.write_serial('stop')

    def command_tick(self):
        now = time.monotonic()
        if now - self.last_command_time > self.command_timeout:
            self.target_twist = (0.0, 0.0, 0.0)

        if self.e_stop_active:
            return
        if self.move_active:
            # The firmware owns motion during a position move; do not stream
            # zero velocity on top of it.
            return

        vx, vy, wz = self.target_twist
        command = f'm {vx:.4f} {vy:.4f} {wz:.4f}'
        if (
            command != self.last_serial_command
            or now - self.last_serial_write >= 0.2
        ):
            self.write_serial(command)
            self.last_serial_command = command
            self.last_serial_write = now

    # ------------------------------------------------------------------
    # Feedback handling
    # ------------------------------------------------------------------

    def read_serial(self):
        if self.serial_port is None:
            self.open_serial()
            return
        try:
            # timeout=0 (논블로킹)에서 readline()은 '\n'이 아직 안 온 라인을
            # 조각으로 반환한다. 조각난 STATE는 티가 안 나지만 단발성
            # DONE,move / ERR 응답이 조각나면 move 결과가 영영 유실된다
            # (2026-07-15 실측: move_active 잔류로 후속 move 전부 거부).
            # 바이트 버퍼에 모아 완성된 라인만 처리한다.
            waiting = self.serial_port.in_waiting
            if waiting:
                self.serial_rx_buffer += self.serial_port.read(waiting)
                while b'\n' in self.serial_rx_buffer:
                    raw, self.serial_rx_buffer = (
                        self.serial_rx_buffer.split(b'\n', 1)
                    )
                    line = raw.decode('utf-8', errors='replace').strip()
                    if line:
                        self.handle_serial_line(line)
        except (OSError, serial.SerialException, termios.error) as exc:
            self.handle_serial_error('mecanum serial read failed', exc)

    def handle_serial_line(self, line):
        sample = parse_mecanum_state_line(line)
        if sample is not None:
            self.handle_state_sample(sample)
            return
        done = parse_move_done_line(line)
        if done is not None:
            self.finish_move(True, 'done', errors_rad=done)
            return
        if line.startswith('ERR move timeout'):
            self.finish_move(False, 'firmware move timeout')
            return
        if line.startswith('ERR') or line.startswith('WARN'):
            self.get_logger().warn('firmware: ' + line)
            return
        if line.startswith('READY'):
            # The Arduino rebooted (for example after a USB reconnect); its
            # RAM tuning is back at defaults, so push the config again.
            if self.move_active:
                # An in-flight move can never complete after a reboot; without
                # this, move_active sticks and every later move is rejected.
                self.finish_move(False, 'firmware rebooted mid-move')
            self.get_logger().info('firmware ready, pushing config')
            try:
                self.push_firmware_config()
            except (OSError, serial.SerialException, termios.error) as exc:
                self.handle_serial_error('config push failed', exc)

    def handle_state_sample(self, sample):
        now = time.monotonic()
        self.firmware_mode = sample.mode
        if self.last_state_sample is not None:
            delta_ms = (
                sample.device_ms - self.last_state_sample.device_ms
            ) & 0xFFFFFFFF
            dt = delta_ms / 1000.0
            if 0.01 <= dt <= 1.0:
                rad_per_tick = 2.0 * math.pi / self.encoder_cpr
                self.measured_wheel_rad_s = [
                    (now_ticks - before_ticks) * rad_per_tick / dt
                    for now_ticks, before_ticks in zip(
                        sample.ticks,
                        self.last_state_sample.ticks,
                    )
                ]
                self.measured_twist = self.kinematics.wheels_to_body(
                    *self.measured_wheel_rad_s
                )
                self.feedback_ok = True

        self.odometry.update(sample.ticks)
        self.last_state_sample = sample
        self.last_feedback_time = now
        self.publish_odometry(sample)

    def finish_move(self, ok, reason, errors_rad=None):
        if not self.move_active and reason == 'done':
            # A DONE can arrive for a firmware-local move (direct serial use).
            return
        elapsed = (
            time.monotonic() - self.move_started_time
            if self.move_started_time is not None
            else None
        )
        self.move_active = False
        self.move_started_time = None
        self.publish_move_result(ok, reason, errors_rad, elapsed)

    def publish_move_result(self, ok, reason, errors_rad=None, elapsed=None):
        result = {
            'ok': bool(ok),
            'reason': reason,
            'wheel_errors_rad': (
                list(errors_rad) if errors_rad is not None else None
            ),
            'elapsed_sec': elapsed,
        }
        msg = String()
        msg.data = json.dumps(result)
        self.move_result_pub.publish(msg)
        if not ok:
            self.get_logger().warn('move failed: ' + reason)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish_odometry(self, sample):
        stamp = self.get_clock().now().to_msg()
        odom_frame = str(self.get_parameter('odom_frame').value)
        base_frame = str(self.get_parameter('base_frame').value)
        orientation = yaw_to_quaternion(self.odometry.yaw)
        vx, vy, wz = self.measured_twist

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = odom_frame
        odom.child_frame_id = base_frame
        odom.pose.pose.position.x = self.odometry.x
        odom.pose.pose.position.y = self.odometry.y
        odom.pose.pose.orientation = orientation
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.angular.z = wz
        self.odom_pub.publish(odom)

        rad_per_tick = 2.0 * math.pi / self.encoder_cpr
        joints = JointState()
        joints.header.stamp = stamp
        joints.name = [name + '_wheel_spin' for name in WHEEL_NAMES]
        joints.position = [ticks * rad_per_tick for ticks in sample.ticks]
        joints.velocity = list(self.measured_wheel_rad_s)
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
        now = time.monotonic()
        if (
            self.last_feedback_time is not None
            and now - self.last_feedback_time > 0.7
        ):
            self.feedback_ok = False

        target_wheels = self.kinematics.body_to_wheels(*self.target_twist)
        status = {
            'drive_type': 'mecanum',
            'connected': self.serial_port is not None,
            'dry_run': self.dry_run,
            'serial_port': self.serial_port_name,
            'serial_error': self.serial_error,
            'e_stop_active': self.e_stop_active,
            'feedback_ok': self.feedback_ok,
            'firmware_mode': self.firmware_mode,
            'position_move_active': self.move_active,
            'commanded_velocity': {
                'linear_x': self.target_twist[0],
                'linear_y': self.target_twist[1],
                'angular_z': self.target_twist[2],
            },
            'measured_velocity': {
                'linear_x': self.measured_twist[0],
                'linear_y': self.measured_twist[1],
                'angular_z': self.measured_twist[2],
            },
            'target_rad_s': list(target_wheels),
            'measured_rad_s': list(self.measured_wheel_rad_s),
            'odom_rad_s': list(self.measured_wheel_rad_s),
            'pwm': (
                list(self.last_state_sample.pwm)
                if self.last_state_sample is not None
                else [0, 0, 0, 0]
            ),
            'wheel_names': list(WHEEL_NAMES),
            'geometry': {
                'wheel_radius_m': self.kinematics.wheel_radius_m,
                'half_length_m': self.kinematics.half_length_m,
                'half_width_m': self.kinematics.half_width_m,
            },
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
        self.write_serial('stop')
        if self.serial_port is not None:
            try:
                self.serial_port.close()
            except (OSError, serial.SerialException, termios.error) as exc:
                self.serial_error = str(exc)
            self.serial_port = None


def main(args=None):
    rclpy.init(args=args)
    node = MecanumBridgeNode()
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
