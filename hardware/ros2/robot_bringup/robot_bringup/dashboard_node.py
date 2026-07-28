#!/usr/bin/env python3

import json
import math
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import rclpy

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Bool, String


class DashboardNode(Node):
    def __init__(self):
        super().__init__('dashboard_node')
        self.declare_parameter('bind_address', '0.0.0.0')
        self.declare_parameter('port', 8080)
        self.declare_parameter('cmd_vel_topic', '/cmd_vel_teleop')

        self.command_pub = self.create_publisher(
            String,
            '/task/command',
            10,
        )
        self.manual_enable_pub = self.create_publisher(
            Bool,
            '/teleop/enable',
            10,
        )
        self.manual_cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel_teleop',
            10,
        )
        self.direct_cmd_pub = self.create_publisher(
            Twist,
            str(self.get_parameter('cmd_vel_topic').value),
            10,
        )
        self.gripper_pub = self.create_publisher(
            String,
            '/gripper/command',
            10,
        )
        self.create_subscription(
            String,
            '/task/current_state',
            lambda msg: self.update_status('mission', msg.data),
            10,
        )
        self.create_subscription(
            String,
            '/motor/state',
            lambda msg: self.update_status('base', msg.data),
            10,
        )
        self.create_subscription(
            String,
            '/gripper/state',
            lambda msg: self.update_status('gripper', msg.data),
            10,
        )
        self.create_subscription(
            String,
            '/semantic/objects',
            lambda msg: self.update_status('detections', msg.data),
            10,
        )
        self.create_subscription(
            String,
            '/depth_yolo_target/status',
            lambda msg: self.update_status('competition', msg.data),
            10,
        )
        self.create_subscription(
            String,
            '/pick_and_place/status',
            lambda msg: self.update_status('pick', msg.data),
            10,
        )
        # 경량 arena control 스택 (현행 실기 경로)
        self.create_subscription(
            String,
            '/arena_lightweight/status',
            lambda msg: self.update_status('arena', msg.data),
            10,
        )
        self.create_subscription(
            String,
            '/match/state',
            lambda msg: self.update_status('match', msg.data),
            10,
        )
        # YOLO 통합 월드맵 (탐색 후 알려진 객체 위치, world 좌표)
        self.create_subscription(
            String,
            '/detected_objects/map_objects',
            lambda msg: self.update_status('map_objects', msg.data),
            10,
        )

        self.status = {
            'mission': {},
            'base': {},
            'gripper': {},
            'detections': {},
            'competition': {},
            'pick': {},
            'arena': {},
            'match': {},
            'map_objects': {},
        }
        self.received_at = {}
        self.status_lock = threading.Lock()
        self.request_queue = queue.SimpleQueue()
        self.queue_timer = self.create_timer(0.02, self.process_requests)

        share = Path(get_package_share_directory('robot_bringup'))
        self.web_dir = share / 'web'
        self.index_html = (share / 'web' / 'index.html').read_bytes()
        handler = self.make_handler()
        address = str(self.get_parameter('bind_address').value)
        port = int(self.get_parameter('port').value)
        self.server = ThreadingHTTPServer((address, port), handler)
        self.server.daemon_threads = True
        self.server_thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
        )
        self.server_thread.start()
        bound_address, bound_port = self.server.server_address[:2]
        self.get_logger().info(
            f'dashboard listening on http://{bound_address}:{bound_port}'
        )

    def update_status(self, key, raw):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = {'raw': raw}
        with self.status_lock:
            self.status[key] = value
            self.received_at[key] = time.monotonic()

    def status_snapshot(self):
        with self.status_lock:
            snapshot = json.loads(json.dumps(self.status))
            now = time.monotonic()
            snapshot['_age_sec'] = {
                key: now - stamp
                for key, stamp in self.received_at.items()
            }
            return snapshot

    def validate_request(self, path, payload):
        if path not in (
            '/api/mission',
            '/api/config',
            '/api/manual',
            '/api/cmd_vel',
            '/api/gripper',
            '/api/estop',
        ):
            return 404, None, 'not found'
        if not isinstance(payload, dict):
            return 400, None, 'JSON payload must be an object'

        if path == '/api/manual':
            enabled = payload.get('enabled', False)
            if not isinstance(enabled, bool):
                return 400, None, 'enabled must be a boolean'
            return 202, {'enabled': enabled}, ''

        if path == '/api/cmd_vel':
            try:
                linear = float(payload.get('linear', 0.0))
                angular = float(payload.get('angular', 0.0))
            except (TypeError, ValueError):
                return 400, None, 'linear and angular must be numbers'
            if not math.isfinite(linear) or not math.isfinite(angular):
                return 400, None, 'linear and angular must be finite'
            return 202, {'linear': linear, 'angular': angular}, ''

        if path == '/api/config':
            config = {'command': 'config'}
            for key in (
                'open_loop_drive_scale',
                'open_loop_turn_scale',
            ):
                if key not in payload:
                    continue
                try:
                    value = float(payload[key])
                except (TypeError, ValueError):
                    return 400, None, f'{key} must be a number'
                if not math.isfinite(value) or value <= 0.0:
                    return 400, None, f'{key} must be a positive finite number'
                config[key] = value
            if len(config) == 1:
                return 400, None, 'config payload is empty'
            return 202, config, ''

        if path == '/api/gripper':
            command = str(payload.get('command', '')).strip().upper()
            if not command:
                return 400, None, 'gripper command is empty'
            return 202, {'command': command}, ''

        return 202, payload, ''

    def make_handler(self):
        node = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format_text, *args):
                node.get_logger().debug(format_text % args)

            def send_bytes(self, status, content_type, payload):
                self.send_response(status)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(payload)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                if self.path == '/':
                    self.send_bytes(
                        200,
                        'text/html; charset=utf-8',
                        node.index_html,
                    )
                    return
                if self.path in ('/hud', '/hud.html'):
                    self.send_bytes(
                        200,
                        'text/html; charset=utf-8',
                        (node.web_dir / 'hud.html').read_bytes(),
                    )
                    return
                if self.path == '/api/status':
                    payload = json.dumps(
                        node.status_snapshot()
                    ).encode('utf-8')
                    self.send_bytes(
                        200,
                        'application/json',
                        payload,
                    )
                    return
                self.send_bytes(404, 'text/plain', b'not found')

            def do_POST(self):
                length = int(self.headers.get('Content-Length', '0'))
                raw = self.rfile.read(length)
                try:
                    payload = json.loads(raw or b'{}')
                except json.JSONDecodeError:
                    self.send_bytes(400, 'text/plain', b'invalid JSON')
                    return
                status, payload, error = node.validate_request(
                    self.path,
                    payload,
                )
                if status != 202:
                    self.send_bytes(
                        status,
                        'text/plain',
                        error.encode('utf-8'),
                    )
                    return
                node.request_queue.put((self.path, payload))
                self.send_bytes(
                    202,
                    'application/json',
                    b'{"accepted":true}',
                )

        return Handler

    def process_requests(self):
        while True:
            try:
                path, payload = self.request_queue.get_nowait()
            except queue.Empty:
                return

            if path == '/api/mission':
                msg = String()
                msg.data = json.dumps(payload)
                self.command_pub.publish(msg)
            elif path == '/api/config':
                msg = String()
                msg.data = json.dumps(payload)
                self.command_pub.publish(msg)
            elif path == '/api/manual':
                msg = Bool()
                msg.data = bool(payload.get('enabled', False))
                self.manual_enable_pub.publish(msg)
            elif path == '/api/cmd_vel':
                msg = Twist()
                msg.linear.x = float(payload.get('linear', 0.0))
                msg.angular.z = float(payload.get('angular', 0.0))
                self.manual_cmd_pub.publish(msg)
            elif path == '/api/gripper':
                msg = String()
                msg.data = payload['command']
                self.gripper_pub.publish(msg)
            elif path == '/api/estop':
                self.publish_estop()

    def publish_estop(self):
        zero = Twist()
        for _ in range(3):
            self.direct_cmd_pub.publish(zero)
            self.manual_cmd_pub.publish(zero)

        manual = Bool()
        manual.data = False
        self.manual_enable_pub.publish(manual)

        mission = String()
        mission.data = json.dumps({'command': 'cancel'})
        self.command_pub.publish(mission)

        gripper = String()
        gripper.data = 'STOP'
        self.gripper_pub.publish(gripper)

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(timeout=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = DashboardNode()
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
