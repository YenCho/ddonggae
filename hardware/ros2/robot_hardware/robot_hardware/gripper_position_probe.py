#!/usr/bin/env python3
# Copyright 2026 mero14
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import statistics
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


POSITION_KEYS = ('present_deg', 'present_raw')


def numeric_values(samples, key):
    values = []
    for sample in samples:
        value = sample.get(key)
        if isinstance(value, bool):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            pass
    return values


def summarize_values(values):
    if not values:
        return None
    return {
        'count': len(values),
        'latest': values[-1],
        'mean': statistics.fmean(values),
        'min': min(values),
        'max': max(values),
    }


def build_summary(samples):
    summary = {
        'samples': len(samples),
        'present_deg': summarize_values(numeric_values(samples, 'present_deg')),
        'present_raw': summarize_values(numeric_values(samples, 'present_raw')),
        'current_raw': summarize_values(numeric_values(samples, 'current_raw')),
        'latest': samples[-1] if samples else None,
    }
    return summary


def format_number(value):
    if value is None:
        return '-'
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f'{value:.2f}'


def format_summary(summary):
    lines = [f"samples: {summary['samples']}"]
    for key in ('present_deg', 'present_raw', 'current_raw'):
        item = summary.get(key)
        if item is None:
            lines.append(f'{key}: -')
            continue
        lines.append(
            f"{key}: latest={format_number(item['latest'])}, "
            f"mean={format_number(item['mean'])}, "
            f"min={format_number(item['min'])}, "
            f"max={format_number(item['max'])}"
        )
    latest = summary.get('latest') or {}
    lines.append(f"torque_enabled: {latest.get('torque_enabled')}")
    lines.append(f"dynamixel_ready: {latest.get('dynamixel_ready')}")
    lines.append(f"last_serial_line: {latest.get('last_serial_line', '')}")
    return '\n'.join(lines)


class GripperProbe(Node):
    def __init__(self, topic):
        super().__init__('gripper_position_probe')
        self.samples = []
        self.create_subscription(String, topic, self.status_callback, 10)

    def status_callback(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if not isinstance(data, dict):
            return
        self.samples.append(data)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', default='/gripper/state')
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--timeout-sec', type=float, default=4.0)
    parser.add_argument('--json', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.samples <= 0:
        raise SystemExit('--samples must be greater than 0')
    if args.timeout_sec <= 0.0:
        raise SystemExit('--timeout-sec must be greater than 0')

    rclpy.init(args=None)
    node = GripperProbe(args.topic)
    deadline = time.monotonic() + args.timeout_sec
    try:
        while rclpy.ok() and len(node.samples) < args.samples:
            if time.monotonic() >= deadline:
                break
            rclpy.spin_once(node, timeout_sec=0.05)
        summary = build_summary(node.samples)
        if args.json:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print(format_summary(summary))
        if not any(summary.get(key) for key in POSITION_KEYS):
            return 1
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
