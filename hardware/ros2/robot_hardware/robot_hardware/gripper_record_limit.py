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
import re
import time
from pathlib import Path

import rclpy
import yaml

from robot_hardware.gripper_position_probe import (
    GripperProbe,
    build_summary,
)

REPO_ROOT = Path(__file__).resolve().parents[4]

DEFAULT_CONFIG = str(REPO_ROOT / 'src' / 'robot_bringup' / 'config' / 'real.yaml')
DEFAULT_FIRMWARE = str(REPO_ROOT / 'firmware' / 'openrb_gripper' / 'openrb_gripper.ino')


def formatted_deg(value):
    return f'{float(value):.2f}'


def limit_value(summary):
    present_deg = summary.get('present_deg')
    if not present_deg:
        raise ValueError('present_deg is unavailable')
    return round(float(present_deg['mean']), 2)


def update_config_data(config, label, value):
    data = dict(config)
    grasp = dict(data.get('gripper_bridge_node') or {})
    grasp_params = dict(grasp.get('ros__parameters') or {})
    mission = dict(data.get('match_state_machine_node') or {})
    mission_params = dict(mission.get('ros__parameters') or {})

    if label == 'open':
        grasp_params['open_deg'] = float(value)
    elif label == 'closed':
        grasp_params['closed_deg'] = float(value)
        mission_params['gripper_closed_deg'] = float(value)
    else:
        raise ValueError(f'unknown label: {label}')

    grasp['ros__parameters'] = grasp_params
    mission['ros__parameters'] = mission_params
    data['gripper_bridge_node'] = grasp
    data['match_state_machine_node'] = mission
    return data


def replace_constant(text, name, value):
    pattern = rf'(const float {re.escape(name)} = )[-+]?\d+(?:\.\d+)?(;)'
    updated, count = re.subn(
        pattern,
        rf'\g<1>{formatted_deg(value)}\2',
        text,
        count=1,
    )
    if count != 1:
        raise ValueError(f'could not update firmware constant {name}')
    return updated


def update_firmware_text(text, label, value):
    if label == 'open':
        text = replace_constant(text, 'FULL_OPEN_DEG', value)
    elif label == 'closed':
        text = replace_constant(text, 'CLOSED_DEG', value)
    else:
        raise ValueError(f'unknown label: {label}')

    closed = extract_constant(text, 'CLOSED_DEG')
    opened = extract_constant(text, 'FULL_OPEN_DEG')
    safe_range = f'SAFE_RANGE_DEG {formatted_deg(closed)} TO {formatted_deg(opened)}'
    text, count = re.subn(
        r'SAFE_RANGE_DEG [-+]?\d+(?:\.\d+)? TO [-+]?\d+(?:\.\d+)?',
        safe_range,
        text,
        count=1,
    )
    if count != 1:
        raise ValueError('could not update SAFE_RANGE_DEG print line')
    return text


def extract_constant(text, name):
    match = re.search(
        rf'const float {re.escape(name)} = ([-+]?\d+(?:\.\d+)?);',
        text,
    )
    if not match:
        raise ValueError(f'could not find firmware constant {name}')
    return float(match.group(1))


def collect_summary(topic, samples, timeout_sec):
    rclpy.init(args=None)
    node = GripperProbe(topic)
    deadline = time.monotonic() + timeout_sec
    try:
        while rclpy.ok() and len(node.samples) < samples:
            if time.monotonic() >= deadline:
                break
            rclpy.spin_once(node, timeout_sec=0.05)
        return build_summary(node.samples)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('label', choices=('open', 'closed'))
    parser.add_argument('--topic', default='/gripper/state')
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--timeout-sec', type=float, default=6.0)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--firmware', default=DEFAULT_FIRMWARE)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--json', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = collect_summary(args.topic, args.samples, args.timeout_sec)
    value = limit_value(summary)
    result = {
        'label': args.label,
        'value_deg': value,
        'summary': summary,
        'config': args.config,
        'firmware': args.firmware,
        'applied': bool(args.apply),
    }

    if args.apply:
        config_path = Path(args.config).expanduser()
        firmware_path = Path(args.firmware).expanduser()

        config_data = yaml.safe_load(config_path.read_text()) or {}
        updated_config = update_config_data(config_data, args.label, value)
        config_path.write_text(
            yaml.safe_dump(updated_config, sort_keys=False)
        )

        firmware_text = firmware_path.read_text()
        firmware_path.write_text(
            update_firmware_text(firmware_text, args.label, value)
        )

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"{args.label}_deg={formatted_deg(value)}")
        print(
            'present_raw_mean='
            + str((summary.get('present_raw') or {}).get('mean'))
        )
        print(
            'current_raw_latest='
            + str((summary.get('current_raw') or {}).get('latest'))
        )
        print(
            'torque_enabled='
            + str((summary.get('latest') or {}).get('torque_enabled'))
        )
        if args.apply:
            print('updated config: ' + args.config)
            print('updated firmware: ' + args.firmware)
            print('rebuild ROS and upload OpenRB firmware before real use')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
