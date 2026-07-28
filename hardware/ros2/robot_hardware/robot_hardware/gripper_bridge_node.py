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

import json
import glob
import os
import time

import serial

# [2026-07-24] pyserial 의 reset_input_buffer()/flush 는 USB 재열거·전원 글리치
# 중 termios.error(5, 'Input/output error') 를 던진다. termios.error 는 OSError 의
# 서브클래스가 아니라서 except (OSError, serial.SerialException) 로는 안 잡히고,
# 그대로 전파되면 노드가 exit 1 로 죽는다 (7/24 실기: gripper_bridge_node 사망 →
# /gripper/state 침묵 → 러너 프리플라이트 필수결손 → 경기 직전 그리퍼 전멸).
# 시리얼 I/O 예외 집합에 termios.error 를 포함해 항상 재연결로 흡수한다.
# 근거/대책: docs/06-troubleshooting.md
try:
    import termios
    _SERIAL_EXTRA_ERRORS = (termios.error,)
except ImportError:  # non-posix (테스트/윈도우) — termios 없음
    _SERIAL_EXTRA_ERRORS = ()
SERIAL_ERRORS = (OSError, serial.SerialException) + _SERIAL_EXTRA_ERRORS

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class GripperBridgeNode(Node):
    def __init__(self):
        super().__init__("gripper_bridge_node")

        self.declare_parameter("serial_port", "/dev/ttyACM0")
        self.declare_parameter(
            "serial_port_candidates",
            "/dev/serial/by-id/usb-ROBOTIS_OpenRB-150*-if00,"
            "/dev/serial/by-id/usb-Arduino_LLC_OpenRB-if00,"
            "/dev/serial/by-path/platform-3610000.usb-usb-0:2.3.3:1.0,"
            "/dev/ttyACM1",
        )
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("dry_run", False)
        self.declare_parameter("control_mode", "command")
        self.declare_parameter("command_topic", "/gripper/command")
        self.declare_parameter("status_topic", "/gripper/state")
        self.declare_parameter("status_poll_interval_sec", 0.5)
        self.declare_parameter("serial_reconnect_interval_sec", 1.0)
        self.declare_parameter(
            "objects_topic",
            "/semantic/objects",
        )
        self.declare_parameter("open_on_shutdown", True)

        self.declare_parameter("target_class", "")
        self.declare_parameter("min_confidence", 0.60)

        self.declare_parameter("min_forward_m", 0.05)
        self.declare_parameter("max_forward_m", 0.35)
        self.declare_parameter("max_abs_left_m", 0.06)
        self.declare_parameter("min_height_m", 0.00)
        self.declare_parameter("max_height_m", 0.20)

        self.declare_parameter("open_deg", 270.95)
        self.declare_parameter("closed_deg", 187.97)

        # --- grasp 성공/실패 판정 (CLOSE 후 전류·위치 기반) ---
        # CLOSE 명령 뒤 delay 만큼 정착 대기 → window 동안 (위치갭, 전류)를
        # 샘플링 → 중앙값으로 HELD/EMPTY 래치. delay+window(<1.3s)는 펌웨어
        # 워치독 재개방(3s)보다 짧아 판정이 항상 물림 구간 안에 든다.
        # 판정 원리(2026-07-17 빈손 실측 기준):
        #  - 주 신호 = 위치갭(present_deg - closed_deg). 빈손은 손가락이 맞닿아
        #    closed_deg 근처까지 닫혀 갭이 작음(빈손 실측 갭 3.2deg). 물체를
        #    물면 물체 폭만큼 덜 닫혀 갭이 큼(수십 deg). → gap>=8.0 을 물림으로.
        #  - 전류는 구별 신호로 부적합: current-based 모드+goal 120이라 빈손도
        #    자기 손가락을 밀며 ~113raw로 포화(물림 ~120과 비슷). 그래서 전류는
        #    "턱이 실제 로드됐나(토크 살아있나)" 확인용 보조 게이트로만 사용.
        # 실측 확정(2026-07-17): 빈손 갭 3.0~3.2deg, 정20면체 물림 갭 14.1deg
        # → gap>=8.0 이 양쪽 중앙(마진 ~5deg)이라 견고. 전류는 빈손 113/물림
        # 120으로 거의 같아 구별엔 못 쓰고 로드 확인(>=60)용 보조로만.
        # 정착 곡선 실측(2026-07-17): CLOSE(열림 214°→닫힘) 후 위치가 빈손은
        # ~577ms, 물체 물림은 ~537ms에 dead-flat 정착(둘 다 <=600ms). 그래서
        # delay 0.6s(빈손 정착 + 마진) + window 0.2s(watchdog 0.2s틱 2샘플)면
        # 판정 ~0.8s — 기존 1.3s 대비 0.5s/파지 단축. 정착 후 값은 완전 고정.
        self.declare_parameter("grasp_check_enabled", True)
        self.declare_parameter("grasp_check_delay_sec", 0.6)
        self.declare_parameter("grasp_check_window_sec", 0.25)
        self.declare_parameter("grasp_pos_gap_deg", 8.0)
        self.declare_parameter("grasp_current_raw_min", 60)
        self.declare_parameter("grasp_topic", "/gripper/grasp")

        self.last_cmd = None
        self.last_cmd_time = 0.0
        self.last_detection_time = 0.0
        self.last_deg = None
        self.last_auto_deg = None
        self.last_status_query_time = 0.0
        self.last_serial_line = ""
        self.dynamixel_ready = None
        self.present_deg = None
        self.present_raw = None
        self.current_raw = None
        self.voltage_raw = None
        self.hardware_error = None
        self.torque_enabled = None
        self.auto_torque_off_ms = None
        # grasp 판정 상태
        self.grasp_pending = False
        self.grasp_close_time = 0.0
        self.grasp_samples = []          # (pos_gap_deg, abs_current_raw)
        self.grasp_state = "unknown"     # unknown / checking / held / empty
        self.grasp_checked = False
        self.grasp_pos_gap_deg_meas = None
        self.grasp_current_raw_meas = None
        self.grasp_result_time = None
        self.serial_port_name = str(self.get_parameter("serial_port").value)
        self.serial_error = ""
        self.baudrate = int(self.get_parameter("baudrate").value)
        self.last_serial_open_attempt_time = 0.0

        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.control_mode = str(
            self.get_parameter("control_mode").value
        ).lower()
        self.ser = None
        if not self.dry_run:
            self.open_serial(force=True)
        else:
            self.get_logger().warn("gripper is in dry_run mode")

        self.sub = self.create_subscription(
            String,
            self.get_parameter("objects_topic").value,
            self.objects_callback,
            10,
        )
        self.command_sub = self.create_subscription(
            String,
            self.get_parameter("command_topic").value,
            self.command_callback,
            10,
        )
        self.status_pub = self.create_publisher(
            String,
            self.get_parameter("status_topic").value,
            10,
        )
        self.grasp_pub = self.create_publisher(
            String,
            self.get_parameter("grasp_topic").value,
            10,
        )
        self.lift_command_sub = self.create_subscription(
            String,
            "/lift/command",
            self.lift_command_callback,
            10,
        )
        self.lift_state_pub = self.create_publisher(String, "/lift/state", 10)
        self.lift_state = {}

        self.timer = self.create_timer(0.2, self.watchdog_callback)

        self.get_logger().info("gripper_bridge_node started")
        self.get_logger().info("control mode: " + self.control_mode)
        if self.ser is not None:
            self.get_logger().info("serial port: " + self.serial_port_name)

    def serial_port_candidates(self):
        patterns = [self.serial_port_name]
        extra = str(self.get_parameter("serial_port_candidates").value)
        patterns.extend(
            part.strip() for part in extra.split(",") if part.strip()
        )

        candidates = []
        seen = set()
        for pattern in patterns:
            if any(ch in pattern for ch in "*?[]"):
                paths = sorted(glob.glob(pattern))
            else:
                paths = [pattern]
            for path in paths:
                if path and path not in seen:
                    candidates.append(path)
                    seen.add(path)
        return candidates

    def open_serial(self, force=False):
        if self.dry_run or self.ser is not None:
            return

        now = time.time()
        reconnect_interval = float(
            self.get_parameter("serial_reconnect_interval_sec").value
        )
        if (
            not force
            and now - self.last_serial_open_attempt_time < reconnect_interval
        ):
            return
        self.last_serial_open_attempt_time = now

        candidates = self.serial_port_candidates()
        present = [path for path in candidates if os.path.exists(path)]
        if not present:
            self.serial_error = (
                "serial port is not present; candidates="
                + ",".join(candidates)
            )
            self.get_logger().warn(self.serial_error, throttle_duration_sec=2.0)
            return

        port_name = present[0]
        try:
            self.ser = serial.Serial(
                port_name,
                self.baudrate,
                timeout=0.05,
                write_timeout=0.2,
                exclusive=True,
            )
            self.serial_port_name = port_name
            time.sleep(2.0)
            self.ser.reset_input_buffer()
            self.serial_error = ""
            self.get_logger().info(
                "gripper serial connected: " + self.serial_port_name
            )
        except SERIAL_ERRORS as exc:
            self.serial_error = str(exc)
            self.get_logger().warn(
                "failed to open gripper serial port "
                + self.serial_port_name
                + ": "
                + self.serial_error
            )
            self.close_serial()

    def close_serial(self):
        if self.ser is None:
            return
        try:
            self.ser.close()
        except SERIAL_ERRORS:
            pass
        self.ser = None

    def handle_serial_error(self, context, exc):
        self.serial_error = str(exc)
        self.get_logger().error(context + ": " + self.serial_error)
        self.close_serial()

    def clamp_deg(self, deg):
        closed_deg = float(self.get_parameter("closed_deg").value)
        open_deg = float(self.get_parameter("open_deg").value)

        if deg < closed_deg:
            return closed_deg

        if deg > open_deg:
            return open_deg

        return deg

    def send_deg(self, deg):
        safe_deg = self.clamp_deg(deg)
        cmd = "SET_DEG " + str(round(safe_deg, 2))
        self.send_cmd(cmd)

    def send_auto_deg(self, deg):
        safe_deg = self.clamp_deg(deg)
        if (
            self.last_auto_deg is not None
            and abs(self.last_auto_deg - safe_deg) < 0.01
        ):
            return
        self.last_auto_deg = safe_deg
        self.send_deg(safe_deg)

    def send_cmd(self, cmd):
        now = time.time()

        if self.last_cmd == cmd and now - self.last_cmd_time < 0.2:
            return

        wrote_serial = False
        if self.ser is None:
            self.open_serial()
        if self.ser is not None:
            try:
                self.ser.write((cmd + "\n").encode("utf-8"))
                wrote_serial = True
                self.serial_error = ""
            except SERIAL_ERRORS as exc:
                self.handle_serial_error(
                    "failed to write gripper serial command",
                    exc,
                )

        self.last_cmd = cmd
        self.last_cmd_time = now
        self.last_status_query_time = 0.0
        if cmd.startswith("SET_DEG "):
            try:
                self.last_deg = float(cmd.split(maxsplit=1)[1])
            except ValueError:
                self.last_deg = None

        if wrote_serial or self.dry_run:
            self.get_logger().info("sent command: " + cmd)
        else:
            self.get_logger().warn("queued command without serial: " + cmd)
        if rclpy.ok():
            self.publish_status()

    def request_status(self):
        if self.ser is None:
            self.open_serial()
        if self.ser is None:
            return
        now = time.time()
        interval = float(
            self.get_parameter("status_poll_interval_sec").value
        )
        if now - self.last_status_query_time < interval:
            return
        try:
            self.ser.write(b"STATUS?\n")
            self.last_status_query_time = now
            self.serial_error = ""
        except SERIAL_ERRORS as exc:
            self.handle_serial_error(
                "failed to request gripper status",
                exc,
            )

    def read_serial(self):
        if self.ser is None:
            self.open_serial()
        if self.ser is None:
            return
        try:
            while self.ser.in_waiting:
                line = self.ser.readline().decode(
                    "utf-8",
                    errors="replace",
                ).strip()
                if not line:
                    continue
                self.last_serial_line = line
                self.parse_serial_line(line)
        except SERIAL_ERRORS as exc:
            self.handle_serial_error(
                "failed to read gripper serial",
                exc,
            )

    def parse_serial_line(self, line):
        parts = line.split()
        if not parts:
            return
        if parts[0] in ("LIFT_STATUS", "LIFT_HOME_SET"):
            self.parse_lift_line(parts)
            return
        if parts[0] != "STATUS":
            return

        values = {}
        index = 1
        while index + 1 < len(parts):
            values[parts[index]] = parts[index + 1]
            index += 2

        def as_float(key):
            try:
                return float(values[key])
            except (KeyError, TypeError, ValueError):
                return None

        def as_int(key):
            try:
                return int(float(values[key]))
            except (KeyError, TypeError, ValueError):
                return None

        if "READY" in values:
            self.dynamixel_ready = as_int("READY") == 1
        if "POS_DEG" in values:
            self.present_deg = as_float("POS_DEG")
        if "POS_RAW" in values:
            self.present_raw = as_int("POS_RAW")
        elif "POS" in values:
            self.present_raw = as_int("POS")
        if "CURRENT_RAW" in values:
            self.current_raw = as_int("CURRENT_RAW")
        elif "CURRENT" in values:
            self.current_raw = as_int("CURRENT")
        if "VOLTAGE_RAW" in values:
            self.voltage_raw = as_int("VOLTAGE_RAW")
        if "HW_ERROR" in values:
            self.hardware_error = as_int("HW_ERROR")
        if "TORQUE" in values:
            self.torque_enabled = as_int("TORQUE") == 1
        if "AUTO_TORQUE_OFF_MS" in values:
            self.auto_torque_off_ms = as_int("AUTO_TORQUE_OFF_MS")

    def publish_status(self):
        now = time.time()
        command_age = None
        if self.last_cmd_time:
            command_age = now - self.last_cmd_time
        detection_age = None
        if self.last_detection_time:
            detection_age = now - self.last_detection_time

        closed_deg = float(self.get_parameter("closed_deg").value)
        open_deg = float(self.get_parameter("open_deg").value)
        target_width_mm = self.deg_to_width_mm(self.last_deg)
        width_mm = self.deg_to_width_mm(self.present_deg)
        current_mA = (
            float(self.current_raw) * 2.69
            if self.current_raw is not None
            else None
        )
        is_moving = (
            self.present_deg is not None
            and self.last_deg is not None
            and abs(float(self.present_deg) - float(self.last_deg)) > 1.0
        )
        target_not_reached = (
            self.present_deg is not None
            and self.last_deg is not None
            and abs(float(self.present_deg) - float(self.last_deg)) > 2.0
        )
        object_detected = (
            self.last_deg is not None
            and abs(float(self.last_deg) - closed_deg) <= 1.0
            and target_not_reached
            and self.current_raw is not None
            and abs(int(self.current_raw)) > 80
        )
        error_msg = self.serial_error
        if self.hardware_error not in (None, 0):
            error_msg = f"hardware_error={self.hardware_error}"

        status = String()
        status.data = json.dumps(
            {
                "width_mm": width_mm,
                "target_width_mm": target_width_mm,
                "current_mA": current_mA,
                "is_moving": is_moving,
                "object_detected": object_detected,
                "error": bool(error_msg),
                "error_msg": error_msg,
                "connected": self.ser is not None,
                "dry_run": self.dry_run,
                "serial_port": self.serial_port_name,
                "serial_error": self.serial_error,
                "control_mode": self.control_mode,
                "command": self.last_cmd,
                "command_age_sec": command_age,
                "last_deg": self.last_deg,
                "last_serial_line": self.last_serial_line,
                "dynamixel_ready": self.dynamixel_ready,
                "present_deg": self.present_deg,
                "present_raw": self.present_raw,
                "current_raw": self.current_raw,
                "voltage_raw": self.voltage_raw,
                "hardware_error": self.hardware_error,
                "torque_enabled": self.torque_enabled,
                "auto_torque_off_ms": self.auto_torque_off_ms,
                "detection_age_sec": detection_age,
                "target_class": self.get_parameter("target_class").value,
                "open_deg": open_deg,
                "closed_deg": closed_deg,
                "grasp_state": self.grasp_state,
                "grasp_checked": self.grasp_checked,
                "grasp_pos_gap_deg": self.grasp_pos_gap_deg_meas,
                "grasp_current_raw": self.grasp_current_raw_meas,
            }
        )
        self.status_pub.publish(status)

    def deg_to_width_mm(self, deg):
        if deg is None:
            return None
        closed_deg = float(self.get_parameter("closed_deg").value)
        open_deg = float(self.get_parameter("open_deg").value)
        if abs(open_deg - closed_deg) < 1e-6:
            return None
        ratio = (float(deg) - closed_deg) / (open_deg - closed_deg)
        ratio = max(0.0, min(1.0, ratio))
        return ratio * 70.0

    def command_callback(self, msg):
        command = msg.data.strip().upper()
        self.last_auto_deg = None
        open_deg = float(self.get_parameter("open_deg").value)
        closed_deg = float(self.get_parameter("closed_deg").value)

        if command == "OPEN":
            self.cancel_grasp_check()
            self.send_deg(open_deg)
        elif command == "CLOSE":
            self.send_deg(closed_deg)
            self.start_grasp_check()
        elif command == "STOP":
            self.cancel_grasp_check()
            self.send_cmd("STOP")
        elif command.startswith("LIFT_"):
            # 카메라 마스트 리프트(ID12) 명령은 그리퍼 상태와 무관하게 그대로 전달
            self.cancel_grasp_check()
            self.send_cmd(command)
        elif command.startswith("SET_DEG "):
            try:
                deg = float(command.split(maxsplit=1)[1])
                self.send_deg(deg)
                # 닫힘 근처로의 SET_DEG 도 파지 시도로 간주(정밀 파지 명령)
                if abs(deg - closed_deg) <= 1.0:
                    self.start_grasp_check()
                else:
                    self.cancel_grasp_check()
            except ValueError:
                self.get_logger().warn("invalid SET_DEG command")
        elif command.startswith("LIFT_"):
            # 마스트 리프트 패스스루. 별도 프로세스가 같은 tty를 다시 열면
            # DTR 리셋으로 보드가 재부팅되어 홈 캡처가 소실되고(LIFT_TO_TOP
            # 하드스톱 돌진 위험) 그리퍼도 강제 OPEN된다 — 리프트 조작은
            # 반드시 이 브리지를 통해야 한다.
            self.send_cmd(command)
        else:
            self.get_logger().warn("unknown gripper command: " + command)

    def lift_command_callback(self, msg):
        command = msg.data.strip().upper()
        if not command.startswith("LIFT_"):
            self.get_logger().warn("lift command must start with LIFT_: " + command)
            return
        self.send_cmd(command)

    def parse_lift_line(self, parts):
        """LIFT_STATUS/LIFT_HOME_SET 텔레메트리 -> /lift/state JSON."""
        if parts[0] == "LIFT_HOME_SET" and len(parts) >= 2:
            self.lift_state["home_capture_raw"] = parts[1]
        else:
            values = {}
            index = 1
            while index + 1 < len(parts):
                values[parts[index]] = parts[index + 1]
                index += 2
            for key in ("READY", "POS_RAW", "CURRENT_RAW", "MOVING",
                        "HW_ERROR", "TORQUE", "HOME_SET", "HOME_RAW", "STROKE"):
                if key in values:
                    try:
                        self.lift_state[key.lower()] = int(float(values[key]))
                    except ValueError:
                        pass
        self.lift_state["stamp"] = time.time()
        msg = String()
        msg.data = json.dumps(self.lift_state)
        self.lift_state_pub.publish(msg)

    def start_grasp_check(self):
        if not bool(self.get_parameter("grasp_check_enabled").value):
            return
        self.grasp_pending = True
        self.grasp_checked = False
        self.grasp_close_time = time.time()
        self.grasp_samples = []
        self.grasp_state = "checking"
        self.grasp_pos_gap_deg_meas = None
        self.grasp_current_raw_meas = None

    def cancel_grasp_check(self):
        self.grasp_pending = False
        if self.grasp_state == "checking":
            self.grasp_state = "unknown"

    def update_grasp_check(self, now):
        """CLOSE 후 정착 윈도우 동안 (위치갭, 전류)를 모아 HELD/EMPTY 판정."""
        if not self.grasp_pending:
            return
        delay = float(self.get_parameter("grasp_check_delay_sec").value)
        window = float(self.get_parameter("grasp_check_window_sec").value)
        elapsed = now - self.grasp_close_time
        if elapsed < delay:
            return
        # 판정 윈도우: 매 틱 최신 STATUS 값을 샘플 (request_status 스로틀은
        # watchdog에서 pending 시 우회하므로 ~0.2s 간격으로 갱신됨).
        closed_deg = float(self.get_parameter("closed_deg").value)
        if self.present_deg is not None and self.current_raw is not None:
            gap = abs(float(self.present_deg) - closed_deg)
            self.grasp_samples.append((gap, abs(int(self.current_raw))))
        if elapsed < delay + window:
            return
        # 윈도우 종료 → 중앙값으로 래치 판정
        self.grasp_pending = False
        self.grasp_checked = True
        self.grasp_result_time = now
        if not self.grasp_samples:
            self.grasp_state = "unknown"
            self.publish_grasp()
            self.get_logger().warn("grasp check: no samples (serial/status?)")
            return
        gaps = sorted(s[0] for s in self.grasp_samples)
        curs = sorted(s[1] for s in self.grasp_samples)
        med_gap = gaps[len(gaps) // 2]
        med_cur = curs[len(curs) // 2]
        self.grasp_pos_gap_deg_meas = round(med_gap, 2)
        self.grasp_current_raw_meas = med_cur
        gap_min = float(self.get_parameter("grasp_pos_gap_deg").value)
        cur_min = int(self.get_parameter("grasp_current_raw_min").value)
        held = med_gap >= gap_min and med_cur >= cur_min
        self.grasp_state = "held" if held else "empty"
        self.publish_grasp()
        self.get_logger().info(
            f"grasp {self.grasp_state}: pos_gap={med_gap:.2f}deg "
            f"current={med_cur}raw n={len(self.grasp_samples)} "
            f"(thr gap>={gap_min} cur>={cur_min})")

    def publish_grasp(self):
        msg = String()
        msg.data = json.dumps({
            "state": self.grasp_state,
            "pos_gap_deg": self.grasp_pos_gap_deg_meas,
            "current_raw": self.grasp_current_raw_meas,
            "samples": len(self.grasp_samples),
            "stamp": self.grasp_result_time,
        })
        self.grasp_pub.publish(msg)

    def normalize_objects(self, data):
        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            if "objects" in data and isinstance(data["objects"], list):
                return data["objects"]

            if "detections" in data and isinstance(data["detections"], list):
                return data["detections"]

            if "results" in data and isinstance(data["results"], list):
                return data["results"]

            return [data]

        return []

    def get_value(self, obj, keys, default=None):
        for key in keys:
            if key in obj:
                return obj[key]
        return default

    def is_graspable(self, obj):
        target_class = self.get_parameter("target_class").value
        min_confidence = float(self.get_parameter("min_confidence").value)

        cls = str(self.get_value(obj, ["class", "class_name", "label"], ""))
        confidence = float(
            self.get_value(obj, ["confidence", "conf", "score"], 0.0)
        )

        if target_class != "" and cls != target_class:
            return False

        if confidence < min_confidence:
            return False

        forward = self.get_value(obj, ["forward_m", "F", "x"])
        left = self.get_value(obj, ["left_m", "L", "y"])
        height = self.get_value(obj, ["height_above_ground_m", "H", "z"])

        if forward is None:
            return False

        if left is None:
            return False

        if height is None:
            return False

        forward = float(forward)
        left = float(left)
        height = float(height)

        min_forward = float(self.get_parameter("min_forward_m").value)
        max_forward = float(self.get_parameter("max_forward_m").value)
        max_abs_left = float(self.get_parameter("max_abs_left_m").value)
        min_height = float(self.get_parameter("min_height_m").value)
        max_height = float(self.get_parameter("max_height_m").value)

        if forward < min_forward:
            return False

        if forward > max_forward:
            return False

        if abs(left) > max_abs_left:
            return False

        if height < min_height:
            return False

        if height > max_height:
            return False

        return True

    def objects_callback(self, msg):
        if self.control_mode != "detection":
            return

        try:
            data = json.loads(msg.data)
        except Exception:
            self.get_logger().warn("failed to parse objects_json")
            return

        objects = self.normalize_objects(data)

        graspable_objects = []

        for obj in objects:
            if self.is_graspable(obj):
                graspable_objects.append(obj)

        closed_deg = float(self.get_parameter("closed_deg").value)
        open_deg = float(self.get_parameter("open_deg").value)

        if len(graspable_objects) > 0:
            self.last_detection_time = time.time()
            self.send_auto_deg(closed_deg)
        else:
            self.send_auto_deg(open_deg)

    def watchdog_callback(self):
        self.read_serial()
        # 파지 판정 중엔 스로틀을 우회해 매 틱 최신 전류/위치를 확보
        if self.grasp_pending:
            self.last_status_query_time = 0.0
        self.request_status()
        self.update_grasp_check(time.time())
        if self.control_mode != "detection":
            self.publish_status()
            return

        now = time.time()

        open_deg = float(self.get_parameter("open_deg").value)

        if now - self.last_detection_time > 1.0:
            self.send_auto_deg(open_deg)

        self.publish_status()


def main(args=None):
    rclpy.init(args=args)

    node = GripperBridgeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if bool(node.get_parameter("open_on_shutdown").value):
            open_deg = float(node.get_parameter("open_deg").value)
            node.send_deg(open_deg)
        if node.ser is not None:
            node.ser.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
