"""Shared serial helpers for the mecanum bringup-day scripts.

All scripts here talk DIRECTLY to the mecanum_encoder_control firmware.
The ROS bridge (mecanum_bridge_node) must NOT be running at the same time
(the serial port is exclusive).
"""

import glob
import sys
import time

try:
    import serial  # pyserial
except ImportError:  # pragma: no cover
    print("pyserial이 없습니다: pip3 install pyserial", file=sys.stderr)
    raise

DEFAULT_CANDIDATES = [
    "/dev/serial/by-id/usb-Arduino__www.arduino.cc__Arduino_14101-if00",
    "/dev/serial/by-path/platform-3610000.usb-usb-0:2.3.1:1.0",
    "/dev/ttyACM0",
    "/dev/ttyACM1",
]

WHEELS = ["front_left", "front_right", "rear_left", "rear_right"]


def find_port(explicit=None):
    candidates = [explicit] if explicit else DEFAULT_CANDIDATES + sorted(
        glob.glob("/dev/ttyACM*")
    )
    for path in candidates:
        if path is None:
            continue
        try:
            with open(path, "rb"):
                return path
        except OSError:
            continue
    raise SystemExit(
        "Arduino 시리얼 포트를 찾지 못했습니다. --port로 직접 지정하세요.\n"
        "확인: ls /dev/ttyACM* /dev/serial/by-id/"
    )


class Mecanum:
    """Line-oriented helper over the firmware serial protocol."""

    def __init__(self, port, baud=115200, boot_delay=2.5, verbose=False):
        self.verbose = verbose
        self.ser = serial.Serial(port, baud, timeout=0.1)
        print(f"[serial] {port} 연결, 부팅 대기 {boot_delay}s...")
        time.sleep(boot_delay)  # UNO auto-reset on open
        self.drain()

    def close(self):
        try:
            self.send("stop")
        except Exception:
            pass
        self.ser.close()

    def drain(self, seconds=0.3):
        end = time.time() + seconds
        while time.time() < end:
            line = self.readline()
            if line is None:
                time.sleep(0.02)

    def readline(self):
        raw = self.ser.readline()
        if not raw:
            return None
        line = raw.decode("ascii", errors="replace").strip()
        if line and self.verbose:
            print(f"  << {line}")
        return line or None

    def send(self, cmd):
        if self.verbose:
            print(f"  >> {cmd}")
        self.ser.write((cmd + "\n").encode("ascii"))
        self.ser.flush()

    def command(self, cmd, expect_prefixes=("OK", "ERR"), timeout=2.0):
        """Send a command and wait for an OK/ERR (or custom prefix) line.

        Returns (matched_line, other_lines_seen)."""
        self.send(cmd)
        others = []
        end = time.time() + timeout
        while time.time() < end:
            line = self.readline()
            if line is None:
                continue
            if any(line.startswith(p) for p in expect_prefixes):
                return line, others
            others.append(line)
        return None, others

    def collect(self, seconds, prefixes=None):
        """Collect lines for a fixed window (optionally filtered by prefix)."""
        out = []
        end = time.time() + seconds
        while time.time() < end:
            line = self.readline()
            if line is None:
                continue
            if prefixes is None or any(line.startswith(p) for p in prefixes):
                out.append(line)
        return out

    def read_ticks(self, timeout=2.0):
        """Read one STATE line and return the 4 cumulative tick counts."""
        self.send("stream 1")
        end = time.time() + timeout
        ticks = None
        while time.time() < end:
            line = self.readline()
            if line and line.startswith("STATE,"):
                parts = line.split(",")
                if len(parts) >= 10:
                    ticks = [int(parts[2 + i]) for i in range(4)]
                    break
        self.send("stream 0")
        self.drain(0.2)
        if ticks is None:
            raise RuntimeError("STATE 라인을 받지 못했습니다 (stream 1 후 무응답)")
        return ticks


def parse_state(line):
    """STATE,<ms>,<t_fl>,<t_fr>,<t_rl>,<t_rr>,<pwm x4>,<mode> -> dict or None."""
    parts = line.split(",")
    if len(parts) != 11 or parts[0] != "STATE":
        return None
    try:
        return {
            "ms": int(parts[1]),
            "ticks": [int(p) for p in parts[2:6]],
            "pwm": [int(p) for p in parts[6:10]],
            "mode": int(parts[10]),
        }
    except ValueError:
        return None
