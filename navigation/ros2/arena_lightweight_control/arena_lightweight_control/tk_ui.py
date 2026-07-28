#!/usr/bin/env python3
import json
import hashlib
import math
import os
import threading
import time
from pathlib import Path
from typing import Optional

import rclpy
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import String

from arena_lightweight_control.map_localization import (
    OccupancyMap,
    Pose2D,
    scan_to_world_points,
)

# MK3 footprint, robot frame (x forward, y left). The origin is the LiDAR
# centre because that is what the localizer actually solves for: scan_to_points
# applies only a yaw offset, never a base_link<-lidar translation, so the pose
# published on /arena_lightweight/status IS the scan origin in map frame.
# Camera mounts are expressed the same way (fieldlib TOP_MOUNT forward_m).
ROBOT_FRONT_M = 0.25       # gripper tip, ahead of the LiDAR
ROBOT_REAR_M = -0.12       # 0.37 m total length
ROBOT_HALF_WIDTH_M = 0.15  # 0.30 m total width
GRIPPER_LEN_M = 0.12       # frontmost 12 cm is the gripper, drawn separately


class ArenaTkUiNode(Node):
    def __init__(self):
        super().__init__("arena_tk_ui")
        self.declare_parameter("map_yaml", default_map_yaml())
        self.declare_parameter("scan_topic", "/laser_scan")
        self.declare_parameter("scan_yaw_offset_rad", 0.0)
        self.declare_parameter("status_topic", "/arena_lightweight/status")
        self.declare_parameter("goal_command_topic", "/arena_lightweight/goal")
        self.declare_parameter("pose_command_topic", "/arena_lightweight/pose")
        self.declare_parameter("control_command_topic", "/arena_lightweight/control")
        self.declare_parameter("objects_topic", "/detected_objects/map_objects")
        self.declare_parameter("object_display_max_age_sec", 1.5)
        self.declare_parameter("gripper_command_topic", "/gripper/command")
        self.declare_parameter(
            "gripper_joint_command_topic",
            "/mk1/gripper_joint_command",
        )
        self.declare_parameter(
            "gripper_joint_names",
            "lower_gripper_slide,upper_gripper_slide",
        )
        self.declare_parameter("gripper_open_positions", "-0.06,0.06")
        self.declare_parameter("gripper_close_positions", "0.05,-0.05")
        self.declare_parameter("gripper_repeat_duration_sec", 1.2)
        self.declare_parameter("gripper_repeat_period_sec", 0.25)
        self.declare_parameter("scan_display_max_points", 120)
        self.declare_parameter("draw_period_ms", 100)

        self.map = OccupancyMap.from_yaml(str(self.get_parameter("map_yaml").value))
        self.scan_yaw_offset_rad = float(
            self.get_parameter("scan_yaw_offset_rad").value
        )
        self.scan_display_max_points = int(
            self.get_parameter("scan_display_max_points").value
        )
        self.draw_period_ms = int(self.get_parameter("draw_period_ms").value)
        self.object_display_max_age_sec = float(
            self.get_parameter("object_display_max_age_sec").value
        )
        self.gripper_joint_names = _parse_csv_strings(
            str(self.get_parameter("gripper_joint_names").value)
        )
        self.gripper_open_positions = _parse_csv_floats(
            str(self.get_parameter("gripper_open_positions").value)
        )
        self.gripper_close_positions = _parse_csv_floats(
            str(self.get_parameter("gripper_close_positions").value)
        )
        self.gripper_repeat_duration_sec = max(
            0.0,
            float(self.get_parameter("gripper_repeat_duration_sec").value),
        )
        self.gripper_repeat_period_sec = max(
            0.05,
            float(self.get_parameter("gripper_repeat_period_sec").value),
        )
        self.lock = threading.Lock()
        self.latest_state: dict = {}
        self.latest_scan: Optional[dict] = None
        self.latest_objects: list[dict] = []
        self.latest_objects_time = 0.0
        self.repeat_gripper_command = ""
        self.repeat_gripper_until = 0.0
        self.last_gripper_repeat_time = 0.0

        self.goal_pub = self.create_publisher(
            String,
            str(self.get_parameter("goal_command_topic").value),
            10,
        )
        self.pose_pub = self.create_publisher(
            String,
            str(self.get_parameter("pose_command_topic").value),
            10,
        )
        self.control_pub = self.create_publisher(
            String,
            str(self.get_parameter("control_command_topic").value),
            10,
        )
        self.gripper_pub = self.create_publisher(
            String,
            str(self.get_parameter("gripper_command_topic").value),
            10,
        )
        self.gripper_joint_pub = self.create_publisher(
            JointState,
            str(self.get_parameter("gripper_joint_command_topic").value),
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("status_topic").value),
            self.status_callback,
            10,
        )
        self.create_subscription(
            LaserScan,
            str(self.get_parameter("scan_topic").value),
            self.scan_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("objects_topic").value),
            self.objects_callback,
            10,
        )
        self.create_timer(
            self.gripper_repeat_period_sec,
            self.repeat_gripper_tick,
        )

    def status_callback(self, msg: String):
        try:
            state = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        with self.lock:
            self.latest_state = state

    def scan_callback(self, msg: LaserScan):
        stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        with self.lock:
            self.latest_scan = {
                "ranges": tuple(float(value) for value in msg.ranges),
                "angle_min": float(msg.angle_min),
                "angle_increment": float(msg.angle_increment),
                "range_min": float(msg.range_min),
                "range_max": float(msg.range_max),
                "stamp_sec": stamp_sec,
            }

    def objects_callback(self, msg: String):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        objects = payload.get("objects") if isinstance(payload, dict) else None
        if not isinstance(objects, list):
            return
        with self.lock:
            self.latest_objects = [obj for obj in objects if isinstance(obj, dict)]
            self.latest_objects_time = time.monotonic()

    def snapshot(self) -> tuple[dict, Optional[dict], list[dict]]:
        with self.lock:
            state = dict(self.latest_state)
            scan = dict(self.latest_scan) if self.latest_scan is not None else None
            objects_fresh = (
                self.latest_objects_time > 0.0
                and time.monotonic() - self.latest_objects_time <= self.object_display_max_age_sec
            )
            objects = [dict(obj) for obj in self.latest_objects] if objects_fresh else []
        return state, scan, objects

    def publish_goal(self, x: float, y: float):
        msg = String()
        msg.data = json.dumps({"x": float(x), "y": float(y)}, separators=(",", ":"))
        self.goal_pub.publish(msg)

    def publish_pose(self, x: float, y: float, yaw: float):
        msg = String()
        msg.data = json.dumps(
            {"x": float(x), "y": float(y), "yaw": float(yaw)},
            separators=(",", ":"),
        )
        self.pose_pub.publish(msg)

    def publish_stop(self):
        msg = String()
        msg.data = "STOP"
        self.control_pub.publish(msg)

    def publish_gripper(self, command: str):
        command = command.strip().upper()
        if not command:
            return
        self.publish_gripper_once(command)
        if command in {"OPEN", "CLOSE"} and self.gripper_repeat_duration_sec > 0.0:
            with self.lock:
                self.repeat_gripper_command = command
                self.repeat_gripper_until = (
                    time.monotonic() + self.gripper_repeat_duration_sec
                )
                self.last_gripper_repeat_time = time.monotonic()

    def publish_gripper_once(self, command: str):
        msg = String()
        msg.data = command
        self.gripper_pub.publish(msg)
        if command in {"OPEN", "CLOSE"}:
            self.publish_gripper_joint(command)

    def publish_gripper_joint(self, command: str):
        positions = (
            self.gripper_open_positions
            if command == "OPEN"
            else self.gripper_close_positions
        )
        if len(self.gripper_joint_names) != len(positions):
            self.get_logger().warn(
                "gripper JointState command skipped: joint name and position "
                "counts differ",
                throttle_duration_sec=2.0,
            )
            return

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.gripper_joint_names)
        msg.position = list(positions)
        self.gripper_joint_pub.publish(msg)

    def repeat_gripper_tick(self):
        now = time.monotonic()
        with self.lock:
            command = self.repeat_gripper_command
            repeat_until = self.repeat_gripper_until
            last_repeat = self.last_gripper_repeat_time
            if not command or now > repeat_until:
                self.repeat_gripper_command = ""
                return
            if now - last_repeat < self.gripper_repeat_period_sec * 0.8:
                return
            self.last_gripper_repeat_time = now
        self.publish_gripper_once(command)


class ArenaTkUi:
    def __init__(self, node: ArenaTkUiNode):
        import tkinter as tk

        self.tk = tk
        self.node = node
        self.map = node.map
        self.mode = "goal"
        self.layout_cache = None
        self.root = tk.Tk()
        self.root.title("Arena Control")
        self.root.geometry("1040x720")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.canvas = tk.Canvas(self.root, bg="#dfe7e9", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        self.sidebar = tk.Frame(self.root, width=320, bg="#ffffff")
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        self.metric_vars = {}
        self.build_sidebar()
        self.occupied_cells = [
            (index % self.map.width, index // self.map.width)
            for index, occupied in enumerate(self.map.occupied)
            if occupied
        ]
        self.static_dirty = True
        self.closed = False

    def build_sidebar(self):
        tk = self.tk
        title = tk.Label(
            self.sidebar,
            text="Arena Control",
            bg="#ffffff",
            fg="#1d2528",
            font=("Arial", 16, "bold"),
            anchor="w",
        )
        title.pack(fill=tk.X, padx=14, pady=(14, 8))

        mode_row = tk.Frame(self.sidebar, bg="#ffffff")
        mode_row.pack(fill=tk.X, padx=14, pady=4)
        self.goal_button = tk.Button(
            mode_row,
            text="Goal",
            command=lambda: self.set_mode("goal"),
            relief=tk.SUNKEN,
        )
        self.goal_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self.pose_button = tk.Button(
            mode_row,
            text="Pose",
            command=lambda: self.set_mode("pose"),
        )
        self.pose_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        stop = tk.Button(
            self.sidebar,
            text="Stop",
            command=self.node.publish_stop,
            bg="#b93830",
            fg="#ffffff",
            activebackground="#a12d27",
        )
        stop.pack(fill=tk.X, padx=14, pady=8)

        for commands in (("OPEN", "CLOSE", "STOP"), ("DXL_POWER_ON", "DXL_POWER_CYCLE")):
            row = tk.Frame(self.sidebar, bg="#ffffff")
            row.pack(fill=tk.X, padx=14, pady=4)
            for command in commands:
                button = tk.Button(
                    row,
                    text=command.replace("DXL_", "").replace("_", " "),
                    command=lambda value=command: self.node.publish_gripper(value),
                )
                button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        metrics = tk.Frame(self.sidebar, bg="#ffffff")
        metrics.pack(fill=tk.X, padx=14, pady=(10, 4))
        for name in (
            "pose",
            "goal",
            "phase",
            "match",
            "scan",
            "objects",
            "front",
            "left",
            "back",
            "right",
            "motor",
            "gripper",
        ):
            row = tk.Frame(metrics, bg="#ffffff")
            row.pack(fill=tk.X, pady=2)
            tk.Label(
                row,
                text=name,
                width=9,
                anchor="w",
                bg="#ffffff",
                fg="#5f6f74",
                font=("Arial", 10),
            ).pack(side=tk.LEFT)
            value = tk.StringVar(value="none")
            self.metric_vars[name] = value
            tk.Label(
                row,
                textvariable=value,
                anchor="w",
                bg="#ffffff",
                fg="#1d2528",
                font=("Arial", 10),
                wraplength=190,
                justify=tk.LEFT,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def set_mode(self, mode: str):
        self.mode = mode
        self.goal_button.configure(relief=self.tk.SUNKEN if mode == "goal" else self.tk.RAISED)
        self.pose_button.configure(relief=self.tk.SUNKEN if mode == "pose" else self.tk.RAISED)

    def on_canvas_resize(self, _event):
        self.static_dirty = True

    def on_canvas_click(self, event):
        state, _scan, _objects = self.node.snapshot()
        view = self.layout()
        point = self.canvas_to_world(event.x, event.y, view)
        if self.mode == "pose":
            pose = state.get("pose") or {}
            self.node.publish_pose(point[0], point[1], float(pose.get("yaw") or 0.0))
        else:
            self.node.publish_goal(point[0], point[1])

    def layout(self):
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        scale = min(width / self.map.width, height / self.map.height)
        offset_x = (width - self.map.width * scale) * 0.5
        offset_y = (height - self.map.height * scale) * 0.5
        return scale, offset_x, offset_y

    def world_to_canvas(self, x: float, y: float, view):
        scale, offset_x, offset_y = view
        mx = (float(x) - self.map.origin[0]) / self.map.resolution
        my = (float(y) - self.map.origin[1]) / self.map.resolution
        return offset_x + mx * scale, offset_y + (self.map.height - my) * scale

    def canvas_to_world(self, px: float, py: float, view):
        scale, offset_x, offset_y = view
        mx = (float(px) - offset_x) / scale
        my = self.map.height - ((float(py) - offset_y) / scale)
        return (
            self.map.origin[0] + mx * self.map.resolution,
            self.map.origin[1] + my * self.map.resolution,
        )

    def draw_static_map(self):
        view = self.layout()
        scale, offset_x, offset_y = view
        self.canvas.delete("static")
        self.canvas.create_rectangle(
            offset_x,
            offset_y,
            offset_x + self.map.width * scale,
            offset_y + self.map.height * scale,
            fill="#f4f7f8",
            outline="#b8c7cc",
            tags="static",
        )
        for col, row in self.occupied_cells:
            x1 = offset_x + col * scale
            y1 = offset_y + row * scale
            self.canvas.create_rectangle(
                x1,
                y1,
                x1 + max(1.0, scale),
                y1 + max(1.0, scale),
                fill="#263238",
                outline="",
                tags="static",
            )
        self.draw_competition_layout(view)
        self.static_dirty = False

    def draw_competition_layout(self, view):
        """공식 배치 격자 42점 + 출발/보관함 구역 + 중앙 스캔점 오버레이."""
        from . import competition_layout as layout

        for name, rect, color in (
            ("STORAGE", layout.STORAGE_RECT_MAP, "#2e7d32"),
            ("START", layout.START_RECT_MAP, "#1565c0"),
        ):
            x1, y1 = self.world_to_canvas(rect[0], rect[1], view)
            x2, y2 = self.world_to_canvas(rect[2], rect[3], view)
            self.canvas.create_rectangle(
                x1, y1, x2, y2, outline=color, width=2, tags="static")
            self.canvas.create_text(
                (x1 + x2) / 2, (y1 + y2) / 2, text=name,
                fill=color, font=("TkDefaultFont", 8, "bold"), tags="static")

        for gx, gy in layout.GRID_POINTS_MAP:
            px, py = self.world_to_canvas(gx, gy, view)
            self.canvas.create_oval(
                px - 3, py - 3, px + 3, py + 3,
                outline="#8e24aa", width=1.5, tags="static")
            self.canvas.create_line(
                px - 6, py, px + 6, py, fill="#ce93d8", tags="static")
            self.canvas.create_line(
                px, py - 6, px, py + 6, fill="#ce93d8", tags="static")

        cx, cy = self.world_to_canvas(*layout.CENTER_SCAN_MAP, view)
        self.canvas.create_oval(
            cx - 8, cy - 8, cx + 8, cy + 8,
            outline="#e65100", width=2, dash=(3, 2), tags="static")
        self.canvas.create_text(
            cx, cy - 14, text="SCAN", fill="#e65100",
            font=("TkDefaultFont", 8, "bold"), tags="static")

    def draw_dynamic(self):
        if self.static_dirty:
            self.draw_static_map()
        view = self.layout()
        state, scan, objects = self.node.snapshot()
        self.canvas.delete("dynamic")

        pose_dict = state.get("pose")
        if pose_dict and scan:
            pose = Pose2D(
                float(pose_dict.get("x", 0.0)),
                float(pose_dict.get("y", 0.0)),
                float(pose_dict.get("yaw", 0.0)),
            )
            points, total = scan_to_world_points(
                scan["ranges"],
                scan["angle_min"],
                scan["angle_increment"],
                scan["range_min"],
                scan["range_max"],
                self.node.scan_yaw_offset_rad,
                pose,
                self.node.scan_display_max_points,
            )
            self.metric_vars["scan"].set(f"{len(points)}/{total} pts")
            for x, y in points:
                px, py = self.world_to_canvas(x, y, view)
                self.canvas.create_oval(
                    px - 2,
                    py - 2,
                    px + 2,
                    py + 2,
                    fill="#1c78c0",
                    outline="",
                    tags="dynamic",
                )
        else:
            self.metric_vars["scan"].set("none")

        self.draw_objects(objects, view)

        goal = state.get("goal")
        if goal:
            self.draw_goal(goal, view)
        if pose_dict:
            self.draw_robot(pose_dict, view)
        self.update_metrics(state)

    def draw_objects(self, objects: list[dict], view):
        if not objects:
            self.metric_vars["objects"].set("none")
            return
        self.metric_vars["objects"].set(str(len(objects)))
        for obj in objects:
            position = obj.get("position") if isinstance(obj, dict) else None
            if not isinstance(position, dict):
                continue
            try:
                x = float(position["x"])
                y = float(position["y"])
            except (KeyError, TypeError, ValueError):
                continue
            px, py = self.world_to_canvas(x, y, view)
            fill = object_color_hex(obj)
            outline = darker_hex(fill)
            label = object_display_label(obj)
            self.canvas.create_oval(
                px - 6,
                py - 6,
                px + 6,
                py + 6,
                fill=fill,
                outline=outline,
                width=2,
                tags="dynamic",
            )
            self.canvas.create_text(
                px + 10,
                py - 10,
                text=label,
                anchor="w",
                fill=outline,
                font=("Arial", 10, "bold"),
                tags="dynamic",
            )

    def draw_goal(self, goal: dict, view):
        px, py = self.world_to_canvas(goal["x"], goal["y"], view)
        radius = 10
        self.canvas.create_oval(
            px - radius,
            py - radius,
            px + radius,
            py + radius,
            outline="#bf7b00",
            width=3,
            tags="dynamic",
        )
        self.canvas.create_line(px - 14, py, px + 14, py, fill="#bf7b00", width=3, tags="dynamic")
        self.canvas.create_line(px, py - 14, px, py + 14, fill="#bf7b00", width=3, tags="dynamic")

    def draw_robot(self, pose: dict, view):
        x = float(pose["x"])
        y = float(pose["y"])
        yaw = float(pose.get("yaw") or 0.0)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        def corners(x_back: float, x_front: float):
            """Rectangle in robot frame -> flat canvas coordinate list."""
            flat = []
            for lx, ly in (
                (x_front, ROBOT_HALF_WIDTH_M),
                (x_front, -ROBOT_HALF_WIDTH_M),
                (x_back, -ROBOT_HALF_WIDTH_M),
                (x_back, ROBOT_HALF_WIDTH_M),
            ):
                flat.extend(
                    self.world_to_canvas(
                        x + lx * cos_yaw - ly * sin_yaw,
                        y + lx * sin_yaw + ly * cos_yaw,
                        view,
                    )
                )
            return flat

        gripper_back = ROBOT_FRONT_M - GRIPPER_LEN_M
        self.canvas.create_polygon(
            corners(ROBOT_REAR_M, gripper_back),
            fill="#1f7a6d",
            outline="#0f3f38",
            width=1,
            tags="dynamic",
        )
        self.canvas.create_polygon(
            corners(gripper_back, ROBOT_FRONT_M),
            fill="#f0a500",
            outline="#0f3f38",
            width=1,
            tags="dynamic",
        )

        # LiDAR centre == the pose the localizer solves for.
        px, py = self.world_to_canvas(x, y, view)
        self.canvas.create_oval(
            px - 4,
            py - 4,
            px + 4,
            py + 4,
            fill="#ffffff",
            outline="#0f3f38",
            width=2,
            tags="dynamic",
        )

    def update_metrics(self, state: dict):
        pose = state.get("pose")
        goal = state.get("goal")
        sectors = state.get("scan_sector_ranges") or {}
        localization = state.get("localization")
        motor_state = state.get("motor_state") or ""
        gripper_status = state.get("gripper_status") or ""
        self.metric_vars["pose"].set(
            f'{pose["x"]:.2f}, {pose["y"]:.2f}, {pose["yaw"]:.2f}' if pose else "none"
        )
        self.metric_vars["goal"].set(
            f'{goal["x"]:.2f}, {goal["y"]:.2f}' if goal else "none"
        )
        self.metric_vars["phase"].set(state.get("command_phase") or "idle")
        self.metric_vars["match"].set(
            f'{localization["latency_ms"]:.1f} ms / {localization["score"]:.3f}'
            if localization
            else "none"
        )
        for name in ("front", "left", "back", "right"):
            value = state.get("obstacle_front_m") if name == "front" else sectors.get(name)
            self.metric_vars[name].set("none" if value is None else f"{float(value):.2f} m")
        self.metric_vars["motor"].set(motor_state[:90] if motor_state else "none")
        self.metric_vars["gripper"].set(
            gripper_status[:90] if gripper_status else "none"
        )

    def tick(self):
        if self.closed:
            return
        self.draw_dynamic()
        self.root.after(max(30, self.node.draw_period_ms), self.tick)

    def run(self):
        self.root.after(100, self.tick)
        self.root.mainloop()

    def close(self):
        self.closed = True
        self.root.destroy()


def default_map_yaml() -> str:
    try:
        return str(Path(get_package_share_directory("example_nav2")) / "maps" / "stadium.yaml")
    except PackageNotFoundError:
        repo_root = find_repo_root()
        return str(repo_root / "src" / "example_nav2" / "maps" / "stadium.yaml")


def find_repo_root() -> Path:
    env_root = os.environ.get("ROBOT_REPO_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend([Path.cwd(), Path(__file__).resolve()])
    for candidate in candidates:
        for root in [candidate, *candidate.parents]:
            if (root / "src" / "example_nav2" / "maps" / "stadium.yaml").exists():
                return root
    return Path.cwd()


def _parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_csv_floats(value: str) -> list[float]:
    values = []
    for item in value.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def object_color_hex(obj: dict) -> str:
    display_color = obj.get("display_color")
    if isinstance(display_color, str) and len(display_color) == 7 and display_color.startswith("#"):
        return display_color
    class_name = str(obj.get("class_name") or "").lower()
    if "apple" in class_name:
        return "#db141a"
    if "orange" in class_name:
        return "#f56b0a"
    if "banana" in class_name:
        return "#f5c71a"
    if "pineapple" in class_name:
        return "#66ad29"
    if "plain" in class_name or "blank" in class_name:
        return "#9ea8b8"
    if "unresolved" in class_name or "unknown" in class_name:
        return "#8c59d1"
    if "too_far" in class_name:
        return "#6b7280"
    key = str(obj.get("id") or obj.get("class_name") or "object")
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    red = 64 + digest[0] % 160
    green = 64 + digest[1] % 160
    blue = 64 + digest[2] % 160
    return f"#{red:02x}{green:02x}{blue:02x}"


def object_display_label(obj: dict) -> str:
    label = obj.get("display_label")
    if isinstance(label, str) and label.strip():
        return label.strip()
    class_name = str(obj.get("class_name") or obj.get("id") or "object")
    if class_name.startswith("fruit_cube:"):
        return class_name.split(":", 1)[1]
    if class_name == "plain_cube":
        return "plain"
    if class_name == "cube_like_unresolved":
        return "unresolved"
    return class_name


def darker_hex(color: str) -> str:
    value = color.lstrip("#")
    if len(value) != 6:
        return "#111827"
    try:
        red = int(value[0:2], 16)
        green = int(value[2:4], 16)
        blue = int(value[4:6], 16)
    except ValueError:
        return "#111827"
    return f"#{int(red * 0.45):02x}{int(green * 0.45):02x}{int(blue * 0.45):02x}"


def main(args=None):
    rclpy.init(args=args)
    node = ArenaTkUiNode()
    spin_thread = threading.Thread(target=_spin_node, args=(node,), daemon=True)
    spin_thread.start()
    try:
        ui = ArenaTkUi(node)
        ui.run()
    finally:
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)
        node.destroy_node()


def _spin_node(node: ArenaTkUiNode):
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, RuntimeError):
        pass


if __name__ == "__main__":
    main()
