import socket
import sys
import urllib.error
import urllib.request
import json
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[3] / "src" / "arena_lightweight_control"

from arena_lightweight_control.web_ui import ArenaWebServer  # noqa: E402


def test_web_ui_falls_back_when_requested_port_is_busy():
    busy = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    busy.bind(("127.0.0.1", 0))
    busy.listen(1)
    busy_port = busy.getsockname()[1]

    server = ArenaWebServer(
        "127.0.0.1",
        busy_port,
        map_provider=lambda: {"width": 1, "height": 1, "occupied": [0]},
        state_provider=lambda: {"ok": True},
        goal_callback=lambda _x, _y, _yaw: None,
        pose_callback=lambda _x, _y, _yaw: None,
        stop_callback=lambda: None,
        gripper_callback=lambda _command: None,
    )
    try:
        server.start()
        assert server.actual_port != busy_port
        with urllib.request.urlopen(server.local_url() + "/state.json", timeout=2.0) as response:
            assert response.read() == b'{"ok":true}'
    finally:
        server.stop()
        busy.close()


def test_goal_post_allows_optional_yaw():
    goals = []
    server = ArenaWebServer(
        "127.0.0.1",
        0,
        map_provider=lambda: {"width": 1, "height": 1, "occupied": [0]},
        state_provider=lambda: {"ok": True},
        goal_callback=lambda x, y, yaw: goals.append((x, y, yaw)),
        pose_callback=lambda _x, _y, _yaw: None,
        stop_callback=lambda: None,
        gripper_callback=lambda _command: None,
    )
    try:
        server.start()
        post_json(server.local_url() + "/goal", {"x": 1.0, "y": 2.0})
        post_json(server.local_url() + "/goal", {"x": 3.0, "y": 4.0, "yaw": 0.5})
    finally:
        server.stop()

    assert goals == [(1.0, 2.0, None), (3.0, 4.0, 0.5)]


def post_json(url, payload):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            assert response.read() == b'{"ok":true}'
    except urllib.error.HTTPError as exc:
        raise AssertionError(exc.read().decode("utf-8")) from exc
