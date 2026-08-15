#!/usr/bin/env python3
"""관객용 실시간 시각화 서버 — 데모/부스 전용.

왜 별도 프로세스인가
--------------------
`match_runner` 는 ROS 로 아무것도 발행하지 않고 stdout + report.json 만 남긴다.
그래서 "지금 무엇을 왜 하는가"를 화면에 띄울 방법이 없었다. 러너에 `/match/state`
발행을 붙이고(그쪽 `publish_match_state`), 이 스크립트가 그걸 구독해 페이지로 만든다.

`arena_control_node` 안에 들어가지 않는 이유는 그게 제어 루프이기 때문이다 —
관객 화면 때문에 로컬라이저가 굶으면 안 된다. `web_ui.py`(18765) 는 조작자 콘솔이라
용도가 다르고, 경기 중에는 끄는 물건이다. 이건 읽기 전용이고 죽어도 경기는 계속된다.

쓰는 법
-------
    ros2 run 없이 그냥:   python3 mission/spectator_server.py
    다른 포트:            python3 mission/spectator_server.py --port 9000
    로봇 없이 화면만 확인: python3 mission/spectator_server.py --no-ros

브라우저에서 http://<robot-ip>:18080/ — 프로젝터/모니터에 전체화면으로 띄운다.

페이지(`spectator.html`)는 **요청마다 디스크에서 읽는다.** 부스에서 노드를 재시작하지
않고 배치나 색을 손보려는 것이다.

아레나 기하는 `/match/state` 페이로드에 실려 온다. 화면은 4m/2m 상수를 하나도
갖고 있지 않다 — 저장소에 이미 격자 테이블 사본이 5개 있고, 여섯 번째를 만들지 않는다.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = Path(__file__).resolve().parent / "spectator.html"
ASSETS = Path(__file__).resolve().parent / "assets"
DEFAULT_PORT = 18080
PORT_TRIES = 25
# 이보다 오래 소식이 없으면 화면이 "대기" 상태로 바뀐다. 부스에서 런과 런 사이
# 화면이 멈춘 것처럼 보이지 않게 하려는 것.
STALE_SEC = 5.0


class StateHub:
    """구독 스레드와 HTTP 스레드가 공유하는 최신 상태. dict 교체만 하므로 락 불필요."""

    def __init__(self):
        self.match = None
        self.match_t = 0.0
        self.arena = None
        self.arena_t = 0.0
        self.overlay = None       # 스캔 오버레이 JPEG 바이트
        self.overlay_rev = 0      # 바뀔 때마다 증가 — 브라우저 캐시 무효화용

    def snapshot(self) -> dict:
        now = time.monotonic()
        return {
            "server_time": time.time(),
            "match": self.match,
            "match_age": round(now - self.match_t, 2) if self.match else None,
            "arena": self.arena,
            "arena_age": round(now - self.arena_t, 2) if self.arena else None,
            "overlay_rev": self.overlay_rev,
            "stale_sec": STALE_SEC,
        }


class Handler(BaseHTTPRequestHandler):
    def __init__(self, hub: StateHub, *a, **kw):
        self.hub = hub
        super().__init__(*a, **kw)

    def log_message(self, *_a):
        pass          # 접근 로그는 부스에서 소음일 뿐이다

    def _send(self, code: int, ctype: str, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass      # 관객이 탭을 닫은 것뿐이다

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            try:
                body = PAGE.read_bytes()
            except OSError as exc:
                self._send(500, "text/plain; charset=utf-8",
                           f"spectator.html 을 읽을 수 없음: {exc}".encode())
                return
            self._send(200, "text/html; charset=utf-8", body)
        elif path == "/state.json":
            body = json.dumps(self.hub.snapshot(), ensure_ascii=False,
                              default=str).encode()
            self._send(200, "application/json; charset=utf-8", body)
        elif path.startswith("/assets/"):
            # 로고 등 정적 자산. 경로 조작 방지를 위해 파일명만 취한다.
            name = Path(path).name
            f = ASSETS / name
            if f.suffix.lower() not in (".png", ".jpg", ".svg") or not f.is_file():
                self._send(404, "text/plain; charset=utf-8", b"not found")
            else:
                ctype = {"png": "image/png", "jpg": "image/jpeg",
                         "svg": "image/svg+xml"}[f.suffix.lower().lstrip(".")]
                self._send(200, ctype, f.read_bytes())
        elif path == "/scan_overlay.jpg":
            blob = self.hub.overlay
            if not blob:
                self._send(404, "text/plain; charset=utf-8", b"no overlay yet")
            else:
                self._send(200, "image/jpeg", blob)
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found")


def serve(hub: StateHub, host: str, port: int) -> ThreadingHTTPServer:
    """포트가 물려 있으면 port..port+24 로 밀어 본다 (web_ui.py 와 같은 관례)."""
    last = None
    for p in range(port, port + PORT_TRIES):
        try:
            httpd = ThreadingHTTPServer((host, p), partial(Handler, hub))
        except OSError as exc:
            last = exc
            continue
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        shown = "127.0.0.1" if host in ("", "0.0.0.0") else host
        print(f"[spectator] http://{shown}:{p}/  (전체화면 권장)", flush=True)
        return httpd
    raise SystemExit(f"[spectator] 포트 {port}..{port + PORT_TRIES - 1} 전부 사용 중: {last}")


def run_ros(hub: StateHub):
    """`/match/state` 와 `/arena_lightweight/status` 를 받아 hub 에 꽂는다."""
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import String

    class SpectatorNode(Node):
        def __init__(self):
            super().__init__("spectator_server")
            self.create_subscription(String, "/match/state", self.on_match, 10)
            self.create_subscription(String, "/arena_lightweight/status",
                                     self.on_arena, 10)
            # 경기당 1장(단발 스캔). depth=1 — 지난 장을 큐에 쌓아둘 이유가 없다.
            self.create_subscription(CompressedImage, "/match/scan_overlay",
                                     self.on_overlay, 1)

        def on_overlay(self, msg):
            hub.overlay = bytes(msg.data)
            hub.overlay_rev += 1

        def on_match(self, msg):
            try:
                hub.match = json.loads(msg.data)
                hub.match_t = time.monotonic()
            except (ValueError, TypeError):
                pass      # 잘린 JSON 한 프레임은 그냥 버린다

        def on_arena(self, msg):
            # 필요한 것만 추린다. status 원문에는 debug_odom·perception·
            # scan_sector_ranges 등이 다 들어 있는데, 10Hz 로 폴링하는 화면에
            # 통째로 넘길 이유가 없다 (전송량과 직렬화 비용 둘 다 줄인다).
            try:
                s = json.loads(msg.data)
            except (ValueError, TypeError):
                return
            loc = s.get("localization") or {}
            imu = s.get("imu_prior") or {}
            hub.arena = {
                "pose": s.get("pose"),
                "loc": {k: loc.get(k) for k in
                        ("success", "score", "latency_ms", "beam_count",
                         "candidate_count", "reason")} if loc else None,
                "imu": {k: imu.get(k) for k in
                        ("fresh", "gate_rejects", "yaw_integrated")} if imu else {},
                "command_phase": s.get("command_phase"),
            }
            hub.arena_t = time.monotonic()

    rclpy.init()
    node = SpectatorNode()
    print("[spectator] 구독: /match/state, /arena_lightweight/status", flush=True)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser(description="DDONGGAE 관객용 실시간 시각화")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-ros", action="store_true",
                    help="ROS 없이 서버만 — 화면 배치/색 확인용")
    args = ap.parse_args()

    hub = StateHub()
    serve(hub, args.host, args.port)
    if args.no_ros:
        print("[spectator] --no-ros: 데이터 없이 대기 화면만 뜬다", flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            return
    try:
        run_ros(hub)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
