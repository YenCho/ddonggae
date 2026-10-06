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
    화면에서 경기 시작:    python3 mission/spectator_server.py --control

`--control` 은 대기 화면에 타깃 선택 + [경기 시작] 을 띄우고 `/start` 를 연다.
**기본은 꺼져 있다** — 이 화면은 위 문단대로 읽기 전용이 계약이고, tailnet 에
붙은 아무 단말이나 로봇을 출발시킬 수 있는 상태를 기본값으로 둘 수 없다.
켜도 러너는 `start_new_session` 으로 떼어 띄우므로 이 서버가 죽어도 경기는 계속된다.

브라우저에서 http://<robot-ip>:18080/ — 프로젝터/모니터에 전체화면으로 띄운다.

페이지(`spectator.html`)는 **요청마다 디스크에서 읽는다.** 부스에서 노드를 재시작하지
않고 배치나 색을 손보려는 것이다.

아레나 기하는 `/match/state` 페이로드에 실려 온다. 화면은 4m/2m 상수를 하나도
갖고 있지 않다 — 저장소에 이미 격자 테이블 사본이 5개 있고, 여섯 번째를 만들지 않는다.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = Path(__file__).resolve().parent / "spectator.html"
ASSETS = Path(__file__).resolve().parent / "assets"
RUNNER = Path(__file__).resolve().parent / "match_runner.py"
REPO = RUNNER.parent.parent
LOG_DIR = REPO / "logs" / "spectator_launch"
DEFAULT_PORT = 18080
PORT_TRIES = 25
# 화면에 버튼을 그리기 위한 목록일 뿐이다. 진짜 검증은 러너의 argparse 가 한다
# (`--target-shape`/`--target-fruit` 의 choices). 여기와 어긋나면 기동이 실패하고
# 그 stderr 가 로그에 남는다 — 목록을 두 곳에서 맞춰야 하는 부담을 지지 않으려고
# 일부러 이쪽을 "표시용"으로만 둔다.
SHAPES = ("cube", "plain", "octahedron", "dodecahedron", "icosahedron")
FRUITS = ("apple", "banana", "orange", "pineapple")
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


class RunnerCtl:
    """`match_runner.py` 프로세스 하나를 붙잡는다 (`--control` 일 때만).

    한 개만 허용하는 이유: 두 러너가 동시에 뜨면 같은 로봇에 두 갈래 주행 명령이
    흐른다. 브리지는 둘 다 받아주므로 아무도 못 막는다.

    import 하지 않고 별도 프로세스로 띄우는 이유는 모듈 docstring 의 계약과 같다 —
    이 화면이 죽어도 경기는 계속되어야 한다. `start_new_session=True` 가 그
    보장이다 (서버를 Ctrl-C 해도 러너는 산다).
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.proc = None
        self.args = None
        self.started_at = None
        self.log_path = None
        self._lock = threading.Lock()

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def status(self) -> dict:
        alive = self.running()
        return {
            "enabled": self.enabled,
            "running": alive,
            "pid": self.proc.pid if self.proc and alive else None,
            "args": self.args,
            "elapsed_sec": (round(time.time() - self.started_at, 1)
                            if self.started_at and alive else None),
            "exit_code": self.proc.poll() if self.proc and not alive else None,
            "log": str(self.log_path) if self.log_path else None,
        }

    def start(self, shape: str, fruit: str):
        if not self.enabled:
            return False, "제어가 꺼져 있다 — --control 없이 띄운 서버다"
        if shape not in SHAPES:
            return False, f"알 수 없는 형상: {shape!r}"
        if fruit not in FRUITS:
            return False, f"알 수 없는 과일: {fruit!r}"
        with self._lock:
            if self.running():
                return False, "이미 경기가 돌고 있다"
            # 쿼터 1/1 — 클래스당 물체가 하나씩 놓인 배치다. 룰북 기본(형상 4 /
            # 과일 3)을 그대로 쓰면 러너가 있지도 않은 나머지를 계속 기다린다.
            cmd = [sys.executable, "-u", str(RUNNER), "--yes",
                   "--target-shape", shape, "--target-fruit", fruit,
                   "--shape-quota", "1", "--fruit-quota", "1"]
            try:
                LOG_DIR.mkdir(parents=True, exist_ok=True)
                self.log_path = LOG_DIR / (time.strftime("%Y%m%d_%H%M%S")
                                           + "_match.log")
                # 로그 핸들은 자식이 물고 있으므로 여기서 닫아도 된다.
                with open(self.log_path, "wb") as fh:
                    self.proc = subprocess.Popen(
                        cmd, cwd=str(REPO), stdout=fh,
                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                        start_new_session=True)
            except OSError as exc:
                return False, f"기동 실패: {exc}"
            self.args = {"shape": shape, "fruit": fruit}
            self.started_at = time.time()
            print(f"[spectator] 경기 시작 — 형상 {shape} / 과일 {fruit} "
                  f"(pid {self.proc.pid}) 로그 {self.log_path}", flush=True)
            return True, f"시작 — {shape} / {fruit}"


class Handler(BaseHTTPRequestHandler):
    def __init__(self, hub: StateHub, ctl: RunnerCtl, *a, **kw):
        self.hub = hub
        self.ctl = ctl
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

    def _json(self, code: int, payload: dict):
        self._send(code, "application/json; charset=utf-8",
                   json.dumps(payload, ensure_ascii=False, default=str).encode())

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path != "/start":
            self._send(404, "text/plain; charset=utf-8", b"not found")
            return
        if not self.ctl.enabled:
            # 제어가 꺼진 서버는 이 경로의 존재 자체를 알릴 이유가 없다.
            self._send(404, "text/plain; charset=utf-8", b"not found")
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(n) or b"{}")
            shape = str(payload.get("shape") or "")
            fruit = str(payload.get("fruit") or "")
        except (ValueError, TypeError, AttributeError):
            self._json(400, {"ok": False, "msg": "잘못된 요청"})
            return
        ok, msg = self.ctl.start(shape, fruit)
        self._json(200 if ok else 409,
                   {"ok": ok, "msg": msg, "runner": self.ctl.status()})

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/options.json":
            self._json(200, {"shapes": list(SHAPES), "fruits": list(FRUITS),
                             "control": self.ctl.enabled})
        elif path in ("/", "/index.html"):
            try:
                body = PAGE.read_bytes()
            except OSError as exc:
                self._send(500, "text/plain; charset=utf-8",
                           f"spectator.html 을 읽을 수 없음: {exc}".encode())
                return
            self._send(200, "text/html; charset=utf-8", body)
        elif path == "/state.json":
            snap = self.hub.snapshot()
            snap["runner"] = self.ctl.status()
            body = json.dumps(snap, ensure_ascii=False, default=str).encode()
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


def serve(hub: StateHub, ctl: RunnerCtl, host: str, port: int) -> ThreadingHTTPServer:
    """포트가 물려 있으면 port..port+24 로 밀어 본다 (web_ui.py 와 같은 관례)."""
    last = None
    for p in range(port, port + PORT_TRIES):
        try:
            httpd = ThreadingHTTPServer((host, p), partial(Handler, hub, ctl))
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
    ap.add_argument("--control", action="store_true",
                    help="대기 화면에 타깃 선택 + [경기 시작] 을 띄운다 "
                         "(기본 꺼짐 — 관객 화면은 읽기 전용이 계약이다)")
    args = ap.parse_args()

    hub = StateHub()
    ctl = RunnerCtl(args.control)
    serve(hub, ctl, args.host, args.port)
    if args.control:
        print("[spectator] --control: 화면에서 경기를 시작할 수 있다 "
              "(러너 쿼터 1/1 고정)", flush=True)
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
