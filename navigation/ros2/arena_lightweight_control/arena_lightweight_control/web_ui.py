#!/usr/bin/env python3
import errno
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Arena Control</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Arial, Helvetica, sans-serif;
      background: #f4f7f8;
      color: #1d2528;
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(420px, 1fr) 320px;
    }
    #map {
      width: 100%;
      height: 100vh;
      display: block;
      background: #dfe7e9;
      cursor: crosshair;
    }
    aside {
      border-left: 1px solid #c8d1d4;
      padding: 16px;
      background: #ffffff;
      overflow: auto;
    }
    h1 {
      font-size: 20px;
      margin: 0 0 14px;
      letter-spacing: 0;
    }
    .row {
      display: flex;
      gap: 8px;
      margin-bottom: 10px;
    }
    button {
      min-height: 38px;
      border: 1px solid #9fb0b5;
      border-radius: 6px;
      background: #f9fbfb;
      color: #1d2528;
      font-weight: 600;
      cursor: pointer;
      flex: 1;
    }
    button.active {
      background: #1f7a6d;
      color: #ffffff;
      border-color: #1f7a6d;
    }
    button.stop {
      background: #b93830;
      color: #ffffff;
      border-color: #b93830;
    }
    .metric {
      display: grid;
      grid-template-columns: 120px 1fr;
      gap: 8px;
      padding: 5px 0;
      font-size: 13px;
      border-bottom: 1px solid #edf1f2;
    }
    .metric span:first-child {
      color: #5f6f74;
    }
    pre {
      white-space: pre-wrap;
      background: #f4f7f8;
      border: 1px solid #d7e0e2;
      border-radius: 6px;
      padding: 10px;
      font-size: 12px;
      line-height: 1.35;
      max-height: 180px;
      overflow: auto;
    }
    @media (max-width: 760px) {
      body {
        grid-template-columns: 1fr;
        grid-template-rows: 70vh auto;
      }
      #map {
        height: 70vh;
      }
      aside {
        border-left: 0;
        border-top: 1px solid #c8d1d4;
      }
    }
  </style>
</head>
<body>
  <canvas id="map"></canvas>
  <aside>
    <h1>Arena Control</h1>
    <div class="row">
      <button id="goalMode" class="active">Goal</button>
      <button id="poseMode">Pose</button>
    </div>
    <div class="row">
      <button class="stop" id="stop">Stop</button>
    </div>
    <div class="row">
      <button data-gripper="OPEN">Open</button>
      <button data-gripper="CLOSE">Close</button>
      <button data-gripper="STOP">Stop</button>
    </div>
    <div class="row">
      <button data-gripper="LIFT_TO_TOP">Mast &#9650;</button>
      <button data-gripper="LIFT_TO_BOTTOM">Mast &#9660;</button>
      <button data-gripper="LIFT_STOP">Mast Stop</button>
    </div>
    <div class="row">
      <button data-gripper="DXL_POWER_ON">Power</button>
      <button data-gripper="DXL_POWER_CYCLE">Recover</button>
    </div>
    <div id="metrics"></div>
    <pre id="status">waiting</pre>
  </aside>
  <script>
    const canvas = document.getElementById('map');
    const ctx = canvas.getContext('2d');
    const metrics = document.getElementById('metrics');
    const status = document.getElementById('status');
    const goalMode = document.getElementById('goalMode');
    const poseMode = document.getElementById('poseMode');
    let mode = 'goal';
    let mapInfo = null;
    let mapCanvas = null;
    let latestState = {};

    function postJson(path, body) {
      return fetch(path, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body || {})
      });
    }

    function setMode(next) {
      mode = next;
      goalMode.classList.toggle('active', mode === 'goal');
      poseMode.classList.toggle('active', mode === 'pose');
    }

    goalMode.onclick = () => setMode('goal');
    poseMode.onclick = () => setMode('pose');
    document.getElementById('stop').onclick = () => postJson('/stop', {});
    document.querySelectorAll('[data-gripper]').forEach((button) => {
      button.onclick = () => postJson('/gripper', {command: button.dataset.gripper});
    });

    function resize() {
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(320, Math.floor(rect.width * devicePixelRatio));
      canvas.height = Math.max(280, Math.floor(rect.height * devicePixelRatio));
      draw();
    }

    function worldToCanvas(x, y, scale, offsetX, offsetY) {
      const mx = (x - mapInfo.origin[0]) / mapInfo.resolution;
      const my = (y - mapInfo.origin[1]) / mapInfo.resolution;
      return {
        x: offsetX + mx * scale,
        y: offsetY + (mapInfo.height - my) * scale
      };
    }

    function canvasToWorld(px, py, scale, offsetX, offsetY) {
      const mx = (px - offsetX) / scale;
      const my = mapInfo.height - ((py - offsetY) / scale);
      return {
        x: mapInfo.origin[0] + mx * mapInfo.resolution,
        y: mapInfo.origin[1] + my * mapInfo.resolution
      };
    }

    function layout() {
      const scale = Math.min(canvas.width / mapInfo.width, canvas.height / mapInfo.height);
      const offsetX = (canvas.width - mapInfo.width * scale) * 0.5;
      const offsetY = (canvas.height - mapInfo.height * scale) * 0.5;
      return {scale, offsetX, offsetY};
    }

    canvas.addEventListener('click', (event) => {
      if (!mapInfo) return;
      const rect = canvas.getBoundingClientRect();
      const px = (event.clientX - rect.left) * devicePixelRatio;
      const py = (event.clientY - rect.top) * devicePixelRatio;
      const view = layout();
      const point = canvasToWorld(px, py, view.scale, view.offsetX, view.offsetY);
      if (mode === 'pose') {
        postJson('/pose', {x: point.x, y: point.y, yaw: latestState.pose?.yaw || 0});
      } else {
        postJson('/goal', {x: point.x, y: point.y});
      }
    });

    function drawRobot(pose, view, color) {
      const p = worldToCanvas(pose.x, pose.y, view.scale, view.offsetX, view.offsetY);
      const yaw = -pose.yaw;
      const size = Math.max(10, 0.16 / mapInfo.resolution * view.scale);
      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.rotate(yaw);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(size, 0);
      ctx.lineTo(-size * 0.65, -size * 0.55);
      ctx.lineTo(-size * 0.65, size * 0.55);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }

    function drawGoal(goal, view) {
      const p = worldToCanvas(goal.x, goal.y, view.scale, view.offsetX, view.offsetY);
      ctx.strokeStyle = '#bf7b00';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 10, 0, Math.PI * 2);
      ctx.moveTo(p.x - 14, p.y);
      ctx.lineTo(p.x + 14, p.y);
      ctx.moveTo(p.x, p.y - 14);
      ctx.lineTo(p.x, p.y + 14);
      ctx.stroke();
    }

    // 경기 공식 레이아웃 — competition_layout.py 에서 주입된다 (아래 _COMP_JSON).
    // 종전에는 격자 루프와 구역 좌표를 여기 JS 에 하드코딩했는데, 저장소에
    // 격자 테이블 사본이 이미 여러 개라 아레나 크기를 바꾸면 이 화면만 조용히
    // 옛 배치를 그리게 된다. 사본을 늘리지 않는다.
    // 스캔점은 아레나 정중앙이 아니라 공식 (225,225)cm = map (0.25,0.25).
    // 격자점이 50cm 간격이라 정중앙에는 물체가 서고, 스캔은 그 사이 street
    // 교차점에서 한다 — 실기 로그의 "[GOTO_CENTER] ... → (+0.25,+0.25)".
    const COMP = __COMP_JSON__;

    function drawCompetitionLayout(view) {
      const zones = [
        ['STORAGE', COMP.storage, '#2e7d32'],
        ['START', COMP.start, '#1565c0']
      ];
      ctx.lineWidth = 2;
      ctx.font = 'bold 11px sans-serif';
      ctx.textAlign = 'center';
      for (const [name, r, color] of zones) {
        const a = worldToCanvas(r[0], r[1], view.scale, view.offsetX, view.offsetY);
        const b = worldToCanvas(r[2], r[3], view.scale, view.offsetX, view.offsetY);
        ctx.strokeStyle = color;
        ctx.strokeRect(Math.min(a.x, b.x), Math.min(a.y, b.y), Math.abs(b.x - a.x), Math.abs(b.y - a.y));
        ctx.fillStyle = color;
        ctx.fillText(name, (a.x + b.x) / 2, (a.y + b.y) / 2 + 4);
      }
      for (const [gx, gy] of COMP.grid) {
        const p = worldToCanvas(gx, gy, view.scale, view.offsetX, view.offsetY);
        ctx.strokeStyle = '#8e24aa';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
        ctx.stroke();
        ctx.strokeStyle = '#ce93d8';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(p.x - 7, p.y); ctx.lineTo(p.x + 7, p.y);
        ctx.moveTo(p.x, p.y - 7); ctx.lineTo(p.x, p.y + 7);
        ctx.stroke();
      }
      const c = worldToCanvas(COMP.center[0], COMP.center[1], view.scale, view.offsetX, view.offsetY);
      ctx.strokeStyle = '#e65100';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.arc(c.x, c.y, 10, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#e65100';
      ctx.fillText('SCAN', c.x, c.y - 14);
    }

    function draw() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (!mapInfo || !mapCanvas) return;
      const view = layout();
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(
        mapCanvas,
        view.offsetX,
        view.offsetY,
        mapInfo.width * view.scale,
        mapInfo.height * view.scale
      );
      drawCompetitionLayout(view);
      if (latestState.goal) drawGoal(latestState.goal, view);
      if (latestState.pose) drawRobot(latestState.pose, view, '#1f7a6d');
    }

    function updateMetrics(state) {
      const sectors = state.scan_sector_ranges || {};
      const fmtRange = (value) => value == null ? 'none' : `${Number(value).toFixed(2)} m`;
      const rows = [
        ['pose', state.pose ? `${state.pose.x.toFixed(2)}, ${state.pose.y.toFixed(2)}, ${state.pose.yaw.toFixed(2)}` : 'none'],
        ['goal', state.goal ? `${state.goal.x.toFixed(2)}, ${state.goal.y.toFixed(2)}` : 'none'],
        ['phase', state.command_phase || 'idle'],
        ['match', state.localization ? `${state.localization.latency_ms.toFixed(1)} ms / ${state.localization.score.toFixed(3)}` : 'none'],
        ['front', fmtRange(state.obstacle_front_m)],
        ['left', fmtRange(sectors.left)],
        ['back', fmtRange(sectors.back)],
        ['right', fmtRange(sectors.right)],
        ['gripper', state.gripper_status || 'none']
      ];
      metrics.innerHTML = rows.map(([name, value]) => `<div class="metric"><span>${name}</span><span>${value}</span></div>`).join('');
      status.textContent = JSON.stringify(state, null, 2);
    }

    async function loadMap() {
      mapInfo = await (await fetch('/map.json')).json();
      mapCanvas = document.createElement('canvas');
      mapCanvas.width = mapInfo.width;
      mapCanvas.height = mapInfo.height;
      const mctx = mapCanvas.getContext('2d');
      const image = mctx.createImageData(mapInfo.width, mapInfo.height);
      for (let i = 0; i < mapInfo.occupied.length; i++) {
        const value = mapInfo.occupied[i] ? 38 : 235;
        image.data[i * 4 + 0] = value;
        image.data[i * 4 + 1] = value;
        image.data[i * 4 + 2] = value;
        image.data[i * 4 + 3] = 255;
      }
      mctx.putImageData(image, 0, 0);
      draw();
    }

    async function poll() {
      try {
        latestState = await (await fetch('/state.json', {cache: 'no-store'})).json();
        updateMetrics(latestState);
        draw();
      } catch (err) {
        status.textContent = String(err);
      }
      setTimeout(poll, 100);
    }

    window.addEventListener('resize', resize);
    loadMap().then(() => { resize(); poll(); });
  </script>
</body>
</html>
"""


def _competition_layout_json() -> str:
    """`competition_layout` 의 상수를 페이지가 쓰는 모양으로 직렬화한다.

    import 시점에 한 번만 치환하므로 요청마다 드는 비용은 없다. 격자 순서는
    `GRID_POINTS_MAP` 그대로 (y 바깥, x 안쪽) — 종전 JS 이중 루프와 동일하다.
    """
    from . import competition_layout as layout

    return json.dumps({
        "grid": [list(p) for p in layout.GRID_POINTS_MAP],
        "storage": list(layout.STORAGE_RECT_MAP),
        "start": list(layout.START_RECT_MAP),
        "center": list(layout.CENTER_SCAN_MAP),
    })


INDEX_HTML = INDEX_HTML.replace("__COMP_JSON__", _competition_layout_json())


class ArenaWebServer:
    def __init__(
        self,
        host: str,
        port: int,
        map_provider: Callable[[], dict],
        state_provider: Callable[[], dict],
        goal_callback: Callable[[float, float, Optional[float]], None],
        pose_callback: Callable[[float, float, float], None],
        stop_callback: Callable[[], None],
        gripper_callback: Callable[[str], None],
    ):
        self.host = host
        self.port = int(port)
        self.map_provider = map_provider
        self.state_provider = state_provider
        self.goal_callback = goal_callback
        self.pose_callback = pose_callback
        self.stop_callback = stop_callback
        self.gripper_callback = gripper_callback
        self.httpd = None
        self.thread = None
        self.actual_host = host
        self.actual_port = int(port)
        self.port_fallback_count = 25

    def start(self):
        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                if self.path in {"/", "/index.html"}:
                    self._send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
                elif self.path == "/state.json":
                    self._send_json(server_ref.state_provider())
                elif self.path == "/map.json":
                    self._send_json(server_ref.map_provider())
                else:
                    self.send_error(404)

            def do_POST(self):  # noqa: N802
                try:
                    body = self._read_json()
                    if self.path == "/goal":
                        yaw = body.get("yaw")
                        server_ref.goal_callback(
                            float(body["x"]),
                            float(body["y"]),
                            None if yaw is None else float(yaw),
                        )
                    elif self.path == "/pose":
                        server_ref.pose_callback(
                            float(body["x"]),
                            float(body["y"]),
                            float(body.get("yaw", 0.0)),
                        )
                    elif self.path == "/stop":
                        server_ref.stop_callback()
                    elif self.path == "/gripper":
                        server_ref.gripper_callback(str(body["command"]))
                    else:
                        self.send_error(404)
                        return
                    self._send_json({"ok": True})
                except (KeyError, TypeError, ValueError) as exc:
                    self.send_error(400, str(exc))

            def log_message(self, _format, *args):
                return

            def _read_json(self):
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    return {}
                return json.loads(self.rfile.read(length).decode("utf-8"))

            def _send_json(self, payload):
                self._send_bytes(
                    json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                    "application/json",
                )

            def _send_bytes(self, payload: bytes, content_type: str):
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)

        self.httpd = self._bind_server(Handler)
        self.actual_host, self.actual_port = self.httpd.server_address[:2]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def local_url(self) -> str:
        host = self.actual_host
        if host in {"", "0.0.0.0"}:
            host = "127.0.0.1"
        elif ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"http://{host}:{self.actual_port}"

    def _bind_server(self, handler):
        if self.port == 0:
            return ReusableThreadingHTTPServer((self.host, 0), handler)

        last_error = None
        for offset in range(self.port_fallback_count + 1):
            port = self.port + offset
            try:
                return ReusableThreadingHTTPServer((self.host, port), handler)
            except OSError as exc:
                last_error = exc
                if not _is_port_unavailable(exc):
                    raise
        raise last_error

    def stop(self):
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def _is_port_unavailable(exc: OSError) -> bool:
    return exc.errno in {
        errno.EADDRINUSE,
        errno.EACCES,
        getattr(errno, "WSAEADDRINUSE", 10048),
        10013,
    } or (
        isinstance(getattr(exc, "winerror", None), int)
        and exc.winerror in {10013, 10048}
    )
