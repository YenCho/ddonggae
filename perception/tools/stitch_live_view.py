#!/usr/bin/env python3
"""실시간 스티치 뷰어 — 두 카메라 합성 결과를 창으로 송출.

  DISPLAY=:0 python3 scripts/dev/field_ops/stitch_live_view.py [--mast up]

- 빨간선 = 이음매(seam). 선 위아래 바닥 무늬가 이어져 보이면 기하 정렬 OK,
  밝기·색이 이어져 보이면 측광 OK.
- 키: [u] up 캘리브 / [d] down 캘리브 전환(마스트 움직일 때),
      [s] 현재 프레임 PNG 저장, [f] 전체화면 토글, [q]/ESC 종료.
- 캘리브는 e2e 와 동일한 `_fieldlib.Stitcher` (data/calibration/stitch/*.json)
  를 쓰므로, 여기서 보이는 그대로가 A1 검출 입력이다.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fieldlib as fl  # noqa: E402

import rclpy  # noqa: E402
from rcl_interfaces.msg import Parameter as ParamMsg  # noqa: E402
from rcl_interfaces.msg import ParameterType, ParameterValue  # noqa: E402
from rcl_interfaces.srv import GetParameters, SetParameters  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402

WIN = "stitch live"
CAM_NODES = ("/camera/top/top", "/camera/bottom/bottom")


def _param(name, value, double=False):
    # white_balance 는 realsense 쪽 선언이 double — int 로 보내면 거부된다
    t = ParameterType.PARAMETER_DOUBLE if double else ParameterType.PARAMETER_INTEGER
    pv = ParameterValue(type=t)
    if double:
        pv.double_value = float(value)
    else:
        pv.integer_value = int(value)
    return ParamMsg(name=name, value=pv)


def _read_photometry(node, timeout=3.0):
    """top 캠의 현재 WB/노출/게인을 슬라이더 초기값으로 (실패 시 런치 기본값)."""
    fallback = (4600, 156, 64)
    cli = node.create_client(GetParameters, f"{CAM_NODES[0]}/get_parameters")
    if not cli.wait_for_service(timeout_sec=timeout):
        return fallback
    fut = cli.call_async(GetParameters.Request(
        names=["rgb_camera.white_balance", "rgb_camera.exposure", "rgb_camera.gain"]))
    deadline = time.time() + timeout
    while not fut.done() and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not fut.done() or fut.result() is None:
        return fallback
    vals = [pv.double_value if pv.type == ParameterType.PARAMETER_DOUBLE
            else pv.integer_value for pv in fut.result().values]
    return tuple(vals) if len(vals) == 3 else fallback


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mast", choices=("up", "down"), default="up")
    ap.add_argument("--scale", type=float, default=1.0, help="표시 배율")
    ap.add_argument("--settle", type=float, default=2.0,
                    help="[a] 스윕에서 측광 설정이 프레임에 반영될 때까지 대기(초)")
    args = ap.parse_args()

    rclpy.init()
    node = Node("stitch_live_view")
    frames = {}

    def cb(name):
        def _cb(msg):
            img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
            # 표시용 BGR (bgr8 이면 그대로, rgb8 이면 뒤집기)
            frames[name] = img.copy() if msg.encoding == "bgr8" else img[:, :, ::-1].copy()
        return _cb

    node.create_subscription(Image, "/camera_19/rgb", cb("top"), qos_profile_sensor_data)
    node.create_subscription(Image, "/camera_54/rgb", cb("near"), qos_profile_sensor_data)

    mast = args.mast
    st = None                     # 첫 프레임 도착 후 해상도에 맞춰 생성
    print(f"[stitch live] mast={mast}  (프레임 대기)")
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    full = False
    n, t0, fps = 0, time.time(), 0.0

    # ── 측광 슬라이더: 두 캠에 동일 값 실시간 반영 (런치가 auto WB/노출을 꺼둔 전제)
    set_clients = [node.create_client(SetParameters, f"{c}/set_parameters")
                   for c in CAM_NODES]
    photo = dict(zip(("wb", "exp", "gain"), _read_photometry(node)))
    photo0 = (int(photo["wb"]), int(photo["exp"]), int(photo["gain"]))  # 종료 시 복원용
    pending = {"changed": False, "sent": 0.0}

    def _e2e_running():
        try:
            return any("e2e_match" in n for n in node.get_node_names())
        except Exception:
            return False

    def _slider(key, offset=0):
        def _cb(v):
            photo[key] = v + offset
            pending["changed"] = True
        return _cb

    cv2.createTrackbar("WB +2800K", WIN, int(photo["wb"]) - 2800, 3700, _slider("wb", 2800))
    cv2.createTrackbar("exposure", WIN, int(photo["exp"]), 1000, _slider("exp"))
    cv2.createTrackbar("gain", WIN, int(photo["gain"]), 128, _slider("gain"))

    def _send_photometry(wb, exp, gain):
        if _e2e_running():
            print("[측광] e2e_match_test 실행 중 — 카메라 파라미터 전송 차단 (경기 보호)")
            return False
        req = SetParameters.Request(parameters=[
            _param("rgb_camera.white_balance", wb, double=True),
            _param("rgb_camera.exposure", max(int(exp), 1)),
            _param("rgb_camera.gain", int(gain)),
        ])
        for cli in set_clients:
            if cli.service_is_ready():
                cli.call_async(req)
        return True

    def _auto_sweep():
        """[a] 자동 촬영: 현재값 주변 WB5×노출5×게인5 = 125조합, 조합마다 --settle초 대기 후 저장."""
        if _e2e_running():
            print("[sweep] e2e_match_test 실행 중 — 스윕 거부 (경기 측광 보호)")
            return
        base = (int(photo["wb"]), int(photo["exp"]), int(photo["gain"]))
        wbs = [min(max(base[0] + d, 2800), 6500) for d in (-600, -300, 0, 300, 600)]
        exps = [min(max(round(base[1] * m), 1), 1000) for m in (0.5, 0.75, 1.0, 1.5, 2.0)]
        gains = [min(max(base[2] + d, 0), 128) for d in (-24, -12, 0, 12, 24)]
        combos = [(w, e, g) for w in wbs for e in exps for g in gains]
        out_dir = (fl.REPO_ROOT / "logs" / "real_validation"
                   / f"stitch_sweep_{datetime.now():%Y%m%d_%H%M%S}")
        out_dir.mkdir(parents=True, exist_ok=True)
        pending["changed"] = False          # 스윕 중 슬라이더 잔여 전송 무효화
        print(f"[sweep] {len(combos)}장 시작 → {out_dir}")
        aborted = False
        for i, (w, e, g) in enumerate(combos):
            _send_photometry(w, e, g)
            t_end, last_draw = time.time() + args.settle, 0.0
            while time.time() < t_end and not aborted:   # 설정이 프레임에 반영될 때까지 대기
                rclpy.spin_once(node, timeout_sec=0.03)
                if time.time() - last_draw > 0.15:
                    disp = st.stitch(frames["top"], frames["near"])
                    if args.scale != 1.0:
                        disp = cv2.resize(disp, None, fx=args.scale, fy=args.scale)
                    cv2.putText(disp,
                                f"SWEEP {i + 1}/{len(combos)}  wb {w}K exp {e} gain {g}  [q]abort",
                                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                    cv2.imshow(WIN, disp)
                    last_draw = time.time()
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    aborted = True
            if aborted:
                break
            shot = st.stitch(frames["top"], frames["near"])  # 대기 후 최신 프레임, 오버레이 없음
            name = f"sweep_{i:03d}_wb{w}_exp{e}_gain{g}"
            cv2.imwrite(str(out_dir / f"{name}.png"), shot)
            (out_dir / f"{name}.json").write_text(json.dumps({
                "png": f"{name}.png", "mast": mast,
                "input_res": [frames["top"].shape[1], frames["top"].shape[0]],
                "wb": w, "exposure": e, "gain": g,
                "stamp": datetime.now().isoformat(timespec="seconds"),
            }, indent=1) + "\n")
            print(f"[sweep] {i + 1}/{len(combos)} {name}.png")
        _send_photometry(*base)                              # 슬라이더 값 복원
        print(f"[sweep] {'중단됨' if aborted else '완료'} — 측광 복원 wb/exp/gain={base}")

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if "top" not in frames or "near" not in frames:
                if time.time() - t0 > 10 and n == 0:
                    sys.exit("카메라 프레임 없음 — 브릿지 확인 (arena-bringup)")
                continue
            if st is None:
                f = frames["top"]
                st = fl.Stitcher(mast, w=f.shape[1], h=f.shape[0])
                print(f"[stitch live] {f.shape[1]}x{f.shape[0]}  calib={st.calib_source}")
            vis = st.stitch(frames["top"], frames["near"])
            cv2.line(vis, (0, st.seam), (vis.shape[1], st.seam), (0, 0, 255), 1)
            n += 1
            if n % 10 == 0:
                now = time.time()
                fps = 10.0 / max(now - t0, 1e-6)
                t0 = now
            cv2.putText(vis, f"mast {mast}  {fps:4.1f}fps  [u/d]calib [s]save [a]sweep125 [f]full [q]quit",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(vis, f"WB {int(photo['wb'])}K  exp {int(photo['exp'])}  gain {int(photo['gain'])}",
                        (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            vis_full = vis                # [s] 저장용 원본 해상도 (표시 축소와 무관)
            if args.scale != 1.0:
                vis = cv2.resize(vis, None, fx=args.scale, fy=args.scale)
            cv2.imshow(WIN, vis)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k in (ord("u"), ord("d")):
                mast = "up" if k == ord("u") else "down"
                f = frames["top"]
                st = fl.Stitcher(mast, w=f.shape[1], h=f.shape[0])
                print(f"[stitch live] mast={mast}  calib={st.calib_source}")
            if k == ord("s"):
                now = datetime.now()
                out = (fl.REPO_ROOT / "logs" / "real_validation"
                       / f"stitch_live_{now:%Y%m%d_%H%M%S}.png")
                cv2.imwrite(str(out), vis_full)
                meta = {
                    "png": out.name,
                    "mast": mast,
                    "input_res": [frames["top"].shape[1], frames["top"].shape[0]],
                    "wb": int(photo["wb"]),
                    "exposure": int(photo["exp"]),
                    "gain": int(photo["gain"]),
                    "stamp": now.isoformat(timespec="seconds"),
                }
                out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
                print(f"저장: {out}  (wb {meta['wb']}K / exp {meta['exposure']} / gain {meta['gain']})")
            if k == ord("f"):
                full = not full
                cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)
            if k == ord("a"):
                _auto_sweep()
            if pending["changed"] and time.time() - pending["sent"] > 0.2:
                _send_photometry(photo["wb"], photo["exp"], photo["gain"])
                pending.update(changed=False, sent=time.time())
    finally:
        try:
            cur = (int(photo["wb"]), int(photo["exp"]), int(photo["gain"]))
            if cur != photo0:
                # 뷰어에서 만진 측광이 이후 E2E 에 새지 않도록 시작 시점 값으로 복원.
                # 좋은 값을 찾았다면 [s] 저장 JSON 을 보고 런치 인자에 옮길 것.
                if _send_photometry(*photo0):
                    print(f"[측광] 종료 복원 {cur} → {photo0}")
                time.sleep(0.3)   # 비동기 서비스 요청이 나갈 시간
        except Exception as e:    # 외부 kill 로 rclpy 가 먼저 닫힌 경우 등
            print(f"[측광] 종료 복원 실패: {e}")
        cv2.destroyAllWindows()
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
