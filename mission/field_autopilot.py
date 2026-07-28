#!/usr/bin/env python3
"""현장 무보조(no-Claude) 원커맨드 자동화 — 기동/점검/기록/종료.

조작자 혼자 실기(Jetson)에서 브리지 전부 자동 기동 + 상태 자동 판정 +
ros2 bag / raw 이미지 / depth / YOLO 디버그 이미지 저장까지 수행한다.
물리 상태(배터리 스위치·출발 배치·마스트 위치)는 y/n 프롬프트로 확인하고
(--yes 로 생략), 소프트웨어 상태(브리지 생존·토픽 신선도·localization 락·
브리지 중복 실행)는 자동 판정한다.

★ 운용 원칙 ★ 조작자는 아래 한 줄씩만 실행하고 화면 지시만 따른다.

사용 (Jetson, ROS 환경 소스 후):
  source /opt/ros/humble/setup.bash && source install/setup.bash
  python3 -u mission/field_autopilot.py up      # 기동+점검+포즈시딩
  python3 -u mission/field_autopilot.py check   # 점검만 (기동 안 함)
  python3 -u mission/field_autopilot.py record \
      --gt-text "apple:150,200;plain:250,300"                 # 기록 (Ctrl-C 종료)
  python3 -u mission/field_autopilot.py down    # 전체 종료+잔존 보고

개발기 자가진단 (ROS 불필요):
  python3 mission/field_autopilot.py 대신 아래처럼:
  python3 mission/field_autopilot.py up --offline --yes

산출물 (기본 logs/field_ops/<YYYYmmdd_HHMMSS>_<cmd>/):
  verdict.json                 up/check 자동 판정 (exit 0=정상 / 2=필수 결손)
  frame_<n>_top_rgb.png / _top_depth.png / _near_rgb.png / _near_depth.png
                               raw 쌍 저장 (depth = uint16 mm, PIL I;16)
  yolo_debug_<n>.png           스티치 프레임 A1+face 오버레이 (박스+conf+면투표)
  frames.jsonl                 프레임별 pose / loc_latency_ms / saved / dets
  gt.json / gt_summary.txt     조작자가 입력한 물체 배치 GT
  bag/ + rosbag.log            진단 rosbag (카메라 4토픽 포함)
  stack_launch.log             스택 기동 로그 (up)

계약:
  - YOLO 추론은 항상 스티치 프레임에서 수행 (단일 캠 프레임에 A1 금지).
    A1 imgsz=896 BGR + 큐브 크롭 face imgsz=224 BGR + face_vote(과일면 우선).
  - depth PNG 는 uint16 mm (PIL mode I;16).
  - 리프트/그리퍼는 절대 tty 직접 열지 않음 — 브리지 토픽 경유만
    (DTR 리셋 → 보드 재부팅 → 리프트 홈 소실 방지).
"""
from __future__ import annotations

import argparse
import json
import math
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perception"))
from fieldlib import (  # noqa: E402
    A1_IMGSZ_STITCHED,
    A1_WEIGHTS,
    BridgeManager,
    DEFAULT_CHECKLIST,
    FACE_IMGSZ,
    FACE_WEIGHTS,
    FRUITS,
    FieldNode,
    HEALTH_SPEC,
    REPO_ROOT,
    START_POSE,
    Stitcher,
    battery_off_signature,
    face_vote,
    gt_summary_text,
    health_report_text,
    healthcheck,
    official_cm_to_map,
    operator_checklist,
    prompt_gt,
    rosbag_record,
    snap_cell,
)


def banner(msg: str):
    print(f"== {msg}")


def _sigterm_raise(_sig, _frm):
    raise KeyboardInterrupt


def _checklist(items, assume_yes: bool) -> dict:
    """operator_checklist 래퍼 — non-tty(EOF)에서도 죽지 않게."""
    try:
        return operator_checklist(items, assume_yes=assume_yes)
    except EOFError:
        print("! 입력 스트림 없음(non-tty) — 체크리스트 통과로 간주 (--yes 사용 권장)")
        return {k: True for k, _ in items}


def _ros_node(name: str):
    """rclpy init + FieldNode. ROS 환경 미소스 시 친절히 안내 후 종료."""
    try:
        import rclpy
    except ImportError:
        print("✗ rclpy import 실패 — ROS 환경을 먼저 소스했는지 확인:")
        print("   source /opt/ros/humble/setup.bash && source install/setup.bash")
        print("  (개발기 자가진단은 --offline 옵션 사용)")
        sys.exit(2)
    rclpy.init()
    return rclpy, FieldNode(name)


def _wait_sub(fn: FieldNode, key: str, timeout: float = 10.0) -> bool:
    """DDS 디스커버리 가드 — 발행 전 구독자 존재 대기 (goal 유실 재발 방지)."""
    t0 = time.monotonic()
    while fn.pub[key].get_subscription_count() == 0 and time.monotonic() - t0 < timeout:
        fn.spin_for(0.2)
    return fn.pub[key].get_subscription_count() > 0


def _wait_stack_boot(fn: FieldNode, timeout: float):
    required = [k for k, _d, _w, req in HEALTH_SPEC if req]
    print(f"스택 부팅 대기 (최대 {timeout:.0f}s) — 필수 토픽 수신까지...")
    deadline = time.monotonic() + timeout
    missing = list(required)
    while time.monotonic() < deadline:
        fn.spin_for(1.0)
        missing = [k for k in required if fn.count.get(k, 0) == 0]
        if not missing:
            print("  필수 토픽 전부 수신 확인")
            return True
        print(f"  대기 중... 미수신 {len(missing)}개: {', '.join(missing)}")
    print(f"! 부팅 대기 초과 — 미수신: {', '.join(missing)} (헬스체크에서 최종 판정)")
    return False


# =========================================================================
# 이미지 저장 (depth = uint16 mm, PIL I;16 계약)
# =========================================================================
def _save_rgb(path: Path, rgb: np.ndarray):
    from PIL import Image as PILImage
    PILImage.fromarray(np.ascontiguousarray(rgb)).save(str(path))


def _save_depth16(path: Path, depth_mm: np.ndarray):
    from PIL import Image as PILImage
    arr = np.ascontiguousarray(depth_mm.astype(np.uint16))
    im = PILImage.frombytes("I;16", (arr.shape[1], arr.shape[0]), arr.tobytes())
    im.save(str(path))


def _save_frame_set(fn: FieldNode, out: Path, n: int):
    """top/near RGB+depth 4장 저장 (raw 쌍 규칙 — 스티치만 저장 금지)."""
    saved = []
    rgb = fn.rgb_pair()
    depth = fn.depth_pair()
    if rgb is not None:
        _save_rgb(out / f"frame_{n}_top_rgb.png", rgb[0])
        saved.append("top_rgb")
        _save_rgb(out / f"frame_{n}_near_rgb.png", rgb[1])
        saved.append("near_rgb")
    if depth is not None:
        _save_depth16(out / f"frame_{n}_top_depth.png", depth[0])
        saved.append("top_depth")
        _save_depth16(out / f"frame_{n}_near_depth.png", depth[1])
        saved.append("near_depth")
    return saved, rgb


# =========================================================================
# YOLO 디버그 (항상 스티치 프레임 — 단일 캠 A1 금지 계약)
# =========================================================================
def _load_yolo(args):
    """A1+face 로드. 실패 시 None (record 는 YOLO 저장만 생략하고 계속)."""
    try:
        import cv2  # noqa: F401
        from ultralytics import YOLO
    except ImportError as e:
        print(f"! ultralytics/cv2 사용 불가 ({e}) — YOLO 디버그 이미지는 생략하고 계속")
        return None
    missing = [str(p) for p in (A1_WEIGHTS, FACE_WEIGHTS) if not p.exists()]
    if missing:
        print("! YOLO weight 없음 — YOLO 디버그 생략:")
        for m in missing:
            print(f"    {m}")
        return None
    print("YOLO 2단계 모델 로드 중 (A1 + face, 최초 ~15s)...")
    a1 = YOLO(str(A1_WEIGHTS))
    face = YOLO(str(FACE_WEIGHTS))
    # 워밍업 (루프 첫 추론 지연 방지)
    a1.predict(np.zeros((480, 640, 3), np.uint8),
               imgsz=A1_IMGSZ_STITCHED, conf=args.conf, verbose=False)
    face.predict(np.zeros((224, 224, 3), np.uint8),
                 imgsz=FACE_IMGSZ, conf=args.face_conf, verbose=False)
    print("  로드 완료")
    return {"a1": a1, "face": face}


def _yolo_debug(models, stitched_rgb: np.ndarray, path: Path, args, header: str):
    """스티치 프레임에 A1(BGR, imgsz=896) + 큐브 크롭 face(BGR, imgsz=224)
    + face_vote 를 돌리고 오버레이 PNG 저장. det 요약 리스트 반환."""
    import cv2
    bgr = np.ascontiguousarray(stitched_rgb[:, :, ::-1])  # ultralytics numpy=BGR 규약
    res = models["a1"].predict(bgr, imgsz=A1_IMGSZ_STITCHED,
                               conf=args.conf, verbose=False)[0]
    canvas = bgr.copy()
    dets = []
    for box in res.boxes:
        cls = res.names[int(box.cls[0])]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        vote = None
        if cls == "cube_like_object":
            pw = int((x2 - x1) * args.crop_pad)
            ph = int((y2 - y1) * args.crop_pad)
            cx1, cy1 = max(0, x1 - pw), max(0, y1 - ph)
            cx2 = min(bgr.shape[1], x2 + pw)
            cy2 = min(bgr.shape[0], y2 + ph)
            crop = np.ascontiguousarray(bgr[cy1:cy2, cx1:cx2])
            faces = []
            if crop.size:
                fr = models["face"].predict(crop, imgsz=FACE_IMGSZ,
                                            conf=args.face_conf, verbose=False)[0]
                faces = [(fr.names[int(b.cls[0])], float(b.conf[0])) for b in fr.boxes]
            ident, vconf = face_vote(faces)
            vote = {"identity": ident, "conf": vconf,
                    "faces": [[n, round(c, 3)] for n, c in faces]}
        color = (0, 200, 0)
        label = f"{cls} {conf:.2f}"
        if vote is not None:
            if vote["identity"] in FRUITS:
                color = (0, 140, 255)
            if vote["identity"]:
                label += f" -> {vote['identity']} {vote['conf']:.2f}"
            else:
                label += " -> face?"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, label, (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        dets.append({"cls": cls, "conf": round(conf, 3),
                     "box": [x1, y1, x2, y2], "face_vote": vote})
    cv2.putText(canvas, header, (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)
    return dets


# =========================================================================
# up — 기동 + 자동 점검 + 포즈 시딩
# =========================================================================
def cmd_up(args, out: Path) -> int:
    banner("field_autopilot UP — 브리지 기동 + 상태 자동 판정")
    print(f"산출물: {out}")

    # ① 물리 체크리스트 (센서로 알 수 없는 것만 사람에게 묻는다)
    if args.yes:
        print("(--yes) 물리 체크리스트 자동 통과 처리")
    checklist = _checklist(DEFAULT_CHECKLIST, assume_yes=args.yes)
    phys_missing = [k for k, v in checklist.items() if not v]
    if phys_missing:
        print(f"! 물리 체크 미충족: {', '.join(phys_missing)} — 해결 후 재실행 권장")

    # ② 브리지 중복 정리 (다중 실행 = 시리얼 스터터 원인)
    bm = BridgeManager(out)
    killed = bm.kill_duplicate_mecanum()
    print(f"모터 브리지 중복 정리: {killed}개 종료" if killed else "모터 브리지 중복 없음")

    # ③ 스택 기동 (이미 떠 있으면 재사용)
    was_running = bm.stack_running()
    launched = False
    if was_running:
        print("스택 이미 기동돼 있음 — 재사용")
    else:
        bm.launch_stack(camera_depth=True)
        launched = True
        print(f"스택 기동 명령 발행 (depth 켬) — 로그: {out / 'stack_launch.log'}")

    # ④ 헬스체크
    rclpy, fn = _ros_node("field_autopilot_up")
    seeded = None
    battery = None
    try:
        if launched:
            _wait_stack_boot(fn, args.boot_wait)
        h = healthcheck(fn, listen_sec=args.listen, require_gripper=False)
        print(health_report_text(h))

        # [2026-07-24] 그리퍼 자가복구(L3) + 필수 판정 반영.
        # 종전 up 은 require_gripper=False 로 죽은 그리퍼를 "[–]"로 묵인하고
        # "✓ 스택 정상"을 찍었다 — 죽은 그리퍼가 2분 뒤 러너 게이트
        # (require_gripper=True)에서야 필수결손으로 드러났다(7/24 사고). 이제
        # up 이 직접 감지→복구(respawn 대기→노드만 재기동)하고, 끝내 실패하면
        # 러너와 동일하게 필수결손으로 처리한다.
        # 근거: docs/06-troubleshooting.md
        g_item = next((it for it in h["items"] if it["key"] == "gripper_board"), None)
        gripper_ok = bool(g_item and g_item.get("alive"))
        gripper_heal = None
        if not gripper_ok:
            print("자가복구: 그리퍼 브리지 미동작 감지 — respawn 복귀 대기 후 "
                  "필요 시 노드만 재기동 (카메라 재초기화 없음)")
            gripper_heal = bm.ensure_gripper_alive(fn, wait_s=6.0)
            gripper_ok = gripper_heal["ok"]
            print(f"  → 자가복구 {'복구됨' if gripper_ok else '복구 실패'}: "
                  f"action={gripper_heal['action']} ({gripper_heal['detail']})")

        # ⑤ START 포즈 시딩 (우하단 북향)
        if args.seed:
            if _wait_sub(fn, "pose_seed"):
                fn.seed_pose(*START_POSE)
                fn.spin_for(1.0)
                p = fn.pose()
                if p is not None and math.hypot(p[0] - START_POSE[0],
                                                p[1] - START_POSE[1]) < 0.35:
                    print(f"✓ 포즈 시딩 OK: ({p[0]:+.2f},{p[1]:+.2f},"
                          f"{math.degrees(p[2]):+.0f}deg) — 실물이 우하단 북향인지 눈으로 확인")
                    seeded = {"ok": True, "pose": list(p)}
                else:
                    print(f"! 시딩 후 pose 불일치/없음: {p} — localization 락 실패 가능. "
                          "로봇 위치·라이다 확인 후 재시딩")
                    seeded = {"ok": False, "pose": list(p) if p else None}
            else:
                print("! /arena_lightweight/pose 구독자 없음 (arena 노드 미기동?) — 시딩 생략")
                seeded = {"ok": False, "pose": None}
        else:
            print("포즈 시딩 생략 (--no-seed)")

        # ⑥ 배터리 프로브 (옵션 — 로봇이 3cm 움직임)
        if args.battery_probe:
            go = _checklist(
                [("probe", "배터리 프로브: 로봇이 3cm 전진 후 복귀합니다. 진행할까요?")],
                assume_yes=args.yes)["probe"]
            if go:
                off = battery_off_signature(fn)
                battery = {"probed": True, "off_signature": bool(off)}
                if off:
                    print("✗ 배터리 OFF 시그니처 감지 (이동 타임아웃 + odom 변위 0) — "
                          "구동 배터리 스위치 확인!")
                else:
                    print("✓ 배터리/구동 정상 (프로브 이동·복귀 확인)")
            else:
                battery = {"probed": False}
                print("배터리 프로브 건너뜀 (조작자 거부)")

        # ⑦ 최종 판정
        reasons = []
        if not h["ok"]:
            reasons.append("필수 토픽 결손")
        if not gripper_ok:
            reasons.append("그리퍼 보드 미동작(자가복구 실패)")
        if not checklist.get("ready", checklist.get("battery", True)):
            reasons.append("물리 체크리스트 미충족 (조작자 응답)")
        if battery and battery.get("off_signature"):
            reasons.append("배터리 OFF 시그니처 (프로브)")
        ok = not reasons
        code = 0 if ok else 2
        if ok:
            banner("최종 판정: 정상 — 진행 가능. 다음: "
                   "python3 -u mission/field_autopilot.py record")
            if phys_missing:
                print(f"  (경고: 물리 체크 미충족 {', '.join(phys_missing)} — 눈으로 재확인)")
        else:
            banner("최종 판정: 필수 결손 — " + " / ".join(reasons))
            print("  조치: 위 ✗ 항목 해결 → 'check' 로 재판정 (스택 재기동은 'down' 후 'up')")

        verdict = {
            "cmd": "up", "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "checklist": checklist, "phys_missing": phys_missing,
            "duplicates_killed": killed, "stack_was_running": was_running,
            "stack_launched": launched, "health": h,
            "gripper_heal": gripper_heal, "gripper_ok": gripper_ok,
            "seeded": seeded, "battery_probe": battery,
            "ok": ok, "exit_code": code,
        }
        (out / "verdict.json").write_text(
            json.dumps(verdict, ensure_ascii=False, indent=2))
        print(f"판정 저장: {out / 'verdict.json'}")
        return code
    finally:
        rclpy.shutdown()


# =========================================================================
# check — 중복 정리 + 헬스체크만
# =========================================================================
def cmd_check(args, out: Path) -> int:
    banner("field_autopilot CHECK — 상태 점검 (기동 안 함)")
    bm = BridgeManager(out)
    killed = bm.kill_duplicate_mecanum()
    print(f"모터 브리지 중복 정리: {killed}개 종료" if killed else "모터 브리지 중복 없음")
    if not bm.stack_running():
        print("! 스택 프로세스 없음 — 다른 방식으로 띄웠으면 무시, 아니면 'up' 먼저")
    rclpy, fn = _ros_node("field_autopilot_check")
    try:
        h = healthcheck(fn, listen_sec=args.listen, require_gripper=False)
        print(health_report_text(h))
        code = 0 if h["ok"] else 2
        (out / "verdict.json").write_text(json.dumps(
            {"cmd": "check", "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
             "duplicates_killed": killed, "health": h,
             "ok": h["ok"], "exit_code": code},
            ensure_ascii=False, indent=2))
        print(f"판정 저장: {out / 'verdict.json'}")
        return code
    finally:
        rclpy.shutdown()


# =========================================================================
# record — GT 입력 + rosbag + raw/depth/YOLO 주기 저장 (Ctrl-C 종료)
# =========================================================================
class Teleop:
    """키보드 텔레옵 (record 병행/단독).

    - w/a/s/d : **월드 좌표계** 이동 (w=+y북, s=-y남, a=-x서, d=+x동) —
      arena pose yaw로 회전 변환해 body twist로 발행. pose 없으면 로봇
      기준(전/후/좌/우)으로 폴백하고 경고.
    - ← / →  : 반시계 / 시계 회전 (방향키. 대체키 q/e)
    - space   : 즉시 정지
    - i       : **initialize** — 로봇을 START(우하단, 북향)에 놓고 누르면
                pose·위치 초기화(리시딩)
    - p       : 현재 pose 출력 / x 또는 ESC 단독: 종료
    - 키를 떼면(오토리핏 끊기면) 0.35s 후 자동 정지.
    발행 토픽: /cmd_vel (arena idle 침묵 패치로 mux 폴스루 — 재빌드 필수).
    """

    KEY_HOLD_SEC = 0.35

    def __init__(self, fn, speed: float = 0.3, turn: float = 0.8):
        from geometry_msgs.msg import Twist
        self._Twist = Twist
        self.fn = fn
        self.speed = speed
        self.turn = turn
        self.pub = fn.node.create_publisher(Twist, "/cmd_vel", 10)
        # 데몬 키 스레드가 finally를 못 타고 죽으면 터미널이 cbreak로 남는다
        # — 종료 경로에서 restore()로 복원할 수 있게 원래 설정을 보관.
        self._term_old = None
        try:
            import termios
            self._term_old = termios.tcgetattr(sys.stdin.fileno())
        except Exception:  # noqa: BLE001 — 비TTY(파이프 실행)면 텔레옵 키 입력 불가
            pass
        self.vx_w = 0.0
        self.vy_w = 0.0
        self.wz = 0.0
        self.deadline = {"x": 0.0, "y": 0.0, "w": 0.0}
        self.quit = False
        self._warned_no_pose = False

    def _key_loop(self):
        import select
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self.quit:
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not r:
                    continue
                ch = sys.stdin.read(1)
                now = time.monotonic()
                if ch == "\x1b":  # ESC 또는 방향키 시퀀스
                    r2, _, _ = select.select([sys.stdin], [], [], 0.05)
                    if not r2:
                        self.quit = True
                        continue
                    seq = sys.stdin.read(2)
                    if seq.endswith("D"):      # ←
                        self.wz = self.turn
                        self.deadline["w"] = now + self.KEY_HOLD_SEC
                    elif seq.endswith("C"):    # →
                        self.wz = -self.turn
                        self.deadline["w"] = now + self.KEY_HOLD_SEC
                elif ch in ("w", "W"):
                    self.vy_w = self.speed
                    self.deadline["y"] = now + self.KEY_HOLD_SEC
                elif ch in ("s", "S"):
                    self.vy_w = -self.speed
                    self.deadline["y"] = now + self.KEY_HOLD_SEC
                elif ch in ("a", "A"):
                    self.vx_w = -self.speed
                    self.deadline["x"] = now + self.KEY_HOLD_SEC
                elif ch in ("d", "D"):
                    self.vx_w = self.speed
                    self.deadline["x"] = now + self.KEY_HOLD_SEC
                elif ch == "q":
                    self.wz = self.turn
                    self.deadline["w"] = now + self.KEY_HOLD_SEC
                elif ch == "e":
                    self.wz = -self.turn
                    self.deadline["w"] = now + self.KEY_HOLD_SEC
                elif ch == " ":
                    self.vx_w = self.vy_w = self.wz = 0.0
                    for k in self.deadline:
                        self.deadline[k] = 0.0
                elif ch in ("i", "I"):
                    self.fn.seed_pose(*START_POSE)
                    print("\n== initialize: START(1.8,-1.8, 북향) 포즈 리시딩 완료")
                elif ch in ("p", "P"):
                    print(f"\npose = {self.fn.pose()}")
                elif ch in ("x", "X"):
                    self.quit = True
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def _pub_loop(self):
        while not self.quit:
            now = time.monotonic()
            if now > self.deadline["x"]:
                self.vx_w = 0.0
            if now > self.deadline["y"]:
                self.vy_w = 0.0
            if now > self.deadline["w"]:
                self.wz = 0.0
            t = self._Twist()
            pose = self.fn.pose()
            if pose is not None:
                c, s = math.cos(pose[2]), math.sin(pose[2])
                t.linear.x = c * self.vx_w + s * self.vy_w
                t.linear.y = -s * self.vx_w + c * self.vy_w
            else:
                if (self.vx_w or self.vy_w) and not self._warned_no_pose:
                    print("\n! pose 없음 — w/s=전/후, a/d=좌/우 로봇 기준으로 동작")
                    self._warned_no_pose = True
                t.linear.x = self.vy_w
                t.linear.y = -self.vx_w
            t.angular.z = self.wz
            self.pub.publish(t)
            time.sleep(0.05)
        stop = self._Twist()
        for _ in range(3):
            self.pub.publish(stop)
            time.sleep(0.05)

    def restore(self):
        """종료 시 호출: 스레드 정지 + 터미널 설정 복원 + 정지 트위스트."""
        self.quit = True
        if self._term_old is not None:
            try:
                import termios
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN,
                                  self._term_old)
            except Exception:  # noqa: BLE001
                pass
        try:
            self.pub.publish(self._Twist())
        except Exception:  # noqa: BLE001
            pass

    def run_blocking(self):
        import threading
        th = threading.Thread(target=self._pub_loop, daemon=True)
        th.start()
        self._key_loop()
        th.join(timeout=1.0)

    def run_background(self):
        import threading
        for target in (self._pub_loop, self._key_loop):
            threading.Thread(target=target, daemon=True).start()


def cmd_teleop(args, out: Path) -> int:
    banner("field_autopilot TELEOP — WASD=월드 xy 이동, ←→=회전, i=포즈 초기화, x=종료")
    if args.offline:
        print("== offline: 텔레옵은 ROS 필요 — 키맵만 표시")
        print(Teleop.__doc__)
        return 0
    rclpy, fn = _ros_node("field_autopilot_teleop")
    fn.spin_for(2.0)
    print(f"현재 pose: {fn.pose()}  (i = START 리시딩)")
    print("주의: arena 재빌드(idle 침묵 패치) 안 됐으면 /cmd_vel이 무시될 수 있음")
    tel = Teleop(fn, speed=args.speed, turn=args.turn)
    # 키 스레드가 stdin을 점유하는 동안 메인 스레드는 rclpy 스핀
    import threading
    kb = threading.Thread(target=tel._key_loop, daemon=True)
    pb = threading.Thread(target=tel._pub_loop, daemon=True)
    kb.start()
    pb.start()
    try:
        while not tel.quit:
            rclpy.spin_once(fn.node, timeout_sec=0.1)
    except KeyboardInterrupt:
        tel.quit = True
    pb.join(timeout=1.0)
    print("\n== 텔레옵 종료 (정지 명령 발행됨)")
    return 0


def cmd_record(args, out: Path) -> int:
    banner("field_autopilot RECORD — GT 입력 + rosbag + 주기 저장 (종료 = Ctrl-C)")
    print(f"산출물: {out}")

    # GT 배치 (우선순위: --gt-text > --gt-file > 대화식 > 없음)
    try:
        gt = prompt_gt(interactive=not args.yes,
                       gt_text=args.gt_text, gt_file=args.gt_file)
    except ValueError as e:
        print(f"✗ GT 파싱 실패: {e}")
        return 2
    print(gt_summary_text(gt))
    (out / "gt_summary.txt").write_text(gt_summary_text(gt) + "\n")
    (out / "gt.json").write_text(json.dumps(
        {"cells": {f"{x},{y}": c for (x, y), c in sorted(gt.items())}},
        ensure_ascii=False, indent=2))

    # 스티치 캘리브는 마스트 위치에 종속 — 실물과 일치 확인
    mast_ok = _checklist(
        [("mast", f"마스트가 '{args.mast}' 위치가 맞나요? (스티치 캘리브 기준)")],
        assume_yes=args.yes)["mast"]
    if not mast_ok:
        print(f"! --mast {args.mast} 와 실물 불일치 — 맞는 값으로 재실행 권장 (일단 계속)")
    stitcher = Stitcher(args.mast)
    models = _load_yolo(args)

    rclpy, fn = _ros_node("field_autopilot_record")
    bag_proc = None
    n = 0
    yolo_saved = 0
    t0 = time.monotonic()
    try:
        print("카메라 4스트림(RGB+depth) 프레임 대기 (최대 10s)...")
        if not fn.wait_fresh_frames(timeout=10.0):
            print("✗ 카메라 4스트림 미수신 — depth 포함 스택인지 확인:")
            print("  'up' 명령이 enable_camera_depth:=true 로 기동함 — 'down' 후 'up' 재실행")
            print("  기록 시작 안 함 (창 낭비 방지).")
            return 2
        for cam, fname in (("top", "top_K.txt"), ("near", "near_K.txt")):
            k = fn.camera_k(cam)
            if k:
                (out / fname).write_text(" ".join(str(v) for v in k) + "\n")
        if fn.pose() is None:
            print("! arena pose 없음 — frames.jsonl 에 pose=null 로 계속 (localization 확인 권장)")

        bag_proc = rosbag_record(out, extra_topics=[
            fn.TOPICS["top_rgb"], fn.TOPICS["top_depth"],
            fn.TOPICS["near_rgb"], fn.TOPICS["near_depth"]])
        print(f"rosbag 기록 시작 → {out / 'bag'} (로그: {out / 'rosbag.log'})")
        if getattr(args, "teleop", False):
            tel = Teleop(fn, speed=args.speed, turn=args.turn)
            globals()["_active_teleop"] = tel
            tel.run_background()
            print("텔레옵 활성: WASD=월드 xy 이동, ←→(또는 q/e)=회전, space=정지, "
                  "i=START 포즈 초기화, p=pose 표시")
        banner(f"기록 루프: raw {args.interval:.1f}s 간격 / YOLO {args.yolo_interval:.1f}s 간격 "
               f"(mast={args.mast}, 스티치 추론) — 종료는 Ctrl-C")

        next_frame = time.monotonic()
        next_yolo = time.monotonic()
        with open(out / "frames.jsonl", "a") as jf:
            while True:
                fn.spin_for(0.1)
                now = time.monotonic()
                if now < next_frame:
                    continue
                saved, rgb = _save_frame_set(fn, out, n)
                pose = fn.pose()
                loc = (fn.arena_status().get("localization") or {})
                entry = {"n": n, "t_sec": round(now - t0, 2),
                         "pose": list(pose) if pose else None,
                         "loc_latency_ms": loc.get("latency_ms"),
                         "loc_reason": loc.get("reason"),
                         "saved": saved, "yolo": None}
                if models is not None and rgb is not None and now >= next_yolo:
                    ypath = out / f"yolo_debug_{n}.png"
                    ptxt = (f"({pose[0]:+.2f},{pose[1]:+.2f},"
                            f"{math.degrees(pose[2]):+.0f}deg)") if pose else "pose?"
                    header = f"n={n} t={entry['t_sec']:.0f}s mast={args.mast} {ptxt}"
                    try:
                        entry["dets"] = _yolo_debug(
                            models, stitcher.stitch(rgb[0], rgb[1]), ypath, args, header)
                        entry["yolo"] = ypath.name
                        yolo_saved += 1
                    except Exception as e:  # 추론 실패해도 raw 기록은 계속
                        print(f"  ! YOLO 디버그 실패(n={n}): {e} — raw 저장은 계속")
                    while next_yolo <= now:
                        next_yolo += args.yolo_interval
                jf.write(json.dumps(entry, ensure_ascii=False) + "\n")
                jf.flush()
                ptag = (f"({pose[0]:+.2f},{pose[1]:+.2f})" if pose else "(pose없음)")
                ytag = (f" yolo={len(entry.get('dets', []))}det"
                        if entry["yolo"] else "")
                print(f"  [{n}] t={entry['t_sec']:6.1f}s 저장 {len(saved)}/4 {ptag}{ytag}")
                n += 1
                while next_frame <= now:
                    next_frame += args.interval
    except KeyboardInterrupt:
        print("\nCtrl-C 수신 — 정상 종료 절차 진행")
    finally:
        tel_active = globals().pop("_active_teleop", None)
        if tel_active is not None:
            tel_active.restore()  # 터미널 cbreak 복원 + 정지 트위스트
        if bag_proc is not None:
            bag_proc.send_signal(signal.SIGINT)
            try:
                bag_proc.wait(timeout=10)
                print("rosbag 정상 종료")
            except subprocess.TimeoutExpired:
                bag_proc.terminate()
                print("! rosbag SIGINT 무응답 — terminate (bag 무결성 확인 필요)")
        rclpy.shutdown()
    elapsed = time.monotonic() - t0
    banner(f"기록 요약: raw {n}프레임 / YOLO 디버그 {yolo_saved}장 / 경과 {elapsed:.0f}s")
    print(f"  저장 위치: {out}")
    return 0


# =========================================================================
# down — rosbag/스택 정지 + 잔존 프로세스 보고
# =========================================================================
def cmd_down(args, out: Path) -> int:
    banner("field_autopilot DOWN — 전체 종료")
    bags = BridgeManager._pgrep("ros2 bag record")
    for pid in bags:
        subprocess.run(["kill", "-INT", pid])
    if bags:
        print(f"rosbag SIGINT → {len(bags)}개")
    bm = BridgeManager(out)
    bm.stop_stack()
    print("스택 SIGINT 발행 — 6s 대기 후 잔존 확인...")
    time.sleep(6.0)
    patterns = [
        ("arena launch", BridgeManager.ARENA_PATTERN),
        ("mecanum 브리지", "mecanum_bridge_node"),
        ("arena 노드", "arena_control_node"),
        ("RealSense 카메라", "realsense2_camera_node"),
        ("라이다", "sllidar_node"),
        ("rosbag", "ros2 bag record"),
    ]
    leftovers = []
    for name, pat in patterns:
        pids = BridgeManager._pgrep(pat)
        if pids:
            leftovers.append((name, pids))
    if leftovers:
        banner("잔존 프로세스 있음 — 종료 중일 수 있음. 잠시 후 'down' 재실행으로 재확인:")
        for name, pids in leftovers:
            print(f"  ! {name}: PID {', '.join(pids)}  (강제: kill -INT {' '.join(pids)})")
    else:
        banner("잔존 프로세스 없음 — 깨끗하게 종료됨")
    return 0


# =========================================================================
# --offline 자가진단 (개발기, ROS 불필요)
# =========================================================================
def offline_diag(args, out: Path) -> int:
    banner(f"오프라인 자가진단 ({args.cmd}) — ROS/rclpy 미사용")
    report = {"cmd": args.cmd, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": []}
    fails = []

    def chk(name, good, detail="", required=True):
        mark = "✓" if good else ("✗" if required else "–")
        print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
        report["checks"].append({"name": name, "ok": bool(good),
                                 "required": required, "detail": detail})
        if required and not good:
            fails.append(name)
        return good

    chk("A1 weight", A1_WEIGHTS.exists(), str(A1_WEIGHTS), required=False)
    chk("face weight", FACE_WEIGHTS.exists(), str(FACE_WEIGHTS), required=False)
    dummy = np.zeros((480, 640, 3), np.uint8)
    for mast in ("down", "up"):
        st = Stitcher(mast)
        s = st.stitch(dummy, dummy.copy())
        ok = (st.to_source(10, 5)[0] == "top"
              and st.to_source(10, s.shape[0] - 5)[0] == "near")
        chk(f"Stitcher({mast})", ok, f"스티치 {s.shape[1]}x{s.shape[0]}px")
    ident, _vconf = face_vote([("plain", 0.9), ("banana", 0.35)])
    chk("face_vote 과일면 우선", ident == "banana")
    cell, err = snap_cell(*official_cm_to_map(150, 200))
    chk("snap_cell 42격자", cell == (150, 200) and err < 0.01)
    chk("START_POSE", True,
        f"({START_POSE[0]:+.2f},{START_POSE[1]:+.2f},"
        f"{math.degrees(START_POSE[2]):+.0f}deg) 우하단 북향", required=False)

    if args.cmd == "record":
        try:
            gt = prompt_gt(interactive=not args.yes,
                           gt_text=args.gt_text, gt_file=args.gt_file)
            print(gt_summary_text(gt))
            (out / "gt_summary.txt").write_text(gt_summary_text(gt) + "\n")
            chk("GT 파싱", True, f"{len(gt)}개 셀")
        except ValueError as e:
            chk("GT 파싱", False, str(e))
        chk("스티치 대상 mast", args.mast in ("down", "up"), args.mast)
        for mod in ("PIL", "cv2", "ultralytics"):
            try:
                __import__(mod)
                chk(f"모듈 {mod}", True, required=False)
            except ImportError:
                chk(f"모듈 {mod}", False, "이 기기엔 없음 — 실기에서 해당 기능만 생략됨",
                    required=False)

    (out / "offline_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    code = 0 if not fails else 2
    if fails:
        banner("오프라인 자가진단 실패 항목: " + ", ".join(fails))
    else:
        banner("오프라인 자가진단 통과 — 실기(Jetson)에서 --offline 빼고 실행")
    print(f"리포트: {out / 'offline_report.json'}")
    return code


# =========================================================================
# main
# =========================================================================
def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default="",
                        help="산출물 디렉토리 (기본 logs/field_ops/<ts>_<cmd>)")
    common.add_argument("--yes", action="store_true",
                        help="모든 y/n 프롬프트 자동 y (무인 실행)")
    common.add_argument("--offline", action="store_true",
                        help="ROS 없이 인자/GT 파싱 자가진단만 (개발기 테스트)")

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_up = sub.add_parser("up", parents=[common],
                          help="체크리스트+중복정리+스택기동+헬스체크+포즈시딩")
    p_up.add_argument("--seed", action=argparse.BooleanOptionalAction, default=True,
                      help="START 포즈 자동 시딩 (기본 켬, --no-seed 로 끔)")
    p_up.add_argument("--battery-probe", action="store_true",
                      help="3cm 프로브 이동으로 배터리 OFF 시그니처 판정 (로봇 움직임)")
    p_up.add_argument("--boot-wait", type=float, default=45.0,
                      help="신규 기동 시 토픽 수신 대기 상한(초)")
    p_up.add_argument("--listen", type=float, default=8.0,
                      help="헬스체크 청취 시간(초)")

    p_chk = sub.add_parser("check", parents=[common],
                           help="중복정리+헬스체크만 (기동 안 함)")
    p_chk.add_argument("--listen", type=float, default=8.0,
                       help="헬스체크 청취 시간(초)")

    p_rec = sub.add_parser("record", parents=[common],
                           help="GT 입력+rosbag+raw/depth/YOLO 주기 저장 (Ctrl-C 종료)")
    p_rec.add_argument("--gt-text", default="",
                       help='GT 배치 텍스트 "apple:150,200;plain:250,300"')
    p_rec.add_argument("--gt-file", default="", help="GT 배치 텍스트 파일 경로")
    p_rec.add_argument("--interval", type=float, default=2.0,
                       help="raw RGB/depth 저장 간격(초)")
    p_rec.add_argument("--yolo-interval", type=float, default=6.0,
                       help="YOLO 디버그 이미지 저장 간격(초)")
    p_rec.add_argument("--mast", choices=("down", "up"), default="down",
                       help="스티치 캘리브 기준 마스트 위치 (실물과 일치 필수)")
    p_rec.add_argument("--conf", type=float, default=0.25, help="A1 conf 문턱")
    p_rec.add_argument("--face-conf", type=float, default=0.10, help="face conf 문턱")
    p_rec.add_argument("--crop-pad", type=float, default=0.18,
                       help="face 크롭 패딩 비율 (85번 계약)")
    p_rec.add_argument("--teleop", action="store_true",
                       help="기록 중 키보드 텔레옵 병행 (WASD 월드 xy, ←→ 회전, i 초기화)")
    p_rec.add_argument("--speed", type=float, default=0.3, help="텔레옵 이동 속도 m/s")
    p_rec.add_argument("--turn", type=float, default=0.8, help="텔레옵 회전 속도 rad/s")

    sub.add_parser("down", parents=[common], help="rosbag/스택 정지 + 잔존 보고")

    p_tel = sub.add_parser("teleop", parents=[common],
                           help="키보드 텔레옵 (WASD=월드 xy, ←→=회전, i=포즈 초기화)")
    p_tel.add_argument("--speed", type=float, default=0.3, help="이동 속도 m/s")
    p_tel.add_argument("--turn", type=float, default=0.8, help="회전 속도 rad/s")

    args = ap.parse_args()
    out = Path(args.out) if args.out else (
        REPO_ROOT / "logs" / "field_ops"
        / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.cmd}")
    out.mkdir(parents=True, exist_ok=True)

    if args.offline:
        sys.exit(offline_diag(args, out))

    signal.signal(signal.SIGTERM, _sigterm_raise)
    handler = {"up": cmd_up, "check": cmd_check, "record": cmd_record,
               "down": cmd_down, "teleop": cmd_teleop}[args.cmd]
    sys.exit(handler(args, out))


if __name__ == "__main__":
    main()
