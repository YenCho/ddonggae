#!/usr/bin/env python3
"""경기장 측광(노출/화이트밸런스) 신속 재조정 — 브라우저 없이 CLI 한 방.

기하 캘리브(up.json/down.json)는 조명과 무관하므로 그대로 두고, 조명이 바뀌는
경기장에서는 **측광만** 다시 잡는다. 이 도구는 그 절차를 자동화한다:

  # 0) 마스트를 올린다 (스캔 = 스티치가 쓰이는 상태에서 잰다)
  ros2 topic pub --once /lift/command std_msgs/String "{data: LIFT_TO_TOP}"   # ~7s

  # 1) 현재 상태 진단 (~8초)
  python3 scripts/dev/field_ops/photometry_tune.py --check

  # 2) 노출 자동 스윕 → 최적값 적용 + 검증 + 런치 인자 출력 (~1분)
  python3 scripts/dev/field_ops/photometry_tune.py --sweep

  # 3) 개발실 기본값으로 원복
  python3 scripts/dev/field_ops/photometry_tune.py --restore

합격 기준 (docs/hardware/camera_stitch_calibration.md §1):
  바닥 Y 100~140 · |ΔY| ≤ 6 · |ΔR/G| ≤ 0.05 · |ΔB/G| ≤ 0.05
  + 흰 물체가 날아가지 않을 것 (겹침 띠 포화율로 근사 감시)

선정 규칙: 바닥 Y 가 목표(기본 115)에 가장 가까운 노출. 단 포화율이
--max-sat(기본 8%) 초과하면 제외 — 반사 심한 바닥에서 X마커/plain 큐브가
날아가는 것을 막는다 (7/20 실험실: 노출 300에서 마커 X 획 소실 확인).

결과는 data/calibration/photometry/<label>_<ts>.json 에 저장되고, 스택 재시작
시 쓸 런치 인자 한 줄을 출력한다. 스택을 재시작하지 않으면 이미 적용된
런타임 값이 그대로 유지된다.

주의:
- 두 캠의 색차(ΔR/G·ΔB/G)는 센서 개체차(D435 vs D435i)로, **런타임 WB 로는
  교정되지 않는다** (7/20 실측: WB 3200→6000 적용 검증 스윕에 ΔR/G 변화 0.004).
  판정에서 '참고'로만 보고하며 무시해도 된다 — 영향은 이음매 걸친 크롭의 색뿐.
  --match-wb 는 폐기(호환용으로만 잔존).
- 마스트 다운 상태로 재면 안 된다 — 다운 띠는 양쪽 렌즈 극단 가장자리 29행이라
  비네팅·정반사로 ΔY/색차가 수 배 부풀려진다 (7/20 실측: 다운 ΔY 28 vs 업 ΔY 6).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fieldlib as fl  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "stitch_calibrator", Path(__file__).resolve().parent / "stitch_calibrator.py")
_sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sc)
photo_apply, photo_read, CAM_NODES = _sc.photo_apply, _sc.photo_read, _sc.CAM_NODES

OUT_DIR = REPO_ROOT / "perception" / "calibration" / "photometry"
DEFAULTS = {"exposure": 156, "gain": 64, "white_balance": 4600,
            "power_line_frequency": 2}
Y_LO, Y_HI = 100.0, 140.0
DY_MAX, DC_MAX = 6.0, 0.05


# ---------------------------------------------------------------- 프레임 취득
def grab_pair(timeout=10.0):
    """/camera_19/rgb + /camera_54/rgb 에서 한 쌍 (RGB uint8)."""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    rclpy.init()
    node = Node("photometry_grab")
    got = {}

    def cb(name):
        def _cb(msg):
            img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
            got[name] = img[:, :, ::-1].copy() if msg.encoding == "bgr8" else img.copy()
        return _cb

    node.create_subscription(Image, "/camera_19/rgb", cb("top"), qos_profile_sensor_data)
    node.create_subscription(Image, "/camera_54/rgb", cb("near"), qos_profile_sensor_data)
    t0 = time.time()
    while rclpy.ok() and len(got) < 2 and time.time() - t0 < timeout:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()
    if len(got) < 2:
        sys.exit("카메라 프레임 타임아웃 — 브릿지가 떠 있는지 확인 (arena-bringup)")
    return got["top"], got["near"]


# ---------------------------------------------------------------- 측광 측정
def measure(top: np.ndarray, near: np.ndarray, mast: str) -> dict:
    """겹침 띠에서 두 캠의 밝기/색 일치도를 잰다.

    캘리브레이터 `photo_stats`/`band` 와 **동일한 방식**(띠 전체 채널 평균)을
    쓴다 — 합격 문턱(|ΔY|≤6 등)이 그 방식 기준으로 정해져 있고, 점 단위
    대응은 렌즈 가장자리 비네팅·국소 질감 때문에 색차가 부풀려진다(7/20 확인).
    입력은 RGB (grab_pair 가 bgr8 을 뒤집어 줌).
    """
    st = fl.Stitcher(mast, w=top.shape[1], h=top.shape[0])   # 해상도 자동 대응
    h_t = top.shape[0]
    c = np.array([[0, 0, 1], [near.shape[1], 0, 1]], np.float64).T
    m = st.A @ c
    v0 = max(0, int(np.floor(float((m[1] / m[2]).min()))))   # near 행0 → top 행
    if h_t <= v0:
        sys.exit(f"겹침 띠 없음 (v0={v0}) — 스티치 캘리브({mast}) 확인")

    def stats(img_rgb, rows):
        a = img_rgb[rows[0]:rows[1]].astype(float)
        r, g, b = (a[:, :, i].mean() for i in range(3))
        y_map = (0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2])
        return {"y": 0.299 * r + 0.587 * g + 0.114 * b,
                "rg": r / max(g, 1e-6), "bg": b / max(g, 1e-6),
                "sat": float((y_map >= 250).mean() * 100),
                "p99": float(np.percentile(y_map, 99))}

    t = stats(top, (v0, h_t))
    n = stats(near, (0, h_t - v0))
    # 띠 좌/우 1/3 밝기차 — 벽·정반사 하이라이트가 띠에 침입하면 좌우가 갈린다
    # (예: START 코너 북향 = 오른쪽 벽 20cm → 우측 절반이 벽을 봄)
    band = near[0:h_t - v0].astype(float)
    w3 = band.shape[1] // 3
    y_l = float((0.299 * band[:, :w3, 0] + 0.587 * band[:, :w3, 1]
                 + 0.114 * band[:, :w3, 2]).mean())
    y_r = float((0.299 * band[:, -w3:, 0] + 0.587 * band[:, -w3:, 1]
                 + 0.114 * band[:, -w3:, 2]).mean())
    return {"band_rows": (v0, h_t),
            "y_top": t["y"], "y_near": n["y"], "dY": t["y"] - n["y"],
            "dRG": t["rg"] - n["rg"], "dBG": t["bg"] - n["bg"],
            "sat_top_pct": t["sat"], "sat_near_pct": n["sat"],
            "p99_top": t["p99"], "p99_near": n["p99"],
            "y_band_left": y_l, "y_band_right": y_r}


def verdict(m: dict, max_sat: float) -> tuple[list[str], list[str]]:
    """(불합격 목록, 참고 목록).

    색차(ΔR/G·ΔB/G)는 참고로만 보고한다 — 2026-07-20 실측: WB 3200→6000
    스윕(적용 검증 포함)에 ΔR/G 변화 0.004 → 두 캠 색차는 런타임 WB 로
    교정 불가한 센서 개체차(D435 vs D435i)로 확정. 영향은 이음매에 걸친
    크롭의 색뿐이다. 단 0.2 초과는 auto WB 가 살아있는 등 설정 사고
    가능성이 있어 불합격으로 취급한다.
    """
    bad, note = [], []
    for cam, y in (("top", m["y_top"]), ("near", m["y_near"])):
        if not (Y_LO <= y <= Y_HI):
            bad.append(f"바닥Y({cam}) {y:.0f} — 목표 {Y_LO:.0f}~{Y_HI:.0f}")
    if abs(m["dY"]) > DY_MAX:
        bad.append(f"|ΔY| {abs(m['dY']):.1f} > {DY_MAX}")
    for k, lbl in (("dRG", "ΔR/G"), ("dBG", "ΔB/G")):
        if abs(m[k]) > 0.2:
            bad.append(f"|{lbl}| {abs(m[k]):.3f} > 0.2 — 설정 사고 의심"
                       " (auto WB 살아있는지 확인)")
        elif abs(m[k]) > DC_MAX:
            note.append(f"|{lbl}| {abs(m[k]):.3f} — 센서 개체차, 교정 불가·무시 가능")
    if max(m["sat_top_pct"], m["sat_near_pct"]) > max_sat:
        bad.append(f"포화 {max(m['sat_top_pct'], m['sat_near_pct']):.1f}% > {max_sat}%"
                   " — 흰 물체 날아감 위험")
    if abs(m["y_band_left"] - m["y_band_right"]) > 25.0:
        bad.append(f"띠 좌우 밝기 갈림 (좌 {m['y_band_left']:.0f} / 우 "
                   f"{m['y_band_right']:.0f}) — 벽·하이라이트 침입 의심. "
                   f"로봇을 벽에서 60cm+ 떼거나 방향을 틀 것")
    return bad, note


def fmt(m: dict) -> str:
    return (f"Y top {m['y_top']:5.1f} / near {m['y_near']:5.1f}   ΔY {m['dY']:+5.1f}   "
            f"ΔR/G {m['dRG']:+.3f}  ΔB/G {m['dBG']:+.3f}   "
            f"포화 {m['sat_top_pct']:.1f}/{m['sat_near_pct']:.1f}%  "
            f"p99 {m['p99_top']:.0f}/{m['p99_near']:.0f}")


# ---------------------------------------------------------------- 아티팩트
def artifact_dir(label: str) -> Path:
    d = (REPO_ROOT / "logs" / "real_validation"
         / f"photometry_{label}_{datetime.now():%Y%m%d_%H%M%S}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_pair(out: Path, tag: str, top, near, mast: str, m: dict):
    """육안 확인용: 원본 2장 + 이음매 표시된 스티치 1장."""
    import cv2
    st = fl.Stitcher(mast, w=top.shape[1], h=top.shape[0])
    top_bgr, near_bgr = top[:, :, ::-1].copy(), near[:, :, ::-1].copy()
    stitched = st.stitch(top_bgr, near_bgr)
    vis = stitched.copy()
    cv2.line(vis, (0, st.seam), (vis.shape[1], st.seam), (0, 0, 255), 1)
    cv2.putText(vis, f"{tag}  Y {m['y_top']:.0f}/{m['y_near']:.0f}  dY {m['dY']:+.1f}",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(str(out / f"{tag}_top.png"), top_bgr)
    cv2.imwrite(str(out / f"{tag}_near.png"), near_bgr)
    cv2.imwrite(str(out / f"{tag}_stitched.png"), vis)


def seam_strip(top, near, mast: str, label: str) -> "np.ndarray":
    """이음매 주변 ±50행 크롭 + 라벨 — 노출 비교 시트용."""
    import cv2
    st = fl.Stitcher(mast, w=top.shape[1], h=top.shape[0])
    stitched = st.stitch(top[:, :, ::-1].copy(), near[:, :, ::-1].copy())
    r0, r1 = max(0, st.seam - 50), min(stitched.shape[0], st.seam + 50)
    strip = stitched[r0:r1].copy()
    cv2.line(strip, (0, st.seam - r0), (strip.shape[1], st.seam - r0), (0, 0, 255), 1)
    cv2.putText(strip, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return strip


# ---------------------------------------------------------------- 적용 헬퍼
def _set_verified(node: str, key: str, value, retries: int = 2) -> bool:
    """set → get 재확인 → 불일치면 재시도. 2026-07-20 경기장에서 ros2 param set 이
    'Node not found' 로 조용히 실패해 WB 스윕 전체가 무효였던 사고의 재발 방지."""
    want = str(value)
    for i in range(retries + 1):
        subprocess.run(["ros2", "param", "set", node, f"rgb_camera.{key}", want],
                       capture_output=True, text=True, timeout=10)
        r = subprocess.run(["ros2", "param", "get", node, f"rgb_camera.{key}"],
                           capture_output=True, text=True, timeout=10)
        got = r.stdout.strip().split(":")[-1].strip()
        if got.lower() == want.lower():
            return True
        if i < retries:
            time.sleep(1.0)
    print(f"  ✗ 적용 실패 확정: {node} {key}={want} (현재 {got!r})")
    return False


def apply_both(exposure, gain, wb) -> bool:
    ok = True
    for cam in CAM_NODES:
        for key, val in (("enable_auto_exposure", "false"),
                         ("enable_auto_white_balance", "false"),
                         ("exposure", int(exposure)), ("gain", int(gain)),
                         ("white_balance", int(wb)), ("power_line_frequency", 2)):
            ok &= _set_verified(CAM_NODES[cam], key, val)
    return ok


def apply_one(cam: str, key: str, value) -> bool:
    return _set_verified(CAM_NODES[cam], key, value)


# ---------------------------------------------------------------- 명령
def cmd_check(args):
    top, near = grab_pair()
    m = measure(top, near, args.mast)
    print(f"[check] {fmt(m)}")
    out = artifact_dir(args.label + "_check")
    save_pair(out, "check", top, near, args.mast, m)
    bad, note = verdict(m, args.max_sat)
    for n in note:
        print("  – " + n)
    if bad:
        print("  ✗ " + "\n  ✗ ".join(bad))
        print("  → --sweep 으로 재조정")
    else:
        print("  ✓ 합격 — 재조정 불필요")
    print(f"  육안 확인: {out.relative_to(REPO_ROOT)}/check_stitched.png")
    return 1 if bad else 0


def cmd_sweep(args):
    exposures = [int(x) for x in args.exposures.split(",")]
    print(f"[sweep] 노출 후보 {exposures} (gain {args.gain}, WB {args.wb}, "
          f"목표 Y≈{args.target_y:.0f}, 포화 한도 {args.max_sat}%)")
    out = artifact_dir(args.label)
    rows, strips = [], []
    for e in exposures:
        apply_both(e, args.gain, args.wb)
        time.sleep(args.settle)
        top, near = grab_pair()
        m = measure(top, near, args.mast)
        ymid = (m["y_top"] + m["y_near"]) / 2
        sat = max(m["sat_top_pct"], m["sat_near_pct"])
        rows.append((e, ymid, sat, m))
        strips.append(seam_strip(top, near, args.mast,
                                 f"exp {e}  Y {ymid:.0f}  dY {m['dY']:+.1f}"))
        print(f"  exp {e:4d}: {fmt(m)}")

    ok_rows = [r for r in rows if r[2] <= args.max_sat]
    pool = ok_rows or rows          # 전부 포화 초과면 그나마 최선
    best = min(pool, key=lambda r: abs(r[1] - args.target_y))
    e_best = best[0]
    if not ok_rows:
        print(f"  ⚠ 모든 후보가 포화 한도 초과 — 가장 가까운 {e_best} 선택. "
              f"후보를 더 낮게 다시: --exposures 로 재실행")

    print(f"\n[적용] exposure {e_best}")
    apply_both(e_best, args.gain, args.wb)
    time.sleep(args.settle)
    top, near = grab_pair()
    m = measure(top, near, args.mast)

    if args.match_wb and abs(m["dRG"]) > DC_MAX:
        print("[match-wb] ⚠ 폐기됨 — 7/20 실측: WB 3200→6000 스윕(적용 검증)에도 "
              "ΔR/G 변화 0.004. 이 카메라들은 런타임 WB 로 색차 교정이 안 된다. "
              "그래도 시도한다:")
        best_wb, best_d = args.wb, abs(m["dRG"]) + abs(m["dBG"])
        for dwb in (-1400, -800, -400, 400, 800):
            wb_try = args.wb + dwb
            if not (2800 <= wb_try <= 6500):
                continue
            if not apply_one("near", "white_balance", wb_try):
                continue    # 적용 실패한 후보는 측정하지 않는다 (무효 데이터 방지)
            time.sleep(args.settle)
            m_try = measure(*grab_pair(), args.mast)
            print(f"  near WB {wb_try}: ΔR/G {m_try['dRG']:+.3f}  ΔB/G {m_try['dBG']:+.3f}")
            score = abs(m_try["dRG"]) + abs(m_try["dBG"])
            if score < best_d:
                best_wb, best_d, m = wb_try, score, m_try
        apply_one("near", "white_balance", best_wb)
        time.sleep(args.settle)
        top, near = grab_pair()
        m = measure(top, near, args.mast)
        near_wb = best_wb
    else:
        near_wb = args.wb

    print(f"\n[최종] {fmt(m)}")
    bad, note = verdict(m, args.max_sat)
    for n in note:
        print("  – " + n)
    print(("  ✗ " + "\n  ✗ ".join(bad)) if bad else "  ✓ 합격")

    # 육안 확인용 아티팩트: 최종 원본/스티치 + 노출 비교 시트
    save_pair(out, "final", top, near, args.mast, m)
    strips.append(seam_strip(top, near, args.mast,
                             f"FINAL exp {e_best}  dY {m['dY']:+.1f}"))
    import cv2
    cv2.imwrite(str(out / "contact_sheet.png"), np.vstack(strips))

    result = {"created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              "label": args.label, "mast": args.mast,
              "applied": {"exposure": e_best, "gain": args.gain,
                          "white_balance_top": args.wb, "white_balance_near": near_wb,
                          "power_line_frequency": 2},
              "measured": m, "pass": not bad,
              "artifacts": str(out.relative_to(REPO_ROOT)),
              "candidates": [{"exposure": r[0], "y_mid": round(r[1], 1),
                              "sat_pct": round(r[2], 2)} for r in rows]}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jpath = OUT_DIR / f"{args.label}_{datetime.now():%Y%m%d_%H%M%S}.json"
    jpath.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n저장: {jpath.relative_to(REPO_ROOT)}")
    print(f"육안 확인: {out.relative_to(REPO_ROOT)}/contact_sheet.png  ← 노출별 이음매 비교")
    print(f"           {out.relative_to(REPO_ROOT)}/final_stitched.png ← 최종 스티치(빨간선=이음매)")
    print("\n스택을 재시작하게 되면 이 인자로 (지금 스택엔 이미 적용됨):")
    print(f"  bash scripts/dev/run_real_competition_bridge.sh \\")
    print(f"      camera_exposure:={e_best} camera_gain:={args.gain} "
          f"camera_white_balance:={args.wb}")
    if near_wb != args.wb:
        print(f"  # 하단캠 WB 만 별도: ros2 param set {CAM_NODES['near']} "
              f"rgb_camera.white_balance {near_wb}  (런치 인자는 공통값만 지원)")
    return 0 if not bad else 1


def cmd_apply(args):
    """스택 재시작 후 15초 복구 — 마지막(또는 지정) 튜닝값 재적용.

    경량 스택(field_autopilot up)은 노출 런치 인자를 받지 못하므로 재시작하면
    무조건 기본값(156)으로 돌아온다. 이 명령이 저장된 JSON 을 다시 적용한다.
    """
    if args.apply == "last":
        cands = sorted(OUT_DIR.glob("*.json"))
        if not cands:
            sys.exit(f"저장된 튜닝값 없음: {OUT_DIR.relative_to(REPO_ROOT)}")
        src = cands[-1]
    else:
        src = Path(args.apply)
    d = json.loads(src.read_text())["applied"]
    print(f"[apply] {src.name}: exposure {d['exposure']} gain {d['gain']} "
          f"WB top {d['white_balance_top']} / near {d['white_balance_near']}")
    apply_both(d["exposure"], d["gain"], d["white_balance_top"])
    if d["white_balance_near"] != d["white_balance_top"]:
        apply_one("near", "white_balance", d["white_balance_near"])
    time.sleep(args.settle)
    m = measure(*grab_pair(), args.mast)
    print(f"  {fmt(m)}")
    bad, note = verdict(m, args.max_sat)
    for n in note:
        print("  – " + n)
    print(("  ✗ " + "\n  ✗ ".join(bad)) if bad else "  ✓ 합격")
    return 0 if not bad else 1


def cmd_restore(args):
    print(f"[restore] 개발실 기본값 {DEFAULTS}")
    apply_both(DEFAULTS["exposure"], DEFAULTS["gain"], DEFAULTS["white_balance"])
    time.sleep(args.settle)
    m = measure(*grab_pair(), args.mast)
    print(f"  {fmt(m)}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="현재 상태 진단만")
    g.add_argument("--sweep", action="store_true", help="노출 스윕 → 최적 적용")
    g.add_argument("--restore", action="store_true", help="기본값(156/64/4600) 원복")
    g.add_argument("--apply", metavar="JSON|last",
                   help="저장된 튜닝값 재적용 (스택 재시작 후 15초 복구). 'last'=최신")
    ap.add_argument("--exposures", default="120,160,200,240,300",
                    help="스윕할 노출 목록 (쉼표)")
    ap.add_argument("--gain", type=int, default=64)
    ap.add_argument("--wb", type=int, default=4600)
    ap.add_argument("--target-y", type=float, default=115.0,
                    help="바닥 Y 목표 (100~140 중앙보다 약간 어둡게 = 포화 여유)")
    ap.add_argument("--max-sat", type=float, default=8.0,
                    help="겹침 띠 포화율 한도 %% (흰 물체 날아감 방지)")
    ap.add_argument("--mast", choices=("up", "down"), default="up",
                    help="겹침 띠 기준 — 스캔이 일어나는 마스트 업이 기본. 마스트를 올리고 잴 것")
    ap.add_argument("--settle", type=float, default=2.5,
                    help="파라미터 적용 후 노출 수렴 대기 s")
    ap.add_argument("--label", default="arena")
    ap.add_argument("--match-wb", action="store_true",
                    help="하단캠 WB 미세 스윕으로 색차(ΔR/G) 축소")
    args = ap.parse_args()
    if args.check:
        sys.exit(cmd_check(args))
    if args.sweep:
        sys.exit(cmd_sweep(args))
    if args.apply:
        sys.exit(cmd_apply(args))
    sys.exit(cmd_restore(args))


if __name__ == "__main__":
    main()
