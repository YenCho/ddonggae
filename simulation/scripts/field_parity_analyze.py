#!/usr/bin/env python3
"""시뮬 캡처 raw 쌍에 **실기 인식 체인 그대로**를 재생하는 오프라인 패리티 분석기.

analyze_street_dataset_sim.py(구판)는 raw 캠 개별에 imgsz=640을 돌렸다 — 실기
계약 위반. 실기(e2e_match_test 스캔 단계)는 **항상 스티치 프레임**에서 추론한다:

  raw top/near → _fieldlib.Stitcher(mast) → A1 seg (imgsz=896, conf=0.25)
  → cube_like 크롭 pad 0.18 → face (imgsz=224, conf=0.1, BGR)
  → face_vote(과일면 conf>=0.30) → depth 중앙값 역투영(원본 캠 픽셀)
  → to_map → 42격자 스냅(snap_err<=0.30) → 셀 투표
  → 셀 확정: 표>=3 + 비대칭 과일투표(강한 과일면 conf>=0.5 K=2, 표차>=2)

이 스크립트는 위 체인을 **재구현하지 않는다** — `e2e_match_test.E2ERunner`를
서브클래스해 `_process_shots_batch`/`_finalize_scan`을 원본 그대로 호출한다
(단일 소스 계약). `_fieldlib`/`e2e_match_test`는 모듈 레벨에서 rclpy를 import
하지 않으므로(FieldNode 안 지연 import) trellis python에서 스텁 없이 import 된다.

입력: street_dataset_sim_capture.py 출력 디렉토리
  shot_<i>_{top,near}.png + _depth.png, top_K.txt/near_K.txt,
  report.json(샷별 arena/gt pose), placement_gt.json(시뮬 GT 배치).

추가 산출:
  - raw 프레임별 측광 패리티 (겹침 띠 Y/p99/포화% — photometry_tune.py 수식,
    실기 타깃 data/calibration/photometry/arena_20260720_212531.json 대비)
  - 42격자 예측 vs GT 매치 표 (report.json + summary.md)
  - e2e_match_test --gt-file 이 그대로 읽는 GT 텍스트 (기본
    logs/sim_validation/gt_seed14.txt — 이후 E2E 풀런이 소비)

실행 (trellis python — torch/ultralytics, rclpy 불필요):
  ~/miniconda3/envs/trellis/bin/python sim/isaacsim/scripts/field_parity_analyze.py \
      --capture-dir logs/sim_validation/street_dataset_sim_<ts>_<label> \
      --mast up --out logs/sim_validation/field_parity_<label>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "perception"))
import fieldlib as fl  # noqa: E402
import match_runner as e2e  # noqa: E402  (실기 스캔 체인 단일 소스)

OUT_BASE = REPO_ROOT / "logs" / "sim_validation"
GT_TEXT_DEFAULT = OUT_BASE / "gt_seed14.txt"
PHOTO_TARGET_JSON = (REPO_ROOT / "perception" / "calibration" / "photometry"
                     / "arena_20260720_212531.json")
# 실기 측광 타깃의 띠 행 (640x480 기준). 타깃 json 이 이 띠로 측정됐으므로
# 640x480 캡처는 같은 띠로 재야 비교 가능 — 다른 해상도는 스티치 A 로 유도.
BAND_REF_ROWS = (419, 480)

MOUNTS = {"up": {"top": fl.TOP_MOUNT_UP, "near": fl.NEAR_MOUNT_UP},
          "down": {"top": fl.TOP_MOUNT_DOWN, "near": fl.NEAR_MOUNT_DOWN}}


# =========================================================================
# 실기 러너 재사용 — ROS 의존부(intr)만 캡처 파일로 대체
# =========================================================================
class OfflineParityRunner(e2e.E2ERunner):
    """fn=None 으로 만들되 intr() 만 캡처의 camera_info K 로 오버라이드.

    스캔 배치 파이프라인(`_process_shots_batch`)과 셀 확정(`_finalize_scan`)은
    e2e 원본이 그대로 돈다 — 여기서 로직을 다시 쓰면 패리티가 깨진다.
    """

    def __init__(self, args_ns, gt: dict, out_dir: Path, k_map: dict):
        super().__init__(args_ns, gt, out_dir, fn=None)
        self._k_map = k_map

    def intr(self, cam: str):
        k = self._k_map.get(cam)
        return e2e.Intrinsics.from_camera_info(k) if k else None


# =========================================================================
# 캡처 로드
# =========================================================================
def load_k_map(cap_dir: Path) -> dict:
    out = {}
    for cam in ("top", "near"):
        p = cap_dir / f"{cam}_K.txt"
        if p.exists():
            out[cam] = [float(v) for v in p.read_text().split()]
    return out


def load_caps(cap_dir: Path, pose_source: str, include_continuous: bool) -> list:
    """capture report.json → e2e `_capture_one` 과 동일 스키마의 caps 목록."""
    entries = json.loads((cap_dir / "report.json").read_text())
    caps = []
    for ent in entries:
        if ent.get("phase") == "continuous" and not include_continuous:
            continue
        pose = ent.get("gt_pose") if pose_source == "gt" else ent.get("pose")
        tag = ent["shot"]
        cap = {"shot": tag, "pose": list(pose) if pose else None,
               "yaw_deg": round(math.degrees(pose[2]), 1) if pose else None,
               "capture_entry": {k: ent.get(k) for k in
                                 ("point", "phase", "pose", "gt_pose", "pose_err")}}
        if pose is None:
            cap["note"] = "pose 없음"
            caps.append(cap)
            continue
        paths = {(cam, kind): cap_dir / f"shot_{tag}_{cam}{suffix}"
                 for cam, kind, suffix in
                 (("top", "rgb", ".png"), ("near", "rgb", ".png"),
                  ("top", "depth", "_depth.png"), ("near", "depth", "_depth.png"))}
        if not all(p.exists() for p in paths.values()):
            cap["note"] = "프레임 없음"
            caps.append(cap)
            continue
        top_rgb = np.asarray(PILImage.open(paths[("top", "rgb")]).convert("RGB"))
        near_rgb = np.asarray(PILImage.open(paths[("near", "rgb")]).convert("RGB"))
        top_d = np.asarray(PILImage.open(paths[("top", "depth")])).astype(np.uint16)
        near_d = np.asarray(PILImage.open(paths[("near", "depth")])).astype(np.uint16)
        cap["frames"] = (top_rgb, near_rgb, top_d, near_d)
        caps.append(cap)
    return caps


def make_stitcher(mast: str, size_wh: tuple, k_map: dict) -> fl.Stitcher:
    """e2e.make_stitcher 와 동일 분기 — 640x480 이면 캘리브 재스케일(크롭 모델),
    다른 해상도면 캡처의 camera_info K 를 intr 로 전달."""
    if size_wh == fl.CALIB_REF_RES:
        return fl.Stitcher(mast)
    intr = {}
    for cam in ("top", "near"):
        k = k_map.get(cam)
        intr[cam] = (float(k[0]), float(k[4]), float(k[2]), float(k[5])) if k else None
    if any(v is None for v in intr.values()):
        intr = None
    return fl.Stitcher(mast, w=int(size_wh[0]), h=int(size_wh[1]), intr=intr)


# =========================================================================
# GT — placement_gt.json → e2e 셀 GT + --gt-file 텍스트
# =========================================================================
def load_gt_cells(cap_dir: Path, gt_file: str) -> dict:
    """{(x_cm,y_cm): cls}. 우선 --gt-file(parse_gt_text 형식), 없으면 캡처의
    placement_gt.json(/sim/objects_state 덤프). 클래스는 cube→plain 정규화."""
    if gt_file:
        gp = Path(gt_file)
        if not gp.exists():
            raise SystemExit(f"--gt-file 없음: {gp}")
        return fl.parse_gt_text(gp.read_text())
    p = cap_dir / "placement_gt.json"
    if not p.exists():
        raise SystemExit(f"GT 소스 없음: {p} (또는 --gt-file) — 캡처가 "
                         "/sim/objects_state 를 못 받았는지 확인")
    gt = {}
    for o in json.loads(p.read_text())["objects"]:
        cls = "plain" if o["class"] == "cube" else o["class"]
        if cls not in fl.CLASSES:
            raise SystemExit(f"placement_gt 클래스 '{o['class']}' 미지원 ({o['prim']})")
        gt[tuple(o["official_cm"])] = cls
    return gt


def write_gt_text(gt: dict, path: Path) -> str:
    """e2e_match_test --gt-file 이 그대로 읽는 형식. 라운드트립 검증 포함."""
    text = ";".join(f"{cls}:{x},{y}" for (x, y), cls in sorted(gt.items()))
    assert fl.parse_gt_text(text) == gt, "gt-text 라운드트립 불일치"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text().strip() != text:
        print(f"[경고] 기존 GT 텍스트와 배치가 다름 — 덮어씀: {path} "
              "(다른 캡처/시드면 --gt-out 으로 경로 분리)", flush=True)
    path.write_text(text + "\n")
    return text


# =========================================================================
# 측광 패리티 (photometry_tune.py measure/stats 수식 그대로)
# =========================================================================
def _band_stats(img_rgb: np.ndarray, rows: tuple) -> dict:
    a = img_rgb[rows[0]:rows[1]].astype(float)
    r, g, b = (a[:, :, i].mean() for i in range(3))
    y_map = (0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2])
    return {"y": 0.299 * r + 0.587 * g + 0.114 * b,
            "sat_pct": float((y_map >= 250).mean() * 100),
            "p99": float(np.percentile(y_map, 99))}


def photometry_shot(top_rgb: np.ndarray, near_rgb: np.ndarray,
                    st: fl.Stitcher) -> dict | None:
    """겹침 띠 밝기 — top 은 [v0,h) 행, near 는 [0,h-v0) 행 (photometry_tune).
    겹침 띠가 없으면 None (photometry_tune 은 sys.exit — 여기선 측광만 생략)."""
    h_t = top_rgb.shape[0]
    if (top_rgb.shape[1], h_t) == fl.CALIB_REF_RES:
        v0 = BAND_REF_ROWS[0]   # 실기 타깃 json 과 동일 띠로 고정 (비교 가능성)
    else:
        c = np.array([[0, 0, 1], [near_rgb.shape[1], 0, 1]], np.float64).T
        m = st.A @ c
        v0 = max(0, int(np.floor(float((m[1] / m[2]).min()))))
        if v0 >= h_t:
            return None
    t = _band_stats(top_rgb, (v0, h_t))
    n = _band_stats(near_rgb, (0, h_t - v0))
    return {"band_rows": [v0, h_t],
            "y_top": round(t["y"], 2), "y_near": round(n["y"], 2),
            "dY": round(t["y"] - n["y"], 2),
            "sat_top_pct": round(t["sat_pct"], 3),
            "sat_near_pct": round(n["sat_pct"], 3),
            "p99_top": round(t["p99"], 1), "p99_near": round(n["p99"], 1)}


def photometry_report(caps: list, st: fl.Stitcher) -> dict:
    per_shot = []
    skipped = 0
    for cap in caps:
        if "frames" not in cap:
            continue
        top_rgb, near_rgb, _, _ = cap["frames"]
        s = photometry_shot(top_rgb, near_rgb, st)
        if s is None:               # 겹침 띠 없음 — 분석은 계속, 측광만 생략
            skipped += 1
            continue
        per_shot.append({"shot": cap["shot"], **s})
    out = {"per_shot": per_shot}
    if skipped:
        out["skipped_no_band"] = skipped
    if per_shot:
        out["mean"] = {k: round(float(np.mean([s[k] for s in per_shot])), 2)
                       for k in ("y_top", "y_near", "dY", "sat_top_pct",
                                 "sat_near_pct", "p99_top", "p99_near")}
    if PHOTO_TARGET_JSON.exists():
        tgt = json.loads(PHOTO_TARGET_JSON.read_text())["measured"]
        out["real_target"] = {k: round(float(tgt[k]), 2) for k in
                              ("y_top", "y_near", "dY", "sat_top_pct",
                               "sat_near_pct", "p99_top", "p99_near")}
        out["real_target_src"] = str(PHOTO_TARGET_JSON.relative_to(REPO_ROOT))
        if per_shot:
            out["delta_vs_target"] = {
                k: round(out["mean"][k] - out["real_target"][k], 2)
                for k in out["mean"]}
    return out


# =========================================================================
# 42격자 매치 표
# =========================================================================
def grid_match_table(cells: dict, gt: dict) -> list:
    """공식좌표 42격자 전 셀: GT/예측/판정 (OK·WRONG·GHOST·MISS·—)."""
    rows = []
    for gy in reversed(fl.GRID_YS_CM):          # 북쪽(y 큰 쪽)부터
        for gx in fl.GRID_XS_CM:
            cell = (gx, gy)
            true = gt.get(cell)
            info = cells.get(cell)
            pred = info["identity"] if info else None
            if true and pred:
                verdict = "OK" if pred == true else "WRONG"
            elif pred:
                verdict = "GHOST"
            elif true:
                verdict = "MISS"
            else:
                verdict = "—"
            rows.append({"cell": list(cell), "gt": true, "pred": pred,
                         "votes": info["votes"] if info else 0,
                         "conflict": info["conflict"] if info else None,
                         "verdict": verdict})
    return rows


def write_summary_md(path: Path, runner, args, st: fl.Stitcher, photo: dict,
                     table: list, gt_text_path: Path, n_frames: int):
    gtc = runner.report["scan"].get("gt_compare", {})
    lines = [
        f"# 시뮬-실기 패리티 분석 — {Path(args.capture_dir).name}",
        "",
        f"- 캡처: `{args.capture_dir}` (프레임 샷 {n_frames})",
        f"- 마스트: **{args.mast}** / pose 소스: {args.pose_source} "
        f"/ range: {args.range_mode}",
        f"- 스티치 캘리브: {st.calib_source} (seam {st.seam}, out_h {st.out_h})",
        f"- 마운트: top {MOUNTS[args.mast]['top']} / near {MOUNTS[args.mast]['near']}",
        f"- 계약: A1 imgsz={fl.A1_IMGSZ_STITCHED} conf={e2e.A1_SCAN_CONF} · "
        f"face imgsz={fl.FACE_IMGSZ} conf=0.1 · pad 0.18 · "
        f"votes>={runner.votes_k} fruit_k={runner.fruit_k} "
        f"fruit_conf>={fl.CELL_FRUIT_CONF} snap<={e2e.MAX_SNAP_ERR_M} · "
        f"pair {'on' if runner.pair is not None else 'off'}",
        f"- GT 텍스트 (e2e --gt-file 용): `{gt_text_path}`",
        "",
        "## 셀 매치 (42격자)",
        "",
        f"정답 {gtc.get('ok', 0)}/{len(runner.gt)} · "
        f"클래스오류 {len(gtc.get('wrong', []))} · "
        f"유령 {len(gtc.get('ghosts', []))} · 누락 {len(gtc.get('missed', []))}",
        "",
        "| cell | GT | 예측 | 표 | 판정 |",
        "|---|---|---|---|---|",
    ]
    for row in table:
        if row["verdict"] == "—":
            continue                      # 빈 셀 30개는 표에서 생략
        conf = f" [{row['conflict']}]" if row["conflict"] else ""
        lines.append(f"| ({row['cell'][0]},{row['cell'][1]}) "
                     f"| {row['gt'] or '—'} | {row['pred'] or '—'}{conf} "
                     f"| {row['votes'] or ''} | {row['verdict']} |")
    lines += ["", "## 측광 패리티 (겹침 띠, photometry_tune 수식)", ""]
    if photo.get("mean"):
        m = photo["mean"]
        lines.append(f"평균: Y_top {m['y_top']} / Y_near {m['y_near']} / "
                     f"dY {m['dY']} / p99 {m['p99_top']}/{m['p99_near']} / "
                     f"sat {m['sat_top_pct']}%/{m['sat_near_pct']}%")
        if photo.get("real_target"):
            t = photo["real_target"]
            d = photo["delta_vs_target"]
            lines.append(f"실기 타깃({photo['real_target_src']}): "
                         f"Y_top {t['y_top']} / Y_near {t['y_near']} — "
                         f"Δ(y_top {d['y_top']:+}, y_near {d['y_near']:+}, "
                         f"p99 {d['p99_top']:+}/{d['p99_near']:+})")
        lines += ["", "| shot | band | Y_top | Y_near | dY | p99 t/n | sat% t/n |",
                  "|---|---|---|---|---|---|---|"]
        for s in photo["per_shot"]:
            lines.append(f"| {s['shot']} | {s['band_rows'][0]}..{s['band_rows'][1]} "
                         f"| {s['y_top']} | {s['y_near']} | {s['dY']} "
                         f"| {s['p99_top']}/{s['p99_near']} "
                         f"| {s['sat_top_pct']}/{s['sat_near_pct']} |")
    else:
        lines.append("(프레임 없음 — 측광 생략)")
    path.write_text("\n".join(lines) + "\n")


# =========================================================================
# main
# =========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture-dir", required=True,
                    help="street_dataset_sim_capture.py 출력 디렉토리")
    ap.add_argument("--mast", required=True, choices=("up", "down"),
                    help="캡처 당시 마스트 상태 (스티치 캘리브/마운트 선택)")
    ap.add_argument("--out", default="",
                    help="출력 디렉토리 (기본 logs/sim_validation/field_parity_<캡처명>_<mast>)")
    ap.add_argument("--pose-source", choices=("arena", "gt"), default="arena",
                    help="샷별 pose: arena(실기 체인과 동일, 기본) / gt(시뮬 진값)")
    ap.add_argument("--range-mode", choices=("depth", "ground"), default="depth",
                    help="e2e 스캔과 동일 기본 depth 역투영")
    ap.add_argument("--pair", choices=("on", "off"), default="on",
                    help="pair(v2) 재검증 — e2e 기본 on, 로드 실패 시 자동 off")
    ap.add_argument("--votes-k", type=int, default=None)
    ap.add_argument("--fruit-k", type=int, default=None)
    ap.add_argument("--gt-file", default="",
                    help="GT 텍스트 파일로 placement_gt.json 대체 (parse_gt_text 형식)")
    ap.add_argument("--gt-out", default=str(GT_TEXT_DEFAULT),
                    help="e2e --gt-file 용 GT 텍스트 출력 경로")
    ap.add_argument("--include-continuous", action="store_true",
                    help="연속회전(phase=continuous) 샷도 포함 (기본 제외)")
    ap.add_argument("--no-images", action="store_true",
                    help="스티치/오버레이 PNG 저장 생략 (report/summary 만)")
    args = ap.parse_args()

    cap_dir = Path(args.capture_dir)
    if not (cap_dir / "report.json").exists():
        raise SystemExit(f"캡처 report.json 없음: {cap_dir}")
    out_dir = Path(args.out) if args.out else \
        OUT_BASE / f"field_parity_{cap_dir.name}_{args.mast}"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = load_gt_cells(cap_dir, args.gt_file)
    gt_text_path = Path(args.gt_out)
    write_gt_text(gt, gt_text_path)
    print(fl.gt_summary_text(gt), flush=True)

    k_map = load_k_map(cap_dir)
    if set(k_map) != {"top", "near"}:
        raise SystemExit(f"camera K 파일 결손: {sorted(k_map)} (top_K.txt/near_K.txt)")
    caps = load_caps(cap_dir, args.pose_source, args.include_continuous)
    with_frames = [c for c in caps if "frames" in c]
    if not with_frames:
        raise SystemExit("프레임 있는 샷 0 — 캡처 디렉토리/파일명 확인")
    size_wh = (with_frames[0]["frames"][0].shape[1],
               with_frames[0]["frames"][0].shape[0])
    st = make_stitcher(args.mast, size_wh, k_map)
    mounts = {c: e2e.CameraMount(**MOUNTS[args.mast][c]) for c in ("top", "near")}

    # e2e 러너 구성 (ROS 없음). Namespace 는 E2ERunner 가 읽는 필드만 채운다.
    ns = argparse.Namespace(
        votes_k=args.votes_k, fruit_k=args.fruit_k, max_v=None,
        speed_profile="normal", range_mode=args.range_mode, pair=args.pair,
        dry_run=False, capture_dir=str(cap_dir), mast=args.mast,
        pose_source=args.pose_source)
    runner = OfflineParityRunner(ns, gt, out_dir, k_map)
    runner.scan_dir.mkdir(parents=True, exist_ok=True)
    if args.no_images:
        runner._save_shot_images = lambda *a, **kw: None

    from ultralytics import YOLO
    t0 = time.monotonic()
    runner.models = (YOLO(str(fl.A1_WEIGHTS)), YOLO(str(fl.FACE_WEIGHTS)))
    if args.pair == "on":
        try:
            runner.pair = fl.PairVerifier()
        except Exception as e:  # noqa: BLE001 — e2e 와 동일: 없으면 face 단독
            runner.log(f"pair 검증기 로드 실패({e}) — face 단독 모드")
    runner.log(f"모델 로드 {time.monotonic() - t0:.1f}s · 스티치 {st.calib_source} "
               f"· 프레임 {len(with_frames)}/{len(caps)}샷 {size_wh[0]}x{size_wh[1]}")

    # --- 실기 스캔 체인 그대로: A1 배치 → 기하/투표 → face/pair 배치 ---
    shots = runner._process_shots_batch(caps, st, mounts)
    runner.report["scan"]["shots"] = shots
    runner._finalize_scan()          # 셀 확정 + grid_map.json/png + GT 비교
    if args.mast != "up":            # e2e 는 up 스캔 전용이라 mast="up" 고정 기록
        gm_path = out_dir / "grid_map.json"
        gm = json.loads(gm_path.read_text())
        gm["mast"] = args.mast
        gm_path.write_text(json.dumps(gm, ensure_ascii=False, indent=2))

    photo = photometry_report(caps, st)
    table = grid_match_table(runner.cells, gt)
    runner.report["parity"] = {
        "capture_dir": str(cap_dir), "mast": args.mast,
        "pose_source": args.pose_source, "image_size": list(size_wh),
        "camera_k": k_map, "mounts": MOUNTS[args.mast],
        "stitch_calib": st.calib_source,
        "gt_text_path": str(gt_text_path),
        "grid_match": table,
    }
    runner.report["photometry"] = photo

    save_th = getattr(runner, "_save_thread", None)
    if save_th is not None:
        save_th.join(timeout=120.0)
    runner.save_report()             # out_dir/report.json (e2e 스키마 + parity 확장)
    write_summary_md(out_dir / "summary.md", runner, args, st, photo, table,
                     gt_text_path, len(with_frames))

    gtc = runner.report["scan"].get("gt_compare", {})
    print(json.dumps({
        "ok": True, "out": str(out_dir),
        "shots": len(with_frames), "cells": len(runner.cells),
        "gt_ok": gtc.get("ok"), "gt_total": len(gt),
        "wrong": len(gtc.get("wrong", [])), "ghosts": len(gtc.get("ghosts", [])),
        "missed": len(gtc.get("missed", [])),
        "photometry_mean": photo.get("mean"),
        "gt_text": str(gt_text_path),
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
