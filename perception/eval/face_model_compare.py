#!/usr/bin/env python3
"""face 모델 일괄 비교 — A1 검출 1회 캐시 후 face 모델만 스왑 (7/19).

스티치 프레임(런타임 경로, 90번과 동일: imgsz=896, 원본 캠 depth 역매핑,
2.5m 초과 presence 강등) 위에서 A1 검출·셀 스냅·크롭을 **한 번만** 계산해
캐시하고, 후보 face 모델들을 순서대로 캐시 크롭에 적용해 동일 잣대
(면투표 과일우선 conf>=0.3 → 셀 비대칭 투표 0.5/K2)로 채점한다.

산출:
  <out>/crops/crop_XXXX.png            큐브 크롭 (face 입력 원본 해상도)
  <out>/detections.json                검출 캐시 (frame/cell/a1/range/src)
  <out>/results_<model>.json           모델별 per-cell/per-object 채점
  <out>/summary.json                   모델 x 데이터셋 비교표 + per-crop 예측

사용 (torch 필요 — conda trellis):
  face_model_compare.py --models "std=perception/models/unified_face/best.pt;cand=<path>/best.pt;..."
"""
import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geometry import CameraMount, Intrinsics  # noqa: E402
from arena_lightweight_control.competition_layout import (  # noqa: E402
    GRID_XS_CM, GRID_YS_CM)

A1_WEIGHTS = REPO_ROOT / "perception" / "models" / "a1_objectseg" / "best.pt"
FRUITS = {"apple", "orange", "banana", "pineapple"}
POLYHEDRA = {"octahedron", "dodecahedron", "icosahedron"}
TOP_MOUNT = CameraMount(forward_m=0.198, height_m=0.360, tilt_deg=18.5)
NEAR_MOUNT = CameraMount(forward_m=0.1874, height_m=0.268, tilt_deg=54.0)
STITCH_DOWN = dict(x_offset=-3, crop_bottom_from_top=0, crop_top_from_bottom=20,
                   blend_overlap=12)
DEFAULT_DATASETS = ["street_dataset_044654", "street_dataset_050614"]


class Stitcher:
    def __init__(self, p, w=640, h=480):
        x = p["x_offset"]
        self.left = max(0, x)
        self.right = min(w, x + w)
        self.bottom_left = self.left - x
        self.top_keep_h = h - min(p["crop_bottom_from_top"], h - 1) - p["blend_overlap"]
        self.crop_top = p["crop_top_from_bottom"]

    def stitch(self, top_img, bottom_img):
        top_a = top_img[:, self.left:self.right]
        bot_a = bottom_img[:, self.bottom_left:self.bottom_left + (self.right - self.left)]
        return np.vstack((top_a[:self.top_keep_h], bot_a[self.crop_top:]))

    def to_source(self, u, v):
        if v < self.top_keep_h:
            return "top", u + self.left, v
        return "near", u + self.bottom_left, (v - self.top_keep_h) + self.crop_top


def mask_bottom_uv(mask):
    ys, xs = np.nonzero(mask)
    b = ys.max()
    return float(xs[ys >= b - 1].mean()), float(b)


def depth_median(dm, u, v, r=3):
    h, w = dm.shape[:2]
    u0, v0 = int(round(u)), int(round(v))
    p = dm[max(0, v0 - r):min(h, v0 + r + 1), max(0, u0 - r):min(w, u0 + r + 1)].astype(float)
    p = p[p > 0]
    return float(np.median(p)) / 1000.0 if p.size else None


def to_map(xr, yf, pose):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return pose[0] + yf * c + xr * s, pose[1] + yf * s - xr * c


def snap_cm(mx, my):
    cx, cy = (mx + 2.0) * 100.0, (my + 2.0) * 100.0
    gx = min(GRID_XS_CM, key=lambda g: abs(g - cx))
    gy = min(GRID_YS_CM, key=lambda g: abs(g - cy))
    return (gx, gy), math.hypot(cx - gx, cy - gy) / 100.0


def build_cache(datasets, out_dir, args):
    from ultralytics import YOLO
    a1 = YOLO(str(A1_WEIGHTS))
    st = Stitcher(STITCH_DOWN)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    dets = []
    crop_idx = 0
    for ds in datasets:
        d = REPO_ROOT / "logs" / "real_validation" / ds
        for line in (d / "frames.jsonl").read_text().splitlines():
            fr = json.loads(line)
            if fr.get("mode") != "discrete":
                continue
            pose = (float(fr["pose"]["x"]), float(fr["pose"]["y"]), float(fr["pose"]["yaw"]))
            pt_dir = d / f"pt{int(fr['point']):02d}"
            base = f"disc_f{int(fr['frame']):03d}"
            paths = {k: pt_dir / f"{base}_{k}.png" for k in
                     ("top_rgb", "top_depth", "near_rgb", "near_depth")}
            if not all(p.exists() for p in paths.values()):
                continue
            rgb = {c: np.asarray(PILImage.open(paths[f"{c}_rgb"]).convert("RGB"))
                   for c in ("top", "near")}
            depth = {c: np.asarray(PILImage.open(paths[f"{c}_depth"])).astype(np.uint16)
                     for c in ("top", "near")}
            stitched = st.stitch(rgb["top"], rgb["near"])
            bgr = stitched[:, :, ::-1].copy()
            r = a1.predict(bgr, imgsz=args.a1_imgsz, conf=args.conf, verbose=False)[0]
            if r.masks is None:
                continue
            h, w = r.orig_shape
            for mask_t, box in zip(r.masks.data, r.boxes):
                mask = mask_t.cpu().numpy()
                if mask.shape != (h, w):
                    yi = np.linspace(0, mask.shape[0] - 1, h).astype(int)
                    xi = np.linspace(0, mask.shape[1] - 1, w).astype(int)
                    mask = mask[yi][:, xi]
                mask = mask > 0.5
                if not mask.any():
                    continue
                u, v = mask_bottom_uv(mask)
                cam, u_src, v_src = st.to_source(u, v)
                dmt = depth_median(depth[cam], u_src, v_src)
                if dmt is None or not (0.1 < dmt < 4.5):
                    continue
                mount = TOP_MOUNT if cam == "top" else NEAR_MOUNT
                intr = Intrinsics.from_camera_info(
                    [float(x) for x in (d / f"{cam}_K.txt").read_text().split()])
                x_cam = (u_src - intr.cx) / intr.fx * dmt
                y_cam = (v_src - intr.cy) / intr.fy * dmt
                tilt = math.radians(mount.tilt_deg)
                y_fwd = dmt * math.cos(tilt) - y_cam * math.sin(tilt) + mount.forward_m
                cell, snap_err = snap_cm(*to_map(x_cam, y_fwd, pose))
                if snap_err > args.max_snap_err:
                    continue
                cls = r.names[int(box.cls)]
                entry = {"dataset": ds, "frame": f"{fr['point']}:{fr['frame']}",
                         "cell": list(cell), "a1": cls, "a1_conf": round(float(box.conf), 3),
                         "range_m": round(dmt, 2), "src_cam": cam,
                         "far": bool(args.max_range_m > 0 and dmt > args.max_range_m)}
                if cls == "cube_like_object":
                    x1, y1, x2, y2 = [int(t) for t in box.xyxy[0].tolist()]
                    pad = int(max(x2 - x1, y2 - y1) * args.crop_pad)
                    crop = bgr[max(0, y1 - pad):min(h, y2 + pad),
                               max(0, x1 - pad):min(w, x2 + pad)]
                    if crop.size:
                        # 테두리 접촉 여부(예외처리 후보 마킹 — 과일면 잘림 가능)
                        entry["edge_touch"] = bool(x1 <= 1 or y1 <= 1
                                                   or x2 >= w - 2 or y2 >= h - 2)
                        cp = crops_dir / f"crop_{crop_idx:04d}.png"
                        PILImage.fromarray(crop[:, :, ::-1]).save(cp)
                        entry["crop"] = cp.name
                        crop_idx += 1
                dets.append(entry)
    (out_dir / "detections.json").write_text(json.dumps(dets, indent=1))
    return dets


def face_vote(faces, face_fruit_min):
    """faces=[(name,conf)...] -> (identity, conf) 면투표(과일 우선)."""
    if not faces:
        return None, None
    fruit = [(n, c) for n, c in faces if n in FRUITS and c >= face_fruit_min]
    return max(fruit, key=lambda t: t[1]) if fruit else max(faces, key=lambda t: t[1])


def evaluate_model(name, weights, dets, out_dir, truths, args):
    from ultralytics import YOLO
    face = YOLO(str(weights))
    crops_dir = out_dir / "crops"
    # 크롭 배치 face 추론 (런타임 계약과 동일: 배치 1회 호출 단위로 처리)
    crop_entries = [e for e in dets if e.get("crop")]
    B = 32
    per_crop = {}
    for i in range(0, len(crop_entries), B):
        batch = crop_entries[i:i + B]
        imgs = [np.asarray(PILImage.open(crops_dir / e["crop"]).convert("RGB"))[:, :, ::-1]
                for e in batch]
        results = face.predict(imgs, imgsz=224, conf=0.1, verbose=False)
        for e, r in zip(batch, results):
            faces = sorted([(r.names[int(b.cls)], round(float(b.conf), 3))
                            for b in (r.boxes or [])], key=lambda t: -t[1])
            per_crop[e["crop"]] = faces
    # 셀 투표 (90과 동일 잣대)
    out = {}
    for ds, truth in truths.items():
        cell_votes = defaultdict(list)
        for e in dets:
            if e["dataset"] != ds or e["far"]:
                continue
            ident, fc = e["a1"], None
            if e.get("crop"):
                ident2, fc = face_vote(per_crop[e["crop"]], args.face_fruit_min)
                if ident2 is not None:
                    ident = ident2
            cell_votes[tuple(e["cell"])].append(
                {"identity": ident, "face_conf": fc, "a1_conf": e["a1_conf"]})
        cells = {}
        for cell, votes in cell_votes.items():
            if len(votes) < args.votes_min:
                continue
            fruit_hits = Counter(v["identity"] for v in votes
                                 if v["identity"] in FRUITS
                                 and (v["face_conf"] or 0) >= args.fruit_conf)
            ident = None
            if fruit_hits:
                best, k = fruit_hits.most_common(1)[0]
                if k >= args.fruit_k:
                    ident = best
            if ident is None:
                tally = defaultdict(float)
                for v in votes:
                    lbl = v["identity"] if v["identity"] in POLYHEDRA else (
                        "plain" if v["identity"] in FRUITS | {"plain"} else v["identity"])
                    tally[lbl] += v["a1_conf"]
                ident = max(tally.items(), key=lambda kv: kv[1])[0]
            cells[cell] = {"identity": ident, "votes": len(votes),
                           "fruit_hits": dict(fruit_hits)}

        def ok(p, t):
            return p == t or (t == "plain" and p == "plain")
        per_obj = []
        for cell, true in sorted(truth.items()):
            r_ = cells.get(cell)
            per_obj.append({"cell": list(cell), "true": true,
                            "pred": r_["identity"] if r_ else None,
                            "ok": bool(r_ and ok(r_["identity"], true)),
                            "votes": r_["votes"] if r_ else 0,
                            "fruit_hits": r_["fruit_hits"] if r_ else {}})
        fruits = [o for o in per_obj if o["true"] in FRUITS]
        wrong_fruit = [o for o in per_obj if o["true"] in FRUITS and o["pred"] in FRUITS
                       and o["pred"] != o["true"]]
        ghost_fruit = [o for o in per_obj if o["true"] == "plain" and o["pred"] in FRUITS]
        out[ds] = {
            "class_ok": sum(1 for o in per_obj if o["ok"]),
            "gt": len(per_obj),
            "fruit_ok": sum(1 for o in fruits if o["ok"]),
            "fruit_n": len(fruits),
            "wrong_fruit_confirm": len(wrong_fruit) + len(ghost_fruit),
            "ghosts": len([c for c in cells if c not in truth]),
            "per_object": per_obj,
        }
    (out_dir / f"results_{name}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return {"per_crop": per_crop, "cells": out}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", required=True,
                    help='"이름=weights경로;이름=경로;..."')
    ap.add_argument("--datasets", default=";".join(DEFAULT_DATASETS))
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "logs" / "real_validation" / "face_model_compare")
    ap.add_argument("--a1-imgsz", type=int, default=896)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--crop-pad", type=float, default=0.18)
    ap.add_argument("--votes-min", type=int, default=3)
    ap.add_argument("--fruit-k", type=int, default=2)
    ap.add_argument("--fruit-conf", type=float, default=0.5)
    ap.add_argument("--face-fruit-min", type=float, default=0.30)
    ap.add_argument("--max-snap-err", type=float, default=0.30)
    ap.add_argument("--max-range-m", type=float, default=2.5)
    ap.add_argument("--reuse-cache", action="store_true")
    args = ap.parse_args()

    datasets = [d for d in args.datasets.split(";") if d]
    args.out.mkdir(parents=True, exist_ok=True)
    truths = {}
    for ds in datasets:
        man = json.loads((REPO_ROOT / "logs" / "real_validation" / ds / "manifest.json")
                         .read_text())
        pl = man.get("placement", man)
        truths[ds] = {(int(o["x_cm"]), int(o["y_cm"])): o["cls"] for o in pl["objects"]}

    if args.reuse_cache and (args.out / "detections.json").exists():
        dets = json.loads((args.out / "detections.json").read_text())
        print(f"cache reused: {len(dets)} detections")
    else:
        dets = build_cache(datasets, args.out, args)
        print(f"cache built: {len(dets)} detections "
              f"({sum(1 for e in dets if e.get('crop'))} cube crops, "
              f"edge_touch {sum(1 for e in dets if e.get('edge_touch'))})")

    models = {}
    for tok in args.models.split(";"):
        if tok:
            n, p = tok.split("=", 1)
            models[n] = p
    summary = {"params": {k: v for k, v in vars(args).items()
                          if k not in ("models", "out")},
               "datasets": datasets, "models": {}, "crop_predictions": {}}
    for name, w in models.items():
        res = evaluate_model(name, w, dets, args.out, truths, args)
        summary["models"][name] = {ds: {k: v for k, v in r.items() if k != "per_object"}
                                   for ds, r in res["cells"].items()}
        summary["crop_predictions"][name] = res["per_crop"]
        line = " | ".join(
            f"{ds.split('_')[-1]}: {r['class_ok']}/{r['gt']} fruit {r['fruit_ok']}/{r['fruit_n']} "
            f"wrongF {r['wrong_fruit_confirm']}" for ds, r in res["cells"].items())
        print(f"[{name}] {line}", flush=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"out: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
