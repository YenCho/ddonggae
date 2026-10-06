from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

JETSON_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = JETSON_ROOT.parent
sys.path.insert(0, str(JETSON_ROOT))

from abc_inference import ABCPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ABC runtime variants on fixed Meta V2 samples.")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/meta_v2_50000_coco_texture_v1"))
    parser.add_argument("--output", type=Path, default=Path("reports/abc_runtime_optimization/exp001_variants.json"))
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--cube-count", type=int, default=4)
    parser.add_argument("--face-count", type=int, default=12)
    parser.add_argument("--meta-scan-limit", type=int, default=20000)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--a2-imgsz", type=int, default=224)
    parser.add_argument("--target-shape", default="cube")
    parser.add_argument("--target-fruit", default="apple")
    parser.add_argument("--disable-stage-timing", action="store_true")
    parser.add_argument("--a1-model", type=Path, default=Path("runs/meta_v2_a1_objectseg/a1_yolo26s_seg_meta_v2_50000/weights/best.pt"))
    parser.add_argument("--a2-model", type=Path, default=Path("runs/meta_v2_a2_faceseg/a2_yolo26n_seg_meta_v2_50000/weights/best.pt"))
    parser.add_argument("--a1-onnx-model", type=Path, default=Path("reports/abc_runtime_optimization/artifacts/a1_yolo26s_seg_meta_v2_50000.onnx"))
    parser.add_argument("--a2-onnx-model", type=Path, default=Path("reports/abc_runtime_optimization/artifacts/a2_yolo26n_seg_meta_v2_50000.onnx"))
    parser.add_argument("--a1-engine-model", type=Path, default=Path("reports/abc_runtime_optimization/artifacts/a1_yolo26s_seg_meta_v2_50000_fp32_dynamic_b1.engine"))
    parser.add_argument("--a2-engine-model", type=Path, default=Path("reports/abc_runtime_optimization/artifacts/a2_yolo26n_seg_meta_v2_50000_fp32_dynamic_b8.engine"))
    parser.add_argument("--b-model", type=Path, default=Path("runs/meta_v2_b_tinyquadnet/b_tinyquadnet_meta_v2_50000/weights/best.pt"))
    parser.add_argument("--c-model", type=Path, default=Path("runs/meta_v2_c_facecls/c_mobilenetv3small_meta_v2_50000_runtimewarp_warmplain_ft/weights/last.pt"))
    parser.add_argument("--c-onnx-model", type=Path, default=Path("reports/abc_runtime_optimization/artifacts/c_mobilenetv3small_runtimewarp_warmplain_ft.onnx"))
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "c_warp128", "no_blacken", "no_refine", "fast_combo"],
        choices=[
            "baseline",
            "c_warp128",
            "no_blacken",
            "no_refine",
            "fast_combo",
            "a1_512",
            "a2_192",
            "a1_512_a2_192",
            "a1_512_a2_192_cwarp128",
            "trace_bc",
            "yolo_half",
            "yolo_half_trace_bc",
            "keep_c_inputs",
            "bbox_only_a1",
            "bbox_only_a1_cwarp128",
            "c_onnx_cpu",
            "c_onnx_cpu_cwarp128",
            "c_onnx_cpu_bbox_cwarp128",
            "c_onnx_cpu_bbox_cwarp128_frame_b",
            "c_onnx_cpu_bbox_cwarp128_frame_bc",
            "c_onnx_cpu_bbox_cwarp128_cuda_fast",
            "c_onnx_cpu_bbox_cwarp128_trace_b",
            "c_onnx_cpu_bbox_cwarp128_b_cpu",
            "c_onnx_cpu_bbox_cwarp128_a2_192",
            "c_onnx_cpu_bbox_cwarp128_pose_fruit",
            "c_onnx_cpu_bbox_cwarp128_pose_fast_iou",
            "c_onnx_cpu_bbox_cwarp128_no_refine",
            "c_onnx_cpu_bbox_cwarp128_fruit_only",
            "c_onnx_cpu_bbox_cwarp128_plain85",
            "c_onnx_cpu_bbox_cwarp128_plain90",
            "c_onnx_cpu_bbox_cwarp128_plain95",
            "c_onnx_cpu_bbox_cwarp128_no_blacken",
            "c_onnx_cpu_bbox_cwarp96",
            "c_onnx_cpu_bbox_cwarp112",
            "c_onnx_cpu_bbox_cwarp88",
            "c_onnx_cpu_bbox_cwarp88_fruit_only",
            "c_onnx_cpu_bbox_cwarp88_no_blacken",
            "c_onnx_cpu_bbox_cwarp80",
            "c_onnx_cpu_bbox_cwarp64",
            "c_onnx_cuda",
            "c_onnx_cuda_cwarp128",
            "c_onnx_cuda_bbox_cwarp128",
            "c_onnx_trt",
            "c_onnx_trt_cwarp128",
            "c_onnx_trt_bbox_cwarp128",
            "a1_onnx",
            "a2_onnx",
            "a1_a2_onnx",
            "a1_a2_onnx_best_c",
            "a1_engine",
            "a1_engine_best_c",
            "a1_engine_best_c_frame_b",
            "a1_engine_best_c_frame_bc",
            "a1_engine_best_c_frame_bc_reindex",
            "a1_engine_best_c_frame_bc_cwarp112",
            "a1_engine_best_c_frame_bc_cwarp96",
            "a1_engine_best_c_frame_bc_reindex_cwarp112",
            "a1_engine_best_c_frame_bc_reindex_cwarp96",
            "a1_engine_best_c_frame_bc_pose_fruit",
            "a1_engine_best_c_frame_bc_pose_fruit_cwarp112",
            "a1_engine_best_c_frame_bc_pose_fruit_cwarp96",
            "a1_engine_best_c_frame_bc_pose_fruit_cconf45",
            "a1_engine_best_c_frame_bc_pose_fruit_cconf35",
            "a1_engine_best_c_frame_bc_pose_fast_iou",
            "a1_engine_best_c_frame_bc_no_refine",
            "a1_engine_best_c_frame_bc_cuda_fast",
            "a1_engine_best_c_frame_bc_yolo_half",
            "a1_engine_best_c_frame_bc_yolo_half_cuda_fast",
            "a1_a2_engine_best_c_frame_bc",
            "a1_a2_engine_best_c_frame_bc_cwarp112",
            "a1_a2_engine_best_c_frame_bc_cwarp96",
            "a1_a2_engine_best_c_frame_bc_a2conf15",
            "a1_a2_engine_best_c_frame_bc_a2conf25",
            "a1_a2_engine_best_c_frame_bc_a2conf30",
            "a1_a2_engine_best_c_frame_bc_cconf45",
            "a1_a2_engine_best_c_frame_bc_cconf35",
            "a1_a2_engine_best_c_frame_bc_pose_fruit",
            "a1_a2_engine_best_c_frame_bc_pose_fast_iou",
            "a1_engine_best_c_cuda_fast",
            "a1_engine_best_c_trace_b",
            "a1_engine_best_c_b_cpu",
            "a1_engine_best_c_a2_192",
            "a1_engine_best_c_pose_fruit",
            "a1_engine_best_c_pose_fast_iou",
            "a1_engine_best_c_no_refine",
            "a1_a2_engine_best_c",
            "a2_engine",
            "a2_engine_best_c",
        ],
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * p))))
    return float(values[idx])


def choose_samples(dataset: Path, cube_count: int, face_count: int, count: int, limit: int) -> list[str]:
    meta_dir = dataset / "_meta" / "train"
    rows: list[tuple[int, int, int, str]] = []
    for idx, path in enumerate(sorted(meta_dir.glob("*.json"))):
        if idx >= limit:
            break
        data = json.loads(path.read_text(encoding="utf-8"))
        meta_v2 = data.get("meta_v2") or {}
        cubes = 0
        good_faces = 0
        visible_pixels = 0
        for obj in meta_v2.get("objects") or []:
            if int(obj.get("a1_class_id", -1)) != 0:
                continue
            cubes += 1
            for face in obj.get("faces") or []:
                pixels = int(face.get("visible_pixels") or 0)
                if face.get("quad_xy") and pixels >= 120:
                    good_faces += 1
                    visible_pixels += pixels
        if cubes == cube_count and good_faces >= max(1, face_count - 1):
            rows.append((abs(good_faces - face_count), -visible_pixels, good_faces, path.stem))

    rows.sort()
    if len(rows) < count:
        raise RuntimeError(f"not enough samples: found {len(rows)}, need {count}")
    return [row[3] for row in rows[:count]]


def load_frames(dataset: Path, stems: list[str]) -> list[tuple[str, np.ndarray]]:
    image_dir = dataset / "images" / "train"
    frames: list[tuple[str, np.ndarray]] = []
    for stem in stems:
        path = image_dir / f"{stem}.jpg"
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image: {path}")
        frames.append((stem, image))
    return frames


def variant_kwargs(name: str) -> dict[str, Any]:
    if name == "baseline":
        return {}
    if name == "c_warp128":
        return {"c_warp_size": 128}
    if name == "no_blacken":
        return {"blacken_c_occluded_face_area": False}
    if name == "no_refine":
        return {"refine_quads": "none"}
    if name == "fast_combo":
        return {
            "c_warp_size": 128,
            "blacken_c_occluded_face_area": False,
            "refine_quads": "none",
        }
    if name == "a1_512":
        return {"imgsz": 512}
    if name == "a2_192":
        return {"a2_imgsz": 192}
    if name == "a1_512_a2_192":
        return {"imgsz": 512, "a2_imgsz": 192}
    if name == "a1_512_a2_192_cwarp128":
        return {"imgsz": 512, "a2_imgsz": 192, "c_warp_size": 128}
    if name == "trace_bc":
        return {"trace_torch_models": True}
    if name == "yolo_half":
        return {"yolo_half": True}
    if name == "yolo_half_trace_bc":
        return {"yolo_half": True, "trace_torch_models": True}
    if name == "keep_c_inputs":
        return {"keep_c_inputs": True}
    if name == "bbox_only_a1":
        return {"use_a1_object_mask": False}
    if name == "bbox_only_a1_cwarp128":
        return {"use_a1_object_mask": False, "c_warp_size": 128}
    if name == "c_onnx_cpu":
        return {"c_onnx_provider": "cpu"}
    if name == "c_onnx_cpu_cwarp128":
        return {"c_onnx_provider": "cpu", "c_warp_size": 128}
    if name == "c_onnx_cpu_bbox_cwarp128":
        return {"c_onnx_provider": "cpu", "use_a1_object_mask": False, "c_warp_size": 128}
    if name == "c_onnx_cpu_bbox_cwarp128_frame_b":
        return {
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "batch_b_across_objects": True,
        }
    if name == "c_onnx_cpu_bbox_cwarp128_frame_bc":
        return {
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "batch_b_across_objects": True,
            "batch_c_across_objects": True,
        }
    if name == "c_onnx_cpu_bbox_cwarp128_cuda_fast":
        return {
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "torch_cuda_fast_path": True,
        }
    if name == "c_onnx_cpu_bbox_cwarp128_trace_b":
        return {
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "trace_torch_models": True,
        }
    if name == "c_onnx_cpu_bbox_cwarp128_b_cpu":
        return {"c_onnx_provider": "cpu", "use_a1_object_mask": False, "c_warp_size": 128, "b_device": "cpu"}
    if name == "c_onnx_cpu_bbox_cwarp128_a2_192":
        return {"c_onnx_provider": "cpu", "use_a1_object_mask": False, "c_warp_size": 128, "a2_imgsz": 192}
    if name == "c_onnx_cpu_bbox_cwarp128_pose_fruit":
        return {
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "refine_quads": "pose_fruit",
        }
    if name == "c_onnx_cpu_bbox_cwarp128_pose_fast_iou":
        return {
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "refine_quads": "pose_fast_iou",
        }
    if name == "c_onnx_cpu_bbox_cwarp128_no_refine":
        return {
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "refine_quads": "none",
        }
    if name == "c_onnx_cpu_bbox_cwarp128_fruit_only":
        return {
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "classify_fruit_faces_only": True,
        }
    if name == "c_onnx_cpu_bbox_cwarp128_plain85":
        return {
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "classify_fruit_faces_only": True,
            "plain_face_fastpath_min_conf": 0.85,
        }
    if name == "c_onnx_cpu_bbox_cwarp128_plain90":
        return {
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "classify_fruit_faces_only": True,
            "plain_face_fastpath_min_conf": 0.90,
        }
    if name == "c_onnx_cpu_bbox_cwarp128_plain95":
        return {
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "classify_fruit_faces_only": True,
            "plain_face_fastpath_min_conf": 0.95,
        }
    if name == "c_onnx_cpu_bbox_cwarp128_no_blacken":
        return {
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "blacken_c_occluded_face_area": False,
        }
    if name == "c_onnx_cpu_bbox_cwarp96":
        return {"c_onnx_provider": "cpu", "use_a1_object_mask": False, "c_warp_size": 96}
    if name == "c_onnx_cpu_bbox_cwarp112":
        return {"c_onnx_provider": "cpu", "use_a1_object_mask": False, "c_warp_size": 112}
    if name == "c_onnx_cpu_bbox_cwarp88":
        return {"c_onnx_provider": "cpu", "use_a1_object_mask": False, "c_warp_size": 88}
    if name == "c_onnx_cpu_bbox_cwarp88_fruit_only":
        return {
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 88,
            "classify_fruit_faces_only": True,
        }
    if name == "c_onnx_cpu_bbox_cwarp88_no_blacken":
        return {
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 88,
            "blacken_c_occluded_face_area": False,
        }
    if name == "c_onnx_cpu_bbox_cwarp80":
        return {"c_onnx_provider": "cpu", "use_a1_object_mask": False, "c_warp_size": 80}
    if name == "c_onnx_cpu_bbox_cwarp64":
        return {"c_onnx_provider": "cpu", "use_a1_object_mask": False, "c_warp_size": 64}
    if name == "c_onnx_cuda":
        return {"c_onnx_provider": "cuda"}
    if name == "c_onnx_cuda_cwarp128":
        return {"c_onnx_provider": "cuda", "c_warp_size": 128}
    if name == "c_onnx_cuda_bbox_cwarp128":
        return {"c_onnx_provider": "cuda", "use_a1_object_mask": False, "c_warp_size": 128}
    if name == "c_onnx_trt":
        return {"c_onnx_provider": "tensorrt"}
    if name == "c_onnx_trt_cwarp128":
        return {"c_onnx_provider": "tensorrt", "c_warp_size": 128}
    if name == "c_onnx_trt_bbox_cwarp128":
        return {"c_onnx_provider": "tensorrt", "use_a1_object_mask": False, "c_warp_size": 128}
    if name == "a1_onnx":
        return {"use_a1_onnx_model": True}
    if name == "a2_onnx":
        return {"use_a2_onnx_model": True}
    if name == "a1_a2_onnx":
        return {"use_a1_onnx_model": True, "use_a2_onnx_model": True}
    if name == "a1_a2_onnx_best_c":
        return {
            "use_a1_onnx_model": True,
            "use_a2_onnx_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
        }
    if name == "a1_engine":
        return {"use_a1_engine_model": True}
    if name == "a1_engine_best_c":
        return {
            "use_a1_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
        }
    if name == "a1_engine_best_c_frame_b":
        return {
            "use_a1_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "batch_b_across_objects": True,
        }
    if name == "a1_engine_best_c_frame_bc":
        return {
            "use_a1_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "batch_b_across_objects": True,
            "batch_c_across_objects": True,
        }
    if name == "a1_engine_best_c_frame_bc_reindex":
        return {
            "use_a1_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "batch_b_across_objects": True,
            "batch_c_across_objects": True,
            "final_face_filter_mode": "reindex",
        }
    if name == "a1_engine_best_c_frame_bc_cwarp112":
        return {
            "use_a1_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 112,
            "batch_b_across_objects": True,
            "batch_c_across_objects": True,
        }
    if name == "a1_engine_best_c_frame_bc_cwarp96":
        return {
            "use_a1_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 96,
            "batch_b_across_objects": True,
            "batch_c_across_objects": True,
        }
    if name == "a1_engine_best_c_frame_bc_reindex_cwarp112":
        return {
            "use_a1_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 112,
            "batch_b_across_objects": True,
            "batch_c_across_objects": True,
            "final_face_filter_mode": "reindex",
        }
    if name == "a1_engine_best_c_frame_bc_reindex_cwarp96":
        return {
            "use_a1_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 96,
            "batch_b_across_objects": True,
            "batch_c_across_objects": True,
            "final_face_filter_mode": "reindex",
        }
    if name == "a1_engine_best_c_frame_bc_pose_fruit":
        return {
            "use_a1_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "batch_b_across_objects": True,
            "batch_c_across_objects": True,
            "refine_quads": "pose_fruit",
        }
    if name == "a1_engine_best_c_frame_bc_pose_fruit_cwarp112":
        return {
            "use_a1_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 112,
            "batch_b_across_objects": True,
            "batch_c_across_objects": True,
            "refine_quads": "pose_fruit",
        }
    if name == "a1_engine_best_c_frame_bc_pose_fruit_cwarp96":
        return {
            "use_a1_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 96,
            "batch_b_across_objects": True,
            "batch_c_across_objects": True,
            "refine_quads": "pose_fruit",
        }
    if name == "a1_engine_best_c_frame_bc_pose_fruit_cconf45":
        return {
            "use_a1_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "batch_b_across_objects": True,
            "batch_c_across_objects": True,
            "refine_quads": "pose_fruit",
            "c_conf": 0.45,
        }
    if name == "a1_engine_best_c_frame_bc_pose_fruit_cconf35":
        return {
            "use_a1_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "batch_b_across_objects": True,
            "batch_c_across_objects": True,
            "refine_quads": "pose_fruit",
            "c_conf": 0.35,
        }
    if name == "a1_engine_best_c_frame_bc_pose_fast_iou":
        return {
            "use_a1_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "batch_b_across_objects": True,
            "batch_c_across_objects": True,
            "refine_quads": "pose_fast_iou",
        }
    if name == "a1_engine_best_c_frame_bc_no_refine":
        return {
            "use_a1_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "batch_b_across_objects": True,
            "batch_c_across_objects": True,
            "refine_quads": "none",
        }
    if name == "a1_engine_best_c_frame_bc_cuda_fast":
        return {
            "use_a1_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "batch_b_across_objects": True,
            "batch_c_across_objects": True,
            "torch_cuda_fast_path": True,
        }
    if name == "a1_engine_best_c_frame_bc_yolo_half":
        return {
            "use_a1_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "batch_b_across_objects": True,
            "batch_c_across_objects": True,
            "yolo_half": True,
        }
    if name == "a1_engine_best_c_frame_bc_yolo_half_cuda_fast":
        return {
            "use_a1_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "batch_b_across_objects": True,
            "batch_c_across_objects": True,
            "yolo_half": True,
            "torch_cuda_fast_path": True,
        }
    if name == "a1_a2_engine_best_c_frame_bc":
        return {
            "use_a1_engine_model": True,
            "use_a2_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "batch_b_across_objects": True,
            "batch_c_across_objects": True,
        }
    if name == "a1_a2_engine_best_c_frame_bc_cwarp112":
        return {
            "use_a1_engine_model": True,
            "use_a2_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 112,
            "batch_b_across_objects": True,
            "batch_c_across_objects": True,
        }
    if name == "a1_a2_engine_best_c_frame_bc_cwarp96":
        return {
            "use_a1_engine_model": True,
            "use_a2_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 96,
            "batch_b_across_objects": True,
            "batch_c_across_objects": True,
        }
    if name == "a1_a2_engine_best_c_frame_bc_a2conf15":
        return {
            "use_a1_engine_model": True,
            "use_a2_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "batch_b_across_objects": True,
            "batch_c_across_objects": True,
            "a2_conf": 0.15,
        }
    if name == "a1_a2_engine_best_c_frame_bc_a2conf25":
        return {
            "use_a1_engine_model": True,
            "use_a2_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "batch_b_across_objects": True,
            "batch_c_across_objects": True,
            "a2_conf": 0.25,
        }
    if name == "a1_a2_engine_best_c_frame_bc_a2conf30":
        return {
            "use_a1_engine_model": True,
            "use_a2_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "batch_b_across_objects": True,
            "batch_c_across_objects": True,
            "a2_conf": 0.30,
        }
    if name == "a1_a2_engine_best_c_frame_bc_cconf45":
        return {
            "use_a1_engine_model": True,
            "use_a2_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "batch_b_across_objects": True,
            "batch_c_across_objects": True,
            "c_conf": 0.45,
        }
    if name == "a1_a2_engine_best_c_frame_bc_cconf35":
        return {
            "use_a1_engine_model": True,
            "use_a2_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "batch_b_across_objects": True,
            "batch_c_across_objects": True,
            "c_conf": 0.35,
        }
    if name == "a1_a2_engine_best_c_frame_bc_pose_fruit":
        return {
            "use_a1_engine_model": True,
            "use_a2_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "batch_b_across_objects": True,
            "batch_c_across_objects": True,
            "refine_quads": "pose_fruit",
        }
    if name == "a1_a2_engine_best_c_frame_bc_pose_fast_iou":
        return {
            "use_a1_engine_model": True,
            "use_a2_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "batch_b_across_objects": True,
            "batch_c_across_objects": True,
            "refine_quads": "pose_fast_iou",
        }
    if name == "a1_engine_best_c_cuda_fast":
        return {
            "use_a1_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "torch_cuda_fast_path": True,
        }
    if name == "a1_engine_best_c_trace_b":
        return {
            "use_a1_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "trace_torch_models": True,
        }
    if name == "a1_engine_best_c_b_cpu":
        return {
            "use_a1_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "b_device": "cpu",
        }
    if name == "a1_engine_best_c_a2_192":
        return {
            "use_a1_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "a2_imgsz": 192,
        }
    if name == "a1_engine_best_c_pose_fruit":
        return {
            "use_a1_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "refine_quads": "pose_fruit",
        }
    if name == "a1_engine_best_c_pose_fast_iou":
        return {
            "use_a1_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "refine_quads": "pose_fast_iou",
        }
    if name == "a1_engine_best_c_no_refine":
        return {
            "use_a1_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
            "refine_quads": "none",
        }
    if name == "a1_a2_engine_best_c":
        return {
            "use_a1_engine_model": True,
            "use_a2_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
        }
    if name == "a2_engine":
        return {"use_a2_engine_model": True}
    if name == "a2_engine_best_c":
        return {
            "use_a2_engine_model": True,
            "c_onnx_provider": "cpu",
            "use_a1_object_mask": False,
            "c_warp_size": 128,
        }
    raise ValueError(name)


def output_signature(output: Any) -> list[dict[str, Any]]:
    signature = []
    for obj in output.objects:
        decision = obj.decision
        signature.append(
            {
                "class": obj.class_name,
                "action": decision.action if decision else "",
                "identity": decision.identity if decision else "",
                "faces": [(face.kind, face.label) for face in obj.faces],
            }
        )
    return signature


def face_labels(output: Any) -> list[str]:
    labels = []
    for obj in output.objects:
        for face in obj.faces:
            labels.append(str(face.label))
    return labels


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)) if values else 0.0,
        "median": float(statistics.median(values)) if values else 0.0,
        "p90": percentile(values, 0.90),
        "min": float(min(values)) if values else 0.0,
        "max": float(max(values)) if values else 0.0,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "measured_ms",
        "a1_ms",
        "a2_ms",
        "b_ms",
        "c_ms",
        "c_source_ms",
        "c_warp_ms",
        "c_filter_ms",
        "c_preprocess_ms",
        "a1_post_ms",
        "a2_prepare_ms",
        "a2_post_ms",
        "a2_face_parse_ms",
        "a2_face_build_ms",
        "a2_face_filter_ms",
        "a2_face_dedupe_ms",
        "a2_refine_sane_ms",
        "a2_refine_ms",
        "c_crop_loop_ms",
        "c_label_apply_ms",
        "model_sum_ms",
        "other_ms",
        "objects",
        "faces",
        "a2_inputs",
        "b_faces",
        "c_faces",
        "c_skipped_plain_faces",
    ]
    summary = {key: summarize([float(row.get(key, 0.0)) for row in rows]) for key in keys}
    summary["fps"] = summarize([1000.0 / max(float(row["measured_ms"]), 1e-9) for row in rows])
    return summary


def make_pipeline(args: argparse.Namespace, variant_name: str) -> ABCPipeline:
    kwargs = {
        "a1_model": resolve(args.a1_model),
        "a2_model": resolve(args.a2_model),
        "b_model": resolve(args.b_model),
        "c_model": resolve(args.c_model),
        "device": args.device,
        "imgsz": args.imgsz,
        "a2_imgsz": args.a2_imgsz,
        "target_shape": args.target_shape,
        "target_fruit": args.target_fruit,
        "enable_timing": not args.disable_stage_timing,
    }
    kwargs.update(variant_kwargs(variant_name))
    if kwargs.pop("use_a1_onnx_model", False):
        kwargs["a1_model"] = resolve(args.a1_onnx_model)
    if kwargs.pop("use_a2_onnx_model", False):
        kwargs["a2_model"] = resolve(args.a2_onnx_model)
    if kwargs.pop("use_a1_engine_model", False):
        kwargs["a1_model"] = resolve(args.a1_engine_model)
    if kwargs.pop("use_a2_engine_model", False):
        kwargs["a2_model"] = resolve(args.a2_engine_model)
    if "c_onnx_provider" in kwargs:
        kwargs["c_onnx_model"] = resolve(args.c_onnx_model)
    return ABCPipeline(**kwargs)


def benchmark_variant(
    args: argparse.Namespace,
    variant_name: str,
    frames: list[tuple[str, np.ndarray]],
) -> dict[str, Any]:
    pipeline = make_pipeline(args, variant_name)
    for idx in range(args.warmup):
        pipeline.process_frame(frames[idx % len(frames)][1])
        sync()

    rows = []
    signatures: dict[str, Any] = {}
    labels: dict[str, list[str]] = {}
    for _rep in range(args.repeats):
        for stem, frame in frames:
            sync()
            start = time.perf_counter()
            output = pipeline.process_frame(frame)
            sync()
            measured_ms = (time.perf_counter() - start) * 1000.0
            timing = dict(pipeline.last_timing)
            model_sum = sum(float(timing.get(key, 0.0)) for key in ("a1_ms", "a2_ms", "b_ms", "c_ms"))
            timing.update(
                {
                    "stem": stem,
                    "measured_ms": measured_ms,
                    "model_sum_ms": model_sum,
                    "other_ms": measured_ms - model_sum,
                    "objects": len(output.objects),
                    "faces": sum(len(obj.faces) for obj in output.objects),
                }
            )
            rows.append(timing)
            signatures[stem] = output_signature(output)
            labels[stem] = face_labels(output)

    return {
        "variant": variant_name,
        "kwargs": variant_kwargs(variant_name),
        "summary": summarize_rows(rows),
        "rows": rows,
        "signatures": signatures,
        "face_labels": labels,
    }


def compare_behavior(baseline: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    stems = sorted(baseline["signatures"].keys())
    exact_matches = 0
    label_matches = 0
    label_total = 0
    mismatches = []
    for stem in stems:
        base_sig = baseline["signatures"].get(stem)
        var_sig = variant["signatures"].get(stem)
        if base_sig == var_sig:
            exact_matches += 1
        else:
            mismatches.append(stem)
        base_labels = baseline["face_labels"].get(stem, [])
        var_labels = variant["face_labels"].get(stem, [])
        for idx in range(min(len(base_labels), len(var_labels))):
            label_total += 1
            if base_labels[idx] == var_labels[idx]:
                label_matches += 1
        label_total += abs(len(base_labels) - len(var_labels))
    return {
        "exact_frame_match_rate": exact_matches / max(len(stems), 1),
        "face_label_match_rate": label_matches / max(label_total, 1),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
    }


def compact_summary(result: dict[str, Any]) -> dict[str, Any]:
    summary = result["summary"]
    return {
        "variant": result["variant"],
        "mean_ms": round(summary["measured_ms"]["mean"], 3),
        "median_ms": round(summary["measured_ms"]["median"], 3),
        "mean_fps": round(summary["fps"]["mean"], 3),
        "median_fps": round(summary["fps"]["median"], 3),
        "a1_ms": round(summary["a1_ms"]["mean"], 3),
        "a2_ms": round(summary["a2_ms"]["mean"], 3),
        "b_ms": round(summary["b_ms"]["mean"], 3),
        "c_ms": round(summary["c_ms"]["mean"], 3),
        "c_source_ms": round(summary["c_source_ms"]["mean"], 3),
        "c_warp_ms": round(summary["c_warp_ms"]["mean"], 3),
        "c_filter_ms": round(summary["c_filter_ms"]["mean"], 3),
        "c_preprocess_ms": round(summary["c_preprocess_ms"]["mean"], 3),
        "a1_post_ms": round(summary["a1_post_ms"]["mean"], 3),
        "a2_prepare_ms": round(summary["a2_prepare_ms"]["mean"], 3),
        "a2_post_ms": round(summary["a2_post_ms"]["mean"], 3),
        "a2_face_parse_ms": round(summary["a2_face_parse_ms"]["mean"], 3),
        "a2_face_build_ms": round(summary["a2_face_build_ms"]["mean"], 3),
        "a2_face_filter_ms": round(summary["a2_face_filter_ms"]["mean"], 3),
        "a2_face_dedupe_ms": round(summary["a2_face_dedupe_ms"]["mean"], 3),
        "a2_refine_sane_ms": round(summary["a2_refine_sane_ms"]["mean"], 3),
        "a2_refine_ms": round(summary["a2_refine_ms"]["mean"], 3),
        "c_crop_loop_ms": round(summary["c_crop_loop_ms"]["mean"], 3),
        "c_label_apply_ms": round(summary["c_label_apply_ms"]["mean"], 3),
        "other_ms": round(summary["other_ms"]["mean"], 3),
        "objects": round(summary["objects"]["mean"], 3),
        "faces": round(summary["faces"]["mean"], 3),
        "c_skipped_plain_faces": round(summary["c_skipped_plain_faces"]["mean"], 3),
    }


def main() -> None:
    args = parse_args()
    dataset = resolve(args.dataset)
    stems = choose_samples(dataset, args.cube_count, args.face_count, args.sample_count, args.meta_scan_limit)
    frames = load_frames(dataset, stems)

    results = []
    for variant in args.variants:
        print(f"benchmarking {variant}...")
        result = benchmark_variant(args, variant, frames)
        print(json.dumps(compact_summary(result), ensure_ascii=False, indent=2))
        results.append(result)

    baseline = next(item for item in results if item["variant"] == "baseline")
    comparisons = {
        item["variant"]: compare_behavior(baseline, item)
        for item in results
        if item["variant"] != "baseline"
    }
    speedups = {
        item["variant"]: baseline["summary"]["measured_ms"]["mean"] / item["summary"]["measured_ms"]["mean"]
        for item in results
        if item["variant"] != "baseline"
    }

    report = {
        "samples": stems,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "args": vars(args),
        "results": results,
        "comparisons_to_baseline": comparisons,
        "speedups_vs_baseline": speedups,
        "compact": [compact_summary(item) for item in results],
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"report={output}")
    print(json.dumps({"compact": report["compact"], "comparisons": comparisons, "speedups": speedups}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
