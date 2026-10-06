from __future__ import annotations

import argparse


ABC_RUNTIME_PRESETS = ("manual", "best_stable_5080", "fast_unverified_a2_engine")


def add_runtime_preset_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runtime-preset",
        choices=ABC_RUNTIME_PRESETS,
        default="manual",
        help=(
            "ABC runtime preset. best_stable_5080 is the current behavior-preserving "
            "RTX 5080 preset; fast_unverified_a2_engine is faster but below the 98%% "
            "face-label match gate in the benchmark."
        ),
    )


def apply_runtime_preset(args: argparse.Namespace) -> None:
    preset = getattr(args, "runtime_preset", "manual")
    if preset == "manual":
        return

    if preset == "best_stable_5080":
        _set_if_present(
            args,
            a1_model="reports/abc_runtime_optimization/artifacts/a1_yolo26s_seg_meta_v2_50000_fp32_dynamic_b1.engine",
            c_onnx_model="reports/abc_runtime_optimization/artifacts/c_mobilenetv3small_runtimewarp_warmplain_ft.onnx",
            c_onnx_provider="cpu",
            c_warp_size=112,
            use_a1_object_mask=False,
            batch_b_across_objects=True,
            batch_c_across_objects=True,
            refine_quads="pose",
            yolo_half=False,
        )
        return

    if preset == "fast_unverified_a2_engine":
        _set_if_present(
            args,
            a1_model="reports/abc_runtime_optimization/artifacts/a1_yolo26s_seg_meta_v2_50000_fp32_dynamic_b1.engine",
            a2_model="reports/abc_runtime_optimization/artifacts/a2_yolo26n_seg_meta_v2_50000_fp32_dynamic_b8.engine",
            c_onnx_model="reports/abc_runtime_optimization/artifacts/c_mobilenetv3small_runtimewarp_warmplain_ft.onnx",
            c_onnx_provider="cpu",
            c_warp_size=128,
            use_a1_object_mask=False,
            batch_b_across_objects=True,
            batch_c_across_objects=True,
            refine_quads="pose",
            yolo_half=False,
        )
        return

    raise ValueError(f"unknown runtime preset: {preset}")


def _set_if_present(args: argparse.Namespace, **values: object) -> None:
    for name, value in values.items():
        if hasattr(args, name):
            setattr(args, name, value)
