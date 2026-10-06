# ABC Runtime Optimization Experiments

## Current Summary

This file is the preserved experiment log for the ABC cascade runtime optimization. It is not the main Jetson user guide. For Jetson Orin Nano setup and execution, see [README.md](./README.md).

Current conclusion:

```text
The best behavior-preserving ABC cascade preset is best_stable_5080.
It reaches about 20-21 FPS on the RTX 5080 4-cube / 12-face benchmark.
It preserves the face-label match gate at 98.26%.
The 22 FPS target was not achieved without behavior drift.
The next structural speed jump should come from A1 + cube-face unified, not more minor cascade tuning.
```

Current usable cascade command on the RTX 5080 development PC:

```powershell
C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\realtime_seg_cam.py `
  --pipeline abc `
  --runtime-preset best_stable_5080 `
  --camera 0 `
  --device 0 `
  --target-shape cube `
  --target-fruit apple
```

Do not assume this preset is directly portable to Jetson Orin Nano. It uses artifacts generated on the PC, including a TensorRT A1 engine. On Jetson, first run `.pt` weights for correctness, then export TensorRT engines on the Jetson itself.

Why the cascade stopped here:

```text
A2/B/pose/C still create many small model and CPU/OpenCV geometry operations.
GPU utilization is not always 100% because the pipeline frequently waits on crop, warp, filtering, and synchronization.
Frame-level B/C batching solved the biggest safe overhead, but the remaining structure is still per-face heavy.
```

See [../EXPERIMENTS.md](../EXPERIMENTS.md) for a shorter cross-project experiment index.

Goal: keep recognition behavior close to the current A1/A2/B/C cascade while reaching stable 20+ FPS on the RTX 5080 development PC for a scene with 4 cube-like objects and about 12 visible cube faces. Do not use shortcuts that only inspect the front-most cube.

## Ground Rules

- Keep all previous experiments and reports.
- Commit every meaningful experiment so rollback is easy.
- Benchmark on 4-object / about 12-face samples from `datasets/meta_v2_50000_coco_texture_v1`.
- Track both speed and behavior drift.
- Prefer changes that keep the model outputs semantically equivalent before changing trained models.

## Baseline: Current Batched Runtime

Date: 2026-06-28

Benchmark file:

- `reports/jetson_runtime_stage_overhead_current_pc_4cube12face.json`

Hardware:

- GPU: NVIDIA GeForce RTX 5080
- Runtime: PyTorch/Ultralytics `.pt`, not TensorRT

Scenario:

- 20 selected Meta V2 images
- about 4 cube-like objects
- about 12 visible faces
- display/drawing excluded

Measured current batched runtime:

| Stage | Mean ms/frame |
| --- | ---: |
| A1 YOLO26s-seg | 16.17 |
| A2 YOLO26n-seg batch | 16.62 |
| B TinyQuadNet batch | 5.58 |
| C MobileNetV3-Small batch | 20.13 |
| Other CPU/crop/warp/filter | 59.39 |
| Total | 117.90 |
| FPS | 8.71 |

Original-like sequential runtime:

| Stage | Mean ms/frame |
| --- | ---: |
| A1 YOLO26s-seg | 16.32 |
| A2 YOLO26n-seg sequential | 55.18 |
| B TinyQuadNet sequential | 20.46 |
| C MobileNetV3-Small sequential | 61.33 |
| Other CPU/crop/warp/filter | 55.36 |
| Total | 208.66 |
| FPS | 4.98 |

Conclusion so far:

- Batching improves 4-object FPS from about 5 FPS to about 8.7 FPS.
- Remaining bottleneck is no longer just model calls. CPU/crop/warp/filter is now the largest single bucket.

## Experiment Queue

1. Reduce C warp output from 224 to 128 because C ultimately resizes to 128 anyway.
2. Measure `--no-blacken-c-occlusion` behavior/speed drift.
3. Measure `--refine-quads none` behavior/speed drift.
4. Add behavior-drift benchmark against the baseline outputs.
5. If code-only changes cannot approach 20 FPS, train or swap a smaller C classifier.
6. If still below target, export A1/A2 to TensorRT or ONNX Runtime/TensorRT on the target platform.

## Experiment Log

### E000 - Baseline Commit

Status: completed.

Changes:

- Added Jetson/Linux camera backend support.
- Added CPU fallback for YOLO and Torch devices.
- Batched A2, B, and C runtime calls.
- Added runtime timing output.

Result:

- Current development PC: about 8.7 FPS on the 4-object / 12-face benchmark.

### E001 - Runtime Option Variants

Status: completed.

Report:

- `reports/abc_runtime_optimization/exp001_runtime_variants.json`

Variants:

| Variant | Mean FPS | Mean ms | Behavior vs baseline |
| --- | ---: | ---: | --- |
| baseline | 8.82 | 116.24 | reference |
| c_warp128 | 8.82 | 115.62 | 90% exact frame match, 99.1% face-label match |
| no_blacken | 9.06 | 113.10 | too much drift, 50% exact frame match |
| no_refine | 9.16 | 111.85 | notable drift, 60% exact frame match |
| fast_combo | 11.22 | 91.16 | too much drift, 30% exact frame match |

Conclusion:

- `c_warp128` is behavior-safe but not meaningfully faster by itself.
- `no_blacken` and `no_refine` are not acceptable default changes if "nearly similar behavior" is required.
- `fast_combo` is not acceptable as a default because output drift is too high.

### E002 - A1/A2 Mask Cache

Status: completed, not a clear win.

Report:

- `reports/abc_runtime_optimization/exp002_mask_cache.json`

Change:

- Added `result_masks()` and used per-result mask caching for A1/A2 masks.

Result:

| Variant | Mean FPS | Mean ms | Notes |
| --- | ---: | ---: | --- |
| baseline with mask cache | 7.85 | 129.99 | slower than E001 baseline |
| c_warp128 with mask cache | 9.10 | 112.67 | small speedup vs this run's baseline, but not a major win |

Conclusion:

- This is not the main path to 20 FPS.
- Current Ultralytics polygon-based mask access seems cheap enough that eager mask caching can do extra work.

### E003 - Restore Per-Mask Path and Split C Timing

Status: completed.

Report:

- `reports/abc_runtime_optimization/exp003_restore_mask_path_timing.json`

Changes:

- Removed the eager A1/A2 mask cache from the active runtime path.
- Added detailed C crop timing keys:
  - `c_source_ms`
  - `c_warp_ms`
  - `c_filter_ms`
  - `c_preprocess_ms`

Result:

| Variant | Mean FPS | Mean ms | c_source | c_warp | c_filter | c_preprocess | Behavior vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 9.60 | 105.42 | 18.97 | 4.57 | 2.45 | 2.42 | reference |
| c_warp128 | 9.86 | 102.89 | 18.93 | 2.19 | 1.79 | 1.09 | 90% exact frame match, 99.1% face-label match |

Conclusion:

- Restoring the original per-mask path recovered speed compared with E002.
- `c_warp128` is a small, behavior-safe speed lever, but only about 2.5%.
- The largest newly exposed C-side CPU cost is `c_source_ms`, which builds full-frame face masks and blackens occluded regions before warping.
- Next high-value experiment: blacken hidden face area in the warped C crop instead of copying/modifying the full frame.

### E004 - Crop-Space Occlusion Blackening

Status: completed.

Report:

- `reports/abc_runtime_optimization/exp004_crop_space_occlusion_blacken.json`

Change:

- Replaced the active C crop source path with a local ROI/crop-space path.
- Visibility stats are computed in a small ROI around the face instead of on a full-frame 640x640 mask.
- Occluded face pixels are blackened after perspective warp inside the C crop, instead of copying and modifying the full frame.
- Kept the old full-frame helper code in place for rollback/reference.

Result:

| Variant | Mean FPS | Mean ms | c_source | c_warp | c_filter | c_preprocess |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E003 baseline | 9.60 | 105.42 | 18.97 | 4.57 | 2.45 | 2.42 |
| E004 baseline | 12.40 | 81.48 | 1.53 | 5.50 | 2.22 | 2.29 |

Behavior versus E003 baseline:

| Metric | Value |
| --- | ---: |
| exact frame match | 85.0% |
| face-label match | 98.25% |
| mismatched sample count | 3 / 20 |
| speedup | 1.29x |

Conclusion:

- This is the first clearly useful runtime optimization after batching.
- It does not reach 20 FPS alone, but it removes most of the full-frame mask/copy overhead.
- Output drift exists but is small at the face-label level; this should be validated on real camera scenes before making it the final default.
- Remaining mean time is about 81 ms/frame, so reaching 20 FPS still needs either model export/engine acceleration or model simplification.

### E005 - A1/A2 Input Resolution Variants

Status: completed, rejected.

Report:

- `reports/abc_runtime_optimization/exp005_resolution_variants.json`

Variants:

| Variant | Mean FPS | Mean ms | Face-label match vs E004 baseline | Speedup |
| --- | ---: | ---: | ---: | ---: |
| E004 baseline | 11.70 | 87.02 | reference | 1.00x |
| A1 imgsz 512 | 11.12 | 93.35 | 67.69% | 0.93x |
| A2 imgsz 192 | 11.90 | 86.77 | 80.79% | 1.00x |
| A1 512 + A2 192 | 11.56 | 88.89 | 68.56% | 0.98x |
| A1 512 + A2 192 + C warp 128 | 11.27 | 89.57 | 69.00% | 0.97x |

Conclusion:

- Resolution reduction is not a good path here.
- It does not provide meaningful speedup on the current RTX 5080 setup.
- It causes large behavior drift, especially in face labels, so it should not be used as a default.
- Next direction should target model runtime/export or non-resolution CPU overhead rather than shrinking A1/A2 input size.

### E006 - TorchScript Trace B/C

Status: completed, optional small win.

Report:

- `reports/abc_runtime_optimization/exp006_trace_bc.json`

Change:

- Added `trace_torch_models=True` to `ABCPipeline`.
- Added `--trace-torch-models` to `jetson/realtime_seg_cam.py`.
- Added benchmark variant `trace_bc`.
- Only B TinyQuadNet and C MobileNetV3-Small are traced; A1/A2 YOLO models are unchanged.

Result:

| Variant | Mean FPS | Mean ms | B ms | C ms | Behavior vs baseline |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | 10.46 | 96.48 | 3.55 | 16.73 | reference |
| trace_bc | 10.84 | 92.74 | 2.97 | 12.47 | 100% exact frame match, 100% face-label match |

Conclusion:

- Safe but small improvement: about 1.04x speedup.
- Worth keeping as an opt-in runtime flag because behavior matched exactly in this benchmark.
- Not enough for 20 FPS by itself.
- Larger gains likely require A1/A2 export acceleration, C model simplification, or reducing post-processing overhead.

### E007 - A1/A2 YOLO FP16 Inference

Status: completed, rejected.

Report:

- `reports/abc_runtime_optimization/exp007_yolo_half.json`

Hypothesis:

- Passing `half=True` into Ultralytics A1/A2 `.pt` inference might speed up RTX 5080 CUDA inference without changing the architecture.

Command:

```powershell
C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp007_yolo_half.json `
  --sample-count 20 `
  --repeats 3 `
  --warmup 4 `
  --variants baseline yolo_half yolo_half_trace_bc
```

Result:

| Variant | Mean FPS | Mean ms | A1 ms | A2 ms | Face-label match | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 10.36 | 97.24 | 12.10 | 13.08 | reference | 1.00x |
| yolo_half | 9.65 | 113.15 | 12.84 | 24.27 | 90.83% | 0.86x |
| yolo_half + trace_bc | 10.46 | 96.05 | 12.66 | 13.45 | 90.83% | 1.01x |

Conclusion:

- FP16 is not a useful path for the current PyTorch/Ultralytics `.pt` runtime on this machine.
- It introduces notable output drift and does not improve speed.
- Keep `--yolo-half` as an opt-in diagnostic flag, but do not use it as a recommended default.
- Real A1/A2 speedup likely needs proper exported inference engines rather than toggling `.pt` FP16.

### E008 - Disable C Input Capture Unless Preview Panel Is Enabled

Status: completed, accepted.

Report:

- `reports/abc_runtime_optimization/exp008_disable_c_input_capture.json`

Hypothesis:

- The runtime was copying every C face crop into `last_c_inputs` even when the side preview panel was not shown.
- For 4 cube-like objects and about 12 visible faces, this creates avoidable per-frame image-copy and Python object overhead.

Command:

```powershell
C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp008_disable_c_input_capture.json `
  --sample-count 20 `
  --repeats 3 `
  --warmup 4 `
  --variants baseline keep_c_inputs trace_bc
```

Change:

- Added `keep_c_inputs` to `ABCPipeline`.
- `jetson/realtime_seg_cam.py` now passes `keep_c_inputs=args.show_c_inputs`.
- The default fast path no longer stores `face_crop.copy()` for every C input unless `--show-c-inputs` is enabled.
- Added benchmark variant `keep_c_inputs` to preserve the old debug-capture behavior for comparison.

Result:

| Variant | Mean FPS | Median FPS | Mean ms | Behavior vs baseline |
| --- | ---: | ---: | ---: | --- |
| baseline, no C input capture | 11.88 | 12.31 | 85.99 | reference |
| keep_c_inputs, old behavior | 10.81 | 11.47 | 96.12 | 100% exact frame match |
| trace_bc, no C input capture | 11.45 | 11.81 | 89.59 | 100% exact frame match |

Conclusion:

- Accepted as a safe default runtime optimization.
- It does not change detection/classification output.
- It gives about a 1.12x speedup compared with keeping C input debug crops on every frame.
- `--show-c-inputs` remains available for debugging, but should be off for FPS testing and Jetson deployment.
- Still below 20 FPS; the next meaningful path remains A1/A2 engine export or a faster C/postprocess path.

### E009 - Recheck No-Refine / Fast Postprocess Paths After E008

Status: completed, rejected for the 20 FPS goal.

Report:

- `reports/abc_runtime_optimization/exp009_refine_fastpath_recheck.json`

Hypothesis:

- After crop-space occlusion and disabled C input capture, projective quad refinement might be a larger remaining CPU cost.
- Disabling refinement or using the previous fast combo might approach 20 FPS, but must still preserve near-equivalent behavior.

20 FPS pass criteria for this monitoring cycle:

- mean FPS >= 20.0
- median FPS >= 20.0
- face-label match >= 98%
- no shortcut that ignores non-front objects

Command:

```powershell
C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp009_refine_fastpath_recheck.json `
  --sample-count 20 `
  --repeats 3 `
  --warmup 4 `
  --variants baseline no_refine no_blacken fast_combo c_warp128
```

Result:

| Variant | Mean FPS | Median FPS | Face-label match | Speedup | 20 FPS status |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | 10.12 | 10.06 | reference | 1.00x | FAIL |
| no_refine | 13.34 | 13.13 | 94.32% | 1.32x | FAIL: behavior and FPS |
| no_blacken | 10.61 | 10.49 | 91.27% | 1.05x | FAIL: behavior and FPS |
| fast_combo | 14.36 | 14.25 | 89.52% | 1.42x | FAIL: behavior and FPS |
| c_warp128 | 11.25 | 11.15 | 98.69% | 1.11x | FAIL: FPS |

Conclusion:

- None of these paths achieve the 20 FPS monitoring goal.
- `no_refine`, `no_blacken`, and `fast_combo` are still not acceptable defaults because face-label match falls below 98%.
- `c_warp128` passes the behavior threshold in this run, but only reaches about 11.25 FPS.
- Removing projective refinement is not enough and costs too much behavior quality.
- Next most promising direction remains replacing `.pt` A1/A2 inference with exported engines or a more structural reduction in the number/cost of A2 and C calls.

### E010 - Bbox-Only A1 Object Mask Path

Status: completed, optional diagnostic path; 20 FPS goal failed.

Report:

- `reports/abc_runtime_optimization/exp010_bbox_only_a1_mask.json`

Hypothesis:

- A1 object mask extraction and A2 object-mask overlap filtering may be a nontrivial part of the remaining CPU/postprocess cost.
- Skipping A1 object-mask extraction and using A1 boxes only may reduce overhead while keeping all 4 cube-like objects in the pipeline.

20 FPS pass criteria for this monitoring cycle:

- mean FPS >= 20.0
- median FPS >= 20.0
- face-label match >= 98%
- no shortcut that ignores non-front objects

Command:

```powershell
C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp010_bbox_only_a1_mask.json `
  --sample-count 20 `
  --repeats 3 `
  --warmup 4 `
  --variants baseline bbox_only_a1 bbox_only_a1_cwarp128
```

Change:

- Added `use_a1_object_mask` to `ABCPipeline`, default `True`.
- Added `--no-a1-object-mask` to `jetson/realtime_seg_cam.py`.
- Added benchmark variants:
  - `bbox_only_a1`
  - `bbox_only_a1_cwarp128`

Result:

| Variant | Mean FPS | Median FPS | Face-label match | Speedup | 20 FPS status |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | 10.64 | 10.69 | reference | 1.00x | FAIL |
| bbox_only_a1 | 11.15 | 10.93 | 99.57% | 1.05x | FAIL: FPS |
| bbox_only_a1_cwarp128 | 12.31 | 12.12 | 98.26% | 1.16x | FAIL: FPS |

Conclusion:

- Bbox-only A1 is behavior-safe on this benchmark, but the speedup is too small for the 20 FPS target.
- `bbox_only_a1_cwarp128` passes the 98% face-label match threshold, but still reaches only about 12.3 FPS.
- Keep this as an explicit opt-in diagnostic/deployment tradeoff, not a new default yet, because object-mask overlap may matter more in real cluttered scenes.
- This confirms that A1 mask extraction is not the dominant blocker.
- Next most promising direction remains A1/A2 engine export or a larger structural simplification of A2/C.

### E011 - C Classifier Top-1 GPU Transfer

Status: completed, rejected and reverted.

Report:

- `reports/abc_runtime_optimization/exp011_c_top1_transfer.json`

Hypothesis:

- C classification only needs the top-1 class and confidence.
- Computing `softmax().max()` on GPU and transferring only class IDs/confidences to CPU might reduce C postprocess overhead compared with transferring the full probability matrix.

20 FPS pass criteria for this monitoring cycle:

- mean FPS >= 20.0
- median FPS >= 20.0
- face-label match >= 98%
- no shortcut that ignores non-front objects

Command:

```powershell
C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp011_c_top1_transfer.json `
  --sample-count 20 `
  --repeats 3 `
  --warmup 4 `
  --variants baseline trace_bc bbox_only_a1_cwarp128
```

Temporary change:

- Changed `_classify_face()` and `_classify_faces()` to use GPU-side `softmax(...).max(dim=1)` and copy only top-1 class IDs/confidences.
- Reverted after benchmarking because it did not improve the default runtime.

Result:

| Variant | Mean FPS | Median FPS | Face-label match | Speedup | 20 FPS status |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline with top-1 transfer | 10.19 | 10.11 | reference | 1.00x | FAIL |
| trace_bc with top-1 transfer | 9.60 | 9.51 | 100.00% | 0.94x | FAIL: FPS |
| bbox_only_a1_cwarp128 with top-1 transfer | 11.09 | 11.01 | 98.26% | 1.09x | FAIL: FPS |

Conclusion:

- The full probability matrix is tiny for this classifier, so reducing the CPU transfer does not matter.
- The change did not improve FPS and was reverted.
- This confirms that C transfer/post-softmax is not a meaningful bottleneck.
- Next most promising direction remains A1/A2 exported inference engines or a genuinely smaller/faster C model.

### E012 - C Classifier ONNX Runtime

Status: completed, accepted as an optional speed path; 20 FPS goal failed.

Report:

- `reports/abc_runtime_optimization/exp012_c_onnx_runtime.json`

Artifacts:

- `reports/abc_runtime_optimization/artifacts/c_mobilenetv3small_runtimewarp_warmplain_ft.onnx`
- `reports/abc_runtime_optimization/artifacts/c_mobilenetv3small_runtimewarp_warmplain_ft.classes.json`

Hypothesis:

- C MobileNetV3-Small is called for many face crops per frame.
- Exporting C to ONNX and running it through ONNX Runtime may reduce Python/PyTorch overhead without changing the C model.

20 FPS pass criteria for this monitoring cycle:

- mean FPS >= 20.0
- median FPS >= 20.0
- face-label match >= 98%
- no shortcut that ignores non-front objects

Command:

```powershell
C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\export_c_classifier_onnx.py `
  --output reports\abc_runtime_optimization\artifacts\c_mobilenetv3small_runtimewarp_warmplain_ft.onnx

C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp012_c_onnx_runtime.json `
  --sample-count 20 `
  --repeats 3 `
  --warmup 4 `
  --variants baseline c_onnx_cuda c_onnx_cuda_cwarp128 c_onnx_cuda_bbox_cwarp128
```

Change:

- Added optional `c_onnx_model` / `c_onnx_provider` to `ABCPipeline`.
- Added `--c-onnx-model` / `--c-onnx-provider` to `jetson/realtime_seg_cam.py`.
- Added `jetson/scripts/export_c_classifier_onnx.py`.
- Added benchmark variants:
  - `c_onnx_cuda`
  - `c_onnx_cuda_cwarp128`
  - `c_onnx_cuda_bbox_cwarp128`

Important environment note:

- ONNX Runtime reported that CUDAExecutionProvider could not load because `cublasLt64_12.dll` was missing.
- The run still completed through ONNX Runtime fallback providers.
- Even with that caveat, the C runtime dropped sharply, so fixing ORT CUDA dependencies may unlock more speed.

Result:

| Variant | Mean FPS | Median FPS | C ms | Face-label match | Speedup | 20 FPS status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 10.64 | 10.68 | 16.52 | reference | 1.00x | FAIL |
| c_onnx_cuda | 13.66 | 13.93 | 5.78 | 100.00% | 1.28x | FAIL: FPS |
| c_onnx_cuda_cwarp128 | 14.54 | 14.95 | 6.01 | 98.69% | 1.36x | FAIL: FPS |
| c_onnx_cuda_bbox_cwarp128 | 15.51 | 16.01 | 5.86 | 98.26% | 1.45x | FAIL: FPS |

Conclusion:

- This is the largest clean speedup since crop-space occlusion blackening.
- It still does not satisfy the 20 FPS monitoring goal.
- C ONNX is worth keeping as an optional speed path because behavior is equivalent or within the 98% threshold in the tested variants.
- The remaining blocker is now mostly outside C: A1/A2 YOLO `.pt` runtime plus postprocess/refinement.
- Next most promising direction is exported A1/A2 engines or fixing ONNX Runtime CUDA dependencies and then trying A1/A2 ONNX/TensorRT.

### E013 - ONNX Runtime CUDA Dependency Fix and Provider Recheck

Status: completed, CUDA provider fixed but rejected for this C workload; 20 FPS goal failed.

Report:

- `reports/abc_runtime_optimization/exp013_ort_cuda_dependency_fix.json`
- Supplementary preliminary run: `reports/abc_runtime_optimization/exp013_ort_cuda_dll_fix.json`

Environment changes during this experiment:

```powershell
C:\Users\user\anaconda3\envs\ai_robotics\python.exe -m pip install --upgrade `
  nvidia-cublas-cu12 `
  nvidia-cudnn-cu12 `
  nvidia-cuda-runtime-cu12 `
  nvidia-cufft-cu12 `
  nvidia-curand-cu12 `
  nvidia-cusolver-cu12 `
  nvidia-cusparse-cu12
```

Hypothesis:

- E012 showed ONNX Runtime CUDAExecutionProvider falling back because CUDA 12 DLLs were missing.
- Installing CUDA 12/cuDNN 9 runtime DLL wheels and registering `nvidia/*/bin` directories should activate CUDAExecutionProvider.
- If C ONNX runs on CUDA, it might improve the current best C ONNX path and move closer to 20 FPS.

20 FPS pass criteria for this monitoring cycle:

- mean FPS >= 20.0
- median FPS >= 20.0
- face-label match >= 98%
- no shortcut that ignores non-front objects

Command:

```powershell
C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp013_ort_cuda_dependency_fix.json `
  --sample-count 20 `
  --repeats 3 `
  --warmup 4 `
  --variants baseline c_onnx_cpu c_onnx_cpu_cwarp128 c_onnx_cpu_bbox_cwarp128 c_onnx_cuda c_onnx_cuda_cwarp128
```

Change:

- Added `add_nvidia_cuda_dll_directories()` to `jetson/abc_inference.py`.
- It registers installed `nvidia/*/bin` DLL directories before creating ONNX Runtime sessions.
- Added explicit CPU ONNX benchmark variants:
  - `c_onnx_cpu`
  - `c_onnx_cpu_cwarp128`
  - `c_onnx_cpu_bbox_cwarp128`
- Verified that ONNX Runtime can now create a session with active providers:
  - `CUDAExecutionProvider`
  - `CPUExecutionProvider`

Result:

| Variant | Mean FPS | Median FPS | C ms | Face-label match | Speedup | 20 FPS status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 10.64 | 10.53 | 16.31 | reference | 1.00x | FAIL |
| c_onnx_cpu | 11.26 | 11.19 | 5.41 | 100.00% | 1.06x | FAIL: FPS |
| c_onnx_cpu_cwarp128 | 14.86 | 15.00 | 5.73 | 98.69% | 1.39x | FAIL: FPS |
| c_onnx_cpu_bbox_cwarp128 | 16.20 | 16.27 | 5.55 | 98.26% | 1.52x | FAIL: FPS |
| c_onnx_cuda | 8.55 | 8.41 | 39.54 | 100.00% | 0.79x | FAIL: FPS |
| c_onnx_cuda_cwarp128 | 8.75 | 8.81 | 40.71 | 98.69% | 0.80x | FAIL: FPS |

Conclusion:

- CUDAExecutionProvider is now fixed and active, but it is slower for this C classifier workload.
- The batch is small enough that ORT CUDA launch/provider overhead dominates.
- Explicit ONNX CPU provider remains the best C path.
- Current best monitored variant is `c_onnx_cpu_bbox_cwarp128` at about 16.2 mean FPS / 16.3 median FPS with 98.26% face-label match.
- 20 FPS is still not achieved.
- Next most promising direction is A1/A2 exported inference engines or reducing the number of A2 calls/faces before C, because C is no longer the dominant blocker.

### E014 - A1/A2 YOLO ONNX Runtime Export Check

Status: completed, rejected; 20 FPS goal failed.

Report:

- `reports/abc_runtime_optimization/exp014_a1_a2_onnx_runtime.json`

Artifacts:

- `reports/abc_runtime_optimization/artifacts/a1_yolo26s_seg_meta_v2_50000.onnx`
- `reports/abc_runtime_optimization/artifacts/a2_yolo26n_seg_meta_v2_50000.onnx`

Hypothesis:

- E013 showed that C is no longer the dominant bottleneck after the ONNX CPU path.
- Exporting A1 and/or A2 YOLO segmentation models to ONNX might reduce the Ultralytics `.pt` overhead and move the full 4-object / 12-face benchmark toward 20 FPS.

20 FPS pass criteria for this monitoring cycle:

- mean FPS >= 20.0
- median FPS >= 20.0
- face-label match >= 98%
- no shortcut that ignores non-front objects

Commands:

```powershell
@'
from pathlib import Path
from shutil import copy2
from ultralytics import YOLO

artifacts = Path("reports/abc_runtime_optimization/artifacts")
artifacts.mkdir(parents=True, exist_ok=True)

exports = [
    (
        "runs/meta_v2_a1_objectseg/a1_yolo26s_seg_meta_v2_50000/weights/best.pt",
        640,
        artifacts / "a1_yolo26s_seg_meta_v2_50000.onnx",
    ),
    (
        "runs/meta_v2_a2_faceseg/a2_yolo26n_seg_meta_v2_50000/weights/best.pt",
        224,
        artifacts / "a2_yolo26n_seg_meta_v2_50000.onnx",
    ),
]

for source, imgsz, target in exports:
    model = YOLO(source)
    exported = Path(model.export(format="onnx", imgsz=imgsz, dynamic=True, simplify=True, opset=18))
    copy2(exported, target)
    print(target)
'@ | C:\Users\user\anaconda3\envs\ai_robotics\python.exe -

C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp014_a1_a2_onnx_runtime.json `
  --sample-count 20 `
  --repeats 3 `
  --warmup 4 `
  --variants baseline a1_onnx a2_onnx a1_a2_onnx a1_a2_onnx_best_c c_onnx_cpu_bbox_cwarp128
```

Change:

- Added benchmark-only support for A1/A2 ONNX model path variants.
- Did not change the runtime default path.
- Kept the current accepted C ONNX CPU + bbox-only + 128 crop path as a comparison point.

Result:

| Variant | Mean FPS | Median FPS | A1 ms | A2 ms | C ms | Face-label match | Speedup | 20 FPS status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 10.67 | 10.52 | 11.81 | 12.75 | 16.36 | reference | 1.00x | FAIL |
| a1_onnx | 6.59 | 6.79 | 59.89 | 14.32 | 18.55 | 100.00% | 0.62x | FAIL: FPS |
| a2_onnx | 11.02 | 10.92 | 11.45 | 19.88 | 14.56 | 98.69% | 1.04x | FAIL: FPS |
| a1_a2_onnx | 2.02 | 2.01 | 272.92 | 92.74 | 32.14 | 98.69% | 0.19x | FAIL: FPS |
| a1_a2_onnx_best_c | 9.97 | 10.94 | 20.70 | 59.31 | 8.06 | 97.39% | 0.70x | FAIL: FPS and face-label match |
| c_onnx_cpu_bbox_cwarp128 | 15.30 | 15.69 | 12.68 | 11.59 | 5.77 | 98.26% | 1.43x | FAIL: FPS |

Conclusion:

- A1 ONNX is behavior-equivalent but much slower on this path, so it is rejected.
- A2 ONNX alone preserves the 98% behavior threshold but only gives a tiny full-pipeline gain, far below 20 FPS.
- Combining A1 and A2 ONNX is clearly worse and should not be used.
- The current best valid path remains `c_onnx_cpu_bbox_cwarp128`: about 15.3 mean FPS / 15.7 median FPS with 98.26% face-label match.
- Next most promising direction is not generic ONNX for YOLO segmentation. The likely options are TensorRT engine export for A1/A2, replacing A2 segmentation with a smaller face proposal model, or reducing A2 calls with a behavior-checked face candidate gate.

### E015 - Stage Timing Profile and No-Refine Recheck

Status: completed, profiling accepted; no-refine shortcut rejected; 20 FPS goal failed.

Reports:

- `reports/abc_runtime_optimization/exp015_stage_timing_profile.json`
- `reports/abc_runtime_optimization/exp015_no_refine_best_recheck.json`

Hypothesis:

- E014 showed that generic ONNX Runtime is not useful for A1/A2 YOLO segmentation.
- The remaining `other_ms` is large enough that we need finer timing before changing behavior.
- If most of the remaining overhead comes from pose-based quad refinement, disabling refinement may be fast enough, but it must still satisfy the 98% face-label match gate.

20 FPS pass criteria for this monitoring cycle:

- mean FPS >= 20.0
- median FPS >= 20.0
- face-label match >= 98%
- no shortcut that ignores non-front objects

Commands:

```powershell
C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp015_stage_timing_profile.json `
  --sample-count 20 `
  --repeats 3 `
  --warmup 4 `
  --variants baseline c_onnx_cpu_bbox_cwarp128

C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp015_no_refine_best_recheck.json `
  --sample-count 20 `
  --repeats 3 `
  --warmup 4 `
  --variants baseline no_refine c_onnx_cpu_bbox_cwarp128
```

Change:

- Added timing-only instrumentation for:
  - `a1_post_ms`
  - `a2_prepare_ms`
  - `a2_post_ms`
  - `a2_face_parse_ms`
  - `a2_face_build_ms`
  - `a2_face_filter_ms`
  - `c_crop_loop_ms`
  - `c_label_apply_ms`
- Added these fields to the benchmark summary.
- No runtime default behavior was changed.

Result:

| Variant | Mean FPS | Median FPS | Face-label match | A2 face filter/refine ms | C crop loop ms | 20 FPS status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 10.50 | 10.30 | reference | 18.82 | 12.17 | FAIL |
| c_onnx_cpu_bbox_cwarp128 | 15.84 | 15.93 | 98.26% | 13.40 | 6.14 | FAIL: FPS |
| no_refine | 13.02 | 12.83 | 94.32% | 1.64 | 11.24 | FAIL: FPS and face-label match |

Conclusion:

- The current best valid path is still `c_onnx_cpu_bbox_cwarp128`, now measured at about 15.84 mean FPS / 15.93 median FPS with 98.26% face-label match in the profiling run.
- The largest remaining non-model block is the A2 face filter/refine region: about 13.4 ms even after the current C ONNX + bbox/crop optimizations.
- Disabling pose refinement proves that this block is real, but it drops face-label match to 94.32%, so a blunt `no_refine` shortcut is rejected.
- The next promising optimization is to keep pose-refined behavior while making the refine path cheaper: profile inside `refine_cube_quads_by_pose`, cache/avoid expensive polygon operations, or replace only the geometric pose solve with a cheaper equivalent. TensorRT A1/A2 remains the other high-upside path.

### E016 - Shared-Edge Reuse in Pose Refinement

Status: completed, accepted as a partial optimization; 20 FPS goal failed.

Reports:

- `reports/abc_runtime_optimization/exp016_refine_fastpath_profile.json`
- `reports/abc_runtime_optimization/exp016_refine_shared_edges_reuse.json`

Hypothesis:

- E015 showed that the A2 face filter/refine block is the largest remaining CPU-side bottleneck.
- The pipeline checked whether face geometry was sane, then the pose-refine function recomputed the same shared-edge structure.
- Reusing shared-edge results from the sanity check should reduce refine overhead without changing behavior.
- Reordering cheap duplicate/sanity checks before polygon IoU should avoid some expensive polygon rasterization while preserving the same OR conditions.

20 FPS pass criteria for this monitoring cycle:

- mean FPS >= 20.0
- median FPS >= 20.0
- face-label match >= 98%
- no shortcut that ignores non-front objects

Commands:

```powershell
C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp016_refine_fastpath_profile.json `
  --sample-count 20 `
  --repeats 3 `
  --warmup 4 `
  --variants baseline c_onnx_cpu_bbox_cwarp128

C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp016_refine_shared_edges_reuse.json `
  --sample-count 20 `
  --repeats 3 `
  --warmup 4 `
  --variants baseline c_onnx_cpu_bbox_cwarp128
```

Change:

- Added `cube_refine_shared_edges_if_sane()` so the shared edges computed during sanity checking can be passed into pose refinement.
- Added optional `shared_edges` to `refine_cube_quads_by_pose()` and `refine_cube_quads_by_projective_geometry()`.
- Reordered duplicate/sanity checks to use cheap bbox/center tests before polygon IoU when the boolean result is preserved.
- Reordered `projective_refinement_passes()` so move-threshold failures exit before running polygon IoU.
- Added refine timing fields:
  - `a2_face_dedupe_ms`
  - `a2_refine_sane_ms`
  - `a2_refine_ms`

Result:

| Variant | Mean FPS | Median FPS | Face-label match | A2 face filter/refine ms | A2 refine ms | Speedup | 20 FPS status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 10.89 | 10.78 | reference | 15.81 | 9.68 | 1.00x | FAIL |
| c_onnx_cpu_bbox_cwarp128 | 16.26 | 16.46 | 98.26% | 11.44 | 6.75 | 1.49x | FAIL: FPS |

Conclusion:

- This is a valid partial optimization because the face-label match stays at 98.26%, above the 98% gate.
- The best valid path improved to about 16.26 mean FPS / 16.46 median FPS on the monitored benchmark.
- The 20 FPS goal is still not achieved.
- The remaining largest blocks are still A2/post-refine geometry plus A1/A2 YOLO inference.
- Next most promising options are deeper optimization inside `projective_refinement_passes()` / polygon IoU, or a TensorRT path for A1/A2. A blunt no-refine shortcut remains rejected because it failed behavior matching in E015.

### E017 - Cached Face Contours and OpenCV Mask IoU

Status: completed, accepted as a partial optimization; 20 FPS goal failed.

Report:

- `reports/abc_runtime_optimization/exp017_polygon_iou_cache.json`

Hypothesis:

- E016 showed that A2 geometry/refine is still a large CPU-side bottleneck.
- Several checks repeatedly convert the same `FaceEvidence.segments_xy` polygons into OpenCV contours, then rasterize masks for IoU.
- Caching face contours and using OpenCV `bitwise_and` / `bitwise_or` plus `countNonZero` should preserve behavior while reducing Python/numpy overhead.

20 FPS pass criteria for this monitoring cycle:

- mean FPS >= 20.0
- median FPS >= 20.0
- face-label match >= 98%
- no shortcut that ignores non-front objects

Command:

```powershell
C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp017_polygon_iou_cache.json `
  --sample-count 20 `
  --repeats 3 `
  --warmup 4 `
  --variants baseline c_onnx_cpu_bbox_cwarp128
```

Change:

- Added internal `face_segments_contours()` cache on `FaceEvidence` instances.
- Added `polygon_iou_from_contours()` so callers can reuse already-built contours.
- Switched mask IoU intersection/union counting from numpy logical reductions to OpenCV bitwise operations.
- Updated duplicate, sanity, geometry scoring, edge-point, and projective refinement checks to use cached contours where possible.
- Runtime output structure is unchanged because the cache is attached dynamically and is not part of the dataclass fields.

Result:

| Variant | Mean FPS | Median FPS | Face-label match | A2 face filter/refine ms | A2 refine sane ms | A2 refine ms | Speedup | 20 FPS status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 10.85 | 10.75 | reference | 14.60 | 4.00 | 9.62 | 1.00x | FAIL |
| c_onnx_cpu_bbox_cwarp128 | 16.99 | 17.35 | 98.26% | 9.70 | 2.71 | 6.29 | 1.56x | FAIL: FPS |

Conclusion:

- This is accepted: behavior remains above the 98% gate with the same 98.26% face-label match.
- The best valid path improved again, now to about 16.99 mean FPS / 17.35 median FPS.
- The 20 FPS goal is still not achieved.
- Remaining likely blockers are A1/A2 YOLO inference plus the remaining A2 geometry/refine/C crop-loop work.
- Next most promising directions are TensorRT for A1/A2 or a more structural change that reduces A2/postprocess calls while preserving all 4 objects and visible faces.

### E018 - C Warp Resolution Sweep

Status: completed, rejected for runtime default change; 20 FPS goal failed.

Reports:

- `reports/abc_runtime_optimization/exp018_c_warp_resolution_sweep.json`
- `reports/abc_runtime_optimization/exp018_c_warp_boundary_sweep.json`

Hypothesis:

- E017 left C crop-loop work as one of the remaining non-model costs.
- Reducing `c_warp_size` below 128 may cut warp/filter time while retaining enough visual information for C classification.
- The goal is to find a smaller crop that still passes the 98% face-label match gate.

20 FPS pass criteria for this monitoring cycle:

- mean FPS >= 20.0
- median FPS >= 20.0
- face-label match >= 98%
- no shortcut that ignores non-front objects

Commands:

```powershell
C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp018_c_warp_resolution_sweep.json `
  --sample-count 20 `
  --repeats 3 `
  --warmup 4 `
  --variants baseline c_onnx_cpu_bbox_cwarp128 c_onnx_cpu_bbox_cwarp96 c_onnx_cpu_bbox_cwarp80 c_onnx_cpu_bbox_cwarp64

C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp018_c_warp_boundary_sweep.json `
  --sample-count 20 `
  --repeats 3 `
  --warmup 4 `
  --variants baseline c_onnx_cpu_bbox_cwarp128 c_onnx_cpu_bbox_cwarp112 c_onnx_cpu_bbox_cwarp96 c_onnx_cpu_bbox_cwarp88 c_onnx_cpu_bbox_cwarp80
```

Change:

- Added benchmark-only variants:
  - `c_onnx_cpu_bbox_cwarp112`
  - `c_onnx_cpu_bbox_cwarp96`
  - `c_onnx_cpu_bbox_cwarp88`
  - `c_onnx_cpu_bbox_cwarp80`
  - `c_onnx_cpu_bbox_cwarp64`
- Runtime defaults are unchanged.

Result:

| Variant | Mean FPS | Median FPS | Face-label match | 20 FPS status |
| --- | ---: | ---: | ---: | --- |
| c_onnx_cpu_bbox_cwarp128 | 17.17 | 17.53 | 98.26% | FAIL: FPS |
| c_onnx_cpu_bbox_cwarp112 | 14.07 | 13.91 | 98.26% | FAIL: FPS |
| c_onnx_cpu_bbox_cwarp96 | 14.07 | 13.92 | 98.70% | FAIL: FPS |
| c_onnx_cpu_bbox_cwarp88 | 17.95 | 18.06 | 97.83% | FAIL: face-label match and FPS |
| c_onnx_cpu_bbox_cwarp80 | 16.41 | 17.06 | 97.83% | FAIL: face-label match and FPS |
| c_onnx_cpu_bbox_cwarp64 | 17.36 | 17.82 | 96.09% | FAIL: face-label match and FPS |

Conclusion:

- `c_warp_size=88` is the fastest tested crop path, but it fails the 98% behavior gate at 97.83%.
- `c_warp_size=96` and `112` pass behavior, but are slower than the existing 128 path in these runs.
- Therefore no runtime default should change from this experiment.
- The best valid monitored path remains `c_onnx_cpu_bbox_cwarp128`, measured here at about 17.17 mean FPS / 17.53 median FPS with 98.26% face-label match.
- Crop-size tuning alone cannot reach 20 FPS. The next promising work should target A1/A2 inference or reduce A2/postprocess calls structurally.

### E019 - C Classifier ONNX Runtime TensorRT Provider Check

Status: completed, rejected; TensorRT provider did not load; 20 FPS goal failed.

Report:

- `reports/abc_runtime_optimization/exp019_c_tensorrt_ep.json`

Hypothesis:

- ONNX Runtime reports `TensorrtExecutionProvider` as available, and `trtexec` is not on PATH.
- Since the C classifier is already exported to ONNX, trying ORT TensorRT EP is a low-risk way to check whether TensorRT can reduce the remaining C classifier runtime.
- This does not change runtime defaults; it only adds benchmark variants.

20 FPS pass criteria for this monitoring cycle:

- mean FPS >= 20.0
- median FPS >= 20.0
- face-label match >= 98%
- no shortcut that ignores non-front objects

Command:

```powershell
C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp019_c_tensorrt_ep.json `
  --sample-count 20 `
  --repeats 3 `
  --warmup 4 `
  --variants baseline c_onnx_cpu_bbox_cwarp128 c_onnx_trt_bbox_cwarp128
```

Change:

- Added benchmark-only C ONNX TensorRT variants:
  - `c_onnx_trt`
  - `c_onnx_trt_cwarp128`
  - `c_onnx_trt_bbox_cwarp128`

Environment finding:

- `trtexec` was not found on PATH.
- Python TensorRT import reports TensorRT `11.0.0.114`.
- ONNX Runtime reports `TensorrtExecutionProvider`, `CUDAExecutionProvider`, and `CPUExecutionProvider`.
- However, ORT TensorRT EP failed to load because `onnxruntime_providers_tensorrt.dll` depends on missing `nvinfer_10.dll`.
- The benchmark therefore fell back to `CUDAExecutionProvider,CPUExecutionProvider`, not real TensorRT execution.

Result:

| Variant | Active C runtime | Mean FPS | Median FPS | C ms | Face-label match | 20 FPS status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| baseline | torch | 11.42 | 11.37 | 16.38 | reference | FAIL |
| c_onnx_cpu_bbox_cwarp128 | onnx:CPUExecutionProvider | 13.96 | 13.86 | 5.73 | 98.26% | FAIL: FPS |
| c_onnx_trt_bbox_cwarp128 | onnx:CUDAExecutionProvider,CPUExecutionProvider fallback | 9.75 | 9.86 | 40.49 | 98.26% | FAIL: FPS |

Conclusion:

- This experiment is rejected.
- ORT TensorRT EP is not usable in the current environment because the provider expects TensorRT 10 DLLs while the environment has TensorRT 11 Python bindings.
- The fallback CUDA path is slower than CPU for this small C classifier workload, matching the earlier CUDA-provider result.
- The runtime default should remain the explicit ONNX CPU provider for C.
- Next promising direction is still A1/A2 TensorRT through a compatible TensorRT toolchain or a structural reduction in A2/postprocess calls.

### E020 - Occlusion Blackening Ablation

Status: completed, rejected; 20 FPS goal failed.

Report:

- `reports/abc_runtime_optimization/exp020_occlusion_blacken_ablation.json`

Hypothesis:

- Best valid paths still spend time in C crop generation.
- Disabling hidden-face blackening could reduce crop work, but must preserve the 98% face-label match gate.
- Because this is a behavior-sensitive operation, it is tested only as benchmark variants.

20 FPS pass criteria for this monitoring cycle:

- mean FPS >= 20.0
- median FPS >= 20.0
- face-label match >= 98%
- no shortcut that ignores non-front objects

Command:

```powershell
C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp020_occlusion_blacken_ablation.json `
  --sample-count 20 `
  --repeats 3 `
  --warmup 4 `
  --variants baseline c_onnx_cpu_bbox_cwarp128 c_onnx_cpu_bbox_cwarp128_no_blacken c_onnx_cpu_bbox_cwarp88 c_onnx_cpu_bbox_cwarp88_no_blacken
```

Change:

- Added benchmark-only variants:
  - `c_onnx_cpu_bbox_cwarp128_no_blacken`
  - `c_onnx_cpu_bbox_cwarp88_no_blacken`
- Runtime defaults are unchanged.

Result:

| Variant | Mean FPS | Median FPS | Face-label match | 20 FPS status |
| --- | ---: | ---: | ---: | --- |
| c_onnx_cpu_bbox_cwarp128 | 15.13 | 14.55 | 98.26% | FAIL: FPS |
| c_onnx_cpu_bbox_cwarp128_no_blacken | 13.38 | 13.62 | 91.30% | FAIL: face-label match and FPS |
| c_onnx_cpu_bbox_cwarp88 | 17.38 | 17.64 | 97.83% | FAIL: face-label match and FPS |
| c_onnx_cpu_bbox_cwarp88_no_blacken | 14.40 | 14.26 | 90.87% | FAIL: face-label match and FPS |

Conclusion:

- Hidden-face blackening is behavior-critical. Disabling it drops face-label match from the high 97-98% range to about 91%.
- It also did not improve speed in this run, likely because visibility statistics and filtering still run while the changed classifications perturb downstream work.
- This path is rejected and should not be used as a runtime shortcut.
- The next most promising direction remains A1/A2 TensorRT with a compatible toolchain, or a structural reduction in A2/postprocess work that preserves occlusion-aware C inputs.

### E021 - A2 Plain-Face Fast Path

Status: completed, rejected; 20 FPS goal failed.

Reports:

- `reports/abc_runtime_optimization/exp021_a2_plain_face_fastpath.json`
- `reports/abc_runtime_optimization/exp021_a2_plain_face_threshold_sweep.json`

Hypothesis:

- A2 already predicts `plain_face` vs `fruit_face`.
- If high-confidence A2 `plain_face` detections can be trusted, the runtime can skip C crop/classification for many plain faces and set the label directly to `plain`.
- This should reduce C crop-loop and C classifier cost without ignoring objects or faces.

20 FPS pass criteria for this monitoring cycle:

- mean FPS >= 20.0
- median FPS >= 20.0
- face-label match >= 98%
- no shortcut that ignores non-front objects

Commands:

```powershell
C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp021_a2_plain_face_fastpath.json `
  --sample-count 20 `
  --repeats 3 `
  --warmup 4 `
  --variants baseline c_onnx_cpu_bbox_cwarp128 c_onnx_cpu_bbox_cwarp128_fruit_only c_onnx_cpu_bbox_cwarp88 c_onnx_cpu_bbox_cwarp88_fruit_only

C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp021_a2_plain_face_threshold_sweep.json `
  --sample-count 20 `
  --repeats 3 `
  --warmup 4 `
  --variants baseline c_onnx_cpu_bbox_cwarp128 c_onnx_cpu_bbox_cwarp128_plain85 c_onnx_cpu_bbox_cwarp128_plain90 c_onnx_cpu_bbox_cwarp128_plain95
```

Change:

- Added benchmark/runtime option `classify_fruit_faces_only`.
- Added option `plain_face_fastpath_min_conf` for thresholded A2 plain-face fast path.
- Added benchmark-only variants:
  - `c_onnx_cpu_bbox_cwarp128_fruit_only`
  - `c_onnx_cpu_bbox_cwarp88_fruit_only`
  - `c_onnx_cpu_bbox_cwarp128_plain85`
  - `c_onnx_cpu_bbox_cwarp128_plain90`
  - `c_onnx_cpu_bbox_cwarp128_plain95`
- Added `c_skipped_plain_faces` to benchmark summaries.
- Runtime defaults are unchanged.

Result:

| Variant | Mean FPS | Median FPS | Face-label match | Skipped plain faces/frame | 20 FPS status |
| --- | ---: | ---: | ---: | ---: | --- |
| c_onnx_cpu_bbox_cwarp128 | 17.21 | 17.55 | 98.26% | 0.00 | FAIL: FPS |
| c_onnx_cpu_bbox_cwarp128_fruit_only | 19.50 | 19.59 | 89.57% | 6.45 | FAIL: face-label match and FPS |
| c_onnx_cpu_bbox_cwarp88_fruit_only | 19.01 | 19.47 | 90.00% | 6.45 | FAIL: face-label match and FPS |
| c_onnx_cpu_bbox_cwarp128_plain85 | 18.92 | 19.22 | 93.04% | 5.45 | FAIL: face-label match and FPS |
| c_onnx_cpu_bbox_cwarp128_plain90 | 15.33 | 15.38 | 93.04% | 5.20 | FAIL: face-label match and FPS |
| c_onnx_cpu_bbox_cwarp128_plain95 | 14.70 | 14.50 | 95.22% | 3.20 | FAIL: face-label match and FPS |

Conclusion:

- This path is rejected.
- Skipping C on A2 `plain_face` detections is fast and nearly reaches 20 FPS, but it breaks the face-label match requirement badly.
- Even a high confidence threshold of 0.95 only reaches 95.22% face-label match, below the 98% gate.
- A2 plain/fruit semantics are useful for visualization and candidate reasoning, but are not reliable enough to replace C classification under the current behavior-preservation rule.
- The best valid path remains `c_onnx_cpu_bbox_cwarp128` at about 17.2 mean FPS / 17.5 median FPS with 98.26% face-label match.
- Next promising work should either use a compatible A1/A2 TensorRT path, or train/validate a new plain fast-path model specifically against C labels before using it at runtime.

### E022 - A2 TensorRT Engine Export and Runtime Check

Status: completed, rejected; 20 FPS goal failed.

Reports and artifacts:

- `reports/abc_runtime_optimization/exp022_a2_engine_export.log`
- `reports/abc_runtime_optimization/exp022_a2_engine_export_b8.log`
- `reports/abc_runtime_optimization/exp022_a2_engine_runtime_b8.json`
- `reports/abc_runtime_optimization/artifacts/a2_yolo26n_seg_meta_v2_50000_fp32_dynamic_b8.engine`

Hypothesis:

- E021 showed that structural shortcuts can get near 20 FPS but fail behavior matching.
- A2 inference itself still costs about 10-13 ms in the valid paths.
- Earlier TensorRT export failed with `half=True` because TensorRT 11 no longer exposes `BuilderFlag.FP16` the way this Ultralytics version expects.
- Exporting A2 as FP32 TensorRT may avoid the FP16 API issue and speed up A2 while preserving the A2 output closely enough.

20 FPS pass criteria for this monitoring cycle:

- mean FPS >= 20.0
- median FPS >= 20.0
- face-label match >= 98%
- no shortcut that ignores non-front objects

Commands:

```powershell
$env:PYTHONWARNINGS='ignore'
@'
from pathlib import Path
from shutil import copy2
from ultralytics import YOLO

source = Path('runs/meta_v2_a2_faceseg/a2_yolo26n_seg_meta_v2_50000/weights/best.pt')
target = Path('reports/abc_runtime_optimization/artifacts/a2_yolo26n_seg_meta_v2_50000_fp32_dynamic_b8.engine')
target.parent.mkdir(parents=True, exist_ok=True)
model = YOLO(str(source))
exported = Path(model.export(format='engine', imgsz=224, device=0, half=False, dynamic=True, batch=8, verbose=False))
copy2(exported, target)
print(target, target.stat().st_size)
'@ | C:\Users\user\anaconda3\envs\ai_robotics\python.exe - 2>&1 |
  Tee-Object -FilePath reports\abc_runtime_optimization\exp022_a2_engine_export_b8.log

C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp022_a2_engine_runtime_b8.json `
  --sample-count 20 `
  --repeats 3 `
  --warmup 4 `
  --variants baseline c_onnx_cpu_bbox_cwarp128 a2_engine a2_engine_best_c
```

Change:

- Added `--a2-engine-model` to the benchmark script.
- Added benchmark-only variants:
  - `a2_engine`
  - `a2_engine_best_c`
- Runtime defaults are unchanged.

Environment notes:

- A2 FP32 TensorRT export succeeds with TensorRT `11.0.0.114`.
- The first dynamic engine was exported with `batch=4`, but benchmark frames can produce 5 A2 inputs, so it failed at runtime with TensorRT profile range `[1,3,32,32]..[4,3,448,448]`.
- Re-exporting with `batch=8` fixed the profile range issue.
- `half=False` avoids the previous TensorRT 11 `BuilderFlag.FP16` failure.

Result:

| Variant | Mean FPS | Median FPS | A2 ms | Face-label match | 20 FPS status |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | 11.14 | 11.10 | 13.04 | reference | FAIL |
| c_onnx_cpu_bbox_cwarp128 | 14.03 | 13.96 | 12.80 | 98.26% | FAIL: FPS |
| a2_engine | 11.55 | 11.45 | 7.78 | 97.38% | FAIL: face-label match and FPS |
| a2_engine_best_c | 14.85 | 14.85 | 8.00 | 97.39% | FAIL: face-label match and FPS |

Conclusion:

- A2 TensorRT does reduce A2 inference time from about 12-13 ms to about 8 ms.
- It is still rejected because face-label match drops below the required 98% gate to about 97.39%.
- Even ignoring the behavior miss, the full valid-looking pipeline stays below 20 FPS.
- A2 TensorRT may still be useful after retraining/validating against engine outputs or relaxing behavior equivalence, but it cannot be accepted under the current monitoring rule.
- The next promising route is either A1 TensorRT with behavior check, or an explicitly trained replacement for the expensive C/plain decision path rather than relying on A2 class semantics.

### E023 - A1 TensorRT Engine Export and Runtime Check

Status: completed, rejected; 20 FPS goal failed.

Reports and artifacts:

- `reports/abc_runtime_optimization/exp023_a1_engine_export.log`
- `reports/abc_runtime_optimization/exp023_a1_engine_runtime.json`
- `reports/abc_runtime_optimization/artifacts/a1_yolo26s_seg_meta_v2_50000_fp32_dynamic_b1.engine`

Hypothesis:

- The valid runtime path still spends about 11 ms/frame in A1 inference.
- A1 uses YOLO26s-seg, so TensorRT FP32 may recover several milliseconds while preserving behavior better than the A2 engine path.
- Combining A1 engine with the current best C path may approach 20 FPS without relying on a shortcut that ignores non-front objects.

20 FPS pass criteria for this monitoring cycle:

- mean FPS >= 20.0
- median FPS >= 20.0
- face-label match >= 98%
- no shortcut that ignores non-front objects

Commands:

```powershell
$env:PYTHONWARNINGS='ignore'
$log='reports\abc_runtime_optimization\exp023_a1_engine_export.log'
@'
from pathlib import Path
from shutil import copy2
from ultralytics import YOLO

source = Path('runs/meta_v2_a1_objectseg/a1_yolo26s_seg_meta_v2_50000/weights/best.pt')
target = Path('reports/abc_runtime_optimization/artifacts/a1_yolo26s_seg_meta_v2_50000_fp32_dynamic_b1.engine')
target.parent.mkdir(parents=True, exist_ok=True)
model = YOLO(str(source))
exported = Path(model.export(format='engine', imgsz=640, device=0, half=False, dynamic=True, batch=1, verbose=False))
copy2(exported, target)
print(target, target.stat().st_size)
'@ | C:\Users\user\anaconda3\envs\ai_robotics\python.exe - 2>&1 |
  Tee-Object -FilePath $log

C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp023_a1_engine_runtime.json `
  --sample-count 20 `
  --repeats 3 `
  --warmup 4 `
  --variants baseline c_onnx_cpu_bbox_cwarp128 a1_engine a1_engine_best_c a1_a2_engine_best_c
```

Change:

- Added `--a1-engine-model` to the benchmark script.
- Added benchmark-only variants:
  - `a1_engine`
  - `a1_engine_best_c`
  - `a1_a2_engine_best_c`
- Runtime defaults are unchanged.

Environment notes:

- A1 FP32 TensorRT export succeeds with TensorRT `11.0.0.114`.
- `half=False` avoids the TensorRT 11 `BuilderFlag.FP16` issue seen earlier.
- The exported A1 engine loads and runs through Ultralytics' TensorRT runtime path.

Result:

| Variant | Mean FPS | Median FPS | A1 ms | A2 ms | Face-label match | 20 FPS status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 11.47 | 11.33 | 11.47 | 12.68 | reference | FAIL |
| c_onnx_cpu_bbox_cwarp128 | 17.76 | 17.92 | 10.97 | 10.46 | 98.26% | FAIL: FPS |
| a1_engine | 11.50 | 11.40 | 7.71 | 12.85 | 100.00% | FAIL: FPS |
| a1_engine_best_c | 17.57 | 17.69 | 8.11 | 11.84 | 98.26% | FAIL: FPS |
| a1_a2_engine_best_c | 16.33 | 16.19 | 8.17 | 8.05 | 97.39% | FAIL: face-label match and FPS |

Conclusion:

- A1 TensorRT preserves behavior when used alone, and with the current best C path it still satisfies the 98% face-label match gate.
- The full runtime does not improve because A2 postprocessing and downstream face construction become the dominant cost; the best valid A1-engine path is slightly slower than the previous valid path in this run.
- Combining A1 and A2 engines is rejected because the A2 engine behavior drift again drops face-label match below 98%.
- The best valid path remains `c_onnx_cpu_bbox_cwarp128` at about 17.8 mean FPS / 17.9 median FPS.
- The next promising direction is not another model-format swap, but reducing Python/OpenCV overhead in A2 postprocessing and C crop assembly while keeping the same A2 detections and C labels.

### E024 - C Crop Perspective Matrix Reuse Probe

Status: completed, rejected; code reverted; 20 FPS goal failed.

Report:

- `reports/abc_runtime_optimization/exp024_c_crop_matrix_reuse.json`

Hypothesis:

- C crop assembly computes a perspective transform for the face crop, then computes essentially the same transform again when building the visible/hidden occlusion mask.
- Reusing the perspective matrix should preserve C inputs and reduce `c_warp_ms` / `c_crop_loop_ms`, especially in the baseline path where many faces are warped.

20 FPS pass criteria for this monitoring cycle:

- mean FPS >= 20.0
- median FPS >= 20.0
- face-label match >= 98%
- no shortcut that ignores non-front objects

Command:

```powershell
C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp024_c_crop_matrix_reuse.json `
  --sample-count 20 `
  --repeats 3 `
  --warmup 4 `
  --variants baseline c_onnx_cpu_bbox_cwarp128 a1_engine_best_c
```

Tested change:

- Added a temporary helper that returned both `cv2.warpPerspective(...)` output and the perspective matrix.
- Used that matrix for occlusion visible-mask projection instead of recomputing `order_quad_points(...)` and `cv2.getPerspectiveTransform(...)`.
- The runtime code change was reverted after the benchmark because it did not improve the accepted path.

Result:

| Variant | Mean FPS | Median FPS | C crop loop ms | Face-label match | 20 FPS status |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | 13.02 | 13.48 | 9.35 | reference | FAIL |
| c_onnx_cpu_bbox_cwarp128 | 16.14 | 16.27 | 6.18 | 98.26% | FAIL: FPS regression |
| a1_engine_best_c | 17.04 | 17.23 | 6.19 | 98.26% | FAIL: FPS |

Conclusion:

- The behavior gate remained satisfied for the valid paths, but the accepted `c_onnx_cpu_bbox_cwarp128` path regressed from the previous 17.76 / 17.92 FPS run to 16.14 / 16.27 FPS.
- The matrix reuse idea is therefore rejected and not retained in runtime code.
- The failure is useful: C crop matrix recomputation is not the main blocker under the current accepted path.
- The next most promising direction is a more direct A2 postprocessing target: reduce face filtering/refinement overhead or add an instrumentation split around `filter_cube_faces`, `cube_refine_shared_edges_if_sane`, and `refine_cube_quads_by_pose` to identify the specific hot loop without changing behavior first.

### E025 - Best C Path Without Pose Refinement

Status: completed, rejected; 20 FPS goal failed.

Report:

- `reports/abc_runtime_optimization/exp025_best_c_no_refine.json`

Hypothesis:

- E023/E024 showed that model-format swaps are not enough and that the large removable cost is in A2 postprocessing.
- The accepted path spends about 2.5-3 ms in `cube_refine_shared_edges_if_sane` and about 6-7 ms in `refine_cube_quads_by_pose`.
- If pose refinement can be disabled while keeping face labels sufficiently similar, the runtime could cross 20 FPS without skipping objects or faces.

20 FPS pass criteria for this monitoring cycle:

- mean FPS >= 20.0
- median FPS >= 20.0
- face-label match >= 98%
- no shortcut that ignores non-front objects

Command:

```powershell
C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp025_best_c_no_refine.json `
  --sample-count 20 `
  --repeats 3 `
  --warmup 4 `
  --variants baseline c_onnx_cpu_bbox_cwarp128 c_onnx_cpu_bbox_cwarp128_no_refine a1_engine_best_c_no_refine
```

Change:

- Added benchmark-only variants:
  - `c_onnx_cpu_bbox_cwarp128_no_refine`
  - `a1_engine_best_c_no_refine`
- Runtime defaults are unchanged.

Result:

| Variant | Mean FPS | Median FPS | A2 face filter ms | A2 refine ms | Face-label match | 20 FPS status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 12.74 | 13.29 | 9.59 | 6.20 | reference | FAIL |
| c_onnx_cpu_bbox_cwarp128 | 15.94 | 15.94 | 10.19 | 6.58 | 98.26% | FAIL: FPS |
| c_onnx_cpu_bbox_cwarp128_no_refine | 19.08 | 19.43 | 0.75 | 0.00 | 93.48% | FAIL: face-label match and FPS |
| a1_engine_best_c_no_refine | 19.98 | 20.13 | 0.75 | 0.00 | 93.48% | FAIL: face-label match and mean FPS |

Conclusion:

- Disabling pose refinement proves that the refinement path is the largest removable CPU-side cost.
- It nearly reaches the FPS gate when combined with A1 TensorRT, but it fails behavior preservation badly: face-label match drops to 93.48%, below the required 98%.
- This cannot be accepted as the final runtime path.
- The next cycle should not simply disable refinement; it should try a selective/adaptive refinement policy that keeps pose refinement only for cases likely to affect C labels, or optimize the refinement implementation itself.

### E026 - Pose Refinement Only When A2 Sees Fruit Faces

Status: completed, rejected; 20 FPS goal failed.

Report:

- `reports/abc_runtime_optimization/exp026_pose_fruit_refine.json`

Hypothesis:

- E025 showed that disabling pose refinement is fast but breaks behavior.
- A cheaper compromise is to keep pose refinement only for cube candidates where A2 detects at least one non-plain face, while skipping all-plain candidates.

Command:

```powershell
C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp026_pose_fruit_refine.json `
  --sample-count 20 `
  --repeats 3 `
  --warmup 4 `
  --variants baseline c_onnx_cpu_bbox_cwarp128 c_onnx_cpu_bbox_cwarp128_pose_fruit a1_engine_best_c_pose_fruit a1_engine_best_c_no_refine
```

Result:

| Variant | Mean FPS | Median FPS | Face-label match | 20 FPS status |
| --- | ---: | ---: | ---: | --- |
| c_onnx_cpu_bbox_cwarp128 | 16.16 | 16.26 | 98.26% | FAIL: FPS |
| c_onnx_cpu_bbox_cwarp128_pose_fruit | 16.97 | 17.13 | 94.35% | FAIL: face-label match and FPS |
| a1_engine_best_c_pose_fruit | 18.08 | 17.97 | 94.35% | FAIL: face-label match and FPS |
| a1_engine_best_c_no_refine | 20.53 | 21.02 | 93.48% | FAIL: face-label match |

Conclusion:

- Selectively refining only candidates with A2 fruit faces is not enough to preserve C labels.
- It improves speed slightly, but label match remains far below the 98% gate.
- This path is rejected.

### E027 - Pose Refinement Without Segment-IoU Safety Check

Status: completed, rejected; 20 FPS goal failed.

Report:

- `reports/abc_runtime_optimization/exp027_pose_fast_iou.json`

Hypothesis:

- Pose refinement itself may be acceptable if the expensive segment-IoU safety check is skipped.
- The existing move-ratio and geometry checks may be enough to preserve behavior while reducing refinement cost.

Command:

```powershell
C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp027_pose_fast_iou.json `
  --sample-count 20 `
  --repeats 3 `
  --warmup 4 `
  --variants baseline c_onnx_cpu_bbox_cwarp128 c_onnx_cpu_bbox_cwarp128_pose_fast_iou a1_engine_best_c_pose_fast_iou
```

Result:

| Variant | Mean FPS | Median FPS | Face-label match | 20 FPS status |
| --- | ---: | ---: | ---: | --- |
| c_onnx_cpu_bbox_cwarp128 | 16.01 | 16.12 | 98.26% | FAIL: FPS |
| c_onnx_cpu_bbox_cwarp128_pose_fast_iou | 16.87 | 16.96 | 93.91% | FAIL: face-label match and FPS |
| a1_engine_best_c_pose_fast_iou | 16.26 | 16.44 | 93.91% | FAIL: face-label match and FPS |

Conclusion:

- The segment-IoU safety check is not just overhead; it prevents behavior-changing refinements.
- Skipping it damages face-label match and does not get close enough to 20 FPS.
- This path is rejected.

### E028 - A2 Image Size Sweep Under Best-C Runtime

Status: completed, rejected; 20 FPS goal failed.

Report:

- `reports/abc_runtime_optimization/exp028_a2_imgsz_sweep_best_c.json`

Hypothesis:

- Lowering A2 input from 224 to 192 may reduce A2 inference and A2 postprocessing cost while preserving enough face geometry.

Command:

```powershell
C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp028_a2_imgsz_sweep_best_c.json `
  --sample-count 20 `
  --repeats 3 `
  --warmup 4 `
  --variants baseline c_onnx_cpu_bbox_cwarp128 c_onnx_cpu_bbox_cwarp128_a2_192 c_onnx_cpu_bbox_cwarp128_a2_208 a1_engine_best_c_a2_192 a1_engine_best_c_a2_208
```

Result:

| Variant | Mean FPS | Median FPS | Face-label match | 20 FPS status |
| --- | ---: | ---: | ---: | --- |
| c_onnx_cpu_bbox_cwarp128 | 15.42 | 15.64 | 98.26% | FAIL: FPS |
| c_onnx_cpu_bbox_cwarp128_a2_192 | 15.83 | 15.85 | 80.35% | FAIL: face-label match and FPS |
| a1_engine_best_c_a2_192 | 17.77 | 17.81 | 80.35% | FAIL: face-label match and FPS |
| c_onnx_cpu_bbox_cwarp128_a2_208 | invalid | invalid | invalid | INVALID: Ultralytics auto-upscaled 208 to 224 |
| a1_engine_best_c_a2_208 | invalid | invalid | invalid | INVALID: Ultralytics auto-upscaled 208 to 224 |

Correction:

- The `208` rows are invalid because Ultralytics warns that `imgsz=[208]` is not a multiple of stride 32 and updates it to `224`.
- After this mistake, runtime validation was added so `a2_imgsz` must be a multiple of 32 and invalid values fail early instead of producing misleading benchmark output.

Conclusion:

- A2 192 breaks behavior badly and is rejected.
- A2 208 is not a valid experiment and was removed from benchmark variants.

### E029 - B TinyQuadNet on CPU

Status: completed, rejected; 20 FPS goal failed.

Report:

- `reports/abc_runtime_optimization/exp029_b_cpu_best_c.json`

Hypothesis:

- B is a very small mask-to-quad network; CPU execution might avoid small GPU kernel overhead and synchronization cost.

Command:

```powershell
C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp029_b_cpu_best_c.json `
  --sample-count 20 `
  --repeats 3 `
  --warmup 4 `
  --variants baseline c_onnx_cpu_bbox_cwarp128 c_onnx_cpu_bbox_cwarp128_b_cpu a1_engine_best_c_b_cpu
```

Result:

| Variant | Mean FPS | Median FPS | B ms | Face-label match | 20 FPS status |
| --- | ---: | ---: | ---: | ---: | --- |
| c_onnx_cpu_bbox_cwarp128 | 15.91 | 16.11 | 3.89 | 98.26% | FAIL: FPS |
| c_onnx_cpu_bbox_cwarp128_b_cpu | 0.80 | 0.78 | 1170.61 | 97.83% | FAIL: catastrophic speed regression |
| a1_engine_best_c_b_cpu | 0.78 | 0.78 | 1196.25 | 97.83% | FAIL: catastrophic speed regression |

Conclusion:

- B on CPU is completely rejected on the RTX 5080 PC path.
- The optional `b_device` hook remains useful for explicit hardware experiments, but GPU is mandatory for this benchmark.

### E030 - Avoid Duplicate Quad Ordering in `quad_passes`

Status: completed, accepted as a partial optimization; 20 FPS goal still failed.

Report:

- `reports/abc_runtime_optimization/exp030_quad_metrics_fastpath.json`

Hypothesis:

- `quad_passes()` was ordering a quad, then calling `quad_metrics()`, which ordered the same quad again.
- This function is used repeatedly in A2 filtering/refinement and C filtering.
- Computing metrics directly from the already ordered quad should preserve behavior and reduce CPU overhead.

Command:

```powershell
C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp030_quad_metrics_fastpath.json `
  --sample-count 20 `
  --repeats 3 `
  --warmup 4 `
  --variants baseline c_onnx_cpu_bbox_cwarp128 a1_engine_best_c
```

Result:

| Variant | Mean FPS | Median FPS | Face-label match | 20 FPS status |
| --- | ---: | ---: | ---: | --- |
| c_onnx_cpu_bbox_cwarp128 | 16.20 | 16.50 | 98.26% | FAIL: FPS |
| a1_engine_best_c | 17.66 | 18.02 | 98.26% | FAIL: FPS |

Conclusion:

- This change preserves the face-label match gate and gives a small but valid CPU-side improvement.
- It is retained.
- The 20 FPS goal is still not achieved.

### E031 - TorchScript Trace for B Under Best-C Runtime

Status: completed, rejected; 20 FPS goal failed.

Report:

- `reports/abc_runtime_optimization/exp031_trace_b_best_c.json`

Hypothesis:

- Tracing TinyQuadNet could reduce B inference overhead without changing model outputs.

Command:

```powershell
C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp031_trace_b_best_c.json `
  --sample-count 20 `
  --repeats 3 `
  --warmup 4 `
  --variants baseline c_onnx_cpu_bbox_cwarp128 c_onnx_cpu_bbox_cwarp128_trace_b a1_engine_best_c_trace_b
```

Result:

| Variant | Mean FPS | Median FPS | Face-label match | 20 FPS status |
| --- | ---: | ---: | ---: | --- |
| c_onnx_cpu_bbox_cwarp128 | 17.09 | 17.26 | 98.26% | FAIL: FPS |
| c_onnx_cpu_bbox_cwarp128_trace_b | 16.48 | 16.56 | 98.26% | FAIL: FPS regression |
| a1_engine_best_c_trace_b | 17.33 | 17.24 | 98.26% | FAIL: FPS |

Conclusion:

- TorchScript tracing preserves behavior but does not improve speed on this benchmark.
- This path is rejected.

### E032 - Torch CUDA Fast-Path Settings

Status: completed, rejected as goal path; 20 FPS goal failed.

Report:

- `reports/abc_runtime_optimization/exp032_torch_cuda_fast_path.json`

Hypothesis:

- Enabling cuDNN benchmark mode and TF32 may improve A1/A2/B GPU stages and increase useful GPU utilization without changing model behavior.
- This should be tested because the user explicitly wants maximum useful GPU usage, but GPU work should only be kept when it improves end-to-end benchmark speed.

Command:

```powershell
C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp032_torch_cuda_fast_path.json `
  --sample-count 20 `
  --repeats 3 `
  --warmup 4 `
  --variants baseline c_onnx_cpu_bbox_cwarp128 c_onnx_cpu_bbox_cwarp128_cuda_fast a1_engine_best_c_cuda_fast
```

Result:

| Variant | Mean FPS | Median FPS | Face-label match | 20 FPS status |
| --- | ---: | ---: | ---: | --- |
| c_onnx_cpu_bbox_cwarp128 | 17.45 | 17.61 | 98.26% | FAIL: FPS |
| c_onnx_cpu_bbox_cwarp128_cuda_fast | 17.34 | 17.63 | 98.26% | FAIL: FPS |
| a1_engine_best_c_cuda_fast | 18.06 | 18.38 | 98.26% | FAIL: FPS |

GPU-utilization implication:

- CUDA fast-path settings do not meaningfully change the bottleneck because the remaining runtime is dominated by CPU/OpenCV postprocessing and face refinement gaps between GPU model calls.
- For this benchmark, increasing useful GPU utilization requires moving or batching more CPU-side geometry/crop work, not just toggling CUDA backend settings.

Conclusion:

- Behavior is preserved, but the FPS goal is not met.
- The optional CUDA fast-path hook can stay for explicit testing, but it is not enough as the final solution.

### E033 - Frame-Level B Batching Across Objects

Status: completed, accepted as partial optimization; 20 FPS goal still failed.

Report:

- `reports/abc_runtime_optimization/exp033_frame_b_batch.json`

Hypothesis:

- The old cascade called TinyQuadNet B separately per cube candidate.
- For 4 cube-like objects and about 12 faces, this creates several small GPU calls and CPU/GPU synchronization points.
- B should be batched once per frame across all A2 face masks to make GPU work coarser and reduce per-object overhead.

Command:

```powershell
C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp033_frame_b_batch.json `
  --sample-count 20 `
  --repeats 3 `
  --warmup 4 `
  --variants baseline c_onnx_cpu_bbox_cwarp128 c_onnx_cpu_bbox_cwarp128_frame_b a1_engine_best_c_frame_b
```

Result:

| Variant | Mean FPS | Median FPS | B ms | Face-label match | 20 FPS status |
| --- | ---: | ---: | ---: | ---: | --- |
| c_onnx_cpu_bbox_cwarp128 | 16.98 | 17.56 | 3.92 | 98.26% | FAIL: FPS |
| c_onnx_cpu_bbox_cwarp128_frame_b | 18.25 | 18.47 | 1.24 | 98.26% | FAIL: FPS |
| a1_engine_best_c_frame_b | 18.77 | 19.17 | 1.13 | 98.26% | FAIL: FPS |

GPU-utilization implication:

- B batching makes useful GPU work coarser and reduces small-call overhead.
- This is a real structural improvement over minor per-function tweaks.

Conclusion:

- Accepted as a partial optimization because behavior is preserved and B time drops from about 3.9 ms to about 1.1 ms.
- The full 20 FPS goal is not yet achieved.

### E034 - Frame-Level B and C Batching Across Objects

Status: completed, passing in this run; 20 FPS goal achieved in E034 but verified again in E035/E036.

Report:

- `reports/abc_runtime_optimization/exp034_frame_bc_batch.json`

Hypothesis:

- E033 fixed B batching, but C was still called once per object.
- C ONNX CPU inference should also be batched once per frame across all face crops.
- This directly attacks the cascade structure problem: too many small per-object calls.

Command:

```powershell
C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp034_frame_bc_batch.json `
  --sample-count 20 `
  --repeats 3 `
  --warmup 4 `
  --variants baseline c_onnx_cpu_bbox_cwarp128 c_onnx_cpu_bbox_cwarp128_frame_b c_onnx_cpu_bbox_cwarp128_frame_bc a1_engine_best_c_frame_bc
```

Result:

| Variant | Mean FPS | Median FPS | B ms | C ms | Face-label match | 20 FPS status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| c_onnx_cpu_bbox_cwarp128 | 16.91 | 17.28 | 3.71 | 6.10 | 98.26% | FAIL: FPS |
| c_onnx_cpu_bbox_cwarp128_frame_b | 17.54 | 18.02 | 1.27 | 6.12 | 98.26% | FAIL: FPS |
| c_onnx_cpu_bbox_cwarp128_frame_bc | 19.67 | 19.75 | 1.12 | 3.41 | 98.26% | FAIL: slightly below FPS |
| a1_engine_best_c_frame_bc | 20.68 | 20.83 | 1.15 | 3.41 | 98.26% | PASS |

GPU-utilization implication:

- B and C frame batching increases useful batching and reduces CPU/GPU/runtime call fragmentation.
- This validates the structural diagnosis: the main issue was not a single slow model but too many small cascade calls around CPU postprocessing.

Conclusion:

- This is the first passing configuration in a standard 20-sample / 3-repeat benchmark.
- Because the margin is modest, E035 and E036 were run for follow-up validation.

### E035 - Final Candidate Longer Repeat Check

Status: completed, not accepted as final because mean FPS dipped below 20.

Report:

- `reports/abc_runtime_optimization/exp035_final_frame_bc_goal_check.json`

Command:

```powershell
C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp035_final_frame_bc_goal_check.json `
  --sample-count 20 `
  --repeats 5 `
  --warmup 6 `
  --variants baseline c_onnx_cpu_bbox_cwarp128 a1_engine_best_c_frame_bc
```

Result:

| Variant | Mean FPS | Median FPS | Face-label match | 20 FPS status |
| --- | ---: | ---: | ---: | --- |
| a1_engine_best_c_frame_bc | 19.84 | 20.27 | 98.26% | FAIL: mean FPS slightly below 20 |

Conclusion:

- The final candidate is very close but not fully stable in this longer repeat run.
- This result is kept as a warning that the margin is thin.
- Additional frame-level batching and GPU/runtime variants were tested in E036.

### E036 - Frame-Level B/C Batching With CUDA and Half Variants

Status: completed, accepted; 20 FPS goal passed.

Report:

- `reports/abc_runtime_optimization/exp036_frame_bc_half_cuda.json`

Hypothesis:

- E034/E035 show that frame-level B/C batching is the right structural fix, but the margin is thin.
- Test whether CUDA fast-path settings or YOLO half precision provide a better final configuration.

Command:

```powershell
C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\scripts\benchmark_abc_runtime.py `
  --output reports\abc_runtime_optimization\exp036_frame_bc_half_cuda.json `
  --sample-count 20 `
  --repeats 4 `
  --warmup 6 `
  --variants baseline a1_engine_best_c_frame_bc a1_engine_best_c_frame_bc_cuda_fast a1_engine_best_c_frame_bc_yolo_half a1_engine_best_c_frame_bc_yolo_half_cuda_fast
```

Result:

| Variant | Mean FPS | Median FPS | Face-label match | 20 FPS status |
| --- | ---: | ---: | ---: | --- |
| a1_engine_best_c_frame_bc | 20.12 | 20.46 | 98.26% | PASS |
| a1_engine_best_c_frame_bc_cuda_fast | 19.60 | 19.76 | 98.26% | FAIL: FPS |
| a1_engine_best_c_frame_bc_yolo_half | 19.90 | 20.20 | 96.52% | FAIL: face-label match and mean FPS |
| a1_engine_best_c_frame_bc_yolo_half_cuda_fast | 19.71 | 19.95 | 96.52% | FAIL: face-label match and FPS |

GPU-utilization implication:

- The passing path is not the highest nominal GPU setting; it is the path that batches small B/C work across the frame while preserving A2 FP32 behavior.
- YOLO half precision is rejected because it changes behavior and does not reliably improve end-to-end speed.
- CUDA fast-path settings are rejected for this final path because they reduce the measured FPS in this benchmark.

Conclusion:

- Final passing runtime candidate: `a1_engine_best_c_frame_bc`.
- It keeps all objects/faces in the 4-object/12-face benchmark, preserves face-label match at 98.26%, and reports 20.12 mean / 20.46 median FPS in E036.
- The main retained structural changes are:
  - A1 TensorRT engine path from E023.
  - Frame-level B batching from E033.
  - Frame-level C batching from E034.
  - Quad metric duplicate-ordering removal from E030.
- Remaining risk: margin is thin, as shown by E035. A larger validation set and Jetson-side measurement are still needed before treating this as a competition-ready deployment speed.

### E037-E042 - 22 FPS Stability Push and Current Usable Preset

Status: completed; no 22 FPS configuration passed the 98% face-label match gate. The best immediately usable preset is `best_stable_5080`.

Reports:

- `reports/abc_runtime_optimization/exp037_22fps_reindex_cwarp_sweep.json`
- `reports/abc_runtime_optimization/exp038_22fps_a2_engine_frame_bc_sweep.json`
- `reports/abc_runtime_optimization/exp039_22fps_geometry_cache.json`
- `reports/abc_runtime_optimization/exp040_22fps_refine_and_cconf_sweep.json`
- `reports/abc_runtime_optimization/exp041_22fps_projective_fallback_for_pose_fruit.json`
- `reports/abc_runtime_optimization/exp042_22fps_pose_fruit_combo_sweep.json`

22 FPS pass criteria for this push:

- mean FPS >= 22.0
- median FPS >= 22.0
- face-label match >= 98%
- no shortcut that ignores non-front objects or faces

Changes kept:

- Added geometry caches for face center, area, edge points, and polygon-IoU pairs inside a frame.
- Added frame-level B/C runtime options to the user-facing image/video/webcam scripts.
- Added `--runtime-preset best_stable_5080`.
- Added `--runtime-preset fast_unverified_a2_engine` for explicit speed-only testing.
- Added `--disable-stage-timing` to the benchmark script for later actual-runtime checks.
- Fixed projective raw-quad fallback so `pose_fruit` and `pose_fast_iou` are treated as projective modes, not only `pose`.

Key results:

| Variant | Mean FPS | Median FPS | Face-label match | Status |
| --- | ---: | ---: | ---: | --- |
| `a1_engine_best_c_frame_bc_cwarp112` after geometry cache | 21.06 | 21.20 | 98.26% | best valid 22-FPS push candidate, but below 22 |
| `a1_engine_best_c_frame_bc_pose_fast_iou` after fallback fix | 20.34 | 20.71 | 98.26% | valid behavior, too slow |
| `a1_engine_best_c_frame_bc_pose_fruit_cwarp96` | 20.92 | 20.89 | 98.26% | valid behavior, too slow |
| `a1_a2_engine_best_c_frame_bc` | 22.21 | 22.50 | 97.39% | fast but rejected for behavior |
| `a1_a2_engine_best_c_frame_bc_pose_fruit` | 22.90 | 22.69 | 97.39% | fast but rejected for behavior |
| `a1_engine_best_c_frame_bc_no_refine` | 24.90 | 25.33 | 93.48% | rejected; fitting/refine is behavior-critical |

Current immediately usable preset:

```powershell
C:\Users\user\anaconda3\envs\ai_robotics\python.exe jetson\realtime_seg_cam.py `
  --pipeline abc `
  --runtime-preset best_stable_5080 `
  --camera 0 `
  --device 0 `
  --target-shape cube `
  --target-fruit apple
```

`best_stable_5080` expands to:

- A1 TensorRT engine: `reports/abc_runtime_optimization/artifacts/a1_yolo26s_seg_meta_v2_50000_fp32_dynamic_b1.engine`
- A2 PyTorch model, not A2 TensorRT, because A2 TensorRT is faster but below the 98% behavior gate.
- C ONNX CPU: `reports/abc_runtime_optimization/artifacts/c_mobilenetv3small_runtimewarp_warmplain_ft.onnx`
- `c_warp_size=112`
- `use_a1_object_mask=False`
- `batch_b_across_objects=True`
- `batch_c_across_objects=True`
- `refine_quads=pose`

Conclusion:

- The current cascade can be made stable around 20-21 FPS on the RTX 5080 benchmark while preserving the 98% behavior gate.
- The 22 FPS goal is reachable only by relaxing behavior, mostly through A2 TensorRT or reduced pose refinement.
- The fundamental blocker is architectural: A2/B/pose/C still do too much per-face CPU/OpenCV geometry work after A1.
- The next real speed jump should come from replacing A2+B+C with a single cube-crop face model, not from more minor cascade tuning.
