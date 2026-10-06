# Unified Runtime Policy and Bottleneck Check - 2026-07-03

## Purpose

Check whether the current runtime/test path follows the cube-face unified training input policy, then measure where the runtime bottleneck is on a fixed 30 second webcam capture.

No new webcam capture was started during this check. The benchmark reused:

```text
reports/share/fair_30s_compare_20260702_155104/capture_30s_mjpg.avi
```

## Model Lineage

A1 object segmentation:

```text
base model: yolo26s-seg.pt
runtime weight: jetson/ABC_model/meta_v2_a1_objectseg/a1_yolo26s_seg_meta_v2_50000/weights/best.pt
```

Cube-face unified face segmentation:

```text
base lineage:
yolo26n-seg.pt
  -> cube_face_unified_yolo26n_seg_v1
  -> cube_face_unified_yolo26n_seg_wholefruit_plainhard_balanced_ft_v2
  -> cube_face_unified_yolo26n_seg_verified_fruit_balanced_from_last_adamw_lr5e5_ft_v1
  -> cube_face_unified_yolo26n_seg_verified_plus_flat_icon_boundary_adamw_lr3e5_ft_v1

runtime weight: jetson/ABC_model/cube_face_unified/preferred_v2/weights/best.pt
```

## Policy Audit

Training export policy in `scripts/export_meta_v2_cube_face_unified_dataset.py`:

```text
A1/meta object bbox
-> expand by max(width, height) * crop_pad
-> crop from the original image
-> cv2.resize(crop, crop_size x crop_size, INTER_AREA)
-> train unified YOLO segmentation on visible face masks
```

Runtime/test mismatch found:

```text
Before this change, the unified runtime passed the raw rectangular A1 crop to YOLO.
That allowed Ultralytics preprocessing to letterbox the input, which differed from the square resized training crop.
```

Runtime/test path was updated to match the training policy:

```text
A1 cube_like_object only
-> expand crop by max(width, height) * 0.18
-> crop from the original frame
-> resize to imgsz x imgsz, default 224 x 224
-> run unified face model
-> scale result boxes/masks back to the original frame for overlay and decision
```

Unified does not use blacken. Occlusion is represented by the raw visible crop pixels and visible face segmentation masks. Blacken remains ABC C-warp specific.

Files updated:

```text
jetson/realtime_seg_cam.py
scripts/evaluate_cube_face_unified_runtime.py
scripts/benchmark_cube_face_unified_runtime.py
```

## Fixed-Capture CPU Benchmark

Conditions:

```text
device: cpu
A1 imgsz: 640
unified face imgsz: 224
A1 conf: 0.25
face conf: 0.25
crop pad: 0.18
frames: 294
average cube crops per frame: 5.00
```

Result:

| Stage | Mean | Median | P90 | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| A1 full-frame predict | 168.7 ms | 168.4 ms | 180.6 ms | 141.9 ms | 271.2 ms |
| Crop + resize prep | 1.3 ms | 1.2 ms | 1.6 ms | 0.7 ms | 2.7 ms |
| Unified face batch per frame | 85.0 ms | 84.9 ms | 96.1 ms | 66.3 ms | 136.6 ms |
| Unified face per cube | 17.0 ms | 17.0 ms | 19.2 ms | 13.3 ms | 27.3 ms |
| A1 + unified model sum | 253.7 ms | 253.2 ms | 274.4 ms | 215.2 ms | 407.8 ms |

Processed FPS:

```text
3.83 FPS
```

## Conclusion

On this Windows CPU fixed-capture run, A1 is the primary bottleneck:

```text
A1 full-frame predict mean: 168.7 ms
Unified face model for 5 cube crops mean: 85.0 ms
Unified face model per cube mean: 17.0 ms
```

Optimization priority should be A1 first. The unified face model is cheaper per cube when crops are batched.

## Notes

ONNX aliases were added for A1 and unified runtime convenience, but the local Windows CPU ONNX Runtime path was slower than PyTorch in the quick test. For Jetson speed work, TensorRT engine export should be preferred over assuming ONNX CPU will be faster.
