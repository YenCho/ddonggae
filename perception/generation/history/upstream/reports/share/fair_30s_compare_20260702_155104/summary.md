# Fair Webcam FPS Compare

- Capture frames: 294
- Capture elapsed: 30.32s
- Capture FPS: 9.70
- Sample frame: 5.0s, index 43
- Device: cpu

| Pipeline | 30s avg FPS | Mean frame ms | Notes |
|---|---:|---:|---|
| Current unified | 3.83 | 261.0 | A1 crop -> unified face seg/class, no warp |
| Old ABC cascade | 1.29 | 775.8 | A1/A2/B/C, C warp panel shown in sample |

## Stage Means

- Current unified: a1_ms=153.5ms, face_ms=100.7ms, total_ms=259.9ms
- Old ABC cascade: a1_ms=553.7ms, a2_ms=78.3ms, b_ms=5.9ms, c_ms=8.3ms, total_ms=713.7ms

## Files

- Side by side: `reports/share/fair_30s_compare_20260702_155104/side_by_side_5s_same_capture_30s_avg.jpg`
- Matched-label side by side: `reports/share/fair_30s_compare_20260702_155104/side_by_side_5s_same_capture_30s_avg_matched_labels.jpg`
- Unified: `reports/share/fair_30s_compare_20260702_155104/current_unified_5s.jpg`
- ABC cascade: `reports/share/fair_30s_compare_20260702_155104/old_abc_cascade_5s_with_warp.jpg`
- Raw capture: `reports/share/fair_30s_compare_20260702_155104/capture_30s_mjpg.avi`
- Raw benchmark JSON: `reports/share/fair_30s_compare_20260702_155104/benchmark.json`

## Unified Sample Lines

- A1 objects=8 cube crops=5 fruit cubes=4 unified faces=14
- o0 A1=cube_like_object conf=0.978 crop=[298,366,482,545]
- A1 octahedron conf=0.976 box=[678,356,788,471]
- A1 icosahedron conf=0.972 box=[780,387,876,484]
- o1 A1=cube_like_object conf=0.968 crop=[325,283,470,419]
- o2 A1=cube_like_object conf=0.960 crop=[429,309,550,442]
- o3 A1=cube_like_object conf=0.951 crop=[559,315,688,457]
- A1 dodecahedron conf=0.944 box=[654,289,729,363]
- o4 A1=cube_like_object conf=0.915 crop=[544,280,658,405]
- o0 decision=fruit_cube:banana action=avoid faces=3 blank=1 fruit={'banana': 2} unknown=0
- o1 decision=fruit_cube:pineapple action=avoid faces=3 blank=1 fruit={'pineapple': 2} unknown=0
- o2 decision=fruit_cube:orange action=avoid faces=3 blank=2 fruit={'orange': 1} unknown=0

## ABC Sample Lines

- 0: octahedron conf=0.984 action=skip identity=octahedron faces=0 blank=0 fruit=- unknown=0 reason=non-target shape
- 1: cube_like_object conf=0.982 action=avoid identity=fruit_cube:banana faces=3 blank=1 fruit=banana:1 unknown=1 reason=non-target fruit is visible; wrong set-2 pickup costs double its 20 point value
- 2: icosahedron conf=0.981 action=skip identity=icosahedron faces=0 blank=0 fruit=- unknown=0 reason=non-target shape
- 3: cube_like_object conf=0.978 action=avoid identity=fruit_cube:orange faces=3 blank=2 fruit=orange:1 unknown=0 reason=non-target fruit is visible; wrong set-2 pickup costs double its 20 point value
- 4: cube_like_object conf=0.975 action=inspect identity=conflicting_fruit_cube faces=3 blank=0 fruit=pineapple:1,orange:1 unknown=1 reason=multiple fruit classes on one cube conflict with the rulebook same-fruit-face constraint
- 5: cube_like_object conf=0.975 action=pickup identity=plain_cube faces=3 blank=3 fruit=- unknown=0 reason=three or more visible faces are confidently blank and no fruit/unknown face evidence is present; target set-1 cube is worth 10 points
- 6: dodecahedron conf=0.966 action=skip identity=dodecahedron faces=0 blank=0 fruit=- unknown=0 reason=non-target shape
- 7: cube_like_object conf=0.961 action=pickup identity=fruit_cube:apple faces=2 blank=0 fruit=apple:2 unknown=0 reason=target fruit face is visible; rulebook set-2 object is worth 20 points

## Matched-label Rerender

- ABC cascade labels were rerendered to final identity only, matching unified.
