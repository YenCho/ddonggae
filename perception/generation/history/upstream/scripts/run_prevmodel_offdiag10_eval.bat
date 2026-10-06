@echo off
cd /d "%~dp0.."
echo === Evaluating PRE-PRUNING model on offdiag10_qr hard probe (120 crops) ===
"C:\Users\user\anaconda3\envs\ai_robotics\python.exe" scripts\evaluate_cube_face_unified_dataset_confusion.py ^
  --dataset datasets\cube_face_unified_hard_probe_offdiag10_qr_v1 ^
  --model runs\segment\cube_face_unified_yolo26n_seg_hsv_ratio_20000_stronger_from_color_shift_last_adamw_lr1e5_ft_v1\weights\last.pt ^
  --output reports\cube_face_unified_eval\hard_probe_offdiag10_qr_v1_prevmodel ^
  --split val ^
  --prediction_mode fruit_for_fruit_truth
echo.
echo Done. Summary: reports\cube_face_unified_eval\hard_probe_offdiag10_qr_v1_prevmodel\summary.md
pause
