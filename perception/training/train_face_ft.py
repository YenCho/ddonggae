import os
from ultralytics import YOLO
# 실사(본선 물체) + 합성 혼합 파인튜닝 — 2026-07-24
# 시작점은 **현재 실기에 배포된 모델**. 기존 train_n_ft.py 계약을 그대로 따르되
# hsv_h 만 0.015 -> 0.005 로 낮춘다: hue 지터가 apple<->orange 경계를 뭉개는데
# 그게 이번 오분류(덜 익은 사과를 orange 0.95 로 오독)의 원인 축이기 때문.
init = os.path.expanduser("~/team14_face/runs/deployed_unified_face_20260719.pt")
m = YOLO(init)
m.train(
    data=os.path.expanduser("~/team14_face/datasets/mix_ft_20260724/data.yaml"),
    epochs=80, imgsz=224, batch=128, device=0, workers=8, cache=False, amp=True,
    optimizer="AdamW", lr0=5e-5, lrf=0.05, cos_lr=True, warmup_epochs=1,
    close_mosaic=0, mosaic=0.0, copy_paste=0.0, mixup=0.0,
    hsv_h=0.005,
    project=os.path.expanduser("~/team14_face/runs"), name="face_ft_real_20260724",
    exist_ok=True, patience=25,
)
