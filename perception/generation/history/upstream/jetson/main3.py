from ultralytics import YOLO
from pathlib import Path
import cv2

# =========================
# 설정
# =========================
model_path = "last.pt"              # 학습된 YOLO 모델 경로
image_dir = "image"                # 입력 이미지 폴더
output_dir = "runs/predict_images"  # 결과 저장 폴더

conf_thres = 0.25                   # confidence threshold
imgsz = 640                         # inference image size

# =========================
# 모델 로드
# =========================
model = YOLO(model_path)

# =========================
# 이미지 경로 수집
# =========================
image_dir = Path(image_dir)
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

image_paths = [
    p for p in image_dir.iterdir()
    if p.suffix.lower() in image_extensions
]

if not image_paths:
    raise FileNotFoundError(f"No image files found in {image_dir}")

# =========================
# 전체 이미지 inference
# =========================
for image_path in image_paths:
    print(f"Processing: {image_path.name}")

    results = model.predict(
        source=str(image_path),
        conf=conf_thres,
        imgsz=imgsz,
        save=False,
        verbose=False
    )

    # YOLO 결과 시각화 이미지 생성
    result = results[0]
    annotated_img = result.plot()

    # 결과 이미지 저장
    save_path = output_dir / image_path.name
    cv2.imwrite(str(save_path), annotated_img)

print(f"\nDone. Results saved to: {output_dir}")