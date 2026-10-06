import cv2
from ultralytics import YOLO

# =========================
# 설정
# =========================
MODEL_PATH = "last.pt"   # 다운로드한 .pt 파일 경로
CAMERA_INDEX = 0         # 기본 웹캠: 0, 외장 카메라: 1 또는 2
CONF_THRESHOLD = 0.25

# =========================
# 모델 로드
# =========================
model = YOLO(MODEL_PATH)

# =========================
# 카메라 열기
# =========================
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다. CAMERA_INDEX를 0, 1, 2 등으로 바꿔보세요.")

print("YOLO webcam detection started.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # YOLO 추론
    results = model.predict(
        source=frame,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    # 결과 시각화
    annotated_frame = results[0].plot()

    # 화면 출력
    cv2.imshow("YOLO Webcam Detection", annotated_frame)

    # q 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()