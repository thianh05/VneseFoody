from ultralytics import YOLO
import cv2

# Tải mô hình YOLO
model = YOLO("best.pt")

# ---- CHỌN NGUỒN NHẬN DIỆN ----
# source = "0" -> webcam
# source = "path/to/image.jpg" -> ảnh
# source = "path/to/video.mp4" -> video
source = 0  # dùng webcam, đổi nếu muốn test ảnh

# Mở webcam hoặc đọc file
cap = cv2.VideoCapture(source)

if not cap.isOpened():
    print("❌ Không mở được nguồn video/ảnh.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Dự đoán bằng YOLO
    results = model(frame)

    # Vẽ kết quả lên ảnh
    annotated_frame = results[0].plot()

    # Hiển thị
    cv2.imshow("🍜 Food Detection", annotated_frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
