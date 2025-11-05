import cv2
import numpy as np
from ultralytics import YOLO
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QTextEdit
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt
import sys


class FoodDetector:
    def __init__(self, model_path="final.pt"):
        print("🔹 Loading YOLOv8 model...")
        self.model = YOLO(model_path)
        print("✅ Model loaded successfully!")

    def detect(self, image_path, top_k=5):
        # Run prediction
        results = self.model.predict(source=image_path, conf=0.3, show=False)
        result = results[0]

        # --- CASE 1: Detection model ---
        if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            data = []
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = self.model.names[cls_id]
                data.append((label, conf))

            data = sorted(data, key=lambda x: x[1], reverse=True)[:top_k]
            info = "\n".join([f"{label} ({conf:.2f})" for label, conf in data])
            img = result.plot()
            return img, info

        # --- CASE 2: Classification model ---
        elif hasattr(result, "probs") and result.probs is not None:
            probs = result.probs.data.cpu().numpy()
            class_names = list(self.model.names.values())
            top_indices = np.argsort(probs)[::-1][:top_k]
            info = "\n".join([f"{class_names[i]} ({probs[i]:.2f})" for i in top_indices])

            img = cv2.imread(image_path)
            return img, info

        else:
            return None, "No objects detected."


class FoodDetectorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.detector = FoodDetector()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("🍱 Food Detector (YOLOv8)")
        self.resize(800, 600)

        layout = QVBoxLayout()

        self.label = QLabel("Click 'Choose Image' to start detection")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        layout.addWidget(self.result_box)

        self.btn = QPushButton("Choose Image")
        self.btn.clicked.connect(self.load_image)
        layout.addWidget(self.btn)

        self.setLayout(layout)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
        )

        if not file_path:
            return

        img, info = self.detector.detect(file_path)
        self.result_box.setText(info)

        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(q_img).scaled(600, 400, Qt.KeepAspectRatio)
            self.image_label.setPixmap(pix)
        else:
            self.image_label.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FoodDetectorGUI()
    window.show()
    sys.exit(app.exec())
