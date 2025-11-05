from ultralytics import YOLO
import torch

def main():
    # Kiểm tra GPU
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        print(f"✅ Using GPU: {device}")
    else:
        print("⚠️ GPU not found! Using CPU instead.")
    
    # Khởi tạo model YOLOv8 (có thể đổi sang yolov8s.pt, yolov8m.pt,...)
    model = YOLO("yolov8n.pt")  # model nhẹ, phù hợp RTX 3060 Ti

    # Train mô hình
    results = model.train(
        data="data.yaml",    # đường dẫn file cấu hình dataset
        epochs=100,          # số epoch
        imgsz=640,           # kích thước ảnh
        batch=16,            # batch size, tùy GPU
        name="food_yolov8",  # tên folder lưu kết quả
        project="runs/train",# thư mục gốc chứa kết quả
        device=0,            # 0 = GPU đầu tiên, -1 = CPU
        workers=4,           # số luồng xử lý dữ liệu
        patience=20,         # early stopping
        save_period=10,      # lưu checkpoint mỗi 10 epoch
        optimizer="SGD",     # hoặc 'Adam', 'AdamW'
        pretrained=True,     # dùng pretrained weights
        verbose=True,        # in log chi tiết
    )

    print("\n✅ Training completed!")
    print(f"Results saved in: {results.save_dir}")

if __name__ == "__main__":
    main()
