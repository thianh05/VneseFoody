from ultralytics import YOLO
from tkinter import Tk, filedialog
import cv2

# Hide Tkinter main window
root = Tk()
root.withdraw()

# Choose an image file
file_path = filedialog.askopenfilename(
    title="Select an image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
)

if not file_path:
    print("No image selected. Exiting.")
else:
    print(f"Selected image: {file_path}")

    # Load model
    model = YOLO("final.pt")

    # Predict (no auto-close)
    results = model.predict(source=file_path, conf=0.3, show=False)

    # Show detected image manually
    for result in results:
        annotated_frame = result.plot()  # Draw boxes and labels
        cv2.imshow("Detection Result - Press ESC to close", annotated_frame)

        print("\nDetected objects:")

        if result.boxes is None or len(result.boxes) == 0:
            print(" - No objects detected.")
        else:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])
                print(f" - {label} ({conf:.2f})")

        # Wait until ESC pressed
        while True:
            key = cv2.waitKey(10)
            if key == 27:  # ESC key
                break

    cv2.destroyAllWindows()
