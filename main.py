from ultralytics import YOLO
import cv2

# Load YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect objects
    results = model(frame)

    # Draw detection boxes
    annotated_frame = results[0].plot()

    # Show output
    cv2.imshow("YOLO Ball Detection", annotated_frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()