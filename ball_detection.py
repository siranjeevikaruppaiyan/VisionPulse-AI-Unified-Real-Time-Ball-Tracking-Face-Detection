from ultralytics import YOLO
import cv2
import time

# ----------------------------
# Load YOLO Model
# ----------------------------
# Recommended:
# model = YOLO("models/yolov8m.pt")
# If you only have nano:
model = YOLO("models/yolov8n.pt")

# ----------------------------
# Open Webcam
# ----------------------------
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # ----------------------------
    # YOLO Detection
    # ----------------------------
    results = model(
        frame,
        conf=0.60,
        iou=0.45,
        verbose=False
    )

    best_ball = None
    best_conf = 0

    # Search only for the best sports ball
    for box in results[0].boxes:

        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        if class_name == "sports ball":

            if confidence > best_conf:
                best_conf = confidence
                best_ball = box

    # ----------------------------
    # Draw only one ball
    # ----------------------------
    if best_ball is not None:

        x1, y1, x2, y2 = map(int, best_ball.xyxy[0])

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        cv2.circle(
            frame,
            (center_x, center_y),
            5,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            f"SPORTS BALL : {best_conf*100:.1f}%",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Center: ({center_x}, {center_y})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "SPORTS BALL : NOT DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # ----------------------------
    # FPS
    # ----------------------------
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS : {int(fps)}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    # ----------------------------
    # Display
    # ----------------------------
    cv2.imshow("Hackronics AI Ball Detection", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()