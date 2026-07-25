import cv2
import mediapipe as mp
import time
import math
from datetime import datetime

# -----------------------------
# MediaPipe Face Detection Init
# -----------------------------
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.7
)

# -----------------------------
# Camera / Geometry Parameters
# -----------------------------
KNOWN_FACE_WIDTH = 0.16      # W: Average real face width in meters
FOCAL_LENGTH = 700           # f: Focal length in pixels

# -----------------------------
# Video Capture Setup
# -----------------------------
cap = cv2.VideoCapture(0)
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Frame center (c_x, c_y)
    frame_c_x = w // 2
    frame_c_y = h // 2

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    # -----------------------------
    # Draw Screen Center Crosshairs
    # -----------------------------
    cv2.line(frame, (frame_c_x - 20, frame_c_y), (frame_c_x + 20, frame_c_y), (255, 255, 255), 2)
    cv2.line(frame, (frame_c_x, frame_c_y - 20), (frame_c_x, frame_c_y + 20), (255, 255, 255), 2)

    if results.detections:
        for face_id, detection in enumerate(results.detections, start=1):
            
            # Extract Bounding Box
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            confidence = detection.score[0] * 100

            # Dynamic Bounding Box Color based on Confidence
            if confidence > 90:
                box_color = (0, 255, 0)     # Green
            elif confidence > 75:
                box_color = (0, 255, 255)   # Yellow
            else:
                box_color = (0, 0, 255)     # Red

            cv2.rectangle(frame, (x, y), (x + width, y + height), box_color, 3)

            # -----------------------------
            # Depth Estimation (Z)
            # -----------------------------
            if width > 0:
                # Z = (f * W) / w_px
                distance = (KNOWN_FACE_WIDTH * FOCAL_LENGTH) / width

                if distance < 0.5:
                    status = "TOO CLOSE"
                    status_color = (0, 0, 255)
                elif distance <= 1.5:
                    status = "SAFE"
                    status_color = (0, 255, 0)
                else:
                    status = "TOO FAR"
                    status_color = (0, 255, 255)
            else:
                distance = 0
                status = "UNKNOWN"
                status_color = (255, 255, 255)

            # -----------------------------
            # Angle Estimation (Theta)
            # -----------------------------
            face_c_x = x + width // 2
            face_c_y = y + height // 2

            offset_x = face_c_x - frame_c_x
            offset_y = face_c_y - frame_c_y

            # Using Precise Pinhole Camera Math: theta = arctan((x - c_x) / f)
            horizontal_angle = math.degrees(math.atan(offset_x / FOCAL_LENGTH))
            vertical_angle = math.degrees(math.atan(offset_y / FOCAL_LENGTH))

            # Directional text
            if abs(offset_x) < 30:
                direction = "CENTER"
            elif offset_x < 0:
                direction = "LEFT"
            else:
                direction = "RIGHT"

            # -----------------------------
            # Information UI Panel
            # -----------------------------
            # Semi-transparent background panel
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (350, 310), (40, 40, 40), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            # Text Data
            cv2.putText(frame, "AI FACE ANALYSIS", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Face ID    : {face_id}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Distance   : {distance:.2f} m", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"H Angle    : {horizontal_angle:.1f} deg", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
            cv2.putText(frame, f"V Angle    : {vertical_angle:.1f} deg", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv2.putText(frame, f"Confidence : {confidence:.1f} %", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Face Size  : {width}x{height} px", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Direction  : {direction}", (20, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Status     : {status}", (20, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    else:
        cv2.putText(frame, "FACE NOT DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # -----------------------------
    # FPS & Clock Display
    # -----------------------------
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
    prev_time = current_time

    cv2.putText(frame, f"FPS : {int(fps)}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    now = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, now, (w - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Render output
    cv2.imshow("AI Monocular Face Modeler", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()