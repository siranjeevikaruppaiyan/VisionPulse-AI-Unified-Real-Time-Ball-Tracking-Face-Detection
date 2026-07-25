# VisionPulse AI – Unified Real-Time Ball Tracking & Face Detection System

VisionPulse AI is an AI-powered computer vision project that performs **real-time ball tracking** and **face detection** using a live camera feed. The project combines **YOLOv8** for accurate object detection with **MediaPipe** for efficient face detection, delivering a fast, reliable, and scalable vision system.

---

## Project Overview

VisionPulse AI continuously processes video frames from a webcam to detect and localize balls and human faces in real time. The system displays bounding boxes, confidence scores, and live FPS (Frames Per Second) to ensure high performance and responsiveness.

This project demonstrates the practical application of Artificial Intelligence and Computer Vision for real-world scenarios such as sports analytics, surveillance, robotics, and intelligent monitoring.

---

## Features

- Real-time ball detection using YOLOv8
- Real-time face detection using MediaPipe
- Live webcam video processing
- Bounding boxes with confidence scores
- FPS (Frames Per Second) monitoring
- Lightweight and scalable architecture
- Easy to extend with new AI features

---

## Technologies Used

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- MediaPipe
- NumPy

---

## Project Structure

```
Track2_AI/
│
├── image/
├── models/
├── output/
├── src/
│   ├── ball_detection.py
│   ├── face_distance.py
│
├── main.py
├── yolov8n.pt
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/siranjeevikaruppaiyan/VisionPulse-AI-Unified-Real-Time-Ball-Tracking-Face-Detection.git
```

### Move to the Project Folder

```bash
cd VisionPulse-AI-Unified-Real-Time-Ball-Tracking-Face-Detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Required Libraries

```
ultralytics
opencv-python
mediapipe
numpy
```

---

## Running the Project

Run the main application:

```bash
python main.py
```

Or execute individual modules:

Ball Detection

```bash
python src/ball_detection.py
```

Face Detection

```bash
python src/face_distance.py
```

---

## Applications

- Sports Analytics
- Smart Surveillance
- Human–Computer Interaction
- Robotics
- AI Vision Systems
- Intelligent Monitoring

---

## Future Enhancements

- Multi-object tracking
- Ball trajectory prediction
- Face recognition
- Person identification
- Distance estimation
- Performance analytics dashboard
- AI event detection
- Mobile and edge-device deployment

---

## Results

- Real-time object detection
- Accurate face localization
- High-speed processing
- Live FPS monitoring
- Robust AI vision pipeline

---

## Author

**Siranjeevi Karuppaiyan**

B.E. Electronics and Communication Engineering

Sri Sairam Engineering College, Chennai

GitHub: https://github.com/siranjeevikaruppaiyan

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- Ultralytics YOLOv8
- OpenCV
- MediaPipe
- Python Community
