# Real-Time Face Detection using OpenCV DNN

A real-time face detection project built using Python and OpenCV DNN. The system uses a pre-trained SSD ResNet model to detect faces from webcam input and display bounding boxes around detected faces.

## Overview

This project demonstrates real-time computer vision using OpenCV's Deep Neural Network module. It captures video from the webcam, processes each frame, detects faces using a pre-trained Caffe model, and displays the result with confidence scores.

## Features

- Real-time face detection using webcam
- OpenCV DNN-based face detector
- SSD ResNet pre-trained model
- Bounding box around detected faces
- Confidence score display
- Simple Python implementation
- Clean project structure

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python |
| Library | OpenCV |
| Model | SSD ResNet Face Detector |
| Model Format | Caffe |
| Tools | Git, GitHub, VS Code |

## Project Structure

```text
face-detection-opencv-dnn/
│
├── src/
│   ├── face_detection.py
│   └── download_models.py
│
├── models/
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
│
├── README.md
├── requirements.txt
└── .gitignore

## Step 6: Requirements update karo

```bash
cat > requirements.txt <<'EOF'
opencv-python
numpy
