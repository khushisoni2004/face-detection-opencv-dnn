# Real-Time Face Detection using OpenCV DNN

A real-time face detection project built using Python and OpenCV DNN. This project uses a pre-trained SSD ResNet Caffe model to detect faces from webcam input and display bounding boxes around detected faces.

## Overview

This project demonstrates real-time computer vision using OpenCV's Deep Neural Network module. It captures video from the webcam, processes each frame, detects faces using a pre-trained model, and shows the result with confidence scores.

This is a beginner-friendly computer vision project useful for learning OpenCV, face detection, webcam processing, and deep learning model integration.

## Features

* Real-time face detection using webcam
* OpenCV DNN-based face detector
* SSD ResNet pre-trained model
* Bounding box around detected faces
* Confidence score display
* Simple Python implementation
* Clean and organized project structure

## Tech Stack

| Category     | Technology               |
| ------------ | ------------------------ |
| Language     | Python                   |
| Library      | OpenCV                   |
| Model        | SSD ResNet Face Detector |
| Model Format | Caffe                    |
| Tools        | Git, GitHub, VS Code     |

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
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/khushisoni2004/face-detection-opencv-dnn.git
cd face-detection-opencv-dnn
```

### 2. Create a virtual environment

For Mac/Linux:

```bash
python3 -m venv venv
```

For Windows:

```bash
python -m venv venv
```

### 3. Activate the virtual environment

For Mac/Linux:

```bash
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the project

```bash
python src/face_detection.py
```

### 6. Stop the project

When the webcam window opens, press:

```text
q
```

to close the webcam window.

## Requirements

The project requires the following Python libraries:

```text
opencv-python
numpy
```

These are already included in the `requirements.txt` file.

## Output

After running the project, the webcam window will open. Detected faces will be highlighted using bounding boxes, and confidence scores will be displayed near the detected face.

## Model Files

This project uses the following model files:

```text
models/deploy.prototxt
models/res10_300x300_ssd_iter_140000.caffemodel
```

These files are required for OpenCV DNN face detection.

## Applications

* Computer vision learning project
* Real-time face detection
* OpenCV DNN practice
* Webcam-based AI project
* Beginner AI/ML portfolio project

## Future Improvements

* Add face recognition
* Add attendance system integration
* Save detected face snapshots
* Add GUI interface
* Add web-based demo
* Add multiple camera support

## Authors

**Khushi Soni**  
GitHub: https://github.com/khushisoni2004

**Dev Malviya**  
GitHub: https://github.com/devmalviya0

**Himanshu Choudhary**  
GitHub: https://github.com/choudharyhim06-star

