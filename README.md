# Real-Time Face Detection using OpenCV DNN

A real-time face detection project built with **Python** and **OpenCV DNN**. It uses a pre-trained **ResNet-10 SSD Caffe model** to detect faces from a live webcam feed and displays bounding boxes, confidence score, FPS, and threshold controls.

## Features

- Real-time face detection using webcam
- OpenCV DNN based SSD face detector
- Confidence score display for each detected face
- Live FPS and face count HUD
- Adjustable confidence threshold using keyboard
- Clean and beginner-friendly Python code

## Tech Stack

- Python
- OpenCV
- OpenCV DNN Module
- Caffe Model

## Project Structure

```text
face-detection-opencv-dnn/
├── face_detection.py
├── download_models.py
├── deploy.prototxt
├── res10_300x300_ssd_iter_140000.caffemodel
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/khushisoni2004/face-detection-opencv-dnn.git
cd face-detection-opencv-dnn
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python3 face_detection.py
```

## Controls

| Key | Action |
|---|---|
| `Q` | Quit camera window |
| `+` | Increase confidence threshold |
| `-` | Decrease confidence threshold |

## Model Information

This project uses the OpenCV pre-trained SSD face detector:

- `deploy.prototxt` — model architecture
- `res10_300x300_ssd_iter_140000.caffemodel` — pre-trained weights

If model files are missing, run:

```bash
python3 download_models.py
```

## Author

**Khushi Soni**  
GitHub: [khushisoni2004](https://github.com/khushisoni2004)

## Collaboration

Contributions are welcome. You can fork this repository, make improvements, and create a pull request.

Suggested contribution ideas:

- Add image/video file input support
- Add face count analytics
- Improve UI overlay
- Add face blur/privacy mode
- Add Streamlit or web dashboard version

## License

This project is open source and available for learning and portfolio use.
