"""
Advanced Real-Time Face Detection using OpenCV DNN (Deep Neural Network)
Model: SSD (Single Shot Multibox Detector) + ResNet-10 backbone (Caffe)

SETUP — download these two model files before running:
  prototxt : https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
  weights  : https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

Place both files in the same folder as this script.

Install dependency (if needed):
  pip install opencv-python
"""

import cv2
import time

# ── Config ────────────────────────────────────────────────────────────────────
PROTO      = "deploy.prototxt"           # Network architecture definition
WEIGHTS    = "res10_300x300_ssd_iter_140000.caffemodel"  # Pre-trained weights
CONFIDENCE = 0.55                        # Minimum detection confidence (0-1)
INPUT_SIZE = (300, 300)                  # SSD was trained at 300x300
MEAN       = (104.0, 177.0, 123.0)       # BGR mean subtraction values (from training)
BOX_COLOR  = (0, 200, 100)              # Green bounding box
TEXT_COLOR = (255, 255, 255)            # White label text
BG_COLOR   = (0, 150, 75)              # Label background
# ──────────────────────────────────────────────────────────────────────────────


def load_model(proto_path, weights_path):
    """Load the pre-trained Caffe DNN face detection model."""
    net = cv2.dnn.readNetFromCaffe(proto_path, weights_path)
    # Optionally use GPU backend (uncomment if you have CUDA):
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return net


def detect_faces(net, frame, conf_threshold):
    """
    Run the DNN model on a single frame.
    
    How it works:
    1. Convert frame to a 'blob' — normalizes size, pixel values, and channel order.
    2. Feed the blob as input to the neural network.
    3. Run a forward pass — this is where inference happens.
    4. Parse the output detections; each row = [_, _, confidence, x1, y1, x2, y2]
       where coordinates are normalized (0.0 to 1.0).
    """
    h, w = frame.shape[:2]

    # blobFromImage: resize to 300x300, subtract mean BGR values, no channel swap
    blob = cv2.dnn.blobFromImage(frame, 1.0, INPUT_SIZE, MEAN, swapRB=False)

    net.setInput(blob)
    detections = net.forward()          # Shape: (1, 1, N, 7)

    faces = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < conf_threshold:
            continue

        # Scale normalized coords back to pixel space
        x1 = int(detections[0, 0, i, 3] * w)
        y1 = int(detections[0, 0, i, 4] * h)
        x2 = int(detections[0, 0, i, 5] * w)
        y2 = int(detections[0, 0, i, 6] * h)

        # Clamp to frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        faces.append((x1, y1, x2, y2, confidence))

    return faces


def draw_face(frame, x1, y1, x2, y2, confidence):
    """Draw a bounding box and confidence label for one detected face."""
    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

    label = f"Face {confidence:.0%}"
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

    # Label background pill (sits just above the box)
    label_y = y1 - 6 if y1 - 6 > th else y1 + th + 6
    cv2.rectangle(frame,
                  (x1, label_y - th - 4),
                  (x1 + tw + 6, label_y + baseline),
                  BG_COLOR, cv2.FILLED)
    cv2.putText(frame, label,
                (x1 + 3, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)


def draw_hud(frame, fps, face_count, conf_threshold):
    """Draw an on-screen heads-up display (FPS, face count, threshold)."""
    lines = [
        f"FPS: {fps:.1f}",
        f"Faces: {face_count}",
        f"Threshold: {conf_threshold:.0%}",
        "Q = quit  +/- = threshold",
    ]
    for i, txt in enumerate(lines):
        y = 24 + i * 22
        # Thin shadow for readability over any background
        cv2.putText(frame, txt, (11, y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, txt, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)


def main():
    print("Loading DNN face detection model...")
    net = load_model(PROTO, WEIGHTS)
    print("Model loaded.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    # Try to set HD resolution (falls back to whatever camera supports)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    conf_threshold = CONFIDENCE
    prev_time = time.time()
    fps = 0.0

    print("Running — press Q to quit, +/- to adjust confidence threshold.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed.")
            break

        # ── FPS calculation ────────────────────────────────────────────────
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))  # EMA smoothing
        prev_time = now

        # ── Detection ─────────────────────────────────────────────────────
        faces = detect_faces(net, frame, conf_threshold)

        for (x1, y1, x2, y2, conf) in faces:
            draw_face(frame, x1, y1, x2, y2, conf)

        # ── HUD ───────────────────────────────────────────────────────────
        draw_hud(frame, fps, len(faces), conf_threshold)

        cv2.imshow("Advanced Face Detection  [DNN/SSD]", frame)

        # ── Key handling ──────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            conf_threshold = min(0.95, conf_threshold + 0.05)
            print(f"Threshold → {conf_threshold:.0%}")
        elif key == ord('-'):
            conf_threshold = max(0.05, conf_threshold - 0.05)
            print(f"Threshold → {conf_threshold:.0%}")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()