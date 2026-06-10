
import urllib.request
import ssl
import certifi
import os

files = {
    "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
}

for filename, url in files.items():
    if os.path.exists(filename):
        print(f"Already exists: {filename}")
        continue
    print(f"Downloading {filename} ...")
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=ctx) as r, open(filename, "wb") as f:
        f.write(r.read())
    print(f"Done: {filename} ({os.path.getsize(filename) / 1024 / 1024:.1f} MB)")

print("\nAll files ready. You can now run: python3 face_detection.py")