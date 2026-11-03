"""Fetch the MediaPipe FaceLandmarker model into lipcolor/models/.

    python download_model.py
"""
import os
import ssl
import urllib.request

URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
       "face_landmarker/float16/latest/face_landmarker.task")
DEST = os.path.join(os.path.dirname(__file__), "lipcolor", "models", "face_landmarker.task")


def main():
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    if os.path.exists(DEST) and os.path.getsize(DEST) > 1_000_000:
        print(f"already present: {DEST} ({os.path.getsize(DEST)} bytes)")
        return
    ctx = ssl.create_default_context()
    req = urllib.request.Request(URL, headers={"User-Agent": "lipcolor/1.0"})
    with urllib.request.urlopen(req, timeout=120, context=ctx) as r:
        data = r.read()
    with open(DEST, "wb") as f:
        f.write(data)
    print(f"downloaded {len(data)} bytes -> {DEST}")


if __name__ == "__main__":
    main()
