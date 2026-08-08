"""Live webcam smoke test. No Pi 5 yet, so this uses the Mac's built-in/USB
camera via OpenCV (source=0) instead of picamera2 -- just to prove
camera->inference->display works before real hardware exists. Real capture
on the Pi uses picamera2 with locked exposure/AWB (PRD S6a/S10), not this.
"""

import sys

from ultralytics import YOLO

from fodcv.paths import resolve_weights


def main():
    weights = resolve_weights()
    print(f"using weights: {weights}")

    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    model = YOLO(weights)
    # stream=True -> generator; must iterate to actually pump frames. Ctrl-C or 'q' in the window to stop.
    for _ in model.predict(source=camera_index, show=True, stream=True, imgsz=640):
        pass


if __name__ == "__main__":
    main()
