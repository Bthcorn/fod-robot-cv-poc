"""Live webcam smoke test. No Pi 5 yet, so this uses the Mac's built-in/USB
camera via OpenCV (source=0) instead of picamera2 -- just to prove
camera->inference->display works before real hardware exists. Real capture
on the Pi uses picamera2 with locked exposure/AWB (PRD §9/§10), not this.
"""

import argparse

from ultralytics import YOLO

from fodcv.paths import CURRENT_RUN, resolve_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("camera_index", nargs="?", type=int, default=0)
    parser.add_argument("--run", default=CURRENT_RUN, help=f"weights from artifacts/<run-id> (default: {CURRENT_RUN})")
    args = parser.parse_args()

    weights = resolve_weights(args.run)
    print(f"using weights: {weights}")

    camera_index = args.camera_index
    model = YOLO(weights)
    # stream=True -> generator; must iterate to actually pump frames. Ctrl-C or 'q' in the window to stop.
    for _ in model.predict(source=camera_index, show=True, stream=True, imgsz=640):
        pass


if __name__ == "__main__":
    main()
