"""Live webcam smoke test on the Mac: proves camera->inference->display works.

Uses OpenCV (source=0), not picamera2. Real capture on the Pi is
fodcv-hailo-camera.
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

    model = YOLO(weights)
    # stream=True returns a generator: iterating is what pumps frames. Ctrl-C or
    # 'q' in the window to stop.
    for _ in model.predict(source=args.camera_index, show=True, stream=True, imgsz=640):
        pass


if __name__ == "__main__":
    main()
