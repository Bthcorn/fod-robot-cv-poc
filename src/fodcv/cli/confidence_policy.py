"""Confidence hysteresis + temporal smoothing demo, on a webcam.

The Mac-side stand-in for what `fodcv.runtime.vision` does on the Pi: it drives the
same tracker from Ultralytics instead of a .hef, so the thresholds can be tuned
without hardware. The tracker itself lives in `fodcv.runtime.policy`, which must
stay importable on the Pi -- hence this loop lives here, where ultralytics is fine.
"""

import argparse

import cv2
from ultralytics import YOLO

from fodcv.paths import CURRENT_RUN, resolve_weights
from fodcv.runtime import policy


def run(source: str = "0", show: bool = True, model_run: str = CURRENT_RUN):
    source = int(source) if str(source).isdigit() else source

    weights = resolve_weights(model_run)
    print(f"using weights: {weights}")
    print(f"CONFIRM_THRESH={policy.CONFIRM_THRESH}  CAUTION_THRESH={policy.CAUTION_THRESH}")

    model = YOLO(weights)
    names = model.names
    tracks = []

    for r in model.predict(source=source, stream=True, imgsz=640, verbose=False):
        detections = []
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
            detections.append((centroid, float(box.conf[0]), names[int(box.cls[0])]))

        tracks = policy.match_tracks(tracks, detections)

        for t in tracks:
            if t.misses == 0:
                print(f"track {t.id}: {t.cls} ema_conf={t.ema_conf:.2f} -> {t.state()}/{t.action()}")

        if show:
            frame = r.plot()
            for t in tracks:
                if t.misses == 0:
                    x, y = map(int, t.centroid)
                    cv2.putText(
                        frame,
                        f"#{t.id} {t.state()}/{t.action()} {t.ema_conf:.2f}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
            cv2.imshow("confidence_policy", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if show:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        nargs="?",
        default="0",
        help="camera index (e.g. 0), image directory, or video path (default: 0)",
    )
    parser.add_argument("--no-show", action="store_true", help="skip the live window, print only")
    parser.add_argument("--run", default=CURRENT_RUN, help=f"weights from artifacts/<run-id> (default: {CURRENT_RUN})")
    args = parser.parse_args()
    run(source=args.source, show=not args.no_show, model_run=args.run)


if __name__ == "__main__":
    main()
