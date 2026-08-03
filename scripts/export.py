"""Export pipeline test: does Ultralytics' export toolchain work on this
machine at all? Tests formats runnable off-Pi (onnx, coreml). Real NCNN/
OpenVINO INT8 accuracy+latency benchmarking (PRD S10) needs the actual
Pi 5 hardware and is out of scope here.
"""

from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "runs" / "train_poc" / "weights" / "best.pt"
VAL_IMAGES = ROOT / "data" / "yolo-subset" / "images" / "val"
FORMATS = ["onnx", "coreml", "tflite"]


def main():
    assert WEIGHTS.exists(), f"no trained weights at {WEIGHTS} -- run train.py first"

    results = {}
    for fmt in FORMATS:
        try:
            model = YOLO(str(WEIGHTS))
            exported_path = model.export(format=fmt)
            reloaded = YOLO(exported_path)
            reloaded.predict(source=str(VAL_IMAGES), max_det=1, verbose=False)
            results[fmt] = f"ok -> {exported_path}"
        except Exception as e:
            results[fmt] = f"FAILED: {e}"

    print("\nexport results:")
    for fmt, status in results.items():
        print(f"  {fmt}: {status}")


if __name__ == "__main__":
    main()
