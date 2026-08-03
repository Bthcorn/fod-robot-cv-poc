"""Smoke test: does the install actually work end to end, before training?

Loads stock (COCO-pretrained) yolo11n and runs it on a few real FOD-A images.
It won't recognize fasteners (wrong classes) -- this only proves the library,
weights download, and inference call all work on this machine.
"""

from pathlib import Path

from ultralytics import YOLO

from fetch_dataset import VOC_ROOT

OUT_DIR = Path(__file__).resolve().parent.parent / "runs" / "smoke_test"
N_SAMPLES = 5


def main():
    images = sorted((VOC_ROOT / "JPEGImages").glob("*.jpg"))[:N_SAMPLES]
    assert images, "no sample images found -- run fetch_dataset.py first"

    model = YOLO("yolo11n.pt")
    results = model.predict(source=images, save=True, project=str(OUT_DIR.parent), name=OUT_DIR.name, exist_ok=True)

    for img, r in zip(images, results):
        print(f"{img.name}: {len(r.boxes)} detections")

    print(f"annotated images saved under: {OUT_DIR}")


if __name__ == "__main__":
    main()
