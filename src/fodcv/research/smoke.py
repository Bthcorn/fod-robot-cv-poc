"""Smoke test: does the install actually work end to end, before training?

Loads stock (COCO-pretrained) yolo11n and runs it on a few images from the
prepared dataset. It won't recognize fasteners (wrong classes) -- this only
proves the library, weights download, and inference call all work here.

Reads the prepared dataset rather than a raw VOC extract, so it works for any
registered dataset, including YOLO sources that were never VOC to begin with.
"""

from ultralytics import YOLO

from fodcv.paths import CURRENT_DATASET, ROOT, dataset_val_images

OUT_DIR = ROOT / "runs" / "smoke_test"
N_SAMPLES = 5


def run(dataset: str = CURRENT_DATASET):
    val_images = dataset_val_images(dataset)
    images = sorted(val_images.glob("*.jpg"))[:N_SAMPLES]
    assert images, (
        f"no images in {val_images} -- run prepare_dataset.py --dataset {dataset}"
    )

    model = YOLO("yolo11n.pt")
    results = model.predict(source=images, save=True, project=str(OUT_DIR.parent), name=OUT_DIR.name, exist_ok=True)

    for img, r in zip(images, results):
        print(f"{img.name}: {len(r.boxes)} detections")

    print(f"annotated images saved under: {OUT_DIR}")
