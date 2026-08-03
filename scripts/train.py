"""Fine-tune trial: proves the train->eval loop runs end to end on this
machine (MPS). Short run on the PoC subset -- a plumbing check, not a real
accuracy result (real training happens later per PRD S10/S14a on the full
self-collected dataset).
"""

from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path(__file__).resolve().parent.parent / "data" / "yolo-subset" / "data.yaml"


def main():
    assert DATA_YAML.exists(), "run remap_classes.py first"

    model = YOLO("yolo11n.pt")
    results = model.train(
        data=str(DATA_YAML),
        imgsz=640,
        epochs=15,
        device="mps",
        project=str(Path(__file__).resolve().parent.parent / "runs"),
        name="train_poc",
        exist_ok=True,
    )
    print(f"training done. results dir: {results.save_dir}")

    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}  mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
