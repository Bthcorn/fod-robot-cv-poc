"""Remap FOD-A's 31 VOC categories to a 4-class scheme: PRD FR-3's committed
targets (nail, screw, bolt) each get their own class, everything else
fastener-adjacent (washer, nut, combo types) collapses into `unknown`. This
is a PoC experiment/comparison -- PRD v4 still specifies training a single
`metal_fastener` class and recovering per-class recall from the seeding log.
Writes a YOLO-format subset dataset for the PoC train/eval smoke test -- not
the real dataset.
"""

import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from fetch_dataset import VOC_ROOT

CLASS_MAP = {
    "Nail": ("nail", 0),
    "Screw": ("screw", 1),
    "Bolt": ("bolt", 2),
    "Washer": ("unknown", 3),
    "Nut": ("unknown", 3),
    "BoltWasher": ("unknown", 3),
    "BoltNutSet": ("unknown", 3),
}
CLASS_NAMES = {cid: name for name, cid in CLASS_MAP.values()}
SUBSET_SIZE = 600  # PoC smoke test, not the real ~2000-2500 own-image dataset (PRD S10)
VAL_FRACTION = 0.15
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "yolo-subset"


def fastener_boxes(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    w = float(root.findtext("size/width"))
    h = float(root.findtext("size/height"))
    boxes = []
    for obj in root.findall("object"):
        mapped = CLASS_MAP.get(obj.findtext("name"))
        if mapped is None:
            continue
        _, class_id = mapped
        b = obj.find("bndbox")
        xmin, ymin = float(b.findtext("xmin")), float(b.findtext("ymin"))
        xmax, ymax = float(b.findtext("xmax")), float(b.findtext("ymax"))
        # YOLO format: class cx cy bw bh, normalized 0-1
        cx, cy = (xmin + xmax) / 2 / w, (ymin + ymax) / 2 / h
        bw, bh = (xmax - xmin) / w, (ymax - ymin) / h
        boxes.append((class_id, cx, cy, bw, bh))
    return boxes


def main() -> Path:
    ann_dir = VOC_ROOT / "Annotations"
    img_dir = VOC_ROOT / "JPEGImages"

    candidates = []
    class_counts = {cid: 0 for cid in CLASS_NAMES}
    for xml_path in ann_dir.glob("*.xml"):
        boxes = fastener_boxes(xml_path)
        if boxes:
            candidates.append((xml_path.stem, boxes))
            for class_id, *_ in boxes:
                class_counts[class_id] += 1

    print(f"images with a fastener box: {len(candidates)}")
    print(f"box counts by class: {[(CLASS_NAMES[cid], n) for cid, n in sorted(class_counts.items())]}")
    random.seed(0)
    random.shuffle(candidates)
    subset = candidates[:SUBSET_SIZE]

    n_val = int(len(subset) * VAL_FRACTION)
    splits = {"val": subset[:n_val], "train": subset[n_val:]}

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    for split, items in splits.items():
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        for stem, boxes in items:
            shutil.copy(img_dir / f"{stem}.jpg", OUT_DIR / "images" / split / f"{stem}.jpg")
            label_lines = [
                f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cid, cx, cy, bw, bh in boxes
            ]
            (OUT_DIR / "labels" / split / f"{stem}.txt").write_text("\n".join(label_lines) + "\n")

    names_block = "\n".join(f"  {cid}: {name}" for cid, name in sorted(CLASS_NAMES.items()))
    data_yaml = OUT_DIR / "data.yaml"
    data_yaml.write_text(
        f"path: {OUT_DIR}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"{names_block}\n"
    )

    print(f"train: {len(splits['train'])} images, val: {len(splits['val'])} images")
    print(f"data.yaml: {data_yaml}")
    return data_yaml


if __name__ == "__main__":
    main()
