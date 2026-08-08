"""Get FOD-A onto disk, then remap it into the PoC's 4-class YOLO subset.

Two halves of one job -- download-and-extract, then remap-and-split -- so they
live in one module and share VOC_ROOT instead of importing it across files.

Download: the FOD-A GitHub repo (FOD-UNOmaha/FOD-data) only ships tools/docs,
not the images -- the actual data lives on Google Drive, linked from its README.

Remap: FOD-A's 31 VOC categories collapse to a 4-class scheme. PRD FR-3's
committed targets (nail, screw, bolt) each get their own class, everything else
fastener-adjacent (washer, nut, combo types) becomes `unknown`. This is a PoC
experiment/comparison -- PRD v4 still specifies training a single
`metal_fastener` class and recovering per-class recall from the seeding log.
The result is a subset for the PoC train/eval smoke test, not the real dataset.
"""

import random
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from fodcv.paths import ROOT

DATA_DIR = ROOT / "data"
ZIP_PATH = DATA_DIR / "fod-a-voc.zip"
VOC_DRIVE_ID = "1RdErcq8PGRXZUOGauaACkQG44T-QyZ4x"  # FOD-A v2.1 Pascal VOC, 300x300
EXTRACT_DIR = DATA_DIR / "fod-a-voc"
VOC_ROOT = EXTRACT_DIR / "FODPascalVOCFormat-V.2.1" / "VOC2007"

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
OUT_DIR = DATA_DIR / "yolo-subset"


def fetch() -> Path:
    """Download + extract the FOD-A Pascal VOC mirror (~432MB). Idempotent."""
    DATA_DIR.mkdir(exist_ok=True)

    if not ZIP_PATH.exists():
        subprocess.run(["gdown", VOC_DRIVE_ID, "-O", str(ZIP_PATH)], check=True)
    else:
        print(f"already downloaded: {ZIP_PATH}")

    if not VOC_ROOT.exists():
        with zipfile.ZipFile(ZIP_PATH) as zf:
            zf.extractall(EXTRACT_DIR)
    else:
        print(f"already extracted: {VOC_ROOT}")

    n_images = len(list((VOC_ROOT / "JPEGImages").glob("*.jpg")))
    print(f"VOC root: {VOC_ROOT} ({n_images} images)")
    return VOC_ROOT


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


def write_data_yaml(data_yaml: Path, train: str = "images/train", val: str = "images/val") -> Path:
    """Write a data.yaml with **no `path:` key**, deliberately.

    Ultralytics resolves the dataset root as
    `data.get("path") or Path(data["yaml_file"]).parent`, so omitting `path:`
    makes every split relative to the yaml's own directory -- correct from any
    working directory and after the tree is copied anywhere.

    Note `path: .` does NOT do this: "." exists, so Ultralytics keeps it and
    resolves the splits against the *current working directory* instead.
    """
    names_block = "\n".join(f"  {cid}: {name}" for cid, name in sorted(CLASS_NAMES.items()))
    data_yaml.write_text(
        f"train: {train}\n"
        f"val: {val}\n"
        "names:\n"
        f"{names_block}\n"
    )
    return data_yaml


def remap() -> Path:
    """VOC -> YOLO subset + data.yaml. Returns the data.yaml path."""
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

    data_yaml = write_data_yaml(OUT_DIR / "data.yaml")

    print(f"train: {len(splits['train'])} images, val: {len(splits['val'])} images")
    print(f"data.yaml: {data_yaml}")
    return data_yaml
