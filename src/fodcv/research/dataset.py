"""Turn a registered source into a prepared dataset at `data/<dataset-id>/`.

Two kinds of source, one output shape. Whatever the input, preparing a dataset
leaves a directory Ultralytics can train and score against directly:

    data/<dataset-id>/
      data.yaml          no `path:` key -- see write_data_yaml
      images/{train,val}
      labels/{train,val}

VOC sources (FOD-A) are downloaded and converted: 31 categories collapse to the
4-class PoC scheme, and only images carrying a mapped box are kept. That is an
experiment/comparison -- PRD FR-3 specifies training a single `metal_fastener`
class and recovering per-class recall from the seeding log.

YOLO sources are already labelled by a tool that exports YOLO, so there is
nothing to download and nothing to convert. They are copied in and validated.

Preparing is scoped to one dataset's own directory and refuses to overwrite an
existing one without `force`. The previous version opened with an unconditional
`shutil.rmtree` on the single shared output directory, so preparing a second
dataset destroyed the first.
"""

import random
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from fodcv.paths import CURRENT_DATASET, DATA_DIR, dataset_dir, dataset_yaml
from fodcv.research.datasets import SOURCES, VocSource, YoloSource, source

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


# --------------------------------------------------------------------------
# shared
# --------------------------------------------------------------------------


def write_data_yaml(data_yaml: Path, class_names: dict[int, str],
                    train: str = "images/train", val: str = "images/val") -> Path:
    """Write a data.yaml with **no `path:` key**, deliberately.

    Ultralytics resolves the dataset root as
    `data.get("path") or Path(data["yaml_file"]).parent`, so omitting `path:`
    makes every split relative to the yaml's own directory -- correct from any
    working directory and after the tree is copied anywhere.

    Note `path: .` does NOT do this: "." exists, so Ultralytics keeps it and
    resolves the splits against the *current working directory* instead.
    """
    names_block = "\n".join(f"  {cid}: {name}" for cid, name in sorted(class_names.items()))
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text(
        f"train: {train}\n"
        f"val: {val}\n"
        "names:\n"
        f"{names_block}\n"
    )
    return data_yaml


def split(items: list, val_fraction: float, seed: int, subset_size: int | None = None) -> dict[str, list]:
    """Deterministic train/val split: shuffle, trim, then cut.

    Seeded, or two machines score different images. The order matters and must
    not be rearranged -- trimming after the cut instead of before would move
    images between train and val, silently changing every mAP already recorded.
    `random.Random(seed)` reproduces the previous global `random.seed(seed)`.
    """
    items = list(items)
    random.Random(seed).shuffle(items)
    if subset_size is not None:
        items = items[:subset_size]
    n_val = int(len(items) * val_fraction)
    return {"val": items[:n_val], "train": items[n_val:]}


def _claim_output(dataset: str, force: bool) -> Path:
    """Clear the dataset's own directory, and only ever its own.

    Checked before any download or parsing so a refusal costs nothing. The
    previous version opened the conversion with an unconditional rmtree of the
    single shared output directory, so preparing a second dataset destroyed the
    first without asking.
    """
    out = dataset_dir(dataset)
    if out.exists():
        assert force, f"{out} already exists -- pass --force to rebuild it"
        shutil.rmtree(out)
    return out


def available() -> list[dict]:
    """Every registered dataset and whether it is prepared on disk. For --list."""
    return [
        {
            "dataset": name,
            "kind": type(src).__name__.removesuffix("Source").lower(),
            "classes": len(src.class_names),
            "prepared": dataset_yaml(name).exists(),
        }
        for name, src in sorted(SOURCES.items())
    ]


def prepare(dataset: str = CURRENT_DATASET, force: bool = False) -> Path:
    """Build `data/<dataset>/` from its registered source. Returns the data.yaml."""
    src = source(dataset)
    out = _claim_output(dataset, force)  # up front: refusing must cost nothing
    if isinstance(src, VocSource):
        return _prepare_voc(dataset, src, out)
    return _prepare_yolo(dataset, src, out)


# --------------------------------------------------------------------------
# Pascal VOC sources
# --------------------------------------------------------------------------


def fetch_voc(src: VocSource) -> Path:
    """Download + extract a VOC zip from Google Drive. Idempotent.

    The FOD-A GitHub repo (FOD-UNOmaha/FOD-data) only ships tools/docs, not the
    images -- the actual data lives on Google Drive, linked from its README.
    """
    DATA_DIR.mkdir(exist_ok=True)
    zip_path = DATA_DIR / src.zip_name
    extract_dir = DATA_DIR / Path(src.zip_name).stem

    if not zip_path.exists():
        subprocess.run(["gdown", src.drive_id, "-O", str(zip_path)], check=True)
    else:
        print(f"already downloaded: {zip_path}")

    voc_root = extract_dir / src.extract_subdir
    if not voc_root.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    else:
        print(f"already extracted: {voc_root}")

    n_images = len(list((voc_root / "JPEGImages").glob("*.jpg")))
    print(f"VOC root: {voc_root} ({n_images} images)")
    return voc_root


def fastener_boxes(xml_path: Path, class_map: dict[str, tuple[str, int]]):
    """Mapped boxes from one VOC annotation, in YOLO normalized centre form."""
    root = ET.parse(xml_path).getroot()
    w = float(root.findtext("size/width"))
    h = float(root.findtext("size/height"))
    boxes = []
    for obj in root.findall("object"):
        mapped = class_map.get(obj.findtext("name"))
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


def _prepare_voc(dataset: str, src: VocSource, out: Path) -> Path:
    voc_root = fetch_voc(src)
    ann_dir, img_dir = voc_root / "Annotations", voc_root / "JPEGImages"

    candidates = []
    class_counts = {cid: 0 for cid in src.class_names}
    for xml_path in ann_dir.glob("*.xml"):
        boxes = fastener_boxes(xml_path, src.class_map)
        if boxes:
            candidates.append((xml_path.stem, boxes))
            for class_id, *_ in boxes:
                class_counts[class_id] += 1

    print(f"images with a mapped box: {len(candidates)}")
    print(f"box counts by class: {[(src.class_names[cid], n) for cid, n in sorted(class_counts.items())]}")

    subset = split(candidates, src.val_fraction, src.seed, src.subset_size)
    for name, items in subset.items():
        (out / "images" / name).mkdir(parents=True, exist_ok=True)
        (out / "labels" / name).mkdir(parents=True, exist_ok=True)
        for stem, boxes in items:
            shutil.copy(img_dir / f"{stem}.jpg", out / "images" / name / f"{stem}.jpg")
            lines = [f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cid, cx, cy, bw, bh in boxes]
            (out / "labels" / name / f"{stem}.txt").write_text("\n".join(lines) + "\n")

    data_yaml = write_data_yaml(dataset_yaml(dataset), src.class_names)
    print(f"train: {len(subset['train'])} images, val: {len(subset['val'])} images")
    print(f"data.yaml: {data_yaml}")
    return data_yaml


# --------------------------------------------------------------------------
# already-YOLO sources
# --------------------------------------------------------------------------


def _label_for(image: Path, root: Path) -> Path:
    """The YOLO convention: labels/ mirrors images/, same stem, .txt."""
    relative = image.relative_to(root / "images")
    return root / "labels" / relative.with_suffix(".txt")


def validate_yolo_export(root: Path, class_names: dict[int, str]) -> list[Path]:
    """Every image has a label, and every class id used is declared.

    A labelling tool exporting a class id the registry does not name produces a
    model whose outputs silently mean the wrong thing, so this is an assert and
    not a warning.
    """
    assert (root / "images").is_dir(), f"no images/ under {root}"
    assert (root / "labels").is_dir(), f"no labels/ under {root}"

    images = sorted(p for p in (root / "images").rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    assert images, f"no images under {root / 'images'}"

    missing = [p for p in images if not _label_for(p, root).exists()]
    assert not missing, f"{len(missing)} image(s) with no label file, first: {missing[0]}"

    seen = set()
    for image in images:
        for line in _label_for(image, root).read_text().split("\n"):
            if line.strip():
                seen.add(int(line.split()[0]))
    undeclared = sorted(seen - set(class_names))
    assert not undeclared, f"label files use class ids {undeclared} not in class_names {class_names}"
    return images


def _prepare_yolo(dataset: str, src: YoloSource, out: Path) -> Path:
    root = Path(src.source_dir).expanduser()
    assert root.is_dir(), f"no such directory: {root}"
    images = validate_yolo_export(root, src.class_names)
    print(f"{len(images)} labelled images in {root}")
    if src.val_fraction is None:
        # The export already carries its own split; take it verbatim.
        shutil.copytree(root / "images", out / "images")
        shutil.copytree(root / "labels", out / "labels")
    else:
        for name, items in split(images, src.val_fraction, src.seed).items():
            (out / "images" / name).mkdir(parents=True, exist_ok=True)
            (out / "labels" / name).mkdir(parents=True, exist_ok=True)
            for image in items:
                shutil.copy(image, out / "images" / name / image.name)
                label = _label_for(image, root)
                shutil.copy(label, out / "labels" / name / label.name)

    data_yaml = write_data_yaml(dataset_yaml(dataset), src.class_names)
    print(f"data.yaml: {data_yaml}")
    return data_yaml


def summary(dataset: str = CURRENT_DATASET) -> dict:
    """Counts for provenance. See migrate_artifacts' run.json."""
    out = dataset_dir(dataset)
    return {
        "dataset": dataset,
        # str keys: json turns int keys into strings anyway, so be explicit
        # rather than have run.json disagree with what was written.
        "classes": {str(cid): name for cid, name in source(dataset).class_names.items()},
        "train_images": len(list((out / "images" / "train").glob("*"))) if out.exists() else 0,
        "val_images": len(list((out / "images" / "val").glob("*"))) if out.exists() else 0,
    }
