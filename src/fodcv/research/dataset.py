"""Turn a registered source into a prepared dataset at `data/<dataset-id>/`.

Two kinds of source, one output shape Ultralytics can train and score directly:

    data/<dataset-id>/
      data.yaml          no `path:` key -- see write_data_yaml
      images/{train,val}
      labels/{train,val}

VOC sources (FOD-A) are downloaded and converted -- 31 categories collapse to
the 4-class PoC scheme, keeping only images with a mapped box. That is the
comparison experiment; PRD FR-3 specifies a single `fod` class.
YOLO sources are already labelled, so they are copied in and validated.

Preparing is scoped to one dataset's own directory and refuses to overwrite an
existing one without `force` -- see _claim_output.
"""

import random
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

from fodcv.paths import CURRENT_DATASET, DATA_DIR, dataset_dir, dataset_yaml
from fodcv.research.datasets import HOLDOUT_STEMS, SOURCES, VocSource, YoloSource, source

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def write_data_yaml(data_yaml: Path, class_names: dict[int, str],
                    train: str = "images/train", val: str = "images/val") -> Path:
    """Write a data.yaml with **no `path:` key**, deliberately.

    Ultralytics resolves the root as `data.get("path") or
    Path(data["yaml_file"]).parent`, so omitting `path:` makes every split
    relative to the yaml itself -- correct from any cwd and after an rsync.

    `path: .` does NOT do this: "." is truthy, so splits resolve against the
    current working directory instead.
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

    That order is load-bearing. Trimming after the cut instead of before moves
    images between train and val, silently changing every mAP already recorded.
    """
    items = list(items)
    random.Random(seed).shuffle(items)
    if subset_size is not None:
        items = items[:subset_size]
    n_val = int(len(items) * val_fraction)
    return {"val": items[:n_val], "train": items[n_val:]}


def ahash(path: Path) -> bytes:
    """16x16 average hash. Two frames of the same scene collide, different
    scenes do not.

    ponytail: a whole-frame average hash, not scene metadata -- a Roboflow
    export carries no scene id, and this is what actually measured the leak.
    Ceiling: it groups near-identical *frames*, not a camera lock that panned.
    Upgrade to grouping on collection metadata when the arena dataset lands
    (docs/dataset-roadmap.md).
    """
    from PIL import Image  # ultralytics already brings pillow; research/ only

    a = np.asarray(Image.open(path).convert("L").resize((16, 16)), dtype=np.float32)
    return (a > a.mean()).tobytes()


def split_grouped(images: list[Path], val_fraction: float, seed: int) -> dict[str, list[Path]]:
    """Split so a near-duplicate cluster never spans train and val.

    Splitting per image puts the same frame on both sides and inflates val mAP.
    RESULT.md section 7 measures that at 74% on fod-a-3k, and both Roboflow
    exports registered here arrive with 14-26% of their own val already
    present in their own train.

    Groups are ordered by hash, NOT by path: the shuffle below must see the
    same sequence on every machine, and absolute paths differ between them.
    That is RESULT.md:310's bug -- filesystem order feeding the shuffle, so one
    seed drew two different splits -- and it would come straight back here.
    """
    groups: dict[bytes, list[Path]] = {}
    for image in images:
        groups.setdefault(ahash(image), []).append(image)
    ordered = [sorted(g) for _, g in sorted(groups.items())]
    cut = split(ordered, val_fraction, seed)
    return {name: [p for group in items for p in group] for name, items in cut.items()}


def _claim_output(dataset: str, force: bool) -> Path:
    """Clear the dataset's own directory, and only ever its own.

    Checked before any download or parsing, so a refusal costs nothing.

    Raises rather than asserts, unlike the rest of this module: `python -O`
    strips asserts, and this is the only guard in front of an rmtree.
    """
    out = dataset_dir(dataset)
    if out.exists():
        if not force:
            raise FileExistsError(f"{out} already exists -- pass --force to rebuild it")
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
    out = _claim_output(dataset, force)
    if isinstance(src, VocSource):
        return _prepare_voc(dataset, src, out)
    return _prepare_yolo(dataset, src, out)


# --- Pascal VOC sources ---


def fetch_voc(src: VocSource) -> Path:
    """Download + extract a VOC zip from Google Drive. Idempotent.

    Drive, not GitHub: FOD-UNOmaha/FOD-data ships only tools and docs.
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


def background_stems(ann_dir: Path, class_map: dict[str, tuple[str, int]],
                     limit: int, seed: int) -> list[str]:
    """Stems with no mapped box, sampled round-robin across VOC categories.

    Round-robin rather than a flat shuffle: FOD-A's negatives are 12% Pliers and
    0.5% Tape, so a flat draw teaches one negative shape well and 23 badly.
    Small categories exhaust and the remainder spreads over the large ones.
    """
    buckets: dict[str, list[str]] = {}
    # sorted, for the reason _prepare_voc's own glob is sorted.
    for xml_path in sorted(ann_dir.glob("*.xml")):
        if xml_path.stem in HOLDOUT_STEMS:
            continue  # scored against, so never trained on -- unconditional
        if fastener_boxes(xml_path, class_map):
            continue  # a positive, handled by the caller
        names = [obj.findtext("name") for obj in ET.parse(xml_path).getroot().findall("object")]
        if names:
            buckets.setdefault(sorted(names)[0], []).append(xml_path.stem)

    rng = random.Random(seed)
    for stems in buckets.values():
        rng.shuffle(stems)

    picked: list[str] = []
    order = sorted(buckets)
    while len(picked) < limit and any(buckets[name] for name in order):
        for name in order:
            if not buckets[name]:
                continue
            picked.append(buckets[name].pop())
            if len(picked) == limit:
                break
    return picked


def _prepare_voc(dataset: str, src: VocSource, out: Path) -> Path:
    voc_root = fetch_voc(src)
    ann_dir, img_dir = voc_root / "Annotations", voc_root / "JPEGImages"

    candidates = []
    class_counts = {cid: 0 for cid in src.class_names}
    # sorted, not bare glob: the split shuffles this list, so filesystem order
    # would hand the Mac and the CUDA box different train/val splits from the
    # same seed -- which is exactly what `seed` is documented to prevent.
    for xml_path in sorted(ann_dir.glob("*.xml")):
        if xml_path.stem in HOLDOUT_STEMS:
            continue  # scored against, so never trained on -- unconditional
        boxes = fastener_boxes(xml_path, src.class_map)
        if boxes:
            candidates.append((xml_path.stem, boxes))
            for class_id, *_ in boxes:
                class_counts[class_id] += 1

    backgrounds = (background_stems(ann_dir, src.class_map, src.background_max, src.seed)
                   if src.background_max else [])

    print(f"holdout images withheld: {len(HOLDOUT_STEMS)}")
    print(f"images with a mapped box: {len(candidates)}")
    print(f"box counts by class: {[(src.class_names[cid], n) for cid, n in sorted(class_counts.items())]}")
    print(f"background images (no mapped box): {len(backgrounds)}")

    subset = split(candidates, src.val_fraction, src.seed, src.subset_size)
    # Split separately, and never append to candidates: split() trims to
    # subset_size after shuffling, so one merged list would both evict positives
    # to make room and move every already-recorded dataset's train/val cut.
    bg_subset = split(backgrounds, src.val_fraction, src.seed)
    for name, items in subset.items():
        (out / "images" / name).mkdir(parents=True, exist_ok=True)
        (out / "labels" / name).mkdir(parents=True, exist_ok=True)
        for stem, boxes in items:
            shutil.copy(img_dir / f"{stem}.jpg", out / "images" / name / f"{stem}.jpg")
            lines = [f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cid, cx, cy, bw, bh in boxes]
            (out / "labels" / name / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        for stem in bg_subset[name]:
            shutil.copy(img_dir / f"{stem}.jpg", out / "images" / name / f"{stem}.jpg")
            # Empty, not absent: Ultralytics reads either as a background, but
            # every image here having a label file is what validate_yolo_export
            # asserts, and a missing file reads as a bug rather than a negative.
            (out / "labels" / name / f"{stem}.txt").write_text("")

    data_yaml = write_data_yaml(dataset_yaml(dataset), src.class_names)
    print(f"backgrounds: {len(bg_subset['train'])} train, {len(bg_subset['val'])} val")
    print(f"train: {len(subset['train'])} images, val: {len(subset['val'])} images")
    print(f"data.yaml: {data_yaml}")
    return data_yaml


# --- already-YOLO sources ---


def _label_for(image: Path) -> Path:
    """The YOLO convention: labels/ mirrors images/, same stem, .txt.

    Locates `images` from the right, so this reads both layouts in use -- this
    repo's `images/<split>/` and a Roboflow export's `<split>/images/`. Taking
    it from the left instead would resolve a Roboflow path to
    `labels/<split>/images/...`, which does not exist, and every image would
    report a missing label.
    """
    i = len(image.parts) - 1 - image.parts[::-1].index("images")
    return Path(*image.parts[:i], "labels", *image.parts[i + 1:]).with_suffix(".txt")


# Roboflow's split names -> ours. `test/` is deliberately absent: the pipeline
# models two splits, and the shipped recipe the published mAP is measured
# against trains on train/ alone.
SHIPPED_SPLITS = {"train": "train", "valid": "val", "val": "val"}


def _shipped_split(image: Path) -> str | None:
    """`<split>/images/x.jpg` -> our split name, or None for a split we skip."""
    i = len(image.parts) - 1 - image.parts[::-1].index("images")
    return SHIPPED_SPLITS.get(image.parts[i - 1])


def _boxed_label(label: Path) -> str:
    """A label file as detection lines, any polygon reduced to its own bounds.

    Roboflow's ARG_Bolts export mixes `cls cx cy w h` with segmentation
    polygons, and in 671 of its 8,334 labelled train files it mixes them in the
    SAME file. Ultralytics decides box-vs-polygon per *file* -- `any(len(x) > 6
    for x in lb)` in ultralytics/data/utils.py -- so one polygon makes it
    reshape every plain box line to (-1, 2) as well, reading `cx cy w h` as the
    two points (cx, cy) and (w, h) and taking their bounds. That silently
    rewrote 26% of this export's instances, median IoU 0.004 against the box
    the file states, on both sides of the split; the loader reports nothing.

    Normalising here means the file Ultralytics opens is uniformly 5-field, so
    that branch never fires. A polygon still becomes exactly the box
    segments2boxes would have given it -- this changes nothing for a file that
    is all polygons, and everything for one that is not.
    """
    lines = []
    for raw in label.read_text().split("\n"):
        parts = raw.split()
        if not parts:
            continue
        if len(parts) == 5:
            lines.append(" ".join(parts))  # verbatim: no float round-trip
            continue
        assert len(parts) > 6 and len(parts) % 2 == 1, f"{label}: unreadable label line {raw!r}"
        xs = [float(v) for v in parts[1::2]]
        ys = [float(v) for v in parts[2::2]]
        lines.append(f"{parts[0]} {(min(xs) + max(xs)) / 2:.6f} {(min(ys) + max(ys)) / 2:.6f} "
                     f"{max(xs) - min(xs):.6f} {max(ys) - min(ys):.6f}")
    # Empty stays empty, not a blank line: a background image has a label file
    # with nothing in it -- see _prepare_voc.
    return "".join(f"{line}\n" for line in lines)


def validate_yolo_export(root: Path, class_names: dict[int, str]) -> list[Path]:
    """Every image has a label, and every class id used is declared.

    An assert, not a warning: a class id the registry does not name produces a
    model whose outputs silently mean the wrong thing.
    """
    images = sorted(p for d in root.rglob("images") if d.is_dir()
                    for p in d.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    assert images, f"no images under any images/ directory in {root}"

    missing = [p for p in images if not _label_for(p).exists()]
    assert not missing, f"{len(missing)} image(s) with no label file, first: {missing[0]}"

    seen = set()
    for image in images:
        for line in _label_for(image).read_text().split("\n"):
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
        # The export already carries its own split; take it verbatim. Anything
        # SHIPPED_SPLITS does not name -- test/ -- is dropped, so the count
        # printed above is the one to read the two below against.
        subset: dict[str, list[Path]] = {"train": [], "val": []}
        for image in images:
            if name := _shipped_split(image):
                subset[name].append(image)
        # One-sided means a layout mismatch -- this repo's own `images/<split>/`,
        # or split names this map does not know -- not a dataset. Crash rather
        # than train on half of it.
        assert subset["train"] and subset["val"], (
            f"{root}: expected <split>/images/ with train/ and valid/, got "
            f"train={len(subset['train'])} val={len(subset['val'])}")
    else:
        # Grouped, not per-image: an export that ships its own split has already
        # been split by someone, and both Roboflow ones split per image.
        subset = split_grouped(images, src.val_fraction, src.seed)

    for name, items in subset.items():
        (out / "images" / name).mkdir(parents=True, exist_ok=True)
        (out / "labels" / name).mkdir(parents=True, exist_ok=True)
        for image in items:
            shutil.copy(image, out / "images" / name / image.name)
            label = _label_for(image)
            (out / "labels" / name / label.name).write_text(_boxed_label(label))

    # split_grouped cuts on group count, so its image-level fraction lands near
    # val_fraction rather than on it. Print what it actually was; a shipped split
    # has no target to miss.
    n_val, n_train = len(subset["val"]), len(subset["train"])
    target = "" if src.val_fraction is None else f", target {src.val_fraction:.0%}"
    print(f"train: {n_train} images, val: {n_val} images "
          f"({n_val / (n_val + n_train):.1%}{target})")

    data_yaml = write_data_yaml(dataset_yaml(dataset), src.class_names)
    print(f"data.yaml: {data_yaml}")
    return data_yaml


def summary(dataset: str = CURRENT_DATASET) -> dict:
    """Counts for provenance. See migrate_artifacts' run.json."""
    out = dataset_dir(dataset)
    return {
        "dataset": dataset,
        # str keys: json stringifies int keys anyway, so be explicit rather than
        # have run.json disagree with what was written.
        "classes": {str(cid): name for cid, name in source(dataset).class_names.items()},
        "train_images": len(list((out / "images" / "train").glob("*"))) if out.exists() else 0,
        "val_images": len(list((out / "images" / "val").glob("*"))) if out.exists() else 0,
    }
