"""One-shot: lift an Ultralytics training run into artifacts/<run-id>/.

Run this once per historical training run. New runs get published the same way,
but you only ever need it after `train.py` -- export, benchmark and deploy all
read artifacts/<run-id>/ from then on.

What moves, and why only this:

  best.pt          the checkpoint. last.pt stays behind; nothing reads it.
  bench_*          the exports already built. Copied rather than rebuilt so the
                   benchmark numbers already recorded stay valid.
  exports.json     rewritten with paths relative to itself, which is what makes
                   the directory survive an rsync to a different absolute path.
  eval/            the val split + a data.yaml with no `path:` key, so the Pi
                   can score mAP without a copy of data/.

Deliberately left behind: the stray un-claimed best.onnx (outside the bench_
namespace, so nothing in the manifest points at it) and runs/detect/val*
(13 orphan Ultralytics dirs no code reads). runs/ itself is untouched.

    uv run scripts/migrate_artifacts.py --from runs/train_poc --run poc-v1
"""

import argparse
import shutil
from pathlib import Path

from fodcv import manifest as mf
from fodcv.paths import CURRENT_RUN, DATASET_DIR, ROOT, run_dir
from fodcv.research.dataset import write_data_yaml


def copy_eval_split(dest):
    """Copy the val split in beside the run, with a location-independent yaml."""
    for kind in ("images", "labels"):
        src = DATASET_DIR / kind / "val"
        assert src.exists(), f"no {src} -- run remap_classes.py first"
        shutil.copytree(src, dest / kind / "val", dirs_exist_ok=True)
    write_data_yaml(dest / "data.yaml", train="images/val", val="images/val")
    return len(list((dest / "images" / "val").glob("*.jpg")))


def migrate(source_dir, run: str):
    weights_dir = source_dir / "weights"
    best = weights_dir / "best.pt"
    assert best.exists(), f"no {best}"

    dest = run_dir(run)
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dest / "best.pt")
    print(f"best.pt -> {dest / 'best.pt'}")

    for path in sorted(weights_dir.glob("bench_*")):
        target = dest / path.name
        if path.is_dir():
            shutil.copytree(path, target, dirs_exist_ok=True)
        else:
            shutil.copy2(path, target)
        print(f"{path.name} -> {target}")

    old_manifest = mf.load(weights_dir / mf.NAME)
    new_manifest_path = dest / mf.NAME
    rewritten = {}
    for key, entry in old_manifest.items():
        if entry.startswith((mf.FAILED, mf.UNSUPPORTED)):
            rewritten[key] = entry  # a sentinel is still the answer for that cell
            continue
        # Old entries are absolute paths into weights_dir; the artifact now sits
        # in dest under the same basename.
        rewritten[key] = mf.entry_for(dest / Path(entry).name, new_manifest_path)
    mf.save(new_manifest_path, rewritten)
    print(f"{mf.NAME} -> {new_manifest_path} ({len(rewritten)} cells, relative paths)")

    n = copy_eval_split(dest / "eval")
    print(f"eval split -> {dest / 'eval'} ({n} images)")

    unresolved = [k for k in rewritten if not rewritten[k].startswith((mf.FAILED, mf.UNSUPPORTED))
                  and not mf.built(rewritten, new_manifest_path, *k.split(":"))]
    assert not unresolved, f"manifest entries do not resolve after migration: {unresolved}"

    print(f"\nready. deploy with:\n  rsync -a {dest}/ pi:cv-poc/artifacts/{run}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="source", default="runs/train_poc",
                        help="Ultralytics run dir holding weights/ (default: runs/train_poc)")
    parser.add_argument("--run", default=CURRENT_RUN, help=f"run-id to create (default: {CURRENT_RUN})")
    args = parser.parse_args()

    source = Path(args.source)
    migrate(source if source.is_absolute() else ROOT / source, args.run)


if __name__ == "__main__":
    main()
