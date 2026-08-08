"""Build a dataset at data/<dataset-id>/. See fodcv.research.dataset.

Replaces fetch_dataset.py + remap_classes.py. Those were two commands because
FOD-A needs download-then-convert; an already-labelled YOLO export needs
neither, so `fetch` would have been a command that exists to do nothing. The
registry entry decides what preparing means.

    uv run fodcv-prepare --list
    uv run fodcv-prepare --dataset fod-a
"""

import argparse

from fodcv.paths import CURRENT_DATASET
from fodcv.research import dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=CURRENT_DATASET,
                        help=f"registered dataset id (default: {CURRENT_DATASET})")
    parser.add_argument("--list", action="store_true", help="show registered datasets and exit")
    parser.add_argument("--force", action="store_true", help="rebuild a dataset already on disk")
    args = parser.parse_args()

    if args.list:
        for row in dataset.available():
            state = "prepared" if row["prepared"] else "not prepared"
            print(f"{row['dataset']:<12} {row['kind']:<5} {row['classes']} classes  [{state}]")
        return

    dataset.prepare(args.dataset, force=args.force)


if __name__ == "__main__":
    main()
