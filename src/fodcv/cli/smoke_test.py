"""Stock yolo11n sanity check on a few dataset images. See fodcv.research.smoke."""

import argparse

from fodcv.paths import CURRENT_DATASET
try:
    from fodcv.research import smoke
except ImportError as e:  # the robot's base install has no ultralytics, by design
    raise SystemExit(f"{e}\nfodcv-smoke needs the research extra: pip install 'fod-vision[research]'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=CURRENT_DATASET,
                        help=f"dataset to sample images from (default: {CURRENT_DATASET})")
    args = parser.parse_args()
    smoke.run(dataset=args.dataset)


if __name__ == "__main__":
    main()
