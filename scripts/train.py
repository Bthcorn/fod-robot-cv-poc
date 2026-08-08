"""Fine-tune yolo11n on the PoC subset. See fodcv.research.training."""

import argparse

from fodcv.paths import CURRENT_DATASET
from fodcv.research import training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--angle-aug",
        action="store_true",
        help="v2: add viewpoint-robustness augmentation, separate run dir",
    )
    parser.add_argument("--dataset", default=CURRENT_DATASET,
                        help=f"dataset to train on (default: {CURRENT_DATASET})")
    args = parser.parse_args()
    training.run(angle_aug=args.angle_aug, dataset=args.dataset)


if __name__ == "__main__":
    main()
