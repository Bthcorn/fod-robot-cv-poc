"""Fine-tune yolo11n on the PoC subset. See fodcv.research.training."""

import argparse

from fodcv.research import training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--angle-aug",
        action="store_true",
        help="v2: add viewpoint-robustness augmentation, write to runs/train_poc_v2",
    )
    args = parser.parse_args()
    training.run(angle_aug=args.angle_aug)


if __name__ == "__main__":
    main()
