"""Fine-tune yolo11n on the PoC subset. See fodcv.research.training."""

import argparse

from fodcv.paths import CURRENT_DATASET, STOCK_WEIGHTS
from fodcv.research import training


def setting(text: str) -> tuple[str, object]:
    """`key=value` for any Ultralytics train argument, typed by what it looks like.

    One flag instead of one per hyperparameter: a sweep varies epochs, imgsz,
    seed, batch, perspective and whatever comes next, and none of those needs
    argparse to know about it.
    """
    key, eq, raw = text.partition("=")
    assert eq and key, f"--set wants key=value, got {text!r}"
    for cast in (int, float):
        try:
            return key, cast(raw)
        except ValueError:
            pass
    if raw.lower() in ("true", "false"):
        return key, raw.lower() == "true"
    return key, raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--angle-aug",
        action="store_true",
        help="v2: add viewpoint-robustness augmentation, separate run dir",
    )
    parser.add_argument("--dataset", default=CURRENT_DATASET,
                        help=f"dataset to train on (default: {CURRENT_DATASET})")
    parser.add_argument("--model", default=STOCK_WEIGHTS, dest="weights",
                        help=f"starting weights (default: {STOCK_WEIGHTS})")
    parser.add_argument("--name", default=None,
                        help="run dir under runs/ (default: train_<dataset>). Pass it when "
                             "sweeping -- two runs sharing a name overwrite each other")
    parser.add_argument("--set", type=setting, action="append", default=[],
                        dest="overrides", metavar="KEY=VALUE",
                        help="any Ultralytics train argument, repeatable: "
                             "--set epochs=60 --set imgsz=480 --set seed=1")
    args = parser.parse_args()
    training.run(angle_aug=args.angle_aug, dataset=args.dataset, weights=args.weights,
                 name=args.name, **dict(args.overrides))


if __name__ == "__main__":
    main()
