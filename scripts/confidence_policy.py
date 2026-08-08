"""Confidence hysteresis + temporal smoothing demo. See fodcv.runtime.policy."""

import argparse

from fodcv.paths import CURRENT_RUN
from fodcv.runtime import policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        nargs="?",
        default="0",
        help="camera index (e.g. 0), image directory, or video path (default: 0)",
    )
    parser.add_argument("--no-show", action="store_true", help="skip the live window, print only")
    parser.add_argument("--run", default=CURRENT_RUN, help=f"weights from artifacts/<run-id> (default: {CURRENT_RUN})")
    args = parser.parse_args()
    policy.run(source=args.source, show=not args.no_show, model_run=args.run)


if __name__ == "__main__":
    main()
