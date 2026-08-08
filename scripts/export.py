"""Build every runtime artifact + exports.json. Mac only. See fodcv.research.export."""

import argparse

from fodcv.matrix import DEFAULT_PRECISIONS, FORMATS, IMGSZ, PRECISIONS
from fodcv.paths import trained_weights
from fodcv.research import export


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=None, help=f"default: {trained_weights()}")
    parser.add_argument("--formats", nargs="+", default=FORMATS)
    parser.add_argument("--precisions", nargs="+", default=DEFAULT_PRECISIONS, choices=list(PRECISIONS))
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--force", action="store_true", help="re-export artifacts already in the manifest")
    args = parser.parse_args()

    export.run(
        weights=args.weights,
        formats=args.formats,
        precisions=args.precisions,
        imgsz=args.imgsz,
        force=args.force,
    )


if __name__ == "__main__":
    main()
