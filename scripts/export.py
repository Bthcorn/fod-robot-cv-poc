"""Build every runtime artifact + exports.json. Mac only. See fodcv.research.export."""

import argparse

from fodcv.research import export
from fodcv.research.export import DEFAULT_PRECISIONS, FORMATS, IMGSZ, PRECISIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=None, help=f"default: {export.default_weights()}")
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
