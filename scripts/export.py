"""Build every runtime artifact + exports.json. Mac only. See fodcv.research.export."""

import argparse

from fodcv.matrix import DEFAULT_PRECISIONS, FORMATS, IMGSZ, PRECISIONS
from fodcv.paths import CURRENT_RUN
from fodcv.research import export


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=CURRENT_RUN, help=f"artifacts/<run-id> (default: {CURRENT_RUN})")
    parser.add_argument("--weights", default=None, help="override artifacts/<run>/best.pt")
    parser.add_argument("--formats", nargs="+", default=FORMATS)
    parser.add_argument("--precisions", nargs="+", default=DEFAULT_PRECISIONS, choices=list(PRECISIONS))
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--force", action="store_true", help="re-export artifacts already in the manifest")
    args = parser.parse_args()

    export.run(
        run=args.run,
        weights=args.weights,
        formats=args.formats,
        precisions=args.precisions,
        imgsz=args.imgsz,
        force=args.force,
    )


if __name__ == "__main__":
    main()
