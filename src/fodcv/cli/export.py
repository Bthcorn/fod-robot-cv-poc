"""Build every runtime artifact + exports.json. Mac only. See fodcv.research.export."""

import argparse

from fodcv.matrix import DEFAULT_PRECISIONS, FORMATS, IMGSZ, PRECISIONS
from fodcv.paths import CURRENT_DATASET, CURRENT_RUN
from fodcv.research import export


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=CURRENT_RUN, help=f"artifacts/<run-id> (default: {CURRENT_RUN})")
    parser.add_argument("--weights", default=None, help="override artifacts/<run>/best.pt")
    parser.add_argument("--dataset", default=CURRENT_DATASET,
                        help=f"dataset used for INT8 calibration (default: {CURRENT_DATASET})")
    parser.add_argument("--formats", nargs="+", default=FORMATS)
    parser.add_argument("--precisions", nargs="+", default=DEFAULT_PRECISIONS, choices=list(PRECISIONS))
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--force", action="store_true", help="re-export artifacts already in the manifest")
    parser.add_argument("--conf", type=float, default=None,
                        help="NMS score threshold to COMPILE INTO the hailo .hef. Ignored by every "
                             "other format, which take conf= at call time. Defaults to the matrix's "
                             "0.001, which exists so the benchmark row keeps its low-confidence tail "
                             "and stays comparable to the host-NMS backends. A deploy .hef wants "
                             "something usable -- 0.10 to 0.25 -- because on the Pi the compiled "
                             "value is a floor that --conf can only filter above")
    args = parser.parse_args()

    export.run(
        run_id=args.run,
        dataset=args.dataset,
        weights=args.weights,
        formats=args.formats,
        precisions=args.precisions,
        imgsz=args.imgsz,
        force=args.force,
        conf=args.conf,
    )


if __name__ == "__main__":
    main()
