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
                        help="NMS score threshold to COMPILE INTO the hailo .hef; ignored by every "
                             "other format. Defaults to the matrix's 0.001, which keeps the "
                             "benchmark row comparable to the host-NMS backends. A deploy .hef "
                             "wants 0.10-0.25 instead -- on the Pi the compiled value is a floor "
                             "that --conf can only filter above")
    parser.add_argument("--calib-fraction", type=float, default=None,
                        help="fraction of the dataset's TRAIN split to calibrate INT8 from. "
                             "Unset uses all of it, which is right for a 3k dataset and "
                             "hours of DFC time for a 10k one. Hailo warns below 1,024 "
                             "images and poc-v2 used 2,550. Ultralytics takes "
                             "im_files[:round(n*fraction)] -- the first N in sorted order, "
                             "not a sample")
    parser.add_argument("--a16-cls", action="store_true",
                        help="hailo only: compile the detect head's class convs at 16-bit. "
                             "Costs latency and size; buys back a class head that INT8 "
                             "silenced. See export.a16_classification_head")
    parser.add_argument("--a16-all", action="store_true",
                        help="hailo only: 16-bit on every output conv, regression included. "
                             "--a16-cls widens only the class convs; if the box branch is "
                             "also losing range this shows it. See export.a16_all_outputs")
    parser.add_argument("--opt-level", type=int, default=None, choices=range(5),
                        help="hailo only: DFC model_optimization_flavor level. Ultralytics "
                             "hardcodes 2; 3-4 add bias correction and AdaRound, which cost "
                             "compile time, not runtime. See export.raise_optimization_level")
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
        calib_fraction=args.calib_fraction,
        a16_cls=args.a16_cls,
        a16_all=args.a16_all,
        opt_level=args.opt_level,
    )


if __name__ == "__main__":
    main()
