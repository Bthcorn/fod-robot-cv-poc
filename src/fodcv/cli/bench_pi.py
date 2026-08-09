"""Runtime benchmark matrix on the Pi 5. See fodcv.bench.pi."""

import argparse

from fodcv.bench import pi
from fodcv.matrix import DEFAULT_PRECISIONS, FORMATS, PRECISIONS
from fodcv.paths import CURRENT_DATASET, CURRENT_RUN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=CURRENT_RUN, help=f"artifacts/<run-id> (default: {CURRENT_RUN})")
    parser.add_argument("--weights", default=None, help="override artifacts/<run>/best.pt")
    parser.add_argument("--dataset", default=CURRENT_DATASET,
                        help=f"eval fallback when the run ships no eval/ (default: {CURRENT_DATASET})")
    parser.add_argument("--models", nargs="+", default=None, help="extra weights, e.g. yolo26n.pt")
    parser.add_argument("--formats", nargs="+", default=FORMATS)
    parser.add_argument("--precisions", nargs="+", default=DEFAULT_PRECISIONS, choices=list(PRECISIONS))
    parser.add_argument("--threads", type=int, default=None, help="intra-op threads; also use taskset")
    parser.add_argument("--no-val", action="store_true", help="skip mAP -- latency-only sweeps")
    parser.add_argument("--soak", type=int, default=0, help="seconds of sustained load instead of the matrix")
    args = parser.parse_args()

    pi.run(
        run_id=args.run,
        dataset=args.dataset,
        weights=args.weights,
        models=args.models,
        formats=args.formats,
        precisions=args.precisions,
        threads=args.threads,
        run_val=not args.no_val,
        soak_seconds=args.soak,
    )


if __name__ == "__main__":
    main()
