"""Runtime benchmark matrix on the Pi 5. See fodcv.bench.pi."""

import argparse

from fodcv.bench import pi
from fodcv.bench.pi import DEFAULT_PRECISIONS, FORMATS, PRECISIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=None, help="default: newest fine-tuned best.pt")
    parser.add_argument("--models", nargs="+", default=None, help="extra weights, e.g. yolo26n.pt")
    parser.add_argument("--formats", nargs="+", default=FORMATS)
    parser.add_argument("--precisions", nargs="+", default=DEFAULT_PRECISIONS, choices=list(PRECISIONS))
    parser.add_argument("--threads", type=int, default=None, help="intra-op threads; also use taskset")
    parser.add_argument("--no-val", action="store_true", help="skip mAP -- latency-only sweeps")
    parser.add_argument("--soak", type=int, default=0, help="seconds of sustained load instead of the matrix")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        return pi.selftest()

    pi.run(
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
