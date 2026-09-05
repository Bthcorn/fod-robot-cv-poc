"""Score trained runs against the scene-disjoint holdout. See fodcv.research.eval.

Takes weights paths rather than run ids because the sweep's outputs live under
runs/plan-*/weights/best.pt, not artifacts/. Several at once, ranked at the end,
since the question is never "how good is this run" but "which of these eleven".
"""

import argparse
from pathlib import Path

from fodcv.paths import HOLDOUT_DIR
from fodcv.research import eval as evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", nargs="+", type=Path,
                        help="one or more best.pt to score")
    parser.add_argument("--eval", type=Path, default=HOLDOUT_DIR, dest="eval_dir",
                        help=f"holdout directory (default: {HOLDOUT_DIR})")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="deploy confidence, for recall and FP counting (default: 0.25)")
    parser.add_argument("--imgsz", type=int, default=480,
                        help="inference size; must match how the run is deployed (default: 480)")
    parser.add_argument("--background", type=Path, default=None,
                        help="directory of unlabelled FOD-free frames, for FP/frame")
    args = parser.parse_args()

    reports = []
    for weights in args.weights:
        assert weights.exists(), f"no weights at {weights}"
        report = evaluation.score(weights, args.eval_dir, conf=args.conf, imgsz=args.imgsz,
                                  background_dir=args.background, run=_run_name(weights))
        reports.append(report)
        print(evaluation.format_report(report), end="\n\n")

    if len(reports) > 1:
        print("ranked by class-agnostic mAP50-95:")
        for i, r in enumerate(sorted(reports, key=lambda r: -r.map50_95), 1):
            print(f"  {i:>2}. {r.map50_95:.4f}  {r.run}")
        print("\nA gap smaller than phase A's seed spread is noise, not a result.")


def _run_name(weights: Path) -> str:
    """runs/plan-a1-seed0/weights/best.pt -> plan-a1-seed0."""
    parent = weights.parent
    return (parent.parent if parent.name == "weights" else parent).name


if __name__ == "__main__":
    main()
