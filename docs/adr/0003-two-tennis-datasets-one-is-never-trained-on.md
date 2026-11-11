# Two tennis-ball datasets; the second one is never trained on

**Superseded 2026-11-11:** the tennis-ball Demo Object is dropped and the tennis pivot (PR #10, never merged) is reverted. FOD stays the target. See the FOD Robot Decision Log, D-12. Kept as history.

**Status: amended 2025-11-03 — the cross-dataset holdout does not exist.**
`rowerup/tennis-ball-8bsra` offers no downloadable export, so there is no
second source. In its place the training set's own shipped `test/` split (195
images, un-augmented, normally discarded by `SHIPPED_SPLITS`) is used as a
clean **same-source** test split. It is *not* a Cross-dataset Holdout and must
never be reported as a generalisation number: same photographer, same scenes,
same 320x240 pre-processing. The real generalisation verdict moves to the
board run of ADR-0002, optionally backed by ~100 self-captured floor frames.
Everything below about *why* a second source was wanted still stands, and the
want is now unmet.

Two sources are registered. `tennis-ball-null` (Roboflow
`lachlan-2jkk9/detect-tennis-ball-with-null`) is the training set, chosen for
the background images its name advertises — this project has already shipped a
model that scored 60/60 on a clean holdout and still put 793 boxes on 120
frames of an empty room, which is why `fod-a-1-neg` exists with 9,360
after-the-fact negatives. A floor-roving robot sees empty floor in most frames.

`tennis-ball-8bsra` (Roboflow `rowerup/tennis-ball-8bsra`) is prepared and
**never trained on**, serving as a cross-dataset holdout. Every Roboflow export
in this repo overlaps its own `valid/` split with `train/` at scene level —
14.5% and 25.6% measured — so an Own-split Score on the training set is not
interpretable as a result. A second photographer's data is the only honest
generalisation number available without collecting our own.

Both numbers get reported, labelled, and the gap between them is treated as a
finding rather than hidden. This is how `RESULT.md` §7 already reports the
FOD-A near-duplicate problem, so it is consistent with the project's existing
standard of evidence.

## Consequences

- **Training on `tennis-ball-8bsra` is irreversible.** The decision is cheap to
  revisit in one direction only: once the holdout has contributed a training
  frame it can never be decontaminated. Nothing in the pipeline enforces this,
  so it is written down here instead.
- The cross-dataset number will be substantially worse than the own-split
  number. That is expected and is the point.
- Both datasets must name the class with the same string — see ADR-0004.

## Rejected

- `mlnomad/imnet1k_tennis_ball` (HuggingFace) and the images.cv tennis-ball
  set: both are **classification** datasets with image-level labels and no
  bounding boxes. Confirmed for the former via the HuggingFace API — its
  features are `image` plus a `label` over the 1000 ImageNet class names.
  Unusable for detection. A possible later source of hard negatives.
- `tennis-lpqkk/tennis-and-player-detection`: carries a `player` class the
  robot has no Action for, and ADR-0001 rejects multi-purpose heads.
