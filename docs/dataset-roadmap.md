# Dataset roadmap — the arena dataset, and what its split still needs

The registry in `src/fodcv/research/datasets.py` currently holds only FOD-A
(`fod-a`, `fod-a-3k`). The real dataset — PRD §10's ~2000–2500 self-collected
arena images — is not registered yet. This file holds its entry and the design
decisions that must land with it, so `--list` does not fail on a `source_dir`
that is not there.

## The entry, when the export exists

`YoloSource`, because the arena images arrive already labelled by a tool that
exports YOLO — nothing to download, nothing to convert.

```python
"arena-v1": YoloSource(
    source_dir=Path("~/Downloads/arena-export").expanduser(),
    class_names={0: "fod"},
    val_fraction=0.15,
),
```

**One class, not four.** PRD FR-3 specifies a single trained class,
`fod`; per-class recall is recovered at evaluation time from the
seeding log. The 4-class scheme (`nail`/`screw`/`bolt`/`unknown`) belongs to the
FOD-A comparison only.

## Read PRD §10 step 4 before registering it

> "Split 70/15/15, grouped so one scene never spans train and test; keep a
> cross-venue holdout from the machine-shop visit."

None of that is implemented. `split()` in `research/dataset.py` is still a plain
per-image shuffle into train/val, which is correct for FOD-A — §10 step 1 calls
FOD-A a *pretraining prior*, not a test set — and wrong for the arena data:

- **Scenes must be grouped first.** A scene is one difference-imaging camera lock
  (§10 step 3: lock the camera, shoot background, place fasteners, shoot
  foreground), so a single scene spans many near-identical images. Shuffling per
  image puts the same scene on both sides of the split and inflates held-out
  mAP. This is the same failure RESULT.md §7 measures on `fod-a-3k`, where 74% of
  validation frames have a near-duplicate in train.
- **There is no test split at all.** `val_fraction` cuts train/val only; 70/15/15
  needs a third slice.
- **The cross-venue holdout is a collection decision**, not something a split
  fraction can express — which shoot is held out. Most likely its own dataset-id
  rather than a slice of this one.

## Why a new id rather than editing an existing one

Preparing a dataset rebuilds its directory and reshuffles which images land in
val. Every mAP already recorded is against `fod-a`'s 90-image split, and those
numbers stay comparable only while that split stays put. `fod-a-3k` exists for
exactly this reason — same source, cap lifted, new id.
