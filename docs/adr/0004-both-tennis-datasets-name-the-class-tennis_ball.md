# Both tennis datasets name the class `tennis_ball`, and a test enforces it

**Superseded 2026-11-11:** the tennis-ball Demo Object is dropped and the tennis pivot (PR #10, never merged) is reverted. FOD stays the target. See the FOD Robot Decision Log, D-12. Kept as history.

**Status: amended 2025-11-03 — the trap described below does not apply to a
1-class model.** `research/eval.py:229` reads `to_holdout = {} if
len(model.names) == 1 else {...}`, so a single-class model never reaches
`holdout_class` and cannot hit the silent-zero. The naming rule is kept for
consistency with the registry and because `tennis_ball` is also the `ACTIONS`
key, but it is now a readability decision, not a safety one. The vendor names
the class `tennis-ball`; we override to `tennis_ball`.

`research/eval.py:holdout_class` maps a model's class name to a holdout's class
id by **exact string match**, falling back to `unknown` and then to `-1`, which
scores as a miss. A cross-dataset holdout (ADR-0003) has no `unknown` class. So
if the training set names the class `tennis_ball` and the holdout names it
`tennis-ball`, `ball` or `Tennis Ball`, every correct detection scores as a
miss, the report reads 0.0000 mAP, and nothing errors — the same silent-zero
signature as the conf-0.001 `.hef`.

`YoloSource.class_names` is this repo's mapping, not the vendor's, so both
sides are ours to set. Both entries hardcode `tennis_ball`: lowercase like
every existing class, specific rather than `ball` because the class name is
what a REPORT says out loud, and underscored because it is also an `ACTIONS`
key. A test in `tests/test_datasets.py` asserts the two registry entries agree,
because the only failure mode is a silent zero.

## Rejected

Naming the class `fod`. It would need no policy change at all — `ACTIONS["fod"]`
is already PICK and `fod-a-1` already uses that taxonomy. It is rejected
because it records a falsehood in the data: ADR-0001 holds that a tennis ball
is a Demo Object and not FOD, and a dataset labelling tennis balls `fod` makes
that distinction unrecoverable for anyone later reading `run.json`. It would
also silently inherit PICK and undo the REPORT decision.
