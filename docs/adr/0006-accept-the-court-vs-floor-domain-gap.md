# Accept the court-vs-floor domain gap; the board demo is the verdict

**Superseded 2026-11-11:** the tennis-ball Demo Object is dropped and the tennis pivot (PR #10, never merged) is reverted. FOD stays the target. See the FOD Robot Decision Log, D-12. Kept as history.

A future reader will reasonably ask why a floor-roving robot is trained on
tennis footage. This records that the gap was seen, measured against, and
accepted deliberately.

The robot's camera is mounted low and forward-tilted (`robot.yaml` target band
10–25°) at a 0.3 m working distance. `RESULT.md` §265 measures a 40 mm screw
there at 49 px of a 480 input, which puts a 67 mm tennis ball at roughly
75–80 px: large, sharp, static, grazing-angle, against flooring. Public
tennis-ball datasets are mostly broadcast and court footage: small, motion
-blurred, airborne balls at 10–30 m against clay or grass, with court
backgrounds behind the `null` frames we chose the dataset for.

This is accepted rather than fixed, because ADR-0001 scopes the claim to the
**toolchain**, and the toolchain's correctness does not depend on what the
pictures contain. The gap cannot invalidate what is actually being claimed. It
can, and probably will, make the model itself unimpressive on a hangar floor —
and saying so plainly is stronger than hoping nobody asks. A tennis ball on a
court and a tennis ball on a workshop floor are different detection problems.

## Pre-registered predictions

Written before training, so that the outcome is evidence rather than a story
told afterwards:

1. Own-split mAP50 will be high — above 0.85 — and will mean very little,
   because the split is scene-contaminated.
2. The Generalisation Gap against `tennis-ball-1-holdout` will be large.
3. The characteristic failure on the board will be **false positives on
   non-court backgrounds**, because the negatives in the training set are court
   negatives. This is the same failure `plan-b3-1class` showed when it put 793
   boxes on 120 frames of an empty room.
4. Detection of a close, grounded, well-lit ball will nonetheless work, because
   the inference-time object is larger and sharper than the training-time one.

The board run from ADR-0002 is the verdict: it is on the real mount, at the
real distance, over the real floor, and it is the only test that answers this
question honestly. No val mAP does.

## Measured after the export arrived (2025-11-03)

Prediction 3's premise was **wrong**, and this is recorded rather than quietly
corrected. The export is not broadcast footage. Ball width on its 320-wide
frames runs min 3 / p25 58 / **median 150** / p75 243 / max 320 px, with only
0.5% of boxes under 10 px. The median ball fills 47% of frame width; the robot
at 0.3 m sees roughly 16%. So training balls are *larger* than deployment
balls, with the p25 sitting close to the robot's size — a better fit than
feared, in the forgiving direction.

A different gap replaces it: every image is **320x240**, while the robot's
Camera Module 3 feeds a 480 input cropped from a 4608x2592 sensor. The model
will be trained on soft, low-resolution balls and deployed on sharp ones.
Prediction 3 is therefore revised to: **false positives on indoor non-court
backgrounds remain the expected failure, and detail-dependent behaviour is
untested by any number this dataset can produce.**

## Remedies held in reserve

Not done now — choosing between them before seeing the data would be guessing:

- Filter the training set by box area, keeping only close-up balls.
- Capture and label ~100 frames of a ball on the actual floor as a third
  holdout. `scripts/capture_raw.py` already does the capture half.
