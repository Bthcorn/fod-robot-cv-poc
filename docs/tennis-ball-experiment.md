# The tennis-ball experiment

A Demo Object carried through the whole toolchain, as evidence that the
toolchain carries a second dataset. **Read `docs/adr/0001`–`0007` first** — they
hold the decisions and this file holds only the procedure and the results.

FOD is still the problem. Nothing here ships: `CURRENT_RUN`, `CURRENT_DATASET`
and `DEPLOY_*` stay pointed at the fastener model throughout.

## The dataset

| | |
|---|---|
| Source | Roboflow `lachlan-2jkk9/detect-tennis-ball-with-null` |
| Version | **1** (2024-08-28), exported 2025-04-11 |
| Licence | CC BY 4.0 |
| Zip sha256 | `40db93c1f4a24b5dc704ed4d3e58b113ac7042390c6840fbb3ae0b0814d8c999` |
| Dataset-id | `tennis-ball-1-neg` |

The version pin is not bookkeeping. Regenerating a Roboflow version with
different augmentation yields different bytes from the same URL, and every
number below would silently stop being reproducible. The download link carries
an account credential and is **not** recorded here — ADR-0007.

Measured from the export itself, not from its web page:

| property | value | why it matters |
|---|---|---|
| Images | 2,970 train / 333 valid / 195 test | `test/` is dropped by `SHIPPED_SPLITS`; we use it by hand, below |
| Empty labels | **1,562 of 2,970 train (53%)** | the reason this export was chosen |
| Image size | **320×240, all of them** | "Resize to 320x240 (Fill with center crop)" — so 640 would upsample and buy nothing |
| Augmentation | 3 versions per source image | ~990 real sources, so `valid/` is **not** scene-disjoint |
| Ball width | min 3 / p25 58 / **median 150** / p75 243 / max 320 px | close-ups, not broadcast footage. 0.5% of boxes under 10 px |

The median ball fills **47% of frame width**; the robot at 0.3 m sees roughly
16%, and `RESULT.md` §265 is where that figure comes from. Training balls are
therefore *larger* than deployment balls, with p25 close to the robot's size.

## Splits, and what each number is allowed to claim

| split | used as | may be reported as |
|---|---|---|
| `train/` 2,970 | training | — |
| `valid/` 333 | Own-split Score | a progress readout, nothing more |
| `test/` 195 | same-source test split | a clean-ish number from **the same source** |

There is **no Cross-dataset Holdout**. The second dataset planned for the role
(`rowerup/tennis-ball-8bsra`) publishes no downloadable export. So no number in
this document is a generalisation result, and none may be quoted as one — the
board run is the only honest verdict available. ADR-0003.

Build the test split by hand, since `fodcv-prepare` models two splits and drops
`test/`:

```bash
SRC=~/Downloads/detect-tennis-ball-with-null.v1i.yolov11
D=data/tennis-ball-1-test
mkdir -p $D/images/val $D/labels/val
cp $SRC/test/images/* $D/images/val/
cp $SRC/test/labels/* $D/labels/val/
printf 'train: images/val\nval: images/val\nnames:\n  0: tennis_ball\n' > $D/data.yaml
```

## Stage 1 — the Mac Gate

Its purpose is to **fail cheaply**. It proves the data, the registry, the
scorer and the export path are wired; it proves nothing about accuracy or
latency, and every number it produces is throwaway.

```bash
unzip -q ~/Downloads/detect-tennis-ball-with-null.v1i.yolov11.zip \
      -d ~/Downloads/detect-tennis-ball-with-null.v1i.yolov11
uv run fodcv-prepare --dataset tennis-ball-1-neg     # expect 2970 / 333
uv run pytest -q                                     # expect 167 passed
uv run fodcv-smoke --dataset tennis-ball-1-neg
uv run fodcv-train --dataset tennis-ball-1-neg --name mac-gate-tennis-320 \
   --set epochs=3 --set imgsz=320 --set batch=16 --set seed=0 --set patience=0
uv run fodcv-eval runs/mac-gate-tennis-320/weights/best.pt \
   --eval data/tennis-ball-1-test --imgsz 320
```

**Result, 2025-11-03 (M4, MPS).** All six steps passed. 3 epochs in 0.164 h
(3.3 min/epoch at 320). Own-val mAP50 **0.9380**; test-split mAP50 **0.9306**,
recall **0.887** at conf 0.25 over 159 instances. Export produced 7 of 8
formats before being interrupted; `litert:int8` was not reached.

Two things that result is **not**: a model worth keeping, and a measurement of
anything. 0.938 after three epochs is what ~990 source images augmented three
ways and split at random look like. ADR-0006 predicted exactly this.

The gate found two bugs in the scorer, both fixed in `d83968a` — a 1-class
model reported `0.000` recall where it meant "undefined", and every report
claimed its eval directory was scene-disjoint whether or not it was.

## Stage 2 — the pod

Follow `docs/autorun-runpod-e2e.md` Stage 0 for setup, with three differences.

**Take a cheap GPU.** That file's 4090 spec is sized for a `yolo11m` sweep.
This is two `yolo11n` runs over 2,970 images at 320 and 480 — 8× fewer pixels
per image than the arg-bolts sweep, which measured 25.2 s/epoch at 640 on 8,891
images.

**`uv sync --extra research` is enough for training**; the `export` extra is
only needed if the compile happens on the same box (it brings torch plus the
CUDA libraries, ~15 GB).

**The dataset travels Mac→pod**, not via Drive — ADR-0007:

```bash
rsync -a ~/Downloads/detect-tennis-ball-with-null.v1i.yolov11 pod:~/Downloads/
# on the pod
uv run fodcv-prepare --dataset tennis-ball-1-neg
```

Then the two runs. One variable moves — `imgsz`. ADR-0005.

```bash
for SZ in 320 480; do
  uv run fodcv-train --dataset tennis-ball-1-neg \
    --name tennis-ball-1-neg-n-$SZ \
    --set imgsz=$SZ --set epochs=60 --set batch=16 --set seed=0 --set patience=0
done
```

`patience=0` is deliberate and is not a typo: `close_mosaic=10` means any
early stop before epoch 50 cuts the run off before mosaic closes and before the
learning rate finishes annealing — its two best phases. `batch` is fixed at 16
across both runs or the comparison is confounded.

Score both against the same split, at the size each was trained at:

```bash
uv run fodcv-eval runs/tennis-ball-1-neg-n-320/weights/best.pt --imgsz 320 \
   --eval data/tennis-ball-1-test
uv run fodcv-eval runs/tennis-ball-1-neg-n-480/weights/best.pt --imgsz 480 \
   --eval data/tennis-ball-1-test
```

**480 wins a tie.** It is 1.87× faster on the Hailo-8 (16.6 ms against 24.4 ms),
it leaves the control loop the other 8 ms, and the conf-0.001 silent-zero bug is
640-specific. 320 wins outright if it matches 480, being faster still. Declare
320 the loser only on a wide margin, since both numbers come from the same
source and small gaps between them mean nothing.

### Results

_Not yet run._

| run | imgsz | own-val mAP50 | test-split mAP50 | recall @0.25 |
|---|---|---|---|---|
| `tennis-ball-1-neg-n-320` | 320 | | | |
| `tennis-ball-1-neg-n-480` | 480 | | | |

## Stage 3 — the board

Compile **only the winner**. `HEF_IMGSZ` must equal the training `imgsz`:
`nms_config.json` carries the input size and `Vision` reads it off the `.hef`,
so a mismatch letterboxes in silence. `HEF_CONF=0.0001`, never 0.001.
`--a16-cls` does not apply — that is for heads above roughly four classes and
this one has one.

Then `fodcv-hailo-camera --preview` on the Pi, with a ball on the floor.

**This is the experiment's only honest verdict.** It is on the real mount, at
the real distance, over the real floor. No val mAP in this document answers the
question it answers. Watch for the pre-registered failure in ADR-0006: false
positives on indoor, non-court backgrounds, because every negative in the
training set is a court negative.

### Board result

_Not yet run._
