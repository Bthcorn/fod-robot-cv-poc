# Live camera runs on real fasteners — 2026-09-06

Manual runs on the Pi 5 with Camera Module 3 (imx708), `fodcv.cli.camera_hailo`,
against the `.hef` builds already on the board. Purpose: the 200-image mAP gate
is dataset images at dataset framing; this is fasteners under the real lens.

Board: `ai@raspberrypi.local`, `~/fod-robot-cv-poc`, governor `performance`,
`lightdm` active (preview needs it). One Hailo session at a time.

## Where things live

The Mac is `/Users/bowornthat/Projects/GraduateProject/cv-poc`. Paths below are
relative to the repo root on whichever machine is named.

| what | Mac | Pi |
|---|---|---|
| `arg-bolts-4-{n,s}-640`, `-480-100e` | yes | yes |
| `fastener-7-480`, `poc-v1-480`, `poc-v2-480{,-3k,-full}` | yes | yes |
| **`plan-b3-1class`** | **absent** | `artifacts/plan-b3-1class/` |
| **`plan-b4-7class-{480,640,a16,dead640}`** | **absent** | `artifacts/plan-b4-7class-*/` |
| this session's camera output | absent | `runs/camera-b3-1class{,-zoom1}/` |
| earlier camera output | `runs/camera-argbolts-n-640/`, `runs/camera_hailo/` | — |

**`plan-b3-1class` and the four `plan-b4-*` runs exist on the Pi only.** They are
the models behind the `bfcccc7` a16 diagnosis and behind this session's finding,
and there is no second copy. Back them up before anything reformats that board.

Named builds on the Mac, the ones worth keeping straight:

| build | Mac path | sha256 |
|---|---|---|
| `n` 640 conf 0.0001 — ships | `artifacts/arg-bolts-4-n-640/SHIPPING_640_conf0001_2026-09-06/` | `d596c9c82f2c` |
| `n` 480 | `artifacts/arg-bolts-4-n-640/SHIPPING_480_2026-09-06/` | `c5cd985e92c4` |
| `s` 640 a16 | `artifacts/arg-bolts-4-s-640/WORKING_a16_2026-09-05/` | `f2daf45d6ea1` |

On the Pi the same three files sit at `bench_int8_hailo_model_conf00001/`,
`bench_int8_hailo_model_480/` and `bench_int8_hailo_model/` — byte-identical,
verified by sha256. Note `Vision` derives class names from
`../run.json` relative to the `.hef`, so a `.hef` nested one level deeper inside
a `SHIPPING_*`/`WORKING_*` folder **will not load** — always point the camera at
the flat `bench_int8_hailo_model*` paths.

To pull this session's frames and timings to the Mac:

    rsync -a ai@raspberrypi.local:fod-robot-cv-poc/runs/camera-b3-1class-zoom1/ runs/camera-b3-1class-zoom1/

## Runs

### `plan-b3-1class` — yolo11n, 1 class (`fod`), imgsz 640, baked NMS floor 0.001

Pi path: `artifacts/plan-b3-1class/bench_int8_hailo_model/best.hef`

| | zoom 0.28 | zoom 1.00 | notes for the decision |
|---|---:|---:|---|
| frames | 600 | 957 | 0.28 run was bounded by me; 1.00 run quit by hand |
| detections | 5,514 | 8,223 | raw count, `--conf 0.25` |
| per frame | 9.2 | 8.6 | both far above the fasteners actually in view — see flicker below |
| first-frame score | 0.36 | **0.86** | **zoom 1.00 wins**; sensor detail beat apparent size |
| sensor px across crop | 645 (upscaled) | 2,304 (native) | 0.28 feeds the model interpolated pixels |
| object scale to model | 1.79N px | 0.50N px | 0.28 makes objects bigger but softer |
| infer median | 15.18 ms | 15.24 ms | zoom is a sensor crop — no effect on model cost |
| end-to-end | 33.46 ms / 29.9 FPS | 33.72 ms / 29.7 FPS | both sit right on the 33 ms budget |
| at training scale for 40 mm? | no (wants 0.63) | no (wants 0.54) | **neither run was correctly framed** |

**Decision:** run zoom 1.00 by default, and re-run at `--zoom 0.54` before
trusting any count — both runs so far were off training scale, so these numbers
are indicative, not a gate.

**Observed by eye (zoom 1.00):** detection on real fasteners is good and frame
rate is good, but it *flickers on other objects* — non-fastener items in the
scene fire intermittently between frames.

The counts corroborate it. 8,223 detections over 957 frames is **8.6 per frame
at `--conf 0.25`**, far more than the number of fasteners actually in view. A
1-class `fod` head has no competing class to lose to, so anything roughly
fastener-shaped clears the threshold on some frames and not others. This is a
precision problem, not a recall problem, and it is exactly what a single-class
formulation buys you. Raising `--conf` trades it against recall; the baked floor
is 0.001 so there is plenty of host-side room to tune.

Backbone is yolo11n: `best.pt` 5,473,690 B against `arg-bolts-4-n-640`'s
5,474,778 B. `arg-bolts-4-s-640` is 19,181,146 B.

### `plan-b4-7class-480` — yolo11n, 7 classes, imgsz 480, baked NMS floor 0.1, **a8**

Pi path: `artifacts/plan-b4-7class-480/bench_int8_hailo_model/best.hef`
Classes: nail, screw, bolt, washer, nut, boltwasher, boltnutset

| | zoom 1.00 | notes for the decision |
|---|---:|---|
| frames | 968 | quit by hand |
| detections | 11,065 | 11.4 per frame — **worse flicker than the 1-class model** |
| first-frame scores | bolt 0.20, washer 0.49, nut 0.19, boltnutset 0.41; nail / screw / boltwasher 0.00 | alive, but weak and uneven |
| infer median | 10.63 ms | fastest of the three runs — 480 input |
| end-to-end | 33.94 ms / 29.5 FPS | sensor-limited, same as every run |
| at training scale for 40 mm? | no (wants 0.71) | off scale again |

**Observed by eye:** *"not that good at detection, with a lot of flickers."*

The numbers agree. 11.4 detections per frame at `--conf 0.25` is the highest of
any run, and the first-frame scores are clustered at 0.19-0.49 where the 1-class
model reached 0.86. Weak, unstable per-class scores across seven classes are
what flicker looks like in the count column — proposals crossing 0.25 on some
frames and not others.

**This build is not dead.** `bfcccc7` diagnosed the 7-class head as having lost
its scores to 8-bit; at 480 with a 0.1 floor the a8 build clearly decodes. That
claim was always specifically about 640, and this does not contradict it — but
it does establish the a8 baseline the a16 arm has to beat.

**Decision: run `plan-b4-7class-a16` next and compare score strength and
detections-per-frame, not just "does it detect".** Both are 480, both floor 0.1,
differing only in the flag. If a16 lifts the scores and cuts the flicker, the
flag earns its keep on evidence that is not confounded by resolution.

### `plan-b4-7class-a16` — same as above but **a16** (`--a16-cls`)

Pi path: `artifacts/plan-b4-7class-a16/bench_int8_hailo_model/best.hef`
Identical to `plan-b4-7class-480` in imgsz (480), NMS floor (0.1) and class list.
The only difference is the flag.

| same 480, same 0.1 floor | a8 | a16 | notes for the decision |
|---|---:|---:|---|
| frames | 968 | 1,628 | different run lengths — not matched |
| detections | 11,065 | 11,476 | raw totals not comparable at different lengths |
| **per frame** | 11.4 | **7.05** | fewer — reads as less flicker |
| best first-frame score | washer 0.49 | washer **0.13** | **weaker**, not stronger |
| classes non-zero, first frame | 4 of 7 | 2 of 7 | fewer classes firing |
| infer median | 10.63 ms | 11.10 ms | a16 costs ~0.5 ms |
| end-to-end | 33.94 ms | 33.90 ms | sensor-limited either way |
| lens at | 0.39 m | 0.42 m | **different scenes** |
| FocusFoM | 345 | 237 | different focus quality |

**Observed by eye:** *"also not good at detection, might need to zoom, and has
some flickers and detecting other objects."* — i.e. indistinguishable from the a8
arm in the ways that matter, plus false positives on non-fasteners.

**a16 does not show a live win.** Detections per frame fell, but the scores fell
with them, which is the opposite of what `--a16-cls` is supposed to buy. Both
readings fit the data: a16 may be suppressing junk, or it may simply be scoring
everything lower so less clears `--conf 0.25`. Nothing here separates the two.

**This pair does not settle the a16 question either.** Same build config, but
uncontrolled input — different scenes, different focus states, different run
lengths, and first-frame scores depend entirely on what happened to be in view
at frame 0. Live camera cannot hold the input fixed, so it cannot do a
controlled A/B, however well matched the builds are.

**Decision: stop trying to settle a16 from live runs.** The free controlled route
is `plan-b4-7class-dead640`'s 1,404-image eval split — same `fod-a-7` lineage,
identical images for both arms:

    ~/.local/bin/uv run fodcv-bench --run plan-b4-7class-480 --imgsz 480 \
      --formats hailo --precisions int8 --val-max 200 --dataset fod-a-7

Two limits on that route: neither 480 run ships a `best.pt`, so it compares a8
against a16 and not either against ground truth; and if those builds trained on
images inside dead640's eval split the absolute numbers are inflated — the
a8-vs-a16 difference survives that, since it hits both arms equally.

### `plan-b4-7class-dead640` — yolo11n, 7 classes, imgsz 640, floor 0.001, a8

Pi path: `artifacts/plan-b4-7class-dead640/bench_int8_hailo_model/best.hef`

| | zoom 1.00 | notes for the decision |
|---|---:|---|
| frames | 256 | stopped early — nothing to see |
| detections | **0** | zero over the whole run |
| first-frame scores | all seven classes **0.00** | not low: zero |
| postprocess median | **0.03 ms** | ~0.7 ms in working runs — nothing reached the host |
| infer median | 15.38 ms | the chip is running; it just returns nothing |
| end-to-end | 33.61 ms / 29.8 FPS | sensor-limited as always |

**Observed by eye:** *"just dead as it"* — nothing on the preview at any point.

**Confirmed dead live.** This is the same signature `arg-bolts-4-n-640` showed at
the identical configuration. The `postprocess` collapse to 0.03 ms is the
clearest tell — the host had no proposals to filter.

**This undercuts `bfcccc7`.** That commit diagnosed the 7-class head as having
lost its scores to 8-bit, and credited `--a16-cls` with the fix — but the fix was
rebuilt at **480**, so a16 and resolution moved together. The control that was
never run then has now been run: `plan-b4-7class-480` is **a8 at 480 and it
works**. The recovery is explained by the resolution change alone. a16 was never
shown to be necessary for this model.

**Caveat:** none of the three `plan-b4-7class-*` builds ships a `best.pt` on the
Pi, so it cannot be verified that they share weights. If they do not, the
comparison is weaker than it reads.

### `arg-bolts-4-n-640/bench_int8_hailo_model_conf015` — floor 0.15

Pi path: `artifacts/arg-bolts-4-n-640/bench_int8_hailo_model_conf015/best.hef`
Same weights as the shipping build and the dead 0.001 build. Third floor.

    scores  bolt 0.00  nut 0.00  screw 0.00  washer 0.00
    0 detections over 189 frames
    postprocess 0.02 ms   infer 15.25 ms   33.30 ms / 30.0 FPS

Dead. This is the run that closed the floor question inside a single model.

**Note on how this nearly went untested.** It was skipped earlier in the session
on the advice that a 0.15 floor was "the same mistake, one step worse" — reasoning
that assumed higher floors are strictly worse. `plan-b4-7class-640` at floor 0.1
appeared to contradict that, which is what made this run worth doing. It was, and
it produced the retraction above.

## The open cell — resolved, and it breaks the explanation

`plan-b4-7class-640` (a8, 640, floor **0.1**) was run live. It is **alive**:

    scores  nail 0.10  screw 0.00  bolt 0.73  washer 0.38
            nut 0.46  boltwasher 0.67  boltnutset 0.00
    28,320 detections over 1,143 frames  (24.8/frame -- very noisy, but alive)
    postprocess 4.74 ms   (0.03 ms on the dead build)
    infer 15.39 ms   end-to-end 32.21 ms / 31.0 FPS

**Observed by eye:** *"this one has detections, and a lot of flickers with
backgrounds and other objects."* At 24.8 detections per frame it is the noisiest
run of the session by a wide margin — roughly 2-3x the others — so "alive" here
means the head decodes, not that the build is usable. Background regions are
clearing `--conf 0.25`.

Against `plan-b4-7class-dead640`: same 640, same 7 classes, same a8, same
lineage, **only the baked floor differs** — 0.001 dead, 0.1 alive.

**Resolution is not the variable. The floor is.** And `--a16-cls` is not the
variable either: a8 works at both 480 and 640 once the floor is off 0.001.

### The floor response is monotonic — an earlier claim in this file was wrong

**Retracted:** an earlier revision of this section claimed the floor response is
non-monotonic, that 0.001 is a poisoned band with working builds on both sides.
That was built on the `plan-b4-7class-*` pair, which are precisely the builds
with no `best.pt` on the Pi — shared weights were never verifiable. The single
model where weights *are* verified says the opposite.

`arg-bolts-4-n-640`, one set of weights, three builds differing only in the
baked floor:

| build | floor | result | max score |
|---|---:|---|---:|
| `_conf00001` | 0.0001 | works — 0.7715 mAP50 | 0.909 |
| `bench_int8_hailo_model` | 0.001 | dead — 0.0000 | ~0.007 |
| `_conf015` | **0.15** | **dead — 0 detections over 189 frames** | 0.00 |

Monotonic: only the lowest floor works. The `plan-b4-7class-640` result (floor
0.1, alive) is the outlier, and it is the one whose weights cannot be checked
against its 0.001 sibling. Treat it as unexplained, not as a counterexample.

**What does not fit a simple threshold.** A runtime filter cannot change the
scores it filters, yet max score fell 0.909 -> 0.007 -> 0.00 as the baked floor
rose. So `HEF_CONF` alters what the compiled model *produces*, not only what the
chip discards. `src/fodcv/matrix.py`'s comment describes the floor as eating the
scores; the practical guidance it gives is correct and measured, and the
mechanism is imprecise rather than wrong. **Do not rewrite that comment on the
strength of this session** — it would need a compile-time investigation to state
the mechanism properly.

### Full floor map, all models

| model | imgsz | floor | result | weights verified? |
|---|---:|---:|---|---|
| `arg-bolts-4-n-640` | 640 | 0.0001 | works — 0.7715 | yes |
| `arg-bolts-4-n-640` | 640 | 0.001 | dead | yes |
| `arg-bolts-4-n-640` | 640 | **0.15** | **dead** | yes |
| `arg-bolts-4-n-640` | 480 | 0.001 | works — 0.7159 | yes |
| `plan-b3-1class` | 640 | 0.001 | works — 0.86 | n/a, single build |
| `plan-b4-7class-dead640` | 640 | 0.001 | dead | no |
| `plan-b4-7class-640` | 640 | 0.1 | works — 0.73 | no |
| `plan-b4-7class-480` | 480 | 0.1 | works | no |
| `arg-bolts-4-s-640` | 640 | 0.001 | works — 0.6549 | yes |

**Decision: `bfcccc7` and the a16 machinery.** The 7-class head was never an
8-bit problem. a8 at 640/0.1 decodes with scores to 0.73. `--a16-cls` was
credited for a recovery that the floor and resolution change explain on their
own. The one place a16 still has evidence is `arg-bolts-4-s-640` (a8 0.0313 vs
a16 0.6549 at a fixed 0.001 floor) — and that pair sits inside the poisoned band,
so it needs re-testing at 0.0001 before it counts for anything.

**Caveat carried forward:** none of the `plan-b4-7class-*` builds ships a
`best.pt` on the Pi, so shared weights across the three cannot be verified.

### `poc-v2-480` — 4 classes (nail, screw, bolt, unknown), imgsz 480, floor 0.1

Pi path: `artifacts/poc-v2-480/bench_int8_hailo_model/best.hef`
The `RESULT.md` production model (0.721 mAP50 recorded), used here as a
known-good reference for what a healthy build looks like on this bench.

| | zoom 1.00 | notes for the decision |
|---|---:|---|
| frames | 709 | quit by hand |
| detections | 3,897 | **5.5 per frame — quietest run of the session** |
| first-frame scores | unknown 0.56; nail / screw / bolt 0.00 | `unknown` is a catch-all class, firing is by design |
| infer median | 10.17 ms | 480 input |
| end-to-end | 34.11 ms / 29.3 FPS | sensor-limited |
| lens at | **0.28 m** | closest of any run |
| at training scale for 40 mm? | **yes** — tool advised 0.99, ran at 1.00 | **the only correctly framed run today** |

**This is the first run of the session framed at training scale**, and it is also
the quietest by a clear margin. The framing came from the operator holding
objects closer (lens at 0.28 m against 0.31-0.51 m elsewhere), not from the zoom
setting — `--zoom` and working distance both feed the same scale calculation.

## Flicker: the metric used all session was the wrong one

**Observed by eye on `poc-v2-480`:** *"this one does detect but not that good,
still flickering with the background and objects."* — so flicker persists even on
the published production model at correct training scale. That rules out framing
as the sole cause and prompted a look at what the preview is actually drawing.

**`camera_hailo` is not showing raw detections.** `vision.snapshot()` returns
tracked targets from `to_targets(tracks)`, and `draw()` colour-codes them by
policy state:

| colour | state | EMA confidence | robot behaviour |
|---|---|---|---|
| green | `CONFIRM` | >= 0.50, latched until below 0.25 | commits to retrieval |
| amber | `CAUTION` | 0.25 - 0.50 | slows down |
| grey | `IGNORE` | < 0.25 | ignores |

`detections_seen` in the CLI counts **all three states**. So every
detections-per-frame figure in this file — 5.5, 7.05, 11.4, 24.8 — includes grey
`IGNORE` tracks the robot would never act on. **That metric measures noise the
policy already discards, and should not be used to rank builds.**

| run | targets/frame (all states) | at training scale? |
|---|---:|---|
| `poc-v2-480` | 5.5 | **yes** |
| `plan-b4-7class-a16` | 7.05 | no |
| `plan-b3-1class` @ 1.00 | 8.6 | no |
| `plan-b3-1class` @ 0.28 | 9.2 | no |
| `plan-b4-7class-480` a8 | 11.4 | no |
| `plan-b4-7class-640` floor 0.1 | 24.8 | no |
| `plan-b4-7class-dead640` | 0 | dead |

Kept for the record, but read it as "how much the tracker was handed", not "how
good the build is". The one row that still means something on its own is the
zero: nothing reached the tracker at all.

**Decision: re-judge every build by CONFIRM-state targets only.** Grey churn is
the two-threshold latch in `policy.py` doing its job —
`CONFIRM_THRESH = 0.5`, `CAUTION_THRESH = 0.25`, `EMA_ALPHA = 0.4`,
`MAX_MISSES = 5`. On the preview, watch the green boxes: if green is stable while
grey flickers, the model is behaving as designed and today's flicker
observations do not indict it.

**The CLI does not report a CONFIRM-only count**, which is why this went
unnoticed all session. That is the one change worth making to the tool before the
next round of live runs.

## Why does 0.0001 work? — not answered, but constrained

Asked directly this session. The honest answer is that it is unknown. What the
evidence rules in and out:

**The compile session is not the variable.** Build times on `arg-bolts-4-n-640`:

| build | compiled | floor | imgsz | result |
|---|---|---:|---:|---|
| `bench_int8_hailo_model` | 09-05 16:25 | 0.001 | 640 | dead |
| `_conf015` | 09-05 16:54 | 0.15 | 640 | dead |
| `_a8` | 09-05 23:20 | 0.001 | 640 | dead |
| `_a16` | 09-06 00:08 | 0.001 | 640 | dead |
| `_480` | 09-06 00:30 | 0.001 | **480** | works — 0.7159 |
| `_conf00001` | 09-06 00:55 | **0.0001** | 640 | works — 0.7715 |

The last four are one pod session. Within it: 640/0.001 died twice, 640/0.0001
worked, 480/0.001 worked. Same host, same calibration. **The floor is causal at
640.**

Note `_conf015` is from the *earlier* session, so its death cannot be cleanly
attributed to the 0.15 floor — that datapoint is weaker than the others.

**`nms_config.json` differs in exactly one field.** Across all three floors:
`nms_iou_th` 0.7, `max_proposals_per_class` 100, `classes` 4,
`regression_length` 16, `background_removal` false — all identical. Only
`nms_scores_th` moves.

**The DFC logs show no difference.** `runs/dfc_logs/dfc_a8.log.gz` (0.001, dead)
and `dfc_v2_conf0001.log.gz` (0.0001, works) both report:

    [info] NMS structure of yolov6 (or equivalent architecture) was detected.
    [info] output_from_conv51|54|62|65|77|80_to_yolov8_nms_postprocess: Pass

Identical. The extra errors in the a8 log are that pod's package-install noise
(`python3-tk`, `libgraphviz-dev`), not compile failures.

**The constraint that matters.** Max score moved 0.909 -> 0.007 with only
`nms_scores_th` changed. A threshold selects among scores; it cannot lower the
highest one. So the value affects **how the model is built**, not what the chip
discards at inference. That is what rules out the explanation currently written
in `src/fodcv/matrix.py`.

It looks like the threshold entering a fixed-point scaling computation on the
score path. **No log confirms that. Do not record it as the cause.**

### Three ways to settle it, cheapest first

1. **Read Hailo's docs on `nms_scores_th`** — whether it is documented as
   affecting quantization rather than only filtering. Free, never checked.
2. **Map the boundary** — compile at 0.0005 and 0.005. A sharp cliff points at a
   representable-range effect; smooth degradation points elsewhere.
   ~35 min / ~$0.45 each.
3. **Raise DFC verbosity** on one compile for the per-layer Layer Noise Analysis.
   Needs a model-script directive; `model_script_patch()` is already the seam.

## Every run so far was off training scale

The operator's instinct on the a16 run — *"might need to zoom"* — is confirmed by
the tool's own geometry report on all four runs:

| run | zoom used | zoom the tool wanted for 40 mm |
|---|---:|---:|
| `plan-b3-1class` | 0.28 | 0.63 |
| `plan-b3-1class` | 1.00 | 0.54 |
| `plan-b4-7class-480` a8 | 1.00 | 0.71 |
| `plan-b4-7class-a16` | 1.00 | 0.66 |

Not one was framed so that a 40 mm fastener lands at the ~53 px training median.
Every detection count and flicker observation above was taken with objects
arriving at the model smaller than anything it trained on.

**Decision: re-run at the advised zoom before treating any of this as a verdict
on a model.** Weak scores and flicker are exactly what off-scale objects produce,
so the current numbers may be measuring the framing rather than the build.

## What that does to Plan 4

Plan 4 Step 1 predicted: *"if a 1-class head also dies at 640/0.001, class count
was never the variable."* It did not die. The test ran and did not falsify —
it points back at head capacity:

| build | backbone | classes | 640 @ floor 0.001 | what it decides |
|---|---|---:|---|---|
| `plan-b3-1class` | yolo11n | 1 | works | class count is back in play as the variable |
| `arg-bolts-4-n-640` | yolo11n | 4 | 0.0000 | 4 classes already too many for this head |
| `plan-b4-7class-dead640` | yolo11n | 7 | 0.0000 (recorded) | consistent, but untested live — do this next |
| `arg-bolts-4-s-640` | yolo11s | 4 | 0.6549 | width fixes it — argues capacity, not class count alone |

Narrow yolo11n head split across 4+ classes dies at that floor; widen the model
or narrow the task and it survives. That is `cls_layers_to_a16`'s original
reasoning, which Plan 4 was preparing to retire.

**Decision: do not retire `--a16-cls` yet.** Plan 4 Step 3's one-line change to
`matrix.py` still stands on its own (already applied on the Mac, not yet on the
Pi), but the case for deleting the a16 machinery is weaker than it looked this
morning.

Does not undo the 2026-09-06 headline: dropping the floor to 0.0001 rescued `n`
to 0.7715, measured. Both hold — the floor decides what is discarded, head
capacity decides whether the scores land above it.

**Confound:** `plan-b3-1class` is `fod-a-1`, a different dataset and training
run. Class count is not cleanly isolated here.

**Next test that would isolate it, free:** `plan-b4-7class-480` (a8) against
`plan-b4-7class-a16` (a16) — both 480, both floor 0.1, differing only in the
flag. Plan 4 records the a16 comparison as confounded because it compared
`dead640` (640) against `a16` (480); this pair is not confounded. Neither has an
`eval/` split so it is live-only, but it is already on the board.

## Zoom is a real trade-off, not a free knob

At 0.28 the object reaches the model 3.6x larger but through only 645 sensor px
(upscaled). At 1.00 it is 3.6x smaller but native 2,304 px. The unfiltered score
was **higher at 1.00** (0.86 vs 0.36), so on this model sensor detail beat
apparent size. Detections per frame were near-identical.

## Capture is a wait on the sensor, not a cost — corrected

**Supersedes an earlier reading in this file** that treated capture as ~18 ms of
overhead and concluded `n` 640 would run ~42 ms live. Three runs falsify it:

| run | infer | capture | infer+capture | total |
|---|---:|---:|---:|---:|
| 1-class 640 @ zoom 0.28 | 15.18 | 15.67 | 30.85 | 33.46 |
| 1-class 640 @ zoom 1.00 | 15.24 | 15.50 | 30.74 | 33.72 |
| 7-class 480 @ zoom 1.00 | **10.63** | **20.66** | 31.29 | 33.94 |

Inference fell 4.6 ms between runs 2 and 3 and capture rose by the same amount.
The total never moved. All three sit on ~33.3 ms, which is **1280x720 at 30 FPS**
— the camera's own cadence. `capture` is the loop blocking on the next sensor
frame, so it absorbs whatever inference does not use.

**Decision, revised:** the ~33 ms figure is the sensor, not a compute ceiling.
`arg-bolts-4-n-640` at 24.4 ms of inference leaves roughly 9 ms of slack inside a
30 FPS cadence and should hold frame rate live. The earlier note that only `n`
480 could fit was wrong. `s` at 50.2 ms still exceeds 33.3 ms and still cannot
hold 30 FPS.

Open question this does not answer: whether the robot loop wants more than 30 FPS
from the sensor. If it does, the camera configuration is the thing to change,
not the model.

## Geometry the tool reported

At zoom 1.00, lens at 0.51 m: a ~74 mm object is at training scale. For 40 mm
fasteners it advised `--zoom 0.54`, or holding at 0.28 m. At zoom 0.28, lens at
0.44 m: ~18 mm is at training scale; for 40 mm, `--zoom 0.63` or ~0.99 m.
Neither run was framed at training scale for a 40 mm fastener.
