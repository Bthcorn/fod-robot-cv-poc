# Autorun: the arg-bolts backbone sweep, end to end

A runbook for an agent to launch `scripts/train_roboflow.sh`, watch it for the
~30 hours it takes, and hand back a decision. Written to be handed to a fresh
agent with no memory of the session that produced it.

**Point an agent at this file and say: "run docs/autorun-argbolts.md".**

Compiling the `.hef` on the same rented box, rather than stopping at
`best.pt`, is `docs/autorun-runpod-e2e.md` — it wraps this file rather than
replacing it.

The sweep is four backbones on one dataset, one axis moving. What it exists to
answer, and the numbers it is read against, are in
[the question](#the-question-this-answers) below — an agent that skips that
section will report a number without knowing whether it is good.

---

## The question this answers

Prep used to hand Ultralytics 26% garbage boxes. `_boxed_label` in
`research/dataset.py` repairs them; the commit message on `c56e321` has the
mechanism. What matters here is the consequence:

The old labels imposed a hard per-class AP50 ceiling. **These are the numbers
every result below is read against:**

| class | old AP50 ceiling | old measured | headroom the repair opened |
|---|---:|---:|---:|
| bolt | 0.839 | 0.63 | small |
| nut | 0.467 | 0.40 | **large** |
| screw | 0.897 | 0.34 | none — screw is genuinely hard |
| washer | 0.419 | 0.28 | **large** |
| **mAP50** | **0.655** | **0.6442** | |

The model was at 98% of what the labels allowed. So every earlier comparison on
this dataset — 60 vs 100 epochs, 480 vs 640 — was reading noise against a
ceiling, and none of those nulls are safe any more.

Two things this sweep decides:

1. **Did the repair alone fix it?** If `n` lands near 0.85, capacity was never
   the limit and runs `s`/`m`/`26n` answer a question nobody has.
2. **If not, is capacity the limit at all?** `m` is the probe. If `m` barely
   beats `n`, the remaining gap is data — screw is 8.6% of instances at a
   39.8 px median — and no deployable backbone closes it.

`m` **cannot ship.** 10.5× yolo11n's FLOPs against ~1.2–1.8× of Hailo headroom
at 640. It is a probe, never a candidate. Do not migrate it.

---

## Preconditions

Verify all of these before launching. Each one has cost a run before.

```bash
git log --oneline -1                       # expect c56e321 or a descendant
uv run pytest -q                           # expect 151 passed
```

**CUDA is real.** PyPI's default torch wheel is CPU-only on native Windows, and
a CPU sweep of this size is weeks:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

False means `uv pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision`.
Re-running `uv sync` reverts it. The script's own preflight catches this, but
catching it here saves a restart.

**The export is on disk**, at whatever `datasets.py` says — do not hardcode the
path, read it:

```bash
uv run python -c "from fodcv.research.datasets import source; print(source('arg-bolts-4').source_dir)"
```

Missing means download the YOLOv11 export of
`https://universe.roboflow.com/mvtec-uihve/arg_bolts_fv` (v3) and unzip it there.

**Disk.** Four runs of weights, plots and val batches. Budget 15 GB free.

**VRAM.** `yolo11m` at `imgsz=640 batch=16` is the cell that OOMs first. If the
card is under ~12 GB, set `BATCH=8` **for the whole sweep** before launching —
never for one run, or the comparison is confounded and the sweep is wasted.

---

## If this is a rented pod

The sweep is ~30 h and `m` is ~20 h of it, which is what makes renting a 4090
cheaper than occupying a desktop for a day. Three things about that:

**The Hailo wheel is not part of training.** `.hef` compilation is a separate
stage, and `pyproject.toml` carries no Hailo dependency at all — `uv sync
--extra research` installs nothing related to it. An agent running *this* file
that goes hunting for the DFC is solving a problem it does not have.

Where that stage runs depends on which runbook is in play. Under this file it
happens afterwards on the Mac or WSL, via `hailo-compile/hailo-compile.sh`,
which mounts the wheel from the host's `~/Downloads`. Under
`docs/autorun-runpod-e2e.md` it happens on this same pod at its Stage 4, and the
wheel is fetched there — still not here, and still not before the sweep is
launched.

**Getting the dataset there — two ways, and the fast one is not the obvious
one.** The export is ~789 MB. Whichever path is used, it must end up unzipped at
the directory `datasets.py` names, because that is what prep resolves:

```bash
uv run python -c "from fodcv.research.datasets import source; print(source('arg-bolts-4').source_dir)"
```

*Fast — the pod pulls it.* The bottleneck in the other path is the human's home
uplink (~3.5 min at 30 Mbps); a pod on datacenter bandwidth fetches the same
file in seconds. Roboflow's dataset page gives a `curl` line under Download
Dataset → YOLOv11 → show download code → Terminal. **The key in that line is a
credential: take it as an env var, never write it into a file in this repo, a
shell history that gets committed, or a log.**

```bash
mkdir -p ~/Downloads/ARG_Bolts_FV.v3i.yolov11
curl -fL "$ROBOFLOW_EXPORT_URL" -o /tmp/rf.zip   # URL carries the key; keep it in the env
unzip -q /tmp/rf.zip -d ~/Downloads/ARG_Bolts_FV.v3i.yolov11 && rm /tmp/rf.zip
```

*Simple — push it from the Mac.* No account on the pod, no key, no link that can
expire. Costs one home upload:

```bash
scp ~/Downloads/ARG_Bolts_FV.v3i.yolov11.zip pod:/workspace/
# on the pod -- ~/Downloads is what datasets.py resolves via expanduser()
mkdir -p ~/Downloads && unzip -q /workspace/ARG_Bolts_FV.v3i.yolov11.zip \
  -d ~/Downloads/ARG_Bolts_FV.v3i.yolov11
```

Either way, verify before launching — a half-unzipped export fails at epoch 0 of
the first run, ~30 h into nothing:

```bash
ls ~/Downloads/ARG_Bolts_FV.v3i.yolov11        # expect train valid test data.yaml
find ~/Downloads/ARG_Bolts_FV.v3i.yolov11 -name '*.jpg' | wc -l   # expect 12678
```

Put it on a persistent network volume, not the pod's ephemeral disk — the pod
gets destroyed when the sweep ends. Do not upload `data/arg-bolts-4/` instead:
prep is deterministic, so rebuilding from the zip on the pod gives a
byte-identical split, and the zip is the file that already exists.

**Only `best.pt` comes back** — 5.2 MB for yolo11n, ~19 MB for s, ~40 MB for m.
Everything else in `runs/` is scratch. Copy each to the same path on the Mac and
migrate there, where the export toolchain and the Pi live.

On Linux the default PyPI torch is already the CUDA build, so the cu-index step
in Preconditions does not apply and cannot be undone by `uv sync`.

## Launch

Background it and keep the log. The run outlives any agent session.

```bash
mkdir -p logs
nohup bash scripts/train_roboflow.sh > logs/argbolts-sweep.log 2>&1 &
echo $! > logs/argbolts-sweep.pid
```

Confirm it got past the preflight and into epoch 1 before trusting it — the
first two failure modes both happen in the first 90 seconds:

```bash
sleep 90 && tail -20 logs/argbolts-sweep.log
```

The script is resumable: a run is skipped once its `results.csv` holds `EPOCHS`
rows. A crash costs the current run and nothing before it. **To resume after
any failure, re-run the same launch command** — do not pass `--resume`, and do
not delete finished run directories.

---

## Monitor

Poll every 20–30 minutes. Not more often: an epoch on this dataset takes
minutes, and a tighter loop burns tokens to watch a number that has not moved.

Each tick, for the run currently training:

```bash
# find, not a glob: zsh aborts the whole command on an unmatched pattern, and
# before the first run exists there is nothing to match.
find runs -maxdepth 1 -name 'arg-bolts-4-*-640' -type d | sort | while read -r d; do
  [ -f "$d/results.csv" ] || continue
  printf "%-22s %3s epochs  last: %s\n" "$(basename "$d")" \
    "$(( $(grep -c '' "$d/results.csv") - 1 ))" "$(tail -1 "$d/results.csv" | cut -d, -f1,2,8,9)"
done
ps -p "$(cat logs/argbolts-sweep.pid)" > /dev/null && echo "alive" || echo "DEAD"
```

Prints `<run> <epochs done> last: epoch,time,mAP50,mAP50-95`. Empty output before
the first epoch lands is correct, not a failure.

`results.csv` columns worth reading: `epoch`, `time` (cumulative seconds),
`metrics/mAP50(B)`, `metrics/mAP50-95(B)`, `train/box_loss`, `train/cls_loss`.

**Healthy** is: row count increasing, `time` per epoch roughly constant, losses
falling, mAP50 rising then flattening. ETA for the run is
`(EPOCHS - epochs_done) * time_per_epoch`.

**Report the real total after epoch 1, not the estimate in this file.** Every
duration here is extrapolated from `yolo11n` on other hardware; one measured
epoch beats all of it, and on rented time it is the number that decides whether
`m` is worth paying for.

**On a rented box, check the GPU is the thing working.** Every image in this
export is already 640x640, so at `imgsz=640` the dataloader only decodes and
augments — but on a fast card with few vCPUs that is still enough to starve it,
and a starved GPU is rent paid for idle silicon:

```bash
nvidia-smi --query-gpu=utilization.gpu --format=csv -l 5   # ctrl-c after a minute
```

Sustained below ~70% during training means dataloader-bound. Add
`--set cache=ram` to `train()` and relaunch: the decoded train split is 10.9 GB
(8,891 x 1.23 MB), 14.0 GB with val, so it needs a box with ~24 GB RAM free.
Caching is pure I/O and changes no gradients, so unlike batch size it does not
invalidate a comparison — but apply it to every run anyway.

### What to do when it is not healthy

| Symptom | Cause | Action |
|---|---|---|
| Process DEAD, log ends in `CUDA out of memory` | `m` at batch 16 | Set `BATCH=8`, delete **all four** run dirs, relaunch. A sweep with one cell at a different batch is not a comparison. |
| Process DEAD, no error at the tail | OOM-killer, or the box slept | Re-run the launch command. It resumes at the run that died. |
| Row count static ≥ 45 min, process alive | Hung dataloader (seen on native Windows) | Kill it, relaunch with `--set workers=4` added to `train()` in the script. Note it in the report. |
| `nan` in any loss column | Diverged | Do not restart blind. Stop, report the run and its `args.yaml`. |
| `time` per epoch climbing steadily | Thermal throttle, or another job on the card | Report it. Do not intervene mid-run; note it as a caveat on that run's number. |
| Log says `no CUDA` | `uv sync` reverted the torch wheel | Reinstall the cu-index wheel, relaunch. |

**Never** change `imgsz`, `epochs`, `batch`, `seed`, or the dataset mid-sweep to
rescue a run. A rescued run is not comparable to the other three, which makes
the whole sweep unreadable. Restart the sweep or report the failure.

---

## After each run finishes

Record the numbers before the next run overwrites your attention. Per-class AP
is the point — mAP50 alone will not tell you whether the repair worked, because
nut and washer are where the ceiling was:

```bash
uv run python -c "
import sys
from ultralytics import YOLO
run = sys.argv[1]
r = YOLO(f'runs/{run}/weights/best.pt').val(data='data/arg-bolts-4/data.yaml', imgsz=640, verbose=False)
print(run, 'mAP50', round(r.box.map50, 4), 'mAP50-95', round(r.box.map, 4))
# ap50 is indexed by the classes actually present, not by names -- zip it
# through ap_class_index or a class with no instances shifts every label.
for i, c in enumerate(r.ap_class_index):
    print(f'  {r.names[c]:8s} AP50 {r.box.ap50[i]:.4f}')
" arg-bolts-4-n-640
```

Verified against `runs/train_fod-a-3k`. It leaves a `runs/detect/val*` directory
behind; that is expected and nothing reads it.

Append each to a table in the final report. Read every number against the
dataset page's published 90.6% and **not** against 0.6414 — different split and
different labels. Say which split anywhere it is quoted.

### Stop early when the answer is already in

- **`n` ≥ ~0.85** — the repair was the whole story. Report it, stop the sweep,
  and say plainly that `s`/`m`/`26n` are now unnecessary. Do not run 25 hours
  to confirm a conclusion already reached.
- **`m` beats `n` by < ~0.02** — capacity is not the limit. Skip `26n` unless
  it is already running; report that the remaining gap is data, and point at
  screw's 8.6% frequency and 39.8 px median size.

Both of these save real money. Take them.

---

## Finish

1. **Migrate only deployable runs.** `n` always; `s` if it won on accuracy.
   Never `m`. `26n` only with the caveat below.

   ```bash
   uv run fodcv-migrate --from runs/arg-bolts-4-n-640 --run arg-bolts-4-n-640 --dataset arg-bolts-4
   ```

   `fodcv-migrate` needs `data/arg-bolts-4/` prepared on the same machine. If
   the sweep ran on a rented box, copy back only `weights/best.pt` (~6–40 MB) to
   the same path on the Mac and migrate there, where the export toolchain and
   the Pi already live.

2. **Latency, for `s` only.** Training cannot answer whether it fits a 33 ms
   frame; this is the other half of the decision and `s` is not a candidate
   without it. Runs on the Pi, not the training box.

   ```bash
   uv run fodcv-export --run arg-bolts-4-s-640 --formats hailo
   uv run fodcv-hailo-camera --hef artifacts/arg-bolts-4-s-640/bench_int8_hailo_model/best.hef \
     --preview --frames 0 --out runs/camera-arg-bolts-4-s-640
   ```

   `s` wins on accuracy but misses 30 FPS → the next run is
   `IMGSZ=480 bash scripts/train_roboflow.sh s`. Do not launch it automatically;
   report the trade and let a human choose.

3. **Report.** Hand back: the per-class table for every run that finished,
   which runs were skipped and why, every caveat noted during monitoring, and
   one recommendation with the number behind it. Attach
   `logs/argbolts-sweep.log`.

   If `26n` won, say so *and* say it is not deployable yet: the Hailo path is
   built on `nms_postprocess(meta_arch=yolov8)` and YOLO26's NMS-free head is
   unproven against it. A win there is a reason to investigate, not to ship.

---

## Things this run cannot tell you

State these in the report rather than letting a reader assume otherwise.

- **Neither number predicts the arena.** This dataset is studio hardware on
  plates and carpet; the arena is a wooden floor at a grazing angle. `imgsz=640`
  was worth +0.003 mAP on this val and +77% detections on real Pi frames — a
  benchmark moving opposite to the deployment.
- **The val split overlaps its own train split by 14.5%, at scene level.** Same
  plate, same pose, fasteners rearranged. That is what makes it comparable to
  the published 90.6% and what makes it not a generalisation result.
- **Latency is not measured by training.** A model that wins here and misses
  the frame budget has not won.
