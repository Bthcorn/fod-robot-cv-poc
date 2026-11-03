# Tennis-ball `.hef`: every failure hit, so the next run is one shot

Written 2026-11-03 after a session that got the compile to **place-and-route and
no further**. Everything below is a wall actually hit this session, in order,
with the fix. Read it once; then the compile is copy-paste.

The winner to compile is **`tennis-ball-1-neg-n-480`** (`docs/tennis-ball-experiment.md`
§ Stage 2). `best.pt` is on the Mac under `runs/tennis-ball-1-neg-n-480/weights/`
(gitignored). The DFC wheel is on the Mac at
`~/Downloads/hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl`
(sha256 `f539ebb5…`, matches the pin) and mirrored on the operator's Drive,
file id `1lpDHRQzMNNM8k3LSgVlx6d6Z56UQvEcg`.

---

## The walls, in the order they appear

### 0. Provisioning — capacity is the first and worst wall
- The **A40 is not reliably re-gettable**. Across this session a single create
  succeeded out of dozens of attempts; the rest returned *"no longer any
  instances available"*. `get-capacity` reported the A40 unavailable seconds
  before a create on it succeeded — **trust `create-pod`, not the capacity
  query** (already in `docs/autorun-runpod-e2e.md`).
- **US was fully dry** this session: 8 cards × SECURE and 10 × COMMUNITY across
  all 16 US data centers, every one refused. **EU-SE-1 was the only stock.**
  That matters because of wall #2 (uplink).
- Fix: iterate `create-pod` across a *list* of cards and both clouds, no region
  constraint, stop at the first that lands. A40 and A6000 both work and were
  the only secures that ever landed; both bill **$0.59/hr**.

### 1. SSH key — the registered key is the `_runpod` one
- The account's registered public key is `…qvOmI runpod`, which matches
  **`~/.ssh/id_ed25519_runpod`**, *not* `~/.ssh/id_ed25519`. Using the latter
  gets `Permission denied`.
- The direct-SSH ip/port is unique per pod and only appears once the runtime is
  up (~30–40 s after create). Read it from `get-pod` →
  `runtime.ports[tcp]` / `ssh.direct`, not from any earlier value.

### 2. Uplink — the Mac→EU-SE-1 route is ~0.5 Mbps, uploads die
- A 5 MB `scp` probe to the EU pod **timed out in 90 s**; a 43 MB tarball got
  5.7 MB through in 300 s. A US pod (US-KS-2) earlier the same session moved
  85 MB in seconds, so this is the transatlantic route, not home upstream alone.
- **Consequence: the 500 MB DFC wheel cannot be pushed from the Mac to an EU
  pod.** But a pod's *downlink* from Google Drive is fast regardless of region
  (measured ~50 MB/s, wheel in ~10 s).
- Fix: **let the pod fetch the wheel from Drive**, never upload it from the Mac:
  ```bash
  mkdir -p ~/.cache/hailo-compile
  uvx --from gdown gdown "1lpDHRQzMNNM8k3LSgVlx6d6Z56UQvEcg" \
    -O ~/.cache/hailo-compile/hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl
  sha256sum ~/.cache/hailo-compile/hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl
  # must be f539ebb5997149ec68ca443a547196a03d28c624fbb072fdcd22a7d37fad9fb1
  ```
  `~/.cache/hailo-compile/` is the first path `hailo-compile.sh` searches, so no
  `HEF_WHEEL` or `HEF_WHEEL_URL` is needed once it is there.

### 3. The small Mac-only payload still has to go up (slowly)
What only exists on the Mac and the pod genuinely needs:
- `best.pt` (the winner) → placed as `artifacts/tennis-ball-1-neg-480/best.pt`
- calibration **images** from `data/tennis-ball-1-neg/images/train`
- the code (`src/fodcv`, `pyproject.toml`, `uv.lock`, `hailo-compile/`, **`README.md`** — see #5)

Minimise it:
- **Skip the `val` split.** It is only for the later Pi eval, not for producing
  the `.hef`. Dropping it, and placing `best.pt` directly, lets you **skip
  `fodcv-migrate`** entirely (migrate's only hard requirement is the val split).
- **Ship only the calibration images you will use.** `HEF_FRACTION=1.0` over the
  **sorted head of 1,485** images (= fraction 0.5 of the 2,970 train, and above
  Hailo's 1,024 floor) is what this session used. Halves the upload (34→18 MB).
  ```bash
  ls data/tennis-ball-1-neg/images/train | sort | head -1485 > /tmp/calib.list
  ```
- `data.yaml` with **both splits pointed at `images/train`** (no val shipped):
  ```
  train: images/train
  val: images/train
  names:
    0: tennis_ball
  ```
- Transfer **one tarball**, not a directory of 1,485 files: rsync of thousands
  of tiny files stalls on this link; a single stream resumes. macOS rsync
  rejects `--info=stats1` and `--append-verify` — use plain `--partial`, in a
  retry loop (the link drops), detached so it survives tool timeouts.
- `tar xzf` on the pod prints harmless `Ignoring unknown extended header keyword
  'LIBARCHIVE.xattr.com.apple.provenance'` — macOS xattrs, ignore.

### 4. Pod base image has no `uv`/`rsync`; needs `python3.10`
- Image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`: default
  `python3` is 3.11, but the DFC venv is built from **`python3.10`**, which is
  present on 22.04 only after `apt install python3.10 python3.10-venv`. Also
  install `sudo` (the script calls it) and `rsync`.
- For the compile you do **not** need `uv sync` at all — `hailo-compile.sh`
  builds its own `python3.10` venv and `pip install -e .`'s the package into it.

### 5. `pip install -e .` fails without `README.md`
- `pyproject.toml` has `readme = "README.md"`; the editable install dies with
  `OSError: Readme file does not exist: README.md` **after** the 500 MB wheel +
  TensorFlow are already installed — an expensive place to fail.
- Fix: **include `README.md` in the payload** (a placeholder is fine for a
  throwaway compile host; its content is only the long-description).

### 6. Place-and-route dies on locale — the costly one
- After QAT (4 epochs, ~6 min on GPU) and Layer Noise Analysis (~2 min) **both
  succeed**, the compile throws
  `BackendAllocatorException: Compilation failed: locale::facet::_S_create_c_locale name not valid`.
  The minimal container has an invalid `LANG`/`LC_*`; the DFC's C++ backend
  builds a `std::locale` from it and throws.
- **Now fixed in the script** (`hailo-compile.sh` exports `LANG=C.UTF-8
  LC_ALL=C.UTF-8`). If running the export by hand, set them yourself. Hitting
  this wastes a whole QAT, so it is the single most important one to pre-empt.

### 7. Non-fatal noise (do not chase these)
- DFC post-install warns `python3-tk / libgraphviz-dev / python3.10-dev not
  found`, then prints `DFC ok` and continues. The script now apt-installs them
  so the log is clean, but they never blocked the compile.
- `fodcv-export` warns `No labels found in …/labels/train` — **benign**, INT8
  calibration and QAT use images only, which is why the payload ships no labels.
- Ultralytics warns `1485 calibration images, Hailo recommends ≥1,024` —
  informational; we are above the floor.
- `USER` unset under `set -u` / `KeyError: 'USER'` in place-and-route — already
  defaulted to `root` at the top of `hailo-compile.sh`.

### 8. Timing — budget, and get it right first time
- Every `fodcv-export` invocation **re-runs QAT from scratch** (~6 min) + LNA
  (~2 min) before place-and-route (~15 min for `yolo11n` at 480, 2 contexts).
  A failure that reaches export costs ~8 min before you learn P&R's verdict —
  which is why #5 and #6 are pre-empted, not discovered.
- Stage-2 **training** on an A6000 was ~25 s/epoch, not the handoff's optimistic
  "<10 min": budget ~45 min for both 320+480 runs if you ever re-train.

---

## The one-shot sequence (EU pod, wheel via Drive)

```bash
# 0. Mac: provision — iterate cards/clouds until one lands (A40/A6000 = $0.59/hr)
#    get the direct SSH from get-pod runtime.ports[tcp]; key is id_ed25519_runpod.

# 1. Mac: build the lean payload (≈18 MB)
cd ~/Projects/GraduateProject/cv-poc
rm -rf /tmp/hefstage && mkdir -p /tmp/hefstage/{src,hailo-compile,artifacts/tennis-ball-1-neg-480,data/tennis-ball-1-neg/images/train}
cp pyproject.toml uv.lock README.md /tmp/hefstage/            # README.md is required (#5)
cp -R src/fodcv /tmp/hefstage/src/
cp hailo-compile/hailo-compile.sh /tmp/hefstage/hailo-compile/
cp runs/tennis-ball-1-neg-n-480/weights/best.pt /tmp/hefstage/artifacts/tennis-ball-1-neg-480/best.pt
ls data/tennis-ball-1-neg/images/train | sort | head -1485 > /tmp/calib.list
(cd data/tennis-ball-1-neg/images/train && rsync -a --files-from=/tmp/calib.list ./ /tmp/hefstage/data/tennis-ball-1-neg/images/train/)
printf 'train: images/train\nval: images/train\nnames:\n  0: tennis_ball\n' > /tmp/hefstage/data/tennis-ball-1-neg/data.yaml
tar czf /tmp/hef-lean.tar.gz -C /tmp/hefstage .

# 2. Mac: push the payload (resumable retry loop, detached) — see #3

# 3. Pod: toolchain + wheel from Drive (fast downlink) + verify sha — see #2, #4
apt-get update -qq && apt-get install -y -qq sudo python3.10 python3.10-venv rsync locales
curl -LsSf https://astral.sh/uv/install.sh | sh ; export PATH="$HOME/.local/bin:$PATH"
mkdir -p ~/.cache/hailo-compile
uvx --from gdown gdown "1lpDHRQzMNNM8k3LSgVlx6d6Z56UQvEcg" \
  -O ~/.cache/hailo-compile/hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl

# 4. Pod: extract + compile (locale + USER now handled inside the script)
mkdir -p /root/cv-poc && cd /root/cv-poc && tar xzf /root/hef-lean.tar.gz
HEF_RUN=tennis-ball-1-neg-480 HEF_DATASET=tennis-ball-1-neg \
HEF_IMGSZ=480 HEF_CONF=0.0001 HEF_FRACTION=1.0 HEF_GPU=1 \
bash hailo-compile/hailo-compile.sh          # no --a16-cls: 1-class head

# 5. Watch for: "DFC ok" · PhysicalDevice(...GPU:0...) · QAT 4/4 · LNA done ·
#    Iteration #N contexts · then artifacts/.../bench_int8_hailo_model/best.hef

# 6. Mac: pull the result, then terminate the pod
rsync -a 'pod:/root/cv-poc/artifacts/tennis-ball-1-neg-480/' artifacts/tennis-ball-1-neg-480/
```

`HEF_IMGSZ` **must** equal the training size (480) — `nms_config.json` carries
it and `Vision` reads it off the `.hef`; a mismatch letterboxes in silence.
`HEF_CONF=0.0001`, never `0.001` (RESULT.md §13).

## What a successful compile still does not prove
A pod has no Hailo device. It proves the `.hef` **built**, not that it loads or
detects anything (`bfcccc7`: a clean compile that decoded nothing), and says
nothing about latency. The only honest verdict is the Pi + Hailo-8 with a ball
on the floor (`docs/tennis-ball-experiment.md` § Stage 3, ADR-0006).
