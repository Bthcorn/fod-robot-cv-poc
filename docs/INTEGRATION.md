# Robot integration — the CV seam

For whoever owns the Pi control loop and the ESP32. You do not need to understand
the detector, the training pipeline or the export toolchain. You need one class,
four fields and one method.

The vision package is a library you import in your own process. There is no
daemon, no socket, no ROS2 node — PRD §5 puts a Python asyncio loop on the Pi with
no middleware, and this is built for exactly that.

---

## 1. Install

Into the Pi's **system** interpreter, not a venv:

```bash
sudo python3.11 -m pip install --break-system-packages \
  'fod-vision @ git+https://github.com/Bthcorn/fod-robot-cv-poc.git@v0.2.0'
```

System 3.11 because apt's `python3-picamera2` is built against it, and a 3.12 venv
is a different C ABI — `import libcamera` fails there no matter what pip does. The
repo's `.python-version` says 3.12; that is for Mac-side training work and does not
apply to you.

The install pulls `numpy` and `opencv-python` and nothing else. No torch, no
ultralytics. Both are already on the Pi via apt, so this adds effectively nothing.

**Pin the tag.** A moving `main` under a running robot is not a thing you want to
debug in week 11.

## 2. Get the model

The `.hef` is not in git. Grab the release bundle for the same tag:

```bash
gh release download v0.2.0 && tar xzf poc-v2-480-full.tar.gz
```

Three files, ~7.8 MB, and all three must stay in that layout — the class names and
the input size are read from the two JSONs, not hardcoded:

```
artifacts/poc-v2-480-full/
├── run.json                       class names + which dataset trained them
└── bench_int8_hailo_model/
    ├── best.hef                   the model
    └── nms_config.json            input size, class count, compiled threshold
```

Paths are relative to your working directory. `hailortcli parse-hef best.hef`
confirms the architecture, class count and score threshold on the board.

## 3. Prove it works, before writing any robot code

```bash
fodcv-hailo-camera --preview        # boxes on screen, focus distance, geometry readout
fodcv-robot-stub --seconds 30       # the loop below, with prints instead of motors
```

Run these first on a new board. They answer "is the camera framed and focused, and
does the chip see anything" while there is still no control loop to blame.
`--preview` needs `DISPLAY=:0`.

## 4. The loop

```python
import asyncio, time
from fodcv.runtime.vision import Vision

CAM_TO_DRUM_M = 0.22        # tape measure on the built chassis (PRD O-3)
HEF = "artifacts/poc-v2-480-full/bench_int8_hailo_model/best.hef"

with Vision(hef=HEF, lookahead=(0.5, 1.0)) as vision:
    hold_until = 0.0
    while sweeping:
        if vision.age > 0.5:
            esp32.write("STOP\n")          # stale vision is NOT a clear floor
            continue

        now = time.monotonic()
        if vision.zone_blocked():
            hold_until = now + CAM_TO_DRUM_M / V_SLOW
        esp32.write(f"SPEED {V_SLOW if now < hold_until else V_FAST}\n")

        await asyncio.sleep(1 / 20)
```

That is PRD FR-4 in full. `zone_blocked()` and `latest()` only take a brief lock,
so both are safe to call from inside the asyncio loop; capture and inference run on
their own daemon thread and never block you.

**The hold-off timer is yours, not the vision package's.** Its length is the
camera-to-drum distance over the current speed, and this package does not know your
speed. Re-arm it every frame the zone is blocked, so it measures from when the
object *left* the zone rather than when it entered.

`lookahead` is a `(lo, hi)` fraction of frame height. **The default is a
placeholder, not a measurement** — the real strip falls out of PRD O-3 (camera
height and tilt, undecided) and M-3 (FOV width `W`, lookahead distance `d`,
unmeasured). RESULT.md §8 narrows the tilt to 10–25° at 15–30 cm and no further.

## 5. The contract

**Stable. Branch on these:**

| | |
|---|---|
| `vision.zone_blocked()` | confirmed detection in the lookahead strip. FR-4's only input |
| `vision.age` | seconds since the last completed frame; `inf` before the first |
| `t.state` | `IGNORE` / `CAUTION` / `CONFIRM` — confidence smoothed over frames |
| `t.action` | `PICK` / `REPORT` / `IGNORE` — what the class means |
| `t.id` | stable while the object stays in view |
| `t.conf` | EMA confidence, 0–1 |

**Diagnostic. Log it, draw it, do not branch on it:**

| | |
|---|---|
| `t.cls` | four classes today; becomes a single `fod` after the arena dataset lands |
| `t.box`, `t.centroid` | pixels in the rotated frame |

`cls` is excluded on purpose. PRD FR-3 mandates one trained class and the arena
dataset ships that way, at which point `nail`/`screw`/`bolt` stop existing. Code
branching on `state` and `action` survives the switch untouched. The names are also
not trustworthy today — RESULT.md §6 has real screws coming back labelled `bolt`,
on 36 training instances.

`state` and `action` are two fields for a reason: a CAUTION screw and a CAUTION
shard both mean *slow down*, and only the class says which one the magnet can lift.

## 6. Failure modes

- **A dead camera raises, it does not return an empty list.** Any exception on the
  capture thread is stored and re-raised from the next `latest()` or
  `zone_blocked()`. "No debris, keep patrolling" is the one thing a broken camera
  must never look like.
- **`age` growing means the thread stopped.** You decide the tolerance; the package
  will not decide it for you. 0.5 s is a starting point.
- **`Vision(...)` as a context manager, always.** `__enter__` waits up to 30 s for
  the Hailo device to configure and raises `TimeoutError` if it does not. `__exit__`
  stops the thread and releases the camera; skipping it leaves the device claimed.
- **One process owns the Hailo.** Do not construct two `Vision` objects, and stop
  `fodcv-hailo-camera` before starting the robot.

## 7. Retuning the hysteresis (PRD M-12)

FR-4 tags the three thresholds MEASURE and M-12 names the retune on the real mount:

```python
from fodcv.runtime import policy
policy.tune(CONFIRM_THRESH=0.6, EMA_ALPHA=0.3)   # before constructing Vision
```

Tunable: `CONFIRM_THRESH`, `CAUTION_THRESH`, `EMA_ALPHA`, `MAX_MATCH_DIST`,
`MAX_MISSES`. A misspelled name raises rather than silently doing nothing.

`MAX_MISSES` is worth knowing about: it is how many frames a track survives without
a detection, and it is what stops `zone_blocked()` chattering when one frame misses.
Lower it and the speed policy gets twitchier.

## 8. Where your latency budget starts

RESULT.md §6, measured on the board, is **capture → postprocess only**:

| stage | median ms |
|---|---:|
| capture | 20.7 |
| preprocess | 1.3 |
| infer | 10.4 |
| postprocess | 1.0 |
| **total** | **33.4** |

30 FPS end to end, **camera-bound, not compute-bound** — inference is 10 ms of a
33 ms frame, so there is ~3× headroom before the chip constrains anything.

PRD M-3 wants capture → serial. The remaining terms — your poll interval, the
serial write, the ESP32 PID settle — are yours to measure. Do not re-measure the
four above; `vision.stats` carries them live if you want them per-run.

## 9. What this does not give you

- **No metres, no floor coordinates.** Boxes are pixels. Ground-plane projection
  needs a mount height and tilt this package cannot know. FR-15's homography is
  optional and unbuilt.
- **No pick point and no steering.** Collection is passive (FR-11: no gripper, no
  targeting), so nothing steers toward an object. If you find yourself wanting a
  bearing, check the requirement again.
- **No serial.** The Pi↔ESP32 line protocol in PRD §5 is entirely yours.
- **Not a general detector.** It is trained on FOD-A, a public dataset with no
  cluttered arena floor in it. See RESULT.md §7 and §11 before quoting any accuracy
  number.

## 10. Open items that block this

| | |
|---|---|
| **O-3** | camera height and tilt — undecided, and `lookahead` cannot leave placeholder status without it |
| **M-3** | FOV width `W` and lookahead distance `d` — unmeasured |
| **M-12** | hysteresis retuned on the real mount — §7 above is the mechanism, the numbers are not taken |
| **FR-13 vs FR-3** | a single-class detector cannot tell ferrous from non-ferrous, so `action` will be constant `PICK` and `REPORT` will never fire. Needs a team decision before anything branches on `action` expecting both |
