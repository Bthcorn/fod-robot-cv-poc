# Tennis-ball pivot — detailed migration plan

Written 2026-11-04 from a cited cross-repo investigation (three read-only
explorers over cv-poc, the `fod-robot` harness, the `fodnav` library and the
esp32 firmware). Companion to `docs/pivot-cross-repo-impact.md` (the verdict) and
ADR-0008 (the decision). This file is the sequenced work list.

**Goal.** Make the shipping detector the tennis 480 model instead of the
arg-bolts fastener model, retire the magnet, and leave the codebase and PRD
coherent for a tennis-ball target. The research thesis (OPEN-2) and the collector
mechanism (OPEN-1) are deliberately not settled here.

Legend: **A** = mechanical swap, safe once the model ships. **B** = collection-model
rework, blocked on OPEN-1 (the collector). **GATE** = waits on the board run (D5).

---

## Two blockers the first pass missed

1. **A pointer flip alone does not boot the robot.** The tennis artifact
   `artifacts/tennis-ball-1-neg-480/` has no `run.json`, and `runtime/vision.py:75-77`
   asserts it exists before the camera opens (`docs/INTEGRATION.md:80-82` says
   Vision refuses to start without it). So `fodcv-hailo-camera` fails at boot
   today, which means **this blocks the board run itself, not just the deploy
   flip.** Fix by hand-writing `run.json`, not by `fodcv-migrate` (migrate reads a
   missing `exports.json` as `{}` and would clobber the hailo export entry —
   `manifest.py:40-41`).

2. **Nav is not yet connected to the firmware, with or without the pivot.** The nav
   library speaks protocol **v1** (`V` velocity plus encoders; `esp32.py:72`
   `PROTO_VERSION = 1`, handshake raises on mismatch at `esp32.py:750-753`). The
   firmware the harness *pins* (submodule on `dev`) speaks **v2** (`P` duty only,
   no encoders; `protocol.h:6` `kProtoVersion = 2`, rejects `V` at `main.cpp:41-44`).
   But this is no longer a design decision: Teemy shipped a nav-compatible **v1**
   firmware on branch `v1-encoders` (`9dc726d`, 2026-10-24; fodnav PR #9), carrying
   both `protocol.md` amendments. The harness just points at the wrong branch. This
   is independent of FOD-vs-tennis and gates any real-robot integration. Tracked as
   OPEN-3; see Phase 6.

---

## Phase 0 — Unblock the board run (cv-poc, do now, no gate)

| Step | File:line | Change |
|---|---|---|
| Hand-write the tennis `run.json` | `artifacts/tennis-ball-1-neg-480/run.json` | create `{"classes": {"0": "tennis_ball"}, "dataset": "tennis-ball-1-neg", ...}` so Vision boots (`vision.py:75-77`). |
| Land D3 (safe now, not gated) | `runtime/policy.py:73` | `tennis_ball` `REPORT` → `PICK`. Only one test asserts the old value (`tests/test_datasets.py:271`); flip it to `PICK`. No cv-poc runtime behaviour changes from this (verified: `in_zone` filters on `state=="CONFIRM"` only, `vision.py:209`). |
| De-magnet the policy comments | `runtime/policy.py:17-19,63-67` ; `research/datasets.py:182-185` | rewrite the "ferrous, the magnet lifts it" rationale to the mechanism-agnostic `PICK`. |

## Phase 1 — Board run (operator; the GATE)

`rsync` the artifact (now with `run.json`) to the Pi, then
`fodcv-hailo-camera --hef artifacts/tennis-ball-1-neg-480/bench_int8_hailo_model/best.hef --preview`
with a real ball. Must answer: does the `.hef` **load and decode** (precedent
`bfcccc7` decoded nothing); false positives on indoor non-court backgrounds
(ADR-0006); the 33 ms frame budget. Pass unlocks Phase 2.

## Phase 2 — Flip the shipping line (cv-poc, GATE on Phase 1)

| Step | File:line | Change |
|---|---|---|
| Move the pointers | `paths.py:32,33,43` | `CURRENT_RUN`/`CURRENT_DATASET`/`DEPLOY_RUN` → `tennis-ball-1-neg-480` / `tennis-ball-1-neg`. They move as a pair (`paths.py:26-31`). |
| Fix the deploy HEF path | `paths.py:48` | drop the arg-bolts-only `_conf00001` suffix; tennis ships on the canonical `bench_int8_hailo_model/best.hef`. |
| Fix the stale guard comment | `paths.py:44-47` | the comment says the canonical bench name must never be the deploy target; tennis makes it the target, so rewrite it or the next reader "fixes" the path. |
| Ship the eval split | `artifacts/tennis-ball-1-neg-480/eval/` | needed by the Pi bench (`bench/pi.py:357`) and the release tar (`scripts/release.sh:59` requires `<run>/run.json`). |
| Re-release the wheel | — | fod-vision is consumed downstream as a wheel; editing installed site-packages changes nothing. Cut a new `fod-vision` tag so the harness can bump its pin. |

## Phase 3 — Make the geometry readout honest (cv-poc, not board-gated; value TBD gated on OPEN-1)

| Step | File:line | Change |
|---|---|---|
| Recompute the training scale | `runtime/vision.py:56` (`TRAIN_SCALE = 0.111`) | recompute from the tennis labels, else the on-screen "at training scale" readout tells the operator to move a 67 mm ball to ~11 cm while ADR-0006 mounts at 0.3 m. Update the pin in `tests/test_camera_geometry.py:17-23` and the text in `camera_hailo.py:58`. |
| Fix the held-object default | `cli/camera_hailo.py:243-244` | `--object-mm` default `40.0` → `67` (tennis diameter, ADR-0006). |
| robot_stub geometry + prose | `cli/robot_stub.py:6,10,15,25-26,37-41,57` | docstrings "screw" → "ball" now; `CAM_TO_DRUM_M = 0.22` and its 1.47 s hold-off become a collection-geometry parameter whose **value is OPEN-1**. |

## Phase 4 — Harness model-swap (fod-robot, class A, after the wheel re-release)

| Step | File:line | Change |
|---|---|---|
| Bump the vision pin | `pyproject.toml:16`, `uv.lock` | `fod-vision` to the new tennis tag. |
| Swap the sim bundle | `sim_bundle/artifacts/.../run.json`, `nms_config.json`, `best.hef` stub, dir name | classes `4→1`, `image_dims 640→480`, taxonomy `{0: tennis_ball}`. Note: these sidecars are **dead to the harness** (nothing reads them; classes reach the harness via `fodcv class_names(hef)`, `sim_vendor/hailo_platform/__init__.py:62`), so this is for honesty, and the live lever is the hef sidecar. |
| Rename the seeded class | `scenarios/*.yaml` (`cls: screw`), `sim/world.py:40`, `scenario.py:108`, mock/contract fixtures | `screw` → `tennis_ball`. |
| Resize the detector recall | `sim/detector.py:38-42` | the recall curve is sized to a 40 mm screw; a ~67 mm ball changes recall and the profile tests (`tests/mock/test_mock_scenarios.py:66-73`) depend on it. |
| Fix the class-flip map | `sim/detector.py:144` | the `{screw↔bolt}` flip is a no-op for one class but still runs `rng.choice` eagerly, shifting seeded results. |

## Phase 5 — Harness collection-model rework (fod-robot, class B, BLOCKED on OPEN-1)

The magnet-drum collection is baked into the simulator and evaluator. Once the
collector is chosen, generalise it rather than find-and-replace:

- **Model the collector as one structure**, a capture-region predicate plus its
  geometry, not a magnet boolean threaded through `sim/world.py:73,83-95`,
  `sim/simrobot.py:123-124,130-131,147,251`, and `mock.py:142`. Today "collected"
  is `magnet on AND inside band AND inside half-width` (`world.py:83-95`).
- `evaluate.py:194-203` `drum_miss` → a collector-miss keyed on the new geometry.
  `picked`/`not_picked` survive unchanged (they only read `picked_t`,
  `evaluate.py:184-191`).
- Rename the `drum` FSM phase and the `picked`/`min_drum_lateral_m` assertions in
  `tests/mock/test_mock_scenarios.py:33,34,50`.
- Nav config, not code: `mission.magnet_on: false` (`_config/nav.yaml:47`) disables
  the magnet with no code change; `swath_source` (`nav.yaml:100`, default `drum`)
  and the servo blind-leg geometry (`servo.py:212-215`, `drum.offset_x_m`) must be
  re-pointed at the new collector once OPEN-1 lands.

## Phase 6 — Connect nav to the v1 firmware (cross-repo, OPEN-3, independent of the pivot)

The v1-vs-v2 question is settled: a nav-compatible v1 firmware already exists
(`Teemy17/fod-robot-esp32` branch `v1-encoders`, `9dc726d`, 2026-10-24; recorded
in fodnav PR #9). No adapter or firmware redesign is needed — the harness just
pins the wrong branch. Remaining work:

- **Re-point the harness firmware submodule** `dev` → `v1-encoders` (fetch it
  first; clean the currently-dirty submodule tree). Bump the firmware pin.
- **Bench-run `link/esp32.py` against his firmware** — `fodnav-teleop` over USB,
  wheels lifted: the handshake, the watchdog, the e-stop. Nothing has ever run
  nav and his code together. Needs the ESP32.
- **Land Teemy's measured constants** in `config/robot.yaml` (`HARDWARE.md` §0,
  ~3 numbers) and have him rebuild with `-D ROBOT_CONSTANTS_MEASURED=1`. Until
  then his `I` line reports `0.000` and `Esp32Link.verify` correctly refuses it.
- **Get Teemy's explicit sign-off on `docs/protocol.md`** — only the two
  amendments were checked against his build, not the whole document.

Owner: team plus Teemy plus advisor. The bench run and the hardware steps
predate the pivot and are the true integration blocker.

---

## Open calls carried out of this plan

- **OPEN-1 collector mechanism** (gates Phase 5, the robot_stub value, and nav's
  servo/coverage geometry). Operator plus firmware plus advisor.
- **OPEN-2 thesis redesign.** Advisor.
- **OPEN-3 nav↔firmware connection.** Not a design decision anymore (v1 firmware
  exists on `v1-encoders`); remaining work is the submodule re-point plus a bench
  run and Teemy's sign-off. See Phase 6. Team plus Teemy plus advisor.
- **Run-id naming.** Default applied: use the artifacts name `tennis-ball-1-neg-480`
  (the code reads that, not the training name `tennis-ball-1-neg-n-480`). Tell me
  to use the training name instead if you prefer symmetry over what the code reads.
- **"FOD" in package and env names** (`fodcv`, `fodrobot`, `FODROBOT_*`, repo
  names). A separate rename decision, deferred.
