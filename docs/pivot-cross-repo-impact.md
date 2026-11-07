# Cross-repo impact of the FOD → tennis-ball pivot (CHECK-1)

Investigated 2026-11-04, after ADR-0008. Question: if the detection target
becomes the tennis ball and the magnet is retired, what breaks in the other
repos? Answer: **the firmware barely notices, the nav *library* is rename-safe by
design, and the real blast radius is the integration harness's simulated
collection + evaluation — which is magnet-drum-shaped and blocked on OPEN-1.**

## The three downstream layers (CHECK-1 "which repo is authoritative", resolved)

| Layer | Repo | Role |
|---|---|---|
| Nav **library** | `cussonspoon/fod-robot-pathing` (`fodnav`) | FSM, projection, approach/servo, `protocol.md` v1. Consumed as a **wheel** (v0.2.2) — not checked out locally. |
| Integration **harness** | `JuniorSE15/fod-robot` (`fodrobot`) | Pins vision+nav+firmware, simulates the robot, scenarios + evaluation. This is the local `../fod-robot` checkout. |
| Firmware | `Teemy17/fod-robot-esp32` | ESP32 motor control. Yellow-TT prototype. Harness submodule still tracks `dev` (v2, duty-only); a nav-compatible v1 build exists on branch `v1-encoders` (`9dc726d`, 2026-10-24). |

The earlier confusion ("is the nav repo `fod-robot` or `fod-robot-pathing`?") is
resolved: **both exist** — `fod-robot-pathing` is the nav *library* (Cusson),
`fod-robot` is the *harness* that imports it (the team / JuniorSE15).

## Firmware — ~zero code impact from retiring the magnet (D2)

The prototype **never had a collector.** From its README and `protocol_test.cpp`:

- *"yellow TT motors without encoders, **no magnetic drum**"*; *"No drum pins are
  configured and **no collector command is implemented**."*
- The `M <0|1>` (magnet) and `DRUM` commands are **already unrecognized and
  ignored** — `parseCommand("M 1") == kNone`, asserted in the host test.

So D2 removes nothing from the firmware. The magnet survives only in *comments*
(`include/pid.h`'s FR-5 rationale "magnet drag rises as debris accumulates";
README's "the eventual robot"). **OPEN-1's collector is greenfield on the firmware
either way** — a new command + driver to design. The pivot does not break
firmware; it only changes *what* the still-unbuilt collector is.
→ **Firmware action: none required now.** When OPEN-1 lands, design the collector
command fresh. Optionally scrub the magnet-drag comment in `pid.h`.

## Nav library (fodnav) — rename-safe by design

Nav consumes vision through the **state/action/projection contract** and drives
at a CONFIRM track whose `action == "PICK"`. Because D3 maps `tennis_ball → PICK`,
the class rename `screw → tennis_ball` is **downstream-neutral** — exactly the
property the harness contract test already asserts
(`tests/contract/test_vision_contract.py`: `targets[0].action == "PICK"`). The
FSM / servo / approach logic needs **no change** for the class pivot.

One honest caveat: nav's job ends at *driving the drum onto the object*. Whether
the object is then collected is the collector's job (OPEN-1), not nav's — so nav
is unaffected, but the *end-to-end* "and then it's collected" claim is not.
→ **Nav-library action: none required for the class pivot.** Re-release only if
the vision contract surface changes.

## Integration harness (fod-robot) — the real blast radius

The magnet/drum is baked into the **simulation** and **evaluation**, in two kinds:

### (A) Mechanical model-swap — do when the D5 board run passes
- `config/profiles/*.yaml` and `sim_bundle/artifacts/arg-bolts-4-n-640/` point at
  the fastener model. `run.json` taxonomy is `{0:bolt,1:nut,2:screw,3:washer}`,
  `nms_config.json` is `classes:4, image_dims [640,640]`. → regenerate the sim
  bundle for tennis: `classes:1`, `image_dims [480,480]`, taxonomy
  `{0:tennis_ball}`, and repoint the `hef`.
- `pyproject.toml` pins `fod-vision @ v0.3.0` (the fastener release). → bump to the
  tennis release once cv-poc ships it.
- Scenario + test fixtures use `cls: screw` (`scenarios/*.yaml`, `tests/mock`,
  `tests/contract`). → cosmetic rename `screw → tennis_ball`.

### (B) Collection-model rework — BLOCKED on OPEN-1
These encode magnet-drum collection specifically and cannot be finalised until
the collector is chosen:
- `sim/world.py`: *"object collected when the drum's capture width passes over it
  **while the magnet is on (`M 1`)**"*; `drum.capture_width_m`, `drum.offset_x_m`,
  `PICKUP_BAND_M`, `min_drum_lateral_m`.
- `sim/simrobot.py`: `magnet_on`, `world.record(now, magnet_on)`, `pickup`
  events, telemetry `"magnet"` / `"picked"`.
- `evaluate.py`: criteria `picked` / `not_picked` / `drum_miss` ("picked by the
  drum", lateral offset at the drum line) — the pass/fail criteria for the mock +
  contract suites.
- `tests/mock`: asserts `phases == ["vision","servoing","blind","drum"]` and
  `min_drum_lateral_m < 0.02`.

→ **Harness action:** (A) is mechanical and tracks the D5 flip. (B) must wait on
OPEN-1, then generalise "drum pickup while magnet on" to "collected when the
\<OPEN-1 collector\>'s capture region passes over the object," and rename the
`drum`/`magnet` phase + telemetry + criteria accordingly. Note the sim currently
models a magnet pickup the **real firmware never performs** — the sim is already
ahead of the hardware here.

## One-line summary per repo

- **esp32 firmware:** unaffected now; collector is greenfield under OPEN-1.
- **fodnav library:** unaffected; class rename is contract-neutral (tennis_ball→PICK).
- **fod-robot harness:** (A) model-swap on the D5 flip; (B) sim+eval collection
  model is magnet-shaped and blocked on OPEN-1 — the deepest coupling in the project.

## Depth pass (2026-11-04) — verifications, corrections, two new blockers

A second, cited investigation (three read-only explorers) sharpened the verdict.
The sequenced work list it produced is `docs/pivot-migration-plan.md`.

- **Verified at the library level:** nav really is rename-neutral. `fodnav` branches
  only on `state`/`action`, never on class — `target.py:50` marks `cls` "diagnostic
  only, never branch on it", `link/vision.py:133` says "never on cls". The earlier
  "rename-safe" was an inference from the harness tests; it now has first-hand code.
- **Correction — nav does send the magnet, but it is config-gated.** `runner.py:153-156`
  sends `link.magnet(...)`, `safe_stop` sends `M 0`. It is gated by `mission.magnet_on`
  (`_config/nav.yaml:47`, `fsm.py:141-146`), so **disabling the magnet is a config flip,
  no code change.** Nav is not coupled to *ferrous*, but it **is** coupled to drum
  *geometry*: the servo blind-leg stop point (`servo.py:212-215`, `drum.offset_x_m`)
  and the coverage swath (`nav.yaml:100`, `swath_source: drum`). OPEN-1's collector
  geometry feeds those. Nav's "collected" counter is an *arrival* event, not a sensed
  pickup (`fsm.py:226-228`).
- **New blocker (cv-poc) — the board run cannot boot yet.** The tennis artifact has no
  `run.json`, and `runtime/vision.py:75-77` asserts it before the camera opens. Hand-write
  it (do not `fodcv-migrate`, which would clobber the hailo export entry). This gates the
  board run itself, not just the deploy flip.
- **Open item (cross-repo) — nav↔firmware not yet connected (OPEN-3).** Nav speaks v1 (`V` +
  encoders, `esp32.py:72`, handshake raises at `esp32.py:750-753`). The firmware the harness
  *pins* (submodule on `dev`) speaks v2 (`P`-only, no encoders, `protocol.h:6`, rejects `V`
  at `main.cpp:41-44`) — so nav cannot drive *that* build. But this is no longer a design
  hole: Teemy shipped a nav-compatible **v1** firmware on branch `v1-encoders` (`9dc726d`,
  2026-10-24; fodnav PR #9), carrying both `protocol.md` amendments. Remaining work is
  mechanical + hardware: re-point the harness submodule to `v1-encoders`, bench-run
  `link/esp32.py` against it, land Teemy's measured constants, and get his sign-off.
  Independent of the pivot.
- **Refinement — D3 is safe to land now.** `tennis_ball → PICK` changes no cv-poc runtime
  behaviour and breaks exactly one test (`tests/test_datasets.py:271`). It need not wait on
  the board run.
- **Refinement — the harness sidecars are partly cosmetic.** `sim_bundle/.../run.json` and
  `nms_config.json` are not read by the harness; classes reach it via `fodcv class_names(hef)`.
  Swap them for honesty, but the live lever is the hef sidecar.
