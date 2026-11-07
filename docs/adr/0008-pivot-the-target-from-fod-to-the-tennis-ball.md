# The detection target pivots from metal FOD to the tennis ball

**Status:** accepted 2026-11-04. Supersedes ADR-0001 (which held that the tennis
ball is a Demo Object and FOD remains the problem). Reframes ADR-0002.

The project's detection target is now the **tennis ball**, permanently, not metal
Foreign Object Debris. The tennis ball stops being a Demo Object — a stand-in
carried through the pipeline to de-risk the toolchain — and becomes the thing the
robot is built to detect and go to. This is a deliberate reversal of ADR-0001,
taken with eyes open to what it costs.

## What this breaks, on purpose

- **The magnetic drum is retired (D2).** A magnet does nothing to a rubber-and-felt
  sphere. The passive magnetic collector, the magnet-gap budget (PRD §9, App B.4)
  and every "the magnet lifts it" claim go with it.
- **`PICK` is redefined (D3).** It no longer means "ferrous, so the drum can lift
  it." It now names *intent to collect, mechanism-agnostic* — the class the
  approach behaviour drives a CONFIRM track onto. `tennis_ball` therefore maps to
  `PICK`, not `REPORT`. The collector that will actually do the collecting is
  **open (OPEN-1)**: the project still wants to collect the ball, but not by
  magnet, and the mechanism — scoop, brush, corral, bin — is undecided and needs
  research + a firmware/advisor decision.
- **The research thesis is not settled here (D4, OPEN-2).** The original
  contribution (perception coupled to a *physical magnetic* collector; the
  adaptive-speed crossover density; pickup-rate-vs-speed for a passive magnet) is
  magnet-shaped and does not survive D2 intact. Whether the adaptive-speed claim
  is re-instantiated on the tennis ball or replaced is a separate task, deferred
  to the advisor. This ADR fixes only the engineering and the docs so the codebase
  and PRD are *coherent for a tennis-ball target* — it does not invent a thesis.

## Considered Options

- **Keep ADR-0001 — tennis ball stays a Demo Object, FOD stays the problem.**
  Rejected: the operator decided to pivot. The honest move is to record the
  reversal, not to keep demoting the tennis ball in a repo that now ships it.
- **Swap the collector for a tennis-ball mechanism and keep the pickup thesis.**
  Not chosen *here*: a late hardware + firmware rebuild that teammates must absorb,
  and it presupposes OPEN-2. Left open as part of OPEN-1/OPEN-2, not decided.
- **Drop collection entirely (detection-only demo).** Rejected: the operator still
  wants the robot to collect the ball — hence OPEN-1 rather than "no collector."

## Consequences

- **The tennis 480 run becomes the shipping line, gated on the board run (D5).**
  `CURRENT_RUN` / `CURRENT_DATASET` / `DEPLOY_*` move to `tennis-ball-1-neg-n-480`
  and its `.hef` — but only after the Pi + Hailo-8 board run proves the `.hef`
  loads and detects. The project has twice trusted a clean compile that decoded
  nothing (`bfcccc7`, `DEPLOY_HEF`'s 0.0000-mAP workaround), so the flip does not
  happen on a compile alone. The fastener artifacts stay on disk as the demoted,
  measured historical baseline.
- **PRD v4 supersedes v3 (D6).** Everything D1–D5 settles is rewritten to the
  tennis target; the thesis sections (§3/§4/§11) carry explicit `OPEN-2`
  placeholders rather than invented claims. Section numbering is preserved because
  `README.md`, `docs/INTEGRATION.md` and `docs/dataset-roadmap.md` cite the PRD by
  section.
- **Cross-repo impact is unknown and must be investigated.** Navigation and
  firmware were written for a ferrous target and a magnetic drum; what they assume
  about the class set, the `PICK` action and the collector is a check in its own
  right (see the impact-investigation task).
