# The tennis-ball demo runs on the board, camera only

**Status:** accepted, not yet executed — the training half comes first; the
compile and the board run are deferred to a later session.

The demonstration is a Pi 5 + Hailo-8 running `fodcv-hailo-camera --preview`
against a tennis ball on the floor. Not a laptop running `best.pt`, and not the
full robot driving at the ball.

Stopping at `best.pt` on a laptop would prove nothing the arg-bolts work has
not already proved. The `.hef` compile is the stage that most needs a second
independent proof: it is the only stage with no Mac fallback (the Dataflow
Compiler is x86_64-Linux-only), and it is where this project's worst surprises
live — `DEPLOY_HEF` is a hardcoded workaround for a build that compiled clean
and scored 0.0000 mAP50, and commit `bfcccc7` is a `.hef` that compiled clean
and decoded nothing. A toolchain claim that stops before the compiler is a
claim about the easy half. The board already runs live at 30 FPS and a compile
is ~35 min and ~$1 of rented CPU, so the marginal cost over a laptop demo is
small.

Driving the robot at the ball is excluded: navigation is another repository and
another author's work, `robot.yaml` is still entirely unmeasured (`v_max_mps`,
`drum.clearance_mm` and the rest are `null`), and the magnetic drum cannot lift
a sphere — so the robot would arrive at the ball and visibly fail to collect
it. That is a worse demonstration than one which never promised collection.

## Consequences

- A tennis-ball Run-id must be compiled to `.hef`, so the imgsz chosen at
  training time must match the compile — and `HEF_CONF` must stay at 0.0001.
- The demo is evidence about the *pipeline*, not about FOD detection
  difficulty. A 67 mm fluorescent sphere is a far easier target than a 40 mm
  dark matte screw; the writeup must not let those blur.
- Full-robot pursuit stays open as a stretch goal once the board run works.
