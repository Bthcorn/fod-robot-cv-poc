# The tennis ball is a Demo Object, not a change of problem

A future reader finding a tennis-ball dataset in a FOD repository will assume
the project pivoted. It did not. FOD remains the problem, the users and the
thesis; the tennis ball is a Demo Object — what gets put on the floor during a
demonstration, because it is visible from the back of a room and does not
require an airfield. It serves a second purpose: it is an independently-sourced
second dataset, and carrying it cleanly through `prepare → train` is direct
evidence for the README's actual claim, that the CV toolchain is de-risked.

## Considered Options

- **Replace FOD with tennis-ball collection.** Rejected: it discards the
  regulatory grounding (FAA AC 150/5210-24, the "FOD locations are not
  recorded" gap that the existing `REPORT` action addresses) and enters a
  market with established consumer products.
- **Detect both fasteners and tennis balls in one model.** Rejected: a ~67 mm
  fluorescent sphere and a ~40 mm dark matte screw have very different scale
  and appearance statistics. `RESULT.md` already records this pipeline
  struggling with the rare class in a 4-class head; one head serving both jobs
  risks degrading the fasteners, which are the ones that matter.

## Consequences

- The tennis ball gets its own Dataset-id and its own Run-id. It never
  displaces `CURRENT_RUN` / `CURRENT_DATASET`, which name the shipping
  fastener model.
- Any claim about tennis-ball accuracy is a claim about the toolchain, not
  about FOD detection performance. The two must not be reported as one number.
