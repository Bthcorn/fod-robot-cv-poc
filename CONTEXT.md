# FOD Robot CV

> **Pivot in progress (2026-11-04, ADR-0008).** The detection target has moved
> from metal FOD to the **tennis ball**, permanently. The magnetic collector is
> retired, `PICK` is redefined mechanism-agnostic, and the research thesis is
> being redesigned (OPEN-2). Entries below marked _(historical)_ or _(original
> problem)_ describe the pre-pivot framing and are kept to explain older docs.

The vision half of the FOD Robot thesis: turning labelled images into a detector
the robot can run, and measuring whether it works. This context owns datasets,
training runs, exported artifacts, and the detection-to-action policy. It does
not own navigation or chassis control — those live in `fodnav`.

## Language

### The problem

**FOD** _(original problem — see ADR-0008)_:
Foreign Object Debris — loose material on a surface an aircraft operates on,
which can damage an aircraft or injure a person. The problem the project was
originally built around. The project has since pivoted its detection target to
the tennis ball (ADR-0008); what real-world problem now frames the work is open
(OPEN-2).
_Avoid_: debris, foreign object damage (that is FO*D*amage, the consequence,
not the object)

**Demo Object** _(historical — see ADR-0008)_:
Under the original FOD framing, an object substituted for real FOD when
demonstrating the system, chosen for visibility and availability rather than
realism. The tennis ball was one. The pivot (ADR-0008) made the tennis ball the
project's actual detection target rather than a stand-in, so this term no longer
describes it; it is kept only to explain pre-pivot docs.

**Test Object**:
One of the specific items FAA AC 150/5210-24 names for certifying FOD removal
equipment: non-ferrous metal, rubber, a 4.3 cm sphere, an asphalt chunk, a
wrench. A regulatory term with a fixed meaning. Do not use it loosely.

### The pipeline

**Dataset-id**:
The name of one prepared dataset, identifying it everywhere — on disk at
`data/<dataset-id>/`, and in every command's `--dataset`. Named
`<source>-<class count>[-modifier]`, after the taxonomy rather than after the
vendor's slug: `arg-bolts-4`, `fod-a-1-neg`.

**Run-id**:
The name of one trained model, identifying it everywhere — on disk at
`artifacts/<run-id>/`, and in every command's `--run`. Named
`<dataset-id>-<backbone>-<imgsz>`, so that the pairing is legible from the
directory name alone. A Run-id and the Dataset-id it was trained on move as a
pair.

**Prepared Dataset**:
A registered source converted to the one on-disk shape the training pipeline
reads. The output of `fodcv-prepare`, and the only form training ever sees.
_Avoid_: export (that is the vendor's zip, the input), processed data

**Source**:
The registry entry describing where a dataset comes from and how to build it.
Either a `VocSource` (fetched and converted) or a `YoloSource` (already
labelled, copied and validated).

**Deploy Unit**:
The directory under `artifacts/<run-id>/` holding everything the robot needs —
weights, exports, manifest and eval split — such that one `rsync` moves it to
the Pi.

### What the robot does about a detection

**Action**:
What the robot should do about an object, keyed by its class name. One of
PICK, REPORT or IGNORE. Deliberately a different axis from State: State is
confidence over time, Action is capability. A class that is not listed is
IGNORE.

**PICK**:
The robot is meant to collect this object — the class the approach behaviour
drives a CONFIRM track onto. **Mechanism-agnostic**: it names *intent to
collect*, not a proven capability. It once meant "ferrous, so the magnetic drum
can lift it"; the pivot off FOD retired that drum (ADR-0008), so *which*
collector does the collecting is now open (OPEN-1). Until a collector is chosen
and measured, PICK is a claim about intent, not hardware.

**REPORT**:
Detected and logged, not collected — whether because the identification is
uncertain or because the object is real but not collectable. Addresses the gap
FAA AC 150/5210-24 records, that FOD locations go unrecorded during manual
inspection.

**State**:
Where a track sits on the confidence-over-time axis — IGNORE, CAUTION or
CONFIRM — governed by hysteresis on an EMA of detection confidence. Says
nothing about what the object is or whether it can be collected.

### Measurement

**Own-split Score**:
A mAP measured against the dataset's own validation split. A progress readout,
never a result: every Roboflow export in use here overlaps its own train split
at scene level, so this number partly measures memorisation.
_Avoid_: val mAP used without qualification

**Scene-disjoint Holdout**:
Images from scenes that contributed no training frame, scored against to get an
honest number. The only kind of score this project reports as a result.
_Avoid_: test set

**Cross-dataset Holdout**:
A whole separate dataset, from a different source, that contributes no
training frame to any run. Scene-disjointness by construction rather than by
bookkeeping. Training on one, even once, destroys it permanently.

**Mac Gate**:
The set of checks run locally before renting a GPU, whose purpose is to fail
cheaply. It proves the data, the registry, the scorer and the export path are
wired correctly; it proves nothing about accuracy, latency or the `.hef`, and
every number it produces is throwaway.

**Generalisation Gap**:
The distance between a run's Own-split Score and its Cross-dataset Holdout
score. Reported as a finding in its own right, not hidden behind whichever
number is larger.
