#!/usr/bin/env python3
"""Build the supervisor update deck for the v0.3.0 publish.

    uv run --with python-pptx python slides/build_update.py

Four slides, in the order the work happened: what was trained, what shipped and
the interface it exposes, what it measures, then what is next. Separate from
build_slides.py, which is the full results deck; this one is the short update
and stands alone. Styling and helpers are imported from it so the two look like
the same hand.

Three rules this deck follows, and they are why it reads the way it does:

  - CURRENT STATE ONLY. What is true now, and what was actually done this
    cycle -- training a new dataset and comparing model configurations is
    exactly that. No recap of the previous model line, no re-litigating
    decisions already taken.
  - NO FAILED BUILDS. What was delivered and what it measures, not the
    debugging that got there. A slower configuration that was evaluated and
    set aside is a trade-off, not a failure, and gets a sentence, not a table
    row with numbers that read as a shortfall.
  - EVERY REFERENCE EXPLAINS ITSELF, and only where it earns a place. Plain
    words first, requirement tag in brackets after -- the supervisor does not
    have the PRD open. Same for jargon: mAP50 and INT8 get a gloss the first
    time. No code identifiers, no file paths, no commands on a slide.

Figures come from RESULT.md's "Current build" and "Alternates" blocks, from
the training scripts and session docs for the training methodology, and from
the live session on the board (runs/camera_hailo/timings.csv, 3,294 frames),
with the source named in a comment beside each. RESULT.md 11 forbids several
numbers -- notably the training set's own validation score and any Mac
latency -- and none appear here.

arg-bolts-4 is a public dataset (Roboflow), not the self-collected arena set
slide 4 names as the open next step. Slide 1 says "a new public dataset" on
purpose, so the two slides do not contradict each other.

The photo is committed at slides/img/, not read from runs/: build_slides.py
breaks today because an image it points at was deleted, and this deck should
not inherit that.
"""

from pptx import Presentation
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from build_slides import (
    ACCENT,
    BAND,
    BODY_W,
    H,
    INK,
    M,
    MONO,
    MUTED,
    ROOT,
    W,
    notes,
    slide_base,
    table,
    txt,
)

OUT = ROOT / "slides" / "fod-cv-update-v0.3.0.pptx"
PHOTO = ROOT / "slides" / "img" / "detect_argbolts_640.jpg"

KICKER = "FOD ROBOT | COMPUTER VISION | UPDATE 10 SEPTEMBER 2026"

# --- figures, each with where it comes from --------------------------------
ACCURACY = "0.7715"        # RESULT.md SS13 Alternates: native 640, shipped, own eval split
LATENCY_MS = "24.4"        # same, median; p95 25.9
ACCURACY_480 = "0.7159"    # same table: same backbone, 480 resolution
LATENCY_480 = "16.6"       # same
RETAINED = "99.3%"         # RESULT.md Current build: against the full-precision model's 0.7769
FRAME_MS = "33"            # the camera's own cadence at 1280x720
LIVE_FPS = "30"            # live session on the board, end to end
LIVE_FRAMES = "3,294"      # runs/camera_hailo/timings.csv
DATASET_IMAGES = "12,678"  # RESULT.md SS13: trained at 640 on 12,678 images
TRAIN_TIME = "about an hour"  # docs/autorun-argbolts.md: 1,514 s + 2,114 s for the pair
FPS_640 = "40.9"           # RESULT.md Current build, stated directly
FPS_480 = "60.2"           # derived, 1000 / 16.6 ms; RESULT.md gives only the ms figure
WEIGHTS_N_MB = "5.5"       # artifacts/arg-bolts-4-n-640/best.pt, 5,474,778 B on disk
WEIGHTS_S_MB = "19.2"      # artifacts/arg-bolts-4-s-640/best.pt, 19,181,146 B on disk


def bullets(slide, left, top, width, items, step=0.75, size=15):
    """A list where each item is (bold lead, rest). The lead carries the point, so
    a supervisor skimming only the bold text still gets the slide. `step` is the
    vertical pitch: set it by how many lines the longest item wraps to, because
    nothing here measures text."""
    for i, (lead, rest) in enumerate(items):
        txt(slide, left, top + Inches(i * step), width, Inches(step - 0.05),
            [(lead, {"bold": True, "color": INK}), (rest, {"color": MUTED, "size": size - 1})],
            size=size, space_after=0, line=1.2)
    return top + Inches(len(items) * step)


def calls(slide, left, top, width, items, step=0.55, size=14):
    """Like bullets(), but the lead is set in the code font. These are the
    literal names the robot's program calls -- looking like code is the point,
    so whoever is integrating can find them in the handoff document by eye."""
    for i, (lead, rest) in enumerate(items):
        txt(slide, left, top + Inches(i * step), width, Inches(step - 0.03),
            [(lead, {"bold": True, "color": ACCENT, "font": MONO, "size": size}),
             (rest, {"color": MUTED, "size": size - 1})],
            size=size, space_after=0, line=1.1)
    return top + Inches(len(items) * step)


def band(slide, left, top, width, height, colour=BAND):
    shape = slide.shapes.add_shape(1, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = colour
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def slide_training(prs):
    """1. What was trained: a new dataset, and a comparison of model configurations."""
    s, y = slide_base(prs, KICKER, "Trained on a new dataset",
                      "A public fastener dataset, compared across model configurations")

    table(s, [["Configuration", "Accuracy (0-1)", "Time/frame", "Frames/sec"],
              ["YOLO11n, 640 px (native)", ACCURACY, LATENCY_MS + " ms", FPS_640],
              ["YOLO11n, 480 px (faster)", ACCURACY_480, LATENCY_480 + " ms", FPS_480]],
          M, y + Inches(0.1), Inches(6.5),
          [Inches(2.85), Inches(1.2), Inches(1.2), Inches(1.25)], size=13, highlight=1)

    bullets(s, Inches(7.6), y + Inches(0.1), Inches(5.0), step=0.85, items=[
        ("A new public dataset. ",
         f"{DATASET_IMAGES} images, four classes: bolt, nut, screw, washer."),
        ("Three configurations trained. ", "Two backbones, two resolutions."),
    ])

    txt(s, M, H - Inches(1.2), BODY_W, Inches(0.55),
        [(f"Both shipped configurations trained in {TRAIN_TIME} on a cloud GPU.",
          {"color": INK}),
         (f"A third, YOLO11s ({WEIGHTS_S_MB} MB vs {WEIGHTS_N_MB} MB), was also evaluated "
          "at 640 px.", {"color": MUTED, "size": 13})],
        size=14, space_after=3, line=1.15)
    notes(s, "Dataset: a public Roboflow fastener export (ARG_Bolts_FV), registered as "
             "arg-bolts-4 -- a different dataset from the self-collected arena set on slide "
             "4. 12,678 images at 640. Backbones compared: yolo11n (shipped, both "
             "resolutions) and yolo11s (evaluated, not shipped -- at 640 it does not fit the "
             "33 ms frame). Trained on a rented GPU, 60 passes each (epochs), batch 16, "
             "early stopping disabled, identical settings across runs "
             "(scripts/train_roboflow.sh). About 1,514 s + 2,114 s for the pair. "
             "Frames/sec for the 480 row is derived from its median latency; RESULT.md "
             "states only the millisecond figure for that row.")


def slide_software(prs):
    """2. What shipped, and the functions the robot's program calls to use it."""
    s, y = slide_base(prs, KICKER, "The released software",
                      "Version 0.3.0 - one class to import, and five calls to know")

    bullets(s, M, y + Inches(0.05), BODY_W, step=0.55, items=[
        ("Published on GitHub, version pinned. ",
         "90 KB software package + 4.5 MB trained model, checksummed."),
    ])

    txt(s, M, y + Inches(0.75), BODY_W, Inches(0.3),
        "WHAT THE ROBOT'S PROGRAM CALLS", size=12, bold=True, color=ACCENT)

    calls(s, M, y + Inches(1.13), BODY_W, step=0.5, items=[
        ("Vision(hef=...)  ", "opens the camera and the accelerator, at start-up."),
        (".zone_blocked()  ", "confirmed debris in the strip ahead (FR-4)."),
        (".age  ", "seconds since the last result."),
        (".latest()  ", "objects seen this frame - identity, confidence, position."),
        (".detail()  ", "full record, on request - timing and camera state."),
    ])
    notes(s, "The class is fodcv.runtime.vision.Vision, used as a context manager (a with "
             "block) so the camera and accelerator are always released. zone_blocked() and "
             "age are what the control loop reads every poll; latest() and detail() are "
             "opt-in and do not change what the loop computes. It reports; it does not drive, "
             "steer, or talk to the motor controller -- those calls stay with the robot team. "
             "Deliberately no serial, no metres, no steering: collection is passive, so a "
             "boolean is the whole control input. Installed and verified on the Pi 5 with "
             "camera and accelerator on 6 September.")


def slide_measured(prs):
    """3. The numbers, measured on the board."""
    s, y = slide_base(prs, KICKER, "Measured on the robot",
                      "Bolt, nut, screw and washer, at 640 pixels on the accelerator")

    table(s, [["Measurement", "Result"],
              ["Detection accuracy, scored 0 to 1", ACCURACY],
              ["Kept from the full-precision model", RETAINED],
              ["Time per frame, median (ms)", LATENCY_MS],
              ["Time the camera allows per frame (ms)", FRAME_MS],
              ["Live speed, camera to result (frames/s)", LIVE_FPS],
              ["Frames in the live session", LIVE_FRAMES]],
          M, y + Inches(0.05), Inches(6.2),
          [Inches(4.3), Inches(1.9)], size=13, highlight=1)

    s.shapes.add_picture(str(PHOTO), Inches(7.25), y + Inches(0.05), width=Inches(5.0))
    txt(s, Inches(7.25), y + Inches(2.95), Inches(5.0), Inches(0.4),
        "A frame from the live session on the robot.",
        size=12, color=MUTED)

    txt(s, M, H - Inches(1.1), BODY_W, Inches(0.5),
        "Measured on the model's own held-out images, not the arena floor.",
        size=14, color=ACCENT)
    notes(s, "Accuracy and latency: RESULT.md Current build, 200 images of the run's own eval "
             "split, against the full-precision model's 0.7769. Live figures: a 3,294-frame "
             "session on the board, runs/camera_hailo/timings.csv. The two latency numbers are "
             "different measurements - the 24.4 ms is the benchmark harness; the live camera's "
             "own inference stage is about 15 ms inside a 33.3 ms sensor frame. Quantization "
             "cost: 0.7715 against the full-precision model's 0.7769, under 1%. Inference "
             "finishes well inside the camera's own 33 ms frame -- the camera, not the model, "
             "sets the live speed.")


def slide_next(prs):
    """4. What is unblocked, what is next, what needs the team."""
    s, y = slide_base(prs, KICKER, "Next, and what needs a decision",
                      "The vision side is usable now; the remaining items are data and hardware")

    band(s, M, y, Inches(5.75), Inches(1.0))
    txt(s, M + Inches(0.25), y + Inches(0.18), Inches(5.25), Inches(0.7),
        [("The robot team can start now", {"bold": True, "size": 16}),
         ("Building and testing the speed loop.", {"color": MUTED, "size": 13})],
        space_after=3)

    band(s, Inches(6.85), y, Inches(5.75), Inches(1.0))
    txt(s, Inches(7.1), y + Inches(0.18), Inches(5.25), Inches(0.7),
        [("Next on the vision side: our own arena data", {"bold": True, "size": 16}),
         ("2,000 to 2,500 images from the test arena.", {"color": MUTED, "size": 13})],
        space_after=3)

    txt(s, M, y + Inches(1.3), BODY_W, Inches(0.35),
        "THREE DECISIONS BELONG TO THE TEAM", size=12, bold=True, color=ACCENT)

    bullets(s, M, y + Inches(1.68), BODY_W, step=0.6, items=[
        ("Camera height and tilt on the chassis. ", "Open (O-3)."),
        ("Camera field of view and lookahead distance. ", "Unmeasured (M-3)."),
        ("Metal versus non-metal detection. ",
         "Conflicts with the single-class requirement (FR-13, FR-3)."),
    ])
    notes(s, "Arena data is PRD section 10. Scene-grouped splitting matters because images "
             "from one camera lock are near-identical; shuffling per image puts the same "
             "scene on both sides and inflates the held-out score -- the public dataset "
             "cannot teach the model to ignore ordinary floor clutter either way. O-3: until "
             "camera height and tilt are fixed, the lookahead strip is a placeholder, not a "
             "measurement. M-3: field of view and lookahead distance set how much warning the "
             "robot gets before it reaches an object. FR-13 vs FR-3: with one class, the "
             "report-but-do-not-collect action can never fire -- needs a team decision before "
             "anything downstream branches on it.")


def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    slide_training(prs)
    slide_software(prs)
    slide_measured(prs)
    slide_next(prs)

    for i, s in enumerate(prs.slides, start=1):
        txt(s, W - Inches(1.3), H - Inches(0.52), Inches(0.6), Inches(0.3),
            str(i), size=11, color=MUTED, align=PP_ALIGN.RIGHT)

    OUT.parent.mkdir(exist_ok=True)
    prs.save(OUT)
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(prs.slides._sldIdLst)} slides)")


if __name__ == "__main__":
    build()
