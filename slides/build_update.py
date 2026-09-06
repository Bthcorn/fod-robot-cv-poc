#!/usr/bin/env python3
"""Build the supervisor update deck for the v0.3.0 publish.

    uv run --with python-pptx python slides/build_update.py

Four slides on what the publish delivered. Separate from build_slides.py, which
is the full results deck; this one is the short update and stands alone. Styling
and helpers are imported from it so the two look like the same hand.

Three rules this deck follows, and they are why it reads the way it does:

  - CURRENT STATE ONLY. What is true now. No before-and-after, no recap of the
    previous model line, no history of decisions already taken.
  - NO FAILED BUILDS. What was delivered and what it measures, not the debugging
    that got there.
  - EVERY REFERENCE EXPLAINS ITSELF, and only where it earns a place. Plain words
    first, requirement tag in brackets after -- the supervisor does not have the
    PRD open. Same for jargon: mAP50 and INT8 get a gloss the first time. No code
    identifiers, no file paths, no commands on a slide.

Figures come from RESULT.md's "Current build" block and from the live session on
the board (runs/camera_hailo/timings.csv, 3,294 frames), with the section named in
a comment beside each. RESULT.md 11 forbids several numbers -- notably the
training set's own validation score and any Mac latency -- and none appear here.

The photo is committed at slides/img/, not read from runs/: build_slides.py
breaks today because an image it points at was deleted, and this deck should not
inherit that.
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

KICKER = "FOD ROBOT | COMPUTER VISION | UPDATE 6 SEPTEMBER 2026"

# --- figures, each with where it comes from --------------------------------
ACCURACY = "0.7715"    # RESULT.md, Current build: 200 images, own eval split
RETAINED = "99.3%"     # same, against the full-precision model's 0.7769
LATENCY_MS = "24.4"    # same, median; p95 25.9
FRAME_MS = "33"        # the camera's own cadence at 1280x720
LIVE_FPS = "30"        # live session on the board, end to end
LIVE_FRAMES = "3,294"  # runs/camera_hailo/timings.csv


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


def band(slide, left, top, width, height, colour=BAND):
    shape = slide.shapes.add_shape(1, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = colour
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def slide_released(prs):
    """1. What exists now: a released, installable component."""
    s, y = slide_base(prs, KICKER, "The vision system is released",
                      "Version 0.3.0, published on GitHub and installed on the robot")

    table(s, [["What is in the release", "Size", "Purpose"],
              ["Software package", "90 KB", "what the robot's program imports"],
              ["Trained model", "4.5 MB", "what runs on the accelerator"],
              ["Checksums", "-", "verify a download is intact"]],
          M, y + Inches(0.1), Inches(6.5),
          [Inches(2.5), Inches(1.1), Inches(2.9)], size=13)

    bullets(s, Inches(7.6), y + Inches(0.05), Inches(5.0), step=0.95, items=[
        ("Two commands to install. ", "Nothing else is needed on the robot."),
        ("No training software on the robot. ", "The heavy machine-learning stack stays on the development machine."),
        ("The version is pinned. ", "The software records which model it expects, so which model is on the robot always has an answer."),
        ("Verified on the hardware. ", "Installed and run on the Raspberry Pi 5 with the camera and accelerator."),
    ])

    txt(s, M, H - Inches(1.35), BODY_W, Inches(0.5),
        "The robot team installs it from the release. They do not need this project's "
        "code, its datasets, or its training tools.",
        size=14, color=MUTED)
    notes(s, "v0.3.0 on GitHub carries the wheel, three model bundles and SHA256SUMS. "
             "Install is a single pip line plus a download-and-extract. Verified end to end "
             "on the board on 6 September with a scripted smoke test.")


def slide_interface(prs):
    """2. What the teammate integrates against."""
    s, y = slide_base(prs, KICKER, "What the robot team receives",
                      "One class to import, and one question answered every frame")

    band(s, M, y + Inches(0.05), BODY_W, Inches(1.15))
    txt(s, M + Inches(0.3), y + Inches(0.42), BODY_W - Inches(0.6), Inches(0.5),
        "Is there confirmed debris in the strip of floor we are about to drive over?",
        size=20, bold=True, color=INK, align=PP_ALIGN.CENTER)

    bullets(s, M, y + Inches(1.5), BODY_W, step=0.72, items=[
        ("That answer drives the speed rule. ",
         "Slow down when it is yes, full speed when it is no - the behaviour the requirements ask for (FR-4)."),
        ("Detail is there when wanted, and out of the way when not. ",
         "Each object's identity, confidence and timings are available on request, off the control path, so the loop stays cheap."),
        ("It ships ready to hand over. ",
         "A written integration document, and a single command that checks a fresh robot end to end."),
    ])

    txt(s, M, H - Inches(1.35), BODY_W, Inches(0.5),
        "The vision system reports. It does not drive, steer, or talk to the motor "
        "controller - those stay with the robot team.",
        size=14, color=MUTED)
    notes(s, "The class is Vision; the per-frame call is zone_blocked(). Detail comes from "
             "latest() and detail(). Deliberately no serial, no metres, no steering: "
             "collection is passive, so a boolean is the whole control input.")


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

    txt(s, M, H - Inches(1.75), BODY_W, Inches(0.9),
        [("Compressing the model to 8-bit for the accelerator costs under 1% of its accuracy, "
          "and inference finishes well inside the camera's frame - the camera, not the model, "
          "sets the speed.", {"color": INK}),
         ("Measured on held-out images from the same dataset: a fair number for comparing "
          "models, not a prediction of accuracy on the real arena floor.", {"color": ACCENT})],
        size=14, space_after=4)
    notes(s, "Accuracy and latency: RESULT.md Current build, 200 images of the run's own eval "
             "split, against the full-precision model's 0.7769. Live figures: a 3,294-frame "
             "session on the board, runs/camera_hailo/timings.csv. The two latency numbers are "
             "different measurements - the 24.4 ms is the benchmark harness; the live camera's "
             "own inference stage is about 15 ms inside a 33.3 ms sensor frame.")


def slide_next(prs):
    """4. What is unblocked, what is next, what needs the team."""
    s, y = slide_base(prs, KICKER, "Next, and what needs a decision",
                      "The vision side is usable now; the remaining items are data and hardware")

    band(s, M, y, Inches(5.75), Inches(1.15))
    txt(s, M + Inches(0.25), y + Inches(0.2), Inches(5.25), Inches(0.8),
        [("The robot team can start now", {"bold": True, "size": 16}),
         ("Building and testing the speed loop against a fixed, installed interface.",
          {"color": MUTED, "size": 13})],
        space_after=3)

    band(s, Inches(6.85), y, Inches(5.75), Inches(1.15))
    txt(s, Inches(7.1), y + Inches(0.2), Inches(5.25), Inches(0.8),
        [("Next on the vision side: our own arena data", {"bold": True, "size": 16}),
         ("2,000 to 2,500 images collected in the test arena, grouped by scene when split.",
          {"color": MUTED, "size": 13})],
        space_after=3)

    txt(s, M, y + Inches(1.45), BODY_W, Inches(0.35),
        "THREE DECISIONS BELONG TO THE TEAM", size=12, bold=True, color=ACCENT)

    bullets(s, M, y + Inches(1.85), BODY_W, step=0.82, items=[
        ("Where the camera sits on the chassis - height and tilt. ",
         "Open (O-3). Until it is fixed, the strip of floor the system watches is a placeholder rather than a measurement."),
        ("How wide the camera sees, and how far ahead it looks. ",
         "Both unmeasured (M-3). They set how much warning the robot gets before it reaches an object."),
        ("Metal versus non-metal, against a single detection class. ",
         "One requirement asks the robot to tell them apart (FR-13); another asks the detector to learn one class only (FR-3). "
         "With one class, the report-but-do-not-collect action can never fire."),
    ])

    txt(s, M, H - Inches(0.95), BODY_W - Inches(1.2), Inches(0.35),
        "The public dataset cannot teach the model to ignore ordinary floor clutter.",
        size=14, color=MUTED)
    notes(s, "Arena data is PRD section 10. Scene-grouped splitting matters because images from "
             "one camera lock are near-identical; shuffling per image puts the same scene on both "
             "sides and inflates the held-out score. O-3 and M-3 are open items in the handoff "
             "document. FR-13 against FR-3 needs a team decision before anything downstream "
             "branches on the action.")


def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    slide_released(prs)
    slide_interface(prs)
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
