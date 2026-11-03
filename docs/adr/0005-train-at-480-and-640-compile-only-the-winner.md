# Train the tennis ball at both 480 and 640; compile only the winner

**Status: amended 2025-11-03 — the pair is 320 and 480, not 480 and 640.**
Every image in the export is **320x240** "Resize to 320x240 (Fill with center
crop)" per its `README.roboflow.txt`. Training at 640 would upsample a 320-wide
source to 640, paying 1.87x the latency for no additional information. So the
two runs are **320** (native) and **480** (does upscaling help), and 640 is
dropped. `RESULT.md` "320 collapses by 37%" was measured on 40 mm fasteners
at 49 px; the median ball here is 150 px of a 320-wide frame, a different
regime entirely. Everything else below — one variable, compile only the
winner, smaller size wins a tie — is unchanged.

Two training runs, one `.hef`. Training is ~25 min each on a rented 4090;
compiling is ~35 min and is where this project's worst failures live, so the
board (ADR-0002) gets one model, not a leaderboard.

640 was adopted for `arg-bolts` to chase small fasteners at the top of the
frame. A 67 mm tennis ball does not have that problem, and `RESULT.md` §363
records that 480 never lost on accuracy anyway — "Same mAP (0.721 vs 0.719) at
1.87× the speed" — while 320 collapses by 37% and is not an option. §406
already lists "`arg-bolts-4` retrained at 480" as unfinished business; running
both here closes the same question on cheaper data.

**480 wins a tie**, on three independent grounds: 1.87× faster (16.6 ms vs
24.4 ms, leaving the control loop the other 8 ms), immune to the conf-0.001
silent-zero bug which §44 and §460 record as 640-specific, and already the
deployment resolution `poc-v2` shipped at. 640 is only declared the winner if
it wins by a wide margin **on the Cross-dataset Holdout** — the gap between two
Own-split Scores on a contaminated split means nothing.

Everything else mirrors the shipping configuration exactly — `yolo11n`, same
epoch count, fixed seed — so that the dataset is the only variable that moved
and the toolchain claim in ADR-0001 is actually supported. `yolo11n` is also
the only backbone that fits the 33 ms frame: `s` at 640 measures 50.2 ms.

## Consequences

- `imgsz` at training must match `HEF_IMGSZ` at compile; `nms_config.json`
  carries the input size and `Vision` reads it off the `.hef`, so a mismatch
  letterboxes silently.
- A single class means `--a16-cls` does not apply; that flag is for heads above
  roughly four classes.

## Not done

- **320.** Potentially ~2× faster again, and the 37% collapse measured on
  fasteners may not transfer to a large high-contrast sphere. It is a third
  variable in an experiment whose purpose is the toolchain, so it is logged as
  future work rather than run.
- **A backbone sweep.** `scripts/train_roboflow.sh` would run n/s/m. It answers
  "what is the best tennis-ball model", which nobody is asking — the ball is a
  Demo Object.
