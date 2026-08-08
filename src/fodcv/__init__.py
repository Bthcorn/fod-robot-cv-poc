"""FOD robot CV proof of concept.

Layout mirrors where the code runs:

- `fodcv.runtime` -- ships to the Pi. Must not import from `fodcv.research`.
- `fodcv.research` -- Mac only: dataset prep, fine-tuning, runtime export.
- `fodcv.bench` -- on-device measurement.
- `fodcv.paths` / `matrix` / `manifest` -- shared by all three.
"""
