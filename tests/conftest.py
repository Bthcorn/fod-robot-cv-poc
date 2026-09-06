"""Some test files need the research stack (ultralytics, torch, yaml) -- at
import, or inside the code under test. On a base install -- CI, or a Pi -- they cannot be collected,
and test_train_cli.py does worse: fodcv.cli.train raises SystemExit with its
install hint, which pytest treats as an internal error and aborts the whole
run. Skip them wholesale there; locally, with the research extra, everything
runs.

ponytail: one list here, not one importorskip per file. A new research test
without an entry fails loudly in CI, which is the right direction.
"""

from importlib.util import find_spec

if find_spec("ultralytics") is None:
    collect_ignore = [
        "test_datasets.py",  # imports clean, reaches yaml inside prepare()
        "test_eval.py",
        "test_export_a16.py",
        "test_matrix.py",
        "test_pi_bench_subset.py",
        "test_pi_probes.py",
        "test_provenance.py",
        "test_train_cli.py",
    ]
