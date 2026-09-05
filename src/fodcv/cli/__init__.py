"""Command-line entry points. One module per command, argparse only.

Kept out of the library modules so runtime/, research/ and bench/ stay
importable without dragging CLI parsing in.

Each main() is wired to a console script in pyproject.toml:

    uv run fodcv-bench --run arg-bolts-4-n-640

ponytail: no scripts/ shim -- `uv sync` installs the package, which is already
how the Pi gets the code.
"""
