"""Command-line entry points. One module per command, argparse only.

Kept out of the library modules so runtime/, research/ and bench/ stay
importable without dragging CLI parsing in -- runtime/policy.py has to load on
the Pi without an argument parser attached to it.

Each module's main() is wired to a console script in pyproject.toml:

    uv run fodcv-bench --run poc-v1

There is no scripts/ shim directory. `uv sync` installs the package, which is
already how the Pi gets the code, so a second spelling of every command would
only be a layer forwarding to this one.
"""
