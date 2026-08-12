"""exports.json -- the handover from the Mac that builds artifacts to the Pi
that measures them.

One cell per `{format}:{precision}` key, holding either an artifact path or a
sentinel explaining why there isn't one:

    "onnx:int8":   "bench_int8.onnx"
    "ncnn:int8":   "UNSUPPORTED: ncnn has no int8 export path"
    "litert:int8": "FAILED: RuntimeError: ..."

Two traps this module exists to close:
  - Paths are stored **relative to this file**, or the run directory stops
    surviving an rsync and every cell silently falls back to a local export.
  - `built()` is the only place a cell is judged usable. Sites that re-derive
    it drift -- filtering FAILED but not UNSUPPORTED reads a skipped cell as
    already built.
"""

import json
from pathlib import Path

NAME = "exports.json"
FAILED = "FAILED"
UNSUPPORTED = "UNSUPPORTED"
_SENTINELS = (FAILED, UNSUPPORTED)


def key(fmt: str, label: str) -> str:
    return f"{fmt}:{label}"


def is_sentinel(entry: str) -> bool:
    """Is this cell an explanation rather than an artifact path?

    The one place the sentinel prefix is matched. Do not re-derive it elsewhere.
    """
    return entry.startswith(_SENTINELS)


def load(manifest_path) -> dict:
    return json.loads(manifest_path.read_text()) if manifest_path.exists() else {}


def save(manifest_path, manifest: dict):
    """Write the whole manifest. Callers save per cell, not once at the end, so
    a 30-minute export that dies late keeps what it already built."""
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def built(manifest: dict, manifest_path, fmt: str, label: str):
    """The artifact path for this cell, or None if there isn't a usable one."""
    entry = manifest.get(key(fmt, label), "")
    if not entry or is_sentinel(entry):
        return None
    path = Path(manifest_path).parent / entry
    return path if path.exists() else None


def entry_for(artifact_path, manifest_path) -> str:
    """How an artifact is written into the manifest: relative to the manifest.

    Absolute fallback if it lands outside the run directory -- better a path
    that works only here than a silently broken one.
    """
    artifact, manifest_dir = Path(artifact_path).resolve(), Path(manifest_path).parent.resolve()
    try:
        return str(artifact.relative_to(manifest_dir))
    except ValueError:
        return str(artifact)
