"""exports.json -- the handover between the Mac that builds artifacts and the Pi
that measures them.

One cell per `{format}:{precision}` key. A cell holds either a path to a usable
artifact, or a sentinel explaining why there isn't one:

    "onnx:int8":   "bench_int8.onnx"
    "ncnn:int8":   "UNSUPPORTED: ncnn has no int8 export path"
    "litert:int8": "FAILED: RuntimeError: ..."

Both sides used to hand-parse this format, and they disagreed: the exporter's
resume check filtered only FAILED while every other site filtered FAILED and
UNSUPPORTED, so an UNSUPPORTED cell read as "already built". One `built()` here
is the whole guard.
"""

import json
from pathlib import Path

NAME = "exports.json"
FAILED = "FAILED"
UNSUPPORTED = "UNSUPPORTED"
_SENTINELS = (FAILED, UNSUPPORTED)


def key(fmt: str, label: str) -> str:
    return f"{fmt}:{label}"


def load(manifest_path) -> dict:
    return json.loads(manifest_path.read_text()) if manifest_path.exists() else {}


def save(manifest_path, manifest: dict):
    """Write the whole manifest.

    Callers save per cell rather than once at the end: a 30-minute export run
    that dies on the last format must not lose the artifacts it already built.
    """
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def built(manifest: dict, fmt: str, label: str):
    """The artifact path for this cell, or None if there isn't a usable one."""
    entry = manifest.get(key(fmt, label), "")
    if not entry or entry.startswith(_SENTINELS):
        return None
    path = Path(entry)
    return path if path.exists() else None
