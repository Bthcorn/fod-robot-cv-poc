"""exports.json -- the handover between the Mac that builds artifacts and the Pi
that measures them.

One cell per `{format}:{precision}` key. A cell holds either a path to a usable
artifact, or a sentinel explaining why there isn't one:

    "onnx:int8":   "bench_int8.onnx"
    "ncnn:int8":   "UNSUPPORTED: ncnn has no int8 export path"
    "litert:int8": "FAILED: RuntimeError: ..."

Paths are stored **relative to this file**, which is what makes the run
directory relocatable. They used to be absolute Mac paths, so after an rsync the
Pi's `Path(entry).exists()` was false for every cell unless its checkout sat at
the identical absolute path -- and each miss fell through to a local export,
which on aarch64 cannot build LiteRT at all. The failure was silent: the
benchmark just reported different numbers.

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


def is_sentinel(entry: str) -> bool:
    """Is this cell an explanation rather than an artifact path?

    The one place the sentinel prefix is matched. migrate_artifacts used to
    re-derive it, which is how the two sides came to disagree in the first place.
    """
    return entry.startswith(_SENTINELS)


def load(manifest_path) -> dict:
    return json.loads(manifest_path.read_text()) if manifest_path.exists() else {}


def save(manifest_path, manifest: dict):
    """Write the whole manifest.

    Callers save per cell rather than once at the end: a 30-minute export run
    that dies on the last format must not lose the artifacts it already built.
    """
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

    Falls back to an absolute path if the artifact somehow lands outside the run
    directory -- better a path that works only here than a silently broken one.
    """
    artifact, manifest_dir = Path(artifact_path).resolve(), Path(manifest_path).parent.resolve()
    try:
        return str(artifact.relative_to(manifest_dir))
    except ValueError:
        return str(artifact)
