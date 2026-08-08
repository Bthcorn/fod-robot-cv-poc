"""Where things live on disk.

One definition of ROOT. Every module used to recompute it with its own
`Path(__file__).resolve().parent.parent`, which silently meant a different
directory the moment a file moved.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
