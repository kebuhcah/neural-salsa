"""Shared paths and import setup for every script in this repo.

Every script starts with these two lines:

    import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from _paths import ROOT, DATA  # noqa: E402

The first makes this file importable from a subfolder; importing it then
puts every code folder on sys.path, so any script can do a flat
`from train_phase import load_all` whichever folder either file lives in.

ROOT  the repository root. Results that are tracked in git are written
      under ROOT / "data".
DATA  where the local, gitignored data lives (features, chroma, audio, saved
      models, beat machine assets). Defaults to ROOT / "data"; set the
      SALSA_DATA environment variable to read it from somewhere else -- e.g.
      a git worktree reading the main checkout's data.

Run scripts from the repository root: some default paths ("data/cv",
"data/ens", ...) are relative to it.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("SALSA_DATA", ROOT / "data"))

# core/: shared library code + main training entry points
# tools/: reusable reports and utilities
# experiments/: one-off investigations, each tied to a NOTES section
# local/: gitignored tools (grid decoder, downloads, feature builders)
for folder in ("local", "experiments", "tools", "core"):
    path = str(ROOT / folder)
    if path not in sys.path:
        sys.path.insert(0, path)
