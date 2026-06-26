"""Pin MOSE-GNN / MotifSAT flat imports (model.py, train.py, …) on sys.path.

Both trainer packages use identical top-level module names. Pytest collection
imports every test module in one process, so the second package must evict the
first package's cached ``sys.modules`` entries before importing.
"""
from __future__ import annotations

import sys
from pathlib import Path

MOSE_TOPLEVEL = ('model', 'train', 'reg_config', 'config', 'run')
MOTIFSAT_TOPLEVEL = MOSE_TOPLEVEL + ('losses', 'motif_modules')


def pin_trainer_imports(
    pkg_dir: Path,
    repo_root: Path,
    toplevel: tuple[str, ...],
) -> None:
    """Drop homonym modules from other packages and prepend *pkg_dir*."""
    pkg = str(pkg_dir.resolve())
    for name in toplevel:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        mod_file = getattr(mod, '__file__', None) or ''
        if not mod_file.startswith(pkg):
            del sys.modules[name]
    while pkg in sys.path:
        sys.path.remove(pkg)
    sys.path.insert(0, pkg)
    root = str(repo_root.resolve())
    if root not in sys.path:
        sys.path.insert(1, root)
