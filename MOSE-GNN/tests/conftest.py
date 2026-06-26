"""Pytest setup: MOSE-GNN top-level modules must win over MotifSAT when collected."""
from __future__ import annotations

import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent.parent
_ROOT = _PKG_DIR.parent

_TOPLEVEL = ('model', 'train', 'reg_config', 'config', 'run')


def _ensure_mose_path() -> None:
    pkg = str(_PKG_DIR)
    for name in _TOPLEVEL:
        mod = sys.modules.get(name)
        mod_file = getattr(mod, '__file__', None) or ''
        if mod is not None and not mod_file.startswith(pkg):
            del sys.modules[name]
    while pkg in sys.path:
        sys.path.remove(pkg)
    sys.path.insert(0, pkg)
    root = str(_ROOT)
    if root not in sys.path:
        sys.path.insert(1, root)


_ensure_mose_path()
