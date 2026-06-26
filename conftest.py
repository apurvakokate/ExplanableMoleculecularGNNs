"""Repo-root pytest hooks for MOSE-GNN / MotifSAT import isolation."""
from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent

# Import after repo root is known (SharedModules is under _REPO).
from SharedModules.tests.pin_pkg_imports import (  # noqa: E402
    MOSE_TOPLEVEL,
    MOTIFSAT_TOPLEVEL,
    pin_trainer_imports,
)


def _norm(path: str) -> str:
    return path.replace('\\', '/')


@pytest.hookimpl(tryfirst=True)
def pytest_collectstart(collector) -> None:
    """Re-pin trainer paths immediately before each test module is imported."""
    nodeid = _norm(getattr(collector, 'nodeid', '') or '')
    fspath = _norm(str(getattr(collector, 'fspath', '') or ''))
    if nodeid.startswith('MotifSAT/tests/') or '/MotifSAT/tests/' in fspath:
        pin_trainer_imports(_REPO / 'MotifSAT', _REPO, MOTIFSAT_TOPLEVEL)
    elif nodeid.startswith('MOSE-GNN/tests/') or '/MOSE-GNN/tests/' in fspath:
        pin_trainer_imports(_REPO / 'MOSE-GNN', _REPO, MOSE_TOPLEVEL)
