"""MOSE-GNN test collection: pin flat imports before test module import."""
from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
_PKG = _REPO / 'MOSE-GNN'

from SharedModules.tests.pin_pkg_imports import MOSE_TOPLEVEL, pin_trainer_imports  # noqa: E402


@pytest.hookimpl(tryfirst=True)
def pytest_collectstart(collector) -> None:
    fspath = str(getattr(collector, 'fspath', '') or '').replace('\\', '/')
    if '/MOSE-GNN/tests/' in fspath:
        pin_trainer_imports(_PKG, _REPO, MOSE_TOPLEVEL)
