"""MOSE-GNN test collection: pin flat imports before test module import."""
from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
_PKG = _REPO / 'MOSE-GNN'

from SharedModules.tests.pin_pkg_imports import MOSE_TOPLEVEL, pin_trainer_imports  # noqa: E402


def _pin() -> None:
    pin_trainer_imports(_PKG, _REPO, MOSE_TOPLEVEL)


@pytest.hookimpl(tryfirst=True)
def pytest_collectstart(collector) -> None:
    fspath = str(getattr(collector, 'fspath', '') or '').replace('\\', '/')
    if '/MOSE-GNN/tests/' in fspath:
        _pin()


@pytest.fixture(autouse=True)
def _pin_mose_imports(request):
    """Re-pin before each test (MotifSAT collection may have swapped sys.modules)."""
    fspath = str(getattr(request.node, 'fspath', '') or '').replace('\\', '/')
    if '/MOSE-GNN/tests/' in fspath or fspath.endswith('test_mose_gnn.py'):
        _pin()
    yield
