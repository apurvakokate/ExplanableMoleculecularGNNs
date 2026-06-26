"""MotifSAT test collection: pin flat imports before test module import."""
from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
_PKG = _REPO / 'MotifSAT'

from SharedModules.tests.pin_pkg_imports import MOTIFSAT_TOPLEVEL, pin_trainer_imports  # noqa: E402


@pytest.hookimpl(tryfirst=True)
def pytest_collectstart(collector) -> None:
    fspath = str(getattr(collector, 'fspath', '') or '').replace('\\', '/')
    if '/MotifSAT/tests/' in fspath:
        pin_trainer_imports(_PKG, _REPO, MOTIFSAT_TOPLEVEL)
