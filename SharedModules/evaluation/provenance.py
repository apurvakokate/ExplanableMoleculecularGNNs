"""Provenance stamping — capture WHICH code + config produced a result, at run time.

A result's provenance must be recorded by the producer (run.py) when the result is made,
not reconstructed at harvest time: a harvest-time git SHA is the code as it is NOW, not the
code that produced a days-old summary. Every summary.json should carry these three fields so
the harvester can tell latest-code rows from stale ones (Rule Set 1, decision #1).
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
from typing import Any, Dict, Optional

PROVENANCE_KEYS = ('git_sha', 'run_timestamp', 'config_hash')


def git_sha(cwd: Optional[str] = None) -> str:
    """Short git SHA of the code, or 'unknown' if unavailable / not a repo."""
    try:
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                             cwd=cwd or os.getcwd(), capture_output=True, text=True, timeout=10)
        sha = out.stdout.strip()
        if out.returncode == 0 and sha:
            dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=cwd or os.getcwd(),
                                   capture_output=True, text=True, timeout=10).stdout.strip()
            return sha + ('-dirty' if dirty else '')
    except Exception:
        pass
    return 'unknown'


def config_hash(config: Any) -> str:
    """Stable 8-char hash of the resolved run config (dict / dataclass / str)."""
    try:
        if hasattr(config, '__dict__'):
            config = vars(config)
        blob = json.dumps(config, sort_keys=True, default=str)
    except Exception:
        blob = str(config)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def provenance_fields(config: Any = None, cwd: Optional[str] = None) -> Dict[str, str]:
    """The three fields every summary.json should carry."""
    return {
        'git_sha': git_sha(cwd),
        'run_timestamp': _dt.datetime.now().isoformat(timespec='seconds'),
        'config_hash': config_hash(config) if config is not None else 'none',
    }
