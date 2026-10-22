#!/usr/bin/env python
"""status.py — per-cell completion of the replication (read-only).

DONE   = results/<bb>/<ds>/seed<k>.json exists and has metric_dict
FAILED = listed in _dispatch/failures.tsv and not DONE
else MISSING (queued, running or never submitted; check squeue).
Prints one line per (backbone, dataset) and the array indices that are not DONE.
"""
import json
import sys
from pathlib import Path

R = Path('/nfs/hpc/share/kokatea/ChemIntuit/Claude+Cursor/gsat_replication_v1')
sys.path.insert(0, str(R / '_scripts'))
import cells  # noqa: E402

failed = set()
ft = R / '_dispatch' / 'failures.tsv'
if ft.exists():
    for line in ft.read_text().splitlines():
        p = line.split('\t')
        if len(p) >= 5:
            failed.add((p[2], p[3], int(p[4])))

todo, tot = [], {'DONE': 0, 'FAILED': 0, 'MISSING': 0}
for (bb, ds) in cells.CELLS:
    st = {'DONE': 0, 'FAILED': 0, 'MISSING': 0}
    for s in cells.SEEDS:
        f = R / 'results' / bb / ds / f'seed{s}.json'
        ok = False
        if f.exists():
            try:
                ok = 'metric_dict' in json.loads(f.read_text())
            except Exception:
                ok = False
        k = 'DONE' if ok else ('FAILED' if (bb, ds, s) in failed else 'MISSING')
        st[k] += 1
        if k != 'DONE':
            todo.append(cells.RUNS.index((bb, ds, s)))
    for k in st:
        tot[k] += st[k]
    print(f'{bb:4s} {ds:16s} done={st["DONE"]:2d} failed={st["FAILED"]:2d} missing={st["MISSING"]:2d}')
print('TOTAL', tot)
print('not-done indices:', ','.join(map(str, todo)) if todo else 'none')
