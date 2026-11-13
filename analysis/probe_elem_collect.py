#!/usr/bin/env python3
"""Concatenate the per-cell probe CSVs into one file per kind.

Each array task writes <out_dir>/<dataset>/<backbone>_<fold>_{gap,perelem,
composition}.csv. This gathers them into <out_dir>/_collected/all_*.csv and
reports which of the expected cells are missing -- it does NOT invent rows for
them, and it does not average anything. Aggregation is a separate, deliberate
step: per the project rule, results are reported per dataset x backbone,
averaging only over folds.

  python3 analysis/probe_elem_collect.py --out_dir $P/probe_elem_v1
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import pandas as pd

DATASETS = ['Alkane_Carbonyl_Verified_GT', 'BBBP', 'Benzene_Verified_GT',
            'Fluoride_Carbonyl_Verified_GT', 'hERG', 'Mutagenicity',
            'esol', 'Lipophilicity']
BACKBONES = ['GCN', 'GIN', 'GAT', 'SAGE', 'PNA']
FOLDS = [f'fold{i}' for i in range(5)]
KINDS = ['gap', 'perelem', 'composition']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--datasets', default=None,
                    help='space- or comma-separated subset (default: all 8)')
    a = ap.parse_args()
    out = Path(a.out_dir)
    dss = (DATASETS if not a.datasets
           else [d for d in a.datasets.replace(',', ' ').split() if d])

    coll = out / '_collected'
    coll.mkdir(parents=True, exist_ok=True)

    for kind in KINDS:
        frames = []
        for ds in dss:
            for f in sorted(glob.glob(str(out / ds / f'*_{kind}.csv'))):
                frames.append(pd.read_csv(f))
        if not frames:
            print(f'{kind:12s} no files')
            continue
        df = pd.concat(frames, ignore_index=True)
        dest = coll / f'all_{kind}.csv'
        df.to_csv(dest, index=False)
        print(f'{kind:12s} {len(df):7d} rows from {len(frames):4d} files -> {dest}')

    # ---- completeness, stated explicitly rather than silently tolerated -----
    missing = [(ds, bb, fd) for ds in dss for bb in BACKBONES for fd in FOLDS
               if not (out / ds / f'{bb}_{fd}.done').exists()]
    print(f'\ncells expected {len(dss)*len(BACKBONES)*len(FOLDS)}  '
          f'done {len(dss)*len(BACKBONES)*len(FOLDS) - len(missing)}  '
          f'MISSING {len(missing)}')
    if missing:
        by_ds = {}
        for ds, bb, fd in missing:
            by_ds.setdefault(ds, []).append(f'{bb}/{fd}')
        for ds, cells in sorted(by_ds.items()):
            print(f'  {ds:32s} {len(cells):3d}  {" ".join(cells[:8])}'
                  + (' ...' if len(cells) > 8 else ''))
        print('\nRe-run `probe_elem_launch.sh` to retry only these '
              '(.done markers make it idempotent).')


if __name__ == '__main__':
    main()
