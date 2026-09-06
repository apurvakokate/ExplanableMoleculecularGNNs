#!/usr/bin/env python3
"""build_tables.py — reduce the per-(dataset,fold) partial CSVs into the 3 tables.

Reads partial/*.csv (from harvest.py), aggregates mean±std over FOLDS only, and
emits (rows = Dataset × Backbone [× View], cols = 9 presets, highlighted):
    t1_auc.tex/.csv     classification AUC (×100, higher-better)
    t1_rmse.tex/.csv    regression RMSE_orig (original units, lower-better)
    t2_grouped_pearson  grouped Pearson, all datasets, Full+Filt
    t3_instance_gtroc   instance GT-ROC, source datasets, Full+Filt
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import common as C


def _load(partials):
    frames = [pd.read_csv(p) for p in sorted(Path(partials).glob('*.csv'))]
    if not frames:
        raise SystemExit(f'no partial CSVs in {partials}')
    return pd.concat(frames, ignore_index=True)


def _agg(df, ds, bb, preset, col):
    m = df[(df.dataset == ds) & (df.backbone == bb) & (df.preset == preset)]
    return C.agg([v for v in m[col].tolist() if v == v])


def _perf_rows(df, datasets, col, with_view=False, hb=True):
    """with_view=False -> Full-only (task perf). with_view=True -> Full+Filt from
    <col>_full / <col>_filt."""
    rows = []
    for ds in datasets:
        for bb in C.BACKBONES:
            if with_view:
                for view, cc in (('Full', f'{col}_full'), ('Filt', f'{col}_filt')):
                    cells = {p: _agg(df, ds, bb, p, cc) for p in C.PRESETS}
                    rows.append(((C.DS_LABEL[ds], bb, view), hb, cells))
            else:
                cells = {p: _agg(df, ds, bb, p, col) for p in C.PRESETS}
                rows.append(((C.DS_LABEL[ds], bb), hb, cells))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--partials', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = _load(a.partials)

    # T1 AUC (classification, ×100, higher-better)
    cls_present = [d for d in C.CLS if d in set(df.dataset)]
    if cls_present:
        rows = _perf_rows(df, cls_present, 'task_value', with_view=False, hb=True)
        C.emit(out / 't1_auc.tex', out / 't1_auc.csv', ['Dataset', 'Backbone'],
               rows, C.PRESETS, higher_better=True, scale=100.0, dp=1)
    # T1 RMSE (regression, original units, lower-better)
    reg_present = [d for d in C.REG if d in set(df.dataset)]
    if reg_present:
        rows = _perf_rows(df, reg_present, 'task_value', with_view=False, hb=False)
        C.emit(out / 't1_rmse.tex', out / 't1_rmse.csv', ['Dataset', 'Backbone'],
               rows, C.PRESETS, higher_better=False, scale=1.0, dp=3)
    # T2 grouped Pearson (all datasets, Full+Filt, higher-better)
    all_present = [d for d in C.DATASETS if d in set(df.dataset)]
    rows = _perf_rows(df, all_present, 'gp', with_view=True, hb=True)
    C.emit(out / 't2_grouped_pearson.tex', out / 't2_grouped_pearson.csv',
           ['Dataset', 'Backbone', 'View'], rows, C.PRESETS, higher_better=True, scale=100.0, dp=1)
    # T3 instance GT-ROC (source datasets, Full+Filt, higher-better)
    src_present = [d for d in C.SOURCE if d in set(df.dataset)]
    if src_present:
        rows = _perf_rows(df, src_present, 'gtroc', with_view=True, hb=True)
        C.emit(out / 't3_instance_gtroc.tex', out / 't3_instance_gtroc.csv',
               ['Dataset', 'Backbone', 'View'], rows, C.PRESETS, higher_better=True, scale=100.0, dp=1)

    print(f'done -> {out}')


if __name__ == '__main__':
    main()
