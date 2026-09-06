#!/usr/bin/env python3
"""harvest.py — compute ALL per-cell metrics for ONE (dataset, fold).

One row per (preset, backbone) -> partial/<dataset>_f<fold>.csv:
    dataset, fold, preset, backbone, task_value, gp_full, gp_filt, gtroc_full, gtroc_filt

The expensive dataset reload (get_loaders, only for source-GT datasets / GT-ROC)
happens ONCE here and is reused across all 45 preset×backbone cells. Fully
parallel across (dataset, fold): 8×5 = 40 independent tasks. Idempotent — writes
its partial atomically; re-running overwrites.
"""
from __future__ import annotations
import os
import argparse
from pathlib import Path

import pandas as pd

import common as C


def harvest_fold(ds, fold, runs, vocab_root, data_root, processed_root):
    kept = C.kept_ids(ds, fold, vocab_root, data_root)
    split_lists = None
    if ds in C.SOURCE:
        split_lists = C.load_fold_split_lists(ds, fold, vocab_root, data_root, processed_root)

    rows = []
    for preset in C.PRESETS:
        stem = C.PRESET_STEM[preset]
        for bb in C.BACKBONES:
            d = C.cell_dir(runs, preset, ds, fold, bb)
            if not (d / 'summary_splits.json').exists() and not (d / 'explainer_importances.json').exists():
                continue
            task = C.read_task_perf(d, stem, ds)
            gp_full, gp_filt = C.grouped_pearson(d, stem, kept)
            if split_lists is not None:
                gt_full, gt_filt = C.gtroc_from_atts(d, stem, split_lists, kept)
            else:
                gt_full = gt_filt = float('nan')
            rows.append(dict(dataset=ds, fold=int(fold), preset=preset, backbone=bb,
                             task_value=task, gp_full=gp_full, gp_filt=gp_filt,
                             gtroc_full=gt_full, gtroc_filt=gt_filt))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--runs', required=True)
    ap.add_argument('--vocab_root', required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--processed_root', required=True)
    ap.add_argument('--out', required=True, help='partial output dir')
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = harvest_fold(a.dataset, a.fold, a.runs, a.vocab_root, a.data_root, a.processed_root)
    dest = out / f'{a.dataset}_f{a.fold}.csv'
    tmp = dest.with_suffix('.csv.tmp')
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, dest)
    print(f'{a.dataset} fold{a.fold}: {len(rows)} cells -> {dest.name}')


if __name__ == '__main__':
    main()
