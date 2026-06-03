#!/usr/bin/env python3
"""make_results_table.py — pivot all_results.csv to dataset×run rows, backbone cols.

Rows  : dataset × family × vocab_variant   ("dataset × run")
Cols  : backbone (GIN, GCN, GAT, SAGE, PNA)
Cells : mean ± std of the chosen metric over folds

Usage
-----
    python analysis/make_results_table.py all_results.csv --metric auc
    python analysis/make_results_table.py all_results.csv --metric auc --md out.md
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

BACKBONE_ORDER = ['GIN', 'GCN', 'GAT', 'SAGE', 'PNA']


def _cell(g: pd.Series) -> str:
    g = g.dropna()
    if len(g) == 0:
        return ''
    if len(g) == 1:
        return f'{g.mean():.3f}'
    return f'{g.mean():.3f} ± {g.std():.3f}'


def build(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    df = df.copy()
    # Family from motif_method (reliable) — exp_dir layout is inconsistent
    # (some paths start with the dataset, not the family).
    if 'motif_method' in df.columns:
        mm = df['motif_method'].fillna('none').astype(str)
        df['family'] = mm.map(lambda m: 'mose' if m == 'mose'
                              else ('motifsat' if m in ('readout', 'node_emb',
                                                        'motif_emb', 'loss')
                                    else 'vanilla'))
    elif 'family' not in df.columns:
        df['family'] = df['exp_dir'].str.split('/').str[0]
    piv = df.pivot_table(
        index=['dataset', 'family', 'vocab_variant'],
        columns='backbone', values=metric, aggfunc=_cell)
    cols = [b for b in BACKBONE_ORDER if b in piv.columns] + \
           [b for b in piv.columns if b not in BACKBONE_ORDER]
    return piv[cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--metric', default='auc',
                    help='Any numeric column in the CSV. Common: auc, val_auc, '
                         'train_auc, gt_roc_auc_mean, pearson, spearman, '
                         'top_k_abs_disc, score_disc_spearman; baseline columns '
                         'like gnnexplainer_mean_pearson also work.')
    ap.add_argument('--md', default=None, help='Optional markdown output path.')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.metric not in df.columns:
        raise SystemExit(f'metric "{args.metric}" not in {args.csv}. '
                         f'Available: {[c for c in df.columns]}')
    table = build(df, args.metric)
    print(f'\n{args.metric}  — rows: dataset × family × variant   '
          f'cols: backbone   (mean ± std over folds)\n')
    print(table.to_string())
    if args.md:
        Path(args.md).write_text(table.to_markdown())
        print(f'\nWrote {args.md}')


if __name__ == '__main__':
    main()
