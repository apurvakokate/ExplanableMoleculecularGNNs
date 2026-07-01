#!/usr/bin/env python3
"""make_results_table.py — pivot all_results.csv to dataset×run rows, backbone cols.

Rows  : dataset × family × synthetic (real|gt) × vocab_variant
Cols  : backbone (GIN, GCN, GAT, SAGE, PNA)
Cells : mean ± std of the chosen metric over folds

The ``synthetic`` axis separates real-label runs from GT-relabelled training
(MOSE/MotifSAT ``*_relabelled`` variants, vanilla/baselines ``*_gt`` dirs).
Without it, headline AUC mixes incompatible label targets.

Usage
-----
    python analysis/make_results_table.py all_results.csv --metric auc
    python analysis/make_results_table.py all_results.csv --metric auc --md out.md
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

BACKBONE_ORDER = ['GIN', 'GCN', 'GAT', 'SAGE', 'PNA']


def _cell(g: pd.Series) -> str:
    g = g.dropna()
    if len(g) == 0:
        return ''
    if len(g) == 1:
        return f'{g.mean():.3f}'
    return f'{g.mean():.3f} ± {g.std():.3f}'


def _ensure_family(df: pd.DataFrame) -> pd.DataFrame:
    """Fill ``family`` only when missing — never overwrite path-derived labels."""
    from analysis.aggregate_experiments import _family, resolve_family

    df = df.copy()
    if 'family' in df.columns:
        fam = df['family'].fillna('').astype(str).str.strip()
        if fam.ne('').all():
            return df
        need = fam.eq('')
    else:
        need = pd.Series(True, index=df.index)
        df['family'] = ''

    exp = df.get('exp_dir', pd.Series([''] * len(df))).astype(str)
    for idx in df.index[need]:
        row = df.loc[idx]
        meta = row.to_dict()
        df.at[idx, 'family'] = resolve_family(meta, exp.at[idx])

    still = df['family'].fillna('').astype(str).str.strip().eq('')
    if still.any() and 'motif_method' in df.columns:
        mm = df.loc[still, 'motif_method'].fillna('none').astype(str)
        df.loc[still, 'family'] = mm.map(
            lambda m: 'mose' if m == 'mose'
            else ('motifsat' if m in ('readout', 'loss') else 'vanilla'))

    if 'exp_dir' in df.columns:
        blank = df['family'].fillna('').astype(str).str.strip().eq('')
        df.loc[blank, 'family'] = df.loc[blank, 'exp_dir'].map(_family)

    return df


def _ensure_synthetic(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ``synthetic`` is ``real`` or ``gt`` (never blank / mixed in pivot)."""
    from analysis.aggregate_experiments import normalize

    df = df.copy()
    if 'exp_dir' not in df.columns:
        df['exp_dir'] = ''
    df = normalize(df)
    syn = df.get('synthetic', pd.Series([''] * len(df))).fillna('').astype(str).str.strip()
    if 'use_gt' in df.columns:
        gt_flag = df['use_gt'].astype(str).str.lower().isin(('true', '1', 'yes'))
        syn = syn.where(~gt_flag, 'gt')
    vv = df.get('vocab_variant', pd.Series([''] * len(df))).astype(str)
    syn = syn.where(~vv.str.endswith('_relabelled'), 'gt')
    df['synthetic'] = syn.replace('', 'real')
    return df


def build(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    df = _ensure_family(df)
    df = _ensure_synthetic(df)
    piv = df.pivot_table(
        index=['dataset', 'family', 'synthetic', 'vocab_variant'],
        columns='backbone', values=metric, aggfunc=_cell)
    cols = [b for b in BACKBONE_ORDER if b in piv.columns] + \
           [b for b in piv.columns if b not in BACKBONE_ORDER]
    return piv[cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--metric', default='auc',
                    help='Any numeric column in the CSV. Common: auc, val_auc, '
                         'train_auc, rmse_orig, mae_orig, gt_roc_auc_mean, '
                         'gt_roc_node_auc_mean, gt_roc_edge_auc_mean, '
                         'gt_roc_node_mean_auc_mean, gt_roc_node_max_auc_mean, '
                         'gt_roc_n_graphs, pearson, spearman, '
                         'top_k_abs_disc, score_disc_spearman; baseline columns '
                         'like gnnexplainer_mean_pearson and '
                         'gnnexplainer_mean_gt_roc_node_auc_mean (also _max_) work.')
    ap.add_argument('--md', default=None, help='Optional markdown output path.')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.metric not in df.columns:
        raise SystemExit(f'metric "{args.metric}" not in {args.csv}. '
                         f'Available: {[c for c in df.columns]}')
    table = build(df, args.metric)
    print(f'\n{args.metric}  — rows: dataset × family × synthetic × variant   '
          f'cols: backbone   (mean ± std over folds)\n')
    print(table.to_string())
    if args.md:
        Path(args.md).write_text(table.to_markdown())
        print(f'\nWrote {args.md}')


if __name__ == '__main__':
    main()
