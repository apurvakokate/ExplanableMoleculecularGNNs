#!/usr/bin/env python3
"""make_results_table.py — pivot all_results.csv to dataset×run rows, backbone cols.

Rows  : dataset × family × synthetic (real|gt) × vocab_base × is_filter
Cols  : backbone (GIN, GCN, GAT, SAGE, PNA)
Cells : mean ± std of the chosen metric over folds (single value when one fold, e.g. mutag)

The ``synthetic`` axis separates real-label runs from GT-relabelled training
(vocab ``*_relabelled``, summary ``use_gt`` for vanilla/baselines). ``vocab_base``
and ``is_filter`` split the bundled ``vocab_variant`` string (E10).

Usage
-----
    python analysis/make_results_table.py all_results.csv --metric auc
    python analysis/make_results_table.py all_results.csv --metric pearson --mode explanation
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
PIVOT_INDEX = ['dataset', 'family', 'synthetic', 'vocab_base', 'is_filter']

PREDICTION_METRICS = frozenset({
    'auc', 'val_auc', 'train_auc', 'rmse', 'mae', 'rmse_orig', 'mae_orig',
})

# Ante-hoc families that produce NODE-level scores: node GT-ROC differs under
# mean- vs max-pooling, so report both (like the post-hoc baselines). MOSE and
# MotifSAT produce MOTIF-level scores, so mean == max — they keep one value.
NODE_SCORING_FAMILIES = frozenset({'gsat'})
NODE_POOL_SRC = {'mean': 'gt_roc_node_mean_auc_mean', 'max': 'gt_roc_node_max_auc_mean'}

# Correlation metrics carry BOTH node->motif poolings (mean/max) as explicit
# summary columns, so they can be reported per-pooling like node GT-ROC. Every
# ante-hoc trainer records these (motif-level MOSE writes mean == max).
CORR_POOL_SRC = {
    'pearson':  {'mean': 'pearson_node_mean',  'max': 'pearson_node_max'},
    'spearman': {'mean': 'spearman_node_mean', 'max': 'spearman_node_max'},
    'pearson_all':  {'mean': 'pearson_node_mean_all',  'max': 'pearson_node_max_all'},
    'spearman_all': {'mean': 'spearman_node_mean_all', 'max': 'spearman_node_max_all'},
}
# Metrics that get split into separate per-pooling report tables (see step_table).
POOLED_TABLE_METRICS = frozenset(CORR_POOL_SRC)


def _expand_node_pooling(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Split ante-hoc rows into mean/max node->motif-pooled rows for *metric*.

    * ``gt_roc_node_auc_mean`` — only genuinely node-scoring families (GSAT)
      differ; motif-level MOSE/MotifSAT keep a single blank-agg row.
    * ``pearson`` / ``spearman`` — every ante-hoc family records both poolings
      (``*_node_mean`` / ``*_node_max``); split them all so each pooling has a
      row (MOSE's mean == max). Rows without the source columns stay blank-agg.
    """
    df = df.copy()
    if 'family' not in df.columns:
        df['explainer_agg'] = ''
        return df
    if metric == 'gt_roc_node_auc_mean':
        pool_src = NODE_POOL_SRC
        eligible = df['family'].astype(str).isin(NODE_SCORING_FAMILIES)
    elif metric in CORR_POOL_SRC:
        pool_src = CORR_POOL_SRC[metric]
        eligible = pd.Series(True, index=df.index)
    else:
        df['explainer_agg'] = ''
        return df
    have_cols = all(c in df.columns for c in pool_src.values())
    split = eligible & have_cols
    keep = df[~split].copy()
    keep['explainer_agg'] = ''
    if not split.any():
        return keep
    src = df[split]
    pieces = [keep]
    for agg, col in pool_src.items():
        sub = src.copy()
        sub['explainer_agg'] = agg
        sub[metric] = src[col]
        pieces.append(sub)
    return pd.concat(pieces, ignore_index=True)


def _cell(g: pd.Series) -> str:
    g = g.dropna()
    if len(g) == 0:
        return ''
    if len(g) == 1:
        return f'{g.mean():.3f}'
    return f'{g.mean():.3f} ± {g.std():.3f}'


def _warn_config_conflation(df: pd.DataFrame, metric: str, index: list) -> None:
    """Loudly flag pivot cells that would average >1 distinct run config (not just
    folds). ``_cell`` collapses every row sharing (index, backbone); those rows
    should differ only in fold. If a cell spans multiple ``config_sig`` values it
    is mixing hyperparameter configs — e.g. old IB-off and new IB-on MotifSAT."""
    if 'config_sig' not in df.columns or metric not in df.columns:
        return
    sub = df[[*index, 'backbone', 'config_sig', metric]].copy()
    sub = sub[pd.to_numeric(sub[metric], errors='coerce').notna()]
    if sub.empty:
        return
    bad = sub.groupby([*index, 'backbone'], dropna=False)['config_sig'].nunique()
    bad = bad[bad > 1]
    if len(bad):
        print(f'  [WARN] {len(bad)} cell(s) for metric {metric!r} average across '
              f'MULTIPLE run configs, not just folds — filter to one config '
              f'(e.g. --vocab_variant, or split by config_sig). First: '
              f'{tuple(bad.index[0])}')


def _pivot(df: pd.DataFrame, metric: str, index: list | None = None) -> pd.DataFrame:
    index = index or PIVOT_INDEX
    _warn_config_conflation(df, metric, index)
    piv = df.pivot_table(
        index=index, columns='backbone', values=metric, aggfunc=_cell)
    cols = [b for b in BACKBONE_ORDER if b in piv.columns] + \
           [b for b in piv.columns if b not in BACKBONE_ORDER]
    return piv[cols]


def select_pooling(table: pd.DataFrame, pool: str) -> pd.DataFrame:
    """Slice a pooled explanation table (``explainer_agg`` = last index level)
    down to one node->motif pooling. Keeps rows tagged ``pool`` AND blank-agg
    rows (motif-level, mean == max, belong in both files), then drops the
    now-redundant ``explainer_agg`` level."""
    names = list(table.index.names or [])
    if table.empty or 'explainer_agg' not in names:
        return table
    agg = table.index.get_level_values('explainer_agg').astype(str)
    sub = table[(agg == pool) | (agg == '')]
    if sub.empty:
        return sub
    return sub.droplevel('explainer_agg')


def build(df: pd.DataFrame, metric: str, *, mode: str = 'auto') -> pd.DataFrame:
    """Pivot rows for *metric*.

    ``mode``:
      * ``prediction`` — ante-hoc families only; drops duplicate baselines (E8)
      * ``explanation`` — ante-hoc + post-hoc explainer rows from baselines (E5)
      * ``auto`` — picks prediction vs explanation from *metric* name
    """
    from analysis.aggregate_experiments import (
        expand_posthoc_explainer_rows,
        filter_prediction_rows,
    )

    if mode == 'auto':
        mode = 'prediction' if metric in PREDICTION_METRICS else 'explanation'

    # Single field-based normalizer — sets family/synthetic/every axis from the
    # run's summary fields and fails fast on anything missing (no path fallback).
    from analysis.aggregate_experiments import normalize
    df = normalize(df)

    if mode == 'prediction':
        df = filter_prediction_rows(df)
        return _pivot(df, metric)

    # explanation: ante-hoc + post-hoc explainer rows. Post-hoc baselines carry
    # BOTH mean- and max-pooled scores; keep them as a distinct index level
    # (explainer_agg) so the two are reported separately, never averaged (E11).
    # ante-hoc: GSAT (node scores) splits into mean/max for node GT-ROC; motif
    # methods (MOSE/MotifSAT) stay single (mean == max).
    ante = _expand_node_pooling(filter_prediction_rows(df), metric)
    posthoc = expand_posthoc_explainer_rows(df)
    if posthoc.empty:
        df = ante
    else:
        df = pd.concat([ante, posthoc], ignore_index=True)
    if 'explainer_agg' not in df.columns:
        df['explainer_agg'] = ''
    # non-node-scoring rows have no pooling variant → blank level (single value)
    df['explainer_agg'] = df['explainer_agg'].fillna('').astype(str)
    return _pivot(df, metric, index=PIVOT_INDEX + ['explainer_agg'])


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
    ap.add_argument('--mode', choices=('auto', 'prediction', 'explanation'),
                    default='auto',
                    help='prediction drops baselines duplicates (E8); '
                         'explanation expands post-hoc explainer rows (E5)')
    ap.add_argument('--md', default=None, help='Optional markdown output path.')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.metric not in df.columns and args.mode != 'explanation':
        raise SystemExit(f'metric "{args.metric}" not in {args.csv}. '
                         f'Available: {[c for c in df.columns]}')
    table = build(df, args.metric, mode=args.mode)
    print(f'\n{args.metric} ({args.mode})  — rows: {" × ".join(PIVOT_INDEX)}   '
          f'cols: backbone   (mean ± std over folds; single value if one fold)\n')
    print(table.to_string())
    if args.md:
        Path(args.md).write_text(table.to_markdown())
        print(f'\nWrote {args.md}')


if __name__ == '__main__':
    main()
