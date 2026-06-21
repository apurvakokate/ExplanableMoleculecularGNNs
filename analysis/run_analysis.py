#!/usr/bin/env python3
"""run_analysis.py — single entry point for all ChemIntuit analysis & plots.

One command to (re)generate metrics, tables, and figures from an experiment
output tree. Wraps the individual analysis modules:

  regenerate   re-run eval-only on existing checkpoints (no retraining), so the
               new explainability metrics land in each run's summary.json + CSVs
  collect      rebuild <out_root>/all_results.csv from all summary.json files
  table        pivot all_results.csv -> dataset×family×variant rows, backbone
               cols (mean±std), written as markdown per metric
  plots        score-vs-impact box-plot grid + per-bin motif-count table
  all          regenerate -> collect -> table -> plots, in order

Subcommands
-----------
    python analysis/run_analysis.py all \
        --out_root results --data_root $DATA_ROOT --vocab_root $VOCAB_ROOT \
        [--processed_root $PROCESSED_ROOT]

    python analysis/run_analysis.py table  --out_root results
    python analysis/run_analysis.py plots  --out_root results
    python analysis/run_analysis.py regenerate --out_root results \
        --data_root $DATA_ROOT --vocab_root $VOCAB_ROOT [--dry_run]
    python analysis/run_analysis.py collect --out_root results

Notes
-----
* `regenerate` requires --data_root/--vocab_root (and ideally --processed_root).
  Pair each checkpoint with the vocab it was TRAINED on.
* The masked-node probe needs a live model in memory, so it is intentionally NOT
  part of this batch CLI; import it instead:
      from analysis.probe_masked_nodes import probe_run
      probe_run(model, test_list, device)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ANALYSIS = REPO / 'analysis'
DEFAULT_METRICS = ['auc', 'val_auc', 'train_auc',
                   'rmse', 'mae', 'rmse_orig', 'mae_orig',
                   'gt_roc_auc_mean', 'gt_roc_node_auc_mean', 'gt_roc_edge_auc_mean',
                   'gt_roc_n_graphs',
                   # Node attention reduced to motif level by mean / max.
                   'gt_roc_node_mean_auc_mean', 'gt_roc_node_max_auc_mean',
                   # Post-hoc baseline GT-ROC (node level): per explainer × agg.
                   'gnnexplainer_mean_gt_roc_node_auc_mean',
                   'gnnexplainer_max_gt_roc_node_auc_mean',
                   'pgexplainer_mean_gt_roc_node_auc_mean',
                   'pgexplainer_max_gt_roc_node_auc_mean',
                   'mage_mean_gt_roc_node_auc_mean',
                   'mage_max_gt_roc_node_auc_mean',
                   'pearson', 'spearman', 'top_k_abs_disc',
                   'score_disc_spearman']


# ── individual steps ──────────────────────────────────────────────────────────

def step_regenerate(args) -> int:
    if not (args.data_root and args.vocab_root):
        print('[regenerate] needs --data_root and --vocab_root; skipping.')
        return 1
    cmd = [sys.executable, str(ANALYSIS / 'regenerate_eval.py'),
           '--out_root', args.out_root,
           '--data_root', args.data_root, '--vocab_root', args.vocab_root]
    if getattr(args, 'mutag_data_root', None):
        cmd += ['--mutag_data_root', args.mutag_data_root]
    if getattr(args, 'ogb_data_root', None):
        cmd += ['--ogb_data_root', args.ogb_data_root]
    if args.processed_root:
        cmd += ['--processed_root', args.processed_root]
    if getattr(args, 'families', None):
        cmd += ['--families', *args.families]
    if getattr(args, 'dry_run', False):
        cmd += ['--dry_run']
    print('\n=== regenerate eval metrics from checkpoints ===')
    return subprocess.run(cmd).returncode


# Column lists inlined in step_collect (below); these were unused duplicates.
# AXIS_COLS = ['family', 'fragmentation', 'threshold', 'synthetic',
#              'norm', 'features', 'injection', 'epochs', 'fold']
# _CONFIG_EXTRA = ['schema', 'encoder_norm', 'weight_vocab_variant', 'seed']


def step_collect(args) -> int:
    """Rebuild all_results.csv from summary.json files.

    For each run we merge (when present) the canonical config.json written by
    run_experiments.py, then fill any axis columns still missing by parsing the
    path with aggregate_experiments.normalize. New canonical runs therefore get
    exact axis columns; legacy runs are decoded from their directory tokens.
    """
    import json
    import pandas as pd
    from analysis.aggregate_experiments import normalize, ALL_AXES, iter_summaries
    from SharedModules.data.dataset_routing import collapse_redundant_folds
    out_root = Path(args.out_root)
    print('\n=== collect summaries -> all_results.csv ===')
    rows = []
    n_cfg = 0
    for p in iter_summaries(out_root, getattr(args, 'exclude', None)):
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f'  [warn] skip corrupt summary {p}: {e}')
            continue
        # Merge the sibling config.json (canonical axes) without letting it clobber
        # the measured metrics in summary.json.
        cfg_path = p.parent / 'config.json'
        if cfg_path.exists():
            try:
                cfg = json.load(open(cfg_path))
                for k, v in cfg.items():
                    d.setdefault(k, v)
                n_cfg += 1
            except Exception as e:
                print(f'  [warn] skip corrupt config {cfg_path}: {e}')
                pass
        d['exp_dir'] = str(p.parent.relative_to(out_root))
        rows.append(d)
    if not rows:
        print('  no summary.json files found.')
        return 1
    df = pd.DataFrame(rows)
    keep = getattr(args, 'vocab_variant', None)
    if keep:
        before = len(df)
        vv = df.get('vocab_variant', pd.Series([''] * len(df))).astype(str)
        df = df[vv.isin(set(keep))].copy()
        print(f'  filtered to vocab_variant in {sorted(set(keep))}: '
              f'{len(df)}/{before} rows')
        if df.empty:
            print('  no rows after vocab_variant filter.')
            return 1
    # Fill/derive the canonical axis columns (prefers explicit config.json values,
    # falls back to path parsing for legacy runs).
    df = normalize(df)
    df = collapse_redundant_folds(df)

    core = [c for c in ['exp_dir', 'family', 'dataset', 'backbone', 'vocab_variant',
                        *ALL_AXES, 'fold',
                        'motif_method', 'noise', 'info_loss_coef',
                        'ent_reg', 'size_reg', 'num_layers', 'explainer_lr', 'gnn_lr',
                        'conv_normalize', 'gin_inner_bn',
                        'loader_kind', 'processed_root', 'data_root',
                        'w_feat', 'w_message', 'w_readout',
                        'mutag_index_maps_path', 'mutag_smiles_csv_path',
                        'mutag_splits_path', 'mutag_seed',
                        'encoder_norm', 'weight_vocab_variant', 'seed',
                        'train_auc', 'val_auc', 'auc', 'rmse', 'mae',
                        'rmse_orig', 'mae_orig',
                        'gt_roc_auc_mean', 'gt_roc_node_auc_mean', 'gt_roc_edge_auc_mean',
                        'gt_roc_n_graphs',
                        'gt_roc_node_mean_auc_mean', 'gt_roc_node_max_auc_mean',
                        'gnnexplainer_mean_gt_roc_node_auc_mean',
                        'gnnexplainer_max_gt_roc_node_auc_mean',
                        'pgexplainer_mean_gt_roc_node_auc_mean',
                        'pgexplainer_max_gt_roc_node_auc_mean',
                        'mage_mean_gt_roc_node_auc_mean',
                        'mage_max_gt_roc_node_auc_mean',
                        'pearson', 'spearman',
                        'top_k_abs_disc', 'mean_abs_disc', 'score_disc_spearman',
                        'score_min', 'score_max', 'score_mean', 'score_std',
                        'score_median', 'score_mode', 'score_count'] if c in df]
    # de-dup while preserving order
    seen = set(); core = [c for c in core if not (c in seen or seen.add(c))]
    extra = sorted(c for c in df.columns if c not in core and any(
        c.startswith(p) for p in ('gnnexplainer_', 'pgexplainer_', 'mage_')))
    want = core + extra
    out = df[want].sort_values(['dataset', 'exp_dir'])
    dest = out_root / 'all_results.csv'
    out.to_csv(dest, index=False)
    print(f'  wrote {dest}  ({len(out)} rows, {len(want)} cols; '
          f'{n_cfg} with config.json)')
    return 0


def step_table(args) -> int:
    import pandas as pd
    from analysis.make_results_table import build
    out_root = Path(args.out_root)
    csv = Path(args.csv) if args.csv else out_root / 'all_results.csv'
    if not csv.exists():
        print(f'[table] {csv} not found — run collect first.')
        return 1
    df = pd.read_csv(csv)
    save_dir = Path(args.save_dir) if args.save_dir else out_root / 'tables'
    save_dir.mkdir(parents=True, exist_ok=True)
    print('\n=== results tables (dataset×family×variant rows, backbone cols) ===')
    metrics = args.metrics or [m for m in DEFAULT_METRICS if m in df.columns]
    for metric in metrics:
        if metric not in df.columns:
            print(f'  [skip] metric {metric} not in CSV')
            continue
        tbl = build(df, metric)
        md = save_dir / f'results_table_{metric}.md'
        try:
            md.write_text(tbl.to_markdown())
        except Exception:
            md.write_text(tbl.to_string())
        print(f'  wrote {md}')
    return 0


def step_plots(args) -> int:
    cmd = [sys.executable, str(ANALYSIS / 'plot_score_vs_impact.py'),
           '--out_root', args.out_root,
           '--group', args.group, '--facet', args.facet,
           '--nbins', str(args.nbins)]
    if args.save_dir:
        cmd += ['--save_dir', args.save_dir]
    print('\n=== score-vs-impact plots + count table ===')
    return subprocess.run(cmd).returncode


# ── dispatch ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Single entry point for ChemIntuit analysis & plots.')
    sub = ap.add_subparsers(dest='command', required=True)

    def path_args(p):
        p.add_argument('--out_root', required=True)
        p.add_argument('--save_dir', default=None)

    def collect_args(p):
        path_args(p)
        p.add_argument('--exclude', nargs='*', default=None,
                       help='extra directory-name prefixes to skip when walking '
                            '--out_root (archive/scratch dirs are always skipped).')
        p.add_argument('--vocab_variant', nargs='*', default=None,
                       help='collect ONLY these vocab variants, e.g. '
                            '--vocab_variant rbrics_old_filter (default: all).')

    def train_args(p):
        p.add_argument('--data_root', default=None)
        p.add_argument('--mutag_data_root', default=os.environ.get('MUTAG_DATA_ROOT'))
        p.add_argument('--ogb_data_root', default=os.environ.get('OGB_DATA_ROOT'))
        p.add_argument('--vocab_root', default=None)
        p.add_argument('--processed_root', default=None)
        p.add_argument('--families', nargs='*',
                       default=['mose', 'motifsat', 'gsat', 'vanilla', 'baselines'])
        p.add_argument('--dry_run', action='store_true')

    p_re = sub.add_parser('regenerate', help='eval-only on existing checkpoints')
    path_args(p_re)
    train_args(p_re)

    p_co = sub.add_parser('collect', help='rebuild all_results.csv')
    collect_args(p_co)

    p_tb = sub.add_parser('table', help='pivot tables per metric')
    path_args(p_tb)
    p_tb.add_argument('--csv', default=None)
    p_tb.add_argument('--metrics', nargs='*', default=None)

    p_pl = sub.add_parser('plots', help='score-vs-impact grid + counts')
    path_args(p_pl)
    p_pl.add_argument('--group', default='family')
    p_pl.add_argument('--facet', default='variant')
    p_pl.add_argument('--nbins', type=int, default=6)

    p_all = sub.add_parser('all', help='regenerate -> collect -> table -> plots')
    collect_args(p_all)
    train_args(p_all)
    p_all.add_argument('--csv', default=None)
    p_all.add_argument('--metrics', nargs='*', default=None)
    p_all.add_argument('--group', default='family')
    p_all.add_argument('--facet', default='variant')
    p_all.add_argument('--nbins', type=int, default=6)
    p_all.add_argument('--skip_regenerate', action='store_true',
                       help='Use existing summaries; do not re-run eval.')

    args = ap.parse_args()

    if args.command == 'regenerate':
        sys.exit(step_regenerate(args))
    if args.command == 'collect':
        sys.exit(step_collect(args))
    if args.command == 'table':
        sys.exit(step_table(args))
    if args.command == 'plots':
        sys.exit(step_plots(args))
    if args.command == 'all':
        rc = 0
        if not args.skip_regenerate:
            if args.data_root and args.vocab_root:
                rc |= step_regenerate(args)
            else:
                print('[all] no --data_root/--vocab_root; skipping regenerate '
                      '(using existing summaries).')
        rc |= step_collect(args)
        rc |= step_table(args)
        rc |= step_plots(args)
        print('\n=== analysis complete ===')
        sys.exit(rc)


if __name__ == '__main__':
    main()
