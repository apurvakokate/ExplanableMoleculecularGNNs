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
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ANALYSIS = REPO / 'analysis'
DEFAULT_METRICS = ['auc', 'val_auc', 'train_auc', 'gt_roc_auc_mean',
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
    if args.processed_root:
        cmd += ['--processed_root', args.processed_root]
    if getattr(args, 'families', None):
        cmd += ['--families', *args.families]
    if getattr(args, 'dry_run', False):
        cmd += ['--dry_run']
    print('\n=== regenerate eval metrics from checkpoints ===')
    return subprocess.run(cmd).returncode


def step_collect(args) -> int:
    """Rebuild all_results.csv from summary.json files (mirrors shell collect)."""
    import json
    import pandas as pd
    out_root = Path(args.out_root)
    print('\n=== collect summaries -> all_results.csv ===')
    rows = []
    for p in out_root.rglob('summary.json'):
        try:
            d = json.load(open(p))
            d['exp_dir'] = str(p.parent.relative_to(out_root))
            rows.append(d)
        except Exception:
            pass
    if not rows:
        print('  no summary.json files found.')
        return 1
    df = pd.DataFrame(rows)
    core = [c for c in ['exp_dir', 'dataset', 'backbone', 'vocab_variant',
                        'motif_method', 'noise', 'info_loss_coef',
                        'ent_reg', 'size_reg', 'num_layers', 'explainer_lr', 'gnn_lr', 'conv_normalize', 'gin_inner_bn',
                        'train_auc', 'val_auc', 'auc', 'gt_roc_auc_mean',
                        'pearson', 'spearman',
                        'top_k_abs_disc', 'mean_abs_disc', 'score_disc_spearman',
                        'score_min', 'score_max', 'score_mean', 'score_std',
                        'score_median', 'score_mode', 'score_count'] if c in df]
    extra = sorted(c for c in df.columns if c not in core and any(
        c.startswith(p) for p in ('gnnexplainer_', 'pgexplainer_', 'mage_')))
    want = core + extra
    out = df[want].sort_values(['dataset', 'exp_dir'])
    dest = out_root / 'all_results.csv'
    out.to_csv(dest, index=False)
    print(f'  wrote {dest}  ({len(out)} rows, {len(want)} cols)')
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

    def common(p, need_train=False):
        p.add_argument('--out_root', required=True)
        p.add_argument('--save_dir', default=None)
        if need_train:
            p.add_argument('--data_root', default=None)
            p.add_argument('--vocab_root', default=None)
            p.add_argument('--processed_root', default=None)
            p.add_argument('--families', nargs='*',
                           default=['mose', 'motifsat', 'vanilla'])
            p.add_argument('--dry_run', action='store_true')

    p_re = sub.add_parser('regenerate', help='eval-only on existing checkpoints')
    common(p_re, need_train=True)

    p_co = sub.add_parser('collect', help='rebuild all_results.csv')
    common(p_co)

    p_tb = sub.add_parser('table', help='pivot tables per metric')
    common(p_tb)
    p_tb.add_argument('--csv', default=None)
    p_tb.add_argument('--metrics', nargs='*', default=None)

    p_pl = sub.add_parser('plots', help='score-vs-impact grid + counts')
    common(p_pl)
    p_pl.add_argument('--group', default='family')
    p_pl.add_argument('--facet', default='variant')
    p_pl.add_argument('--nbins', type=int, default=6)

    p_all = sub.add_parser('all', help='regenerate -> collect -> table -> plots')
    common(p_all, need_train=True)
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
