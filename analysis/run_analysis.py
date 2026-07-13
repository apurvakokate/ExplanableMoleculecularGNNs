#!/usr/bin/env python3
"""run_analysis.py — single entry point for all ChemIntuit analysis & plots.

By default, ``all`` only **collects** existing summary.json files and writes
tables/plots — it does not load checkpoints, retrain, or regenerate metrics.

  collect      rebuild <out_root>/all_results.csv from all summary.json files
  table        pivot all_results.csv -> dataset×family×variant rows, backbone
               cols (mean±std), written as markdown per metric
  plots        score-vs-impact box-plot grid + per-bin motif-count table
  all          collect -> table -> plots  (default; no model I/O)

Optional (explicit subcommands / flags — never run by default in ``all``):

  regenerate   re-run ``--eval_only`` on existing MOSE/MotifSAT/GSAT checkpoints
  multi_explanation  post-hoc H0/H1/H2 on MOSE / MotifSAT / GSAT
  probe          masked-node feature-recovery probe on ante-hoc checkpoints

Subcommands
-----------
    # Default: collect + tables + plots only
    python analysis/run_analysis.py all --out_root results \\
        --extra_out_root results_motifsat_ib

    # Opt-in: refresh eval metrics on existing ante-hoc checkpoints first
    python analysis/run_analysis.py all --out_root results --regenerate \\
        --data_root $DATA_ROOT --vocab_root $VOCAB_ROOT

    python analysis/run_analysis.py table  --out_root results
    python analysis/run_analysis.py plots  --out_root results
    python analysis/run_analysis.py regenerate --out_root results \\
        --data_root $DATA_ROOT --vocab_root $VOCAB_ROOT [--dry_run]
    python analysis/run_analysis.py collect --out_root results

Notes
-----
* ``--regenerate`` never creates missing runs or trains models. It only re-runs
  eval on dirs that already have ``best_model.pt`` + ``summary.json``.
  Missing baselines/ → ``bash run_experiments.sh phase5_baselines``.
* To include vanilla/baselines in regenerate (re-fits post-hoc explainers):
  ``--regenerate --families mose motifsat gsat vanilla baselines``
* ``multi_explanation`` and ``probe`` are separate subcommands, not part of ``all``.
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

# Identity / config columns — never treat as pivot metrics.
_NON_METRIC_COLS = frozenset({
    'fold', 'seed', 'mutag_seed', 'num_layers', 'gt_roc_n_graphs', 'epochs',
    'info_loss_coef', 'motif_loss_coef', 'within_node_coef', 'between_motif_coef',
    'ent_reg', 'size_reg', 'explainer_lr', 'gnn_lr', 'hidden_dim',
    'init_r', 'final_r', 'decay_interval', 'decay_r', 'learn_edge_att',
    'w_feat', 'w_message', 'w_readout', 'gin_inner_bn', 'score_count',
})


_EXPLAINER_PREFIXES = ('gnnexplainer_', 'pgexplainer_', 'mage_')


def discover_table_metrics(df) -> list[str]:
    """Metrics to pivot into tables.

    Two deliberate policies:

    * **No dedicated per-explainer tables.** Post-hoc explainers (GNNExplainer /
      PGExplainer / MAGE) already surface as FAMILY ROWS inside every generic
      metric table (via ``build``'s ``expand_posthoc``), so the redundant
      ``{explainer}_{agg}_*`` columns are never tabled on their own.
    * **Regression performance in ORIGINAL units.** ``rmse_orig``/``mae_orig``
      (predictions inverse-transformed to the target's real scale) are the
      reported regression metrics; the normalised ``rmse``/``mae`` are tabled
      only as a fallback when no denormalised counterpart exists (i.e. the model
      was trained without target normalisation).
    """
    import pandas as pd
    from analysis.aggregate_experiments import DEFAULT_REPORT_METRICS, PERF

    metrics: list[str] = []
    seen: set[str] = set()
    for m in DEFAULT_REPORT_METRICS:
        if m == PERF:
            perf_cols = ['auc', 'rmse_orig', 'mae_orig']
            # Normalised rmse/mae only when the original-scale version is absent.
            if 'rmse_orig' not in df.columns:
                perf_cols.append('rmse')
            if 'mae_orig' not in df.columns:
                perf_cols.append('mae')
            for col in perf_cols:
                if col in df.columns and col not in seen:
                    metrics.append(col)
                    seen.add(col)
        elif m.startswith(_EXPLAINER_PREFIXES):
            continue  # per-explainer tables are redundant (see docstring)
        elif m in df.columns and m not in seen:
            metrics.append(m)
            seen.add(m)
    for c in sorted(df.columns):
        if c in seen or c in _NON_METRIC_COLS:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        # Auto-add bare score_* distribution stats and the diagnostic sidecar
        # families (topbot_* = top-K vs bottom-K impact; gtvo_* = GT vs non-GT
        # motif discrimination). Explainer-prefixed columns are excluded (no
        # dedicated per-explainer tables — they surface as family rows instead).
        if c.startswith(_EXPLAINER_PREFIXES):
            continue
        if c.startswith(('score_', 'topbot_', 'gtvo_')):
            metrics.append(c)
            seen.add(c)
    # Register the GENERIC topbot_/gtvo_ metric when only the per-explainer
    # (prefixed) columns exist — e.g. baselines were re-run but the ante-hoc
    # runs (which write the unprefixed columns via sidecar) were not. build()'s
    # expand_posthoc then surfaces the baseline rows for these tables.
    for c in df.columns:
        if not c.startswith(_EXPLAINER_PREFIXES):
            continue
        for pref in _EXPLAINER_PREFIXES:
            if c.startswith(pref):
                tail = c[len(pref):]
                for agg in ('mean_', 'max_'):
                    if tail.startswith(agg):
                        gen = tail[len(agg):]
                        if gen.startswith(('topbot_', 'gtvo_')) and gen not in seen:
                            metrics.append(gen)
                            seen.add(gen)
                        break
                break
    return metrics


# ── individual steps ──────────────────────────────────────────────────────────

def _datasets_arg(args) -> list[str] | None:
    return getattr(args, 'dataset', None) or None


def _fill_pooled_correlation(d: dict, run_dir) -> None:
    """Backfill ``{pearson,spearman}_node_{mean,max}`` for runs whose summary.json
    predates them. Prefer the sidecar ``correlation_att_{mean,max}.csv``
    (MotifSAT / GSAT write one per pooling); otherwise fall back to the headline
    ``pearson``/``spearman`` (motif-level runs where mean == max). New runs
    already carry the columns and are left untouched."""
    import math
    import pandas as pd

    def _present(v) -> bool:
        return v is not None and not (isinstance(v, float) and math.isnan(v))

    for base in ('pearson', 'spearman'):
        for agg in ('mean', 'max'):
            key = f'{base}_node_{agg}'
            if _present(d.get(key)):
                continue
            val = None
            side = run_dir / f'correlation_att_{agg}.csv'
            if side.exists():
                try:
                    sdf = pd.read_csv(side)
                    if base in sdf.columns and len(sdf):
                        val = float(sdf.iloc[0][base])
                except Exception:
                    val = None
            if not _present(val):
                val = d.get(base)  # motif-level headline (mean == max)
            if _present(val):
                d[key] = val


def _fill_diagnostic_sidecars(d: dict, run_dir) -> None:
    """Aggregate per-run diagnostic CSVs into flat summary columns so the analysis
    phase can table them WITHOUT re-running eval. Ante-hoc trainers
    (MOSE/MotifSAT/GSAT) write these next to summary.json:

      * ``top_bottom_summary.csv`` → ``topbot_*`` (top-K vs bottom-K impact:
        do the highest-scored motifs actually have the largest removal impact?)
      * ``gt_vs_outside.csv``      → ``gtvo_<subset>_*`` + subset-invariant
        ``gtvo_score_auc`` / ``gtvo_gt_impact_rank`` (do GT motifs outrank
        non-GT motifs in score and impact, over 3 example subsets).

    Uses setdefault so anything already promoted into summary.json wins.
    """
    import math
    import pandas as pd

    def _num(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return None if math.isnan(f) else f

    # top_bottom_summary.csv — one row of scalar summaries.
    tb = run_dir / 'top_bottom_summary.csv'
    if tb.exists():
        try:
            row = pd.read_csv(tb).iloc[0]
            for c in ('top_mean_score', 'bottom_mean_score', 'top_mean_impact',
                      'bottom_mean_impact', 'impact_ratio'):
                v = _num(row.get(c))
                if v is not None:
                    d.setdefault(f'topbot_{c}', v)
        except Exception:
            pass

    # gt_vs_outside.csv — one row per example subset.
    gv = run_dir / 'gt_vs_outside.csv'
    if gv.exists():
        try:
            g = pd.read_csv(gv)
            per_subset = ('gt_mean_impact', 'non_gt_mean_impact',
                          'gt_mean_score', 'non_gt_mean_score')
            for _, r in g.iterrows():
                subset = str(r.get('subset', '')).strip()
                if not subset:
                    continue
                for c in per_subset:
                    v = _num(r.get(c))
                    if v is not None:
                        d.setdefault(f'gtvo_{subset}_{c}', v)
            # score_auc / gt_impact_rank are subset-invariant — record once.
            for c in ('score_auc', 'gt_impact_rank'):
                if c in g.columns:
                    for _, r in g.iterrows():
                        v = _num(r.get(c))
                        if v is not None:
                            d.setdefault(f'gtvo_{c}', v)
                            break
        except Exception:
            pass

    # Baseline explainers: reconstruct per-explainer top_bottom from the saved
    # {ex}_{agg}_score_vs_impact.csv (score + per-motif impact_mean) WITHOUT
    # re-running the explainers — writes PREFIXED {ex}_{agg}_topbot_* so
    # expand_posthoc surfaces baseline rows in the topbot_* tables. (gt_vs_outside
    # needs the model + per-graph structure, so it is NOT reconstructable here —
    # use run_vanilla --reuse_explainer_scores for that.)
    # K for top-K vs bottom-K — use the run's recorded top_k (summary field) so
    # the reconstruction matches what the native run computed; default 10.
    _topbot_k = 10
    try:
        _topbot_k = int(d.get('top_k') or 10)
    except (TypeError, ValueError):
        _topbot_k = 10

    def _topbot_from_svi(path, k: int) -> dict | None:
        try:
            sv = pd.read_csv(path)
        except Exception:
            return None
        if 'method' in sv.columns:
            sv = sv[sv['method'].astype(str) == 'own']  # own-LOO impact (headline)
        if not {'motif_id', 'score'}.issubset(sv.columns):
            return None
        imp_col = ('impact_mean' if 'impact_mean' in sv.columns
                   else 'impact' if 'impact' in sv.columns else None)
        if imp_col is None or sv.empty:
            return None
        # score + impact_mean are constant per motif → one value each.
        scores, impacts = {}, {}
        for mid, sub in sv.dropna(subset=['motif_id']).groupby('motif_id'):
            s = _num(sub['score'].iloc[0]); imp = _num(sub[imp_col].iloc[0])
            if s is not None:
                scores[int(mid)] = s
            if imp is not None:
                impacts[int(mid)] = imp
        common = sorted(set(scores) & set(impacts))
        if len(common) < 2:
            return None
        ranked = sorted(common, key=lambda m: scores[m], reverse=True)
        k = min(k, len(ranked) // 2)               # top/bottom must not overlap
        if k < 1:
            return None
        top, bot = ranked[:k], ranked[-k:]
        _mean = lambda xs: sum(xs) / len(xs)
        tmi, bmi = _mean([impacts[m] for m in top]), _mean([impacts[m] for m in bot])
        return {
            'top_mean_score':     _mean([scores[m] for m in top]),
            'bottom_mean_score':  _mean([scores[m] for m in bot]),
            'top_mean_impact':    tmi,
            'bottom_mean_impact': bmi,
            'impact_ratio':       (tmi / bmi) if bmi > 1e-9 else float('nan'),
        }

    for ex in ('gnnexplainer', 'pgexplainer', 'mage'):
        for agg in ('mean', 'max'):
            if f'{ex}_{agg}_topbot_impact_ratio' in d:
                continue  # already promoted into summary.json (real / reuse run)
            svi = run_dir / f'{ex}_{agg}_score_vs_impact.csv'
            if not svi.exists():
                continue
            tb = _topbot_from_svi(svi, _topbot_k)
            if tb:
                for kk, vv in tb.items():
                    d.setdefault(f'{ex}_{agg}_topbot_{kk}', vv)


def _regenerate_families(args) -> list[str]:
    if getattr(args, 'families', None):
        return list(args.families)
    return ['mose', 'motifsat', 'gsat']


def step_regenerate(args) -> int:
    if not (args.data_root and args.vocab_root):
        print('[regenerate] needs --data_root and --vocab_root; skipping.')
        return 1
    fams = _regenerate_families(args)
    print(f'  regenerate families: {fams}')
    cmd = [sys.executable, str(ANALYSIS / 'regenerate_eval.py'),
           '--out_root', args.out_root,
           '--data_root', args.data_root, '--vocab_root', args.vocab_root,
           '--families', *fams]
    if getattr(args, 'mutag_data_root', None):
        cmd += ['--mutag_data_root', args.mutag_data_root]
    if getattr(args, 'ogb_data_root', None):
        cmd += ['--ogb_data_root', args.ogb_data_root]
    if args.processed_root:
        cmd += ['--processed_root', args.processed_root]
    if _datasets_arg(args):
        cmd += ['--dataset', *_datasets_arg(args)]
    if getattr(args, 'dry_run', False):
        cmd += ['--dry_run']
    print('\n=== regenerate eval metrics from checkpoints ===')
    return subprocess.run(cmd).returncode


# Column lists inlined in step_collect (below); these were unused duplicates.
# AXIS_COLS = ['family', 'fragmentation', 'threshold', 'synthetic',
#              'norm', 'features', 'injection', 'epochs', 'fold']
# _CONFIG_EXTRA = ['schema', 'encoder_norm', 'weight_vocab_variant', 'seed']


def _out_roots(args) -> list[Path]:
    """Primary --out_root plus optional --extra_out_root trees (e.g. results_motifsat_ib)."""
    roots = [Path(args.out_root)]
    extra = getattr(args, 'extra_out_root', None) or []
    for p in extra:
        ep = Path(p)
        if ep not in roots:
            roots.append(ep)
    return roots


def _collect_args_extra(p):
    p.add_argument('--extra_out_root', nargs='*', default=None,
                   help='additional result trees to merge into collect/plots '
                        '(e.g. results_motifsat_ib for IB MotifSAT reruns)')


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
    roots = _out_roots(args)
    print('\n=== collect summaries -> all_results.csv ===')
    rows = []
    n_cfg = 0
    datasets = _datasets_arg(args)
    if datasets:
        print(f'  dataset filter: {sorted(set(datasets))}')
    for root in roots:
        print(f'  scanning {root}')
        for p in iter_summaries(root, getattr(args, 'exclude', None), datasets):
            try:
                with open(p, encoding='utf-8') as f:
                    d = json.load(f)
            except Exception as e:
                print(f'  [warn] skip corrupt summary {p}: {e}')
                continue
            cfg_path = p.parent / 'config.json'
            if cfg_path.exists():
                try:
                    with open(cfg_path, encoding='utf-8') as f:
                        cfg = json.load(f)
                    for k, v in cfg.items():
                        d.setdefault(k, v)
                    n_cfg += 1
                except Exception as e:
                    print(f'  [warn] skip corrupt config {cfg_path}: {e}')
            _fill_pooled_correlation(d, p.parent)
            _fill_diagnostic_sidecars(d, p.parent)
            d['results_root'] = str(root)
            d['exp_dir'] = str(p.parent.relative_to(root))
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

    core = [c for c in ['results_root', 'exp_dir', 'config_sig',
                        'family', 'dataset', 'backbone', 'vocab_variant',
                        'vocab_base', 'is_filter', 'is_relabelled', 'use_gt',
                        'node_encoder',
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
                        'gnnexplainer_pearson_instance', 'gnnexplainer_spearman_instance',
                        'pgexplainer_pearson_instance', 'pgexplainer_spearman_instance',
                        'mage_pearson_instance', 'mage_spearman_instance',
                        'gnnexplainer_pearson_instance_agnostic', 'gnnexplainer_spearman_instance_agnostic',
                        'pgexplainer_pearson_instance_agnostic', 'pgexplainer_spearman_instance_agnostic',
                        'mage_pearson_instance_agnostic', 'mage_spearman_instance_agnostic',
                        'pearson', 'spearman',
                        'pearson_motif', 'spearman_motif',
                        'pearson_instance', 'spearman_instance',
                        'pearson_instance_all', 'spearman_instance_all',
                        'pearson_instance_agnostic', 'spearman_instance_agnostic',
                        'pearson_instance_agnostic_all', 'spearman_instance_agnostic_all',
                        'pearson_node_mean', 'pearson_node_max',
                        'spearman_node_mean', 'spearman_node_max',
                        'pearson_all', 'spearman_all',
                        'pearson_motif_all', 'spearman_motif_all',
                        'pearson_node_mean_all', 'pearson_node_max_all',
                        'spearman_node_mean_all', 'spearman_node_max_all',
                        'gt_roc_auc_mean_all', 'gt_roc_node_auc_mean_all',
                        'gt_roc_edge_auc_mean_all',
                        'gt_roc_node_mean_auc_mean_all',
                        'gt_roc_node_max_auc_mean_all',
                        'gt_roc_n_graphs_all',
                        'top_k_abs_disc', 'mean_abs_disc', 'score_disc_spearman',
                        'score_min', 'score_max', 'score_mean', 'score_std',
                        'score_median', 'score_mode', 'score_count'] if c in df]
    # de-dup while preserving order
    seen = set(); core = [c for c in core if not (c in seen or seen.add(c))]
    extra = sorted(c for c in df.columns if c not in core and any(
        c.startswith(p) for p in ('gnnexplainer_', 'pgexplainer_', 'mage_')))
    seen = set(core) | set(extra)
    rest = [c for c in df.columns if c not in seen]
    want = core + extra + rest
    out = df[want].sort_values(['dataset', 'exp_dir'])
    dest = out_root / 'all_results.csv'
    out.to_csv(dest, index=False)
    print(f'  wrote {dest}  ({len(out)} rows, {len(want)} cols; '
          f'{n_cfg} with config.json)')
    return 0


def step_table(args) -> int:
    import pandas as pd
    from analysis.make_results_table import (
        build, PREDICTION_METRICS, POOLED_TABLE_METRICS, select_pooling,
    )
    out_root = Path(args.out_root)
    csv = Path(args.csv) if args.csv else out_root / 'all_results.csv'
    if not csv.exists():
        print(f'[table] {csv} not found — run collect first.')
        return 1
    df = pd.read_csv(csv)
    datasets = _datasets_arg(args)
    if datasets:
        allowed = set(datasets)
        if 'dataset' in df.columns:
            before = len(df)
            df = df[df['dataset'].astype(str).isin(allowed)].copy()
            print(f'  dataset filter {sorted(allowed)}: {len(df)}/{before} rows')
        if df.empty:
            print('  no rows after dataset filter.')
            return 1
    save_dir = Path(args.save_dir) if args.save_dir else out_root / 'tables'
    save_dir.mkdir(parents=True, exist_ok=True)
    print('\n=== results tables (dataset×family×synthetic×vocab_base×filter rows) ===')
    metrics = args.metrics or discover_table_metrics(df)
    for metric in metrics:
        mode = 'prediction' if metric in PREDICTION_METRICS else 'explanation'
        if metric not in df.columns and mode == 'explanation':
            # Post-hoc metrics live on expanded explainer rows, not raw CSV cols.
            pass
        elif metric not in df.columns:
            print(f'  [skip] metric {metric} not in CSV')
            continue
        tbl = build(df, metric, mode=mode)
        if tbl.empty:
            print(f'  [skip] metric {metric}: no rows after pivot')
            continue
        suffix = '' if mode == 'auto' else f'_{mode}'

        def _write(table, path):
            try:
                path.write_text(table.to_markdown())
            except Exception:
                path.write_text(table.to_string())
            print(f'  wrote {path}  ({mode}, {len(table)} rows)')

        if metric in POOLED_TABLE_METRICS:
            # Split into separate per-pooling files (mean/max node->motif),
            # e.g. results_table_pearson_explanation_mean.md / _max.md.
            for pool in ('mean', 'max'):
                sub = select_pooling(tbl, pool)
                if sub.empty:
                    print(f'  [skip] metric {metric} ({pool}): no rows')
                    continue
                _write(sub, save_dir / f'results_table_{metric}{suffix}_{pool}.md')
        else:
            _write(tbl, save_dir / f'results_table_{metric}{suffix}.md')
    return 0


def step_multi_explanation(args) -> int:
    if not (args.data_root and args.vocab_root):
        print('[multi_explanation] needs --data_root and --vocab_root; skipping.')
        return 1
    cmd = [sys.executable, str(ANALYSIS / 'run_multi_explanation.py'),
           '--out_root', args.out_root,
           '--data_root', args.data_root, '--vocab_root', args.vocab_root]
    if _datasets_arg(args):
        cmd += ['--dataset', *_datasets_arg(args)]
    print('\n=== post-hoc multi-explanation (H0/H1/H2) ===')
    return subprocess.run(cmd).returncode


def step_probe(args) -> int:
    if not (args.data_root and args.vocab_root):
        print('[probe] needs --data_root and --vocab_root; skipping.')
        return 1
    cmd = [sys.executable, str(ANALYSIS / 'probe_masked_nodes.py'),
           '--out_root', args.out_root,
           '--data_root', args.data_root, '--vocab_root', args.vocab_root,
           '--save', 'masked_node_probe.csv']
    if _datasets_arg(args):
        cmd += ['--dataset', *_datasets_arg(args)]
    print('\n=== masked-node feature probe ===')
    return subprocess.run(cmd).returncode


def step_plots(args) -> int:
    # Emit the full matrix: {instance, global} aggregation × {own, agnostic}
    # baseline impact. Directories split by AGGREGATION (plots_instance/,
    # plots_global/); within each, own & agnostic figures coexist via distinct
    # filenames (..._own_..., ..._agnostic_...). Motif-aware models (MOSE/
    # MotifSAT/GSAT) have a single impact, so their figures are identical across
    # own/agnostic (harmless repeat). If --save_dir is given, the aggregation is
    # appended so instance/global never overwrite each other.
    rc = 0
    for impact_agg in ('instance', 'global'):
        for impact_kind in ('own', 'agnostic'):
            cmd = [sys.executable, str(ANALYSIS / 'plot_score_vs_impact.py'),
                   '--out_root', args.out_root,
                   '--nbins', str(args.nbins),
                   '--impact_agg', impact_agg,
                   '--impact_kind', impact_kind]
            extra = getattr(args, 'extra_out_root', None) or []
            if extra:
                cmd += ['--extra_out_root', *extra]
            if getattr(args, 'score_min', None) is not None:
                cmd += ['--score_min', str(args.score_min)]
            if getattr(args, 'score_max', None) is not None:
                cmd += ['--score_max', str(args.score_max)]
            if args.save_dir:
                cmd += ['--save_dir', f'{args.save_dir.rstrip("/")}_{impact_agg}']
            if _datasets_arg(args):
                cmd += ['--dataset', *_datasets_arg(args)]
            print(f'\n=== score-vs-impact plots ({impact_agg}, {impact_kind} impact) ===')
            rc = subprocess.run(cmd).returncode or rc
    return rc


# ── dispatch ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Single entry point for ChemIntuit analysis & plots.')
    sub = ap.add_subparsers(dest='command', required=True)

    def path_args(p):
        p.add_argument('--out_root', required=True)
        p.add_argument('--save_dir', default=None)

    def filter_args(p):
        p.add_argument('--dataset', nargs='*', default=None,
                       help='only include these dataset(s), e.g. --dataset mutag')

    def collect_args(p, *, with_dataset: bool = True):
        path_args(p)
        _collect_args_extra(p)
        p.add_argument('--exclude', nargs='*', default=None,
                       help='extra directory-name prefixes to skip when walking '
                            '--out_root (archive/scratch dirs are always skipped).')
        p.add_argument('--vocab_variant', nargs='*', default=None,
                       help='collect ONLY these vocab variants, e.g. '
                            '--vocab_variant rbrics_old_filter (default: all).')
        if with_dataset:
            filter_args(p)

    def train_args(p, *, with_dataset: bool = True):
        p.add_argument('--data_root', default=None)
        p.add_argument('--mutag_data_root', default=os.environ.get('MUTAG_DATA_ROOT'))
        p.add_argument('--ogb_data_root', default=os.environ.get('OGB_DATA_ROOT'))
        p.add_argument('--vocab_root', default=None)
        p.add_argument('--processed_root', default=None)
        p.add_argument('--families', nargs='*', default=None,
                       help='regenerate: checkpoint families (default when '
                            '--regenerate: mose motifsat gsat). Add vanilla '
                            'baselines to re-fit post-hoc explainers.')
        p.add_argument('--dry_run', action='store_true')
        if with_dataset:
            filter_args(p)

    p_re = sub.add_parser('regenerate', help='eval-only on existing checkpoints')
    path_args(p_re)
    train_args(p_re)

    p_co = sub.add_parser('collect', help='rebuild all_results.csv')
    collect_args(p_co)

    p_tb = sub.add_parser('table', help='pivot tables per metric')
    path_args(p_tb)
    p_tb.add_argument('--csv', default=None)
    p_tb.add_argument('--metrics', nargs='*', default=None)
    filter_args(p_tb)

    p_me = sub.add_parser('multi_explanation', help='post-hoc H0/H1/H2 analysis')
    path_args(p_me)
    train_args(p_me)

    p_pr = sub.add_parser('probe', help='masked-node feature-recovery probe')
    path_args(p_pr)
    train_args(p_pr)

    p_pl = sub.add_parser('plots', help='score-vs-impact grid + counts')
    path_args(p_pl)
    _collect_args_extra(p_pl)
    filter_args(p_pl)
    p_pl.add_argument('--nbins', type=int, default=6)
    p_pl.add_argument('--score_min', type=float, default=None)
    p_pl.add_argument('--score_max', type=float, default=None)

    p_all = sub.add_parser('all',
                           help='collect -> table -> plots (default; no model I/O)')
    collect_args(p_all, with_dataset=False)
    train_args(p_all, with_dataset=False)
    filter_args(p_all)
    p_all.add_argument('--regenerate', action='store_true',
                       help='Before collect: re-run eval-only on existing '
                            'MOSE/MotifSAT/GSAT checkpoints (requires '
                            '--data_root and --vocab_root). Default off.')
    p_all.add_argument('--csv', default=None)
    p_all.add_argument('--metrics', nargs='*', default=None)
    p_all.add_argument('--nbins', type=int, default=6)
    p_all.add_argument('--score_min', type=float, default=None)
    p_all.add_argument('--score_max', type=float, default=None)

    args = ap.parse_args()

    if args.command == 'regenerate':
        sys.exit(step_regenerate(args))
    if args.command == 'multi_explanation':
        sys.exit(step_multi_explanation(args))
    if args.command == 'probe':
        sys.exit(step_probe(args))
    if args.command == 'collect':
        sys.exit(step_collect(args))
    if args.command == 'table':
        sys.exit(step_table(args))
    if args.command == 'plots':
        sys.exit(step_plots(args))
    if args.command == 'all':
        rc = 0
        if getattr(args, 'regenerate', False):
            if args.data_root and args.vocab_root:
                rc |= step_regenerate(args)
            else:
                print('[all] --regenerate requires --data_root and --vocab_root.')
                rc |= 1
        rc |= step_collect(args)
        rc |= step_table(args)
        rc |= step_plots(args)
        print('\n=== analysis complete ===')
        sys.exit(rc)


if __name__ == '__main__':
    main()
