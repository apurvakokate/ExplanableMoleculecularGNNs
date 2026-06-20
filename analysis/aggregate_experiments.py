#!/usr/bin/env python3
"""aggregate_experiments.py — separate ChemIntuit results BY EXPERIMENT.

An "experiment" is one fixed combination of every config axis *except*
``architecture``, ``dataset`` and ``fold``:

    experiment = (fragmentation, threshold, synthetic_gt, norm, features,
                  injection, epochs)

Within an experiment the only things that vary are the backbone (rows),
the dataset (columns) and the fold (averaged over, with the folds that ran
recorded). Every model family (vanilla, baselines, mose, motifsat, gsat) is
placed in the same experiment so the table is directly comparable.

Because the vanilla GNN and its post-hoc baselines (GNNExplainer / PGExplainer
/ MAGE) do not depend on the injection axis, their rows are *broadcast* into
every injection variant that shares the remaining axes, so each per-experiment
table/plot is complete.

This module is robust to the THREE historical directory schemes that coexist in
``all_results.csv``:

  * priority sweep        ``A0_B0_C0/<family>/<dataset>/fold<k>/<variant_tag>``
  * phased pipeline       ``<family>/<variant>/<dataset>/fold<k>/<variant_tag>``
  * grid driver           ``<family>/.../enc-..._inj..._ep..._real/<dataset>/fold<k>/<variant_tag>``

Most axes are read from real CSV columns (dataset, backbone, vocab_variant,
conv_normalize); the few that only live in the path (features, injection,
synthetic, epochs) are parsed from ``exp_dir`` / the variant tag.

Usage
-----
    # from an existing combined CSV
    python analysis/aggregate_experiments.py --csv "all_results (2).csv" \
        --save_dir experiment_tables

    # or collect straight from an output tree first
    python analysis/aggregate_experiments.py --out_root results \
        --save_dir results/experiment_tables
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import importlib.util

_schema_path = _REPO / 'SharedModules' / 'data' / 'dataset_schema.py'
_spec = importlib.util.spec_from_file_location('dataset_schema', _schema_path)
_schema = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_schema)
TASK_TYPE = _schema.TASK_TYPE

FAMILIES = ('vanilla', 'baselines', 'mose', 'motifsat', 'gsat', 'base_gsat')
# Families whose identity does NOT include the injection axis. Their rows are
# broadcast across every injection value present for the ante-hoc families.
INJECTION_AGNOSTIC = {'vanilla', 'baselines'}

# every config axis we normalize (recorded as columns on every tidy row)
ALL_AXES = ['fragmentation', 'threshold', 'synthetic',
            'norm', 'features', 'injection', 'epochs']

# axes that define an experiment BY DEFAULT (everything but architecture/
# dataset/fold). Injection is intentionally excluded so that vanilla, MOSE,
# MotifSAT and GSAT (which carry different per-family injection defaults) all
# land in the SAME experiment table. Promote it via --experiment_axes when you
# are deliberately sweeping injection for a single family.
DEFAULT_EXPERIMENT_AXES = ['fragmentation', 'threshold', 'synthetic',
                           'norm', 'features', 'epochs']

REGRESSION_DATASETS = {
    ds for ds, task in TASK_TYPE.items() if task == 'Regression'
}

# Directory-name prefixes excluded from the results walk by default, so archived
# / scratch runs under <out_root> are not re-collected (see RESULTS_LAYOUT.md).
ARCHIVE_PREFIXES = ('_archive', '_trash', '_old')


def iter_summaries(root, extra_excludes=()):
    """Yield every summary.json under ``root`` whose relative path does NOT pass
    through an excluded directory (archive/scratch dirs by default)."""
    root = Path(root)
    excl = tuple(ARCHIVE_PREFIXES) + tuple(extra_excludes or ())
    for p in root.rglob('summary.json'):
        rel = p.relative_to(root)
        if any(part.startswith(excl) for part in rel.parts):
            continue
        yield p


# ── normalization helpers ──────────────────────────────────────────────────────

def _family(exp_dir: str) -> str:
    parts = [p for p in str(exp_dir).split('/') if p]
    for p in parts:
        if p in FAMILIES:
            return 'gsat' if p == 'base_gsat' else p
    return ''


def _fold(exp_dir: str):
    m = re.search(r'fold(\d+)', str(exp_dir))
    return int(m.group(1)) if m else None


def _priority_axes(exp_dir: str):
    """Decode the A{a}_B{b}_C{c} prefix of the priority sweep, if present.

    A = fragmentation (0=rbrics, 1=all_fallback_bpe), B = labels (0=real,1=gt),
    C = normalization (0=none, 1=l2).  Returns (synthetic, norm) overrides or
    (None, None) when the prefix is absent.
    """
    m = re.match(r'A(\d)_B(\d)_C(\d)', str(exp_dir))
    if not m:
        return None, None
    _, b, c = m.groups()
    synthetic = 'gt' if b == '1' else 'real'
    norm = 'l2' if c == '1' else 'none'
    return synthetic, norm


def _features(exp_dir: str) -> str:
    s = str(exp_dir)
    m = re.search(r'enc-([A-Za-z_]+?)_', s)
    if m:
        return m.group(1)
    for tok in ('atom_encoder', 'onehot', 'linear'):
        if re.search(rf'_{tok}_', s) or s.endswith(f'_{tok}'):
            return tok
    return 'onehot'


def _injection(exp_dir: str, family: str) -> str:
    s = str(exp_dir)
    m = re.search(r'inj(\d{3})', s)
    if m:
        return m.group(1)
    # derive from variant-tag injection token: wf / wm / wr joined by '+'
    bits = ['0', '0', '0']
    if re.search(r'(^|_)wf(\+|_)', s):
        bits[0] = '1'
    if re.search(r'\+wm(\+|_)', s) or re.search(r'(^|_)wm(\+|_)', s):
        bits[1] = '1'
    if re.search(r'\+wr(\+|_)', s) or re.search(r'(^|_)wr(\+|_)', s):
        bits[2] = '1'
    if bits != ['0', '0', '0']:
        return ''.join(bits)
    if family in INJECTION_AGNOSTIC:
        return 'na'
    return 'na'


def _synthetic(exp_dir: str) -> str:
    s = str(exp_dir)
    if re.search(r'(^|[_/])gt([_/]|$)', s):
        return 'gt'
    return 'real'


def _epochs(exp_dir: str):
    m = re.search(r'ep(\d+)', str(exp_dir))
    return int(m.group(1)) if m else None


def _norm(row) -> str:
    v = str(row.get('conv_normalize', '') or '').strip()
    if v and v.lower() != 'nan':
        return v
    s = str(row.get('exp_dir', ''))
    m = re.search(r'norm-([A-Za-z0-9]+)', s)
    if m:
        return m.group(1)
    if 'noLN' in s:
        return 'none'
    return ''


def _prefer(df: pd.DataFrame, col: str, parsed: pd.Series) -> pd.Series:
    """Use an EXISTING explicit column (e.g. merged from config.json) wherever it
    is present, falling back to the path-parsed value otherwise. This lets new
    canonical runs (which carry real axis columns) bypass the legacy regex
    parsing while old runs are still decoded from their paths."""
    if col not in df.columns:
        return parsed
    existing = df[col]
    has = existing.notna() & (existing.astype(str).str.strip().ne(''))
    return existing.where(has, parsed)


def resolve_family(meta: dict, exp_dir: str = '') -> str:
    """Resolve model family: path segment (gsat/baselines/…) preferred over summary fields."""
    fam = _family(exp_dir) if exp_dir else ''
    if fam:
        return fam
    mt = (meta.get('model_type') or '').lower()
    mm = (meta.get('motif_method') or '').lower()
    if 'mose' in mt or mm == 'mose':
        return 'mose'
    if 'motifsat' in mt or 'gsat' in mt:
        return 'gsat' if mm == 'none' else 'motifsat'
    if mm in ('readout', 'loss'):
        return 'motifsat'
    if 'vanilla' in mt or mm == 'none':
        return 'vanilla'
    return mt or mm or 'unknown'


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized axis columns derived from explicit columns (config.json,
    when present) + exp_dir parsing (legacy fallback)."""
    df = df.copy()
    exp = df['exp_dir'].astype(str)

    fam = exp.map(_family)
    # fall back to motif_method only to separate mose/motifsat from the rest
    if 'motif_method' in df.columns:
        need = fam == ''
        mm = df['motif_method'].fillna('none').astype(str)
        fam = fam.where(~need, mm.map(
            lambda m: 'mose' if m == 'mose'
            else ('motifsat' if m in ('readout', 'loss')
                  else 'vanilla')))
    df['family'] = _prefer(df, 'family', fam)

    df['fold'] = _prefer(df, 'fold', exp.map(_fold))

    # fragmentation + threshold from the (reliable) vocab_variant column
    vv = df.get('vocab_variant', pd.Series([''] * len(df))).fillna('').astype(str)
    df['threshold'] = _prefer(
        df, 'threshold', vv.map(lambda v: 'on' if v.endswith('_filter') else 'off'))
    df['fragmentation'] = _prefer(
        df, 'fragmentation',
        vv.map(lambda v: v[:-len('_filter')] if v.endswith('_filter') else v))

    # priority-prefix overrides (only where present)
    pri = exp.map(_priority_axes)
    pri_syn = pri.map(lambda t: t[0])
    pri_norm = pri.map(lambda t: t[1])

    syn = exp.map(_synthetic)
    syn = syn.where(pri_syn.isna(), pri_syn)
    df['synthetic'] = _prefer(df, 'synthetic', syn)

    df['features'] = _prefer(df, 'features', exp.map(_features))
    inj = pd.Series([_injection(e, f) for e, f in zip(exp, df['family'])],
                    index=df.index)
    df['injection'] = _prefer(df, 'injection', inj)
    df['epochs'] = _prefer(df, 'epochs', exp.map(_epochs))

    nrm = df.apply(_norm, axis=1)
    nrm = nrm.where(pri_norm.isna(), pri_norm)
    df['norm'] = _prefer(df, 'norm', nrm)

    return df


def experiment_id(row, axes) -> str:
    parts = []
    for a in axes:
        v = row.get(a)
        v = '' if v is None or (isinstance(v, float) and pd.isna(v)) else v
        parts.append(f'{a}={v}')
    return ' | '.join(parts)


# ── metric resolution ──────────────────────────────────────────────────────────

# Special pseudo-metric: predictive performance, auto-resolved per task type
# (auc for classification, pearson for regression).
PERF = 'performance'

# Metrics reported per experiment by default (filtered to those present). This is
# the FULL result set, not just model performance: prediction + every
# explainability metric the eval pipeline writes into summary.json.
DEFAULT_REPORT_METRICS = [PERF, 'pearson', 'spearman',
                          'gt_roc_auc_mean', 'gt_roc_node_auc_mean', 'gt_roc_edge_auc_mean',
                          'gt_roc_node_mean_auc_mean', 'gt_roc_node_max_auc_mean',
                          'gnnexplainer_mean_gt_roc_node_auc_mean',
                          'gnnexplainer_max_gt_roc_node_auc_mean',
                          'pgexplainer_mean_gt_roc_node_auc_mean',
                          'pgexplainer_max_gt_roc_node_auc_mean',
                          'mage_mean_gt_roc_node_auc_mean',
                          'mage_max_gt_roc_node_auc_mean',
                          'top_k_abs_disc', 'mean_abs_disc', 'score_disc_spearman']

METRIC_LABELS = {
    PERF:                  'predictive performance (auc / rmse_orig|mae_orig for regression)',
    'pearson':             'score-vs-impact correlation (pearson)',
    'spearman':            'score-vs-impact correlation (spearman)',
    'gt_roc_auc_mean':     'explanation GT-ROC AUC (primary level)',
    'gt_roc_node_auc_mean': 'explanation GT-ROC AUC (node level, raw per-node)',
    'gt_roc_edge_auc_mean': 'explanation GT-ROC AUC (edge level)',
    'gt_roc_node_mean_auc_mean': 'explanation GT-ROC AUC (node, mean-of-motif)',
    'gt_roc_node_max_auc_mean':  'explanation GT-ROC AUC (node, max-of-motif)',
    'gnnexplainer_mean_gt_roc_node_auc_mean': 'GNNExplainer GT-ROC AUC (node, mean)',
    'gnnexplainer_max_gt_roc_node_auc_mean':  'GNNExplainer GT-ROC AUC (node, max)',
    'pgexplainer_mean_gt_roc_node_auc_mean':  'PGExplainer GT-ROC AUC (node, mean)',
    'pgexplainer_max_gt_roc_node_auc_mean':   'PGExplainer GT-ROC AUC (node, max)',
    'mage_mean_gt_roc_node_auc_mean':         'MAGE GT-ROC AUC (node, mean)',
    'mage_max_gt_roc_node_auc_mean':          'MAGE GT-ROC AUC (node, max)',
    'top_k_abs_disc':      'top-k motif |discriminativeness|',
    'mean_abs_disc':       'mean motif |discriminativeness|',
    'score_disc_spearman': 'score-vs-discriminativeness (spearman)',
}


def _perf_score(row) -> float:
    """Task-aware predictive metric: auc for classification; RMSE/MAE for regression."""
    ds = str(row.get('dataset', ''))
    if ds in REGRESSION_DATASETS:
        for col in ('rmse_orig', 'mae_orig', 'rmse', 'mae'):
            v = pd.to_numeric(row.get(col), errors='coerce')
            if pd.notna(v):
                return v
        return float('nan')
    v = pd.to_numeric(row.get('auc'), errors='coerce')
    if pd.isna(v):
        v = pd.to_numeric(row.get('pearson'), errors='coerce')
    return v


def _mean_std(series):
    s = pd.to_numeric(series, errors='coerce').dropna()
    if len(s) == 0:
        return None, None
    return (round(float(s.mean()), 6),
            round(float(s.std()), 6) if len(s) > 1 else 0.0)


# ── aggregation ─────────────────────────────────────────────────────────────────

def build_tidy(df: pd.DataFrame, metrics, exp_axes) -> pd.DataFrame:
    """Long/tidy table: one row per (experiment, family, backbone, dataset) with
    fold-averaged mean/std for EVERY requested metric, the fold count, and the
    folds that ran. Columns are ``<metric>__mean`` / ``<metric>__std``.

    ``exp_axes`` are the config axes that define an experiment. When 'injection'
    is one of them, the injection-agnostic families (vanilla/baselines) are
    broadcast across every injection value present for the ante-hoc families so
    each experiment table stays complete.
    """
    df = df.copy()
    df[PERF] = df.apply(_perf_score, axis=1)
    df['_exp'] = df.apply(lambda r: experiment_id(r, exp_axes), axis=1)
    # base experiment id = experiment axes minus injection (broadcast key)
    base_axes = [a for a in exp_axes if a != 'injection']
    df['_base_exp'] = df.apply(lambda r: experiment_id(r, base_axes), axis=1)

    inj_is_sep = 'injection' in exp_axes
    antehoc = df[~df['family'].isin(INJECTION_AGNOSTIC)]
    inj_by_base = (antehoc.groupby('_base_exp')['injection']
                   .agg(lambda s: sorted(set(s))).to_dict()) if inj_is_sep else {}

    records = []
    grp_cols = ['_exp', '_base_exp', 'family', 'backbone', 'dataset'] + ALL_AXES
    for keys, g in df.groupby(grp_cols, dropna=False):
        rec = dict(zip(grp_cols, keys))
        folds = sorted(int(f) for f in g['fold'].dropna().unique())
        rec['n_folds'] = len(folds)
        rec['folds'] = ','.join(map(str, folds))
        for m in metrics:
            src = g[m] if m in g.columns else None
            mean, std = _mean_std(src) if src is not None else (None, None)
            rec[f'{m}__mean'] = mean
            rec[f'{m}__std'] = std
        records.append(rec)
    tidy = pd.DataFrame.from_records(records)

    if inj_is_sep and not tidy.empty:
        broadcast_rows = []
        for _, r in tidy[tidy['family'].isin(INJECTION_AGNOSTIC)].iterrows():
            injs = [i for i in inj_by_base.get(r['_base_exp'], []) if i and i != 'na']
            for inj in injs:
                nr = r.copy()
                nr['injection'] = inj
                nr['_exp'] = experiment_id(nr, exp_axes)
                broadcast_rows.append(nr)
        if broadcast_rows:
            tidy = pd.concat([tidy, pd.DataFrame(broadcast_rows)], ignore_index=True)
            tidy = tidy.drop_duplicates(subset=['_exp', 'family', 'backbone', 'dataset'])

    return tidy.drop(columns=['_base_exp']).rename(columns={'_exp': 'experiment'})


def write_per_experiment_tables(tidy: pd.DataFrame, save_dir: Path, metrics) -> None:
    """One markdown file per experiment; within it a section per metric, and
    within each metric a backbone×dataset pivot per model family."""
    save_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for exp_id, g in tidy.groupby('experiment'):
        safe = re.sub(r'[^A-Za-z0-9]+', '_', exp_id).strip('_')[:120]
        lines = [f'# Experiment: {exp_id}\n']
        present_metrics = []
        for m in metrics:
            mcol, scol = f'{m}__mean', f'{m}__std'
            if mcol not in g.columns or g[mcol].notna().sum() == 0:
                continue
            present_metrics.append(m)
            lines.append(f'\n## {METRIC_LABELS.get(m, m)}\n')

            def cell(row, mcol=mcol, scol=scol):
                mv = row.get(mcol)
                if mv is None or pd.isna(mv):
                    return f"– (folds={row['folds']})"
                sv = row.get(scol)
                std = '' if (sv is None or pd.isna(sv) or not sv) else f"±{sv:.3f}"
                return f"{mv:.4f}{std} [f:{row['folds']}]"

            for fam, gf in g.groupby('family'):
                gf = gf.assign(_cell=gf.apply(cell, axis=1))
                piv = gf.pivot_table(index='backbone', columns='dataset',
                                     values='_cell', aggfunc='first')
                lines.append(f'\n### {fam}\n')
                try:
                    lines.append(piv.to_markdown())
                except Exception:
                    lines.append(piv.to_string())
                lines.append('\n')
        (save_dir / f'experiment__{safe}.md').write_text('\n'.join(lines))
        index.append({'experiment': exp_id, 'file': f'experiment__{safe}.md',
                      'metrics': ','.join(present_metrics), 'rows': len(g)})
    pd.DataFrame(index).sort_values('experiment').to_csv(
        save_dir / 'experiments_index.csv', index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--csv', help='existing combined results CSV')
    src.add_argument('--out_root', help='output tree to collect summary.json from')
    ap.add_argument('--save_dir', default='experiment_tables')
    ap.add_argument('--metrics', nargs='*', default=None,
                    help='metric columns to report per experiment. Use the '
                         f'pseudo-metric "{PERF}" for auc/pearson-by-task. '
                         f'Default reports {DEFAULT_REPORT_METRICS} (filtered to '
                         'columns present). Pass any summary.json column, e.g. '
                         'gnnexplainer_mean_pearson, to add explainer metrics.')
    ap.add_argument('--metric', default=None,
                    help='(legacy) single metric column; same as --metrics <col>.')
    ap.add_argument('--experiment_axes', default=','.join(DEFAULT_EXPERIMENT_AXES),
                    help='comma list of axes that define an experiment. '
                         f'Choose from {ALL_AXES}. '
                         'Default excludes injection so all model families share '
                         'one table; add "injection" when sweeping it.')
    ap.add_argument('--exclude', nargs='*', default=None,
                    help='extra directory-name prefixes to skip when walking '
                         f'--out_root (always skips {ARCHIVE_PREFIXES}).')
    ap.add_argument('--vocab_variant', nargs='*', default=None,
                    help='keep ONLY these vocab variants, e.g. '
                         '--vocab_variant rbrics_old_filter (default: all).')
    args = ap.parse_args()

    exp_axes = [a for a in args.experiment_axes.split(',') if a]
    bad = [a for a in exp_axes if a not in ALL_AXES]
    if bad:
        raise SystemExit(f'unknown experiment axes {bad}; choose from {ALL_AXES}')

    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        rows = []
        root = Path(args.out_root)
        n_cfg = 0
        for p in iter_summaries(root, args.exclude):
            try:
                d = json.load(open(p))
            except Exception as e:
                print(f'  [warn] skip corrupt summary {p}: {e}')
                continue
            cfg_path = p.parent / 'config.json'
            if cfg_path.exists():
                try:
                    cfg = json.load(open(cfg_path))
                    for k, v in cfg.items():
                        d.setdefault(k, v)
                    n_cfg += 1
                except Exception as e:
                    print(f'  [warn] skip corrupt config {cfg_path}: {e}')
            d['exp_dir'] = str(p.parent.relative_to(root))
            rows.append(d)
        if not rows:
            raise SystemExit('no summary.json files found under --out_root')
        df = pd.DataFrame(rows)
        if n_cfg:
            print(f'  merged config.json for {n_cfg} run(s)')

    if args.vocab_variant:
        keep = set(args.vocab_variant)
        before = len(df)
        vv = df.get('vocab_variant', pd.Series([''] * len(df))).astype(str)
        df = df[vv.isin(keep)].copy()
        print(f'filtered to vocab_variant in {sorted(keep)}: '
              f'{len(df)}/{before} rows')
        if df.empty:
            raise SystemExit(f'no rows with vocab_variant in {sorted(keep)}')

    df = normalize(df)

    # resolve which metrics to report
    if args.metrics:
        requested = args.metrics
    elif args.metric:
        requested = [args.metric]
    else:
        requested = DEFAULT_REPORT_METRICS
    metrics = []
    for m in requested:
        if m == PERF or m in df.columns:
            metrics.append(m)
        else:
            print(f'  [skip] metric {m!r} not present in data')
    if not metrics:
        metrics = [PERF]

    tidy = build_tidy(df, metrics, exp_axes)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    tidy_path = save_dir / 'results_tidy.csv'
    tidy.sort_values(['experiment', 'family', 'backbone', 'dataset']).to_csv(
        tidy_path, index=False)
    write_per_experiment_tables(tidy, save_dir, metrics)

    n_exp = tidy['experiment'].nunique()
    print(f'normalized {len(df)} rows -> {len(tidy)} tidy rows')
    print(f'{n_exp} distinct experiment(s); metrics: {metrics}')
    print(f'wrote {tidy_path}')
    print(f'wrote per-experiment tables under {save_dir}/')


if __name__ == '__main__':
    main()
