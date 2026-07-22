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
/ Motif-Occlusion) do not depend on the injection axis, their rows are *broadcast* into
every injection variant that shares the remaining axes, so each per-experiment
table/plot is complete.

This module is robust to multiple directory schemes in ``all_results.csv``:

  * phased pipeline (shell) ``<family>/<variant>/<dataset>/fold<k>/<variant_tag>``
  * grid driver           ``<family>/<ds>/fold<k>/<variant>/enc-..._inj..._ep..._real/``
  * legacy priority sweep ``A0_B0_C0/...`` (parser disabled; re-enable _priority_axes if needed)

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

POSTHOC_EXPLAINERS = ('gnnexplainer', 'pgexplainer', 'motif_occlusion', 'mage_official')
EXPLAINER_AGGS = ('mean', 'max')


def _alias_legacy_motif_occlusion_keys(d: dict) -> dict:
    """Backward-compat: pre-rename runs saved the Motif-Occlusion baseline (then
    mislabelled "MAGE") under bare ``mage_*`` summary keys. Official MAGE now uses
    the distinct ``mage_official_*`` namespace, so any bare ``mage_*`` key (never
    ``mage_official_*``) is unambiguously the legacy Motif-Occlusion — copy it onto
    the ``motif_occlusion_*`` key. Mutates and returns ``d``."""
    legacy = [k for k in d if str(k).startswith('mage_')
              and not str(k).startswith('mage_official_')]
    for k in legacy:
        d.setdefault('motif_occlusion_' + str(k)[len('mage_'):], d[k])
    return d

# Prediction AUC is identical on vanilla vs its baselines row (epochs=0 reload).
PREDICTION_FAMILIES = ('vanilla', 'mose', 'motifsat', 'gsat', 'base_gsat')
# Families whose identity does NOT include the injection axis. Their rows are
# broadcast across every injection value present for the ante-hoc families.
INJECTION_AGNOSTIC = {'vanilla', 'baselines'}

FAMILIES = ('vanilla', 'baselines', 'mose', 'motifsat', 'gsat', 'base_gsat')

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


def dataset_allowed(run_path: Path, datasets: set[str] | None) -> bool:
    """True when *run_path* (run dir or summary.json) belongs to an allowed dataset."""
    if not datasets:
        return True
    allowed = set(datasets)
    run_dir = run_path.parent if run_path.name == 'summary.json' else Path(run_path)
    if any(part in allowed for part in run_dir.parts):
        return True
    sj = run_dir / 'summary.json'
    if sj.exists():
        try:
            with open(sj, encoding='utf-8') as f:
                meta = json.load(f)
            return str(meta.get('dataset', '')) in allowed
        except Exception:
            pass
    return False


def iter_summaries(root, extra_excludes=(), datasets=None):
    """Yield every summary.json under ``root`` whose relative path does NOT pass
    through an excluded directory (archive/scratch dirs by default).

    When *datasets* is set (e.g. ``{'mutag'}``), only yields summaries for those
    datasets (matched by path segment or ``summary.json`` metadata).
    """
    root = Path(root)
    excl = tuple(ARCHIVE_PREFIXES) + tuple(extra_excludes or ())
    allowed = set(datasets) if datasets else None
    for p in root.rglob('summary.json'):
        rel = p.relative_to(root)
        if any(part.startswith(excl) for part in rel.parts):
            continue
        if not dataset_allowed(p, allowed):
            continue
        yield p


# ── normalization helpers ──────────────────────────────────────────────────────

def _truthy(v) -> bool:
    if v is True:
        return True
    if v is False or v is None:
        return False
    return str(v).strip().lower() in ('1', 'true', 'yes', 'on')


_RULE_TIERS = ('easy', 'medium', 'hard')


def parse_vocab_variant(vv: str) -> tuple[str, bool, bool, str]:
    """Split bundled vocab_variant into (base, is_filter, is_relabelled, tier).

    Suffix order (outermost last): ``{base}[_filter]_relabelled[_<tier>]``. The
    difficulty tier (easy/medium/hard, from RULE_TIERS=1) is only valid directly
    after ``_relabelled``; it is stripped first, then ``_relabelled``, then
    ``_filter``. ``tier`` is ``'none'`` for a non-tiered (single-best-rule) run."""
    v = (vv or '').strip()
    is_relabelled = False
    is_filter = False
    tier = 'none'
    for _t in _RULE_TIERS:
        if v.endswith(f'_relabelled_{_t}'):
            tier = _t
            v = v[:-len(f'_{_t}')]          # drop the tier, leaving …_relabelled
            break
    if v.endswith('_relabelled'):
        is_relabelled = True
        v = v[:-len('_relabelled')]
    if v.endswith('_filter'):
        is_filter = True
        v = v[:-len('_filter')]
    return v, is_filter, is_relabelled, tier


def _config_sig(exp_dir: str) -> str:
    """Fold-invariant run signature = the run path with the ``fold<k>`` segment
    removed. The run-dir tag encodes EVERY config axis (backbone, method, vocab,
    noise/info_loss_level, and the hp-hash of info_loss_coef / final_r / init_r /
    size_reg / ent_reg / num_layers / lr / dropout / … — see loader._HP_HASHED /
    _HP_SPELLED), while ``fold`` is a separate path segment. So two runs share a
    ``config_sig`` iff they are the SAME configuration on different folds. Adding
    it to the aggregation key guarantees only folds are collapsed — runs that
    differ in any hyperparameter stay separate rows instead of being averaged."""
    parts = [p for p in str(exp_dir).split('/')
             if p and not re.fullmatch(r'fold\d+', p)]
    return '/'.join(parts)


def family_of(meta: dict) -> str:
    """The run's model family, read from the authoritative ``family`` field in
    summary.json (base_gsat normalised to gsat). Fails fast if absent — every
    run.py records it, so a run without it is an incomplete summary. No path
    fallback."""
    fam = str(meta.get('family') or '').strip()
    if not fam or fam.lower() in ('nan', 'none', 'null'):
        raise ValueError(
            "summary.json is missing the 'family' field. Every run.py records it "
            "(mose / motifsat / gsat / vanilla / baselines); re-run or regenerate "
            "to produce a complete summary. No path fallback.")
    return 'gsat' if fam == 'base_gsat' else fam


# Every run.py writes these to summary.json (family + training_summary_extras).
# Analysis reads them DIRECTLY — no path-token parsing, no silent fallback.
REQUIRED_SUMMARY_FIELDS = (
    'family', 'dataset', 'backbone', 'fold', 'vocab_variant',
    'node_encoder', 'conv_normalize', 'use_gt', 'epochs',
)
# Injection flags are required only for injection-BEARING families. vanilla /
# baselines are injection-agnostic (injection axis = 'na', see _inj) and
# legitimately record these as null; requiring them there would wrongly abort
# the whole collect. Ante-hoc families (mose/motifsat/gsat) must record them.
INJECTION_REQUIRED_FIELDS = ('w_feat', 'w_message', 'w_readout')


def _require_field(df: pd.DataFrame, col: str) -> pd.Series:
    """Return df[col], failing fast if the column is absent or any value is
    blank/null. No path fallback — a run missing it is a broken summary."""
    if col not in df.columns:
        raise ValueError(
            f"normalize: required field '{col}' is absent from the collected "
            f"summaries. Every run.py writes it to summary.json (family + "
            f"training_summary_extras); re-collect from complete summaries. "
            f"No path fallback.")
    s = df[col]
    # Real nulls (NaN/None) are caught by isna(); the string 'none' is NOT blank
    # — it is a valid value (e.g. conv_normalize='none' = no normalization).
    blank = s.isna() | s.astype(str).str.strip().str.lower().isin(('', 'nan', 'null'))
    if blank.any():
        ex = df.loc[blank, 'exp_dir'].iloc[0] if 'exp_dir' in df.columns else '?'
        raise ValueError(
            f"normalize: field '{col}' is blank/missing for {int(blank.sum())} "
            f"run(s) (e.g. exp_dir={ex!r}). Fix the summary or re-run — '{col}' "
            f"has no fallback.")
    return s


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized axis columns read DIRECTLY from summary fields.

    No path-token parsing and no fallback: every axis value comes from an
    explicit field that every run.py records in summary.json. A run missing a
    field fails loudly (see ``_require_field``) rather than being guessed. Only
    ``config_sig`` uses the run's own directory — as its unique storage identity
    (the merge key), never to infer an axis value.
    """
    df = df.copy()
    if 'exp_dir' not in df.columns:
        df['exp_dir'] = ''

    for col in REQUIRED_SUMMARY_FIELDS:
        _require_field(df, col)

    # family (authoritative field; base_gsat normalised to gsat)
    df['family'] = df['family'].astype(str).map(
        lambda s: 'gsat' if s == 'base_gsat' else s)

    # Injection flags: required only for injection-bearing families (mirrors _inj).
    # Injection-agnostic families (vanilla/baselines) may leave them null.
    antehoc = df[~df['family'].isin(INJECTION_AGNOSTIC)]
    if not antehoc.empty:
        for col in INJECTION_REQUIRED_FIELDS:
            _require_field(antehoc, col)

    # fold (numeric)
    df['fold'] = pd.to_numeric(df['fold'], errors='coerce').astype('Int64')

    # vocab_variant → base + filter/relabel flags (parsing a FIELD, not the path)
    vv = df['vocab_variant'].astype(str)
    parsed = vv.map(parse_vocab_variant)
    df['vocab_base'] = parsed.map(lambda t: t[0])
    df['is_filter'] = parsed.map(lambda t: t[1])
    df['is_relabelled'] = parsed.map(lambda t: t[2])
    # Difficulty tier: prefer the authoritative `gt_tier` summary field (the tier is
    # NOT in vocab_variant — that stays the base vocab for loading). Fall back to any
    # `_relabelled_<tier>` suffix parsed from the variant name (legacy path-encoded).
    _parsed_tier = parsed.map(lambda t: t[3])        # easy/medium/hard or 'none'
    if 'gt_tier' in df.columns:
        _field_tier = df['gt_tier'].astype(str).str.strip().str.lower()
        _field_tier = _field_tier.where(_field_tier.isin(_RULE_TIERS), other='none')
        df['tier'] = _field_tier.where(_field_tier != 'none', _parsed_tier)
    else:
        df['tier'] = _parsed_tier
    # A GT-tier run is relabelled even though its vocab_variant is the base name.
    df['is_relabelled'] = df['is_relabelled'] | (df['tier'] != 'none')
    df['threshold'] = df['is_filter'].map(lambda x: 'on' if x else 'off')
    df['fragmentation'] = df['vocab_base']

    # synthetic = the recorded use_gt (relabelled vocab implies gt too)
    use_gt = df['use_gt'].apply(_truthy) | df['is_relabelled']
    df['synthetic'] = use_gt.map(lambda g: 'gt' if g else 'real')
    df['use_gt'] = use_gt.values

    # features = the recorded node_encoder
    df['features'] = df['node_encoder'].astype(str).str.strip()

    # norm = the recorded conv_normalize
    df['norm'] = df['conv_normalize'].astype(str).str.strip()

    # epochs = the recorded epochs
    df['epochs'] = pd.to_numeric(df['epochs'], errors='coerce').astype('Int64')

    # injection = the recorded w_feat/w_message/w_readout as a 3-bit string;
    # 'na' for injection-agnostic families (vanilla/baselines have no injection).
    def _inj(row):
        if row['family'] in INJECTION_AGNOSTIC:
            return 'na'
        return ''.join('1' if _truthy(row.get(c)) else '0'
                       for c in ('w_feat', 'w_message', 'w_readout'))
    df['injection'] = df.apply(_inj, axis=1)

    # config_sig = the run's own directory minus the fold segment: a unique,
    # fold-invariant identity that guarantees distinct configs never merge and
    # only folds collapse. This is the run's storage key, not a parsed axis value.
    df['config_sig'] = df['exp_dir'].astype(str).map(_config_sig)

    return df


def filter_prediction_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop baselines rows for predictive metrics — they duplicate vanilla (E8)."""
    fam = df.get('family', pd.Series([''] * len(df))).astype(str)
    return df[fam.isin(PREDICTION_FAMILIES)].copy()


def expand_posthoc_explainer_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Expand baselines summary columns into explainer-family rows (E5).

    Post-hoc GNNExplainer/PGExplainer/Motif-Occlusion metrics live as ``{expl}_{agg}_*``
    columns on baselines runs, not as separate run dirs.
    """
    fam = df.get('family', pd.Series([''] * len(df))).astype(str)
    bl = df[fam == 'baselines'].copy()
    if bl.empty:
        return bl

    pieces = []
    metric_map = [
        ('pearson', 'pearson'),
        ('spearman', 'spearman'),
        ('pearson_all', 'pearson_all'),
        ('spearman_all', 'spearman_all'),
        ('pearson_motif', 'pearson_motif'),
        ('spearman_motif', 'spearman_motif'),
        ('pearson_motif_all', 'pearson_motif_all'),
        ('spearman_motif_all', 'spearman_motif_all'),
        # Per-instance is TEST-scope for post-hoc baselines (explainers run on
        # test_list only), so no *_instance_all here — those exist for ante-hoc
        # trainers (via all_list) and come straight off their summary columns.
        ('pearson_instance', 'pearson_instance'),
        ('spearman_instance', 'spearman_instance'),
        ('pearson_instance_agnostic', 'pearson_instance_agnostic'),
        ('spearman_instance_agnostic', 'spearman_instance_agnostic'),
        ('gt_roc_node_auc_mean', 'gt_roc_node_auc_mean'),
        ('gt_roc_edge_auc_mean', 'gt_roc_edge_auc_mean'),
        ('gt_roc_node_auc_mean_all', 'gt_roc_node_auc_mean_all'),
        ('gt_roc_edge_auc_mean_all', 'gt_roc_edge_auc_mean_all'),
        ('top_k_abs_disc', 'top_k_abs_disc'),
        ('score_disc_spearman', 'score_disc_spearman'),
    ]
    # top_bottom (top-K vs bottom-K impact) + gt_vs_outside (GT vs non-GT motif
    # discrimination) per explainer — run_vanilla writes {ex}_{agg}_topbot_* /
    # {ex}_{agg}_gtvo_*; map them onto the generic columns so baseline explainer
    # rows populate the topbot_*/gtvo_* tables like the ante-hoc families do.
    for _k in ('top_mean_score', 'bottom_mean_score', 'top_mean_impact',
               'bottom_mean_impact', 'impact_ratio'):
        metric_map.append((f'topbot_{_k}', f'topbot_{_k}'))
    for _sub in ('all', 'positive_class', 'correct_positive_class'):
        for _k in ('gt_mean_impact', 'non_gt_mean_impact',
                   'gt_mean_score', 'non_gt_mean_score'):
            metric_map.append((f'gtvo_{_sub}_{_k}', f'gtvo_{_sub}_{_k}'))
    for _k in ('score_auc', 'gt_impact_rank'):
        metric_map.append((f'gtvo_{_k}', f'gtvo_{_k}'))
    for ex in POSTHOC_EXPLAINERS:
        for agg in EXPLAINER_AGGS:
            sub = bl.copy()
            sub['family'] = ex
            sub['explainer_agg'] = agg
            any_metric = False
            for dst, suffix in metric_map:
                src = f'{ex}_{agg}_{suffix}'
                if src in sub.columns:
                    sub[dst] = sub[src]
                    any_metric = True
            if any_metric:
                pieces.append(sub)
    if not pieces:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(pieces, ignore_index=True)


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
DEFAULT_REPORT_METRICS = [PERF, 'train_auc', 'val_auc',
                          'pearson', 'spearman',
                          'pearson_all', 'spearman_all',
                          'pearson_motif', 'spearman_motif',
                          'pearson_motif_all', 'spearman_motif_all',
                          # TRUE per-instance (per-graph weight vs per-graph impact)
                          # score-vs-impact — the instance-based counterpart of the
                          # mean-over-motifs (motif) tables above. 'own' = model's
                          # own-attention LOO impact; '_agnostic' = uniform-weight
                          # LOO impact. Baseline per-instance is test-scope only
                          # (ante-hoc also emit *_instance_all over all splits).
                          'pearson_instance', 'spearman_instance',
                          'pearson_instance_agnostic', 'spearman_instance_agnostic',
                          'pearson_instance_all', 'spearman_instance_all',
                          'pearson_instance_agnostic_all', 'spearman_instance_agnostic_all',
                          'gt_roc_auc_mean', 'gt_roc_node_auc_mean', 'gt_roc_edge_auc_mean',
                          'gt_roc_auc_mean_all', 'gt_roc_node_auc_mean_all',
                          'gt_roc_edge_auc_mean_all',
                          'gt_roc_node_mean_auc_mean', 'gt_roc_node_max_auc_mean',
                          'gt_roc_node_mean_auc_mean_all', 'gt_roc_node_max_auc_mean_all',
                          'gnnexplainer_mean_gt_roc_node_auc_mean',
                          'gnnexplainer_max_gt_roc_node_auc_mean',
                          'pgexplainer_mean_gt_roc_node_auc_mean',
                          'pgexplainer_max_gt_roc_node_auc_mean',
                          'motif_occlusion_mean_gt_roc_node_auc_mean',
                          'motif_occlusion_max_gt_roc_node_auc_mean',
                          'mage_official_mean_gt_roc_node_auc_mean',
                          'mage_official_max_gt_roc_node_auc_mean',
                          'top_k_abs_disc', 'mean_abs_disc', 'score_disc_spearman']

METRIC_LABELS = {
    PERF:                  'predictive performance (auc / rmse_orig|mae_orig for regression)',
    'pearson':             'score-vs-impact correlation (pearson)',
    'spearman':            'score-vs-impact correlation (spearman)',
    'pearson_all':         'score-vs-impact correlation (pearson, train+valid+test)',
    'spearman_all':        'score-vs-impact correlation (spearman, train+valid+test)',
    'pearson_motif':       'score-vs-impact correlation (pearson, motif-level aggregated)',
    'spearman_motif':      'score-vs-impact correlation (spearman, motif-level aggregated)',
    'pearson_motif_all':   'score-vs-impact correlation (pearson, motif-level, train+valid+test)',
    'spearman_motif_all':  'score-vs-impact correlation (spearman, motif-level, train+valid+test)',
    'gt_roc_auc_mean':     'explanation GT-ROC AUC (primary level)',
    'gt_roc_node_auc_mean': 'explanation GT-ROC AUC (node level, raw per-node)',
    'gt_roc_edge_auc_mean': 'explanation GT-ROC AUC (edge level)',
    'gt_roc_auc_mean_all':     'explanation GT-ROC AUC (primary, train+valid+test)',
    'gt_roc_node_auc_mean_all': 'explanation GT-ROC AUC (node, train+valid+test)',
    'gt_roc_edge_auc_mean_all': 'explanation GT-ROC AUC (edge, train+valid+test)',
    'gt_roc_node_mean_auc_mean': 'explanation GT-ROC AUC (node, mean-of-motif)',
    'gt_roc_node_max_auc_mean':  'explanation GT-ROC AUC (node, max-of-motif)',
    'gt_roc_node_mean_auc_mean_all': 'explanation GT-ROC AUC (node, mean-of-motif, train+valid+test)',
    'gt_roc_node_max_auc_mean_all':  'explanation GT-ROC AUC (node, max-of-motif, train+valid+test)',
    'gnnexplainer_mean_gt_roc_node_auc_mean': 'GNNExplainer GT-ROC AUC (node, mean)',
    'gnnexplainer_max_gt_roc_node_auc_mean':  'GNNExplainer GT-ROC AUC (node, max)',
    'pgexplainer_mean_gt_roc_node_auc_mean':  'PGExplainer GT-ROC AUC (node, mean)',
    'pgexplainer_max_gt_roc_node_auc_mean':   'PGExplainer GT-ROC AUC (node, max)',
    'motif_occlusion_mean_gt_roc_node_auc_mean': 'Motif-Occlusion GT-ROC AUC (node, mean)',
    'motif_occlusion_max_gt_roc_node_auc_mean':  'Motif-Occlusion GT-ROC AUC (node, max)',
    'mage_official_mean_gt_roc_node_auc_mean': 'MAGE (official) GT-ROC AUC (node, mean)',
    'mage_official_max_gt_roc_node_auc_mean':  'MAGE (official) GT-ROC AUC (node, max)',
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
    # config_sig is the fold-invariant run identity: including it in the group
    # key guarantees the ONLY axis collapsed is fold. Without it, runs that share
    # the coarse axes but differ in a hyperparameter (info_loss_coef, final_r,
    # noise, info_loss_level, num_layers, lr, size_reg/ent_reg, …) are silently
    # averaged into one cell.
    if 'config_sig' not in df.columns:
        df['config_sig'] = df.get('exp_dir', '').map(_config_sig)
    grp_cols = ['_exp', '_base_exp', 'config_sig',
                'family', 'backbone', 'dataset'] + ALL_AXES
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
            tidy = tidy.drop_duplicates(
                subset=['_exp', 'config_sig', 'family', 'backbone', 'dataset'])

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
                with open(p, encoding='utf-8') as f:
                    d = json.load(f)
                _alias_legacy_motif_occlusion_keys(d)
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
    from SharedModules.data.dataset_routing import collapse_redundant_folds
    df = collapse_redundant_folds(df)

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
